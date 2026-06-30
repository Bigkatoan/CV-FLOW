#!/usr/bin/env python3
"""
scripts/bench_naive_pipeline.py — "naive" (no DAM) baseline for comparing
against cv-flow's full pipeline overhead.

Runs the EXACT same algorithm as scripts/smoke_pipeline.py — letterbox
resize, normalize, YOLOv8 TensorRT inference, NMS, ByteTrack, draw — but as
plain sequential Python function calls in one process, with NO Node/
PortBus/shared-memory hop between stages. This isolates the cost of
cv-flow's DAM layer (struct packing, shared-memory writes/reads, JSON
metadata) from the underlying compute, by reusing the identical pure
functions/classes cv-flow's nodes call internally:
  - cv_flow.nodes.preprocess.letterbox_resize / normalize_chw
  - onnxruntime.InferenceSession directly (same providers/trt_cache_dir)
  - cv_flow.nodes.postprocess.run_nms
  - cv_flow.nodes.tracking.ByteTrackLite
  - cv_flow.nodes.draw.draw_boxes

Two modes:
  --source synthetic   uncapped throughput — no camera framerate ceiling,
                        isolates pure pipeline speed (random frames, looped)
  --source camera       live USB camera — realistic deployment latency,
                        capped by the camera's own hardware framerate

Usage:
    python scripts/bench_naive_pipeline.py --source synthetic \
        --model tests/fixtures/yolov8n.onnx --frames 200
    python scripts/bench_naive_pipeline.py --source camera --device-index 4 \
        --model tests/fixtures/yolov8n.onnx --frames 150
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from cv_flow.nodes.draw import draw_boxes
from cv_flow.nodes.postprocess import run_nms
from cv_flow.nodes.preprocess import letterbox_resize, normalize_chw
from cv_flow.nodes.tracking import ByteTrackLite


def _make_session(model_path: str, device: str, trt_cache_dir: str | None):
    if device.startswith("cuda"):
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options: list[dict] = [{}, {}, {}]
        if trt_cache_dir:
            import os
            os.makedirs(trt_cache_dir, exist_ok=True)
            provider_options[0] = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
            }
        sess = ort.InferenceSession(model_path, providers=providers,
                                     provider_options=provider_options)
    else:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"naive pipeline: active ONNX Runtime provider = {sess.get_providers()[0]}")
    return sess


def _run_one_frame(frame, session, tracker, *, max_detections=100, max_tracks=100):
    resized = letterbox_resize(frame, 640, 640, keep_aspect=True)
    tensor = normalize_chw(resized, normalize="[0,1]")
    raw = session.run(["output0"], {"images": tensor})[0]
    boxes, scores, class_ids = run_nms(raw, confidence_threshold=0.35,
                                        max_detections=max_detections,
                                        output_layout="features_first")
    t_boxes, t_scores, t_cls, t_ids = tracker.update(boxes, scores, class_ids)
    annotated = draw_boxes(frame, t_boxes, scores=t_scores, class_ids=t_cls)
    return annotated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "camera"], default="synthetic")
    ap.add_argument("--device-index", type=int, default=4)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
    ap.add_argument("--trt-cache-dir", default=".trt_cache")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", default="/tmp/cv_flow_naive_out.mp4")
    args = ap.parse_args()

    session = _make_session(args.model, args.device, args.trt_cache_dir)
    tracker = ByteTrackLite(max_age=30, min_hits=3, iou_threshold=0.3)

    cap = None
    synthetic_frame = None
    if args.source == "camera":
        cap = cv2.VideoCapture(args.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read from /dev/video{args.device_index}")
        h, w = frame.shape[:2]
    else:
        h, w = args.height, args.width
        synthetic_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                              float(args.fps), (w, h))

    # Warm up (excludes TensorRT engine build / first-call overhead from the timed loop).
    warm_frame = synthetic_frame if synthetic_frame is not None else frame
    for _ in range(3):
        _run_one_frame(warm_frame, session, tracker)

    print(f"Running {args.frames} frames through the naive pipeline "
          f"(source={args.source}, device={args.device})...")
    n = 0
    t0 = time.monotonic()
    for i in range(args.frames):
        if args.source == "camera":
            ok, frame = cap.read()
            if not ok or frame is None:
                break
        else:
            frame = synthetic_frame
        annotated = _run_one_frame(frame, session, tracker)
        writer.write(annotated)
        n += 1
    elapsed = time.monotonic() - t0

    writer.release()
    if cap is not None:
        cap.release()

    fps = n / elapsed if elapsed > 0 else 0.0
    print(f"\nDone: {n} frames in {elapsed:.2f}s -> {fps:.1f} FPS")
    print(f"Output video: {args.out}")


if __name__ == "__main__":
    main()
