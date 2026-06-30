#!/usr/bin/env python3
"""
scripts/smoke_pipeline.py — manual end-to-end hardware smoke test.

Real USB/CSI camera -> Preprocess -> YoloInference (TensorRT) -> NMS ->
ObjectTracker -> DrawBbox -> VideoWriter, run for a fixed number of frames
on real hardware. Not a pytest test — run by hand on the target device and
inspect the printed FPS + the output video:

    python scripts/smoke_pipeline.py --device-index 4 --model tests/fixtures/yolov8n.onnx \
        --frames 150 --out /tmp/smoke_out.mp4

Use `v4l2-ctl --list-devices` / try a few --device-index values if unsure
which /dev/video* node is the actual color stream (multi-stream depth
cameras like RealSense expose several non-color /dev/video* nodes too).
"""
from __future__ import annotations

import argparse
import time

import cv2

import cv_flow
from cv_flow.executor import Executor
from cv_flow.nodes.camera import CameraSource
from cv_flow.nodes.draw import DrawBbox
from cv_flow.nodes.inference import YoloInference
from cv_flow.nodes.output import VideoWriter
from cv_flow.nodes.postprocess import NMS
from cv_flow.nodes.preprocess import Preprocess
from cv_flow.nodes.tee import Tee
from cv_flow.nodes.tracking import ObjectTracker
from cv_flow.topic.topic import Topic
from cv_flow.topic.types import FieldDef, PortDef, TopicDef


def _probe_camera_frame_shape(device_index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(
            f"/dev/video{device_index} did not return a real BGR frame "
            f"(ok={ok}, frame={'None' if frame is None else frame.shape}). "
            "Try a different --device-index."
        )
    return frame.shape[:2]  # (h, w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", required=True, help="Path to yolov8 .onnx model")
    ap.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
    ap.add_argument("--trt-cache-dir", default=".trt_cache")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", default="/tmp/cv_flow_smoke_out.mp4")
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--max-tracks", type=int, default=100)
    args = ap.parse_args()

    print(f"Probing /dev/video{args.device_index} for real frame shape...")
    cam_h, cam_w = _probe_camera_frame_shape(
        args.device_index, args.width, args.height, args.fps,
    )
    print(f"Camera frame shape: ({cam_h}, {cam_w}, 3)")

    frame_field = FieldDef.build("frame", "bgr8", (cam_h, cam_w))
    Topic(TopicDef(name="camera_frame", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[frame_field])))
    # camera_frame is consumed by Tee only — Tee fans it out to two
    # independent topics so Preprocess and DrawBbox each get every frame
    # instead of competing for one single-reader FIFO (see Tee docstring).
    Topic(TopicDef(name="camera_frame_infer", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[frame_field])))
    Topic(TopicDef(name="camera_frame_draw", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[frame_field])))

    tensor_field = FieldDef.build("tensor", "float32", (1, 3, 640, 640))
    Topic(TopicDef(name="yolo_input", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[tensor_field])))

    raw_field = FieldDef.build("raw", "float32", (1, 84, 8400))
    Topic(TopicDef(name="yolo_raw", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[raw_field])))

    md = args.max_detections
    det_fields = [
        FieldDef.build("boxes", "float32", (md, 4)),
        FieldDef.build("scores", "float32", (md,)),
        FieldDef.build("class_ids", "int32", (md,)),
    ]
    Topic(TopicDef(name="detections", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=det_fields)))

    mt = args.max_tracks
    track_fields = [
        FieldDef.build("boxes", "float32", (mt, 4)),
        FieldDef.build("scores", "float32", (mt,)),
        FieldDef.build("class_ids", "int32", (mt,)),
        FieldDef.build("track_ids", "int32", (mt,)),
    ]
    Topic(TopicDef(name="tracked", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=track_fields)))

    Topic(TopicDef(name="annotated_frame", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[frame_field])))

    camera = CameraSource("camera_frame", device_index=args.device_index,
                            width=args.width, height=args.height, fps=args.fps)
    tee = Tee("camera_frame", ["camera_frame_infer", "camera_frame_draw"])
    preprocess = Preprocess("camera_frame_infer", "yolo_input", width=640, height=640,
                              normalize="[0,1]")  # YOLOv8 expects [0,1] RGB, not ImageNet norm
    infer = YoloInference("yolo_input", "yolo_raw", model_path=args.model,
                            device=args.device, trt_cache_dir=args.trt_cache_dir)
    nms = NMS("yolo_raw", "detections", confidence_threshold=0.35,
               max_detections=md, output_layout="features_first")
    tracker = ObjectTracker("detections", "tracked", max_tracks=mt)
    drawer = DrawBbox("camera_frame_draw", "tracked", "annotated_frame")
    writer = VideoWriter("annotated_frame", output_path=args.out, fps=float(args.fps))

    counter = {"n": 0}
    orig_spin = camera.spin_once

    def _counted_spin():
        orig_spin()
        counter["n"] += 1
        if counter["n"] >= args.frames:
            raise StopIteration

    camera.spin_once = _counted_spin

    nodes = [camera, tee, preprocess, infer, nms, tracker, drawer, writer]
    executor = Executor(nodes)

    print(f"Running {args.frames} frames through the full pipeline "
          f"(device={args.device}, model={args.model})...")
    t0 = time.monotonic()
    executor.spin()
    elapsed = time.monotonic() - t0

    fps = counter["n"] / elapsed if elapsed > 0 else 0.0
    print(f"\nDone: {counter['n']} frames in {elapsed:.1f}s -> {fps:.1f} FPS")
    print(f"Output video: {args.out}")

    for node in nodes:
        for p in getattr(node, "_publishers", []):
            p._bus.close(unlink=True)


if __name__ == "__main__":
    main()
