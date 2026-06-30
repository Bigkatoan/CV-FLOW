#!/usr/bin/env python3
"""
scripts/bench_cv_flow_pipeline.py — cv-flow full pipeline benchmark,
uncapped throughput (file input, no camera framerate ceiling) or live
camera (realistic deployment), for direct comparison against
scripts/bench_naive_pipeline.py (same algorithm, no DAM layer).

Same node chain as scripts/smoke_pipeline.py (CameraSource/VideoFileSource
-> Tee -> Preprocess -> YoloInference -> NMS -> ObjectTracker -> DrawBbox ->
VideoWriter), through the real DAM/Topic/Node/Executor stack.

Usage:
    python scripts/bench_cv_flow_pipeline.py --source file --input synth.mp4 \
        --model tests/fixtures/yolov8n.onnx --frames 200
    python scripts/bench_cv_flow_pipeline.py --source camera --device-index 4 \
        --model tests/fixtures/yolov8n.onnx --frames 150
"""
from __future__ import annotations

import argparse
import time

import cv2

import cv_flow
from cv_flow.executor import Executor
from cv_flow.nodes.camera import CameraSource, VideoFileSource
from cv_flow.nodes.draw import DrawBbox
from cv_flow.nodes.inference import YoloInference
from cv_flow.nodes.output import VideoWriter
from cv_flow.nodes.postprocess import NMS
from cv_flow.nodes.preprocess import Preprocess
from cv_flow.nodes.tee import Tee
from cv_flow.nodes.tracking import ObjectTracker
from cv_flow.topic.topic import Topic
from cv_flow.topic.types import FieldDef, PortDef, TopicDef


def _probe_frame_shape(source: str, device_index: int, input_path: str | None,
                        width: int, height: int, fps: int):
    if source == "camera":
        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(input_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from source={source}")
    return frame.shape[:2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["file", "camera"], default="file")
    ap.add_argument("--input", default=None, help="Video file path (source=file)")
    ap.add_argument("--device-index", type=int, default=4)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
    ap.add_argument("--trt-cache-dir", default=".trt_cache")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", default="/tmp/cv_flow_bench_out.mp4")
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--max-tracks", type=int, default=100)
    args = ap.parse_args()

    cam_h, cam_w = _probe_frame_shape(args.source, args.device_index, args.input,
                                       args.width, args.height, args.fps)
    print(f"Frame shape: ({cam_h}, {cam_w}, 3)")

    frame_field = FieldDef.build("frame", "bgr8", (cam_h, cam_w))
    Topic(TopicDef(name="camera_frame", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[frame_field])))
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

    if args.source == "camera":
        source = CameraSource("camera_frame", device_index=args.device_index,
                                width=args.width, height=args.height, fps=args.fps)
    else:
        # fps=1000 -> negligible inter-frame pacing delay, i.e. effectively
        # uncapped throughput limited only by the pipeline itself.
        source = VideoFileSource("camera_frame", path=args.input, loop=True, fps=1000.0)

    tee = Tee("camera_frame", ["camera_frame_infer", "camera_frame_draw"])
    preprocess = Preprocess("camera_frame_infer", "yolo_input", width=640, height=640,
                              normalize="[0,1]")
    infer = YoloInference("yolo_input", "yolo_raw", model_path=args.model,
                            device=args.device, trt_cache_dir=args.trt_cache_dir)
    nms = NMS("yolo_raw", "detections", confidence_threshold=0.35,
               max_detections=md, output_layout="features_first")
    tracker = ObjectTracker("detections", "tracked", max_tracks=mt)
    drawer = DrawBbox("camera_frame_draw", "tracked", "annotated_frame")
    writer = VideoWriter("annotated_frame", output_path=args.out, fps=float(args.fps))

    counter = {"n": 0}
    orig_spin = source.spin_once

    def _counted_spin():
        orig_spin()
        counter["n"] += 1
        if counter["n"] >= args.frames:
            raise StopIteration

    source.spin_once = _counted_spin

    nodes = [source, tee, preprocess, infer, nms, tracker, drawer, writer]
    executor = Executor(nodes)

    print(f"Running {args.frames} frames through cv-flow's full pipeline "
          f"(source={args.source}, device={args.device})...")
    t0 = time.monotonic()
    executor.spin()
    elapsed = time.monotonic() - t0

    fps = counter["n"] / elapsed if elapsed > 0 else 0.0
    print(f"\nDone: {counter['n']} frames in {elapsed:.2f}s -> {fps:.1f} FPS")
    print(f"Output video: {args.out}")

    for node in nodes:
        for p in getattr(node, "_publishers", []):
            p._bus.close(unlink=True)


if __name__ == "__main__":
    main()
