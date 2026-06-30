#!/usr/bin/env python3
"""
scripts/bench_inference.py — manual latency benchmark for ONNX Runtime
execution providers (CPU / CUDA / TensorRT) on this machine.

Not a pytest test — run by hand on the target device:

    python scripts/bench_inference.py --model tests/fixtures/yolov8n.onnx

TensorRT's first engine build is slow (~1-2 min on Jetson Orin Nano); pass
--trt-cache-dir to reuse a cached engine across runs (see cv_flow.nodes
.inference.OnnxInference(trt_cache_dir=...) for the same option in-pipeline).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


def bench_provider(model_path: str, providers: list[str], provider_options: list[dict],
                    input_name: str, input_shape: tuple, warmup: int, iters: int) -> dict:
    x = np.random.rand(*input_shape).astype(np.float32)

    t0 = time.monotonic()
    try:
        sess = ort.InferenceSession(model_path, providers=providers,
                                     provider_options=provider_options)
    except Exception as exc:  # noqa: BLE001 - report and move on
        return {"providers": providers, "error": str(exc)}
    load_s = time.monotonic() - t0

    active = sess.get_providers()
    for _ in range(warmup):
        sess.run(None, {input_name: x})

    t0 = time.monotonic()
    for _ in range(iters):
        sess.run(None, {input_name: x})
    avg_ms = (time.monotonic() - t0) / iters * 1000

    return {
        "providers": providers,
        "active": active,
        "load_s": load_s,
        "avg_ms": avg_ms,
        "fps": 1000.0 / avg_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to .onnx model")
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--shape", default="1,3,640,640", help="Comma-separated input shape")
    ap.add_argument("--trt-cache-dir", default=None,
                     help="Enable TensorRT engine disk cache at this path")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))

    trt_options: dict = {}
    if args.trt_cache_dir:
        import os
        os.makedirs(args.trt_cache_dir, exist_ok=True)
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": args.trt_cache_dir,
        }

    configs = [
        ("CPU", ["CPUExecutionProvider"], [{}]),
        ("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]),
        ("TensorRT", ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
         [trt_options, {}, {}]),
    ]

    print(f"Model: {args.model}  input={args.input_name} shape={shape}  "
          f"warmup={args.warmup} iters={args.iters}\n")
    print(f"{'Provider':<10} {'Active[0]':<28} {'Load(s)':>8} {'Avg(ms)':>9} {'FPS':>7}")
    print("-" * 68)

    for label, providers, provider_options in configs:
        result = bench_provider(args.model, providers, provider_options,
                                 args.input_name, shape, args.warmup, args.iters)
        if "error" in result:
            print(f"{label:<10} FAILED: {result['error']}")
            continue
        print(f"{label:<10} {result['active'][0]:<28} {result['load_s']:>8.2f} "
              f"{result['avg_ms']:>9.2f} {result['fps']:>7.1f}")


if __name__ == "__main__":
    main()
