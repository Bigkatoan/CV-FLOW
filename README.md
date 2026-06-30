# CV-FLOW

DAM-backed Computer Vision pipeline framework, ROS2-inspired and topic-based.
Build CV pipelines (camera → preprocess → inference → postprocess → tracking →
output) as small `Node` classes wired together over shared-memory `.topic`
channels, run them with an `Executor`, and optionally drive them from a
FastAPI backend meant for a future visual editor.

Validated on an NVIDIA Jetson Orin Nano (JetPack 6, CUDA 12.6) with real CSI
camera output via GStreamer (and real or USB camera), TensorRT-accelerated
ONNX inference, and SQLite-backed pipeline persistence. See
[`PROJECT_REFERENCE.md`](PROJECT_REFERENCE.md) for full architecture/changelog
and [`CLAUDE.md`](CLAUDE.md) for AI-agent-oriented usage notes.

## Install

### On a regular machine (x86_64 / generic aarch64)

```bash
pip install "cv-flow[cpu] @ git+https://github.com/Bigkatoan/CV-FLOW.git@v0.3.0"
# or for GPU inference:
pip install "cv-flow[gpu] @ git+https://github.com/Bigkatoan/CV-FLOW.git@v0.3.0"
```

### On a Jetson (JetPack)

**Do not** `pip install` plain `opencv-python`, `torch`, or `onnxruntime-gpu`
on a Jetson — NVIDIA ships Jetson-specific builds of these (with
GStreamer/CUDA/TensorRT support) outside of PyPI, and the generic PyPI
wheels will either fail to install (no aarch64+CUDA wheel) or silently
replace a working GPU-accelerated install with a CPU-only one.

```bash
# 1. Make sure your venv can see the JetPack-provided system packages
#    (cv2 with GStreamer support lives in /usr/lib/python3.X/dist-packages,
#    not in any pip wheel):
#    edit <venv>/pyvenv.cfg -> include-system-site-packages = true

# 2. Install cv-flow itself WITHOUT pulling its declared deps — torch /
#    onnxruntime-gpu / opencv are already provided by the system / JetPack:
pip install -e . --no-deps

# 3. Install only the genuinely-missing pure-Python deps:
pip install pytest pytest-asyncio httpx fastapi uvicorn sqlalchemy aiosqlite \
    websockets paho-mqtt build

# 4. If cv2 import fails with "_ARRAY_API not found" / numpy ABI errors,
#    the JetPack cv2 was built against numpy 1.x — pin numpy accordingly:
pip install "numpy<2,>=1.26.0"
```

Verify nothing got clobbered:

```bash
python -c "import torch, onnxruntime, cv2; \
print(torch.__version__, torch.cuda.is_available()); \
print(onnxruntime.get_available_providers()); \
print(cv2.__version__, 'GStreamer' in cv2.getBuildInformation())"
```

## Quickstart

```python
import cv_flow

cv_flow.load_topics("topics/")  # parse + register every *.topic in a directory

class MyNode(cv_flow.Node):
    def initialize(self):
        self.sub = self.subscribe("camera_frame")
        self.pub = self.advertise("annotated_frame")

    def spin_once(self):
        frame = self.sub.read(timeout_ms=30)
        if frame is None:
            return
        # ... process ...
        self.pub.write(frame)

executor = cv_flow.Executor([MyNode()], hz=30)
executor.spin()  # blocking; Ctrl+C to stop, or raise StopIteration in spin_once()
```

Using built-in nodes:

```python
from cv_flow.nodes import VideoFileSource, YoloInference, NMS, DrawBbox, VideoWriter

source = VideoFileSource("camera_frame", path="input.mp4")
infer  = YoloInference("yolo_input", "yolo_raw",
                        model_path="yolov8n.onnx", device="cuda:0",
                        trt_cache_dir=".trt_cache")
...
cv_flow.Executor([source, ..., writer]).spin()
```

### Elastic auto-scaling (real multiprocessing)

`ElasticStage` runs N real worker processes for one transform stage
(e.g. inference) and merges results back in order — drop it in wherever
the wrapped node would otherwise go:

```python
import functools
from cv_flow.elastic import ElasticStage
from cv_flow.nodes import YoloInference

stage = ElasticStage(
    "yolo_input", "yolo_raw",
    node_factory=functools.partial(YoloInference, model_path="yolov8n.onnx",
                                    device="cuda:0", trt_cache_dir=".trt_cache"),
    initial_replicas=1, max_replicas=2,
)
cv_flow.Executor([..., stage, ...], elastic=True).spin()
```

On an 8GB Jetson Orin Nano, keep `max_replicas` small — each replica loads
its own full model and CUDA context, so VRAM/RAM cost scales linearly with
replica count.

## Benchmarks: cv-flow vs naive pipeline vs GStreamer vs DeepStream

Real numbers, measured on the same Jetson Orin Nano 8GB (JetPack 6, CUDA
12.6) used throughout this repo's development, with the **same model**
(`tests/fixtures/yolov8n.onnx`, TensorRT FP16), **same camera** (USB UVC,
1280×720), and **same post-processing math** (confidence 0.35, NMS IoU
0.45, features-first YOLOv8 decode) across every system, so the comparison
isolates *architecture* differences, not algorithm differences. Reproduce
with `scripts/bench_naive_pipeline.py`, `scripts/bench_cv_flow_pipeline.py`,
and `scripts/deepstream_bench/` (see that directory's README for the
DeepStream setup).

This camera maxes out at **30 FPS at 1280×720** (a hardware limit, not a
software one — confirmed independently via `v4l2-ctl` and a raw
`cv2.VideoCapture`/GStreamer capture test). Any system below shows two
numbers: **uncapped** (video file input, no camera ceiling — the system's
real maximum throughput) and **live camera** (USB camera, realistic
deployment latency, capped at 30 FPS regardless of how fast the pipeline
could otherwise go).

| System | Uncapped FPS | Live camera FPS | What's actually being measured |
|---|---|---|---|
| TensorRT inference only (no pipeline) | 113.0 | — | `onnxruntime` `session.run()` in a loop, no pre/post-processing |
| **naive pipeline** (plain Python, no DAM) | 25.0 | 21.8 | preprocess→TensorRT→NMS→track→draw, direct function calls, one process |
| **cv-flow** (full DAM pipeline) | 10.4 | 11.1 | identical algorithm, routed through `Node`/`PortBus`/`Executor` |
| GStreamer (`v4l2src`, capture only) | — | ~28.7–30.0 | raw camera capture + color convert, no inference (camera-bound) |
| **DeepStream** (`nvinfer` + custom YOLOv8 parser) | **205.4** | 29.98 | same model, in-graph TensorRT, NVMM zero-copy throughout |

Takeaways from these specific numbers:

- **cv-flow's DAM layer costs ~2x throughput** vs the identical algorithm
  called directly (21.8 → 11.1 FPS live camera, 25.0 → 10.4 FPS uncapped).
  That's the real, measured price of routing every stage through
  POSIX-shared-memory `PortBus`es — struct packing, a full memcpy per hop,
  and JSON metadata serialization, *seven* topic hops deep in this
  pipeline (camera→Tee→×2→preprocess→infer→NMS→track→draw). It buys
  process isolation, hot-swappable nodes, and (since v0.4.0) real
  multi-process elastic scaling — naive/plain-function pipelines can't
  offer any of that without rewriting the whole thing by hand.
- **DeepStream is ~20x cv-flow's real throughput** (205.4 vs 10.4 FPS
  uncapped) and is the only system here that isn't actually bottlenecked
  by this particular camera (it hits the live-camera number purely because
  the *camera* caps at 30 FPS, not because DeepStream ran out of headroom
  — 205 FPS uncapped proves there's a lot of margin left). The gap comes
  from architecture, not just "C++ vs Python": NVMM zero-copy buffers flow
  GPU-to-GPU from decode through inference to OSD with no CPU round-trip,
  and `nvinfer` pipelines/double-buffers inference asynchronously instead
  of the synchronous request-response loop both the naive and cv-flow
  benchmarks use.
- **GStreamer alone has no DNN inference** — `nvinfer` *is* DeepStream
  (DeepStream is built on top of GStreamer, not a separate competing
  framework). Plain GStreamer's advantage is purely at the capture/
  decode/colorspace-convert stage (hardware-accelerated, ~zero overhead
  vs the camera's own ceiling); to run a model in a plain GStreamer graph
  without DeepStream you'd write a custom `appsink` → Python/C++ → model
  bridge — architecturally the same category as this benchmark's "naive
  pipeline", not a third option.

### Trade-offs (not just the numbers)

| | cv-flow | GStreamer (plain) | DeepStream |
|---|---|---|---|
| **Raw throughput** | Lowest of the four real systems measured | N/A (no inference) | Highest by far |
| **Language / dev speed** | Pure Python, write a `Node` in minutes | C (or Python bindings), steeper API | C, custom parsers needed per model family (this repo had to write one for YOLOv8) |
| **Scaling model** | Real OS processes (`ElasticStage`), works for CPU *or* GPU stages, simple to reason about | N/A | In-graph batching across multiple streams; far more efficient per-stream but only for what `nvinfer`/plugins support |
| **Hardware portability** | Any machine with Python + the model runtime (CPU, CUDA, or — per `CHANGELOG.md` — anywhere TensorRT/CUDA/CPU onnxruntime runs) | NVIDIA Jetson/dGPU for the accelerated elements; the base framework itself runs anywhere | NVIDIA hardware only (Jetson or dGPU with matching JetPack/driver/TensorRT versions) |
| **Setup complexity** | `pip install`, write Python | Install GStreamer + plugins | Install DeepStream SDK (~632MB), write `.txt`/`.yml` configs, often a custom parser `.so` per model — this benchmark's YOLOv8 support took a full custom parser to get working |
| **Debuggability** | Plain Python stack traces, `pytest`, normal tooling | `gst-launch`/`GST_DEBUG`, less familiar to most Python devs | Same as GStreamer plus DeepStream-specific metadata/probe APIs |
| **Best fit** | Prototyping, research, pipelines that need elastic CPU/GPU worker scaling, teams that live in Python | Building custom capture/streaming pipelines without DNN inference, or as the substrate under a custom inference bridge | Production video analytics at scale on NVIDIA hardware where every FPS/watt matters and the model is well-supported (or you're willing to write the parser) |

In short: cv-flow trades raw throughput for Python-level development speed,
process-level isolation, and elastic scaling that's simple to reason
about. DeepStream trades all of that for an order of magnitude more
throughput on the same hardware, at the cost of NVIDIA-only portability,
C-level tooling, and per-model integration work. Neither is strictly
"better" — they're optimized for different points on the
throughput-vs-development-speed curve, and this repo's numbers (not just
qualitative claims) show exactly how large that gap is on real hardware.

List everything available:

```bash
cv-flow list-nodes
```

## Optional dependency groups

| Extra     | What it adds                                              |
|-----------|-------------------------------------------------------------|
| `cpu`     | `onnxruntime` (CPU-only inference)                          |
| `gpu`     | `torch` + `onnxruntime-gpu` (CUDA/TensorRT inference) — **do not use this extra on Jetson, see above** |
| `mqtt`    | `paho-mqtt`, for `MqttPublisher`                             |
| `stream`  | `websockets`, for `StreamViewer`                              |
| `backend` | `fastapi`, `uvicorn`, `sqlalchemy`, `aiosqlite` — the pipeline editor API |
| `dev`     | `pytest`, `pytest-asyncio`, `httpx` — test suite              |

## Offline / local reuse on another machine

```bash
python -m build --wheel        # or: scripts/build_wheel.sh
pip install --no-index --find-links ~/wheels "cv-flow[gpu]"
```

## Tests

```bash
pytest tests/ -v                 # full suite
pytest tests/ -m "not gpu"       # skip slow TensorRT-build tests
pytest tests/ -m hardware        # tests that need physically attached hardware
```
