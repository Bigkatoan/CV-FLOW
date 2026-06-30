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
