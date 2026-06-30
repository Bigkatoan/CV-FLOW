# CLAUDE.md — using cv-flow as a dependency

This file is for an AI agent (Claude) working on a *different* project that
depends on `cv-flow`, or working inside this repo. It is not a user-facing
README (see `README.md`) — it exists so you don't have to re-derive API
shape, gotchas, and current real/stub status from scratch every time.

**Last updated:** v0.4.0 (2026-06-30, Phase 2 of the Jetson hardening plan).
Re-check `CHANGELOG.md`'s latest entry before trusting anything below if a
lot of time has passed — this file is a snapshot, not a live source of truth.

## Install (read this before running pip)

On a Jetson, **do not** let `pip` reinstall `torch`, `onnxruntime-gpu`, or
`opencv-python` — NVIDIA ships Jetson-specific builds outside PyPI (CUDA/
TensorRT/GStreamer support baked in) and a generic pip install will either
fail or silently swap in a CPU-only/no-GStreamer build.

```bash
pip install -e . --no-deps      # cv-flow itself, no dependency resolution
pip install pytest pytest-asyncio httpx fastapi uvicorn sqlalchemy \
    aiosqlite websockets paho-mqtt build   # genuinely-missing pure-Python deps
```

If `import cv2` fails with `_ARRAY_API not found` / a numpy ABI error, the
system OpenCV was built against numpy 1.x — run `pip install "numpy<2,>=1.26.0"`.
Full reasoning in `README.md` → "Install → On a Jetson".

On non-Jetson machines, normal `pip install cv-flow[gpu]` or `cv-flow[cpu]`
is fine — there's nothing special to avoid.

## Minimal API to build a pipeline

```python
import cv_flow

cv_flow.load_topics("topics/")   # parse + register every *.topic in a dir

class MyNode(cv_flow.Node):
    def initialize(self):                       # called once, before spin
        self.sub = self.subscribe("camera_frame")
        self.pub = self.advertise("annotated_frame")

    def spin_once(self):                         # called once per executor tick
        frame = self.sub.read(timeout_ms=30)
        if frame is None:
            return
        self.pub.write(frame)

cv_flow.Executor([MyNode()], hz=30).spin()        # blocking; Ctrl+C or StopIteration to stop
```

Real example pulled from `scripts/smoke_pipeline.py` (verified working on
real hardware, see below):

```python
from cv_flow.nodes import (CameraSource, Tee, Preprocess, YoloInference,
                            NMS, ObjectTracker, DrawBbox, VideoWriter)

camera = CameraSource("camera_frame", device_index=4, width=1280, height=720, fps=30)
tee    = Tee("camera_frame", ["camera_frame_infer", "camera_frame_draw"])
prep   = Preprocess("camera_frame_infer", "yolo_input", width=640, height=640, normalize="[0,1]")
infer  = YoloInference("yolo_input", "yolo_raw", model_path="yolov8n.onnx",
                        device="cuda:0", trt_cache_dir=".trt_cache")
nms    = NMS("yolo_raw", "detections", output_layout="features_first")
track  = ObjectTracker("detections", "tracked")
draw   = DrawBbox("camera_frame_draw", "tracked", "annotated_frame")
writer = VideoWriter("annotated_frame", output_path="out.mp4", fps=30.0)

cv_flow.Executor([camera, tee, prep, infer, nms, track, draw, writer]).spin()
```

List everything available with parameters: `cv-flow list-nodes`, or read
`cv_flow/nodes/_catalog.py` (`NODE_CATALOG`) directly.

## Conventions you must know before touching this codebase

- **Every `.topic` bus is a single-reader FIFO**, not a broadcast channel.
  The read cursor lives in the shared-memory header and is shared by every
  `Subscriber` created on that bus name. **If two different nodes both
  `subscribe()` the same topic, they compete for the same queue** — each
  only sees *some* frames, not every frame. This bit us for real building
  the Phase-1 smoke pipeline (a detect-then-draw pipeline needs the
  original frame both for `Preprocess` and later for `DrawBbox`). Fix:
  route through `Tee` first (`cv_flow/nodes/tee.py`) to fan one topic out
  to N independent topics. Symptom if you forget this: one of the
  competing subscribers' downstream bus fills up and starts logging
  `"queue full — dropped frame"` almost every iteration.
- **Bus naming is `f"{topic_name}_{session_id}"`.** `Executor` injects
  `session_id` (a uuid) into every node before `initialize()`. Writing a
  manual test without an `Executor`? Set `node._session_id = "..."` yourself
  before calling `node.initialize()`.
- **`peek()` vs `read()`:** `PortBus.peek()` inspects the head slot without
  advancing the read cursor. Only `MergeBus` needs this (compares several
  candidate buses' seq_no before deciding which one to actually `read()`).
  Don't `read()` a bus just to "look" at it — that consumes the frame.
- **Topic shapes are always fixed** — no variable-length data in this DAM
  model. Nodes whose output count varies (`NMS`, `ObjectTracker`) take a
  `max_*` parameter and pad with the sentinel `class_id = -1` ("no
  detection here"). Downstream consumers (`DrawBbox`, `ObjectTracker`)
  already skip `class_id == -1` entries.
- **`.topic` files are NOT YAML** — hand-rolled regex/line parser
  (`cv_flow/topic/parser.py`), intentionally, to avoid a YAML dependency
  and keep the format eyeball-readable. Don't "fix" this by swapping in a
  YAML lib.
- **`NMS(output_layout=...)`**: default `"features_first"` matches the
  standard YOLOv8 ONNX export shape `(1, 84, N)`. Don't use
  `output_layout="auto"` unless you specifically need the old
  shape-comparison heuristic (it's wrong whenever box count < 84 — kept
  only for backward compat, logs a WARNING when used).
- **`YoloInference`/`OnnxInference(device="cuda:0")`** tries providers in
  order `TensorrtExecutionProvider` → `CUDAExecutionProvider` →
  `CPUExecutionProvider`. Pass `trt_cache_dir=...` or every process start
  pays a ~1-2 min TensorRT engine rebuild. Logs which provider actually
  activated — check the log if inference seems slow, it may have silently
  fallen back to CPU.
- **`Preprocess(normalize=...)` matters for correctness, not just style.**
  YOLOv8 expects `normalize="[0,1]"` (plain 0-1 scaled RGB), NOT the
  default `"imagenet"` (mean/std normalized) — the default is meant for
  ImageNet-classifier-style models. Using the wrong one silently produces
  garbage detections, no error.
- **Real elastic scaling exists via `cv_flow.elastic.ElasticStage`** (since
  v0.4.0) — NOT via the `elastic`/`max_replicas` fields in `.topic` files
  (those are still parsed into `TopicDef` but not auto-consumed; elastic
  scaling is opt-in and explicit, not config-driven). Put `ElasticStage`
  directly in your node list wherever the wrapped transform node would
  otherwise go:
  ```python
  import functools
  from cv_flow.elastic import ElasticStage
  from cv_flow.nodes import YoloInference

  stage = ElasticStage(
      "yolo_input", "yolo_raw",
      node_factory=functools.partial(YoloInference, model_path="yolov8n.onnx",
                                      device="cuda:0", trt_cache_dir=".trt_cache"),
      initial_replicas=1, max_replicas=2,   # keep small on an 8GB Jetson — each
                                              # replica loads its own full model
  )
  cv_flow.Executor([..., stage, ...], elastic=True).spin()
  ```
  `node_factory` must be picklable (spawn start method) — a class +
  `functools.partial(..., kw=...)` works, a lambda does not. Real
  correctness (no lost/duplicated/misordered frames across real worker
  processes) is unit-tested in `tests/test_elastic.py` — read that file's
  module docstring before changing `cv_flow/elastic.py`, it documents two
  real concurrency bugs that were found and fixed building this (worker
  output seq numbering, and a genuine `PortBus` cross-process race — see
  the next bullet).
- **`PortBus`/`MergeBus` are NOT safe for a real concurrent writer process
  + reader process on the same bus** without external locking — only for
  the common case of one side at a time (true for every plain
  single-process pipeline in this codebase). `ElasticStage` works around
  this with a `multiprocessing.Lock` per worker; if you write NEW code with
  a genuine concurrent writer+reader on one `PortBus` (not through
  `ElasticStage`), you need the same kind of locking — see the concurrency
  notes in `cv_flow/dam/bus.py` and `cv_flow/dam/merge.py`.
- **`CudaPortBus` does NOT do real CUDA IPC, and a real attempt confirmed
  it isn't viable on this Jetson's integrated GPU** (see `CHANGELOG.md`
  `[0.4.0]`) — it always round-trips GPU tensors through CPU RAM, despite
  the class name. This is the final status, not a "later" gap — don't
  attempt the same `torch.multiprocessing.reductions.reduce_tensor()`
  approach again expecting a different result on Jetson hardware.

## Hardware verification status (don't assume more than this)

| Component | Status as of v0.4.0 |
|---|---|
| USB camera capture (`CameraSource(device_index=...)`) | **Verified on real hardware** (Intel RealSense color stream) |
| CSI camera (`CameraSource(gstreamer_pipeline=...)`, `build_nvargus_pipeline()`) | Code + unit-tested string generation only — **no physical CSI sensor tested yet** |
| RTSP (`RtspSource` reconnect backoff) | Backoff logic unit-tested (mocked) — **no real RTSP server tested yet** |
| ONNX inference (CPU / CUDA / TensorRT) | **Verified on real hardware** with a real exported yolov8n.onnx — see `scripts/bench_inference.py` output in `CHANGELOG.md` |
| Full pipeline (camera → infer → NMS → track → draw → video) | **Verified end-to-end on real hardware**, `scripts/smoke_pipeline.py`, ~7.9 FPS at 1280×720 input |
| `StreamViewer` (WebSocket) | Code reviewed only, no real client connected |
| `MqttPublisher` | Code reviewed only, no real broker connected |
| `CudaPortBus` real CUDA IPC | **Investigated for real, confirmed not viable on this hardware** — stays CPU-roundtrip, documented honestly (not a "later" gap) |
| `ElasticStage` real multiprocessing auto-scale | **Verified on real hardware**: real worker process spawn/scale-up/scale-down/shutdown, zero lost/duplicated/misordered frames across repeated stress runs — `tests/test_elastic.py`. NOT yet exercised with a real GPU-bound node_factory (only a trivial CPU numpy worker) — combining with real `YoloInference` workers is logically straightforward (same `node_factory` pattern already used in this file's example) but not itself separately verified end-to-end. |
| Backend `PipelineStore` (SQLite) | **Verified**: a record written by one process is readable after a simulated restart |
| Local wheel build + offline install | **Verified**: `scripts/build_wheel.sh` → installed into an isolated env via `--no-index --find-links` → imports work |

## Where to look for more

- `README.md` — human-facing install/quickstart/extras table.
- `PROJECT_REFERENCE.md` — architecture deep-dive, coding principles
  (§4, includes the Tee/single-reader-FIFO rule above), full file tree.
- `CHANGELOG.md` — what changed and when, in Keep-a-Changelog format.
- `cv_flow/nodes/_catalog.py` — machine-readable metadata for every node
  (parameters, types, defaults, descriptions) — the source of truth for
  what a node accepts, more reliable than guessing from the constructor.
