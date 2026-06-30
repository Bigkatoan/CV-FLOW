# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

## [0.4.0] — 2026-06-30 — Real elastic multiprocessing; honest CUDA IPC status

Phase 2 of the Jetson hardening plan: complete the originally-designed
architecture for real — real multiprocessing-backed elastic auto-scaling,
and a real (not just attempted) verdict on CUDA IPC zero-copy.

### Added
- `cv_flow.elastic.ElasticStage` — a drop-in replacement for a single
  transform Node (e.g. `YoloInference`) that internally fans work out to N
  real `multiprocessing.Process` workers (each running its own instance of
  a `node_factory`, in total isolation — own CUDA context, own model load)
  and merges results back in original order. `Executor.scale_up()`/
  `scale_down()` now have a REAL default implementation: they call
  `add_worker()`/`remove_worker()` on any node that exposes them (previously
  empty `logger.debug(...)` hooks).
- `RoundRobinBus.add_bus(name=...)` — optional explicit shared-memory
  segment name (previously always auto-generated with a random suffix),
  needed so a separately-spawned worker process can derive and attach to
  the exact same segment.

### Fixed — real bugs found building and stress-testing ElasticStage
- **`PortBus` cross-process concurrency**: `_write_header()` writes
  write_count/read_count/drop_count together based on whatever the caller
  last read, with no cross-process lock. A genuine concurrent writer
  process and reader process on the same bus (exactly what elastic workers
  are) can race and silently duplicate or drop a frame. This had never
  manifested before because every existing pipeline in this codebase is
  single-process/sequential. Documented prominently in `PortBus`'s,
  `RoundRobinBus`'s, and `MergeBus`'s docstrings. Fix is scoped to
  `ElasticStage`, not PortBus itself: every worker gets its own
  `multiprocessing.Lock`, held by both the main process and that worker
  around every operation on that worker's bus pair.
- **`MergeBus` ordering is best-effort, not strict**: `read()` only
  compares buses that already have data *right now* — it doesn't wait for
  a momentarily-empty bus that might still produce an earlier seq a few ms
  later, so a faster worker's later-seq result can be returned before a
  slower worker's earlier-seq one. `MergeBus`'s own docstring previously
  overstated this as a strict guarantee — corrected. `ElasticStage` adds
  its own reorder buffer on top to get real strict ordering, with a
  configurable stall-timeout skip-ahead so one genuinely lost upstream
  frame can't stall the pipeline's output forever.
- **Worker output seq numbering**: a worker's `Publisher.write()` defaults
  to an independent per-publisher auto-incrementing counter — fine for a
  single node, but means two different ElasticStage workers' "seq=1" are
  unrelated frames once there's more than one worker, silently breaking
  downstream ordering. Fixed by transparently wrapping each worker's
  publish call to carry the same seq_no as its most recent read.

### Investigated — CUDA IPC real zero-copy: not viable on this hardware
- Attempted real CUDA IPC for `CudaPortBus` using
  `torch.multiprocessing.reductions.reduce_tensor()` (the same mechanism
  torch itself uses for cross-process GPU tensor sharing). Confirmed NOT
  viable on this Jetson Orin Nano (JetPack 6): the receiving process's
  rebuild call fails with `torch.AcceleratorError: CUDA error: invalid
  argument` from `torch.UntypedStorage._new_shared_cuda`, consistently,
  regardless of explicit `torch.cuda.set_device()`/`init()` in the child.
  Consistent with Jetson's integrated-GPU (unified host/device memory)
  architecture not supporting the `cudaIpcGetMemHandle`/
  `cudaIpcOpenMemHandle` mechanism, which targets discrete GPUs with
  separate VRAM addressable over PCIe from multiple processes.
  `CudaPortBus` keeps its existing CPU-roundtrip behavior; its docstring
  now explains this honestly instead of implying real IPC. While
  investigating, also found and fixed a real, separate, more serious bug:
  whenever `using_cuda=True`, the bus slot was hardcoded to 64 bytes
  (sized for a CUDA IPC handle that was never actually used), silently
  truncating any tensor larger than that on write and always returning a
  shapeless/dtypeless flat uint8 buffer on read — i.e. `CudaPortBus` never
  correctly round-tripped any realistic-sized tensor before this fix. Now
  uses the full declared `slot_bytes` and carries dtype/shape through the
  bus's existing per-slot metadata channel.

### Added — tooling
- `scripts/build_wheel.sh` — builds a `cv-flow` wheel into `~/wheels` (or a
  given directory) for fully offline reuse: `pip install --no-index
  --find-links ~/wheels "cv-flow[gpu]"` from another project. Verified
  end-to-end: built, installed into an isolated location, imported successfully.

## [0.3.0] — 2026-06-30 — Real hardware validation on Jetson Orin Nano

Phase 1 of hardening CV-FLOW for production deployment on an NVIDIA Jetson
Orin Nano 8GB (ASUS PE1100N, JetPack 6 / L4T R36.4.4, CUDA 12.6). Every item
below was verified against real hardware, not just reviewed/mocked — see
`PROJECT_REFERENCE.md` and `CLAUDE.md` for the up-to-date architecture and
usage reference.

### Added
- `cv_flow.nodes.camera.build_nvargus_pipeline()` — generates a GStreamer
  `nvarguscamerasrc` pipeline string for Jetson CSI cameras.
- `CameraSource(gstreamer_pipeline=...)` — opens via `cv2.CAP_GSTREAMER`
  instead of a plain device index, for CSI camera support.
- `RtspSource` exponential reconnect backoff (`max_reconnect_delay_s`),
  resets to the base delay on a successful read, logs each reconnect.
- `OnnxInference`/`YoloInference(trt_cache_dir=...)` — enables ONNX Runtime's
  TensorRT engine disk cache so the (1-2 min) engine build only happens
  once instead of on every process start.
- `cv_flow.nodes.tee.Tee` — fans a single topic out to N independent output
  topics. Required for any pipeline where more than one downstream node
  needs the same upstream frame (e.g. detect-and-draw): every `.topic` bus
  in this DAM model is a single-reader FIFO, so two nodes subscribing to
  the same topic directly would compete for one queue instead of each
  seeing every frame. Discovered via the real end-to-end smoke test below.
- `NMS(output_layout=...)` (`"features_first"` / `"boxes_first"` / `"auto"`)
  — explicit raw-tensor layout instead of guessing from shape, which was
  silently wrong whenever box count < 84 (the old default/only behavior,
  now `output_layout="auto"`, logs a warning when used).
- `backend/app/db.py` + SQLite-backed `PipelineStore` — pipeline specs now
  survive a process restart (previously an in-memory dict, lost on restart).
  Public `PipelineStore` interface unchanged; API routers required no edits.
- `scripts/bench_inference.py` — manual CPU/CUDA/TensorRT latency benchmark.
- `scripts/smoke_pipeline.py` — manual real-camera → TensorRT inference →
  NMS → tracking → draw → video-file end-to-end hardware smoke test.
- `README.md`, `LICENSE` (Apache-2.0), this `CHANGELOG.md`, `CLAUDE.md`.
- `pyproject.toml`: `authors`/`license`/`readme`/`classifiers`/`urls`;
  split the `onnxruntime` dependency into `cpu`/`gpu` extras (the package
  names `onnxruntime` vs `onnxruntime-gpu` conflict, so both must never be
  declared as a single hard dependency); registered `gpu`/`multiprocess`/
  `hardware` pytest markers.

### Fixed
- `OnnxInference`/`YoloInference`: ONNX Runtime provider list now tries
  `TensorrtExecutionProvider` before `CUDAExecutionProvider` on
  `device="cuda:0"` (previously CUDA-only, missing a measured **13x**
  CPU→TensorRT speedup on this hardware — 117ms → 8.85ms per yolov8n frame
  at 640×640). Logs the actually-active provider and warns on silent CPU
  fallback.

### Verified on real Jetson Orin Nano hardware (not mocked)
- `cv2` 4.8.0 with GStreamer 1.20.3 + CUDA support (JetPack apt package,
  exposed to the project venv via `include-system-site-packages = true` —
  the generic PyPI `opencv-python` wheel lacks GStreamer entirely).
- `torch` 2.11.0 (`cuda.is_available() == True`, device "Orin"),
  `onnxruntime-gpu` 1.24.0 (`TensorrtExecutionProvider`,
  `CUDAExecutionProvider` both available) — both Jetson-specific builds,
  confirmed untouched by the `pip install -e . --no-deps` install flow.
  `numpy` pinned to `<2,>=1.26.0` (the JetPack `cv2` build is numpy-1.x ABI).
- Real YOLOv8n ONNX inference (model exported from `ultralytics`) on CPU,
  CUDA, and TensorRT execution providers — measured FPS: CPU 8.5,
  CUDA 67.6, TensorRT 113.0 (640×640 input).
- Real USB camera capture (Intel RealSense color stream, `/dev/video4`).
  CSI capture (`build_nvargus_pipeline`) and RTSP backoff are
  code-complete and unit-tested but **not yet verified against physical
  CSI/RTSP hardware** — flagged explicitly in `CLAUDE.md`.
- Full real pipeline, single process: `CameraSource` → `Tee` →
  `Preprocess` → `YoloInference` (TensorRT) → `NMS` → `ObjectTracker` →
  `DrawBbox` → `VideoWriter`, ~7.9 FPS end-to-end at 1280×720 input /
  640×640 inference on this board.
- SQLite `PipelineStore`: a record written by one instance is readable by
  a second instance pointed at the same db file (simulated process restart).

### Known gaps (unchanged from v0.2.0, deferred to Phase 2 / v0.4.0)
- `CudaPortBus` still round-trips through CPU RAM — no real CUDA IPC
  zero-copy yet.
- `Executor.scale_up()`/`scale_down()` are still empty hooks — no real
  multiprocessing-based elastic auto-scaling yet.

## [0.2.0] — 2026-06-30 — Rewrite as topic-based DAM pipeline framework

See `PROJECT_REFERENCE.md` §1 for the full original changelog (the
codebase was rewritten from scratch in this release: DAM shared-memory
buses, `.topic` files, `Node`/`Executor`, built-in nodes, FastAPI backend).
