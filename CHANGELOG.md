# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

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
