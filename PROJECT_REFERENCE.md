# CV-FLOW Project Reference

Generated 2026-06-25. Covers every source file in the repository.

---

## Overall Architecture

CV-FLOW is a three-tier computer-vision pipeline builder:

```
┌─────────────────────────────┐
│  Frontend (React/ReactFlow)  │  browser  – drag-drop node graph editor
└────────────┬────────────────┘
             │ REST (HTTP)          │ WebSocket ws://localhost:8765
┌────────────▼────────────────┐   ┌───────────────────────────┐
│  Backend (FastAPI / Python) │   │ Engine WebSocket Server    │
│  – pipeline CRUD            │◄──│  ws_server.py (background  │
│  – model registry           │   │  thread inside engine)     │
│  – execution lifecycle      │   └────────────┬──────────────┘
│  – C++ compiler service     │                │ push JPEG / events
└────────────┬────────────────┘   ┌────────────▼──────────────┐
             │ subprocess Popen   │  Engine Process             │
             └───────────────────►│  engine/main.py            │
                                  │  – PipelineRunner          │
                                  │  – ordered node graph      │
                                  └───────────────────────────┘
```

### How the pieces connect

1. The **Backend** (FastAPI, port 8000) stores pipeline JSON and model metadata in SQLite. On `POST /api/execution/start` it serialises the pipeline to a temp JSON file and spawns an **Engine** subprocess (`engine/main.py`).
2. The **Engine** subprocess loads the pipeline JSON, performs a topological sort (Kahn's algorithm) in `pipeline_builder.py`, instantiates node objects, and runs the blocking frame loop in `PipelineRunner.run()`. It also starts a WebSocket server (port 8765) in a background thread.
3. Every frame, the engine calls `node.process(ctx: FrameContext)` on each node in topological order. `FrameContext` is the data bus that carries the BGR frame, detections list, and a metadata dict.
4. The **Frontend** connects to port 8765 WebSocket channels to receive live JPEG frames and JSON events, and connects to port 8000 REST API for all CRUD operations.

### Global conventions

| Convention | Detail |
|---|---|
| **FrameContext flow** | Each `process()` receives and returns a `FrameContext`. Nodes mutate it in-place and return it. Raising `StopIteration` signals source exhaustion and stops the pipeline. |
| **`@register` decorator** | `engine/core/node_registry.py` maps string type keys to Python classes. Usage: `@register("my_type")`. Multiple keys can be registered to one class (e.g. `@register("postprocess_nms")` and `@register("nms")`). |
| **Port naming** | Most nodes use `in` / `out`. Split-output nodes use `frame` + `raw` (model_inference) or `frame` + `dets` (nms, object_tracker, filter). Multi-input nodes accept `frame` / `dets` / `raw` by name. These IDs are defined in `nodes.js / NODE_PORTS`. |
| **CVFLOW_MODELS_DIR / CVFLOW_COMPILED_DIR** | The backend sets these env vars when launching the engine subprocess so engine nodes can locate models and compiled .so files without importing backend code. |
| **Config dict** | Each node's `config` dict comes from the pipeline JSON `nodes[i].config`. Defaults are documented per-node below. |
| **Metadata bus** | `ctx.metadata` is a plain dict. Nodes write results here (`preprocessed_tensor`, `model_output`, `model_config`, `line_<id>`, `zone_<id>_count`, `counter_<id>`, etc.). |
| **Hot-reload** | On POSIX: `SIGUSR1` → `PipelineRunner.request_reload()` → `ModelInferenceNode.reload_model()`. On Windows the signal is skipped; use the API endpoint instead. |

---

## Backend

### `backend/app/main.py`

Application entry point for the FastAPI server.

**Purpose:** Creates the FastAPI app, mounts CORS middleware, serves the frontend static files, and runs DB initialisation on startup.

| Symbol | Description |
|---|---|
| `NoCacheStaticFiles(StaticFiles)` | Subclass of Starlette `StaticFiles` that adds `Cache-Control: no-cache` headers to every response, ensuring the browser always fetches the latest frontend JS. |
| `lifespan(app)` | `@asynccontextmanager` startup handler. Creates `storage/models`, `storage/compiled`, and `storage/tmp` directories, then calls `create_tables()`. |
| `app` | The FastAPI application instance. Title `CV-FLOW API`, version `0.1.0`. |
| `health()` | `GET /health` — returns `{"status": "ok"}`. |
| `serve_ui()` | `GET /` — returns `frontend/static/index.html` if it exists. |
| Static mount | Mounts `frontend/static/` under `/` using `NoCacheStaticFiles`. |

---

### `backend/app/config.py`

**Purpose:** Application-wide settings loaded from environment variables / `.env` file using `pydantic-settings`.

**Class: `Settings(BaseSettings)`**

| Field / Property | Type | Default | Description |
|---|---|---|---|
| `database_url` | `str` | `sqlite+aiosqlite:///./cv_flow.db` | SQLAlchemy async DSN |
| `storage_path` | `Path` | `./storage` | Root directory for all file storage |
| `engine_ws_port` | `int` | `8765` | Port for the engine WebSocket server |
| `engine_python` | `str` | `python` | Python executable used to spawn the engine subprocess |
| `cors_origins` | `str` | `http://localhost:5173` | Comma-separated list of allowed CORS origins |
| `cors_origins_list` | `list[str]` | *(property)* | Splits `cors_origins` on comma |
| `models_dir` | `Path` | *(property)* | `storage_path / "models"` |
| `compiled_dir` | `Path` | *(property)* | `storage_path / "compiled"` |
| `pipelines_tmp_dir` | `Path` | *(property)* | `storage_path / "tmp"` — pipeline JSON and engine log files live here |

Singleton instance exported as `settings`.

---

### `backend/app/database.py`

**Purpose:** Async SQLAlchemy engine, session factory, ORM base class, and DB-creation helper.

| Symbol | Description |
|---|---|
| `engine` | `create_async_engine(settings.database_url)` — SQLite async engine |
| `AsyncSessionLocal` | `async_sessionmaker` producing `AsyncSession` instances with `expire_on_commit=False` |
| `Base` | `DeclarativeBase` subclass — all ORM models inherit from this |
| `get_db()` | Async generator dependency; yields a session and closes it after the request |
| `create_tables()` | Runs `Base.metadata.create_all` inside an async connection; called from `lifespan` |

---

### `backend/app/api/router.py`

**Purpose:** Aggregates all sub-routers into a single `APIRouter` with prefix `/api`.

Includes: `pipelines.router`, `models.router`, `execution.router`, `compiler.router`, `system.router`.

---

### `backend/app/api/compiler.py`

**Purpose:** REST endpoints for compiling C++ node source code to a shared library.

Router prefix: `/api/compile`, tag: `compiler`.

| Endpoint | Description |
|---|---|
| `POST /api/compile` | Accepts `CompileRequest`, calls `compiler_service.compile_node()`, returns `CompileResponse` with status, so_hash, stderr, timestamp |
| `GET /api/compile/sdk` | Streams a ZIP archive of the C++ SDK headers and CMakeLists.txt template from `shared/cpp/` |

---

### `backend/app/api/execution.py`

**Purpose:** REST endpoints for starting, stopping, and monitoring pipeline execution sessions. Also provides Server-Sent Events (SSE) for log streaming.

Router prefix: `/api/execution`, tag: `execution`.

| Endpoint | Description |
|---|---|
| `POST /api/execution/start` | Looks up pipeline from DB, calls `execution_service.start_session()`, returns `{session_id}` |
| `POST /api/execution/stop/{session_id}` | Terminates engine subprocess via `execution_service.stop_session()` |
| `GET /api/execution/status/{session_id}` | Returns `SessionStatusResponse` with status string (`running / stopped / error / completed`) |
| `GET /api/execution/sessions` | Lists all active session IDs |
| `GET /api/execution/logs/{session_id}` | Returns last `tail` (default 200) log lines as a JSON array |
| `GET /api/execution/logs/{session_id}/stream` | **SSE endpoint** — streams engine stdout/stderr log file in real time. Waits up to 5 s for the log file to appear. Polls for new content every 0.3 s. Sends `event: done` sentinel when the engine exits. Media type: `text/event-stream`. |

---

### `backend/app/api/models.py`

**Purpose:** REST endpoints for the model registry — upload, list, download, delete, hot-reload, and one-click YOLO download.

Router prefix: `/api/models`, tag: `models`.

| Endpoint | Description |
|---|---|
| `GET /api/models` | List all models (name, version, task, uploaded_at) |
| `GET /api/models/{model_id}` | Full model details including config JSON |
| `POST /api/models/upload` | Multipart upload of `model_file` (.onnx) and `config_file` (.json). Required config fields: `name, version, task, format:"onnx", input_shape, output_shapes`. Saves to `storage/models/{uuid}/` and registers in DB. |
| `DELETE /api/models/{model_id}` | Removes model files and DB entry |
| `POST /api/models/{model_id}/reload` | Sends `SIGUSR1` to all running engine processes to trigger model hot-reload |
| `GET /api/models/{model_id}/download` | Streams the `.onnx` file as a file response |
| `GET /api/models/defaults/list` | Returns the full `MODEL_CATALOG` from `yolo_downloader.py` |
| `POST /api/models/defaults/download/{model_key}` | Downloads a YOLO model via `ultralytics`, exports to ONNX, and registers it. Runs synchronously in an executor. |

---

### `backend/app/api/pipelines.py`

**Purpose:** CRUD endpoints for pipeline persistence.

Router prefix: `/api/pipelines`, tag: `pipelines`.

| Endpoint | Description |
|---|---|
| `GET /api/pipelines` | List all pipelines (id, name, description, created_at, updated_at) ordered by updated_at desc |
| `POST /api/pipelines` | Create a pipeline from `PipelineCreate`. Auto-generates UUID if `id` not provided. Stores full graph as JSON in `config_json`. |
| `GET /api/pipelines/{pipeline_id}` | Retrieve full pipeline (nodes + edges) |
| `PUT /api/pipelines/{pipeline_id}` | Update name, description, and graph |
| `DELETE /api/pipelines/{pipeline_id}` | Delete pipeline |
| `POST /api/pipelines/validate` | Validates edges reference existing node IDs. Returns `{valid, errors[]}`. |

---

### `backend/app/api/system.py`

**Purpose:** System utilities — currently exposes a `pip install` endpoint for installing Python packages into the server's environment.

Router prefix: `/api/system`, tag: `system`.

| Symbol | Description |
|---|---|
| `PipInstallRequest` | Pydantic model with a single `command: str` field (accepts `"pip install X"` or just `"X Y"`) |
| `_parse_packages(raw)` | Strips leading `pip install` tokens and returns a list of package names / flags |
| `_stream_pip(packages)` | Generator that runs `pip install` in a subprocess and yields SSE-formatted bytes for each stdout line |
| `POST /api/system/pip-install` | Accepts `PipInstallRequest`, streams pip output as SSE (`text/event-stream`). Sends `event: done data: ok` or `data: error` when complete. |

---

### `backend/app/models/pipeline.py`

**Purpose:** SQLAlchemy ORM model for the `pipelines` table.

**Class: `Pipeline(Base)`**

| Column | Type | Description |
|---|---|---|
| `id` | `String(36)` PK | UUID string |
| `name` | `String(128)` | Pipeline display name |
| `description` | `Text` nullable | Optional description |
| `config_json` | `Text` | Full graph JSON (nodes + edges serialised) |
| `created_at` | `DateTime(timezone=True)` | Auto-set to UTC now |
| `updated_at` | `DateTime(timezone=True)` | Auto-updated on every write |

---

### `backend/app/models/compiled_node.py`

**Purpose:** SQLAlchemy ORM model for caching compiled C++ node shared libraries.

**Class: `CompiledNode(Base)`** — table `compiled_nodes`

| Column | Type | Description |
|---|---|---|
| `source_hash` | `String(64)` PK | SHA-256 hex of `source_code + compile_flags` |
| `so_path` | `Text` | Absolute path to the compiled `.so` file |
| `compile_flags` | `Text` nullable | JSON array string of compiler flags |
| `stderr_output` | `Text` nullable | Compiler stderr/stdout output |
| `status` | `String(16)` | `"ok"` or `"error"` |
| `compiled_at` | `DateTime(timezone=True)` | Compilation timestamp |

---

### `backend/app/models/model_registry.py`

**Purpose:** SQLAlchemy ORM model for the ONNX model registry.

**Class: `ModelEntry(Base)`** — table `model_registry`

Unique constraint on `(name, version)`.

| Column | Type | Description |
|---|---|---|
| `id` | `String(36)` PK | UUID |
| `name` | `String(128)` | Model name |
| `version` | `String(32)` | Version string |
| `task` | `String(32)` | `detection`, `segmentation`, `pose`, `classification` |
| `file_path` | `Text` | Absolute path to `model.onnx` |
| `config_json` | `Text` | Full config JSON including class names and I/O shapes |
| `uploaded_at` | `DateTime(timezone=True)` | Upload timestamp |

---

### `backend/app/schemas/pipeline.py`

**Purpose:** Pydantic schemas for pipeline API request/response validation.

| Schema | Fields | Notes |
|---|---|---|
| `PipelineCreate` | `version, id, name, description, nodes: list[dict], edges: list[dict]` | `id` auto-generates UUID; used for both create and update |
| `PipelineUpdate` | `name?, description?, nodes?, edges?` | Partial update (not currently wired to a PATCH endpoint) |
| `PipelineResponse` | `id, version, name, description, nodes, edges, created_at, updated_at` | Returned from GET/POST/PUT |
| `PipelineListItem` | `id, name, description, created_at, updated_at` | Returned from list |
| `ValidateResponse` | `valid: bool, errors: list[str]` | Returned from /validate |

---

### `backend/app/schemas/compiler.py`

| Schema | Fields | Notes |
|---|---|---|
| `CompileRequest` | `node_id: str, source_code: str, compile_flags: list[str] = ["-O2", "-march=native"], extra_libs: list[str] = []` | Sent to `POST /api/compile` |
| `CompileResponse` | `status: str, so_hash: str?, stderr_output: str?, compiled_at: datetime?` | Returned from compile endpoint; `status` is `"ok"` or `"error"` |

---

### `backend/app/schemas/execution.py`

| Schema | Fields | Notes |
|---|---|---|
| `ExecutionStartRequest` | `pipeline_id: str, params_override: dict? = None` | Params in the dict override matching keys in any node's config |
| `ExecutionStartResponse` | `session_id: str` | UUID of the new engine session |
| `SessionStatusResponse` | `session_id, pipeline_id, status, started_at, stopped_at?, error_msg?` | Status values: `running / stopped / error / completed` |

---

### `backend/app/schemas/model_registry.py`

| Schema | Fields |
|---|---|
| `ModelResponse` | `id, name, version, task, config: dict, uploaded_at` |
| `ModelListItem` | `id, name, version, task, uploaded_at` |

---

### `backend/app/services/compiler_service.py`

**Purpose:** On-the-fly C++ node compilation service. Compiles user source code to a `.so` shared library using CMake and caches the result by source hash.

| Symbol | Description |
|---|---|
| `_SDK_INCLUDE` | Path to `shared/cpp/include/` — SDK headers copied to each build |
| `_CMAKE_TEMPLATE` | Path to `shared/cpp/CMakeLists.txt.template` |
| `_source_hash(source_code, compile_flags)` | Returns SHA-256 hex of `source_code + "||" + sorted_flags_json`. Used as cache key. |
| `compile_node(source_code, compile_flags?, extra_libs?)` | Main compilation function. Checks the hash-keyed cache at `storage/compiled/{hash}/node.so`. On cache miss: writes source to a temp dir, fills the CMake template, copies SDK headers, runs `cmake -B build` then `cmake --build build --parallel` (with 60 s configure / 120 s build timeouts). Copies the resulting `node.so` to the cache. Returns dict with `status, so_hash, so_path, stderr_output, compiled_at`. |

---

### `backend/app/services/execution_service.py`

**Purpose:** Manages the lifecycle of engine subprocesses (one per session). Stores process handles and metadata in module-level dicts.

| Symbol | Description |
|---|---|
| `_sessions` | `dict[str, subprocess.Popen]` — session_id → process |
| `_meta` | `dict[str, dict]` — session_id → metadata dict with `pipeline_id`, `started_at`, `log_path` |
| `get_running_sessions()` | Returns `_sessions` |
| `_kill_all_running()` | Terminates all running sessions (used before starting a new one because only one engine can own the WS port) |
| `start_session(session_id, pipeline_json, params_override?)` | Kills any existing sessions, writes pipeline JSON to `tmp/{session_id}.json`, opens a line-buffered log file at `tmp/{session_id}.log`, launches `engine/main.py` via `subprocess.Popen`. Engine receives env vars `CVFLOW_MODELS_DIR` and `CVFLOW_COMPILED_DIR`. Returns the `Popen` handle. |
| `stop_session(session_id)` | Calls `proc.terminate()`, waits up to 5 s, force-kills if needed. Returns `True` if the session existed. |
| `session_status(session_id)` | Returns `"running"` / `"error"` / `"completed"` / `"stopped"` based on `proc.poll()`. |
| `session_meta(session_id)` | Returns the metadata dict for a session, or `{}` if not found. |
| `iter_logs(session_id, tail=200)` | Reads the log file and yields the last `tail` lines. |

---

## Engine Core

### `engine/main.py`

**Purpose:** Entry point for the engine subprocess. Parses CLI arguments, starts the WebSocket server, builds the pipeline, installs the hot-reload handler, and runs the frame loop.

CLI arguments:
- `--pipeline-json` (required) — path to pipeline JSON file
- `--session-id` (required) — session UUID passed from the backend
- `--ws-port` (default 8765) — port for the WebSocket server
- `--params-override` (default `"{}"`) — JSON dict; matching keys in any node's config are overridden

| Function | Description |
|---|---|
| `main()` | Parses args, loads pipeline JSON, applies params_override (adds to any node config that already has the key), starts the WS server, calls `build_pipeline()` and `PipelineRunner.run()`, installs `install_hot_reload(runner)`. |

---

### `engine/core/frame_context.py`

**Purpose:** Defines `FrameContext` (the per-frame data bus), `Detection` (a single detected object), and the ctypes structs that mirror the C++ `CVFlowCtx` layout for the C++ bridge.

**Class: `Detection`** (dataclass)

| Field | Type | Description |
|---|---|---|
| `x1, y1, x2, y2` | `float` | Bounding box in pixel coordinates (absolute, matching frame dimensions) |
| `confidence` | `float` | Detection score 0–1 |
| `class_id` | `int` | Integer class index |
| `class_name` | `str` | String class label |
| `track_id` | `int` | `-1` if not tracked |
| `metadata` | `dict` | Per-detection metadata |
| `bbox_xyxy` | property | Returns `(x1, y1, x2, y2)` tuple |
| `center` | property | Returns `((x1+x2)/2, (y1+y2)/2)` |
| `area` | property | Returns bounding box area |

**Class: `FrameContext`** (dataclass)

| Field | Type | Description |
|---|---|---|
| `frame` | `np.ndarray | None` | CPU BGR uint8 frame. `None` if data is on GPU. |
| `frame_number` | `int` | Zero-based frame counter |
| `timestamp` | `float` | Unix timestamp of frame capture |
| `session_id` | `str` | Session UUID (set by PipelineRunner) |
| `detections` | `list[Detection]` | Detection results; populated by postprocess nodes |
| `metadata` | `dict[str, Any]` | Key-value bus for inter-node communication |
| `frame_gpu` | `Any` | `cv2.cuda.GpuMat` or `torch.Tensor`; `None` for CPU pipelines |
| `_gpu_valid` | `bool` | True when the GPU copy is the authoritative frame |
| `on_gpu` | property | `True` when `_gpu_valid and frame_gpu is not None` |
| `ensure_cpu()` | method | Returns `self.frame`; if None, downloads from GPU (via `.download()` for cv2 or `.cpu()` for PyTorch) and caches. Raises `RuntimeError` if no frame data. |
| `set_frame_gpu(gpu_mat, *, invalidate_cpu=True)` | method | Stores a GPU frame. If `invalidate_cpu=True` sets `frame=None`. |
| `copy_frame()` | method | Returns `ensure_cpu().copy()` |

**ctypes structs (C++ bridge):**

- `CVFlowDetectionC` — ctypes.Structure matching C++ `CVFlowDetection`. Fields: `x1, y1, x2, y2` (float), `confidence` (float), `class_id` (int), `class_name` (char[64]), `track_id` (int), `metadata_json` (char[256]).
- `CVFlowCtxC` — ctypes.Structure matching C++ `CVFlowCtx`. Fields: `frame_data` (ptr to uint8), `width, height, channels` (int), `frame_number` (int), `timestamp` (double), `session_id` (char[64]), `detections` (ptr to CVFlowDetectionC), `detection_count` (int), `detection_capacity` (int), `metadata_json` (char[4096]).

Constants: `CVFLOW_CLASS_NAME_LEN=64`, `CVFLOW_DET_META_LEN=256`, `CVFLOW_SESSION_ID_LEN=64`, `CVFLOW_METADATA_LEN=4096`.

---

### `engine/core/node_registry.py`

**Purpose:** Maps string node-type keys to Python `BaseNode` subclasses via a registry dict. Imports all node modules at module load time to trigger `@register` decorations.

| Symbol | Description |
|---|---|
| `_registry` | `dict[str, type[BaseNode]]` — private module-level registry |
| `register(node_type)` | Decorator factory: `@register("my_type")` adds the class to `_registry`. |
| `get_node_class(node_type)` | Looks up `node_type` in `_registry`. Raises `ValueError` if not found (error message includes the list of registered types). |
| `_import_all()` | Imports every node submodule so their `@register` calls execute. Called at module bottom. |

---

### `engine/core/pipeline_builder.py`

**Purpose:** Parses pipeline JSON and returns an ordered list of `BaseNode` instances in topological execution order.

| Function | Description |
|---|---|
| `build_pipeline(pipeline_json)` | Accepts the full pipeline dict (`{nodes, edges}`). Builds an adjacency map and in-degree map. Runs Kahn's topological sort (BFS). Raises `ValueError("Pipeline graph contains a cycle")` if the graph is cyclic. Instantiates each node in order via `get_node_class(type)()` then `instance.setup(node_id, config)`. Returns `list[BaseNode]`. |

---

### `engine/core/pipeline_runner.py`

**Purpose:** Blocking frame loop — executes nodes in topological order, manages live previews, auto-streaming, hot-reload, and teardown.

**Class: `PipelineRunner`**

| Method | Description |
|---|---|
| `__init__(nodes, session_id)` | Stores node list and session ID. Initialises stop/reload threading events. Detects whether a `StreamViewerNode` is present (enables auto-stream fallback if not). |
| `request_stop()` | Sets `_stop_event`. |
| `request_reload()` | Sets `_reload_flag` (called from SIGUSR1 handler). |
| `_try_send_node_preview(node, ctx)` | If the frontend is watching this `node.node_id` (via `ws_server.is_node_watched()`), encodes the current frame as JPEG (quality 65) with detections overlaid, and calls `ws_server.send_node_preview()`. Throttled to `_NODE_PREVIEW_FPS=10` fps. |
| `_try_auto_stream(ctx)` | When no `StreamViewerNode` is present, encodes the final frame as JPEG (quality 75, max 15 fps) and calls `ws_server.send_frame()`. |
| `run()` | Blocking loop. Creates a fresh `FrameContext` for each frame, calls each node's `process(ctx)` in order. If a node raises `StopIteration` the pipeline stops cleanly. Catches and logs other exceptions but continues. Calls `teardown()` on all nodes in reverse order in the `finally` block. Calls `ws_server.cleanup_session()`. |
| `_do_reload()` | Finds all `ModelInferenceNode` instances and calls `reload_model()` on each. |

Constants: `_NODE_PREVIEW_FPS=10`, `_NODE_PREVIEW_QUALITY=65`, `_AUTO_STREAM_FPS=15`, `_AUTO_STREAM_QUALITY=75`.

---

## Engine Nodes

### `engine/nodes/base.py`

**Purpose:** Abstract base class for all engine nodes.

**Class: `BaseNode(ABC)`**

| Method | Description |
|---|---|
| `setup(node_id, config)` | Sets `self.node_id` and `self.config`, then calls `self.initialize()`. Called by `pipeline_builder`. |
| `initialize()` | Override to open files, load weights, initialise state. Default: no-op. |
| `process(ctx: FrameContext) -> FrameContext` | **Abstract.** Must be implemented by every node. Modify `ctx` in-place and return it. |
| `teardown()` | Override to release resources. Called in reverse order when the pipeline stops. |

---

### `engine/nodes/input/camera.py`

**Purpose:** Generic camera source node. Handles USB and RTSP/HTTP sources with auto-reconnect.

Registered as: `"camera"`

| Config key | Default | Description |
|---|---|---|
| `source_type` | `"usb"` | `"usb"`, `"rtsp"`, or `"http"` |
| `device_index` | `0` | Integer device index for USB |
| `url` | `""` | URL for RTSP/HTTP sources |
| `fps_limit` | `0` | Maximum FPS (0 = unlimited) |
| `reconnect_delay_s` | `3.0` | Seconds to wait before reconnect attempt |

Key methods:
- `_open_cap()` — Opens `cv2.VideoCapture`. On Windows falls back to `CAP_DSHOW` if the default backend fails. Discards first 3 frames for sensor warm-up.
- `process()` — Reads a frame; if read fails, sleeps `reconnect_delay_s`, re-opens the source, and retries. Raises `StopIteration` if reconnect fails.

---

### `engine/nodes/input/usb_camera.py`

**Purpose:** Convenience subclass of `CameraNode` that forces `source_type="usb"` and optionally sets frame resolution.

Registered as: `"usb_camera"`

Extra config keys: `width` (0 = camera default), `height` (0 = camera default).

---

### `engine/nodes/input/video_file.py`

**Purpose:** Reads frames from a local video file with optional looping and FPS limiting.

Registered as: `"video_file"`

| Config key | Default | Description |
|---|---|---|
| `file_path` | `""` | Path to video file |
| `loop` | `False` | Seek to frame 0 when file ends instead of stopping |
| `fps_limit` | `0` | Override FPS (0 = native video FPS) |

Raises `StopIteration("Video file exhausted")` when file ends and `loop=False`.

---

### `engine/nodes/input/image_directory.py`

**Purpose:** Iterates through images in a directory matching a glob pattern.

Registered as: `"image_directory"`

| Config key | Default | Description |
|---|---|---|
| `directory_path` | `""` | Directory to scan |
| `pattern` | `"*.jpg"` | Glob pattern |
| `delay_ms` | `100` | Milliseconds to sleep between frames |

Raises `StopIteration` when all images have been processed.

---

### `engine/nodes/input/rtsp_stream.py`

**Purpose:** Subclass of `CameraNode` specialised for RTSP streams with reconnect logic.

Registered as: `"rtsp_stream"`

Config: `url` (RTSP URL), `fps_limit`, `reconnect_delay_s`. On frame read failure: sleeps, releases, re-opens via `cv2.VideoCapture(url)`. Raises `StopIteration("RTSP stream lost")` if still failing.

---

### `engine/nodes/processing/preprocess.py`

**Purpose:** Image preprocessing — crop, resize, and normalisation. Puts a float32 CHW tensor into `ctx.metadata["preprocessed_tensor"]` for the inference node.

Registered as: `"preprocess"`

| Config key | Default | Description |
|---|---|---|
| `crop` | `None` | Dict `{x, y, width, height}` for ROI crop before resize |
| `resize_w` | `0` | Target width (0 = no resize) |
| `resize_h` | `0` | Target height (0 = no resize) |
| `resize` | `{}` | Legacy nested dict `{width, height, keep_aspect}` |
| `normalize` | `"none"` | `"none"`, `"0_1"`, `"imagenet"`, or `"custom"` |
| `mean` | `[0,0,0]` | Used when `normalize="custom"` |
| `std` | `[1,1,1]` | Used when `normalize="custom"` |

Side-effect: On normalisation, writes `ctx.metadata["preprocessed_tensor"]` as a `[1, C, H, W]` float32 numpy array (CHW, batch dim=1). `ctx.frame` remains uint8 so downstream draw nodes can use it.

---

### `engine/nodes/processing/model_inference.py`

**Purpose:** Runs an ONNX model via `onnxruntime`. Reads the model from `storage/models/{model_id}/`.

Registered as: `"model_inference"`

| Config key | Default | Description |
|---|---|---|
| `model_id` | `""` | UUID of the model in the registry |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `conf_threshold` | `0.5` | Passed to metadata for downstream NMS node |

Key methods:
- `_load_model()` — Resolves model path from `CVFLOW_MODELS_DIR` env var or default relative path. Reads `config.json`. Creates `ort.InferenceSession` with CPU or CUDA providers.
- `reload_model()` — Re-runs `_load_model()` in-place (used by hot-reload).
- `process(ctx)` — Uses `ctx.metadata.get("preprocessed_tensor")`. If not present, auto-resizes frame to model input dimensions. Runs inference. Writes `ctx.metadata["model_output"]`, `ctx.metadata["model_config"]`, and `ctx.metadata["model_conf_threshold"]`.

---

### `engine/nodes/processing/postprocess_nms.py`

**Purpose:** Decodes raw ONNX model output into `Detection` objects and populates `ctx.detections`. Supports three output formats automatically.

Registered as: `"postprocess_nms"` and `"nms"`

| Config key | Default | Description |
|---|---|---|
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `conf_threshold` | `0.25` | Minimum confidence to keep a detection |
| `max_detections` | `300` | Maximum detections per frame |

Format detection logic (`_detect_format`):
- 2+ outputs → `"rtdetr"` (RT-DETR: boxes + scores)
- Single output, last dim 6 → `"yolov10"` (already decoded x1,y1,x2,y2,score,cls)
- Otherwise → `"yolov8"` (YOLOv8/v9/v11: [4+num_cls, num_anchors])

Parsers:
- `_parse_yolov8` — Transposes to [N, 4+C], argmax class scores, applies `cv2.dnn.NMSBoxes`, scales from input resolution to frame resolution.
- `_parse_yolov10` — Reads pre-decoded [max_det, 6] rows; no NMS needed (model is NMS-free); scales coordinates.
- `_parse_rtdetr` — Reads normalised cx/cy/w/h boxes and class logits; applies sigmoid if needed; scales to frame pixels.

---

### `engine/nodes/processing/draw_bbox.py`

**Purpose:** Draws bounding boxes and labels for all detections onto the frame.

Registered as: `"draw_bbox"`

| Config key | Default | Description |
|---|---|---|
| `thickness` | `2` | Rectangle line thickness |
| `show_label` | `True` | Draw class name |
| `show_confidence` | `True` | Draw score |
| `show_track_id` | `True` | Draw `#track_id` if `track_id >= 0` |
| `font_scale` | `0.45` | `cv2.FONT_HERSHEY_SIMPLEX` font scale |

Uses a 10-colour palette `_PALETTE` indexed by `class_id % 10`. Converts grayscale frames to BGR before drawing.

---

### `engine/nodes/processing/blur.py`

**Purpose:** Applies Gaussian, box, or median blur to the frame.

Registered as: `"blur"`

| Config key | Default | Options |
|---|---|---|
| `type` | `"gaussian"` | `"gaussian"`, `"box"`, `"median"` |
| `kernel_size` | `5` | Kernel size (forced to odd) |
| `sigma` | `0` | Gaussian sigma (0 = auto) |

---

### `engine/nodes/processing/edge_detect.py`

**Purpose:** Detects edges using Canny, Sobel, or Laplacian algorithms. Converts to grayscale internally; outputs BGR.

Registered as: `"edge_detect"`

| Config key | Default | Options |
|---|---|---|
| `algorithm` | `"canny"` | `"canny"`, `"sobel"`, `"laplacian"` |
| `threshold1` | `50` | Canny low threshold |
| `threshold2` | `150` | Canny high threshold |

---

### `engine/nodes/processing/corner_detect.py`

**Purpose:** Detects corners using Harris, FAST, or Shi-Tomasi algorithms. Draws detected corners as red dots/keypoints.

Registered as: `"corner_detect"`

| Config key | Default | Options |
|---|---|---|
| `algorithm` | `"harris"` | `"harris"`, `"fast"`, `"shitomasi"` |
| `block_size` | `2` | Harris block size |
| `k` | `0.04` | Harris k parameter |
| `quality` | `0.01` | Quality threshold (Harris = fraction of max response; Shi-Tomasi = min quality) |
| `max_corners` | `100` | Max corners returned (FAST, Shi-Tomasi) |
| `min_dist` | `10.0` | Min distance between corners (Shi-Tomasi) |

---

### `engine/nodes/processing/threshold.py`

**Purpose:** Applies binary, inverse binary, Otsu, or adaptive threshold to the frame. Converts to grayscale then back to BGR.

Registered as: `"threshold"`

| Config key | Default | Options |
|---|---|---|
| `type` | `"binary"` | `"binary"`, `"binary_inv"`, `"otsu"`, `"adaptive"` |
| `threshold` | `127` | Threshold value (not used by `otsu` or `adaptive`) |
| `max_val` | `255` | Maximum output value |

---

### `engine/nodes/processing/color_convert.py`

**Purpose:** Converts the frame between colour spaces.

Registered as: `"color_convert"`

| Config key | Default |
|---|---|
| `conversion` | `"bgr2gray"` |

Supported conversions: `bgr2gray`, `bgr2hsv`, `bgr2rgb`, `bgr2lab`, `bgr2yuv`, `gray2bgr`, `hsv2bgr`. Grayscale outputs are expanded back to 3-channel BGR.

---

### `engine/nodes/processing/morph.py`

**Purpose:** Applies morphological operations (erode, dilate, open, close, gradient, tophat, blackhat).

Registered as: `"morph"`

| Config key | Default | Options |
|---|---|---|
| `operation` | `"erode"` | `"erode"`, `"dilate"`, `"open"`, `"close"`, `"gradient"`, `"tophat"`, `"blackhat"` |
| `kernel_size` | `3` | Square kernel side length |
| `iterations` | `1` | Number of iterations |

---

### `engine/nodes/processing/resize.py`

**Purpose:** Resizes the frame to a target resolution.

Registered as: `"resize"`

| Config key | Default | Options |
|---|---|---|
| `width` | `640` | Target width in pixels |
| `height` | `480` | Target height in pixels |
| `interpolation` | `"area"` | `"nearest"`, `"linear"`, `"cubic"`, `"area"`, `"lanczos"` |

---

### `engine/nodes/processing/affine_transform.py`

**Purpose:** Applies a rotation + translation + scale affine transform to the frame.

Registered as: `"affine_transform"`

| Config key | Default | Description |
|---|---|---|
| `tx` | `0` | X translation in pixels |
| `ty` | `0` | Y translation in pixels |
| `angle` | `0` | Rotation angle in degrees (counter-clockwise) |
| `scale` | `1.0` | Uniform scale factor |

Rotation centre is the frame centre. Uses `cv2.BORDER_REPLICATE` to fill border pixels.

---

### `engine/nodes/spatial/draw_roi.py`

**Purpose:** Draws a semi-transparent polygon ROI on the frame. Optionally filters detections to only those whose centre is inside the polygon.

Registered as: `"draw_roi"`

| Config key | Default | Description |
|---|---|---|
| `zone_id` | `"zone_1"` | ID string stored in `ctx.metadata["zone_{zone_id}_count"]` |
| `color` | `[0, 255, 0]` | BGR colour for the polygon outline |
| `draw_on_frame` | `True` | Draw the polygon on the frame |
| `filter_outside` | `True` | Remove detections whose centre is outside the polygon |
| `polygon` | `[]` | List of `[x_pct, y_pct]` points (0–100 percent of frame dimensions) |

Side-effects: Writes `ctx.metadata[f"zone_{zone_id}_count"]` = count of detections inside the zone.

---

### `engine/nodes/spatial/draw_line.py`

**Purpose:** Draws a directed trip line on the frame and stores its geometry in metadata for the CounterNode.

Registered as: `"draw_line"`

| Config key | Default | Description |
|---|---|---|
| `line_id` | `"line_1"` | ID used as metadata key `line_{line_id}` |
| `color` | `[0, 0, 255]` | BGR colour |
| `direction` | `"both"` | `"both"`, `"up"`, `"down"` — stored in metadata for CounterNode |
| `line` | `[[10,50],[90,50]]` | Two `[x_pct, y_pct]` endpoints (0–100 percent of frame) |

Side-effects: Writes `ctx.metadata[f"line_{line_id}"] = {p0, p1, direction}`. Draws an arrow circle at the line midpoint. Has a second output port `line_ref` (purely visual; same FrameContext is passed) to explicitly connect to a `counter` node's `line_ref` input, making the dependency visible in the graph.

---

### `engine/nodes/spatial/object_tracker.py`

**Purpose:** Assigns persistent track IDs to detections across frames using DeepSORT or ByteTrack. Writes `track_id` back into each `Detection` object.

Registered as: `"object_tracker"`

| Config key | Default | Options |
|---|---|---|
| `algorithm` | `"bytetrack"` | `"bytetrack"`, `"deepsort"` |
| `max_age` | `30` | Max frames a track can be lost before deletion |
| `iou_threshold` | `0.3` | IOU threshold for matching |

Requires `pip install bytetracker` or `pip install deep-sort-realtime`. If the library is not installed, a warning is logged and `track_id` remains `-1`.

Helper function `_iou(a, b)` — computes intersection-over-union between two `[x1,y1,x2,y2]` boxes.

---

### `engine/nodes/spatial/counter.py`

**Purpose:** Counts object line crossings or zone entry/exit events. Requires tracked detections (track_id >= 0).

Registered as: `"counter"`

| Config key | Default | Options |
|---|---|---|
| `trigger_type` | `"line_cross"` | `"line_cross"`, `"zone_enter"`, `"zone_exit"` |
| `trigger_id` | `"line_1"` | ID of the line or zone to monitor |
| `label` | `"Count"` | Display label shown in the frame overlay |
| `show_overlay` | `True` | Draw count text directly onto the frame |
| `count_classes` | `[]` | List of class names to count (empty = all classes) |

Side-effects: Writes `ctx.metadata[f"counter_{self.node_id}"]` = cumulative count. Draws an overlay text on the frame if `show_overlay=True`.

Mechanism (line_cross): Reads `ctx.metadata[f"line_{trigger_id}"]` from `DrawLineNode`. Tracks the cross-product sign of each detection's centre relative to the line. A sign change means the object crossed the line.

---

### `engine/nodes/output/stream_viewer.py`

**Purpose:** Encodes the current frame as a JPEG and pushes it to the WebSocket stream channel. The primary output node for browser previewing.

Registered as: `"stream_viewer"`

| Config key | Default | Description |
|---|---|---|
| `jpeg_quality` | `80` | JPEG encode quality 1–100 |
| `draw_detections` | `True` | Overlay bounding boxes before encoding |

Functions:
- `draw_detections(frame, detections)` — module-level helper; draws boxes with the same 10-colour `_COLORS` palette.
- `process()` — Calls `ws_server.send_frame(session_id, jpeg_bytes)` and `ws_server.send_event()` with a `"frame"` type event including `detection_count` and `frame_number`.

---

### `engine/nodes/output/video_writer.py`

**Purpose:** Writes frames to an MP4 (or other codec) video file. Lazily initialises the writer on the first frame to pick up frame dimensions.

Registered as: `"video_writer"`

| Config key | Default | Description |
|---|---|---|
| `output_path` | `"./output.mp4"` | Output file path |
| `codec` | `"mp4v"` | FourCC codec string |
| `fps` | `30` | Output video FPS |

`teardown()` releases the `cv2.VideoWriter`.

---

### `engine/nodes/output/trigger_webhook.py`

**Purpose:** Sends HTTP POST or MQTT messages when detection events occur. Rate-limited.

Registered as: `"trigger_webhook"`

| Config key | Default | Description |
|---|---|---|
| `protocol` | `"http"` | `"http"` or `"mqtt"` |
| `url` | `""` | HTTP POST URL or MQTT broker host |
| `trigger_on` | `"count_change"` | `"count_change"` or `"detection"` |
| `rate_limit_s` | `2.0` | Minimum seconds between sends |
| `payload_template` | `""` | Reserved for future use |
| `mqtt_broker` | `"localhost"` | MQTT broker (when `protocol="mqtt"`) |
| `mqtt_topic` | `"cv/events"` | MQTT topic (when `protocol="mqtt"`) |

Payload sent: `{timestamp, frame, session, count}` or `{timestamp, frame, session, detection_count, classes}`.

---

### `engine/nodes/output/mqtt_publish.py`

**Purpose:** Publishes detection events to an MQTT broker. Uses `paho-mqtt`.

Registered as: `"mqtt_publish"`

| Config key | Default | Description |
|---|---|---|
| `broker` | `"localhost"` | MQTT broker hostname |
| `port` | `1883` | MQTT broker port |
| `topic` | `"cv_flow/events"` | MQTT topic |
| `qos` | `0` | MQTT QoS (0, 1, or 2) |
| `trigger_on` | `"detection"` | `"every_frame"`, `"detection"`, `"count_change"` |
| `rate_limit_s` | `0.5` | Minimum seconds between publishes |

Payload: `{frame_number, timestamp, detection_count, counters}`. No-op if `paho-mqtt` not installed.

---

### `engine/nodes/output/kafka_produce.py`

**Purpose:** Publishes detection events to a Kafka topic. Uses `kafka-python`.

Registered as: `"kafka_produce"`

| Config key | Default | Description |
|---|---|---|
| `bootstrap_servers` | `"localhost:9092"` | Comma-separated Kafka broker list |
| `topic` | `"cv_flow_events"` | Kafka topic |
| `trigger_on` | `"detection"` | `"every_frame"`, `"detection"`, `"count_change"` |
| `rate_limit_s` | `0.0` | Minimum seconds between produce calls |

Payload: `{frame_number, timestamp, detection_count, counters}`. `teardown()` calls `flush(timeout=5)` then `close()`. No-op if `kafka-python` not installed.

---

### `engine/nodes/utility/python_function.py`

**Purpose:** Executes arbitrary user Python code on each frame. User defines a `process()` function; the node calls it with `(frame, detections, params)`.

Registered as: `"python_function"`

| Config key | Default | Description |
|---|---|---|
| `code` | `"def process(frame, detections, params):\n    return frame, detections\n"` | Python source code |

Available in user namespace: `np`, `cv2`, `Detection`.

Supported return types from user `process()`:
- `np.ndarray` → updates `ctx.frame`
- `(frame, detections)` → updates frame and detections
- `(frame, detections, metadata_dict)` → updates all three
- `(frame, metadata_dict)` → updates frame and merges metadata
- `dict` with optional keys `frame`, `detections`, `metadata`
- `None` → no change

---

### `engine/nodes/utility/filter_node.py`

**Purpose:** Filters `ctx.detections` by class name, minimum confidence, and minimum area percentage.

Registered as: `"filter"`

| Config key | Default | Description |
|---|---|---|
| `allowed_classes` | `[]` | List of class name strings. Empty = allow all. |
| `min_confidence` | `0.0` | Minimum confidence score |
| `min_area_pct` | `0.0` | Minimum bounding box area as percent of frame area |

---

### `engine/nodes/utility/param_node.py`

**Purpose:** Injects a static parameter dictionary into `ctx.metadata["params"]`. Existing params take precedence (allows live API overrides).

Registered as: `"param"`

Config key: `params: dict` — the parameter values to inject.

---

### `engine/nodes/utility/pipeline_output.py`

**Purpose:** Runtime no-op marker node. Its presence tells the frontend that named outputs exist when saving a pipeline as a reusable template.

Registered as: `"pipeline_output"`

Config: `label: str` (the output name), `description: str`.

---

### `engine/nodes/cpp/cpp_node.py`

**Purpose:** Loads a compiled C++ `.so` shared library and calls its `cv_flow_process()` function on every frame via the ctypes bridge.

Registered as: `"cpp_function"`

| Config key | Description |
|---|---|
| `compiled_so_hash` | SHA-256 hash of the compilation; used to locate `storage/compiled/{hash}/node.so` |

Lifecycle:
- `initialize()` — Locates the `.so` using `CVFLOW_COMPILED_DIR` env var or fallback relative path. Calls `load_cpp_node(so_path)` to wire ctypes signatures. Calls `cv_flow_setup(config_json_bytes)`. Logs the optional `cv_flow_version()` string.
- `process(ctx)` — Calls `self._bridge.python_to_c(ctx)` to fill the shared struct, then `self._lib.cv_flow_process(ctypes.byref(c_ctx))`, then `self._bridge.c_to_python(ctx)` to read back changes.
- `teardown()` — Calls `cv_flow_teardown()`.

---

## Engine C++ Bridge

### `engine/cpp_bridge/context_shm.py`

**Purpose:** Zero-copy Python ↔ C++ data bridge. Converts `FrameContext` to `CVFlowCtxC` and back, sharing the frame buffer by pointer.

**Class: `CppBridge`**

| Method | Description |
|---|---|
| `__init__()` | Pre-allocates a `CVFlowDetectionC` array of 512 elements and a `CVFlowCtxC` struct. These are reused every frame to avoid GC overhead. |
| `python_to_c(ctx: FrameContext) -> CVFlowCtxC` | Ensures `ctx.frame` is C-contiguous. Sets `frame_data` pointer (zero-copy). Marshals up to 512 Python `Detection` objects to the C array. Serialises `ctx.metadata` to JSON (truncated to 4095 chars) into `metadata_json`. Returns the cached C struct. |
| `c_to_python(ctx: FrameContext) -> FrameContext` | Reads back `detection_count` from the C struct and reconstructs Python `Detection` objects. Parses `metadata_json` back to a dict if valid JSON. Frame memory is shared — no copy needed. |

Constant: `_MAX_DETECTIONS = 512`

---

### `engine/cpp_bridge/loader.py`

**Purpose:** Loads a compiled `.so` file and sets up ctypes function signatures for the four C++ ABI functions.

| Function | Description |
|---|---|
| `load_cpp_node(so_path: str) -> ctypes.CDLL` | Loads the shared library. Wires `cv_flow_setup(const char*)`, `cv_flow_process(CVFlowCtxC*)`, `cv_flow_teardown()`. Optionally wires `cv_flow_version() -> const char*` if the symbol exists. Returns the `CDLL` handle. |

C++ ABI expected in every custom node:
```c
void cv_flow_setup(const char* config_json);
void cv_flow_process(CVFlowCtx* ctx);
void cv_flow_teardown(void);
const char* cv_flow_version(void);  // optional
```

---

## Engine Model Hub

### `engine/model_hub/hot_reload.py`

**Purpose:** Installs a POSIX `SIGUSR1` signal handler that triggers model reloading in the `PipelineRunner` without stopping the pipeline.

| Function | Description |
|---|---|
| `install(runner)` | Stores the runner reference. On POSIX, installs a signal handler for `SIGUSR1` that calls `runner.request_reload()`. On Windows, logs a message and skips signal installation (use the API reload endpoint instead). |

---

### `engine/model_hub/yolo_downloader.py`

**Purpose:** Curated model catalog and download/export utility for YOLO and RT-DETR models via the `ultralytics` library.

**Constants:**

- `COCO_CLASSES` — List of 80 COCO object class names.
- `COCO_KEYPOINTS` — List of 17 COCO pose keypoint names.
- `MODEL_CATALOG` — `dict[str, dict]` mapping model keys to metadata. Keys cover: `yolo11n/s/m/l/x`, `yolov8n/s/m/l/x`, `yolov9c/e`, `yolov10n/s/m`, `rtdetr-l/x`, `yolo11n/s/m-seg`, `yolov8n/s/m-seg`, `yolo11n/s/m-pose`, `yolov8n/s/m-pose`, `yolo11n/s/m-cls`, `yolov8n/s/m-cls`. Each entry has: `name, desc, category, task, version, size_mb, badge?, input_shape, output_shapes, class_names, keypoint_names?`.
- `YOLO_MODELS` — Alias for `MODEL_CATALOG` (backward compat).

| Function | Description |
|---|---|
| `download_yolo_model(model_key, models_dir)` | Requires `ultralytics`. Downloads `{model_key}.pt`, exports to ONNX (`simplify=True, opset=12, dynamic=False`). Uses `imgsz=224` for classification models, `640` for all others. Copies `.onnx` to `models_dir/{uuid}/model.onnx`, writes `config.json`. Returns the full config dict with `model_id`. |

---

## Engine Streaming

### `engine/streaming/ws_server.py`

**Purpose:** Async WebSocket server running in a background daemon thread. Provides three channel types per session.

**Channel URL patterns:**
- `ws://host:8765/ws/stream/{session_id}` — receives raw JPEG bytes (pipeline video output)
- `ws://host:8765/ws/events/{session_id}` — receives JSON event strings
- `ws://host:8765/ws/node-preview/{session_id}/{node_id}` — receives per-node JPEG frames for live debugging

**Module-level state:**
- `_stream_queues`, `_event_queues` — `dict[session_id, asyncio.Queue(maxsize=5)]`
- `_node_preview_queues` — `dict["{session_id}:{node_id}", asyncio.Queue(maxsize=5)]`
- `_watched_nodes` — `dict[session_id, set[node_id]]` — tracks which nodes have active preview clients
- `_loop` — the event loop running in the background thread
- `_ready` — `threading.Event` set when the server is listening

| Function | Description |
|---|---|
| `start_server(host, port)` | Starts the background daemon thread running `_run_server()`. Blocks until `_ready` is set (up to 5 s). |
| `send_frame(session_id, jpeg_bytes)` | Thread-safe non-blocking enqueue of a JPEG frame to the stream queue. Creates the queue if it doesn't exist. |
| `send_event(session_id, event: dict)` | Thread-safe non-blocking enqueue of a JSON event (maxsize=20). |
| `is_node_watched(session_id, node_id)` | Returns `True` if a frontend client is watching this node's preview stream. Used to avoid JPEG encoding when no client is connected. |
| `send_node_preview(session_id, node_id, jpeg_bytes)` | Thread-safe push to the node-preview queue. No-op if no client is watching. |
| `cleanup_session(session_id)` | Sends `None` sentinel to all stream, event, and node-preview queues for the session to signal end-of-stream to connected clients. |
| `_put_nowait_safe(queue, item)` | Schedules `queue.put_nowait(item)` on the server's event loop using `call_soon_threadsafe`. Drops the item silently if the queue is full (`asyncio.QueueFull`). |

---

## Frontend

### `frontend/static/index.html`

**Purpose:** Single-page application shell. Loads React 18, ReactFlow 11, and `app.js` via ES module import map (using `jspm` shims for browser compatibility). Applies dark theme CSS overrides for ReactFlow elements.

Key details:
- Import map: `react` → `esm.sh/react@18`, `react-dom` → `esm.sh/react-dom@18`, `reactflow` → `esm.sh/reactflow@11`
- Loads `app.js?v=5` as a module script
- Global CSS: dark background `#0d1117`, ReactFlow edge stroke `#58a6ff`, handle colour, minimap, selected node outline
- `@keyframes blink` — used for status indicator LEDs

---

### `frontend/static/nodes.js`

**Purpose:** Node visual definitions — colour groups, metadata, port schemas, and the React component factory.

**Exported constants:**

`GROUP_COLOR: Record<string, string>` — CSS background colour per group:
- `input: "#1e3a5f"`, `processing: "#1a3d2e"`, `visualize: "#2d1a3d"`, `vision: "#1a2d3d"`, `spatial: "#3d2e0a"`, `utility: "#2d1a4a"`, `cpp: "#0a2d3d"`, `output: "#3d1a1a"`

`NODE_META: Record<string, {group, icon, label}>` — metadata for every registered node type. Complete list with groups:
- **input**: `camera, usb_camera, video_file, image_directory, rtsp_stream`
- **processing**: `preprocess, model_inference, nms`
- **visualize**: `draw_bbox`
- **vision**: `blur, edge_detect, corner_detect, threshold, color_convert, morph, resize, affine_transform`
- **spatial**: `draw_roi, draw_line, object_tracker, counter`
- **utility**: `python_function, filter, param`
- **cpp**: `cpp_function`
- **output**: `pipeline_output, stream_viewer, video_writer, trigger_webhook, mqtt_publish, kafka_produce`

`NODE_PORTS: Record<string, {inputs: Port[], outputs: Port[]}>` — Port definitions where `Port = {id: string, label: string}`. Key port schemas:
- Source nodes (camera, usb_camera, video_file, image_directory, rtsp_stream): `inputs: [], outputs: [{id:"out", label:"Frame"}]`
- `model_inference`: `inputs: [{id:"in"}], outputs: [{id:"frame"}, {id:"raw", label:"Raw Output"}]`
- `nms`: `inputs: [{id:"frame"}, {id:"raw"}], outputs: [{id:"frame"}, {id:"dets", label:"Detections"}]`
- `draw_bbox`: `inputs: [{id:"frame"}, {id:"dets"}], outputs: [{id:"out"}]`
- `object_tracker`: `inputs: [{id:"frame"}, {id:"dets"}], outputs: [{id:"frame"}, {id:"dets", label:"Tracked"}]`
- `filter`: `inputs: [{id:"frame"}, {id:"dets"}], outputs: [{id:"frame"}, {id:"dets", label:"Filtered"}]`
- Sink nodes (stream_viewer, video_writer, mqtt_publish, kafka_produce, trigger_webhook): `inputs: [{id:"in"}], outputs: []`

**Exported functions:**

`makeNode(type) -> React.Component` — factory that produces the visual node component for a given type. Layout constants: `HEADER_H=34px`, `PREVIEW_H=46px`, `PORT_H=22px`. Each node renders: input handles (left, blue `#58a6ff`), a colour-coded header strip with icon + label, a config preview area (using the internal `Preview` component), a port label grid, and output handles (right, green `#3fb950`). Supports `NodeResizer` when selected.

`nodeTypes: Record<string, Component>` — pre-built node type map (passed to ReactFlow's `nodeTypes` prop). Mutable at runtime.

`registerNodeType(type, meta, ports)` — Adds a new type to `NODE_META`, `NODE_PORTS`, and `nodeTypes` at runtime (used by the custom node / template feature).

**Internal component `Preview({type, cfg})`** — renders 1–2 lines of key config values as compact grey text inside each node card. Per type: shows the most important config values (device index, FPS, model ID, blur type/kernel, etc.).

---

### `frontend/static/app.js`

**Purpose:** Main application — full React SPA implementing the pipeline editor, execution control, model hub, log viewer, stream viewer, and all modals.

#### Top-level utility functions

| Function | Description |
|---|---|
| `apiFetch(method, path, body?)` | Thin wrapper around `fetch` targeting `http://localhost:8000/api`. Throws on non-2xx. Returns `null` for 204 No Content. |
| `apiUploadModel(onnxFile, configFile)` | Multipart POST to `/api/models/upload` with two file fields. |
| `parsePySignature(code)` | Parses a Python function signature to extract input parameter names and defaults. Returns `[{name, def}]` or `null`. Used to auto-derive input ports from the user's Python function code. |
| `parsePyOutputs(code)` | Finds the last `return` statement in user Python code and parses variable names as output port IDs. Handles nested parens/brackets when splitting on commas. Returns `[{id, label}]`. |
| `validatePipeline(nodes, edges)` | Client-side validation. Checks: at least one source node, at least one sink node, all nodes with inputs have at least one incoming edge, all nodes with outputs have at least one outgoing edge, `model_inference` has a model_id set, `rtsp_stream` has a URL. Returns `string[]` of warnings. |
| `autoLayout(nodes, edges)` | BFS topological layout. Assigns `x = level * 240 + 60`, `y` centred per column. Returns updated node array with new `position` values. |

**`DEFAULT_CONFIG`** — `Record<node_type, config_dict>` — default config values for every node type when dropped onto the canvas. Contains all the config keys and their initial values.

**`BASE_GROUPS`** — Array of `{label, types[]}` defining the node palette sections: Input, Processing, Visualize, Vision, Spatial, Utility, C++, Output.

#### React Components

**`Field({label, children})`** — Layout wrapper for a labelled form field.

**`PortsInfo({node})`** — Renders port chips (blue for inputs, green for outputs) at the bottom of the Properties panel.

**`NodePreviewCanvas({sessionId, nodeId, nodeType, config, onConfigChange})`** — Live node preview component. Opens a WebSocket to `ws://localhost:8765/ws/node-preview/{sessionId}/{nodeId}` and displays incoming JPEG frames. For `draw_roi` and `draw_line` nodes, renders an interactive HTML5 canvas overlay:
- `draw_roi`: Click to add polygon vertex, double-click to remove, drag to move vertices
- `draw_line`: Drag A/B endpoints to reposition the trip line
Coordinate system: polygon/line points are stored as `[x_pct, y_pct]` (0–100% of frame dimensions).

**`PythonFunctionFields({node, cfg, onUpdate})`** — Specialised form for `python_function` nodes. Auto-detects input ports from the function signature (via `parsePySignature`) and output ports from the return statement (via `parsePyOutputs`). Allows manual addition/removal/renaming of output ports.

**`PropertiesPanel({node, onUpdate, onDuplicate, sessionId, running})`** — Right-panel node configuration. Contains a large `switch(node.type)` block with a bespoke form layout for every node type. When the pipeline is running, also renders `NodePreviewCanvas`. Includes a Delete button that calls `onUpdate(id, null)`.

**`ModelHubModal({onClose})`** — Modal dialog with four tabs:
- **Library**: Lists registered models with Copy ID / Hot Reload / Delete buttons
- **Download Models**: Shows the `MODEL_CATALOG` grouped by category (Object Detection, Segmentation, Pose Estimation, Classification) with download buttons. Calls `POST /api/models/defaults/download/{key}` and auto-copies the returned model ID to clipboard.
- **Packages**: Terminal-style pip install UI. Streams SSE from `POST /api/system/pip-install`. Quick-pill buttons for: `ultralytics`, `torch torchvision torchaudio`, `onnxruntime-gpu`, `opencv-python`, `paho-mqtt`, `kafka-python`.
- **Upload Custom**: File inputs for `.onnx` + `config.json`, calls `apiUploadModel`.

**`CustomNodeModal({onClose, onSave, currentNodes, currentEdges})`** — "Save as Template" modal. Two tabs:
- **From Current Pipeline**: Wraps the current canvas as a reusable template node. Auto-detects `pipeline_output` nodes to define output ports. If the pipeline has source nodes, the template has no input port. Saves as `tmpl_{name}` type.
- **Define Manually**: Form for creating a node type stub with custom name, icon, colour group, and arbitrary port lists.
Both modes call `onSave({type, meta, ports, config})` which calls `registerNodeType` and persists to `localStorage["cvflow_custom_nodes"]`.

**`ShortcutsModal({onClose})`** — Keyboard shortcuts reference modal.
Shortcuts: `Ctrl+S` Save, `Ctrl+Z` Undo, `Ctrl+Y` Redo, `Ctrl+D` Duplicate, `Ctrl+K` Quick-add, `Delete` Remove, `Ctrl+A` Select all, `Esc` Deselect, `?` Help, `Ctrl+Shift+L` Auto-layout, `Ctrl+E` Export JSON.

**`QuickAddModal({onClose, onAdd})`** — Spotlight-style search-and-add. Searches `NODE_META` by label and type string. Keyboard navigable (arrow keys + Enter). Shows top 12 results.

**`ValidateModal({warnings, onClose, onRunAnyway})`** — Shows pipeline validation warnings with "Run Anyway" and "Fix Issues" buttons.

**`ContextMenu({x, y, node, onClose, onDuplicate, onDelete, onResetConfig})`** — Right-click context menu with Duplicate, Reset Config, and Delete actions.

**`Palette({groups, onAddCustom})`** — Left sidebar node palette. Searchable. Shows "Recent" group (stored in `localStorage["cvflow_recent_types"]`, max 6 items) at the top. Drag-and-drop via `e.dataTransfer.setData("application/cvflow", type)`. "Save as Template" button at the bottom.

**`EngineLogViewer({sessionId, height})`** — SSE-based log viewer. Connects to `GET /api/execution/logs/{sessionId}/stream`. Colour-codes lines by severity (red=ERROR, yellow=WARNING, green=pipeline events). Supports auto-scroll (pinned) / manual scroll mode. Max 500 lines in buffer. Copy and Clear buttons.

**`StreamPanel({sessionId, wsPort, counters, height})`** — Live video stream via WebSocket to `ws://localhost:8765/ws/stream/{sessionId}`. Retries connection up to 20 times (1.5 s intervals) to handle engine startup delay. Displays FPS counter. Shows `counters` dict as key-value overlays below the stream.

**`useHistory(nodes, edges, setNodes, setEdges)`** — Custom hook implementing undo/redo. Stack depth: 50 snapshots. `snapshot()` pushes the current state. `undo()` / `redo()` restore states and temporarily set `skip=true` to avoid re-snapshotting the restored state.

**`saveDraft(name, nodes, edges)` / `loadDraft()`** — Persist/restore the canvas to `localStorage["cvflow_draft"]`. Auto-saves every 30 s and on `beforeunload`.

**`loadCustomNodes()`** — Loads custom node definitions from `localStorage["cvflow_custom_nodes"]`.

**`App()`** — Root component. Manages all application state:
- `nodes, edges` — ReactFlow graph state
- `selId` — ID of selected node
- `name` — pipeline name
- `dirty` — unsaved changes indicator
- `pipelineId` — DB ID of the saved pipeline
- `sessionId` — running engine session ID
- `running` — boolean execution state
- `counters` — dict of counter values from WebSocket events
- Modal visibility flags: `showSamples, showModels, showCustomNode, showShortcuts, showQuickAdd`
- `contextMenu`, `validateWarns`, `toast`
- `rightW` — right panel width (resizable, default 280 px)
- `streamH` — stream panel height (resizable, default 200 px)
- `logH` — log viewer height (resizable, default 160 px)

Key App functions:
| Function | Description |
|---|---|
| `onConnect(params)` | Adds edges via ReactFlow `addEdge` |
| `onDrop(e)` | Reads drag type from `e.dataTransfer`, computes canvas position via ReactFlow `project()`, creates new node |
| `onUpdate(id, newCfg, newLabel?, newPorts?)` | Updates a node's config/label/ports. Pass `newCfg=null` to delete the node and its edges. |
| `duplicateNode(id)` | Clones a node at +30px offset |
| `resetNodeConfig(id)` | Restores `DEFAULT_CONFIG[type]` to the node |
| `handleAutoLayout()` | Calls `autoLayout()` then `fitView()` |
| `loadSample(s)` | Replaces canvas with a sample pipeline from `SAMPLES` |
| `handleExport()` | Serialises nodes+edges to JSON and triggers a file download |
| `handleImport(e)` | Reads a JSON file and loads it into the canvas |
| `save()` | POST or PUT to `/api/pipelines`, updates `pipelineId` |
| `startRun()` | Saves pipeline, POSTs to `/api/execution/start`, connects events WebSocket |
| `run()` | Validates pipeline first; if warnings exist, shows `ValidateModal`; otherwise calls `startRun()` |
| `stop()` | POSTs to `/api/execution/stop/{sessionId}`, closes events WS |
| `saveCustomNode(cn)` | Calls `registerNodeType`, persists to localStorage, updates palette groups |
| `quickAddNode(type)` | Adds a node at a random position without drag |

---

### `frontend/static/samples.js`

**Purpose:** Exports eight pre-built sample pipeline definitions.

**Exported constant `SAMPLES: SamplePipeline[]`**

Each sample has `{name, description, nodes[], edges[]}` in ReactFlow format (node `data.config` matches the engine config keys).

| # | Name | Description |
|---|---|---|
| 1 | USB Camera Stream | `usb_camera → stream_viewer` |
| 2 | Video File Playback | `video_file → stream_viewer` |
| 3 | Object Detection | `usb_camera → preprocess → model_inference → nms → draw_bbox → stream_viewer` |
| 4 | Filtered Detection | Object detection + `filter` node (only "person" class, conf≥0.4, area≥0.5%) |
| 5 | Object Tracking + Counter | Detection → NMS → `object_tracker` → `draw_bbox` + `draw_line` + `counter` → `stream_viewer` |
| 6 | RTSP Alert via MQTT | `rtsp_stream → model_inference → nms → draw_bbox → stream_viewer` + `mqtt_publish` |
| 7 | Edge Detection | `usb_camera → color_convert(bgr2gray) → edge_detect(canny) → color_convert(gray2bgr) → stream_viewer` |
| 8 | Morphology Pipeline | `usb_camera → color_convert → threshold → morph(erode) → morph(dilate) → color_convert → stream_viewer` |

Port naming convention used in samples:
- Single-output nodes: `sourceHandle: "out"`
- `model_inference` outputs: `"frame"` and `"raw"`
- `nms` outputs: `"frame"` and `"dets"`
- Split-input nodes: `targetHandle: "frame"` / `"dets"` / `"raw"`
- Single-input nodes: `targetHandle: "in"`
