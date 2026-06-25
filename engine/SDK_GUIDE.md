# CV-FLOW Node SDK Guide

This guide covers everything you need to write, test, and deploy custom nodes
for the CV-FLOW pipeline system.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Writing Your First Node](#writing-your-first-node)
3. [Python Node — Full Reference](#python-node--full-reference)
4. [C++ Node — Full Reference](#c-node--full-reference)
5. [Cross-Language Communication](#cross-language-communication)
6. [Resource Limits & Auto-Scaling](#resource-limits--auto-scaling)
7. [Testing Nodes](#testing-nodes)
8. [Profiling & Debugging](#profiling--debugging)
9. [Practical Examples](#practical-examples)

---

## Architecture Overview

```
┌──────────────┐    PortBus (RAM)     ┌──────────────┐    PortBus (VRAM)    ┌──────────────┐
│ Python Node  │ ──────────────────▶  │  C++ Node    │ ──────────────────▶  │ Python Node  │
│ (Process)    │                      │  (Process)   │                      │  (Thread)    │
└──────────────┘                      └──────────────┘                      └──────────────┘
```

Each node runs in its **own OS process** (Python nodes) or **thread** (C++ nodes,
which release the GIL via ctypes).  Nodes communicate exclusively via **PortBus**
— a shared memory region that is zero-copy on the receiving side.

The pipeline runs as an **async DAG**: nodes do not wait for each other.  A slow
inference node will accumulate frames in its input PortBus; the AutoScaler
detects the build-up and spawns additional worker processes for that node.

---

## Writing Your First Node

### Option A — Function decorator (simplest)

```python
# engine/nodes/my_nodes.py
from engine.nodes.sdk import cv_node_fn

@cv_node_fn(
    label="Grayscale",
    group="processing",
    inputs=["frame"],
    outputs=["frame"],
)
def grayscale(frame):
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

**That's it.** The decorator creates a BaseNode subclass, registers it in the
SDK registry, and wires input/output ports automatically.

### Option B — Class decorator (full control)

```python
from engine.nodes.sdk import cv_node
from engine.nodes.base import BaseNode
import cv2

@cv_node(label="Edge Detect", group="processing")
class EdgeDetectNode(BaseNode):
    def initialize(self):
        self.low  = self.config.get("low_threshold",  50)
        self.high = self.config.get("high_threshold", 150)

    def process(self, ctx):
        ctx.frame = cv2.Canny(ctx.frame, self.low, self.high)
        return ctx
```

### Adding to the frontend palette

Register the node type string in `frontend/static/nodes.js`:

```js
// In NODE_TYPES object:
"grayscale": {
    label: "Grayscale",
    group: "processing",
    color: "#1a3d2e",
    icon: "photo",
    inputs:  [{ id: "in",  label: "Frame" }],
    outputs: [{ id: "out", label: "Frame" }],
},
```

---

## Python Node — Full Reference

### FrameContext fields

| Field | Type | Description |
|-------|------|-------------|
| `ctx.frame` | `np.ndarray` (H×W×3 uint8) | BGR frame.  Modify in-place or assign new. |
| `ctx.detections` | `List[Detection]` | Detection objects from NMS/tracker. |
| `ctx.metadata` | `dict` | Free-form inter-node data bus. |
| `ctx.frame_number` | `int` | Monotonic frame counter. |
| `ctx.timestamp` | `float` | Unix epoch of frame capture. |
| `ctx.frame_gpu` | `cv2.cuda.GpuMat \| torch.Tensor \| None` | GPU frame, if on GPU path. |

### Detection fields

```python
det.x1, det.y1, det.x2, det.y2  # bounding box in absolute pixels
det.confidence                    # float 0–1
det.class_id                      # int
det.class_name                    # str
det.track_id                      # int, -1 if untracked
det.metadata                      # dict, per-detection extras
```

### Lifecycle

```python
class MyNode(BaseNode):
    def initialize(self):
        """Called once when pipeline starts.  Load models, open files here."""
        self.model = load_model(self.config["model_path"])

    def process(self, ctx: FrameContext) -> FrameContext | None:
        """
        Called every frame (sequential) or when a frame arrives (multi-process).

        Return ctx to pass downstream.
        Return None to DROP this frame — it will NOT be forwarded.
        Raise StopIteration to signal source exhausted (input nodes only).
        """
        result = self.model.run(ctx.frame)
        if result is None:
            return None  # drop bad frame
        ctx.metadata["my_result"] = result
        return ctx

    def teardown(self):
        """Called once when pipeline stops.  Release resources here."""
        self.model.close()
```

### UI helpers (sequential mode only)

These work in `process()` when running in the browser UI:

```python
# Display an image in the node's preview panel
self.show_image("debug", ctx.frame)

# Display text in the node's info panel
self.show_text("count", str(len(ctx.detections)))
```

### Frame drop semantics

```python
def process(self, ctx):
    # Drop frame if no detections — downstream nodes won't run for this frame
    if not ctx.detections:
        return None
    # ... process ...
    return ctx
```

### Accessing config

Config values are passed from the pipeline JSON:

```json
{ "type": "my_node", "config": { "threshold": 0.5, "max_objects": 10 } }
```

```python
def initialize(self):
    self.threshold   = self.config.get("threshold", 0.5)
    self.max_objects = self.config.get("max_objects", 10)
```

---

## C++ Node — Full Reference

### Sequential mode (ctypes — simple)

The Python runner calls your .so via ctypes.  Implement three C functions:

```cpp
#include <cv_flow/helpers.hpp>   // frame_mat(), add_detection(), set_metadata()

extern "C" {

void cv_flow_setup(const char* config_json) {
    // Called once — parse config, initialize state
}

void cv_flow_process(CVFlowCtx* ctx) {
    // Called every frame
    // ctx->frame_data points into the Python numpy array — zero-copy!
    cv::Mat frame = cvflow::frame_mat(ctx);  // wrap with no copy
    cv::GaussianBlur(frame, frame, {15, 15}, 0.0);
    // Modify frame in-place — changes are visible to Python immediately
}

void cv_flow_teardown(void) {
    // Called once — release resources
}

}
```

Build as a shared library:

```bash
cmake -B build -DCVFLOW_MODE=sequential
cmake --build build
# → build/node.so
```

Upload `node.so` via the C++ Node editor in the UI, or reference it in pipeline JSON.

### Multi-process mode (subprocess — full parallelism)

Add `cv_flow_run()` / `cv_flow_stop()` and `cv_flow_setup_bus()`:

```cpp
#include <cv_flow/port_bus.h>

static CVFlowPortBus* g_in  = NULL;
static CVFlowPortBus* g_out = NULL;
static volatile int   g_run = 1;

void cv_flow_setup_bus(const char* cfg,
                       const char** in_names, const char** out_names) {
    cv_flow_setup(cfg);
    g_in  = cv_flow_bus_attach(in_names[0],  1280, 720, 3);
    g_out = cv_flow_bus_open  (out_names[0], 1280, 720, 3);
}

void cv_flow_run(void) {
    CVFlowCtx ctx = {};
    ctx.detections         = (CVFlowDetection*)calloc(512, sizeof(CVFlowDetection));
    ctx.detection_capacity = 512;
    while (g_run) {
        if (cv_flow_bus_read(g_in, &ctx, 30) != 0) continue;
        cv_flow_process(&ctx);
        cv_flow_bus_write(g_out, &ctx);
    }
    free(ctx.detections);
}

void cv_flow_stop(void) { g_run = 0; }
```

Build as executable:

```bash
cmake -B build -DCVFLOW_MODE=multiprocess
cmake --build build
# → build/my_node  (executable)
```

The orchestrator spawns this executable with `--input-bus` and `--output-bus` args.

### CVFlowCtx fields (C++)

```c
ctx->frame_data        // uint8_t* — BGR row-major [H × W × 3]
ctx->width             // int32_t
ctx->height            // int32_t
ctx->channels          // int32_t (always 3)
ctx->frame_number      // int32_t
ctx->timestamp         // double (Unix epoch)
ctx->detections        // CVFlowDetection* array
ctx->detection_count   // int32_t
ctx->detection_capacity// int32_t (do NOT exceed this)
ctx->metadata_json     // char[4096] — JSON string, read/write freely
```

### PortBus API (C++)

```c
// Producer side (create=true)
CVFlowPortBus* bus = cv_flow_bus_open("bus_name", max_w, max_h, max_c);
cv_flow_bus_write(bus, &ctx);   // writes frame+dets+meta into shared memory

// Consumer side (create=false)
CVFlowPortBus* bus = cv_flow_bus_attach("bus_name", max_w, max_h, max_c);
if (cv_flow_bus_read(bus, &ctx, timeout_ms=30) == 0) {
    // ctx.frame_data points into shared memory — zero copy!
}

// Monitor
uint64_t depth = cv_flow_bus_depth(bus);   // unread frame count

// Cleanup
cv_flow_bus_close(bus);
```

---

## Cross-Language Communication

Python and C++ nodes share the **same binary PortBus layout**.  You can chain
them in any order — Python → C++ → Python is fully supported.

### Memory layout compatibility

| Field | Python (struct format) | C++ (type) |
|-------|----------------------|------------|
| Header seq_no | `Q` (uint64) | `uint64_t` |
| Frame region | `np.uint8` array | `uint8_t*` |
| Detection x1..conf | `5f` (float32×5) | `float[5]` |
| Detection class_id | `i` (int32) | `int32_t` |
| Detection class_name | `64s` | `char[64]` |
| Metadata JSON | UTF-8 bytes | `char[4096]` |

Both sides use the same struct sizes with no padding — binary-compatible.

### Example: Python preprocess → C++ YOLO → Python draw

```python
# engine/samples/py_cpp_pipeline.py
pipeline = {
    "nodes": [
        {"id": "cam", "type": "usb_camera", "config": {"device_index": 0}},
        {"id": "pre", "type": "preprocess",  "config": {"width": 640, "height": 640}},
        {"id": "yolo","type": "cpp_node",    "config": {"source_code": "..."}},
        {"id": "draw","type": "draw_bbox",   "config": {}},
        {"id": "out", "type": "stream_viewer","config": {}},
    ],
    "edges": [
        {"source": "cam", "sourceHandle": "out",  "target": "pre",  "targetHandle": "in"},
        {"source": "pre", "sourceHandle": "out",  "target": "yolo", "targetHandle": "in"},
        {"source": "yolo","sourceHandle": "out",  "target": "draw", "targetHandle": "in"},
        {"source": "yolo","sourceHandle": "dets", "target": "draw", "targetHandle": "dets"},
        {"source": "draw","sourceHandle": "out",  "target": "out",  "targetHandle": "in"},
    ]
}
```

The runner allocates one PortBus per edge.  The C++ YOLO node reads from the
pre→yolo PortBus, writes detections + frame to the yolo→draw PortBus.

---

## Resource Limits & Auto-Scaling

### Per-node resource limits

In pipeline JSON, add a `resources` key to any node:

```json
{
  "id": "inference_0",
  "type": "model_inference",
  "config": { "model_id": "abc" },
  "resources": {
    "cpu_cores":           [2, 3],
    "max_memory_mb":       2048,
    "gpu_memory_fraction": 0.25,
    "max_fps":             null,
    "priority":            0
  }
}
```

| Parameter | Effect |
|-----------|--------|
| `cpu_cores` | Pin to specific CPU cores (Linux: `sched_setaffinity`, Windows: `SetProcessAffinityMask`) |
| `max_memory_mb` | RSS memory soft limit (Linux only via `RLIMIT_AS`) |
| `gpu_memory_fraction` | ONNX `device_memory_limit` (0.25 = 25% of total GPU VRAM) |
| `max_fps` | Rate-limit this node (frames/sec); dropped frames don't stall pipeline |
| `priority` | Unix nice level (−20=highest, 19=lowest) or Windows priority class |

### Auto-scaling

Add a `scaling` key to enable dynamic worker scaling:

```json
{
  "id": "inference_0",
  "type": "model_inference",
  "scaling": {
    "min_workers":       1,
    "max_workers":       4,
    "scale_up_buffer":   10,
    "scale_down_buffer": 2,
    "cooldown_s":        10.0
  }
}
```

When the input PortBus has > `scale_up_buffer` frames queued, the AutoScaler
spawns an additional worker process.  Frames are distributed round-robin.
When buffer < `scale_down_buffer`, the extra worker is gracefully stopped.

**Example — 12 cameras, inference bottleneck:**

1. 12 cameras run at 15 FPS → 180 frames/sec input.
2. One inference worker handles ~30 frames/sec → buffer grows.
3. AutoScaler detects depth > 10 → spawns worker #2.
4. Two workers → 60 fps → buffer still grows → spawns #3, #4.
5. Four workers → 120 fps → buffer stabilizes → no more scaling.
6. If load drops (e.g., 4 cameras disconnected), buffer drains → scale down.

---

## Testing Nodes

### NodeTestHarness

```python
from engine.nodes.sdk import NodeTestHarness, load_test_image, make_detection
import numpy as np

# Test a built-in node
from engine.nodes.processing.draw_bbox import DrawBboxNode

with NodeTestHarness(DrawBboxNode, config={"thickness": 2}) as h:
    frame = load_test_image()
    ctx   = h.make_context(
        frame=frame,
        detections=[make_detection(10, 10, 200, 200, class_name="car")],
    )
    result = h.run_frame(ctx)
    assert result is not None
    assert result.frame is not None
    print("DrawBboxNode test passed!")
```

### Testing your SDK node

```python
from engine.nodes.samples.fps_limiter import FpsLimiterNode
from engine.nodes.sdk import NodeTestHarness
import numpy as np, time

with NodeTestHarness(FpsLimiterNode, config={"target_fps": 5.0}) as h:
    frame  = np.zeros((480, 640, 3), dtype=np.uint8)
    # First frame: should pass (no previous timestamp)
    r1 = h.run_frame(frame=frame)
    assert r1 is not None, "First frame should pass"
    # Second frame immediately: should be dropped (< 200ms interval)
    r2 = h.run_frame(frame=frame)
    assert r2 is None, "Second immediate frame should be dropped"
    # After waiting > 200ms: should pass again
    time.sleep(0.25)
    r3 = h.run_frame(frame=frame)
    assert r3 is not None, "Frame after interval should pass"
    print("FpsLimiterNode test passed!")
```

### Running tests

```bash
# Run a single node test
python -m pytest engine/tests/test_fps_limiter.py -v

# Or run inline
python engine/nodes/samples/fps_limiter.py
```

---

## Profiling & Debugging

### Per-node stats (sequential mode)

After running a pipeline, read stats from the runner:

```python
# In your code or via API:
# GET /api/execution/stats/{session_id}
stats = runner.get_stats()  # dict: node_id → {avg_ms, p95_ms, fps, errors}
```

Via WebSocket events (pushed every 5s while running):

```json
{
  "type": "node_stats",
  "stats": {
    "cam": { "avg_ms": 0.3, "p95_ms": 0.5, "fps": 30.0, "errors": 0 },
    "inference_0": { "avg_ms": 18.3, "p95_ms": 22.1, "fps": 29.8, "errors": 0 },
    "draw": { "avg_ms": 1.2, "p95_ms": 1.8, "fps": 29.8, "errors": 0 }
  }
}
```

### Per-node preview (UI)

Click the eye icon on a node in the canvas to open a live JPEG preview of that
node's output frame.  Useful for debugging mid-pipeline.

### Structured logging

Each node worker logs to the engine log file (`storage/tmp/{session_id}.log`).
View in UI via Engine Logs panel, or via SSE:

```bash
curl http://localhost:8000/api/execution/logs/{session_id}/stream
```

### Buffer depth monitoring (multi-process mode)

```python
# Check if a specific node's input buffer is backing up:
bus = runner._buses["cam:out__pre:in"]
print(f"Buffer depth: {bus.get_buffer_depth()}")
```

---

## Practical Examples

### 1. Minimal detection filter

```python
from engine.nodes.sdk import cv_node_fn

@cv_node_fn(
    label="High-Confidence Filter",
    group="processing",
    inputs=["frame"],
    outputs=["frame"],
)
def high_conf_filter(frame, detections, *, min_confidence=0.7):
    detections[:] = [d for d in detections if d.confidence >= min_confidence]
    return frame, detections
```

### 2. Zone-based ROI filter (class-based)

```python
from engine.nodes.sdk import cv_node
from engine.nodes.base import BaseNode
import cv2

@cv_node(label="Zone Filter", group="spatial")
class ZoneFilterNode(BaseNode):
    def initialize(self):
        # Zone as percentage of frame (0–100)
        self.x1 = self.config.get("zone_x1", 20)
        self.y1 = self.config.get("zone_y1", 20)
        self.x2 = self.config.get("zone_x2", 80)
        self.y2 = self.config.get("zone_y2", 80)

    def process(self, ctx):
        h, w = ctx.frame.shape[:2]
        x1 = int(self.x1 * w / 100)
        y1 = int(self.y1 * h / 100)
        x2 = int(self.x2 * w / 100)
        y2 = int(self.y2 * h / 100)
        # Draw zone
        cv2.rectangle(ctx.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Filter detections
        ctx.detections = [
            d for d in ctx.detections
            if x1 <= (d.x1 + d.x2) / 2 <= x2
            and y1 <= (d.y1 + d.y2) / 2 <= y2
        ]
        return ctx
```

### 3. Event publisher

```python
import httpx
from engine.nodes.sdk import cv_node_fn
from engine.nodes.base import BaseNode

@cv_node_fn(
    label="Webhook Alert",
    group="output",
    inputs=["frame"],
    outputs=["frame"],
)
def webhook_alert(frame, detections, metadata, *,
                  url="http://localhost:9000/event",
                  min_detections=1, cooldown_s=5.0,
                  _node_id=""):
    import time
    from engine.nodes.samples import _webhook_state
    last_t = _webhook_state.get(_node_id, 0.0)
    if len(detections) >= min_detections and (time.monotonic() - last_t) >= cooldown_s:
        _webhook_state[_node_id] = time.monotonic()
        try:
            httpx.post(url, json={
                "count": len(detections),
                "classes": list({d.class_name for d in detections}),
            }, timeout=1.0)
        except Exception:
            pass
    return frame
```

---

## Quick Reference Card

| Task | Code / JSON |
|------|-------------|
| Create function node | `@cv_node_fn(label="...", group="...", inputs=[...], outputs=[...])` |
| Create class node | `@cv_node(label="...", group="...")` + `class MyNode(BaseNode)` |
| Drop a frame | `return None` from `process()` |
| Signal source exhausted | `raise StopIteration` from `process()` |
| Access config | `self.config.get("key", default)` |
| Access metadata bus | `ctx.metadata["my_key"] = value` |
| Test a node | `NodeTestHarness(MyNode, config={...})` |
| Limit node CPU | `"resources": {"cpu_cores": [0, 1]}` in pipeline JSON |
| Enable auto-scaling | `"scaling": {"max_workers": 4, "scale_up_buffer": 8}` in pipeline JSON |
| C++ sequential build | `cmake -B build -DCVFLOW_MODE=sequential && cmake --build build` |
| C++ multi-process build | `cmake -B build -DCVFLOW_MODE=multiprocess && cmake --build build` |
