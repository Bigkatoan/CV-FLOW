# CV-FLOW — Master Implementation Plan

> **Phiên bản**: 2.0 (Đã bổ sung đầy đủ sau review)
> **Cập nhật**: 2026-06-26

---

## Trạng Thái Source Code Hiện Tại (Baseline)

Trước khi implement bất cứ điều gì, cần nắm rõ trạng thái hiện tại:

| File / Module | Trạng thái | Ghi chú |
|---|---|---|
| `frontend/static/app.js` | Tồn tại, v=33 | 4300 dòng, có `parseParams()` regex ở dòng 36–48 |
| `frontend/static/nodes.js` | Tồn tại, v=30 | 149 dòng, chưa có `useState`/`useReactFlow` |
| `frontend/static/index.html` | Tồn tại | `app.js?v=33` — bump lên v=34 khi hoàn tất |
| `backend/app/services/execution_service.py` | Tồn tại | Sync subprocess, `_sessions`/`_meta` in-memory dict, **không có Lock** |
| `backend/app/api/execution.py` | Tồn tại | Thiếu session persistence, `GET /execution/logs/` trả `list[str]` thẳng |
| `backend/app/api/system.py` | Tồn tại | Chỉ có `POST /system/pip-install`, **không có** `GET /system/info` |
| `backend/app/api/router.py` | Tồn tại | 7 routers, chưa có `datahub` |
| `backend/app/models/model_registry.py` | Tồn tại | 7 columns cơ bản, thiếu `slug`, `ports_json`, `tag`, v.v. |
| `backend/app/database.py` | Tồn tại | `create_tables()` dùng `create_all` — không alter existing tables |
| `engine/core/pipeline_builder.py` | Tồn tại | Hardcode chỉ `python_node`/`cpp_node` — skip mọi type khác |
| `engine/core/node_registry.py` | Tồn tại | `@register` là **no-op**, `get_node_class` raise ValueError — legacy shim |
| `engine/nodes/spatial/counter.py` | Tồn tại | Trigger type là `line_cross`/`zone_enter`/`zone_exit` — **không phải** `roi` |
| `engine/nodes/spatial/draw_roi.py` | Tồn tại | ✅ có |
| `backend/requirements.txt` | Tồn tại | Có `alembic==1.13.2`, `httpx==0.27.0` — thiếu `mcp`, `psutil`, `pynvml` |
| `engine/requirements.txt` | Tồn tại | Thiếu `onnx` cho introspection |

**Nhận xét chính**:
- `_sessions` dict không thread-safe → race condition khi concurrent requests
- Session data mất khi server restart (in-memory)
- `pipeline_builder` không extend được — phải sửa để support `model_node`
- Schema migration: `create_all` không alter existing table → phải dùng Alembic

---

---

## Part 1: Inline Node Parameter & Resource UI

### Context

Tất cả params của node (slider, checkbox, text_input) chỉ hiển thị trong PropertiesPanel — phải chọn node trước mới thấy. ResourceLimits (cpu_cores, max_fps, max_memory_mb, gpu_memory_fraction) đã implement ở engine nhưng không có UI.

**Mục tiêu**:
1. Mỗi node card có section **Params** hiển thị inline sliders/checkboxes/text inputs — không cần mở panel
2. Mỗi node card có section **Resources** (mặc định có trên tất cả node) — cấu hình FPS limit, CPU cores, memory limit ngay trên node
3. Giá trị chỉnh inline được lưu vào pipeline JSON và engine đọc được

---

### Architecture Decisions

**State update từ node component**: Dùng `useReactFlow().setNodes()` bên trong component trả về từ `makeNode()`. ReactFlow truyền `id` prop cho custom nodes → dùng `id` để filter đúng node khi update.

**Resources storage**: Riêng `node.data.resources` (không nằm trong `config`) → `save()` xuất thành `node["resources"]` trong pipeline JSON. Engine đọc `node_json.get("resources", {})` → không cần sửa engine.

**Dirty tracking**: Inline edits gọi `setNodes()` trực tiếp → dùng ReactFlow `onNodesChange` callback ở App để set dirty flag. **Quan trọng**: Chỉ set dirty khi change type KHÔNG phải là `position`, `dimensions`, hoặc `select`. Các change types của ReactFlow gồm: `position`, `dimensions`, `select`, `remove`, `add`, `reset` — chỉ `remove` và `add` cần trigger dirty ngoài data changes.

**Hiển thị**: Cả hai section collapse mặc định. Badge trên header cho thấy có params/resource active.

---

### 1.1 `frontend/static/nodes.js` — Thay đổi lớn nhất

**Thêm import** ở dòng 1 (hiện tại chỉ có `createElement`):
```js
import { createElement, useState } from "react";
import { Handle, Position, NodeResizer, useReactFlow } from "reactflow";
import htm from "https://esm.sh/htm@3";
const html = htm.bind(createElement);
```

**Thêm `parseParamsInline(code)`** — copy EXACT regex từ `app.js` dòng 36–48, không viết lại:
```js
function parseParamsInline(code) {
  if (!code) return [];
  const out = [];
  // Exact same regex as app.js parseParams()
  for (const m of code.matchAll(/\bslider\s*\(\s*["'](\w+)["']\s*,\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\)/g))
    out.push({ type: "slider",   name: m[1], min: +m[2], max: +m[3], default: +m[4] });
  for (const m of code.matchAll(/\bcheckbox\s*\(\s*["'](\w+)["']\s*,\s*(True|False)\s*\)/gi))
    out.push({ type: "checkbox", name: m[1], default: m[2].toLowerCase() === "true" });
  for (const m of code.matchAll(/\btext_input\s*\(\s*["'](\w+)["']\s*,\s*["']([^"']*)["']\s*\)/g))
    out.push({ type: "text",     name: m[1], default: m[2] });
  return out;  // button không cần inline control
}
```

> ⚠️ **Không viết lại regex** — phải giống 100% `app.js:36-48` để hai nơi parse nhất quán.

**Sửa `makeNode(type)`** — thêm `id` vào props (ReactFlow tự truyền prop `id` cho custom nodes):
```js
export function makeNode(type) {
  const meta  = NODE_META[type] ?? { group: "core", icon: "◻", label: type };
  const hdrBg = GROUP_COLOR[meta.group] ?? "#21262d";

  const Comp = ({ id, data, selected }) => {   // ← thêm `id`
    const { setNodes } = useReactFlow();        // ← hook mới
    const [paramsOpen, setParamsOpen] = useState(false);
    const [resOpen,    setResOpen]    = useState(false);

    const cfg = data.config ?? {};
    const res = data.resources ?? {};

    // Update a config key — immutable update qua setNodes
    const setParam = (key, val) => setNodes(ns => ns.map(n =>
      n.id !== id ? n : { ...n, data: { ...n.data, config: { ...n.data.config, [key]: val } } }
    ));

    // Update a resource key
    const setRes = (key, val) => setNodes(ns => ns.map(n =>
      n.id !== id ? n : { ...n, data: { ...n.data, resources: { ...(n.data.resources ?? {}), [key]: val } } }
    ));

    // ... rest of render
  };
  Comp.displayName = type;
  return Comp;
}
```

**Params section** (chỉ render khi `type === "python_node"` VÀ có ít nhất 1 param):
```js
const params = type === "python_node" ? parseParamsInline(cfg.code) : [];

// Collapsed badge: "⚙ 3" khi collapsed và params.length > 0
// Header: click để toggle paramsOpen
// Expanded: render slider/checkbox/text cho mỗi param
//   - slider: <input type="range"> + số value bên cạnh, onChange gọi setParam(p.name, +e.target.value)
//   - checkbox: <input type="checkbox">, onChange gọi setParam(p.name, e.target.checked)
//   - text: <input type="text" style={{width: 80}}>, onBlur gọi setParam(p.name, e.target.value)
// Current value: cfg[p.name] ?? p.default
```

**Resources section** (render trên TẤT CẢ node types):

Giá trị hiển thị:
- `max_fps`: `res.max_fps ?? ""` — number input, 0 = unlimited
- `cpu_cores`: convert array → string: `(res.cpu_cores ?? []).join(",")` — text input
- `max_memory_mb`: `res.max_memory_mb ?? ""` — number input

Khi save:
- `max_fps`: `parseInt(val) || 0`
- `cpu_cores`: `val.trim() ? val.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n)) : []`
- `max_memory_mb`: `parseInt(val) || 0`

**Type contract engine expects** (từ `engine/nodes/base.py` và ResourceLimits):
```
resources.max_fps         → int (0 = unlimited)
resources.cpu_cores       → list[int] ([] = all cores)
resources.max_memory_mb   → int (0 = unlimited)
resources.gpu_memory_fraction → float (0.0–1.0, 0 = unlimited)
```

Collapsed badge: `"⚡ 30fps"` khi `max_fps > 0`, `"⚡ 0.5 cores"` nếu cpu_cores set, v.v.

**Node card layout mới**:
```
┌─────────────────────────────┐
│ [icon] Label          LOOP  │  ← header (unchanged)
├──────────┬──────────────────┤
│ ○ in     │        out ●     │  ← ports (unchanged)
├──────────┴──────────────────┤
│ ⚙ 2 params            [▾]  │  ← params toggle (chỉ python_node)
│  threshold ──────●── 128    │  ← inline slider (expanded)
│  ☑ enabled                  │  ← inline checkbox
├─────────────────────────────┤
│ ⚡ 30fps                [▾] │  ← resources toggle (badge)
│  FPS    [30]  CPU [0,1]     │  ← resource inputs (expanded)
│  Mem    [   ] MB            │
└─────────────────────────────┘
```

**Style constants**:
- Section header background: `#0d1117` (darker than card `#161b22`)
- Font: 10px, padding: 4px 8px
- Number input width: 52px, text input width: 80px
- Slider: `flex: 1`, `min-width: 60px`
- Section header height: 24px

---

### 1.2 `frontend/static/app.js` — Thay đổi nhỏ

**`save()` function** — thêm `resources` vào exported JSON:

Tìm đoạn `nodes: nodes.map(n => ({` trong hàm save (khoảng dòng 800–850 trong app.js hiện tại), thêm:
```js
nodes: nodes.map(n => ({
  id: n.id,
  type: n.type,
  label: n.data.label,
  position: n.position,
  config: n.data.config ?? {},
  resources: n.data.resources ?? {},   // ← THÊM
})),
```

**`loadPipeline()` / sample load** — restore `resources` vào `node.data`:
```js
data: {
  label: n.label,
  config: n.config ?? {},
  resources: n.resources ?? {},    // ← THÊM
  ports: ...,
}
```

**Dirty tracking qua `onNodesChange`**: Tìm handler `onNodesChange` trong app.js. Sửa để chỉ set dirty khi change thực sự liên quan đến data:
```js
const handleNodesChange = useCallback((changes) => {
  onNodesChange(changes);  // built-in ReactFlow handler
  // Chỉ mark dirty với data/structure changes, không phải UI-only changes
  const hasDataChange = changes.some(c =>
    c.type !== "position" &&
    c.type !== "dimensions" &&
    c.type !== "select"     // click chọn node KHÔNG phải dirty
  );
  if (hasDataChange) setDirty(true);
}, [onNodesChange]);
```

**Bump version**: `index.html` → `app.js?v=34`, import trong `app.js` dòng 8 → `nodes.js?v=31`.

---

### 1.3 `frontend/static/index.html` — Nhỏ

```html
<script type="module" src="./app.js?v=34"></script>
```

---

### Verification — Part 1

- [ ] Load sample "Edge Detection" → node card có Resources section (tất cả nodes)
- [ ] Expand Resources, set Max FPS=15 → Save → JSON có `"resources": {"max_fps": 15}`
- [ ] `python_node` với `slider("threshold", 0, 255, 128)` trong code → Params section có range input
- [ ] Kéo slider → value badge update real-time (không cần save)
- [ ] Run pipeline → engine log có `ResourceLimits applied: max_fps=15`
- [ ] Click chọn node → dirty flag **KHÔNG** bật (verify bằng title bar "unsaved" indicator)
- [ ] Reload page → resources persist (đã save JSON correct)
- [ ] PropertiesPanel vẫn hoạt động bình thường — không bị break

---

---

## Part 2: Data Hub

### Context

CV-FLOW hiện có:
- SQLite (`cv_flow.db`) với 3 bảng: `pipelines`, `model_registry`, `compiled_nodes` — dùng SQLAlchemy async + aiosqlite
- Face vector DB tự viết: cosine similarity trên numpy, lưu `.npy` + `identities.json` trong `storage/facedb/`
- Không có persistence cho: execution history, pipeline metrics, detection events, generic vector collections

**Mục tiêu**: Thêm Data Hub — quản lý hai loại database riêng biệt, hiển thị trong UI.

---

### 2.1 Critical Issue: Async/Sync Bridge cho DB Writes

**Vấn đề**: `execution_service.py` là pure sync code (subprocess management), không thể `await` async SQLAlchemy. Backend `create_tables()` và session factories đều async.

**Giải pháp được chọn: Dùng sync SQLite3 trực tiếp cho execution logging** (không qua SQLAlchemy async):

```python
# backend/app/services/execution_db.py  [NEW]
"""
Sync SQLite3 wrapper cho execution session logging.
Tách biệt khỏi SQLAlchemy async engine để dùng được trong execution_service.py (sync context).
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from app.config import settings

def _get_conn() -> sqlite3.Connection:
    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "").replace("./", "")
    return sqlite3.connect(db_path, check_same_thread=False)

def insert_session(session_id: str, pipeline_id: str) -> None:
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO execution_sessions
            (id, pipeline_id, started_at, status, frame_count)
            VALUES (?, ?, ?, 'running', 0)
        """, (session_id, pipeline_id, datetime.now(timezone.utc).isoformat()))

def update_session_stopped(session_id: str, status: str, frame_count: int = 0, error_msg: str = None) -> None:
    with _get_conn() as conn:
        conn.execute("""
            UPDATE execution_sessions
            SET ended_at=?, status=?, frame_count=?, error_msg=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), status, frame_count, error_msg, session_id))

def insert_node_metrics(session_id: str, metrics: dict) -> None:
    """metrics: {node_id: {avg_ms, p95_ms, fps, errors}}"""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.executemany("""
            INSERT INTO node_metrics (session_id, node_id, avg_ms, p95_ms, fps, errors, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (session_id, nid, m.get("avg_ms", 0), m.get("p95_ms", 0),
             m.get("fps", 0), m.get("errors", 0), now)
            for nid, m in metrics.items()
        ])
```

> **Tại sao không dùng asyncio.run()?** Vì `execution_service.start_session()` được gọi từ FastAPI async handler — asyncio.run() sẽ raise "cannot be called from a running event loop".
>
> **Tại sao không tạo new event loop?** Fragile, có thể conflict với uvicorn event loop.
>
> **Kết luận**: Sync sqlite3 cho execution logging là pattern an toàn, đã dùng rộng rãi.

---

### 2.2 Thread Safety cho `_sessions` Dict

**Vấn đề**: `execution_service.py` không có lock. Hai concurrent requests → race condition trên `active_count` → cùng `ws_port`.

**Sửa `execution_service.py`**:
```python
import threading
_lock = threading.Lock()

def start_session(session_id, pipeline_json, params_override=None, mode="sequential"):
    with _lock:
        if mode == "sequential":
            _kill_all_running_locked()  # internal version, assumes lock held
        # ... existing logic
        active_count = sum(1 for p in _sessions.values() if p.poll() is None)
        ws_port = settings.engine_ws_port + (active_count if mode != "sequential" else 0)
        # ... create proc
        _sessions[session_id] = proc
        _meta[session_id] = {...}
    
    # DB write OUTSIDE lock (sync, không block long)
    from app.services.execution_db import insert_session
    insert_session(session_id, pipeline_json.get("id", ""))
    return proc
```

Tương tự thêm lock cho `stop_session()` và `session_status()`.

---

### 2.3 ORM Models — `backend/app/models/datahub.py` [NEW]

```python
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Integer, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

def _utcnow():
    return datetime.now(timezone.utc)

class ExecutionSession(Base):
    __tablename__ = "execution_sessions"

    id:          Mapped[str]           = mapped_column(String(36), primary_key=True)
    pipeline_id: Mapped[str]           = mapped_column(String(36), nullable=False)
    started_at:  Mapped[datetime]      = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at:    Mapped[datetime|None] = mapped_column(DateTime(timezone=True), nullable=True)
    status:      Mapped[str]           = mapped_column(String(16), default="running")
    # status values: "running" | "completed" | "error" | "stopped"
    frame_count: Mapped[int]           = mapped_column(Integer, default=0)
    error_msg:   Mapped[str|None]      = mapped_column(Text, nullable=True)
    mode:        Mapped[str]           = mapped_column(String(16), default="sequential")

class NodeMetric(Base):
    __tablename__ = "node_metrics"

    id:          Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:  Mapped[str]      = mapped_column(String(36), ForeignKey("execution_sessions.id"))
    node_id:     Mapped[str]      = mapped_column(String(64))
    avg_ms:      Mapped[float]    = mapped_column(Float)
    p95_ms:      Mapped[float]    = mapped_column(Float)
    fps:         Mapped[float]    = mapped_column(Float)
    errors:      Mapped[int]      = mapped_column(Integer)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:   Mapped[str]      = mapped_column(String(36), ForeignKey("execution_sessions.id"))
    pipeline_id:  Mapped[str]      = mapped_column(String(36))
    node_id:      Mapped[str]      = mapped_column(String(64))
    timestamp:    Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    event_type:   Mapped[str]      = mapped_column(String(32))
    # event_type values: "object_detected" | "face_matched" | "face_unknown" | "counter_update"
    payload_json: Mapped[str]      = mapped_column(Text)  # JSON: bbox, class, confidence, identity_id, count, etc.
```

**Migration strategy**:

Vì dự án đã có `alembic==1.13.2` trong `backend/requirements.txt` và `backend/migrations/` folder, nhưng `create_all()` không alter existing tables, cần dùng Alembic cho các bảng MỚI trong Part 2 (ok với `create_all`) và QUAN TRỌNG cho Part 4 (alter existing `model_registry`).

Với 3 bảng mới của Part 2: `create_all` + `checkfirst=True` là đủ vì chúng chưa tồn tại.

`backend/app/main.py` lifespan đã gọi `await create_tables()` → import `datahub.py` models vào `database.py` hoặc `models/__init__.py` là đủ để chúng được tạo.

---

### 2.4 Wire-up `execution_service.py` với DB

**`start_session()`** — thêm DB insert sau khi tạo proc:
```python
# Sau _sessions[session_id] = proc
from app.services.execution_db import insert_session
insert_session(session_id, pipeline_json.get("id", ""), mode=mode)
```

**`stop_session()`** — thêm DB update sau khi terminate:
```python
# Sau proc.terminate() và wait
from app.services.execution_db import update_session_stopped, insert_node_metrics
status_str = "stopped"
# Read stats file nếu tồn tại
stats_path = Path(_meta.get(session_id, {}).get("stats_path", ""))
if stats_path.exists():
    try:
        metrics = json.loads(stats_path.read_text())
        insert_node_metrics(session_id, metrics)
        # Lấy frame_count từ stats nếu có
        frame_count = max((m.get("fps", 0) for m in metrics.values()), default=0)
    except Exception:
        pass
update_session_stopped(session_id, status_str)
_sessions.pop(session_id, None)
_meta.pop(session_id, None)
```

---

### 2.5 Backend API — `backend/app/api/datahub.py` [NEW]

```
# Relational
GET  /api/datahub/relational/tables           — list tables + row counts (whitelist only: execution_sessions, node_metrics, detection_events, pipelines, model_registry)
GET  /api/datahub/relational/sessions         — execution history (query params: pipeline_id, status, limit=50, offset=0)
GET  /api/datahub/relational/sessions/{id}    — chi tiết 1 session + node metrics
GET  /api/datahub/relational/events           — detection events (query params: event_type, pipeline_id, limit=100, offset=0)
GET  /api/datahub/relational/export/{table}   — download CSV (table phải nằm trong whitelist)

# Vector
GET    /api/datahub/vector/collections                — list collections + count + dim
POST   /api/datahub/vector/collections                — create new collection (body: {name, dim})
DELETE /api/datahub/vector/collections/{name}         — delete collection
GET    /api/datahub/vector/collections/{name}/records — list entries (query: limit=50, offset=0)
DELETE /api/datahub/vector/collections/{name}/{id}    — delete 1 entry
POST   /api/datahub/vector/collections/{name}/search  — cosine search (body: {embedding: [...], top_k: 5})
GET    /api/datahub/vector/collections/{name}/export  — download index.npy + meta.json as zip
```

> **Security**: `export/{table}` và mọi table-name param phải whitelist cứng:
> ```python
> ALLOWED_TABLES = {"execution_sessions", "node_metrics", "detection_events", "pipelines", "model_registry"}
> if table not in ALLOWED_TABLES:
>     raise HTTPException(400, "Table not allowed")
> ```

**Wire-up trong `backend/app/api/router.py`**:
```python
from app.api import pipelines, models, execution, compiler, system, facedb, python_lsp, datahub  # ← thêm

api_router.include_router(datahub.router)  # ← thêm
```

---

### 2.6 VectorStore Abstraction — `engine/data/vector_store.py` [NEW]

```python
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import threading

@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict

class VectorStore:
    """Named collection of embeddings, numpy-backed, cosine similarity.

    Thread-safety: read/write có RLock. Concurrent reads OK, write exclusive.
    Save strategy: write-through sau mỗi add()/delete(). Không dùng periodic flush
    để tránh data loss khi crash.
    """

    def __init__(self, name: str, storage_dir: Path, dim: int = 512) -> None:
        self.name = name
        self._dir = Path(storage_dir) / name
        self._dim = dim
        self._lock = threading.RLock()
        self._embeddings: np.ndarray | None = None  # shape (N, dim)
        self._meta: list[dict] = []  # [{id, label, metadata}]
        self._dir.mkdir(parents=True, exist_ok=True)
        if (self._dir / "index.npy").exists():
            self.load()

    def add(self, id: str, embedding: np.ndarray, metadata: dict = {}) -> None:
        emb = np.array(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self._dim:
            raise ValueError(f"Expected dim={self._dim}, got {emb.shape[0]}")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm  # normalize to unit vector for cosine via dot product
        with self._lock:
            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, emb])
            self._meta.append({"id": id, "label": id, **metadata})
            self._save_locked()

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        q = np.array(query, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        with self._lock:
            if self._embeddings is None or len(self._embeddings) == 0:
                return []
            scores = self._embeddings @ q
            idxs = np.argsort(scores)[::-1][:top_k]
            return [
                SearchResult(
                    id=self._meta[i]["id"],
                    score=float(scores[i]),
                    metadata={k: v for k, v in self._meta[i].items() if k != "id"}
                )
                for i in idxs
            ]

    def delete(self, id: str) -> bool:
        with self._lock:
            idxs = [i for i, m in enumerate(self._meta) if m["id"] == id]
            if not idxs:
                return False
            mask = np.ones(len(self._meta), dtype=bool)
            for i in idxs:
                mask[i] = False
            self._embeddings = self._embeddings[mask] if self._embeddings is not None else None
            self._meta = [m for i, m in enumerate(self._meta) if mask[i]]
            self._save_locked()
            return True

    def _save_locked(self) -> None:
        """Must be called with self._lock held."""
        if self._embeddings is not None:
            np.save(str(self._dir / "index.npy"), self._embeddings)
        (self._dir / "meta.json").write_text(json.dumps(self._meta, indent=2))

    def load(self) -> None:
        with self._lock:
            idx_path = self._dir / "index.npy"
            meta_path = self._dir / "meta.json"
            if idx_path.exists():
                self._embeddings = np.load(str(idx_path))
                self._dim = self._embeddings.shape[1] if self._embeddings.ndim == 2 else self._dim
            if meta_path.exists():
                self._meta = json.loads(meta_path.read_text())

    @property
    def count(self) -> int:
        return len(self._meta)

    @property
    def dim(self) -> int:
        return self._dim
```

**Storage layout**:
```
storage/
├── vectordb/
│   ├── faces/
│   │   ├── index.npy    ← all embeddings stacked (N × dim)
│   │   └── meta.json    ← [{id, label, metadata}]
│   ├── products/
│   └── features/
├── relationaldb/        ← user-added SQLite files (ngoài cv_flow.db)
│   ├── analytics.db
│   └── events.db
```

**Collections management** — backend giữ singleton dict `{name: VectorStore}` trong module-level. API router load/create on demand.

---

### 2.7 Frontend UI — Data Hub Tab (`frontend/static/datahub.js`) [NEW]

Import trong `app.js` và thêm tab "Data Hub" vào sidebar (cùng level "Pipelines", "Models").

**State management** (React useState hooks trong DataHubPanel component):
```js
const [activeTab, setActiveTab]     = useState("relational");  // "relational" | "vector"
const [tables, setTables]           = useState([]);            // list of {name, row_count}
const [collections, setCollections] = useState([]);            // list of {name, count, dim}

// Browse modal state
const [browseTable, setBrowseTable] = useState(null);  // table name or null
const [browseRows,  setBrowseRows]  = useState([]);
const [browseTotal, setBrowseTotal] = useState(0);
const [browsePage,  setBrowsePage]  = useState(0);     // 0-indexed
const BROWSE_LIMIT = 20;

// Vector browse state
const [vecCollection, setVecCollection] = useState(null);
const [vecRecords,    setVecRecords]    = useState([]);
const [vecPage,       setVecPage]       = useState(0);
const VEC_LIMIT = 50;

// Loading / error state (separate per operation để không block UI)
const [loadingTables,    setLoadingTables]    = useState(false);
const [loadingBrowse,    setLoadingBrowse]    = useState(false);
const [error,            setError]            = useState(null);
```

**Data fetching**:
- `useEffect([])` khi DataHubPanel mount → fetch `/api/datahub/relational/tables` và `/api/datahub/vector/collections`
- Browse modal open → fetch page 0 ngay lập tức
- Page change → fetch new page (không refetch nếu same page)
- Error → hiện error banner trong modal, không dismiss modal

**Pagination component** (tái sử dụng giữa relational và vector browse):
```js
function Pagination({ page, total, limit, onPageChange }) {
  const totalPages = Math.ceil(total / limit);
  return html`
    <div style=${{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
      <button disabled=${page === 0} onClick=${() => onPageChange(page - 1)}>← Prev</button>
      <span style=${{ fontSize: 11, color: "#8b949e" }}>
        ${page * limit + 1}–${Math.min((page + 1) * limit, total)} of ${total}
      </span>
      <button disabled=${(page + 1) * limit >= total} onClick=${() => onPageChange(page + 1)}>Next →</button>
    </div>`;
}
```

**Browse modal Relational**:
- Table: columns = keys của first row (dynamic, không hardcode)
- Export button → `GET /api/datahub/relational/export/{table}` → trigger browser download
- Filter text → client-side filter trên current page (không re-fetch)

**Browse modal Vector**:
- List records với id, label, metadata JSON preview (truncated 80 chars)
- Search form: text input nhập comma-separated floats hoặc file `.npy` upload → POST `/api/datahub/vector/collections/{name}/search`

**Layout**:
```
┌─── Data Hub ────────────────────┐
│ [Relational]  [Vector]          │  ← tab switcher
├─────────────────────────────────┤
│ RELATIONAL DATABASES            │
│ ┌──────────────────────────────┐│
│ │ 📋 cv_flow.db (main)         ││
│ │    execution_sessions  5 rows││
│ │    node_metrics       20 rows││
│ │    detection_events  100 rows││
│ │    [Browse]  [Export CSV]    ││
│ └──────────────────────────────┘│
├─────────────────────────────────┤
│ VECTOR DATABASES                │
│ ┌──────────────────────────────┐│
│ │ 🔷 faces   (512-dim, 24 vecs)││
│ │    [Browse] [Search] [Export]││
│ └──────────────────────────────┘│
│ [+ New Vector Collection]       │
└─────────────────────────────────┘
```

---

### 2.8 Requirements Update

**`backend/requirements.txt`** — không cần thêm gì cho Part 2 vì `sqlite3` là built-in Python.

---

### Verification — Part 2

- [ ] Server restart → `execution_sessions` bảng tồn tại trong DB (create_all chạy khi khởi động)
- [ ] Start pipeline → record xuất hiện trong `execution_sessions` với status="running"
- [ ] Stop pipeline → record update ended_at, status="stopped", node_metrics inserted
- [ ] Hai concurrent `/execution/start` → không cùng ws_port (lock hoạt động)
- [ ] `GET /api/datahub/relational/tables` → trả list với row counts
- [ ] Browse `execution_sessions` → bảng render đúng, pagination hoạt động
- [ ] Export CSV `execution_sessions` → download file với columns đúng
- [ ] `GET /api/datahub/relational/export/sqlite_master` → HTTP 400 (whitelist block)
- [ ] Create vector collection → thư mục `storage/vectordb/{name}/` được tạo
- [ ] Add embedding → `index.npy` và `meta.json` updated ngay (write-through)
- [ ] Search cosine → kết quả có score trong [−1, 1]

---

---

## Part 3: CV-FLOW MCP Server

### Context

Thêm MCP server để Claude có thể giao tiếp trực tiếp với hệ thống CV-FLOW — tạo/sửa pipeline, chạy/dừng execution, đọc logs và stats. Transport: **stdio** (chuẩn cho Claude Desktop).

---

### 3.1 Prerequisites — `GET /api/system/info` [THIẾU]

`backend/app/api/system.py` hiện chỉ có `POST /system/pip-install`. Cần thêm endpoint info.

**Dependencies cần thêm vào `backend/requirements.txt`**:
```
psutil>=5.9
```
> `pynvml` optional — GPU info chỉ available nếu có NVIDIA GPU. Dùng try/import.

**Endpoint mới trong `system.py`**:
```python
@router.get("/info")
async def system_info():
    """Return CPU, RAM, GPU information."""
    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    result = {
        "cpu_count": cpu_count,
        "cpu_percent": cpu_percent,
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_used_gb":  round(ram.used  / (1024**3), 2),
        "gpu": [],
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode()
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            result["gpu"].append({
                "index": i,
                "name": name,
                "vram_total_mb": mem.total // (1024**2),
                "vram_used_mb":  mem.used  // (1024**2),
            })
    except Exception:
        pass  # no GPU or pynvml not installed
    return result
```

---

### 3.2 Fix API Response Contract — `GET /api/execution/logs/{session_id}`

**Vấn đề**: API hiện tại (`execution.py` dòng 65–67) trả `list[str]` thẳng. MCP tool skeleton trong plan cũ gọi `data.get("lines", [])` → lỗi vì response là list, không phải dict.

**Lựa chọn**: Sửa MCP tool để match API hiện có (đừng sửa API vì có thể break frontend):

```python
@mcp.tool()
async def get_logs(session_id: str, tail: int = 100) -> str:
    """Get last N lines of engine logs for a running or completed session."""
    data = await _get(f"/execution/logs/{session_id}?tail={tail}")
    # API trả list[str] thẳng, không phải {lines: [...]}
    if isinstance(data, list):
        return "\n".join(data)
    return str(data)
```

---

### 3.3 MCP Server — `backend/mcp_server.py` [NEW]

**Dependencies thêm vào `backend/requirements.txt`**:
```
mcp>=1.0
```
(`httpx>=0.27.0` đã có sẵn)

```python
# backend/mcp_server.py
"""
CV-FLOW MCP Server — expose CV-FLOW control plane to Claude.
Transport: stdio (Claude Desktop standard).
Requires: pip install mcp
Requires: CV-FLOW backend running on localhost:8000
"""
import asyncio
import json
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("CV-FLOW", instructions="""
You have direct access to the CV-FLOW computer vision pipeline system.
Use these tools to create, run, and monitor CV pipelines on the user's machine.
When creating pipelines, prefer python_node type with Python code.
""")

BASE = "http://localhost:8000/api"
TIMEOUT = 15.0  # seconds — download/compile can be slow

# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _get(path: str) -> dict | list:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.get(f"{BASE}{path}")
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to CV-FLOW backend (localhost:8000). Is it running?"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

async def _post(path: str, body: dict = {}) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{BASE}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to CV-FLOW backend (localhost:8000). Is it running?"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

async def _delete(path: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.delete(f"{BASE}{path}")
            r.raise_for_status()
            if r.status_code == 204:
                return {"deleted": True}
            return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to CV-FLOW backend"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

# ── Pipeline tools ────────────────────────────────────────────────────────────

@mcp.tool()
async def list_pipelines() -> list[dict]:
    """List all saved CV-FLOW pipelines (id, name, node count)."""
    return await _get("/pipelines")

@mcp.tool()
async def get_pipeline(pipeline_id: str) -> dict:
    """Get full details of a pipeline including nodes and edges."""
    return await _get(f"/pipelines/{pipeline_id}")

@mcp.tool()
async def create_pipeline(name: str, nodes: list[dict], edges: list[dict]) -> dict:
    """Create a new pipeline. nodes and edges follow ReactFlow JSON format."""
    return await _post("/pipelines", {"name": name, "nodes": nodes, "edges": edges})

@mcp.tool()
async def update_pipeline(pipeline_id: str, nodes: list[dict], edges: list[dict], name: str | None = None) -> dict:
    """Update an existing pipeline's nodes and edges."""
    body = {"nodes": nodes, "edges": edges}
    if name:
        body["name"] = name
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.put(f"{BASE}/pipelines/{pipeline_id}", json=body)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()

@mcp.tool()
async def delete_pipeline(pipeline_id: str) -> dict:
    """Delete a pipeline."""
    return await _delete(f"/pipelines/{pipeline_id}")

# ── Execution tools ───────────────────────────────────────────────────────────

@mcp.tool()
async def run_pipeline(pipeline_id: str, mode: str = "sequential") -> dict:
    """Start executing a pipeline. mode: 'sequential' or 'multiprocess'. Returns session_id."""
    return await _post("/execution/start", {"pipeline_id": pipeline_id, "mode": mode})

@mcp.tool()
async def stop_pipeline(session_id: str) -> dict:
    """Stop a running pipeline session."""
    return await _post(f"/execution/stop/{session_id}")

@mcp.tool()
async def get_status(session_id: str) -> dict:
    """Get status of a pipeline session (running/stopped/completed/error)."""
    return await _get(f"/execution/status/{session_id}")

@mcp.tool()
async def get_logs(session_id: str, tail: int = 100) -> str:
    """Get last N lines of engine logs for a session."""
    data = await _get(f"/execution/logs/{session_id}?tail={tail}")
    if isinstance(data, list):
        return "\n".join(data)  # API returns list[str] directly
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    return str(data)

@mcp.tool()
async def get_node_stats(session_id: str) -> dict:
    """Get per-node fps/latency stats for a session."""
    return await _get(f"/execution/stats/{session_id}")

@mcp.tool()
async def list_sessions() -> list[str]:
    """List all active session IDs."""
    return await _get("/execution/sessions")

# ── Model tools ───────────────────────────────────────────────────────────────

@mcp.tool()
async def list_models(task: str | None = None) -> list[dict]:
    """List available models. Optionally filter by task (detection/segmentation/pose/face_detect/face_embed)."""
    path = "/models"
    if task:
        path += f"?task={task}"
    return await _get(path)

@mcp.tool()
async def get_system_info() -> dict:
    """Get CPU, RAM, and GPU info of the host running CV-FLOW."""
    return await _get("/system/info")

# ── MCP Resources ─────────────────────────────────────────────────────────────

@mcp.resource("cvflow://pipelines")
async def resource_pipelines() -> str:
    """List of all pipelines as JSON."""
    data = await _get("/pipelines")
    return json.dumps(data, indent=2)

@mcp.resource("cvflow://pipeline/{pipeline_id}")
async def resource_pipeline(pipeline_id: str) -> str:
    """Full JSON content of a specific pipeline."""
    data = await _get(f"/pipelines/{pipeline_id}")
    return json.dumps(data, indent=2)

@mcp.resource("cvflow://session/{session_id}/logs")
async def resource_logs(session_id: str) -> str:
    """Engine logs for a session."""
    data = await _get(f"/execution/logs/{session_id}?tail=200")
    if isinstance(data, list):
        return "\n".join(data)
    return str(data)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # stdio transport
```

---

### 3.4 Claude Desktop Config

Sau khi implement, user thêm vào `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "cv-flow": {
      "command": "python",
      "args": ["C:/Users/trandangan/CV-FLOW/backend/mcp_server.py"]
    }
  }
}
```

---

### Verification — Part 3

- [ ] `pip install mcp` → no errors
- [ ] `python backend/mcp_server.py` → khởi động không crash
- [ ] `GET /api/system/info` → trả CPU/RAM data (psutil installed)
- [ ] Claude Desktop config → Claude nhận tools: `list_pipelines`, `run_pipeline`, `get_logs`, `list_models`, `get_system_info`
- [ ] "liệt kê pipeline" → `list_pipelines()` trả kết quả đúng
- [ ] "chạy pipeline X" → `run_pipeline()` → session_id trả về
- [ ] "xem logs" → `get_logs(session_id)` → engine stdout lines (không phải empty string)
- [ ] Backend down → MCP tool trả lỗi user-friendly, không phải Python traceback

---

---

## Part 4: Model Hub Redesign (GitHub-like)

### Context

Model hub hiện tại có 7 columns cơ bản. Các điểm yếu chính:
- Không có version history (chỉ có version string, không có lineage)
- ONNX I/O đã được introspect khi load nhưng không expose ra API
- Model không phải là node với ports rõ ràng
- Không có search/filter
- Models với nhiều input, dynamic shapes, custom dtype bị handle sai

---

### 4.1 CRITICAL: Schema Migration với Alembic

**Vấn đề**: `model_registry` đã tồn tại với 7 columns. Cần thêm ~10 columns mới. `create_all` sẽ không alter existing table.

**Giải pháp: Alembic migration** (alembic đã có trong requirements.txt):

```bash
# Nếu Alembic chưa được init:
alembic init migrations  # (hoặc kiểm tra migrations/ folder đã có gì)

# Tạo migration mới:
alembic revision --autogenerate -m "add_model_registry_v2_columns"
```

Migration script sẽ chứa:
```python
def upgrade():
    op.add_column("model_registry", sa.Column("slug", sa.String(128), nullable=True))
    op.add_column("model_registry", sa.Column("tag", sa.String(16), nullable=True, server_default="stable"))
    op.add_column("model_registry", sa.Column("is_latest", sa.Boolean(), nullable=True, server_default="1"))
    op.add_column("model_registry", sa.Column("parent_id", sa.String(36), nullable=True))
    op.add_column("model_registry", sa.Column("changelog", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("description", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("ports_json", sa.Text(), nullable=True))  # nullable: existing records
    op.add_column("model_registry", sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("model_registry", sa.Column("size_bytes", sa.Integer(), nullable=True))
    op.add_column("model_registry", sa.Column("author", sa.String(128), nullable=True))
    op.add_column("model_registry", sa.Column("license", sa.String(64), nullable=True))
    op.add_column("model_registry", sa.Column("extra_meta", sa.Text(), nullable=True))
    # Unique constraint cho slug (nullable nên phải partial index):
    op.create_index("ix_model_registry_slug", "model_registry", ["slug"], unique=False)
    # Backfill slug từ existing records: name + version, replace spaces với hyphens
    op.execute("UPDATE model_registry SET slug = lower(name || '-' || version) WHERE slug IS NULL")

def downgrade():
    for col in ["slug","tag","is_latest","parent_id","changelog","description",
                "ports_json","last_used_at","size_bytes","author","license","extra_meta"]:
        op.drop_column("model_registry", col)
```

> ⚠️ **`ports_json` phải nullable** vì existing records chưa có introspection data. Code phải handle `ports_json = None` → trả `{"inputs": [], "outputs": []}`.

**Chạy migration**: `alembic upgrade head` — phải chạy trước khi start server sau khi deploy.

---

### 4.2 ORM Model Update — `backend/app/models/model_registry.py`

```python
class ModelEntry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_name_version"),)

    # Existing columns (giữ nguyên)
    id:          Mapped[str]           = mapped_column(String(36), primary_key=True)
    name:        Mapped[str]           = mapped_column(String(128), nullable=False)
    version:     Mapped[str]           = mapped_column(String(32), nullable=False)
    task:        Mapped[str]           = mapped_column(String(32), nullable=False)
    file_path:   Mapped[str]           = mapped_column(Text, nullable=False)
    config_json: Mapped[str]           = mapped_column(Text, nullable=False)
    uploaded_at: Mapped[datetime]      = mapped_column(DateTime(timezone=True), default=_utcnow)

    # NEW columns (tất cả nullable để không break existing data)
    slug:          Mapped[str|None]      = mapped_column(String(128), nullable=True, index=True)
    tag:           Mapped[str|None]      = mapped_column(String(16), nullable=True, default="stable")
    # tag values: "stable" | "experimental" | "deprecated"
    is_latest:     Mapped[bool|None]     = mapped_column(Boolean, nullable=True, default=True)
    parent_id:     Mapped[str|None]      = mapped_column(String(36), ForeignKey("model_registry.id"), nullable=True)
    changelog:     Mapped[str|None]      = mapped_column(Text, nullable=True)
    description:   Mapped[str|None]      = mapped_column(Text, nullable=True)
    ports_json:    Mapped[str|None]      = mapped_column(Text, nullable=True)
    # ports_json = None → use {} as default. Always handle None in code.
    last_used_at:  Mapped[datetime|None] = mapped_column(DateTime(timezone=True), nullable=True)
    size_bytes:    Mapped[int|None]      = mapped_column(Integer, nullable=True)
    author:        Mapped[str|None]      = mapped_column(String(128), nullable=True)
    license:       Mapped[str|None]      = mapped_column(String(64), nullable=True)
    extra_meta:    Mapped[str|None]      = mapped_column(Text, nullable=True)
```

---

### 4.3 `ports_json` Schema — The Critical New Field

```json
{
  "inputs": [
    {
      "name": "image",
      "tensor_name": "images",
      "type": "image",
      "shape": [1, 3, 640, 640],
      "dtype": "float32",
      "dynamic_axes": [0, 2, 3],
      "desc": "Preprocessed BGR image, normalized 0-1"
    }
  ],
  "outputs": [
    {
      "name": "detections",
      "tensor_name": "output0",
      "type": "detections",
      "shape": [1, 84, 8400],
      "dtype": "float32",
      "desc": "Raw YOLO detection output [batch, 4+classes, anchors]"
    }
  ]
}
```

**Port types enum** (8 semantic types):

| type | Semantic | Example shape |
|------|----------|---------------|
| `image` | BGR/RGB frame tensor | `[1,3,H,W]` |
| `tensor` | Raw float tensor (unknown) | any |
| `detections` | Bounding box array | `[1,N,6]` or `[1,6,N]` |
| `embeddings` | Feature vector(s) | `[1,512]` |
| `mask` | Binary/multi-class mask | `[1,C,H,W]` |
| `keypoints` | Pose landmarks | `[1,17,3]` |
| `class_scores` | Classification softmax output | `[1,1000]` |
| `landmarks` | Face 5-point landmarks | `[1,5,2]` |

---

### 4.4 ONNX Auto-Introspection — `engine/model_hub/onnx_inspector.py` [NEW]

**Thêm vào `engine/requirements.txt`**:
```
onnx>=1.16
```

```python
"""
ONNX model introspection — extract I/O metadata and auto-guess semantic port types.
"""
import onnx
from onnx import TensorProto
from pathlib import Path


_ONNX_DTYPE_MAP = {
    TensorProto.FLOAT:   "float32",
    TensorProto.DOUBLE:  "float64",
    TensorProto.INT32:   "int32",
    TensorProto.INT64:   "int64",
    TensorProto.UINT8:   "uint8",
    TensorProto.INT8:    "int8",
    TensorProto.BOOL:    "bool",
    TensorProto.STRING:  "string",
}


def _extract_shape(tensor_type_shape) -> list[int]:
    """Extract shape as list[int], using -1 for dynamic dims."""
    if tensor_type_shape is None:
        return []
    dims = tensor_type_shape.dim
    result = []
    for d in dims:
        if d.HasField("dim_param"):
            result.append(-1)  # symbolic / dynamic
        elif d.HasField("dim_value"):
            result.append(d.dim_value)
        else:
            result.append(-1)
    return result


def _onnx_dtype_to_str(elem_type: int) -> str:
    return _ONNX_DTYPE_MAP.get(elem_type, f"unknown({elem_type})")


def _friendly_name(tensor_name: str, fallback: str) -> str:
    """Convert tensor name like 'images' or 'output0' to friendly port name."""
    n = tensor_name.strip().lower()
    if not n or n.startswith("/"):
        return fallback
    return n


def _guess_input_type(name: str, shape: list[int], dtype: str) -> str:
    """Heuristic: guess semantic type for an input tensor."""
    positive = [d for d in shape if d > 0]
    name_lower = name.lower()
    if dtype in ("float32", "float64") and len(shape) == 4:
        return "image"
    if len(positive) == 1 and 128 <= positive[-1] <= 2048:
        return "embeddings"
    return "tensor"


def _guess_output_type(name: str, shape: list[int], dtype: str, total_outputs: int) -> str:
    """
    Heuristic: guess semantic type for an output tensor.

    Precedence (order matters):
    1. Specific shape patterns checked before generic ones
    2. Name-based hints only used as tiebreaker

    Known edge cases:
    - ResNet output [1, 2048] → embeddings (not class_scores), because dim > 1000
    - MobileFaceNet output [1, 128] → embeddings (< 1000 but embedding model)
    - Use total_outputs context: single output detection models are "detections" if shape matches
    """
    if not shape:
        return "tensor"
    positive = [d for d in shape if d > 0]
    name_lower = name.lower()

    # 4D → mask (segmentation proto or feature map)
    if len(shape) == 4:
        return "mask"

    # 3D patterns
    if len(shape) == 3:
        last = shape[-1] if shape[-1] > 0 else None
        mid  = shape[-2] if len(shape) >= 2 and shape[-2] > 0 else None
        # Pose: [..., 17, 3]
        if mid == 17 and last == 3:
            return "keypoints"
        # Landmarks: [..., 5, 2] or [..., 10, 2]
        if last == 2 and mid in (5, 10):
            return "landmarks"
        # Detection: last dim is class+bbox count OR bbox+class
        if last in (4, 5, 6, 85, 116) or (last is not None and last < 200):
            return "detections"
        if mid is not None and mid in (4, 5, 6, 85, 116):
            return "detections"
        return "tensor"

    # 2D patterns (or 1D after squeeze)
    if len(positive) == 1:
        dim = positive[0]
        # Large embedding dims (typical: 128, 256, 512, 1024, 2048)
        if dim in (64, 128, 256, 512, 1024, 2048) or (128 <= dim <= 2048 and dim not in range(1001)):
            return "embeddings"
        # Classification softmax
        if dim <= 1000:
            return "class_scores"
        # Larger embedding (e.g. 4096)
        return "embeddings"

    # name-based fallback
    if "embed" in name_lower or "feature" in name_lower or "repr" in name_lower:
        return "embeddings"
    if "cls" in name_lower or "class" in name_lower or "score" in name_lower:
        return "class_scores"
    if "det" in name_lower or "box" in name_lower or "pred" in name_lower:
        return "detections"
    if "kpt" in name_lower or "pose" in name_lower:
        return "keypoints"
    if "mask" in name_lower or "seg" in name_lower:
        return "mask"

    return "tensor"


def inspect_onnx(onnx_path: str | Path) -> dict:
    """
    Load ONNX and extract I/O metadata + auto-guess semantic port types.
    Returns ports_json-compatible dict.

    Handles:
    - Dynamic shapes (dim_param → -1)
    - INT8 quantized models → dtype: "int8", warns preprocessing
    - Multiple inputs (initializers filtered out)
    - Opset < 11 → flagged as legacy
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Filter out initializers from inputs (they are weights, not runtime inputs)
    initializer_names = {init.name for init in graph.initializer}

    inputs = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue  # skip weight tensors
        t = inp.type.tensor_type
        shape  = _extract_shape(t.shape)
        dtype  = _onnx_dtype_to_str(t.elem_type)
        dyn    = [i for i, d in enumerate(shape) if d == -1]
        ptype  = _guess_input_type(inp.name, shape, dtype)
        entry  = {
            "name":          _friendly_name(inp.name, f"input_{len(inputs)}"),
            "tensor_name":   inp.name,
            "type":          ptype,
            "shape":         shape,
            "dtype":         dtype,
            "dynamic_axes":  dyn,
            "desc":          "",
        }
        if dtype == "int8":
            entry["warning"] = "INT8 quantized — preprocessing may need dequantization"
        inputs.append(entry)

    outputs = []
    for out in graph.output:
        t = out.type.tensor_type
        shape = _extract_shape(t.shape)
        dtype = _onnx_dtype_to_str(t.elem_type)
        ptype = _guess_output_type(out.name, shape, dtype, len(graph.output))
        outputs.append({
            "name":        _friendly_name(out.name, f"output_{len(outputs)}"),
            "tensor_name": out.name,
            "type":        ptype,
            "shape":       shape,
            "dtype":       dtype,
            "optional":    False,
            "desc":        "",
        })

    # Check opset version
    opset = model.opset_import[0].version if model.opset_import else 0
    meta = {
        "opset":       opset,
        "ir_version":  model.ir_version,
    }
    if opset < 11:
        meta["warning"] = f"Opset {opset} < 11 — may have compatibility issues with onnxruntime"

    return {
        "inputs":  inputs,
        "outputs": outputs,
        "meta":    meta,
    }
```

---

### 4.5 API Redesign — `backend/app/api/models.py`

**Endpoints mới/sửa**:

```
# List với filter
GET  /api/models?task=&q=&tag=&sort=&include_deprecated=false

# Detail + ports
GET  /api/models/{id}       — trả full detail + ports_json

# Versions
GET  /api/models/versions/{slug}     — all versions của model family

# Upload flow
POST /api/models/inspect    — upload ONNX, trả introspection (không save)
POST /api/models/upload     — upload + ports_json reviewed by user

# Lifecycle
PUT  /api/models/{id}/ports — update port definitions
PUT  /api/models/{id}/tag   — set tag: stable/experimental/deprecated
POST /api/models/{id}/fork  — tạo version mới từ existing

# Catalog (đổi tên từ /defaults và /face)
GET  /api/models/catalog    — unified catalog: YOLO + Face với metadata đầy đủ
POST /api/models/catalog/{key}/download  — download + auto-introspect + register

# Existing (giữ nguyên)
GET  /api/models/{id}/download
POST /api/models/{id}/reload
DELETE /api/models/{id}
```

**`GET /api/models` query params**:
```python
@router.get("")
async def list_models(
    task: str | None = None,
    q: str | None = None,
    tag: str | None = None,
    sort: str = "uploaded_at",          # "name" | "uploaded_at" | "last_used_at" | "size"
    include_deprecated: bool = False,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ModelEntry)
    if task:
        stmt = stmt.where(ModelEntry.task == task)
    if q:
        stmt = stmt.where(
            (ModelEntry.name.ilike(f"%{q}%")) |
            (ModelEntry.description.ilike(f"%{q}%"))
        )
    if tag:
        stmt = stmt.where(ModelEntry.tag == tag)
    if not include_deprecated:
        stmt = stmt.where(ModelEntry.tag != "deprecated")
    # sort
    sort_col = {
        "name": ModelEntry.name,
        "uploaded_at": ModelEntry.uploaded_at,
        "last_used_at": ModelEntry.last_used_at,
        "size": ModelEntry.size_bytes,
    }.get(sort, ModelEntry.uploaded_at)
    stmt = stmt.order_by(sort_col.desc())
    ...
```

**`POST /api/models/inspect`** — upload ONNX, trả introspection:
```python
@router.post("/inspect")
async def inspect_model(
    model_file: UploadFile = File(..., description=".onnx model file"),
):
    """Introspect an ONNX file and return auto-detected port definitions (without saving)."""
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        shutil.copyfileobj(model_file.file, tmp)
        tmp_path = tmp.name
    try:
        from engine.model_hub.onnx_inspector import inspect_onnx
        return inspect_onnx(tmp_path)
    except Exception as e:
        raise HTTPException(400, f"ONNX inspection failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

**`POST /api/models/upload`** — upload với ports_json:
```python
@router.post("/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    config_file: UploadFile = File(...),
    ports_json: str = Form(None),  # JSON string, optional (user can skip)
):
    config_data = json.loads(await config_file.read())
    # Validate required fields
    required = {"name", "version", "task", "format", "input_shape", "output_shapes"}
    missing = required - config_data.keys()
    if missing:
        raise HTTPException(422, f"config.json missing: {missing}")
    if config_data.get("format") != "onnx":
        raise HTTPException(422, "Only ONNX format is supported")

    model_id = str(uuid.uuid4())
    # ... save files ...

    # Auto-introspect nếu user không cung cấp ports_json
    if not ports_json:
        try:
            from engine.model_hub.onnx_inspector import inspect_onnx
            ports = inspect_onnx(onnx_path)
            ports_json_str = json.dumps(ports)
        except Exception:
            ports_json_str = None
    else:
        ports_json_str = ports_json

    row = ModelEntry(
        id=model_id,
        name=config_data["name"],
        version=config_data["version"],
        task=config_data["task"],
        file_path=str(onnx_path),
        config_json=json.dumps(config_data),
        ports_json=ports_json_str,
        size_bytes=onnx_path.stat().st_size,
        slug=f"{config_data['name'].lower().replace(' ', '-')}-{config_data['version']}",
    )
    ...
```

---

### 4.6 `pipeline_builder.py` — Hỗ Trợ `model_node` [CRITICAL]

**Vấn đề**: Hiện tại `pipeline_builder.py` hardcode chỉ support `python_node`/`cpp_node`, skip mọi type khác.

**Giải pháp**: `model_node` sẽ được convert thành `PythonCodeNode` với code được generate tự động từ `ports_json`. Không cần class mới — reuse PythonCodeNode infrastructure.

**Cơ chế**:

Khi `pipeline_builder.py` gặp `node_type == "model_node"`:
1. Lookup `config.model_id` → đọc `config.json` trong `storage/models/{model_id}/`
2. Đọc `ports_json` → biết input/output tensor names
3. Generate Python code tương đương `model_inference` node với named outputs
4. Tạo `PythonCodeNode` với generated code

```python
# Trong pipeline_builder.py, thêm function:

def _generate_model_node_code(model_id: str, ports_json: dict) -> str:
    """Generate PythonCodeNode code for a model_node based on its ports_json."""
    models_dir = os.environ.get("CVFLOW_MODELS_DIR", "storage/models")
    input_port  = ports_json["inputs"][0]["tensor_name"]  if ports_json.get("inputs")  else "images"
    output_defs = ports_json.get("outputs", [])

    output_writes = "\n    ".join(
        f"ctx.metadata['{o['name']}'] = outputs['{o['tensor_name']}']"
        for o in output_defs
    )

    return f'''
import onnxruntime as ort
import numpy as np

def setup():
    global session
    import json
    from pathlib import Path
    model_id = config.get("model_id", "")
    models_dir = config.get("models_dir", "{models_dir}")
    onnx_path = Path(models_dir) / model_id / "model.onnx"
    session = ort.InferenceSession(str(onnx_path))

def loop(frame):
    global session
    # Default preprocessing: BGR uint8 → float32 normalized
    h, w = frame.shape[:2]
    inp = frame.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]  # HWC → NCHW
    outputs_raw = session.run(None, {{"{input_port}": inp}})
    output_names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(output_names, outputs_raw))
    {output_writes}
    return frame  # pass frame downstream unchanged

def teardown():
    pass
'''

# Trong build_pipeline():
elif node_type == "model_node":
    model_id = node_config.get("model_id")
    if not model_id:
        logger.warning("model_node %s missing model_id — skipping", nid)
        continue
    # Load ports_json from model config file
    models_dir = os.environ.get("CVFLOW_MODELS_DIR", "storage/models")
    cfg_path = Path(models_dir) / model_id / "config.json"
    ports_json = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            ports_raw = cfg.get("ports_json", "{}")
            if isinstance(ports_raw, str):
                ports_json = json.loads(ports_raw)
            elif isinstance(ports_raw, dict):
                ports_json = ports_raw
        except Exception:
            logger.warning("model_node %s: failed to load ports_json", nid)
    generated_code = _generate_model_node_code(model_id, ports_json)
    node_config = {**node_config, "code": generated_code, "mode": "loop", "models_dir": models_dir}
    instance = PythonCodeNode()
```

> **Lý do chọn code generation thay vì class mới**: Tái sử dụng toàn bộ PythonCodeNode infrastructure (timing, error handling, viz, teardown). Không cần sửa pipeline_runner hay pipeline_runner_mp. Đơn giản và maintainable hơn.

---

### 4.7 Model-as-Node trong Frontend

**Khi user click "Use in pipeline"** trong Model Hub sidebar:

1. Frontend đã có model detail (fetch từ `GET /api/models/{id}` khi expand card)
2. Parse `ports_json` từ model detail
3. Create node với type `"model_node"` và ports từ `ports_json`:

```js
// Trong app.js — hàm handleUseModel(model)
const ports = {
  inputs:  (model.ports?.inputs  ?? []).map(p => ({ id: p.name, label: p.name })),
  outputs: (model.ports?.outputs ?? []).map(p => ({ id: p.name, label: p.name })),
};
if (!ports.inputs.length)  ports.inputs  = [{ id: "in",  label: "in"  }];
if (!ports.outputs.length) ports.outputs = [{ id: "out", label: "out" }];

addNode({
  type: "model_node",
  label: `${model.name} v${model.version}`,
  config: {
    model_id: model.id,
    mode: "loop",
  },
  ports,
});
```

`model_node` phải được đăng ký trong `nodes.js`:
```js
export const NODE_META = {
  python_node: { group: "core", icon: "🐍", label: "Python Node" },
  cpp_node:    { group: "core", icon: "⚙️", label: "C++ Node" },
  model_node:  { group: "core", icon: "🤖", label: "Model Node" },  // ← THÊM
};
```

**Stale ports problem**: Nếu model được update ports sau khi node đã add → stale. Solution: `model_node` config lưu `model_id` + `ports` snapshot. Engine dùng actual file — frontend chỉ dùng ports cho display/validation.

---

### 4.8 Frontend Model Hub UI Redesign

Redesign Model Hub sidebar tab trong `app.js`:

```
┌─── Model Hub ─────────────────────────┐
│ [🔍 Search models...]                 │
│ [All][Detection][Seg][Pose][Emb][Cls] │  ← task filter chips
│ [stable ▾]  Sort: [Last used ▾]       │
├───────────────────────────────────────┤
│ ┌───────────────────────────────────┐ │
│ │ 🟢 YOLOv8n Detection  v1.0.0     │ │  ← stable badge
│ │ ◉→[image] ·········→ [det]→◉    │ │  ← port preview
│ │ 640×640 · 6.2MB · float32 · ONNX │ │
│ │ [Use in pipeline] [Versions ▾]   │ │
│ └───────────────────────────────────┘ │
│ ┌───────────────────────────────────┐ │
│ │ 🟡 MobileFaceNet  v1.0.0         │ │  ← experimental
│ │ ◉→[image] ·········→ [emb]→◉    │ │
│ │ 112×112 · 4.1MB                  │ │
│ │ [Use in pipeline] [Versions ▾]   │ │
│ └───────────────────────────────────┘ │
│ [+ Upload Model]  [Browse Catalog]    │
└───────────────────────────────────────┘
```

**Badge colors**: `🟢` stable / `🟡` experimental / `🔴` deprecated

**"Versions ▾" dropdown**:
```
v1.0.0 (stable) ← current
v0.9.0 (deprecated)
[Fork to new version]
```

**Upload flow** (2-step):
1. Upload ONNX → `POST /api/models/inspect` → hiện introspection result
2. User review/edit port names/types → `POST /api/models/upload` với ports_json đã review

---

### 4.9 Testing Plan

**`engine/requirements.txt`** — thêm:
```
onnx>=1.16
pytest>=8.0
```

**Unit tests — `tests/test_onnx_inspector.py`**:
```python
# Test 1: Standard YOLO detection — 1 input [1,3,640,640], 1 output [1,84,8400]
#   Expected: input type="image", output type="detections"
# Test 2: YOLO segmentation — 1 input, 2 outputs ([1,84,8400] + [1,32,160,160])
#   Expected: output[0] type="detections", output[1] type="mask"
# Test 3: YOLO pose — output [1,56,8400]
#   Expected: output type="detections" (contains bbox+keypoints combined)
# Test 4: SCRFD face detection — 1 input, 6+ outputs (cls+bbox+kps per stride)
#   Expected: outputs type="detections" or "landmarks" based on shape
# Test 5: ArcFace embedding — output [1,512]
#   Expected: output type="embeddings"
# Test 6: Dynamic shapes — input [B,-1,-1,3] → dynamic_axes=[0,1,2]
#   Expected: shape=[-1,-1,-1,3], dynamic_axes=[0,1,2]
# Test 7: INT8 quantized — dtype should be "int8", warning present
# Test 8: Multi-input model — 2 inputs, initializers filtered out
#   Expected: only runtime inputs returned, not weight tensors
# Test 9: Classification ResNet — output [1,1000]
#   Expected: type="class_scores"
# Test 10: ResNet-50 penultimate layer output [1,2048]
#   Expected: type="embeddings" (dim > 1000, in embedding range)
# Test 11: Opset < 11 — meta.warning present
# Test 12: Face embed MobileFaceNet output [1,128]
#   Expected: type="embeddings" (even though 128 < 1000, it's in embedding dims set)
```

**Integration tests — `tests/test_model_api.py`**:
```python
# Test upload custom ONNX → inspect returns inputs/outputs with shapes
# Test port type auto-detection matches expected for YOLO det
# Test fork: create v2 from v1, verify parent_id set, is_latest updated
# Test tag update: mark as deprecated → not in default list, in list with include_deprecated=True
# Test search: ?q=yolo&task=detection → only detection models returned
# Test model-as-node: Use in pipeline → ports match model ports_json
# Test upload without ports_json → auto-introspected ports attached
```

**Engine integration — `tests/test_model_node_e2e.py`**:
```python
# Test model_node với YOLO det → ctx.metadata["detections"] populated
# Test model_node với face embed → ctx.metadata["embeddings"] populated
# Test pipeline: model_node → passthrough → verify frame unchanged
# Test model_node với invalid model_id → graceful error, không crash pipeline
```

---

### 4.10 MCP Tools for Model Hub

Thêm vào `backend/mcp_server.py`:
```python
@mcp.tool()
async def get_model_ports(model_id: str) -> dict:
    """Get input/output port definitions for a model — tensor names, shapes, dtypes."""
    data = await _get(f"/models/{model_id}")
    return data.get("ports", {}) if isinstance(data, dict) else data

@mcp.tool()
async def list_model_versions(slug: str) -> list[dict]:
    """List all versions of a model family by slug."""
    return await _get(f"/models/versions/{slug}")

@mcp.tool()
async def set_model_tag(model_id: str, tag: str) -> dict:
    """Set model lifecycle tag: 'stable', 'experimental', or 'deprecated'."""
    if tag not in ("stable", "experimental", "deprecated"):
        return {"error": "tag must be one of: stable, experimental, deprecated"}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.put(f"{BASE}/models/{model_id}/tag", json={"tag": tag})
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()

@mcp.tool()
async def download_model(catalog_key: str) -> dict:
    """Download a model from the built-in catalog and register it."""
    return await _post(f"/models/catalog/{catalog_key}/download")
```

---

### Verification — Part 4

- [ ] `alembic upgrade head` → no errors, `model_registry` has new columns
- [ ] Existing records → `ports_json = NULL`, không crash API
- [ ] `GET /api/models` → trả `slug`, `tag`, `ports_json` fields
- [ ] `POST /api/models/inspect` + upload YOLO ONNX → trả inputs/outputs đúng
- [ ] Upload YOLO seg → auto-detect 2 outputs: `detections` + `mask`
- [ ] Upload ArcFace → auto-detect output type `embeddings` shape [1,512]
- [ ] Upload ResNet-50 penultimate → type `embeddings` (không nhầm là class_scores)
- [ ] `?q=yolo&task=detection` → chỉ trả YOLO det models
- [ ] `PUT /api/models/{id}/tag` → deprecated model không xuất hiện trong default list
- [ ] Click "Use in pipeline" → tạo `model_node` với ports đúng
- [ ] Run pipeline với `model_node` → engine không skip node, output populated
- [ ] `pytest tests/test_onnx_inspector.py -v` → 12/12 pass
- [ ] `pytest tests/test_model_api.py -v` → tất cả pass

---

---

## Part 5: Sample Pipeline + Benchmark Node + Data Hub Seed

### 5.1 Clarification: Node System Architecture

**QUAN TRỌNG** trước khi viết sample pipeline:

Hệ thống hiện tại (`pipeline_builder.py` + `node_registry.py`) chỉ support `python_node` và `cpp_node`. Các node như `counter`, `draw_roi`, `face_detect`, v.v. trong `engine/nodes/` có `@register` decorator nhưng decorator này là **no-op** (legacy shim). Chúng không được load bởi pipeline_builder.

**Do đó, sample pipeline "Face ROI Counter" KHÔNG thể dùng type `face_detect`, `draw_roi`, `counter` trực tiếp**. Thay vào đó, phải dùng `python_node` với code gọi đến các class đó, **hoặc** phải sửa `pipeline_builder.py` để load engine nodes từ registry.

**Lựa chọn được chọn**: Extend `pipeline_builder.py` để support registered node types bằng dynamic import + registry. Đây là groundwork cần thiết cho `model_node` và sample pipeline.

**Sửa `pipeline_builder.py`**:
```python
# Thêm node registry loader
_REGISTERED_NODES: dict[str, type] = {}

def _discover_nodes() -> None:
    """Import all engine node modules để trigger @register decorators."""
    # Re-implement registry: decorator lưu class vào _REGISTERED_NODES
    import importlib, pkgutil, engine.nodes
    for finder, name, ispkg in pkgutil.walk_packages(
        path=engine.nodes.__path__,
        prefix="engine.nodes.",
        onerror=lambda x: None,
    ):
        try:
            importlib.import_module(name)
        except Exception as e:
            logger.debug("Failed to import %s: %s", name, e)

# Thực ra @register hiện là no-op → cần sửa node_registry.py để thực sự register:
```

**Sửa `engine/core/node_registry.py`**:
```python
"""
Node registry — maps type string → BaseNode subclass.
@register stores class. get_node_class raises if not found.
"""
_REGISTRY: dict[str, type] = {}

def register(node_type: str):
    """Register a node class under a type string."""
    def decorator(cls):
        _REGISTRY[node_type] = cls
        return cls
    return decorator

def get_node_class(node_type: str):
    if node_type not in _REGISTRY:
        raise ValueError(f"Unknown node type: {node_type!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[node_type]

def get_registry() -> dict[str, type]:
    return dict(_REGISTRY)
```

**Sửa `pipeline_builder.py`** để dùng registry — full updated `build_pipeline()` flow:
```python
from engine.core.node_registry import get_node_class, get_registry
import importlib, pkgutil

_NODES_DISCOVERED = False

def _ensure_nodes_discovered() -> None:
    """Lazy import all engine node modules once to populate @register registry.
    Called at the top of build_pipeline() — idempotent via _NODES_DISCOVERED flag.
    """
    global _NODES_DISCOVERED
    if _NODES_DISCOVERED:
        return
    import engine.nodes  # ensure package importable
    for finder, mod_name, ispkg in pkgutil.walk_packages(
        path=engine.nodes.__path__,
        prefix="engine.nodes.",
        onerror=lambda x: None,
    ):
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            logger.debug("Node discovery: failed to import %s: %s", mod_name, e)
    _NODES_DISCOVERED = True
    logger.info("Node registry populated: %d types", len(get_registry()))

def build_pipeline(pipeline_json: dict) -> list[BaseNode]:
    _ensure_nodes_discovered()  # ← gọi đầu tiên, lazy, idempotent
    _registered = get_registry()  # {type_str: class} sau khi discover
    # ... rest of existing code ...
    for nid in ordered_ids:
        node_data   = node_map[nid]
        node_type   = node_data["type"]
        node_config = node_data.get("config", {})
        if node_type in _MARKER_TYPES:
            continue
        if node_type == "python_node":
            instance = PythonCodeNode()
        elif node_type == "cpp_node":
            instance = CppCodeNode()
        elif node_type == "model_node":
            # Convert model_node to PythonCodeNode with generated code
            instance = _build_model_node(nid, node_config)  # see Part 4.6
            if instance is None:
                continue
        elif node_type in _registered:
            # Registered engine node (face_detect, draw_roi, counter, benchmark, etc.)
            klass    = get_node_class(node_type)
            instance = klass()
        else:
            logger.warning("Unknown node type %r (node %s) — skipping", node_type, nid)
            continue
        instance.setup(node_id=nid, config=node_config, node_type=node_type)
        ordered_nodes.append(instance)
```

> **Tại sao lazy thay vì module-level**: `pipeline_builder.py` được import sớm trong engine startup. Lazy discovery tránh circular import nếu node modules import engine core modules.

---

### 5.2 Face ROI Counter — Corrected Sample Pipeline

**File**: `engine/samples/face_roi_counter.json` + thêm vào `frontend/static/samples.js`

**Xác nhận node types tồn tại** (sau khi sửa registry):
- `face_detect` — `engine/nodes/face/face_detect.py` ✅
- `face_align` — `engine/nodes/face/face_align.py` ✅
- `face_embed` — `engine/nodes/face/face_embed.py` ✅
- `face_vector_db` — `engine/nodes/face/face_vector_db.py` ✅
- `draw_roi` — `engine/nodes/spatial/draw_roi.py` ✅
- `counter` — `engine/nodes/spatial/counter.py` ✅
- `stream_viewer` — `engine/nodes/output/stream_viewer.py` ✅

**Counter node actual API** (từ `counter.py` thực tế):
```python
# trigger_type: "line_cross" | "zone_enter" | "zone_exit"  (KHÔNG phải "roi")
# trigger_id: key để lookup trong ctx.metadata
```
→ Dùng `trigger_type: "zone_enter"` với `trigger_id: "face_roi"` thay vì `mode: "roi"`.

**Pipeline JSON** (corrected):
```json
{
  "id": "face_roi_counter_sample",
  "name": "Face ROI Counter",
  "nodes": [
    {
      "id": "n1", "type": "python_node",
      "config": {
        "mode": "loop",
        "code": "import cv2\ndef setup():\n    global cap\n    idx = config.get('device_index', 0)\n    cap = cv2.VideoCapture(idx)\ndef loop():\n    ret, frame = cap.read()\n    if not ret: raise StopIteration\n    return frame\ndef teardown():\n    cap.release()",
        "device_index": 0
      }
    },
    { "id": "n2", "type": "face_detect",  "config": {"conf_threshold": 0.5} },
    { "id": "n3", "type": "face_align",   "config": {} },
    { "id": "n4", "type": "face_embed",   "config": {} },
    { "id": "n5", "type": "face_vector_db", "config": {"db_dir": "storage/facedb", "threshold": 0.35} },
    {
      "id": "n6", "type": "draw_roi",
      "config": {"polygon": [[100,100],[540,100],[540,380],[100,380]], "label": "Count Zone"}
    },
    {
      "id": "n7", "type": "counter",
      "config": {"trigger_type": "zone_enter", "trigger_id": "face_roi", "label": "Faces in ROI"}
    },
    { "id": "n8", "type": "stream_viewer", "config": {} }
  ],
  "edges": [
    {"id": "e1", "source": "n1", "target": "n2"},
    {"id": "e2", "source": "n2", "target": "n3"},
    {"id": "e3", "source": "n3", "target": "n4"},
    {"id": "e4", "source": "n4", "target": "n5"},
    {"id": "e5", "source": "n5", "target": "n6"},
    {"id": "e6", "source": "n6", "target": "n7"},
    {"id": "e7", "source": "n7", "target": "n8"}
  ]
}
```

---

### 5.3 Benchmark Node — `engine/nodes/utility/benchmark.py` [NEW]

**Requires**: Phải được registered sau khi node_registry được sửa.

```python
"""
BenchmarkNode — pass-through node that records frame timing to CSV.

Config:
  label (str):        identifies this benchmark point, e.g. "after_inference"
  output_path (str):  path to write CSV (default: storage/tmp/benchmark_{session_id}.csv)
  window (int):       rolling window for avg/p95/p99 (default: 100)

Output: ctx unchanged.
CSV columns: frame_no, wall_time_ms, elapsed_since_start_ms, rolling_avg_ms, p95_ms, p99_ms
"""
import csv
import os
import time
from pathlib import Path
from collections import deque
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
import logging

logger = logging.getLogger(__name__)


@register("benchmark")
class BenchmarkNode(BaseNode):

    def initialize(self) -> None:
        self._label      = self.config.get("label", self.node_id)
        self._window_sz  = int(self.config.get("window", 100))
        self._window     = deque(maxlen=self._window_sz)
        self._start_time = time.monotonic()
        self._frame_no   = 0

        session_id   = os.environ.get("CVFLOW_SESSION_ID", "unknown")
        default_path = f"storage/tmp/benchmark_{session_id}_{self._label}.csv"
        out_path     = self.config.get("output_path", default_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        self._csv_file = open(out_path, "w", newline="", buffering=1)
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow(
            ["frame_no", "wall_time_ms", "elapsed_since_start_ms",
             "rolling_avg_ms", "p95_ms", "p99_ms", "label"]
        )
        logger.info("[Benchmark] %s → %s", self._label, out_path)

    def process(self, ctx: FrameContext) -> FrameContext:
        now     = time.monotonic()
        elapsed = (now - self._start_time) * 1000
        self._window.append(elapsed)
        self._frame_no += 1

        sorted_w = sorted(self._window)
        n = len(sorted_w)
        avg  = sum(sorted_w) / n
        p95  = sorted_w[int(n * 0.95)]
        p99  = sorted_w[int(n * 0.99)]

        self._writer.writerow([
            ctx.frame_number,
            round(now * 1000, 1),
            round(elapsed, 2),
            round(avg, 2),
            round(p95, 2),
            round(p99, 2),
            self._label,
        ])
        return ctx

    def teardown(self) -> None:
        """Flush and close CSV file."""
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception as e:
            logger.warning("[Benchmark] %s teardown error: %s", self._label, e)
```

**Cách dùng benchmark**:
```
pipeline: camera → benchmark(label="start") → face_detect → benchmark(label="after_detect") → ...
1. Run với mode=sequential  → CSV timing file A
2. Run với mode=multiprocess → CSV timing file B
3. So sánh p95_ms: multiprocess cải thiện inference stage ≥ 20%
```

**Lưu ý**: Pass `CVFLOW_SESSION_ID` environment variable xuống engine subprocess. Sửa `execution_service.py`:
```python
engine_env = {
    **os.environ,
    "CVFLOW_MODELS_DIR":   str(settings.models_dir),
    "CVFLOW_COMPILED_DIR": str(settings.compiled_dir),
    "CVFLOW_STATS_PATH":   str(stats_path),
    "CVFLOW_SESSION_ID":   session_id,  # ← THÊM
}
```

---

### 5.4 Add Benchmark to samples.js

Thêm sample "Benchmark: Sequential vs Multiprocess" vào `frontend/static/samples.js`:
```js
{
  name: "Benchmark Sequential vs Multiprocess",
  description: "Place BenchmarkNode before and after inference to compare modes. Run twice: sequential then multiprocess. Compare CSV p95_ms.",
  nodes: [
    { id: "cam",  type: "python_node", ... },
    { id: "bm1",  type: "benchmark",   config: { label: "before_inference" } },
    { id: "inf",  type: "face_detect", config: { conf_threshold: 0.5 } },
    { id: "bm2",  type: "benchmark",   config: { label: "after_inference" } },
    { id: "view", type: "stream_viewer", config: {} },
  ],
  edges: [cam→bm1, bm1→inf, inf→bm2, bm2→view],
}
```

---

### 5.5 Data Hub Seed Script — `scripts/seed_datahub.py` [NEW]

```python
"""
Seed Data Hub với test data để demo UI.
Chạy một lần: python scripts/seed_datahub.py
Không cần backend server.
"""
import sqlite3
import json
import uuid
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = "cv_flow.db"
STORAGE = Path("storage")

def seed_relational():
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc)

    # 5 execution sessions
    sessions = [
        (str(uuid.uuid4()), "pipeline-001", (now - timedelta(hours=2)).isoformat(),
         (now - timedelta(hours=1, minutes=50)).isoformat(), "completed", 3600, None, "sequential"),
        (str(uuid.uuid4()), "pipeline-001", (now - timedelta(hours=1)).isoformat(),
         (now - timedelta(minutes=48)).isoformat(), "completed", 2880, None, "multiprocess"),
        (str(uuid.uuid4()), "pipeline-002", (now - timedelta(minutes=30)).isoformat(),
         (now - timedelta(minutes=25)).isoformat(), "error", 0, "ONNX model not found", "sequential"),
        (str(uuid.uuid4()), "pipeline-001", (now - timedelta(minutes=10)).isoformat(),
         None, "running", 0, None, "sequential"),
        (str(uuid.uuid4()), "pipeline-003", (now - timedelta(days=1)).isoformat(),
         (now - timedelta(hours=23)).isoformat(), "completed", 7200, None, "sequential"),
    ]

    conn.executemany("""
        INSERT OR IGNORE INTO execution_sessions
        (id, pipeline_id, started_at, ended_at, status, frame_count, error_msg, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sessions)

    # node_metrics cho 2 completed sessions
    for sid, _, _, _, status, fc, _, mode in sessions:
        if status != "completed": continue
        conn.executemany("""
            INSERT INTO node_metrics (session_id, node_id, avg_ms, p95_ms, fps, errors, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (sid, "face_detect", 45.2, 62.1, 22.0, 0, now.isoformat()),
            (sid, "face_embed",  12.8, 18.4, 22.0, 0, now.isoformat()),
        ])

    # 50 detection events
    event_types = ["face_matched", "face_unknown", "counter_update"]
    for i in range(50):
        event_sid = sessions[0][0]
        conn.execute("""
            INSERT INTO detection_events (session_id, pipeline_id, node_id, timestamp, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event_sid, "pipeline-001", "face_detect",
            (now - timedelta(seconds=50-i)).isoformat(),
            event_types[i % 3],
            json.dumps({"bbox": [100+i,100,200+i,200], "confidence": 0.9, "identity": f"person_{i%5}"})
        ))

    conn.commit()
    conn.close()
    print("✅ Relational DB seeded")

def seed_vector():
    vectordb_dir = STORAGE / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)

    # Collection: faces (empty)
    (vectordb_dir / "faces").mkdir(exist_ok=True)
    np.save(str(vectordb_dir / "faces" / "index.npy"), np.zeros((0, 512), dtype=np.float32))
    (vectordb_dir / "faces" / "meta.json").write_text("[]")

    # Collection: test_embeddings (3 random 512-dim vecs)
    (vectordb_dir / "test_embeddings").mkdir(exist_ok=True)
    vecs = np.random.randn(3, 512).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(str(vectordb_dir / "test_embeddings" / "index.npy"), vecs)
    meta = [
        {"id": "vec_001", "label": "Test Vector 1", "source": "seed"},
        {"id": "vec_002", "label": "Test Vector 2", "source": "seed"},
        {"id": "vec_003", "label": "Test Vector 3", "source": "seed"},
    ]
    (vectordb_dir / "test_embeddings" / "meta.json").write_text(json.dumps(meta, indent=2))
    print("✅ Vector DB seeded: faces (0 vecs), test_embeddings (3 vecs)")

if __name__ == "__main__":
    seed_relational()
    seed_vector()
    print("✅ Data Hub seed complete")
```

---

### Verification — Part 5

- [ ] `python scripts/seed_datahub.py` → no errors
- [ ] Open Data Hub → `execution_sessions` có 5 rows, `detection_events` có 50 rows
- [ ] Browse `detection_events` → bảng render đúng, filter theo event_type hoạt động
- [ ] Vector DB tab → "test_embeddings" hiện 3 records, "faces" hiện 0
- [ ] Export CSV `node_metrics` → download file với đúng columns
- [ ] Sample "Face ROI Counter" xuất hiện trong samples list
- [ ] Load sample → Load thành công, nodes display đúng
- [ ] `benchmark` node: Run pipeline → CSV file xuất hiện trong `storage/tmp/`
- [ ] CSV có đúng columns: frame_no, wall_time_ms, elapsed_since_start_ms, rolling_avg_ms, p95_ms, p99_ms, label
- [ ] BenchmarkNode teardown: kill pipeline đột ngột → CSV file được flush/closed (không bị corrupt)
- [ ] Sequential vs multiprocess benchmark: p95_ms after_inference stage cải thiện ≥ 20%

---

---

## Dependencies Summary (Cần Update)

### `backend/requirements.txt`

```diff
 fastapi==0.111.0
 uvicorn[standard]==0.30.1
 sqlalchemy==2.0.31
 aiosqlite==0.20.0
 alembic==1.13.2
 pydantic==2.7.4
 pydantic-settings==2.3.4
 python-multipart==0.0.9
 jsonschema==4.22.0
 httpx==0.27.0
 python-jose==3.3.0
+mcp>=1.0
+psutil>=5.9
+pynvml>=11.0; platform_system=="Windows" or platform_system=="Linux"
```

### `engine/requirements.txt`

```diff
 opencv-python==4.10.0.84
 numpy==1.26.4
 onnxruntime==1.18.1
 websockets==12.0
 httpx==0.27.0
 paho-mqtt==2.1.0
+onnx>=1.16
+pytest>=8.0
```

---

## Files Summary — Toàn Bộ Plan

| File | Action | Part |
|------|--------|------|
| `frontend/static/nodes.js` | MODIFY | 1 |
| `frontend/static/app.js` | MODIFY | 1, 4 |
| `frontend/static/index.html` | MODIFY (version bump) | 1 |
| `frontend/static/datahub.js` | **NEW** | 2 |
| `frontend/static/samples.js` | MODIFY | 5 |
| `backend/app/services/execution_db.py` | **NEW** | 2 |
| `backend/app/services/execution_service.py` | MODIFY (thread safety + DB wire-up) | 2 |
| `backend/app/models/datahub.py` | **NEW** | 2 |
| `backend/app/models/model_registry.py` | MODIFY (new columns) | 4 |
| `backend/app/api/datahub.py` | **NEW** | 2 |
| `backend/app/api/models.py` | MODIFY (inspect, catalog, fork, versions, tag) | 4 |
| `backend/app/api/system.py` | MODIFY (add GET /system/info) | 3 |
| `backend/app/api/router.py` | MODIFY (add datahub) | 2 |
| `backend/mcp_server.py` | **NEW** | 3 |
| `backend/requirements.txt` | MODIFY | 3 |
| `engine/core/node_registry.py` | MODIFY (make register functional) | 5 |
| `engine/core/pipeline_builder.py` | MODIFY (support registered nodes + model_node) | 4, 5 |
| `engine/data/vector_store.py` | **NEW** | 2 |
| `engine/model_hub/onnx_inspector.py` | **NEW** | 4 |
| `engine/nodes/utility/benchmark.py` | **NEW** | 5 |
| `engine/samples/face_roi_counter.json` | **NEW** | 5 |
| `engine/requirements.txt` | MODIFY | 4 |
| `backend/migrations/versions/{hash}_add_model_registry_v2.py` | **NEW** (Alembic) | 4 |
| `scripts/seed_datahub.py` | **NEW** | 5 |
| `tests/test_onnx_inspector.py` | **NEW** | 4 |
| `tests/test_model_api.py` | **NEW** | 4 |
| `tests/test_model_node_e2e.py` | **NEW** | 4 |

---

## Execution Order (Quan Trọng — Dependencies)

```
Phase 1: Foundations (không dependency)
  └─ engine/core/node_registry.py   ← sửa @register thành functional (Part 5 cần, làm trước)
  └─ backend/app/services/execution_db.py  ← NEW sync DB writer (Part 2)
  └─ engine/model_hub/onnx_inspector.py    ← NEW (Part 4)

Phase 2: Backend Core
  └─ backend/app/models/datahub.py         ← NEW models
  └─ backend/app/models/model_registry.py  ← MODIFY + Alembic migration
  └─ backend/app/services/execution_service.py  ← thread safety + DB wire-up

Phase 3: Backend API
  └─ backend/app/api/system.py    ← thêm /info endpoint
  └─ backend/app/api/datahub.py   ← NEW
  └─ backend/app/api/models.py    ← redesign
  └─ backend/app/api/router.py    ← wire datahub

Phase 4: Engine
  └─ engine/core/pipeline_builder.py  ← support model_node + registered nodes
  └─ engine/data/vector_store.py      ← NEW
  └─ engine/nodes/utility/benchmark.py ← NEW

Phase 5: Frontend
  └─ frontend/static/nodes.js    ← inline params + resources
  └─ frontend/static/app.js      ← save/load resources, dirty fix, Model Hub redesign
  └─ frontend/static/datahub.js  ← NEW Data Hub panel
  └─ frontend/static/samples.js  ← NEW samples
  └─ frontend/static/index.html  ← version bump

Phase 6: Integrations & Scripts
  └─ backend/mcp_server.py       ← NEW MCP server
  └─ scripts/seed_datahub.py     ← NEW seed script

Phase 7: Tests
  └─ tests/test_onnx_inspector.py
  └─ tests/test_model_api.py
  └─ tests/test_model_node_e2e.py
```

---

## End-to-End Verification Checklist

### Part 1: Inline Params + Resources
- [ ] Load sample "Edge Detection" → node card có Resources section
- [ ] Expand Resources, set Max FPS=15 → Save → JSON có `"resources": {"max_fps": 15}`
- [ ] `python_node` với `slider(...)` → Params section với range input chỉnh được
- [ ] Click chọn node → dirty KHÔNG bật
- [ ] Run pipeline → engine log có ResourceLimits applied
- [ ] Reload page → resources persist

### Part 2: Data Hub
- [ ] `scripts/seed_datahub.py` → no errors
- [ ] Open Data Hub tab → thấy "cv_flow.db" với tables + row counts
- [ ] Browse `detection_events` → table render đúng, filter hoạt động
- [ ] Vector DB → "test_embeddings" hiện 3 vectors
- [ ] Export CSV → download file với đúng columns
- [ ] `GET /api/datahub/relational/export/sqlite_master` → HTTP 400
- [ ] Start pipeline → session logged; stop → node_metrics logged

### Part 3: MCP Server
- [ ] `python backend/mcp_server.py` → no errors (stdio waits for input)
- [ ] Claude Desktop config → Claude nhận đủ tools
- [ ] `get_logs()` → trả log lines (không phải empty string)
- [ ] Backend down → MCP tool trả friendly error message
- [ ] `get_system_info()` → trả CPU/RAM data

### Part 4: Model Hub Redesign
- [ ] `alembic upgrade head` → no errors
- [ ] Existing models → không crash API (ports_json nullable)
- [ ] `POST /api/models/inspect` + YOLO ONNX → trả inputs/outputs với shapes
- [ ] YOLO seg → 2 outputs: detections + mask
- [ ] ArcFace → output type embeddings [1,512]
- [ ] ResNet-50 [1,2048] → type embeddings (không phải class_scores)
- [ ] Search/filter hoạt động đúng
- [ ] model_node: drag to canvas → ports đúng; run → engine không skip node
- [ ] `pytest tests/test_onnx_inspector.py -v` → 12/12 pass
- [ ] `pytest tests/test_model_api.py -v` → all pass

### Part 5: Sample + Benchmark
- [ ] Sample "Face ROI Counter" load được (không crash với registered node types)
- [ ] `benchmark` node: CSV xuất hiện, columns đúng
- [ ] Kill pipeline giữa chừng → CSV file không corrupt (teardown flush)
- [ ] Sequential vs multiprocess: p95_ms improvement documented
