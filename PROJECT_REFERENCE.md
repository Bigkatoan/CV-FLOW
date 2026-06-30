# CV-FLOW — Project Reference

Tài liệu tổng hợp: kiến trúc hiện tại, hướng dẫn sử dụng, nguyên tắc viết code, và những gì chưa được kiểm thử.

> Changelog đã chuyển sang [`CHANGELOG.md`](CHANGELOG.md) (chuẩn Keep a Changelog).
> README hướng tới người dùng package nằm ở [`README.md`](README.md). Tài liệu
> hướng tới AI agent (Claude) dùng package này trong dự án khác nằm ở
> [`CLAUDE.md`](CLAUDE.md). File này (`PROJECT_REFERENCE.md`) là tài liệu kiến
> trúc/nguyên tắc nội bộ — phần changelog dưới đây giữ nguyên cho bối cảnh lịch
> sử của bản v0.2.0 rewrite, các thay đổi sau đó xem `CHANGELOG.md`.

---

## 1. Changelog (lịch sử — xem CHANGELOG.md cho các bản sau v0.2.0)

### v0.2.0 — 2026-06-30 — Viết lại toàn bộ theo mô hình Topic-based DAM

Toàn bộ source code cũ (`engine/`, `cv_flow/` bản DAG, frontend ReactFlow, backend FastAPI bản cũ) đã bị **xóa hoàn toàn** và viết lại từ đầu theo yêu cầu chuyển sang mô hình publish/subscribe kiểu ROS2, chạy trên DAM (Direct Access Memory — shared memory RAM/VRAM) thay vì DAG node-graph thuần.

**Đã hoàn thành (build theo đúng thứ tự, mỗi module đều có test suite riêng và pass 100% trước khi sang module tiếp):**

1. `cv_flow/dam/bus.py` — `PortBus`: ring buffer trên POSIX shared memory, ordered queue mặc định, drop+log khi đầy.
2. `cv_flow/dam/round_robin.py` — `RoundRobinBus`: elastic fan-out (1 writer → N readers).
3. `cv_flow/dam/merge.py` — `MergeBus`: elastic fan-in (N writers → 1 reader), sắp xếp lại theo `seq_no` để xử lý race condition khi nhiều worker trả kết quả cùng lúc.
4. `cv_flow/dam/cuda_bus.py` — `CudaPortBus`: kênh VRAM, tự fallback về CPU khi không có GPU.
5. `cv_flow/topic/types.py` — `DTYPE_MAP`, `FieldDef`, `PortDef`, `TopicDef`.
6. `cv_flow/topic/parser.py` — parser cho file `.topic` (không dùng thư viện YAML, tự viết regex).
7. `cv_flow/topic/publisher.py` + `subscriber.py` — đóng gói/giải mã dict/ndarray/Tensor ↔ bytes.
8. `cv_flow/topic/topic.py` — class `Topic` + registry toàn cục (get/list/clear/load).
9. `cv_flow/node.py` + `cv_flow/executor.py` — base class `Node`, `Executor` (lifecycle, hz limiter, SIGINT, elastic monitor).
10. `cv_flow/nodes/_catalog.py` — `NODE_CATALOG`: metadata cho visual editor.
11. 10 file `.topic` mẫu trong `cv_flow/topic_templates/`.
12. Toàn bộ node có sẵn trong `cv_flow/nodes/` (camera, preprocess, inference, postprocess, tracking, draw, output) + integration test end-to-end (video file → detect → draw → ghi video).
13. `cv_flow/__init__.py`, `cv_flow/cli.py` (lệnh `cv-flow run / validate / list-nodes`).
14. Backend API (`backend/app/`): catalog node, CRUD pipeline spec + validate, sinh AI deployment guide (Markdown) từ Pipeline Spec JSON.

**Kết quả kiểm thử tại thời điểm này: 139 passed, 4 skipped (do máy không có GPU), 0 failed.**

**Các bug đã phát hiện và sửa trong quá trình build** (xem thêm trong memory của agent):
- `PortBus.__init__`: header `queue_depth`/`slot_bytes` pack bằng format khác format đọc → sai offset.
- `PortBus.read(timeout_ms=0)`: deadline = now + 0 nên loop không bao giờ chạy, luôn trả `None`.
- `MergeBus`: gọi `read()` (có side-effect tiêu thụ frame) trên mọi bus ứng viên rồi mới chọn cái seq nhỏ nhất → các bus không được chọn bị mất frame vĩnh viễn. → Thêm `PortBus.peek()` (đọc không tiêu thụ) để `MergeBus` chỉ `read()` đúng bus thắng.
- `Subscriber.has_seq_gap()`: so sánh nhầm với seq vừa đọc (đã bị `read()` cập nhật) thay vì seq trước đó → luôn báo gap sai. Sửa bằng cách lưu thêm `_prev_seq`.
- `NMS.run_nms`: heuristic transpose `(features, N) → (N, features)` dựa trên so sánh kích thước 2 chiều — chỉ đúng khi N (số box) > số feature (84); ghi chú rõ giả định này trong code.
- `ObjectTracker`: output có số lượng track biến đổi nhưng topic DAM yêu cầu shape cố định → thêm tham số `max_tracks` để pad giống cách `NMS` pad `max_detections`.
- Plugin pytest của ROS2 Humble (`launch_testing`, `launch_pytest`, `launch_ros`) được global entry-point nạp tự động và làm crash pytest ngay từ lúc khởi động → cài `lark` + thêm `addopts = "-p no:launch_testing -p no:launch_pytest -p no:launch_ros -p no:ament_*"` vào `pyproject.toml`.
- FastAPI 0.111.0: endpoint `DELETE` với `status_code=204` cần thêm `response_model=None`, nếu không bị `AssertionError`.

---

## 2. Hiện tại hệ thống có gì (kiến trúc)

```
cv_flow/
├── __init__.py            # export Node, Executor, Topic, get_topic, load_topics, ...
├── cli.py                 # cv-flow run / validate / list-nodes
├── dam/                   # tầng IPC shared-memory (RAM/VRAM)
│   ├── bus.py              PortBus       — ring buffer 1 writer/1 reader, ordered, drop+log
│   ├── round_robin.py       RoundRobinBus — elastic fan-out
│   ├── merge.py             MergeBus      — elastic fan-in, sort theo seq_no
│   └── cuda_bus.py          CudaPortBus   — kênh VRAM, fallback CPU
├── topic/                 # định nghĩa & I/O cho topic kiểu ROS2
│   ├── types.py             DTYPE_MAP, FieldDef, PortDef, TopicDef
│   ├── parser.py             parse_topic_file(), load_topics_dir()
│   ├── publisher.py          Publisher  (dict/ndarray/Tensor → bytes)
│   ├── subscriber.py         Subscriber (bytes → ndarray/dict/Tensor)
│   └── topic.py              Topic class + registry (get_topic/list_topics/load_topics)
├── node.py                 Node base class (advertise()/subscribe()/initialize()/spin_once()/shutdown())
├── executor.py              Executor (spin loop, hz limiter, SIGINT, elastic monitor; scale_up()/
│                            scale_down() mặc định gọi add_worker()/remove_worker() nếu node có)
├── elastic.py               ElasticStage — elastic auto-scale THẬT (multiprocessing.Process worker
│                            thật, kể từ v0.4.0). Tự quản lý round-robin fan-out + reorder buffer
│                            fan-in qua PortBus + multiprocessing.Lock per-worker (không dùng
│                            RoundRobinBus/MergeBus trực tiếp — xem ghi chú concurrency bên dưới).
├── nodes/                  # node có sẵn
│   ├── _catalog.py          NODE_CATALOG — metadata cho visual editor & deployment guide
│   ├── camera.py             CameraSource, RtspSource, VideoFileSource
│   ├── preprocess.py         Preprocess (letterbox+normalize), GrayscaleConvert
│   ├── inference.py          OnnxInference, YoloInference
│   ├── postprocess.py        NMS (pure-numpy)
│   ├── tracking.py           ObjectTracker (ByteTrackLite — IOU tracker thuần Python)
│   ├── tee.py                 Tee — fan-out 1 topic ra N topic (xem mục 4.10)
│   ├── draw.py               DrawBbox (cv2)
│   └── output.py             VideoWriter, StreamViewer (websocket), MqttPublisher
└── topic_templates/        # 10 file .topic mẫu: camera_frame, depth_frame, yolo_input,
                             # yolo_raw, detections, tracked, annotated_frame, stream_jpeg,
                             # audio_pcm, embedding

backend/app/
├── main.py                 FastAPI app, mount toàn bộ router
├── schemas.py               Pydantic models: TopicSpec, NodeSpec, PipelineSpec, ValidationResult
├── db.py                    SQLAlchemy engine + bảng `pipelines` (JSON blob theo id)
├── pipeline_store.py        PipelineStore — SQLite thật (kể từ v0.3.0), survive restart.
│                            Interface (create/get/update/delete/list_all/clear) không đổi.
├── validator.py             validate_pipeline() — kiểm tra node type, topic reference, required params
├── guide_generator.py       generate_deployment_guide() — PipelineSpec → Markdown
└── api/
    ├── nodes.py              GET /api/nodes, GET /api/nodes/{type}
    ├── topics.py              GET /api/topics/templates
    ├── pipeline.py            POST/GET/PUT/DELETE /api/pipeline[/{id}], POST /api/pipeline/validate
    └── guide.py               POST /api/pipeline/generate-guide

scripts/
├── bench_inference.py       Benchmark thủ công CPU/CUDA/TensorRT latency (không phải pytest)
└── smoke_pipeline.py        Smoke test thủ công full pipeline trên hardware thật (không phải pytest)

tests/                       # mirror cấu trúc cv_flow/ — xem CHANGELOG.md cho số lượng test mới nhất
├── dam/
├── topic/
├── nodes/        (gồm cả test_inference.py, test_camera.py, test_tee.py — dùng model/camera thật)
├── backend/      (gồm test_pipeline_store.py — test persistence SQLite thật)
├── conftest.py   pytest_configure() trỏ DB pipeline_store sang file tạm cho cả phiên test
└── test_*.py ở root: catalog, cli, executor, node
```

**Chưa có / nằm ngoài phạm vi đã build:**
- Không có frontend UI kéo-thả thực sự (chỉ có backend API phục vụ cho UI đó).
- Không có runtime C++ (CPU/GPU) — yêu cầu gốc có nhắc tới multi-runtime (numpy / torch+pycuda / C++ CPU / C++ GPU) nhưng hiện tại **chỉ có runtime Python (numpy / torch)**.
- CUDA IPC zero-copy thật cho `CudaPortBus` — **đã thử thật (Phase 2), xác nhận không khả thi trên
  GPU tích hợp của Jetson** (xem CHANGELOG.md `[0.4.0]`). Đây là kết luận cuối, không phải gap còn
  treo — `CudaPortBus` giữ nguyên CPU-roundtrip, docstring đã sửa cho trung thực.
- Elastic multiprocessing thật: **đã hoàn thành ở Phase 2 (v0.4.0)** qua `cv_flow.elastic.ElasticStage`
  — xem mục 2 (kiến trúc) và CHANGELOG.md.

---

## 3. Hướng dẫn sử dụng

### Cài đặt & chạy test
Trên máy Jetson hiện tại, môi trường dùng là `/home/orin/venv` (không phải venv riêng trong repo)
— xem [`README.md`](README.md) mục "Install → On a Jetson" cho lý do và lệnh cài đặt đầy đủ
(`include-system-site-packages=true`, `pip install -e . --no-deps`, ghim `numpy<2`). Tóm tắt:

```bash
/home/orin/venv/bin/python3 -m pytest tests/ -v          # chạy toàn bộ test suite
/home/orin/venv/bin/python3 -m pytest tests/ -m "not gpu" # bỏ qua test build TensorRT engine (chậm)
```
> Lưu ý: máy này có ROS2 Humble cài global, plugin pytest của nó sẽ làm crash pytest nếu thiếu 2 chỗ fix đã có sẵn trong `pyproject.toml` (`addopts` disable plugin) và venv (`pip install lark`). Không cần set `AMENT_PREFIX_PATH` hay biến môi trường nào khác.

### Định nghĩa một topic (`.topic` file)
```
# topics/camera_frame.topic — topic nguồn (source), chỉ có output
output: -> cpu
   - frame : bgr8 shape=[720, 1280]
   - seq   : uint64
```
```
# topics/yolo_raw.topic — topic transform, elastic (auto-scale N worker)
elastic: true
max_replicas: 4
queue_depth: 8

input: -> cpu
   - tensor : float32 shape=[1, 3, 640, 640]
output: -> cpu
   - raw : float32 shape=[1, 84, 8400]
```
Dtype hỗ trợ: xem `cv_flow/topic/types.py::DTYPE_MAP` (bgr8, rgb8, bgra8, mono8, 16UC1, 32FC1, 32FC3, float16/32/64, int8/16/32/64, uint8/16/32/64, bool).

### Viết một Node
```python
import cv_flow

cv_flow.load_topics("topics/")  # parse + đăng ký toàn bộ *.topic trong thư mục

class MyNode(cv_flow.Node):
    def initialize(self):
        self.sub = self.subscribe("camera_frame")
        self.pub = self.advertise("annotated_frame")

    def spin_once(self):
        frame = self.sub.read(timeout_ms=30)
        if frame is None:
            return
        # ... xử lý ...
        self.pub.write(frame)

executor = cv_flow.Executor([MyNode()], hz=30)
executor.spin()   # blocking; Ctrl+C để dừng, hoặc raise StopIteration trong spin_once()
```

### Dùng node có sẵn
```python
from cv_flow.nodes import VideoFileSource, DrawBbox, VideoWriter

source = VideoFileSource("camera_frame", path="input.mp4")
writer = VideoWriter("annotated_frame", output_path="out.mp4", fps=30.0)
cv_flow.Executor([source, ..., writer]).spin()
```
Xem toàn bộ node + tham số trong `cv_flow/nodes/_catalog.py` (`NODE_CATALOG`), hoặc chạy:
```bash
python -m cv_flow.cli list-nodes
```

### Chạy backend API
```bash
source venv/bin/activate
python -m uvicorn backend.app.main:app --reload --port 8000
```
Endpoint chính: `GET /api/nodes`, `GET /api/topics/templates`, `POST /api/pipeline/validate`, `POST /api/pipeline/generate-guide`, CRUD `/api/pipeline[/{id}]`. Xem schema đầy đủ tại `http://localhost:8000/docs` (Swagger UI tự sinh).

### CLI
```bash
cv-flow validate topics/        # parse + validate toàn bộ .topic trong thư mục
cv-flow list-nodes              # in catalog node có sẵn
cv-flow run launch.py           # chạy script launch (gọi Executor.spin())
```

---

## 4. Nguyên tắc khi viết code (bắt buộc tuân theo)

1. **Test-driven theo từng module**: viết xong 1 file/module → viết test cho nó → chạy pytest → confirm pass → mới sang module tiếp theo. Không batch nhiều module rồi mới test chung.
2. **Ordered queue là default**: `PortBus` mặc định giữ thứ tự ghi, khi đầy thì drop slot cũ nhất + log WARNING + tăng `drop_count`. Chỉ dùng `drop_mode=True` khi cố ý muốn silent newest-wins (ví dụ stream JPEG không cần giữ frame cũ).
3. **Shape topic luôn cố định**: DAM không hỗ trợ dữ liệu độ dài biến đổi. Node có output số lượng thay đổi (NMS, ObjectTracker, v.v.) phải có tham số `max_*` để pad về kích thước cố định, dùng sentinel `class_id = -1` để đánh dấu "không có detection" ở vị trí đó.
4. **Không tiêu thụ (consume) PortBus khi chỉ muốn xem trước**: nếu cần so sánh nhiều bus (như `MergeBus`) trước khi quyết định đọc bus nào, dùng `PortBus.peek()` (không advance `read_count`), chỉ `read()` đúng bus đã chọn.
5. **Bus naming theo session_id**: `Node.advertise()`/`subscribe()` tạo tên shared-memory bus theo dạng `f"{topic_name}_{session_id}"`. `Executor` tự inject `session_id` (uuid) vào mọi node trước khi `initialize()`. Khi viết test thủ công không qua `Executor`, phải tự set `node._session_id` trước khi gọi `initialize()`.
6. **Không dùng thư viện YAML cho `.topic` parser** — giữ parser thuần Python (regex/line-based) như đã thiết kế, để tránh dependency thừa và giữ format đơn giản, dễ đọc bằng mắt.
7. **Không thêm abstraction/feature ngoài yêu cầu**: không tạo class trừu tượng hoá sớm, không viết fallback cho trường hợp không thể xảy ra trong code nội bộ.
8. **FastAPI**: mọi endpoint `DELETE` với `status_code=204` phải có `response_model=None` (bug riêng của FastAPI 0.111.0 đang dùng trong venv này).
9. **Không sửa lại `pyproject.toml`'s `addopts` (ROS2 plugin disable list)** trừ khi thật sự cần thêm plugin mới — đây là fix cho môi trường máy chủ, không phải config tuỳ chọn.
10. **Mỗi topic/bus chỉ có 1 reader thật sự** (FIFO, read cursor nằm trong shared-memory header, dùng chung cho mọi `Subscriber` cùng tên bus): nếu 2 node khác nhau cần cùng 1 dữ liệu nguồn (vd cả `Preprocess` lẫn `DrawBbox` đều cần frame gốc), **không** cho cả 2 cùng `subscribe()` 1 topic — chúng sẽ tranh nhau đọc và rớt dữ liệu. Dùng `Tee` (`cv_flow/nodes/tee.py`) để fan-out ra N topic riêng trước. Bug này chỉ lộ ra khi chạy thật nhiều frame liên tục (test với 1 frame đơn lẻ không phát hiện được) — phát hiện qua `scripts/smoke_pipeline.py` trên hardware thật.
11. **`PortBus` không an toàn khi có writer process và reader process THẬT chạy đồng thời** trên cùng 1 bus — header (`write_count`/`read_count`/`drop_count`) được ghi đè nguyên cụm trong 1 lần gọi `_write_header()`, không có lock cross-process, nên 2 process thao tác song song có thể ghi đè tiến trình đọc/ghi của nhau (gây duplicate hoặc mất frame). Đây là lỗi tồn tại từ bản gốc, chỉ lộ ra khi xây `ElasticStage` (worker process thật đầu tiên đọc/ghi đồng thời với process chính). Mọi pipeline 1-process tuần tự (toàn bộ phần còn lại của codebase) KHÔNG bị ảnh hưởng. Nếu viết code mới có writer/reader THẬT khác process trên cùng 1 `PortBus` (ngoài `ElasticStage`, vốn đã tự xử lý bằng `multiprocessing.Lock` per-worker), phải tự thêm lock tương tự — không sửa `PortBus` dùng chung cho việc này.

---

## 5. Những gì CHƯA được test (gaps)

Cập nhật sau Phase 2 (v0.4.0) — đã verify thật trên Jetson Orin Nano: GPU
(`torch`/`onnxruntime-gpu` thật), `YoloInference`/`OnnxInference` với model
ONNX thật (CPU/CUDA/TensorRT), `CameraSource` với USB camera thật,
`PipelineStore` SQLite sống sót qua restart (giả lập), và **elastic
multiprocessing thật** qua `ElasticStage` (spawn/scale up/scale down/shutdown
+ zero lost/duplicate/misorder frame qua nhiều lần stress test lặp lại —
xem `tests/test_elastic.py`). CUDA IPC zero-copy đã thử thật và xác nhận
không khả thi trên phần cứng này (không phải gap, là kết luận cuối). Chi
tiết xem `CHANGELOG.md` mục `[0.3.0]` và `[0.4.0]`. Các mục còn lại dưới
đây là gap thật sự còn tồn tại:

| Thành phần | Vì sao chưa test | Mức độ rủi ro |
|---|---|---|
| `ElasticStage` với `node_factory` là node GPU thật (vd `YoloInference`) | Test thật mới dùng worker CPU đơn giản (nhân đôi giá trị) để cô lập đúng vấn đề concurrency; chưa test tổ hợp N worker GPU thật chạy song song (CUDA context riêng mỗi worker, có thể tốn VRAM/tranh GPU) | Trung bình — cơ chế lõi (spawn/scale/lock/reorder) đã verify đúng, nhưng chưa đo hiệu năng/ổn định khi worker thật sự nặng |
| `CameraSource(gstreamer_pipeline=...)` với camera CSI vật lý | Board chưa gắn module cảm biến CSI lúc build — chỉ test được chuỗi pipeline sinh ra đúng cú pháp (`build_nvargus_pipeline`), chưa mở camera CSI thật | Trung bình — code đã review kỹ theo cú pháp nvarguscamerasrc chuẩn, nhưng chưa chạy thật |
| `RtspSource` reconnect với stream thật | Backoff logic đã test kỹ bằng mock (doubles/caps/resets đúng), nhưng chưa nối với 1 RTSP server thật để xác nhận hành vi qua mất kết nối mạng thật | Trung bình |
| `StreamViewer` (WebSocket JPEG broadcast) | Chưa viết test có client WebSocket thật kết nối vào, chỉ review logic | Trung bình |
| `MqttPublisher` | Không có MQTT broker trong môi trường test | Trung bình — code path connect/publish chưa từng chạy thật |
| `ObjectTracker` qua video dài thật với occlusion thật | Đã verify qua smoke test thật (100 frame, model thật) nhưng chưa test với video dài/occlusion phức tạp | Thấp-Trung bình |
| Visual editor frontend (kéo-thả) | Không nằm trong scope đã build — chỉ có backend API | N/A — chưa làm |
| C++ CPU/GPU runtime (LibTorch/CUDA) | Không nằm trong scope đã build — chỉ có Python runtime | N/A — chưa làm |

---

## 6. Trạng thái Git

Toàn bộ source code cũ (`engine/`, `cv_flow/` bản DAG, frontend, backend bản cũ, migrations, Dockerfile, v.v.) đã được xóa khỏi working tree và thay bằng cấu trúc mới ở trên. `.gitignore` được tạo lại để loại `__pycache__/`, `venv/`, `.pytest_cache/`, v.v.
