// Pre-built sample pipelines using ONLY python_node / cpp_node / model_node.
// Every CV operation is expressed as user-written Python code that runs inside
// the engine's PythonCodeNode executor.
//
// Built-in helpers available in every python_node:
//   send_frame(img, quality=80)  — broadcast JPEG frame to the frontend viewer
//   send_event(dict)             — push a custom event via WebSocket
//   show_image(img, label="")    — show thumbnail in Properties panel
//   show_text(text)              — show text in Properties panel
//   config                       — the node's config dict (read-only recommended)
//   np                           — numpy (always pre-imported)
//
// Parameter routing in loop()/iteration():
//   "frame"      → ctx.frame  (numpy BGR ndarray)
//   "metadata"   → ctx.metadata  (the shared per-frame dict — mutable!)
//   anything else → ctx.metadata.get(name)

// ── Shared code snippets ───────────────────────────────────────────────────────

const CAM_CODE = `\
import cv2

def setup():
    global cap
    idx = config.get("device_index", 0)
    w   = config.get("width", 1280)
    h   = config.get("height", 720)
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {idx}")

def loop():
    ret, frame = cap.read()
    if not ret:
        raise StopIteration("Camera disconnected")
    return frame

def teardown():
    global cap
    if cap is not None:
        cap.release()
`;

const STREAM_CODE = `\
def loop(frame):
    if frame is not None:
        send_frame(frame)
    return frame
`;

const EDGE_CODE = `\
import cv2

def setup():
    global t1, t2
    t1 = config.get("threshold1", 50)
    t2 = config.get("threshold2", 150)

def loop(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    out   = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    show_image(out, "Edges")
    show_text(f"thresholds: {t1} / {t2}")
    return out

def teardown():
    pass
`;

const FACE_DETECT_CODE = `\
import cv2

def setup():
    global detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    show_text("Face detector ready (OpenCV Haar cascade)")

def loop(frame, metadata):
    scale = config.get("scale_factor", 1.1)
    neigh = config.get("min_neighbors", 4)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neigh)
    faces = [{"bbox": [int(x), int(y), int(x + w), int(y + h)]} for (x, y, w, h) in rects]
    metadata["faces"] = faces
    show_text(f"Detected: {len(faces)} face(s)")
    return frame
`;

const ROI_COUNTER_CODE = `\
import cv2
import numpy as np

def setup():
    global polygon, zone_id
    pts     = config.get("polygon", [[100, 100], [540, 100], [540, 380], [100, 380]])
    polygon = np.array(pts, dtype=np.int32)
    zone_id = config.get("zone_id", "zone_1")

def loop(frame, metadata):
    faces = metadata.get("faces", [])
    out   = frame.copy()

    # Draw zone boundary
    cv2.polylines(out, [polygon], True, (0, 255, 100), 2)

    # Count faces whose centre falls inside the polygon
    count = 0
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
        inside  = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0
        color   = (0, 255, 0) if inside else (128, 128, 128)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if inside:
            count += 1

    # Expose count in metadata so downstream nodes or MCP can read it
    metadata[f"counter_{zone_id}"] = count

    cv2.putText(out, f"In Zone: {count}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2, cv2.LINE_AA)
    show_image(out, "ROI Counter")
    show_text(f"Faces in zone: {count}")
    return out
`;

const BENCHMARK_CODE = `\
import time, csv, os, collections

def setup():
    global _times, _start, _count, _csv, _label
    _label  = config.get("label", "checkpoint")
    window  = config.get("window", 100)
    _times  = collections.deque(maxlen=window)
    _start  = None
    _count  = 0

    path = config.get("output_path", "")
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f    = open(path, "w", newline="")
        _csv = csv.writer(f)
        _csv.writerow(["frame", "fps_instant", "fps_avg", "latency_ms"])
    else:
        _csv = None

def loop(frame):
    global _start, _count
    now = time.perf_counter()
    if _start is None:
        _start = now
    _count += 1

    if _times:
        dt         = now - _times[-1]
        fps_now    = 1.0 / dt if dt > 0 else 0.0
        fps_avg    = _count / (now - _start) if now > _start else 0.0
        latency_ms = dt * 1000
        show_text(f"[{_label}] {fps_now:.1f} fps  avg {fps_avg:.1f}  {latency_ms:.0f}ms")
        if _csv:
            _csv.writerow([_count, round(fps_now, 2), round(fps_avg, 2), round(latency_ms, 2)])

    _times.append(now)
    return frame
`;

// ── Face pipeline nodes ────────────────────────────────────────────────────────

// SCRFD-10G face detection (InsightFace model, UUID ba91a664-…)
const SCRFD_CODE = `\
import cv2, numpy as np, os, onnxruntime as ort
from pathlib import Path

_sess = None
_SZ   = (640, 640)   # model input size

def _pre(img):
    scale = min(_SZ[0] / img.shape[1], _SZ[1] / img.shape[0])
    nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
    pad = np.zeros((_SZ[1], _SZ[0], 3), np.uint8)
    pad[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob = pad.astype(np.float32)
    blob = (blob - 127.5) / 128.0
    return blob.transpose(2, 0, 1)[np.newaxis], scale

def _decode(outs, scale, thr):
    strides = [8, 16, 32]
    all_bx, all_sc, all_kp = [], [], []
    for si, s in enumerate(strides):
        sc = outs[si].flatten()
        bx = outs[si+3].reshape(-1, 4)
        kp = outs[si+6].reshape(-1, 10)
        H, W = _SZ[1] // s, _SZ[0] // s
        cy, cx = np.mgrid[:H, :W]
        ctr = np.stack([cx, cy], -1).astype(np.float32) * s
        ctr = np.repeat(ctr.reshape(-1, 2), 2, 0)
        pos = sc >= thr
        if not pos.any(): continue
        c = ctr[pos]; b = bx[pos]; k = kp[pos]
        x1=(c[:,0]-b[:,0]*s)/scale; y1=(c[:,1]-b[:,1]*s)/scale
        x2=(c[:,0]+b[:,2]*s)/scale; y2=(c[:,1]+b[:,3]*s)/scale
        all_bx.append(np.stack([x1,y1,x2,y2], 1))
        all_sc.append(sc[pos])
        kx = c[:,0:1] + k[:,0::2]*s; ky = c[:,1:2] + k[:,1::2]*s
        all_kp.append(np.stack([kx/scale, ky/scale], 2))
    if not all_bx: return [], [], []
    bx=np.vstack(all_bx); sc=np.hstack(all_sc); kp=np.vstack(all_kp)
    x1,y1,x2,y2 = bx[:,0],bx[:,1],bx[:,2],bx[:,3]
    areas=(x2-x1)*(y2-y1); order=sc.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        iou=(np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
             /(areas[i]+areas[order[1:]]-np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)+1e-6))
        order=order[1:][iou<=0.4]
    k=np.array(keep)
    return bx[k].astype(int).tolist(), sc[k].tolist(), kp[k].tolist()

def setup():
    global _sess
    mid = "ba91a664-690d-47bd-96b9-3ece7691fd78"
    p = Path(os.environ.get("CVFLOW_MODELS_DIR", ".")) / mid / "model.onnx"
    _sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    show_text("SCRFD-10G face detector ready")

def loop(frame, metadata):
    if frame is None: return frame
    blob, scale = _pre(frame)
    outs = _sess.run(None, {"input.1": blob})
    thr = config.get("threshold", 0.45)
    bboxes, scores, kpss = _decode(outs, scale, thr)
    metadata["face_bboxes"] = bboxes
    metadata["face_kpss"]   = kpss
    out = frame.copy()
    for (x1,y1,x2,y2), sc in zip(bboxes, scores):
        cv2.rectangle(out, (x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(out, f"{sc:.2f}", (x1,max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)
    show_image(out, f"{len(bboxes)} face(s)")
    show_text(f"Detected {len(bboxes)} face(s)")
    return out
`;

// MobileFaceNet embedding (InsightFace model, UUID 6dc15c96-…)
const EMBED_CODE = `\
import cv2, numpy as np, os, onnxruntime as ort
from pathlib import Path

_sess = None
_SRC  = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

def _align(img, kps):
    dst = np.array(kps, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(dst, _SRC)
    if M is None: return None
    return cv2.warpAffine(img, M, (112, 112))

def _embed(face):
    x = face.astype(np.float32)
    x = (x - 127.5) / 128.0
    x = x[:, :, ::-1]          # BGR -> RGB
    x = x.transpose(2, 0, 1)[np.newaxis]
    out = _sess.run(None, {"input.1": x})[0][0]
    norm = np.linalg.norm(out)
    return (out / norm).tolist() if norm > 0 else out.tolist()

def setup():
    global _sess
    mid = "6dc15c96-68b6-4c36-8353-7ace6737f6e7"
    p = Path(os.environ.get("CVFLOW_MODELS_DIR", ".")) / mid / "model.onnx"
    _sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    show_text("MobileFaceNet embedding model ready")

def loop(frame, metadata):
    bboxes = metadata.get("face_bboxes", [])
    kpss   = metadata.get("face_kpss", [])
    if frame is None or not bboxes:
        metadata["face_embeddings"] = []
        return frame
    embeddings = []
    for bbox, kps in zip(bboxes, kpss):
        x1,y1,x2,y2 = bbox
        face = _align(frame, kps) if kps else None
        if face is None:
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            face = cv2.resize(crop, (112,112)) if crop.size > 0 else None
        if face is not None:
            embeddings.append(_embed(face))
    metadata["face_embeddings"] = embeddings
    show_text(f"Embedded {len(embeddings)} face(s)")
    return frame
`;

const ENROLL_CODE = `\
import cv2, numpy as np, requests, time

_last = 0.0

def setup():
    show_text(f"Ready to enroll: {config.get('person_name', 'Unknown')}")

def loop(frame, metadata):
    global _last
    bboxes  = metadata.get("face_bboxes", [])
    embeds  = metadata.get("face_embeddings", [])
    person  = config.get("person_name", "Unknown")
    coll    = config.get("collection", "faces")
    cooldown = config.get("cooldown_s", 1.5)
    url     = "http://localhost:8000/api"

    if not bboxes or not embeds:
        show_text(f"No face — waiting... ({person})")
        return frame

    out = frame.copy()
    x1,y1,x2,y2 = bboxes[0]
    now = time.monotonic()

    if now - _last >= cooldown:
        try:
            r = requests.post(
                f"{url}/datahub/vector/collections/{coll}/records",
                json={"embedding": embeds[0], "label": person},
                timeout=2
            )
            if r.ok:
                _last = now
                rid = r.json().get("id", "")[:8]
                cv2.rectangle(out, (x1,y1),(x2,y2),(0,255,220),3)
                cv2.putText(out, f"ENROLLED: {person}", (x1,max(0,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,220), 2)
                show_text(f"Enrolled '{person}' (id={rid})")
        except Exception as e:
            show_text(f"Enroll error: {e}")
    else:
        rem = cooldown - (now - _last)
        cv2.rectangle(out, (x1,y1),(x2,y2),(255,140,0),2)
        cv2.putText(out, f"Next in {rem:.1f}s", (x1,max(0,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,140,0), 2)
        show_text(f"Cooldown {rem:.1f}s — ({person})")

    send_frame(out)
    return out
`;

const RECOGNIZE_CODE = `\
import cv2, numpy as np, requests

def setup():
    show_text(f"Recognizer ready — collection: {config.get('collection','faces')}")

def loop(frame, metadata):
    bboxes = metadata.get("face_bboxes", [])
    embeds = metadata.get("face_embeddings", [])
    coll   = config.get("collection", "faces")
    thr    = config.get("threshold", 0.40)
    url    = "http://localhost:8000/api"

    if frame is None: return frame
    out = frame.copy()

    for emb, bbox in zip(embeds, bboxes):
        x1,y1,x2,y2 = bbox
        name  = "Unknown"
        score = 0.0
        try:
            r = requests.post(
                f"{url}/datahub/vector/collections/{coll}/search",
                json={"embedding": emb, "top_k": 1},
                timeout=2
            )
            if r.ok:
                res = r.json()
                if res and res[0]["score"] >= thr:
                    hit   = res[0]
                    name  = hit.get("metadata", {}).get("label", hit["id"][:8])
                    score = hit["score"]
        except Exception:
            pass

        color = (0,220,0) if name != "Unknown" else (0,60,220)
        cv2.rectangle(out, (x1,y1),(x2,y2), color, 2)
        label = f"{name} {score:.2f}" if score > 0 else "Unknown"
        cv2.putText(out, label, (x1,max(0,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    show_image(out, f"{len(bboxes)} face(s)")
    show_text(f"Recognizing {len(bboxes)} face(s)")
    return out
`;

// ── Face Tracking & Logging — GPU-optimised nodes ─────────────────────────────
// Detection (SCRFD) and embedding (MobileFaceNet) use CUDAExecutionProvider when
// onnxruntime-gpu is installed; fall back to CPU automatically.

// SCRFD-10G with GPU provider + min-face-size filter + face_scores output
const DETECT_FACE_CODE = `\
import cv2, numpy as np, os, onnxruntime as ort
from pathlib import Path

_sess = None
_SZ   = (640, 640)
_PROV = "CPU"

def _add_cuda_dll_dirs():
    # Windows: explicitly register CUDA lib dirs so LoadLibrary can find cublasLt etc.
    # (PATH alone is insufficient when the process inherits a restricted environment)
    if not hasattr(os, "add_dll_directory"): return
    cuda_env = os.environ.get("CUDA_PATH", "")
    search = []
    if cuda_env:
        search += [os.path.join(cuda_env,"bin"), os.path.join(cuda_env,"bin","x64")]
    cuda_root = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    for ver in ["v13.0","v12.6","v12.4","v12.2","v12.0","v11.8"]:
        base = cuda_root + "/" + ver
        search += [base + "/bin", base + "/bin/x64"]
    for d in search:
        if os.path.isdir(d):
            try: os.add_dll_directory(d)
            except: pass

def _pre(img):
    scale = min(_SZ[0]/img.shape[1], _SZ[1]/img.shape[0])
    nw, nh = int(img.shape[1]*scale), int(img.shape[0]*scale)
    pad = np.zeros((_SZ[1],_SZ[0],3), np.uint8)
    pad[:nh,:nw] = cv2.resize(img, (nw,nh))
    blob = pad.astype(np.float32)
    blob = (blob - 127.5) / 128.0
    return blob.transpose(2,0,1)[np.newaxis], scale

def _decode(outs, scale, thr):
    strides=[8,16,32]; all_bx,all_sc,all_kp=[],[],[]
    for si,s in enumerate(strides):
        sc=outs[si].flatten(); bx=outs[si+3].reshape(-1,4); kp=outs[si+6].reshape(-1,10)
        H,W=_SZ[1]//s,_SZ[0]//s
        cy,cx=np.mgrid[:H,:W]
        ctr=np.stack([cx,cy],-1).astype(np.float32)*s
        ctr=np.repeat(ctr.reshape(-1,2),2,0)
        pos=sc>=thr
        if not pos.any(): continue
        c=ctr[pos]; b=bx[pos]; k=kp[pos]
        x1=(c[:,0]-b[:,0]*s)/scale; y1=(c[:,1]-b[:,1]*s)/scale
        x2=(c[:,0]+b[:,2]*s)/scale; y2=(c[:,1]+b[:,3]*s)/scale
        all_bx.append(np.stack([x1,y1,x2,y2],1))
        all_sc.append(sc[pos])
        kx=c[:,0:1]+k[:,0::2]*s; ky=c[:,1:2]+k[:,1::2]*s
        all_kp.append(np.stack([kx/scale,ky/scale],2))
    if not all_bx: return [],[],[]
    bx=np.vstack(all_bx); sc=np.hstack(all_sc); kp=np.vstack(all_kp)
    x1,y1,x2,y2=bx[:,0],bx[:,1],bx[:,2],bx[:,3]
    areas=(x2-x1)*(y2-y1); order=sc.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        iou=(np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
             /(areas[i]+areas[order[1:]]-np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)+1e-6))
        order=order[1:][iou<=0.4]
    k=np.array(keep)
    return bx[k].astype(int).tolist(), sc[k].tolist(), kp[k].tolist()

def setup():
    global _sess, _PROV
    mid = "ba91a664-690d-47bd-96b9-3ece7691fd78"
    p = Path(os.environ.get("CVFLOW_MODELS_DIR", ".")) / mid / "model.onnx"

    # Must call BEFORE InferenceSession — registers CUDA DLL search dirs
    _add_cuda_dll_dirs()

    # Use get_available_providers() — this is the ground truth list that actually works
    avail = ort.get_available_providers()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
    try:
        _sess = ort.InferenceSession(str(p), providers=providers)
        # get_providers() on the session returns what is ACTUALLY registered, not just requested
        active = _sess.get_providers()
        _PROV = "CUDA" if active and active[0] == "CUDAExecutionProvider" else "CPU"
    except Exception as e:
        _sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        _PROV = "CPU"
        show_text(f"⚠ GPU init failed: {e}")
    show_text(f"SCRFD-10G ready [{_PROV}]  avail={avail}")

def loop(frame, metadata):
    if frame is None: return frame
    blob, scale = _pre(frame)
    outs = _sess.run(None, {"input.1": blob})
    thr    = slider('threshold',    10, 100, 45) / 100.0
    min_px = slider('min_face_px',  20, 200, 40)
    bboxes, scores, kpss = _decode(outs, scale, thr)
    filtered = [(b,s,k) for b,s,k in zip(bboxes,scores,kpss) if b[3]-b[1]>=min_px]
    bboxes=[x[0] for x in filtered]
    scores=[x[1] for x in filtered]
    kpss  =[x[2] for x in filtered]
    metadata["face_bboxes"]=bboxes; metadata["face_kpss"]=kpss; metadata["face_scores"]=scores
    out=frame.copy()
    for (x1,y1,x2,y2),sc in zip(bboxes,scores):
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(out,f"{sc:.2f}",(x1,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,0),1)
    show_image(out,f"{len(bboxes)} face(s)")
    show_text(f"Detected {len(bboxes)} face(s) [{_PROV}]")
    return out
`;

// Face crop with 5-point landmark alignment + quality filter
// Outputs face_crops: list of aligned numpy crops (HWC BGR)
const CROP_FACE_CODE = `\
import cv2, numpy as np

_SRC = np.array([
    [38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
    [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)

def _align(img, kps, size):
    dst = _SRC * size / 112.0
    M, _ = cv2.estimateAffinePartial2D(
        np.array(kps,dtype=np.float32), dst, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(img, M, (size,size))

def loop(frame, metadata):
    if frame is None: return frame
    bboxes  = metadata.get("face_bboxes",[])
    kpss    = metadata.get("face_kpss",[])
    img_sz  = slider('image_size',  64, 256, 112)
    min_q   = slider('min_quality', 10, 200,  40)
    crops = []
    for bbox,kps in zip(bboxes,kpss):
        h = bbox[3]-bbox[1]
        if h < min_q: continue
        crop = _align(frame, kps, img_sz)
        if crop is not None:
            crops.append(crop)
    metadata["face_crops"] = crops
    if crops:
        show_image(crops[0], f"{len(crops)} crop(s)")
    show_text(f"{len(crops)} face crop(s) aligned (size={img_sz})")
    return frame
`;

// MobileFaceNet batch embedding — GPU inference via CUDAExecutionProvider
// Reads face_crops from metadata, outputs face_embeddings as list of Python lists
const EMBED_FACE_CODE = `\
import numpy as np, os, onnxruntime as ort
from pathlib import Path

_sess = None
_PROV = "CPU"

def _add_cuda_dll_dirs():
    if not hasattr(os, "add_dll_directory"): return
    cuda_env = os.environ.get("CUDA_PATH", "")
    search = []
    if cuda_env:
        search += [os.path.join(cuda_env,"bin"), os.path.join(cuda_env,"bin","x64")]
    cuda_root = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    for ver in ["v13.0","v12.6","v12.4","v12.2","v12.0","v11.8"]:
        base = cuda_root + "/" + ver
        search += [base + "/bin", base + "/bin/x64"]
    for d in search:
        if os.path.isdir(d):
            try: os.add_dll_directory(d)
            except: pass

def setup():
    global _sess, _PROV
    mid = "6dc15c96-68b6-4c36-8353-7ace6737f6e7"
    p = Path(os.environ.get("CVFLOW_MODELS_DIR", ".")) / mid / "model.onnx"
    _add_cuda_dll_dirs()
    avail = ort.get_available_providers()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
    try:
        _sess = ort.InferenceSession(str(p), providers=providers)
        active = _sess.get_providers()
        _PROV = "CUDA" if active and active[0] == "CUDAExecutionProvider" else "CPU"
    except Exception as e:
        _sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        _PROV = "CPU"
        show_text(f"⚠ GPU init failed: {e}")
    show_text(f"MobileFaceNet ready [{_PROV}]  avail={avail}")

def loop(frame, metadata):
    crops = metadata.get("face_crops",[])
    if not crops:
        metadata["face_embeddings"] = []
        return frame
    # Batch preprocessing: BGR→RGB, normalise, CHW → (N,3,H,W)
    batch = np.stack([
        ((c[:,:,::-1].astype(np.float32)-127.5)/128.0).transpose(2,0,1)
        for c in crops
    ])  # (N,3,112,112)
    embs = _sess.run(None, {"input.1": batch})[0]  # (N,512)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.maximum(norms, 1e-8)
    metadata["face_embeddings"] = [e.tolist() for e in embs]
    show_text(f"Embedded {len(embs)} face(s) [{_PROV}]")
    return frame
`;

// Dedup + HTTP POST to LoggingDB API — non-blocking via background queue
// Reads face_crops + face_embeddings, logs unique faces with cooldown
const TRACK_LOG_CODE = `\
import base64, cv2, requests, queue, threading

_q = None

def setup():
    global _q
    _q = queue.Queue(maxsize=50)

    def _worker():
        while True:
            item = _q.get()
            if item is None:
                break
            url, payload = item
            try:
                requests.post(url, json=payload, timeout=3)
            except Exception:
                pass
            _q.task_done()

    threading.Thread(target=_worker, daemon=True).start()
    show_text(f"Track & Log ready → {config.get('collection','face_log')}")

def loop(frame, metadata):
    crops  = metadata.get("face_crops",[])
    embs   = metadata.get("face_embeddings",[])
    coll      = text_input('collection',    'face_log')
    cooldown  = slider('cooldown_min',  1,  60,  5) * 60
    sim_thr   = slider('sim_threshold', 30, 100, 60) / 100.0

    queued = 0
    for crop, emb in zip(crops, embs):
        if _q is None or _q.full():
            continue
        _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()
        try:
            _q.put_nowait((
                f"http://localhost:8000/api/logging/{coll}/add",
                {"embedding": emb, "image_b64": b64,
                 "cooldown_sec": cooldown, "sim_threshold": sim_thr},
            ))
            queued += 1
        except queue.Full:
            pass

    show_text(f"Queued {queued}/{len(crops)} face(s) | backlog={_q.qsize() if _q else 0}")
    return frame
`;

// ── Sample pipeline definitions ────────────────────────────────────────────────

export const SAMPLES = [

  // ── 1. USB Camera Display ──────────────────────────────────────────────────
  {
    name: "USB Camera Display",
    description: "Camera source (loop with no params) → Stream Viewer (send_frame broadcasts to frontend). Edit device_index in Camera config.",
    nodes: [
      { id: "cam",    type: "python_node", position: { x: 160, y: 220 },
        data: {
          label: "USB Camera",
          config: { mode: "loop", active_key: "active", device_index: 0, width: 1280, height: 720, code: CAM_CODE },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "viewer", type: "python_node", position: { x: 480, y: 220 },
        data: {
          label: "Stream Viewer",
          config: { mode: "loop", code: STREAM_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam", target: "viewer", sourceHandle: "out", targetHandle: "in" },
    ],
  },

  // ── 2. Edge Detection ──────────────────────────────────────────────────────
  {
    name: "Edge Detection",
    description: "Camera → Canny edge detect → stream. Adjust threshold1/threshold2 in Edge Detect config to tune sensitivity.",
    nodes: [
      { id: "cam",    type: "python_node", position: { x:  60, y: 200 },
        data: {
          label: "USB Camera",
          config: { mode: "loop", code: CAM_CODE, device_index: 0 },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "edge",   type: "python_node", position: { x: 360, y: 200 },
        data: {
          label: "Edge Detect",
          config: { mode: "loop", code: EDGE_CODE, threshold1: 50, threshold2: 150 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "edges" }] },
        }},
      { id: "viewer", type: "python_node", position: { x: 660, y: 200 },
        data: {
          label: "Stream Viewer",
          config: { mode: "loop", code: STREAM_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam",  target: "edge",   sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "edge", target: "viewer", sourceHandle: "out", targetHandle: "in" },
    ],
  },

  // ── 3. Face ROI Counter ────────────────────────────────────────────────────
  {
    name: "Face ROI Counter",
    description: "Camera → Haar-cascade face detection → polygon zone counter → stream. Uses only OpenCV — no extra install needed.",
    nodes: [
      { id: "cam",     type: "python_node", position: { x:  60, y: 200 },
        data: {
          label: "USB Camera",
          config: { mode: "loop", code: CAM_CODE, device_index: 0 },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "detect",  type: "python_node", position: { x: 360, y: 200 },
        data: {
          label: "Face Detect",
          config: { mode: "loop", code: FACE_DETECT_CODE, scale_factor: 1.1, min_neighbors: 4 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame+faces" }] },
        }},
      { id: "counter", type: "python_node", position: { x: 660, y: 200 },
        data: {
          label: "ROI Counter",
          config: {
            mode: "loop", code: ROI_COUNTER_CODE,
            polygon: [[100, 100], [540, 100], [540, 380], [100, 380]],
            zone_id: "face_zone",
          },
          ports: { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "annotated" }] },
        }},
      { id: "viewer",  type: "python_node", position: { x: 960, y: 200 },
        data: {
          label: "Stream Viewer",
          config: { mode: "loop", code: STREAM_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam",     target: "detect",  sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "detect",  target: "counter", sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "counter", target: "viewer",  sourceHandle: "out", targetHandle: "in" },
    ],
  },

  // ── 4. Benchmark Sequential vs Multiprocess ───────────────────────────────
  {
    name: "Benchmark Sequential vs Multiprocess",
    description: "Wrap any node between two Benchmark nodes to measure latency. Compare sequential vs multiprocess run modes.",
    nodes: [
      { id: "cam",  type: "python_node", position: { x:  60, y: 200 },
        data: {
          label: "USB Camera",
          config: { mode: "loop", code: CAM_CODE, device_index: 0 },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "bm1",  type: "python_node", position: { x: 340, y: 200 },
        data: {
          label: "Benchmark ①",
          config: { mode: "loop", code: BENCHMARK_CODE, label: "before_edge", window: 100 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "proc", type: "python_node", position: { x: 620, y: 200 },
        data: {
          label: "Edge Detect",
          config: { mode: "loop", code: EDGE_CODE, threshold1: 50, threshold2: 150 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "edges" }] },
        }},
      { id: "bm2",  type: "python_node", position: { x: 900, y: 200 },
        data: {
          label: "Benchmark ②",
          config: { mode: "loop", code: BENCHMARK_CODE, label: "after_edge", window: 100, output_path: "storage/tmp/bench.csv" },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "view", type: "python_node", position: { x: 1180, y: 200 },
        data: {
          label: "Stream Viewer",
          config: { mode: "loop", code: STREAM_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam",  target: "bm1",  sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "bm1",  target: "proc", sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "proc", target: "bm2",  sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "bm2",  target: "view", sourceHandle: "out", targetHandle: "in" },
    ],
  },

  // ── 5. Face Enrollment ────────────────────────────────────────────────────
  {
    name: "Face Enrollment",
    description: "Camera → SCRFD face detect → MobileFaceNet embed → save to DataHub 'faces' collection. Set person_name in the Enroll node config. Run alongside Face Recognition to add identities.",
    nodes: [
      { id: "cam",    type: "python_node", position: { x:  60, y: 220 },
        data: {
          label: "Camera",
          config: { mode: "loop", code: CAM_CODE, device_index: 0 },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "detect", type: "python_node", position: { x: 360, y: 220 },
        data: {
          label: "Face Detect (SCRFD)",
          config: { mode: "loop", code: SCRFD_CODE, threshold: 0.45 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "embed",  type: "python_node", position: { x: 660, y: 220 },
        data: {
          label: "Face Embed (MobileFaceNet)",
          config: { mode: "loop", code: EMBED_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "enroll", type: "python_node", position: { x: 960, y: 220 },
        data: {
          label: "Enroll to DataHub",
          config: { mode: "loop", code: ENROLL_CODE, person_name: "Alice", collection: "faces", cooldown_s: 1.5 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "viewer", type: "python_node", position: { x: 1260, y: 220 },
        data: {
          label: "Stream Viewer",
          config: { mode: "loop", code: STREAM_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam",    target: "detect", sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "detect", target: "embed",  sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "embed",  target: "enroll", sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "enroll", target: "viewer", sourceHandle: "out", targetHandle: "in" },
    ],
  },

  // ── 6. Face Tracking & Logging ───────────────────────────────────────────
  {
    name: "Face Tracking & Logging",
    description: "Camera → SCRFD detect (GPU) → 5-point crop → MobileFaceNet embed (GPU) → dedup + log to face_log collection. Same face suppressed for 5 min (configurable). Requires onnxruntime-gpu for GPU inference; falls back to CPU automatically.",
    nodes: [
      { id: "cam",      type: "python_node", position: { x:   60, y: 220 },
        data: {
          label: "Camera",
          config: { mode: "loop", code: CAM_CODE, device_index: 0 },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "detect",   type: "python_node", position: { x:  360, y: 220 },
        data: {
          label: "Face Detect (GPU)",
          config: { mode: "loop", code: DETECT_FACE_CODE, device: "cuda:0", gpu_memory_fraction: 0.30 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "crop",     type: "python_node", position: { x:  660, y: 220 },
        data: {
          label: "Face Crop",
          config: { mode: "loop", code: CROP_FACE_CODE, image_size: 112, min_quality: 40 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "embed",    type: "python_node", position: { x:  960, y: 220 },
        data: {
          label: "Face Embed (GPU)",
          config: { mode: "loop", code: EMBED_FACE_CODE, device: "cuda:0", gpu_memory_fraction: 0.20 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "tracklog", type: "python_node", position: { x: 1260, y: 220 },
        data: {
          label: "Track & Log",
          config: { mode: "loop", code: TRACK_LOG_CODE, collection: "face_log", cooldown_min: 5, sim_threshold: 60 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam",    target: "detect",   sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "detect", target: "crop",     sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "crop",   target: "embed",    sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "embed",  target: "tracklog", sourceHandle: "out", targetHandle: "in" },
    ],
  },

  // ── 7. Face Recognition ───────────────────────────────────────────────────
  {
    name: "Face Recognition",
    description: "Camera → SCRFD detect → MobileFaceNet embed → search DataHub 'faces' collection → draw name + score. Run Face Enrollment first to populate the database.",
    nodes: [
      { id: "cam",    type: "python_node", position: { x:  60, y: 220 },
        data: {
          label: "Camera",
          config: { mode: "loop", code: CAM_CODE, device_index: 0 },
          ports:  { inputs: [], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "detect", type: "python_node", position: { x: 360, y: 220 },
        data: {
          label: "Face Detect (SCRFD)",
          config: { mode: "loop", code: SCRFD_CODE, threshold: 0.45 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "embed",  type: "python_node", position: { x: 660, y: 220 },
        data: {
          label: "Face Embed (MobileFaceNet)",
          config: { mode: "loop", code: EMBED_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "recog",  type: "python_node", position: { x: 960, y: 220 },
        data: {
          label: "Face Recognize",
          config: { mode: "loop", code: RECOGNIZE_CODE, collection: "faces", threshold: 0.40 },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [{ id: "out", label: "frame" }] },
        }},
      { id: "viewer", type: "python_node", position: { x: 1260, y: 220 },
        data: {
          label: "Stream Viewer",
          config: { mode: "loop", code: STREAM_CODE },
          ports:  { inputs: [{ id: "in", label: "frame" }], outputs: [] },
        }},
    ],
    edges: [
      { id: "e1", source: "cam",    target: "detect", sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "detect", target: "embed",  sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "embed",  target: "recog",  sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "recog",  target: "viewer", sourceHandle: "out", targetHandle: "in" },
    ],
  },
];
