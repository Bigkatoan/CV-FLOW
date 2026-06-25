// Pre-built sample pipelines using the function-based python_node system.
// Parameters become input ports; return value becomes output (ctx.frame if ndarray).

const CAM_CODE = `\
import cv2

def setup():
    global cap
    idx = config.get("device_index", 0)
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {idx}")

def loop():
    ret, frame = cap.read()
    if not ret:
        raise StopIteration
    show_image(frame, "Camera")
    return frame

def teardown():
    global cap
    if cap is not None:
        cap.release()
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

export const SAMPLES = [
  // ── 1. USB Camera Display ────────────────────────────────────────────────────
  {
    name: "USB Camera Display",
    description: "Single Python Node reads from USB camera and streams the output. loop() has no params → no input port.",
    nodes: [
      { id: "cam_1", type: "python_node", position: { x: 180, y: 220 },
        data: {
          label: "USB Camera",
          ports: { inputs: [], outputs: [{ id: "frame", label: "frame" }] },
          config: { mode: "loop", active_key: "active", code: CAM_CODE },
        },
      },
    ],
    edges: [],
  },

  // ── 2. Edge Detection ────────────────────────────────────────────────────────
  {
    name: "Edge Detection",
    description: "Camera → Canny edge detect. loop(frame) receives ctx.frame; show_image/show_text show live output in Properties.",
    nodes: [
      { id: "cam_1",  type: "python_node", position: { x:  80, y: 220 },
        data: {
          label: "USB Camera",
          ports: { inputs: [], outputs: [{ id: "frame", label: "frame" }] },
          config: { mode: "loop", active_key: "active", code: CAM_CODE },
        },
      },
      { id: "edge_1", type: "python_node", position: { x: 400, y: 220 },
        data: {
          label: "Edge Detect",
          ports: { inputs: [{ id: "frame", label: "frame" }], outputs: [{ id: "out", label: "out" }] },
          config: {
            mode: "loop", active_key: "active", code: EDGE_CODE,
            threshold1: 50, threshold2: 150,
          },
        },
      },
    ],
    edges: [
      { id: "e1", source: "cam_1", target: "edge_1", sourceHandle: "frame", targetHandle: "frame" },
    ],
  },
];
