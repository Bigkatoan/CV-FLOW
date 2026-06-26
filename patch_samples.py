import re

with open("frontend/static/samples.js", "r", encoding="utf-8") as f:
    content = f.read()

new_samples = """
  // ── 3. Face ROI Counter ──────────────────────────────────────────────────────
  {
    name: "Face ROI Counter",
    description: "Camera → Face Detect → Align → Embed → Vector DB → Draw ROI → Counter",
    nodes: [
      { id: "n1", type: "python_node", position: { x: 50, y: 100 },
        data: { label: "Camera", config: { mode: "loop", code: CAM_CODE }, ports: { inputs: [], outputs: [{id:"out",label:"out"}] } } },
      { id: "n2", type: "face_detect", position: { x: 250, y: 100 },
        data: { label: "Face Detect", config: { conf_threshold: 0.5 } } },
      { id: "n3", type: "face_align", position: { x: 450, y: 100 },
        data: { label: "Face Align", config: {} } },
      { id: "n4", type: "face_embed", position: { x: 650, y: 100 },
        data: { label: "Face Embed", config: {} } },
      { id: "n5", type: "face_vector_db", position: { x: 850, y: 100 },
        data: { label: "Face DB", config: { db_dir: "storage/facedb", threshold: 0.35 } } },
      { id: "n6", type: "draw_roi", position: { x: 250, y: 250 },
        data: { label: "Count Zone", config: { polygon: [[100,100],[540,100],[540,380],[100,380]], zone_id: "face_roi" } } },
      { id: "n7", type: "counter", position: { x: 450, y: 250 },
        data: { label: "Faces in ROI", config: { trigger_type: "zone_enter", trigger_id: "face_roi", label: "Faces in ROI" } } },
      { id: "n8", type: "stream_viewer", position: { x: 650, y: 250 },
        data: { label: "Viewer", config: {} } }
    ],
    edges: [
      { id: "e1", source: "n1", target: "n2", sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "n2", target: "n3", sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "n3", target: "n4", sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "n4", target: "n5", sourceHandle: "out", targetHandle: "in" },
      { id: "e5", source: "n5", target: "n6", sourceHandle: "out", targetHandle: "in" },
      { id: "e6", source: "n6", target: "n7", sourceHandle: "out", targetHandle: "in" },
      { id: "e7", source: "n7", target: "n8", sourceHandle: "out", targetHandle: "in" }
    ]
  },

  // ── 4. Benchmark Sequential vs MP ────────────────────────────────────────────
  {
    name: "Benchmark Sequential vs Multiprocess",
    description: "Place BenchmarkNode before and after inference to compare modes.",
    nodes: [
      { id: "cam", type: "python_node", position: { x: 50, y: 100 },
        data: { label: "Camera", config: { mode: "loop", code: CAM_CODE }, ports: { inputs: [], outputs: [{id:"out",label:"out"}] } } },
      { id: "bm1", type: "benchmark", position: { x: 250, y: 100 },
        data: { label: "BM Before", config: { label: "before_inference" } } },
      { id: "inf", type: "face_detect", position: { x: 450, y: 100 },
        data: { label: "Face Detect", config: { conf_threshold: 0.5 } } },
      { id: "bm2", type: "benchmark", position: { x: 650, y: 100 },
        data: { label: "BM After", config: { label: "after_inference" } } },
      { id: "view", type: "stream_viewer", position: { x: 850, y: 100 },
        data: { label: "Viewer", config: {} } }
    ],
    edges: [
      { id: "eb1", source: "cam", target: "bm1", sourceHandle: "out", targetHandle: "in" },
      { id: "eb2", source: "bm1", target: "inf", sourceHandle: "out", targetHandle: "in" },
      { id: "eb3", source: "inf", target: "bm2", sourceHandle: "out", targetHandle: "in" },
      { id: "eb4", source: "bm2", target: "view", sourceHandle: "out", targetHandle: "in" }
    ]
  }
];
"""

content = content.replace("];\n", new_samples)

with open("frontend/static/samples.js", "w", encoding="utf-8") as f:
    f.write(content)

print("Added samples")
