import { createElement, useState, useCallback, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactFlow, Background, Controls, MiniMap,
  addEdge, useNodesState, useEdgesState, useReactFlow, ReactFlowProvider,
} from "reactflow";
import htm from "https://esm.sh/htm@3";
import { nodeTypes, NODE_META, GROUP_COLOR } from "./nodes.js";
import { SAMPLES } from "./samples.js";

const html = htm.bind(createElement);

// ── API ──────────────────────────────────────────────────────────────────────
const API = "http://localhost:8000/api";
async function apiFetch(method, path, body) {
  const r = await fetch(API + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) { const t = await r.text(); throw new Error(`${r.status}: ${t}`); }
  if (r.status === 204) return null;
  return r.json();
}

// ── DEFAULT CONFIGS ──────────────────────────────────────────────────────────
const DEFAULT_CONFIG = {
  camera:          { source_type: "usb", device_index: 0, fps_limit: 30 },
  video_file:      { file_path: "", loop: false },
  image_directory: { directory_path: "", pattern: "*.jpg", delay_ms: 100 },
  preprocess:      { normalize: "none" },
  model_inference: { model_id: "", device: "cpu", conf_threshold: 0.5 },
  nms:             { iou_threshold: 0.45, conf_threshold: 0.25, max_detections: 300 },
  draw_roi:        { zone_id: "zone_1", polygon: [[10,10],[90,10],[90,90],[10,90]] },
  draw_line:       { line_id: "line_1", line: [[10,50],[90,50]], direction: "both" },
  object_tracker:  { algorithm: "bytetrack", max_age: 30, min_hits: 3 },
  counter:         { trigger_type: "line_cross", trigger_id: "line_1", reset_on_start: true },
  python_function: { code: "def process(frame, detections, params):\n    return frame, detections\n" },
  filter:          { allowed_classes: [], min_confidence: 0.0 },
  param:           { params: {} },
  cpp_function:    { source_code: "", compile_status: "uncompiled", compile_flags: ["-O2"] },
  stream_viewer:   { jpeg_quality: 80, max_fps: 30 },
  video_writer:    { output_path: "./output.mp4", codec: "mp4v", fps: 30 },
  trigger_webhook: { protocol: "http", trigger_on: "count_change", rate_limit_s: 2.0 },
};

// ── PALETTE ──────────────────────────────────────────────────────────────────
const GROUPS = [
  { label: "Input",      types: ["camera", "video_file", "image_directory"] },
  { label: "Processing", types: ["preprocess", "model_inference", "nms"] },
  { label: "Spatial",    types: ["draw_roi", "draw_line", "object_tracker", "counter"] },
  { label: "Utility",    types: ["python_function", "filter", "param"] },
  { label: "C++",        types: ["cpp_function"] },
  { label: "Output",     types: ["stream_viewer", "video_writer", "trigger_webhook"] },
];

function Palette() {
  const onDragStart = (e, type) => {
    e.dataTransfer.setData("application/cvflow", type);
    e.dataTransfer.effectAllowed = "move";
  };
  return html`
    <div style=${{ width: 162, flexShrink: 0, borderRight: "1px solid #30363d",
                   overflowY: "auto", background: "#0d1117", padding: "8px 6px" }}>
      ${GROUPS.map(g => html`
        <div key=${g.label} style=${{ marginBottom: 8 }}>
          <div style=${{ fontSize: 10, color: "#8b949e", textTransform: "uppercase",
                         letterSpacing: ".6px", padding: "2px 4px 5px", fontWeight: 700 }}>
            ${g.label}
          </div>
          ${g.types.map(type => {
            const m = NODE_META[type];
            const bg = GROUP_COLOR[m.group];
            return html`
              <div key=${type} draggable=${true} onDragStart=${e => onDragStart(e, type)}
                style=${{ display: "flex", alignItems: "center", gap: 7, padding: "5px 8px",
                           borderRadius: 6, cursor: "grab", marginBottom: 3,
                           background: bg + "60", border: "1px solid " + bg,
                           userSelect: "none", fontSize: 12, color: "#c9d1d9",
                           transition: "opacity .1s" }}
                onMouseEnter=${e => e.currentTarget.style.opacity = ".8"}
                onMouseLeave=${e => e.currentTarget.style.opacity = "1"}>
                <span>${m.icon}</span>
                <span style=${{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>${m.label}</span>
              </div>`;
          })}
        </div>`)}
    </div>`;
}

// ── FORM FIELDS ───────────────────────────────────────────────────────────────
const inp = {
  width: "100%", background: "#21262d", border: "1px solid #30363d",
  borderRadius: 5, color: "#c9d1d9", padding: "4px 8px", fontSize: 12, outline: "none",
  fontFamily: "inherit",
};
const row = { marginBottom: 10 };
const lbl = { display: "block", fontSize: 10, color: "#8b949e",
              textTransform: "uppercase", letterSpacing: ".5px", marginBottom: 3 };

function Field({ label, children }) {
  return html`<div style=${row}><label style=${lbl}>${label}</label>${children}</div>`;
}

function PropertiesPanel({ node, onUpdate }) {
  if (!node) return html`
    <div style=${{ padding: 16, color: "#8b949e", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
      Click a node to edit its config
    </div>`;

  const cfg = node.data.config ?? {};
  const set = (key, val) => onUpdate(node.id, { ...cfg, [key]: val });

  let fields;
  switch (node.type) {
    case "camera": fields = html`
      <${Field} label="Source">
        <select style=${inp} value=${cfg.source_type ?? "usb"} onChange=${e => set("source_type", e.target.value)}>
          <option value="usb">USB Device</option><option value="rtsp">RTSP</option><option value="http">HTTP/MJPEG</option>
        </select>
      <//>
      <${Field} label="Device Index">
        <input type="number" style=${inp} value=${cfg.device_index ?? 0} onChange=${e => set("device_index", +e.target.value)} />
      <//>
      <${Field} label="FPS Limit">
        <input type="number" style=${inp} value=${cfg.fps_limit ?? 30} onChange=${e => set("fps_limit", +e.target.value)} />
      <//>
      ${cfg.source_type !== "usb" && cfg.source_type != null && html`
        <${Field} label="URL">
          <input style=${inp} value=${cfg.url ?? ""} onChange=${e => set("url", e.target.value)} placeholder="rtsp://..." />
        <//>
      `}`; break;

    case "video_file": fields = html`
      <${Field} label="File Path">
        <input style=${inp} value=${cfg.file_path ?? ""} onChange=${e => set("file_path", e.target.value)} placeholder="C:/video.mp4" />
      <//>
      <${Field} label="Loop">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${!!cfg.loop} onChange=${e => set("loop", e.target.checked)} />
          <span style=${{ color: "#c9d1d9", fontSize: 12 }}>Loop video</span>
        </label>
      <//>
      <${Field} label="FPS Limit (0 = native)">
        <input type="number" style=${inp} value=${cfg.fps_limit ?? 0} onChange=${e => set("fps_limit", +e.target.value)} />
      <//>
      `; break;

    case "image_directory": fields = html`
      <${Field} label="Directory Path">
        <input style=${inp} value=${cfg.directory_path ?? ""} onChange=${e => set("directory_path", e.target.value)} placeholder="C:/images" />
      <//>
      <${Field} label="File Pattern">
        <input style=${inp} value=${cfg.pattern ?? "*.jpg"} onChange=${e => set("pattern", e.target.value)} />
      <//>
      <${Field} label="Delay Between Frames (ms)">
        <input type="number" style=${inp} value=${cfg.delay_ms ?? 100} onChange=${e => set("delay_ms", +e.target.value)} />
      <//>
      `; break;

    case "model_inference": fields = html`
      <${Field} label="Model ID">
        <input style=${{ ...inp, fontFamily: "monospace", fontSize: 11 }} value=${cfg.model_id ?? ""}
          onChange=${e => set("model_id", e.target.value)} placeholder="UUID from Models page" />
      <//>
      <${Field} label="Device">
        <select style=${inp} value=${cfg.device ?? "cpu"} onChange=${e => set("device", e.target.value)}>
          <option value="cpu">CPU</option><option value="cuda">CUDA (GPU)</option>
        </select>
      <//>
      <${Field} label=${"Confidence Threshold: " + (cfg.conf_threshold ?? 0.5)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="1" step="0.05"
          value=${cfg.conf_threshold ?? 0.5} onChange=${e => set("conf_threshold", +e.target.value)} />
      <//>
      `; break;

    case "nms": fields = html`
      <${Field} label=${"IoU Threshold: " + (cfg.iou_threshold ?? 0.45)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="1" step="0.05"
          value=${cfg.iou_threshold ?? 0.45} onChange=${e => set("iou_threshold", +e.target.value)} />
      <//>
      <${Field} label=${"Confidence Threshold: " + (cfg.conf_threshold ?? 0.25)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="1" step="0.05"
          value=${cfg.conf_threshold ?? 0.25} onChange=${e => set("conf_threshold", +e.target.value)} />
      <//>
      <${Field} label="Max Detections">
        <input type="number" style=${inp} value=${cfg.max_detections ?? 300} onChange=${e => set("max_detections", +e.target.value)} />
      <//>
      `; break;

    case "object_tracker": fields = html`
      <${Field} label="Algorithm">
        <select style=${inp} value=${cfg.algorithm ?? "bytetrack"} onChange=${e => set("algorithm", e.target.value)}>
          <option value="bytetrack">ByteTrack</option><option value="sort">SORT</option>
        </select>
      <//>
      <${Field} label="Max Age (frames)">
        <input type="number" style=${inp} value=${cfg.max_age ?? 30} onChange=${e => set("max_age", +e.target.value)} />
      <//>
      <${Field} label="Min Hits to confirm track">
        <input type="number" style=${inp} value=${cfg.min_hits ?? 3} onChange=${e => set("min_hits", +e.target.value)} />
      <//>
      `; break;

    case "draw_line": fields = html`
      <${Field} label="Line ID">
        <input style=${inp} value=${cfg.line_id ?? "line_1"} onChange=${e => set("line_id", e.target.value)} />
      <//>
      <${Field} label="Direction">
        <select style=${inp} value=${cfg.direction ?? "both"} onChange=${e => set("direction", e.target.value)}>
          <option value="both">Both directions</option>
          <option value="up">Up only</option>
          <option value="down">Down only</option>
        </select>
      <//>
      `; break;

    case "counter": fields = html`
      <${Field} label="Trigger Type">
        <select style=${inp} value=${cfg.trigger_type ?? "line_cross"} onChange=${e => set("trigger_type", e.target.value)}>
          <option value="line_cross">Line Cross</option>
          <option value="zone_enter">Zone Enter</option>
          <option value="zone_exit">Zone Exit</option>
        </select>
      <//>
      <${Field} label="Trigger ID">
        <input style=${inp} value=${cfg.trigger_id ?? ""} onChange=${e => set("trigger_id", e.target.value)} placeholder="line_1 or zone_1" />
      <//>
      `; break;

    case "stream_viewer": fields = html`
      <${Field} label=${"JPEG Quality: " + (cfg.jpeg_quality ?? 80)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="10" max="100" step="5"
          value=${cfg.jpeg_quality ?? 80} onChange=${e => set("jpeg_quality", +e.target.value)} />
      <//>
      <${Field} label="Max FPS">
        <input type="number" style=${inp} value=${cfg.max_fps ?? 30} onChange=${e => set("max_fps", +e.target.value)} />
      <//>
      `; break;

    case "video_writer": fields = html`
      <${Field} label="Output Path">
        <input style=${inp} value=${cfg.output_path ?? "./output.mp4"} onChange=${e => set("output_path", e.target.value)} />
      <//>
      <${Field} label="FPS">
        <input type="number" style=${inp} value=${cfg.fps ?? 30} onChange=${e => set("fps", +e.target.value)} />
      <//>
      `; break;

    case "python_function": fields = html`
      <${Field} label="Code">
        <textarea style=${{ ...inp, height: 160, resize: "vertical", lineHeight: 1.5,
                             fontFamily: "Consolas,'Courier New',monospace", fontSize: 11 }}
          value=${cfg.code ?? ""} onChange=${e => set("code", e.target.value)} />
      <//>
      `; break;

    case "cpp_function": fields = html`
      <${Field} label="Compile Status">
        <div style=${{ fontSize: 12, padding: "2px 0",
                        color: cfg.compile_status === "ok" ? "#3fb950" :
                               cfg.compile_status === "error" ? "#f85149" : "#8b949e" }}>
          ${cfg.compile_status ?? "uncompiled"}
        </div>
      <//>
      <${Field} label="so_hash (for pipeline JSON)">
        <input style=${{ ...inp, fontFamily: "monospace", fontSize: 10 }} readOnly value=${cfg.compiled_so_hash ?? "(compile first)"} />
      <//>
      `; break;

    default: fields = html`
      <${Field} label="Config JSON">
        <textarea style=${{ ...inp, height: 180, resize: "vertical", lineHeight: 1.5,
                             fontFamily: "Consolas,'Courier New',monospace", fontSize: 11 }}
          value=${JSON.stringify(cfg, null, 2)}
          onChange=${e => { try { onUpdate(node.id, JSON.parse(e.target.value)); } catch {} }} />
      <//>
      `;
  }

  const meta = NODE_META[node.type];
  return html`
    <div style=${{ padding: "12px 14px", overflowY: "auto", flex: 1 }}>
      <div style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14,
                      paddingBottom: 10, borderBottom: "1px solid #30363d" }}>
        <div style=${{ width: 28, height: 28, borderRadius: 6, background: GROUP_COLOR[meta?.group] ?? "#21262d",
                        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15 }}>
          ${meta?.icon ?? "◻"}
        </div>
        <div>
          <div style=${{ fontWeight: 700, fontSize: 13, color: "#e2e8f0" }}>${meta?.label ?? node.type}</div>
          <div style=${{ fontSize: 10, color: "#8b949e" }}>${node.id}</div>
        </div>
      </div>

      <${Field} label="Label">
        <input style=${inp} value=${node.data.label ?? ""}
          onChange=${e => onUpdate(node.id, cfg, e.target.value)} />
      <//>

      ${fields}

      <button onClick=${() => { if (confirm("Delete this node?")) onUpdate(node.id, null); }}
        style=${{ width: "100%", marginTop: 8, padding: "6px 0", borderRadius: 6,
                   background: "transparent", border: "1px solid #f85149",
                   color: "#f85149", cursor: "pointer", fontSize: 12 }}>
        Delete Node
      </button>
    </div>`;
}

// ── STREAM PANEL ─────────────────────────────────────────────────────────────
function StreamPanel({ sessionId, wsPort, counters }) {
  const imgRef  = useRef(null);
  const [status, setStatus] = useState("idle");
  const [fps, setFps] = useState(0);
  const prevUrl = useRef(null);
  const fpsRef  = useRef({ n: 0, t: Date.now() });

  useEffect(() => {
    if (!sessionId) { setStatus("idle"); return; }
    const ws = new WebSocket(`ws://localhost:${wsPort}/ws/stream/${sessionId}`);
    ws.binaryType = "blob";
    setStatus("connecting");
    ws.onopen  = () => setStatus("live");
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("error");
    ws.onmessage = e => {
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      const u = URL.createObjectURL(e.data);
      if (imgRef.current) imgRef.current.src = u;
      prevUrl.current = u;
      fpsRef.current.n++;
      const now = Date.now();
      if (now - fpsRef.current.t >= 1000) {
        setFps(Math.round(fpsRef.current.n * 1000 / (now - fpsRef.current.t)));
        fpsRef.current = { n: 0, t: now };
      }
    };
    return () => ws.close();
  }, [sessionId, wsPort]);

  const dot = { idle: "#8b949e", connecting: "#d29922", live: "#3fb950",
                disconnected: "#f85149", error: "#f85149" }[status];

  return html`
    <div style=${{ borderTop: "1px solid #30363d", flexShrink: 0 }}>
      <div style=${{ display: "flex", alignItems: "center", gap: 8, padding: "7px 14px",
                      borderBottom: "1px solid #30363d" }}>
        <div style=${{ width: 7, height: 7, borderRadius: "50%", background: dot,
                        animation: status === "live" ? "blink 2s infinite" : "none" }} />
        <span style=${{ fontSize: 10, color: "#8b949e", textTransform: "uppercase", letterSpacing: ".5px" }}>
          Live Stream
        </span>
        ${status === "live" && html`<span style=${{ marginLeft: "auto", fontSize: 11, color: "#3fb950" }}>${fps} fps</span>`}
      </div>
      <div style=${{ background: "#000", height: 150, display: "flex", alignItems: "center", justifyContent: "center" }}>
        ${!sessionId
          ? html`<span style=${{ color: "#8b949e", fontSize: 11 }}>Run pipeline to stream</span>`
          : html`<img ref=${imgRef} style=${{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />`}
      </div>
      ${Object.keys(counters).length > 0 && html`
        <div style=${{ padding: "6px 14px" }}>
          ${Object.entries(counters).map(([k, v]) => html`
            <div key=${k} style=${{ display: "flex", justifyContent: "space-between",
                                     fontSize: 12, color: "#c9d1d9", marginBottom: 2 }}>
              <span style=${{ color: "#8b949e" }}>${k}</span><b>${v}</b>
            </div>`)}
        </div>`}
    </div>`;
}

// ── MAIN APP ─────────────────────────────────────────────────────────────────
let _id = 0;
const uid = () => ++_id;

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selId, setSelId]           = useState(null);
  const [name, setName]             = useState("Untitled Pipeline");
  const [pipelineId, setPid]        = useState(null);
  const [sessionId, setSid]         = useState(null);
  const [running, setRunning]       = useState(false);
  const [counters, setCounters]     = useState({});
  const [showSamples, setShowSamples] = useState(false);
  const [toast, setToast]           = useState(null);
  const evWsRef                     = useRef(null);
  const { project }                 = useReactFlow();

  const showToast = (msg, ok = true) => { setToast({ msg, ok }); setTimeout(() => setToast(null), 3000); };
  const selNode = nodes.find(n => n.id === selId) ?? null;

  const onConnect = useCallback(p => setEdges(es => addEdge({ ...p, sourceHandle: "out", targetHandle: "in" }, es)), [setEdges]);

  const onDragOver = useCallback(e => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }, []);
  const onDrop = useCallback(e => {
    e.preventDefault();
    const type = e.dataTransfer.getData("application/cvflow");
    if (!type) return;
    const pos  = project({ x: e.clientX - 162, y: e.clientY - 44 }); // subtract palette + toolbar width
    const meta = NODE_META[type];
    setNodes(ns => [...ns, {
      id: `${type}_${uid()}`, type,
      position: pos,
      data: { label: meta.label, config: { ...DEFAULT_CONFIG[type] } },
    }]);
  }, [project, setNodes]);

  const onNodeClick = useCallback((_, n) => setSelId(n.id), []);
  const onPaneClick = useCallback(() => setSelId(null), []);

  const onUpdate = useCallback((id, newCfg, newLabel) => {
    if (newCfg === null) { setNodes(ns => ns.filter(n => n.id !== id)); setSelId(null); return; }
    setNodes(ns => ns.map(n => n.id !== id ? n : {
      ...n, data: { ...n.data, config: newCfg, ...(newLabel !== undefined ? { label: newLabel } : {}) }
    }));
  }, [setNodes]);

  const loadSample = s => {
    _id = 0;
    setNodes(s.nodes); setEdges(s.edges); setName(s.name); setPid(null);
    setSelId(null); setShowSamples(false);
    showToast("Loaded: " + s.name);
  };

  const save = async () => {
    const payload = {
      version: "1.0", name,
      nodes: nodes.map(n => ({ id: n.id, type: n.type, label: n.data.label, position: n.position, config: n.data.config ?? {} })),
      edges: edges.map(e => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle ?? "out", targetHandle: e.targetHandle ?? "in" })),
    };
    try {
      const result = pipelineId
        ? await apiFetch("PUT", "/pipelines/" + pipelineId, payload)
        : await apiFetch("POST", "/pipelines", payload);
      if (!pipelineId) setPid(result.id);
      showToast(pipelineId ? "Updated" : "Saved");
      return result.id ?? pipelineId;
    } catch (e) { showToast("Save failed: " + e.message, false); return null; }
  };

  const run = async () => {
    let pid = pipelineId ?? await save();
    if (!pid) return;
    try {
      const r = await apiFetch("POST", "/execution/start", { pipeline_id: pid });
      setSid(r.session_id); setRunning(true); setCounters({});
      showToast("Started: " + r.session_id.slice(0, 8) + "…");
      if (evWsRef.current) evWsRef.current.close();
      const ws = new WebSocket(`ws://localhost:8765/ws/events/${r.session_id}`);
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "counter_update")
            setCounters(c => ({ ...c, [msg.counter_id ?? msg.node_id]: msg.value }));
        } catch {}
      };
      evWsRef.current = ws;
    } catch (e) { showToast("Start failed: " + e.message, false); }
  };

  const stop = async () => {
    if (!sessionId) return;
    try {
      await apiFetch("POST", "/execution/stop/" + sessionId);
      setRunning(false); setSid(null);
      if (evWsRef.current) { evWsRef.current.close(); evWsRef.current = null; }
      showToast("Stopped");
    } catch (e) { showToast("Stop failed: " + e.message, false); }
  };

  const btn = (variant, disabled) => ({
    padding: "4px 14px", borderRadius: 6, cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 12, fontWeight: 500, opacity: disabled ? .45 : 1,
    border: "1px solid", fontFamily: "inherit",
    ...({
      run:  { background: "#1a3d2e", borderColor: "#3fb950", color: "#3fb950" },
      stop: { background: "#3d1a1a", borderColor: "#f85149", color: "#f85149" },
      save: { background: "#1f3a5e", borderColor: "#58a6ff", color: "#58a6ff" },
      def:  { background: "#21262d", borderColor: "#30363d", color: "#c9d1d9" },
    }[variant] ?? {}),
  });

  return html`
    <div style=${{ display: "flex", flexDirection: "column", height: "100vh",
                   background: "#0d1117", color: "#c9d1d9", fontSize: 13 }}>

      <!-- TOOLBAR (44px) -->
      <div style=${{ display: "flex", alignItems: "center", gap: 8, padding: "0 12px",
                      height: 44, background: "#161b22", borderBottom: "1px solid #30363d", flexShrink: 0 }}>
        <span style=${{ fontSize: 15, fontWeight: 800, color: "#58a6ff", letterSpacing: 1.5, marginRight: 4 }}>CV-FLOW</span>

        <!-- Samples dropdown -->
        <div style=${{ position: "relative" }}>
          <button style=${btn("def")} onClick=${() => setShowSamples(s => !s)}>Samples ▾</button>
          ${showSamples && html`
            <div style=${{ position: "absolute", top: "110%", left: 0, zIndex: 200,
                            background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
                            minWidth: 280, boxShadow: "0 8px 24px rgba(0,0,0,.6)" }}>
              ${SAMPLES.map((s, i) => html`
                <div key=${i} onClick=${() => loadSample(s)}
                  style=${{ padding: "10px 14px", cursor: "pointer",
                              borderBottom: i < SAMPLES.length - 1 ? "1px solid #30363d" : "none" }}
                  onMouseEnter=${e => e.currentTarget.style.background = "#21262d"}
                  onMouseLeave=${e => e.currentTarget.style.background = ""}>
                  <div style=${{ fontWeight: 600, fontSize: 12, color: "#e2e8f0" }}>${s.name}</div>
                  <div style=${{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>${s.description}</div>
                </div>`)}
            </div>`}
        </div>

        <!-- Pipeline name -->
        <input value=${name} onChange=${e => setName(e.target.value)}
          style=${{ background: "transparent", border: "1px solid transparent", borderRadius: 5,
                     color: "#e2e8f0", fontSize: 13, padding: "3px 8px", outline: "none",
                     minWidth: 160, maxWidth: 260, fontFamily: "inherit" }}
          onFocus=${e => e.target.style.borderColor = "#30363d"}
          onBlur=${e  => e.target.style.borderColor = "transparent"} />

        <div style=${{ width: 1, height: 20, background: "#30363d" }} />
        <button style=${btn("save")}  onClick=${save}>Save</button>
        <button style=${btn("run",  running)} onClick=${run}  disabled=${running}>▶ Run</button>
        <button style=${btn("stop", !running)} onClick=${stop} disabled=${!running}>■ Stop</button>

        <!-- Session status -->
        <div style=${{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
          <div style=${{ width: 7, height: 7, borderRadius: "50%",
                          background: running ? "#3fb950" : "#8b949e",
                          animation: running ? "blink 1.8s infinite" : "none" }} />
          <span style=${{ color: running ? "#3fb950" : "#8b949e" }}>
            ${running ? (sessionId?.slice(0, 8) + "…") : "Idle"}
          </span>
        </div>
      </div>

      <!-- BODY -->
      <div style=${{ display: "flex", flex: 1, overflow: "hidden" }}>
        <${Palette} />

        <!-- CANVAS -->
        <div style=${{ flex: 1, position: "relative" }}>
          <${ReactFlow}
            nodes=${nodes} edges=${edges}
            onNodesChange=${onNodesChange} onEdgesChange=${onEdgesChange}
            onConnect=${onConnect} nodeTypes=${nodeTypes}
            onDrop=${onDrop} onDragOver=${onDragOver}
            onNodeClick=${onNodeClick} onPaneClick=${onPaneClick}
            fitView deleteKeyCode="Delete"
            defaultEdgeOptions=${{ style: { stroke: "#58a6ff", strokeWidth: 2 }, animated: false }}>
            <${Background} color="#21262d" gap=${24} size=${1} />
            <${Controls} />
            <${MiniMap} nodeColor=${n => GROUP_COLOR[NODE_META[n.type]?.group] ?? "#21262d"}
              maskColor="rgba(0,0,0,.6)" />
          <//>
          ${nodes.length === 0 && html`
            <div style=${{ position: "absolute", inset: 0, display: "flex", alignItems: "center",
                            justifyContent: "center", pointerEvents: "none" }}>
              <div style=${{ textAlign: "center", color: "#8b949e" }}>
                <div style=${{ fontSize: 32, marginBottom: 10 }}>◻</div>
                <div style=${{ fontSize: 14, marginBottom: 6 }}>Drag nodes from the palette</div>
                <div style=${{ fontSize: 12 }}>or click <b style=${{ color: "#58a6ff" }}>Samples ▾</b> to load a pre-built pipeline</div>
              </div>
            </div>`}
        </div>

        <!-- RIGHT PANEL (268px) -->
        <div style=${{ width: 268, flexShrink: 0, borderLeft: "1px solid #30363d",
                        display: "flex", flexDirection: "column", background: "#0d1117" }}>
          <div style=${{ padding: "7px 14px", borderBottom: "1px solid #30363d",
                          fontSize: 10, textTransform: "uppercase", letterSpacing: ".5px", color: "#8b949e" }}>
            Properties
          </div>
          <div style=${{ flex: 1, overflowY: "auto" }}>
            <${PropertiesPanel} node=${selNode} onUpdate=${onUpdate} />
          </div>
          <${StreamPanel} sessionId=${running ? sessionId : null} wsPort=${8765} counters=${counters} />
        </div>
      </div>

      <!-- TOAST -->
      ${toast && html`
        <div style=${{ position: "fixed", bottom: 20, right: 20, zIndex: 9999,
                        padding: "10px 16px", borderRadius: 8, fontSize: 12,
                        background: toast.ok ? "#1a3d2e" : "#3d1a1a",
                        border: "1px solid " + (toast.ok ? "#3fb950" : "#f85149"),
                        color: toast.ok ? "#3fb950" : "#f85149",
                        boxShadow: "0 4px 16px rgba(0,0,0,.5)", pointerEvents: "none" }}>
          ${toast.msg}
        </div>`}
    </div>`;
}

// ── BOOT ─────────────────────────────────────────────────────────────────────
createRoot(document.getElementById("root")).render(html`
  <${ReactFlowProvider}><${App} /><//>
`);
