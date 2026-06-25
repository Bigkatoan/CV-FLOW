import { createElement, useState, useCallback, useRef, useEffect, useMemo } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactFlow, Background, Controls, MiniMap,
  addEdge, useNodesState, useEdgesState, useReactFlow, ReactFlowProvider,
} from "reactflow";
import htm from "https://esm.sh/htm@3";
import { nodeTypes, NODE_META, GROUP_COLOR, NODE_PORTS, makeNode, registerNodeType } from "./nodes.js?v=30";
import { SAMPLES } from "./samples.js?v=30";

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
async function apiUploadModel(onnxFile, configFile) {
  const fd = new FormData();
  fd.append("model_file", onnxFile);
  fd.append("config_file", configFile);
  const r = await fetch(API + "/models/upload", { method: "POST", body: fd });
  if (!r.ok) { const t = await r.text(); throw new Error(`${r.status}: ${t}`); }
  return r.json();
}

// ── UTILITIES ─────────────────────────────────────────────────────────────────
// Parse UI-block helper calls from python code (slider / checkbox / text_input / button)
function parseParams(code) {
  if (!code) return [];
  const out = [];
  for (const m of code.matchAll(/\bslider\s*\(\s*["'](\w+)["']\s*,\s*([\d.-]+)\s*,\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\)/g))
    out.push({ type: "slider",   name: m[1], min: +m[2], max: +m[3], default: +m[4] });
  for (const m of code.matchAll(/\bcheckbox\s*\(\s*["'](\w+)["']\s*,\s*(True|False)\s*\)/gi))
    out.push({ type: "checkbox", name: m[1], default: m[2].toLowerCase() === "true" });
  for (const m of code.matchAll(/\btext_input\s*\(\s*["'](\w+)["']\s*,\s*["']([^"']*)["']\s*\)/g))
    out.push({ type: "text",     name: m[1], default: m[2] });
  for (const m of code.matchAll(/\bbutton\s*\(\s*["'](\w+)["']\s*\)/g))
    out.push({ type: "button",   name: m[1] });
  return out;
}

function parsePySignature(code) {
  // Prefer loop() or iteration() for function-based style
  const loopM = code.match(/def\s+(loop|iteration)\s*\(([^)]*)\)/);
  const fnName   = loopM ? loopM[1] : null;
  const paramStr = loopM
    ? loopM[2]
    : (code.match(/def\s+\w+\s*\(([^)]*)\)/) ?? [])[1] ?? "";
  if (!paramStr.trim()) return null;
  return paramStr.split(",").map(s => {
    const [namePart, ...rest] = s.trim().split("=");
    const name = namePart.split(":")[0].trim();
    const def  = rest.join("=").trim() || null;
    if (!name || name === "self") return null;
    // "active" in iteration is a control signal, not an input port
    if (fnName === "iteration" && name === "active") return null;
    return { name, def };
  }).filter(Boolean);
}

function parsePyOutputs(code) {
  // Find the last `return` line to infer output port names.
  const lines = code.split("\n");
  let returnExpr = null;
  for (let i = lines.length - 1; i >= 0; i--) {
    const t = lines[i].trim();
    if (t.startsWith("return ")) { returnExpr = t.slice(7).trim(); break; }
    if (t === "return")          { return [];                             }
  }
  if (!returnExpr || returnExpr === "None") return [];

  // Strip outer parens: return (frame, dets) → frame, dets
  if (returnExpr.startsWith("(") && returnExpr.endsWith(")"))
    returnExpr = returnExpr.slice(1, -1).trim();

  // Split on top-level commas (skip nested parens/brackets)
  const parts = [];
  let depth = 0, cur = "";
  for (const ch of returnExpr) {
    if ("([{".includes(ch)) depth++;
    else if (")]}".includes(ch)) depth--;
    else if (ch === "," && depth === 0) { parts.push(cur.trim()); cur = ""; continue; }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());

  return parts.map(expr => {
    // Extract the bare variable name: strip .method(), [idx], (args)
    const name = expr.replace(/\[.*/, "").replace(/\(.*/, "").replace(/\..*/, "").trim();
    return { id: name || "out", label: name || "out" };
  }).filter((p, i, arr) => p.id && arr.findIndex(x => x.id === p.id) === i); // dedupe
}

// ── DEFAULT CODE TEMPLATES ────────────────────────────────────────────────────
const DEFAULT_LOOP_CODE = `\
import cv2

def setup():
    # Runs once — import libs, load models, open resources.
    # "config" dict is available as a module-level global.
    pass

def loop(frame):
    # Called every frame. Parameters become input ports.
    #   "frame"     → ctx.frame (numpy BGR array)
    #   other names → ctx.metadata.get(name)
    # Return value (ndarray) → ctx.frame. StopIteration stops pipeline.
    if isinstance(frame, np.ndarray):
        show_image(frame)
        show_text(f"shape: {frame.shape}")
    else:
        show_text(f"frame: {type(frame).__name__}")
    return frame

def teardown():
    # Release resources (cameras, files, models).
    pass
`;

const DEFAULT_ITERATION_CODE = `\
def setup():
    pass

def iteration(frame, active):
    # Called every frame. Body runs only when active=True.
    # active_key in config sets which ctx.metadata key controls this.
    # "active" is a control signal — not an input port.
    if active:
        show_text("Active!")
    return frame

def teardown():
    pass
`;

const DEFAULT_CPP_CODE = `// C++ Node — compiled to a shared library by the backend.
// Entry points called by the engine:
//   cv_flow_setup(const char* config_json)
//   cv_flow_process(CvFlowContext* ctx)
//   cv_flow_teardown()

#include <opencv2/opencv.hpp>
#include <string>

extern "C" {

void cv_flow_setup(const char* config_json) {
    // Initialize once: parse config, load models, open files.
}

void cv_flow_process(CvFlowContext* ctx) {
    // Called every frame.
    // ctx->frame points to the current BGR frame (shared memory).
}

void cv_flow_teardown() {
    // Release resources.
}

} // extern "C"
`;

// ── DEFAULT CONFIGS ──────────────────────────────────────────────────────────
const DEFAULT_CONFIG = {
  python_node: { mode: "loop", active_key: "active", code: DEFAULT_LOOP_CODE },
  cpp_node:    { mode: "loop", source_code: DEFAULT_CPP_CODE, compile_status: "uncompiled" },
};

// ── BASE PALETTE GROUPS ───────────────────────────────────────────────────────
const BASE_GROUPS = [
  { label: "Core", types: ["python_node", "cpp_node"] },
];

// ── FORM FIELD STYLES ─────────────────────────────────────────────────────────
const inp = {
  width: "100%", background: "#21262d", border: "1px solid #30363d",
  borderRadius: 5, color: "#c9d1d9", padding: "4px 8px", fontSize: 12,
  outline: "none", fontFamily: "inherit",
};
const row  = { marginBottom: 10 };
const lbl  = { display: "block", fontSize: 10, color: "#8b949e", textTransform: "uppercase", letterSpacing: ".5px", marginBottom: 3 };

function Field({ label, children }) {
  return html`<div style=${row}><label style=${lbl}>${label}</label>${children}</div>`;
}

// ── PIPELINE VALIDATOR ────────────────────────────────────────────────────────
function validatePipeline(nodes, edges) {
  const warnings = [];
  if (nodes.length === 0) {
    warnings.push("Pipeline is empty — add at least one Python Node");
    return warnings;
  }
  const connectedTargets = new Set(edges.map(e => e.target));
  const connectedSources = new Set(edges.map(e => e.source));
  nodes.forEach(n => {
    const cfg = n.data.config ?? {};
    if ((n.type === "python_node" || n.type === "cpp_node") && !cfg.code?.trim() && !cfg.source_code?.trim())
      warnings.push(`"${n.data.label}" — code is empty`);
    const ports = n.data.ports ?? NODE_PORTS[n.type];
    if (!ports) return;
    if (nodes.length > 1) {
      if (ports.inputs.length > 0 && !connectedTargets.has(n.id))
        warnings.push(`"${n.data.label}" has no incoming connection`);
      if (ports.outputs.length > 0 && !connectedSources.has(n.id))
        warnings.push(`"${n.data.label}" has no outgoing connection`);
    }
  });
  return warnings;
}

// ── AUTO-LAYOUT (BFS left-to-right) ──────────────────────────────────────────
function autoLayout(nodes, edges) {
  if (!nodes.length) return nodes;
  const adj = {};
  const indegree = {};
  nodes.forEach(n => { adj[n.id] = []; indegree[n.id] = 0; });
  edges.forEach(e => {
    if (adj[e.source]) { adj[e.source].push(e.target); indegree[e.target] = (indegree[e.target] || 0) + 1; }
  });

  const queue = nodes.filter(n => !indegree[n.id]).map(n => n.id);
  const levels = {};
  queue.forEach(id => { levels[id] = 0; });

  const visited = new Set(queue);
  let qi = 0;
  while (qi < queue.length) {
    const cur = queue[qi++];
    (adj[cur] || []).forEach(next => {
      levels[next] = Math.max(levels[next] ?? 0, (levels[cur] ?? 0) + 1);
      if (!visited.has(next)) { visited.add(next); queue.push(next); }
    });
  }

  // Assign nodes not yet reached (disconnected)
  nodes.forEach(n => { if (levels[n.id] === undefined) levels[n.id] = 0; });

  // Count nodes per level for vertical spacing
  const levelCount = {};
  nodes.forEach(n => { const lv = levels[n.id]; levelCount[lv] = (levelCount[lv] || 0) + 1; });
  const levelIdx = {};

  return nodes.map(n => {
    const lv = levels[n.id] ?? 0;
    const idx = levelIdx[lv] = (levelIdx[lv] ?? 0) + 1;
    const total = levelCount[lv] ?? 1;
    return { ...n, position: { x: lv * 240 + 60, y: (idx - 1) * 180 + 60 - (total - 1) * 90 } };
  });
}

// ── PORTS INFO ────────────────────────────────────────────────────────────────
function PortsInfo({ node }) {
  const ports = node.data.ports ?? NODE_PORTS[node.type] ?? { inputs: [], outputs: [] };
  if (!ports.inputs.length && !ports.outputs.length) return null;
  const chip = (p, color, bg) => html`
    <span key=${p.id} style=${{ display: "inline-block", fontSize: 10, color: "#e2e8f0",
                                  background: bg, borderRadius: 3, padding: "1px 7px",
                                  marginRight: 4, marginBottom: 4, border: "1px solid " + color + "55" }}>
      <span style=${{ color }}>${p.id}</span>${p.label !== p.id ? html` · ${p.label}` : ""}
    </span>`;
  return html`
    <div style=${{ marginTop: 12, paddingTop: 10, borderTop: "1px solid #30363d" }}>
      <div style=${lbl}>Ports</div>
      ${ports.inputs.length > 0 && html`
        <div style=${{ marginBottom: 5 }}>
          <span style=${{ fontSize: 10, color: "#8b949e", marginRight: 4 }}>IN</span>
          ${ports.inputs.map(p => chip(p, "#58a6ff", "#1a2a40"))}
        </div>`}
      ${ports.outputs.length > 0 && html`
        <div>
          <span style=${{ fontSize: 10, color: "#8b949e", marginRight: 4 }}>OUT</span>
          ${ports.outputs.map(p => chip(p, "#3fb950", "#1a3a20"))}
        </div>`}
    </div>`;
}

// ── NODE LIVE PREVIEW ─────────────────────────────────────────────────────────
// Shows JPEG frames streamed from the engine for the selected node while running.
// For draw_roi / draw_line nodes the canvas is interactive — drag handles to
// reposition polygon vertices or line endpoints.
function NodePreviewCanvas({ sessionId, nodeId, nodeType, config, onConfigChange }) {
  const canvasRef    = useRef(null);
  const draggingRef  = useRef(null);    // { index } while mouse is down on a handle
  const prevUrlRef   = useRef(null);
  const frameSizeRef = useRef({ w: 640, h: 480 });  // natural frame dimensions
  const [frameSrc,  setFrameSrc]  = useState(null);
  const [connected, setConnected] = useState(false);

  const interactive = nodeType === "draw_roi" || nodeType === "draw_line";

  // Open WebSocket to receive JPEG frames from the engine
  useEffect(() => {
    setFrameSrc(null);
    setConnected(false);
    if (!sessionId || !nodeId) return;

    const ws = new WebSocket(`ws://localhost:8765/ws/node-preview/${sessionId}/${nodeId}`);
    ws.binaryType = "blob";
    ws.onopen  = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
      const url = URL.createObjectURL(e.data);
      prevUrlRef.current = url;
      setFrameSrc(url);
    };
    return () => {
      ws.close();
      if (prevUrlRef.current) { URL.revokeObjectURL(prevUrlRef.current); prevUrlRef.current = null; }
    };
  }, [sessionId, nodeId]);

  // Redraw canvas whenever a new frame or config change arrives
  useEffect(() => {
    if (!interactive || !canvasRef.current || !frameSrc) return;
    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const displayW = canvas.parentElement?.clientWidth ?? 240;
      const aspect   = img.naturalHeight / img.naturalWidth;
      canvas.width   = displayW;
      canvas.height  = Math.round(displayW * aspect);
      canvas.style.height = canvas.height + "px";
      frameSizeRef.current = { w: img.naturalWidth, h: img.naturalHeight };
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      _drawOverlay(ctx, canvas.width, canvas.height);
    };
    img.src = frameSrc;
  }, [frameSrc, config, interactive]);

  const _drawOverlay = (ctx, cw, ch) => {
    const { w: fw, h: fh } = frameSizeRef.current;
    const sx = cw / fw, sy = ch / fh;
    ctx.save();

    if (nodeType === "draw_roi") {
      const pts = (config.polygon ?? []).map(([x, y]) => [x * sx, y * sy]);
      if (pts.length >= 2) {
        ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
        ctx.beginPath();
        pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
        ctx.closePath(); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = "#3fb95022"; ctx.fill();
      }
      pts.forEach(([x, y], i) => {
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#3fb950"; ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = "9px monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(i, x, y);
      });

    } else if (nodeType === "draw_line") {
      const raw = config.line ?? [[10, 50], [90, 50]];
      const [[x1, y1], [x2, y2]] = raw.map(([x, y]) => [x * sx, y * sy]);
      ctx.strokeStyle = "#f0a040"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      [["A", x1, y1], ["B", x2, y2]].forEach(([label, x, y]) => {
        ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fillStyle = "#f0a040"; ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = "bold 8px monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(label, x, y);
      });
    }
    ctx.restore();
  };

  const _canvasPoint = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { w: fw, h: fh } = frameSizeRef.current;
    return {
      cx, cy,
      fx: Math.round(Math.max(0, Math.min(cx * fw / canvas.width,  fw))),
      fy: Math.round(Math.max(0, Math.min(cy * fh / canvas.height, fh))),
    };
  };

  const _hitPoint = (cx, cy, pts) => {
    const canvas = canvasRef.current;
    const { w: fw, h: fh } = frameSizeRef.current;
    const sx = canvas.width / fw, sy = canvas.height / fh;
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = pts[i];
      if (Math.hypot(cx - px * sx, cy - py * sy) < 10) return i;
    }
    return -1;
  };

  const onMouseDown = (e) => {
    if (!interactive) return;
    const pt = _canvasPoint(e);
    if (!pt) return;
    const { cx, cy, fx, fy } = pt;

    if (nodeType === "draw_roi") {
      const pts = config.polygon ?? [];
      const hit = _hitPoint(cx, cy, pts);
      if (hit >= 0) { draggingRef.current = { index: hit }; return; }
      onConfigChange({ ...config, polygon: [...pts, [fx, fy]] });
    } else if (nodeType === "draw_line") {
      const line = config.line ?? [[10, 50], [90, 50]];
      const hit  = _hitPoint(cx, cy, line);
      if (hit >= 0) draggingRef.current = { index: hit };
    }
  };

  const onMouseMove = (e) => {
    if (!interactive || !draggingRef.current) return;
    const pt = _canvasPoint(e);
    if (!pt) return;
    const { fx, fy } = pt;
    if (nodeType === "draw_roi") {
      const pts = [...(config.polygon ?? [])];
      pts[draggingRef.current.index] = [fx, fy];
      onConfigChange({ ...config, polygon: pts });
    } else if (nodeType === "draw_line") {
      const line = [...(config.line ?? [[10, 50], [90, 50]])];
      line[draggingRef.current.index] = [fx, fy];
      onConfigChange({ ...config, line });
    }
  };

  const onMouseUp    = () => { draggingRef.current = null; };
  const onDblClick   = (e) => {
    if (nodeType !== "draw_roi") return;
    const pt = _canvasPoint(e);
    if (!pt) return;
    const { cx, cy } = pt;
    const pts = config.polygon ?? [];
    const hit = _hitPoint(cx, cy, pts);
    if (hit >= 0 && pts.length > 3)
      onConfigChange({ ...config, polygon: pts.filter((_, j) => j !== hit) });
  };

  return html`
    <div style=${{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #30363d" }}>
      <div style=${{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
        <span style=${{ fontSize: 10, color: "#8b949e", textTransform: "uppercase", letterSpacing: ".5px" }}>
          Live Preview
        </span>
        <span style=${{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style=${{ width: 6, height: 6, borderRadius: "50%",
                          background: connected ? "#3fb950" : "#555d68",
                          animation: connected ? "blink 2s infinite" : "none" }} />
          ${interactive && html`<span style=${{ fontSize: 9, color: "#58a6ff" }}>
            ${nodeType === "draw_roi" ? "Click=add · Dbl=remove · Drag=move" : "Drag A/B endpoints"}
          </span>`}
        </span>
      </div>

      ${!frameSrc && html`
        <div style=${{ background: "#21262d", borderRadius: 4, height: 80,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 11, color: "#555d68" }}>
          ${connected ? "Waiting for frame…" : "Connecting…"}
        </div>`}

      ${interactive
        ? html`<canvas ref=${canvasRef}
              style=${{ width: "100%", display: frameSrc ? "block" : "none",
                         borderRadius: 4, cursor: "crosshair", background: "#000", userSelect: "none" }}
              onMouseDown=${onMouseDown}
              onMouseMove=${onMouseMove}
              onMouseUp=${onMouseUp}
              onMouseLeave=${onMouseUp}
              onDblClick=${onDblClick} />`
        : frameSrc && html`<img src=${frameSrc}
              style=${{ width: "100%", borderRadius: 4, display: "block", background: "#000" }} alt="" />`}

      ${interactive && html`
        <div style=${{ fontSize: 9, color: "#555d68", marginTop: 3, textAlign: "right" }}>
          ⚠ Config changes apply on next Run
        </div>`}
    </div>`;
}

// ── PORT EDITOR (shared by python_node / cpp_node) ────────────────────────────
function PortEditor({ node, onUpdate }) {
  const cfg = node.data.config ?? {};
  const DEFAULT_PORTS = { inputs: [{ id: "in", label: "in" }], outputs: [{ id: "out", label: "out" }] };
  const ports = node.data.ports ?? NODE_PORTS[node.type] ?? DEFAULT_PORTS;

  const push = (newPorts) => onUpdate(node.id, cfg, undefined, newPorts);

  const addPort = (side) => {
    const n = ports[side].length + 1;
    push({ ...ports, [side]: [...ports[side], { id: "p" + n, label: "p" + n }] });
  };
  const removePort = (side, i) => {
    push({ ...ports, [side]: ports[side].filter((_, j) => j !== i) });
  };
  const renamePort = (side, i, val) => {
    const id = val.trim().toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "") || ("p" + (i + 1));
    push({ ...ports, [side]: ports[side].map((p, j) => j === i ? { id, label: val.trim() || p.label } : p) });
  };

  const sideSection = (side, color, label) => html`
    <div style=${{ flex: 1 }}>
      <div style=${{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
        <span style=${{ fontSize: 10, color: "#8b949e", textTransform: "uppercase", letterSpacing: ".5px" }}>${label}</span>
        <button onClick=${() => addPort(side)}
          style=${{ fontSize: 10, background: "none", border: "1px solid " + color + "55",
                     color, borderRadius: 4, padding: "1px 7px", cursor: "pointer" }}>+ Add</button>
      </div>
      ${ports[side].map((p, i) => html`
        <div key=${i} style=${{ display: "flex", gap: 4, marginBottom: 4, alignItems: "center" }}>
          <span style=${{ color, fontSize: 11 }}>●</span>
          <input defaultValue=${p.label}
            onBlur=${e => renamePort(side, i, e.target.value)}
            onKeyDown=${e => e.key === "Enter" && (renamePort(side, i, e.target.value), e.target.blur())}
            style=${{ ...inp, flex: 1, fontSize: 11, fontFamily: "monospace" }}
            placeholder="label" />
          ${ports[side].length > 1 && html`
            <button onClick=${() => removePort(side, i)}
              style=${{ background: "none", border: "none", color: "#f85149", cursor: "pointer", fontSize: 13, padding: "0 4px" }}>×</button>`}
        </div>`)}
    </div>`;

  return html`
    <div style=${row}>
      <label style=${lbl}>Ports</label>
      <div style=${{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        ${sideSection("inputs",  "#58a6ff", "Inputs")}
        ${sideSection("outputs", "#3fb950", "Outputs")}
      </div>
    </div>`;
}

// ── PYTHON FUNCTION FIELDS ─────────────────────────────────────────────────────
function PythonFunctionFields({ node, cfg, onUpdate }) {
  const currentPorts = node.data.ports ?? NODE_PORTS.python_function ?? { inputs: [], outputs: [] };

  // Keep a local draft of output port labels so the user can rename before committing
  const [outputs, setOutputs] = useState(() => currentPorts.outputs);

  // Sync if another update arrives from outside (e.g. code change)
  const prevPortsRef = useRef(currentPorts);
  if (prevPortsRef.current !== currentPorts) {
    prevPortsRef.current = currentPorts;
    setOutputs(currentPorts.outputs);
  }

  const pushPorts = (newOutputs, newInputs) => {
    const ins = newInputs ?? currentPorts.inputs;
    onUpdate(node.id, cfg, undefined, { inputs: ins, outputs: newOutputs });
    setOutputs(newOutputs);
  };

  const handleCode = (code) => {
    const parsedIn  = parsePySignature(code);
    const parsedOut = parsePyOutputs(code);
    const newInputs = parsedIn
      ? parsedIn.map(p => ({ id: p.name, label: p.name + (p.def ? ` (=${p.def})` : "") }))
      : currentPorts.inputs;
    const newOutputs = parsedOut.length > 0 ? parsedOut : outputs;
    setOutputs(newOutputs);
    onUpdate(node.id, { ...cfg, code }, undefined, { inputs: newInputs, outputs: newOutputs });
  };

  const addOutput = () => {
    const n = outputs.length + 1;
    const next = [...outputs, { id: "out" + n, label: "out" + n }];
    pushPorts(next);
  };

  const removeOutput = (i) => {
    const next = outputs.filter((_, idx) => idx !== i);
    pushPorts(next.length ? next : []);
  };

  const renameOutput = (i, val) => {
    const name = val.trim() || ("out" + (i + 1));
    const next = outputs.map((p, idx) => idx === i ? { id: name, label: name } : p);
    pushPorts(next);
  };

  const smallBtn = (label, onClick, color="#8b949e") => html`
    <button onClick=${onClick}
      style=${{ padding: "1px 7px", fontSize: 10, cursor: "pointer", borderRadius: 4,
                 border: "1px solid #30363d", background: "#21262d", color, fontFamily: "inherit" }}>
      ${label}
    </button>`;

  return html`
    <${Field} label="Python Code">
      <textarea
        style=${{ ...inp, height: 160, resize: "vertical", lineHeight: 1.5,
                   fontFamily: "Consolas,'Courier New',monospace", fontSize: 11 }}
        value=${cfg.code ?? ""}
        onInput=${e => handleCode(e.target.value)} />
    <//>

    <!-- Inputs (read-only — auto from signature) -->
    <div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 2 }}>
      Inputs (from function params)
    </div>
    <div style=${{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 10 }}>
      ${currentPorts.inputs.length === 0
        ? html`<span style=${{ fontSize: 10, color: "#555d68" }}>none</span>`
        : currentPorts.inputs.map(p => html`
          <span key=${p.id} style=${{ fontSize: 10, background: "#1a2a40", color: "#58a6ff",
                                       borderRadius: 3, padding: "1px 7px", border: "1px solid #1a3a5e" }}>
            ● ${p.label}
          </span>`)}
    </div>

    <!-- Outputs (editable) -->
    <div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 4 }}>
      Outputs <span style=${{ color: "#555d68" }}>(auto-detected from return, or add manually)</span>
    </div>
    ${outputs.map((p, i) => html`
      <div key=${i} style=${{ display: "flex", gap: 6, alignItems: "center", marginBottom: 5 }}>
        <span style=${{ fontSize: 12, color: "#3fb950" }}>●</span>
        <input
          value=${p.label}
          onBlur=${e => renameOutput(i, e.target.value)}
          onKeyDown=${e => e.key === "Enter" && renameOutput(i, e.target.value)}
          style=${{ ...inp, flex: 1, fontSize: 11, fontFamily: "monospace" }} />
        ${smallBtn("×", () => removeOutput(i), "#f85149")}
      </div>`)}
    <div style=${{ marginBottom: 10 }}>
      ${smallBtn("+ Add Output", addOutput, "#3fb950")}
    </div>`;
}

// Returns true if this node type produces a frame/image output (so live preview makes sense)
function nodeHasFrameOutput(type) {
  const ports = NODE_PORTS[type];
  if (!ports || !ports.outputs.length) return false;
  return ports.outputs.some(p =>
    p.id === "frame" ||
    (p.id === "out" && p.label.toLowerCase() !== "params")
  );
}

// ── PIPELINE OUTPUT PORT EDITOR ───────────────────────────────────────────────
function PipelineOutputPortEditor({ node, cfg, onUpdate }) {
  const DEFAULT_INPUTS = [{ id: "frame", label: "Frame" }, { id: "dets", label: "Detections" }];
  const [inputs, setInputs] = useState(
    () => node.data.ports?.inputs ?? NODE_PORTS.pipeline_output?.inputs ?? DEFAULT_INPUTS
  );

  // Sync when ports change externally (e.g. undo/redo)
  useEffect(() => {
    const ext = node.data.ports?.inputs;
    if (ext) setInputs(ext);
  }, [node.data.ports]);

  const push = (newInputs) => {
    setInputs(newInputs);
    onUpdate(node.id, cfg, undefined, { inputs: newInputs, outputs: [] });
  };
  const add = () => {
    const n = inputs.length + 1;
    push([...inputs, { id: "port_" + n, label: "Port " + n }]);
  };
  const remove = (i) => push(inputs.filter((_, j) => j !== i));
  const rename = (i, label) => {
    const id = label.trim().toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "") || ("port_" + (i + 1));
    push(inputs.map((p, j) => j === i ? { id, label: label.trim() || p.label } : p));
  };

  const smallBtn = (lbl, onClick, color) => html`
    <button onClick=${onClick}
      style=${{ background: "none", border: "1px solid " + color, color, borderRadius: 5,
                 padding: "2px 8px", fontSize: 11, cursor: "pointer" }}>
      ${lbl}
    </button>`;

  return html`
    <${Field} label="Input Ports">
      ${inputs.map((p, i) => html`
        <div key=${i} style=${{ display: "flex", gap: 6, alignItems: "center", marginBottom: 5 }}>
          <span style=${{ fontSize: 12, color: "#58a6ff", flexShrink: 0 }}>●</span>
          <input
            defaultValue=${p.label}
            onBlur=${e => rename(i, e.target.value)}
            onKeyDown=${e => e.key === "Enter" && (rename(i, e.target.value), e.target.blur())}
            style=${{ ...inp, flex: 1, fontSize: 11 }}
            placeholder="Port label" />
          ${inputs.length > 1 && smallBtn("×", () => remove(i), "#f85149")}
        </div>`)}
      <div style=${{ marginTop: 4 }}>${smallBtn("+ Add Input", add, "#3fb950")}</div>
      <div style=${{ fontSize: 10, color: "#555d68", marginTop: 5 }}>
        Rename: click the label, edit, then press Enter or click away.
      </div>
    <//>`;
}

// ── CODEMIRROR PYTHON EDITOR ──────────────────────────────────────────────────
// Lazy-loads all CodeMirror packages once; subsequent node-remounts are instant.
let _cm = null;
async function _loadCM() {
  if (_cm) return _cm;
  const [stMod, vwMod, cmdMod, pyMod, acMod, thmMod] = await Promise.all([
    import("@codemirror/state"),
    import("@codemirror/view"),
    import("@codemirror/commands"),
    import("@codemirror/lang-python"),
    import("@codemirror/autocomplete"),
    import("@codemirror/theme-one-dark"),
  ]);
  _cm = { stMod, vwMod, cmdMod, pyMod, acMod, thmMod };
  return _cm;
}

function PythonEditor({ value, onChange, onBlur, minHeight = 260 }) {
  const containerRef  = useRef(null);
  const viewRef       = useRef(null);
  const remoteRef     = useRef(false); // true while we dispatch an external sync
  const onChangeRef   = useRef(onChange);
  const onBlurRef     = useRef(onBlur);
  onChangeRef.current = onChange;
  onBlurRef.current   = onBlur;

  // Jedi completion source — calls /api/python/complete
  const jediSource = async (context) => {
    const word = context.matchBefore(/[\w.]+/);
    if (!word && !context.explicit) return null;
    const code   = context.state.doc.toString();
    const before = code.slice(0, context.pos);
    const lines  = before.split("\n");
    const builtins = [
      { label: "show_image", type: "function", detail: "(img, label='') → None" },
      { label: "show_text",  type: "function", detail: "(text) → None" },
      { label: "config",     type: "variable", detail: "node config dict" },
    ];
    try {
      const r = await apiFetch("POST", "/python/complete", {
        code, line: lines.length, column: lines[lines.length - 1].length,
      });
      const jediOpts = (r.completions ?? []).map(c => ({
        label:  c.name,
        type:   c.type === "function" ? "function"
               : c.type === "class"   ? "class"
               : c.type === "module"  ? "namespace"
               : c.type === "keyword" ? "keyword"
               : "variable",
        detail: c.description || undefined,
      }));
      return { from: word ? word.from : context.pos, options: [...builtins, ...jediOpts] };
    } catch {
      return { from: word ? word.from : context.pos, options: builtins };
    }
  };

  useEffect(() => {
    let view;
    (async () => {
      if (!containerRef.current) return;
      const { stMod, vwMod, cmdMod, pyMod, acMod, thmMod } = await _loadCM();
      if (!containerRef.current) return; // unmounted while loading
      const { EditorState }                                                          = stMod;
      const { EditorView, keymap, lineNumbers, highlightActiveLine,
              highlightActiveLineGutter, drawSelection }                             = vwMod;
      const { history, historyKeymap, defaultKeymap, indentWithTab }                = cmdMod;
      const { python }                                                               = pyMod;
      const { autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap } = acMod;
      const { oneDark }                                                              = thmMod;

      view = new EditorView({
        state: EditorState.create({
          doc: value ?? "",
          extensions: [
            history(),
            lineNumbers(),
            highlightActiveLine(),
            highlightActiveLineGutter(),
            drawSelection(),
            keymap.of([
              indentWithTab,
              ...closeBracketsKeymap,
              ...defaultKeymap,
              ...historyKeymap,
              ...completionKeymap,
            ]),
            closeBrackets(),
            python(),
            autocompletion({ override: [jediSource], activateOnTyping: true }),
            oneDark,
            EditorView.theme({
              "&":                        { background: "#0d1117", minHeight: minHeight + "px" },
              ".cm-content":              { caretColor: "#79c0ff",  minHeight: minHeight + "px" },
              ".cm-gutters":              { background: "#0d1117",  borderRight: "1px solid #21262d", color: "#555d68" },
              ".cm-activeLine":           { background: "#ffffff08" },
              ".cm-activeLineGutter":     { background: "#ffffff08" },
              ".cm-scroller":             { fontFamily: "Consolas,'Fira Code','Courier New',monospace", fontSize: "11px", lineHeight: "1.65", overflow: "auto" },
              ".cm-tooltip.cm-tooltip-autocomplete": { background: "#161b22", border: "1px solid #30363d" },
              ".cm-tooltip-autocomplete ul li":      { color: "#c9d1d9" },
              ".cm-tooltip-autocomplete ul li[aria-selected]": { background: "#1f6feb", color: "#fff" },
            }),
            EditorView.updateListener.of(update => {
              if (update.docChanged && !remoteRef.current)
                onChangeRef.current?.(update.state.doc.toString());
              if (update.focusChanged && !update.view.hasFocus)
                onBlurRef.current?.(update.state.doc.toString());
            }),
          ],
        }),
        parent: containerRef.current,
      });
      viewRef.current = view;
    })();

    return () => { view?.destroy(); viewRef.current = null; };
  }, []); // mount once — use key prop on parent to remount when node changes

  // Sync value changes that originate externally (e.g. mode switch loads new template)
  useEffect(() => {
    const v = viewRef.current;
    if (!v || v.hasFocus) return;
    const cur = v.state.doc.toString();
    if (cur === (value ?? "")) return;
    remoteRef.current = true;
    v.dispatch({ changes: { from: 0, to: cur.length, insert: value ?? "" } });
    remoteRef.current = false;
  }, [value]);

  return html`<div ref=${containerRef} style=${{ border: "1px solid #30363d", borderRadius: 6, overflow: "hidden" }} />`;
}

// ── PROPERTIES PANEL ──────────────────────────────────────────────────────────
function PropertiesPanel({ node, onUpdate, onDuplicate, sessionId, running, nodeDataMap, nodeVizMap, onEditCode, onSaveNode }) {
  if (!node) return html`
    <div style=${{ padding: 16, color: "#8b949e", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
      <div style=${{ fontSize: 24, marginBottom: 8 }}>◻</div>
      Click a node to edit its config
      <div style=${{ marginTop: 20, fontSize: 11, lineHeight: 1.9, color: "#555d68" }}>
        <div>Ctrl+Z / Ctrl+Y — Undo / Redo</div>
        <div>Ctrl+D — Duplicate selected</div>
        <div>Ctrl+S — Save</div>
        <div>Ctrl+K — Quick-add node</div>
        <div>Delete — Remove selected node</div>
        <div>? — Shortcuts help</div>
      </div>
    </div>`;

  const cfg = node.data.config ?? {};
  const set = (key, val) => onUpdate(node.id, { ...cfg, [key]: val });

  // ── Python Node ──────────────────────────────────────────────────────────────
  const renderPythonFields = () => {
    const mode = cfg.mode ?? "loop";
    const setMode = (m) => {
      const next = { ...cfg, mode: m };
      if (!cfg.code?.trim()) next.code = m === "iteration" ? DEFAULT_ITERATION_CODE : DEFAULT_LOOP_CODE;
      onUpdate(node.id, next);
    };

    const modeBtn = (m, label, desc) => html`
      <button onClick=${() => setMode(m)}
        style=${{
          flex: 1, padding: "6px 10px", borderRadius: 6, cursor: "pointer",
          fontSize: 11, fontFamily: "inherit", textAlign: "center",
          background: mode === m ? (m === "iteration" ? "#2d1a4a" : "#1a2d1a") : "#21262d",
          border: "1px solid " + (mode === m ? (m === "iteration" ? "#9a6bdc" : "#3fb950") : "#30363d"),
          color:  mode === m ? (m === "iteration" ? "#c9a0ff" : "#3fb950")  : "#8b949e",
        }}>
        <div style=${{ fontWeight: 700, fontSize: 12 }}>${label}</div>
        <div style=${{ fontSize: 9, opacity: 0.8, marginTop: 1 }}>${desc}</div>
      </button>`;

    // ── Hyperparameter blocks (parsed from slider/checkbox/... calls in setup()) ──
    const params = parseParams(cfg.code ?? "");

    const renderParam = (p) => {
      const cur = cfg[p.name] ?? p.default;
      switch (p.type) {
        case "slider": return html`
          <div key=${p.name} style=${{ marginBottom: 10 }}>
            <div style=${{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <label style=${{ fontSize: 11, color: "#c9d1d9" }}>${p.name}</label>
              <span style=${{ fontSize: 11, color: "#79c0ff", fontFamily: "monospace", minWidth: 36, textAlign: "right" }}>
                ${typeof cur === "number" ? (Number.isInteger(cur) ? cur : cur.toFixed(2)) : cur}
              </span>
            </div>
            <input type="range" min=${p.min} max=${p.max}
              step=${(p.max - p.min) >= 2 ? 1 : 0.01}
              value=${cur}
              onInput=${e => set(p.name, +e.target.value)}
              style=${{ width: "100%", accentColor: "#58a6ff", cursor: "pointer" }} />
            <div style=${{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#555d68" }}>
              <span>${p.min}</span><span>${p.max}</span>
            </div>
          </div>`;

        case "checkbox": return html`
          <div key=${p.name} style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <input type="checkbox" id=${"cb-" + node.id + "-" + p.name}
              checked=${!!cur}
              onChange=${e => set(p.name, e.target.checked)}
              style=${{ accentColor: "#58a6ff", cursor: "pointer", width: 14, height: 14 }} />
            <label for=${"cb-" + node.id + "-" + p.name}
              style=${{ fontSize: 12, color: "#c9d1d9", cursor: "pointer" }}>${p.name}</label>
          </div>`;

        case "text": return html`
          <div key=${p.name} style=${{ marginBottom: 8 }}>
            <label style=${{ fontSize: 11, color: "#8b949e", display: "block", marginBottom: 3 }}>${p.name}</label>
            <input style=${inp} value=${cur}
              onBlur=${e => set(p.name, e.target.value)}
              onKeyDown=${e => e.key === "Enter" && set(p.name, e.target.value)} />
          </div>`;

        case "button": return html`
          <button key=${p.name}
            style=${{ width: "100%", marginBottom: 8, padding: "6px 0", borderRadius: 6, cursor: "pointer",
                       background: "#21262d", border: "1px solid #30363d", color: "#c9d1d9",
                       fontSize: 12, fontFamily: "inherit" }}>
            ${p.name}
          </button>`;

        default: return null;
      }
    };

    const ports    = node.data.ports ?? { inputs: [], outputs: [] };
    const vizSnap  = nodeVizMap?.[node.id];
    const vizItems = vizSnap?.items ?? [];

    const portChip = (p, color, bg, border) => html`
      <span key=${p.id} style=${{
        fontSize: 10, background: bg, color, borderRadius: 3,
        padding: "1px 8px", border: "1px solid " + border,
      }}>${p.label}</span>`;

    return html`
      <!-- Mode selector -->
      <div style=${row}>
        <label style=${lbl}>Execution Mode</label>
        <div style=${{ display: "flex", gap: 6 }}>
          ${modeBtn("loop",      "LOOP",      "always runs")}
          ${modeBtn("iteration", "ITERATION", "runs when active")}
        </div>
      </div>

      <!-- Active key (iteration only) -->
      ${mode === "iteration" && html`
        <${Field} label="Active Key — ctx.metadata key that activates this node">
          <input style=${inp} value=${cfg.active_key ?? "active"}
            onChange=${e => set("active_key", e.target.value)}
            placeholder="active" />
          <div style=${{ fontSize: 10, color: "#555d68", marginTop: 3 }}>
            When ctx.metadata["${cfg.active_key ?? "active"}"] is True, the node body runs.
          </div>
        <//>
      `}

      <!-- Edit Code button -->
      <button onClick=${onEditCode}
        style=${{ width: "100%", padding: "8px 0", borderRadius: 6, cursor: "pointer", marginBottom: 12,
                   background: "#1a2d4a", border: "1px solid #1f6feb",
                   color: "#58a6ff", fontSize: 12, fontFamily: "inherit", fontWeight: 600 }}>
        ✏  Open Code Editor
      </button>

      <!-- Hyperparameters (auto-parsed from slider/checkbox/... in setup()) -->
      ${params.length > 0 && html`
        <div style=${{ marginBottom: 8 }}>
          <div style=${{ fontSize: 10, fontWeight: 700, color: "#8b949e", letterSpacing: .5,
                          textTransform: "uppercase", marginBottom: 8, borderBottom: "1px solid #21262d", paddingBottom: 4 }}>
            Parameters
          </div>
          ${params.map(renderParam)}
        </div>`}

      <!-- Ports (read-only chips) -->
      <div style=${{ display: "flex", gap: 14, marginBottom: 12 }}>
        <div style=${{ flex: 1 }}>
          <div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 4 }}>Inputs</div>
          <div style=${{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            ${ports.inputs.length === 0
              ? html`<span style=${{ fontSize: 10, color: "#555d68" }}>none</span>`
              : ports.inputs.map(p => portChip(p, "#58a6ff", "#0d1a2d", "#1a3a5e"))}
          </div>
        </div>
        <div style=${{ flex: 1 }}>
          <div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 4 }}>Outputs</div>
          <div style=${{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            ${ports.outputs.length === 0
              ? html`<span style=${{ fontSize: 10, color: "#555d68" }}>none</span>`
              : ports.outputs.map(p => portChip(p, "#3fb950", "#0d1f0d", "#1a3a1a"))}
          </div>
        </div>
      </div>

      <!-- Live viz output (show_image / show_text) -->
      ${vizItems.length > 0 && html`
        <div style=${{ border: "1px solid #30363d", borderRadius: 6, overflow: "hidden", marginBottom: 8 }}>
          <div style=${{ padding: "4px 10px", background: "#21262d", fontSize: 10, color: "#8b949e",
                          letterSpacing: 0.5, textTransform: "uppercase", borderBottom: "1px solid #30363d" }}>
            Live Output
          </div>
          ${vizItems.map((item, i) =>
            item.type === "viz_image"
              ? html`<div key=${i} style=${{ padding: "8px 10px", background: "#0d1117" }}>
                  ${item.label && html`<div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 4 }}>${item.label}</div>`}
                  <img src=${"data:image/jpeg;base64," + item.b64}
                    style=${{ width: "100%", borderRadius: 4, display: "block" }} />
                </div>`
              : html`<pre key=${i} style=${{
                  margin: 0, padding: "8px 10px", fontSize: 11, color: "#79c0ff",
                  fontFamily: "Consolas,'Fira Code',monospace", background: "#0d1117",
                  borderTop: "1px solid #21262d", whiteSpace: "pre-wrap", wordBreak: "break-word",
                }}>${item.data}</pre>`
          )}
        </div>`}`;
  };

  // ── C++ Node ─────────────────────────────────────────────────────────────────
  const renderCppFields = () => html`
    <${Field} label="Compile Status">
      <div style=${{ fontSize: 12, padding: "2px 0",
                      color: cfg.compile_status === "ok" ? "#3fb950"
                           : cfg.compile_status === "error" ? "#f85149"
                           : "#8b949e" }}>
        ${cfg.compile_status ?? "uncompiled"}
      </div>
    <//>
    <${Field} label="Compiled .so Hash">
      <input style=${{ ...inp, fontFamily: "monospace", fontSize: 10 }}
        readOnly value=${cfg.compiled_so_hash ?? "(compile first)"} />
    <//>

    <${PortEditor} node=${node} onUpdate=${onUpdate} />

    <${Field} label="C++ Source Code">
      <div style=${{ fontSize: 10, color: "#555d68", marginBottom: 4 }}>
        Use <code style=${{ color: "#79c0ff" }}>extern "C"</code> entry points:
        <code style=${{ color: "#79c0ff" }}>cv_flow_setup</code>,
        <code style=${{ color: "#79c0ff" }}>cv_flow_process</code>,
        <code style=${{ color: "#79c0ff" }}>cv_flow_teardown</code>
      </div>
      <textarea
        style=${{ ...inp, height: 280, resize: "vertical", lineHeight: 1.55,
                   fontFamily: "Consolas,'Fira Code','Courier New',monospace", fontSize: 11 }}
        value=${cfg.source_code ?? ""}
        onInput=${e => set("source_code", e.target.value)} />
    <//>`;

  // ── Template node ─────────────────────────────────────────────────────────────
  const renderTemplateFields = () => {
    const pj = cfg.pipeline_json;
    const cfgKeys = Object.keys(cfg).filter(k => k !== "pipeline_json");
    return html`
      ${pj && html`
        <div style=${{ background: "#0d1117", border: "1px solid #21262d", borderRadius: 6,
                        padding: "8px 12px", marginBottom: 8, fontSize: 11 }}>
          <div style=${{ fontWeight: 600, color: "#d29922", marginBottom: 5 }}>⬡ Embedded Pipeline</div>
          <div style=${{ color: "#8b949e" }}>${(pj.nodes ?? []).length} nodes · ${(pj.edges ?? []).length} edges</div>
          <div style=${{ color: "#555d68", fontSize: 10, marginTop: 4 }}>
            ${(pj.nodes ?? []).map(n => n.type).join(" → ")}
          </div>
        </div>`}
      <${PortsInfo} node=${node} />
      ${cfgKeys.length > 0 && html`
        <${Field} label="Config">
          <textarea style=${{ ...inp, height: 100, resize: "vertical",
                               fontFamily: "Consolas,'Courier New',monospace", fontSize: 10 }}
            value=${JSON.stringify(Object.fromEntries(cfgKeys.map(k => [k, cfg[k]])), null, 2)}
            onChange=${e => { try {
              const parsed = JSON.parse(e.target.value);
              onUpdate(node.id, { ...parsed, ...(pj ? { pipeline_json: pj } : {}) });
            } catch {} }} />
        <//>
      `}`;
  };

  let fields;
  switch (node.type) {
    case "python_node": fields = renderPythonFields(); break;
    case "cpp_node":    fields = renderCppFields();    break;
    default:
      if (node.type.startsWith("tmpl_")) { fields = renderTemplateFields(); break; }
      // Unknown node type — show raw config
      fields = html`
        <${PortsInfo} node=${node} />
        <${Field} label="Config (JSON)">
          <textarea style=${{ ...inp, height: 120, resize: "vertical",
                               fontFamily: "Consolas,'Courier New',monospace", fontSize: 10 }}
            value=${JSON.stringify(cfg, null, 2)}
            onChange=${e => { try { onUpdate(node.id, JSON.parse(e.target.value)); } catch {} }} />
        <//>`;
  }

  if (false) { // legacy dead code — all old node types removed
    switch (node.type) {
    case "usb_camera": fields = html`
      <${Field} label="Device Index (0 = default camera)"><input type="number" style=${inp} value=${cfg.device_index ?? 0} onChange=${e => set("device_index", +e.target.value)} /><//>
      <${Field} label="FPS Limit (0 = unlimited)"><input type="number" style=${inp} value=${cfg.fps_limit ?? 30} onChange=${e => set("fps_limit", +e.target.value)} /><//>
      <${Field} label="Width (0 = camera default)"><input type="number" style=${inp} value=${cfg.width ?? 0} onChange=${e => set("width", +e.target.value)} /><//>
      <${Field} label="Height (0 = camera default)"><input type="number" style=${inp} value=${cfg.height ?? 0} onChange=${e => set("height", +e.target.value)} /><//>
      `; break;

    case "rtsp_stream": fields = html`
      <${Field} label="RTSP URL"><input style=${inp} value=${cfg.url ?? ""} onChange=${e => set("url", e.target.value)} placeholder="rtsp://192.168.1.x:554/stream" /><//>
      <${Field} label="FPS Limit (0 = stream rate)"><input type="number" style=${inp} value=${cfg.fps_limit ?? 30} onChange=${e => set("fps_limit", +e.target.value)} /><//>
      <${Field} label="Reconnect Delay (s)"><input type="number" style=${inp} value=${cfg.reconnect_delay_s ?? 3} onChange=${e => set("reconnect_delay_s", +e.target.value)} /><//>
      `; break;

    case "video_file": fields = html`
      <${Field} label="File Path"><input style=${inp} value=${cfg.file_path ?? ""} onChange=${e => set("file_path", e.target.value)} placeholder="C:/video.mp4" /><//>
      <${Field} label="Loop">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${!!cfg.loop} onChange=${e => set("loop", e.target.checked)} />
          <span style=${{ color: "#c9d1d9", fontSize: 12 }}>Loop video</span>
        </label>
      <//>
      <${Field} label="FPS Limit (0 = native)"><input type="number" style=${inp} value=${cfg.fps_limit ?? 0} onChange=${e => set("fps_limit", +e.target.value)} /><//>
      `; break;

    case "image_directory": fields = html`
      <${Field} label="Directory Path"><input style=${inp} value=${cfg.directory_path ?? ""} onChange=${e => set("directory_path", e.target.value)} placeholder="C:/images" /><//>
      <${Field} label="File Pattern"><input style=${inp} value=${cfg.pattern ?? "*.jpg"} onChange=${e => set("pattern", e.target.value)} /><//>
      <${Field} label="Delay Between Frames (ms)"><input type="number" style=${inp} value=${cfg.delay_ms ?? 100} onChange=${e => set("delay_ms", +e.target.value)} /><//>
      `; break;

    case "preprocess": fields = html`
      <${Field} label="Normalize">
        <select style=${inp} value=${cfg.normalize ?? "none"} onChange=${e => set("normalize", e.target.value)}>
          <option value="none">None</option><option value="imagenet">ImageNet (÷255, mean/std)</option><option value="min_max">Min-Max [0,1]</option>
        </select>
      <//>
      <${Field} label="Resize Width (0 = no resize)"><input type="number" style=${inp} value=${cfg.resize_w ?? 0} onChange=${e => set("resize_w", +e.target.value)} /><//>
      <${Field} label="Resize Height (0 = no resize)"><input type="number" style=${inp} value=${cfg.resize_h ?? 0} onChange=${e => set("resize_h", +e.target.value)} /><//>
      `; break;

    case "model_inference": fields = html`
      <${Field} label="Model ID">
        <input style=${{ ...inp, fontFamily: "monospace", fontSize: 11 }} value=${cfg.model_id ?? ""}
          onChange=${e => set("model_id", e.target.value)} placeholder="UUID — copy from Models panel" />
      <//>
      ${!cfg.model_id && html`<div style=${{ fontSize: 10, color: "#f85149", marginBottom: 8 }}>⚠ No model selected — open Models panel to upload</div>`}
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
      <${Field} label="Max Detections"><input type="number" style=${inp} value=${cfg.max_detections ?? 300} onChange=${e => set("max_detections", +e.target.value)} /><//>
      `; break;

    case "draw_bbox": fields = html`
      <${Field} label="Thickness">
        <input type="number" style=${inp} min="1" max="8" value=${cfg.thickness ?? 2}
          onChange=${e => set("thickness", +e.target.value)} />
      <//>
      <${Field} label=${"Font Scale: " + (cfg.font_scale ?? 0.45)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0.2" max="1.2" step="0.05"
          value=${cfg.font_scale ?? 0.45} onChange=${e => set("font_scale", +e.target.value)} />
      <//>
      <${Field} label="Show Label">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.show_label !== false}
            onChange=${e => set("show_label", e.target.checked)} />
          <span style=${{ fontSize: 12, color: "#c9d1d9" }}>Class name</span>
        </label>
      <//>
      <${Field} label="Show Confidence">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.show_confidence !== false}
            onChange=${e => set("show_confidence", e.target.checked)} />
          <span style=${{ fontSize: 12, color: "#c9d1d9" }}>Score</span>
        </label>
      <//>
      <${Field} label="Show Track ID">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.show_track_id !== false}
            onChange=${e => set("show_track_id", e.target.checked)} />
          <span style=${{ fontSize: 12, color: "#c9d1d9" }}>Track #ID (if tracked)</span>
        </label>
      <//>
      `; break;

    case "crop_bbox": fields = html`
      <${Field} label="Output Size (px)">
        <input type="number" style=${inp} min="8" max="1024" step="8"
          value=${cfg.image_size ?? 112} onChange=${e => set("image_size", +e.target.value)} />
        <div style=${{ fontSize: 10, color: "#8b949e", marginTop: 3 }}>Each crop is resized to N×N pixels</div>
      <//>
      <${Field} label=${"Padding: " + (cfg.padding ?? 0)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="0.5" step="0.05"
          value=${cfg.padding ?? 0} onChange=${e => set("padding", +e.target.value)} />
        <div style=${{ fontSize: 10, color: "#8b949e", marginTop: 3 }}>Expand each bbox by this fraction before cropping</div>
      <//>
      `; break;

    case "blur": fields = html`
      <${Field} label="Blur Type">
        <select style=${inp} value=${cfg.type ?? "gaussian"} onChange=${e => set("type", e.target.value)}>
          <option value="gaussian">Gaussian</option><option value="box">Box (Average)</option><option value="median">Median</option>
        </select>
      <//>
      <${Field} label="Kernel Size (odd number)"><input type="number" style=${inp} min="1" step="2" value=${cfg.kernel_size ?? 5} onChange=${e => set("kernel_size", +e.target.value)} /><//>
      ${(cfg.type ?? "gaussian") === "gaussian" && html`
        <${Field} label="Sigma (0 = auto)"><input type="number" style=${inp} value=${cfg.sigma ?? 0} onChange=${e => set("sigma", +e.target.value)} /><//>
      `}`; break;

    case "edge_detect": fields = html`
      <${Field} label="Algorithm">
        <select style=${inp} value=${cfg.algorithm ?? "canny"} onChange=${e => set("algorithm", e.target.value)}>
          <option value="canny">Canny</option><option value="sobel">Sobel</option><option value="laplacian">Laplacian</option>
        </select>
      <//>
      ${(cfg.algorithm ?? "canny") === "canny" && html`
        <${Field} label=${"Threshold 1: " + (cfg.threshold1 ?? 50)}>
          <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="500" step="5"
            value=${cfg.threshold1 ?? 50} onChange=${e => set("threshold1", +e.target.value)} />
        <//>
        <${Field} label=${"Threshold 2: " + (cfg.threshold2 ?? 150)}>
          <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="500" step="5"
            value=${cfg.threshold2 ?? 150} onChange=${e => set("threshold2", +e.target.value)} />
        <//>
      `}`; break;

    case "corner_detect": fields = html`
      <${Field} label="Algorithm">
        <select style=${inp} value=${cfg.algorithm ?? "harris"} onChange=${e => set("algorithm", e.target.value)}>
          <option value="harris">Harris</option><option value="fast">FAST</option><option value="shitomasi">Shi-Tomasi</option>
        </select>
      <//>
      <${Field} label="Max Corners"><input type="number" style=${inp} value=${cfg.max_corners ?? 100} onChange=${e => set("max_corners", +e.target.value)} /><//>
      <${Field} label=${"Quality: " + (cfg.quality ?? 0.01)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0.001" max="0.1" step="0.001"
          value=${cfg.quality ?? 0.01} onChange=${e => set("quality", +e.target.value)} />
      <//>
      <${Field} label="Min Distance (px)"><input type="number" style=${inp} value=${cfg.min_dist ?? 10} onChange=${e => set("min_dist", +e.target.value)} /><//>
      `; break;

    case "threshold": fields = html`
      <${Field} label="Type">
        <select style=${inp} value=${cfg.type ?? "binary"} onChange=${e => set("type", e.target.value)}>
          <option value="binary">Binary</option><option value="binary_inv">Binary Inverse</option>
          <option value="otsu">Otsu (auto)</option><option value="adaptive">Adaptive Gaussian</option>
        </select>
      <//>
      ${(cfg.type !== "otsu" && cfg.type !== "adaptive") && html`
        <${Field} label=${"Threshold: " + (cfg.threshold ?? 127)}>
          <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="255" step="1"
            value=${cfg.threshold ?? 127} onChange=${e => set("threshold", +e.target.value)} />
        <//>
      `}
      <${Field} label="Max Value"><input type="number" style=${inp} value=${cfg.max_val ?? 255} onChange=${e => set("max_val", +e.target.value)} /><//>
      `; break;

    case "color_convert": fields = html`
      <${Field} label="Conversion">
        <select style=${inp} value=${cfg.conversion ?? "bgr2gray"} onChange=${e => set("conversion", e.target.value)}>
          <option value="bgr2gray">BGR → Grayscale</option><option value="bgr2hsv">BGR → HSV</option>
          <option value="bgr2rgb">BGR → RGB</option><option value="bgr2lab">BGR → Lab</option>
          <option value="bgr2yuv">BGR → YUV</option><option value="gray2bgr">Grayscale → BGR</option>
          <option value="hsv2bgr">HSV → BGR</option>
        </select>
      <//>
      `; break;

    case "morph": fields = html`
      <${Field} label="Operation">
        <select style=${inp} value=${cfg.operation ?? "erode"} onChange=${e => set("operation", e.target.value)}>
          <option value="erode">Erode</option><option value="dilate">Dilate</option>
          <option value="open">Open (erode→dilate)</option><option value="close">Close (dilate→erode)</option>
          <option value="gradient">Gradient</option><option value="tophat">Top Hat</option><option value="blackhat">Black Hat</option>
        </select>
      <//>
      <${Field} label="Kernel Size"><input type="number" style=${inp} min="1" step="2" value=${cfg.kernel_size ?? 3} onChange=${e => set("kernel_size", +e.target.value)} /><//>
      <${Field} label="Iterations"><input type="number" style=${inp} min="1" value=${cfg.iterations ?? 1} onChange=${e => set("iterations", +e.target.value)} /><//>
      `; break;

    case "resize": fields = html`
      <${Field} label="Width (px)"><input type="number" style=${inp} value=${cfg.width ?? 640} onChange=${e => set("width", +e.target.value)} /><//>
      <${Field} label="Height (px)"><input type="number" style=${inp} value=${cfg.height ?? 480} onChange=${e => set("height", +e.target.value)} /><//>
      <${Field} label="Interpolation">
        <select style=${inp} value=${cfg.interpolation ?? "area"} onChange=${e => set("interpolation", e.target.value)}>
          <option value="nearest">Nearest (fastest)</option><option value="linear">Linear</option>
          <option value="cubic">Cubic</option><option value="area">Area (best for shrink)</option><option value="lanczos">Lanczos</option>
        </select>
      <//>
      `; break;

    case "affine_transform": fields = html`
      <${Field} label="Translate X (px)"><input type="number" style=${inp} value=${cfg.tx ?? 0} onChange=${e => set("tx", +e.target.value)} /><//>
      <${Field} label="Translate Y (px)"><input type="number" style=${inp} value=${cfg.ty ?? 0} onChange=${e => set("ty", +e.target.value)} /><//>
      <${Field} label=${"Rotation Angle: " + (cfg.angle ?? 0) + "°"}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="-180" max="180" step="1"
          value=${cfg.angle ?? 0} onChange=${e => set("angle", +e.target.value)} />
      <//>
      <${Field} label=${"Scale: " + (cfg.scale ?? 1.0)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0.1" max="3.0" step="0.05"
          value=${cfg.scale ?? 1.0} onChange=${e => set("scale", +e.target.value)} />
      <//>
      `; break;

    case "draw_roi": fields = html`
      <${Field} label="Zone ID"><input style=${inp} value=${cfg.zone_id ?? "zone_1"} onChange=${e => set("zone_id", e.target.value)} /><//>
      ${(running && sessionId)
        ? html`<div style=${{ fontSize: 10, color: "#58a6ff", marginBottom: 8 }}>
            ↓ Use the canvas below to drag / add / remove polygon points
          </div>`
        : html`<${Field} label="Polygon Points (JSON array — or run & use canvas)">
            <textarea style=${{ ...inp, height: 80, resize: "vertical", fontFamily: "monospace", fontSize: 10 }}
              value=${JSON.stringify(cfg.polygon ?? [[10,10],[90,10],[90,90],[10,90]])}
              onChange=${e => { try { set("polygon", JSON.parse(e.target.value)); } catch {} }} />
          <//>
        `}
      `; break;

    case "draw_line": fields = html`
      <${Field} label="Line ID"><input style=${inp} value=${cfg.line_id ?? "line_1"} onChange=${e => set("line_id", e.target.value)} /><//>
      <${Field} label="Direction">
        <select style=${inp} value=${cfg.direction ?? "both"} onChange=${e => set("direction", e.target.value)}>
          <option value="both">Both directions</option><option value="up">Up only</option><option value="down">Down only</option>
        </select>
      <//>
      ${(running && sessionId)
        ? html`<div style=${{ fontSize: 10, color: "#58a6ff", marginBottom: 8 }}>
            ↓ Drag the A / B endpoints on the canvas below to reposition the line
          </div>`
        : html`<${Field} label="Line Points [x1,y1],[x2,y2]">
            <textarea style=${{ ...inp, height: 48, resize: "none", fontFamily: "monospace", fontSize: 10 }}
              value=${JSON.stringify(cfg.line ?? [[10,50],[90,50]])}
              onChange=${e => { try { set("line", JSON.parse(e.target.value)); } catch {} }} />
          <//>
        `}
      `; break;

    case "object_tracker": fields = html`
      <${Field} label="Algorithm">
        <select style=${inp} value=${cfg.algorithm ?? "bytetrack"} onChange=${e => set("algorithm", e.target.value)}>
          <option value="bytetrack">ByteTrack (IoU, no ReID — pip install bytetracker)</option>
          <option value="deepsort">DeepSORT (IoU + appearance — pip install deep-sort-realtime)</option>
        </select>
      <//>
      <${Field} label="Max Age (frames lost before drop)"><input type="number" style=${inp} value=${cfg.max_age ?? 30} onChange=${e => set("max_age", +e.target.value)} /><//>
      <${Field} label="IoU Threshold"><input type="number" step="0.05" min="0" max="1" style=${inp} value=${cfg.iou_threshold ?? 0.3} onChange=${e => set("iou_threshold", +e.target.value)} /><//>
      <div style=${{ fontSize: 10, color: "#8b949e", padding: "2px 0 4px" }}>
        Track IDs are written to each detection. Draw BBox reads them when <b>Show Track ID</b> is enabled.
        Add a <b>Track DB</b> node after this to store position history and motion trails.
      </div>
      `; break;

    case "track_db": fields = html`
      <${Field} label="Max Tracks (RAM limit)"><input type="number" style=${inp} value=${cfg.max_tracks ?? 200} onChange=${e => set("max_tracks", +e.target.value)} /><//>
      <${Field} label="Position History (frames)"><input type="number" style=${inp} value=${cfg.history_frames ?? 30} onChange=${e => set("history_frames", +e.target.value)} /><//>
      <${Field} label="Draw Motion Trails on frame">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.draw_trails !== false}
            onChange=${e => set("draw_trails", e.target.checked)} />
          <span style=${{ fontSize: 11, color: "#8b949e" }}>Coloured trail per track ID</span>
        </label>
      <//>
      <div style=${{ fontSize: 10, color: "#8b949e", padding: "4px 0" }}>
        Stores: class, age, travel distance (px), last position, colour histogram per ID.
        Downstream nodes can read <code style=${{ color: "#79c0ff" }}>ctx.metadata["track_db"]</code>.
      </div>
      `; break;

    case "counter": fields = html`
      <${Field} label="Display Label"><input style=${inp} value=${cfg.label ?? "Count"} onChange=${e => set("label", e.target.value)} placeholder="Count" /><//>
      <${Field} label="Trigger Type">
        <select style=${inp} value=${cfg.trigger_type ?? "line_cross"} onChange=${e => set("trigger_type", e.target.value)}>
          <option value="line_cross">Line Cross (connect draw_line → line_ref)</option>
          <option value="zone_enter">Zone Enter (connect draw_roi → frame)</option>
          <option value="zone_exit">Zone Exit (connect draw_roi → frame)</option>
        </select>
      <//>
      <${Field} label="Trigger ID — matches line_id / zone_id"><input style=${inp} value=${cfg.trigger_id ?? "line_1"} onChange=${e => set("trigger_id", e.target.value)} placeholder="line_1" /><//>
      <${Field} label="Count Classes (comma-sep, empty = all)">
        <input style=${inp}
          value=${Array.isArray(cfg.count_classes) ? cfg.count_classes.join(", ") : ""}
          onChange=${e => set("count_classes", e.target.value.split(",").map(s => s.trim()).filter(Boolean))}
          placeholder="person, car  (empty = all)" />
      <//>
      <${Field} label="Show count overlay on frame">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.show_overlay !== false}
            onChange=${e => set("show_overlay", e.target.checked)} />
          <span style=${{ fontSize: 11, color: "#8b949e" }}>Draw count text on frame</span>
        </label>
      <//>
      <div style=${{ fontSize: 10, color: "#8b949e", padding: "4px 0 2px" }}>
        ⓘ Trigger condition: <b>center point</b> of bounding box crosses the line
        (sign of cross-product changes). Not boundary touch. Requires tracked detections
        (track_id ≥ 0) — connect an <b>object_tracker</b> node upstream.
        Connect <b>draw_line.line_ref → counter.line_ref</b> to auto-sync the trigger ID.
      </div>
      `; break;

    case "filter": fields = html`
      <${Field} label="Allowed Classes (comma-separated, empty = all)">
        <input style=${inp}
          value=${Array.isArray(cfg.allowed_classes) ? cfg.allowed_classes.join(", ") : ""}
          onChange=${e => set("allowed_classes", e.target.value.split(",").map(s => s.trim()).filter(Boolean))}
          placeholder="person, car, truck" />
      <//>
      <${Field} label=${"Min Confidence: " + (cfg.min_confidence ?? 0)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="1" step="0.05"
          value=${cfg.min_confidence ?? 0} onChange=${e => set("min_confidence", +e.target.value)} />
      <//>
      `; break;

    case "param": fields = html`
      <${Field} label="Params (JSON object)">
        <textarea style=${{ ...inp, height: 120, resize: "vertical", lineHeight: 1.5, fontFamily: "Consolas,'Courier New',monospace", fontSize: 11 }}
          value=${JSON.stringify(cfg.params ?? {}, null, 2)}
          onChange=${e => { try { set("params", JSON.parse(e.target.value)); } catch {} }} />
      <//>
      `; break;

    case "python_function":
      fields = html`<${PythonFunctionFields} node=${node} cfg=${cfg} onUpdate=${onUpdate} />`;
      break;

    case "cpp_function": fields = html`
      <${Field} label="Compile Status">
        <div style=${{ fontSize: 12, padding: "2px 0",
                        color: cfg.compile_status === "ok" ? "#3fb950" : cfg.compile_status === "error" ? "#f85149" : "#8b949e" }}>
          ${cfg.compile_status ?? "uncompiled"}
        </div>
      <//>
      <${Field} label="Compiled .so Hash">
        <input style=${{ ...inp, fontFamily: "monospace", fontSize: 10 }} readOnly value=${cfg.compiled_so_hash ?? "(compile first)"} />
      <//>
      `; break;

    case "stream_viewer": fields = html`
      <${Field} label=${"JPEG Quality: " + (cfg.jpeg_quality ?? 80)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="10" max="100" step="5"
          value=${cfg.jpeg_quality ?? 80} onChange=${e => set("jpeg_quality", +e.target.value)} />
      <//>
      <${Field} label="Max FPS"><input type="number" style=${inp} value=${cfg.max_fps ?? 30} onChange=${e => set("max_fps", +e.target.value)} /><//>
      <${Field} label="Draw Detections">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.draw_detections !== false}
            onChange=${e => set("draw_detections", e.target.checked)} />
          <span style=${{ color: "#c9d1d9", fontSize: 12 }}>Overlay bounding boxes from detections</span>
        </label>
      <//>
      `; break;

    case "video_writer": fields = html`
      <${Field} label="Output Path"><input style=${inp} value=${cfg.output_path ?? "./output.mp4"} onChange=${e => set("output_path", e.target.value)} /><//>
      <${Field} label="FPS"><input type="number" style=${inp} value=${cfg.fps ?? 30} onChange=${e => set("fps", +e.target.value)} /><//>
      `; break;

    case "trigger_webhook": fields = html`
      <${Field} label="Protocol">
        <select style=${inp} value=${cfg.protocol ?? "http"} onChange=${e => set("protocol", e.target.value)}>
          <option value="http">HTTP POST</option><option value="mqtt">MQTT</option>
        </select>
      <//>
      <${Field} label="URL / Broker">
        <input style=${inp}
          value={cfg.url ?? ""}
          onChange=${e => set("url", e.target.value)}
          placeholder="https://..." />
      <//>
      <${Field} label="Trigger On">
        <select style=${inp} value=${cfg.trigger_on ?? "count_change"} onChange=${e => set("trigger_on", e.target.value)}>
          <option value="every_frame">Every Frame</option><option value="detection">On Detection</option><option value="count_change">On Count Change</option>
        </select>
      <//>
      <${Field} label=${"Rate Limit: " + (cfg.rate_limit_s ?? 2.0) + "s"}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="10" step="0.5"
          value=${cfg.rate_limit_s ?? 2.0} onChange=${e => set("rate_limit_s", +e.target.value)} />
      <//>
      `; break;

    case "mqtt_publish": fields = html`
      <${Field} label="Broker"><input style=${inp} value=${cfg.broker ?? "localhost"} onChange=${e => set("broker", e.target.value)} /><//>
      <${Field} label="Port"><input type="number" style=${inp} value=${cfg.port ?? 1883} onChange=${e => set("port", +e.target.value)} /><//>
      <${Field} label="Topic"><input style=${inp} value=${cfg.topic ?? "cv_flow/events"} onChange=${e => set("topic", e.target.value)} /><//>
      <${Field} label="QoS">
        <select style=${inp} value=${cfg.qos ?? 0} onChange=${e => set("qos", +e.target.value)}>
          <option value="0">0 — At most once</option><option value="1">1 — At least once</option><option value="2">2 — Exactly once</option>
        </select>
      <//>
      <${Field} label="Trigger On">
        <select style=${inp} value=${cfg.trigger_on ?? "detection"} onChange=${e => set("trigger_on", e.target.value)}>
          <option value="every_frame">Every Frame</option><option value="detection">On Detection</option><option value="count_change">On Count Change</option>
        </select>
      <//>
      <${Field} label=${"Rate Limit: " + (cfg.rate_limit_s ?? 0.5) + "s"}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0" max="5" step="0.1"
          value=${cfg.rate_limit_s ?? 0.5} onChange=${e => set("rate_limit_s", +e.target.value)} />
      <//>
      `; break;

    case "kafka_produce": fields = html`
      <${Field} label="Bootstrap Servers">
        <input style=${inp} value=${cfg.bootstrap_servers ?? "localhost:9092"} onChange=${e => set("bootstrap_servers", e.target.value)} placeholder="host1:9092,host2:9092" />
      <//>
      <${Field} label="Topic"><input style=${inp} value=${cfg.topic ?? "cv_flow_events"} onChange=${e => set("topic", e.target.value)} /><//>
      <${Field} label="Trigger On">
        <select style=${inp} value=${cfg.trigger_on ?? "detection"} onChange=${e => set("trigger_on", e.target.value)}>
          <option value="every_frame">Every Frame</option><option value="detection">On Detection</option><option value="count_change">On Count Change</option>
        </select>
      <//>
      <${Field} label="Rate Limit (s, 0 = unlimited)">
        <input type="number" style=${inp} min="0" step="0.1" value=${cfg.rate_limit_s ?? 0} onChange=${e => set("rate_limit_s", +e.target.value)} />
      <//>
      `; break;

    case "face_detect": fields = html`
      <${Field} label="Model Key">
        <select style=${inp} value=${cfg.model_key ?? "scrfd_10g"} onChange=${e => set("model_key", e.target.value)}>
          <option value="scrfd_10g">SCRFD-10G (High accuracy, ~16MB) — Recommended</option>
          <option value="scrfd_500m">SCRFD-500M (Lightweight, ~2MB) — CPU Real-time</option>
        </select>
      <//>
      <${Field} label=${"Confidence: " + (cfg.conf_threshold ?? 0.5)}>
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0.1" max="0.95" step="0.05"
          value=${cfg.conf_threshold ?? 0.5} onChange=${e => set("conf_threshold", +e.target.value)} />
      <//>
      <${Field} label="Device">
        <select style=${inp} value=${cfg.device ?? "cpu"} onChange=${e => set("device", e.target.value)}>
          <option value="cpu">CPU</option><option value="cuda">CUDA (GPU)</option>
        </select>
      <//>
      <${Field} label="Min Face Size (px)">
        <input type="number" style=${inp} min="10" value=${cfg.min_face_size_px ?? 20} onChange=${e => set("min_face_size_px", +e.target.value)} />
      <//>
      <${Field} label="Return Largest Only">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.return_largest === true}
            onChange=${e => set("return_largest", e.target.checked)} />
          <span style=${{ fontSize: 12, color: "#c9d1d9" }}>Only output the largest detected face</span>
        </label>
      <//>
      <div style=${{ fontSize: 11, color: "#8b949e", lineHeight: 1.5, marginTop: 6, padding: 8, background: "#21262d", borderRadius: 5, border: "1px solid #30363d" }}>
        <b style=${{ color: "#d29922" }}>Auto-download</b> — If the selected model is not present, it downloads automatically on first run via InsightFace. Requires: <code>pip install insightface onnxruntime</code>
      </div>
      `; break;

    case "embedding": fields = html`
      <${Field} label="Model">
        <select style=${inp} value=${cfg.model_key ?? "mobilefacenet"} onChange=${e => set("model_key", e.target.value)}>
          <option value="mobilefacenet">MobileFaceNet (~4 MB, CPU real-time)</option>
          <option value="arcface_r50">ArcFace R50 (~166 MB, higher accuracy)</option>
        </select>
      <//>
      <${Field} label="L2 Normalize Output">
        <label style=${{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked=${cfg.normalize !== false}
            onChange=${e => set("normalize", e.target.checked)} />
          <span style=${{ fontSize: 12, color: "#c9d1d9" }}>Normalize embedding vectors (recommended)</span>
        </label>
      <//>
      `; break;

    case "face_db": fields = html`
      <${Field} label="Match Threshold">
        <input type="range" style=${{ width: "100%", accentColor: "#58a6ff" }} min="0.1" max="0.99" step="0.01"
          value=${cfg.threshold ?? 0.5} onChange=${e => set("threshold", +e.target.value)} />
        <div style=${{ fontSize: 10, color: "#8b949e", marginTop: 2 }}>Current: ${cfg.threshold ?? 0.5} — cosine similarity required to count as a match</div>
      <//>
      <${Field} label="Identity Name (Enroll Label)">
        <input type="text" style=${inp} value=${cfg.name ?? "Person"}
          onChange=${e => set("name", e.target.value)} placeholder="Person" />
        <div style=${{ fontSize: 10, color: "#8b949e", marginTop: 2 }}>Name stored when the Enroll dot receives True</div>
      <//>
      <${Field} label="Max Embeddings to Save per Identity">
        <input type="number" style=${inp} min="1" max="200" step="1"
          value=${cfg.max_save ?? 10} onChange=${e => set("max_save", +e.target.value)} />
        <div style=${{ fontSize: 10, color: "#8b949e", marginTop: 2 }}>Stop saving after this many embeddings for the same name</div>
      <//>
      <${Field} label="DB Path">
        <input type="text" style=${inp} value=${cfg.db_path ?? "storage/facedb"}
          onChange=${e => set("db_path", e.target.value)} />
      <//>
      `; break;

    case "pipeline_output": fields = html`
      <${Field} label="Output Label">
        <input style=${inp} value=${cfg.label ?? "Output"}
          onChange=${e => set("label", e.target.value)} placeholder="Output" />
      <//>
      <${Field} label="Description">
        <input style=${inp} value=${cfg.description ?? ""}
          onChange=${e => set("description", e.target.value)} placeholder="What this output carries" />
      <//>
      <${PipelineOutputPortEditor} node=${node} cfg=${cfg} onUpdate=${onUpdate} />
      <div style=${{ fontSize: 11, color: "#8b949e", lineHeight: 1.5, marginTop: 6, padding: "8px", background: "#21262d", borderRadius: 5, border: "1px solid #30363d" }}>
        <b style=${{ color: "#d29922" }}>Pipeline Output</b> marks where data exits this pipeline when saved as a reusable template. Add as many input ports as you need.
      </div>
      `; break;

    default: {
      const isTemplate = node.type.startsWith("tmpl_");
      const pj = cfg.pipeline_json;
      const customNodes = loadCustomNodes().find(c => c.type === node.type);
      const cfgKeys = Object.keys(cfg).filter(k => k !== "pipeline_json");

      fields = html`
        <!-- Node label (visual header) -->
        <${Field} label="Node Label">
          <input style=${inp} value=${node.data.label ?? node.type}
            onChange=${e => onUpdate(node.id, cfg, e.target.value)} />
        <//>

        ${isTemplate && pj && html`
          <!-- Template pipeline summary -->
          <div style=${{ background: "#0d1117", border: "1px solid #21262d", borderRadius: 6,
                          padding: "8px 12px", marginBottom: 8, fontSize: 11 }}>
            <div style=${{ fontWeight: 600, color: "#d29922", marginBottom: 5 }}>⬡ Embedded Pipeline</div>
            <div style=${{ color: "#8b949e" }}>${(pj.nodes ?? []).length} nodes · ${(pj.edges ?? []).length} edges</div>
            <div style=${{ color: "#555d68", fontSize: 10, marginTop: 4 }}>
              ${(pj.nodes ?? []).map(n => n.type).join(" → ")}
            </div>
          </div>`}

        <!-- Ports summary (read-only) -->
        ${(() => {
          const ports = node.data.ports ?? NODE_PORTS[node.type];
          if (!ports) return null;
          return html`
            <div style=${{ marginBottom: 8 }}>
              <div style=${{ fontSize: 10, color: "#555d68", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>Ports</div>
              <div style=${{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                ${ports.inputs.map(p => html`
                  <span key=${p.id} style=${{ fontSize: 10, background: "#1a2d40", color: "#58a6ff",
                                               border: "1px solid #1e4060", borderRadius: 4, padding: "1px 7px" }}>
                    ● ${p.label}
                  </span>`)}
                ${ports.outputs.map(p => html`
                  <span key=${p.id} style=${{ fontSize: 10, background: "#1a3020", color: "#3fb950",
                                               border: "1px solid #1e5030", borderRadius: 4, padding: "1px 7px" }}>
                    ${p.label} ●
                  </span>`)}
              </div>
            </div>`;
        })()}

        <!-- Editable config (non-pipeline_json keys) -->
        ${cfgKeys.length > 0 && html`
          <${Field} label="Config">
            <textarea
              style=${{ ...inp, height: 100, resize: "vertical", fontFamily: "Consolas,'Courier New',monospace", fontSize: 10 }}
              value=${JSON.stringify(Object.fromEntries(cfgKeys.map(k => [k, cfg[k]])), null, 2)}
              onChange=${e => {
                try { const parsed = JSON.parse(e.target.value);
                  onUpdate(node.id, { ...parsed, ...(pj ? { pipeline_json: pj } : {}) }); } catch {}
              }} />
          <//>
        `}`;
    }; break;
    }  // end switch (legacy)
  }  // end if(false)

  const meta = NODE_META[node.type] ?? { icon: "◻", label: node.type, group: "utility" };
  return html`
    <div style=${{ padding: "12px 14px", overflowY: "auto", flex: 1 }}>
      <div style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14, paddingBottom: 10, borderBottom: "1px solid #30363d" }}>
        <div style=${{ width: 28, height: 28, borderRadius: 6, background: GROUP_COLOR[meta.group] ?? "#21262d",
                        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15 }}>
          ${meta.icon}
        </div>
        <div style=${{ flex: 1 }}>
          <div style=${{ fontWeight: 700, fontSize: 13, color: "#e2e8f0" }}>${meta.label}</div>
          <div style=${{ fontSize: 9, color: "#555d68", fontFamily: "monospace" }}>${node.id}</div>
        </div>
        <button onClick=${() => onDuplicate(node.id)} title="Duplicate (Ctrl+D)"
          style=${{ background: "none", border: "1px solid #30363d", color: "#8b949e", cursor: "pointer",
                     borderRadius: 5, padding: "3px 8px", fontSize: 11 }}>⎘</button>
      </div>

      <${Field} label="Label">
        <input style=${inp} value=${node.data.label ?? ""}
          onChange=${e => onUpdate(node.id, cfg, e.target.value)} />
      <//>

      ${fields}

      ${(() => {
          const snap = nodeDataMap?.[node.id];
          return snap ? html`<${NodeDataPanel} nodeType=${snap.node_type} data=${snap.data} />` : null;
        })()}

      <${PortsInfo} node=${node} />

      ${node.type === "python_node" && html`
        <button onClick=${onSaveNode}
          style=${{ width: "100%", marginTop: 10, padding: "6px 0", borderRadius: 6,
                     background: "transparent", border: "1px solid #d29922",
                     color: "#d29922", cursor: "pointer", fontSize: 12, fontFamily: "inherit" }}>
          ⬡ Save to Library
        </button>`}

      <button onClick=${() => { if (confirm("Delete this node?")) onUpdate(node.id, null); }}
        style=${{ width: "100%", marginTop: 6, padding: "6px 0", borderRadius: 6,
                   background: "transparent", border: "1px solid #f85149",
                   color: "#f85149", cursor: "pointer", fontSize: 12 }}>
        Delete Node
      </button>
    </div>`;
}

// ── MODEL HUB MODAL ───────────────────────────────────────────────────────────
const CATALOG_CATEGORIES = ["All", "Object Detection", "Segmentation", "Pose Estimation", "Classification"];
const TASK_ICON = { detection: "🎯", segmentation: "🗺️", pose: "🦴", classification: "🏷️" };
const CAT_COLOR = {
  "Object Detection": { bg: "#1a2d40", border: "#1a4a7e", accent: "#58a6ff" },
  "Segmentation":     { bg: "#1a2d22", border: "#1a4a30", accent: "#3fb950" },
  "Pose Estimation":  { bg: "#2d2218", border: "#4a3618", accent: "#e3b341" },
  "Classification":   { bg: "#2a1a2d", border: "#4a1a4e", accent: "#bc8cff" },
};

function ModelHubModal({ onClose }) {
  const [models, setModels]           = useState([]);
  const [loading, setLoading]         = useState(true);
  const [uploading, setUploading]     = useState(false);
  const [uploadMsg, setUploadMsg]     = useState(null);
  const [catalog, setCatalog]         = useState([]);
  const [faceCatalog, setFaceCatalog] = useState([]);
  const [downloading, setDownloading] = useState(null);
  const [dlMsg, setDlMsg]             = useState(null);
  const [tab, setTab]                 = useState("library");   // "library" | "catalog" | "face" | "upload" | "packages"
  const [catFilter, setCatFilter]     = useState("All");
  const [pkgInput, setPkgInput]       = useState("");
  const [pkgLog, setPkgLog]           = useState([]);   // array of {text, type} — type: "cmd"|"out"|"ok"|"err"
  const [pkgRunning, setPkgRunning]   = useState(false);
  const pkgLogRef = useRef(null);
  const onnxRef = useRef(null);
  const cfgRef  = useRef(null);

  const reload = useCallback(() => {
    apiFetch("GET", "/models").then(setModels).catch(() => {}).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    reload();
    apiFetch("GET", "/models/defaults/list").then(setCatalog).catch(() => {});
    apiFetch("GET", "/models/face/list").then(setFaceCatalog).catch(() => {});
  }, []);

  const upload = async () => {
    const onnx = onnxRef.current?.files[0];
    const cfg  = cfgRef.current?.files[0];
    if (!onnx || !cfg) { setUploadMsg("Select both model.onnx and config.json"); return; }
    setUploading(true); setUploadMsg(null);
    try {
      const result = await apiUploadModel(onnx, cfg);
      setModels(ms => [result, ...ms]);
      setUploadMsg("Uploaded: " + result.name);
      onnxRef.current.value = ""; cfgRef.current.value = "";
      setTab("library");
    } catch (e) { setUploadMsg("Error: " + e.message); }
    finally { setUploading(false); }
  };

  const downloadDefault = async (key, name) => {
    setDownloading(key); setDlMsg(null);
    try {
      const result = await apiFetch("POST", "/models/defaults/download/" + key);
      setModels(ms => [result, ...ms]);
      setDlMsg({ ok: true, msg: name + " downloaded — ID copied to clipboard!" });
      await navigator.clipboard.writeText(result.id).catch(() => {});
    } catch (e) { setDlMsg({ ok: false, msg: "Error: " + e.message }); }
    finally { setDownloading(null); }
  };

  const downloadFaceModel = async (key, name) => {
    setDownloading(key); setDlMsg(null);
    try {
      const result = await apiFetch("POST", "/models/face/download/" + key);
      setModels(ms => ms.find(m => m.id === result.id) ? ms : [result, ...ms]);
      setDlMsg({ ok: true, msg: name + " downloaded." });
    } catch (e) { setDlMsg({ ok: false, msg: "Error: " + e.message }); }
    finally { setDownloading(null); }
  };

  const isFaceDownloaded = (key, name) => models.some(m => m.name === name);

  const isDownloaded = name => models.some(m => m.name === name);

  const runPip = async () => {
    const cmd = pkgInput.trim();
    if (!cmd || pkgRunning) return;
    setPkgRunning(true);
    setPkgLog([{ text: "$ " + cmd, type: "cmd" }]);

    try {
      const res = await fetch("/api/system/pip-install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd }),
      });
      if (!res.ok || !res.body) {
        setPkgLog(l => [...l, { text: "Error: " + res.statusText, type: "err" }]);
        return;
      }

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // Parse SSE: split on double newline
        const events = buf.split("\n\n");
        buf = events.pop(); // keep incomplete chunk
        for (const ev of events) {
          if (!ev.trim()) continue;
          // Check for "event: done"
          const isDone = ev.includes("event: done");
          const isOk   = isDone && ev.includes("data: ok");
          // Extract data lines
          const lines = ev.split("\n")
            .filter(l => l.startsWith("data: "))
            .map(l => l.slice(6));
          for (const line of lines) {
            if (line === "ok" || line === "error") continue; // sentinel
            const type = isDone ? (isOk ? "ok" : "err") : "out";
            setPkgLog(l => {
              const updated = [...l, { text: line || " ", type }];
              // Auto-scroll after state update
              setTimeout(() => {
                if (pkgLogRef.current) pkgLogRef.current.scrollTop = pkgLogRef.current.scrollHeight;
              }, 0);
              return updated;
            });
          }
          if (isDone) { setPkgRunning(false); return; }
        }
      }
    } catch (e) {
      setPkgLog(l => [...l, { text: "Error: " + e.message, type: "err" }]);
    } finally {
      setPkgRunning(false);
    }
  };

  const del = async (id) => {
    if (!confirm("Delete this model?")) return;
    try { await apiFetch("DELETE", "/models/" + id); setModels(ms => ms.filter(m => m.id !== id)); }
    catch (e) { alert("Delete failed: " + e.message); }
  };

  const hotReload = async (id) => {
    try { await apiFetch("POST", "/models/" + id + "/reload"); }
    catch (e) { alert("Reload failed: " + e.message); }
  };

  const tabBtn = (key, label) => html`
    <button onClick=${() => setTab(key)}
      style=${{ padding: "5px 12px", border: "none", cursor: "pointer", fontSize: 12,
                 background: tab === key ? "#1f3a5e" : "transparent",
                 color: tab === key ? "#58a6ff" : "#8b949e",
                 borderBottom: tab === key ? "2px solid #58a6ff" : "2px solid transparent",
                 fontFamily: "inherit" }}>
      ${label}
    </button>`;

  // Group catalog items by category for section headers
  const filteredCatalog = catFilter === "All"
    ? catalog
    : catalog.filter(d => d.category === catFilter);

  // Build category sections preserving catalog order
  const catalogSections = [];
  let lastCat = null;
  filteredCatalog.forEach(d => {
    if (d.category !== lastCat) { catalogSections.push({ cat: d.category, items: [] }); lastCat = d.category; }
    catalogSections[catalogSections.length - 1].items.push(d);
  });

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(0,0,0,.72)",
                    display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
                      width: 680, maxHeight: "88vh", display: "flex", flexDirection: "column",
                      boxShadow: "0 16px 48px rgba(0,0,0,.8)" }}>

        <!-- Header -->
        <div style=${{ display: "flex", alignItems: "center", padding: "14px 18px", borderBottom: "1px solid #30363d" }}>
          <span style=${{ fontWeight: 700, fontSize: 15, color: "#e2e8f0", flex: 1 }}>🧠 Model Hub</span>
          <button onClick=${onClose} style=${{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18, padding: "0 4px" }}>✕</button>
        </div>

        <!-- Tabs -->
        <div style=${{ display: "flex", borderBottom: "1px solid #30363d", paddingLeft: 10 }}>
          ${tabBtn("library", "Library (" + models.length + ")")}
          ${tabBtn("catalog", "YOLO Models")}
          ${tabBtn("face", "Face Models")}
          ${tabBtn("packages", "Packages")}
          ${tabBtn("upload", "Upload Custom")}
        </div>

        <!-- Tab: Library -->
        ${tab === "library" && html`
          <div style=${{ overflowY: "auto", flex: 1, padding: "10px 18px" }}>
            ${loading && html`<div style=${{ color: "#8b949e", fontSize: 12, textAlign: "center", padding: 24 }}>Loading…</div>`}
            ${!loading && models.length === 0 && html`
              <div style=${{ color: "#8b949e", fontSize: 12, textAlign: "center", padding: 24 }}>
                No models yet — download from catalog or upload a custom ONNX model.
              </div>`}
            ${models.map(m => html`
              <div key=${m.id} style=${{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8, padding: "10px 14px", marginBottom: 8 }}>
                <div style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span style=${{ fontSize: 18 }}>${TASK_ICON[m.task] ?? "🧠"}</span>
                  <span style=${{ fontWeight: 600, fontSize: 13, color: "#e2e8f0", flex: 1 }}>${m.name}</span>
                  <span style=${{ fontSize: 10, background: "#1a2a40", color: "#58a6ff", borderRadius: 4, padding: "1px 6px" }}>v${m.version}</span>
                  <span style=${{ fontSize: 10, background: "#1a3a20", color: "#3fb950", borderRadius: 4, padding: "1px 6px" }}>${m.task}</span>
                </div>
                <div style=${{ fontSize: 10, color: "#8b949e", fontFamily: "monospace", marginBottom: 8, wordBreak: "break-all" }}>ID: ${m.id}</div>
                <div style=${{ display: "flex", gap: 6 }}>
                  <button onClick=${() => navigator.clipboard.writeText(m.id)}
                    style=${{ padding: "3px 10px", background: "#21262d", border: "1px solid #30363d", color: "#c9d1d9", borderRadius: 5, cursor: "pointer", fontSize: 11 }}>
                    Copy ID
                  </button>
                  <button onClick=${() => hotReload(m.id)}
                    style=${{ padding: "3px 10px", background: "#1f3a5e", border: "1px solid #58a6ff", color: "#58a6ff", borderRadius: 5, cursor: "pointer", fontSize: 11 }}>
                    Hot Reload
                  </button>
                  <button onClick=${() => del(m.id)}
                    style=${{ padding: "3px 10px", background: "#3d1a1a", border: "1px solid #f85149", color: "#f85149", borderRadius: 5, cursor: "pointer", fontSize: 11, marginLeft: "auto" }}>
                    Delete
                  </button>
                </div>
              </div>`)}
          </div>`}

        <!-- Tab: Download Models Catalog -->
        ${tab === "catalog" && html`
          <div style=${{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>

            <!-- Category filter bar -->
            <div style=${{ display: "flex", gap: 6, padding: "10px 16px", borderBottom: "1px solid #21262d", flexWrap: "wrap" }}>
              ${CATALOG_CATEGORIES.map(cat => {
                const active = catFilter === cat;
                const clr = cat === "All" ? { bg: "#1f3a5e", border: "#58a6ff", accent: "#58a6ff" } : (CAT_COLOR[cat] || {});
                return html`
                  <button key=${cat} onClick=${() => setCatFilter(cat)}
                    style=${{ padding: "4px 12px", borderRadius: 20, border: "1px solid",
                               cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                               background: active ? (clr.bg || "#1f3a5e") : "#0d1117",
                               borderColor: active ? (clr.accent || "#58a6ff") : "#30363d",
                               color: active ? (clr.accent || "#58a6ff") : "#8b949e" }}>
                    ${cat}
                  </button>`;
              })}
            </div>

            <!-- Info bar -->
            <div style=${{ padding: "6px 16px", fontSize: 11, color: "#555d68", borderBottom: "1px solid #21262d" }}>
              Requires <code style=${{ color: "#58a6ff" }}>pip install ultralytics</code> · Downloads .pt weights then exports to ONNX automatically
            </div>

            <!-- Status message -->
            ${dlMsg && html`
              <div style=${{ margin: "8px 16px 0", padding: "6px 10px", borderRadius: 6, fontSize: 11,
                              background: dlMsg.ok ? "#1a3d2e" : "#3d1a1a",
                              border: "1px solid " + (dlMsg.ok ? "#3fb950" : "#f85149"),
                              color: dlMsg.ok ? "#3fb950" : "#f85149" }}>
                ${dlMsg.msg}
              </div>`}

            <!-- Catalog sections -->
            <div style=${{ overflowY: "auto", flex: 1, padding: "10px 16px" }}>
              ${catalog.length === 0 && html`<div style=${{ color: "#8b949e", fontSize: 12, textAlign: "center", padding: 24 }}>Loading catalog…</div>`}

              ${catalogSections.map(({ cat, items }) => {
                const clr = CAT_COLOR[cat] || { bg: "#1a2a40", border: "#30363d", accent: "#58a6ff" };
                return html`
                  <div key=${cat} style=${{ marginBottom: 18 }}>
                    <!-- Section header -->
                    <div style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <span style=${{ fontSize: 13, color: clr.accent, fontWeight: 600 }}>${cat}</span>
                      <div style=${{ flex: 1, height: 1, background: clr.border }}></div>
                    </div>

                    <!-- Grid of cards (2 per row) -->
                    <div style=${{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      ${items.map(d => {
                        const done = isDownloaded(d.name);
                        const busy = downloading === d.key;
                        return html`
                          <div key=${d.key}
                            style=${{ background: clr.bg, border: "1px solid " + clr.border,
                                       borderRadius: 8, padding: "10px 12px",
                                       display: "flex", flexDirection: "column", gap: 5 }}>

                            <!-- Top row: icon + name + badge -->
                            <div style=${{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                              <span style=${{ fontSize: 20, lineHeight: 1 }}>${TASK_ICON[d.task] ?? "🧠"}</span>
                              <div style=${{ flex: 1, minWidth: 0 }}>
                                <div style=${{ fontWeight: 600, fontSize: 12, color: "#e2e8f0",
                                               whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                  ${d.name}
                                </div>
                                <div style=${{ display: "flex", gap: 4, marginTop: 2, flexWrap: "wrap" }}>
                                  ${d.badge && html`
                                    <span style=${{ fontSize: 9, background: "#3d2700", color: "#e3b341",
                                                     border: "1px solid #7d5800", borderRadius: 3, padding: "0 4px" }}>
                                      ${d.badge}
                                    </span>`}
                                  <span style=${{ fontSize: 9, color: "#555d68" }}>~${d.size_mb} MB</span>
                                  ${done && html`
                                    <span style=${{ fontSize: 9, background: "#1a3d2e", color: "#3fb950",
                                                     border: "1px solid #1a5a3a", borderRadius: 3, padding: "0 4px" }}>
                                      ✓ Downloaded
                                    </span>`}
                                </div>
                              </div>
                            </div>

                            <!-- Description -->
                            <div style=${{ fontSize: 10, color: "#8b949e", lineHeight: 1.4 }}>${d.desc}</div>

                            <!-- Download button -->
                            <button
                              onClick=${() => !done && downloadDefault(d.key, d.name)}
                              disabled=${!!downloading || done}
                              style=${{
                                marginTop: 2, padding: "5px 0", borderRadius: 5, fontSize: 11,
                                cursor: done ? "default" : (downloading ? "wait" : "pointer"),
                                border: "1px solid", fontFamily: "inherit", textAlign: "center",
                                background: done ? "#0d1117" : (busy ? "#1a2a40" : clr.bg),
                                borderColor: done ? "#30363d" : (busy ? "#58a6ff" : clr.accent),
                                color: done ? "#555d68" : (busy ? "#58a6ff" : clr.accent),
                                opacity: (downloading && !busy && !done) ? 0.45 : 1,
                              }}>
                              ${done ? "Already in Library" : (busy ? "Downloading…" : "Download")}
                            </button>
                          </div>`;
                      })}
                    </div>
                  </div>`;
              })}
            </div>
          </div>`}

        <!-- Tab: Face Models -->
        ${tab === "face" && html`
          <div style=${{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>

            <!-- Info bar -->
            <div style=${{ padding: "8px 16px", fontSize: 11, color: "#555d68", borderBottom: "1px solid #21262d", lineHeight: 1.6 }}>
              Requires <code style=${{ color: "#db61a2" }}>pip install insightface onnxruntime</code>
              · Downloads from InsightFace buffalo packs, registers into Library automatically
            </div>

            <!-- Status message -->
            ${dlMsg && html`
              <div style=${{ margin: "8px 16px 0", padding: "6px 10px", borderRadius: 6, fontSize: 11,
                              background: dlMsg.ok ? "#1a3d2e" : "#3d1a1a",
                              border: "1px solid " + (dlMsg.ok ? "#3fb950" : "#f85149"),
                              color: dlMsg.ok ? "#3fb950" : "#f85149" }}>
                ${dlMsg.msg}
              </div>`}

            <div style=${{ overflowY: "auto", flex: 1, padding: "12px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
              ${faceCatalog.length === 0
                ? html`<div style=${{ color: "#8b949e", fontSize: 12, textAlign: "center", padding: 24 }}>Loading…</div>`
                : faceCatalog.map(d => {
                    const done = isFaceDownloaded(d.key, d.name);
                    const busy = downloading === d.key;
                    const isDetect = d.category === "Face Detection";
                    const clr = isDetect
                      ? { bg: "#2d1a2d", border: "#5c2d5c", accent: "#db61a2" }
                      : { bg: "#1a2d3d", border: "#1a4a6e", accent: "#58a6ff" };
                    return html`
                      <div key=${d.key}
                        style=${{ background: clr.bg, border: "1px solid " + clr.border,
                                   borderRadius: 8, padding: "12px 14px",
                                   display: "flex", alignItems: "center", gap: 12 }}>

                        <div style=${{ fontSize: 22, lineHeight: 1, flexShrink: 0 }}>
                          ${isDetect ? "👤" : "🧬"}
                        </div>

                        <div style=${{ flex: 1, minWidth: 0 }}>
                          <div style=${{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                            <span style=${{ fontWeight: 600, fontSize: 13, color: "#e2e8f0" }}>${d.name}</span>
                            ${d.badge && html`
                              <span style=${{ fontSize: 9, background: "#3d2700", color: "#e3b341",
                                               border: "1px solid #7d5800", borderRadius: 3, padding: "0 5px" }}>
                                ${d.badge}
                              </span>`}
                            <span style=${{ fontSize: 9, color: "#555d68", marginLeft: 2 }}>~${d.size_mb} MB</span>
                            ${done && html`
                              <span style=${{ fontSize: 9, background: "#1a3d2e", color: "#3fb950",
                                               border: "1px solid #1a5a3a", borderRadius: 3, padding: "0 5px" }}>
                                ✓ In Library
                              </span>`}
                          </div>
                          <div style=${{ fontSize: 11, color: "#8b949e", lineHeight: 1.4 }}>${d.desc}</div>
                        </div>

                        <button
                          onClick=${() => !done && downloadFaceModel(d.key, d.name)}
                          disabled=${!!downloading || done}
                          style=${{
                            flexShrink: 0, padding: "6px 16px", borderRadius: 6, fontSize: 11,
                            cursor: done ? "default" : (downloading ? "wait" : "pointer"),
                            border: "1px solid", fontFamily: "inherit", whiteSpace: "nowrap",
                            background: done ? "#0d1117" : (busy ? "#1a2a40" : clr.bg),
                            borderColor: done ? "#30363d" : (busy ? "#58a6ff" : clr.accent),
                            color: done ? "#555d68" : (busy ? "#58a6ff" : clr.accent),
                            opacity: (downloading && !busy && !done) ? 0.45 : 1,
                          }}>
                          ${done ? "Downloaded" : (busy ? "Downloading…" : "Download")}
                        </button>
                      </div>`;
                  })}
            </div>
          </div>`}

        <!-- Tab: Packages -->
        ${tab === "packages" && html`
          <div style=${{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, padding: "12px 16px", gap: 10 }}>

            <!-- Input row -->
            <div style=${{ display: "flex", gap: 8, alignItems: "center" }}>
              <div style=${{ flex: 1, display: "flex", alignItems: "center", gap: 0,
                              background: "#0d1117", border: "1px solid #30363d", borderRadius: 7, overflow: "hidden" }}>
                <span style=${{ padding: "0 10px", color: "#555d68", fontSize: 12, fontFamily: "monospace", userSelect: "none" }}>$</span>
                <input
                  value=${pkgInput}
                  onInput=${e => setPkgInput(e.target.value)}
                  onKeyDown=${e => {
                    if (e.key === "Enter" && !pkgRunning && pkgInput.trim()) runPip();
                  }}
                  placeholder=${"pip install ultralytics   or   torch torchvision torchaudio"}
                  disabled=${pkgRunning}
                  style=${{
                    flex: 1, background: "none", border: "none", outline: "none", padding: "8px 0",
                    color: "#e2e8f0", fontSize: 12, fontFamily: "monospace",
                  }} />
              </div>
              <button
                onClick=${runPip}
                disabled=${pkgRunning || !pkgInput.trim()}
                style=${{
                  padding: "7px 18px", borderRadius: 7, border: "1px solid",
                  cursor: (pkgRunning || !pkgInput.trim()) ? "default" : "pointer",
                  fontSize: 12, fontFamily: "inherit",
                  background: pkgRunning ? "#1a2a40" : "#1a3d2e",
                  borderColor: pkgRunning ? "#58a6ff" : "#3fb950",
                  color: pkgRunning ? "#58a6ff" : "#3fb950",
                  minWidth: 80, textAlign: "center",
                }}>
                ${pkgRunning ? "Installing…" : "Install"}
              </button>
            </div>

            <!-- Quick pills -->
            <div style=${{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              ${["ultralytics", "torch torchvision torchaudio", "onnxruntime-gpu", "opencv-python", "paho-mqtt", "kafka-python"].map(pkg => html`
                <button key=${pkg}
                  onClick=${() => setPkgInput("pip install " + pkg)}
                  style=${{ padding: "3px 10px", borderRadius: 20, fontSize: 10, cursor: "pointer",
                              background: "#161b22", border: "1px solid #30363d",
                              color: "#8b949e", fontFamily: "monospace" }}>
                  ${pkg}
                </button>`)}
            </div>

            <!-- Terminal output -->
            <div ref=${pkgLogRef}
              style=${{
                flex: 1, overflowY: "auto", background: "#0d1117",
                border: "1px solid #21262d", borderRadius: 7,
                padding: "10px 12px", fontFamily: "monospace", fontSize: 11,
                lineHeight: 1.6, minHeight: 120,
              }}>
              ${pkgLog.length === 0 && html`
                <span style=${{ color: "#555d68" }}>Output will appear here…</span>`}
              ${pkgLog.map((line, i) => html`
                <div key=${i} style=${{ color: line.type === "cmd" ? "#58a6ff"
                                              : line.type === "ok"  ? "#3fb950"
                                              : line.type === "err" ? "#f85149"
                                              : "#c9d1d9",
                                         whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                  ${line.text}
                </div>`)}
            </div>

          </div>`}

        <!-- Tab: Upload Custom -->
        ${tab === "upload" && html`
          <div style=${{ padding: "14px 18px", overflowY: "auto", flex: 1 }}>
            <div style=${{ display: "flex", gap: 8, alignItems: "flex-end", flexWrap: "wrap" }}>
              <div>
                <div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 3 }}>model.onnx *</div>
                <input type="file" accept=".onnx" ref=${onnxRef}
                  style=${{ fontSize: 11, color: "#c9d1d9", background: "#21262d", border: "1px solid #30363d", borderRadius: 5, padding: "4px 8px" }} />
              </div>
              <div>
                <div style=${{ fontSize: 10, color: "#8b949e", marginBottom: 3 }}>config.json *</div>
                <input type="file" accept=".json" ref=${cfgRef}
                  style=${{ fontSize: 11, color: "#c9d1d9", background: "#21262d", border: "1px solid #30363d", borderRadius: 5, padding: "4px 8px" }} />
              </div>
              <button onClick=${upload} disabled=${uploading}
                style=${{ padding: "6px 14px", background: uploading ? "#21262d" : "#1a3d2e", border: "1px solid #3fb950", color: "#3fb950", borderRadius: 6, cursor: uploading ? "default" : "pointer", fontSize: 12 }}>
                ${uploading ? "Uploading…" : "Upload"}
              </button>
            </div>
            ${uploadMsg && html`<div style=${{ marginTop: 8, fontSize: 11, color: uploadMsg.startsWith("Error") ? "#f85149" : "#3fb950" }}>${uploadMsg}</div>`}
            <div style=${{ marginTop: 10, fontSize: 10, color: "#8b949e", lineHeight: 1.7 }}>
              config.json required fields: <code>name, version, task, format:"onnx", input_shape, output_shapes</code>
            </div>
          </div>`}

      </div>
    </div>`;
}

// ── SAVE AS TEMPLATE MODAL ────────────────────────────────────────────────────
function CustomNodeModal({ onClose, onSave, currentNodes, currentEdges }) {
  const [tab,  setTab]  = useState("pipeline"); // "pipeline" | "manual"
  const [name, setName] = useState("");
  const [icon, setIcon] = useState("📦");

  // Detect pipeline_output nodes in the current canvas
  const outputNodes = (currentNodes || []).filter(n => n.type === "pipeline_output");
  // Detect source nodes
  const SOURCE_TYPES = new Set(["camera","usb_camera","rtsp_stream","video_file","image_directory"]);
  const sourceNodes  = (currentNodes || []).filter(n => SOURCE_TYPES.has(n.type));

  // Build output ports from pipeline_output nodes — read data.ports.inputs (custom) or fallback to config.label
  const DEFAULT_PO = [{ id: "frame", label: "Frame" }, { id: "dets", label: "Detections" }];
  const autoPorts = outputNodes.flatMap(n =>
    n.data.ports?.inputs ?? (n.data.config?.label
      ? [{ id: n.data.config.label.toLowerCase().replace(/\s+/g, "_"), label: n.data.config.label }]
      : DEFAULT_PO)
  );

  // ── Manual tab state ──────────────────────────────────────────────────────
  const [mForm, setMForm] = useState({
    name: "", icon: "◻", group: "utility", groupLabel: "Custom",
    inputs:  [{ id: "in",  label: "Frame" }],
    outputs: [{ id: "out", label: "Frame" }],
  });
  const mf = (k, v) => setMForm(p => ({ ...p, [k]: v }));

  const addPort = (side) => {
    const key = side === "in" ? "inputs" : "outputs";
    mf(key, [...mForm[key], { id: "p" + Date.now(), label: "Port" }]);
  };
  const rmPort  = (side, i) => {
    const key = side === "in" ? "inputs" : "outputs";
    mf(key, mForm[key].filter((_, j) => j !== i));
  };
  const updPort = (side, i, field, val) => {
    const key = side === "in" ? "inputs" : "outputs";
    mf(key, mForm[key].map((p, j) => j === i ? { ...p, [field]: val } : p));
  };

  const portRow = (side, p, i) => html`
    <div key=${i} style=${{ display: "flex", gap: 4, marginBottom: 4, alignItems: "center" }}>
      <input placeholder="id" style=${{ ...inp, flex: 1, padding: "3px 5px" }} value=${p.id}
        onChange=${e => updPort(side, i, "id", e.target.value)} />
      <input placeholder="label" style=${{ ...inp, flex: 2, padding: "3px 5px" }} value=${p.label}
        onChange=${e => updPort(side, i, "label", e.target.value)} />
      <button onClick=${() => rmPort(side, i)}
        style=${{ background: "none", border: "none", color: "#f85149", cursor: "pointer", padding: "2px 5px", fontSize: 13 }}>✕</button>
    </div>`;

  const savePipeline = () => {
    if (!name.trim()) return;
    const type = "tmpl_" + name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
    const inputs = sourceNodes.length > 0
      ? []  // template takes no inputs — it has its own source
      : [{ id: "in", label: "Frame" }];
    const outputs = autoPorts.length > 0 ? autoPorts : [{ id: "out", label: "Output" }];
    const pipelineJson = {
      nodes: (currentNodes || []).map(n => ({ id: n.id, type: n.type, config: n.data.config ?? {} })),
      edges: (currentEdges || []).map(e => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle })),
    };
    onSave({
      type,
      meta:  { group: "visualize", icon, label: name, group_label: "Templates" },
      ports: { inputs, outputs },
      config: { pipeline_json: pipelineJson },
    });
    onClose();
  };

  const saveManual = () => {
    if (!mForm.name.trim()) return;
    const type = mForm.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
    onSave({ type, meta: { group: mForm.group, icon: mForm.icon, label: mForm.name, group_label: mForm.groupLabel }, ports: { inputs: mForm.inputs, outputs: mForm.outputs }, config: {} });
    onClose();
  };

  const tabBtn = (id, label) => html`
    <button onClick=${() => setTab(id)}
      style=${{ padding: "5px 14px", borderRadius: 5, cursor: "pointer", fontSize: 12,
                 background: tab === id ? "#1f3a5e" : "transparent",
                 border: "1px solid " + (tab === id ? "#58a6ff" : "#30363d"),
                 color: tab === id ? "#58a6ff" : "#8b949e" }}>
      ${label}
    </button>`;

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 1001, background: "rgba(0,0,0,.75)",
                    display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
                      width: 500, maxHeight: "85vh", overflowY: "auto",
                      boxShadow: "0 16px 48px rgba(0,0,0,.8)", padding: 20 }}>

        <div style=${{ display: "flex", alignItems: "center", marginBottom: 14 }}>
          <span style=${{ fontWeight: 700, fontSize: 15, color: "#e2e8f0", flex: 1 }}>⬡ Save as Template</span>
          <button onClick=${onClose} style=${{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18 }}>✕</button>
        </div>

        <!-- Tabs -->
        <div style=${{ display: "flex", gap: 6, marginBottom: 16 }}>
          ${tabBtn("pipeline", "From Current Pipeline")}
          ${tabBtn("manual",   "Define Manually")}
        </div>

        ${tab === "pipeline" && html`
          <!-- Pipeline summary -->
          <div style=${{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8,
                          padding: "10px 14px", marginBottom: 14, fontSize: 11 }}>
            <div style=${{ color: "#8b949e", marginBottom: 6 }}>Current pipeline</div>
            <div style=${{ color: "#e2e8f0" }}>${(currentNodes || []).length} nodes · ${(currentEdges || []).length} edges</div>
            ${sourceNodes.length > 0 && html`
              <div style=${{ marginTop: 4, color: "#d29922" }}>
                ⚠ Contains ${sourceNodes.length} source node(s) — template will be self-contained (no Frame input)
              </div>`}
          </div>

          <!-- Pipeline output ports detected -->
          <div style=${{ marginBottom: 14 }}>
            <div style=${{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
              <label style=${lbl}>Output Ports (from Pipeline Output nodes)</label>
            </div>
            ${autoPorts.length === 0
              ? html`<div style=${{ fontSize: 11, color: "#d29922", padding: "8px 10px", background: "#2d1a00", border: "1px solid #d2992244", borderRadius: 5 }}>
                  No "Pipeline Output" nodes found. Add one to the canvas to define outputs, or the template will use a single default output.
                </div>`
              : autoPorts.map((p, i) => html`
                  <div key=${i} style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <div style=${{ width: 8, height: 8, borderRadius: "50%", background: "#3fb950", flexShrink: 0 }} />
                    <span style=${{ fontSize: 12, color: "#3fb950" }}>${p.label}</span>
                    <span style=${{ fontSize: 10, color: "#555d68", fontFamily: "monospace" }}>${p.id}</span>
                  </div>`)}
          </div>

          <!-- Name + Icon -->
          <div style=${{ display: "flex", gap: 8, marginBottom: 14 }}>
            <div style=${{ flex: 1 }}><${Field} label="Template Name"><input style=${inp} value=${name} onChange=${e => setName(e.target.value)} placeholder="e.g. YOLOv8 Detector" /><//></div>
            <div><label style=${lbl}>Icon</label>
              <input style=${{ ...inp, width: 48, textAlign: "center" }} value=${icon} onChange=${e => setIcon(e.target.value)} /></div>
          </div>

          <button onClick=${savePipeline} disabled=${!name.trim()}
            style=${{ width: "100%", padding: "8px 0", borderRadius: 6,
                       background: name.trim() ? "#1a3d2e" : "#21262d",
                       border: "1px solid " + (name.trim() ? "#3fb950" : "#30363d"),
                       color: name.trim() ? "#3fb950" : "#8b949e",
                       cursor: name.trim() ? "pointer" : "default", fontSize: 13, fontWeight: 600 }}>
            Save Pipeline as Template
          </button>
          <div style=${{ marginTop: 8, fontSize: 10, color: "#555d68", textAlign: "center" }}>
            Template appears in the palette under "Templates" and can be dropped into any pipeline
          </div>`}

        ${tab === "manual" && html`
          <${Field} label="Node Name"><input style=${inp} value=${mForm.name} onChange=${e => mf("name", e.target.value)} placeholder="My Custom Node" /><//>
          <div style=${{ display: "flex", gap: 8, marginBottom: 10 }}>
            <div><label style=${lbl}>Icon</label>
              <input style=${{ ...inp, width: 48, textAlign: "center" }} value=${mForm.icon} onChange=${e => mf("icon", e.target.value)} /></div>
            <div style=${{ flex: 1 }}><label style=${lbl}>Color Group</label>
              <select style=${inp} value=${mForm.group} onChange=${e => mf("group", e.target.value)}>
                ${["input","processing","visualize","vision","spatial","utility","cpp","output"].map(g => html`<option key=${g} value=${g}>${g}</option>`)}
              </select></div>
            <div style=${{ flex: 1 }}><label style=${lbl}>Palette Section</label>
              <input style=${inp} value=${mForm.groupLabel} onChange=${e => mf("groupLabel", e.target.value)} /></div>
          </div>
          <div style=${{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
            <div>
              <div style=${{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <label style=${lbl}>Inputs</label>
                <button onClick=${() => addPort("in")} style=${{ fontSize: 10, background: "#1a2a40", border: "1px solid #58a6ff44", color: "#58a6ff", borderRadius: 4, padding: "2px 7px", cursor: "pointer" }}>+ Add</button>
              </div>
              ${mForm.inputs.map((p, i) => portRow("in", p, i))}
            </div>
            <div>
              <div style=${{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <label style=${lbl}>Outputs</label>
                <button onClick=${() => addPort("out")} style=${{ fontSize: 10, background: "#1a3a20", border: "1px solid #3fb95044", color: "#3fb950", borderRadius: 4, padding: "2px 7px", cursor: "pointer" }}>+ Add</button>
              </div>
              ${mForm.outputs.map((p, i) => portRow("out", p, i))}
            </div>
          </div>
          <button onClick=${saveManual} disabled=${!mForm.name.trim()}
            style=${{ width: "100%", padding: "8px 0", borderRadius: 6,
                       background: mForm.name.trim() ? "#1f3a5e" : "#21262d",
                       border: "1px solid " + (mForm.name.trim() ? "#58a6ff" : "#30363d"),
                       color: mForm.name.trim() ? "#58a6ff" : "#8b949e",
                       cursor: mForm.name.trim() ? "pointer" : "default", fontSize: 13, fontWeight: 600 }}>
            Add to Palette
          </button>`}

      </div>
    </div>`;
}

// ── SAVE NODE TO LIBRARY MODAL ────────────────────────────────────────────────
function SaveNodeModal({ node, onClose, onSave }) {
  const defaultName = node.data.label ?? node.type;
  const defaultSlug = defaultName.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
  const [name,  setName]  = useState(defaultName);
  const [icon,  setIcon]  = useState("⚙");
  const [group, setGroup] = useState("utility");

  const cfg = node.data.config ?? {};
  const code = cfg.code ?? "";

  // Infer ports from code signature (same logic as parsePySignature + parsePyOutputs)
  const inputs  = (parsePySignature(code) ?? []).map(p => ({ id: p.name, label: p.name }));
  const outputs = parsePyOutputs(code);

  const doSave = () => {
    if (!name.trim()) return;
    const slug = name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "") || defaultSlug;
    onSave({
      type: slug,
      meta: { label: name.trim(), group, icon, group_label: "My Nodes" },
      ports: { inputs, outputs: outputs.length ? outputs : [{ id: "out", label: "out" }] },
      config: { ...cfg },
    });
    onClose();
  };

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,.72)",
                    display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
                      width: 400, boxShadow: "0 16px 48px rgba(0,0,0,.8)", padding: 20 }}>

        <div style=${{ display: "flex", alignItems: "center", marginBottom: 14 }}>
          <span style=${{ fontWeight: 700, fontSize: 15, color: "#e2e8f0", flex: 1 }}>⬡ Save Node to Library</span>
          <button onClick=${onClose} style=${{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18 }}>✕</button>
        </div>

        <div style=${{ fontSize: 11, color: "#8b949e", marginBottom: 14, background: "#0d1117",
                        border: "1px solid #21262d", borderRadius: 6, padding: "8px 12px" }}>
          Node will appear in the palette under <b style=${{ color: "#d29922" }}>My Nodes</b>.
          Drop it into any pipeline — it will have the same code and config as this node.
          ${inputs.length > 0 && html`<br/><br/>Auto-detected ports: <b>${inputs.map(p => p.id).join(", ")}</b>`}
        </div>

        <div style=${{ display: "flex", gap: 8, marginBottom: 12 }}>
          <div style=${{ flex: 1 }}>
            <label style=${lbl}>Name</label>
            <input style=${inp} value=${name} onChange=${e => setName(e.target.value)} placeholder="My Node" />
          </div>
          <div>
            <label style=${lbl}>Icon</label>
            <input style=${{ ...inp, width: 48, textAlign: "center" }} value=${icon} onChange=${e => setIcon(e.target.value)} />
          </div>
        </div>

        <div style=${{ marginBottom: 16 }}>
          <label style=${lbl}>Color Group</label>
          <select style=${inp} value=${group} onChange=${e => setGroup(e.target.value)}>
            ${["input","processing","visualize","vision","spatial","utility","cpp","output"].map(g =>
              html`<option key=${g} value=${g}>${g}</option>`)}
          </select>
        </div>

        <button onClick=${doSave} disabled=${!name.trim()}
          style=${{ width: "100%", padding: "8px 0", borderRadius: 6,
                     background: name.trim() ? "#1a3d2e" : "#21262d",
                     border: "1px solid " + (name.trim() ? "#3fb950" : "#30363d"),
                     color: name.trim() ? "#3fb950" : "#8b949e",
                     cursor: name.trim() ? "pointer" : "default", fontSize: 13, fontWeight: 600,
                     fontFamily: "inherit" }}>
          Save to Library
        </button>
      </div>
    </div>`;
}

// ── KEYBOARD SHORTCUTS HELP ───────────────────────────────────────────────────
function ShortcutsModal({ onClose }) {
  const shortcuts = [
    ["Ctrl + S",       "Save pipeline"],
    ["Ctrl + Z",       "Undo"],
    ["Ctrl + Y",       "Redo"],
    ["Ctrl + D",       "Duplicate selected node"],
    ["Ctrl + K",       "Quick-add node (spotlight)"],
    ["Delete",         "Delete selected node / edge"],
    ["Ctrl + A",       "Select all nodes"],
    ["Escape",         "Deselect / close modals"],
    ["?",              "Show this help"],
    ["Ctrl + Shift+L", "Auto-layout nodes"],
    ["Ctrl + E",       "Export pipeline JSON"],
  ];
  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,.7)",
                    display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
                      width: 380, boxShadow: "0 16px 48px rgba(0,0,0,.8)", padding: 20 }}>
        <div style=${{ display: "flex", alignItems: "center", marginBottom: 14 }}>
          <span style=${{ fontWeight: 700, fontSize: 15, color: "#e2e8f0", flex: 1 }}>⌨ Keyboard Shortcuts</span>
          <button onClick=${onClose} style=${{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18 }}>✕</button>
        </div>
        ${shortcuts.map(([key, desc]) => html`
          <div key=${key} style=${{ display: "flex", alignItems: "center", marginBottom: 8 }}>
            <code style=${{ background: "#21262d", border: "1px solid #30363d", borderRadius: 4,
                             padding: "2px 8px", fontSize: 11, color: "#58a6ff", minWidth: 140, textAlign: "center", marginRight: 12 }}>
              ${key}
            </code>
            <span style=${{ fontSize: 12, color: "#c9d1d9" }}>${desc}</span>
          </div>`)}
      </div>
    </div>`;
}

// ── QUICK-ADD SPOTLIGHT ───────────────────────────────────────────────────────
function QuickAddModal({ onClose, onAdd }) {
  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef(null);
  useEffect(() => { inputRef.current?.focus(); }, []);

  const all = useMemo(() =>
    Object.entries(NODE_META).map(([type, m]) => ({ type, ...m }))
  , []);

  const results = useMemo(() => {
    const lq = q.toLowerCase();
    if (!lq) return all.slice(0, 12);
    return all.filter(n => n.label.toLowerCase().includes(lq) || n.type.includes(lq)).slice(0, 12);
  }, [q, all]);

  useEffect(() => { setCursor(0); }, [results.length]);

  const select = (type) => { onAdd(type); onClose(); };

  const onKey = (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor(c => Math.min(c + 1, results.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)); }
    else if (e.key === "Enter") { if (results[cursor]) select(results[cursor].type); }
    else if (e.key === "Escape") onClose();
  };

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,.6)",
                    display: "flex", alignItems: "flex-start", justifyContent: "center", paddingTop: 100 }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
                      width: 440, boxShadow: "0 20px 60px rgba(0,0,0,.9)" }}>
        <div style=${{ padding: "10px 14px", borderBottom: "1px solid #30363d" }}>
          <input ref=${inputRef} value=${q} onChange=${e => setQ(e.target.value)} onKeyDown=${onKey}
            placeholder="Search and add a node…"
            style=${{ width: "100%", background: "transparent", border: "none", outline: "none",
                       color: "#e2e8f0", fontSize: 14, fontFamily: "inherit" }} />
        </div>
        <div style=${{ maxHeight: 360, overflowY: "auto", padding: "6px 0" }}>
          ${results.length === 0 && html`<div style=${{ padding: "16px", color: "#8b949e", fontSize: 12, textAlign: "center" }}>No nodes found</div>`}
          ${results.map((n, i) => html`
            <div key=${n.type} onClick=${() => select(n.type)}
              style=${{ display: "flex", alignItems: "center", gap: 10, padding: "8px 14px", cursor: "pointer",
                         background: cursor === i ? "#21262d" : "transparent",
                         borderLeft: cursor === i ? "2px solid #58a6ff" : "2px solid transparent" }}
              onMouseEnter=${() => setCursor(i)}>
              <div style=${{ width: 26, height: 26, borderRadius: 6,
                              background: GROUP_COLOR[n.group] ?? "#21262d",
                              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, flexShrink: 0 }}>
                ${n.icon}
              </div>
              <div>
                <div style=${{ fontSize: 13, color: "#e2e8f0", fontWeight: 500 }}>${n.label}</div>
                <div style=${{ fontSize: 10, color: "#555d68" }}>${n.type} · ${n.group}</div>
              </div>
            </div>`)}
        </div>
        <div style=${{ padding: "6px 14px", borderTop: "1px solid #30363d",
                        fontSize: 10, color: "#555d68", display: "flex", gap: 12 }}>
          <span>↑↓ navigate</span><span>↵ add node</span><span>Esc close</span>
        </div>
      </div>
    </div>`;
}

// ── VALIDATE WARNINGS MODAL ───────────────────────────────────────────────────
function ValidateModal({ warnings, onClose, onRunAnyway }) {
  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,.7)",
                    display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #d29922", borderRadius: 12,
                      width: 420, boxShadow: "0 16px 48px rgba(0,0,0,.8)", padding: 20 }}>
        <div style=${{ fontWeight: 700, fontSize: 15, color: "#e2e8f0", marginBottom: 14 }}>⚠ Pipeline Warnings</div>
        ${warnings.map((w, i) => html`
          <div key=${i} style=${{ display: "flex", gap: 8, marginBottom: 8, alignItems: "flex-start" }}>
            <span style=${{ color: "#d29922", flexShrink: 0 }}>·</span>
            <span style=${{ fontSize: 12, color: "#c9d1d9", lineHeight: 1.5 }}>${w}</span>
          </div>`)}
        <div style=${{ display: "flex", gap: 8, marginTop: 16 }}>
          <button onClick=${onRunAnyway}
            style=${{ flex: 1, padding: "7px 0", borderRadius: 6, background: "#3d2a00", border: "1px solid #d29922", color: "#d29922", cursor: "pointer", fontSize: 12 }}>
            Run Anyway
          </button>
          <button onClick=${onClose}
            style=${{ flex: 1, padding: "7px 0", borderRadius: 6, background: "#1f3a5e", border: "1px solid #58a6ff", color: "#58a6ff", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
            Fix Issues
          </button>
        </div>
      </div>
    </div>`;
}

// ── NODE CONTEXT MENU ─────────────────────────────────────────────────────────
function ContextMenu({ x, y, node, onClose, onDuplicate, onDelete, onResetConfig, onEditCode, onSaveNode }) {
  const items = [
    ...(node.type === "python_node"
        ? [
            { icon: "✏", label: "Edit Code",       action: onEditCode,  primary: true },
            { icon: "⬡", label: "Save to Library", action: onSaveNode  },
          ]
        : []),
    { icon: "⎘", label: "Duplicate", action: onDuplicate },
    { icon: "↺", label: "Reset Config", action: onResetConfig },
    { icon: "🗑", label: "Delete",     action: onDelete, danger: true },
  ];
  useEffect(() => {
    const close = () => onClose();
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);
  return html`
    <div style=${{ position: "fixed", left: x, top: y, zIndex: 3000,
                    background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
                    minWidth: 160, boxShadow: "0 8px 24px rgba(0,0,0,.7)", padding: "4px 0" }}>
      <div style=${{ padding: "6px 12px 4px", fontSize: 10, color: "#555d68", borderBottom: "1px solid #30363d", marginBottom: 2 }}>
        ${node.data.label}
      </div>
      ${items.map((it, i) => html`
        ${it.danger && html`<div key=${"sep-" + i} style=${{ borderTop: "1px solid #30363d", margin: "4px 0" }} />`}
        <button key=${it.label} onClick=${e => { e.stopPropagation(); it.action(); onClose(); }}
          style=${{ display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "7px 12px",
                     background: "none", border: "none", cursor: "pointer", fontSize: 12,
                     color: it.danger ? "#f85149" : it.primary ? "#58a6ff" : "#c9d1d9", textAlign: "left",
                     fontWeight: it.primary ? 600 : 400 }}
          onMouseEnter=${e => e.currentTarget.style.background = "#21262d"}
          onMouseLeave=${e => e.currentTarget.style.background = ""}>
          <span>${it.icon}</span><span>${it.label}</span>
        </button>`)}
    </div>`;
}

// ── CODE EDITOR OVERLAY ───────────────────────────────────────────────────────
// Full-screen editor opened via right-click → "Edit Code" on a python_node.
function CodeEditorOverlay({ node, onClose, onUpdate }) {
  const cfg   = node?.data?.config ?? {};
  const ports = node?.data?.ports  ?? { inputs: [], outputs: [] };
  const meta  = NODE_META[node?.type] ?? {};
  const mode  = cfg.mode ?? "loop";

  const handleChange = useCallback((code) => {
    onUpdate(node.id, { ...cfg, code });
  }, [node.id, JSON.stringify(cfg)]);

  const handleBlur = useCallback((code) => {
    const parsedIn  = parsePySignature(code);
    const parsedOut = parsePyOutputs(code);
    const cur = node.data.ports ?? { inputs: [], outputs: [] };
    const newIn  = parsedIn  ? parsedIn.map(p => ({ id: p.name, label: p.name })) : cur.inputs;
    const newOut = parsedOut.length > 0 ? parsedOut : cur.outputs;
    onUpdate(node.id, { ...cfg, code }, undefined, { inputs: newIn, outputs: newOut });
  }, [node.id, JSON.stringify(cfg)]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!node) return null;

  const portRow = (p, color) => html`
    <div key=${p.id} style=${{ fontSize: 12, color, padding: "4px 0", display: "flex", alignItems: "center", gap: 6 }}>
      <span>●</span><span style=${{ fontFamily: "monospace" }}>${p.label}</span>
    </div>`;

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 1500, background: "#0d1117",
                    display: "flex", flexDirection: "column" }}>

      <!-- Header bar -->
      <div style=${{ display: "flex", alignItems: "center", gap: 10, padding: "0 20px", height: 48,
                      borderBottom: "1px solid #30363d", background: "#161b22", flexShrink: 0 }}>
        <span style=${{ fontSize: 15 }}>${meta.icon ?? "🐍"}</span>
        <span style=${{ fontWeight: 700, color: "#e2e8f0", fontSize: 14 }}>${node.data.label}</span>
        <span style=${{
          fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
          background: mode === "iteration" ? "#2d1a4a" : "#1a2d1a",
          color:      mode === "iteration" ? "#c9a0ff" : "#3fb950",
        }}>${mode.toUpperCase()}</span>
        <div style=${{ flex: 1 }} />
        <span style=${{ fontSize: 11, color: "#555d68", marginRight: 8 }}>Esc to close</span>
        <button onClick=${onClose} style=${{
          background: "#21262d", border: "1px solid #30363d", color: "#c9d1d9",
          cursor: "pointer", borderRadius: 6, padding: "5px 16px", fontSize: 12, fontFamily: "inherit",
        }}>✕  Close</button>
      </div>

      <!-- Editor + Ports -->
      <div style=${{ display: "flex", flex: 1, overflow: "hidden" }}>

        <!-- CodeMirror editor -->
        <div style=${{ flex: 1, overflow: "auto" }}>
          <${PythonEditor}
            key=${node.id}
            value=${cfg.code ?? ""}
            onChange=${handleChange}
            onBlur=${handleBlur}
            minHeight=${700} />
        </div>

        <!-- Ports + hints sidebar -->
        <div style=${{
          width: 240, padding: "20px 16px", overflowY: "auto", flexShrink: 0,
          borderLeft: "1px solid #30363d", background: "#161b22", display: "flex", flexDirection: "column", gap: 0,
        }}>
          <div style=${{ fontSize: 10, fontWeight: 700, color: "#8b949e", letterSpacing: .8,
                          textTransform: "uppercase", marginBottom: 8 }}>Input Ports</div>
          ${ports.inputs.length === 0
            ? html`<div style=${{ fontSize: 11, color: "#555d68" }}>none — no params in loop()</div>`
            : ports.inputs.map(p => portRow(p, "#58a6ff"))}

          <div style=${{ fontSize: 10, fontWeight: 700, color: "#8b949e", letterSpacing: .8,
                          textTransform: "uppercase", marginBottom: 8, marginTop: 20 }}>Output Ports</div>
          ${ports.outputs.length === 0
            ? html`<div style=${{ fontSize: 11, color: "#555d68" }}>none — no return value</div>`
            : ports.outputs.map(p => portRow(p, "#3fb950"))}

          <div style=${{ flex: 1 }} />

          <div style=${{ padding: 12, background: "#0d1117", borderRadius: 6, border: "1px solid #21262d",
                          fontSize: 10, color: "#555d68", lineHeight: 1.8, marginTop: 20 }}>
            <div style=${{ color: "#8b949e", fontWeight: 600, marginBottom: 4 }}>Routing</div>
            "frame" → ctx.frame<br/>
            other → ctx.metadata[name]<br/>
            return ndarray → ctx.frame
            <div style=${{ color: "#8b949e", fontWeight: 600, marginTop: 8, marginBottom: 4 }}>Globals</div>
            show_image(img, label="")<br/>
            show_text(text)<br/>
            config: dict<br/>
            slider(name, min, max, default)<br/>
            checkbox(name, default)<br/>
            text_input(name, default)<br/>
            button(name)
            <div style=${{ color: "#8b949e", fontWeight: 600, marginTop: 8, marginBottom: 4 }}>Shortcuts</div>
            Tab → indent<br/>
            Ctrl+Space → complete<br/>
            Ctrl+Z / Ctrl+Y → undo/redo
          </div>
        </div>
      </div>
    </div>`;
}

// ── PALETTE ───────────────────────────────────────────────────────────────────
const RECENT_KEY = "cvflow_recent_types";

function addRecent(type) {
  try {
    const prev = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    const next = [type, ...prev.filter(t => t !== type)].slice(0, 6);
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  } catch {}
}
function getRecent() {
  try { return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); }
  catch { return []; }
}

const _BASE_TYPES = new Set(BASE_GROUPS.flatMap(g => g.types));

function Palette({ groups, onAddCustom, onDeleteTemplate }) {
  const [query, setQuery]     = useState("");
  const [recent, setRecent]   = useState(getRecent);
  const [ctxMenu, setCtxMenu] = useState(null); // { type, label, x, y }
  const q = query.toLowerCase();

  // Close context menu on any outside click
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [ctxMenu]);

  const recentGroup = recent.filter(t => NODE_META[t]);
  const visible = q
    ? groups.map(g => ({
        ...g,
        types: g.types.filter(t => {
          const m = NODE_META[t];
          return m && (m.label.toLowerCase().includes(q) || t.includes(q));
        }),
      })).filter(g => g.types.length > 0)
    : (recentGroup.length > 0
        ? [{ label: "Recent", types: recentGroup }, ...groups]
        : groups);

  const handleDragStart = (e, type) => {
    e.dataTransfer.setData("application/cvflow", type);
    e.dataTransfer.effectAllowed = "move";
    addRecent(type);
    setRecent(getRecent());
  };

  const handleContextMenu = (e, type) => {
    if (_BASE_TYPES.has(type)) return; // built-in nodes cannot be deleted
    e.preventDefault();
    e.stopPropagation();
    const m = NODE_META[type];
    setCtxMenu({ type, label: m?.label ?? type, x: e.clientX, y: e.clientY });
  };

  const handleDelete = () => {
    if (!ctxMenu) return;
    onDeleteTemplate(ctxMenu.type);
    setRecent(prev => prev.filter(t => t !== ctxMenu.type));
    setCtxMenu(null);
  };

  return html`
    <div style=${{ width: 172, flexShrink: 0, borderRight: "1px solid #30363d",
                   background: "#0d1117", display: "flex", flexDirection: "column" }}>
      <div style=${{ padding: "6px 6px 4px", flexShrink: 0 }}>
        <input placeholder="🔍  Search nodes  (Ctrl+K)" value=${query}
          onChange=${e => setQuery(e.target.value)}
          style=${{ width: "100%", background: "#21262d", border: "1px solid #30363d",
                     borderRadius: 6, color: "#c9d1d9", padding: "5px 8px", fontSize: 11,
                     outline: "none", fontFamily: "inherit", boxSizing: "border-box" }} />
      </div>
      <div style=${{ flex: 1, overflowY: "auto", padding: "2px 5px" }}>
        ${visible.map(g => html`
          <div key=${g.label} style=${{ marginBottom: 8 }}>
            <div style=${{ fontSize: 10, color: g.label === "Recent" ? "#58a6ff" : "#8b949e",
                           textTransform: "uppercase", letterSpacing: ".6px",
                           padding: "2px 4px 4px", fontWeight: 700 }}>
              ${g.label === "Recent" ? "⏱ " : ""}${g.label}
            </div>
            ${g.types.map(type => {
              const m = NODE_META[type];
              if (!m) return null;
              const bg = GROUP_COLOR[m.group] ?? "#21262d";
              const deletable = !_BASE_TYPES.has(type);
              return html`
                <div key=${type} draggable=${true}
                  onDragStart=${e => handleDragStart(e, type)}
                  onContextMenu=${e => handleContextMenu(e, type)}
                  title=${deletable ? type + "\n(right-click to delete)" : type}
                  style=${{ display: "flex", alignItems: "center", gap: 6, padding: "5px 7px",
                             borderRadius: 6, cursor: "grab", marginBottom: 3,
                             background: bg + "55", border: "1px solid " + bg + "88",
                             userSelect: "none", fontSize: 11, color: "#c9d1d9", transition: "opacity .1s" }}
                  onMouseEnter=${e => e.currentTarget.style.opacity = ".72"}
                  onMouseLeave=${e => e.currentTarget.style.opacity = "1"}>
                  <span style=${{ flexShrink: 0 }}>${m.icon}</span>
                  <span style=${{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>${m.label}</span>
                  ${deletable && html`<span style=${{ fontSize: 9, color: "#555d68", flexShrink: 0 }}>⋮</span>`}
                </div>`;
            })}
          </div>`)}
      </div>
      <div style=${{ padding: "5px 5px 6px", borderTop: "1px solid #30363d", flexShrink: 0 }}>
        <button onClick=${onAddCustom}
          style=${{ width: "100%", padding: "5px 0", background: "transparent",
                     border: "1px dashed #30363d", color: "#8b949e", borderRadius: 6,
                     cursor: "pointer", fontSize: 11, fontFamily: "inherit" }}>
          ⬡ Save as Template
        </button>
      </div>

      ${ctxMenu && html`
        <div onClick=${e => e.stopPropagation()}
             style=${{ position: "fixed", left: ctxMenu.x, top: ctxMenu.y, zIndex: 9999,
                        background: "#161b22", border: "1px solid #30363d", borderRadius: 7,
                        padding: "4px 0", fontSize: 12, color: "#c9d1d9",
                        boxShadow: "0 6px 20px rgba(0,0,0,.7)", minWidth: 160 }}>
          <div style=${{ padding: "5px 12px 7px", fontSize: 10, color: "#555d68",
                          borderBottom: "1px solid #21262d", userSelect: "none" }}>
            ${ctxMenu.label}
          </div>
          <div style=${{ padding: "7px 12px", cursor: "pointer", display: "flex",
                          alignItems: "center", gap: 8, color: "#f85149",
                          borderRadius: "0 0 6px 6px" }}
               onClick=${handleDelete}
               onMouseEnter=${e => e.currentTarget.style.background = "#2d1010"}
               onMouseLeave=${e => e.currentTarget.style.background = "transparent"}>
            🗑 Xóa template
          </div>
        </div>`}
    </div>`;
}

// ── ENGINE LOG VIEWER ─────────────────────────────────────────────────────────
const MAX_LOG_LINES = 500;

function logLineColor(l) {
  if (l.includes("[ERROR]") || l.includes("ERROR") || l.includes("Traceback") ||
      l.includes("Error:") || l.includes("Exception")) return "#f85149";
  if (l.includes("[WARNING]") || l.includes("WARNING") || l.includes("Warning")) return "#d29922";
  if (l.includes("[INFO]") && (l.includes("Pipeline") || l.includes("started") ||
      l.includes("Session") || l.includes("listening"))) return "#3fb950";
  return "#8b949e";
}

function EngineLogViewer({ sessionId, height = 160 }) {
  const [lines,   setLines]   = useState([{ text: "— Waiting for pipeline —", sys: true }]);
  const [pinned,  setPinned]  = useState(true);   // auto-scroll to bottom
  const scrollRef = useRef(null);
  const esRef     = useRef(null);
  const prevSid   = useRef(null);

  // Start SSE stream when sessionId changes
  useEffect(() => {
    if (sessionId === prevSid.current) return;
    prevSid.current = sessionId;

    // Close any previous stream
    if (esRef.current) { esRef.current.close(); esRef.current = null; }

    if (!sessionId) return;

    setLines([{ text: "▶ Pipeline starting…", sys: true }]);
    setPinned(true);

    const es = new EventSource(`/api/execution/logs/${sessionId}/stream`);
    esRef.current = es;

    es.onmessage = e => {
      const text = e.data;
      setLines(prev => {
        const next = [...prev, { text, sys: false }];
        return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next;
      });
    };

    es.addEventListener("done", ev => {
      const status = ev.data;
      const msg = status === "error" ? "✖ Pipeline stopped with error"
                : status === "completed" ? "✔ Pipeline finished"
                : "■ Pipeline stopped";
      setLines(prev => [...prev, { text: msg, sys: true }]);
      es.close();
      esRef.current = null;
    });

    es.onerror = () => {
      es.close();
      esRef.current = null;
    };

    return () => { es.close(); esRef.current = null; };
  }, [sessionId]);

  // Auto-scroll when pinned
  useEffect(() => {
    if (pinned && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, pinned]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    // Unpin if user scrolled up; re-pin if they scroll back to bottom
    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 10);
  };

  return html`
    <div style=${{ borderTop: "1px solid #30363d", display: "flex", flexDirection: "column" }}>
      <!-- Header -->
      <div style=${{ display: "flex", alignItems: "center", padding: "5px 10px 5px 14px",
                      borderBottom: "1px solid #21262d" }}>
        <span style=${{ fontSize: 10, color: "#555d68", textTransform: "uppercase",
                         letterSpacing: ".5px", flex: 1 }}>Engine Log</span>
        ${!pinned && html`
          <button onClick=${() => { setPinned(true); }}
            title="Scroll to bottom"
            style=${{ background: "none", border: "none", cursor: "pointer", color: "#58a6ff",
                       fontSize: 11, padding: "0 4px" }}>↓</button>`}
        <button onClick=${() => {
            const text = lines.map(l => l.text).join("\n");
            navigator.clipboard.writeText(text).catch(() => {});
          }}
          title="Copy all logs"
          style=${{ background: "none", border: "none", cursor: "pointer", color: "#555d68",
                     fontSize: 11, padding: "0 4px" }}>⎘</button>
        <button onClick=${() => setLines([{ text: "— Log cleared —", sys: true }])}
          title="Clear log"
          style=${{ background: "none", border: "none", cursor: "pointer", color: "#555d68",
                     fontSize: 11, padding: "0 4px" }}>✕</button>
      </div>

      <!-- Log body -->
      <div ref=${scrollRef} onScroll=${onScroll}
        style=${{ height: height, overflowY: "auto", padding: "6px 10px",
                   fontFamily: "Consolas,'Courier New',monospace", fontSize: 10.5, lineHeight: 1.65,
                   background: "#0a0d12" }}>
        ${lines.map((l, i) => html`
          <div key=${i} style=${{
            color: l.sys ? "#4a9970" : logLineColor(l.text),
            whiteSpace: "pre-wrap", wordBreak: "break-all",
            opacity: l.sys ? 0.8 : 1,
          }}>${l.text}</div>`)}
      </div>
    </div>`;
}

// ── STREAM PANEL ──────────────────────────────────────────────────────────────
// The engine subprocess takes 1–3 s to start its WS server after the backend
// returns the session_id.  We retry the connection every 1.5 s (up to 20 times)
// so the stream lights up automatically once the engine is ready.
function StreamPanel({ sessionId, wsPort, counters, height = 180 }) {
  const imgRef     = useRef(null);
  const wsRef      = useRef(null);
  const retryRef   = useRef(null);
  const prevUrl    = useRef(null);
  const fpsRef     = useRef({ n: 0, t: Date.now() });
  const attemptsRef = useRef(0);

  const [status, setStatus] = useState("idle");
  const [fps,    setFps]    = useState(0);
  const [retryN, setRetryN] = useState(0);
  const MAX_RETRIES = 20;

  useEffect(() => {
    if (!sessionId) {
      setStatus("idle"); setFps(0); setRetryN(0);
      return;
    }

    attemptsRef.current = 0;

    const connect = () => {
      // Abort any previous socket cleanly before creating a new one
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        try { wsRef.current.close(); } catch {}
        wsRef.current = null;
      }

      const ws = new WebSocket(`ws://localhost:${wsPort}/ws/stream/${sessionId}`);
      wsRef.current = ws;
      ws.binaryType = "blob";
      setStatus("connecting");

      ws.onopen = () => {
        attemptsRef.current = 0;
        setRetryN(0);
        setStatus("live");
      };

      ws.onerror = () => { /* onclose fires right after — handled there */ };

      ws.onclose = () => {
        if (attemptsRef.current < MAX_RETRIES) {
          const n = ++attemptsRef.current;
          setRetryN(n);
          setStatus("reconnecting");
          retryRef.current = setTimeout(connect, 1500);
        } else {
          setStatus("error");
        }
      };

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
    };

    connect();

    return () => {
      clearTimeout(retryRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        try { wsRef.current.close(); } catch {}
        wsRef.current = null;
      }
      if (prevUrl.current) {
        URL.revokeObjectURL(prevUrl.current);
        prevUrl.current = null;
      }
    };
  }, [sessionId, wsPort]);

  const statusColor = {
    idle: "#8b949e", connecting: "#d29922", reconnecting: "#d29922",
    live: "#3fb950", disconnected: "#f85149", error: "#f85149",
  }[status] ?? "#8b949e";
  const blinking = status === "live" || status === "reconnecting" || status === "connecting";

  return html`
    <div style=${{ borderTop: "1px solid #30363d", flexShrink: 0 }}>
      <div style=${{ display: "flex", alignItems: "center", gap: 8, padding: "7px 14px", borderBottom: "1px solid #30363d" }}>
        <div style=${{ width: 7, height: 7, borderRadius: "50%", background: statusColor,
                        animation: blinking ? "blink 2s infinite" : "none" }} />
        <span style=${{ fontSize: 10, color: "#8b949e", textTransform: "uppercase", letterSpacing: ".5px" }}>Live Stream</span>
        ${status === "live"         && html`<span style=${{ marginLeft: "auto", fontSize: 11, color: "#3fb950" }}>${fps} fps</span>`}
        ${status === "reconnecting" && html`<span style=${{ marginLeft: "auto", fontSize: 10, color: "#d29922" }}>connecting… ${retryN}/${MAX_RETRIES}</span>`}
        ${status === "error"        && html`<span style=${{ marginLeft: "auto", fontSize: 10, color: "#f85149" }}>engine error</span>`}
      </div>

      <div style=${{ background: "#000", height: height, display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
        ${!sessionId && html`<span style=${{ color: "#8b949e", fontSize: 11 }}>Run pipeline to stream</span>`}
        ${sessionId && status === "error" && html`
          <span style=${{ color: "#f85149", fontSize: 11, textAlign: "center", padding: "0 12px", lineHeight: 1.6 }}>
            Engine error<br/>
            <span style=${{ fontSize: 10, color: "#8b949e" }}>Check camera / logs</span>
          </span>`}
        ${sessionId && status !== "error" && html`
          <img ref=${imgRef} style=${{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} alt="" />`}
      </div>

      ${Object.keys(counters).length > 0 && html`
        <div style=${{ padding: "6px 14px" }}>
          ${Object.entries(counters).map(([k, v]) => html`
            <div key=${k} style=${{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#c9d1d9", marginBottom: 2 }}>
              <span style=${{ color: "#8b949e" }}>${k}</span><b>${v}</b>
            </div>`)}
        </div>`}

    </div>`;
}

// ── HISTORY HOOK ──────────────────────────────────────────────────────────────
function useHistory(nodes, edges, setNodes, setEdges) {
  const stack  = useRef([]);
  const future = useRef([]);
  const skip   = useRef(false);

  const snapshot = useCallback(() => {
    if (skip.current) return;
    stack.current = [...stack.current.slice(-49), { nodes, edges }];
    future.current = [];
  }, [nodes, edges]);

  const undo = useCallback(() => {
    if (!stack.current.length) return;
    future.current = [{ nodes, edges }, ...future.current];
    const prev = stack.current.pop();
    skip.current = true;
    setNodes(prev.nodes);
    setEdges(prev.edges);
    setTimeout(() => { skip.current = false; }, 0);
  }, [nodes, edges, setNodes, setEdges]);

  const redo = useCallback(() => {
    if (!future.current.length) return;
    stack.current.push({ nodes, edges });
    const next = future.current.shift();
    skip.current = true;
    setNodes(next.nodes);
    setEdges(next.edges);
    setTimeout(() => { skip.current = false; }, 0);
  }, [nodes, edges, setNodes, setEdges]);

  return { snapshot, undo, redo };
}

// ── LOCAL STORAGE AUTO-SAVE ───────────────────────────────────────────────────
const LS_DRAFT = "cvflow_draft";
function saveDraft(name, nodes, edges) {
  try { localStorage.setItem(LS_DRAFT, JSON.stringify({ name, nodes, edges, ts: Date.now() })); }
  catch {}
}
function loadDraft() {
  try { return JSON.parse(localStorage.getItem(LS_DRAFT)); } catch { return null; }
}

// ── NODE DATA PANEL ───────────────────────────────────────────────────────────
// Shows live non-frame metadata emitted by the engine (landmarks, embeddings,
// match results, detection counts, etc.) for the currently selected node.
function NodeDataPanel({ nodeType, data }) {
  if (!data) return null;

  const s = {
    wrap:   { marginTop: 10, paddingTop: 10, borderTop: "1px solid #30363d" },
    title:  { ...lbl, marginBottom: 6 },
    card:   { background: "#0d1117", border: "1px solid #21262d", borderRadius: 6,
               padding: "7px 10px", marginBottom: 6, fontSize: 11 },
    key:    { color: "#8b949e", fontSize: 10, marginRight: 4 },
    val:    { color: "#e2e8f0" },
    badge:  (ok) => ({ display: "inline-block", borderRadius: 4, padding: "1px 7px",
                        fontSize: 10, fontWeight: 600,
                        background: ok ? "#1a3d2e" : "#3d1a1a",
                        border: "1px solid " + (ok ? "#3fb950" : "#f85149"),
                        color: ok ? "#3fb950" : "#f85149" }),
    mono:   { fontFamily: "Consolas,'Courier New',monospace", fontSize: 10, color: "#8b949e",
               wordBreak: "break-all" },
  };

  const KV = ({ k, v }) => html`
    <div style=${{ display: "flex", gap: 4, marginBottom: 2 }}>
      <span style=${s.key}>${k}</span>
      <span style=${s.val}>${typeof v === "boolean" ? (v ? "true" : "false") : String(v ?? "—")}</span>
    </div>`;

  let body = null;

  // ── face_detect ──────────────────────────────────────────────────────────
  if (nodeType === "face_detect") {
    body = html`
      <div style=${s.card}>
        <${KV} k="Faces detected" v=${data.faces_detected} />
      </div>
      ${(data.detections ?? []).map((det, i) => html`
        <div key=${i} style=${{ ...s.card, marginBottom: 4 }}>
          <div style=${{ color: "#58a6ff", fontSize: 10, fontWeight: 600, marginBottom: 3 }}>Face ${i + 1}</div>
          <${KV} k="bbox"       v=${"[" + det.bbox.join(", ") + "]"} />
          <${KV} k="confidence" v=${(det.confidence * 100).toFixed(1) + "%"} />
          ${data.landmarks?.[i] && html`
            <div style=${{ marginTop: 3 }}>
              <div style=${s.key}>landmarks (5pt)</div>
              <div style=${s.mono}>${data.landmarks[i].map(p => "[" + p.join(",") + "]").join("  ")}</div>
            </div>`}
        </div>`)}`;
  }

  // ── object_tracker ────────────────────────────────────────────────────────
  else if (nodeType === "object_tracker") {
    body = html`
      <div style=${s.card}><${KV} k="Tracked objects" v=${data.tracked_count} /></div>
      ${(data.tracks ?? []).map((t, i) => html`
        <div key=${i} style=${s.card}>
          <div style=${{ color: "#58a6ff", fontSize: 10, fontWeight: 600, marginBottom: 2 }}>
            #${t.id} · ${t.class}
          </div>
          <${KV} k="conf" v=${(t.conf * 100).toFixed(1) + "%"} />
          <${KV} k="bbox" v=${"[" + t.bbox.join(", ") + "]"} />
        </div>`)}`;
  }

  // ── track_db ──────────────────────────────────────────────────────────────
  else if (nodeType === "track_db") {
    body = html`
      <div style=${s.card}><${KV} k="Active tracks" v=${data.active_tracks} /></div>
      ${(data.tracks ?? []).map((t, i) => html`
        <div key=${i} style=${s.card}>
          <div style=${{ color: "#58a6ff", fontSize: 10, fontWeight: 600, marginBottom: 2 }}>
            #${t.id} · ${t.class}
          </div>
          <${KV} k="age" v=${t.age + " frames"} />
          ${t.pos?.length > 0 && html`<${KV} k="pos" v=${"[" + t.pos.map(Math.round).join(", ") + "]"} />`}
        </div>`)}`;
  }

  // ── counter ───────────────────────────────────────────────────────────────
  else if (nodeType === "counter") {
    body = html`
      <div style=${{ ...s.card, textAlign: "center" }}>
        <div style=${{ fontSize: 32, fontWeight: 700, color: "#58a6ff", lineHeight: 1.2 }}>${data.count}</div>
        <div style=${{ fontSize: 10, color: "#555d68", marginTop: 2 }}>total count</div>
      </div>`;
  }

  // ── detection nodes (nms / filter / draw_bbox) ───────────────────────────
  else if (["nms", "filter", "draw_bbox", "model_inference"].includes(nodeType)) {
    body = html`
      <div style=${s.card}><${KV} k="Detections" v=${data.detection_count ?? data.raw_shape?.join("×") ?? "—"} /></div>
      ${(data.detections ?? []).map((d, i) => html`
        <div key=${i} style=${s.card}>
          <div style=${{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
            ${d.id !== undefined && html`<span style=${{ color: "#58a6ff", fontSize: 10 }}>#${d.id}</span>`}
            <span style=${{ color: "#e2e8f0", fontWeight: 600 }}>${d.class}</span>
            <span style=${{ color: "#8b949e", fontSize: 10 }}>${(d.conf * 100).toFixed(1)}%</span>
          </div>
          <div style=${s.mono}>${"[" + d.bbox.join(", ") + "]"}</div>
        </div>`)}`;
  }

  // ── corner_detect ─────────────────────────────────────────────────────────
  else if (nodeType === "corner_detect") {
    body = html`
      <div style=${s.card}><${KV} k="Corners found" v=${data.corner_count} /></div>
      ${data.corners?.length > 0 && html`
        <div style=${s.card}>
          <div style=${s.key}>Coordinates (first 12)</div>
          <div style=${s.mono}>${data.corners.map(p => "[" + p.join(",") + "]").join("  ")}</div>
        </div>`}`;
  }

  // ── embedding ─────────────────────────────────────────────────────────────
  else if (nodeType === "embedding") {
    const embs = data.embeddings ?? [];
    body = html`
      <div style=${s.card}><${KV} k="Embeddings" v=${data.embedding_count} /></div>
      ${embs.map((e, i) => html`
        <div key=${i} style=${s.card}>
          <div style=${{ color: "#58a6ff", fontSize: 10, fontWeight: 600, marginBottom: 3 }}>Embedding ${i + 1}</div>
          <${KV} k="shape"   v=${"(" + e.shape.join(", ") + ")"} />
          <${KV} k="L2 norm" v=${e.norm} />
          <div style=${{ marginTop: 3 }}>
            <div style=${s.key}>first 6 dims</div>
            <div style=${s.mono}>${e.preview.map(x => x.toFixed(4)).join("  ")}</div>
          </div>
        </div>`)}`;
  }

  // ── face_db ───────────────────────────────────────────────────────────────
  else if (nodeType === "face_db") {
    body = html`
      <div style=${{ ...s.card, display: "flex", alignItems: "center", gap: 10 }}>
        <div>
          <${KV} k="Match"      v=${data.matched ? "✓ YES" : "✗ NO"} />
          <${KV} k="Similarity" v=${data.similarity !== undefined ? (data.similarity * 100).toFixed(1) + "%" : "—"} />
          ${data.name && html`<${KV} k="Identity" v=${data.name} />`}
        </div>
        ${data.matched !== undefined && html`
          <span style=${{ marginLeft: "auto", ...s.badge(data.matched) }}>
            ${data.matched ? "MATCH" : "NO MATCH"}
          </span>`}
      </div>
      <div style=${s.card}>
        <${KV} k="DB entries" v=${data.db_count ?? "—"} />
        <${KV} k="Threshold"  v=${data.threshold} />
      </div>`;
  }

  // ── crop_bbox ─────────────────────────────────────────────────────────────
  else if (nodeType === "crop_bbox") {
    body = html`
      <div style=${s.card}>
        <${KV} k="Crops produced" v=${data.crop_count} />
        <${KV} k="Output size"    v=${data.image_size + " × " + data.image_size + " px"} />
      </div>
      ${(data.iou_pairs ?? []).length > 0 && html`
        <div style=${s.card}>
          <div style=${{ ...s.key, marginBottom: 5 }}>Pairwise IoU (${data.iou_pairs.length} pair${data.iou_pairs.length > 1 ? "s" : ""})</div>
          ${data.iou_pairs.map((p, i) => html`
            <div key=${i} style=${{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
              <span style=${s.mono}>crop ${p.i} ↔ crop ${p.j}</span>
              <span style=${{ ...s.val, marginLeft: "auto",
                              color: p.iou > 0.5 ? "#f85149" : p.iou > 0.2 ? "#d29922" : "#3fb950" }}>
                ${(p.iou * 100).toFixed(1)}%
              </span>
            </div>`)}
        </div>`}
      ${data.iou_pairs?.length === 0 && data.crop_count > 0 && html`
        <div style=${s.card}>
          <span style=${{ ...s.key }}>No overlap between crops</span>
        </div>`}`;
  }

  if (!body) return null;

  return html`
    <div style=${s.wrap}>
      <div style=${s.title}>Live Data</div>
      ${body}
    </div>`;
}

// ── FACE ENROLL MODAL ─────────────────────────────────────────────────────────
function FaceEnrollModal({ prompt, onClose }) {
  const [name, setName] = useState(prompt.suggested_name ?? "");
  const [busy, setBusy] = useState(false);
  const [err,  setErr]  = useState("");

  const confirm = async () => {
    const n = name.trim();
    if (!n) { setErr("Name is required"); return; }
    setBusy(true);
    try {
      await apiFetch("POST", "/facedb/enroll", { pending_id: prompt.pending_id, name: n });
      onClose();
    } catch (e) { setErr(e.message); setBusy(false); }
  };

  const overlay = {
    position: "fixed", inset: 0, background: "rgba(0,0,0,.7)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 9999,
  };
  const box = {
    background: "#161b22", border: "1px solid #30363d", borderRadius: 10,
    padding: 24, width: 320, fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
    color: "#c9d1d9",
  };

  return html`
    <div style=${overlay} onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${box}>
        <div style=${{ fontWeight: 700, fontSize: 14, marginBottom: 16, color: "#e2e8f0" }}>
          ✍ Enroll Face
        </div>

        ${prompt.crop_b64 && html`
          <div style=${{ textAlign: "center", marginBottom: 14 }}>
            <img src=${"data:image/jpeg;base64," + prompt.crop_b64}
              style=${{ width: 96, height: 96, objectFit: "cover", borderRadius: 8,
                         border: "2px solid #3d1a2d" }} />
          </div>`}

        <div style=${{ marginBottom: 12 }}>
          <label style=${lbl}>Name</label>
          <input style=${inp} value=${name} onInput=${e => setName(e.target.value)}
            onKeyDown=${e => e.key === "Enter" && confirm()} autoFocus />
        </div>

        ${prompt.similarity > 0 && html`
          <div style=${{ fontSize: 11, color: "#8b949e", marginBottom: 12 }}>
            Best match: <b style=${{ color: "#58a6ff" }}>${prompt.suggested_name}</b>
            (similarity ${(prompt.similarity * 100).toFixed(1)}%)
          </div>`}

        ${err && html`<div style=${{ color: "#f85149", fontSize: 11, marginBottom: 8 }}>${err}</div>`}

        <div style=${{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick=${onClose}
            style=${{ background: "none", border: "1px solid #30363d", color: "#8b949e",
                       borderRadius: 6, padding: "5px 14px", cursor: "pointer", fontSize: 12 }}>
            Cancel
          </button>
          <button onClick=${confirm} disabled=${busy}
            style=${{ background: "#3d1a2d", border: "1px solid #db61a2", color: "#db61a2",
                       borderRadius: 6, padding: "5px 14px", cursor: busy ? "not-allowed" : "pointer",
                       fontSize: 12, opacity: busy ? 0.6 : 1 }}>
            ${busy ? "Enrolling…" : "Enroll"}
          </button>
        </div>
      </div>
    </div>`;
}

// ── SEEN FACES PANEL ──────────────────────────────────────────────────────────
function SeenFacesModal({ entries, onClose }) {
  const overlay = {
    position: "fixed", inset: 0, background: "rgba(0,0,0,.75)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 9999,
  };
  const box = {
    background: "#161b22", border: "1px solid #30363d", borderRadius: 10,
    padding: 0, width: 480, maxHeight: "75vh", overflow: "hidden", display: "flex",
    flexDirection: "column",
    fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif", color: "#c9d1d9",
  };

  const sorted = [...entries].reverse(); // newest first

  return html`
    <div style=${overlay} onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${box}>
        <div style=${{ padding: "12px 16px", borderBottom: "1px solid #30363d",
                        fontWeight: 700, fontSize: 13, color: "#e2e8f0",
                        display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span>📋 Seen Faces (${entries.length})</span>
          <button onClick=${onClose}
            style=${{ background: "none", border: "none", color: "#8b949e",
                       cursor: "pointer", fontSize: 16, lineHeight: 1, padding: 0 }}>×</button>
        </div>
        <div style=${{ overflowY: "auto", flex: 1, padding: 12 }}>
          ${sorted.length === 0
            ? html`<div style=${{ textAlign: "center", color: "#555d68", padding: 24, fontSize: 12 }}>No faces seen yet</div>`
            : sorted.map((e, i) => html`
                <div key=${i} style=${{ display: "flex", gap: 10, padding: "8px 0",
                                         borderBottom: "1px solid #21262d", alignItems: "center" }}>
                  <div style=${{ width: 48, height: 48, borderRadius: 6,
                                  background: "#21262d", flexShrink: 0,
                                  display: "flex", alignItems: "center", justifyContent: "center",
                                  fontSize: 20 }}>
                    👤
                  </div>
                  <div style=${{ flex: 1, minWidth: 0 }}>
                    <div style=${{ fontWeight: 600, fontSize: 12, color: e.in_db ? "#e2e8f0" : "#8b949e" }}>
                      ${e.name}
                      ${e.in_db ? html`<span style=${{ color: "#3fb950", fontSize: 10 }}> ✓</span>`
                                : html`<span style=${{ color: "#d29922", fontSize: 10 }}> ?</span>`}
                    </div>
                    <div style=${{ fontSize: 10, color: "#555d68" }}>${e.timestamp}</div>
                    <div style=${{ fontSize: 10, color: "#8b949e" }}>
                      sim: ${((e.similarity ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>`)}
        </div>
      </div>
    </div>`;
}

// ── CUSTOM NODES HELPER ───────────────────────────────────────────────────────
function loadCustomNodes() {
  try { return JSON.parse(localStorage.getItem("cvflow_custom_nodes") || "[]"); }
  catch { return []; }
}

// ── OPEN PIPELINE MODAL ───────────────────────────────────────────────────────
function OpenModal({ onLoad, onClose }) {
  const [list,    setList]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(null);

  useEffect(() => {
    apiFetch("GET", "/pipelines")
      .then(data => setList(data))
      .catch(() => setList([]))
      .finally(() => setLoading(false));
  }, []);

  const handleLoad = async (item) => {
    try {
      const p = await apiFetch("GET", "/pipelines/" + item.id);
      onLoad(p);
    } catch (e) { alert("Failed to load pipeline: " + e.message); }
  };

  const handleDelete = async (item, e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete "${item.name}"?`)) return;
    setDeleting(item.id);
    try {
      await apiFetch("DELETE", "/pipelines/" + item.id);
      setList(l => l.filter(x => x.id !== item.id));
    } catch (err) { alert("Delete failed: " + err.message); }
    setDeleting(null);
  };

  const fmt = (iso) => {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,.7)",
                    display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
                      width: 480, maxHeight: "70vh", display: "flex", flexDirection: "column",
                      boxShadow: "0 16px 48px rgba(0,0,0,.8)" }}>
        <!-- header -->
        <div style=${{ display: "flex", alignItems: "center", padding: "14px 18px",
                        borderBottom: "1px solid #30363d" }}>
          <span style=${{ fontWeight: 700, fontSize: 15, color: "#e2e8f0", flex: 1 }}>📂 Open Pipeline</span>
          <button onClick=${onClose} style=${{ background: "none", border: "none", color: "#8b949e",
                                               cursor: "pointer", fontSize: 18 }}>✕</button>
        </div>
        <!-- list -->
        <div style=${{ overflowY: "auto", flex: 1 }}>
          ${loading && html`<div style=${{ padding: 24, color: "#8b949e", textAlign: "center" }}>Loading…</div>`}
          ${!loading && list.length === 0 && html`
            <div style=${{ padding: 24, color: "#8b949e", textAlign: "center" }}>
              No saved pipelines yet.<br/>
              <span style=${{ fontSize: 12 }}>Use <b>Save</b> in the toolbar to save your first pipeline.</span>
            </div>`}
          ${list.map((item, i) => html`
            <div key=${item.id} onClick=${() => handleLoad(item)}
              style=${{ padding: "12px 18px", cursor: "pointer", display: "flex", alignItems: "center", gap: 12,
                          borderBottom: i < list.length - 1 ? "1px solid #21262d" : "none" }}
              onMouseEnter=${e => e.currentTarget.style.background = "#21262d"}
              onMouseLeave=${e => e.currentTarget.style.background = ""}>
              <div style=${{ flex: 1, minWidth: 0 }}>
                <div style=${{ fontWeight: 600, fontSize: 13, color: "#e2e8f0",
                                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  ${item.name || "Untitled"}
                </div>
                <div style=${{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>
                  Saved ${fmt(item.updated_at)}
                </div>
              </div>
              <button onClick=${(e) => handleDelete(item, e)}
                disabled=${deleting === item.id}
                style=${{ background: "none", border: "1px solid #f8514933", color: "#f85149", borderRadius: 5,
                            fontSize: 11, padding: "3px 8px", cursor: "pointer", flexShrink: 0 }}
                onMouseEnter=${e => e.currentTarget.style.background = "#f8514922"}
                onMouseLeave=${e => e.currentTarget.style.background = "none"}>
                ${deleting === item.id ? "…" : "Delete"}
              </button>
            </div>`)}
        </div>
      </div>
    </div>`;
}

// ── MAIN APP ──────────────────────────────────────────────────────────────────
let _id = 0;
const uid = () => ++_id;

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selId, setSelId]     = useState(null);
  const [name,  setName]      = useState("Untitled Pipeline");
  const [dirty, setDirty]     = useState(false);
  const [pipelineId, setPid]  = useState(null);
  const [sessionId, setSid]   = useState(null);
  const [running, setRunning] = useState(false);
  const [enrollPrompt, setEnrollPrompt] = useState(null); // face_enroll_prompt payload
  const [seenFaces,   setSeenFaces]     = useState([]);   // recent face_seen entries
  const [showSeenLog, setShowSeenLog]   = useState(false);
  const [nodeDataMap, setNodeDataMap]   = useState({});   // node_id → JSON metadata snapshot
  const [nodeVizMap,  setNodeVizMap]    = useState({});   // node_id → { items: [{type,b64,data,...}] }
  const [codeEditId,  setCodeEditId]    = useState(null); // node id whose code editor overlay is open

  const [showOpen,        setShowOpen]         = useState(false);
  const [showSamples,     setShowSamples]     = useState(false);
  const [showModels,      setShowModels]       = useState(false);
  const [showCustomNode,  setShowCustomNode]   = useState(false);
  const [showShortcuts,   setShowShortcuts]    = useState(false);
  const [showQuickAdd,    setShowQuickAdd]     = useState(false);
  const [saveNodeTarget,  setSaveNodeTarget]   = useState(null); // node to save to library
  const [contextMenu,     setContextMenu]      = useState(null); // {x,y,node}
  const [validateWarns,   setValidateWarns]    = useState(null); // warnings[]
  const [toast,           setToast]            = useState(null);

  const [rightW, setRightW]   = useState(280);
  const [logH,   setLogH]     = useState(160);
  const resizerWRef  = useRef(false);
  const resizerLHRef = useRef(false);

  const [groups, setGroups] = useState(() => {
    const customs = loadCustomNodes();
    let grps = BASE_GROUPS.map(g => ({ ...g, types: [...g.types] }));
    for (const cn of customs) {
      registerNodeType(cn.type, cn.meta, cn.ports);
      if (cn.config) DEFAULT_CONFIG[cn.type] = cn.config;
      const gi = grps.findIndex(g => g.label === (cn.meta.group_label || "Custom"));
      if (gi >= 0) grps[gi] = { ...grps[gi], types: [...grps[gi].types, cn.type] };
      else grps = [...grps, { label: cn.meta.group_label || "Custom", types: [cn.type] }];
    }
    return grps;
  });

  const evWsRef   = useRef(null);
  const importRef = useRef(null);
  const { project, fitView } = useReactFlow();

  // Panel resize handlers
  useEffect(() => {
    const onMove = e => {
      if (resizerWRef.current) {
        const newW = window.innerWidth - e.clientX;
        setRightW(Math.max(220, Math.min(600, newW)));
      }
      if (resizerLHRef.current) {
        const panelEl = resizerLHRef.current;
        const rect = panelEl.getBoundingClientRect();
        const newH = rect.bottom - e.clientY;
        setLogH(Math.max(60, Math.min(400, newH)));
      }
    };
    const onUp = () => {
      resizerWRef.current = false;
      resizerLHRef.current = false;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  const { snapshot, undo, redo } = useHistory(nodes, edges, setNodes, setEdges);

  const selNode = nodes.find(n => n.id === selId) ?? null;

  const showToast = useCallback((msg, ok = true) => {
    setToast({ msg, ok }); setTimeout(() => setToast(null), 3000);
  }, []);

  // Poll engine status while running — auto-stop + toast if the engine crashes
  useEffect(() => {
    if (!running || !sessionId) return;
    const t = setInterval(async () => {
      try {
        const r = await apiFetch("GET", "/execution/status/" + sessionId);
        if (r.status === "error") {
          setRunning(false); setSid(null);
          if (evWsRef.current) { evWsRef.current.close(); evWsRef.current = null; }
          showToast("Engine crashed — check camera / console", false);
        } else if (r.status === "completed") {
          setRunning(false); setSid(null);
          showToast("Pipeline finished");
        }
      } catch {}
    }, 3000);
    return () => clearInterval(t);
  }, [running, sessionId]);

  // Dirty tracking
  useEffect(() => { setDirty(true); }, [nodes, edges]);

  // Auto-save draft every 30s
  useEffect(() => {
    const t = setInterval(() => saveDraft(name, nodes, edges), 30000);
    return () => clearInterval(t);
  }, [name, nodes, edges]);
  useEffect(() => {
    const handler = () => saveDraft(name, nodes, edges);
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [name, nodes, edges]);

  // Draft restore prompt on first load
  useEffect(() => {
    const draft = loadDraft();
    if (draft && draft.nodes?.length > 0) {
      const age = Math.round((Date.now() - draft.ts) / 60000);
      if (confirm(`Restore unsaved draft "${draft.name}" (${age}m ago) with ${draft.nodes.length} nodes?`)) {
        setNodes(draft.nodes); setEdges(draft.edges); setName(draft.name);
      }
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") {
        if (e.key === "Escape") e.target.blur();
        return;
      }
      const ctrl = e.ctrlKey || e.metaKey;
      if (ctrl && e.key === "s") { e.preventDefault(); save(); }
      else if (ctrl && e.key === "z") { e.preventDefault(); snapshot(); undo(); }
      else if (ctrl && (e.key === "y" || (e.shiftKey && e.key === "Z"))) { e.preventDefault(); redo(); }
      else if (ctrl && e.key === "d") { e.preventDefault(); if (selId) duplicateNode(selId); }
      else if (ctrl && e.key === "k") { e.preventDefault(); setShowQuickAdd(true); }
      else if (ctrl && e.shiftKey && e.key === "L") { e.preventDefault(); handleAutoLayout(); }
      else if (ctrl && e.key === "e") { e.preventDefault(); handleExport(); }
      else if (e.key === "?" && !ctrl) { setShowShortcuts(true); }
      else if (e.key === "Escape") {
        setShowSamples(false); setShowModels(false); setShowCustomNode(false);
        setShowShortcuts(false); setShowQuickAdd(false); setContextMenu(null); setValidateWarns(null);
        setEnrollPrompt(null); setShowSeenLog(false); setCodeEditId(null);
        setSelId(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selId, nodes, edges, name]);

  const onConnect = useCallback(p => {
    snapshot();
    // Auto-sync counter.trigger_id when draw_line.line_ref → counter.line_ref
    if (p.sourceHandle === "line_ref" && p.targetHandle === "line_ref") {
      const src = nodes.find(n => n.id === p.source);
      const tgt = nodes.find(n => n.id === p.target);
      if (src?.type === "draw_line" && tgt?.type === "counter") {
        const lineId = src.data?.config?.line_id ?? "line_1";
        setNodes(nds => nds.map(n => n.id === tgt.id
          ? { ...n, data: { ...n.data, config: { ...n.data.config, trigger_id: lineId } } } : n));
      }
    }
    // Auto-sync counter.trigger_id when draw_roi.out → counter.frame (zone triggers)
    if (p.sourceHandle === "out" && p.targetHandle === "frame") {
      const src = nodes.find(n => n.id === p.source);
      const tgt = nodes.find(n => n.id === p.target);
      if (src?.type === "draw_roi" && tgt?.type === "counter") {
        const zoneId = src.data?.config?.zone_id ?? "zone_1";
        setNodes(nds => nds.map(n => n.id === tgt.id
          ? { ...n, data: { ...n.data, config: { ...n.data.config, trigger_id: zoneId } } } : n));
      }
    }
    setEdges(es => addEdge({ ...p }, es));
    setDirty(true);
  }, [snapshot, nodes, setNodes, setEdges]);
  const onDragOver   = useCallback(e => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }, []);
  const onDrop       = useCallback(e => {
    e.preventDefault();
    const type = e.dataTransfer.getData("application/cvflow");
    if (!type) return;
    const pos  = project({ x: e.clientX - 172, y: e.clientY - 44 });
    const meta = NODE_META[type];
    snapshot();
    setNodes(ns => [...ns, {
      id: `${type}_${uid()}`, type, position: pos,
      data: { label: meta?.label ?? type, config: { ...(DEFAULT_CONFIG[type] ?? {}) } },
    }]);
    setDirty(true);
    addRecent(type);
  }, [project, setNodes, snapshot]);

  const onNodeClick       = useCallback((_, n) => setSelId(n.id), []);
  const onPaneClick       = useCallback(() => { setSelId(null); setContextMenu(null); }, []);
  const onNodeContextMenu = useCallback((e, n) => {
    e.preventDefault(); e.stopPropagation();
    setContextMenu({ x: e.clientX, y: e.clientY, node: n });
    setSelId(n.id);
  }, []);

  const onUpdate = useCallback((id, newCfg, newLabel, newPorts) => {
    if (newCfg === null) {
      snapshot();
      setNodes(ns => ns.filter(n => n.id !== id));
      setEdges(es => es.filter(e => e.source !== id && e.target !== id));
      setSelId(null);
      setDirty(true);
      return;
    }
    snapshot();
    setNodes(ns => ns.map(n => n.id !== id ? n : {
      ...n, data: {
        ...n.data,
        config: newCfg,
        ...(newLabel !== undefined ? { label: newLabel } : {}),
        ...(newPorts !== undefined ? { ports: newPorts } : {}),
      }
    }));
    setDirty(true);
  }, [snapshot, setNodes, setEdges]);

  const duplicateNode = useCallback((id) => {
    const n = nodes.find(n => n.id === id);
    if (!n) return;
    snapshot();
    const newId = `${n.type}_${uid()}`;
    setNodes(ns => [...ns, { ...n, id: newId, position: { x: n.position.x + 30, y: n.position.y + 30 }, selected: false }]);
    setDirty(true);
    showToast("Duplicated: " + n.data.label);
  }, [nodes, snapshot, setNodes, showToast]);

  const resetNodeConfig = useCallback((id) => {
    const n = nodes.find(n => n.id === id);
    if (!n) return;
    const def = DEFAULT_CONFIG[n.type];
    if (!def) return;
    snapshot();
    setNodes(ns => ns.map(x => x.id !== id ? x : { ...x, data: { ...x.data, config: { ...def } } }));
    setDirty(true);
    showToast("Config reset to defaults");
  }, [nodes, snapshot, setNodes, showToast]);

  const handleNew = useCallback(() => {
    if (dirty && nodes.length > 0) {
      if (!window.confirm("Discard unsaved changes and create a new pipeline?")) return;
    }
    _id = 0; snapshot();
    setNodes([]); setEdges([]);
    setName("Untitled"); setPid(null); setSelId(null); setDirty(false);
    showToast("New pipeline");
  }, [dirty, nodes, snapshot, setNodes, setEdges, showToast]);

  const handleAutoLayout = useCallback(() => {
    snapshot();
    setNodes(ns => autoLayout(ns, edges));
    setTimeout(() => fitView({ padding: 0.15, duration: 400 }), 50);
    setDirty(true);
    showToast("Auto-layout applied");
  }, [nodes, edges, snapshot, setNodes, fitView, showToast]);

  const loadSample = (s) => {
    _id = 0; snapshot();
    setNodes(s.nodes); setEdges(s.edges); setName(s.name);
    setPid(null); setSelId(null); setShowSamples(false); setDirty(true);
    setTimeout(() => fitView({ padding: 0.15, duration: 400 }), 50);
    showToast("Loaded: " + s.name);
  };

  const _toRFNodes = (raw) => (raw ?? []).map(n => ({
    id: n.id, type: n.type,
    position: n.position ?? { x: 0, y: 0 },
    data: n.data ?? { label: n.label, config: n.config ?? {} },
  }));
  const _toRFEdges = (raw) => (raw ?? []).map(e => ({
    id: e.id, source: e.source, target: e.target,
    sourceHandle: e.sourceHandle ?? "out",
    targetHandle: e.targetHandle ?? "in",
  }));

  const loadPipeline = (p) => {
    _id = 0; snapshot();
    setNodes(_toRFNodes(p.nodes)); setEdges(_toRFEdges(p.edges));
    setName(p.name ?? "Untitled"); setPid(p.id); setSelId(null); setDirty(false);
    setShowOpen(false);
    setTimeout(() => fitView({ padding: 0.15, duration: 400 }), 50);
    showToast("Opened: " + (p.name || "Untitled"));
  };

  const handleExport = () => {
    const payload = {
      version: "1.0", name,
      nodes: nodes.map(n => ({ id: n.id, type: n.type, label: n.data.label, position: n.position, config: n.data.config ?? {} })),
      edges: edges.map(e => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle ?? "out", targetHandle: e.targetHandle ?? "in" })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (name || "pipeline").replace(/\s+/g, "_") + ".json";
    a.click();
    URL.revokeObjectURL(a.href);
    showToast("Exported JSON");
  };

  const handleImport = (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        snapshot();
        setNodes(_toRFNodes(data.nodes)); setEdges(_toRFEdges(data.edges));
        setName(data.name ?? file.name.replace(/\.json$/, ""));
        setPid(null); setSelId(null); setDirty(true);
        setTimeout(() => fitView({ padding: 0.15, duration: 400 }), 50);
        showToast("Imported: " + (data.name ?? file.name));
      } catch (err) { showToast("Import failed: " + err.message, false); }
      e.target.value = "";
    };
    reader.readAsText(file);
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
      setDirty(false);
      saveDraft(name, nodes, edges);
      showToast(pipelineId ? "Saved ✓" : "Created ✓");
      return result.id ?? pipelineId;
    } catch (e) { showToast("Save failed: " + e.message, false); return null; }
  };

  const startRun = async () => {
    let pid = pipelineId ?? await save();
    if (!pid) return;
    try {
      const r = await apiFetch("POST", "/execution/start", { pipeline_id: pid });
      setSid(r.session_id); setRunning(true); setNodeDataMap({}); setNodeVizMap({});
      showToast("Started");
      if (evWsRef.current) evWsRef.current.close();
      const ws = new WebSocket(`ws://localhost:8765/ws/events/${r.session_id}`);
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "face_enroll_prompt")
            setEnrollPrompt(msg);
          else if (msg.type === "face_enrolled")
            showToast("Enrolled: " + msg.name);
          else if (msg.type === "face_gate_event")
            showToast((msg.allowed ? "✓ Allowed: " : "✗ Blocked: ") + msg.identity, msg.allowed);
          else if (msg.type === "face_seen_update")
            setSeenFaces(msg.entries ?? []);
          else if (msg.type === "face_seen")
            showToast("Seen: " + msg.name + (msg.in_db ? "" : " (unknown)"));
          else if (msg.type === "node_data")
            setNodeDataMap(m => ({ ...m, [msg.node_id]: { node_type: msg.node_type, data: msg.data } }));
          else if (msg.type === "node_viz")
            setNodeVizMap(m => ({ ...m, [msg.node_id]: { items: msg.items } }));
        } catch {}
      };
      evWsRef.current = ws;
    } catch (e) { showToast("Start failed: " + e.message, false); }
  };

  const run = async () => {
    const warns = validatePipeline(nodes, edges);
    if (warns.length > 0) { setValidateWarns(warns); return; }
    await startRun();
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

  const quickAddNode = (type) => {
    const meta = NODE_META[type];
    setNodes(ns => [...ns, {
      id: `${type}_${uid()}`, type,
      position: { x: 300 + Math.random() * 200, y: 200 + Math.random() * 100 },
      data: { label: meta?.label ?? type, config: { ...(DEFAULT_CONFIG[type] ?? {}) } },
    }]);
    addRecent(type);
    setDirty(true);
  };

  const saveCustomNode = (cn) => {
    registerNodeType(cn.type, cn.meta, cn.ports);
    if (cn.config) DEFAULT_CONFIG[cn.type] = cn.config;
    const existing = loadCustomNodes();
    localStorage.setItem("cvflow_custom_nodes", JSON.stringify([...existing.filter(c => c.type !== cn.type), cn]));
    setGroups(prev => {
      const label = cn.meta.group_label || "Custom";
      const gi = prev.findIndex(g => g.label === label);
      if (gi >= 0) {
        const updated = [...prev];
        if (!updated[gi].types.includes(cn.type))
          updated[gi] = { ...updated[gi], types: [...updated[gi].types, cn.type] };
        return updated;
      }
      return [...prev, { label, types: [cn.type] }];
    });
    showToast("Saved: " + cn.meta.label);
  };

  const deleteCustomNode = (type) => {
    localStorage.setItem("cvflow_custom_nodes",
      JSON.stringify(loadCustomNodes().filter(c => c.type !== type)));
    try {
      const prev = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
      localStorage.setItem(RECENT_KEY, JSON.stringify(prev.filter(t => t !== type)));
    } catch {}
    setGroups(prev => prev
      .map(g => ({ ...g, types: g.types.filter(t => t !== type) }))
      .filter(g => g.types.length > 0 || BASE_GROUPS.some(bg => bg.label === g.label))
    );
    showToast("Đã xóa template");
  };

  const btn = (variant, disabled) => ({
    padding: "4px 12px", borderRadius: 6, cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 12, fontWeight: 500, opacity: disabled ? .45 : 1,
    border: "1px solid", fontFamily: "inherit",
    ...({
      run:  { background: "#1a3d2e", borderColor: "#3fb950", color: "#3fb950" },
      stop: { background: "#3d1a1a", borderColor: "#f85149", color: "#f85149" },
      save: { background: "#1f3a5e", borderColor: "#58a6ff", color: "#58a6ff" },
      def:  { background: "#21262d", borderColor: "#30363d", color: "#c9d1d9" },
    }[variant] ?? {}),
  });

  const divider = html`<div style=${{ width: 1, height: 20, background: "#30363d", margin: "0 2px" }} />`;

  return html`
    <div style=${{ display: "flex", flexDirection: "column", height: "100vh",
                   background: "#0d1117", color: "#c9d1d9", fontSize: 13 }}>

      <!-- TOOLBAR -->
      <div style=${{ display: "flex", alignItems: "center", gap: 6, padding: "0 10px",
                      height: 44, background: "#161b22", borderBottom: "1px solid #30363d", flexShrink: 0 }}>
        <span style=${{ fontSize: 15, fontWeight: 800, color: "#58a6ff", letterSpacing: 1.5, marginRight: 6 }}>CV-FLOW</span>

        <button style=${btn("def")} onClick=${handleNew} title="New pipeline">+ New</button>
        <button style=${btn("def")} onClick=${() => setShowOpen(true)} title="Open saved pipeline">📂 Open</button>

        <!-- Samples dropdown -->
        <div style=${{ position: "relative" }}>
          <button style=${btn("def")} onClick=${() => setShowSamples(s => !s)}>Samples ▾</button>
          ${showSamples && html`
            <div style=${{ position: "absolute", top: "110%", left: 0, zIndex: 200,
                            background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
                            minWidth: 300, maxHeight: "70vh", overflowY: "auto",
                            boxShadow: "0 8px 24px rgba(0,0,0,.6)" }}>
              ${SAMPLES.map((s, i) => html`
                <div key=${i} onClick=${() => { loadSample(s); setShowSamples(false); }}
                  style=${{ padding: "9px 14px", cursor: "pointer",
                              borderBottom: i < SAMPLES.length - 1 ? "1px solid #30363d" : "none" }}
                  onMouseEnter=${e => e.currentTarget.style.background = "#21262d"}
                  onMouseLeave=${e => e.currentTarget.style.background = ""}>
                  <div style=${{ fontWeight: 600, fontSize: 12, color: "#e2e8f0" }}>${s.name}</div>
                  <div style=${{ fontSize: 11, color: "#8b949e", marginTop: 2, lineHeight: 1.4 }}>${s.description}</div>
                </div>`)}
            </div>`}
        </div>

        <!-- Pipeline name + dirty indicator -->
        <div style=${{ display: "flex", alignItems: "center", gap: 4 }}>
          ${dirty && html`<span title="Unsaved changes" style=${{ color: "#d29922", fontSize: 14, lineHeight: 1 }}>●</span>`}
          <input value=${name} onChange=${e => { setName(e.target.value); setDirty(true); }}
            style=${{ background: "transparent", border: "1px solid transparent", borderRadius: 5,
                       color: "#e2e8f0", fontSize: 13, padding: "3px 8px", outline: "none",
                       minWidth: 140, maxWidth: 240, fontFamily: "inherit" }}
            onFocus=${e => e.target.style.borderColor = "#30363d"}
            onBlur=${e  => e.target.style.borderColor = "transparent"} />
        </div>

        ${divider}

        <button style=${btn("save")} onClick=${save} title="Save (Ctrl+S)">Save</button>
        <button style=${btn("run",  running)} onClick=${run}  disabled=${running} title="Run pipeline">▶ Run</button>
        <button style=${btn("stop", !running)} onClick=${stop} disabled=${!running} title="Stop pipeline">■ Stop</button>

        ${divider}

        <button style=${btn("def")} onClick=${() => setShowModels(true)} title="Model Hub">🧠 Models</button>
        ${running && html`
          <button style=${{ ...btn("def"), position: "relative" }} onClick=${() => setShowSeenLog(true)} title="Seen Faces log">
            📋 Faces
            ${seenFaces.length > 0 && html`<span style=${{ position: "absolute", top: -4, right: -4, background: "#db61a2",
              color: "#fff", borderRadius: "50%", fontSize: 9, width: 14, height: 14, lineHeight: "14px",
              textAlign: "center", display: "inline-block" }}>${seenFaces.length > 99 ? "99+" : seenFaces.length}</span>`}
          </button>`}
        <button style=${btn("def")} onClick=${handleExport} title="Export JSON (Ctrl+E)">↓ Export</button>
        <button style=${btn("def")} onClick=${() => importRef.current?.click()} title="Import JSON">↑ Import</button>
        <input ref=${importRef} type="file" accept=".json" onChange=${handleImport} style=${{ display: "none" }} />

        ${divider}

        <button style=${{ ...btn("def"), padding: "4px 8px" }} onClick=${handleAutoLayout} title="Auto-layout (Ctrl+Shift+L)">⬡ Layout</button>
        <button style=${{ ...btn("def"), padding: "4px 8px" }} onClick=${() => fitView({ padding: 0.12, duration: 400 })} title="Fit view">⊞</button>

        <!-- Stats -->
        <span style=${{ marginLeft: 4, fontSize: 11, color: "#555d68" }}>
          ${nodes.length} nodes · ${edges.length} edges
        </span>

        <div style=${{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <div style=${{ display: "flex", alignItems: "center", gap: 5, fontSize: 11 }}>
            <div style=${{ width: 7, height: 7, borderRadius: "50%",
                            background: running ? "#3fb950" : "#8b949e",
                            animation: running ? "blink 1.8s infinite" : "none" }} />
            <span style=${{ color: running ? "#3fb950" : "#8b949e" }}>
              ${running ? (sessionId?.slice(0, 8) + "…") : "Idle"}
            </span>
          </div>
          <button onClick=${() => setShowShortcuts(true)} title="Keyboard shortcuts (?)"
            style=${{ background: "#21262d", border: "1px solid #30363d", color: "#8b949e",
                       borderRadius: "50%", width: 22, height: 22, cursor: "pointer",
                       fontSize: 11, display: "flex", alignItems: "center", justifyContent: "center",
                       padding: 0 }}>
            ?
          </button>
        </div>
      </div>

      <!-- BODY -->
      <div style=${{ display: "flex", flex: 1, overflow: "hidden" }}>
        <${Palette} groups=${groups} onAddCustom=${() => setShowCustomNode(true)} onDeleteTemplate=${deleteCustomNode} />

        <div style=${{ flex: 1, position: "relative" }}>
          <${ReactFlow}
            nodes=${nodes} edges=${edges}
            onNodesChange=${(chg) => { onNodesChange(chg); }}
            onEdgesChange=${onEdgesChange}
            onConnect=${onConnect}
            nodeTypes=${nodeTypes}
            onDrop=${onDrop} onDragOver=${onDragOver}
            onNodeClick=${onNodeClick}
            onPaneClick=${onPaneClick}
            onNodeContextMenu=${onNodeContextMenu}
            fitView deleteKeyCode="Delete"
            defaultEdgeOptions=${{ style: { stroke: "#58a6ff", strokeWidth: 2 }, animated: false }}>
            <${Background} color="#21262d" gap=${24} size=${1} />
            <${Controls} />
            <${MiniMap}
              nodeColor=${n => GROUP_COLOR[NODE_META[n.type]?.group] ?? "#21262d"}
              maskColor="rgba(0,0,0,.6)" />
          <//>
          ${nodes.length === 0 && html`
            <div style=${{ position: "absolute", inset: 0, display: "flex", alignItems: "center",
                            justifyContent: "center", pointerEvents: "none" }}>
              <div style=${{ textAlign: "center", color: "#8b949e" }}>
                <div style=${{ fontSize: 32, marginBottom: 10 }}>◻</div>
                <div style=${{ fontSize: 14, marginBottom: 6 }}>Drag nodes from the palette</div>
                <div style=${{ fontSize: 12, marginBottom: 4 }}>or click <b style=${{ color: "#58a6ff" }}>Samples ▾</b> to load a pipeline</div>
                <div style=${{ fontSize: 11, color: "#555d68" }}>Ctrl+K to quick-add a node</div>
              </div>
            </div>`}
        </div>

        <!-- PANEL RESIZE HANDLE (drag to widen/narrow right panel) -->
        <div
          onMouseDown=${() => { resizerWRef.current = true; }}
          style=${{ width: 5, cursor: "col-resize", background: "transparent", flexShrink: 0,
                     transition: "background .15s", zIndex: 10 }}
          onMouseEnter=${e => e.currentTarget.style.background = "#58a6ff55"}
          onMouseLeave=${e => e.currentTarget.style.background = "transparent"}
        />

        <!-- RIGHT PANEL -->
        <div style=${{ width: rightW, flexShrink: 0, borderLeft: "1px solid #30363d",
                        display: "flex", flexDirection: "column", background: "#0d1117" }}>
          <div style=${{ padding: "7px 14px", borderBottom: "1px solid #30363d",
                          fontSize: 10, textTransform: "uppercase", letterSpacing: ".5px", color: "#8b949e",
                          display: "flex", alignItems: "center" }}>
            <span style=${{ flex: 1 }}>Properties</span>
            <span style=${{ fontSize: 9, color: "#555d68" }}>${rightW}px ↔</span>
          </div>
          <div style=${{ flex: 1, overflowY: "auto" }}>
            <${PropertiesPanel} node=${selNode} onUpdate=${onUpdate} onDuplicate=${duplicateNode}
              sessionId=${running ? sessionId : null} running=${running}
              nodeDataMap=${nodeDataMap} nodeVizMap=${nodeVizMap}
              onEditCode=${() => selNode?.type === "python_node" && setCodeEditId(selNode.id)}
              onSaveNode=${() => selNode?.type === "python_node" && setSaveNodeTarget(selNode)} />
          </div>

          <!-- Engine log resize handle -->
          <div
            onMouseDown=${e => { resizerLHRef.current = e.currentTarget.nextElementSibling; }}
            style=${{ height: 5, cursor: "row-resize", background: "transparent", flexShrink: 0,
                       transition: "background .15s" }}
            onMouseEnter=${e => e.currentTarget.style.background = "#58a6ff55"}
            onMouseLeave=${e => e.currentTarget.style.background = "transparent"}
          />

          <${EngineLogViewer} sessionId=${sessionId} height=${logH} />
        </div>
      </div>

      <!-- MODALS -->
      ${showOpen       && html`<${OpenModal}        onLoad=${loadPipeline} onClose=${() => setShowOpen(false)} />`}
      ${showModels     && html`<${ModelHubModal}    onClose=${() => setShowModels(false)} />`}
      ${showCustomNode && html`<${CustomNodeModal}  onClose=${() => setShowCustomNode(false)} onSave=${saveCustomNode} currentNodes=${nodes} currentEdges=${edges} />`}
      ${showShortcuts  && html`<${ShortcutsModal}   onClose=${() => setShowShortcuts(false)} />`}
      ${showQuickAdd   && html`<${QuickAddModal}    onClose=${() => setShowQuickAdd(false)} onAdd=${quickAddNode} />`}
      ${enrollPrompt   && html`<${FaceEnrollModal}  prompt=${enrollPrompt} onClose=${() => setEnrollPrompt(null)} />`}
      ${showSeenLog    && html`<${SeenFacesModal}   entries=${seenFaces} onClose=${() => setShowSeenLog(false)} />`}
      ${validateWarns  && html`<${ValidateModal}    warnings=${validateWarns}
                                  onClose=${() => setValidateWarns(null)}
                                  onRunAnyway=${() => { setValidateWarns(null); startRun(); }} />`}

      <!-- CONTEXT MENU -->
      ${contextMenu && html`
        <${ContextMenu}
          x=${contextMenu.x} y=${contextMenu.y} node=${contextMenu.node}
          onClose=${() => setContextMenu(null)}
          onDuplicate=${() => duplicateNode(contextMenu.node.id)}
          onDelete=${() => { onUpdate(contextMenu.node.id, null); }}
          onResetConfig=${() => resetNodeConfig(contextMenu.node.id)}
          onEditCode=${() => { setCodeEditId(contextMenu.node.id); setContextMenu(null); }}
          onSaveNode=${() => { setSaveNodeTarget(contextMenu.node); setContextMenu(null); }} />`}

      ${saveNodeTarget && html`
        <${SaveNodeModal}
          node=${saveNodeTarget}
          onClose=${() => setSaveNodeTarget(null)}
          onSave=${saveCustomNode} />`}

      <!-- CODE EDITOR OVERLAY -->
      ${codeEditId && (() => {
        const codeNode = nodes.find(n => n.id === codeEditId);
        return codeNode ? html`
          <${CodeEditorOverlay}
            node=${codeNode}
            onClose=${() => setCodeEditId(null)}
            onUpdate=${onUpdate} />` : null;
      })()}

      <!-- TOAST -->
      ${toast && html`
        <div style=${{ position: "fixed", bottom: 20, right: 20, zIndex: 9999,
                        padding: "10px 16px", borderRadius: 8, fontSize: 12,
                        background: toast.ok ? "#1a3d2e" : "#3d1a1a",
                        border: "1px solid " + (toast.ok ? "#3fb950" : "#f85149"),
                        color: toast.ok ? "#3fb950" : "#f85149",
                        boxShadow: "0 4px 16px rgba(0,0,0,.5)", pointerEvents: "none",
                        transition: "opacity .3s" }}>
          ${toast.msg}
        </div>`}
    </div>`;
}

// ── BOOT ──────────────────────────────────────────────────────────────────────
createRoot(document.getElementById("root")).render(html`
  <${ReactFlowProvider}><${App} /><//>
`);
