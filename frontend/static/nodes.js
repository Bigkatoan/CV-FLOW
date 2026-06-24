import { createElement } from "react";
import { Handle, Position } from "reactflow";
import htm from "https://esm.sh/htm@3";
const html = htm.bind(createElement);

export const GROUP_COLOR = {
  input:      "#1e3a5f",
  processing: "#1a3d2e",
  vision:     "#1a2d3d",
  spatial:    "#3d2e0a",
  utility:    "#2d1a4a",
  cpp:        "#0a2d3d",
  output:     "#3d1a1a",
};

export const NODE_META = {
  // Input
  camera:           { group: "input",      icon: "📷", label: "Camera" },
  usb_camera:       { group: "input",      icon: "📹", label: "USB Camera" },
  video_file:       { group: "input",      icon: "🎞️", label: "Video File" },
  image_directory:  { group: "input",      icon: "🗂️", label: "Image Dir" },
  rtsp_stream:      { group: "input",      icon: "🌐", label: "RTSP Stream" },
  // Processing (ML)
  preprocess:       { group: "processing", icon: "⚙️",  label: "Preprocess" },
  model_inference:  { group: "processing", icon: "🧠", label: "Model" },
  nms:              { group: "processing", icon: "🔲", label: "NMS Filter" },
  // Vision (OpenCV)
  blur:             { group: "vision",     icon: "💧", label: "Blur" },
  edge_detect:      { group: "vision",     icon: "▣",  label: "Edge Detect" },
  corner_detect:    { group: "vision",     icon: "✦",  label: "Corner Detect" },
  threshold:        { group: "vision",     icon: "◑",  label: "Threshold" },
  color_convert:    { group: "vision",     icon: "🎨", label: "Color Convert" },
  morph:            { group: "vision",     icon: "◈",  label: "Morphology" },
  resize:           { group: "vision",     icon: "⤢",  label: "Resize" },
  affine_transform: { group: "vision",     icon: "⤡",  label: "Affine" },
  // Spatial
  draw_roi:         { group: "spatial",    icon: "⬡",  label: "ROI Zone" },
  draw_line:        { group: "spatial",    icon: "↕",  label: "Trip Line" },
  object_tracker:   { group: "spatial",    icon: "🎯", label: "Tracker" },
  counter:          { group: "spatial",    icon: "🔢", label: "Counter" },
  // Utility
  python_function:  { group: "utility",    icon: "🐍", label: "Python Fn" },
  filter:           { group: "utility",    icon: "🔍", label: "Filter" },
  param:            { group: "utility",    icon: "🔧", label: "Param" },
  // C++
  cpp_function:     { group: "cpp",        icon: "⚡", label: "C++ Node" },
  // Output
  stream_viewer:    { group: "output",     icon: "📺", label: "Stream" },
  video_writer:     { group: "output",     icon: "💾", label: "Video Out" },
  trigger_webhook:  { group: "output",     icon: "🔔", label: "Webhook" },
  mqtt_publish:     { group: "output",     icon: "📡", label: "MQTT Publish" },
  kafka_produce:    { group: "output",     icon: "🌊", label: "Kafka Produce" },
};

export const NODE_PORTS = {
  // Source nodes — no inputs
  camera:           { inputs: [],                                      outputs: [{ id: "out",  label: "Frame" }] },
  usb_camera:       { inputs: [],                                      outputs: [{ id: "out",  label: "Frame" }] },
  video_file:       { inputs: [],                                      outputs: [{ id: "out",  label: "Frame" }] },
  image_directory:  { inputs: [],                                      outputs: [{ id: "out",  label: "Frame" }] },
  rtsp_stream:      { inputs: [],                                      outputs: [{ id: "out",  label: "Frame" }] },
  // Processing
  preprocess:       { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  model_inference:  { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" },
                                                                                  { id: "dets", label: "Detections" }] },
  nms:              { inputs: [{ id: "in",   label: "Frame+Dets" }],   outputs: [{ id: "out",  label: "Frame+Dets" }] },
  // Vision
  blur:             { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  edge_detect:      { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Edges" }] },
  corner_detect:    { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame+Pts" }] },
  threshold:        { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Mask" }] },
  color_convert:    { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  morph:            { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  resize:           { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  affine_transform: { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  // Spatial
  draw_roi:         { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  draw_line:        { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  object_tracker:   { inputs: [{ id: "in",   label: "Frame+Dets" }],   outputs: [{ id: "out",  label: "Tracked" }] },
  counter:          { inputs: [{ id: "in",   label: "Frame" }],        outputs: [{ id: "out",  label: "Frame" }] },
  // Utility (python/cpp ports are dynamic via data.ports)
  python_function:  { inputs: [{ id: "in",   label: "frame" }],        outputs: [{ id: "out",  label: "frame" }] },
  filter:           { inputs: [{ id: "in",   label: "Dets" }],         outputs: [{ id: "out",  label: "Filtered" }] },
  param:            { inputs: [],                                       outputs: [{ id: "out",  label: "Params" }] },
  cpp_function:     { inputs: [{ id: "in",   label: "frame" }],        outputs: [{ id: "out",  label: "frame" }] },
  // Output nodes — no outputs
  stream_viewer:    { inputs: [{ id: "in",   label: "Frame" }],        outputs: [] },
  video_writer:     { inputs: [{ id: "in",   label: "Frame" }],        outputs: [] },
  trigger_webhook:  { inputs: [{ id: "in",   label: "Frame" }],        outputs: [] },
  mqtt_publish:     { inputs: [{ id: "in",   label: "Frame" }],        outputs: [] },
  kafka_produce:    { inputs: [{ id: "in",   label: "Frame" }],        outputs: [] },
};

const DEFAULT_PORTS = { inputs: [{ id: "in", label: "in" }], outputs: [{ id: "out", label: "out" }] };

function trunc(s, n = 20) {
  s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function Preview({ type, cfg }) {
  let lines = [];
  if      (type === "camera")           lines = [`device: ${cfg.device_index ?? 0}`, `fps: ${cfg.fps_limit ?? 30}`];
  else if (type === "usb_camera")       lines = [`device: ${cfg.device_index ?? 0}`, `fps: ${cfg.fps_limit ?? 30}`];
  else if (type === "rtsp_stream")      lines = [trunc(cfg.url || "(no URL)", 24), `fps: ${cfg.fps_limit ?? 30}`];
  else if (type === "video_file")       lines = [trunc(cfg.file_path || "(no file)"), cfg.loop ? "loop ✓" : "loop ✗"];
  else if (type === "image_directory")  lines = [trunc(cfg.directory_path || "(no dir)"), cfg.pattern || "*.jpg"];
  else if (type === "model_inference")  lines = [cfg.model_id ? `id: ${trunc(cfg.model_id, 16)}` : "⚠ no model", `${cfg.device ?? "cpu"} | conf ${cfg.conf_threshold ?? 0.5}`];
  else if (type === "nms")              lines = [`iou: ${cfg.iou_threshold ?? 0.45}`, `conf: ${cfg.conf_threshold ?? 0.25}`];
  else if (type === "blur")             lines = [`${cfg.type ?? "gaussian"} · k=${cfg.kernel_size ?? 5}`];
  else if (type === "edge_detect")      lines = [`${cfg.algorithm ?? "canny"} · ${cfg.threshold1 ?? 50}/${cfg.threshold2 ?? 150}`];
  else if (type === "corner_detect")    lines = [`${cfg.algorithm ?? "harris"} · max=${cfg.max_corners ?? 100}`];
  else if (type === "threshold")        lines = [`${cfg.type ?? "binary"} · t=${cfg.threshold ?? 127}`];
  else if (type === "color_convert")    lines = [cfg.conversion ?? "bgr2gray"];
  else if (type === "morph")            lines = [`${cfg.operation ?? "erode"} · k=${cfg.kernel_size ?? 3}`];
  else if (type === "resize")           lines = [`${cfg.width ?? 640} × ${cfg.height ?? 480}`];
  else if (type === "affine_transform") lines = [`tx=${cfg.tx ?? 0} ty=${cfg.ty ?? 0}`, `ang=${cfg.angle ?? 0}° sc=${cfg.scale ?? 1}`];
  else if (type === "draw_line")        lines = [`id: ${cfg.line_id ?? "line_1"}`, `dir: ${cfg.direction ?? "both"}`];
  else if (type === "object_tracker")   lines = [`algo: ${cfg.algorithm ?? "bytetrack"}`];
  else if (type === "counter")          lines = [`trigger: ${cfg.trigger_id ?? "—"}`];
  else if (type === "stream_viewer")    lines = [`q:${cfg.jpeg_quality ?? 80} fps:${cfg.max_fps ?? 30}`];
  else if (type === "video_writer")     lines = [trunc(cfg.output_path || "./output.mp4")];
  else if (type === "cpp_function")     lines = [`status: ${cfg.compile_status ?? "uncompiled"}`];
  else if (type === "python_function")  lines = ["custom Python code"];
  else if (type === "mqtt_publish")     lines = [`${cfg.broker ?? "localhost"}:${cfg.port ?? 1883}`, trunc(cfg.topic || "cv_flow/events")];
  else if (type === "kafka_produce")    lines = [trunc(cfg.bootstrap_servers || "localhost:9092"), trunc(cfg.topic || "cv_flow_events")];
  else                                  lines = [];

  if (!lines.length) return html`<div style=${{ height: 4 }} />`;
  return html`
    <div style=${{ lineHeight: 1.6 }}>
      ${lines.map((l, i) => html`
        <div key=${i} style=${{ color: "#8b949e", fontSize: 11, whiteSpace: "nowrap",
                                 overflow: "hidden", textOverflow: "ellipsis" }}>
          ${l}
        </div>`)}
    </div>`;
}

// Layout constants (px) — must match across nodes.js and app.js if referenced
const HEADER_H = 34;
const PREVIEW_H = 46;
const PORT_H = 22;

export function makeNode(type) {
  const meta  = NODE_META[type] ?? { group: "utility", icon: "◻", label: type };
  const hdrBg = GROUP_COLOR[meta.group] ?? "#21262d";

  const Comp = ({ data, selected }) => {
    const ports   = data.ports ?? NODE_PORTS[type] ?? DEFAULT_PORTS;
    const maxRows = Math.max(ports.inputs.length, ports.outputs.length);
    // Vertical center of port row i, measured from top of node element
    const portTop = (i) => HEADER_H + PREVIEW_H + (i + 0.5) * PORT_H;

    return html`
      <div style=${{
        background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
        minWidth: 186, fontSize: 12, color: "#c9d1d9",
        boxShadow: selected ? "0 0 0 2px #58a6ff" : "0 2px 8px rgba(0,0,0,.5)",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}>

        ${ports.inputs.map((port, i) => html`
          <${Handle} key=${port.id} type="target" position=${Position.Left} id=${port.id}
            style=${{ width: 10, height: 10, background: "#58a6ff",
                       border: "2px solid #0d1117", left: -6, top: portTop(i) }} />`)}

        <div style=${{ background: hdrBg, borderRadius: "7px 7px 0 0",
                        padding: "0 10px", height: HEADER_H,
                        display: "flex", alignItems: "center", gap: 7 }}>
          <span style=${{ fontSize: 14, lineHeight: 1, flexShrink: 0 }}>${meta.icon}</span>
          <span style=${{ fontWeight: 600, fontSize: 12, color: "#e2e8f0",
                           overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
            ${data.label ?? meta.label}
          </span>
        </div>

        <div style=${{ height: PREVIEW_H, overflow: "hidden", padding: "6px 10px 0" }}>
          <${Preview} type=${type} cfg=${data.config ?? {}} />
        </div>

        ${maxRows > 0 && html`
          <div style=${{ borderTop: "1px solid #30363d" }}>
            ${Array.from({ length: maxRows }, (_, i) => html`
              <div key=${i} style=${{
                display: "grid", gridTemplateColumns: "1fr 1fr",
                height: PORT_H, alignItems: "center",
                borderBottom: i < maxRows - 1 ? "1px solid #21262d" : "none",
              }}>
                <div style=${{ paddingLeft: 14, fontSize: 10, color: "#58a6ff",
                                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  ${ports.inputs[i] ? "● " + ports.inputs[i].label : ""}
                </div>
                <div style=${{ paddingRight: 14, fontSize: 10, color: "#3fb950",
                                textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  ${ports.outputs[i] ? ports.outputs[i].label + " ●" : ""}
                </div>
              </div>`)}
          </div>`}

        ${ports.outputs.map((port, i) => html`
          <${Handle} key=${port.id} type="source" position=${Position.Right} id=${port.id}
            style=${{ width: 10, height: 10, background: "#3fb950",
                       border: "2px solid #0d1117", right: -6, top: portTop(i) }} />`)}
      </div>`;
  };

  Comp.displayName = type;
  return Comp;
}

// Mutable registry — custom nodes extend this at runtime
export const nodeTypes = Object.fromEntries(
  Object.keys(NODE_META).map(t => [t, makeNode(t)])
);

export function registerNodeType(type, meta, ports) {
  NODE_META[type] = meta;
  NODE_PORTS[type] = ports ?? DEFAULT_PORTS;
  nodeTypes[type] = makeNode(type);
}
