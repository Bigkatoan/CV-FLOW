import { createElement, useState } from "react";
import { Handle, Position, NodeResizer, useReactFlow } from "reactflow";
import htm from "https://esm.sh/htm@3";
const html = htm.bind(createElement);

export const GROUP_COLOR = {
  core:  "#1a2d4a",
  model: "#2a2010",
};

// Only three node types exist — all behaviour is expressed as user code.
export const NODE_META = {
  python_node: { group: "core",  icon: "🐍", label: "Python Node" },
  cpp_node:    { group: "core",  icon: "⚙️", label: "C++ Node"   },
  model_node:  { group: "model", icon: "🧠", label: "Model Node"  },
};

const _IN  = [{ id: "in",  label: "in"  }];
const _OUT = [{ id: "out", label: "out" }];

export const NODE_PORTS = {
  python_node: { inputs: _IN, outputs: _OUT },
  cpp_node:    { inputs: _IN, outputs: _OUT },
  model_node:  { inputs: _IN, outputs: _OUT },
};

const DEFAULT_PORTS = { inputs: [{ id: "in", label: "in" }], outputs: [{ id: "out", label: "out" }] };

// Layout constants (px)
const HEADER_H  = 36;
const BADGE_H   = 26;
const PREVIEW_H = 38;
const PORT_H    = 22;

function CodePreview({ code }) {
  if (!code) return html`<div style=${{ height: 4 }} />`;
  const lines = code.trim().split("\n").filter(l => l.trim()).slice(0, 3);
  return html`
    <div style=${{ fontFamily: "'SF Mono','Fira Code',monospace", fontSize: 10, lineHeight: 1.5 }}>
      ${lines.map((l, i) => html`
        <div key=${i} style=${{
          color: "#6e7681", whiteSpace: "nowrap",
          overflow: "hidden", textOverflow: "ellipsis",
        }}>${l}</div>`)}
    </div>`;
}

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

export function makeNode(type) {
  const meta  = NODE_META[type] ?? { group: "core", icon: "◻", label: type };
  const hdrBg = GROUP_COLOR[meta.group] ?? "#21262d";

  const Comp = ({ id, data, selected }) => {
    const { setNodes } = useReactFlow();
    const [paramsOpen, setParamsOpen] = useState(false);
    const [resOpen,    setResOpen]    = useState(false);

    const cfg       = data.config ?? {};
    const ports     = data.ports ?? NODE_PORTS[type] ?? DEFAULT_PORTS;
    const mode      = cfg.mode ?? "loop";
    const res       = data.resources ?? {};
    const isCodeNode = type === "python_node" || type === "cpp_node";

    const params = isCodeNode ? parseParamsInline(cfg.code) : [];
    const maxRows = Math.max(ports.inputs.length, ports.outputs.length);

    // Top of the ports section, measured from top of node element
    const portTop = (i) => HEADER_H + BADGE_H + PREVIEW_H + (i + 0.5) * PORT_H;

    // Update a config key — immutable update qua setNodes
    const setParam = (key, val) => setNodes(ns => ns.map(n =>
      n.id !== id ? n : { ...n, data: { ...n.data, config: { ...n.data.config, [key]: val } } }
    ));

    // Update a resource key
    const setRes = (key, val) => setNodes(ns => ns.map(n =>
      n.id !== id ? n : { ...n, data: { ...n.data, resources: { ...(n.data.resources ?? {}), [key]: val } } }
    ));

    let resBadge = "";
    if (res.max_fps > 0) resBadge = `⚡ ${res.max_fps}fps`;
    else if ((res.cpu_cores || []).length > 0) resBadge = `⚡ ${(res.cpu_cores || []).length} cores`;
    else if (res.max_memory_mb > 0) resBadge = `⚡ ${res.max_memory_mb}MB`;
    else resBadge = "⚡ All resources";

    return html`
      <${NodeResizer} minWidth=${180} minHeight=${100} isVisible=${selected}
        lineStyle=${{ borderColor: "#58a6ff55" }}
        handleStyle=${{ width: 8, height: 8, background: "#58a6ff", border: "none", borderRadius: 2 }} />

      <div style=${{
        background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
        minWidth: 180, width: "100%", height: "100%", fontSize: 12, color: "#c9d1d9",
        boxShadow: selected ? "0 0 0 2px #58a6ff" : "0 2px 8px rgba(0,0,0,.5)",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        overflow: "hidden", display: "flex", flexDirection: "column",
      }}>

        ${ports.inputs.map((port, i) => html`
          <${Handle} key=${port.id} type="target" position=${Position.Left} id=${port.id}
            style=${{ width: 10, height: 10, background: "#58a6ff",
                       border: "2px solid #0d1117", left: -6, top: portTop(i) }} />`)}

        <!-- Header -->
        <div style=${{
          background: hdrBg, borderRadius: "7px 7px 0 0",
          padding: "0 10px", height: HEADER_H,
          display: "flex", alignItems: "center", gap: 7,
        }}>
          <span style=${{ fontSize: 15, lineHeight: 1, flexShrink: 0 }}>${meta.icon}</span>
          <span style=${{
            fontWeight: 600, fontSize: 12, color: "#e2e8f0",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1,
          }}>${data.label ?? meta.label}</span>
        </div>

        <!-- Badge row: mode for python/cpp, MODEL badge for model_node -->
        <div style=${{
          height: BADGE_H, padding: "0 10px",
          display: "flex", alignItems: "center", gap: 6,
          borderBottom: "1px solid #21262d",
        }}>
          ${isCodeNode ? html`
            <span style=${{
              fontSize: 10, fontWeight: 700, letterSpacing: 0.6,
              padding: "2px 6px", borderRadius: 4,
              background: mode === "iteration" ? "#2d1a4a" : "#1a2d1a",
              color:      mode === "iteration" ? "#c9a0ff" : "#3fb950",
            }}>${mode === "iteration" ? "ITER" : "LOOP"}</span>
            ${mode === "iteration" && html`
              <span style=${{ fontSize: 10, color: "#6e7681" }}>← ${cfg.active_key ?? "active"}</span>`}
          ` : html`
            <span style=${{
              fontSize: 10, fontWeight: 700, letterSpacing: 0.6,
              padding: "2px 6px", borderRadius: 4,
              background: "#2a2010", color: "#e3b341",
            }}>MODEL</span>
            <span style=${{ fontSize: 10, color: "#6e7681", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              ${cfg.model_id ? cfg.model_id.slice(0, 16) + "…" : "— select model —"}
            </span>
          `}
        </div>

        <!-- Preview: code snippet for python/cpp, model path for model_node -->
        <div style=${{ height: PREVIEW_H, overflow: "hidden", padding: "5px 10px 0", flexShrink: 0 }}>
          ${isCodeNode ? html`<${CodePreview} code=${cfg.code ?? cfg.source_code} />` : html`
            <div style=${{ fontSize: 11, lineHeight: 1.5, paddingTop: 1 }}>
              <div style=${{ color: cfg.model_id ? "#e3b341" : "#555d68",
                              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                ${cfg.model_id ? cfg.model_id : "No model selected — use Properties"}
              </div>
            </div>
          `}
        </div>

        <!-- Port rows -->
        ${maxRows > 0 && html`
          <div style=${{ borderTop: "1px solid #30363d", flex: 1 }}>
            ${Array.from({ length: maxRows }, (_, i) => html`
              <div key=${i} style=${{
                display: "grid", gridTemplateColumns: "1fr 1fr",
                height: PORT_H, alignItems: "center",
                borderBottom: i < maxRows - 1 ? "1px solid #21262d" : "none",
              }}>
                <div style=${{ paddingLeft: 14, fontSize: 10, color: "#58a6ff",
                                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  ${ports.inputs[i]  ? "● " + ports.inputs[i].label  : ""}
                </div>
                <div style=${{ paddingRight: 14, fontSize: 10, color: "#3fb950",
                                textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  ${ports.outputs[i] ? ports.outputs[i].label + " ●" : ""}
                </div>
              </div>`)}
          </div>`}
        
        <!-- Params Section -->
        ${params.length > 0 && html`
          <div>
            <div style=${{
              background: "#0d1117", borderTop: "1px solid #30363d", height: 24,
              padding: "4px 8px", fontSize: 10, color: "#8b949e", cursor: "pointer",
              display: "flex", justifyContent: "space-between", userSelect: "none"
            }} onClick=${() => setParamsOpen(!paramsOpen)}>
              <span>⚙ ${params.length} params</span>
              <span>${paramsOpen ? "▾" : "▸"}</span>
            </div>
            ${paramsOpen && html`
              <div style=${{ padding: "4px 8px", borderTop: "1px solid #21262d", display: "flex", flexDirection: "column", gap: 4 }}>
                ${params.map(p => {
                  const val = cfg[p.name] ?? p.default;
                  if (p.type === "slider") return html`
                    <div key=${p.name} style=${{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
                      <span style=${{ width: 50, overflow: "hidden", textOverflow: "ellipsis" }}>${p.name}</span>
                      <input type="range" min=${p.min} max=${p.max} value=${val}
                             onChange=${e => setParam(p.name, +e.target.value)}
                             style=${{ flex: 1, minWidth: 60 }} />
                      <span style=${{ width: 24, textAlign: "right" }}>${val}</span>
                    </div>`;
                  if (p.type === "checkbox") return html`
                    <div key=${p.name} style=${{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
                      <label style=${{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                        <input type="checkbox" checked=${val} onChange=${e => setParam(p.name, e.target.checked)} />
                        ${p.name}
                      </label>
                    </div>`;
                  if (p.type === "text") return html`
                    <div key=${p.name} style=${{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
                      <span style=${{ width: 50, overflow: "hidden", textOverflow: "ellipsis" }}>${p.name}</span>
                      <input type="text" value=${val}
                             onChange=${e => setParam(p.name, e.target.value)}
                             style=${{ width: 80, background: "#0d1117", border: "1px solid #30363d", color: "#c9d1d9", padding: "2px 4px", borderRadius: 4 }} />
                    </div>`;
                  return null;
                })}
              </div>
            `}
          </div>
        `}

        <!-- Resources Section -->
        <div>
          <div style=${{
            background: "#0d1117", borderTop: "1px solid #30363d", height: 24,
            padding: "4px 8px", fontSize: 10, color: "#8b949e", cursor: "pointer",
            display: "flex", justifyContent: "space-between", userSelect: "none"
          }} onClick=${() => setResOpen(!resOpen)}>
            <span>${resBadge}</span>
            <span>${resOpen ? "▾" : "▸"}</span>
          </div>
          ${resOpen && html`
            <div style=${{ padding: "6px 8px", borderTop: "1px solid #21262d", display: "flex", flexDirection: "column", gap: 6 }}>
              <div style=${{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
                <span style=${{ width: 30 }}>FPS</span>
                <input type="number" value=${res.max_fps || ""} placeholder="no limit"
                       onChange=${e => setRes("max_fps", parseInt(e.target.value) || 0)}
                       style=${{ width: 52, background: "#0d1117", border: "1px solid #30363d", color: "#c9d1d9", padding: "2px 4px", borderRadius: 4 }} />
                <span style=${{ width: 30, marginLeft: 4 }}>CPU</span>
                <input type="text" value=${(res.cpu_cores || []).join(",")} placeholder="all"
                       onChange=${e => {
                         const val = e.target.value;
                         setRes("cpu_cores", val.trim() ? val.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n)) : []);
                       }}
                       style=${{ width: 52, background: "#0d1117", border: "1px solid #30363d", color: "#c9d1d9", padding: "2px 4px", borderRadius: 4 }} />
              </div>
              <div style=${{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
                <span style=${{ width: 30 }}>Mem</span>
                <input type="number" value=${res.max_memory_mb || ""} placeholder="no limit"
                       onChange=${e => setRes("max_memory_mb", parseInt(e.target.value) || 0)}
                       style=${{ width: 52, background: "#0d1117", border: "1px solid #30363d", color: "#c9d1d9", padding: "2px 4px", borderRadius: 4 }} />
                <span>MB</span>
              </div>
            </div>
          `}
        </div>

        <!-- Live stats strip (shown when pipeline is running) -->
        ${data.liveStats && (() => {
          const s = data.liveStats;
          const ms = s.avg_ms ?? 0;
          const barColor = ms < 33 ? "#3fb950" : ms < 100 ? "#e3b341" : "#f85149";
          return html`
            <div style=${{
              borderTop: "1px solid #30363d", background: "#0a0f14",
              padding: "3px 8px", display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style=${{ fontSize: 10, color: barColor, fontVariantNumeric: "tabular-nums" }}>
                ${(s.fps ?? 0).toFixed(1)} fps
              </span>
              <span style=${{ fontSize: 10, color: barColor, fontVariantNumeric: "tabular-nums" }}>
                ${ms.toFixed(0)}ms
              </span>
              ${(s.errors ?? 0) > 0 && html`
                <span style=${{ fontSize: 10, color: "#f85149" }}>⚠ ${s.errors}</span>
              `}
            </div>`;
        })()}

        ${ports.outputs.map((port, i) => html`
          <${Handle} key=${port.id} type="source" position=${Position.Right} id=${port.id}
            style=${{ width: 10, height: 10, background: "#3fb950",
                       border: "2px solid #0d1117", right: -6, top: portTop(i) }} />`)}
      </div>`;
  };

  Comp.displayName = type;
  return Comp;
}

// Mutable registry — all known types (core + engine + custom templates)
export const nodeTypes = Object.fromEntries(
  Object.keys(NODE_META).map(t => [t, makeNode(t)])
);

export function registerNodeType(type, meta, ports) {
  NODE_META[type] = meta;
  NODE_PORTS[type] = ports ?? DEFAULT_PORTS;
  nodeTypes[type] = makeNode(type);
}
