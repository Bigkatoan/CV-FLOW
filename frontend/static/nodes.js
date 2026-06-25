import { createElement } from "react";
import { Handle, Position, NodeResizer } from "reactflow";
import htm from "https://esm.sh/htm@3";
const html = htm.bind(createElement);

export const GROUP_COLOR = {
  core: "#1a2d4a",
};

export const NODE_META = {
  python_node: { group: "core", icon: "🐍", label: "Python Node" },
  cpp_node:    { group: "core", icon: "⚙️", label: "C++ Node" },
};

export const NODE_PORTS = {
  python_node: { inputs: [{ id: "in", label: "in" }],  outputs: [{ id: "out", label: "out" }] },
  cpp_node:    { inputs: [{ id: "in", label: "in" }],  outputs: [{ id: "out", label: "out" }] },
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

export function makeNode(type) {
  const meta  = NODE_META[type] ?? { group: "core", icon: "◻", label: type };
  const hdrBg = GROUP_COLOR[meta.group] ?? "#21262d";

  const Comp = ({ data, selected }) => {
    const cfg    = data.config ?? {};
    const ports  = data.ports ?? NODE_PORTS[type] ?? DEFAULT_PORTS;
    const mode   = cfg.mode ?? "loop";
    const maxRows = Math.max(ports.inputs.length, ports.outputs.length);

    // Top of the ports section, measured from top of node element
    const portTop = (i) => HEADER_H + BADGE_H + PREVIEW_H + (i + 0.5) * PORT_H;

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

        <!-- Mode badge -->
        <div style=${{
          height: BADGE_H, padding: "0 10px",
          display: "flex", alignItems: "center", gap: 6,
          borderBottom: "1px solid #21262d",
        }}>
          <span style=${{
            fontSize: 10, fontWeight: 700, letterSpacing: 0.6,
            padding: "2px 6px", borderRadius: 4,
            background: mode === "iteration" ? "#2d1a4a" : "#1a2d1a",
            color:      mode === "iteration" ? "#c9a0ff" : "#3fb950",
          }}>${mode === "iteration" ? "ITER" : "LOOP"}</span>
          ${mode === "iteration" && html`
            <span style=${{ fontSize: 10, color: "#6e7681" }}>
              ← ${cfg.active_key ?? "active"}
            </span>`}
        </div>

        <!-- Code preview -->
        <div style=${{ height: PREVIEW_H, overflow: "hidden", padding: "5px 10px 0", flexShrink: 0 }}>
          <${CodePreview} code=${cfg.code} />
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

        ${ports.outputs.map((port, i) => html`
          <${Handle} key=${port.id} type="source" position=${Position.Right} id=${port.id}
            style=${{ width: 10, height: 10, background: "#3fb950",
                       border: "2px solid #0d1117", right: -6, top: portTop(i) }} />`)}
      </div>`;
  };

  Comp.displayName = type;
  return Comp;
}

// Mutable registry — custom template nodes extend this at runtime
export const nodeTypes = Object.fromEntries(
  Object.keys(NODE_META).map(t => [t, makeNode(t)])
);

export function registerNodeType(type, meta, ports) {
  NODE_META[type] = meta;
  NODE_PORTS[type] = ports ?? DEFAULT_PORTS;
  nodeTypes[type] = makeNode(type);
}
