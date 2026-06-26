import { createElement, useState, useEffect, useCallback } from "react";
import htm from "https://esm.sh/htm@3";
const html = htm.bind(createElement);

// Basic UI components
const inp = {
  width: "100%", background: "#0d1117", border: "1px solid #30363d", borderRadius: 6,
  color: "#c9d1d9", padding: "6px 10px", fontSize: 12, outline: "none", fontFamily: "inherit"
};

const btn = {
  padding: "6px 12px", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 500,
  background: "#21262d", border: "1px solid #30363d", color: "#c9d1d9", fontFamily: "inherit"
};

async function apiFetch(method, endpoint, body) {
  const opts = { method, headers: {} };
  if (body) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const res = await fetch("/api" + endpoint, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}

function Pagination({ page, total, limit, onPageChange }) {
  const totalPages = Math.ceil(total / limit);
  return html`
    <div style=${{ display: "flex", gap: 8, alignItems: "center", marginTop: 12, justifyContent: "center" }}>
      <button disabled=${page === 0} onClick=${() => onPageChange(page - 1)} style=${btn}>← Prev</button>
      <span style=${{ fontSize: 11, color: "#8b949e" }}>
        ${Math.min(page * limit + 1, total)}–${Math.min((page + 1) * limit, total)} of ${total}
      </span>
      <button disabled=${(page + 1) * limit >= total} onClick=${() => onPageChange(page + 1)} style=${btn}>Next →</button>
    </div>`;
}

export function DataHubPanel({ onClose }) {
  const [activeTab, setActiveTab]     = useState("relational");  // "relational" | "vector"
  const [tables, setTables]           = useState([]);
  const [collections, setCollections] = useState([]);

  // Browse modal state (Relational)
  const [browseTable, setBrowseTable] = useState(null);
  const [browseRows,  setBrowseRows]  = useState([]);
  const [browseTotal, setBrowseTotal] = useState(0);
  const [browsePage,  setBrowsePage]  = useState(0);
  const BROWSE_LIMIT = 20;

  // Vector browse state
  const [vecCollection, setVecCollection] = useState(null);
  const [vecRecords,    setVecRecords]    = useState([]);
  const [vecTotal,      setVecTotal]      = useState(0);
  const [vecPage,       setVecPage]       = useState(0);
  const [vecSearch,     setVecSearch]     = useState("");
  const [vecSearchResults, setVecSearchResults] = useState(null);
  const VEC_LIMIT = 20;

  // Loading / error state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Create collection modal
  const [showCreate, setShowCreate] = useState(false);
  const [newName,    setNewName]    = useState("");
  const [newDim,     setNewDim]     = useState("512");

  const fetchTables = useCallback(async () => {
    try {
      const res = await apiFetch("GET", "/datahub/relational/tables");
      setTables(res);
    } catch (e) { setError("Failed to load tables: " + e.message); }
  }, []);

  const fetchCollections = useCallback(async () => {
    try {
      const res = await apiFetch("GET", "/datahub/vector/collections");
      setCollections(res);
    } catch (e) { setError("Failed to load vector collections: " + e.message); }
  }, []);

  // Initial load + auto-refresh every 5s while panel is open
  useEffect(() => {
    if (activeTab === "relational") {
      fetchTables();
      const id = setInterval(fetchTables, 5000);
      return () => clearInterval(id);
    } else {
      fetchCollections();
      const id = setInterval(fetchCollections, 5000);
      return () => clearInterval(id);
    }
  }, [activeTab, fetchTables, fetchCollections]);

  const openVecCollection = async (name, page = 0) => {
    setVecCollection(name); setVecPage(page); setVecSearchResults(null); setVecSearch("");
    try {
      setLoading(true);
      const res = await apiFetch("GET", `/datahub/vector/collections/${name}/records?limit=${VEC_LIMIT}&offset=${page * VEC_LIMIT}`);
      setVecRecords(res.records ?? res.items ?? []); setVecTotal(res.total ?? 0);
    } catch (e) { setError("Failed to load records: " + e.message); }
    finally { setLoading(false); }
  };

  const vecSearchExec = async () => {
    if (!vecSearch.trim() || !vecCollection) return;
    try {
      setLoading(true);
      const nums = vecSearch.split(",").map(Number).filter(n => !isNaN(n));
      if (nums.length === 0) { setError("Enter comma-separated floats as query vector"); return; }
      const res = await apiFetch("POST", `/datahub/vector/collections/${vecCollection}/search`, { embedding: nums, top_k: 5 });
      setVecSearchResults(res.results ?? res ?? []);
    } catch (e) { setError("Search failed: " + e.message); }
    finally { setLoading(false); }
  };

  // Browse Relational
  const openBrowseTable = async (table) => {
    setBrowseTable(table);
    setBrowsePage(0);
    await loadBrowseData(table, 0);
  };

  const loadBrowseData = async (table, page) => {
    try {
      setLoading(true);
      let endpoint = `/datahub/relational/${table}?limit=${BROWSE_LIMIT}&offset=${page * BROWSE_LIMIT}`;
      if (table === "execution_sessions") endpoint = `/datahub/relational/sessions?limit=${BROWSE_LIMIT}&offset=${page * BROWSE_LIMIT}`;
      else if (table === "detection_events") endpoint = `/datahub/relational/events?limit=${BROWSE_LIMIT}&offset=${page * BROWSE_LIMIT}`;
      
      const res = await apiFetch("GET", endpoint);
      setBrowseRows(res.data ?? []);
      setBrowseTotal(res.total ?? 0);
    } catch (e) { setError("Failed to load data for " + table + ": " + e.message); }
    finally { setLoading(false); }
  };

  const exportTable = (table) => {
    window.open(`/api/datahub/relational/export/${table}`, "_blank");
  };

  const tabBtn = (key, label) => html`
    <button onClick=${() => setActiveTab(key)}
      style=${{ padding: "6px 14px", border: "none", cursor: "pointer", fontSize: 13, fontWeight: 600,
                 background: activeTab === key ? "#1f3a5e" : "transparent",
                 color: activeTab === key ? "#58a6ff" : "#8b949e",
                 borderBottom: activeTab === key ? "2px solid #58a6ff" : "2px solid transparent",
                 fontFamily: "inherit" }}>
      ${label}
    </button>`;

  return html`
    <div style=${{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(0,0,0,.75)", display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick=${e => e.target === e.currentTarget && onClose()}>
      
      <div style=${{ background: "#161b22", border: "1px solid #30363d", borderRadius: 12, width: 800, height: "85vh", display: "flex", flexDirection: "column", boxShadow: "0 16px 48px rgba(0,0,0,.8)" }}>
        
        <div style=${{ display: "flex", alignItems: "center", padding: "14px 18px", borderBottom: "1px solid #30363d" }}>
          <span style=${{ fontWeight: 700, fontSize: 16, color: "#e2e8f0", flex: 1 }}>🗄️ Data Hub</span>
          <button onClick=${onClose} style=${{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18, padding: "0 4px" }}>✕</button>
        </div>

        <div style=${{ display: "flex", borderBottom: "1px solid #30363d", paddingLeft: 10 }}>
          ${tabBtn("relational", "Relational DBs")}
          ${tabBtn("vector", "Vector Collections")}
        </div>

        ${error && html`
          <div style=${{ margin: "14px 18px 0", padding: "8px 12px", borderRadius: 6, fontSize: 12, background: "#3d1a1a", border: "1px solid #f85149", color: "#f85149" }}>
            ${error}
            <button onClick=${() => setError(null)} style=${{ float: "right", background: "none", border: "none", color: "#f85149", cursor: "pointer" }}>✕</button>
          </div>
        `}

        <div style=${{ flex: 1, overflowY: "auto", padding: "18px" }}>
          ${activeTab === "relational" && !browseTable && html`
            <div>
              <div style=${{ fontSize: 11, fontWeight: 700, color: "#8b949e", letterSpacing: 1, textTransform: "uppercase", marginBottom: 12 }}>Relational Databases</div>
              ${loading && html`<div style=${{ color: "#8b949e", fontSize: 12 }}>Loading...</div>`}
              <div style=${{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))" }}>
                ${tables.map(t => html`
                  <div key=${t.name} style=${{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8, padding: "14px" }}>
                    <div style=${{ display: "flex", alignItems: "center", marginBottom: 8 }}>
                      <span style=${{ fontSize: 16, marginRight: 8 }}>📋</span>
                      <span style=${{ fontWeight: 600, fontSize: 14, color: "#e2e8f0", flex: 1 }}>${t.name}</span>
                      <span style=${{ fontSize: 12, color: "#8b949e", background: "#21262d", padding: "2px 8px", borderRadius: 12 }}>${t.row_count} rows</span>
                    </div>
                    <div style=${{ display: "flex", gap: 8, marginTop: 12 }}>
                      <button onClick=${() => openBrowseTable(t.name)} style=${btn}>Browse Data</button>
                      <button onClick=${() => exportTable(t.name)} style=${btn}>Export CSV</button>
                    </div>
                  </div>
                `)}
              </div>
            </div>
          `}

          ${activeTab === "vector" && !vecCollection && html`
            <div>
              <div style=${{ display: "flex", alignItems: "center", marginBottom: 12 }}>
                <div style=${{ fontSize: 11, fontWeight: 700, color: "#8b949e", letterSpacing: 1, textTransform: "uppercase", flex: 1 }}>Vector Databases</div>
                <button onClick=${() => { setNewName(""); setNewDim("512"); setShowCreate(true); }} style=${{ ...btn, background: "#1f3a5e", color: "#58a6ff", border: "1px solid #1f6feb" }}>+ New Collection</button>
              </div>
              ${showCreate && html`
                <div style=${{ background: "#0d1117", border: "1px solid #58a6ff", borderRadius: 8, padding: 16, marginBottom: 16 }}>
                  <div style=${{ fontWeight: 600, fontSize: 13, color: "#e2e8f0", marginBottom: 12 }}>Create Vector Collection</div>
                  <div style=${{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                    <div style=${{ flex: 2 }}>
                      <div style=${{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>Name</div>
                      <input style=${inp} placeholder="faces, embeddings_v2, ..." value=${newName} onInput=${e => setNewName(e.target.value)} />
                    </div>
                    <div style=${{ flex: 1 }}>
                      <div style=${{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>Dimensions</div>
                      <input style=${inp} type="number" value=${newDim} onInput=${e => setNewDim(e.target.value)} min="1" max="65536" />
                    </div>
                    <button style=${{ ...btn, background: "#1a3d2e", color: "#3fb950", border: "1px solid #3fb950" }}
                      onClick=${async () => {
                        if (!newName.trim()) { setError("Name is required"); return; }
                        const dim = parseInt(newDim) || 512;
                        try {
                          await apiFetch("POST", "/datahub/vector/collections", { name: newName.trim(), dim });
                          setShowCreate(false);
                          fetchCollections();
                        } catch (e) { setError("Create failed: " + e.message); }
                      }}>Create</button>
                    <button style=${btn} onClick=${() => setShowCreate(false)}>Cancel</button>
                  </div>
                </div>
              `}
              <div style=${{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))" }}>
                ${collections.map(c => html`
                  <div key=${c.name} style=${{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8, padding: "14px" }}>
                    <div style=${{ display: "flex", alignItems: "center", marginBottom: 8 }}>
                      <span style=${{ fontSize: 16, marginRight: 8 }}>🔷</span>
                      <span style=${{ fontWeight: 600, fontSize: 14, color: "#e2e8f0", flex: 1 }}>${c.name}</span>
                      <span style=${{ fontSize: 12, color: "#8b949e", background: "#21262d", padding: "2px 8px", borderRadius: 12 }}>${c.count} vecs · ${c.dim}d</span>
                    </div>
                    <div style=${{ display: "flex", gap: 8 }}>
                      <button onClick=${() => openVecCollection(c.name)} style=${btn}>Browse</button>
                      <button onClick=${() => window.open(`/api/datahub/vector/collections/${c.name}/export`, "_blank")} style=${btn}>Export ZIP</button>
                    </div>
                  </div>
                `)}
                ${collections.length === 0 && html`<div style=${{ color: "#555d68", fontSize: 12 }}>No vector collections yet.</div>`}
              </div>
            </div>
          `}

          ${activeTab === "vector" && vecCollection && html`
            <div style=${{ display: "flex", flexDirection: "column", height: "100%" }}>
              <div style=${{ display: "flex", alignItems: "center", marginBottom: 12, gap: 8 }}>
                <button onClick=${() => setVecCollection(null)} style=${btn}>← Back</button>
                <span style=${{ fontWeight: 600, fontSize: 15, color: "#e2e8f0", flex: 1 }}>🔷 ${vecCollection}</span>
                <span style=${{ fontSize: 12, color: "#8b949e" }}>Total: ${vecTotal}</span>
                <button onClick=${() => window.open(`/api/datahub/vector/collections/${vecCollection}/export`, "_blank")} style=${btn}>Export ZIP</button>
              </div>

              <!-- Similarity Search -->
              <div style=${{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8, padding: 12, marginBottom: 12 }}>
                <div style=${{ fontSize: 11, fontWeight: 700, color: "#8b949e", marginBottom: 8 }}>SIMILARITY SEARCH</div>
                <div style=${{ display: "flex", gap: 8 }}>
                  <input style=${{ ...inp, flex: 1 }} placeholder="Paste comma-separated floats (e.g. 0.1,-0.5,0.3,...)"
                    value=${vecSearch} onInput=${e => setVecSearch(e.target.value)}
                    onKeyDown=${e => e.key === "Enter" && vecSearchExec()} />
                  <button onClick=${vecSearchExec} style=${{ ...btn, background: "#1f3a5e", color: "#58a6ff" }}>Search</button>
                </div>
                ${vecSearchResults && html`
                  <div style=${{ marginTop: 10 }}>
                    <div style=${{ fontSize: 11, color: "#8b949e", marginBottom: 6 }}>Top ${vecSearchResults.length} results:</div>
                    ${vecSearchResults.map((r, i) => html`
                      <div key=${i} style=${{ display: "flex", gap: 10, padding: "5px 0", borderBottom: "1px solid #21262d", fontSize: 11 }}>
                        <span style=${{ color: "#e3b341", minWidth: 32 }}>${(r.score ?? r.similarity ?? 0).toFixed(3)}</span>
                        <span style=${{ color: "#c9d1d9", flex: 1 }}>${r.id ?? r.meta?.id ?? "?"}</span>
                        <span style=${{ color: "#8b949e" }}>${r.label ?? r.meta?.label ?? ""}</span>
                      </div>
                    `)}
                  </div>
                `}
              </div>

              <!-- Records table -->
              <div style=${{ flex: 1, overflow: "auto", border: "1px solid #30363d", borderRadius: 6, background: "#0d1117" }}>
                ${loading ? html`<div style=${{ padding: 20, textAlign: "center", color: "#8b949e" }}>Loading...</div>` : html`
                  <table style=${{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead style=${{ position: "sticky", top: 0, background: "#161b22" }}>
                      <tr>
                        <th style=${{ padding: "8px 12px", textAlign: "left", color: "#8b949e", borderBottom: "1px solid #30363d" }}>ID</th>
                        <th style=${{ padding: "8px 12px", textAlign: "left", color: "#8b949e", borderBottom: "1px solid #30363d" }}>Label</th>
                        <th style=${{ padding: "8px 12px", textAlign: "left", color: "#8b949e", borderBottom: "1px solid #30363d" }}>Embedding (first 6)</th>
                        <th style=${{ padding: "8px 12px", textAlign: "left", color: "#8b949e", borderBottom: "1px solid #30363d" }}>Metadata</th>
                      </tr>
                    </thead>
                    <tbody>
                      ${vecRecords.map((r, i) => html`
                        <tr key=${i} style=${{ borderBottom: "1px solid #21262d" }}>
                          <td style=${{ padding: "6px 12px", color: "#8b949e", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>${r.id}</td>
                          <td style=${{ padding: "6px 12px", color: "#c9d1d9" }}>${r.label ?? ""}</td>
                          <td style=${{ padding: "6px 12px", color: "#555d68", fontFamily: "monospace", fontSize: 10 }}>
                            [${(r.embedding ?? []).slice(0,6).map(v => v.toFixed(3)).join(", ")}${(r.embedding?.length ?? 0) > 6 ? ", …" : ""}]
                          </td>
                          <td style=${{ padding: "6px 12px", color: "#555d68", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            ${JSON.stringify(r.metadata ?? r.meta ?? {})}
                          </td>
                        </tr>
                      `)}
                    </tbody>
                  </table>
                  ${vecRecords.length === 0 && html`<div style=${{ padding: 20, textAlign: "center", color: "#555d68" }}>No records</div>`}
                `}
              </div>
              ${vecTotal > VEC_LIMIT && html`
                <${Pagination} page=${vecPage} total=${vecTotal} limit=${VEC_LIMIT}
                  onPageChange=${p => openVecCollection(vecCollection, p)} />
              `}
            </div>
          `}

          ${activeTab === "relational" && browseTable && html`
            <div style=${{ display: "flex", flexDirection: "column", height: "100%" }}>
              <div style=${{ display: "flex", alignItems: "center", marginBottom: 12 }}>
                <button onClick=${() => setBrowseTable(null)} style=${{ ...btn, marginRight: 12 }}>← Back</button>
                <span style=${{ fontWeight: 600, fontSize: 15, color: "#e2e8f0", flex: 1 }}>${browseTable}</span>
                <span style=${{ fontSize: 12, color: "#8b949e", marginRight: 12 }}>Total: ${browseTotal}</span>
                <button onClick=${() => exportTable(browseTable)} style=${btn}>Export CSV</button>
              </div>
              
              <div style=${{ flex: 1, overflow: "auto", border: "1px solid #30363d", borderRadius: 6, background: "#0d1117" }}>
                ${loading ? html`<div style=${{ padding: 20, textAlign: "center", color: "#8b949e" }}>Loading...</div>` : html`
                  <table style=${{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead style=${{ position: "sticky", top: 0, background: "#161b22", zIndex: 1 }}>
                      <tr>
                        ${browseRows.length > 0 && Object.keys(browseRows[0]).map(k => html`
                          <th key=${k} style=${{ padding: "8px 12px", textAlign: "left", color: "#c9d1d9", borderBottom: "1px solid #30363d", whiteSpace: "nowrap" }}>${k}</th>
                        `)}
                      </tr>
                    </thead>
                    <tbody>
                      ${browseRows.map((r, i) => html`
                        <tr key=${i} style=${{ borderBottom: "1px solid #21262d" }}>
                          ${Object.values(r).map((v, j) => html`
                            <td key=${j} style=${{ padding: "6px 12px", color: "#8b949e", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title=${String(v)}>
                              ${String(v)}
                            </td>
                          `)}
                        </tr>
                      `)}
                    </tbody>
                  </table>
                  ${browseRows.length === 0 && html`<div style=${{ padding: 20, textAlign: "center", color: "#555d68" }}>No data</div>`}
                `}
              </div>
              
              ${browseTotal > BROWSE_LIMIT && html`
                <${Pagination} page=${browsePage} total=${browseTotal} limit=${BROWSE_LIMIT} onPageChange=${p => { setBrowsePage(p); loadBrowseData(browseTable, p); }} />
              `}
            </div>
          `}
        </div>
      </div>
    </div>
  `;
}
