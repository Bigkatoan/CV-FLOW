"""
CV-FLOW MCP Server — expose CV-FLOW control plane to Claude.
Transport: stdio (Claude Desktop standard).
Requires: pip install mcp
Requires: CV-FLOW backend running on localhost:8000
"""
import asyncio
import json
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("CV-FLOW", instructions="""
You have direct access to the CV-FLOW computer vision pipeline system.
Use these tools to create, run, and monitor CV pipelines on the user's machine.
When creating pipelines, prefer python_node type with Python code.
""")

BASE = "http://localhost:8000/api"
TIMEOUT = 15.0  # seconds — download/compile can be slow

# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _get(path: str) -> dict | list:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.get(f"{BASE}{path}")
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to CV-FLOW backend (localhost:8000). Is it running?"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

async def _post(path: str, body: dict = {}) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{BASE}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to CV-FLOW backend (localhost:8000). Is it running?"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

async def _delete(path: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.delete(f"{BASE}{path}")
            r.raise_for_status()
            if r.status_code == 204:
                return {"deleted": True}
            return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to CV-FLOW backend"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

# ── Pipeline tools ────────────────────────────────────────────────────────────

@mcp.tool()
async def list_pipelines() -> list[dict]:
    """List all saved CV-FLOW pipelines (id, name, node count)."""
    return await _get("/pipelines")

@mcp.tool()
async def get_pipeline(pipeline_id: str) -> dict:
    """Get full details of a pipeline including nodes and edges."""
    return await _get(f"/pipelines/{pipeline_id}")

@mcp.tool()
async def create_pipeline(name: str, nodes: list[dict], edges: list[dict]) -> dict:
    """Create a new pipeline. nodes and edges follow ReactFlow JSON format."""
    return await _post("/pipelines", {"name": name, "nodes": nodes, "edges": edges})

@mcp.tool()
async def update_pipeline(pipeline_id: str, nodes: list[dict], edges: list[dict], name: str | None = None) -> dict:
    """Update an existing pipeline's nodes and edges."""
    body = {"nodes": nodes, "edges": edges}
    if name:
        body["name"] = name
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.put(f"{BASE}/pipelines/{pipeline_id}", json=body)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()

@mcp.tool()
async def delete_pipeline(pipeline_id: str) -> dict:
    """Delete a pipeline."""
    return await _delete(f"/pipelines/{pipeline_id}")

# ── Execution tools ───────────────────────────────────────────────────────────

@mcp.tool()
async def run_pipeline(pipeline_id: str, mode: str = "sequential") -> dict:
    """Start executing a pipeline. mode: 'sequential' or 'multiprocess'. Returns session_id."""
    return await _post("/execution/start", {"pipeline_id": pipeline_id, "mode": mode})

@mcp.tool()
async def stop_pipeline(session_id: str) -> dict:
    """Stop a running pipeline session."""
    return await _post(f"/execution/stop/{session_id}")

@mcp.tool()
async def get_status(session_id: str) -> dict:
    """Get status of a pipeline session (running/stopped/completed/error)."""
    return await _get(f"/execution/status/{session_id}")

@mcp.tool()
async def get_logs(session_id: str, tail: int = 100) -> str:
    """Get last N lines of engine logs for a session."""
    data = await _get(f"/execution/logs/{session_id}?tail={tail}")
    if isinstance(data, list):
        return "\\n".join(data)  # API returns list[str] directly
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    return str(data)

@mcp.tool()
async def get_node_stats(session_id: str) -> dict:
    """Get per-node fps/latency stats for a session."""
    return await _get(f"/execution/stats/{session_id}")

@mcp.tool()
async def list_sessions() -> list[str]:
    """List all active session IDs."""
    return await _get("/execution/sessions")

# ── Model tools ───────────────────────────────────────────────────────────────

@mcp.tool()
async def list_models(task: str | None = None) -> list[dict]:
    """List available models. Optionally filter by task (detection/segmentation/pose/face_detect/face_embed)."""
    path = "/models"
    if task:
        path += f"?task={task}"
    return await _get(path)

@mcp.tool()
async def get_system_info() -> dict:
    """Get CPU, RAM, and GPU info of the host running CV-FLOW."""
    return await _get("/system/info")

# ── Data Hub tools ───────────────────────────────────────────────────────────

@mcp.tool()
async def get_model_ports(model_id: str) -> dict:
    """Get input/output port definitions for a specific model (tensor names, shapes, semantic types)."""
    data = await _get(f"/models/{model_id}")
    if isinstance(data, dict) and "error" not in data:
        ports = data.get("ports_json")
        if isinstance(ports, str):
            try:
                import json as _json
                return _json.loads(ports)
            except Exception:
                return {}
        return ports or {}
    return data

@mcp.tool()
async def list_datahub_tables() -> list[dict]:
    """List all Data Hub tables with their row counts."""
    return await _get("/datahub/relational/tables")

@mcp.tool()
async def list_execution_history(limit: int = 20, status: str | None = None) -> dict:
    """List pipeline execution history. status filter: 'running'|'completed'|'stopped'|'error'."""
    qs = f"?limit={limit}"
    if status:
        qs += f"&status={status}"
    return await _get(f"/datahub/relational/sessions{qs}")

@mcp.tool()
async def get_node_performance(session_id: str) -> dict:
    """Get per-node performance stats (fps, avg_ms, p95_ms, errors) for any session."""
    return await _get(f"/execution/stats/{session_id}")

# ── MCP Resources ─────────────────────────────────────────────────────────────

@mcp.resource("cvflow://pipelines")
async def resource_pipelines() -> str:
    """List of all pipelines as JSON."""
    data = await _get("/pipelines")
    return json.dumps(data, indent=2)

@mcp.resource("cvflow://pipeline/{pipeline_id}")
async def resource_pipeline(pipeline_id: str) -> str:
    """Full JSON content of a specific pipeline."""
    data = await _get(f"/pipelines/{pipeline_id}")
    return json.dumps(data, indent=2)

@mcp.resource("cvflow://session/{session_id}/logs")
async def resource_logs(session_id: str) -> str:
    """Engine logs for a session."""
    data = await _get(f"/execution/logs/{session_id}?tail=200")
    if isinstance(data, list):
        return "\\n".join(data)
    return str(data)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # stdio transport
