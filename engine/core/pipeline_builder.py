"""
Parse PipelineJSON and build an ordered list of node instances via topological sort (Kahn's algorithm).

All nodes are either PythonCodeNode (type="python_node") or CppCodeNode (type="cpp_node").
Template nodes (type starts with "tmpl_") are expanded inline before sorting.
"""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Any
import logging

from engine.nodes.base import BaseNode
from engine.nodes.python_code_node import PythonCodeNode
from engine.nodes.cpp.cpp_node import CppCodeNode
from engine.core.node_registry import get_node_class, get_registry
import importlib
import pkgutil
import textwrap

logger = logging.getLogger(__name__)

_NODES_DISCOVERED = False

def _ensure_nodes_discovered() -> None:
    """Lazy import all engine node modules once to populate @register registry."""
    global _NODES_DISCOVERED
    if _NODES_DISCOVERED:
        return
    import engine.nodes
    for finder, mod_name, ispkg in pkgutil.walk_packages(
        path=engine.nodes.__path__,
        prefix="engine.nodes.",
        onerror=lambda x: None,
    ):
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            logger.debug("Node discovery: failed to import %s: %s", mod_name, e)
    _NODES_DISCOVERED = True
    logger.info("Node registry populated: %d types", len(get_registry()))

def _build_model_node(node_id: str, config: dict) -> PythonCodeNode | None:
    """Generate a PythonCodeNode wrapper for an ONNX model inference."""
    model_id = config.get("model_id")
    if not model_id:
        logger.warning("model_node %r missing model_id", node_id)
        return None
    
    ports_json = config.get("ports", {})
    inputs = ports_json.get("inputs", [])
    outputs = ports_json.get("outputs", [])
    
    # Auto-generate python code for the node
    code = [
        "import onnxruntime as ort",
        "import os",
        "from pathlib import Path",
        "",
        "session = None",
        "",
        "def setup():",
        "    global session",
        f'    model_dir = Path(os.environ.get("CVFLOW_MODELS_DIR", "."))',
        f'    model_path = model_dir / "{model_id}"',
        '    if not model_path.exists():',
        f'        raise FileNotFoundError(f"Model file not found: {{model_path}}")',
        '    # Fallback to CPU if CUDA/TensorRT not available',
        '    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]',
        '    session = ort.InferenceSession(str(model_path), providers=providers)',
        "",
        "def process(ctx):",
        "    # Prepare inputs",
        "    ort_inputs = {}",
    ]
    
    for inp in inputs:
        # ctx.get('port_name') defaults to None
        code.append(f'    val_{inp["name"]} = ctx.get("{inp["name"]}")')
        code.append(f'    if val_{inp["name"]} is not None:')
        code.append(f'        ort_inputs["{inp["tensor_name"]}"] = val_{inp["name"]}')
        
    code.append("")
    code.append("    if not ort_inputs:")
    code.append("        return  # No inputs ready")
    code.append("")
    code.append("    # Run inference")
    out_names = [out["tensor_name"] for out in outputs]
    code.append(f'    ort_outs = session.run({out_names}, ort_inputs)')
    code.append("")
    code.append("    # Map outputs back to FrameContext")
    
    for i, out in enumerate(outputs):
        code.append(f'    ctx.set("{out["name"]}", ort_outs[{i}])')
        
    code_str = "\n".join(code)
    
    # Create a PythonCodeNode and inject the generated code
    config_copy = dict(config)
    config_copy["code"] = code_str
    
    instance = PythonCodeNode()
    # We delay calling setup() until build_pipeline loop
    return instance

# Frontend-only marker types — never instantiated in the engine
_MARKER_TYPES = {"pipeline_output", "pipeline_input"}


def _expand_templates(pipeline_json: dict[str, Any]) -> dict[str, Any]:
    """
    Replace every template node (type starts with 'tmpl_') with the nodes/edges
    embedded in its config.pipeline_json.  Expansion is iterative so nested
    templates are handled automatically.
    """
    nodes: list[dict] = list(pipeline_json.get("nodes", []))
    edges: list[dict] = list(pipeline_json.get("edges", []))

    changed = True
    while changed:
        changed = False
        new_nodes: list[dict] = []
        for node in nodes:
            if not node["type"].startswith("tmpl_"):
                new_nodes.append(node)
                continue

            sub = node.get("config", {}).get("pipeline_json") or {}
            sub_nodes_raw: list[dict] = sub.get("nodes", [])
            sub_edges_raw: list[dict] = sub.get("edges", [])

            if not sub_nodes_raw:
                logger.warning(
                    "Template node %r has no embedded pipeline_json — skipping expansion",
                    node["id"],
                )
                new_nodes.append(node)
                continue

            tmpl_id = node["id"]
            prefix  = tmpl_id + "__"

            id_map = {n["id"]: prefix + n["id"] for n in sub_nodes_raw}

            sub_targets = {e["target"] for e in sub_edges_raw}
            entry_ids   = [n["id"] for n in sub_nodes_raw if n["id"] not in sub_targets]

            sub_sources = {e["source"] for e in sub_edges_raw}
            po_ids      = {n["id"] for n in sub_nodes_raw if n["type"] == "pipeline_output"}
            exit_ids    = [
                n["id"] for n in sub_nodes_raw
                if n["id"] not in sub_sources and n["type"] != "pipeline_output"
            ]
            if not exit_ids:
                exit_ids = [e["source"] for e in sub_edges_raw if e["target"] in po_ids]

            new_edges: list[dict] = []
            for e in edges:
                src, tgt = e["source"], e["target"]
                if tgt == tmpl_id:
                    for eid in entry_ids:
                        new_edges.append({**e,
                            "id": e["id"] + "_xe_" + eid,
                            "target": id_map.get(eid, eid),
                        })
                elif src == tmpl_id:
                    for eid in exit_ids:
                        new_edges.append({**e,
                            "id": e["id"] + "_xe_" + eid,
                            "source": id_map.get(eid, eid),
                        })
                else:
                    new_edges.append(e)
            edges = new_edges

            for e in sub_edges_raw:
                if e["source"] in id_map and e["target"] in id_map:
                    edges.append({**e,
                        "id": prefix + e["id"],
                        "source": id_map[e["source"]],
                        "target": id_map[e["target"]],
                    })

            for n in sub_nodes_raw:
                if n["type"] == "pipeline_output":
                    continue
                new_nodes.append({**n, "id": id_map[n["id"]]})

            logger.info(
                "Expanded template %r → %d sub-nodes",
                tmpl_id, len(sub_nodes_raw) - len(po_ids),
            )
            changed = True

        nodes = new_nodes

    return {"nodes": nodes, "edges": edges}


def build_pipeline(pipeline_json: dict[str, Any]) -> list[BaseNode]:
    """
    Returns nodes in topological execution order.
    Raises ValueError if the graph has cycles or references unknown node types.
    Template nodes (type starts with 'tmpl_') are expanded inline before sorting.
    """
    _ensure_nodes_discovered()
    _registered = get_registry()
    
    pipeline_json = _expand_templates(pipeline_json)
    nodes_data: list[dict] = pipeline_json.get("nodes", [])
    edges_data: list[dict] = pipeline_json.get("edges", [])

    # Build adjacency map: node_id → list of successor node_ids
    successors: dict[str, list[str]] = defaultdict(list)
    in_degree:  dict[str, int]       = {n["id"]: 0 for n in nodes_data}

    for edge in edges_data:
        src = edge["source"]
        dst = edge["target"]
        if src not in in_degree or dst not in in_degree:
            continue  # skip edges referencing unknown nodes (template expansion artefacts)
        successors[src].append(dst)
        in_degree[dst] += 1

    # Kahn's topological sort
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    ordered_ids: list[str] = []

    while queue:
        nid = queue.popleft()
        ordered_ids.append(nid)
        for succ in successors[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(ordered_ids) != len(nodes_data):
        raise ValueError("Pipeline graph contains a cycle — cannot execute")

    node_map = {n["id"]: n for n in nodes_data}

    ordered_nodes: list[BaseNode] = []
    for nid in ordered_ids:
        node_data  = node_map[nid]
        node_type  = node_data["type"]
        node_config = node_data.get("config", {})

        if node_type in _MARKER_TYPES:
            continue

        if node_type == "python_node":
            instance = PythonCodeNode()
        elif node_type == "cpp_node":
            instance = CppCodeNode()
        elif node_type == "model_node":
            instance = _build_model_node(nid, node_config)
            if instance is None:
                continue
        elif node_type in _registered:
            klass = get_node_class(node_type)
            instance = klass()
        else:
            logger.warning(
                "Unknown node type %r (node %s) — skipping. ",
                node_type, nid,
            )
            continue

        instance.setup(node_id=nid, config=node_config, node_type=node_type)
        ordered_nodes.append(instance)

    return ordered_nodes
