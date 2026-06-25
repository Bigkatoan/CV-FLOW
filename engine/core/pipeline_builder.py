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

logger = logging.getLogger(__name__)

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
        else:
            logger.warning(
                "Unknown node type %r (node %s) — skipping. "
                "Only 'python_node' and 'cpp_node' are supported.",
                node_type, nid,
            )
            continue

        instance.setup(node_id=nid, config=node_config, node_type=node_type)
        ordered_nodes.append(instance)

    return ordered_nodes
