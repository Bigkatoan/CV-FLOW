"""Parse PipelineJSON and build an ordered list of node instances via topological sort (Kahn's algorithm)."""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Any

from engine.nodes.base import BaseNode
from engine.core.node_registry import get_node_class


def build_pipeline(pipeline_json: dict[str, Any]) -> list[BaseNode]:
    """
    Returns nodes in topological execution order.
    Raises ValueError if the graph has cycles or references unknown node types.
    """
    nodes_data: list[dict] = pipeline_json.get("nodes", [])
    edges_data: list[dict] = pipeline_json.get("edges", [])

    # Build adjacency map: node_id → list of successor node_ids
    successors: dict[str, list[str]] = defaultdict(list)
    in_degree:  dict[str, int]       = {n["id"]: 0 for n in nodes_data}

    for edge in edges_data:
        src = edge["source"]
        dst = edge["target"]
        successors[src].append(dst)
        in_degree[dst] = in_degree.get(dst, 0) + 1

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

    # Build node map for fast lookup
    node_map = {n["id"]: n for n in nodes_data}

    # Instantiate nodes in order
    ordered_nodes: list[BaseNode] = []
    for nid in ordered_ids:
        node_data = node_map[nid]
        node_type = node_data["type"]
        node_config = node_data.get("config", {})
        cls = get_node_class(node_type)
        instance = cls()
        instance.setup(node_id=nid, config=node_config)
        ordered_nodes.append(instance)

    return ordered_nodes
