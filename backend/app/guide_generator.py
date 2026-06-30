"""
backend.app.guide_generator — Generate an AI deployment guide (Markdown)
from a Pipeline Spec.

The output is meant to be precise enough that an AI assistant (or a human)
can implement the pipeline's topic files and launch script without
needing to look anything else up.
"""
from __future__ import annotations

from backend.app.schemas import PipelineSpec, TopicSpec
from cv_flow.nodes._catalog import NODE_CATALOG


def _field_lines(fields) -> list[str]:
    lines = []
    for f in fields:
        shape = f"shape=[{', '.join(str(d) for d in f.shape)}]" if f.shape else ""
        lines.append(f"   - {f.name} : {f.dtype} {shape}".rstrip())
    return lines


def _render_topic_file(topic: TopicSpec) -> str:
    lines: list[str] = []
    if topic.elastic:
        lines.append("elastic: true")
        lines.append(f"max_replicas: {topic.max_replicas}")
    if topic.queue_depth != 8:
        lines.append(f"queue_depth: {topic.queue_depth}")
    if topic.drop_mode:
        lines.append("drop_mode: true")
    if lines:
        lines.append("")

    if topic.fields_in or topic.input_device:
        lines.append(f"input: -> {topic.input_device or 'cpu'}")
        lines.extend(_field_lines(topic.fields_in))
        lines.append("")
    if topic.fields_out or topic.output_device:
        lines.append(f"output: -> {topic.output_device or 'cpu'}")
        lines.extend(_field_lines(topic.fields_out))
    elif topic.fields and not topic.fields_in:
        # Single-port topic (source or sink) declared via device + fields.
        lines.append(f"output: -> {topic.device or 'cpu'}")
        lines.extend(_field_lines(topic.fields))

    return "\n".join(lines).strip() + "\n"


def generate_deployment_guide(spec: PipelineSpec) -> str:
    """Render a complete Markdown deployment guide for `spec`."""
    md: list[str] = []

    md.append(f"# Pipeline: {spec.name}")
    md.append("")
    if spec.description:
        md.append(spec.description)
        md.append("")

    node_chain = " -> ".join(n.type for n in spec.nodes)
    md.append("## Overview")
    md.append(f"{len(spec.nodes)}-node pipeline: {node_chain}")
    md.append("")

    md.append("## Step 1: Create the project structure")
    md.append("```bash")
    md.append("mkdir -p my_pipeline/topics")
    md.append("```")
    md.append("")

    md.append("## Step 2: Create topic files")
    md.append("")
    for topic in spec.topics:
        md.append(f"### `topics/{topic.name}.topic`")
        md.append("```")
        md.append(_render_topic_file(topic).rstrip())
        md.append("```")
        md.append("")

    md.append("## Step 3: Node configuration reference")
    md.append("")
    for node in spec.nodes:
        meta = NODE_CATALOG.get(node.type, {})
        md.append(f"### `{node.id}` ({node.type})")
        if meta.get("description"):
            md.append(meta["description"])
        if node.config:
            md.append("")
            md.append("Configuration:")
            for k, v in node.config.items():
                md.append(f"- `{k}` = `{v}`")
        if node.connections_in:
            md.append("")
            md.append("Subscribes to: " +
                       ", ".join(f"`{c.topic}`" for c in node.connections_in))
        if node.connections_out:
            md.append("Publishes to: " +
                       ", ".join(f"`{c.topic}`" for c in node.connections_out))
        md.append("")

    md.append("## Step 4: Write launch.py")
    md.append("```python")
    md.append("import cv_flow")
    md.append("")
    md.append('cv_flow.load_topics("topics/")')
    md.append("")
    for node in spec.nodes:
        md.append(f"# {node.id}: {node.type}")
        md.append(f"# from cv_flow.nodes import {node.type}")
    md.append("")
    md.append("executor = cv_flow.Executor([...])  # instantiate nodes above, in order")
    md.append("executor.spin()")
    md.append("```")
    md.append("")

    md.append("## Step 5: Run")
    md.append("```bash")
    md.append("cv-flow run my_pipeline/launch.py")
    md.append("```")

    return "\n".join(md) + "\n"
