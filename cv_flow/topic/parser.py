"""
cv_flow.topic.parser — Parse .topic files into TopicDef.

File format (pure text, no external YAML library):

    # comment
    [elastic: true|false]
    [max_replicas: N]
    [queue_depth: N]
    [drop_mode: true|false]

    input: -> <device>
       - <name> : <dtype> shape=[<d>, ...]
       - ...

    output: -> <device>
       - <name> : <dtype> shape=[<d>, ...]
       - ...

Sections are optional: omitting "input:" makes a source topic (is_none input),
omitting "output:" makes a sink topic (is_none output).
"""
from __future__ import annotations

import re
from pathlib import Path

from cv_flow.topic.types import FieldDef, PortDef, TopicDef


class ParseError(ValueError):
    """Raised when a .topic file cannot be parsed."""
    def __init__(self, message: str, line_no: int | None = None) -> None:
        loc = f" (line {line_no})" if line_no is not None else ""
        super().__init__(f"ParseError{loc}: {message}")
        self.line_no = line_no


# ── Regex patterns ─────────────────────────────────────────────────────────────
_RE_SECTION     = re.compile(r"^(input|output)\s*:\s*->\s*(\S+)\s*$", re.I)
_RE_FIELD       = re.compile(
    r"^\s*-\s+(\w+)\s*:\s*(\S+)(?:\s+shape\s*=\s*\[([^\]]*)\])?\s*$"
)
_RE_OPTION_BOOL = re.compile(r"^(elastic|drop_mode)\s*:\s*(true|false)\s*$", re.I)
_RE_OPTION_INT  = re.compile(r"^(max_replicas|queue_depth)\s*:\s*(\d+)\s*$", re.I)


def _parse_shape(raw: str | None) -> tuple:
    """Parse "720, 1280" → (720, 1280); None → ()."""
    if not raw or not raw.strip():
        return ()
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_topic_file(path: str | Path) -> TopicDef:
    """
    Parse one .topic file and return a TopicDef.

    Parameters
    ----------
    path : Path to the .topic file. The topic name is derived from the stem.

    Raises
    ------
    ParseError if the file is malformed.
    FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    name = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()

    elastic      = False
    max_replicas = 4
    queue_depth  = 8
    drop_mode    = False

    input_device:  str | None      = None
    output_device: str | None      = None
    input_fields:  list[FieldDef]  = []
    output_fields: list[FieldDef]  = []
    current_section: str | None    = None

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        # — Skip blank lines and comments —
        if not stripped or stripped.startswith("#"):
            continue

        # — Section header: input: -> cpu —
        m = _RE_SECTION.match(stripped)
        if m:
            current_section = m.group(1).lower()
            device           = m.group(2)
            if current_section == "input":
                input_device = device
            else:
                output_device = device
            continue

        # — Field line: - name : dtype shape=[...] —
        m = _RE_FIELD.match(raw)   # use raw to preserve indentation match
        if m:
            if current_section is None:
                raise ParseError("Field declared outside any section", lineno)
            fname, dtype_str, shape_raw = m.group(1), m.group(2), m.group(3)
            try:
                f = FieldDef.build(fname, dtype_str, _parse_shape(shape_raw))
            except ValueError as exc:
                raise ParseError(str(exc), lineno) from exc
            if current_section == "input":
                input_fields.append(f)
            else:
                output_fields.append(f)
            continue

        # — Option: elastic: true —
        m = _RE_OPTION_BOOL.match(stripped)
        if m:
            key, val = m.group(1).lower(), m.group(2).lower() == "true"
            if key == "elastic":
                elastic = val
            else:
                drop_mode = val
            continue

        # — Option: max_replicas: 4 —
        m = _RE_OPTION_INT.match(stripped)
        if m:
            key, val = m.group(1).lower(), int(m.group(2))
            if key == "max_replicas":
                max_replicas = val
            else:
                queue_depth = val
            continue

        raise ParseError(f"Unexpected line: {stripped!r}", lineno)

    # — Build ports —
    if input_device is None:
        in_port = PortDef.none_port()
    else:
        in_port = PortDef(device=input_device, fields=input_fields)

    if output_device is None:
        out_port = PortDef.none_port()
    else:
        out_port = PortDef(device=output_device, fields=output_fields)

    return TopicDef(
        name=name,
        input_port=in_port,
        output_port=out_port,
        elastic=elastic,
        max_replicas=max_replicas,
        queue_depth=queue_depth,
        drop_mode=drop_mode,
    )


def load_topics_dir(directory: str | Path) -> dict[str, TopicDef]:
    """
    Load all *.topic files in a directory and return a {name → TopicDef} dict.

    Raises ParseError on the first malformed file.
    """
    directory = Path(directory)
    result: dict[str, TopicDef] = {}
    for topic_file in sorted(directory.glob("*.topic")):
        td = parse_topic_file(topic_file)
        result[td.name] = td
    return result
