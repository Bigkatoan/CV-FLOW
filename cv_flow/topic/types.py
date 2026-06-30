"""
cv_flow.topic.types — DTYPE_MAP, FieldDef, PortDef, TopicDef dataclasses.

ROS2-compatible dtype strings → numpy dtype + shape expansion rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ── DTYPE_MAP ─────────────────────────────────────────────────────────────────
# Maps dtype string → {"numpy": np.dtype, "expand": fn(base_shape) → full_shape}

def _same(s):    return tuple(s)
def _append3(s): return (*tuple(s), 3)
def _append4(s): return (*tuple(s), 4)


DTYPE_MAP: dict[str, dict] = {
    # Image formats
    "bgr8":    {"numpy": np.dtype(np.uint8),   "expand": _append3},
    "rgb8":    {"numpy": np.dtype(np.uint8),   "expand": _append3},
    "bgra8":   {"numpy": np.dtype(np.uint8),   "expand": _append4},
    "rgba8":   {"numpy": np.dtype(np.uint8),   "expand": _append4},
    "mono8":   {"numpy": np.dtype(np.uint8),   "expand": _same},
    "16UC1":   {"numpy": np.dtype(np.uint16),  "expand": _same},
    "32FC1":   {"numpy": np.dtype(np.float32), "expand": _same},
    "32FC3":   {"numpy": np.dtype(np.float32), "expand": _append3},
    # Generic numeric
    "float32": {"numpy": np.dtype(np.float32), "expand": _same},
    "float64": {"numpy": np.dtype(np.float64), "expand": _same},
    "float16": {"numpy": np.dtype(np.float16), "expand": _same},
    "int8":    {"numpy": np.dtype(np.int8),    "expand": _same},
    "int16":   {"numpy": np.dtype(np.int16),   "expand": _same},
    "int32":   {"numpy": np.dtype(np.int32),   "expand": _same},
    "int64":   {"numpy": np.dtype(np.int64),   "expand": _same},
    "uint8":   {"numpy": np.dtype(np.uint8),   "expand": _same},
    "uint16":  {"numpy": np.dtype(np.uint16),  "expand": _same},
    "uint32":  {"numpy": np.dtype(np.uint32),  "expand": _same},
    "uint64":  {"numpy": np.dtype(np.uint64),  "expand": _same},
    "bool":    {"numpy": np.dtype(np.bool_),   "expand": _same},
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FieldDef:
    """One field in a topic port (e.g. "frame : bgr8 shape=[720, 1280]")."""
    name:        str
    dtype_str:   str
    numpy_dtype: np.dtype
    base_shape:  tuple        # as declared in .topic file
    full_shape:  tuple        # after dtype expansion (e.g. bgr8 adds channel dim)
    n_bytes:     int          # total bytes for one element

    @classmethod
    def build(cls, name: str, dtype_str: str, base_shape: tuple) -> "FieldDef":
        if dtype_str not in DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype '{dtype_str}'. "
                f"Valid dtypes: {', '.join(sorted(DTYPE_MAP))}"
            )
        entry      = DTYPE_MAP[dtype_str]
        numpy_dt   = entry["numpy"]
        full_shape = entry["expand"](base_shape)
        n_bytes    = int(np.prod(full_shape) if full_shape else 1) * numpy_dt.itemsize
        return cls(
            name=name,
            dtype_str=dtype_str,
            numpy_dtype=numpy_dt,
            base_shape=base_shape,
            full_shape=full_shape,
            n_bytes=n_bytes,
        )


@dataclass
class PortDef:
    """One side (input or output) of a topic."""
    device:      str               # "cpu", "cuda:0", etc.
    fields:      list[FieldDef]   = field(default_factory=list)
    is_none:     bool              = False   # True for source output-only / sink input-only

    @property
    def total_bytes(self) -> int:
        return sum(f.n_bytes for f in self.fields)

    @classmethod
    def none_port(cls) -> "PortDef":
        return cls(device="", fields=[], is_none=True)


@dataclass
class TopicDef:
    """Complete definition of a DAM topic (parsed from a .topic file)."""
    name:         str
    input_port:   PortDef
    output_port:  PortDef
    elastic:      bool = False
    max_replicas: int  = 4
    queue_depth:  int  = 8
    drop_mode:    bool = False
