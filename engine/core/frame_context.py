"""FrameContext — the data bus passed between every node in the pipeline."""
from __future__ import annotations
import ctypes
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Python-side Detection object ──────────────────────────────────────────────

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int = -1
    metadata: dict = field(default_factory=dict)

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# ── Python FrameContext ────────────────────────────────────────────────────────

@dataclass
class FrameContext:
    frame: np.ndarray          # BGR uint8 C-contiguous [H × W × 3]
    frame_number: int
    timestamp: float
    session_id: str
    detections: list[Detection] = field(default_factory=list)
    metadata: dict[str, Any]   = field(default_factory=dict)

    def copy_frame(self) -> np.ndarray:
        return self.frame.copy()


# ── ctypes mirror of CVFlowCtx (must match shared/cpp/include/cv_flow/context.h) ──

CVFLOW_CLASS_NAME_LEN = 64
CVFLOW_DET_META_LEN   = 256
CVFLOW_SESSION_ID_LEN = 64
CVFLOW_METADATA_LEN   = 4096


class CVFlowDetectionC(ctypes.Structure):
    _fields_ = [
        ("x1",            ctypes.c_float),
        ("y1",            ctypes.c_float),
        ("x2",            ctypes.c_float),
        ("y2",            ctypes.c_float),
        ("confidence",    ctypes.c_float),
        ("class_id",      ctypes.c_int),
        ("class_name",    ctypes.c_char * CVFLOW_CLASS_NAME_LEN),
        ("track_id",      ctypes.c_int),
        ("metadata_json", ctypes.c_char * CVFLOW_DET_META_LEN),
    ]


class CVFlowCtxC(ctypes.Structure):
    _fields_ = [
        ("frame_data",        ctypes.POINTER(ctypes.c_uint8)),
        ("width",             ctypes.c_int),
        ("height",            ctypes.c_int),
        ("channels",          ctypes.c_int),
        ("frame_number",      ctypes.c_int),
        ("timestamp",         ctypes.c_double),
        ("session_id",        ctypes.c_char * CVFLOW_SESSION_ID_LEN),
        ("detections",        ctypes.POINTER(CVFlowDetectionC)),
        ("detection_count",   ctypes.c_int),
        ("detection_capacity",ctypes.c_int),
        ("metadata_json",     ctypes.c_char * CVFLOW_METADATA_LEN),
    ]
