"""FrameContext — the data bus passed between every node in the pipeline."""
from __future__ import annotations
import ctypes
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from engine.core.dam import PortBus


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
    frame: np.ndarray | None    # CPU BGR uint8 — None if data lives only on GPU
    frame_number: int
    timestamp: float
    session_id: str
    detections: list[Detection] = field(default_factory=list)
    metadata: dict[str, Any]   = field(default_factory=dict)
    # GPU path — cv2.cuda.GpuMat or torch.Tensor.  None for CPU-only pipelines.
    frame_gpu: Any = field(default=None, repr=False)
    _gpu_valid: bool = field(default=False, repr=False, compare=False)

    # ── Device helpers ──────────────────────────────────────────────────────────

    @property
    def on_gpu(self) -> bool:
        """True when the authoritative copy is on the GPU (CPU copy may be stale)."""
        return self._gpu_valid and self.frame_gpu is not None

    def ensure_cpu(self) -> np.ndarray:
        """Return the CPU frame, downloading from GPU if needed (caches the result)."""
        if self.frame is not None:
            return self.frame
        if self._gpu_valid and self.frame_gpu is not None:
            # cv2.cuda.GpuMat
            if hasattr(self.frame_gpu, "download"):
                self.frame = self.frame_gpu.download()
                return self.frame
            # PyTorch tensor
            if hasattr(self.frame_gpu, "cpu"):
                t = self.frame_gpu.cpu()
                if t.dtype != np.uint8:
                    t = t.mul(255).byte()
                self.frame = t.numpy()
                return self.frame
        raise RuntimeError("FrameContext has no frame data (frame is None and no GPU frame)")

    def set_frame_gpu(self, gpu_mat: Any, *, invalidate_cpu: bool = True) -> None:
        """Store a GPU frame.  Pass invalidate_cpu=False to keep a stale CPU cache."""
        self.frame_gpu = gpu_mat
        self._gpu_valid = True
        if invalidate_cpu:
            self.frame = None

    def copy_frame(self) -> np.ndarray:
        return self.ensure_cpu().copy()

    # ── DAM integration ────────────────────────────────────────────────────────

    def to_bus(self, bus: "PortBus") -> None:
        """Serialize this context into a PortBus slot (zero-copy for C++)."""
        frame = self.ensure_cpu()
        bus.write(frame, self.detections, self.metadata)

    @classmethod
    def from_bus(
        cls,
        bus: "PortBus",
        session_id: str = "",
        timeout_ms: int = 30,
    ) -> "FrameContext | None":
        """
        Deserialize a FrameContext from a PortBus.
        Returns None when no new frame arrives within timeout_ms.
        """
        result = bus.read(timeout_ms=timeout_ms)
        if result is None:
            return None
        frame, detections, metadata = result
        frame_number = int(metadata.pop("__frame_number__", 0))
        timestamp    = float(metadata.pop("__timestamp__", 0.0))
        return cls(
            frame=frame,
            frame_number=frame_number,
            timestamp=timestamp,
            session_id=session_id,
            detections=detections,
            metadata=metadata,
        )

    @classmethod
    def from_buses(
        cls,
        buses: "dict[str, PortBus]",
        session_id: str = "",
        timeout_ms: int = 30,
    ) -> "FrameContext | None":
        """
        Deserialize from a named dict of PortBuses (port_name → bus).
        Primary bus must be named 'frame'.  Other buses contribute to metadata.
        """
        primary = buses.get("frame")
        if primary is None:
            return None
        ctx = cls.from_bus(primary, session_id=session_id, timeout_ms=timeout_ms)
        if ctx is None:
            return None
        for port_name, bus in buses.items():
            if port_name == "frame":
                continue
            extra = bus.read(timeout_ms=0)
            if extra is not None:
                _, extra_dets, extra_meta = extra
                ctx.metadata.update(extra_meta)
                ctx.detections.extend(extra_dets)
        return ctx

    def to_buses(self, buses: "dict[str, PortBus]") -> None:
        """
        Serialize context to a named dict of output PortBuses.
        The primary 'frame' bus carries frame + detections + metadata.
        """
        meta_with_frame_info = dict(self.metadata)
        meta_with_frame_info["__frame_number__"] = self.frame_number
        meta_with_frame_info["__timestamp__"]    = self.timestamp

        frame_bus = buses.get("frame")
        if frame_bus is not None:
            frame = self.ensure_cpu()
            frame_bus.write(frame, self.detections, meta_with_frame_info)


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
