"""
Python ↔ C++ zero-copy bridge.

Converts FrameContext Python object ↔ CVFlowCtxC ctypes struct.
The frame numpy array is passed by pointer (zero-copy) so C++ can
modify it in-place and the result is immediately visible in Python.
"""
from __future__ import annotations
import ctypes
import json

import numpy as np

from engine.core.frame_context import (
    FrameContext, Detection,
    CVFlowCtxC, CVFlowDetectionC,
    CVFLOW_CLASS_NAME_LEN, CVFLOW_SESSION_ID_LEN, CVFLOW_METADATA_LEN,
)

_MAX_DETECTIONS = 512   # Pre-allocated C detection array size


class CppBridge:
    """
    Holds the pre-allocated C arrays so they don't get GC'd between frames.
    Create one instance per CppNode and reuse it every frame.
    """

    def __init__(self):
        self._det_array = (CVFlowDetectionC * _MAX_DETECTIONS)()
        self._c_ctx = CVFlowCtxC()
        self._c_ctx.detection_capacity = _MAX_DETECTIONS
        self._c_ctx.detections = self._det_array

    def python_to_c(self, ctx: FrameContext) -> CVFlowCtxC:
        """Fill the cached CVFlowCtxC from a Python FrameContext. Zero-copy for frame."""
        frame = ctx.frame
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
            ctx.frame = frame   # update in-place so caller sees contiguous version

        h, w, c = frame.shape
        self._c_ctx.frame_data = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._c_ctx.width      = w
        self._c_ctx.height     = h
        self._c_ctx.channels   = c
        self._c_ctx.frame_number = ctx.frame_number
        self._c_ctx.timestamp    = ctx.timestamp
        sid = ctx.session_id.encode()[:CVFLOW_SESSION_ID_LEN - 1]
        ctypes.memmove(self._c_ctx.session_id, sid, len(sid))
        self._c_ctx.session_id[len(sid)] = 0

        # Marshal Python detections → C array
        n = min(len(ctx.detections), _MAX_DETECTIONS)
        for i, d in enumerate(ctx.detections[:n]):
            cd = self._det_array[i]
            cd.x1 = d.x1; cd.y1 = d.y1; cd.x2 = d.x2; cd.y2 = d.y2
            cd.confidence = d.confidence
            cd.class_id   = d.class_id
            cd.track_id   = d.track_id
            cname = d.class_name.encode()[:CVFLOW_CLASS_NAME_LEN - 1]
            ctypes.memmove(cd.class_name, cname, len(cname))
            cd.class_name[len(cname)] = 0
            cd.metadata_json[0] = 0
        self._c_ctx.detection_count = n

        # Metadata JSON bus
        meta_str = json.dumps(ctx.metadata)[:CVFLOW_METADATA_LEN - 1].encode()
        ctypes.memmove(self._c_ctx.metadata_json, meta_str, len(meta_str))
        self._c_ctx.metadata_json[len(meta_str)] = 0

        return self._c_ctx

    def c_to_python(self, ctx: FrameContext) -> FrameContext:
        """Read back C changes into the Python FrameContext.
        Frame memory is shared — no copy needed for the frame itself.
        """
        # Read back detections (C++ may have changed count or individual fields)
        n = self._c_ctx.detection_count
        updated: list[Detection] = []
        for i in range(min(n, _MAX_DETECTIONS)):
            cd = self._det_array[i]
            updated.append(Detection(
                x1=cd.x1, y1=cd.y1, x2=cd.x2, y2=cd.y2,
                confidence=cd.confidence,
                class_id=cd.class_id,
                class_name=cd.class_name.decode(errors="replace").rstrip("\x00"),
                track_id=cd.track_id,
            ))
        ctx.detections = updated

        # Read back metadata JSON
        meta_bytes = bytes(self._c_ctx.metadata_json).rstrip(b"\x00")
        if meta_bytes:
            try:
                ctx.metadata = json.loads(meta_bytes)
            except json.JSONDecodeError:
                pass   # Keep old metadata if C++ wrote invalid JSON

        return ctx
