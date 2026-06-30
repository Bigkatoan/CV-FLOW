"""
cv_flow.topic.subscriber — Subscriber: reads PortBus bytes → unpacks to dict/ndarray.
"""
from __future__ import annotations

from typing import Optional, Any

import numpy as np

from cv_flow.dam.bus import PortBus, Detection
from cv_flow.topic.types import PortDef


class Subscriber:
    """
    Reads from a PortBus and unpacks bytes into typed numpy arrays.

    Single-field topic → returns ndarray directly.
    Multi-field topic  → returns dict {name: ndarray}.
    """

    def __init__(
        self,
        bus:        PortBus,
        port_def:   PortDef,
        *,
        output_device: str = "cpu",
    ) -> None:
        self._bus           = bus
        self._port_def      = port_def
        self._output_device = output_device
        self._last_seq: int | None = None
        self._prev_seq: int | None = None  # seq before the most recent read

    # ── read ─────────────────────────────────────────────────────────────────

    def read(self, timeout_ms: int = 30) -> Optional[Any]:
        """
        Read the next frame and unpack it.

        Single-field topic → np.ndarray (or torch.Tensor if output_device is cuda).
        Multi-field topic  → dict {name: ndarray}.
        Returns None on timeout.
        """
        result = self._bus.read(timeout_ms=timeout_ms)
        if result is None:
            return None
        data, seq, _, _ = result
        self._prev_seq, self._last_seq = self._last_seq, seq
        return self._unpack(data)

    def read_full(
        self, timeout_ms: int = 30
    ) -> Optional[tuple[Any, list[Detection], dict, int]]:
        """
        Read and return (unpacked_data, detections, metadata, seq_no).
        Returns None on timeout.
        """
        result = self._bus.read(timeout_ms=timeout_ms)
        if result is None:
            return None
        data, seq, dets, meta = result
        self._prev_seq, self._last_seq = self._last_seq, seq
        return self._unpack(data), dets, meta, seq

    # ── unpack ────────────────────────────────────────────────────────────────

    def _unpack(self, raw: bytes) -> Any:
        fields = self._port_def.fields
        offset = 0
        results: dict[str, np.ndarray] = {}

        for f in fields:
            chunk = raw[offset : offset + f.n_bytes]
            arr   = np.frombuffer(chunk, dtype=f.numpy_dtype)
            if f.full_shape:
                arr = arr.reshape(f.full_shape)
            else:
                arr = arr.reshape(())
            results[f.name] = arr.copy()
            offset += f.n_bytes

        # Move to target device if CUDA requested
        if self._output_device.startswith("cuda"):
            try:
                import torch
                results = {
                    k: torch.from_numpy(v.copy()).to(self._output_device)
                    for k, v in results.items()
                }
            except ImportError:
                pass

        if len(fields) == 1:
            return results[fields[0].name]
        return results

    # ── seq gap detection ─────────────────────────────────────────────────────

    @property
    def last_seq(self) -> int | None:
        return self._last_seq

    def has_seq_gap(self, current_seq: int) -> bool:
        """Return True if current_seq is not consecutive with the PREVIOUS read.

        Checks against _prev_seq (the seq before the most recent read_full/read),
        so calling this right after a read is meaningful.
        """
        if self._prev_seq is None:
            return False
        return current_seq != self._prev_seq + 1

    def __repr__(self) -> str:
        return (f"Subscriber(bus={self._bus.name!r}, "
                f"fields={len(self._port_def.fields)}, "
                f"device={self._output_device!r})")
