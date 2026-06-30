"""
cv_flow.topic.publisher — Publisher: packs dict/ndarray → bytes → PortBus.
"""
from __future__ import annotations

import struct
from typing import Any, Union

import numpy as np

from cv_flow.dam.bus import PortBus, Detection
from cv_flow.topic.types import PortDef


class Publisher:
    """
    Packs topic data and writes to a PortBus.

    Supports:
      - dict {field_name: value}  — for multi-field topics
      - np.ndarray                — single-field shortcut
      - torch.Tensor              — converted to CPU bytes before writing

    Fields are packed sequentially in the order defined in PortDef.
    """

    def __init__(self, bus: PortBus, port_def: PortDef) -> None:
        self._bus      = bus
        self._port_def = port_def
        self._seq      = 0

    # ── write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        data:        Any,
        *,
        seq:         int | None = None,
        detections:  list[Detection] = [],
        metadata:    dict            = {},
    ) -> bool:
        """
        Pack data and write to bus.

        Parameters
        ----------
        data      : dict, np.ndarray, or torch.Tensor.
        seq       : Optional explicit sequence number. Auto-increments if None.
        detections: Optional Detection list.
        metadata  : Optional metadata dict.

        Returns
        -------
        True if a slot was dropped, False otherwise.

        Raises
        ------
        TypeError  if data is wrong type for the declared port.
        ValueError if array shape doesn't match field definition.
        """
        if seq is None:
            self._seq += 1
            seq = self._seq

        raw = self._pack(data)
        return self._bus.write(raw, seq, detections=detections, metadata=metadata)

    # ── packing ───────────────────────────────────────────────────────────────

    def _pack(self, data: Any) -> bytes:
        fields = self._port_def.fields

        # — Single-field shortcut: accept ndarray or tensor directly —
        if len(fields) == 1 and not isinstance(data, dict):
            return self._pack_field(fields[0], data)

        if not isinstance(data, dict):
            raise TypeError(
                f"Publisher has {len(fields)} fields; expected dict, "
                f"got {type(data).__name__}"
            )

        parts: list[bytes] = []
        for f in fields:
            if f.name not in data:
                raise KeyError(f"Field '{f.name}' missing from data dict")
            parts.append(self._pack_field(f, data[f.name]))
        return b"".join(parts)

    def _pack_field(self, field_def, value) -> bytes:
        from cv_flow.topic.types import FieldDef
        f = field_def

        # — torch.Tensor → numpy → bytes —
        try:
            import torch
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
        except ImportError:
            pass

        # — numpy scalar or ndarray —
        if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            arr = np.array(value, dtype=f.numpy_dtype)
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            raise TypeError(
                f"Field '{f.name}': expected ndarray or scalar, "
                f"got {type(value).__name__}"
            )

        arr = arr.astype(f.numpy_dtype, copy=False)

        if f.full_shape:
            if arr.shape != f.full_shape:
                raise ValueError(
                    f"Field '{f.name}': expected shape {f.full_shape}, "
                    f"got {arr.shape}"
                )
        else:
            arr = arr.reshape(())

        return arr.tobytes()

    # ── monitoring ────────────────────────────────────────────────────────────

    @property
    def seq(self) -> int:
        return self._seq

    def __repr__(self) -> str:
        return f"Publisher(bus={self._bus.name!r}, fields={len(self._port_def.fields)})"
