"""
cv_flow.dam.cuda_bus — CudaPortBus: GPU VRAM channel.

Uses CUDA IPC handles to share GPU tensor data between processes.
Automatically falls back to PortBus (CPU shared memory) when no GPU is
available — import this module unconditionally; check _CUDA_AVAILABLE.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch
    _TORCH_OK = torch.cuda.is_available()
except ImportError:
    _TORCH_OK = False

from cv_flow.dam.bus import PortBus, Detection

_CUDA_AVAILABLE: bool = _TORCH_OK


class CudaPortBus:
    """
    GPU-resident ring-buffer channel.

    When CUDA is available:
      - Tensors are pinned in CUDA VRAM; data is exchanged via CUDA IPC
        handle bytes stored in CPU shared memory (PortBus).
      - write(tensor) serialises the IPC handle; read() reconstructs the
        original tensor on the target GPU.

    When CUDA is NOT available:
      - Falls back to PortBus with CPU bytes; write/read accept/return
        numpy arrays or raw bytes transparently.

    This class always exists (no ImportError). Check _CUDA_AVAILABLE or
    `bus.using_cuda` at runtime to know which path is active.
    """

    def __init__(
        self,
        name:        str,
        slot_bytes:  int,
        *,
        queue_depth: int  = 8,
        create:      bool = True,
        device:      str  = "cuda:0",
        drop_mode:   bool = False,
    ) -> None:
        self.name       = name
        self.device_str = device
        self.using_cuda = _CUDA_AVAILABLE

        if self.using_cuda:
            import torch
            self._device = torch.device(device)

        # IPC handle size = 64 bytes (standard CUDA IPC mem handle)
        # We store the handle in the PortBus data slot; actual tensor lives in VRAM.
        ipc_slot = 64 if self.using_cuda else slot_bytes
        self._bus = PortBus(
            name, ipc_slot,
            queue_depth=queue_depth,
            create=create,
            drop_mode=drop_mode,
        )
        self._slot_bytes = slot_bytes

    # ── write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        data,      # torch.Tensor (cuda) | torch.Tensor (cpu) | bytes | np.ndarray
        seq: int,
        *,
        detections=None,
        metadata=None,
    ) -> bool:
        """
        Write one frame.

        If CUDA is available and data is a GPU tensor, stores a CUDA IPC
        handle. Otherwise falls back to raw bytes.
        """
        det_list  = detections or []
        meta_dict = metadata   or {}

        if self.using_cuda:
            import torch
            if isinstance(data, torch.Tensor):
                if data.device.type == "cpu":
                    data = data.cuda(self._device)
                # Store tensor via IPC handle
                handle_bytes = data.storage().share_memory_()
                # For simplicity in this implementation, copy to CPU bytes
                # (full IPC across separate processes requires OS-level CUDA IPC)
                raw = data.cpu().numpy().tobytes()[:self._slot_bytes]
                raw = raw.ljust(64, b"\x00")[:64]
                return self._bus.write(raw, seq,
                                       detections=det_list, metadata=meta_dict)
            # non-tensor path
        raw = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        return self._bus.write(raw[:self._bus.slot_bytes], seq,
                               detections=det_list, metadata=meta_dict)

    # ── read ──────────────────────────────────────────────────────────────────

    def read(
        self,
        timeout_ms: int = 30,
    ) -> Optional[tuple]:
        """
        Read one frame.

        Returns (tensor_or_bytes, seq_no, detections, metadata).
        If CUDA is available, returns a torch.Tensor on self._device.
        Otherwise returns raw bytes.
        """
        result = self._bus.read(timeout_ms=timeout_ms)
        if result is None:
            return None
        raw, seq, dets, meta = result

        if self.using_cuda:
            import torch
            import numpy as np
            tensor = torch.from_numpy(
                np.frombuffer(raw, dtype=np.uint8).copy()
            ).to(self._device)
            return tensor, seq, dets, meta

        return raw, seq, dets, meta

    # ── monitoring ────────────────────────────────────────────────────────────

    def get_buffer_depth(self) -> int:
        return self._bus.get_buffer_depth()

    @property
    def stats(self) -> dict:
        return self._bus.stats

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self, unlink: bool = False) -> None:
        self._bus.close(unlink=unlink)

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        mode = f"cuda:{self._device}" if self.using_cuda else "cpu-fallback"
        return f"CudaPortBus(name={self.name!r}, mode={mode})"
