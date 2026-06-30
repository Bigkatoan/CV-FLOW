"""
cv_flow.dam.cuda_bus — CudaPortBus: GPU-tensor-aware channel.

Despite the name, this does NOT do real CUDA IPC (zero-copy cross-process
GPU memory sharing) — every write/read round-trips the tensor through CPU
shared memory (PortBus). This was attempted for real using
torch.multiprocessing.reductions.reduce_tensor() (the standard mechanism
torch itself uses for cross-process CUDA tensor sharing) and confirmed NOT
viable on Jetson Orin Nano (JetPack 6, tested 2026-06-30): the rebuild call
in the receiving process fails with
`torch.AcceleratorError: CUDA error: invalid argument` from
`torch.UntypedStorage._new_shared_cuda`, consistently, regardless of
explicit `torch.cuda.set_device()`/`torch.cuda.init()` in the child. This
matches Jetson's integrated-GPU (unified host/device memory) architecture
not supporting the standard `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle`
mechanism, which is designed for discrete GPUs with separate VRAM
addressable over PCIe from multiple processes.

Practical effect: GPU-bound elastic worker processes on this hardware must
each do their own host<->device copy (exactly what this class already
does) rather than share one GPU buffer — there is currently no known
zero-copy alternative on Jetson without dropping to very low-level CUDA
driver calls (cudaHostAlloc/cudaHostGetDevicePointer) that are out of
scope here. Import this module unconditionally; check _CUDA_AVAILABLE.
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

        # No real CUDA IPC (see module docstring) — every tensor round-trips
        # through CPU shared memory, so the slot must fit the FULL tensor,
        # same as plain PortBus. (A previous version capped this at the
        # 64-byte size of a CUDA IPC handle and silently truncated any
        # tensor larger than that — fixed.)
        self._bus = PortBus(
            name, slot_bytes,
            queue_depth=queue_depth,
            create=create,
            drop_mode=drop_mode,
        )
        self._slot_bytes = slot_bytes

    # ── write ─────────────────────────────────────────────────────────────────

    # Reserved metadata key used to carry the original tensor's dtype/shape
    # across the CPU round-trip so read() can reconstruct it exactly
    # (a flat byte buffer alone loses both).
    _TENSOR_INFO_KEY = "__cv_flow_cuda_tensor__"

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

        If CUDA is available and data is a torch.Tensor, copies it to CPU
        and stores the *full* tensor bytes (not truncated — see module
        docstring for why this is not zero-copy real CUDA IPC) plus its
        dtype/shape (so read() can reconstruct it exactly). Otherwise
        writes raw bytes, same as plain PortBus.
        """
        det_list  = detections or []
        meta_dict = dict(metadata or {})

        if self.using_cuda:
            import torch
            if isinstance(data, torch.Tensor):
                cpu_tensor = data.cpu() if data.device.type != "cpu" else data
                raw = cpu_tensor.numpy().tobytes()
                if len(raw) > self._slot_bytes:
                    raise ValueError(
                        f"CudaPortBus '{self.name}': tensor is {len(raw)} bytes, "
                        f"exceeds slot_bytes={self._slot_bytes}"
                    )
                meta_dict[self._TENSOR_INFO_KEY] = {
                    "dtype": str(cpu_tensor.dtype).removeprefix("torch."),
                    "shape": list(cpu_tensor.shape),
                }
                return self._bus.write(raw, seq,
                                       detections=det_list, metadata=meta_dict)
            # non-tensor path falls through to raw bytes below
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
        If CUDA is available and the frame was written from a tensor,
        returns a torch.Tensor on self._device with the original
        dtype/shape restored. Otherwise returns raw bytes.
        """
        result = self._bus.read(timeout_ms=timeout_ms)
        if result is None:
            return None
        raw, seq, dets, meta = result

        tensor_info = meta.pop(self._TENSOR_INFO_KEY, None) if isinstance(meta, dict) else None
        if self.using_cuda and tensor_info is not None:
            import torch
            import numpy as np
            np_dtype = np.dtype(getattr(np, tensor_info["dtype"]))
            shape = tensor_info["shape"]
            n_elements = 1
            for dim in shape:
                n_elements *= dim
            n_bytes = n_elements * np_dtype.itemsize
            # raw is padded out to the bus's fixed slot_bytes — slice back
            # down to the tensor's actual byte length before reshaping.
            arr = np.frombuffer(raw[:n_bytes], dtype=np_dtype).reshape(shape).copy()
            tensor = torch.from_numpy(arr).to(self._device)
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
