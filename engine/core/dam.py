"""
Direct Access Memory (DAM) — zero-copy inter-node communication.

PortBus wraps a named SharedMemory region so two OS processes can exchange
frames without any data copy. The binary layout in shared memory matches the
C struct in shared/cpp/include/cv_flow/port_bus.h, so Python and C++ nodes
can be connected to the same PortBus transparently.

Layout per PortBus (pre-allocated at creation, never reallocated):
  [BusHeader  :   64 bytes]   ← seq_no, dims, counters
  [Frame      :   W×H×C    ]  ← BGR uint8 or float32 CHW
  [Detections : 512×348    ]  ← CVFlowDetection structs (C-compatible)
  [Metadata   : 4096 bytes ]  ← UTF-8 JSON string

Concurrency model:
  Single writer + single reader per PortBus (no locking in hot path).
  Writer increments seq_no AFTER writing data; reader compares seq_no to its
  cached last_seq to detect new frames (non-blocking).
  write_count / read_count in the header let AutoScaler compute buffer depth.

  For one-writer-many-readers: use RoundRobinBus (fan-out).
  For many-writers-one-reader: use MergeBus (fan-in).
"""
from __future__ import annotations

import json
import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from engine.core.frame_context import (
    CVFLOW_CLASS_NAME_LEN,
    CVFLOW_DET_META_LEN,
    CVFLOW_METADATA_LEN,
    Detection,
)

# ── Header layout — 64 bytes, little-endian ───────────────────────────────────
# Must match CVFlowBusHeader in shared/cpp/include/cv_flow/port_bus.h
#
#  Offset  Type      Field         Description
#   0      uint64    seq_no        Monotonic write counter (reader detects new frame)
#   8      int32     width         Frame width actually written (≤ max_w)
#  12      int32     height        Frame height actually written (≤ max_h)
#  16      uint8     channels      1 or 3
#  17      uint8     flags         Bit 0=FRAME_VALID, 1=DET_VALID, 2=META_VALID
#  18      uint16    det_count     Number of valid detections (≤ MAX_DETS)
#  20      uint32    meta_len      Bytes of valid JSON in metadata region
#  24      uint64    write_count   Total frames written (producer increments)
#  32      uint64    read_count    Total frames consumed (consumer increments)
#  40      bytes[24] reserved      Future use / padding
_HDR_FMT  = "<QiiBBHIQQ24s"
_HDR_SIZE = struct.calcsize(_HDR_FMT)
assert _HDR_SIZE == 64, f"Header size must be 64, got {_HDR_SIZE}"

# Detection layout (must match CVFlowDetection in context.h)
#  5×float  x1,y1,x2,y2,confidence   20 bytes
#  int32    class_id                   4 bytes
#  char[64] class_name                64 bytes
#  int32    track_id                   4 bytes
#  char[256] metadata_json           256 bytes
#  Total: 348 bytes
_DET_FMT    = f"<5fi{CVFLOW_CLASS_NAME_LEN}si{CVFLOW_DET_META_LEN}s"
_DET_STRIDE = struct.calcsize(_DET_FMT)
assert _DET_STRIDE == 348, f"Detection struct size must be 348, got {_DET_STRIDE}"

MAX_DETS = 512

FLAG_FRAME_VALID: int = 0x01
FLAG_DET_VALID:   int = 0x02
FLAG_META_VALID:  int = 0x04


def _shm_size(max_w: int, max_h: int, max_c: int = 3) -> int:
    return _HDR_SIZE + max_w * max_h * max_c + MAX_DETS * _DET_STRIDE + CVFLOW_METADATA_LEN


def _encode_class_name(name: str) -> bytes:
    b = name.encode("utf-8", errors="replace")[: CVFLOW_CLASS_NAME_LEN - 1]
    return b.ljust(CVFLOW_CLASS_NAME_LEN, b"\x00")


def _encode_meta_json(d: dict) -> bytes:
    b = json.dumps(d, default=str).encode("utf-8")[: CVFLOW_DET_META_LEN - 1]
    return b.ljust(CVFLOW_DET_META_LEN, b"\x00")


# ── PortBus ───────────────────────────────────────────────────────────────────

class PortBus:
    """
    Zero-copy CPU frame channel backed by POSIX/Windows shared memory.

    A PortBus has ONE frame slot — the producer always overwrites with the
    latest frame.  This gives low-latency 'newest frame wins' semantics ideal
    for live video.  Buffer depth = write_count − read_count tracks how far
    behind the consumer is, which the AutoScaler uses to trigger scale-up.

    Parameters
    ----------
    name:    Unique shared memory name (no slashes; Python adds platform prefix).
    max_w:   Maximum frame width pre-allocated.  Actual frame may be smaller.
    max_h:   Maximum frame height pre-allocated.
    max_c:   Maximum channels (3 for BGR, 1 for grayscale).
    create:  True  → allocate new region (creator process).
             False → attach to existing region (consumer process).
    """

    def __init__(
        self,
        name: str,
        max_w: int = 1280,
        max_h: int = 720,
        max_c: int = 3,
        *,
        create: bool = True,
    ) -> None:
        self.name  = name
        self.max_w = max_w
        self.max_h = max_h
        self.max_c = max_c

        size = _shm_size(max_w, max_h, max_c)
        if create:
            self._shm = SharedMemory(name=name, create=True, size=size)
            # Zero header so readers see no valid frame initially
            self._shm.buf[:_HDR_SIZE] = b"\x00" * _HDR_SIZE
        else:
            self._shm = SharedMemory(name=name, create=False)

        self._owner = create

        # Pre-compute byte offsets (constant for lifetime of this PortBus)
        self._frame_off = _HDR_SIZE
        self._frame_sz  = max_w * max_h * max_c
        self._det_off   = self._frame_off + self._frame_sz
        self._meta_off  = self._det_off + MAX_DETS * _DET_STRIDE

        # Numpy view into the frame region — writing here goes straight to shm
        self._frame_view: np.ndarray = np.frombuffer(
            self._shm.buf, dtype=np.uint8,
            offset=self._frame_off,
            count=self._frame_sz,
        ).reshape(max_h, max_w, max_c)

        self._last_seq: int = 0  # tracks last seq_no seen by reader

    # ── write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        metadata: dict,
    ) -> None:
        """Write one frame into the bus (overwrites previous unread frame)."""
        h, w = frame.shape[:2]
        c    = frame.shape[2] if frame.ndim == 3 else 1
        ww   = min(w, self.max_w)
        hh   = min(h, self.max_h)
        cc   = min(c, self.max_c)

        # Read current counters without full unpack (offsets 24 and 32)
        write_count = struct.unpack_from("<Q", self._shm.buf, 24)[0]
        read_count  = struct.unpack_from("<Q", self._shm.buf, 32)[0]
        seq_no      = struct.unpack_from("<Q", self._shm.buf, 0)[0]

        flags = 0

        # — Frame —
        src = frame[:hh, :ww] if cc == c else frame[:hh, :ww, :cc]
        np.copyto(self._frame_view[:hh, :ww, :cc], src)
        flags |= FLAG_FRAME_VALID

        # — Detections —
        det_count = min(len(detections), MAX_DETS)
        for i, det in enumerate(detections[:det_count]):
            struct.pack_into(
                _DET_FMT,
                self._shm.buf,
                self._det_off + i * _DET_STRIDE,
                det.x1, det.y1, det.x2, det.y2, det.confidence,
                det.class_id,
                _encode_class_name(det.class_name),
                det.track_id,
                _encode_meta_json(det.metadata if det.metadata else {}),
            )
        flags |= FLAG_DET_VALID

        # — Metadata JSON —
        meta_bytes = json.dumps(metadata, default=str).encode("utf-8")
        meta_len   = min(len(meta_bytes), CVFLOW_METADATA_LEN - 1)
        self._shm.buf[self._meta_off : self._meta_off + meta_len] = meta_bytes[:meta_len]
        self._shm.buf[self._meta_off + meta_len] = 0
        flags |= FLAG_META_VALID

        # — Header (seq_no written last to signal reader) —
        struct.pack_into(
            _HDR_FMT,
            self._shm.buf, 0,
            seq_no + 1,
            ww, hh, cc,
            flags,
            det_count,
            meta_len,
            write_count + 1,
            read_count,
            b"\x00" * 24,
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def read(
        self,
        timeout_ms: int = 30,
    ) -> Optional[tuple[np.ndarray, list[Detection], dict]]:
        """
        Read the latest frame.  Returns None on timeout (no new frame in time).
        The returned frame is a copy — safe to hold while producer writes next frame.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            hdr = struct.unpack_from(_HDR_FMT, self._shm.buf)
            seq_no, width, height, channels, flags, det_count, meta_len, wc, rc, _ = hdr

            if seq_no == self._last_seq or not (flags & FLAG_FRAME_VALID):
                time.sleep(0.0005)
                continue

            # — Frame (copy from shared memory) —
            frame_size = width * height * channels
            frame = np.frombuffer(
                self._shm.buf, dtype=np.uint8,
                offset=self._frame_off,
                count=frame_size,
            ).reshape(height, width, channels).copy()

            # — Detections —
            detections: list[Detection] = []
            for i in range(det_count):
                offset = self._det_off + i * _DET_STRIDE
                x1, y1, x2, y2, conf, cls_id, cn_b, track_id, mj_b = struct.unpack_from(
                    _DET_FMT, self._shm.buf, offset
                )
                class_name = cn_b.rstrip(b"\x00").decode("utf-8", errors="replace")
                try:
                    det_meta = json.loads(
                        mj_b.rstrip(b"\x00").decode("utf-8", errors="replace") or "{}"
                    )
                except Exception:
                    det_meta = {}
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id,
                    metadata=det_meta,
                ))

            # — Metadata —
            meta_raw = bytes(self._shm.buf[self._meta_off : self._meta_off + meta_len])
            try:
                metadata = json.loads(meta_raw.decode("utf-8", errors="replace")) if meta_raw else {}
            except Exception:
                metadata = {}

            # — Update tracking —
            self._last_seq = seq_no
            # Increment read_count (best-effort, no lock)
            struct.pack_into("<Q", self._shm.buf, 32, rc + 1)

            return frame, detections, metadata

        return None

    # ── monitoring ────────────────────────────────────────────────────────────

    def get_buffer_depth(self) -> int:
        """Frames written but not yet consumed.  Used by AutoScaler."""
        wc = struct.unpack_from("<Q", self._shm.buf, 24)[0]
        rc = struct.unpack_from("<Q", self._shm.buf, 32)[0]
        return max(0, int(wc - rc))

    @property
    def shm_name(self) -> str:
        """Shared memory name — pass to C++ workers via --input-bus / --output-bus."""
        return self.name

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self, *, unlink: bool = False) -> None:
        """
        Release this process's mapping.
        The creator should call close(unlink=True) to free the OS resource.
        Consumers call close() (unlink=False).
        """
        if hasattr(self, "_shm") and self._shm is not None:
            try:
                self._shm.close()
                if unlink or self._owner:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PortBus(name={self.name!r}, max={self.max_w}×{self.max_h})"


# ── CudaPortBus ───────────────────────────────────────────────────────────────

class CudaPortBus(PortBus):
    """
    GPU-to-GPU zero-copy channel using CUDA IPC handles.
    Falls back to CPU PortBus if torch/CUDA is unavailable.

    Frame data lives in a shared CUDA tensor; metadata and detections still
    use CPU shared memory (they are small and already struct-packed).
    The returned frame from read() is a torch.Tensor on the same GPU.
    """

    def __init__(
        self,
        name: str,
        max_w: int = 1280,
        max_h: int = 720,
        max_c: int = 3,
        device: str = "cuda:0",
        *,
        create: bool = True,
    ) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available; falling back to CPU PortBus")
            self._torch  = torch
            self._device = torch.device(device)
        except (ImportError, RuntimeError) as e:
            import warnings
            warnings.warn(f"CudaPortBus: {e} — using CPU PortBus", stacklevel=2)
            super().__init__(name, max_w, max_h, max_c, create=create)
            self._cuda_ok = False
            return

        self._cuda_ok = True
        # CPU shm for header + detections + metadata
        super().__init__(name, max_w, max_h, max_c, create=create)

        if create:
            # Allocate pinned CUDA tensor for the frame
            self._gpu_frame: "torch.Tensor" = torch.zeros(
                (max_h, max_w, max_c), dtype=torch.uint8, device=self._device
            )
            self._gpu_frame.share_memory_()
            # Store IPC handle in shared memory metadata region (base64-encoded)
            import base64, pickle
            handle = self._gpu_frame.storage()._share_cuda_()[1]
            handle_b = base64.b64encode(pickle.dumps(handle))
            self._shm.buf[self._meta_off : self._meta_off + len(handle_b)] = handle_b
        else:
            # Attach to existing GPU tensor via IPC handle
            import base64, pickle
            handle_b = bytes(self._shm.buf[self._meta_off : self._meta_off + 512]).rstrip(b"\x00")
            handle = pickle.loads(base64.b64decode(handle_b))
            storage = torch.cuda.storage.UntypedStorage._new_shared_cuda(*handle)
            self._gpu_frame = torch.tensor(storage, dtype=torch.uint8).view(max_h, max_w, max_c)

    def write(self, frame, detections, metadata):  # type: ignore[override]
        if not self._cuda_ok:
            super().write(frame, detections, metadata)
            return
        # Accept numpy (CPU) or torch tensor (GPU)
        if isinstance(frame, np.ndarray):
            gpu = self._torch.from_numpy(frame).to(self._device)
        else:
            gpu = frame.to(self._device)
        h, w = gpu.shape[:2]
        hh, ww = min(h, self.max_h), min(w, self.max_w)
        self._gpu_frame[:hh, :ww].copy_(gpu[:hh, :ww])
        # Write header + dets + meta to CPU shm (reuse parent logic, set frame_valid)
        # Pass empty frame array to avoid double-write of frame bytes
        dummy = np.empty((hh, ww, 3), dtype=np.uint8)
        super().write(dummy, detections, metadata)

    def read(self, timeout_ms=30):  # type: ignore[override]
        if not self._cuda_ok:
            return super().read(timeout_ms)
        result = super().read(timeout_ms)
        if result is None:
            return None
        _, detections, metadata = result
        # Return GPU tensor slice (zero-copy)
        hdr = struct.unpack_from(_HDR_FMT, self._shm.buf)
        width, height = hdr[1], hdr[2]
        gpu_frame = self._gpu_frame[:height, :width].clone()
        return gpu_frame, detections, metadata


# ── make_port_bus — factory ───────────────────────────────────────────────────

def make_port_bus(
    name: str,
    src_device: str = "cpu",
    dst_device: str = "cpu",
    *,
    max_w: int = 1280,
    max_h: int = 720,
    max_c: int = 3,
    create: bool = True,
) -> PortBus:
    """
    Create the right PortBus variant for the src→dst device pair.

    cpu   → cpu   : PortBus (CPU shared memory, zero-copy on same machine)
    cuda  → cuda  : CudaPortBus (CUDA IPC, zero-copy on GPU)
    mixed          : PortBus (transparent CPU↔GPU copy in write/read)

    The 'create' flag controls whether shared memory is allocated (True,
    creator/writer side) or attached (False, consumer/reader side).
    """
    same_device = src_device == dst_device
    if same_device and src_device.startswith("cuda"):
        return CudaPortBus(name, max_w, max_h, max_c, device=src_device, create=create)
    return PortBus(name, max_w, max_h, max_c, create=create)


# ── RoundRobinBus — fan-out ───────────────────────────────────────────────────

class RoundRobinBus:
    """
    Distributes writes round-robin across N PortBuses (one writer, N readers).
    Used by AutoScaler when a node is scaled to M workers: each worker gets its
    own PortBus and the RoundRobinBus feeds them evenly.
    """

    def __init__(self, buses: list[PortBus]) -> None:
        assert buses, "Need at least one bus"
        self._buses = buses
        self._idx   = 0

    def write(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        metadata: dict,
    ) -> None:
        self._buses[self._idx].write(frame, detections, metadata)
        self._idx = (self._idx + 1) % len(self._buses)

    def get_buffer_depth(self) -> float:
        """Average depth across all downstream buses."""
        return sum(b.get_buffer_depth() for b in self._buses) / len(self._buses)

    def add_bus(self, bus: PortBus) -> None:
        self._buses.append(bus)

    def remove_bus(self) -> PortBus:
        """Remove and return the last bus (for scale-down)."""
        if len(self._buses) > 1:
            return self._buses.pop()
        raise RuntimeError("Cannot remove last bus from RoundRobinBus")

    @property
    def buses(self) -> list[PortBus]:
        return self._buses


# ── MergeBus — fan-in ─────────────────────────────────────────────────────────

class MergeBus:
    """
    Reads from N PortBuses and returns the first available frame (N writers,
    1 reader).  Used on the output side of a scaled node pool — all N worker
    outputs merge back into one stream for the downstream node.
    """

    def __init__(self, buses: list[PortBus]) -> None:
        assert buses, "Need at least one bus"
        self._buses = buses
        self._poll  = 0  # round-robin poll cursor

    def read(
        self,
        timeout_ms: int = 30,
    ) -> Optional[tuple[np.ndarray, list[Detection], dict]]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            for _ in range(len(self._buses)):
                result = self._buses[self._poll].read(timeout_ms=0)
                self._poll = (self._poll + 1) % len(self._buses)
                if result is not None:
                    return result
            time.sleep(0.0005)
        return None

    def add_bus(self, bus: PortBus) -> None:
        self._buses.append(bus)

    def remove_bus(self) -> PortBus:
        if len(self._buses) > 1:
            return self._buses.pop()
        raise RuntimeError("Cannot remove last bus from MergeBus")

    @property
    def buses(self) -> list[PortBus]:
        return self._buses
