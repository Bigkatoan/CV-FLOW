"""
cv_flow.dam.bus — PortBus: Named shared memory ring-buffer channel.

Binary layout (shared memory):
  [HEADER       :   64 bytes]
  [DATA REGION  :   queue_depth × slot_bytes]
  [DET REGION   :   queue_depth × MAX_DETS × DET_STRIDE]
  [META REGION  :   queue_depth × META_LEN]

Header (64 bytes, little-endian):
  +00 uint64  write_count   total writes (monotonic)
  +08 uint64  read_count    total reads  (monotonic)
  +16 uint64  drop_count    total drops  (queue full)
  +24 uint32  queue_depth   N slots
  +28 uint32  slot_bytes    bytes per data slot
  +32 uint8   flags         reserved (always 0 for now)
  +33 uint8[31] reserved

Ordering semantics (DEFAULT — ordered queue):
  Writer → slot[write_count % queue_depth]
  If queue full (depth >= queue_depth):
    - Overwrite oldest slot (advance read_count by 1)
    - Increment drop_count
    - Emit WARNING log

Drop mode (fastest, no ordering):
  Same as above but no logging; newest-wins semantics.

Concurrency note: write()/read() are safe for the common case of ONE
process advertising and ONE process subscribing where each side only ever
runs sequentially relative to itself (true for every plain single-process
or single-writer/single-reader pipeline in this codebase). They are NOT
safe for a genuine concurrent writer process and reader process touching
the SAME bus at the same time: `_write_header()` writes write_count,
read_count, and drop_count together in one call based on whatever the
caller last read, so a writer's drop-oldest path (which also advances
read_count) can race a concurrent read() and clobber its progress, or vice
versa — no cross-process lock protects this. This was discovered building
`cv_flow.elastic.ElasticStage`, which spawns real multiprocessing workers
that genuinely read/write concurrently with the main process; the fix
there is a `multiprocessing.Lock` held by both sides around each bus
operation (see `cv_flow/elastic.py` module docstring) rather than changing
PortBus itself. If you add new code with a real concurrent writer+reader
on one PortBus, you need the same kind of external locking.
"""
from __future__ import annotations

import json
import logging
import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

logger = logging.getLogger("cv_flow.dam")

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_DETS   = 256           # max detections per frame
META_LEN   = 4096          # bytes for metadata JSON per slot
CLASS_LEN  = 64            # bytes for class_name string
DET_META_L = 256           # bytes for per-detection metadata JSON

# Detection struct: 5×float(20) + int(4) + char[64](64) + int(4) + char[256](256) = 348 bytes
_DET_FMT    = f"<5fi{CLASS_LEN}si{DET_META_L}s"
DET_STRIDE  = struct.calcsize(_DET_FMT)
assert DET_STRIDE == 348, f"DET_STRIDE must be 348, got {DET_STRIDE}"

# Header format
_HDR_FMT  = "<QQQIIBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 8+8+8+4+4+1+31pad = 64
_HDR_SIZE = struct.calcsize(_HDR_FMT)
assert _HDR_SIZE == 64, f"Header must be 64 bytes, got {_HDR_SIZE}"


def _shm_size(queue_depth: int, slot_bytes: int) -> int:
    return (
        _HDR_SIZE
        + queue_depth * slot_bytes
        + queue_depth * MAX_DETS * DET_STRIDE
        + queue_depth * META_LEN
    )


def _encode_class(s: str) -> bytes:
    b = s.encode("utf-8", errors="replace")[: CLASS_LEN - 1]
    return b.ljust(CLASS_LEN, b"\x00")


def _encode_det_meta(d: dict) -> bytes:
    b = json.dumps(d, default=str).encode("utf-8")[: DET_META_L - 1]
    return b.ljust(DET_META_L, b"\x00")


class Detection:
    """Simple detection struct (mirrors CVFlowDetection)."""
    __slots__ = ("x1", "y1", "x2", "y2", "confidence", "class_id",
                 "class_name", "track_id", "metadata")

    def __init__(self, x1=0.0, y1=0.0, x2=0.0, y2=0.0, confidence=0.0,
                 class_id=0, class_name="", track_id=-1, metadata=None):
        self.x1 = float(x1); self.y1 = float(y1)
        self.x2 = float(x2); self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id   = int(class_id)
        self.class_name = str(class_name)
        self.track_id   = int(track_id)
        self.metadata   = metadata or {}

    def __repr__(self):
        return (f"Detection({self.class_name!r} conf={self.confidence:.2f} "
                f"bbox=[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}])")


# ── PortBus ───────────────────────────────────────────────────────────────────

class PortBus:
    """
    Named shared memory ring-buffer channel.

    One writer + one reader (lock-free via monotonic counters).
    Ordered queue semantics by default: reader sees frames in write order.
    When queue is full, oldest slot is dropped with a WARNING log.

    Parameters
    ----------
    name        : Unique shared memory name (alphanumeric + underscore).
    slot_bytes  : Bytes per data slot (total payload per frame).
    queue_depth : Ring buffer depth (number of slots). Default 8.
    create      : True = allocate new region; False = attach to existing.
    drop_mode   : True = silent newest-wins (no log). Default False.
    """

    def __init__(
        self,
        name:        str,
        slot_bytes:  int,
        *,
        queue_depth: int  = 8,
        create:      bool = True,
        drop_mode:   bool = False,
    ) -> None:
        self.name        = name
        self.slot_bytes  = slot_bytes
        self.queue_depth = queue_depth
        self.drop_mode   = drop_mode

        size = _shm_size(queue_depth, slot_bytes)
        if create:
            self._shm   = SharedMemory(name=name, create=True, size=size)
            # Zero header
            self._shm.buf[:_HDR_SIZE] = b"\x00" * _HDR_SIZE
            # Write queue_depth and slot_bytes into header (use canonical format)
            struct.pack_into(_HDR_FMT, self._shm.buf, 0,
                             0, 0, 0, queue_depth, slot_bytes, 0)
        else:
            self._shm = SharedMemory(name=name, create=False)
            # Read actual queue_depth / slot_bytes from header
            wc, rc, dc, qd, sb, _flags = struct.unpack_from(_HDR_FMT, self._shm.buf)
            self.queue_depth = qd
            self.slot_bytes  = sb

        self._owner = create

        # Pre-compute region offsets
        self._data_off = _HDR_SIZE
        self._det_off  = self._data_off + self.queue_depth * self.slot_bytes
        self._meta_off = self._det_off  + self.queue_depth * MAX_DETS * DET_STRIDE

    # ── internal helpers ──────────────────────────────────────────────────────

    def _data_slot_offset(self, idx: int) -> int:
        return self._data_off + idx * self.slot_bytes

    def _det_slot_offset(self, idx: int) -> int:
        return self._det_off + idx * MAX_DETS * DET_STRIDE

    def _meta_slot_offset(self, idx: int) -> int:
        return self._meta_off + idx * META_LEN

    def _read_header(self) -> tuple[int, int, int]:
        """Returns (write_count, read_count, drop_count)."""
        return struct.unpack_from("<QQQ", self._shm.buf, 0)

    def _write_header(self, wc: int, rc: int, dc: int) -> None:
        struct.pack_into("<QQQ", self._shm.buf, 0, wc, rc, dc)

    # ── write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        data:        bytes,
        seq:         int,
        *,
        detections:  list[Detection] = [],
        metadata:    dict            = {},
    ) -> bool:
        """
        Write one frame into the ring buffer.

        Returns True if a slot was dropped (queue was full), False otherwise.
        If drop_mode=False (default), logs a WARNING when a drop occurs.
        """
        wc, rc, dc = self._read_header()
        depth  = wc - rc
        dropped = False

        if depth >= self.queue_depth:
            # Queue full — drop oldest slot (advance read_count)
            rc     += 1
            dc     += 1
            dropped = True
            if not self.drop_mode:
                logger.warning(
                    "[cv_flow.dam] Bus '%s': queue full — dropped frame #%d "
                    "(total drops: %d)", self.name, seq, dc,
                )

        slot_idx = int(wc % self.queue_depth)

        # — Data —
        data_bytes = bytes(data)
        n = min(len(data_bytes), self.slot_bytes)
        off = self._data_slot_offset(slot_idx)
        self._shm.buf[off : off + n] = data_bytes[:n]
        if n < self.slot_bytes:
            self._shm.buf[off + n : off + self.slot_bytes] = b"\x00" * (self.slot_bytes - n)

        # — Store seq in first 8 bytes of det region for this slot (reuse DET header) —
        det_base = self._det_slot_offset(slot_idx)
        struct.pack_into("<Q", self._shm.buf, det_base, seq)

        # — Detections (after the 8-byte seq header) —
        det_count = min(len(detections), MAX_DETS)
        struct.pack_into("<I", self._shm.buf, det_base + 8, det_count)
        for i, det in enumerate(detections[:det_count]):
            struct.pack_into(
                _DET_FMT,
                self._shm.buf,
                det_base + 12 + i * DET_STRIDE,
                det.x1, det.y1, det.x2, det.y2, det.confidence,
                det.class_id,
                _encode_class(det.class_name),
                det.track_id,
                _encode_det_meta(det.metadata),
            )

        # — Metadata JSON —
        meta_raw = json.dumps(metadata, default=str).encode("utf-8")
        meta_len = min(len(meta_raw), META_LEN - 1)
        meta_off = self._meta_slot_offset(slot_idx)
        self._shm.buf[meta_off : meta_off + meta_len] = meta_raw[:meta_len]
        self._shm.buf[meta_off + meta_len] = 0

        # — Update header (write_count last → reader detects new data) —
        self._write_header(wc + 1, rc, dc)

        return dropped

    # ── internal slot decoder ────────────────────────────────────────────────

    def _decode_slot(
        self, slot_idx: int
    ) -> tuple[bytes, int, list[Detection], dict]:
        """Decode data, seq, detections, and metadata from a ring-buffer slot."""
        off  = self._data_slot_offset(slot_idx)
        data = bytes(self._shm.buf[off : off + self.slot_bytes])

        det_base  = self._det_slot_offset(slot_idx)
        seq_no    = struct.unpack_from("<Q", self._shm.buf, det_base)[0]
        det_count = struct.unpack_from("<I", self._shm.buf, det_base + 8)[0]
        detections: list[Detection] = []
        for i in range(min(det_count, MAX_DETS)):
            x1, y1, x2, y2, conf, cls_id, cn_b, trk, mj_b = struct.unpack_from(
                _DET_FMT, self._shm.buf, det_base + 12 + i * DET_STRIDE
            )
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=cls_id,
                class_name=cn_b.rstrip(b"\x00").decode("utf-8", errors="replace"),
                track_id=trk,
                metadata=json.loads(
                    mj_b.rstrip(b"\x00").decode("utf-8", errors="replace") or "{}"
                ),
            ))

        meta_off = self._meta_slot_offset(slot_idx)
        meta_end = meta_off
        while meta_end < meta_off + META_LEN and self._shm.buf[meta_end] != 0:
            meta_end += 1
        meta_raw = bytes(self._shm.buf[meta_off:meta_end])
        try:
            metadata = json.loads(meta_raw.decode("utf-8", errors="replace")) if meta_raw else {}
        except Exception:
            metadata = {}

        return data, seq_no, detections, metadata

    # ── read ──────────────────────────────────────────────────────────────────

    def peek(self) -> Optional[tuple[bytes, int, list[Detection], dict]]:
        """
        Read head frame WITHOUT advancing read_count (non-destructive).

        Used by MergeBus to inspect seq_no before deciding which bus to consume.
        Returns None if queue is empty.
        """
        wc, rc, dc = self._read_header()
        if wc <= rc:
            return None
        return self._decode_slot(int(rc % self.queue_depth))

    def read(
        self,
        timeout_ms: int = 30,
    ) -> Optional[tuple[bytes, int, list[Detection], dict]]:
        """
        Read the next frame from the ring buffer (in write order).

        Blocks up to timeout_ms waiting for a new frame.

        Returns
        -------
        (data: bytes, seq_no: int, detections: list, metadata: dict)
        or None on timeout.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            wc, rc, dc = self._read_header()
            if wc > rc:
                result = self._decode_slot(int(rc % self.queue_depth))
                self._write_header(wc, rc + 1, dc)
                return result
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)

    # ── monitoring ────────────────────────────────────────────────────────────

    def get_buffer_depth(self) -> int:
        """Frames written but not yet read (0 = consumer is caught up)."""
        wc, rc, _ = self._read_header()
        return max(0, int(wc - rc))

    @property
    def stats(self) -> dict:
        """Return current bus statistics."""
        wc, rc, dc = self._read_header()
        return {
            "writes": int(wc),
            "reads":  int(rc),
            "drops":  int(dc),
            "depth":  max(0, int(wc - rc)),
        }

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self, unlink: bool = False) -> None:
        """
        Release this process's mapping.
        Creator calls close(unlink=True); consumers call close().
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
        return (f"PortBus(name={self.name!r}, slot={self.slot_bytes}B, "
                f"depth={self.queue_depth})")
