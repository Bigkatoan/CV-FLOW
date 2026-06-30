"""
Tests for cv_flow.dam.cuda_bus.CudaPortBus

T-CUDA-01: no GPU → automatically falls back to PortBus, no crash
T-CUDA-02: (skip if no GPU) torch.Tensor write → torch.Tensor read, data correct
T-CUDA-03: (skip if no GPU) tensor is on the right device after read()
T-CUDA-04: (skip if no GPU) a tensor LARGER than 64 bytes round-trips with
           every value intact and the original dtype/shape restored — this
           is the regression test for a real bug found while building
           Phase 2: a previous version hardcoded a 64-byte slot whenever
           using_cuda=True (sized for a CUDA IPC handle that was never
           actually used), silently truncating any larger tensor and
           losing dtype/shape entirely on read (always returned a flat
           uint8 buffer). The old T-CUDA-02/03 used 8-byte tensors and
           never checked exact values, so they never caught this.
T-CUDA-05: (marked multiprocess, skip if no GPU) a real separate child
           process reads a tensor written by the parent via CudaPortBus
           and gets back the exact same values/dtype/shape. This is NOT a
           zero-copy / shared-VRAM test — real CUDA IPC via
           torch.multiprocessing.reductions.reduce_tensor() was tried for
           real and confirmed not viable on this Jetson (see
           cv_flow/dam/cuda_bus.py module docstring): rebuild_cuda_tensor
           fails with `CUDA error: invalid argument` in the receiving
           process, consistently. This test instead verifies the
           CPU-roundtrip path (the only viable path on this hardware)
           is correct across real OS processes, not just within one.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.dam.cuda_bus import CudaPortBus, _CUDA_AVAILABLE


def _unique(prefix="cuda"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── T-CUDA-01 ─────────────────────────────────────────────────────────────────

def test_cuda_01_fallback_no_crash():
    """CudaPortBus can be created regardless of GPU availability."""
    name = _unique()
    bus  = CudaPortBus(name, slot_bytes=64, queue_depth=4, create=True)
    try:
        assert bus is not None
        assert hasattr(bus, "using_cuda")
        assert bus.using_cuda == _CUDA_AVAILABLE

        # Write and read raw bytes — must work on both CPU and GPU paths
        payload = b"\xAB" * 64
        bus.write(payload, seq=1)
        result = bus.read(timeout_ms=100)

        assert result is not None
        _, seq, _, _ = result
        assert seq == 1
    finally:
        bus.close(unlink=True)


# ── T-CUDA-02 ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA GPU available")
def test_cuda_02_tensor_round_trip():
    """torch.Tensor write → torch.Tensor read, data matches exactly."""
    import torch
    name = _unique()
    bus  = CudaPortBus(name, slot_bytes=64, queue_depth=4,
                       create=True, device="cuda:0")
    try:
        t = torch.ones(8, dtype=torch.uint8, device="cuda:0")
        bus.write(t, seq=42)
        result = bus.read(timeout_ms=100)

        assert result is not None
        tensor, seq, _, _ = result
        assert seq == 42
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == t.shape
        assert tensor.dtype == t.dtype
        assert torch.equal(tensor.cpu(), t.cpu())
    finally:
        bus.close(unlink=True)


# ── T-CUDA-03 ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA GPU available")
def test_cuda_03_tensor_on_correct_device():
    """Tensor returned by read() is on the correct CUDA device."""
    import torch
    name = _unique()
    bus  = CudaPortBus(name, slot_bytes=64, queue_depth=4,
                       create=True, device="cuda:0")
    try:
        t = torch.zeros(8, dtype=torch.uint8, device="cuda:0")
        bus.write(t, seq=7)
        result = bus.read(timeout_ms=100)

        assert result is not None
        tensor, seq, _, _ = result
        assert tensor.device.type == "cuda", \
            f"Expected cuda tensor, got device={tensor.device}"
    finally:
        bus.close(unlink=True)


# ── T-CUDA-04 ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA GPU available")
def test_cuda_04_large_tensor_not_truncated():
    """A tensor much larger than 64 bytes round-trips with every value and
    the original dtype/shape intact (regression test for the truncation bug)."""
    import torch
    name = _unique()
    shape = (1, 3, 64, 64)
    n_bytes = 1 * 3 * 64 * 64 * 4  # float32
    bus = CudaPortBus(name, slot_bytes=n_bytes, queue_depth=2,
                       create=True, device="cuda:0")
    try:
        t = torch.arange(1 * 3 * 64 * 64, dtype=torch.float32, device="cuda:0").reshape(shape)
        bus.write(t, seq=1)
        result = bus.read(timeout_ms=200)

        assert result is not None
        tensor, seq, _, _ = result
        assert tensor.shape == t.shape, f"shape lost: got {tensor.shape}, expected {t.shape}"
        assert tensor.dtype == t.dtype, f"dtype lost: got {tensor.dtype}, expected {t.dtype}"
        assert torch.equal(tensor.cpu(), t.cpu()), "values corrupted/truncated in round-trip"
    finally:
        bus.close(unlink=True)


def test_cuda_04b_oversized_tensor_raises_instead_of_silently_truncating():
    """Writing a tensor bigger than slot_bytes raises, rather than silently
    dropping data (the original failure mode of the truncation bug)."""
    import torch
    if not _CUDA_AVAILABLE:
        pytest.skip("No CUDA GPU available")
    name = _unique()
    bus = CudaPortBus(name, slot_bytes=16, queue_depth=2, create=True, device="cuda:0")
    try:
        t = torch.ones(100, dtype=torch.float32, device="cuda:0")  # 400 bytes > 16
        with pytest.raises(ValueError):
            bus.write(t, seq=1)
    finally:
        bus.close(unlink=True)


# ── T-CUDA-05 (multiprocess, real cross-process correctness) ──────────────────

def _cuda_bus_child_proc(bus_name: str, slot_bytes: int, q) -> None:
    import torch
    from cv_flow.dam.cuda_bus import CudaPortBus
    bus = CudaPortBus(bus_name, slot_bytes=slot_bytes, queue_depth=2,
                       create=False, device="cuda:0")
    try:
        result = bus.read(timeout_ms=5000)
        if result is None:
            q.put(("error", "no data received"))
            return
        tensor, seq, _, _ = result
        q.put((
            "ok",
            {
                "seq": seq,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device_type": tensor.device.type,
                "sum": float(tensor.sum().item()),
            },
        ))
    except Exception as exc:  # noqa: BLE001
        q.put(("error", repr(exc)))
    finally:
        bus.close(unlink=False)


@pytest.mark.gpu
@pytest.mark.multiprocess
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA GPU available")
def test_cuda_05_real_cross_process_round_trip():
    """A real child OS process reads a GPU tensor written by the parent via
    CudaPortBus and gets back the exact same values (CPU-roundtrip path,
    not zero-copy — see module docstring for why real CUDA IPC isn't used)."""
    import multiprocessing as mp
    import torch

    mp.set_start_method("spawn", force=True)

    name = _unique()
    shape = (4, 4)
    n_bytes = 4 * 4 * 4  # float32
    parent_bus = CudaPortBus(name, slot_bytes=n_bytes, queue_depth=2,
                              create=True, device="cuda:0")
    try:
        t = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(shape)
        expected_sum = float(t.sum().item())

        q = mp.Queue()
        p = mp.Process(target=_cuda_bus_child_proc, args=(name, n_bytes, q))
        p.start()

        parent_bus.write(t, seq=99)

        status, payload = q.get(timeout=30)
        p.join(timeout=10)

        assert status == "ok", f"child failed: {payload}"
        assert payload["seq"] == 99
        assert payload["shape"] == list(shape)
        assert payload["dtype"] == "torch.float32"
        assert payload["device_type"] == "cuda"
        assert payload["sum"] == expected_sum
    finally:
        parent_bus.close(unlink=True)
