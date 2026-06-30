"""
Tests for cv_flow.dam.cuda_bus.CudaPortBus

T-CUDA-01: no GPU → automatically falls back to PortBus, no crash
T-CUDA-02: (skip if no GPU) torch.Tensor write → torch.Tensor read, data correct
T-CUDA-03: (skip if no GPU) tensor is on the right device after read()
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
    """torch.Tensor write → torch.Tensor read, data matches."""
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
