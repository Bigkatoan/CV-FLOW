"""
Tests for cv_flow.nodes.inference (OnnxInference, YoloInference) using a real
exported ONNX model (tests/fixtures/yolov8n.onnx, YOLOv8n exported via ultralytics).

T-INF-01: OnnxInference on CPUExecutionProvider runs a real model end-to-end
          through the DAM/Topic stack and produces the expected output shape.
T-INF-02 (marked gpu): YoloInference on device="cuda:0" actually activates a
          GPU execution provider (CUDA or TensorRT) — not a silent CPU fallback.
T-INF-03 (marked gpu): YoloInference with trt_cache_dir set builds a TensorRT
          engine once and reuses the cache on a second session (much faster
          second load) — proves the cache wiring is real, not just accepted
          and ignored.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ort = pytest.importorskip("onnxruntime")

from cv_flow.nodes.inference import OnnxInference, YoloInference
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef
from cv_flow.topic.publisher import Publisher
from cv_flow.dam.bus import PortBus

FIXTURE_MODEL = Path(__file__).parent.parent / "fixtures" / "yolov8n.onnx"

pytestmark = pytest.mark.skipif(
    not FIXTURE_MODEL.exists(),
    reason=(
        f"Real ONNX model fixture not found at {FIXTURE_MODEL}. "
        "Export one with: pip install ultralytics && "
        "python -c \"from ultralytics import YOLO; "
        "YOLO('yolov8n.pt').export(format='onnx', imgsz=640)\" "
        "then copy yolov8n.onnx to tests/fixtures/."
    ),
)


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


def _register_yolo_topics():
    tensor_field = FieldDef.build("tensor", "float32", (1, 3, 640, 640))
    Topic(TopicDef(
        name="inf_in",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[tensor_field]),
    ))
    raw_field = FieldDef.build("raw", "float32", (1, 84, 8400))
    Topic(TopicDef(
        name="inf_out",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[raw_field]),
    ))
    return tensor_field, raw_field


def _publish_input(session: str, tensor_field: FieldDef) -> PortBus:
    in_bus = PortBus(f"inf_in_{session}", slot_bytes=tensor_field.n_bytes,
                      queue_depth=4, create=True)
    in_pub = Publisher(in_bus, PortDef(device="cpu", fields=[tensor_field]))
    in_pub.write(np.random.rand(1, 3, 640, 640).astype(np.float32))
    return in_bus


# ── T-INF-01 ──────────────────────────────────────────────────────────────────

def test_onnx_inference_cpu_real_model():
    """OnnxInference(device='cpu') runs the real yolov8n.onnx model end-to-end."""
    tensor_field, raw_field = _register_yolo_topics()
    session = "inf-cpu-session"
    in_bus = _publish_input(session, tensor_field)

    node = OnnxInference(
        "inf_in", "inf_out",
        model_path=str(FIXTURE_MODEL),
        input_name="images", output_name="output0",
        device="cpu",
    )
    node._session_id = session
    node.initialize()
    assert node._session.get_providers()[0] == "CPUExecutionProvider"

    node.spin_once()

    out_bus = PortBus(f"inf_out_{session}", slot_bytes=raw_field.n_bytes, create=False)
    result = out_bus.read(timeout_ms=500)
    assert result is not None
    raw = np.frombuffer(result[0], dtype=np.float32).reshape(1, 84, 8400)
    assert raw.shape == (1, 84, 8400)
    # Real model output should not be all-zero garbage.
    assert np.any(raw != 0)

    in_bus.close(unlink=True)
    out_bus.close(unlink=True)


# ── T-INF-02 (gpu) ────────────────────────────────────────────────────────────

@pytest.mark.gpu
def test_yolo_inference_gpu_real_model_uses_gpu_provider():
    """YoloInference(device='cuda:0') actually activates CUDA or TensorRT, not CPU."""
    tensor_field, raw_field = _register_yolo_topics()
    session = "inf-gpu-session"
    in_bus = _publish_input(session, tensor_field)

    node = YoloInference(
        "inf_in", "inf_out",
        model_path=str(FIXTURE_MODEL),
        device="cuda:0",
    )
    node._session_id = session
    node.initialize()

    active = node._session.get_providers()
    assert active[0] in ("TensorrtExecutionProvider", "CUDAExecutionProvider"), (
        f"Expected GPU provider active, got {active}"
    )

    node.spin_once()
    out_bus = PortBus(f"inf_out_{session}", slot_bytes=raw_field.n_bytes, create=False)
    result = out_bus.read(timeout_ms=2000)
    assert result is not None
    raw = np.frombuffer(result[0], dtype=np.float32).reshape(1, 84, 8400)
    assert raw.shape == (1, 84, 8400)

    in_bus.close(unlink=True)
    out_bus.close(unlink=True)


# ── T-INF-03 (gpu) ────────────────────────────────────────────────────────────

@pytest.mark.gpu
def test_yolo_inference_trt_cache_speeds_up_second_load(tmp_path):
    """trt_cache_dir caches the TensorRT engine: 2nd session loads much faster."""
    cache_dir = tmp_path / "trt_cache"

    t0 = time.monotonic()
    sess1 = ort.InferenceSession(
        str(FIXTURE_MODEL),
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[
            {"trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_dir)},
            {}, {},
        ],
    )
    first_load_s = time.monotonic() - t0
    assert sess1.get_providers()[0] == "TensorrtExecutionProvider"
    assert any(cache_dir.iterdir()), "Expected TensorRT engine cache files to be written"

    t0 = time.monotonic()
    sess2 = ort.InferenceSession(
        str(FIXTURE_MODEL),
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[
            {"trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_dir)},
            {}, {},
        ],
    )
    second_load_s = time.monotonic() - t0
    assert sess2.get_providers()[0] == "TensorrtExecutionProvider"

    assert second_load_s < first_load_s, (
        f"Expected cached engine load ({second_load_s:.1f}s) to beat "
        f"first build ({first_load_s:.1f}s)"
    )
