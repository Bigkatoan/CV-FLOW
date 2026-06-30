"""
Tests for cv_flow.nodes.camera (CameraSource, RtspSource, build_nvargus_pipeline).

T-CAM-01: build_nvargus_pipeline() produces a correct GStreamer pipeline string.
T-CAM-02: CameraSource(gstreamer_pipeline=...) opens via cv2.CAP_GSTREAMER, not index.
T-CAM-03/04/05: RtspSource exponential backoff — doubles delay on repeated
                failure, caps at max_reconnect_delay_s, resets to base delay
                after a successful read.
T-CAM-06 (marked hardware): CameraSource opens a *real* USB camera, reads N
          real frames, and verifies shape/dtype. Requires a /dev/video* device
          physically attached — skipped with a clear reason otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
cv2 = pytest.importorskip("cv2")

from cv_flow.nodes.camera import CameraSource, RtspSource, build_nvargus_pipeline
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── T-CAM-01 ──────────────────────────────────────────────────────────────────

def test_build_nvargus_pipeline_string():
    pipeline = build_nvargus_pipeline(sensor_id=0, width=1280, height=720, fps=30)
    assert "nvarguscamerasrc sensor-id=0" in pipeline
    assert "width=1280, height=720" in pipeline
    assert "framerate=30/1" in pipeline
    assert pipeline.strip().endswith("appsink drop=true max-buffers=1")
    assert "video/x-raw, format=BGR" in pipeline  # final stage must be BGR for cv2


def test_build_nvargus_pipeline_different_sensor_and_flip():
    pipeline = build_nvargus_pipeline(sensor_id=1, width=640, height=480, fps=15, flip_method=2)
    assert "sensor-id=1" in pipeline
    assert "width=640, height=480" in pipeline
    assert "framerate=15/1" in pipeline
    assert "flip-method=2" in pipeline


# ── T-CAM-02 ──────────────────────────────────────────────────────────────────

def test_camera_source_uses_gstreamer_when_pipeline_set():
    """CameraSource opens via cv2.CAP_GSTREAMER with the pipeline string, not device_index."""
    pipeline = build_nvargus_pipeline()
    source = CameraSource("frame_topic", gstreamer_pipeline=pipeline)

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=fake_cap) as mock_ctor:
        cap = source._open_capture()

    mock_ctor.assert_called_once_with(pipeline, cv2.CAP_GSTREAMER)
    assert cap is fake_cap


def test_camera_source_uses_device_index_when_no_pipeline():
    """Without gstreamer_pipeline, CameraSource falls back to plain device_index."""
    source = CameraSource("frame_topic", device_index=2)

    fake_cap = MagicMock()
    with patch("cv2.VideoCapture", return_value=fake_cap) as mock_ctor:
        source._open_capture()

    mock_ctor.assert_called_once_with(2)


# ── RtspSource backoff (T-CAM-03/04/05) ────────────────────────────────────────

def _make_rtsp_source(monkeypatch, **kwargs):
    monkeypatch.setattr("time.sleep", lambda s: None)  # don't actually wait in tests
    source = RtspSource(
        "frame_topic", url="rtsp://example.invalid/stream",
        reconnect_delay_s=1.0, max_reconnect_delay_s=8.0, **kwargs,
    )
    source._cap = MagicMock()
    monkeypatch.setattr(source, "_open_capture", lambda: MagicMock())
    return source


def test_rtsp_backoff_doubles_on_repeated_failure(monkeypatch):
    source = _make_rtsp_source(monkeypatch)
    assert source._current_delay_s == 1.0

    source._on_read_failure()
    assert source._current_delay_s == 2.0
    assert source._consecutive_failures == 1

    source._on_read_failure()
    assert source._current_delay_s == 4.0
    assert source._consecutive_failures == 2


def test_rtsp_backoff_caps_at_max(monkeypatch):
    source = _make_rtsp_source(monkeypatch)
    for _ in range(10):
        source._on_read_failure()
    assert source._current_delay_s == 8.0  # capped, never exceeds max_reconnect_delay_s


def test_rtsp_backoff_resets_on_success(monkeypatch):
    source = _make_rtsp_source(monkeypatch)
    source._on_read_failure()
    source._on_read_failure()
    assert source._current_delay_s == 4.0
    assert source._consecutive_failures == 2

    source._on_read_success()
    assert source._current_delay_s == 1.0
    assert source._consecutive_failures == 0


def test_rtsp_full_spin_once_failure_then_recovery(monkeypatch):
    """End-to-end through spin_once(): failing reads reconnect+backoff, then recover."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    frame_field = FieldDef.build("frame", "bgr8", (4, 4))
    Topic(TopicDef(
        name="rtsp_test_topic",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    source = RtspSource("rtsp_test_topic", url="rtsp://example.invalid/stream",
                         reconnect_delay_s=0.5, max_reconnect_delay_s=4.0)
    source._session_id = "rtsp-test-session"

    fail_cap = MagicMock()
    fail_cap.read.return_value = (False, None)
    ok_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    ok_cap = MagicMock()
    ok_cap.read.return_value = (True, ok_frame)

    open_calls = {"n": 0}

    def fake_open():
        open_calls["n"] += 1
        return fail_cap if open_calls["n"] < 3 else ok_cap

    monkeypatch.setattr(source, "_open_capture", fake_open)
    source.initialize()  # first _open_capture() call -> fail_cap

    source.spin_once()  # read fails -> reconnect (2nd open -> still fail_cap)
    assert source._consecutive_failures == 1
    assert source._current_delay_s == 1.0

    source.spin_once()  # read fails again -> reconnect (3rd open -> ok_cap)
    assert source._consecutive_failures == 2
    assert source._current_delay_s == 2.0

    source.spin_once()  # now reads successfully -> resets backoff
    assert source._consecutive_failures == 0
    assert source._current_delay_s == 0.5

    for p in source._publishers:
        p._bus.close(unlink=True)


# ── T-CAM-06 (hardware, real USB camera) ───────────────────────────────────────

def _find_color_video_device() -> int | None:
    """
    Probe /dev/video* and return the index of the first device that yields a
    real 3-channel (BGR) frame. Multi-stream UVC devices (e.g. RealSense
    depth cameras) expose several /dev/video* nodes — depth (Z16, 2D) and
    infrared (grayscale, 2D) streams will NOT match; only an actual color
    stream will.
    """
    import glob
    indices = sorted(
        int(d.replace("/dev/video", "")) for d in glob.glob("/dev/video*")
    )
    for index in indices:
        cap = cv2.VideoCapture(index)
        try:
            ok, frame = cap.read()
            if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                return index
        finally:
            cap.release()
    return None


@pytest.mark.hardware
def test_camera_source_real_usb_camera():
    """Open a real USB camera and read real frames. Requires /dev/video* attached."""
    index = _find_color_video_device()
    if index is None:
        pytest.skip(
            "No working color (3-channel) /dev/video* device found — plug in "
            "a USB webcam to run this hardware verification test."
        )

    frame_field = FieldDef.build("frame", "bgr8", (480, 640))
    Topic(TopicDef(
        name="usb_cam_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    source = CameraSource("usb_cam_test", device_index=index, width=640, height=480, fps=30)
    source._session_id = "usb-cam-session"
    source.initialize()

    try:
        n_ok = 0
        for _ in range(10):
            try:
                source.spin_once()
                n_ok += 1
            except StopIteration:
                break
        assert n_ok > 0, "Could not read any frame from the real USB camera"
    finally:
        source.shutdown()
        for p in source._publishers:
            p._bus.close(unlink=True)
