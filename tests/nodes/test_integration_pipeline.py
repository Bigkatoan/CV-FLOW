"""
End-to-end integration test: VideoFileSource -> DrawBbox -> VideoWriter.

Builds a real synthetic .mp4 file on disk, runs it through the full
DAM/Topic/Node/Executor stack with real PortBus shared memory (not mocks),
and verifies an annotated output video file is produced.

This is the integration test required by the build order for step 13
("nodes/*.py -> end-to-end pipeline runs").
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
cv2 = pytest.importorskip("cv2")

from cv_flow.nodes.camera import VideoFileSource
from cv_flow.nodes.draw import DrawBbox
from cv_flow.nodes.output import VideoWriter
from cv_flow.node import Node
from cv_flow.executor import Executor
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


def _make_test_video(path: Path, n_frames: int = 5, w: int = 64, h: int = 48) -> None:
    """Write a small synthetic .mp4 with n_frames solid-color frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _DetSource(Node):
    """Synthetic detection generator — feeds a fixed box into the dets topic."""

    def __init__(self, output_topic, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._output_topic = output_topic

    def initialize(self) -> None:
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        boxes     = np.array([[5, 5, 30, 30], [0, 0, 0, 0]], dtype=np.float32)
        scores    = np.array([0.99, 0], dtype=np.float32)
        class_ids = np.array([0, -1], dtype=np.int32)
        self._pub.write({"boxes": boxes, "scores": scores, "class_ids": class_ids})


def test_full_pipeline_video_to_annotated_video():
    """VideoFileSource -> DrawBbox -> VideoWriter produces a real output file."""
    tmp = Path(tempfile.mkdtemp())
    input_video  = tmp / "input.mp4"
    output_video = tmp / "output.mp4"
    n_frames = 5
    w, h = 64, 48
    _make_test_video(input_video, n_frames=n_frames, w=w, h=h)

    frame_field = FieldDef.build("frame", "bgr8", (h, w))
    Topic(TopicDef(
        name="it_frame",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    det_fields = [
        FieldDef.build("boxes",     "float32", (2, 4)),
        FieldDef.build("scores",    "float32", (2,)),
        FieldDef.build("class_ids", "int32",   (2,)),
    ]
    Topic(TopicDef(
        name="it_dets",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=det_fields),
    ))

    Topic(TopicDef(
        name="it_annotated",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    class _StoppingSource(VideoFileSource):
        """VideoFileSource that signals StopIteration once frames are exhausted."""

    source   = _StoppingSource("it_frame", path=str(input_video), loop=False, fps=0.0)
    det_src  = _DetSource("it_dets")
    drawer   = DrawBbox("it_frame", "it_dets", "it_annotated")
    writer   = VideoWriter("it_annotated", output_path=str(output_video), fps=10.0)

    # Run for exactly n_frames iterations, then stop deterministically.
    counter = {"n": 0}
    orig_spin = source.spin_once

    def _counted_spin():
        orig_spin()
        counter["n"] += 1
        if counter["n"] >= n_frames:
            raise StopIteration

    source.spin_once = _counted_spin

    exec_ = Executor([source, det_src, drawer, writer])
    exec_.spin()

    assert output_video.exists(), "Output video file was not created"
    assert output_video.stat().st_size > 0, "Output video file is empty"

    # Verify the output video is actually readable and has frames
    cap = cv2.VideoCapture(str(output_video))
    read_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_count += 1
        assert frame.shape == (h, w, 3)
    cap.release()
    assert read_count > 0, "Output video has no readable frames"

    for node in [source, det_src, drawer, writer]:
        for p in getattr(node, "_publishers", []):
            p._bus.close(unlink=True)
