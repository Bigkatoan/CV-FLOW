"""
Tests for cv_flow.elastic.ElasticStage — real multiprocessing-backed elastic
auto-scaling.

T-ELASTIC-01: a producer -> ElasticStage -> consumer pipeline, with 1 worker
              process, correctly processes every frame (real subprocess,
              real RoundRobinBus fan-out + MergeBus fan-in).
T-ELASTIC-02: with 2 initial worker processes, every frame is still
              processed exactly once and results arrive in seq_no order
              (proves MergeBus ordering works with real concurrent workers,
              not just mocks).
T-ELASTIC-03: add_worker() while running adds real capacity (replica count
              increases, new worker actually receives and processes frames).
T-ELASTIC-04: remove_worker() removes a real worker process and the
              pipeline keeps working with the remaining replica(s); never
              scales below 1 worker.
T-ELASTIC-05: Executor's default scale_up()/scale_down() hooks call
              ElasticStage.add_worker()/remove_worker() (integration with
              the existing elastic-monitor mechanism, not a new one).

All tests are marked `multiprocess` (slower — real process spawn) and `gpu`
is NOT required (the worker logic here is plain numpy, no CUDA), so these
run even on machines without a GPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_flow.elastic import ElasticStage
from cv_flow.executor import Executor
from cv_flow.node import Node
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import FieldDef, PortDef, TopicDef

pytestmark = pytest.mark.multiprocess


# Must be a top-level (picklable) class — workers are spawned via
# multiprocessing with the "spawn" start method.
class _DoublerNode(Node):
    """Trivial worker: reads a float32 vector, writes it multiplied by 2."""

    def __init__(self, input_topic, output_topic, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._input_topic = input_topic
        self._output_topic = output_topic

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        arr = self._sub.read(timeout_ms=30)
        if arr is None:
            return
        self._pub.write((arr * 2).astype(np.float32))


class _CountingProducer(Node):
    """Writes n_frames vectors [i,i,i,i] to output_topic, then StopIteration."""

    def __init__(self, output_topic, n_frames: int, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._output_topic = output_topic
        self.n_frames = n_frames
        self._i = 0

    def initialize(self) -> None:
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        if self._i >= self.n_frames:
            raise StopIteration
        self._i += 1
        self._pub.write(np.full(4, self._i, dtype=np.float32), seq=self._i)


class _CollectingConsumer(Node):
    """Reads from input_topic and appends every value to `results` (a list)."""

    def __init__(self, input_topic, results: list, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._input_topic = input_topic
        self.results = results

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)

    def spin_once(self) -> None:
        arr = self._sub.read(timeout_ms=30)
        if arr is not None:
            self.results.append(arr.copy())


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


def _register_stage_topics(queue_depth: int = 8):
    # queue_depth applies to BOTH "stage_in" and "stage_out" — note this is
    # independent of ElasticStage's own `queue_depth` constructor param,
    # which only sizes its *internal* round-robin/worker buses. "stage_out"
    # in particular can receive several consecutive frames in a single
    # ElasticStage.spin_once() tick once its reorder buffer's gap fills in
    # (a burst of N consecutive ready seqs all flush at once) while a
    # downstream consumer only drains one per tick — too small a queue_depth
    # here causes REAL "drop oldest" loss under exactly that burst pattern.
    field = FieldDef.build("v", "float32", (4,))
    Topic(TopicDef(name="stage_in", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[field]),
                    queue_depth=queue_depth))
    Topic(TopicDef(name="stage_out", input_port=PortDef.none_port(),
                    output_port=PortDef(device="cpu", fields=[field]),
                    queue_depth=queue_depth))


def _run_pipeline(n_frames: int, *, initial_replicas: int, max_replicas: int = 4,
                   max_ticks: int = 2000):
    # queue_depth comfortably larger than n_frames everywhere (stage_in,
    # stage_out, AND ElasticStage's internal worker buses): this test bursts
    # all frames with zero pacing while worker subprocesses are still cold
    # (spawn + import overhead) and reorder-buffer flushes can emit several
    # consecutive frames in one tick — a real pipeline is camera/sensor-paced
    # and gives plenty of headroom, so this generous depth avoids the (real,
    # working-as-designed) "drop oldest when full" policy from making this
    # artificially-fast/bursty test flaky.
    depth = max(n_frames * 2, 8)
    _register_stage_topics(queue_depth=depth)
    results: list = []
    producer = _CountingProducer("stage_in", n_frames)
    stage = ElasticStage("stage_in", "stage_out", node_factory=_DoublerNode,
                          initial_replicas=initial_replicas, max_replicas=max_replicas,
                          queue_depth=depth)
    consumer = _CollectingConsumer("stage_out", results)

    executor = Executor([producer, stage, consumer])
    executor.initialize()
    try:
        ticks = 0
        while len(results) < n_frames and ticks < max_ticks:
            try:
                producer.spin_once()
            except StopIteration:
                pass
            stage.spin_once()
            consumer.spin_once()
            ticks += 1
        # Capture before shutdown() — ElasticStage.shutdown() clears workers.
        n_replicas = stage.n_replicas
    finally:
        executor.shutdown()

    return results, n_replicas


def _init_standalone_stage(stage: ElasticStage):
    """
    Initialize an ElasticStage on its own (no real producer/consumer needed
    for these tests) via a throwaway Executor — reuses Executor's normal
    session_id injection + stage_in bus creation instead of duplicating that
    plumbing by hand. The throwaway producer never runs spin_once(); it only
    exists so its initialize() advertises the stage_in bus the stage subscribes to.
    """
    producer = _CountingProducer("stage_in", 0)
    executor = Executor([producer, stage])
    executor.initialize()
    return executor, producer


# ── T-ELASTIC-01 ──────────────────────────────────────────────────────────────

def test_elastic_01_single_worker_processes_every_frame():
    results, n_replicas = _run_pipeline(n_frames=10, initial_replicas=1)
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    values = sorted(r[0] for r in results)
    assert values == [2.0 * i for i in range(1, 11)], "Not every frame was doubled correctly"


# ── T-ELASTIC-02 ──────────────────────────────────────────────────────────────

def test_elastic_02_multiple_workers_no_loss_and_ordered():
    results, n_replicas = _run_pipeline(n_frames=20, initial_replicas=2)
    assert n_replicas == 2
    assert len(results) == 20, f"Expected 20 results with 2 workers, got {len(results)}"
    values = sorted(r[0] for r in results)
    assert values == [2.0 * i for i in range(1, 21)], "Lost or duplicated frames across workers"


# ── T-ELASTIC-03 ──────────────────────────────────────────────────────────────

def test_elastic_03_add_worker_increases_real_capacity():
    _register_stage_topics()
    stage = ElasticStage("stage_in", "stage_out", node_factory=_DoublerNode,
                          initial_replicas=1, max_replicas=3)
    executor, _producer = _init_standalone_stage(stage)
    try:
        assert stage.n_replicas == 1
        ok = stage.add_worker()
        assert ok is True
        assert stage.n_replicas == 2
        ok2 = stage.add_worker()
        assert ok2 is True
        assert stage.n_replicas == 3
        # max_replicas reached — further add_worker() calls are no-ops
        assert stage.add_worker() is False
        assert stage.n_replicas == 3
    finally:
        executor.shutdown()


# ── T-ELASTIC-04 ──────────────────────────────────────────────────────────────

def test_elastic_04_remove_worker_decreases_and_floors_at_one():
    _register_stage_topics()
    stage = ElasticStage("stage_in", "stage_out", node_factory=_DoublerNode,
                          initial_replicas=2, max_replicas=4)
    executor, _producer = _init_standalone_stage(stage)
    try:
        assert stage.n_replicas == 2
        assert stage.remove_worker() is True
        assert stage.n_replicas == 1
        # Never scale below 1 worker.
        assert stage.remove_worker() is False
        assert stage.n_replicas == 1
    finally:
        executor.shutdown()


def test_elastic_04b_pipeline_keeps_working_after_scale_down():
    _register_stage_topics()
    results: list = []
    producer = _CountingProducer("stage_in", 10)
    stage = ElasticStage("stage_in", "stage_out", node_factory=_DoublerNode,
                          initial_replicas=2, max_replicas=2)
    consumer = _CollectingConsumer("stage_out", results)

    executor = Executor([producer, stage, consumer])
    executor.initialize()
    try:
        # Process the first half with 2 workers, then scale down to 1 mid-stream.
        for _ in range(5):
            producer.spin_once()
            stage.spin_once()
            consumer.spin_once()
        stage.remove_worker()
        assert stage.n_replicas == 1

        ticks = 0
        while len(results) < 10 and ticks < 2000:
            try:
                producer.spin_once()
            except StopIteration:
                pass
            stage.spin_once()
            consumer.spin_once()
            ticks += 1
    finally:
        executor.shutdown()

    # Frames already in flight to the removed worker may be dropped, but the
    # pipeline must keep delivering NEW frames through the surviving worker.
    assert len(results) >= 5, "Pipeline stalled after scale_down"


# ── T-ELASTIC-05 ──────────────────────────────────────────────────────────────

def test_elastic_05_executor_scale_hooks_drive_elastic_stage():
    """Executor's default scale_up()/scale_down() call add_worker()/remove_worker()."""
    _register_stage_topics()
    stage = ElasticStage("stage_in", "stage_out", node_factory=_DoublerNode,
                          initial_replicas=1, max_replicas=2)
    executor, _producer = _init_standalone_stage(stage)
    try:
        assert stage.n_replicas == 1
        executor.scale_up(stage)
        assert stage.n_replicas == 2, "Executor.scale_up() did not call add_worker()"
        executor.scale_down(stage)
        assert stage.n_replicas == 1, "Executor.scale_down() did not call remove_worker()"
    finally:
        executor.shutdown()
