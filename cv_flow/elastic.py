"""
cv_flow.elastic — ElasticStage: real multiprocessing-backed auto-scaling.

A drop-in replacement for a single transform Node (e.g. YoloInference) in a
pipeline's node list. Internally it:
  - subscribes to its input topic and advertises its output topic exactly
    like any other Node (so upstream/downstream nodes need no special
    awareness of elasticity),
  - fans every frame out to one of N real `multiprocessing.Process` workers,
    each through its own dedicated PortBus pair (input + output),
  - each worker runs its own instance of `node_factory(in_topic, out_topic)`
    (e.g. `functools.partial(YoloInference, model_path=..., device="cuda:0")`)
    in total isolation (own CUDA context, own model load),
  - merges worker results back in original seq_no order and republishes
    them downstream.

`Executor.scale_up()`/`scale_down()` call `add_worker()`/`remove_worker()`
on any node that has them (see executor.py) — ElasticStage is such a node.

Worker processes are spawned with multiprocessing's "spawn" start method
(required for CUDA safety) — node_factory and its bound arguments must be
picklable (a plain class + simple constructor kwargs, or
functools.partial(SomeNodeClass, kw=...), works; a lambda does not).

Concurrency note (read before changing this file): PortBus's header
(write_count/read_count/drop_count) is a single struct updated via one
`_write_header()` call that writes all three fields based on whatever the
caller last read — safe when a single writer and single reader are
strictly sequential (the common case: one Executor, one process, one
thread), but NOT safe when a real writer process and a real reader process
touch the SAME bus concurrently (no cross-process atomicity/locking exists
in PortBus itself). ElasticStage is the first place in this codebase where
that actually happens (a worker process reads while the main process
writes, on each worker's dedicated bus pair), and it manifested as rare
duplicate/dropped frames during testing. Fix: every worker gets its own
`multiprocessing.Lock`, held by BOTH sides (main process write / worker
process read on the input bus; worker process write / main process
read+peek on the output bus) for the duration of each bus operation. This
is scoped to ElasticStage's specific concurrent-access pattern rather than
adding locking to PortBus's general API.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import uuid
from typing import Callable

from cv_flow.dam.bus import PortBus
from cv_flow.node import Node
from cv_flow.topic.topic import Topic
from cv_flow.topic.types import PortDef, TopicDef

logger = logging.getLogger("cv_flow.elastic")


def _wrap_with_lock(node: Node, lock) -> None:
    """
    Make a worker node's subscribe-read and publish-write calls hold `lock`
    for their duration, matching the lock the main process holds around its
    writes to this worker's input bus and reads from its output bus.

    Also makes output carry the SAME seq_no as input (Publisher's default —
    an independent per-publisher auto-incrementing counter — would make two
    different workers' "seq=1" unrelated frames, breaking the global
    ordering ElasticStage reconstructs downstream). Assumes the common
    single-subscriber/single-publisher transform shape used by every
    elastic_capable node in NODE_CATALOG (YoloInference, OnnxInference,
    Preprocess, ...): `self._sub` / `self._pub`. Nodes that don't expose
    that shape are left untouched.
    """
    sub = getattr(node, "_sub", None)
    pub = getattr(node, "_pub", None)
    if sub is None or pub is None:
        return

    original_read = sub.read
    original_write = pub.write

    def _locked_read(timeout_ms=30):
        with lock:
            return original_read(timeout_ms=timeout_ms)

    def _write_preserving_seq(data, *, seq=None, detections=[], metadata={}):
        if seq is None:
            seq = sub.last_seq
        with lock:
            return original_write(data, seq=seq, detections=detections, metadata=metadata)

    sub.read = _locked_read
    pub.write = _write_preserving_seq


def _worker_main(
    node_factory: Callable,
    in_topic_name: str,
    out_topic_name: str,
    input_port: PortDef,
    output_port: PortDef,
    queue_depth: int,
    worker_session_id: str,
    lock,
    stop_event,
    ready_event,
    error_queue,
) -> None:
    """Top-level (picklable) entry point run inside each worker process."""
    try:
        Topic(TopicDef(
            name=in_topic_name, input_port=PortDef.none_port(),
            output_port=input_port, queue_depth=queue_depth,
        ))
        Topic(TopicDef(
            name=out_topic_name, input_port=PortDef.none_port(),
            output_port=output_port, queue_depth=queue_depth,
        ))

        node = node_factory(in_topic_name, out_topic_name)
        node._session_id = worker_session_id
        node.initialize()
        _wrap_with_lock(node, lock)
    except Exception as exc:  # noqa: BLE001
        error_queue.put(repr(exc))
        ready_event.set()
        return

    ready_event.set()
    try:
        while not stop_event.is_set():
            try:
                node.spin_once()
            except StopIteration:
                break
            except Exception:  # noqa: BLE001
                logger.exception("elastic worker %s: spin_once() error", worker_session_id)
    finally:
        try:
            node.shutdown()
        except Exception:  # noqa: BLE001
            pass


class ElasticStage(Node):
    """
    See module docstring. Behaves as a Node: place it in a pipeline's node
    list exactly where the wrapped transform node would otherwise go.
    """

    _elastic_capable = True

    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        *,
        node_factory: Callable,
        max_replicas: int = 4,
        initial_replicas: int = 1,
        queue_depth: int = 8,
        worker_ready_timeout_s: float = 120.0,
        reorder_stall_timeout_s: float = 30.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic
        self.node_factory  = node_factory
        self.max_replicas  = max(1, max_replicas)
        self._initial_replicas = max(1, min(initial_replicas, self.max_replicas))
        self.queue_depth   = queue_depth
        self.worker_ready_timeout_s = worker_ready_timeout_s
        self.reorder_stall_timeout_s = reorder_stall_timeout_s

        self._lock = threading.Lock()
        self._workers: list[dict] = []
        self._instance_id = uuid.uuid4().hex[:8]
        self._mp_ctx = mp.get_context("spawn")
        self._rr_counter = 0  # manual round-robin index into self._workers

        # Reorder buffer: workers can finish out of arrival order (one
        # briefly slower than another), so results are buffered here and
        # only released downstream in strict consecutive seq order.
        self._next_seq_to_emit: int | None = None
        self._reorder_buffer: dict[int, tuple] = {}
        # None until the first upstream frame arrives — avoids the stall
        # clock appearing already-expired due to worker spawn time elapsed
        # between __init__() and the first real spin_once() tick.
        self._last_emit_monotonic: float | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

        in_port  = self._sub._port_def
        out_port = self._pub._port_def
        self._in_port_bytes  = max(in_port.total_bytes, 1)
        self._out_port_bytes = max(out_port.total_bytes, 1)
        self._in_port_def    = in_port
        self._out_port_def   = out_port

        for _ in range(self._initial_replicas):
            self.add_worker()

    def spin_once(self) -> None:
        # Fan upstream frame out to a worker (manual round-robin; each
        # worker's bus write is guarded by that worker's own lock — see
        # module docstring's concurrency note).
        upstream = self._sub._bus.read(timeout_ms=30)
        if upstream is not None:
            raw, seq, dets, meta = upstream
            if self._next_seq_to_emit is None:
                self._next_seq_to_emit = seq
                self._last_emit_monotonic = time.monotonic()
            with self._lock:
                workers = list(self._workers)
            if workers:
                worker = workers[self._rr_counter % len(workers)]
                self._rr_counter += 1
                with worker["lock"]:
                    worker["in_bus"].write(raw, seq, detections=dets, metadata=meta)

        # Drain whatever worker results are ready this tick into the
        # reorder buffer (each peek+read happens under that worker's own
        # lock, so it can't race the worker's own write — see concurrency
        # note in the module docstring).
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            with worker["lock"]:
                result = worker["out_bus"].read(timeout_ms=0)
            if result is not None:
                raw, seq, dets, meta = result
                self._reorder_buffer[seq] = (raw, dets, meta)

        # If the buffer has been stuck on a gap (e.g. the upstream "stage_in"
        # bus itself dropped a frame under backpressure) for too long, skip
        # ahead to the lowest seq actually available instead of stalling the
        # pipeline forever — consistent with this project's existing
        # drop-oldest-and-continue policy elsewhere (PortBus.write()).
        if (self._reorder_buffer and self._last_emit_monotonic is not None
                and time.monotonic() - self._last_emit_monotonic > self.reorder_stall_timeout_s):
            skip_to = min(self._reorder_buffer)
            logger.warning(
                "ElasticStage '%s': reorder buffer stalled waiting for seq=%s for "
                "%.1fs — skipping ahead to seq=%s (upstream likely dropped a frame).",
                self.name, self._next_seq_to_emit, self.reorder_stall_timeout_s, skip_to,
            )
            self._next_seq_to_emit = skip_to

        # Emit every consecutive run starting at the next expected seq.
        while (self._next_seq_to_emit is not None
               and self._next_seq_to_emit in self._reorder_buffer):
            raw, dets, meta = self._reorder_buffer.pop(self._next_seq_to_emit)
            self._pub._bus.write(raw, self._next_seq_to_emit, detections=dets, metadata=meta)
            self._next_seq_to_emit += 1
            self._last_emit_monotonic = time.monotonic()

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers.clear()
        for w in workers:
            w["stop_event"].set()
        for w in workers:
            w["process"].join(timeout=10)
            if w["process"].is_alive():
                w["process"].terminate()
            w["in_bus"].close(unlink=True)
            w["out_bus"].close(unlink=True)

    # ── scale ─────────────────────────────────────────────────────────────────

    def add_worker(self) -> bool:
        with self._lock:
            if len(self._workers) >= self.max_replicas:
                return False
            idx = len(self._workers)

        worker_session = f"{self._instance_id}_{idx}"
        in_topic_name  = f"{self.name}_{self._instance_id}__wkin"
        out_topic_name = f"{self.name}_{self._instance_id}__wkout"

        # Names chosen so the worker's own Node.subscribe()/advertise() calls
        # (which derive bus names as f"{topic_name}_{session_id}") resolve to
        # exactly these same shared-memory segments.
        in_bus_name  = f"{in_topic_name}_{worker_session}"
        out_bus_name = f"{out_topic_name}_{worker_session}"
        in_bus = PortBus(in_bus_name, self._in_port_bytes, queue_depth=self.queue_depth,
                          create=True)

        lock = self._mp_ctx.Lock()
        stop_event  = self._mp_ctx.Event()
        ready_event = self._mp_ctx.Event()
        error_queue = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(
            target=_worker_main,
            args=(self.node_factory, in_topic_name, out_topic_name,
                  self._in_port_def, self._out_port_def, self.queue_depth,
                  worker_session, lock, stop_event, ready_event, error_queue),
            daemon=True,
        )
        proc.start()

        if not ready_event.wait(timeout=self.worker_ready_timeout_s):
            logger.warning("ElasticStage '%s': worker %d did not become ready within %.0fs",
                           self.name, idx, self.worker_ready_timeout_s)
        if not error_queue.empty():
            err = error_queue.get()
            logger.error("ElasticStage '%s': worker %d failed to initialize: %s",
                         self.name, idx, err)
            proc.join(timeout=5)
            in_bus.close(unlink=True)
            return False

        out_bus = PortBus(out_bus_name, self._out_port_bytes, create=False)

        with self._lock:
            self._workers.append({
                "process": proc, "stop_event": stop_event, "lock": lock,
                "in_bus": in_bus, "out_bus": out_bus,
            })
        logger.info("ElasticStage '%s': scaled up to %d worker(s)", self.name, idx + 1)
        return True

    def remove_worker(self) -> bool:
        with self._lock:
            if len(self._workers) <= 1:
                return False  # always keep at least 1 worker
            worker = self._workers.pop()

        worker["stop_event"].set()
        worker["process"].join(timeout=10)
        if worker["process"].is_alive():
            worker["process"].terminate()
        worker["in_bus"].close(unlink=True)
        worker["out_bus"].close(unlink=True)

        logger.info("ElasticStage '%s': scaled down to %d worker(s)",
                    self.name, len(self._workers))
        return True

    # ── monitoring ────────────────────────────────────────────────────────────

    def get_buffer_depth(self) -> float:
        with self._lock:
            workers = list(self._workers)
        if not workers:
            return 0.0
        return sum(w["in_bus"].get_buffer_depth() for w in workers) / len(workers)

    @property
    def n_replicas(self) -> int:
        with self._lock:
            return len(self._workers)
