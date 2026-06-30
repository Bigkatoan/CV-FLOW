"""
Tests for cv_flow.executor.Executor

T-EXEC-01: initialize() all nodes before spin
T-EXEC-02: spin_once() called in node order
T-EXEC-03: StopIteration from 1 node → shutdown all, no crash
T-EXEC-04: stop() → graceful shutdown
T-EXEC-05: shutdown() calls node.shutdown() in reverse order
T-EXEC-06: hz=30 → loop no faster than 30 iter/s
T-EXEC-07: spin_background() → returns Thread, non-blocking
T-EXEC-08: elastic=True → monitor thread starts
T-EXEC-09: buffer_depth > threshold → scale_up called (mock)
T-EXEC-10: buffer_depth == 0 many times → scale_down called (mock)
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cv_flow.node import Node
from cv_flow.executor import Executor


class TrackNode(Node):
    """Records call order for init, spin, shutdown."""
    log: list

    def __init__(self, log: list, name: str, max_spins: int = 3):
        super().__init__(name=name)
        self.log       = log
        self._max      = max_spins
        self._spun     = 0

    def initialize(self):
        self.log.append(f"init:{self.name}")

    def spin_once(self):
        self._spun += 1
        self.log.append(f"spin:{self.name}:{self._spun}")
        if self._spun >= self._max:
            raise StopIteration

    def shutdown(self):
        self.log.append(f"shutdown:{self.name}")


# ── T-EXEC-01 ─────────────────────────────────────────────────────────────────

def test_exec_01_initialize_before_spin():
    """initialize() on all nodes happens before any spin_once()."""
    log = []
    n1  = TrackNode(log, "n1", max_spins=1)
    n2  = TrackNode(log, "n2", max_spins=1)
    Executor([n1, n2]).spin()

    init_positions  = [i for i, e in enumerate(log) if e.startswith("init:")]
    spin_positions  = [i for i, e in enumerate(log) if e.startswith("spin:")]
    assert max(init_positions) < min(spin_positions), \
        "All init() calls must finish before first spin()"


# ── T-EXEC-02 ─────────────────────────────────────────────────────────────────

def test_exec_02_spin_order():
    """spin_once() is called in declared node order each iteration."""
    log = []

    class _InfiniteNode(Node):
        def __init__(self, name, log): super().__init__(name=name); self.log = log
        def spin_once(self): self.log.append(self.name)

    class _StopAfter3(Node):
        def __init__(self, name, log):
            super().__init__(name=name); self.log = log; self._n = 0
        def spin_once(self):
            self.log.append(self.name); self._n += 1
            if self._n >= 3: raise StopIteration

    n1 = _InfiniteNode("A", log)
    n2 = _InfiniteNode("B", log)
    n3 = _StopAfter3("C",  log)
    Executor([n1, n2, n3]).spin()

    # In every iteration, A must come before B which must come before C
    iterations = []
    itr = []
    for entry in log:
        itr.append(entry)
        if entry == "C":
            iterations.append(list(itr)); itr = []

    for itr in iterations:
        assert itr.index("A") < itr.index("B") < itr.index("C")


# ── T-EXEC-03 ─────────────────────────────────────────────────────────────────

def test_exec_03_stopiteration_shuts_all():
    """StopIteration from one node → all nodes get shutdown(), no crash."""
    log = []
    n1  = TrackNode(log, "n1", max_spins=1)
    n2  = TrackNode(log, "n2", max_spins=999)  # would run forever
    Executor([n1, n2]).spin()

    assert "shutdown:n1" in log
    assert "shutdown:n2" in log


# ── T-EXEC-04 ─────────────────────────────────────────────────────────────────

def test_exec_04_stop_graceful():
    """stop() causes spin() to exit gracefully."""
    spins = [0]

    class _InfiniteNode(Node):
        def spin_once(self): spins[0] += 1; time.sleep(0.01)

    exec_ = Executor([_InfiniteNode()])
    t = exec_.spin_background()
    time.sleep(0.05)
    exec_.stop()
    t.join(timeout=2.0)

    assert not t.is_alive(), "Executor did not stop after stop()"
    assert spins[0] > 0


# ── T-EXEC-05 ─────────────────────────────────────────────────────────────────

def test_exec_05_shutdown_reverse_order():
    """shutdown() calls node.shutdown() in REVERSE node order."""
    log = []
    n1  = TrackNode(log, "n1", max_spins=1)
    n2  = TrackNode(log, "n2", max_spins=999)
    n3  = TrackNode(log, "n3", max_spins=999)
    Executor([n1, n2, n3]).spin()

    shutdowns = [e for e in log if e.startswith("shutdown:")]
    names = [s.split(":")[1] for s in shutdowns]
    assert names == ["n3", "n2", "n1"], f"Expected reverse order, got {names}"


# ── T-EXEC-06 ─────────────────────────────────────────────────────────────────

def test_exec_06_hz_rate_limit():
    """hz=30 → loop does not exceed ~30 iterations/s."""
    spins = [0]
    t0    = [0.0]

    class _SpinCounter(Node):
        def spin_once(self):
            spins[0] += 1
            if spins[0] >= 10:
                raise StopIteration

    t0[0] = time.monotonic()
    Executor([_SpinCounter()], hz=30).spin()
    elapsed = time.monotonic() - t0[0]

    # 10 iterations at 30Hz should take ≥ 10/30 ≈ 0.333s
    assert elapsed >= 0.30, f"Too fast: {elapsed:.3f}s for 10 spins at 30Hz"
    assert elapsed < 2.0,   f"Too slow: {elapsed:.3f}s"


# ── T-EXEC-07 ─────────────────────────────────────────────────────────────────

def test_exec_07_spin_background():
    """spin_background() returns a Thread and doesn't block."""
    class _Quick(Node):
        def __init__(self): super().__init__(); self._n = 0
        def spin_once(self):
            self._n += 1
            time.sleep(0.01)
            if self._n >= 3:
                raise StopIteration

    exec_ = Executor([_Quick()])
    t0 = time.monotonic()
    thr = exec_.spin_background()
    elapsed = time.monotonic() - t0

    assert isinstance(thr, threading.Thread)
    assert elapsed < 0.1, "spin_background() should return immediately"
    thr.join(timeout=2.0)


# ── T-EXEC-08 ─────────────────────────────────────────────────────────────────

def test_exec_08_elastic_monitor_starts():
    """elastic=True → the elastic monitor thread starts."""
    threads_before = threading.active_count()

    class _Quick(Node):
        def __init__(self): super().__init__(); self._n = 0
        def spin_once(self):
            self._n += 1
            if self._n >= 2:
                raise StopIteration

    exec_ = Executor([_Quick()], elastic=True)
    thr = exec_.spin_background()
    time.sleep(0.05)
    threads_during = threading.active_count()
    thr.join(timeout=2.0)

    # At least the elastic monitor thread was alive during spin
    assert threads_during > threads_before


# ── T-EXEC-09 ─────────────────────────────────────────────────────────────────

def test_exec_09_scale_up_called():
    """scale_up() is called when a node's buffer_depth > threshold."""
    scale_up_calls = [0]

    class _MockElasticNode(Node):
        _elastic_capable = True
        def get_buffer_depth(self): return 10  # always high
        def spin_once(self): time.sleep(0.01)

    class _TrackedExecutor(Executor):
        def scale_up(self, node):
            scale_up_calls[0] += 1
            self.stop()

    node  = _MockElasticNode()
    exec_ = _TrackedExecutor([node], elastic=True)
    thr   = exec_.spin_background()
    thr.join(timeout=3.0)

    assert scale_up_calls[0] > 0, "scale_up() was never called"


# ── T-EXEC-10 ─────────────────────────────────────────────────────────────────

def test_exec_10_scale_down_called():
    """scale_down() is called when buffer_depth == 0 consistently."""
    scale_down_calls = [0]

    class _IdleElasticNode(Node):
        _elastic_capable = True
        def get_buffer_depth(self): return 0   # always idle
        def spin_once(self): time.sleep(0.01)

    class _TrackedExecutor(Executor):
        def scale_down(self, node):
            scale_down_calls[0] += 1
            self.stop()

    node  = _IdleElasticNode()
    exec_ = _TrackedExecutor([node], elastic=True)
    thr   = exec_.spin_background()
    thr.join(timeout=5.0)

    assert scale_down_calls[0] > 0, "scale_down() was never called"
