"""
AutoScaler — dynamic node worker scaling based on PortBus buffer depth.

When a node's input buffer accumulates more frames than scale_up_threshold,
the AutoScaler spawns an additional NodeWorker for that node.  Frames are
distributed round-robin across workers via RoundRobinBus and merged back via
MergeBus.  When buffer drains below scale_down_threshold the extra worker is
gracefully stopped.

The AutoScaler runs in the pipeline runner's monitor thread (not in a separate
process) — it only manages worker lifecycles, not data.

Scaling invariants:
  - min_workers_per_node ≤ active_workers ≤ max_workers_per_node
  - cooldown_s seconds must pass between any two scale events per node
  - The primary worker (worker_index=0) is never removed
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from engine.core.dam import MergeBus, PortBus, RoundRobinBus
from engine.core.node_worker import NodeWorker, ResourceLimits

logger = logging.getLogger(__name__)


@dataclass
class ScalingPolicy:
    """Per-node scaling parameters.  Loaded from pipeline JSON 'scaling' key."""
    min_workers: int   = 1
    max_workers: int   = 4
    scale_up_buffer:   int   = 10   # depth threshold to add a worker
    scale_down_buffer: int   = 2    # depth threshold to remove a worker
    cooldown_s: float  = 10.0       # min seconds between scale events

    @classmethod
    def from_dict(cls, d: dict) -> "ScalingPolicy":
        return cls(
            min_workers=d.get("min_workers", 1),
            max_workers=d.get("max_workers", 4),
            scale_up_buffer=d.get("scale_up_buffer", 10),
            scale_down_buffer=d.get("scale_down_buffer", 2),
            cooldown_s=d.get("cooldown_s", 10.0),
        )


@dataclass
class _NodeScaleState:
    workers: list[NodeWorker] = field(default_factory=list)
    last_scale_t: float = 0.0
    policy: ScalingPolicy = field(default_factory=ScalingPolicy)


class AutoScaler:
    """
    Monitors PortBus buffer depths and scales NodeWorker pools up/down.

    Usage
    -----
    autoscaler = AutoScaler()
    autoscaler.register(
        node_id="inference_0",
        workers=[primary_worker],
        input_bus=round_robin_bus,    # PortBus or RoundRobinBus that feeds workers
        output_bus=merge_bus,         # MergeBus or PortBus collecting output
        spawn_fn=lambda idx: make_worker(idx),
        policy=ScalingPolicy(max_workers=4, scale_up_buffer=8),
    )
    # In monitor loop:
    autoscaler.tick()
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeScaleState] = {}
        self._input_buses:  dict[str, PortBus | RoundRobinBus] = {}
        self._spawn_fns:    dict[str, Callable[[int], NodeWorker]] = {}

    def register(
        self,
        node_id: str,
        workers: list[NodeWorker],
        input_bus: "PortBus | RoundRobinBus",
        spawn_fn: Callable[[int], NodeWorker],
        policy: ScalingPolicy | None = None,
    ) -> None:
        """Register a node pool for auto-scaling monitoring."""
        self._nodes[node_id] = _NodeScaleState(
            workers=list(workers),
            policy=policy or ScalingPolicy(),
        )
        self._input_buses[node_id] = input_bus
        self._spawn_fns[node_id]   = spawn_fn

    def tick(self) -> None:
        """
        Check all registered nodes and scale if needed.
        Call this in the monitor thread at ~1 Hz.
        """
        now = time.monotonic()
        for node_id, state in self._nodes.items():
            bus    = self._input_buses[node_id]
            policy = state.policy

            # Get current buffer depth
            if isinstance(bus, RoundRobinBus):
                depth = bus.get_buffer_depth()
            elif isinstance(bus, PortBus):
                depth = bus.get_buffer_depth()
            else:
                depth = 0.0

            n = len(state.workers)
            cooldown_ok = (now - state.last_scale_t) >= policy.cooldown_s

            if depth > policy.scale_up_buffer and n < policy.max_workers and cooldown_ok:
                self._scale_up(node_id, state, now)

            elif depth < policy.scale_down_buffer and n > policy.min_workers and cooldown_ok:
                self._scale_down(node_id, state, now)

    def _scale_up(self, node_id: str, state: _NodeScaleState, now: float) -> None:
        new_idx    = len(state.workers)
        spawn_fn   = self._spawn_fns[node_id]
        new_worker = spawn_fn(new_idx)
        new_worker.start()
        state.workers.append(new_worker)
        state.last_scale_t = now

        # Tell the RoundRobinBus about the new worker's input bus
        input_bus = self._input_buses[node_id]
        if isinstance(input_bus, RoundRobinBus):
            from engine.core.dam import make_port_bus
            new_bus = make_port_bus(
                name=f"{node_id}_in_{new_idx}",
                max_w=new_worker._input_buses.get("frame", PortBus("_")).max_w,
                max_h=new_worker._input_buses.get("frame", PortBus("_")).max_h,
                create=True,
            )
            input_bus.add_bus(new_bus)

        logger.info(
            "AutoScaler: scaled UP node %s → %d workers (buffer depth was high)",
            node_id, len(state.workers),
        )

    def _scale_down(self, node_id: str, state: _NodeScaleState, now: float) -> None:
        victim = state.workers.pop()
        victim.stop(timeout=3.0)
        state.last_scale_t = now

        # Remove the bus from RoundRobinBus
        input_bus = self._input_buses[node_id]
        if isinstance(input_bus, RoundRobinBus) and len(input_bus.buses) > 1:
            removed_bus = input_bus.remove_bus()
            removed_bus.close(unlink=True)

        logger.info(
            "AutoScaler: scaled DOWN node %s → %d workers (buffer depth was low)",
            node_id, len(state.workers),
        )

    def stop_all(self) -> None:
        """Stop all managed workers (called at pipeline shutdown)."""
        for state in self._nodes.values():
            for w in state.workers:
                try:
                    w.stop(timeout=5.0)
                except Exception as e:
                    logger.warning("Worker stop error: %s", e)

    def get_stats(self) -> dict[str, dict]:
        """Aggregate stats for all managed nodes."""
        result = {}
        for node_id, state in self._nodes.items():
            worker_stats = [w.get_stats().to_dict() for w in state.workers]
            bus = self._input_buses.get(node_id)
            depth = bus.get_buffer_depth() if bus else 0
            result[node_id] = {
                "workers": len(state.workers),
                "buffer_depth": depth,
                "per_worker": worker_stats,
            }
        return result
