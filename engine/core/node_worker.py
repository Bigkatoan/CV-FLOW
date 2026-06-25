"""
NodeWorker — runs a single node in an isolated OS process (Python) or thread (C++).

Python nodes run in separate processes to bypass the GIL.  C++ nodes, whose
ctypes calls already release the GIL, run in threads to avoid IPC overhead.
All communication happens exclusively via PortBus (DAM layer).

ResourceLimits lets the orchestrator constrain each node's CPU affinity,
memory, GPU fraction, and max throughput at spawn time.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Type

from engine.core.dam import MergeBus, PortBus, RoundRobinBus
from engine.core.frame_context import FrameContext

logger = logging.getLogger(__name__)

# ── ResourceLimits ────────────────────────────────────────────────────────────

@dataclass
class ResourceLimits:
    """
    Per-node resource constraints applied at worker spawn time.

    cpu_cores: list[int] | None
        Pin this worker to specific CPU cores.
        e.g. [0, 1] → only cores 0 and 1.
        None = no affinity (OS decides).

    max_memory_mb: int | None
        Soft RSS limit.  Worker is logged if it exceeds this; hard kill is
        not performed by default (set kill_on_oom=True to enable).

    gpu_memory_fraction: float | None
        For ONNX GPU sessions: sets device_memory_limit option.
        Range 0.0–1.0 relative to total GPU memory.

    max_fps: float | None
        Rate-limit this node to at most max_fps frames per second.
        Useful for input sources (cameras) or expensive non-critical nodes.

    priority: int
        Unix nice level (−20 = highest, 19 = lowest, 0 = default).
        On Windows maps to BELOW_NORMAL / NORMAL / ABOVE_NORMAL process class.
    """
    cpu_cores:            Optional[list[int]] = None
    max_memory_mb:        Optional[int]       = None
    gpu_memory_fraction:  Optional[float]     = None
    max_fps:              Optional[float]     = None
    priority:             int                 = 0

    @classmethod
    def from_dict(cls, d: dict) -> "ResourceLimits":
        return cls(
            cpu_cores=d.get("cpu_cores"),
            max_memory_mb=d.get("max_memory_mb"),
            gpu_memory_fraction=d.get("gpu_memory_fraction"),
            max_fps=d.get("max_fps"),
            priority=d.get("priority", 0),
        )


# ── Stats shared memory ───────────────────────────────────────────────────────

@dataclass
class NodeStats:
    avg_ms:   float = 0.0
    p95_ms:   float = 0.0
    fps:      float = 0.0
    frames:   int   = 0
    errors:   int   = 0

    def to_dict(self) -> dict:
        return {
            "avg_ms":  round(self.avg_ms, 2),
            "p95_ms":  round(self.p95_ms, 2),
            "fps":     round(self.fps, 2),
            "frames":  self.frames,
            "errors":  self.errors,
        }


def _make_stats_shm():
    """Create a small shared memory block for stats (10 float64 = 80 bytes)."""
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(create=True, size=80)
    return shm


def _read_stats(shm) -> NodeStats:
    import struct
    vals = struct.unpack_from("<8d", shm.buf)
    # [avg_ms, p95_ms, fps, frames, errors, _reserved×3]
    return NodeStats(
        avg_ms=vals[0], p95_ms=vals[1], fps=vals[2],
        frames=int(vals[3]), errors=int(vals[4]),
    )


def _update_stats_shm(shm, elapsed_ms: float, window: list, t_last: list) -> None:
    """Called by worker after each frame to update rolling stats."""
    import struct
    window.append(elapsed_ms)
    if len(window) > 100:
        window.pop(0)
    now = time.monotonic()
    dt  = now - t_last[0] if t_last[0] else 1.0
    t_last[0] = now

    avg  = sum(window) / len(window) if window else 0.0
    p95  = sorted(window)[int(len(window) * 0.95)] if window else 0.0
    fps  = 1.0 / dt if dt > 0 else 0.0
    frames = struct.unpack_from("<d", shm.buf, 24)[0] + 1
    errors = struct.unpack_from("<d", shm.buf, 32)[0]

    struct.pack_into("<8d", shm.buf, 0, avg, p95, fps, frames, errors, 0.0, 0.0, 0.0)


def _inc_error_shm(shm) -> None:
    import struct
    errors = struct.unpack_from("<d", shm.buf, 32)[0] + 1
    struct.pack_into("<d", shm.buf, 32, errors)


# ── Resource application helpers ──────────────────────────────────────────────

def _apply_resource_limits(limits: Optional[ResourceLimits]) -> None:
    """Called inside the worker process/thread right after startup."""
    if limits is None:
        return

    # CPU affinity
    if limits.cpu_cores:
        try:
            if sys.platform == "win32":
                mask = sum(1 << c for c in limits.cpu_cores)
                import ctypes
                ctypes.windll.kernel32.SetProcessAffinityMask(  # type: ignore[attr-defined]
                    ctypes.windll.kernel32.GetCurrentProcess(), mask)
            else:
                os.sched_setaffinity(0, set(limits.cpu_cores))
        except Exception as e:
            logger.warning("CPU affinity failed: %s", e)

    # Priority / nice
    if limits.priority != 0:
        try:
            if sys.platform == "win32":
                import ctypes
                # Map nice to Windows priority class
                pclass = 0x00004000 if limits.priority > 5 else (
                         0x00008000 if limits.priority < -5 else 0x00000020)
                ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
                    ctypes.windll.kernel32.GetCurrentProcess(), pclass)
            else:
                os.nice(limits.priority)
        except Exception as e:
            logger.warning("Priority change failed: %s", e)

    # Memory limit (Linux only via resource module)
    if limits.max_memory_mb and sys.platform != "win32":
        try:
            import resource
            limit_bytes = limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))
        except Exception as e:
            logger.warning("Memory limit failed: %s", e)


# ── Worker main functions (run inside child process/thread) ───────────────────

def _python_node_worker_main(
    node_class_module: str,
    node_class_name: str,
    node_id: str,
    node_type: str,
    config: dict,
    input_bus_specs: list[tuple[str, str, int, int, int]],   # (port_name, shm_name, w, h, c)
    output_bus_specs: list[tuple[str, str, int, int, int]],
    stats_shm_name: str,
    shutdown_event: multiprocessing.Event,  # type: ignore[type-arg]
    limits: Optional[ResourceLimits],
) -> None:
    """Entry point for a Python node worker process."""
    import importlib
    from multiprocessing.shared_memory import SharedMemory

    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [{node_id}] %(levelname)s %(message)s")
    _apply_resource_limits(limits)

    # Open stats shm
    stats_shm = SharedMemory(name=stats_shm_name, create=False)

    # Attach to PortBuses (consumer side — create=False)
    input_buses: dict[str, PortBus] = {}
    for port_name, shm_name, w, h, c in input_bus_specs:
        input_buses[port_name] = PortBus(shm_name, max_w=w, max_h=h, max_c=c, create=False)

    output_buses: dict[str, PortBus] = {}
    for port_name, shm_name, w, h, c in output_bus_specs:
        output_buses[port_name] = PortBus(shm_name, max_w=w, max_h=h, max_c=c, create=False)

    # Instantiate and setup node
    mod = importlib.import_module(node_class_module)
    node_class = getattr(mod, node_class_name)
    node = node_class()
    try:
        node.setup(node_id, config, node_type)
    except Exception as e:
        logger.error("Node %s initialize() failed: %s", node_id, e, exc_info=True)
        return

    # FPS limiter state
    fps_limit  = limits.max_fps if limits else None
    min_period = (1.0 / fps_limit) if fps_limit else 0.0
    last_t     = 0.0

    window: list[float] = []
    t_last  = [time.monotonic()]

    logger.info("Node %s started (type=%s)", node_id, node_type)

    try:
        while not shutdown_event.is_set():
            ctx = FrameContext.from_buses(input_buses, timeout_ms=30)
            if ctx is None:
                continue

            # Rate limiting
            if fps_limit:
                now = time.monotonic()
                elapsed = now - last_t
                if elapsed < min_period:
                    time.sleep(min_period - elapsed)
                last_t = time.monotonic()

            t0 = time.perf_counter()
            try:
                result = node.process(ctx)
            except StopIteration:
                logger.info("Node %s raised StopIteration — shutting down", node_id)
                break
            except Exception as e:
                logger.error("Node %s process() error: %s", node_id, e, exc_info=True)
                _inc_error_shm(stats_shm)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            _update_stats_shm(stats_shm, elapsed_ms, window, t_last)

            if result is None:
                continue  # node dropped this frame

            result.to_buses(output_buses)

    finally:
        node.teardown()
        for bus in input_buses.values():
            bus.close(unlink=False)
        for bus in output_buses.values():
            bus.close(unlink=False)
        stats_shm.close()
        logger.info("Node %s stopped", node_id)


def _cpp_node_worker_main(
    so_path: str,
    node_id: str,
    config_json: str,
    input_bus_specs: list[tuple[str, str, int, int, int]],
    output_bus_specs: list[tuple[str, str, int, int, int]],
    stats_shm_name: str,
    shutdown_event: threading.Event,
    limits: Optional[ResourceLimits],
) -> None:
    """Entry point for a C++ node worker (runs in a thread)."""
    _apply_resource_limits(limits)

    # Build subprocess command
    # The .so must export main() that parses --input-bus / --output-bus / --config
    # OR we use a generic runner that loads the .so via dlopen
    cmd = [so_path]
    cmd += ["--config", config_json]
    for _, shm_name, w, h, c in input_bus_specs:
        cmd += ["--input-bus", f"{shm_name}:{w}:{h}:{c}"]
    for _, shm_name, w, h, c in output_bus_specs:
        cmd += ["--output-bus", f"{shm_name}:{w}:{h}:{c}"]
    cmd += ["--stats-shm", stats_shm_name]

    logger.info("Spawning C++ worker: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Relay stdout/stderr to our logger
        def _log_output(stream, level):
            for line in iter(stream.readline, b""):
                logger.log(level, "[%s] %s", node_id, line.decode(errors="replace").rstrip())
            stream.close()

        threading.Thread(target=_log_output, args=(proc.stdout, logging.INFO),  daemon=True).start()
        threading.Thread(target=_log_output, args=(proc.stderr, logging.WARNING), daemon=True).start()

        # Wait for shutdown signal
        while not shutdown_event.is_set():
            if proc.poll() is not None:
                logger.warning("C++ worker %s exited early (code=%d)", node_id, proc.returncode)
                break
            time.sleep(0.1)

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception as e:
        logger.error("C++ worker %s failed: %s", node_id, e)


# ── NodeWorker ────────────────────────────────────────────────────────────────

class NodeWorker:
    """
    Wraps one node instance in an OS process (Python) or thread (C++),
    connecting it to input and output PortBuses.

    Parameters
    ----------
    node_class:      Python class or path to compiled .so for C++ nodes.
    node_id:         Unique string identifier (matches pipeline JSON).
    node_type:       Type string (e.g. "model_inference").
    config:          Node config dict from pipeline JSON.
    input_buses:     {port_name → PortBus}  (already created by runner).
    output_buses:    {port_name → PortBus}  (already created by runner).
    resource_limits: CPU/GPU/memory constraints.
    worker_index:    0 for primary, 1+ for auto-scaled replicas.
    use_thread:      True for C++ nodes (they release GIL via ctypes).
    """

    def __init__(
        self,
        node_class,             # type: Type | str   (class or .so path)
        node_id: str,
        node_type: str,
        config: dict,
        input_buses:   dict[str, PortBus | RoundRobinBus | MergeBus],
        output_buses:  dict[str, PortBus | RoundRobinBus | MergeBus],
        resource_limits: Optional[ResourceLimits] = None,
        worker_index: int = 0,
        use_thread: bool = False,
    ) -> None:
        self.node_id      = node_id
        self.node_type    = node_type
        self.worker_index = worker_index

        self._node_class     = node_class
        self._config         = config
        self._input_buses    = input_buses
        self._output_buses   = output_buses
        self._limits         = resource_limits
        self._use_thread     = use_thread

        self._process: Optional[multiprocessing.Process] = None
        self._thread:  Optional[threading.Thread]        = None
        self._shutdown = multiprocessing.Event() if not use_thread else threading.Event()

        self._stats_shm = _make_stats_shm()
        self._stats_shm_name = self._stats_shm.name

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bus_specs(self, buses: dict) -> list[tuple]:
        specs = []
        for port_name, bus in buses.items():
            if isinstance(bus, PortBus):
                specs.append((port_name, bus.name, bus.max_w, bus.max_h, bus.max_c))
            # RoundRobinBus / MergeBus: pick the worker-indexed sub-bus
            elif isinstance(bus, (RoundRobinBus, MergeBus)):
                idx = min(self.worker_index, len(bus.buses) - 1)
                b   = bus.buses[idx]
                specs.append((port_name, b.name, b.max_w, b.max_h, b.max_c))
        return specs

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        in_specs  = self._bus_specs(self._input_buses)
        out_specs = self._bus_specs(self._output_buses)

        if self._use_thread or isinstance(self._node_class, str):
            # C++ node → thread
            self._thread = threading.Thread(
                target=_cpp_node_worker_main,
                args=(
                    self._node_class,
                    self.node_id,
                    __import__("json").dumps(self._config),
                    in_specs,
                    out_specs,
                    self._stats_shm_name,
                    self._shutdown,
                    self._limits,
                ),
                daemon=True,
                name=f"cpp-{self.node_id}-{self.worker_index}",
            )
            self._thread.start()
        else:
            # Python node → process
            module_name = self._node_class.__module__
            class_name  = self._node_class.__name__
            self._process = multiprocessing.Process(
                target=_python_node_worker_main,
                args=(
                    module_name,
                    class_name,
                    self.node_id,
                    self.node_type,
                    self._config,
                    in_specs,
                    out_specs,
                    self._stats_shm_name,
                    self._shutdown,
                    self._limits,
                ),
                daemon=True,
                name=f"py-{self.node_id}-{self.worker_index}",
            )
            self._process.start()
        logger.info("Started worker %s[%d] pid=%s",
                    self.node_id, self.worker_index,
                    self._process.pid if self._process else "thread")

    def stop(self, timeout: float = 5.0) -> None:
        self._shutdown.set()

        if self._process and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.0)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._stats_shm.close()
        try:
            self._stats_shm.unlink()
        except Exception:
            pass

    def is_alive(self) -> bool:
        if self._process:
            return self._process.is_alive()
        if self._thread:
            return self._thread.is_alive()
        return False

    def get_stats(self) -> NodeStats:
        return _read_stats(self._stats_shm)

    def __repr__(self) -> str:
        return f"NodeWorker({self.node_id!r}[{self.worker_index}] alive={self.is_alive()})"
