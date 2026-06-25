"""
Async WebSocket server running in a background thread.

Channels per session:
  /ws/stream/{session_id}              — JPEG frames (pipeline output)
  /ws/events/{session_id}              — JSON events (counters, logs, status)
  /ws/node-preview/{session_id}/{nid}  — per-node JPEG frames (for live debugging)
"""
from __future__ import annotations
import asyncio
import json
import logging
import threading
from typing import Callable

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

# ── Per-session stream / event queues ─────────────────────────────────────────
_stream_queues: dict[str, asyncio.Queue] = {}
_event_queues:  dict[str, asyncio.Queue] = {}

# ── Per-node preview queues  key: "{session_id}:{node_id}" ────────────────────
_node_preview_queues: dict[str, asyncio.Queue] = {}
_watched_nodes: dict[str, set[str]] = {}  # session_id → set of node_ids being watched

_loop: asyncio.AbstractEventLoop | None = None
_ready = threading.Event()  # set when server is actually listening


def _make_queue(maxsize: int = 5) -> asyncio.Queue:
    return asyncio.Queue(maxsize=maxsize)


def _get_or_create_queue(store: dict, key: str, maxsize: int = 5) -> asyncio.Queue:
    if key not in store:
        store[key] = _make_queue(maxsize)
    return store[key]


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def _handler(ws: WebSocketServerProtocol):
    # websockets v10+ exposes the path on the connection object
    path = getattr(ws, "path", None) or getattr(getattr(ws, "request", None), "path", "/")
    parts = path.strip("/").split("/")
    # Expected paths:
    #   ws/stream/{session_id}
    #   ws/events/{session_id}
    #   ws/node-preview/{session_id}/{node_id}
    if len(parts) < 3 or parts[0] != "ws":
        await ws.close(1008, "Invalid path")
        return

    channel    = parts[1]
    session_id = parts[2]

    if channel == "stream":
        queue = _get_or_create_queue(_stream_queues, session_id)
        try:
            while True:
                data = await queue.get()
                if data is None:
                    break
                await ws.send(data)   # raw JPEG bytes
        except websockets.exceptions.ConnectionClosed:
            pass

    elif channel == "events":
        queue = _get_or_create_queue(_event_queues, session_id)
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                await ws.send(json.dumps(msg))
                # Check for client→server messages without blocking
                try:
                    _ = await asyncio.wait_for(ws.recv(), timeout=0.001)
                except asyncio.TimeoutError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass

    elif channel == "node-preview" and len(parts) >= 4:
        node_id = parts[3]
        key     = f"{session_id}:{node_id}"

        # Register as watcher
        _watched_nodes.setdefault(session_id, set()).add(node_id)
        queue = _get_or_create_queue(_node_preview_queues, key)
        logger.debug("Node preview client connected: %s/%s", session_id, node_id)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Ping to detect stale connections
                    await ws.ping()
                    continue
                if data is None:
                    break
                await ws.send(data)   # raw JPEG bytes
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as exc:
            logger.debug("Node preview handler error (%s/%s): %s", session_id, node_id, exc)
        finally:
            _watched_nodes.get(session_id, set()).discard(node_id)
            _node_preview_queues.pop(key, None)
            logger.debug("Node preview client disconnected: %s/%s", session_id, node_id)

    else:
        await ws.close(1008, "Unknown channel")


# ── Server startup ────────────────────────────────────────────────────────────

def _run_server(host: str, port: int):
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def main():
        async with websockets.serve(_handler, host, port):
            _ready.set()
            logger.info("WebSocket server listening on ws://%s:%d", host, port)
            await asyncio.Future()   # run forever

    _loop.run_until_complete(main())


def start_server(host: str = "0.0.0.0", port: int = 8765):
    _ready.clear()
    t = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    t.start()
    # Wait until the server is actually accepting connections before returning
    if not _ready.wait(timeout=5.0):
        logger.warning("WebSocket server did not start within 5s")


# ── Send helpers ──────────────────────────────────────────────────────────────

def _put_nowait_safe(queue: asyncio.Queue, item) -> None:
    """Thread-safe, non-blocking enqueue. Drops item silently if queue is full."""
    if not _loop:
        return

    def _do_put():
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # frame dropped — consumer is not keeping up

    _loop.call_soon_threadsafe(_do_put)


def send_frame(session_id: str, jpeg_bytes: bytes) -> None:
    """Push a JPEG frame to the pipeline output stream. Called from pipeline thread.
    Pre-creates the queue so frames buffer even before a WS client connects."""
    if not _loop:
        return
    # Create queue eagerly so frames don't drop before frontend connects
    if session_id not in _stream_queues:
        _stream_queues[session_id] = _make_queue()
    _put_nowait_safe(_stream_queues[session_id], jpeg_bytes)


def send_event(session_id: str, event: dict) -> None:
    """Push a JSON event to the events channel. Called from pipeline thread."""
    if not _loop:
        return
    if session_id not in _event_queues:
        _event_queues[session_id] = _make_queue(maxsize=20)
    _put_nowait_safe(_event_queues[session_id], event)


def is_node_watched(session_id: str, node_id: str) -> bool:
    """Return True if a frontend client is watching this node's preview."""
    return node_id in _watched_nodes.get(session_id, set())


def send_node_preview(session_id: str, node_id: str, jpeg_bytes: bytes) -> None:
    """Push a JPEG preview frame for a specific node. Called from pipeline thread."""
    key = f"{session_id}:{node_id}"
    q   = _node_preview_queues.get(key)
    if q:
        _put_nowait_safe(q, jpeg_bytes)


def cleanup_session(session_id: str) -> None:
    """
    Signal end-of-session to all connected WebSocket clients by sending
    a None sentinel.  Call this from the pipeline runner's finally block.
    """
    # Main stream
    sq = _stream_queues.get(session_id)
    if sq:
        _put_nowait_safe(sq, None)

    # Events
    eq = _event_queues.get(session_id)
    if eq:
        _put_nowait_safe(eq, None)

    # Node previews
    for key, q in list(_node_preview_queues.items()):
        if key.startswith(f"{session_id}:"):
            _put_nowait_safe(q, None)

    _watched_nodes.pop(session_id, None)
