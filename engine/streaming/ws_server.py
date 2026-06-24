"""
Async WebSocket server running in a background thread.
Provides two channels per session:
  /ws/stream/{session_id}  — JPEG frames
  /ws/events/{session_id}  — JSON events (counters, logs, status)
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

# Queues keyed by session_id
_stream_queues: dict[str, asyncio.Queue] = {}
_event_queues:  dict[str, asyncio.Queue] = {}
_loop: asyncio.AbstractEventLoop | None = None


def _get_or_create_queue(store: dict, key: str) -> asyncio.Queue:
    if key not in store:
        store[key] = asyncio.Queue(maxsize=5)
    return store[key]


async def _handler(ws: WebSocketServerProtocol, path: str):
    parts = path.strip("/").split("/")
    # Expected: ws/stream/{session_id} or ws/events/{session_id}
    if len(parts) < 3 or parts[0] != "ws":
        await ws.close(1008, "Invalid path")
        return

    channel, session_id = parts[1], parts[2]

    if channel == "stream":
        queue = _get_or_create_queue(_stream_queues, session_id)
        try:
            while True:
                data = await queue.get()
                if data is None:
                    break
                await ws.send(data)   # raw bytes (JPEG)
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
                # Handle client messages
                try:
                    client_msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                    # Pass to registered handler (future: live param update)
                except asyncio.TimeoutError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
    else:
        await ws.close(1008, "Unknown channel")


def _run_server(host: str, port: int):
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def main():
        async with websockets.serve(_handler, host, port):
            logger.info("WebSocket server listening on ws://%s:%d", host, port)
            await asyncio.Future()   # run forever

    _loop.run_until_complete(main())


def start_server(host: str = "0.0.0.0", port: int = 8765):
    t = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    t.start()


def send_frame(session_id: str, jpeg_bytes: bytes):
    """Called from pipeline thread. Non-blocking — drops frame if queue full."""
    queue = _stream_queues.get(session_id)
    if queue and _loop:
        try:
            _loop.call_soon_threadsafe(queue.put_nowait, jpeg_bytes)
        except asyncio.QueueFull:
            pass   # Drop frame — viewer can't keep up


def send_event(session_id: str, event: dict):
    """Called from pipeline thread. Non-blocking."""
    queue = _event_queues.get(session_id)
    if queue and _loop:
        try:
            _loop.call_soon_threadsafe(queue.put_nowait, event)
        except asyncio.QueueFull:
            pass
