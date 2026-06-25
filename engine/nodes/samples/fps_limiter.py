"""
FPS Limiter Node — SDK sample (function-based decorator style).

Demonstrates:
  - @cv_node_fn decorator with keyword-only config params
  - Frame drop via return None
  - Per-node state using a closure variable
  - Testing with NodeTestHarness

Usage in pipeline JSON:
  { "type": "fps_limiter", "config": { "target_fps": 10.0 } }

Test:
  python -c "
  from engine.nodes.samples.fps_limiter import FpsLimiterNode
  from engine.nodes.sdk import NodeTestHarness
  import numpy as np, time

  with NodeTestHarness(FpsLimiterNode, config={'target_fps': 5.0}) as h:
      frames = [np.zeros((480, 640, 3), np.uint8)] * 10
      results = h.run_frames(frames)
      passed = sum(r is not None for r in results)
      print(f'Passed {passed}/10 frames at 5 FPS limit (instantaneous — expect 1)')
  "
"""
import time

from engine.nodes.sdk import cv_node_fn

_state: dict[str, float] = {}  # node_id → last_pass_timestamp


@cv_node_fn(
    label="FPS Limiter",
    group="utility",
    icon="timer",
    inputs=["frame"],
    outputs=["frame"],
    default_config={"target_fps": 15.0},
)
def fps_limiter(frame, *, _node_id: str = "", target_fps: float = 15.0):
    """
    Limits the downstream frame rate to target_fps.
    Dropped frames are silently discarded — the pipeline continues at the
    next frame from the source.  Useful for reducing CPU/GPU load on
    inference nodes when the camera runs faster than the model.
    """
    now      = time.monotonic()
    interval = 1.0 / max(0.1, target_fps)
    last_t   = _state.get(_node_id, 0.0)
    if now - last_t < interval:
        return None  # DROP — too soon
    _state[_node_id] = now
    return frame


# Expose the generated class with a clean name
FpsLimiterNode = fps_limiter
