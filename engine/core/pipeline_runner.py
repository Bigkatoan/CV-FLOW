"""Main frame loop — calls each node's process() in topological order."""
from __future__ import annotations
import logging
import threading
import time

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, nodes: list[BaseNode], session_id: str):
        self.nodes = nodes
        self.session_id = session_id
        self._stop_event = threading.Event()
        self._reload_flag = threading.Event()
        self._frame_number = 0

    def request_stop(self):
        self._stop_event.set()

    def request_reload(self):
        """Called from SIGUSR1 handler; engine reloads model weights after current frame."""
        self._reload_flag.set()

    def run(self):
        """Blocking frame loop. Returns when stop is requested or source exhausted."""
        logger.info("Pipeline started with %d nodes", len(self.nodes))
        try:
            while not self._stop_event.is_set():
                # The first node (input) creates the FrameContext
                ctx = FrameContext(
                    frame=None,   # type: ignore  # input node fills this
                    frame_number=self._frame_number,
                    timestamp=time.time(),
                    session_id=self.session_id,
                )

                stop_pipeline = False
                for node in self.nodes:
                    try:
                        ctx = node.process(ctx)
                    except StopIteration:
                        # Input node signals end of source (e.g., end of video file)
                        stop_pipeline = True
                        break
                    except Exception as e:
                        logger.error("Node %s error on frame %d: %s",
                                     node.node_id, self._frame_number, e, exc_info=True)
                        # Continue to next frame; don't crash the whole pipeline

                if stop_pipeline:
                    logger.info("Input source exhausted — pipeline complete")
                    break

                self._frame_number += 1

                # Handle hot-reload after frame completes
                if self._reload_flag.is_set():
                    self._reload_flag.clear()
                    self._do_reload()

        finally:
            logger.info("Pipeline stopping, calling teardown on %d nodes", len(self.nodes))
            for node in reversed(self.nodes):
                try:
                    node.teardown()
                except Exception as e:
                    logger.warning("Teardown error for node %s: %s", node.node_id, e)

    def _do_reload(self):
        """Reload model weights in any ModelInference nodes without stopping the loop."""
        from engine.nodes.processing.model_inference import ModelInferenceNode
        for node in self.nodes:
            if isinstance(node, ModelInferenceNode):
                try:
                    node.reload_model()
                    logger.info("Hot-reloaded model in node %s", node.node_id)
                except Exception as e:
                    logger.error("Hot-reload failed for %s: %s", node.node_id, e)
