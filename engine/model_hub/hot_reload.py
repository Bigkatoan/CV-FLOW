"""SIGUSR1 handler for hot-reload of model weights without stopping the pipeline."""
import logging
import signal

logger = logging.getLogger(__name__)

_runner = None   # Set by engine/main.py after PipelineRunner is created


def install(runner) -> None:
    global _runner
    _runner = runner

    def _handler(signum, frame):
        logger.info("SIGUSR1 received — requesting model hot-reload after current frame")
        if _runner:
            _runner.request_reload()

    signal.signal(signal.SIGUSR1, _handler)
    logger.info("Hot-reload signal handler installed (SIGUSR1)")
