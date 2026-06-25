"""Hot-reload handler — triggers weight reload without stopping the pipeline.

On POSIX (Linux/macOS): uses SIGUSR1.
On Windows: SIGUSR1 does not exist, so the signal handler is skipped.
Hot-reload can still be triggered programmatically via runner.request_reload().
"""
import logging
import signal
import sys

logger = logging.getLogger(__name__)

_runner = None   # Set by engine/main.py after PipelineRunner is created


def install(runner) -> None:
    global _runner
    _runner = runner

    if sys.platform == "win32":
        # SIGUSR1 is a POSIX signal not available on Windows.
        logger.info("Hot-reload signal handler skipped (Windows — use API endpoint instead)")
        return

    def _handler(signum, frame):
        logger.info("SIGUSR1 received — requesting model hot-reload after current frame")
        if _runner:
            _runner.request_reload()

    signal.signal(signal.SIGUSR1, _handler)
    logger.info("Hot-reload signal handler installed (SIGUSR1)")
