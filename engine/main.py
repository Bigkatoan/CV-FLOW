#!/usr/bin/env python3
"""
CV-FLOW Engine Entry Point
==========================
Usage:
  python engine/main.py \\
    --pipeline-json /path/to/pipeline.json \\
    --session-id <uuid> \\
    --ws-port 8765 \\
    [--params-override '{"conf_threshold": 0.7}']
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so both `engine` and `app` packages resolve
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core.pipeline_builder import build_pipeline
from engine.core.pipeline_runner  import PipelineRunner
from engine.streaming.ws_server   import start_server
from engine.model_hub.hot_reload  import install as install_hot_reload


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cv_flow.engine")


def main():
    parser = argparse.ArgumentParser(description="CV-FLOW pipeline execution engine")
    parser.add_argument("--pipeline-json", required=True)
    parser.add_argument("--session-id",    required=True)
    parser.add_argument("--ws-port",       type=int, default=8765)
    parser.add_argument("--params-override", default="{}")
    parser.add_argument(
        "--mode",
        choices=["sequential", "multiprocess"],
        default="sequential",
        help="sequential: single-process loop (default, backward-compatible). "
             "multiprocess: async DAG — each node runs in its own process/thread.",
    )
    args = parser.parse_args()

    pipeline_path = Path(args.pipeline_json)
    if not pipeline_path.exists():
        logger.error("Pipeline JSON not found: %s", pipeline_path)
        sys.exit(1)

    pipeline_json = json.loads(pipeline_path.read_text())

    # Apply params override
    params_override = json.loads(args.params_override)
    if params_override:
        for node in pipeline_json.get("nodes", []):
            node.setdefault("config", {}).update(
                {k: v for k, v in params_override.items() if k in node["config"]}
            )

    # Start WebSocket server in background thread
    start_server(port=args.ws_port)
    logger.info("Session: %s | WS port: %d", args.session_id, args.ws_port)

    if args.mode == "multiprocess":
        from engine.core.pipeline_runner_mp import MultiProcessPipelineRunner
        runner = MultiProcessPipelineRunner(pipeline_json, session_id=args.session_id)
        runner.start()
        # Block until stop requested (Ctrl+C or signal)
        import signal, threading
        stop_evt = threading.Event()
        def _sig(sig, frame):
            runner.request_stop()
            stop_evt.set()
        signal.signal(signal.SIGINT,  _sig)
        signal.signal(signal.SIGTERM, _sig)
        stop_evt.wait()
        runner.stop()
    else:
        # Build and run pipeline (sequential — default, backward-compatible)
        nodes = build_pipeline(pipeline_json)
        runner = PipelineRunner(nodes, session_id=args.session_id)
        # Install hot-reload signal handler (SIGUSR1 / polling)
        install_hot_reload(runner)
        runner.run()

    logger.info("Engine exiting cleanly")


if __name__ == "__main__":
    main()
