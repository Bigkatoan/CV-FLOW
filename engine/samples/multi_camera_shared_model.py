"""
multi_camera_shared_model.py — Multi-camera orchestrator demo.

Runs 4 cameras simultaneously, all sharing a single YOLO model inference
worker pool.  Demonstrates how to use the Orchestrator API and the automatic
resource savings when cameras share model weights.

Usage:
    python engine/samples/multi_camera_shared_model.py \\
        --pipeline engine/samples/yolo_detection_usb.json \\
        --cameras  cam0:usb:0 cam1:rtsp:rtsp://... cam2:rtsp:rtsp://... \\
        --mode     multiprocess

Requirements:
    - At least one camera source available.
    - A model registered in the Model Hub (update MODEL_ID below).
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("multi_camera_demo")

# ── Model to use (update with your model ID) ──────────────────────────────────
DEFAULT_MODEL_ID = "REPLACE_WITH_MODEL_ID"

# ── Default pipeline (relative to repo root) ──────────────────────────────────
DEFAULT_PIPELINE = Path(__file__).parent / "yolo_detection_usb.json"


def load_pipeline(path: str, model_id: str) -> dict:
    with open(path) as f:
        pipeline = json.load(f)
    # Patch model_id into any inference node
    for node in pipeline.get("nodes", []):
        if node.get("type") == "model_inference":
            node["config"]["model_id"] = model_id
    return pipeline


def parse_camera_spec(spec: str) -> dict:
    """
    Parse 'cam_id:type:source' into a camera config dict.
    Examples:
        cam0:usb:0          → USB camera device_index=0
        cam1:rtsp:rtsp://…  → RTSP stream
        cam2:video:/path    → Video file
    """
    parts = spec.split(":", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid camera spec: {spec!r}. Expected cam_id:type:source")
    cam_id, cam_type, source = parts

    if cam_type == "usb":
        return {"id": cam_id, "type": "usb_camera", "device_index": int(source)}
    elif cam_type == "rtsp":
        return {"id": cam_id, "type": "rtsp_stream", "url": source}
    elif cam_type == "video":
        return {"id": cam_id, "type": "video_file", "path": source}
    else:
        raise ValueError(f"Unknown camera type: {cam_type!r}")


def run_demo(
    pipeline_path: str,
    camera_specs: list[str],
    model_id: str,
    mode: str,
) -> None:
    from engine.core.orchestrator import Orchestrator

    pipeline = load_pipeline(pipeline_path, model_id)
    cameras  = [parse_camera_spec(s) for s in camera_specs]

    logger.info("Starting %d cameras with pipeline: %s", len(cameras), pipeline["name"])
    logger.info("Mode: %s", mode)

    orchestrator = Orchestrator()
    group = orchestrator.create_group(
        group_id="demo_group",
        pipeline_json=pipeline,
        cameras=cameras,
        mode=mode,
    )

    logger.info("Camera group started.  Press Ctrl+C to stop.")

    stop_requested = False

    def _handle_signal(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_requested:
            time.sleep(5.0)
            stats = group.get_stats()
            logger.info(
                "Stats: %d cameras active | %s",
                stats.get("active_cameras", 0),
                json.dumps(stats.get("node_stats", {}), indent=None),
            )
    finally:
        logger.info("Stopping orchestrator...")
        orchestrator.delete_group("demo_group")
        logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV-FLOW multi-camera shared model demo"
    )
    parser.add_argument(
        "--pipeline", default=str(DEFAULT_PIPELINE),
        help="Path to pipeline JSON (default: yolo_detection_usb.json)",
    )
    parser.add_argument(
        "--cameras", nargs="+",
        default=["cam0:usb:0"],
        help="Camera specs: cam_id:type:source  (type: usb|rtsp|video)",
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID,
        help="Model ID from the Model Hub",
    )
    parser.add_argument(
        "--mode", choices=["sequential", "multiprocess"], default="multiprocess",
        help="Runner mode (multiprocess recommended for multiple cameras)",
    )
    args = parser.parse_args()

    run_demo(args.pipeline, args.cameras, args.model_id, args.mode)


if __name__ == "__main__":
    main()
