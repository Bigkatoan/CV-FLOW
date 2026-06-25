"""Downloads face analysis models from InsightFace model zoo for CV-FLOW.

Models land in ~/.insightface/models/{pack}/ via the insightface library,
then are registered into CV-FLOW's model registry.

Catalog keys used in node configs: scrfd_10g, scrfd_500m, mobilefacenet, arcface_r50
"""
from __future__ import annotations
import json
import logging
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping from CV-FLOW model key → insightface internal details
FACE_MODEL_CATALOG: dict[str, dict] = {
    "scrfd_10g": {
        "name":          "SCRFD-10G",
        "desc":          "High-accuracy face detector with 5-point landmarks. ~16 MB.",
        "category":      "Face Detection",
        "task":          "face_detect",
        "pack":          "buffalo_l",
        "onnx_name":     "det_10g.onnx",
        "size_mb":       16,
        "badge":         "Recommended",
        "input_shape":   [1, 3, 640, 640],
        "output_shapes": [[1, 12800, 15]],
        "class_names":   ["face"],
        "keypoint_names": ["leye", "reye", "nose", "lmouth", "rmouth"],
    },
    "scrfd_500m": {
        "name":          "SCRFD-500M",
        "desc":          "Lightweight face detector. CPU real-time. ~2 MB.",
        "category":      "Face Detection",
        "task":          "face_detect",
        "pack":          "buffalo_s",
        "onnx_name":     "det_500m.onnx",
        "size_mb":       2,
        "badge":         "Lite",
        "input_shape":   [1, 3, 640, 640],
        "output_shapes": [[1, 12800, 15]],
        "class_names":   ["face"],
        "keypoint_names": ["leye", "reye", "nose", "lmouth", "rmouth"],
    },
    "mobilefacenet": {
        "name":          "MobileFaceNet",
        "desc":          "Lightweight face embedding. CPU real-time. ~4 MB.",
        "category":      "Face Recognition",
        "task":          "face_embed",
        "pack":          "buffalo_s",
        "onnx_name":     "w600k_mbf.onnx",
        "size_mb":       4,
        "badge":         "Lite",
        "input_shape":   [1, 3, 112, 112],
        "output_shapes": [[1, 512]],
        "class_names":   [],
    },
    "arcface_r50": {
        "name":          "ArcFace-R50",
        "desc":          "High-accuracy face embedding (ResNet-50). GPU recommended. ~166 MB.",
        "category":      "Face Recognition",
        "task":          "face_embed",
        "pack":          "buffalo_l",
        "onnx_name":     "w600k_r50.onnx",
        "size_mb":       166,
        "badge":         "High Accuracy",
        "input_shape":   [1, 3, 112, 112],
        "output_shapes": [[1, 512]],
        "class_names":   [],
    },
}

# Convenience alias used by the API
FACE_MODELS = FACE_MODEL_CATALOG


def find_existing_face_model(model_key: str, models_dir: Path) -> str | None:
    """Return model_id if this face model is already registered in models_dir."""
    info = FACE_MODEL_CATALOG.get(model_key)
    if not info:
        return None
    for d in models_dir.iterdir():
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("face_model_key") == model_key:
                return cfg.get("model_id") or d.name
        except Exception:
            continue
    return None


def download_face_model(model_key: str, models_dir: Path) -> dict:
    """
    Download a face model via InsightFace model zoo and register it in CV-FLOW.

    Steps:
      1. Use insightface to download the model pack to ~/.insightface/models/{pack}/
      2. Copy the specific ONNX file to models_dir/{uuid}/model.onnx
      3. Write config.json

    Returns the config dict including model_id.
    Raises ImportError if insightface is not installed.
    """
    if model_key not in FACE_MODEL_CATALOG:
        raise ValueError(f"Unknown face model key: {model_key!r}. "
                         f"Available: {list(FACE_MODEL_CATALOG)}")

    # Check if already registered
    existing = find_existing_face_model(model_key, models_dir)
    if existing:
        logger.info("[FaceDownloader] %s already registered (model_id=%s)", model_key, existing)
        cfg_path = models_dir / existing / "config.json"
        return json.loads(cfg_path.read_text()) if cfg_path.exists() else {"model_id": existing}

    info = FACE_MODEL_CATALOG[model_key]
    pack = info["pack"]

    logger.info("[FaceDownloader] Downloading %s (pack: %s) — this may take a moment …",
                info["name"], pack)

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "insightface is required to download face models. "
            "Install with: pip install insightface"
        )

    # Trigger insightface download to ~/.insightface/models/{pack}/
    try:
        app = FaceAnalysis(name=pack, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        logger.warning("[FaceDownloader] FaceAnalysis.prepare warning: %s", e)

    # Locate the downloaded ONNX — scan pack dir first, fall back to catalog name
    home_models = Path.home() / ".insightface" / "models" / pack
    src_onnx = None

    if home_models.exists():
        # List all ONNX files actually present so we can match by name stem
        present = sorted(home_models.glob("*.onnx"))
        logger.info("[FaceDownloader] Files in %s: %s", home_models,
                    [f.name for f in present])

        # 1. Exact match on catalog onnx_name
        candidate = home_models / info["onnx_name"]
        if candidate.exists():
            src_onnx = candidate

        # 2. Fuzzy: for recognition models, pick the largest non-detector ONNX
        if src_onnx is None and info["task"] == "face_embed" and present:
            det_stems = {"det_10g", "det_500m", "2d106det", "genderage"}
            emb_files = [f for f in present if f.stem not in det_stems]
            if emb_files:
                src_onnx = max(emb_files, key=lambda f: f.stat().st_size)
                logger.info("[FaceDownloader] Using fuzzy match for embed model: %s", src_onnx.name)

        # 3. For detection models, pick by prefix
        if src_onnx is None and info["task"] == "face_detect" and present:
            prefix = "det_10g" if "10g" in info["onnx_name"] else "det_500m"
            det_files = [f for f in present if f.stem.startswith(prefix.split("_")[0])]
            if det_files:
                src_onnx = det_files[0]
                logger.info("[FaceDownloader] Using fuzzy match for detect model: %s", src_onnx.name)

    if src_onnx is None:
        # Also check model root (insightface < 0.7 layout)
        alt = Path.home() / ".insightface" / "models" / info["onnx_name"]
        if alt.exists():
            src_onnx = alt

    if src_onnx is None or not src_onnx.exists():
        present_names = [f.name for f in home_models.glob("*.onnx")] if home_models.exists() else []
        raise FileNotFoundError(
            f"Could not find ONNX for {model_key!r} in {home_models}. "
            f"Files present: {present_names}. "
            f"Run: python -c \"from insightface.app import FaceAnalysis; "
            f"FaceAnalysis(name='{pack}').prepare(ctx_id=-1)\""
        )

    # Copy into CV-FLOW models directory
    model_id  = str(uuid.uuid4())
    dest_dir  = models_dir / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_onnx), str(dest_dir / "model.onnx"))

    config = {
        "model_id":       model_id,
        "name":           info["name"],
        "version":        "1.0",
        "task":           info["task"],
        "format":         "onnx",
        "input_shape":    info["input_shape"],
        "output_shapes":  info["output_shapes"],
        "class_names":    info["class_names"],
        "face_model_key": model_key,
        "source":         f"insightface/{pack}/{info['onnx_name']}",
    }
    (dest_dir / "config.json").write_text(json.dumps(config, indent=2))

    logger.info("[FaceDownloader] %s registered → model_id=%s", info["name"], model_id)
    return config
