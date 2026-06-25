import json
import logging
import os
import time
import numpy as np
from pathlib import Path
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


def _models_dir() -> Path:
    """Resolve models directory — use env var set by backend, fall back to relative path."""
    env = os.environ.get("CVFLOW_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent.parent / "backend" / "storage" / "models"


try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


@register("model_inference")
class ModelInferenceNode(BaseNode):
    _session = None
    _model_config: dict = {}

    def initialize(self):
        if not _ORT_AVAILABLE:
            logger.warning(
                "onnxruntime not installed — ModelInferenceNode will be a no-op. "
                "Install with: pip install onnxruntime"
            )
            return
        self._load_model()

    def _load_model(self):
        model_id = self.config.get("model_id", "")
        if not model_id:
            logger.warning("[ModelInference] No model_id set — node will be a no-op")
            return
        model_dir  = _models_dir() / model_id
        onnx_path  = model_dir / "model.onnx"
        config_path = model_dir / "config.json"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"[ModelInference] ONNX file not found: {onnx_path}\n"
                f"  Models dir: {_models_dir()}\n"
                f"  Model ID: {model_id!r}"
            )
        self._model_config = json.loads(config_path.read_text()) if config_path.exists() else {}

        input_shape = self._model_config.get("input_shape", [1, 3, 640, 640])
        logger.info(
            "[ModelInference] Loading model %r — input %s, task=%s",
            model_id, input_shape, self._model_config.get("task", "unknown"),
        )

        device = self.config.get("device", "cpu")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)

        self._input_name    = self._model_config.get("input_name", "images")
        self._conf_threshold = self.config.get("conf_threshold", 0.5)

        # Log actual I/O shapes reported by the runtime
        for inp in self._session.get_inputs():
            logger.info("[ModelInference] Input  '%s': shape=%s dtype=%s", inp.name, inp.shape, inp.type)
        for out in self._session.get_outputs():
            logger.info("[ModelInference] Output '%s': shape=%s dtype=%s", out.name, out.shape, out.type)

    def reload_model(self):
        self._load_model()

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._session is None:
            return ctx

        import cv2
        input_shape = self._model_config.get("input_shape", [1, 3, 640, 640])
        expected_h, expected_w = input_shape[-2], input_shape[-1]

        tensor = ctx.metadata.get("preprocessed_tensor")

        if tensor is None:
            # No preprocess node — build tensor from raw frame
            resized = cv2.resize(ctx.frame, (expected_w, expected_h))
            tensor  = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
            logger.debug(
                "[ModelInference] No preprocessed_tensor — auto-resized frame "
                "from %s to (%d,%d)", ctx.frame.shape[:2], expected_h, expected_w,
            )
        elif tensor.shape[-2] != expected_h or tensor.shape[-1] != expected_w:
            # Preprocess produced wrong dimensions — auto-correct
            logger.warning(
                "[ModelInference] Tensor shape %s does not match model input %s. "
                "Check that Preprocess → resize_w=%d, resize_h=%d. Auto-correcting.",
                tensor.shape, input_shape, expected_w, expected_h,
            )
            resized = cv2.resize(ctx.frame, (expected_w, expected_h))
            tensor  = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        ms = (time.perf_counter() - t0) * 1000
        logger.debug("[ModelInference] Inference %.1f ms — outputs: %s",
                     ms, [o.shape for o in outputs])

        ctx.metadata["model_output"]          = outputs
        ctx.metadata["model_config"]          = self._model_config
        ctx.metadata["model_conf_threshold"]  = self._conf_threshold
        return ctx
