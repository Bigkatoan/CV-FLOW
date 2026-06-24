import json
import numpy as np
from pathlib import Path
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

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
            import logging
            logging.getLogger(__name__).warning(
                "onnxruntime not installed — ModelInferenceNode will be a no-op. "
                "Install with: pip install onnxruntime"
            )
            return
        self._load_model()

    def _load_model(self):
        from app.config import settings  # noqa — lazy import to avoid circular dep
        model_id = self.config.get("model_id", "")
        model_dir = Path(settings.models_dir) / model_id
        onnx_path = model_dir / "model.onnx"
        config_path = model_dir / "config.json"
        if not onnx_path.exists():
            raise FileNotFoundError(f"Model ONNX not found: {onnx_path}")
        self._model_config = json.loads(config_path.read_text())
        device = self.config.get("device", "cpu")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self._model_config.get("input_name", "images")
        self._conf_threshold = self.config.get("conf_threshold", 0.5)

    def reload_model(self):
        """Hot-reload: recreate ONNXRuntime session with new weights."""
        self._load_model()

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._session is None:
            return ctx

        tensor = ctx.metadata.get("preprocessed_tensor")
        if tensor is None:
            import cv2
            input_shape = self._model_config.get("input_shape", [1, 3, 640, 640])
            th, tw = input_shape[-2], input_shape[-1]
            resized = cv2.resize(ctx.frame, (tw, th))
            tensor = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        outputs = self._session.run(None, {self._input_name: tensor})
        ctx.metadata["model_output"] = outputs
        ctx.metadata["model_config"] = self._model_config
        ctx.metadata["model_conf_threshold"] = self._conf_threshold
        return ctx
