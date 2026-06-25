import logging
import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("preprocess")
class PreprocessNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        frame = ctx.frame
        if frame is None:
            return ctx

        fh, fw = frame.shape[:2]

        # Crop
        if crop := self.config.get("crop"):
            x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
            frame = frame[y:y+h, x:x+w]

        # Resize — support flat keys (resize_w / resize_h) AND legacy nested dict
        rw = int(self.config.get("resize_w") or 0)
        rh = int(self.config.get("resize_h") or 0)
        if rw <= 0 or rh <= 0:
            _rd = self.config.get("resize") or {}
            rw = int(_rd.get("width") or 0)
            rh = int(_rd.get("height") or 0)

        if rw > 0 and rh > 0:
            keep_aspect = (self.config.get("resize") or {}).get("keep_aspect", False)
            if keep_aspect:
                scale = min(rw / fw, rh / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            else:
                frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
            logger.debug("Preprocess: resized %dx%d → %dx%d", fw, fh, rw, rh)

        # Normalize — produces a float32 CHW tensor stored in metadata
        # frame stays uint8 so downstream draw nodes can still use it
        normalize = self.config.get("normalize", "none")
        if normalize != "none":
            fh2, fw2 = frame.shape[:2]
            f32 = frame.astype(np.float32)
            if normalize == "0_1":
                f32 /= 255.0
            elif normalize == "imagenet":
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                f32 = cv2.cvtColor(f32, cv2.COLOR_BGR2RGB) / 255.0
                f32 = (f32 - mean) / std
            elif normalize == "custom":
                mean = np.array(self.config.get("mean", [0, 0, 0]), dtype=np.float32)
                std  = np.array(self.config.get("std",  [1, 1, 1]), dtype=np.float32)
                f32 = (f32 - mean) / std
            tensor = np.ascontiguousarray(f32.transpose(2, 0, 1)[np.newaxis])  # [1,C,H,W]
            ctx.metadata["preprocessed_tensor"] = tensor
            logger.debug("Preprocess: tensor shape %s, normalize=%s", tensor.shape, normalize)

        ctx.frame = frame
        return ctx
