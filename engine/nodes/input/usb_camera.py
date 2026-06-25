import cv2
from engine.nodes.input.camera import CameraNode
from engine.core.node_registry import register


@register("usb_camera")
class USBCameraNode(CameraNode):
    def initialize(self):
        self.config.setdefault("source_type", "usb")
        super().initialize()
        # Apply requested resolution (0 = let the driver decide)
        w = self.config.get("width", 0)
        h = self.config.get("height", 0)
        if w > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
