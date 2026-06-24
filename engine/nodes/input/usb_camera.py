from engine.nodes.input.camera import CameraNode
from engine.core.node_registry import register


@register("usb_camera")
class USBCameraNode(CameraNode):
    def initialize(self):
        self.config.setdefault("source_type", "usb")
        super().initialize()
