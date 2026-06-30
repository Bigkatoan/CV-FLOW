"""cv_flow.nodes — Built-in pipeline nodes."""
from cv_flow.nodes._catalog import NODE_CATALOG, validate_catalog
from cv_flow.nodes.camera import CameraSource, RtspSource, VideoFileSource, build_nvargus_pipeline
from cv_flow.nodes.preprocess import Preprocess, GrayscaleConvert
from cv_flow.nodes.inference import YoloInference, OnnxInference
from cv_flow.nodes.postprocess import NMS
from cv_flow.nodes.tracking import ObjectTracker
from cv_flow.nodes.tee import Tee
from cv_flow.nodes.draw import DrawBbox
from cv_flow.nodes.output import StreamViewer, VideoWriter, MqttPublisher

__all__ = [
    "NODE_CATALOG", "validate_catalog",
    "CameraSource", "RtspSource", "VideoFileSource", "build_nvargus_pipeline",
    "Preprocess", "GrayscaleConvert",
    "YoloInference", "OnnxInference",
    "NMS",
    "ObjectTracker",
    "Tee",
    "DrawBbox",
    "StreamViewer", "VideoWriter", "MqttPublisher",
]
