"""
cv_flow.nodes.inference — YoloInference, OnnxInference: ONNX Runtime model nodes.
"""
from __future__ import annotations

from cv_flow.node import Node


class OnnxInference(Node):
    """Generic ONNX model runner. Input/output shapes are model-defined."""

    def __init__(
        self,
        input_topic,
        output_topic,
        *,
        model_path: str,
        input_name: str = "images",
        output_name: str = "output0",
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic
        self.model_path  = model_path
        self.input_name  = input_name
        self.output_name = output_name
        self.device      = device
        self._session = None

    def initialize(self) -> None:
        import onnxruntime as ort

        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self.model_path, providers=providers)

    def spin_once(self) -> None:
        tensor = self._sub.read(timeout_ms=30)
        if tensor is None:
            return
        outputs = self._session.run([self.output_name], {self.input_name: tensor})
        self._pub.write(outputs[0])


class YoloInference(OnnxInference):
    """
    Runs a YOLO (v5/v8) ONNX model on a preprocessed CHW float32 tensor.
    Elastic-capable: deploy N replicas behind a RoundRobinBus/MergeBus pair.
    """

    def __init__(
        self,
        input_topic,
        output_topic,
        *,
        model_path: str,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        super().__init__(
            input_topic, output_topic,
            model_path=model_path,
            input_name="images",
            output_name="output0",
            device=device,
            name=name,
        )
