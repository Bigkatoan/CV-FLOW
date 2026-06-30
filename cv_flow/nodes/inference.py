"""
cv_flow.nodes.inference — YoloInference, OnnxInference: ONNX Runtime model nodes.
"""
from __future__ import annotations

import logging

from cv_flow.node import Node

logger = logging.getLogger("cv_flow.nodes.inference")


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
        trt_cache_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic
        self.model_path  = model_path
        self.input_name  = input_name
        self.output_name = output_name
        self.device      = device
        self.trt_cache_dir = trt_cache_dir
        self._session = None

    def initialize(self) -> None:
        import onnxruntime as ort

        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

        if self.device.startswith("cuda"):
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            trt_options: dict = {}
            if self.trt_cache_dir:
                import os
                os.makedirs(self.trt_cache_dir, exist_ok=True)
                trt_options = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": self.trt_cache_dir,
                }
            provider_options = [trt_options, {}, {}]
            self._session = ort.InferenceSession(
                self.model_path, providers=providers, provider_options=provider_options,
            )
        else:
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.model_path, providers=providers)

        active = self._session.get_providers()
        logger.info("%s: requested providers=%s, active providers=%s",
                    self.name, providers, active)
        if self.device.startswith("cuda") and active and active[0] == "CPUExecutionProvider":
            logger.warning(
                "%s: requested device=%r but ONNX Runtime fell back to CPUExecutionProvider "
                "— GPU acceleration is NOT active for this session.",
                self.name, self.device,
            )

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
        trt_cache_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            input_topic, output_topic,
            model_path=model_path,
            input_name="images",
            output_name="output0",
            device=device,
            trt_cache_dir=trt_cache_dir,
            name=name,
        )
