import pytest
import numpy as np
import tempfile
from pathlib import Path
import onnx
from onnx import helper, TensorProto
from engine.model_hub.onnx_inspector import inspect_onnx

@pytest.fixture
def dummy_onnx_path(tmp_path):
    # Create a dummy ONNX model
    X = helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 640, 640])
    Y = helper.make_tensor_value_info('det', TensorProto.FLOAT, [1, 25200, 85])
    
    node = helper.make_node('Identity', inputs=['image'], outputs=['det'])
    graph = helper.make_graph([node], 'test-model', [X], [Y])
    model = helper.make_model(graph, producer_name='pytest')
    
    path = tmp_path / "dummy.onnx"
    onnx.save(model, str(path))
    return path

def test_onnx_inspector(dummy_onnx_path):
    info = inspect_onnx(str(dummy_onnx_path))
    
    assert "inputs" in info
    assert "outputs" in info
    
    inputs = info["inputs"]
    assert len(inputs) == 1
    assert inputs[0]["name"] == "image"
    assert inputs[0]["shape"] == [1, 3, 640, 640]
    assert inputs[0]["dtype"] == "float32"
    
    outputs = info["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["name"] == "det"
    assert outputs[0]["shape"] == [1, 25200, 85]
    assert outputs[0]["dtype"] == "float32"
