"""
ONNX model introspection — extract I/O metadata and auto-guess semantic port types.

Handles:
- Dynamic shapes (dim_param → -1)
- INT8 / UINT8 quantized models → warns about preprocessing
- Multiple inputs with initializer filtering (weights ≠ runtime inputs)
- Legacy opset < 11 → flagged in meta.warning
- Multiple outputs with per-shape heuristics
"""
from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto


# ── dtype map ─────────────────────────────────────────────────────────────────

_ONNX_DTYPE_MAP: dict[int, str] = {
    TensorProto.FLOAT:   "float32",
    TensorProto.DOUBLE:  "float64",
    TensorProto.INT32:   "int32",
    TensorProto.INT64:   "int64",
    TensorProto.UINT8:   "uint8",
    TensorProto.INT8:    "int8",
    TensorProto.BOOL:    "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.STRING:  "string",
}


# ── shape extraction ──────────────────────────────────────────────────────────

def _extract_shape(tensor_type_shape) -> list[int]:
    """Extract shape as list[int], using -1 for dynamic / symbolic dims."""
    if tensor_type_shape is None:
        return []
    result: list[int] = []
    for d in tensor_type_shape.dim:
        if d.HasField("dim_param"):      # symbolic name, e.g. "batch_size"
            result.append(-1)
        elif d.HasField("dim_value"):
            result.append(d.dim_value)
        else:
            result.append(-1)
    return result


def _onnx_dtype_to_str(elem_type: int) -> str:
    return _ONNX_DTYPE_MAP.get(elem_type, f"unknown({elem_type})")


def _friendly_name(tensor_name: str, fallback: str) -> str:
    """Return a clean port name from a tensor name.

    Strips leading slashes (ONNX graph node names) and trims whitespace.
    Falls back to the provided fallback if the result would be empty.
    """
    n = tensor_name.strip().lstrip("/")
    return n if n else fallback


# ── type heuristics ───────────────────────────────────────────────────────────

# Typical embedding dimensions (exact set to distinguish from class_scores ≤ 1000)
_EMBEDDING_DIMS = {64, 128, 192, 256, 320, 384, 512, 768, 1024, 1280, 1536, 2048, 4096}


def _guess_input_type(name: str, shape: list[int], dtype: str) -> str:
    """Heuristic: guess semantic type for an input tensor."""
    positive = [d for d in shape if d > 0]
    if dtype in ("float32", "float16", "float64") and len(shape) == 4:
        return "image"
    if len(positive) == 1 and positive[0] in _EMBEDDING_DIMS:
        return "embeddings"
    return "tensor"


def _guess_output_type(name: str, shape: list[int], dtype: str, total_outputs: int) -> str:
    """
    Heuristic: guess semantic type for an output tensor.

    Precedence (checked in order — more specific wins):
    1. 4-D → mask  (segmentation proto / feature map)
    2. 3-D with specific last-dim patterns → detections / keypoints / landmarks
    3. 1-D or 2-D (squeeze-safe):
       a. dim in _EMBEDDING_DIMS and > 1000 → embeddings
       b. dim in _EMBEDDING_DIMS → embeddings (MobileFaceNet [1,128] still embeddings)
       c. dim ≤ 1000 → class_scores (ImageNet etc.)
       d. dim > 1000 but not in set → embeddings
    4. Name-based fallback
    5. tensor (unknown)

    Edge cases documented:
    - ResNet-50 penultimate output [1, 2048]: 2048 ∈ _EMBEDDING_DIMS → embeddings ✓
    - MobileFaceNet output [1, 128]:          128 ∈ _EMBEDDING_DIMS → embeddings ✓
    - ResNet-50 final softmax [1, 1000]:       1000 ≤ 1000 → class_scores ✓
    - YOLO pose output [1, 56, 8400]:          len=3, last=8400 not in det set →
      mid=56 matches det heuristic? No → falls through to tensor. Caller should
      override manually. (pose+bbox fused format needs manual port editing)
    """
    if not shape:
        return "tensor"

    positive = [d for d in shape if d > 0]
    name_lower = name.lower()

    # 4D → mask / segmentation proto
    if len(shape) == 4:
        return "mask"

    # 3D patterns
    if len(shape) == 3:
        last = positive[-1] if positive else None
        mid  = positive[-2] if len(positive) >= 2 else None
        # Pose: [..., 17, 3]
        if mid == 17 and last == 3:
            return "keypoints"
        # Landmarks: [..., 5, 2] or [..., 10, 2]
        if last == 2 and mid in (5, 10):
            return "landmarks"
        # Detection: last dim is class+bbox count (common YOLO formats)
        _DET_LAST = {4, 5, 6, 85, 116}
        _DET_MID  = {4, 5, 6, 85, 116}
        if last in _DET_LAST:
            return "detections"
        if mid in _DET_MID:
            return "detections"
        # Generic 3D: many detection models have [B, N, C] where C varies
        if last is not None and last < 200:
            return "detections"
        return "tensor"

    # 2D / effectively 1D (after batch dim)
    if len(positive) == 1:
        dim = positive[0]
        if dim in _EMBEDDING_DIMS:
            return "embeddings"
        if dim <= 1000:
            return "class_scores"
        # Large unusual dim → treat as embedding
        return "embeddings"

    # 2D: [B, dim]
    if len(shape) == 2:
        dim = shape[-1] if shape[-1] > 0 else (shape[0] if shape[0] > 0 else 0)
        if dim in _EMBEDDING_DIMS:
            return "embeddings"
        if 0 < dim <= 1000:
            return "class_scores"

    # Name-based fallback
    if any(k in name_lower for k in ("embed", "feature", "repr", "latent")):
        return "embeddings"
    if any(k in name_lower for k in ("cls", "class", "score", "prob", "logit")):
        return "class_scores"
    if any(k in name_lower for k in ("det", "box", "pred", "bbox", "anchor")):
        return "detections"
    if any(k in name_lower for k in ("kpt", "pose", "joint", "landmark")):
        return "keypoints"
    if any(k in name_lower for k in ("mask", "seg", "proto")):
        return "mask"

    return "tensor"


# ── main API ──────────────────────────────────────────────────────────────────

def inspect_onnx(onnx_path: str | Path) -> dict:
    """Load an ONNX model and extract structured I/O metadata.

    Returns a ``ports_json``-compatible dict::

        {
            "inputs":  [{name, tensor_name, type, shape, dtype, dynamic_axes, desc, warning?}, ...],
            "outputs": [{name, tensor_name, type, shape, dtype, optional, desc}, ...],
            "meta":    {opset, ir_version, warning?},
        }

    Args:
        onnx_path: Path to the .onnx file.

    Raises:
        FileNotFoundError: if the file does not exist.
        Exception: propagates onnx.load errors (corrupt file, version mismatch, etc.)
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Initializers are weight tensors embedded in the graph — not runtime inputs
    initializer_names: set[str] = {init.name for init in graph.initializer}

    # ── Inputs ────────────────────────────────────────────────────────────────
    inputs: list[dict] = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue  # skip weight tensors

        t      = inp.type.tensor_type
        shape  = _extract_shape(t.shape)
        dtype  = _onnx_dtype_to_str(t.elem_type)
        dyn    = [i for i, d in enumerate(shape) if d == -1]
        ptype  = _guess_input_type(inp.name, shape, dtype)

        entry: dict = {
            "name":         _friendly_name(inp.name, f"input_{len(inputs)}"),
            "tensor_name":  inp.name,
            "type":         ptype,
            "shape":        shape,
            "dtype":        dtype,
            "dynamic_axes": dyn,
            "desc":         "",
        }
        if dtype in ("int8", "uint8"):
            entry["warning"] = (
                f"Quantized model (dtype={dtype}). "
                "Preprocessing may need dequantization or integer-range inputs."
            )
        inputs.append(entry)

    # ── Outputs ───────────────────────────────────────────────────────────────
    outputs: list[dict] = []
    for out in graph.output:
        t     = out.type.tensor_type
        shape = _extract_shape(t.shape)
        dtype = _onnx_dtype_to_str(t.elem_type)
        ptype = _guess_output_type(out.name, shape, dtype, len(graph.output))

        outputs.append({
            "name":        _friendly_name(out.name, f"output_{len(outputs)}"),
            "tensor_name": out.name,
            "type":        ptype,
            "shape":       shape,
            "dtype":       dtype,
            "optional":    False,
            "desc":        "",
        })

    # ── Model metadata ────────────────────────────────────────────────────────
    opset = model.opset_import[0].version if model.opset_import else 0
    meta: dict = {
        "opset":      opset,
        "ir_version": model.ir_version,
    }
    if opset < 11:
        meta["warning"] = (
            f"Opset {opset} < 11 — may have compatibility issues with "
            "recent versions of onnxruntime."
        )

    return {
        "inputs":  inputs,
        "outputs": outputs,
        "meta":    meta,
    }
