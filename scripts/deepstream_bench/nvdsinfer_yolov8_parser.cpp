/*
 * nvdsinfer_yolov8_parser.cpp — custom NvDsInfer bounding-box parser for
 * YOLOv8 ONNX output, used only for the cv-flow vs DeepStream benchmark in
 * scripts/deepstream_bench/. Not part of the cv-flow Python package.
 *
 * DeepStream ships parsers for YOLOv3/v4-style outputs only; YOLOv8's ONNX
 * export has a different raw layout, so DeepStream needs a custom parser
 * (the documented nvinfer "parse-bbox-func-name" extension point) to decode
 * it. Written from scratch against the raw tensor layout, using the exact
 * same decode math as cv_flow/nodes/postprocess.py::run_nms() (features-
 * first, cx/cy/w/h -> xyxy, per-box argmax over class scores) so this is a
 * fair comparison of the SAME model/algorithm running on a different
 * runtime, not a different detector.
 *
 * Output tensor layout: (84, 8400) — channels-first, exactly as the ONNX
 * model exports it (no batch dim once nvinfer hands it to the parser):
 *   row 0..3   : cx, cy, w, h (in the model's 640x640 input space)
 *   row 4..83  : per-class confidence (80 COCO classes), no separate
 *                objectness score (already folded into class score)
 * NMS/clustering itself is NOT done here — nvinfer's own cluster-mode
 * (configured as NMS in the .txt config) handles that downstream, same
 * separation of concerns as cv_flow's NMS node.
 */
#include <algorithm>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

#define NUM_CLASSES_YOLOV8 80
#define NUM_BOXES_YOLOV8 8400

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        return false;
    }
    const float* buffer = static_cast<const float*>(outputLayersInfo[0].buffer);

    const unsigned int numBoxes = NUM_BOXES_YOLOV8;
    const unsigned int numClasses = NUM_CLASSES_YOLOV8;
    const float inputW = static_cast<float>(networkInfo.width);   // 640
    const float inputH = static_cast<float>(networkInfo.height);  // 640

    // Single global threshold (config supplies the same value for every
    // class via pre-cluster-threshold) — matches run_nms()'s
    // confidence_threshold=0.35 default used throughout this benchmark.
    float confThreshold = 0.35f;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        confThreshold = detectionParams.perClassPreclusterThreshold[0];
    }

    objectList.reserve(numBoxes);

    for (unsigned int box = 0; box < numBoxes; ++box) {
        float bestScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = buffer[(4 + c) * numBoxes + box];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestScore < confThreshold) {
            continue;
        }

        float cx = buffer[0 * numBoxes + box];
        float cy = buffer[1 * numBoxes + box];
        float w  = buffer[2 * numBoxes + box];
        float h  = buffer[3 * numBoxes + box];

        float left = cx - w / 2.0f;
        float top  = cy - h / 2.0f;

        // Clamp to the model's input frame (640x640) — nvinfer maps this
        // back to the original frame coordinates downstream.
        left = std::max(0.0f, std::min(left, inputW));
        top  = std::max(0.0f, std::min(top, inputH));
        w    = std::max(0.0f, std::min(w, inputW - left));
        h    = std::max(0.0f, std::min(h, inputH - top));

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = w;
        obj.height = h;
        obj.detectionConfidence = bestScore;
        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
