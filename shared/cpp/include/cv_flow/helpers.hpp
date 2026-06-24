#pragma once
#include "context.h"
#include <opencv2/core.hpp>
#include <string>
#include <cstring>

/*
 * CV-FLOW C++ Helper Layer
 * ========================
 * Optional C++ convenience wrappers. Include this header instead of node.h
 * when you want idiomatic C++ access to the CVFlowCtx.
 *
 * These are header-only — no separate compilation unit needed.
 */

namespace cvflow {

/* Wrap the raw frame pointer as a cv::Mat (zero-copy, BGR, 8UC3).
   The Mat shares memory with the engine's numpy array — writes are visible
   to the Python side immediately after cv_flow_process() returns. */
inline cv::Mat frame_mat(CVFlowCtx* ctx) {
    return cv::Mat(ctx->height, ctx->width, CV_8UC3, ctx->frame_data);
}

/* Get current metadata JSON as std::string. */
inline std::string metadata(const CVFlowCtx* ctx) {
    return std::string(ctx->metadata_json);
}

/* Overwrite the metadata JSON bus. Silently truncates if content exceeds
   CVFLOW_METADATA_LEN-1 bytes. */
inline void set_metadata(CVFlowCtx* ctx, const std::string& json) {
    std::strncpy(ctx->metadata_json, json.c_str(), CVFLOW_METADATA_LEN - 1);
    ctx->metadata_json[CVFLOW_METADATA_LEN - 1] = '\0';
}

/* Append one detection. Returns false if detection_capacity is exhausted
   (engine pre-allocates a fixed array; the C++ node must not exceed it). */
inline bool add_detection(CVFlowCtx* ctx,
                          float x1, float y1, float x2, float y2,
                          float confidence, int class_id,
                          const char* class_name, int track_id = -1) {
    if (ctx->detection_count >= ctx->detection_capacity) return false;
    CVFlowDetection& d = ctx->detections[ctx->detection_count++];
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    d.confidence = confidence;
    d.class_id   = class_id;
    d.track_id   = track_id;
    std::strncpy(d.class_name, class_name, CVFLOW_CLASS_NAME_LEN - 1);
    d.class_name[CVFLOW_CLASS_NAME_LEN - 1] = '\0';
    d.metadata_json[0] = '\0';
    return true;
}

/* Remove all detections (reset count to zero). */
inline void clear_detections(CVFlowCtx* ctx) {
    ctx->detection_count = 0;
}

} /* namespace cvflow */
