#pragma once
#include "context.h"

/*
 * CV-FLOW C++ Node Interface
 * ==========================
 * Implement these three functions in your .so. The Python engine calls them
 * using ctypes — they must be extern "C" (no name mangling).
 *
 * Lifecycle per pipeline session:
 *   cv_flow_setup()    — called once at pipeline start
 *   cv_flow_process()  — called for every frame in the pipeline loop
 *   cv_flow_teardown() — called once when pipeline stops
 *
 * Thread safety: all three are called from the same thread. No locking needed
 * unless your node spawns its own worker threads internally.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * cv_flow_setup — initialize node state.
 * @param config_json  Node config from PipelineJSON as a JSON string.
 *                     Parse it with your preferred JSON library (nlohmann/json,
 *                     cJSON, rapidjson, or a simple sscanf for trivial configs).
 */
void cv_flow_setup(const char* config_json);

/*
 * cv_flow_process — process one frame.
 * @param ctx  Pointer to the shared FrameContext. Modify ctx->frame_data,
 *             ctx->detections, and ctx->metadata_json in-place.
 *             Never free or reallocate ctx or its internal pointers.
 */
void cv_flow_process(CVFlowCtx* ctx);

/*
 * cv_flow_teardown — release resources (close file handles, free heap, etc.).
 */
void cv_flow_teardown(void);

/*
 * cv_flow_version — optional, return a version string for diagnostics.
 * Default: return "1.0.0" if not implemented.
 */
const char* cv_flow_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
