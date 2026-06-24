#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Max sizes for fixed-length fields in CVFlowDetection and CVFlowCtx.
   Keep in sync with engine/cpp_bridge/context_shm.py. */
#define CVFLOW_CLASS_NAME_LEN  64
#define CVFLOW_DET_META_LEN   256
#define CVFLOW_SESSION_ID_LEN  64
#define CVFLOW_METADATA_LEN  4096

typedef struct {
    float x1, y1, x2, y2;              /* Bounding box — xyxy absolute pixels */
    float confidence;
    int   class_id;
    char  class_name[CVFLOW_CLASS_NAME_LEN];
    int   track_id;                     /* -1 if not tracked */
    char  metadata_json[CVFLOW_DET_META_LEN]; /* per-detection arbitrary JSON */
} CVFlowDetection;

typedef struct {
    /* --- Frame buffer (zero-copy: points into numpy array memory) --- */
    uint8_t* frame_data;                /* BGR packed bytes, row-major [H × W × 3] */
    int      width;
    int      height;
    int      channels;                  /* Always 3 (BGR) */

    /* --- Frame metadata --- */
    int    frame_number;
    double timestamp;                   /* Unix epoch, seconds */
    char   session_id[CVFLOW_SESSION_ID_LEN];

    /* --- Detections array (pre-allocated by Python engine) ---
       C++ nodes may modify existing entries and update detection_count,
       but MUST NOT exceed detection_capacity or free/reallocate the pointer. */
    CVFlowDetection* detections;
    int detection_count;
    int detection_capacity;

    /* --- Metadata bus (arbitrary JSON string) ---
       Read and write structured data here for inter-node communication.
       Downstream nodes parse this JSON to access upstream results. */
    char metadata_json[CVFLOW_METADATA_LEN];
} CVFlowCtx;

#ifdef __cplusplus
} /* extern "C" */
#endif
