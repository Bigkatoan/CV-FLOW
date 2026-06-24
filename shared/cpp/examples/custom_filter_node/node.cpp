/*
 * custom_filter_node — Example CV-FLOW C++ node
 * ===============================================
 * Filters detections by minimum confidence and draws colored boxes.
 * Config JSON: { "min_conf": 0.5, "box_thickness": 2 }
 */

#include <cv_flow/helpers.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstring>

static float g_min_conf      = 0.5f;
static int   g_box_thickness = 2;

/* Simple color palette indexed by class_id */
static cv::Scalar class_color(int class_id) {
    static const cv::Scalar palette[] = {
        {255, 56,  56},  {255, 157, 151}, {255, 112, 31},
        {255, 178, 29},  {207, 210, 49},  {72,  249, 10},
        {146, 204, 23},  {61,  219, 134}, {26,  147, 52},
        {0,   212, 187}, {44,  153, 168}, {0,   194, 255},
    };
    return palette[class_id % 12];
}

extern "C" {

void cv_flow_setup(const char* config_json) {
    float mc = 0.5f; int bt = 2;
    sscanf(config_json, "{\"min_conf\":%f,\"box_thickness\":%d}", &mc, &bt);
    g_min_conf      = (mc > 0.0f && mc <= 1.0f) ? mc : 0.5f;
    g_box_thickness = (bt > 0) ? bt : 2;
}

void cv_flow_process(CVFlowCtx* ctx) {
    cv::Mat frame = cvflow::frame_mat(ctx);
    int kept = 0;

    for (int i = 0; i < ctx->detection_count; ++i) {
        CVFlowDetection& d = ctx->detections[i];
        if (d.confidence < g_min_conf) continue;

        /* Keep detection (compact in-place) */
        if (kept != i) ctx->detections[kept] = d;
        ++kept;

        /* Draw bounding box */
        cv::rectangle(frame,
                      cv::Point((int)d.x1, (int)d.y1),
                      cv::Point((int)d.x2, (int)d.y2),
                      class_color(d.class_id),
                      g_box_thickness);

        /* Draw label */
        char label[96];
        snprintf(label, sizeof(label), "%s %.2f", d.class_name, d.confidence);
        cv::putText(frame, label,
                    cv::Point((int)d.x1, (int)d.y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    class_color(d.class_id), 1);
    }
    ctx->detection_count = kept;
}

void cv_flow_teardown(void) {}

const char* cv_flow_version(void) { return "1.0.0"; }

} /* extern "C" */
