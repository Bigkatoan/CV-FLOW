/*
 * blur_node — Example CV-FLOW C++ node
 * =====================================
 * Applies Gaussian blur to the incoming frame.
 * Config JSON: { "kernel_size": 15, "sigma": 0.0 }
 *
 * Compile:
 *   cmake -B build && cmake --build build
 *   # Produces: build/node.so
 *
 * Or paste this code directly into the C++ Function Node editor in the
 * CV-FLOW web UI and click "Compile".
 */

#include <cv_flow/helpers.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>

static int    g_kernel_size = 15;
static double g_sigma       = 0.0;

extern "C" {

void cv_flow_setup(const char* config_json) {
    /* Simple sscanf parse — use nlohmann/json for complex configs */
    int ks = 15; double sigma = 0.0;
    sscanf(config_json, "{\"kernel_size\":%d,\"sigma\":%lf}", &ks, &sigma);
    /* Kernel size must be odd and positive */
    g_kernel_size = (ks > 0 && ks % 2 == 1) ? ks : 15;
    g_sigma = (sigma >= 0) ? sigma : 0.0;
}

void cv_flow_process(CVFlowCtx* ctx) {
    cv::Mat frame = cvflow::frame_mat(ctx);
    cv::GaussianBlur(frame, frame,
                     cv::Size(g_kernel_size, g_kernel_size),
                     g_sigma);
    /* Detections pass through unchanged */
}

void cv_flow_teardown(void) {
    /* Nothing to release */
}

const char* cv_flow_version(void) {
    return "1.0.0";
}

} /* extern "C" */
