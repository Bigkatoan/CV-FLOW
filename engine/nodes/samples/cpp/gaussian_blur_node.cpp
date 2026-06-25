/*
 * gaussian_blur_node.cpp — C++ Node SDK sample
 * =============================================
 * Demonstrates:
 *   1. Sequential mode  (ctypes call from Python runner)
 *   2. Multi-process mode  (standalone subprocess using PortBus)
 *
 * Build (sequential / ctypes mode):
 *   cmake -B build -DCVFLOW_MODE=sequential
 *   cmake --build build
 *   → produces: build/node.so
 *
 * Build (multi-process mode):
 *   cmake -B build -DCVFLOW_MODE=multiprocess
 *   cmake --build build
 *   → produces: build/gaussian_blur_node  (executable)
 *
 * Config JSON: { "kernel_size": 15, "sigma": 0.0 }
 */

#include <cv_flow/helpers.hpp>    /* frame_mat, add_detection, etc. */
#include <cv_flow/port_bus.h>     /* cv_flow_bus_open/attach/read/write/close */
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>

/* ── Node state ─────────────────────────────────────────────────────────── */

static int    g_kernel_size = 15;
static double g_sigma       = 0.0;

static void parse_config(const char* config_json) {
    int ks = 15; double sigma = 0.0;
    if (config_json && config_json[0]) {
        /* Simple parse — for complex configs use nlohmann/json */
        sscanf(config_json, "{\"kernel_size\":%d,\"sigma\":%lf}", &ks, &sigma);
    }
    g_kernel_size = (ks > 0 && ks % 2 == 1) ? ks : (ks + 1);
    g_sigma       = (sigma >= 0.0) ? sigma : 0.0;
}

/* ── Sequential (ctypes) API ────────────────────────────────────────────── */

extern "C" {

void cv_flow_setup(const char* config_json) {
    parse_config(config_json);
    fprintf(stderr, "[gaussian_blur] kernel=%d sigma=%.2f\n", g_kernel_size, g_sigma);
}

void cv_flow_process(CVFlowCtx* ctx) {
    if (!ctx || !ctx->frame_data || ctx->width == 0) return;
    cv::Mat frame = cvflow::frame_mat(ctx);
    cv::GaussianBlur(frame, frame,
                     cv::Size(g_kernel_size, g_kernel_size),
                     g_sigma);
    /* Detections, metadata, and frame_number pass through unchanged */
}

void cv_flow_teardown(void) { /* nothing to release */ }

const char* cv_flow_version(void) { return "2.0.0"; }

/* ── Multi-process API (new) ──────────────────────────────────────────── */

static CVFlowPortBus* g_in_bus  = NULL;
static CVFlowPortBus* g_out_bus = NULL;
static volatile int   g_running = 1;

void cv_flow_setup_bus(
    const char* config_json,
    const char** input_bus_names,
    const char** output_bus_names)
{
    parse_config(config_json);

    /* Default resolution: 1280×720 — orchestrator sets actual dims via
       --input-bus name:W:H:C and --output-bus name:W:H:C arguments.
       Here we read them from a static parsed by main() below. */
    if (input_bus_names && input_bus_names[0])
        g_in_bus  = cv_flow_bus_attach(input_bus_names[0],  1280, 720, 3);
    if (output_bus_names && output_bus_names[0])
        g_out_bus = cv_flow_bus_open  (output_bus_names[0], 1280, 720, 3);

    fprintf(stderr, "[gaussian_blur_mp] in=%s out=%s kernel=%d\n",
            input_bus_names  ? input_bus_names[0]  : "(none)",
            output_bus_names ? output_bus_names[0] : "(none)",
            g_kernel_size);
}

void cv_flow_run(void) {
    CVFlowCtx ctx = {};
    ctx.detections         = (CVFlowDetection*)calloc(512, sizeof(CVFlowDetection));
    ctx.detection_capacity = 512;

    while (g_running) {
        if (cv_flow_bus_read(g_in_bus, &ctx, 30) != 0)
            continue;

        /* Process frame in-place — frame_data points into shared memory */
        cv_flow_process(&ctx);

        if (g_out_bus)
            cv_flow_bus_write(g_out_bus, &ctx);
    }
    free(ctx.detections);
}

void cv_flow_stop(void) {
    g_running = 0;
}

} /* extern "C" */

/* ── main() — only compiled in multiprocess mode ────────────────────────── */

#ifdef CVFLOW_MULTIPROCESS

static void _sig_handler(int) { cv_flow_stop(); }

int main(int argc, char* argv[]) {
    /* Parse: --config JSON --input-bus name:W:H:C --output-bus name:W:H:C */
    const char* config_json  = "{}";
    const char* in_spec      = NULL;
    const char* out_spec     = NULL;

    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--config")      == 0) config_json  = argv[i+1];
        if (strcmp(argv[i], "--input-bus")   == 0) in_spec      = argv[i+1];
        if (strcmp(argv[i], "--output-bus")  == 0) out_spec     = argv[i+1];
    }

    /* Parse "name:W:H:C" spec */
    char in_name[256] = {}, out_name[256] = {};
    int  in_w=1280, in_h=720, in_c=3, out_w=1280, out_h=720, out_c=3;
    if (in_spec)  sscanf(in_spec,  "%255[^:]:%d:%d:%d", in_name,  &in_w,  &in_h,  &in_c);
    if (out_spec) sscanf(out_spec, "%255[^:]:%d:%d:%d", out_name, &out_w, &out_h, &out_c);

    parse_config(config_json);
    if (in_spec)
        g_in_bus  = cv_flow_bus_attach(in_name,  in_w,  in_h,  in_c);
    if (out_spec)
        g_out_bus = cv_flow_bus_open  (out_name, out_w, out_h, out_c);

    signal(SIGINT,  _sig_handler);
    signal(SIGTERM, _sig_handler);

    cv_flow_run();

    if (g_in_bus)  cv_flow_bus_close(g_in_bus);
    if (g_out_bus) cv_flow_bus_close(g_out_bus);
    return 0;
}

#endif /* CVFLOW_MULTIPROCESS */
