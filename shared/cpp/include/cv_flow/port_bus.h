#pragma once
/*
 * CV-FLOW PortBus — C++ API for Direct Access Memory inter-node communication
 * ===========================================================================
 * A PortBus is a named shared-memory region that lets C++ and Python nodes
 * exchange frames with ZERO copies.  The producer writes into the shared buffer;
 * the consumer reads from it directly — no sockets, no serialisation.
 *
 * Binary layout (same offsets in Python dam.py and here):
 *
 *   [CVFlowBusHeader : 64 bytes]    — seq_no, dims, flags, counters
 *   [Frame region   : W×H×C bytes]  — BGR uint8 or float32 CHW
 *   [Detections     : 512×348 bytes] — CVFlowDetection[] (see context.h)
 *   [Metadata JSON  : 4096 bytes]   — UTF-8 null-terminated JSON string
 *
 * Usage — producer (writer) side:
 *
 *   CVFlowPortBus* bus = cv_flow_bus_open("my_bus_name");
 *   cv_flow_bus_write(bus, &ctx);
 *   // ... repeat for every frame
 *   cv_flow_bus_close(bus);
 *
 * Usage — consumer (reader) side:
 *
 *   CVFlowPortBus* bus = cv_flow_bus_attach("my_bus_name");
 *   CVFlowCtx ctx = {};
 *   while (running) {
 *       if (cv_flow_bus_read(bus, &ctx, 30) == 0) {
 *           // process ctx ...
 *       }
 *   }
 *   cv_flow_bus_close(bus);
 *
 * Multi-process C++ node loop (new in multi-process mode):
 *
 *   void cv_flow_setup_bus(config_json, input_bus_names[], output_bus_names[]);
 *   void cv_flow_run();   // blocking loop: read → process → write
 *   void cv_flow_stop();  // signal shutdown from orchestrator
 *
 * The Python orchestrator passes --input-bus and --output-bus arguments to the
 * C++ node subprocess, which calls cv_flow_setup_bus() to open those buses.
 */

#include "context.h"
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Header struct (matches dam.py _HDR_FMT "<QiiBBHIQQ24s") ─────────────── */

#define CVFLOW_BUS_HDR_SIZE   64
#define CVFLOW_BUS_MAX_DETS  512
#define CVFLOW_BUS_MAX_META  4096

typedef struct {
    uint64_t seq_no;        /* Monotonic write counter — reader compares to detect new frame */
    int32_t  width;         /* Frame width actually written */
    int32_t  height;        /* Frame height actually written */
    uint8_t  channels;      /* Channels in frame region (usually 3) */
    uint8_t  flags;         /* FLAG_FRAME_VALID | FLAG_DET_VALID | FLAG_META_VALID */
    uint16_t det_count;     /* Number of detections written */
    uint32_t meta_len;      /* Bytes of JSON in metadata region */
    uint64_t write_count;   /* Total frames written (for AutoScaler buffer depth) */
    uint64_t read_count;    /* Total frames consumed (consumer updates) */
    uint8_t  reserved[24];
} CVFlowBusHeader;

#define CVFLOW_FLAG_FRAME_VALID 0x01
#define CVFLOW_FLAG_DET_VALID   0x02
#define CVFLOW_FLAG_META_VALID  0x04

/* ── PortBus handle ──────────────────────────────────────────────────────── */

typedef struct {
    CVFlowBusHeader* header;   /* Points to start of shared memory */
    uint8_t*         shm_base; /* Same as header, kept for unmap/close */
    size_t           shm_size; /* Total size of shared memory region */
    int              is_owner; /* 1 if this process created the shm (will unlink on close) */
    uint64_t         last_seq; /* Last seq_no seen by reader */
    int32_t          max_w;
    int32_t          max_h;
    int32_t          max_c;

    /* Region offsets within shm_base */
    size_t           frame_off;
    size_t           det_off;
    size_t           meta_off;

#ifdef _WIN32
    HANDLE           win_handle;
#endif
} CVFlowPortBus;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

/*
 * cv_flow_bus_open — create a new PortBus (producer/owner side).
 * Allocates shared memory of the given capacity.
 * Returns NULL on error.
 */
static inline CVFlowPortBus* cv_flow_bus_open(
    const char* name,
    int32_t     max_w,
    int32_t     max_h,
    int32_t     max_c)
{
    size_t frame_off = CVFLOW_BUS_HDR_SIZE;
    size_t frame_sz  = (size_t)max_w * max_h * max_c;
    size_t det_off   = frame_off + frame_sz;
    size_t det_sz    = CVFLOW_BUS_MAX_DETS * sizeof(CVFlowDetection);
    size_t meta_off  = det_off + det_sz;
    size_t total     = meta_off + CVFLOW_BUS_MAX_META;

    CVFlowPortBus* bus = (CVFlowPortBus*)calloc(1, sizeof(CVFlowPortBus));
    if (!bus) return NULL;

    bus->max_w     = max_w;
    bus->max_h     = max_h;
    bus->max_c     = max_c;
    bus->frame_off = frame_off;
    bus->det_off   = det_off;
    bus->meta_off  = meta_off;
    bus->shm_size  = total;
    bus->is_owner  = 1;

#ifdef _WIN32
    bus->win_handle = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        (DWORD)(total >> 32), (DWORD)(total & 0xFFFFFFFF), name);
    if (!bus->win_handle) { free(bus); return NULL; }
    bus->shm_base = (uint8_t*)MapViewOfFile(
        bus->win_handle, FILE_MAP_ALL_ACCESS, 0, 0, total);
    if (!bus->shm_base) { CloseHandle(bus->win_handle); free(bus); return NULL; }
#else
    /* POSIX shm_open */
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "/%s", name);
    int fd = shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) { free(bus); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) { close(fd); free(bus); return NULL; }
    bus->shm_base = (uint8_t*)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
    close(fd);
    if (bus->shm_base == MAP_FAILED) { free(bus); return NULL; }
#endif

    bus->header = (CVFlowBusHeader*)bus->shm_base;
    memset(bus->header, 0, CVFLOW_BUS_HDR_SIZE);
    return bus;
}

/*
 * cv_flow_bus_attach — attach to an existing PortBus (consumer side).
 * Does NOT allocate; the creator's size must match max_w/max_h/max_c.
 */
static inline CVFlowPortBus* cv_flow_bus_attach(
    const char* name,
    int32_t     max_w,
    int32_t     max_h,
    int32_t     max_c)
{
    size_t frame_off = CVFLOW_BUS_HDR_SIZE;
    size_t frame_sz  = (size_t)max_w * max_h * max_c;
    size_t det_off   = frame_off + frame_sz;
    size_t det_sz    = CVFLOW_BUS_MAX_DETS * sizeof(CVFlowDetection);
    size_t meta_off  = det_off + det_sz;
    size_t total     = meta_off + CVFLOW_BUS_MAX_META;

    CVFlowPortBus* bus = (CVFlowPortBus*)calloc(1, sizeof(CVFlowPortBus));
    if (!bus) return NULL;

    bus->max_w     = max_w;
    bus->max_h     = max_h;
    bus->max_c     = max_c;
    bus->frame_off = frame_off;
    bus->det_off   = det_off;
    bus->meta_off  = meta_off;
    bus->shm_size  = total;
    bus->is_owner  = 0;

#ifdef _WIN32
    bus->win_handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (!bus->win_handle) { free(bus); return NULL; }
    bus->shm_base = (uint8_t*)MapViewOfFile(
        bus->win_handle, FILE_MAP_ALL_ACCESS, 0, 0, total);
    if (!bus->shm_base) { CloseHandle(bus->win_handle); free(bus); return NULL; }
#else
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "/%s", name);
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd < 0) { free(bus); return NULL; }
    bus->shm_base = (uint8_t*)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
    close(fd);
    if (bus->shm_base == MAP_FAILED) { free(bus); return NULL; }
#endif

    bus->header = (CVFlowBusHeader*)bus->shm_base;
    return bus;
}

/*
 * cv_flow_bus_close — release shared memory mapping.
 * Owner automatically unlinks the shm object on POSIX.
 */
static inline void cv_flow_bus_close(CVFlowPortBus* bus) {
    if (!bus) return;
#ifdef _WIN32
    if (bus->shm_base) UnmapViewOfFile(bus->shm_base);
    if (bus->win_handle) CloseHandle(bus->win_handle);
#else
    if (bus->shm_base && bus->shm_base != MAP_FAILED)
        munmap(bus->shm_base, bus->shm_size);
    /* Owner unlinks on close so OS releases backing store */
    /* Python will also unlink; double-unlink is harmless on most OSes */
#endif
    free(bus);
}

/* ── Write ───────────────────────────────────────────────────────────────── */

/*
 * cv_flow_bus_write — copy ctx into the PortBus shared memory slot.
 * frame_data is copied into the frame region; detections and metadata_json
 * are copied into their respective regions.  seq_no is incremented last so
 * readers can safely detect the new frame without a lock.
 */
static inline void cv_flow_bus_write(CVFlowPortBus* bus, const CVFlowCtx* ctx) {
    if (!bus || !ctx) return;

    CVFlowBusHeader* hdr = bus->header;
    uint8_t* frame_dst   = bus->shm_base + bus->frame_off;
    CVFlowDetection* det_dst = (CVFlowDetection*)(bus->shm_base + bus->det_off);
    char*    meta_dst    = (char*)(bus->shm_base + bus->meta_off);

    int32_t ww = (ctx->width  < bus->max_w) ? ctx->width  : bus->max_w;
    int32_t hh = (ctx->height < bus->max_h) ? ctx->height : bus->max_h;
    int32_t cc = (ctx->channels < bus->max_c) ? ctx->channels : bus->max_c;
    uint8_t flags = 0;

    /* Frame */
    if (ctx->frame_data) {
        size_t row_bytes_src = (size_t)ctx->width   * ctx->channels;
        size_t row_bytes_dst = (size_t)bus->max_w   * bus->max_c;
        for (int32_t r = 0; r < hh; ++r)
            memcpy(frame_dst + r * row_bytes_dst,
                   ctx->frame_data + r * row_bytes_src,
                   (size_t)ww * cc);
        flags |= CVFLOW_FLAG_FRAME_VALID;
    }

    /* Detections */
    uint16_t det_count = 0;
    if (ctx->detection_count > 0) {
        det_count = (ctx->detection_count < CVFLOW_BUS_MAX_DETS)
                    ? (uint16_t)ctx->detection_count
                    : CVFLOW_BUS_MAX_DETS;
        memcpy(det_dst, ctx->detections, det_count * sizeof(CVFlowDetection));
        flags |= CVFLOW_FLAG_DET_VALID;
    }

    /* Metadata JSON */
    uint32_t meta_len = 0;
    if (ctx->metadata_json[0]) {
        meta_len = (uint32_t)strnlen(ctx->metadata_json, CVFLOW_METADATA_LEN - 1);
        uint32_t copy_len = (meta_len < CVFLOW_BUS_MAX_META - 1)
                            ? meta_len : CVFLOW_BUS_MAX_META - 1;
        memcpy(meta_dst, ctx->metadata_json, copy_len);
        meta_dst[copy_len] = '\0';
        flags |= CVFLOW_FLAG_META_VALID;
    }

    /* Update header — seq_no last (memory barrier equivalent via volatile write) */
    hdr->width       = ww;
    hdr->height      = hh;
    hdr->channels    = (uint8_t)cc;
    hdr->flags       = flags;
    hdr->det_count   = det_count;
    hdr->meta_len    = meta_len;
    hdr->write_count = hdr->write_count + 1;
    /* seq_no written last so reader sees consistent frame */
    hdr->seq_no = hdr->seq_no + 1;
}

/* ── Read ────────────────────────────────────────────────────────────────── */

/*
 * cv_flow_bus_read — wait for a new frame and populate ctx.
 * Returns 0 on success, -1 on timeout.
 * timeout_ms: milliseconds to wait (0 = non-blocking).
 *
 * Frame pointer in ctx points INTO the shared memory region — zero-copy.
 * Do NOT free or hold it after cv_flow_bus_close().
 */
static inline int cv_flow_bus_read(
    CVFlowPortBus* bus,
    CVFlowCtx*     ctx,
    int            timeout_ms)
{
    if (!bus || !ctx) return -1;

    CVFlowBusHeader* hdr   = bus->header;
    uint8_t*         frame = bus->shm_base + bus->frame_off;
    CVFlowDetection* dets  = (CVFlowDetection*)(bus->shm_base + bus->det_off);
    char*            meta  = (char*)(bus->shm_base + bus->meta_off);

#ifdef _WIN32
    DWORD deadline = GetTickCount() + (DWORD)timeout_ms;
#else
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif

    while (1) {
        uint64_t seq = hdr->seq_no;
        if (seq != bus->last_seq && (hdr->flags & CVFLOW_FLAG_FRAME_VALID)) {
            /* New frame available — populate ctx (zero-copy for frame) */
            ctx->frame_data      = frame;           /* Points into shm */
            ctx->width           = hdr->width;
            ctx->height          = hdr->height;
            ctx->channels        = hdr->channels;
            ctx->frame_number    = (int)hdr->write_count;
            ctx->detection_count = hdr->det_count;
            ctx->detections      = dets;            /* Points into shm */
            if (hdr->flags & CVFLOW_FLAG_META_VALID) {
                uint32_t mlen = hdr->meta_len < CVFLOW_METADATA_LEN - 1
                                ? hdr->meta_len : CVFLOW_METADATA_LEN - 1;
                memcpy(ctx->metadata_json, meta, mlen);
                ctx->metadata_json[mlen] = '\0';
            }
            bus->last_seq = seq;
            hdr->read_count = hdr->read_count + 1;
            return 0;
        }

        /* Check timeout */
        if (timeout_ms == 0) return -1;
#ifdef _WIN32
        if ((int)(GetTickCount() - deadline) > 0) return -1;
        Sleep(1);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        long elapsed_ms = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000
                        + (ts_now.tv_nsec - ts_start.tv_nsec) / 1000000;
        if (elapsed_ms >= timeout_ms) return -1;
        struct timespec sleep_ts = {0, 500000}; /* 0.5ms */
        nanosleep(&sleep_ts, NULL);
#endif
    }
}

/* ── Buffer depth ────────────────────────────────────────────────────────── */

/*
 * cv_flow_bus_depth — frames produced but not yet consumed.
 * AutoScaler calls this to decide whether to scale up a node.
 */
static inline uint64_t cv_flow_bus_depth(const CVFlowPortBus* bus) {
    if (!bus) return 0;
    uint64_t wc = bus->header->write_count;
    uint64_t rc = bus->header->read_count;
    return (wc > rc) ? wc - rc : 0;
}

/* ── Multi-process C++ node API ──────────────────────────────────────────── */

/*
 * In multi-process mode the orchestrator spawns the C++ node as a subprocess
 * and passes bus names via command-line arguments.  The C++ node then:
 *
 *   1. Calls cv_flow_setup_bus(config, input_names, output_names).
 *   2. Calls cv_flow_run() — blocks, processing frames in a loop.
 *   3. The orchestrator calls cv_flow_stop() (via signal or pipe) to exit.
 *
 * Implement these in your node.cpp alongside cv_flow_setup / cv_flow_process /
 * cv_flow_teardown.  The run loop below is provided as a convenience template.
 *
 * Example node.cpp snippet:
 *
 *   static CVFlowPortBus* g_in  = NULL;
 *   static CVFlowPortBus* g_out = NULL;
 *   static volatile int   g_run = 1;
 *
 *   void cv_flow_setup_bus(const char* cfg,
 *                           const char** in_names, const char** out_names) {
 *       cv_flow_setup(cfg);
 *       g_in  = cv_flow_bus_attach(in_names[0],  1280, 720, 3);
 *       g_out = cv_flow_bus_open  (out_names[0], 1280, 720, 3);
 *   }
 *
 *   void cv_flow_run(void) {
 *       CVFlowCtx ctx = {};
 *       while (g_run) {
 *           if (cv_flow_bus_read(g_in, &ctx, 30) == 0) {
 *               cv_flow_process(&ctx);
 *               cv_flow_bus_write(g_out, &ctx);
 *           }
 *       }
 *   }
 *
 *   void cv_flow_stop(void) { g_run = 0; }
 */

/*
 * Declarations — implement in your node.cpp for multi-process mode.
 * Not required for sequential (ctypes) mode.
 */
void cv_flow_setup_bus(const char* config_json,
                       const char** input_bus_names,
                       const char** output_bus_names);
void cv_flow_run(void);
void cv_flow_stop(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
