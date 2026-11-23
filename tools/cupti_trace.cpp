// Optional CUPTI activity tracer for RAMforge benchmark attribution.
//
// Build:
//   g++ -O2 -shared -fPIC -o /tmp/libforge_cupti_trace.so cupti_trace.cpp \
//       -I/usr/include -L/usr/lib64 -lcupti -ldl
//
// Run:
//   LD_PRELOAD=/tmp/libforge_cupti_trace.so \
//   FORGE_CUPTI_OUT=/tmp/forge-cupti.csv \
//   ramforge benchmark ...
//
// This is validation tooling, not a runtime dependency. It records kernel,
// memcpy, runtime API and driver API intervals. Timestamps are CUPTI device/
// host timestamps in nanoseconds.

#include <cupti.h>
#include <cupti_activity.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
std::mutex g_mutex;
FILE * g_out = nullptr;

const char * kind_name(CUpti_ActivityKind kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL: return "kernel";
        case CUPTI_ACTIVITY_KIND_MEMCPY: return "memcpy";
        case CUPTI_ACTIVITY_KIND_MEMCPY2: return "memcpy2";
        case CUPTI_ACTIVITY_KIND_RUNTIME: return "runtime";
        case CUPTI_ACTIVITY_KIND_DRIVER: return "driver";
        case CUPTI_ACTIVITY_KIND_MEMSET: return "memset";
        default: return "other";
    }
}

void CUPTIAPI buffer_requested(uint8_t ** buffer, size_t * size, size_t * max_records) {
    *size = 8 * 1024 * 1024;
    *buffer = static_cast<uint8_t *>(std::malloc(*size));
    *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t * buffer, size_t, size_t valid) {
    CUpti_Activity * record = nullptr;
    std::lock_guard<std::mutex> lock(g_mutex);
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        const char * kind = kind_name(record->kind);
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                auto * kernel = reinterpret_cast<CUpti_ActivityKernel11 *>(record);
                std::fprintf(g_out, "%s,%llu,%llu,%u,%s\n", kind,
                        static_cast<unsigned long long>(kernel->start),
                        static_cast<unsigned long long>(kernel->end),
                        kernel->streamId, kernel->name ? kernel->name : "");
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto * copy = reinterpret_cast<CUpti_ActivityMemcpy6 *>(record);
                std::fprintf(g_out, "%s,%llu,%llu,%u,%llu;kind=%u\n", kind,
                        static_cast<unsigned long long>(copy->start),
                        static_cast<unsigned long long>(copy->end),
                        copy->streamId, static_cast<unsigned long long>(copy->bytes),
                        static_cast<unsigned>(copy->copyKind));
                break;
            }
            case CUPTI_ACTIVITY_KIND_RUNTIME:
            case CUPTI_ACTIVITY_KIND_DRIVER: {
                auto * api = reinterpret_cast<CUpti_ActivityAPI *>(record);
                std::fprintf(g_out, "%s,%llu,%llu,%u,%llu\n", kind,
                        static_cast<unsigned long long>(api->start),
                        static_cast<unsigned long long>(api->end),
                        api->threadId, static_cast<unsigned long long>(api->cbid));
                break;
            }
            default:
                break;
        }
    }
    std::fflush(g_out);
    std::free(buffer);
}

__attribute__((constructor)) void forge_cupti_init() {
    const char * path = std::getenv("FORGE_CUPTI_OUT");
    g_out = std::fopen(path ? path : "/tmp/forge-cupti.csv", "w");
    if (g_out == nullptr) {
        std::fprintf(stderr, "forge-cupti: cannot open output file\n");
        return;
    }
    std::fprintf(g_out, "kind,start_ns,end_ns,stream_or_thread,detail\n");
    CUptiResult rc = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (rc != CUPTI_SUCCESS) {
        std::fprintf(stderr, "forge-cupti: register failed: %d\n", rc);
        return;
    }
    std::vector<CUpti_ActivityKind> kinds = {CUPTI_ACTIVITY_KIND_KERNEL,
            CUPTI_ACTIVITY_KIND_MEMCPY};
    if (std::getenv("FORGE_CUPTI_API") == nullptr ||
        std::strcmp(std::getenv("FORGE_CUPTI_API"), "0") != 0) {
        kinds.push_back(CUPTI_ACTIVITY_KIND_RUNTIME);
        kinds.push_back(CUPTI_ACTIVITY_KIND_DRIVER);
    }
    for (CUpti_ActivityKind kind : kinds) {
        rc = cuptiActivityEnable(kind);
        if (rc != CUPTI_SUCCESS) {
            std::fprintf(stderr, "forge-cupti: enable %d failed: %d\n", kind, rc);
        }
    }
}

__attribute__((destructor)) void forge_cupti_fini() {
    if (g_out == nullptr) {
        return;
    }
    cuptiActivityFlushAll(0);
    std::fclose(g_out);
    g_out = nullptr;
}
} // namespace
