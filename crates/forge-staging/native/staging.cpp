#include "staging.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUdeviceptr = unsigned long long;
using CUstream = void *;
using CUevent = void *;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_NOT_READY = 600;

struct CudaApi {
    void * lib = nullptr;
    CUresult (* init)(unsigned int) = nullptr;
    CUresult (* device_get_count)(int *) = nullptr;
    CUresult (* device_get)(CUdevice *, int) = nullptr;
    CUresult (* device_get_name)(char *, int, CUdevice) = nullptr;
    CUresult (* driver_get_version)(int *) = nullptr;
    CUresult (* ctx_create)(CUcontext *, unsigned int, CUdevice) = nullptr;
    CUresult (* ctx_destroy)(CUcontext) = nullptr;
    CUresult (* ctx_push_current)(CUcontext) = nullptr;
    CUresult (* ctx_pop_current)(CUcontext *) = nullptr;
    CUresult (* mem_alloc)(CUdeviceptr *, size_t) = nullptr;
    CUresult (* mem_free)(CUdeviceptr) = nullptr;
    CUresult (* mem_alloc_host)(void **, size_t) = nullptr;
    CUresult (* mem_free_host)(void *) = nullptr;
    CUresult (* memcpy_htod_async)(CUdeviceptr, const void *, size_t, CUstream) = nullptr;
    CUresult (* memcpy_dtoh_async)(void *, CUdeviceptr, size_t, CUstream) = nullptr;
    CUresult (* stream_create)(CUstream *, unsigned int) = nullptr;
    CUresult (* stream_destroy)(CUstream) = nullptr;
    CUresult (* stream_synchronize)(CUstream) = nullptr;
    CUresult (* event_create)(CUevent *, unsigned int) = nullptr;
    CUresult (* event_record)(CUevent, CUstream) = nullptr;
    CUresult (* event_synchronize)(CUevent) = nullptr;
    CUresult (* event_query)(CUevent) = nullptr;
    CUresult (* event_elapsed)(float *, CUevent, CUevent) = nullptr;
    CUresult (* event_destroy)(CUevent) = nullptr;
    CUresult (* error_name)(CUresult, const char **) = nullptr;
    std::string load_error;
};

CudaApi & api() {
    static CudaApi instance = [] {
        CudaApi a;
        const char * names[] = {"libcuda.so.1", "libcuda.so"};
        for (const char * name : names) {
            a.lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (a.lib != nullptr) {
                break;
            }
        }
        if (a.lib == nullptr) {
            a.load_error = "libcuda is not available";
            return a;
        }
        auto load = [&](auto & fn, const char * symbol) {
            fn = reinterpret_cast<std::decay_t<decltype(fn)>>(dlsym(a.lib, symbol));
            if (fn == nullptr) {
                a.load_error = std::string("missing CUDA driver symbol: ") + symbol;
            }
        };
        load(a.init, "cuInit");
        load(a.device_get_count, "cuDeviceGetCount");
        load(a.device_get, "cuDeviceGet");
        load(a.device_get_name, "cuDeviceGetName");
        load(a.driver_get_version, "cuDriverGetVersion");
        load(a.ctx_create, "cuCtxCreate_v2");
        load(a.ctx_destroy, "cuCtxDestroy_v2");
        load(a.ctx_push_current, "cuCtxPushCurrent_v2");
        load(a.ctx_pop_current, "cuCtxPopCurrent_v2");
        load(a.mem_alloc, "cuMemAlloc_v2");
        load(a.mem_free, "cuMemFree_v2");
        load(a.mem_alloc_host, "cuMemAllocHost_v2");
        load(a.mem_free_host, "cuMemFreeHost");
        load(a.memcpy_htod_async, "cuMemcpyHtoDAsync_v2");
        load(a.memcpy_dtoh_async, "cuMemcpyDtoHAsync_v2");
        load(a.stream_create, "cuStreamCreate");
        load(a.stream_destroy, "cuStreamDestroy_v2");
        load(a.stream_synchronize, "cuStreamSynchronize");
        load(a.event_create, "cuEventCreate");
        load(a.event_record, "cuEventRecord");
        load(a.event_synchronize, "cuEventSynchronize");
        load(a.event_query, "cuEventQuery");
        load(a.event_elapsed, "cuEventElapsedTime");
        load(a.event_destroy, "cuEventDestroy_v2");
        load(a.error_name, "cuGetErrorName");
        if (a.load_error.empty() && a.init(0) != CUDA_SUCCESS) {
            a.load_error = "cuInit failed";
        }
        return a;
    }();
    return instance;
}

thread_local std::string g_last_error;

int32_t fail(int32_t code, const std::string & message) {
    g_last_error = message;
    return code;
}

int32_t cuda_fail(CUresult result, const char * operation) {
    const char * name = nullptr;
    if (api().error_name != nullptr) {
        api().error_name(result, &name);
    }
    std::string message = operation;
    message += " failed";
    if (name != nullptr) {
        message += ": ";
        message += name;
    }
    return fail(FSTAGE_CUDA_ERROR, message);
}

uint64_t next_handle() {
    static std::mutex mutex;
    static uint64_t next = 1;
    std::lock_guard<std::mutex> lock(mutex);
    return next++;
}

uint64_t now_ns() {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
}

struct Engine {
    uint64_t id = 0;
    uint32_t device_index = 0;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUevent start = nullptr;
    CUevent end = nullptr;
    bool inflight = false;
    bool usable = true;
    uint64_t src = 0;
    uint64_t dst = 0;
    uint64_t bytes = 0;
    uint32_t direction = 0;
    uint64_t submit_started_ns = 0;
    uint64_t last_submit_ns = 0;
    std::unordered_set<uint64_t> buffers;
    fstage_stats stats{};
    std::string device_name;
    int driver_version = 0;
};

struct Buffer {
    uint64_t id = 0;
    uint64_t engine = 0;
    uint32_t memory_type = 0;
    uint32_t state = FSTAGE_BUFFER_ALLOCATED;
    uint64_t capacity = 0;
    uint64_t allocated = 0;
    void * host = nullptr;
    CUdeviceptr device = 0;
    bool inflight = false;
    bool host_pinned = false;
    uint32_t device_index = 0;
};

std::mutex g_mutex;
std::unordered_map<uint64_t, Engine> g_engines;
std::unordered_map<uint64_t, Buffer> g_buffers;

struct ContextGuard {
    Engine & engine;
    CUcontext previous = nullptr;
    bool ok = false;
    explicit ContextGuard(Engine & e) : engine(e) {
        ok = api().ctx_push_current(engine.context) == CUDA_SUCCESS;
    }
    ~ContextGuard() {
        if (ok) {
            api().ctx_pop_current(&previous);
        }
    }
};

Engine * find_engine(uint64_t handle) {
    auto it = g_engines.find(handle);
    return it == g_engines.end() ? nullptr : &it->second;
}

Buffer * find_buffer(uint64_t handle) {
    auto it = g_buffers.find(handle);
    return it == g_buffers.end() ? nullptr : &it->second;
}

bool buffer_is_host(const Buffer & buffer) {
    return buffer.memory_type == FSTAGE_MEMORY_PAGEABLE_HOST ||
           buffer.memory_type == FSTAGE_MEMORY_PINNED_HOST;
}

void update_buffer_state(Buffer & buffer) {
    if (buffer.state == FSTAGE_BUFFER_RELEASED) {
        return;
    }
    buffer.state = buffer.inflight ? FSTAGE_BUFFER_IN_FLIGHT : FSTAGE_BUFFER_ALLOCATED;
}

void clear_inflight_buffers(Engine & engine) {
    Buffer * src = find_buffer(engine.src);
    Buffer * dst = find_buffer(engine.dst);
    if (src != nullptr) {
        src->inflight = false;
        update_buffer_state(*src);
    }
    if (dst != nullptr) {
        dst->inflight = false;
        update_buffer_state(*dst);
    }
}

void complete_transfer(Engine & engine, float elapsed_ms) {
    engine.inflight = false;
    engine.src = 0;
    engine.dst = 0;
    engine.bytes = 0;
    engine.direction = 0;
    engine.stats.transfer_completed++;
    engine.stats.in_flight_operations = 0;
    engine.stats.last_duration_ns =
            static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1e6);
    engine.stats.total_duration_ns += engine.stats.last_duration_ns;
}

} // namespace

extern "C" {

int32_t fstage_engine_create(uint32_t device_index, uint64_t * out_engine) {
    if (out_engine == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out_engine is null");
    }
    *out_engine = 0;
    try {
        const CudaApi & a = api();
        if (!a.load_error.empty()) {
            return fail(FSTAGE_UNSUPPORTED, a.load_error);
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        int count = 0;
        CUresult rc = a.device_get_count(&count);
        if (rc != CUDA_SUCCESS) {
            return cuda_fail(rc, "cuDeviceGetCount");
        }
        if (count <= 0 || device_index >= static_cast<uint32_t>(count)) {
            return fail(FSTAGE_INVALID_ARGUMENT, "CUDA device index is out of range");
        }
        Engine engine;
        engine.id = next_handle();
        engine.device_index = device_index;
        rc = a.device_get(&engine.device, static_cast<int>(device_index));
        if (rc != CUDA_SUCCESS) {
            return cuda_fail(rc, "cuDeviceGet");
        }
        char name[128] = {0};
        if (a.device_get_name != nullptr &&
            a.device_get_name(name, sizeof(name), engine.device) == CUDA_SUCCESS) {
            engine.device_name = name;
        }
        if (a.driver_get_version != nullptr) {
            a.driver_get_version(&engine.driver_version);
        }
        rc = a.ctx_create(&engine.context, 0, engine.device);
        if (rc != CUDA_SUCCESS) {
            return cuda_fail(rc, "cuCtxCreate");
        }
        rc = a.stream_create(&engine.stream, 1 /* non-blocking */);
        if (rc != CUDA_SUCCESS) {
            a.ctx_destroy(engine.context);
            return cuda_fail(rc, "cuStreamCreate");
        }
        rc = a.event_create(&engine.start, 0);
        if (rc != CUDA_SUCCESS) {
            a.stream_destroy(engine.stream);
            a.ctx_destroy(engine.context);
            return cuda_fail(rc, "cuEventCreate(start)");
        }
        rc = a.event_create(&engine.end, 0);
        if (rc != CUDA_SUCCESS) {
            a.event_destroy(engine.start);
            a.stream_destroy(engine.stream);
            a.ctx_destroy(engine.context);
            return cuda_fail(rc, "cuEventCreate(end)");
        }
        CUcontext previous = nullptr;
        a.ctx_pop_current(&previous);
        *out_engine = engine.id;
        g_engines.emplace(engine.id, std::move(engine));
        return FSTAGE_OK;
    } catch (const std::exception & error) {
        return fail(FSTAGE_INTERNAL, error.what());
    } catch (...) {
        return fail(FSTAGE_INTERNAL, "unknown native exception");
    }
}

int32_t fstage_engine_destroy(uint64_t engine_handle) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        Engine * engine = find_engine(engine_handle);
        if (engine == nullptr) {
            return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
        }
        if (engine->inflight) {
            return fail(FSTAGE_BUSY, "engine has an in-flight transfer");
        }
        if (!engine->buffers.empty()) {
            return fail(FSTAGE_BUSY, "engine still owns allocated buffers");
        }
        const CudaApi & a = api();
        CUcontext context = engine->context;
        {
            ContextGuard guard(*engine);
            if (!guard.ok) {
                return fail(FSTAGE_CUDA_ERROR, "cannot make the staging context current");
            }
            a.event_destroy(engine->end);
            a.event_destroy(engine->start);
            a.stream_destroy(engine->stream);
        }
        a.ctx_destroy(context);
        g_engines.erase(engine_handle);
        return FSTAGE_OK;
    } catch (const std::exception & error) {
        return fail(FSTAGE_INTERNAL, error.what());
    } catch (...) {
        return fail(FSTAGE_INTERNAL, "unknown native exception");
    }
}

int32_t fstage_engine_get_info(uint64_t engine_handle, fstage_engine_info * out) {
    if (out == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out is null");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Engine * engine = find_engine(engine_handle);
    if (engine == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
    }
    std::memset(out, 0, sizeof(*out));
    out->device_index = engine->device_index;
    std::snprintf(out->device_name, sizeof(out->device_name), "%s", engine->device_name.c_str());
    out->driver_version = engine->driver_version;
    out->runtime_version = 0;
    return FSTAGE_OK;
}

int32_t fstage_engine_stats(uint64_t engine_handle, fstage_stats * out) {
    if (out == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out is null");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Engine * engine = find_engine(engine_handle);
    if (engine == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
    }
    *out = engine->stats;
    out->in_flight_operations = engine->inflight ? 1 : 0;
    return FSTAGE_OK;
}

int32_t fstage_buffer_create(
        uint64_t engine_handle, uint32_t memory_type, uint64_t bytes, uint64_t * out) {
    if (out == nullptr || bytes == 0) {
        return fail(FSTAGE_INVALID_ARGUMENT, "buffer size must be non-zero");
    }
    *out = 0;
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        Engine * engine = find_engine(engine_handle);
        if (engine == nullptr) {
            return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
        }
        if (!engine->usable) {
            return fail(FSTAGE_CUDA_ERROR, "engine is unusable after a prior failure");
        }
        if (memory_type != FSTAGE_MEMORY_PAGEABLE_HOST &&
            memory_type != FSTAGE_MEMORY_PINNED_HOST &&
            memory_type != FSTAGE_MEMORY_DEVICE) {
            return fail(FSTAGE_INVALID_ARGUMENT, "unknown memory type");
        }
        engine->stats.allocation_requests++;
        Buffer buffer;
        buffer.id = next_handle();
        buffer.engine = engine_handle;
        buffer.memory_type = memory_type;
        buffer.capacity = bytes;
        buffer.allocated = bytes;
        buffer.device_index = engine->device_index;
        ContextGuard guard(*engine);
        if (!guard.ok && memory_type != FSTAGE_MEMORY_PAGEABLE_HOST) {
            engine->stats.allocation_failures++;
            return fail(FSTAGE_CUDA_ERROR, "cannot make the staging context current");
        }
        if (memory_type == FSTAGE_MEMORY_PAGEABLE_HOST) {
            buffer.host = std::malloc(static_cast<size_t>(bytes));
        } else if (memory_type == FSTAGE_MEMORY_PINNED_HOST) {
            CUresult rc = api().mem_alloc_host(
                    &buffer.host, static_cast<size_t>(bytes));
            if (rc != CUDA_SUCCESS) {
                engine->stats.allocation_failures++;
                return cuda_fail(rc, "cuMemAllocHost");
            }
            buffer.host_pinned = true;
        } else {
            CUresult rc = api().mem_alloc(&buffer.device, static_cast<size_t>(bytes));
            if (rc != CUDA_SUCCESS) {
                engine->stats.allocation_failures++;
                return cuda_fail(rc, "cuMemAlloc");
            }
        }
        if (buffer.host == nullptr && buffer.device == 0) {
            engine->stats.allocation_failures++;
            return fail(FSTAGE_OUT_OF_MEMORY, "staging allocation returned no memory");
        }
        engine->stats.allocation_successes++;
        engine->stats.current_bytes += bytes;
        engine->stats.high_water_bytes =
                std::max(engine->stats.high_water_bytes, engine->stats.current_bytes);
        engine->buffers.insert(buffer.id);
        *out = buffer.id;
        g_buffers.emplace(buffer.id, buffer);
        return FSTAGE_OK;
    } catch (const std::exception & error) {
        return fail(FSTAGE_INTERNAL, error.what());
    } catch (...) {
        return fail(FSTAGE_INTERNAL, "unknown native exception");
    }
}

int32_t fstage_buffer_release(uint64_t buffer_handle) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        Buffer * buffer = find_buffer(buffer_handle);
        if (buffer == nullptr) {
            return fail(FSTAGE_INVALID_HANDLE, "buffer handle is unknown");
        }
        if (buffer->state == FSTAGE_BUFFER_RELEASED) {
            return fail(FSTAGE_ALREADY_RELEASED, "buffer was already released");
        }
        if (buffer->inflight) {
            return fail(FSTAGE_BUSY, "buffer is referenced by an in-flight transfer");
        }
        Engine * engine = find_engine(buffer->engine);
        if (engine != nullptr) {
            if (buffer->memory_type == FSTAGE_MEMORY_PAGEABLE_HOST) {
                std::free(buffer->host);
            } else {
                ContextGuard guard(*engine);
                if (!guard.ok) {
                    return fail(FSTAGE_CUDA_ERROR, "cannot make the staging context current");
                }
                if (buffer->memory_type == FSTAGE_MEMORY_PINNED_HOST) {
                    api().mem_free_host(buffer->host);
                } else {
                    api().mem_free(buffer->device);
                }
            }
            engine->buffers.erase(buffer_handle);
            engine->stats.released++;
            engine->stats.current_bytes =
                    engine->stats.current_bytes > buffer->allocated
                    ? engine->stats.current_bytes - buffer->allocated
                    : 0;
        }
        buffer->host = nullptr;
        buffer->device = 0;
        buffer->state = FSTAGE_BUFFER_RELEASED;
        buffer->inflight = false;
        return FSTAGE_OK;
    } catch (const std::exception & error) {
        return fail(FSTAGE_INTERNAL, error.what());
    } catch (...) {
        return fail(FSTAGE_INTERNAL, "unknown native exception");
    }
}

int32_t fstage_buffer_get_info(uint64_t buffer_handle, fstage_buffer_info * out) {
    if (out == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out is null");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Buffer * buffer = find_buffer(buffer_handle);
    if (buffer == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "buffer handle is unknown");
    }
    std::memset(out, 0, sizeof(*out));
    out->handle = buffer->id;
    out->memory_type = buffer->memory_type;
    out->state = buffer->state;
    out->capacity = buffer->capacity;
    out->allocated = buffer->allocated;
    out->in_flight = buffer->inflight ? 1 : 0;
    out->host_pinned = buffer->host_pinned ? 1 : 0;
    out->device_index = buffer->device_index;
    return FSTAGE_OK;
}

int32_t fstage_buffer_write(
        uint64_t buffer_handle, uint64_t offset, const void * data, uint64_t size) {
    if (data == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "data is null");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Buffer * buffer = find_buffer(buffer_handle);
    if (buffer == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "buffer handle is unknown");
    }
    if (buffer->state == FSTAGE_BUFFER_RELEASED) {
        return fail(FSTAGE_ALREADY_RELEASED, "buffer was released");
    }
    if (buffer->inflight) {
        return fail(FSTAGE_BUSY, "buffer is in flight");
    }
    if (!buffer_is_host(*buffer)) {
        return fail(FSTAGE_INVALID_ARGUMENT, "buffer is not host-accessible");
    }
    if (offset > buffer->capacity || size > buffer->capacity - offset) {
        return fail(FSTAGE_INVALID_ARGUMENT, "write exceeds buffer capacity");
    }
    std::memcpy(static_cast<char *>(buffer->host) + offset, data, static_cast<size_t>(size));
    return FSTAGE_OK;
}

int32_t fstage_buffer_read(
        uint64_t buffer_handle, uint64_t offset, void * out, uint64_t size) {
    if (out == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out is null");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Buffer * buffer = find_buffer(buffer_handle);
    if (buffer == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "buffer handle is unknown");
    }
    if (buffer->state == FSTAGE_BUFFER_RELEASED) {
        return fail(FSTAGE_ALREADY_RELEASED, "buffer was released");
    }
    if (buffer->inflight) {
        return fail(FSTAGE_BUSY, "buffer is in flight");
    }
    if (!buffer_is_host(*buffer)) {
        return fail(FSTAGE_INVALID_ARGUMENT, "buffer is not host-accessible");
    }
    if (offset > buffer->capacity || size > buffer->capacity - offset) {
        return fail(FSTAGE_INVALID_ARGUMENT, "read exceeds buffer capacity");
    }
    std::memcpy(out, static_cast<const char *>(buffer->host) + offset, static_cast<size_t>(size));
    return FSTAGE_OK;
}

int32_t fstage_transfer_submit(
        uint64_t engine_handle,
        uint64_t src_handle, uint64_t src_offset,
        uint64_t dst_handle, uint64_t dst_offset,
        uint64_t bytes, uint32_t direction) {
    if (bytes == 0) {
        return fail(FSTAGE_INVALID_ARGUMENT, "transfer size must be non-zero");
    }
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        Engine * engine = find_engine(engine_handle);
        Buffer * src = find_buffer(src_handle);
        Buffer * dst = find_buffer(dst_handle);
        if (engine == nullptr) {
            return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
        }
        if (src == nullptr || dst == nullptr) {
            return fail(FSTAGE_INVALID_HANDLE, "transfer buffer handle is unknown");
        }
        if (!engine->usable) {
            return fail(FSTAGE_CUDA_ERROR, "engine is unusable after a prior failure");
        }
        if (engine->inflight) {
            return fail(FSTAGE_BUSY, "engine already has an in-flight transfer");
        }
        if (src->engine != engine_handle || dst->engine != engine_handle) {
            return fail(FSTAGE_INVALID_ARGUMENT, "transfer buffers belong to different engines");
        }
        if (src->state == FSTAGE_BUFFER_RELEASED || dst->state == FSTAGE_BUFFER_RELEASED) {
            return fail(FSTAGE_ALREADY_RELEASED, "transfer buffer was released");
        }
        if (src->inflight || dst->inflight) {
            return fail(FSTAGE_BUSY, "transfer buffer is already in flight");
        }
        if (src_offset > src->capacity || bytes > src->capacity - src_offset ||
            dst_offset > dst->capacity || bytes > dst->capacity - dst_offset) {
            return fail(FSTAGE_INVALID_ARGUMENT, "transfer range exceeds buffer capacity");
        }
        if (direction == FSTAGE_DIRECTION_HOST_TO_DEVICE) {
            if (!buffer_is_host(*src) || dst->memory_type != FSTAGE_MEMORY_DEVICE) {
                return fail(FSTAGE_INVALID_ARGUMENT, "H2D requires host source and device destination");
            }
        } else if (direction == FSTAGE_DIRECTION_DEVICE_TO_HOST) {
            if (src->memory_type != FSTAGE_MEMORY_DEVICE || !buffer_is_host(*dst)) {
                return fail(FSTAGE_INVALID_ARGUMENT, "D2H requires device source and host destination");
            }
        } else {
            return fail(FSTAGE_INVALID_ARGUMENT, "unknown transfer direction");
        }
        ContextGuard guard(*engine);
        if (!guard.ok) {
            return fail(FSTAGE_CUDA_ERROR, "cannot make the staging context current");
        }
        const uint64_t submit_started = now_ns();
        CUresult rc = api().event_record(engine->start, engine->stream);
        if (rc != CUDA_SUCCESS) {
            engine->usable = false;
            return cuda_fail(rc, "cuEventRecord(start)");
        }
        if (direction == FSTAGE_DIRECTION_HOST_TO_DEVICE) {
            const char * source = static_cast<const char *>(src->host) + src_offset;
            rc = api().memcpy_htod_async(
                    dst->device + dst_offset, source, static_cast<size_t>(bytes), engine->stream);
        } else {
            char * destination = static_cast<char *>(dst->host) + dst_offset;
            rc = api().memcpy_dtoh_async(
                    destination, src->device + src_offset, static_cast<size_t>(bytes), engine->stream);
        }
        if (rc != CUDA_SUCCESS) {
            engine->stats.transfer_requests++;
            engine->stats.transfer_failed++;
            engine->usable = false;
            return cuda_fail(rc, "CUDA async copy submission");
        }
        rc = api().event_record(engine->end, engine->stream);
        if (rc != CUDA_SUCCESS) {
            engine->stats.transfer_requests++;
            engine->stats.transfer_failed++;
            engine->usable = false;
            return cuda_fail(rc, "cuEventRecord(end)");
        }
        engine->stats.transfer_requests++;
        engine->stats.transfer_bytes += bytes;
        engine->last_submit_ns = now_ns() - submit_started;
        engine->stats.submit_ns += engine->last_submit_ns;
        engine->inflight = true;
        engine->src = src_handle;
        engine->dst = dst_handle;
        engine->bytes = bytes;
        engine->direction = direction;
        src->inflight = true;
        dst->inflight = true;
        update_buffer_state(*src);
        update_buffer_state(*dst);
        return FSTAGE_OK;
    } catch (const std::exception & error) {
        return fail(FSTAGE_INTERNAL, error.what());
    } catch (...) {
        return fail(FSTAGE_INTERNAL, "unknown native exception");
    }
}

int32_t fstage_transfer_query(uint64_t engine_handle, fstage_transfer_report * out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Engine * engine = find_engine(engine_handle);
    if (engine == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
    }
    if (out == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out is null");
    }
    if (!engine->inflight) {
        return fail(FSTAGE_NO_OPERATION, "no transfer is in flight");
    }
    ContextGuard guard(*engine);
    if (!guard.ok) {
        return fail(FSTAGE_CUDA_ERROR, "cannot make the staging context current");
    }
    CUresult rc = api().event_query(engine->end);
    if (rc == CUDA_ERROR_NOT_READY) {
        std::memset(out, 0, sizeof(*out));
        out->bytes = engine->bytes;
        out->submit_ns = engine->last_submit_ns;
        out->completed = 0;
        out->direction = engine->direction;
        return FSTAGE_OK;
    }
    if (rc != CUDA_SUCCESS) {
        engine->inflight = false;
        engine->stats.transfer_failed++;
        Buffer * src = find_buffer(engine->src);
        Buffer * dst = find_buffer(engine->dst);
        if (src != nullptr) { src->inflight = false; update_buffer_state(*src); }
        if (dst != nullptr) { dst->inflight = false; update_buffer_state(*dst); }
        engine->usable = false;
        return cuda_fail(rc, "cuEventQuery");
    }
    float elapsed_ms = 0.0f;
    rc = api().event_elapsed(&elapsed_ms, engine->start, engine->end);
    if (rc != CUDA_SUCCESS) {
        engine->usable = false;
        return cuda_fail(rc, "cuEventElapsedTime");
    }
    const uint64_t bytes = engine->bytes;
    const uint64_t submit = engine->last_submit_ns;
    const uint64_t direction = engine->direction;
    engine->stats.last_duration_ns =
            static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1e6);
    engine->stats.total_duration_ns += engine->stats.last_duration_ns;
    clear_inflight_buffers(*engine);
    complete_transfer(*engine, elapsed_ms);
    std::memset(out, 0, sizeof(*out));
    out->bytes = bytes;
    out->submit_ns = submit;
    out->duration_ns = engine->stats.last_duration_ns;
    out->completed = 1;
    out->direction = direction;
    return FSTAGE_OK;
}

int32_t fstage_transfer_wait(uint64_t engine_handle, fstage_transfer_report * out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Engine * engine = find_engine(engine_handle);
    if (engine == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
    }
    if (out == nullptr) {
        return fail(FSTAGE_INVALID_ARGUMENT, "out is null");
    }
    if (!engine->inflight) {
        return fail(FSTAGE_NO_OPERATION, "no transfer is in flight");
    }
    ContextGuard guard(*engine);
    if (!guard.ok) {
        return fail(FSTAGE_CUDA_ERROR, "cannot make the staging context current");
    }
    const uint64_t wait_started = now_ns();
    CUresult rc = api().event_synchronize(engine->end);
    const uint64_t wait_ns = now_ns() - wait_started;
    if (rc != CUDA_SUCCESS) {
        engine->inflight = false;
        engine->stats.transfer_failed++;
        Buffer * src = find_buffer(engine->src);
        Buffer * dst = find_buffer(engine->dst);
        if (src != nullptr) { src->inflight = false; update_buffer_state(*src); }
        if (dst != nullptr) { dst->inflight = false; update_buffer_state(*dst); }
        engine->usable = false;
        return cuda_fail(rc, "cuEventSynchronize");
    }
    float elapsed_ms = 0.0f;
    rc = api().event_elapsed(&elapsed_ms, engine->start, engine->end);
    if (rc != CUDA_SUCCESS) {
        engine->usable = false;
        return cuda_fail(rc, "cuEventElapsedTime");
    }
    const uint64_t bytes = engine->bytes;
    const uint64_t submit = engine->last_submit_ns;
    const uint64_t direction = engine->direction;
    engine->stats.wait_ns += wait_ns;
    engine->stats.last_duration_ns =
            static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1e6);
    engine->stats.total_duration_ns += engine->stats.last_duration_ns;
    clear_inflight_buffers(*engine);
    complete_transfer(*engine, elapsed_ms);
    std::memset(out, 0, sizeof(*out));
    out->bytes = bytes;
    out->submit_ns = submit;
    out->wait_ns = wait_ns;
    out->duration_ns = engine->stats.last_duration_ns;
    out->completed = 1;
    out->direction = direction;
    return FSTAGE_OK;
}

int32_t fstage_transfer_cancel(uint64_t engine_handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Engine * engine = find_engine(engine_handle);
    if (engine == nullptr) {
        return fail(FSTAGE_INVALID_HANDLE, "engine handle is not live");
    }
    engine->stats.cancellation_requests++;
    if (engine->inflight) {
        return fail(FSTAGE_BUSY, "an in-flight CUDA copy cannot be cancelled safely; wait for completion");
    }
    return fail(FSTAGE_NO_OPERATION, "no transfer is in flight");
}

int32_t fstage_last_error(char * out, uint64_t size) {
    if (out == nullptr || size == 0) {
        return FSTAGE_INVALID_ARGUMENT;
    }
    std::snprintf(out, static_cast<size_t>(size), "%s", g_last_error.c_str());
    return FSTAGE_OK;
}

} // extern "C"
