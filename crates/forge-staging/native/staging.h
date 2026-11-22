#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FSTAGE_API __attribute__((visibility("default")))

typedef enum fstage_memory_type {
    FSTAGE_MEMORY_PAGEABLE_HOST = 1,
    FSTAGE_MEMORY_PINNED_HOST = 2,
    FSTAGE_MEMORY_DEVICE = 3,
} fstage_memory_type;

typedef enum fstage_buffer_state {
    FSTAGE_BUFFER_EMPTY = 0,
    FSTAGE_BUFFER_ALLOCATED = 1,
    FSTAGE_BUFFER_IN_FLIGHT = 2,
    FSTAGE_BUFFER_RELEASED = 3,
    FSTAGE_BUFFER_FAILED = 4,
} fstage_buffer_state;

typedef enum fstage_direction {
    FSTAGE_DIRECTION_HOST_TO_DEVICE = 1,
    FSTAGE_DIRECTION_DEVICE_TO_HOST = 2,
} fstage_direction;

typedef enum fstage_error {
    FSTAGE_OK = 0,
    FSTAGE_INVALID_ARGUMENT = 1,
    FSTAGE_INVALID_HANDLE = 2,
    FSTAGE_UNSUPPORTED = 3,
    FSTAGE_OUT_OF_MEMORY = 4,
    FSTAGE_BUSY = 5,
    FSTAGE_NO_OPERATION = 6,
    FSTAGE_CUDA_ERROR = 7,
    FSTAGE_INTERNAL = 8,
    FSTAGE_ALREADY_RELEASED = 9,
    FSTAGE_IO_ERROR = 10,
} fstage_error;

typedef struct fstage_engine_info {
    uint32_t device_index;
    char device_name[128];
    int32_t driver_version;
    int32_t runtime_version;
} fstage_engine_info;

typedef struct fstage_buffer_info {
    uint64_t handle;
    uint32_t memory_type;
    uint32_t state;
    uint64_t capacity;
    uint64_t allocated;
    uint8_t in_flight;
    uint8_t host_pinned;
    uint32_t device_index;
    uint64_t device_ptr;
} fstage_buffer_info;

typedef struct fstage_stats {
    uint64_t allocation_requests;
    uint64_t allocation_successes;
    uint64_t allocation_failures;
    uint64_t released;
    uint64_t current_bytes;
    uint64_t high_water_bytes;
    uint64_t transfer_requests;
    uint64_t transfer_completed;
    uint64_t transfer_failed;
    uint64_t transfer_bytes;
    uint64_t submit_ns;
    uint64_t wait_ns;
    uint64_t last_duration_ns;
    uint64_t total_duration_ns;
    uint64_t in_flight_operations;
    uint64_t cancellation_requests;
    uint64_t read_requests;
    uint64_t read_bytes;
    uint64_t read_ns;
} fstage_stats;

typedef struct fstage_transfer_report {
    uint64_t bytes;
    uint64_t submit_ns;
    uint64_t wait_ns;
    uint64_t duration_ns;
    uint32_t completed;
    uint32_t direction;
} fstage_transfer_report;

FSTAGE_API int32_t fstage_engine_create(uint32_t device_index, uint64_t * out_engine);
FSTAGE_API int32_t fstage_engine_destroy(uint64_t engine);
FSTAGE_API int32_t fstage_engine_get_info(uint64_t engine, fstage_engine_info * out);
FSTAGE_API int32_t fstage_engine_stats(uint64_t engine, fstage_stats * out);

FSTAGE_API int32_t fstage_buffer_create(
        uint64_t engine, uint32_t memory_type, uint64_t bytes, uint64_t * out_buffer);
FSTAGE_API int32_t fstage_buffer_release(uint64_t buffer);
FSTAGE_API int32_t fstage_buffer_get_info(uint64_t buffer, fstage_buffer_info * out);
FSTAGE_API int32_t fstage_buffer_write(
        uint64_t buffer, uint64_t offset, const void * data, uint64_t size);
FSTAGE_API int32_t fstage_buffer_read(
        uint64_t buffer, uint64_t offset, void * out, uint64_t size);
/* Read a file range directly into a host-accessible staging buffer using
 * pread(2). Retries EINTR and rejects short reads. The caller owns the fd. */
FSTAGE_API int32_t fstage_buffer_read_fd(
        uint64_t buffer, uint64_t buffer_offset,
        int32_t fd, uint64_t file_offset, uint64_t size);

FSTAGE_API int32_t fstage_transfer_submit(
        uint64_t engine,
        uint64_t src, uint64_t src_offset,
        uint64_t dst, uint64_t dst_offset,
        uint64_t bytes, uint32_t direction);
/* Host-to-device copy into an externally owned CUDA device pointer. The
 * pointer must belong to the same primary CUDA context/device. This is the
 * graph-integration path used to populate independently allocated model
 * tensors; it does not take ownership of the destination. */
FSTAGE_API int32_t fstage_transfer_submit_to_ptr(
        uint64_t engine, uint64_t src, uint64_t src_offset,
        uint64_t device_ptr, uint64_t bytes);
/* Submit one H2D batch into externally owned CUDA device pointers on the
 * engine's private stream. All copies are enqueued before returning; the
 * caller must use query/wait before reusing the source buffer. */
FSTAGE_API int32_t fstage_transfer_submit_batch_to_ptr(
        uint64_t engine, uint64_t src,
        const uint64_t * src_offsets, const uint64_t * device_ptrs,
        const uint64_t * sizes, uint64_t count);
/* Wait until the first operation of the current batch has started on the
 * device. Does not wait for completion. */
FSTAGE_API int32_t fstage_transfer_wait_started(uint64_t engine);
FSTAGE_API int32_t fstage_transfer_query(
        uint64_t engine, fstage_transfer_report * out);
FSTAGE_API int32_t fstage_transfer_wait(
        uint64_t engine, fstage_transfer_report * out);
FSTAGE_API int32_t fstage_transfer_cancel(uint64_t engine);

FSTAGE_API int32_t fstage_last_error(char * out, uint64_t size);

#ifdef __cplusplus
}
#endif
