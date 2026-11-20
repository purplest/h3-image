/* Shared public types for the h3-image engine.
 * Trimmed from h3.c's h3.h (commit 8974cc0): only the device/component info
 * structs used by the GPU probe and the weights inventory survive the rebuild.
 * The video public API (context/params/generate entry points) is intentionally dropped. */
#ifndef H3I_H
#define H3I_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3I_VERSION "0.1.0"

typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;
    int metal4;
    int unified_memory;
} h3i_device_info;

typedef struct {
    uint64_t bytes;
    uint64_t tensor_bytes;
    size_t files;
    size_t tensors;
} h3i_component_info;

#ifdef __cplusplus
}
#endif
#endif
