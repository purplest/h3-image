#ifndef H3I_WEIGHTS_H
#define H3I_WEIGHTS_H

#include "gpu/gpu.h"
#include "weights/safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3i_weight_store h3i_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3i_weight_store *h3i_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3i_weight_store_free(h3i_weight_store *store);
size_t h3i_weight_store_shards(const h3i_weight_store *store);

const h3i_st_tensor *h3i_weight_find(const h3i_weight_store *store,
                                   const char *name,
                                   const h3i_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3i_gpu_tensor *h3i_weight_load_bf16(const h3i_weight_store *store, h3i_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3i_gpu_tensor *h3i_weight_load_f32(const h3i_weight_store *store, h3i_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
