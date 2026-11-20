/* Interim compatibility API from h3.c, implemented over the spec weight
 * source (weights/source.c). The donor model files (dit.c, text_encoder.c,
 * vision.c, vae_encoder.c, dit_schedule.c) use it until Phases 6/7/10 rewire
 * them to h3i_source directly, at which point this file is removed. */
#ifndef H3I_WEIGHTS_H
#define H3I_WEIGHTS_H

#include "gpu/gpu.h"
#include "weights/gguf.h"
#include "weights/source.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3i_weight_store h3i_weight_store;

/* Open every .safetensors/.gguf file in a component directory (headers only,
 * tensor payloads read on demand). */
h3i_weight_store *h3i_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3i_weight_store_free(h3i_weight_store *store);

/* First match in insertion order, then the one-way prefix aliases. */
const h3i_tensor_rec *h3i_weight_find(const h3i_weight_store *store,
                                    const char *name);

/* Validate an exact shape, then load as BF16/F32 on the GPU (dequantizing
 * quantized storage through the caller's open command buffer). */
h3i_gpu_tensor *h3i_weight_load_bf16(const h3i_weight_store *store, h3i_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3i_gpu_tensor *h3i_weight_load_f32(const h3i_weight_store *store, h3i_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
