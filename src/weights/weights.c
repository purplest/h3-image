/* Interim h3.c weight-store compatibility layer over h3i_source (SPEC §2.4).
 * See weights.h: removed once the donor model files use the source directly. */
#include "weights/weights.h"

#include "app/error.h"
#include "weights/gguf.h"

#include <stdlib.h>

struct h3i_weight_store {
    h3i_source source;
};

h3i_weight_store *h3i_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    h3i_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        h3i_fail(error, error_size, "out of memory opening weights");
        return NULL;
    }
    if (!h3i_source_add(&store->source, directory, error, error_size)) {
        free(store);
        return NULL;
    }
    return store;
}

void h3i_weight_store_free(h3i_weight_store *store) {
    if (!store) return;
    h3i_source_free(&store->source);
    free(store);
}

const h3i_tensor_rec *h3i_weight_find(const h3i_weight_store *store,
                                    const char *name) {
    if (!store) return NULL;
    return h3i_source_find(&store->source, name);
}

static h3i_gpu_tensor *load_tensor(const h3i_weight_store *store, h3i_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape, int dtype,
                                   char *error, size_t error_size) {
    const h3i_tensor_rec *tensor = h3i_weight_find(store, name);
    if (!tensor) {
        h3i_fail(error, error_size, "missing tensor %s", name);
        return NULL;
    }
    if (!tensor->supported) {
        h3i_fail(error, error_size, "tensor %s has an unsupported stored type",
                 name);
        return NULL;
    }
    if (tensor->ndim != ndim) {
        h3i_fail(error, error_size,
                 "weight %s has rank %d, expected %d", name, tensor->ndim,
                 ndim);
        return NULL;
    }
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            h3i_fail(error, error_size,
                     "weight %s shape mismatch at dimension %d", name,
                     dimension);
            return NULL;
        }
    }
    return h3i_source_load(&store->source, gpu, name, dtype, 0, error,
                           error_size);
}

h3i_gpu_tensor *h3i_weight_load_bf16(const h3i_weight_store *store, h3i_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3I_TYPE_BF16, error,
                       error_size);
}

h3i_gpu_tensor *h3i_weight_load_f32(const h3i_weight_store *store, h3i_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3I_TYPE_F32, error,
                       error_size);
}
