/* Weight source: one lookup over many files of either format (SPEC §4). */
#ifndef H3I_WEIGHTS_SOURCE_H
#define H3I_WEIGHTS_SOURCE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *name;
    int type;           /* ggml id; -1 for unsupported safetensors dtypes */
    int supported;
    int ndim;
    uint64_t shape[8];  /* PyTorch order: outermost first, [out, in] matrices */
    uint64_t elements;
    uint64_t bytes;
    char *path;
    uint64_t offset;    /* absolute file offset */
} h3i_tensor_rec;

typedef struct {
    h3i_tensor_rec *recs;
    size_t count;
    size_t cap;
} h3i_source;

/* Add a .gguf/.safetensors file, or a directory of them (sorted by strcmp,
 * non-recursive listing, hidden files included). May be called repeatedly
 * on the same source; earlier insertions shadow later duplicates. */
int h3i_source_add(h3i_source *s, const char *path, char *error, size_t error_size);

/* Exact name match, first in insertion order wins; then one-way prefix
 * aliases (model.language_model.→model., model.visual.→visual.). */
const h3i_tensor_rec *h3i_source_find(const h3i_source *s, const char *name);

/* CPU read + dequantize to a malloc'd row-major F32 array (caller frees).
 * expected_elements != 0 enforces an exact element count. */
float *h3i_source_read_f32(const h3i_source *s, const char *name,
                           uint64_t expected_elements, char *error, size_t error_size);

/* pread exactly t->bytes from t->offset into dst+dst_offset. uncached sets
 * F_NOCACHE on Darwin. Opens/closes the file per call; thread-safe. */
int h3i_source_read_raw(const h3i_tensor_rec *t, void *dst, uint64_t dst_offset,
                        int uncached, char *error, size_t error_size);

/* SPEC §4.4: load as an F32 or BF16 (dtype = H3I_TYPE_F32/H3I_TYPE_BF16) GPU
 * tensor. Stored type equal to requested: raw byte copy (no command buffer
 * needed). Otherwise: stage raw bytes and encode h3i_gpu_convert_raw into
 * the caller's OPEN command buffer (h3i_gpu_begin); results are visible
 * after the caller's h3i_gpu_submit. expected_elements != 0 enforces count.
 * Implemented only where the Metal GPU runtime exists (__APPLE__). */
typedef struct h3i_gpu h3i_gpu;
typedef struct h3i_gpu_tensor h3i_gpu_tensor;
h3i_gpu_tensor *h3i_source_load(const h3i_source *s, h3i_gpu *gpu,
                                const char *name, int dtype,
                                uint64_t expected_elements,
                                char *error, size_t error_size);

void h3i_source_free(h3i_source *s);

#endif /* H3I_WEIGHTS_SOURCE_H */
