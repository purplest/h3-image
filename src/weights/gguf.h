/* GGUF parsing and CPU dequantization reference (SPEC part 2). */
#ifndef H3I_WEIGHTS_GGUF_H
#define H3I_WEIGHTS_GGUF_H

#include <stddef.h>
#include <stdint.h>

/* ggml tensor type ids, used everywhere internally (SPEC §2.1). */
enum {
    H3I_TYPE_F32  = 0,
    H3I_TYPE_F16  = 1,
    H3I_TYPE_Q8_0 = 8,
    H3I_TYPE_Q2_K = 10,
    H3I_TYPE_Q3_K = 11,
    H3I_TYPE_Q4_K = 12,
    H3I_TYPE_Q5_K = 13,
    H3I_TYPE_Q6_K = 14,
    H3I_TYPE_BF16 = 30,
};

typedef struct {
    int block_values;   /* values per block (0 = unsupported type) */
    int block_bytes;    /* bytes per block */
    int chunks;         /* GPU chunks per block */
    int chunk_values;   /* values per GPU chunk */
} h3i_type_info;

h3i_type_info h3i_type_info_get(int type);
int h3i_type_supported(int type);

typedef struct {
    char *name;
    int type;           /* ggml id */
    int supported;
    int ndim;           /* 1..4 */
    uint64_t ne[4];     /* ggml order: ne[0] is the contiguous axis */
    uint64_t elements;
    uint64_t bytes;     /* 0 when unsupported */
    uint64_t offset;    /* absolute file offset */
} h3i_gguf_tensor;

typedef struct {
    char *path;
    h3i_gguf_tensor *tensors;
    size_t count;
    uint32_t alignment;
    uint64_t file_size;
} h3i_gguf;

/* Parse the header only (prefix strategy, SPEC §2.1); tensor data is read on
 * demand. Returns 1 on success, 0 on failure with error filled. */
int h3i_gguf_open(const char *path, h3i_gguf *out, char *error, size_t error_size);
void h3i_gguf_close(h3i_gguf *g);

/* IEEE binary16 (2 LE bytes worth, as u16) → F32, exact. */
float h3i_half_to_f32(uint16_t h);

/* CPU dequantization, bit-exact with ggml (SPEC §5). fma where written. */
void h3i_dequant_block(int type, const uint8_t *block, float *out);
void h3i_dequant_row(int type, const uint8_t *raw, uint64_t elements, float *out);

#endif /* H3I_WEIGHTS_GGUF_H */
