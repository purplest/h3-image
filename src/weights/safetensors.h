/* Safetensors parsing (SPEC §3). */
#ifndef H3I_WEIGHTS_SAFETENSORS_H
#define H3I_WEIGHTS_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *name;
    int type;           /* internal ggml id (F32=0, F16=1, BF16=30); -1 other */
    int supported;
    int ndim;           /* 0..8, shape kept as written (outermost first) */
    uint64_t shape[8];
    uint64_t elements;
    uint64_t bytes;
    uint64_t offset;    /* absolute file offset */
} h3i_st_tensor;

typedef struct {
    char *path;
    h3i_st_tensor *tensors;
    size_t count;
    uint64_t file_size;
} h3i_safetensors;

int h3i_safetensors_open(const char *path, h3i_safetensors *out,
                         char *error, size_t error_size);
void h3i_safetensors_close(h3i_safetensors *st);

#endif /* H3I_WEIGHTS_SAFETENSORS_H */
