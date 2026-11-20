#ifndef H3I_SAFETENSORS_H
#define H3I_SAFETENSORS_H

#include "h3i.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3I_DTYPE_UNKNOWN = 0,
    H3I_DTYPE_BOOL,
    H3I_DTYPE_I8,
    H3I_DTYPE_U8,
    H3I_DTYPE_I16,
    H3I_DTYPE_U16,
    H3I_DTYPE_F16,
    H3I_DTYPE_BF16,
    H3I_DTYPE_I32,
    H3I_DTYPE_U32,
    H3I_DTYPE_F32,
    H3I_DTYPE_I64,
    H3I_DTYPE_U64,
    H3I_DTYPE_F64
} h3i_dtype;

typedef struct {
    char *name;
    h3i_dtype dtype;
    int ndim;
    uint64_t shape[8];
    uint64_t data_begin;
    uint64_t data_end;
    uint64_t file_offset;
} h3i_st_tensor;

typedef struct {
    char *path;
    uint64_t file_size;
    uint64_t header_size;
    h3i_st_tensor *tensors;
    size_t tensor_count;
} h3i_st_header;

int h3i_st_read_header(const char *path, h3i_st_header *header,
                      char *error, size_t error_size);
void h3i_st_free_header(h3i_st_header *header);
const h3i_st_tensor *h3i_st_find(const h3i_st_header *header, const char *name);
uint64_t h3i_st_tensor_elements(const h3i_st_tensor *tensor);
int h3i_st_read_data(const h3i_st_header *header, const h3i_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size);
size_t h3i_dtype_size(h3i_dtype dtype);
const char *h3i_dtype_name(h3i_dtype dtype);

int h3i_st_inventory_dir(const char *directory, h3i_component_info *info,
                        char *error, size_t error_size);

#endif
