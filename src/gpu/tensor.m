/* Shared-storage GPU tensors and file I/O (SPEC §6.3.1, §6.3.3). */
#import "gpu_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "app/error.h"

@implementation H3ITensor
@end

static h3i_gpu_tensor *wrap_tensor(H3IGPU *g, id<MTLBuffer> buffer, h3i_dtype dt,
                                   size_t elements, char *error, size_t es)
{
    if (buffer == nil) {
        h3i_fail(error, es, "cannot allocate %zu-byte Metal buffer",
                 elements * dtype_size(dt));
        return NULL;
    }
    H3ITensor *t = [H3ITensor new];
    t.buffer = buffer;
    t.dtype = dt;
    t.elements = elements;
    t.bytes = elements * dtype_size(dt);
    t.owner = g;
    [g noteAlloc:t.bytes];
    /* Zero so padding rows are deterministic (SPEC §5.6 relies on this). */
    memset(buffer.contents, 0, t.bytes);
    h3i_gpu_tensor *out = malloc(sizeof *out);
    if (out == NULL) {
        h3i_fail(error, es, "out of memory");
        return NULL;
    }
    out->obj = (void *)CFBridgingRetain(t);
    return out;
}

h3i_gpu_tensor *h3i_gpu_tensor_new(h3i_gpu *g, h3i_dtype dt, size_t elements)
{
    H3IGPU *gg = gpu_obj(g);
    size_t bytes = elements * dtype_size(dt);
    id<MTLBuffer> buffer = [gg.device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
    return wrap_tensor(gg, buffer, dt, elements, NULL, 0);
}

h3i_gpu_tensor *h3i_gpu_tensor_new_f32(h3i_gpu *g, size_t elements)
{
    return h3i_gpu_tensor_new(g, H3I_DTYPE_F32, elements);
}

h3i_gpu_tensor *h3i_gpu_tensor_new_bf16(h3i_gpu *g, size_t elements)
{
    return h3i_gpu_tensor_new(g, H3I_DTYPE_BF16, elements);
}

h3i_gpu_tensor *h3i_gpu_tensor_new_i8(h3i_gpu *g, size_t elements)
{
    return h3i_gpu_tensor_new(g, H3I_DTYPE_I8, elements);
}

h3i_gpu_tensor *h3i_gpu_tensor_new_u32(h3i_gpu *g, size_t elements)
{
    return h3i_gpu_tensor_new(g, H3I_DTYPE_U32, elements);
}

static h3i_gpu_tensor *from_host(h3i_gpu *g, h3i_dtype dt, const void *host, size_t n)
{
    h3i_gpu_tensor *t = h3i_gpu_tensor_new(g, dt, n);
    if (t != NULL && host != NULL && n > 0) {
        memcpy(tensor_obj(t).buffer.contents, host, n * dtype_size(dt));
    }
    return t;
}

h3i_gpu_tensor *h3i_gpu_tensor_from_f32(h3i_gpu *g, const float *host, size_t elements)
{
    return from_host(g, H3I_DTYPE_F32, host, elements);
}

h3i_gpu_tensor *h3i_gpu_tensor_from_bf16(h3i_gpu *g, const uint16_t *host, size_t elements)
{
    return from_host(g, H3I_DTYPE_BF16, host, elements);
}

h3i_gpu_tensor *h3i_gpu_tensor_from_u32(h3i_gpu *g, const uint32_t *host, size_t elements)
{
    return from_host(g, H3I_DTYPE_U32, host, elements);
}

void h3i_gpu_tensor_free(h3i_gpu_tensor *t)
{
    if (t == NULL) {
        return;
    }
    H3ITensor *tt = tensor_obj(t);
    [tt.owner noteFree:tt.bytes];
    CFBridgingRelease(t->obj);
    free(t);
}

void *h3i_gpu_tensor_contents(h3i_gpu_tensor *t)
{
    return tensor_obj(t).buffer.contents;
}

size_t h3i_gpu_tensor_elements(const h3i_gpu_tensor *t)
{
    return tensor_obj((h3i_gpu_tensor *)t).elements;
}

size_t h3i_gpu_tensor_bytes(const h3i_gpu_tensor *t)
{
    return tensor_obj((h3i_gpu_tensor *)t).bytes;
}

h3i_dtype h3i_gpu_tensor_dtype(const h3i_gpu_tensor *t)
{
    return tensor_obj((h3i_gpu_tensor *)t).dtype;
}

int h3i_gpu_read_f32(h3i_gpu_tensor *t, float *out, size_t elements)
{
    H3ITensor *tt = tensor_obj(t);
    if (tt.dtype != H3I_DTYPE_F32 || elements > tt.elements) {
        return 0;
    }
    memcpy(out, tt.buffer.contents, elements * 4);
    return 1;
}

int h3i_gpu_read_bf16(h3i_gpu_tensor *t, uint16_t *out, size_t elements)
{
    H3ITensor *tt = tensor_obj(t);
    if (tt.dtype != H3I_DTYPE_BF16 || elements > tt.elements) {
        return 0;
    }
    memcpy(out, tt.buffer.contents, elements * 2);
    return 1;
}

/* pread loop into a buffer; shared by load_* and read_file_bytes. */
static int pread_into(void *dst, uint64_t dst_offset, const char *path,
                      uint64_t file_offset, uint64_t bytes, int uncached,
                      char *error, size_t es)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        h3i_fail(error, es, "cannot open weights %s", path);
        return 0;
    }
    if (uncached) {
        fcntl(fd, F_NOCACHE, 1);
    }
    uint8_t *out = (uint8_t *)dst + dst_offset;
    uint64_t got = 0;
    while (got < bytes) {
        ssize_t n = pread(fd, out + got, (size_t)(bytes - got),
                          (off_t)(file_offset + got));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (n == 0) {
            break;
        }
        got += (uint64_t)n;
    }
    close(fd);
    if (got != bytes) {
        h3i_fail(error, es, "cannot read %llu bytes from %s: unexpected end of file",
                 (unsigned long long)bytes, path);
        return 0;
    }
    return 1;
}

static h3i_gpu_tensor *load_typed(h3i_gpu *g, h3i_dtype dt, const char *path,
                                  uint64_t offset, size_t elements,
                                  char *error, size_t es)
{
    h3i_gpu_tensor *t = h3i_gpu_tensor_new(g, dt, elements);
    if (t == NULL) {
        h3i_fail(error, es, "cannot allocate %zu-byte Metal buffer",
                 elements * dtype_size(dt));
        return NULL;
    }
    if (!pread_into(tensor_obj(t).buffer.contents, 0, path, offset,
                    elements * dtype_size(dt), 1, error, es)) {
        h3i_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

h3i_gpu_tensor *h3i_gpu_load_bf16(h3i_gpu *g, const char *path, uint64_t offset,
                                  size_t elements, char *error, size_t error_size)
{
    return load_typed(g, H3I_DTYPE_BF16, path, offset, elements, error, error_size);
}

h3i_gpu_tensor *h3i_gpu_load_f32(h3i_gpu *g, const char *path, uint64_t offset,
                                 size_t elements, char *error, size_t error_size)
{
    return load_typed(g, H3I_DTYPE_F32, path, offset, elements, error, error_size);
}

int h3i_gpu_read_file_bytes(h3i_gpu *g, h3i_gpu_tensor *t, uint64_t tensor_offset,
                            const char *path, uint64_t file_offset, uint64_t bytes,
                            int uncached, char *error, size_t error_size)
{
    (void)g;
    H3ITensor *tt = tensor_obj(t);
    if (tensor_offset + bytes > tt.bytes) {
        h3i_fail(error, error_size, "read_file_bytes range exceeds tensor");
        return 0;
    }
    return pread_into(tt.buffer.contents, tensor_offset, path, file_offset, bytes,
                      uncached, error, error_size);
}
