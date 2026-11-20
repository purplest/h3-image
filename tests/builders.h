/* Synthetic weight-file builders for unit tests (no weights needed). */
#ifndef H3I_TESTS_BUILDERS_H
#define H3I_TESTS_BUILDERS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static inline void put_u32(FILE *f, uint32_t v)
{
    uint8_t b[4] = {(uint8_t)(v & 255), (uint8_t)((v >> 8) & 255),
                    (uint8_t)((v >> 16) & 255), (uint8_t)((v >> 24) & 255)};
    fwrite(b, 1, 4, f);
}

static inline void put_u64(FILE *f, uint64_t v)
{
    put_u32(f, (uint32_t)v);
    put_u32(f, (uint32_t)(v >> 32));
}

static inline void put_str(FILE *f, const char *s)
{
    put_u64(f, strlen(s));
    fwrite(s, 1, strlen(s), f);
}

typedef struct {
    const char *name;
    uint32_t ndim;
    uint64_t ne[8];       /* ggml order (8 slots so bad-ndim tests stay in bounds) */
    uint32_t type;
    uint64_t offset;      /* relative to data start */
    const void *data;
    size_t data_bytes;    /* may be shorter than the type implies (error tests) */
} tb_tensor;

/* Minimal GGUF: one metadata entry (general.alignment = alignment).
 * Returns the absolute data offset, or -1 on failure. */
static inline long build_gguf(const char *path, uint32_t version, uint32_t alignment,
                       const tb_tensor *ts, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return -1;
    }
    fwrite("GGUF", 1, 4, f);
    put_u32(f, version);
    put_u64(f, n);
    put_u64(f, 1);
    put_str(f, "general.alignment");
    put_u32(f, 4); /* UINT32 */
    put_u32(f, alignment);
    for (size_t i = 0; i < n; i++) {
        put_str(f, ts[i].name);
        put_u32(f, ts[i].ndim);
        for (uint32_t d = 0; d < ts[i].ndim; d++) {
            put_u64(f, ts[i].ne[d]);
        }
        put_u32(f, ts[i].type);
        put_u64(f, ts[i].offset);
    }
    long pos = ftell(f);
    long pad = (long)((alignment - (uint32_t)pos % alignment) % alignment);
    for (long i = 0; i < pad; i++) {
        fputc(0, f);
    }
    long data_off = ftell(f);
    for (size_t i = 0; i < n; i++) {
        if (ts[i].data != NULL && ts[i].data_bytes > 0) {
            fseek(f, data_off + (long)ts[i].offset, SEEK_SET);
            fwrite(ts[i].data, 1, ts[i].data_bytes, f);
        }
    }
    fclose(f);
    return data_off;
}

static inline void build_st(const char *path, const char *json, const void *data, size_t nbytes)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return;
    }
    put_u64(f, strlen(json));
    fwrite(json, 1, strlen(json), f);
    if (data != NULL && nbytes > 0) {
        fwrite(data, 1, nbytes, f);
    }
    fclose(f);
}

static inline void test_mkdir(const char *path)
{
    mkdir(path, 0755);
}

#endif /* H3I_TESTS_BUILDERS_H */
