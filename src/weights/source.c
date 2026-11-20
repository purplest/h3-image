/* Weight source abstraction (SPEC §4). */
#include "weights/source.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "app/error.h"
#include "weights/gguf.h"
#include "weights/safetensors.h"

static int ends_with(const char *s, const char *suffix)
{
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static int is_weight_name(const char *name)
{
    return ends_with(name, ".gguf") || ends_with(name, ".safetensors");
}

static h3i_tensor_rec *source_grow(h3i_source *s)
{
    if (s->count == s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 64;
        h3i_tensor_rec *nr = realloc(s->recs, nc * sizeof *nr);
        if (nr == NULL) {
            return NULL;
        }
        s->recs = nr;
        s->cap = nc;
    }
    h3i_tensor_rec *t = &s->recs[s->count++];
    memset(t, 0, sizeof *t);
    return t;
}

static int add_gguf(h3i_source *s, const char *path, char *error, size_t es)
{
    h3i_gguf g;
    if (!h3i_gguf_open(path, &g, error, es)) {
        return 0;
    }
    for (size_t i = 0; i < g.count; i++) {
        const h3i_gguf_tensor *gt = &g.tensors[i];
        h3i_tensor_rec *t = source_grow(s);
        if (t == NULL) {
            h3i_gguf_close(&g);
            h3i_fail(error, es, "%s: out of memory", path);
            return 0;
        }
        t->name = strdup(gt->name);
        t->path = strdup(path);
        if (t->name == NULL || t->path == NULL) {
            h3i_gguf_close(&g);
            h3i_fail(error, es, "%s: out of memory", path);
            return 0;
        }
        t->type = gt->type;
        t->supported = gt->supported;
        t->ndim = gt->ndim;
        /* GGUF dims are ggml order (innermost first): reverse to PyTorch. */
        for (int d = 0; d < gt->ndim; d++) {
            t->shape[d] = gt->ne[gt->ndim - 1 - d];
        }
        t->elements = gt->elements;
        t->bytes = gt->bytes;
        t->offset = gt->offset;
    }
    h3i_gguf_close(&g);
    return 1;
}

static int add_safetensors(h3i_source *s, const char *path, char *error, size_t es)
{
    h3i_safetensors st;
    if (!h3i_safetensors_open(path, &st, error, es)) {
        return 0;
    }
    for (size_t i = 0; i < st.count; i++) {
        const h3i_st_tensor *tt = &st.tensors[i];
        h3i_tensor_rec *t = source_grow(s);
        if (t == NULL) {
            h3i_safetensors_close(&st);
            h3i_fail(error, es, "%s: out of memory", path);
            return 0;
        }
        t->name = strdup(tt->name);
        t->path = strdup(path);
        if (t->name == NULL || t->path == NULL) {
            h3i_safetensors_close(&st);
            h3i_fail(error, es, "%s: out of memory", path);
            return 0;
        }
        t->type = tt->type;
        t->supported = tt->supported;
        t->ndim = tt->ndim;
        for (int d = 0; d < tt->ndim; d++) {
            t->shape[d] = tt->shape[d];
        }
        t->elements = tt->elements;
        t->bytes = tt->bytes;
        t->offset = tt->offset;
    }
    h3i_safetensors_close(&st);
    return 1;
}

static int cmp_names(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

int h3i_source_add(h3i_source *s, const char *path, char *error, size_t error_size)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        h3i_fail(error, error_size, "cannot open weights %s", path);
        return 0;
    }
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (d == NULL) {
            h3i_fail(error, error_size, "cannot open weights %s", path);
            return 0;
        }
        char **names = NULL;
        size_t n = 0, cap = 0;
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (!is_weight_name(e->d_name)) {
                continue;
            }
            if (n == cap) {
                cap = cap ? cap * 2 : 16;
                char **nn = realloc(names, cap * sizeof *nn);
                if (nn == NULL) {
                    closedir(d);
                    for (size_t i = 0; i < n; i++) {
                        free(names[i]);
                    }
                    free(names);
                    h3i_fail(error, error_size, "%s: out of memory", path);
                    return 0;
                }
                names = nn;
            }
            names[n] = strdup(e->d_name);
            if (names[n] == NULL) {
                closedir(d);
                for (size_t i = 0; i < n; i++) {
                    free(names[i]);
                }
                free(names);
                h3i_fail(error, error_size, "%s: out of memory", path);
                return 0;
            }
            n++;
        }
        closedir(d);
        if (n == 0) {
            free(names);
            h3i_fail(error, error_size,
                     "%s contains no .safetensors or .gguf files", path);
            return 0;
        }
        qsort(names, n, sizeof *names, cmp_names);
        int ok = 1;
        for (size_t i = 0; i < n && ok; i++) {
            char full[4096];
            snprintf(full, sizeof full, "%s/%s", path, names[i]);
            ok = h3i_source_add(s, full, error, error_size);
        }
        for (size_t i = 0; i < n; i++) {
            free(names[i]);
        }
        free(names);
        return ok;
    }
    if (!S_ISREG(st.st_mode)) {
        h3i_fail(error, error_size, "cannot open weights %s", path);
        return 0;
    }
    if (ends_with(path, ".gguf")) {
        return add_gguf(s, path, error, error_size);
    }
    if (ends_with(path, ".safetensors")) {
        return add_safetensors(s, path, error, error_size);
    }
    h3i_fail(error, error_size, "%s: expected a .gguf or .safetensors file", path);
    return 0;
}

const h3i_tensor_rec *h3i_source_find(const h3i_source *s, const char *name)
{
    for (size_t i = 0; i < s->count; i++) {
        if (strcmp(s->recs[i].name, name) == 0) {
            return &s->recs[i];
        }
    }
    /* One-way prefix aliases (SPEC §4.2), truncated to 511 bytes. */
    char aliased[512];
    const char *replacement = NULL;
    size_t prefix_len = 0;
    if (strncmp(name, "model.language_model.", 21) == 0) {
        replacement = "model.";
        prefix_len = 21;
    } else if (strncmp(name, "model.visual.", 13) == 0) {
        replacement = "visual.";
        prefix_len = 13;
    }
    if (replacement != NULL) {
        snprintf(aliased, sizeof aliased, "%s%s", replacement, name + prefix_len);
        for (size_t i = 0; i < s->count; i++) {
            if (strcmp(s->recs[i].name, aliased) == 0) {
                return &s->recs[i];
            }
        }
    }
    return NULL;
}

int h3i_source_read_raw(const h3i_tensor_rec *t, void *dst, uint64_t dst_offset,
                        int uncached, char *error, size_t error_size)
{
    int fd = open(t->path, O_RDONLY);
    if (fd < 0) {
        h3i_fail(error, error_size, "cannot open weights %s", t->path);
        return 0;
    }
#ifdef __APPLE__
    if (uncached) {
        fcntl(fd, F_NOCACHE, 1);
    }
#else
    (void)uncached;
#endif
    uint8_t *out = (uint8_t *)dst + dst_offset;
    uint64_t got = 0;
    while (got < t->bytes) {
        ssize_t n = pread(fd, out + got, (size_t)(t->bytes - got),
                          (off_t)(t->offset + got));
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
    if (got != t->bytes) {
        h3i_fail(error, error_size,
                 "cannot read %llu bytes from %s: unexpected end of file",
                 (unsigned long long)t->bytes, t->path);
        return 0;
    }
    return 1;
}

float *h3i_source_read_f32(const h3i_source *s, const char *name,
                           uint64_t expected_elements, char *error, size_t error_size)
{
    const h3i_tensor_rec *t = h3i_source_find(s, name);
    if (t == NULL) {
        h3i_fail(error, error_size, "missing tensor %s", name);
        return NULL;
    }
    if (!t->supported) {
        h3i_fail(error, error_size, "tensor %s has an unsupported type", name);
        return NULL;
    }
    if (expected_elements != 0 && t->elements != expected_elements) {
        h3i_fail(error, error_size, "tensor %s has %llu elements, expected %llu",
                 name, (unsigned long long)t->elements,
                 (unsigned long long)expected_elements);
        return NULL;
    }
    if (t->elements > SIZE_MAX / 4) {
        h3i_fail(error, error_size, "tensor %s is too large", name);
        return NULL;
    }
    float *out = malloc((size_t)t->elements * 4);
    uint8_t *raw = malloc((size_t)t->bytes);
    if (out == NULL || raw == NULL) {
        free(out);
        free(raw);
        h3i_fail(error, error_size, "out of memory reading tensor %s", name);
        return NULL;
    }
    if (!h3i_source_read_raw(t, raw, 0, 1, error, error_size)) {
        free(out);
        free(raw);
        return NULL;
    }
    h3i_dequant_row(t->type, raw, t->elements, out);
    free(raw);
    return out;
}

void h3i_source_free(h3i_source *s)
{
    if (s == NULL) {
        return;
    }
    for (size_t i = 0; i < s->count; i++) {
        free(s->recs[i].name);
        free(s->recs[i].path);
    }
    free(s->recs);
    s->recs = NULL;
    s->count = s->cap = 0;
}

#ifdef __APPLE__
#include "gpu/gpu.h"

h3i_gpu_tensor *h3i_source_load(const h3i_source *s, h3i_gpu *gpu,
                                const char *name, int dtype,
                                uint64_t expected_elements,
                                char *error, size_t error_size)
{
    const h3i_tensor_rec *t = h3i_source_find(s, name);
    if (!t) {
        h3i_fail(error, error_size, "missing tensor %s", name);
        return NULL;
    }
    if (!t->supported) {
        h3i_fail(error, error_size, "tensor %s has an unsupported stored type",
                 name);
        return NULL;
    }
    if (expected_elements != 0 && t->elements != expected_elements) {
        h3i_fail(error, error_size, "tensor %s has %llu elements, expected %llu",
                 name, (unsigned long long)t->elements,
                 (unsigned long long)expected_elements);
        return NULL;
    }
    if (dtype != H3I_TYPE_F32 && dtype != H3I_TYPE_BF16) {
        h3i_fail(error, error_size, "tensor %s: invalid load dtype", name);
        return NULL;
    }
    if (t->elements > SIZE_MAX || t->bytes > SIZE_MAX) {
        h3i_fail(error, error_size, "tensor %s is too large", name);
        return NULL;
    }
    h3i_gpu_tensor *out = dtype == H3I_TYPE_BF16
        ? h3i_gpu_tensor_new_bf16(gpu, (size_t)t->elements)
        : h3i_gpu_tensor_new_f32(gpu, (size_t)t->elements);
    if (!out) {
        h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
        return NULL;
    }
    if (t->type == dtype) {
        if (!h3i_source_read_raw(t, h3i_gpu_tensor_contents(out), 0, 0,
                                 error, error_size)) {
            h3i_gpu_tensor_free(out);
            return NULL;
        }
        return out;
    }
    h3i_gpu_tensor *raw = h3i_gpu_tensor_new_i8(gpu, (size_t)t->bytes);
    if (!raw) {
        h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
        h3i_gpu_tensor_free(out);
        return NULL;
    }
    if (!h3i_source_read_raw(t, h3i_gpu_tensor_contents(raw), 0, 1,
                             error, error_size)) {
        h3i_gpu_tensor_free(raw);
        h3i_gpu_tensor_free(out);
        return NULL;
    }
    int ok = h3i_gpu_convert_raw(gpu, out, 0, raw, 0, t->type,
                                 (size_t)t->elements);
    if (!ok) {
        h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
    }
    /* Metal retains encoded buffers until the command buffer completes, so
     * the staging tensor may be released immediately after encoding. */
    h3i_gpu_tensor_free(raw);
    if (!ok) {
        h3i_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}
#endif /* __APPLE__ */
