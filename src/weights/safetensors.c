/* Safetensors header parsing (SPEC §3): u64 LE header size, one JSON object,
 * tensor data after it. Hand-written JSON with the spec's exact semantics:
 * digits-only unsigned integers, control chars rejected in strings, escapes
 * incl. \uXXXX encoded to UTF-8 1–3 bytes (surrogate pairs NOT combined),
 * skipped scalars are unvalidated runs, duplicate keys not detected. */
#include "weights/safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "app/error.h"
#include "weights/gguf.h"

typedef struct {
    const char *p;
    const char *end;
    const char *path;
    char *error;
    size_t es;
    int failed;
} json_ctx;

static void jfail(json_ctx *c, const char *fmt, ...) H3I_PRINTF_LIKE(2, 3);
static void jfail(json_ctx *c, const char *fmt, ...)
{
    if (c->failed) {
        return;
    }
    c->failed = 1;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->error, c->es, fmt, ap);
    va_end(ap);
}

static void j_ws(json_ctx *c)
{
    while (c->p < c->end &&
           (*c->p == ' ' || *c->p == '\n' || *c->p == '\r' || *c->p == '\t')) {
        c->p++;
    }
}

static int j_hex4(json_ctx *c, uint32_t *out)
{
    if (c->end - c->p < 4) {
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char ch = c->p[i];
        uint32_t d;
        if (ch >= '0' && ch <= '9') {
            d = (uint32_t)(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            d = (uint32_t)(ch - 'a') + 10;
        } else if (ch >= 'A' && ch <= 'F') {
            d = (uint32_t)(ch - 'A') + 10;
        } else {
            jfail(c, "%s: invalid safetensors header", c->path);
            return 0;
        }
        v = (v << 4) | d;
    }
    c->p += 4;
    *out = v;
    return 1;
}

/* Append a code point as UTF-8 (1–3 bytes; surrogates encoded as-is). */
static void utf8_append(char *buf, size_t *n, uint32_t cp)
{
    if (cp < 0x80) {
        buf[(*n)++] = (char)cp;
    } else if (cp < 0x800) {
        buf[(*n)++] = (char)(0xC0 | (cp >> 6));
        buf[(*n)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        buf[(*n)++] = (char)(0xE0 | (cp >> 12));
        buf[(*n)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*n)++] = (char)(0x80 | (cp & 0x3F));
    }
}

/* Parse a JSON string into a malloc'd NUL-terminated UTF-8 buffer. */
static char *j_string(json_ctx *c)
{
    if (c->failed || c->p >= c->end || *c->p != '"') {
        jfail(c, "%s: invalid safetensors header", c->path);
        return NULL;
    }
    c->p++;
    size_t cap = 32, n = 0;
    char *buf = malloc(cap);
    if (buf == NULL) {
        jfail(c, "%s: out of memory", c->path);
        return NULL;
    }
    for (;;) {
        if (c->p >= c->end) {
            jfail(c, "%s: invalid safetensors header", c->path);
            free(buf);
            return NULL;
        }
        unsigned char b = (unsigned char)*c->p++;
        if (b == '"') {
            break;
        }
        if (n + 4 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (nb == NULL) {
                jfail(c, "%s: out of memory", c->path);
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        if (b < 0x20) { /* control chars rejected */
            jfail(c, "%s: invalid safetensors header", c->path);
            free(buf);
            return NULL;
        }
        if (b != '\\') {
            buf[n++] = (char)b;
            continue;
        }
        if (c->p >= c->end) {
            jfail(c, "%s: invalid safetensors header", c->path);
            free(buf);
            return NULL;
        }
        char e = *c->p++;
        switch (e) {
        case '"': buf[n++] = '"'; break;
        case '\\': buf[n++] = '\\'; break;
        case '/': buf[n++] = '/'; break;
        case 'b': buf[n++] = '\b'; break;
        case 'f': buf[n++] = '\f'; break;
        case 'n': buf[n++] = '\n'; break;
        case 'r': buf[n++] = '\r'; break;
        case 't': buf[n++] = '\t'; break;
        case 'u': {
            uint32_t cp;
            if (!j_hex4(c, &cp)) {
                free(buf);
                return NULL;
            }
            utf8_append(buf, &n, cp);
            break;
        }
        default:
            jfail(c, "%s: invalid safetensors header", c->path);
            free(buf);
            return NULL;
        }
    }
    buf[n] = '\0';
    return buf;
}

/* Skip any JSON value. depth guards recursion. */
static int j_skip(json_ctx *c, int depth)
{
    if (c->failed) {
        return 0;
    }
    if (depth > 256) {
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }
    j_ws(c);
    if (c->p >= c->end) {
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }
    char ch = *c->p;
    if (ch == '{' || ch == '[') {
        char close = (ch == '{') ? '}' : ']';
        c->p++;
        j_ws(c);
        if (c->p < c->end && *c->p == close) {
            c->p++;
            return 1;
        }
        for (;;) {
            if (ch == '{') {
                char *key = j_string(c);
                if (key == NULL) {
                    return 0;
                }
                free(key);
                j_ws(c);
                if (c->p >= c->end || *c->p != ':') {
                    jfail(c, "%s: invalid safetensors header", c->path);
                    return 0;
                }
                c->p++;
            }
            if (!j_skip(c, depth + 1)) {
                return 0;
            }
            j_ws(c);
            if (c->p >= c->end) {
                jfail(c, "%s: invalid safetensors header", c->path);
                return 0;
            }
            if (*c->p == ',') {
                c->p++;
                j_ws(c);
                continue;
            }
            if (*c->p == close) {
                c->p++;
                return 1;
            }
            jfail(c, "%s: invalid safetensors header", c->path);
            return 0;
        }
    }
    if (ch == '"') {
        char *s = j_string(c);
        if (s == NULL) {
            return 0;
        }
        free(s);
        return 1;
    }
    /* scalar: any run up to , } ] or whitespace, not validated */
    const char *start = c->p;
    while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' &&
           *c->p != ' ' && *c->p != '\n' && *c->p != '\r' && *c->p != '\t') {
        c->p++;
    }
    if (c->p == start) {
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }
    return 1;
}

/* Unsigned decimal integer, digits only, overflow rejected. */
static int j_uint(json_ctx *c, uint64_t *out)
{
    j_ws(c);
    const char *start = c->p;
    uint64_t v = 0;
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
        uint64_t d = (uint64_t)(*c->p - '0');
        if (v > (UINT64_MAX - d) / 10) {
            jfail(c, "%s: invalid safetensors header", c->path);
            return 0;
        }
        v = v * 10 + d;
        c->p++;
    }
    if (c->p == start) {
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }
    *out = v;
    return 1;
}

/* ------------------------------------------------------------------ */
/* dtype table (SPEC §3)                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    int size;
    int internal; /* ggml id, -1 = indexed but unsupported */
} st_dtype;

static const st_dtype st_dtypes[] = {
    {"BOOL", 1, -1}, {"I8", 1, -1}, {"U8", 1, -1},
    {"I16", 2, -1}, {"U16", 2, -1}, {"F16", 2, H3I_TYPE_F16},
    {"BF16", 2, H3I_TYPE_BF16},
    {"I32", 4, -1}, {"U32", 4, -1}, {"F32", 4, H3I_TYPE_F32},
    {"I64", 8, -1}, {"U64", 8, -1}, {"F64", 8, -1},
};

static const st_dtype *find_dtype(const char *name)
{
    for (size_t i = 0; i < sizeof st_dtypes / sizeof st_dtypes[0]; i++) {
        if (strcmp(st_dtypes[i].name, name) == 0) {
            return &st_dtypes[i];
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Header parsing                                                      */
/* ------------------------------------------------------------------ */

static void free_tensors(h3i_safetensors *st)
{
    for (size_t i = 0; i < st->count; i++) {
        free(st->tensors[i].name);
    }
    free(st->tensors);
    st->tensors = NULL;
    st->count = 0;
}

/* Parse one tensor descriptor object (after the name and ':'). */
static int parse_tensor(json_ctx *c, h3i_st_tensor *t, uint64_t data_start,
                        uint64_t data_size)
{
    j_ws(c);
    if (c->p >= c->end || *c->p != '{') {
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }
    c->p++;

    int have_dtype = 0, have_shape = 0, have_offsets = 0;
    const st_dtype *dtype = NULL;
    uint64_t begin = 0, end = 0;
    t->ndim = 0;

    j_ws(c);
    if (c->p < c->end && *c->p == '}') {
        c->p++;
        jfail(c, "incomplete tensor descriptor");
        return 0;
    }
    for (;;) {
        char *key = j_string(c);
        if (key == NULL) {
            return 0;
        }
        j_ws(c);
        if (c->p >= c->end || *c->p != ':') {
            free(key);
            jfail(c, "%s: invalid safetensors header", c->path);
            return 0;
        }
        c->p++;
        if (strcmp(key, "dtype") == 0) {
            char *s = j_string(c);
            if (s == NULL) {
                free(key);
                return 0;
            }
            dtype = find_dtype(s); /* unknown string ⇒ missing dtype */
            have_dtype = (dtype != NULL);
            free(s);
        } else if (strcmp(key, "shape") == 0) {
            j_ws(c);
            if (c->p >= c->end || *c->p != '[') {
                free(key);
                jfail(c, "%s: invalid safetensors header", c->path);
                return 0;
            }
            c->p++;
            t->ndim = 0;
            j_ws(c);
            if (c->p < c->end && *c->p == ']') {
                c->p++;
            } else {
                for (;;) {
                    uint64_t d;
                    if (!j_uint(c, &d)) {
                        free(key);
                        return 0;
                    }
                    if (t->ndim >= 8) {
                        free(key);
                        jfail(c, "tensor rank exceeds 8");
                        return 0;
                    }
                    t->shape[t->ndim++] = d;
                    j_ws(c);
                    if (c->p >= c->end) {
                        free(key);
                        jfail(c, "%s: invalid safetensors header", c->path);
                        return 0;
                    }
                    if (*c->p == ',') {
                        c->p++;
                        continue;
                    }
                    if (*c->p == ']') {
                        c->p++;
                        break;
                    }
                    free(key);
                    jfail(c, "%s: invalid safetensors header", c->path);
                    return 0;
                }
            }
            have_shape = 1;
        } else if (strcmp(key, "data_offsets") == 0) {
            j_ws(c);
            if (c->p >= c->end || *c->p != '[') {
                free(key);
                jfail(c, "%s: invalid safetensors header", c->path);
                return 0;
            }
            c->p++;
            if (!j_uint(c, &begin)) {
                free(key);
                return 0;
            }
            j_ws(c);
            if (c->p >= c->end || *c->p != ',') {
                free(key);
                jfail(c, "%s: invalid safetensors header", c->path);
                return 0;
            }
            c->p++;
            if (!j_uint(c, &end)) {
                free(key);
                return 0;
            }
            j_ws(c);
            if (c->p >= c->end || *c->p != ']') {
                free(key);
                jfail(c, "%s: invalid safetensors header", c->path);
                return 0;
            }
            c->p++;
            have_offsets = 1;
        } else {
            if (!j_skip(c, 0)) {
                free(key);
                return 0;
            }
        }
        free(key);
        j_ws(c);
        if (c->p >= c->end) {
            jfail(c, "%s: invalid safetensors header", c->path);
            return 0;
        }
        if (*c->p == ',') {
            c->p++;
            j_ws(c);
            continue;
        }
        if (*c->p == '}') {
            c->p++;
            break;
        }
        jfail(c, "%s: invalid safetensors header", c->path);
        return 0;
    }

    if (!have_dtype || !have_shape || !have_offsets) {
        jfail(c, "incomplete tensor descriptor");
        return 0;
    }
    if (end < begin || end > data_size) {
        jfail(c, "tensor data offsets exceed file");
        return 0;
    }
    uint64_t elements = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] != 0 && elements > UINT64_MAX / t->shape[i]) {
            jfail(c, "%s: tensor is too large", c->path);
            return 0;
        }
        elements *= t->shape[i];
    }
    if (elements > UINT64_MAX / (uint64_t)dtype->size) {
        jfail(c, "%s: tensor is too large", c->path);
        return 0;
    }
    uint64_t bytes = (uint64_t)dtype->size * elements;
    if (bytes != end - begin) {
        jfail(c, "tensor shape does not match byte offsets");
        return 0;
    }
    t->type = dtype->internal;
    t->supported = (dtype->internal >= 0);
    t->elements = elements;
    t->bytes = bytes;
    t->offset = data_start + begin;
    return 1;
}

int h3i_safetensors_open(const char *path, h3i_safetensors *out,
                         char *error, size_t error_size)
{
    memset(out, 0, sizeof *out);
    out->path = strdup(path);
    if (out->path == NULL) {
        h3i_fail(error, error_size, "%s: out of memory", path);
        return 0;
    }

    struct stat st;
    int fd = open(path, O_RDONLY);
    if (fd < 0 || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        h3i_fail(error, error_size, "%s: invalid file", path);
        if (fd >= 0) {
            close(fd);
        }
        free(out->path);
        out->path = NULL;
        return 0;
    }
    out->file_size = (uint64_t)st.st_size;
    if (out->file_size < 8) {
        h3i_fail(error, error_size, "%s: invalid file", path);
        close(fd);
        free(out->path);
        out->path = NULL;
        return 0;
    }

    uint8_t hdr[8];
    ssize_t n = pread(fd, hdr, 8, 0);
    if (n != 8) {
        h3i_fail(error, error_size, "%s: invalid file", path);
        close(fd);
        free(out->path);
        out->path = NULL;
        return 0;
    }
    uint64_t hsize = (uint64_t)hdr[0] | ((uint64_t)hdr[1] << 8) |
                     ((uint64_t)hdr[2] << 16) | ((uint64_t)hdr[3] << 24) |
                     ((uint64_t)hdr[4] << 32) | ((uint64_t)hdr[5] << 40) |
                     ((uint64_t)hdr[6] << 48) | ((uint64_t)hdr[7] << 56);
    if (hsize > 268435456u || hsize > out->file_size - 8) {
        h3i_fail(error, error_size, "%s: invalid header size", path);
        close(fd);
        free(out->path);
        out->path = NULL;
        return 0;
    }

    char *buf = malloc((size_t)hsize + 1);
    if (buf == NULL) {
        h3i_fail(error, error_size, "%s: out of memory", path);
        close(fd);
        free(out->path);
        out->path = NULL;
        return 0;
    }
    uint64_t got = 0;
    while (got < hsize) {
        n = pread(fd, buf + got, (size_t)(hsize - got), (off_t)(8 + got));
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
    if (got != hsize) {
        h3i_fail(error, error_size, "%s: invalid file", path);
        free(buf);
        free(out->path);
        out->path = NULL;
        return 0;
    }
    buf[hsize] = '\0';

    json_ctx c = {buf, buf + hsize, path, error, error_size, 0};
    uint64_t data_start = 8 + hsize;
    uint64_t data_size = out->file_size - data_start;

    j_ws(&c);
    if (c.p >= c.end || *c.p != '{') {
        jfail(&c, "%s: invalid safetensors header", path);
    } else {
        c.p++;
        j_ws(&c);
        if (c.p < c.end && *c.p == '}') {
            c.p++; /* empty object: 0 tensors, valid */
        } else {
            for (;;) {
                char *key = j_string(&c);
                if (key == NULL) {
                    break;
                }
                j_ws(&c);
                if (c.p >= c.end || *c.p != ':') {
                    free(key);
                    jfail(&c, "%s: invalid safetensors header", path);
                    break;
                }
                c.p++;
                if (strcmp(key, "__metadata__") == 0) {
                    free(key);
                    if (!j_skip(&c, 0)) {
                        break;
                    }
                } else {
                    h3i_st_tensor *nt = realloc(out->tensors,
                                                (out->count + 1) * sizeof *nt);
                    if (nt == NULL) {
                        free(key);
                        jfail(&c, "%s: out of memory", path);
                        break;
                    }
                    out->tensors = nt;
                    h3i_st_tensor *t = &out->tensors[out->count];
                    memset(t, 0, sizeof *t);
                    t->name = key; /* ownership moves */
                    if (!parse_tensor(&c, t, data_start, data_size)) {
                        break;
                    }
                    out->count++;
                }
                j_ws(&c);
                if (c.p >= c.end) {
                    jfail(&c, "%s: invalid safetensors header", path);
                    break;
                }
                if (*c.p == ',') {
                    c.p++;
                    j_ws(&c);
                    continue;
                }
                if (*c.p == '}') {
                    c.p++;
                    break;
                }
                jfail(&c, "%s: invalid safetensors header", path);
                break;
            }
        }
    }

    if (!c.failed) {
        /* after the object only whitespace is allowed */
        while (c.p < c.end &&
               (*c.p == ' ' || *c.p == '\n' || *c.p == '\r' || *c.p == '\t')) {
            c.p++;
        }
        if (c.p != c.end) {
            jfail(&c, "trailing data in safetensors header");
        }
    }

    free(buf);
    if (c.failed) {
        free_tensors(out);
        free(out->path);
        out->path = NULL;
        return 0;
    }
    return 1;
}

void h3i_safetensors_close(h3i_safetensors *st)
{
    if (st == NULL) {
        return;
    }
    free_tensors(st);
    free(st->path);
    st->path = NULL;
}
