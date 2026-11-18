/* GGUF v2/v3 header parsing (SPEC §2) and CPU dequantization (SPEC §5).
 * The dequantization is written to be bit-identical with ggml and with the
 * GPU convert kernels: fma(a, q, -m) where the spec writes fma, everything
 * else separately-rounded F32 in the order shown (FP contraction OFF). */
#include "weights/gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "app/error.h"
#include "io/host.h"

/* ------------------------------------------------------------------ */
/* Type geometry (SPEC §2.1)                                           */
/* ------------------------------------------------------------------ */

h3i_type_info h3i_type_info_get(int type)
{
    switch (type) {
    case H3I_TYPE_F32:  return (h3i_type_info){1, 4, 1, 1};
    case H3I_TYPE_F16:  return (h3i_type_info){1, 2, 1, 1};
    case H3I_TYPE_BF16: return (h3i_type_info){1, 2, 1, 1};
    case H3I_TYPE_Q8_0: return (h3i_type_info){32, 34, 1, 32};
    case H3I_TYPE_Q2_K: return (h3i_type_info){256, 84, 16, 16};
    case H3I_TYPE_Q3_K: return (h3i_type_info){256, 110, 16, 16};
    case H3I_TYPE_Q4_K: return (h3i_type_info){256, 144, 8, 32};
    case H3I_TYPE_Q5_K: return (h3i_type_info){256, 176, 8, 32};
    case H3I_TYPE_Q6_K: return (h3i_type_info){256, 210, 8, 32};
    default:            return (h3i_type_info){0, 0, 0, 0};
    }
}

int h3i_type_supported(int type)
{
    return h3i_type_info_get(type).block_values != 0;
}

/* ------------------------------------------------------------------ */
/* Little-endian readers and float helpers                             */
/* ------------------------------------------------------------------ */

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

float h3i_half_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal: mant * 2^-24, normalized exactly */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / NaN */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float f32_from_bits(uint32_t b)
{
    float f;
    memcpy(&f, &b, 4);
    return f;
}

/* ------------------------------------------------------------------ */
/* Dequantization (SPEC §5.2)                                          */
/* ------------------------------------------------------------------ */

static void scale_min_k4(int j, const uint8_t *s, int *sc, int *mn)
{
    if (j < 4) {
        *sc = s[j] & 63;
        *mn = s[j + 4] & 63;
    } else {
        *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

void h3i_dequant_block(int type, const uint8_t *block, float *out)
{
    switch (type) {
    case H3I_TYPE_F32:
        out[0] = f32_from_bits(rd32(block));
        return;
    case H3I_TYPE_F16:
        out[0] = h3i_half_to_f32(rd16(block));
        return;
    case H3I_TYPE_BF16:
        out[0] = h3i_bf16_to_f32(rd16(block));
        return;
    case H3I_TYPE_Q8_0: {
        float d = h3i_half_to_f32(rd16(block));
        const int8_t *qs = (const int8_t *)(block + 2);
        for (int j = 0; j < 32; j++) {
            out[j] = (float)qs[j] * d;
        }
        return;
    }
    case H3I_TYPE_Q2_K: {
        const uint8_t *scales = block;
        const uint8_t *qs = block + 16;
        float d = h3i_half_to_f32(rd16(block + 80));
        float dmin = h3i_half_to_f32(rd16(block + 82));
        for (int c = 0; c < 16; c++) {
            const uint8_t *q = qs + 32 * (c / 8);
            int shift = 2 * ((c % 8) / 2);
            int second = 16 * (c % 2);
            float dl = d * (float)(scales[c] & 0xF);
            float ml = dmin * (float)(scales[c] >> 4);
            for (int l = 0; l < 16; l++) {
                out[16 * c + l] = fmaf(dl, (float)((q[l + second] >> shift) & 3), -ml);
            }
        }
        return;
    }
    case H3I_TYPE_Q3_K: {
        const uint8_t *hmask = block;
        const uint8_t *qs = block + 32;
        uint32_t aux0 = rd32(block + 96);
        uint32_t aux1 = rd32(block + 100);
        uint32_t tmp = rd32(block + 104);
        const uint32_t k1 = 0x03030303u, k2 = 0x0F0F0F0Fu;
        uint32_t w[4];
        w[0] = (aux0 & k2) | (((tmp >> 0) & k1) << 4);
        w[1] = (aux1 & k2) | (((tmp >> 2) & k1) << 4);
        w[2] = ((aux0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
        w[3] = ((aux1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
        float d = h3i_half_to_f32(rd16(block + 108));
        for (int c = 0; c < 16; c++) {
            int scale = (int8_t)((w[c / 4] >> (8 * (c % 4))) & 0xFF);
            float dl = d * (float)(scale - 32);
            int n = c / 8, j = (c % 8) / 2;
            int second = 16 * (c % 2), shift = 2 * j;
            const uint8_t *q = qs + 32 * n;
            int m = 1 << (4 * n + j);
            for (int l = 0; l < 16; l++) {
                int i = l + second;
                int v = ((q[i] >> shift) & 3) - ((hmask[i] & m) ? 0 : 4);
                out[16 * c + l] = dl * (float)v;
            }
        }
        return;
    }
    case H3I_TYPE_Q4_K: {
        float d = h3i_half_to_f32(rd16(block));
        float dmin = h3i_half_to_f32(rd16(block + 2));
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;
        for (int c = 0; c < 8; c++) {
            int sc, mn;
            scale_min_k4(c, scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            const uint8_t *q = qs + 32 * (c / 2);
            for (int l = 0; l < 32; l++) {
                int nib = (c % 2) ? (q[l] >> 4) : (q[l] & 0xF);
                out[32 * c + l] = fmaf(d1, (float)nib, -m1);
            }
        }
        return;
    }
    case H3I_TYPE_Q5_K: {
        float d = h3i_half_to_f32(rd16(block));
        float dmin = h3i_half_to_f32(rd16(block + 2));
        const uint8_t *scales = block + 4;
        const uint8_t *qh = block + 16;
        const uint8_t *qs = block + 48;
        for (int c = 0; c < 8; c++) {
            int sc, mn;
            scale_min_k4(c, scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            int j = c / 2;
            const uint8_t *ql = qs + 32 * j;
            int u = 1 << (2 * j + c % 2);
            for (int l = 0; l < 32; l++) {
                int low = (c % 2) ? (ql[l] >> 4) : (ql[l] & 0xF);
                out[32 * c + l] = fmaf(d1, (float)(low + ((qh[l] & u) ? 16 : 0)), -m1);
            }
        }
        return;
    }
    case H3I_TYPE_Q6_K: {
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        float d = h3i_half_to_f32(rd16(block + 208));
        for (int c = 0; c < 8; c++) {
            int n = c / 4, k = c % 4;
            const uint8_t *L = ql + 64 * n;
            const uint8_t *H = qh + 32 * n;
            const int8_t *S = scales + 8 * n;
            for (int l = 0; l < 32; l++) {
                int q;
                switch (k) {
                case 0: q = (L[l] & 0xF) | (((H[l] >> 0) & 3) << 4); break;
                case 1: q = (L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4); break;
                case 2: q = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4); break;
                default: q = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4); break;
                }
                q -= 32;
                out[32 * c + l] = (d * (float)S[l / 16 + 2 * k]) * (float)q;
            }
        }
        return;
    }
    default:
        return; /* unsupported: caller checks before calling */
    }
}

void h3i_dequant_row(int type, const uint8_t *raw, uint64_t elements, float *out)
{
    h3i_type_info ti = h3i_type_info_get(type);
    if (ti.block_values == 0) {
        return;
    }
    uint64_t blocks = elements / (uint64_t)ti.block_values;
    for (uint64_t b = 0; b < blocks; b++) {
        h3i_dequant_block(type, raw + b * (uint64_t)ti.block_bytes,
                          out + b * (uint64_t)ti.block_values);
    }
}

/* ------------------------------------------------------------------ */
/* Header parsing (SPEC §2.2–2.4)                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *p;
    size_t len;
    size_t pos;
    int truncated;
    int failed;
    const char *path;
    char *error;
    size_t es;
} gguf_ctx;

static const uint8_t *need(gguf_ctx *c, size_t n)
{
    if (c->failed || c->truncated) {
        return NULL;
    }
    if (n > c->len - c->pos) {
        c->truncated = 1;
        return NULL;
    }
    const uint8_t *p = c->p + c->pos;
    c->pos += n;
    return p;
}

static uint32_t get_u32(gguf_ctx *c)
{
    const uint8_t *p = need(c, 4);
    return p ? rd32(p) : 0;
}

static uint64_t get_u64(gguf_ctx *c)
{
    const uint8_t *p = need(c, 8);
    return p ? rd64(p) : 0;
}

/* String: u64 length + bytes, copied and NUL-terminated. Caller frees. */
static char *get_string(gguf_ctx *c)
{
    uint64_t len = get_u64(c);
    if (c->failed || c->truncated) {
        return NULL;
    }
    if (len > c->len - c->pos) {
        c->truncated = 1;
        return NULL;
    }
    char *s = malloc((size_t)len + 1);
    if (s == NULL) {
        c->failed = 1;
        h3i_fail(c->error, c->es, "%s: out of memory", c->path);
        return NULL;
    }
    memcpy(s, c->p + c->pos, (size_t)len);
    s[len] = '\0';
    c->pos += (size_t)len;
    return s;
}

static void fail(gguf_ctx *c, const char *fmt, ...) H3I_PRINTF_LIKE(2, 3);
static void fail(gguf_ctx *c, const char *fmt, ...)
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

static size_t fixed_size(uint32_t type)
{
    switch (type) {
    case 0: case 1: case 7: return 1;   /* UINT8 INT8 BOOL */
    case 2: case 3: return 2;           /* UINT16 INT16 */
    case 4: case 5: case 6: return 4;   /* UINT32 INT32 FLOAT32 */
    case 10: case 11: case 12: return 8; /* UINT64 INT64 FLOAT64 */
    default: return 0;
    }
}

static int skip_value(gguf_ctx *c, uint32_t type, int depth);

static int skip_array(gguf_ctx *c, int depth)
{
    uint32_t item_type = get_u32(c);
    uint64_t count = get_u64(c);
    if (c->failed || c->truncated) {
        return 0;
    }
    size_t fs = fixed_size(item_type);
    if (fs != 0) {
        if (count > SIZE_MAX / fs) {
            fail(c, "%s: invalid metadata array", c->path);
            return 0;
        }
        return need(c, (size_t)count * fs) != NULL;
    }
    if (item_type == 8) { /* strings: item by item */
        for (uint64_t i = 0; i < count; i++) {
            char *s = get_string(c);
            if (s == NULL) {
                return 0;
            }
            free(s);
        }
        return 1;
    }
    if (item_type == 9) { /* nested arrays */
        if (depth + 1 > 8) {
            fail(c, "%s: GGUF metadata nesting too deep", c->path);
            return 0;
        }
        for (uint64_t i = 0; i < count; i++) {
            if (!skip_array(c, depth + 1)) {
                return 0;
            }
        }
        return 1;
    }
    fail(c, "%s: invalid metadata value type %u", c->path, item_type);
    return 0;
}

static int skip_value(gguf_ctx *c, uint32_t type, int depth)
{
    size_t fs = fixed_size(type);
    if (fs != 0) {
        return need(c, fs) != NULL;
    }
    if (type == 8) {
        char *s = get_string(c);
        if (s == NULL) {
            return 0;
        }
        free(s);
        return 1;
    }
    if (type == 9) {
        if (depth + 1 > 8) {
            fail(c, "%s: GGUF metadata nesting too deep", c->path);
            return 0;
        }
        return skip_array(c, depth + 1);
    }
    fail(c, "%s: invalid metadata value type %u", c->path, type);
    return 0;
}

static uint64_t align_up(uint64_t x, uint32_t a)
{
    return (x + a - 1) & ~((uint64_t)a - 1);
}

/* Parse one prefix attempt. Returns: 1 ok, 0 hard error, -1 truncated. */
static int parse_header(gguf_ctx *c, h3i_gguf *out)
{
    const uint8_t *magic = need(c, 4);
    if (magic == NULL) {
        return -1;
    }
    if (memcmp(magic, "GGUF", 4) != 0) {
        fail(c, "%s: not a GGUF file", c->path);
        return 0;
    }
    uint32_t version = get_u32(c);
    if (c->truncated) {
        return -1;
    }
    if (version < 2 || version > 3) {
        fail(c, "%s: unsupported GGUF version %u", c->path, version);
        return 0;
    }
    uint64_t tensor_count = get_u64(c);
    uint64_t metadata_count = get_u64(c);
    if (c->truncated) {
        return -1;
    }
    if (tensor_count > 1000000 || metadata_count > 1000000) {
        fail(c, "%s: implausible GGUF counts", c->path);
        return 0;
    }

    out->alignment = 32;
    for (uint64_t i = 0; i < metadata_count; i++) {
        char *key = get_string(c);
        if (key == NULL) {
            return c->truncated ? -1 : 0;
        }
        uint32_t vtype = get_u32(c);
        if (c->truncated) {
            free(key);
            return -1;
        }
        if (strcmp(key, "general.alignment") == 0 && vtype == 4) {
            uint32_t a = get_u32(c);
            if (c->truncated) {
                free(key);
                return -1;
            }
            if (a == 0 || (a & (a - 1)) != 0) {
                fail(c, "%s: invalid alignment", c->path);
                free(key);
                return 0;
            }
            out->alignment = a;
        } else {
            if (!skip_value(c, vtype, 0)) {
                free(key);
                return c->truncated ? -1 : 0;
            }
        }
        free(key);
    }

    out->tensors = calloc((size_t)tensor_count, sizeof(h3i_gguf_tensor));
    if (out->tensors == NULL && tensor_count > 0) {
        fail(c, "%s: out of memory", c->path);
        return 0;
    }
    out->count = (size_t)tensor_count;

    for (uint64_t i = 0; i < tensor_count; i++) {
        h3i_gguf_tensor *t = &out->tensors[i];
        t->name = get_string(c);
        if (t->name == NULL) {
            return c->truncated ? -1 : 0;
        }
        uint32_t ndim = get_u32(c);
        if (c->truncated) {
            return -1;
        }
        if (ndim < 1 || ndim > 4) {
            fail(c, "%s: tensor %s has %u dimensions", c->path, t->name, ndim);
            return 0;
        }
        t->ndim = (int)ndim;
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            t->ne[d] = get_u64(c);
            if (c->truncated) {
                return -1;
            }
            if (t->ne[d] != 0 && elements > UINT64_MAX / t->ne[d]) {
                fail(c, "%s: tensor %s is too large", c->path, t->name);
                return 0;
            }
            elements *= t->ne[d];
        }
        t->elements = elements;
        uint32_t type = get_u32(c);
        uint64_t offset = get_u64(c);
        if (c->truncated) {
            return -1;
        }
        t->type = (int)type;
        h3i_type_info ti = h3i_type_info_get((int)type);
        if (ti.block_values != 0 && t->ne[0] % (uint64_t)ti.block_values == 0) {
            t->supported = 1;
            t->bytes = elements / (uint64_t)ti.block_values * (uint64_t)ti.block_bytes;
        } else {
            t->supported = 0;
            t->bytes = 0;
        }
        if (offset % out->alignment != 0) {
            fail(c, "misaligned offset");
            return 0;
        }
        t->offset = offset; /* relative; absolute set after data_offset known */
    }

    uint64_t data_offset = align_up((uint64_t)c->pos, out->alignment);
    for (size_t i = 0; i < out->count; i++) {
        h3i_gguf_tensor *t = &out->tensors[i];
        if (t->offset > UINT64_MAX - data_offset) {
            fail(c, "extends past the end of the file");
            return 0;
        }
        t->offset += data_offset;
        if (t->bytes != 0 &&
            (t->offset > out->file_size || t->bytes > out->file_size - t->offset)) {
            fail(c, "extends past the end of the file");
            return 0;
        }
    }
    return 1;
}

static void free_tensors(h3i_gguf *g)
{
    for (size_t i = 0; i < g->count; i++) {
        free(g->tensors[i].name);
    }
    free(g->tensors);
    g->tensors = NULL;
    g->count = 0;
}

int h3i_gguf_open(const char *path, h3i_gguf *out, char *error, size_t error_size)
{
    memset(out, 0, sizeof *out);
    out->path = strdup(path);
    if (out->path == NULL) {
        h3i_fail(error, error_size, "%s: out of memory", path);
        return 0;
    }

    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        h3i_fail(error, error_size, "%s: not a GGUF file", path);
        free(out->path);
        out->path = NULL;
        return 0;
    }
    out->file_size = (uint64_t)st.st_size;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        h3i_fail(error, error_size, "%s: not a GGUF file", path);
        free(out->path);
        out->path = NULL;
        return 0;
    }

    /* Prefix strategy (SPEC §2.1): start at min(1 MiB, file size), quadruple
     * on truncation, stop when the prefix covers the file or reaches 512 MiB. */
    uint64_t prefix = out->file_size < (1u << 20) ? out->file_size : (1u << 20);
    uint8_t *buf = NULL;
    int rc = 0;
    for (;;) {
        if (prefix == 0) {
            h3i_fail(error, error_size, "%s: GGUF header is truncated", path);
            rc = 0;
            break;
        }
        uint8_t *nb = realloc(buf, (size_t)prefix);
        if (nb == NULL) {
            h3i_fail(error, error_size, "%s: out of memory", path);
            rc = 0;
            break;
        }
        buf = nb;
        size_t got = 0;
        while (got < prefix) {
            ssize_t n = pread(fd, buf + got, (size_t)(prefix - got), (off_t)got);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            if (n == 0) {
                break;
            }
            got += (size_t)n;
        }
        if (got != prefix) {
            h3i_fail(error, error_size, "%s: GGUF header is truncated", path);
            rc = 0;
            break;
        }

        free_tensors(out);
        gguf_ctx c = {buf, (size_t)prefix, 0, 0, 0, path, error, error_size};
        int pr = parse_header(&c, out);
        if (pr == 1) {
            rc = 1;
            break;
        }
        if (pr == 0) {
            rc = 0;
            break;
        }
        /* truncated */
        if (prefix == out->file_size || prefix >= (512u << 20)) {
            h3i_fail(error, error_size, "%s: GGUF header is truncated", path);
            rc = 0;
            break;
        }
        prefix *= 4;
        if (prefix > out->file_size) {
            prefix = out->file_size;
        }
    }

    free(buf);
    close(fd);
    if (!rc) {
        free_tensors(out);
        free(out->path);
        out->path = NULL;
    }
    return rc;
}

void h3i_gguf_close(h3i_gguf *g)
{
    if (g == NULL) {
        return;
    }
    free_tensors(g);
    free(g->path);
    g->path = NULL;
}
