/* A1 gate (CPU half): dequantization bit-exact against the vendored ggml
 * reference for every quant type, on random blocks; half/bf16/f32 element
 * conversions; GGUF header parsing on synthetic files. The GPU half
 * (convert_raw vs CPU) is added with the Metal runtime in M1.
 * With real weights present, also checks tensor counts and finiteness. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app/error.h"
#include "io/host.h"
#include "weights/gguf.h"
#include "weights/source.h"
#include "builders.h"
#include "check.h"
#include "ggml_ref.h"

#ifdef __APPLE__
#include "bf16_util.h"
#include "gpu_test.h"
#endif

#define TMP "build/test-tmp"

static uint32_t bits_of(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

/* Independent half decode via exact double arithmetic (finite values). */
static float ref_half(uint16_t h)
{
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 31;
    int mant = h & 1023;
    double v;
    if (exp == 0) {
        v = ldexp((double)mant, -24);
    } else if (exp == 31) {
        v = mant ? (double)NAN : (double)INFINITY;
    } else {
        v = ldexp(1.0 + (double)mant / 1024.0, exp - 15);
    }
    return (float)(sign ? -v : v);
}

static uint16_t rand_half_finite(void)
{
    uint16_t sign = (uint16_t)((rand() & 1) << 15);
    uint16_t exp = (uint16_t)((rand() % 31) << 10); /* 0..30: finite, incl subnormal */
    uint16_t mant = (uint16_t)(rand() & 0x3FF);
    return (uint16_t)(sign | exp | mant);
}

static void fill_random(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        p[i] = (uint8_t)(rand() & 255);
    }
}

static void put_half(uint8_t *p, uint16_t h)
{
    p[0] = (uint8_t)(h & 255);
    p[1] = (uint8_t)(h >> 8);
}

/* One quant type: 256 random blocks, our dequant vs ggml, bit for bit. */
static void oracle_q8_0(const void *r, float *y, int64_t k) { dequantize_row_q8_0((const block_q8_0 *)r, y, k); }
static void oracle_q2_K(const void *r, float *y, int64_t k) { dequantize_row_q2_K((const block_q2_K *)r, y, k); }
static void oracle_q3_K(const void *r, float *y, int64_t k) { dequantize_row_q3_K((const block_q3_K *)r, y, k); }
static void oracle_q4_K(const void *r, float *y, int64_t k) { dequantize_row_q4_K((const block_q4_K *)r, y, k); }
static void oracle_q5_K(const void *r, float *y, int64_t k) { dequantize_row_q5_K((const block_q5_K *)r, y, k); }
static void oracle_q6_K(const void *r, float *y, int64_t k) { dequantize_row_q6_K((const block_q6_K *)r, y, k); }

static void check_quant(int type, int block_bytes, size_t d_off, size_t dmin_off,
                        void (*oracle)(const void *, float *, int64_t))
{
    enum { NB = 256 };
    static uint8_t raw[NB * 256]; /* max block 210 B, 256 values */
    static float ours[NB * 256], ref[NB * 256];
    size_t total_bytes = (size_t)NB * (size_t)block_bytes;
    fill_random(raw, total_bytes);
    for (int b = 0; b < NB; b++) {
        uint8_t *blk = raw + (size_t)b * (size_t)block_bytes;
        put_half(blk + d_off, rand_half_finite());
        if (dmin_off != SIZE_MAX) {
            put_half(blk + dmin_off, rand_half_finite());
        }
    }
    h3i_dequant_row(type, raw, (uint64_t)NB * 256, ours);
    oracle(raw, ref, NB * 256);
    int bad = -1;
    for (int i = 0; i < NB * 256; i++) {
        if (bits_of(ours[i]) != bits_of(ref[i])) {
            bad = i;
            break;
        }
    }
    CHECK(bad < 0, "type %d dequant bit-exact vs ggml (first diff at %d: %08x vs %08x)",
          type, bad, bad >= 0 ? bits_of(ours[bad]) : 0, bad >= 0 ? bits_of(ref[bad]) : 0);
}

static void test_dequant(void)
{
    srand(1234);
    check_quant(H3I_TYPE_Q8_0, 34, 0, SIZE_MAX, oracle_q8_0);
    check_quant(H3I_TYPE_Q2_K, 84, 80, 82, oracle_q2_K);
    check_quant(H3I_TYPE_Q3_K, 110, 108, SIZE_MAX, oracle_q3_K);
    check_quant(H3I_TYPE_Q4_K, 144, 0, 2, oracle_q4_K);
    check_quant(H3I_TYPE_Q5_K, 176, 0, 2, oracle_q5_K);
    check_quant(H3I_TYPE_Q6_K, 210, 208, SIZE_MAX, oracle_q6_K);
}

static void test_elements(void)
{
    /* F16: all 65536 patterns vs the independent decode. */
    int bad = -1;
    for (uint32_t h = 0; h < 65536; h++) {
        float a = h3i_half_to_f32((uint16_t)h);
        float b = ref_half((uint16_t)h);
        int both_nan = isnan(a) && isnan(b);
        if (!both_nan && bits_of(a) != bits_of(b)) {
            bad = (int)h;
            break;
        }
    }
    CHECK(bad < 0, "half→f32 all patterns (first diff at 0x%04x)", bad);

    /* directed values */
    CHECK(h3i_half_to_f32(0x3C00) == 1.0f, "half 1.0");
    CHECK(h3i_half_to_f32(0xC000) == -2.0f, "half -2.0");
    CHECK(h3i_half_to_f32(0x7BFF) == 65504.0f, "half max 65504");
    CHECK(h3i_half_to_f32(0x0001) == 0x1p-24f, "half min subnormal");
    CHECK(isinf(h3i_half_to_f32(0x7C00)), "half inf");
    CHECK(isnan(h3i_half_to_f32(0x7E00)), "half NaN");
    CHECK(bits_of(h3i_half_to_f32(0x8000)) == 0x80000000u, "half -0");

    /* F32 / BF16 / F16 element rows */
    uint8_t raw[8] = {0x00, 0x00, 0x80, 0x3F, 0x4C, 0x3E, 0x00, 0x3C};
    float out[2];
    h3i_dequant_row(H3I_TYPE_F32, raw, 1, out);
    CHECK(out[0] == 1.0f, "f32 element");
    h3i_dequant_row(H3I_TYPE_BF16, raw + 4, 1, out);
    CHECK(bits_of(out[0]) == 0x3E4C0000u, "bf16 element (shift)");
    h3i_dequant_row(H3I_TYPE_F16, raw + 6, 1, out);
    CHECK(out[0] == 1.0f, "f16 element");
}

/* ------------------------------------------------------------------ */
/* GGUF parsing                                                        */
/* ------------------------------------------------------------------ */

static void test_gguf_valid(void)
{
    char path[256], err[512];
    snprintf(path, sizeof path, "%s/valid.gguf", TMP);
    float fdata[4] = {1.0f, -2.0f, 3.5f, 0.25f};
    uint8_t qdata[34];
    memset(qdata, 0, sizeof qdata);
    qdata[0] = 0x00; qdata[1] = 0x3C; /* d = 1.0 */
    for (int i = 0; i < 32; i++) {
        qdata[2 + i] = (uint8_t)(i - 16);
    }
    tb_tensor ts[2] = {
        {"w.f32", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, fdata, sizeof fdata},
        {"w.q8", 1, {32, 1, 1, 1}, H3I_TYPE_Q8_0, 32, qdata, sizeof qdata},
    };
    long data_off = build_gguf(path, 3, 32, ts, 2);
    CHECK(data_off > 0, "build gguf");

    h3i_gguf g;
    CHECK(h3i_gguf_open(path, &g, err, sizeof err), "open gguf: %s", err);
    CHECK(g.count == 2, "tensor count %zu", g.count);
    CHECK(g.alignment == 32, "alignment %u", g.alignment);
    CHECK(strcmp(g.tensors[0].name, "w.f32") == 0, "name 0");
    CHECK(g.tensors[0].supported && g.tensors[0].bytes == 16 &&
          g.tensors[0].elements == 4, "f32 tensor info");
    CHECK(g.tensors[0].offset == (uint64_t)data_off, "f32 offset");
    CHECK(g.tensors[1].supported && g.tensors[1].bytes == 34, "q8 tensor info");
    CHECK(g.tensors[1].offset == (uint64_t)data_off + 32, "q8 offset");
    h3i_gguf_close(&g);

    /* v2 accepted */
    snprintf(path, sizeof path, "%s/v2.gguf", TMP);
    build_gguf(path, 2, 32, ts, 2);
    CHECK(h3i_gguf_open(path, &g, err, sizeof err), "gguf v2 accepted: %s", err);
    h3i_gguf_close(&g);

    /* alignment 64 from metadata */
    snprintf(path, sizeof path, "%s/a64.gguf", TMP);
    data_off = build_gguf(path, 3, 64, ts, 1);
    CHECK(h3i_gguf_open(path, &g, err, sizeof err), "gguf align 64: %s", err);
    CHECK(g.alignment == 64 && data_off % 64 == 0, "alignment applied");
    h3i_gguf_close(&g);

    /* read through the source */
    snprintf(path, sizeof path, "%s/valid.gguf", TMP);
    h3i_source s = {0};
    CHECK(h3i_source_add(&s, path, err, sizeof err), "source add: %s", err);
    float *v = h3i_source_read_f32(&s, "w.f32", 4, err, sizeof err);
    CHECK(v != NULL && memcmp(v, fdata, sizeof fdata) == 0, "read_f32 values");
    free(v);
    v = h3i_source_read_f32(&s, "w.q8", 32, err, sizeof err);
    int ok = v != NULL;
    for (int i = 0; ok && i < 32; i++) {
        ok = v[i] == (float)(i - 16);
    }
    CHECK(ok, "read_f32 q8 dequant");
    free(v);
    h3i_source_free(&s);
}

static void test_gguf_errors(void)
{
    char path[256], err[512];
    h3i_gguf g;
    float fdata[4] = {0, 0, 0, 0};
    tb_tensor ts[1] = {{"x", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, fdata, sizeof fdata}};

    snprintf(path, sizeof path, "%s/bad-magic.gguf", TMP);
    build_gguf(path, 3, 32, ts, 1);
    FILE *f = fopen(path, "r+b");
    if (f) { fwrite("NOPE", 1, 4, f); fclose(f); }
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "bad magic rejected");
    CHECK(strstr(err, "not a GGUF file") != NULL, "msg: %s", err);

    snprintf(path, sizeof path, "%s/bad-ver.gguf", TMP);
    build_gguf(path, 1, 32, ts, 1);
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "version 1 rejected");
    CHECK(strstr(err, "unsupported GGUF version 1") != NULL, "msg: %s", err);

    snprintf(path, sizeof path, "%s/bad-ndim.gguf", TMP);
    ts[0].ndim = 5;
    build_gguf(path, 3, 32, ts, 1);
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "ndim 5 rejected");
    CHECK(strstr(err, "tensor x has 5 dimensions") != NULL, "msg: %s", err);
    ts[0].ndim = 1;

    snprintf(path, sizeof path, "%s/misalign.gguf", TMP);
    ts[0].offset = 1;
    build_gguf(path, 3, 32, ts, 1);
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "misaligned offset rejected");
    CHECK(strcmp(err, "misaligned offset") == 0, "msg: %s", err);
    ts[0].offset = 0;

    snprintf(path, sizeof path, "%s/past-end.gguf", TMP);
    ts[0].data_bytes = 4; /* declares 16 bytes, writes 4 */
    build_gguf(path, 3, 32, ts, 1);
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "past-end rejected");
    CHECK(strcmp(err, "extends past the end of the file") == 0, "msg: %s", err);
    ts[0].data_bytes = sizeof fdata;

    snprintf(path, sizeof path, "%s/truncated.gguf", TMP);
    f = fopen(path, "wb");
    if (f) {
        fwrite("GGUF", 1, 4, f);
        put_u32(f, 3);
        put_u64(f, 1);   /* tensor count */
        put_u64(f, 0);   /* metadata count */
        put_u64(f, 1000000); /* string length far past EOF */
        fclose(f);
    }
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "truncated rejected");
    CHECK(strstr(err, "GGUF header is truncated") != NULL, "msg: %s", err);

    snprintf(path, sizeof path, "%s/implausible.gguf", TMP);
    f = fopen(path, "wb");
    if (f) {
        fwrite("GGUF", 1, 4, f);
        put_u32(f, 3);
        put_u64(f, 1000001);
        put_u64(f, 0);
        fclose(f);
    }
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "implausible counts rejected");
    CHECK(strstr(err, "implausible GGUF counts") != NULL, "msg: %s", err);

    /* unsupported type: indexed but bytes = 0 */
    snprintf(path, sizeof path, "%s/unsupported.gguf", TMP);
    ts[0].type = 99;
    ts[0].data = NULL;
    ts[0].data_bytes = 0;
    build_gguf(path, 3, 32, ts, 1);
    CHECK(h3i_gguf_open(path, &g, err, sizeof err), "unsupported type indexed: %s", err);
    CHECK(g.count == 1 && !g.tensors[0].supported && g.tensors[0].bytes == 0,
          "unsupported tensor info");
    h3i_gguf_close(&g);
    ts[0].type = H3I_TYPE_F32;
    ts[0].data = fdata;
    ts[0].data_bytes = sizeof fdata;

    /* ne[0] not block-aligned ⇒ unsupported */
    snprintf(path, sizeof path, "%s/unaligned-block.gguf", TMP);
    ts[0].type = H3I_TYPE_Q8_0;
    ts[0].ne[0] = 33;
    ts[0].data = NULL;
    ts[0].data_bytes = 0;
    build_gguf(path, 3, 32, ts, 1);
    CHECK(h3i_gguf_open(path, &g, err, sizeof err), "block-unaligned indexed: %s", err);
    CHECK(!g.tensors[0].supported && g.tensors[0].bytes == 0, "block-unaligned unsupported");
    h3i_gguf_close(&g);

    /* metadata skipping: string, array of scalars, nested arrays */
    snprintf(path, sizeof path, "%s/meta.gguf", TMP);
    f = fopen(path, "wb");
    if (f) {
        fwrite("GGUF", 1, 4, f);
        put_u32(f, 3);
        put_u64(f, 1);
        put_u64(f, 4);
        put_str(f, "general.name"); put_u32(f, 8); put_str(f, "test model");
        put_str(f, "general.alignment"); put_u32(f, 4); put_u32(f, 32);
        put_str(f, "arr"); put_u32(f, 9); put_u32(f, 4); put_u64(f, 3);
        put_u32(f, 1); put_u32(f, 2); put_u32(f, 3);
        put_str(f, "nested"); put_u32(f, 9); put_u32(f, 9); put_u64(f, 1);
        put_u32(f, 8); put_u64(f, 1); put_str(f, "s");
        /* tensor info */
        put_str(f, "x"); put_u32(f, 1); put_u64(f, 4); put_u32(f, H3I_TYPE_F32); put_u64(f, 0);
        long pos = ftell(f);
        long pad = (32 - pos % 32) % 32;
        for (long i = 0; i < pad; i++) fputc(0, f);
        fwrite(fdata, 1, sizeof fdata, f);
        fclose(f);
    }
    CHECK(h3i_gguf_open(path, &g, err, sizeof err), "metadata skipped: %s", err);
    CHECK(g.count == 1 && g.alignment == 32, "metadata parse result");
    h3i_gguf_close(&g);

    /* invalid alignment */
    snprintf(path, sizeof path, "%s/bad-align.gguf", TMP);
    build_gguf(path, 3, 3, ts, 1);
    CHECK(!h3i_gguf_open(path, &g, err, sizeof err), "alignment 3 rejected");
    CHECK(strstr(err, "invalid alignment") != NULL, "msg: %s", err);
}

static void test_real_weights(void)
{
    const char *home = getenv("HOME");
    if (home == NULL) {
        printf("skip: real weights (no HOME)\n");
        return;
    }
    char path[4096], err[512];
    snprintf(path, sizeof path, "%s/models/MiniMax-H3-Image-GGUF/minimax_h3_image_pruned-Q4_K_M.gguf", home);
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        printf("skip: real GGUF weights (%s absent)\n", path);
        return;
    }
    fclose(f);
    h3i_source s = {0};
    CHECK(h3i_source_add(&s, path, err, sizeof err), "open real DiT: %s", err);
    CHECK(s.count == 528, "DiT tensor count %zu (want 528)", s.count);
    const h3i_tensor_rec *t = h3i_source_find(&s, "blocks.0.attn.qkv_proj.weight");
    CHECK(t != NULL && t->type == H3I_TYPE_Q4_K, "qkv tensor present, Q4_K");
    if (t != NULL) {
        CHECK(t->elements == 21504ULL * 5376ULL, "qkv elements");
        /* dequantize the first block row and check finiteness */
        uint8_t raw[144];
        CHECK(h3i_source_read_raw(t, raw, 0, 1, err, sizeof err), "read raw block: %s", err);
        float out[256];
        h3i_dequant_block(t->type, raw, out);
        int finite = 1;
        for (int i = 0; i < 256; i++) {
            if (!isfinite(out[i])) {
                finite = 0;
            }
        }
        CHECK(finite, "real Q4_K block finite");
    }
    h3i_source_free(&s);
}

#ifdef __APPLE__
/* A1 GPU half: convert_raw bit-identical to the CPU reference for every
 * type, with nonzero output/raw offsets. */
static void test_dequant_gpu(void)
{
    h3i_gpu *g = test_gpu_create();
    if (g == NULL) {
        return;
    }
    srand(4321);
    const int types[] = {H3I_TYPE_F32, H3I_TYPE_F16, H3I_TYPE_BF16, H3I_TYPE_Q8_0,
                         H3I_TYPE_Q2_K, H3I_TYPE_Q3_K, H3I_TYPE_Q4_K, H3I_TYPE_Q5_K,
                         H3I_TYPE_Q6_K};
    for (size_t ti_i = 0; ti_i < sizeof types / sizeof types[0]; ti_i++) {
        int type = types[ti_i];
        h3i_type_info info = h3i_type_info_get(type);
        enum { NB = 64 };
        size_t elements = (size_t)NB * (size_t)info.block_values;
        size_t bytes = (size_t)NB * (size_t)info.block_bytes;
        uint8_t *raw = malloc(bytes);
        float *cpu = malloc(elements * 4);
        float *got = malloc(elements * 4);
        uint16_t *gotb = malloc(elements * 2);
        fill_random(raw, bytes);
        for (int b = 0; b < NB; b++) {
            uint8_t *blk = raw + (size_t)b * (size_t)info.block_bytes;
            if (type == H3I_TYPE_Q8_0) put_half(blk, rand_half_finite());
            if (type == H3I_TYPE_Q2_K) { put_half(blk + 80, rand_half_finite()); put_half(blk + 82, rand_half_finite()); }
            if (type == H3I_TYPE_Q3_K) put_half(blk + 108, rand_half_finite());
            if (type == H3I_TYPE_Q4_K || type == H3I_TYPE_Q5_K) { put_half(blk, rand_half_finite()); put_half(blk + 2, rand_half_finite()); }
            if (type == H3I_TYPE_Q6_K) put_half(blk + 208, rand_half_finite());
            if (type == H3I_TYPE_F16) {
                for (size_t j = 0; j < bytes; j += 2) {
                    raw[j] = (uint8_t)(rand() & 255);
                    raw[j + 1] = (uint8_t)((rand() & 0x7B) | ((rand() & 1) << 7));
                }
            }
        }
        h3i_dequant_row(type, raw, elements, cpu);

        h3i_gpu_tensor *traw = h3i_gpu_tensor_new_i8(g, bytes + 8);
        memcpy((uint8_t *)h3i_gpu_tensor_contents(traw) + 3, raw, bytes);
        h3i_gpu_tensor *tf = h3i_gpu_tensor_new_f32(g, elements + 7);
        h3i_gpu_tensor *tb = h3i_gpu_tensor_new_bf16(g, elements);

        CHECK(h3i_gpu_begin(g), "gpu begin type %d", type);
        CHECK(h3i_gpu_convert_raw(g, tf, 7, traw, 3, type, elements),
              "convert f32 type %d: %s", type, h3i_gpu_error(g));
        CHECK(h3i_gpu_convert_raw(g, tb, 0, traw, 3, type, elements),
              "convert bf16 type %d: %s", type, h3i_gpu_error(g));
        CHECK(h3i_gpu_submit(g), "gpu submit type %d: %s", type, h3i_gpu_error(g));
        h3i_gpu_read_f32(tf, got, elements + 7);
        h3i_gpu_read_bf16(tb, gotb, elements);
        int bad = -1;
        for (size_t i = 0; i < elements; i++) {
            if (bits_of(got[7 + i]) != bits_of(cpu[i])) {
                bad = (int)i;
                break;
            }
        }
        CHECK(bad < 0, "GPU F32 convert type %d bit-exact (diff at %d)", type, bad);
        bad = -1;
        for (size_t i = 0; i < elements; i++) {
            if (gotb[i] != f_bf16(cpu[i])) {
                bad = (int)i;
                break;
            }
        }
        CHECK(bad < 0, "GPU BF16 convert type %d bit-exact (diff at %d)", type, bad);

        h3i_gpu_tensor_free(traw);
        h3i_gpu_tensor_free(tf);
        h3i_gpu_tensor_free(tb);
        free(raw);
        free(cpu);
        free(got);
        free(gotb);
    }
    h3i_gpu_free(g);
}
#endif /* __APPLE__ */

int main(void)
{
    test_mkdir("build");
    test_mkdir(TMP);
    test_dequant();
    test_elements();
    test_gguf_valid();
    test_gguf_errors();
    test_real_weights();
#ifdef __APPLE__
    test_dequant_gpu();
#endif
    TEST_RESULT();
}
