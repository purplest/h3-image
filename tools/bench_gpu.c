/* bench_gpu — SPEC.md part 6 §6 (Phase 1 microbenchmarks).
 * Random-data speed of the DiT's big BF16 operations (MPSGraph path):
 * QKV projection, fused SwiGLU MLP, and full attention at the 1 MP (1024²)
 * and 4 MP (2048²) image token counts. The Q4_K→BF16 decode benchmark
 * lands in Phase 2 with h3i_gpu_convert_raw.
 * Usage: build/bench_gpu [seq]        (default 4096)
 * Shaders: build/h3i_shaders.metal, or $H3I_SHADERS. No weights needed. */
#include "gpu/gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIDDEN 5376u
#define INNER 7168u
#define QKV (3u * INNER)
#define FFN 14336u
#define HEADS 56u
#define HEAD_DIM 128u

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint16_t to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static uint64_t lcg = UINT64_C(0x243F6A8885A308D3);

static void fill_random_bf16(h3i_gpu_tensor *tensor, size_t elements) {
    uint16_t *host = malloc(elements * sizeof(uint16_t));
    if (!host) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (size_t i = 0; i < elements; i++) {
        lcg = lcg * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        float v = (float)((double)(lcg >> 33) / 2147483648.0 * 2.0 - 1.0);
        host[i] = to_bf16(v * 0.05f);
    }
    if (!h3i_gpu_tensor_write_bf16(tensor, host, elements)) {
        fprintf(stderr, "write failed\n");
        exit(1);
    }
    free(host);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Median of 3 timed runs (after one warmup), each a full submit+wait. */
static double time_op(h3i_gpu *gpu, int (*op)(h3i_gpu *, void *), void *ctx) {
    double samples[3];
    for (int run = -1; run < 3; run++) {
        double start = now_seconds();
        if (!h3i_gpu_begin(gpu) || !op(gpu, ctx) || !h3i_gpu_submit(gpu)) {
            fprintf(stderr, "bench op failed: %s\n", h3i_gpu_error(gpu));
            exit(1);
        }
        if (run >= 0) samples[run] = now_seconds() - start;
    }
    qsort(samples, 3, sizeof(double), cmp_double);
    return samples[1];
}

typedef struct {
    h3i_gpu_tensor *x, *w, *b, *out;
    uint32_t rows, in, out_dim;
} linear_ctx;

static int run_linear(h3i_gpu *gpu, void *opaque) {
    linear_ctx *c = opaque;
    return h3i_gpu_linear_bf16(gpu, c->out, c->x, c->w, c->b, c->rows, c->in,
                               c->out_dim);
}

typedef struct {
    h3i_gpu_tensor *x, *fc1, *fc2, *out;
    uint32_t rows;
} mlp_ctx;

static int run_mlp(h3i_gpu *gpu, void *opaque) {
    mlp_ctx *c = opaque;
    return h3i_gpu_mlp_bf16(gpu, c->out, c->x, c->fc1, c->fc2, c->rows, HIDDEN,
                            FFN, HIDDEN);
}

typedef struct {
    h3i_gpu_tensor *q, *k, *v, *out;
    uint32_t seq;
} sdpa_ctx;

static int run_sdpa(h3i_gpu *gpu, void *opaque) {
    sdpa_ctx *c = opaque;
    return h3i_gpu_sdpa_bf16(gpu, c->out, c->q, c->k, c->v, c->seq, HEADS,
                             HEAD_DIM, 1.0f / sqrtf(128.0f));
}

static void bench_qkv(h3i_gpu *gpu, uint32_t seq) {
    linear_ctx c = {0};
    c.rows = seq; c.in = HIDDEN; c.out_dim = QKV;
    c.x = h3i_gpu_tensor_new_bf16(gpu, (size_t)seq * HIDDEN);
    c.w = h3i_gpu_tensor_new_bf16(gpu, (size_t)QKV * HIDDEN);
    c.out = h3i_gpu_tensor_new_bf16(gpu, (size_t)seq * QKV);
    fill_random_bf16(c.x, (size_t)seq * HIDDEN);
    fill_random_bf16(c.w, (size_t)QKV * HIDDEN);
    double t = time_op(gpu, run_linear, &c);
    double flops = 2.0 * seq * HIDDEN * QKV;
    printf("qkv   seq=%5u: %8.2f ms  %6.2f TFLOP/s\n", seq, t * 1e3,
           flops / t / 1e12);
    h3i_gpu_tensor_free(c.x); h3i_gpu_tensor_free(c.w); h3i_gpu_tensor_free(c.out);
}

static void bench_mlp(h3i_gpu *gpu, uint32_t seq) {
    mlp_ctx c = {0};
    c.rows = seq;
    c.x = h3i_gpu_tensor_new_bf16(gpu, (size_t)seq * HIDDEN);
    c.fc1 = h3i_gpu_tensor_new_bf16(gpu, (size_t)FFN * 2 * HIDDEN);
    c.fc2 = h3i_gpu_tensor_new_bf16(gpu, (size_t)HIDDEN * FFN);
    c.out = h3i_gpu_tensor_new_bf16(gpu, (size_t)seq * HIDDEN);
    fill_random_bf16(c.x, (size_t)seq * HIDDEN);
    fill_random_bf16(c.fc1, (size_t)FFN * 2 * HIDDEN);
    fill_random_bf16(c.fc2, (size_t)HIDDEN * FFN);
    double t = time_op(gpu, run_mlp, &c);
    double flops = 2.0 * seq * HIDDEN * FFN * 2 + 2.0 * seq * FFN * HIDDEN;
    printf("mlp   seq=%5u: %8.2f ms  %6.2f TFLOP/s\n", seq, t * 1e3,
           flops / t / 1e12);
    h3i_gpu_tensor_free(c.x); h3i_gpu_tensor_free(c.fc1);
    h3i_gpu_tensor_free(c.fc2); h3i_gpu_tensor_free(c.out);
}

static void bench_attention(h3i_gpu *gpu, uint32_t seq, const char *label) {
    sdpa_ctx c = {0};
    c.seq = seq;
    size_t n = (size_t)seq * HEADS * HEAD_DIM;
    c.q = h3i_gpu_tensor_new_bf16(gpu, n);
    c.k = h3i_gpu_tensor_new_bf16(gpu, n);
    c.v = h3i_gpu_tensor_new_bf16(gpu, n);
    c.out = h3i_gpu_tensor_new_bf16(gpu, n);
    fill_random_bf16(c.q, n);
    fill_random_bf16(c.k, n);
    fill_random_bf16(c.v, n);
    double t = time_op(gpu, run_sdpa, &c);
    double flops = 4.0 * (double)seq * seq * HEADS * HEAD_DIM;
    printf("sdpa  seq=%5u (%s): %8.2f ms  %6.2f TFLOP/s\n", seq, label,
           t * 1e3, flops / t / 1e12);
    h3i_gpu_tensor_free(c.q); h3i_gpu_tensor_free(c.k);
    h3i_gpu_tensor_free(c.v); h3i_gpu_tensor_free(c.out);
}

int main(int argc, char **argv) {
    uint32_t seq = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4096;
    if (!seq) seq = 4096;
    const char *shaders = getenv("H3I_SHADERS");
    if (!shaders || !*shaders) shaders = "build/h3i_shaders.metal";
    char error[512];
    h3i_gpu *gpu = h3i_gpu_create(shaders, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "bench_gpu: %s\n", error);
        return 1;
    }
    printf("device working set: %.2f GiB\n",
           (double)h3i_gpu_recommended_working_set(gpu) / 1073741824.0);
    bench_qkv(gpu, seq);
    bench_mlp(gpu, seq);
    /* 1 MP image (1024²): (1024/16/2)² = 1024 target tokens.
     * 4 MP image (2048²): 4096 target tokens. */
    bench_attention(gpu, 1024, "1 MP");
    bench_attention(gpu, 4096, "4 MP");
    printf("q4_k decode: benchmark lands in Phase 2 (h3i_gpu_convert_raw)\n");
    h3i_gpu_free(gpu);
    return 0;
}
