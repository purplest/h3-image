/* bench_gpu [seq] — random-data microbenchmarks (SPEC §6.6): DiT QKV/MLP
 * BF16 matmuls, attention at 1×/4× token counts, Q4_K→BF16 decode speed. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app/clock.h"
#include "app/paths.h"
#include "gpu/gpu.h"
#include "weights/gguf.h"

static uint16_t rand_bf16_small(void)
{
    /* values in ±[0.5, 1): no overflow in accumulated matmuls */
    return (uint16_t)(0x3F00u | (uint32_t)(rand() & 0x007F) |
                      ((uint32_t)(rand() & 1) << 15));
}

static void fill_bf16(uint16_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        p[i] = rand_bf16_small();
    }
}

static double bench_linear(h3i_gpu *g, size_t rows, size_t in_w, size_t out_w, int mlp)
{
    uint16_t *x = malloc(rows * in_w * 2);
    uint16_t *w1 = malloc((mlp ? 2 * 14336 : out_w) * in_w * 2);
    uint16_t *w2 = mlp ? malloc(5376 * 14336 * 2) : NULL;
    uint16_t *y = malloc(rows * out_w * 2);
    fill_bf16(x, rows * in_w);
    fill_bf16(w1, (mlp ? 2 * 14336 : out_w) * in_w);
    if (w2) {
        fill_bf16(w2, 5376 * 14336);
    }
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(g, x, rows * in_w);
    h3i_gpu_tensor *tw1 = h3i_gpu_tensor_from_bf16(g, w1, (mlp ? 2 * 14336 : out_w) * in_w);
    h3i_gpu_tensor *tw2 = w2 ? h3i_gpu_tensor_from_bf16(g, w2, 5376 * 14336) : NULL;
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, rows * out_w);
    double best = 1e30;
    for (int it = 0; it < 3; it++) {
        double t0 = h3i_now();
        if (!h3i_gpu_begin(g) ||
            !(mlp ? h3i_gpu_mlp_bf16(g, ty, tx, tw1, tw2, (uint32_t)rows, (uint32_t)in_w,
                           14336, (uint32_t)out_w)
                  : h3i_gpu_linear_bf16(g, ty, tx, tw1, NULL, (uint32_t)rows,
                          (uint32_t)in_w, (uint32_t)out_w)) ||
            !h3i_gpu_submit(g)) {
            fprintf(stderr, "bench failed: %s\n", h3i_gpu_error(g));
            exit(1);
        }
        double dt = h3i_now() - t0;
        if (dt < best) {
            best = dt;
        }
    }
    (void)y;
    h3i_gpu_tensor_free(tx);
    h3i_gpu_tensor_free(tw1);
    if (tw2) {
        h3i_gpu_tensor_free(tw2);
    }
    h3i_gpu_tensor_free(ty);
    free(x);
    free(w1);
    free(w2);
    free(y);
    return best;
}

int main(int argc, char **argv)
{
    size_t seq = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 1024;
    char path[4096], err[512];
    if (h3i_shader_path(path, sizeof path) == NULL) {
        fprintf(stderr, "bench_gpu: cannot find build/h3i_shaders.metal (run make)\n");
        return 1;
    }
    h3i_gpu *g = h3i_gpu_create(path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "bench_gpu: %s\n", err);
        return 1;
    }
    srand(1);

    double t_qkv = bench_linear(g, seq, 5376, 21504, 0);
    double flops_qkv = 2.0 * (double)seq * 5376.0 * 21504.0;
    printf("QKV  [%zu,5376]x[5376,21504]: %.3f s  %.2f TFLOP/s\n",
           seq, t_qkv, flops_qkv / t_qkv / 1e12);

    double t_mlp = bench_linear(g, seq, 5376, 5376, 1);
    double flops_mlp = 2.0 * (double)seq * 5376.0 * 28672.0 +
                       2.0 * (double)seq * 14336.0 * 5376.0;
    printf("MLP  [%zu,5376] SwiGLU 14336:    %.3f s  %.2f TFLOP/s\n",
           seq, t_mlp, flops_mlp / t_mlp / 1e12);

    for (int mult = 1; mult <= 4; mult *= 4) {
        size_t s = seq * (size_t)mult;
        size_t n = s * 56 * 128;
        uint16_t *qkv = malloc(n * 2);
        fill_bf16(qkv, n);
        h3i_gpu_tensor *tq = h3i_gpu_tensor_from_bf16(g, qkv, n);
        h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, n);
        double best = 1e30;
        for (int it = 0; it < 3; it++) {
            double t0 = h3i_now();
            if (!h3i_gpu_begin(g) ||
                !h3i_gpu_sdpa_bf16(g, ty, tq, tq, tq, (uint32_t)s, 56, 128,
                             1.0f / sqrtf(128.0f)) ||
                !h3i_gpu_submit(g)) {
                fprintf(stderr, "sdpa failed: %s\n", h3i_gpu_error(g));
                return 1;
            }
            double dt = h3i_now() - t0;
            if (dt < best) {
                best = dt;
            }
        }
        double flops = 4.0 * (double)s * (double)s * 56.0 * 128.0;
        printf("SDPA seq %zu (56 heads):          %.3f s  %.2f TFLOP/s\n",
               s, best, flops / best / 1e12);
        h3i_gpu_tensor_free(tq);
        h3i_gpu_tensor_free(ty);
        free(qkv);
    }

    /* Q4_K → BF16 decode: one QKV-sized matrix */
    size_t elements = 21504ULL * 5376ULL;
    size_t raw_bytes = elements / 256 * 144;
    h3i_gpu_tensor *traw = h3i_gpu_tensor_new_i8(g, raw_bytes);
    memset(h3i_gpu_tensor_contents(traw), 0x55, raw_bytes);
    h3i_gpu_tensor *tout = h3i_gpu_tensor_new_bf16(g, elements);
    double best = 1e30;
    for (int it = 0; it < 3; it++) {
        double t0 = h3i_now();
        if (!h3i_gpu_begin(g) ||
            !h3i_gpu_convert_raw(g, tout, 0, traw, 0, H3I_TYPE_Q4_K, elements) ||
            !h3i_gpu_submit(g)) {
            fprintf(stderr, "convert failed: %s\n", h3i_gpu_error(g));
            return 1;
        }
        double dt = h3i_now() - t0;
        if (dt < best) {
            best = dt;
        }
    }
    printf("Q4_K→BF16 %zu values:            %.3f s  %.1f GB/s raw\n",
           elements, best, (double)raw_bytes / best / 1e9);
    h3i_gpu_tensor_free(traw);
    h3i_gpu_tensor_free(tout);

    h3i_gpu_free(g);
    return 0;
}
