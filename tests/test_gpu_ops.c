/* test_gpu_ops — SPEC.md part 6 §5 (Phase 1 gate).
 * GPU operations vs double-precision CPU references.
 * Tolerances: RMS/LayerNorm ≤2 BF16 steps, AdaLN ≤3, SwiGLU/SiLU-mul ≤1,
 * gate & add exact, linear max err ≤1% of max|ref|, SDPA rel L2 <0.02.
 * Also pins the §3.2 command model and tensor I/O round trips.
 * Usage: build/test_gpu_ops [shaders.metal]   (default build/h3i_shaders.metal)
 * Requires a Metal device; no weights. */
#include "gpu/gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Expected-value math is separately rounded (SPEC conventions). */
#pragma STDC FP_CONTRACT OFF

static int checks;
static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        checks++;                                                              \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

/* ---- BF16 helpers (SPEC §5.1 RNE, no NaN special case) ---- */

static uint16_t to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Distance in BF16 ULP steps (monotone key over sign-magnitude bits). */
static int bf16_steps(uint16_t a, uint16_t b) {
    if (from_bf16(a) == from_bf16(b)) return 0; /* ±0 equal */
    int ka = (a & 0x8000) ? (a ^ 0xFFFF) : (a | 0x8000);
    int kb = (b & 0x8000) ? (b ^ 0xFFFF) : (b | 0x8000);
    return ka > kb ? ka - kb : kb - ka;
}

/* ---- deterministic host data ---- */

static uint64_t lcg_state = UINT64_C(0x243F6A8885A308D3);

static double uniform(void) {
    lcg_state = lcg_state * UINT64_C(6364136223846793005) +
                UINT64_C(1442695040888963407);
    return (double)(lcg_state >> 33) / 2147483648.0;
}

static double range(double lo, double hi) { return lo + (hi - lo) * uniform(); }

static void fill_bf16(uint16_t *values, size_t count, double lo, double hi) {
    for (size_t i = 0; i < count; i++) values[i] = to_bf16((float)range(lo, hi));
}

static void fill_f32(float *values, size_t count, double lo, double hi) {
    for (size_t i = 0; i < count; i++) values[i] = (float)range(lo, hi);
}

/* ---- references ---- */

static double silu_d(double x) { return x / (1.0 + exp(-x)); }

static void rms_reference(const uint16_t *x, const uint16_t *w, size_t rows,
                          size_t width, double eps, uint16_t *out) {
    for (size_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (size_t c = 0; c < width; c++) {
            double v = (double)from_bf16(x[r * width + c]);
            ss += v * v;
        }
        double inv = 1.0 / sqrt(ss / (double)width + eps);
        for (size_t c = 0; c < width; c++)
            out[r * width + c] =
                to_bf16((float)((double)from_bf16(x[r * width + c]) * inv *
                                (double)from_bf16(w[c])));
    }
}

static void layer_norm_reference(const uint16_t *x, const uint16_t *w,
                                 const uint16_t *b, size_t rows, size_t width,
                                 double eps, uint16_t *out) {
    for (size_t r = 0; r < rows; r++) {
        double mean = 0.0;
        for (size_t c = 0; c < width; c++)
            mean += (double)from_bf16(x[r * width + c]);
        mean /= (double)width;
        double var = 0.0;
        for (size_t c = 0; c < width; c++) {
            double d = (double)from_bf16(x[r * width + c]) - mean;
            var += d * d;
        }
        var /= (double)width;
        double inv = 1.0 / sqrt(var + eps);
        for (size_t c = 0; c < width; c++) {
            double norm = ((double)from_bf16(x[r * width + c]) - mean) * inv;
            out[r * width + c] =
                to_bf16((float)(norm * (double)from_bf16(w[c]) +
                                (double)from_bf16(b[c])));
        }
    }
}

/* ---- tests ---- */

static void test_command_model(h3i_gpu *gpu) {
    h3i_gpu_tensor *a = h3i_gpu_tensor_new_bf16(gpu, 64);
    h3i_gpu_tensor *w = h3i_gpu_tensor_new_bf16(gpu, 64);
    h3i_gpu_tensor *out = h3i_gpu_tensor_new_bf16(gpu, 64);
    CHECK(a && w && out, "alloc");
    /* op without begin fails with the exact message */
    CHECK(h3i_gpu_rms_norm_bf16(gpu, out, a, w, 1, 64, 1e-5f) == 0,
          "op without begin succeeded");
    CHECK(strcmp(h3i_gpu_error(gpu), "h3i_gpu_begin() was not called") == 0,
          "wrong error: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_begin(gpu) == 1, "begin");
    CHECK(h3i_gpu_begin(gpu) == 0, "double begin succeeded");
    CHECK(h3i_gpu_rms_norm_bf16(gpu, out, a, w, 1, 64, 1e-5f) == 1, "encode");
    CHECK(h3i_gpu_continue(gpu) == 1, "continue");
    CHECK(h3i_gpu_rms_norm_bf16(gpu, out, a, w, 1, 64, 1e-5f) == 1,
          "encode after continue");
    CHECK(h3i_gpu_submit(gpu) == 1, "submit chain: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_begin(gpu) == 1, "begin after submit");
    CHECK(h3i_gpu_submit(gpu) == 1, "empty submit");
    h3i_gpu_tensor_free(a);
    h3i_gpu_tensor_free(w);
    h3i_gpu_tensor_free(out);
}

static void test_tensor_io(h3i_gpu *gpu) {
    enum { N = 257 };
    uint16_t host_bf16[N], back_bf16[N];
    float host_f32[N], back_f32[N];
    fill_bf16(host_bf16, N, -2.0, 2.0);
    fill_f32(host_f32, N, -2.0, 2.0);
    h3i_gpu_tensor *b = h3i_gpu_tensor_from_bf16(gpu, host_bf16, N);
    h3i_gpu_tensor *f = h3i_gpu_tensor_from_f32(gpu, host_f32, N);
    CHECK(b && f, "from_* alloc");
    CHECK(h3i_gpu_tensor_read_bf16(b, back_bf16, N) == 1, "read_bf16");
    CHECK(h3i_gpu_tensor_read_f32(f, back_f32, N) == 1, "read_f32");
    CHECK(memcmp(host_bf16, back_bf16, sizeof(host_bf16)) == 0,
          "bf16 round trip");
    CHECK(memcmp(host_f32, back_f32, sizeof(host_f32)) == 0, "f32 round trip");
    CHECK(h3i_gpu_tensor_elements(b) == N, "elements");
    CHECK(h3i_gpu_tensor_dtype(b) == H3I_GPU_BF16, "dtype bf16");
    CHECK(h3i_gpu_tensor_dtype(f) == H3I_GPU_F32, "dtype f32");
    h3i_gpu_tensor_free(b);
    h3i_gpu_tensor_free(f);
}

static void test_add_exact(h3i_gpu *gpu) {
    enum { N = 4096 };
    uint16_t *a = malloc(N * 2), *b = malloc(N * 2), *got = malloc(N * 2);
    fill_bf16(a, N, -4.0, 4.0);
    fill_bf16(b, N, -4.0, 4.0);
    h3i_gpu_tensor *ta = h3i_gpu_tensor_from_bf16(gpu, a, N);
    h3i_gpu_tensor *tb = h3i_gpu_tensor_from_bf16(gpu, b, N);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, N);
    CHECK(h3i_gpu_begin(gpu) && h3i_gpu_add_bf16(gpu, to, ta, tb, N) &&
              h3i_gpu_submit(gpu),
          "add encode/submit: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, N) == 1, "add read");
    int worst = 0;
    for (size_t i = 0; i < N; i++) {
        uint16_t want = to_bf16(from_bf16(a[i]) + from_bf16(b[i]));
        int steps = bf16_steps(got[i], want);
        if (steps > worst) worst = steps;
    }
    CHECK(worst == 0, "add not exact: worst %d steps", worst);
    free(a); free(b); free(got);
    h3i_gpu_tensor_free(ta); h3i_gpu_tensor_free(tb); h3i_gpu_tensor_free(to);
}

static void test_silu_mul(h3i_gpu *gpu) {
    enum { N = 4096 };
    uint16_t *g = malloc(N * 2), *u = malloc(N * 2), *got = malloc(N * 2);
    fill_bf16(g, N, -6.0, 6.0);
    fill_bf16(u, N, -2.0, 2.0);
    h3i_gpu_tensor *tg = h3i_gpu_tensor_from_bf16(gpu, g, N);
    h3i_gpu_tensor *tu = h3i_gpu_tensor_from_bf16(gpu, u, N);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, N);
    CHECK(h3i_gpu_begin(gpu) && h3i_gpu_silu_mul_bf16(gpu, to, tg, tu, N) &&
              h3i_gpu_submit(gpu),
          "silu_mul: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, N) == 1, "silu_mul read");
    int worst = 0;
    for (size_t i = 0; i < N; i++) {
        double gd = (double)from_bf16(g[i]), ud = (double)from_bf16(u[i]);
        uint16_t want = to_bf16((float)(silu_d(gd) * ud));
        int steps = bf16_steps(got[i], want);
        if (steps > worst) worst = steps;
    }
    CHECK(worst <= 1, "silu_mul worst %d steps (limit 1)", worst);
    free(g); free(u); free(got);
    h3i_gpu_tensor_free(tg); h3i_gpu_tensor_free(tu); h3i_gpu_tensor_free(to);
}

static void test_swiglu(h3i_gpu *gpu) {
    enum { ROWS = 16, WIDTH = 512 };
    size_t fused_n = (size_t)ROWS * WIDTH * 2, out_n = (size_t)ROWS * WIDTH;
    uint16_t *fused = malloc(fused_n * 2), *got = malloc(out_n * 2);
    fill_bf16(fused, fused_n, -5.0, 5.0);
    h3i_gpu_tensor *tf = h3i_gpu_tensor_from_bf16(gpu, fused, fused_n);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, out_n);
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_swiglu_bf16(gpu, to, tf, ROWS, WIDTH) &&
              h3i_gpu_submit(gpu),
          "swiglu: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, out_n) == 1, "swiglu read");
    int worst = 0;
    for (size_t r = 0; r < ROWS; r++)
        for (size_t c = 0; c < WIDTH; c++) {
            double gate = (double)from_bf16(fused[r * WIDTH * 2 + c]);
            double up = (double)from_bf16(fused[r * WIDTH * 2 + WIDTH + c]);
            uint16_t want = to_bf16((float)(silu_d(gate) * up));
            int steps = bf16_steps(got[r * WIDTH + c], want);
            if (steps > worst) worst = steps;
        }
    CHECK(worst <= 1, "swiglu worst %d steps (limit 1)", worst);
    free(fused); free(got);
    h3i_gpu_tensor_free(tf); h3i_gpu_tensor_free(to);
}

static void test_rms_norm_bf16(h3i_gpu *gpu) {
    enum { ROWS = 8, WIDTH = 5376 };
    size_t n = (size_t)ROWS * WIDTH;
    uint16_t *x = malloc(n * 2), *w = malloc(WIDTH * 2), *got = malloc(n * 2),
             *want = malloc(n * 2);
    fill_bf16(x, n, -2.0, 2.0);
    fill_bf16(w, WIDTH, 0.5, 1.5);
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(gpu, x, n);
    h3i_gpu_tensor *tw = h3i_gpu_tensor_from_bf16(gpu, w, WIDTH);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, n);
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_rms_norm_bf16(gpu, to, tx, tw, ROWS, WIDTH, 1e-5f) &&
              h3i_gpu_submit(gpu),
          "rms_norm_bf16: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, n) == 1, "rms read");
    rms_reference(x, w, ROWS, WIDTH, 1e-5, want);
    int worst = 0;
    for (size_t i = 0; i < n; i++) {
        int steps = bf16_steps(got[i], want[i]);
        if (steps > worst) worst = steps;
    }
    CHECK(worst <= 2, "rms_norm_bf16 worst %d steps (limit 2)", worst);
    free(x); free(w); free(got); free(want);
    h3i_gpu_tensor_free(tx); h3i_gpu_tensor_free(tw); h3i_gpu_tensor_free(to);
}

static void test_rms_norm_f32(h3i_gpu *gpu) {
    enum { ROWS = 4, WIDTH = 1024 };
    size_t n = (size_t)ROWS * WIDTH;
    float *x = malloc(n * 4), *w = malloc(WIDTH * 4), *got = malloc(n * 4);
    fill_f32(x, n, -2.0, 2.0);
    fill_f32(w, WIDTH, 0.5, 1.5);
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_f32(gpu, x, n);
    h3i_gpu_tensor *tw = h3i_gpu_tensor_from_f32(gpu, w, WIDTH);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_f32(gpu, n);
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_rms_norm_f32(gpu, to, tx, tw, ROWS, WIDTH, 1e-5f) &&
              h3i_gpu_submit(gpu),
          "rms_norm_f32: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_f32(to, got, n) == 1, "rms f32 read");
    double num = 0.0, den = 0.0;
    for (size_t r = 0; r < ROWS; r++) {
        double ss = 0.0;
        for (size_t c = 0; c < WIDTH; c++)
            ss += (double)x[r * WIDTH + c] * (double)x[r * WIDTH + c];
        double inv = 1.0 / sqrt(ss / (double)WIDTH + 1e-5);
        for (size_t c = 0; c < WIDTH; c++) {
            double want = (double)x[r * WIDTH + c] * inv * (double)w[c];
            double d = (double)got[r * WIDTH + c] - want;
            num += d * d;
            den += want * want;
        }
    }
    double rel = sqrt(num / (den > 0 ? den : 1));
    CHECK(rel < 1e-5, "rms_norm_f32 rel L2 %g (limit 1e-5)", rel);
    free(x); free(w); free(got);
    h3i_gpu_tensor_free(tx); h3i_gpu_tensor_free(tw); h3i_gpu_tensor_free(to);
}

static void test_layer_norm_bf16(h3i_gpu *gpu) {
    enum { ROWS = 8, WIDTH = 1152 };
    size_t n = (size_t)ROWS * WIDTH;
    uint16_t *x = malloc(n * 2), *w = malloc(WIDTH * 2), *b = malloc(WIDTH * 2),
             *got = malloc(n * 2), *want = malloc(n * 2);
    fill_bf16(x, n, -2.0, 2.0);
    fill_bf16(w, WIDTH, 0.5, 1.5);
    fill_bf16(b, WIDTH, -0.1, 0.1);
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(gpu, x, n);
    h3i_gpu_tensor *tw = h3i_gpu_tensor_from_bf16(gpu, w, WIDTH);
    h3i_gpu_tensor *tb = h3i_gpu_tensor_from_bf16(gpu, b, WIDTH);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, n);
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_layer_norm_bf16(gpu, to, tx, tw, tb, ROWS, WIDTH, 1e-6f) &&
              h3i_gpu_submit(gpu),
          "layer_norm_bf16: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, n) == 1, "layer_norm read");
    layer_norm_reference(x, w, b, ROWS, WIDTH, 1e-6, want);
    int worst = 0;
    for (size_t i = 0; i < n; i++) {
        int steps = bf16_steps(got[i], want[i]);
        if (steps > worst) worst = steps;
    }
    CHECK(worst <= 2, "layer_norm_bf16 worst %d steps (limit 2)", worst);
    free(x); free(w); free(b); free(got); free(want);
    h3i_gpu_tensor_free(tx); h3i_gpu_tensor_free(tw);
    h3i_gpu_tensor_free(tb); h3i_gpu_tensor_free(to);
}

static void test_adaln_gate(h3i_gpu *gpu) {
    enum { ROWS = 16, WIDTH = 1024, SLOTS = 6, MOD_ROWS = 7 };
    size_t n = (size_t)ROWS * WIDTH;
    size_t mod_n = (size_t)MOD_ROWS * SLOTS * WIDTH;
    uint16_t *x = malloc(n * 2), *w = malloc(WIDTH * 2),
             *mod = malloc(mod_n * 2), *got = malloc(n * 2);
    uint32_t row_map[ROWS];
    fill_bf16(x, n, -2.0, 2.0);
    fill_bf16(w, WIDTH, 0.5, 1.5);
    fill_bf16(mod, mod_n, -0.25, 0.25);
    for (size_t r = 0; r < ROWS; r++) row_map[r] = (uint32_t)(r % MOD_ROWS);
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(gpu, x, n);
    h3i_gpu_tensor *tw = h3i_gpu_tensor_from_bf16(gpu, w, WIDTH);
    h3i_gpu_tensor *tm = h3i_gpu_tensor_from_bf16(gpu, mod, mod_n);
    h3i_gpu_tensor *tr = h3i_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, n);

    /* AdaLN: norm(x)*(1+scale)+shift, slots 0/1 */
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_adaln_bf16(gpu, to, tx, tw, tm, tr, ROWS, WIDTH, SLOTS,
                                 0, 1, 1e-5f) &&
              h3i_gpu_submit(gpu),
          "adaln: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, n) == 1, "adaln read");
    /* The kernel keeps the normalized value in F32 (no intermediate BF16
     * rounding) and rounds once at the output: bf16((x*inv*w)*(1+scale)+shift).
     * Mirror that in double; F32 reduction + rsqrt differences stay well
     * inside the 3-step limit. */
    int worst = 0;
    for (size_t r = 0; r < ROWS; r++) {
        double ss = 0.0;
        for (size_t c = 0; c < WIDTH; c++) {
            double v = (double)from_bf16(x[r * WIDTH + c]);
            ss += v * v;
        }
        double inv = 1.0 / sqrt(ss / (double)WIDTH + 1e-5);
        size_t base = (size_t)row_map[r] * SLOTS * WIDTH;
        for (size_t c = 0; c < WIDTH; c++) {
            double norm = (double)from_bf16(x[r * WIDTH + c]) * inv *
                          (double)from_bf16(w[c]);
            double scale = (double)from_bf16(mod[base + WIDTH + c]);
            double shift = (double)from_bf16(mod[base + c]);
            uint16_t expect = to_bf16((float)(norm * (1.0 + scale) + shift));
            int steps = bf16_steps(got[r * WIDTH + c], expect);
            if (steps > worst) worst = steps;
        }
    }
    CHECK(worst <= 3, "adaln worst %d steps (limit 3)", worst);

    /* Gate: res + branch*gate, slot 2. The kernel computes the F32 expression
     * in one statement; Metal may or may not contract it to fma. Accept either
     * faithful rounding (both catch intermediate BF16 rounding bugs). */
    uint16_t *branch = malloc(n * 2);
    fill_bf16(branch, n, -2.0, 2.0);
    h3i_gpu_tensor *tb = h3i_gpu_tensor_from_bf16(gpu, branch, n);
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_gate_bf16(gpu, to, tx, tb, tm, tr, ROWS, WIDTH, SLOTS, 2) &&
              h3i_gpu_submit(gpu),
          "gate: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, n) == 1, "gate read");
    worst = 0;
    for (size_t r = 0; r < ROWS; r++) {
        size_t base = (size_t)row_map[r] * SLOTS * WIDTH;
        for (size_t c = 0; c < WIDTH; c++) {
            size_t i = r * WIDTH + c;
            float res = from_bf16(x[i]), br = from_bf16(branch[i]);
            float gate = from_bf16(mod[base + 2 * WIDTH + c]);
            uint16_t separate = to_bf16(res + br * gate);
            uint16_t fused = to_bf16(fmaf(br, gate, res));
            int steps = bf16_steps(got[i], separate);
            int steps_f = bf16_steps(got[i], fused);
            int best = steps < steps_f ? steps : steps_f;
            if (best > worst) worst = best;
        }
    }
    CHECK(worst == 0, "gate not exact: worst %d steps", worst);
    free(x); free(w); free(mod); free(got); free(branch);
    h3i_gpu_tensor_free(tx); h3i_gpu_tensor_free(tw); h3i_gpu_tensor_free(tm);
    h3i_gpu_tensor_free(tr); h3i_gpu_tensor_free(to); h3i_gpu_tensor_free(tb);
}

static void linear_check(h3i_gpu *gpu, uint32_t rows, uint32_t in,
                         uint32_t out, int use_bias, int bf16) {
    size_t xn = (size_t)rows * in, wn = (size_t)out * in, on = (size_t)rows * out;
    double scale = 1.0 / sqrt((double)in);
    void *x = malloc(xn * (bf16 ? 2 : 4)), *w = malloc(wn * (bf16 ? 2 : 4)),
         *got = malloc(on * (bf16 ? 2 : 4));
    double *xd = malloc(xn * 8), *wd = malloc(wn * 8), *bd = NULL;
    if (bf16) {
        fill_bf16(x, xn, -1.0, 1.0);
        fill_bf16(w, wn, -scale, scale);
        for (size_t i = 0; i < xn; i++)
            xd[i] = (double)from_bf16(((uint16_t *)x)[i]);
        for (size_t i = 0; i < wn; i++)
            wd[i] = (double)from_bf16(((uint16_t *)w)[i]);
    } else {
        fill_f32(x, xn, -1.0, 1.0);
        fill_f32(w, wn, -scale, scale);
        for (size_t i = 0; i < xn; i++) xd[i] = (double)((float *)x)[i];
        for (size_t i = 0; i < wn; i++) wd[i] = (double)((float *)w)[i];
    }
    h3i_gpu_tensor *tx = bf16 ? h3i_gpu_tensor_from_bf16(gpu, x, xn)
                              : h3i_gpu_tensor_from_f32(gpu, x, xn);
    h3i_gpu_tensor *tw = bf16 ? h3i_gpu_tensor_from_bf16(gpu, w, wn)
                              : h3i_gpu_tensor_from_f32(gpu, w, wn);
    h3i_gpu_tensor *tb = NULL;
    if (use_bias) {
        bd = malloc(out * 8);
        if (bf16) {
            uint16_t *b = malloc(out * 2);
            fill_bf16(b, out, -0.1, 0.1);
            for (uint32_t i = 0; i < out; i++) bd[i] = (double)from_bf16(b[i]);
            tb = h3i_gpu_tensor_from_bf16(gpu, b, out);
            free(b);
        } else {
            float *b = malloc(out * 4);
            fill_f32(b, out, -0.1, 0.1);
            for (uint32_t i = 0; i < out; i++) bd[i] = (double)b[i];
            tb = h3i_gpu_tensor_from_f32(gpu, b, out);
            free(b);
        }
    }
    h3i_gpu_tensor *to = bf16 ? h3i_gpu_tensor_new_bf16(gpu, on)
                              : h3i_gpu_tensor_new_f32(gpu, on);
    int ok = h3i_gpu_begin(gpu);
    ok = ok && (bf16 ? h3i_gpu_linear_bf16(gpu, to, tx, tw, tb, rows, in, out)
                     : h3i_gpu_linear_f32(gpu, to, tx, tw, tb, rows, in, out));
    ok = ok && h3i_gpu_submit(gpu);
    CHECK(ok, "linear %dx%dx%d bf16=%d bias=%d: %s", rows, in, out, bf16,
          use_bias, h3i_gpu_error(gpu));
    ok = bf16 ? h3i_gpu_tensor_read_bf16(to, got, on)
              : h3i_gpu_tensor_read_f32(to, got, on);
    CHECK(ok == 1, "linear read");
    double max_err = 0.0, max_ref = 0.0;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t o = 0; o < out; o++) {
            double acc = bd ? bd[o] : 0.0;
            for (uint32_t i = 0; i < in; i++)
                acc += xd[(size_t)r * in + i] * wd[(size_t)o * in + i];
            double gotd = bf16
                ? (double)from_bf16(((uint16_t *)got)[(size_t)r * out + o])
                : (double)((float *)got)[(size_t)r * out + o];
            double err = fabs(gotd - acc);
            if (err > max_err) max_err = err;
            if (fabs(acc) > max_ref) max_ref = fabs(acc);
        }
    CHECK(max_err <= 0.01 * max_ref,
          "linear %dx%dx%d bf16=%d: max err %g > 1%% of max|ref| %g", rows, in,
          out, bf16, max_err, max_ref);
    free(x); free(w); free(got); free(xd); free(wd); free(bd);
    h3i_gpu_tensor_free(tx); h3i_gpu_tensor_free(tw);
    h3i_gpu_tensor_free(tb); h3i_gpu_tensor_free(to);
}

static void test_sdpa(h3i_gpu *gpu) {
    enum { SEQ = 48, HEADS = 4, HD = 64 };
    size_t n = (size_t)SEQ * HEADS * HD;
    uint16_t *q = malloc(n * 2), *k = malloc(n * 2), *v = malloc(n * 2),
             *got = malloc(n * 2);
    fill_bf16(q, n, -1.0, 1.0);
    fill_bf16(k, n, -1.0, 1.0);
    fill_bf16(v, n, -1.0, 1.0);
    h3i_gpu_tensor *tq = h3i_gpu_tensor_from_bf16(gpu, q, n);
    h3i_gpu_tensor *tk = h3i_gpu_tensor_from_bf16(gpu, k, n);
    h3i_gpu_tensor *tv = h3i_gpu_tensor_from_bf16(gpu, v, n);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(gpu, n);
    float scale = 1.0f / 8.0f;
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_gpu_sdpa_bf16(gpu, to, tq, tk, tv, SEQ, HEADS, HD, scale) &&
              h3i_gpu_submit(gpu),
          "sdpa: %s", h3i_gpu_error(gpu));
    CHECK(h3i_gpu_tensor_read_bf16(to, got, n) == 1, "sdpa read");
    double num = 0.0, den = 0.0;
    double scores[SEQ];
    for (int i = 0; i < SEQ; i++)
        for (int h = 0; h < HEADS; h++) {
            double max_score = -(double)INFINITY;
            for (int j = 0; j < SEQ; j++) {
                double dot = 0.0;
                for (int d = 0; d < HD; d++)
                    dot += (double)from_bf16(q[((size_t)i * HEADS + h) * HD + d]) *
                           (double)from_bf16(k[((size_t)j * HEADS + h) * HD + d]);
                scores[j] = dot * (double)scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            double sum = 0.0;
            for (int j = 0; j < SEQ; j++) {
                scores[j] = exp(scores[j] - max_score);
                sum += scores[j];
            }
            for (int d = 0; d < HD; d++) {
                double acc = 0.0;
                for (int j = 0; j < SEQ; j++)
                    acc += scores[j] / sum *
                           (double)from_bf16(v[((size_t)j * HEADS + h) * HD + d]);
                size_t idx = ((size_t)i * HEADS + h) * HD + d;
                double gotd = (double)from_bf16(got[idx]);
                num += (gotd - acc) * (gotd - acc);
                den += acc * acc;
            }
        }
    double rel = sqrt(num / (den > 0 ? den : 1));
    CHECK(rel < 0.02, "sdpa rel L2 %g (limit 0.02)", rel);
    free(q); free(k); free(v); free(got);
    h3i_gpu_tensor_free(tq); h3i_gpu_tensor_free(tk);
    h3i_gpu_tensor_free(tv); h3i_gpu_tensor_free(to);
}

int main(int argc, char **argv) {
    const char *shaders = argc > 1 ? argv[1] : "build/h3i_shaders.metal";
    /* Spec convention: skip (and pass) when a prerequisite is absent; a
     * present-but-broken shader source or GPU still fails loudly. */
    FILE *probe = fopen(shaders, "r");
    if (!probe) {
        printf("skip: %s not found (run make)\n", shaders);
        printf("ok: 0 checks, 0 failures\n");
        return 0;
    }
    fclose(probe);
    char error[512];
    h3i_gpu *gpu = h3i_gpu_create(shaders, error, sizeof(error));
    if (!gpu) {
        printf("FAIL %s:%d: cannot create GPU: %s\n", __FILE__, __LINE__, error);
        printf("FAIL: 1 checks, 1 failures\n");
        return 1;
    }
    CHECK(h3i_gpu_recommended_working_set(gpu) > 0, "working set accessor");
    test_command_model(gpu);
    test_tensor_io(gpu);
    test_add_exact(gpu);
    test_silu_mul(gpu);
    test_swiglu(gpu);
    test_rms_norm_bf16(gpu);
    test_rms_norm_f32(gpu);
    test_layer_norm_bf16(gpu);
    test_adaln_gate(gpu);
    /* linear: MPSGraph path (rows>=32, in>=256, out>=256) and custom path */
    linear_check(gpu, 64, 512, 256, 1, 1);
    linear_check(gpu, 8, 64, 64, 0, 1);
    linear_check(gpu, 64, 256, 256, 1, 0);
    linear_check(gpu, 4, 32, 16, 0, 0);
    test_sdpa(gpu);
    h3i_gpu_free(gpu);
    printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "ok", checks,
           failures);
    return failures != 0;
}
