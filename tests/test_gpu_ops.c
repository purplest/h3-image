/* GPU operation checks vs double-precision CPU references (SPEC §6.5):
 * rms/layer ≤2 BF16 steps, AdaLN ≤3, SwiGLU/SiLU-mul ≤1, gate & add exact,
 * linear max err ≤1% of max|ref|, SDPA <0.02. Requires macOS/Metal. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"

#ifdef __APPLE__

#include "bf16_util.h"
#include "gpu_test.h"
#include "io/host.h"

static double rnd(void)
{
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

static void rand_bf16(uint16_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        p[i] = f_bf16((float)rnd());
    }
}

static int max_steps(const uint16_t *a, const uint16_t *b, size_t n)
{
    int worst = 0;
    for (size_t i = 0; i < n; i++) {
        int s = bf16_steps(a[i], b[i]);
        if (s > worst) {
            worst = s;
        }
    }
    return worst;
}

static void test_norms(h3i_gpu *g)
{
    enum { R = 8, W = 5376 };
    uint16_t x[R * W], w[W], y[R * W], ref[R * W];
    rand_bf16(x, R * W);
    for (int i = 0; i < W; i++) {
        w[i] = f_bf16((float)(0.5 + (double)rand() / RAND_MAX));
    }
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(g, x, R * W);
    h3i_gpu_tensor *tw = h3i_gpu_tensor_from_bf16(g, w, W);
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, R * W);

    CHECK(h3i_gpu_begin(g), "begin");
    CHECK(h3i_gpu_rms_norm_bf16(g, ty, tx, tw, R, W, 1e-5f), "rms op");
    CHECK(h3i_gpu_submit(g), "submit: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * W);
    for (int r = 0; r < R; r++) {
        double ss = 0;
        for (int i = 0; i < W; i++) {
            double v = bf16_f(x[r * W + i]);
            ss += v * v;
        }
        double inv = 1.0 / sqrt(ss / W + 1e-5);
        for (int i = 0; i < W; i++) {
            ref[r * W + i] = f_bf16((float)(bf16_f(x[r * W + i]) * inv * bf16_f(w[i])));
        }
    }
    int ms = max_steps(y, ref, R * W);
    CHECK(ms <= 2, "rms_norm_bf16 worst %d steps (≤2)", ms);

    /* layer norm with bias */
    uint16_t b[W];
    rand_bf16(b, W);
    h3i_gpu_tensor *tb = h3i_gpu_tensor_from_bf16(g, b, W);
    CHECK(h3i_gpu_begin(g), "begin ln");
    CHECK(h3i_gpu_layer_norm_bf16(g, ty, tx, tw, tb, R, W, 1e-6f), "layer op");
    CHECK(h3i_gpu_submit(g), "submit ln: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * W);
    for (int r = 0; r < R; r++) {
        double mean = 0;
        for (int i = 0; i < W; i++) {
            mean += bf16_f(x[r * W + i]);
        }
        mean /= W;
        double var = 0;
        for (int i = 0; i < W; i++) {
            double d = bf16_f(x[r * W + i]) - mean;
            var += d * d;
        }
        var /= W;
        double inv = 1.0 / sqrt(var + 1e-6);
        for (int i = 0; i < W; i++) {
            double n = (bf16_f(x[r * W + i]) - mean) * inv;
            ref[r * W + i] = f_bf16((float)(n * bf16_f(w[i]) + bf16_f(b[i])));
        }
    }
    ms = max_steps(y, ref, R * W);
    CHECK(ms <= 2, "layer_norm_bf16 worst %d steps (≤2)", ms);

    h3i_gpu_tensor_free(tx);
    h3i_gpu_tensor_free(tw);
    h3i_gpu_tensor_free(tb);
    h3i_gpu_tensor_free(ty);
}

static void test_adaln_gate(h3i_gpu *g)
{
    enum { R = 16, W = 1024, SLOTS = 6 };
    uint16_t x[R * W], nw[W], mod[2 * SLOTS * W], y[R * W], ref[R * W];
    uint32_t map[R];
    rand_bf16(x, R * W);
    rand_bf16(nw, W);
    rand_bf16(mod, 2 * SLOTS * W);
    for (int r = 0; r < R; r++) {
        map[r] = (uint32_t)(rand() % 2);
    }
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(g, x, R * W);
    h3i_gpu_tensor *tnw = h3i_gpu_tensor_from_bf16(g, nw, W);
    h3i_gpu_tensor *tmod = h3i_gpu_tensor_from_bf16(g, mod, 2 * SLOTS * W);
    h3i_gpu_tensor *tmap = h3i_gpu_tensor_from_u32(g, map, R);
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, R * W);

    CHECK(h3i_gpu_begin(g), "begin adaln");
    CHECK(h3i_gpu_adaln_bf16(g, ty, tx, tnw, tmod, tmap, R, W, SLOTS, 0, 1, 1e-5f), "adaln op");
    CHECK(h3i_gpu_submit(g), "submit adaln: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * W);
    for (int r = 0; r < R; r++) {
        double ss = 0;
        for (int i = 0; i < W; i++) {
            double v = bf16_f(x[r * W + i]);
            ss += v * v;
        }
        double inv = 1.0 / sqrt(ss / W + 1e-5);
        size_t mb = (size_t)map[r] * SLOTS * W;
        for (int i = 0; i < W; i++) {
            double n = bf16_f(x[r * W + i]) * inv * bf16_f(nw[i]);
            double sc = 1.0 + bf16_f(mod[mb + W + i]);
            double sh = bf16_f(mod[mb + i]);
            ref[r * W + i] = f_bf16((float)(n * sc + sh));
        }
    }
    int ms = max_steps(y, ref, R * W);
    CHECK(ms <= 3, "adaln_bf16 worst %d steps (≤3)", ms);

    /* gate: exact vs separately-rounded F32 CPU math */
    uint16_t branch[R * W], gate_mod[2 * SLOTS * W];
    rand_bf16(branch, R * W);
    rand_bf16(gate_mod, 2 * SLOTS * W);
    h3i_gpu_tensor *tbr = h3i_gpu_tensor_from_bf16(g, branch, R * W);
    h3i_gpu_tensor *tgm = h3i_gpu_tensor_from_bf16(g, gate_mod, 2 * SLOTS * W);
    CHECK(h3i_gpu_begin(g), "begin gate");
    CHECK(h3i_gpu_gate_bf16(g, ty, tx, tbr, tgm, tmap, R, W, SLOTS, 2), "gate op");
    CHECK(h3i_gpu_submit(g), "submit gate: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * W);
    int exact = 1;
    for (int r = 0; r < R && exact; r++) {
        size_t mb = (size_t)map[r] * SLOTS * W + 2 * W;
        for (int i = 0; i < W; i++) {
            float gv = bf16_f(gate_mod[mb + i]);
            float a = bf16_f(branch[r * W + i]) * gv;
            uint16_t want = f_bf16(bf16_f(x[r * W + i]) + a);
            if (y[r * W + i] != want) {
                exact = 0;
                break;
            }
        }
    }
    CHECK(exact, "gate_bf16 bit-exact");

    /* add: exact */
    h3i_gpu_tensor *ty2 = h3i_gpu_tensor_new_bf16(g, R * W);
    CHECK(h3i_gpu_begin(g), "begin add");
    CHECK(h3i_gpu_add_bf16(g, ty2, tx, tbr, R * W), "add op");
    CHECK(h3i_gpu_submit(g), "submit add: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty2, y, R * W);
    exact = 1;
    for (size_t i = 0; i < R * W; i++) {
        if (y[i] != f_bf16(bf16_f(x[i]) + bf16_f(branch[i]))) {
            exact = 0;
            break;
        }
    }
    CHECK(exact, "add_bf16 bit-exact");

    h3i_gpu_tensor_free(tx);
    h3i_gpu_tensor_free(tnw);
    h3i_gpu_tensor_free(tmod);
    h3i_gpu_tensor_free(tmap);
    h3i_gpu_tensor_free(ty);
    h3i_gpu_tensor_free(ty2);
    h3i_gpu_tensor_free(tbr);
    h3i_gpu_tensor_free(tgm);
}

static void test_swiglu(h3i_gpu *g)
{
    enum { R = 8, W = 4096 };
    uint16_t fused[R * 2 * W], y[R * W], ref[R * W];
    rand_bf16(fused, R * 2 * W);
    h3i_gpu_tensor *tf = h3i_gpu_tensor_from_bf16(g, fused, R * 2 * W);
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, R * W);
    CHECK(h3i_gpu_begin(g), "begin swiglu");
    CHECK(h3i_gpu_swiglu_bf16(g, ty, tf, R, W), "swiglu op");
    CHECK(h3i_gpu_submit(g), "submit swiglu: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * W);
    for (int r = 0; r < R; r++) {
        for (int i = 0; i < W; i++) {
            double gate = bf16_f(fused[r * 2 * W + i]);
            double up = bf16_f(fused[r * 2 * W + W + i]);
            double s = gate / (1.0 + exp(-gate));
            ref[r * W + i] = f_bf16((float)(s * up));
        }
    }
    int ms = max_steps(y, ref, R * W);
    CHECK(ms <= 1, "swiglu_bf16 worst %d steps (≤1)", ms);

    /* silu_mul */
    uint16_t up[R * W];
    rand_bf16(up, R * W);
    h3i_gpu_tensor *tu = h3i_gpu_tensor_from_bf16(g, up, R * W);
    CHECK(h3i_gpu_begin(g), "begin silu_mul");
    CHECK(h3i_gpu_silu_mul_bf16(g, ty, tf, tu, R * W), "silu_mul op");
    CHECK(h3i_gpu_submit(g), "submit silu_mul: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * W);
    for (size_t i = 0; i < R * W; i++) {
        double gate = bf16_f(fused[i]);
        double s = gate / (1.0 + exp(-gate));
        ref[i] = f_bf16((float)(s * bf16_f(up[i])));
    }
    ms = max_steps(y, ref, R * W);
    CHECK(ms <= 1, "silu_mul_bf16 worst %d steps (≤1)", ms);
    h3i_gpu_tensor_free(tf);
    h3i_gpu_tensor_free(tu);
    h3i_gpu_tensor_free(ty);
}

static void test_linear(h3i_gpu *g)
{
    /* custom small-shape BF16 path */
    enum { R = 8, IN = 64, OUT = 32 };
    uint16_t x[R * IN], W[OUT * IN], b[OUT], y[R * OUT];
    rand_bf16(x, R * IN);
    rand_bf16(W, OUT * IN);
    rand_bf16(b, OUT);
    h3i_gpu_tensor *tx = h3i_gpu_tensor_from_bf16(g, x, R * IN);
    h3i_gpu_tensor *tW = h3i_gpu_tensor_from_bf16(g, W, OUT * IN);
    h3i_gpu_tensor *tb = h3i_gpu_tensor_from_bf16(g, b, OUT);
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, R * OUT);
    CHECK(h3i_gpu_begin(g), "begin lin");
    CHECK(h3i_gpu_linear_bf16(g, ty, tx, tW, tb, R, IN, OUT), "linear op");
    CHECK(h3i_gpu_submit(g), "submit lin: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, R * OUT);
    double maxref = 0, maxerr = 0;
    for (int r = 0; r < R; r++) {
        for (int o = 0; o < OUT; o++) {
            double acc = 0;
            for (int i = 0; i < IN; i++) {
                acc += (double)bf16_f(x[r * IN + i]) * bf16_f(W[o * IN + i]);
            }
            acc += bf16_f(b[o]);
            double got = bf16_f(y[r * OUT + o]);
            if (fabs(acc) > maxref) {
                maxref = fabs(acc);
            }
            if (fabs(got - acc) > maxerr) {
                maxerr = fabs(got - acc);
            }
        }
    }
    CHECK(maxerr <= 0.01 * maxref, "linear_bf16 custom err %.4g (≤1%% of %.4g)", maxerr, maxref);

    /* MPSGraph path */
    enum { R2 = 64, IN2 = 512, OUT2 = 256 };
    static uint16_t x2[R2 * IN2], W2[OUT2 * IN2], b2[OUT2], y2[R2 * OUT2];
    rand_bf16(x2, R2 * IN2);
    rand_bf16(W2, OUT2 * IN2);
    rand_bf16(b2, OUT2);
    h3i_gpu_tensor *tx2 = h3i_gpu_tensor_from_bf16(g, x2, R2 * IN2);
    h3i_gpu_tensor *tW2 = h3i_gpu_tensor_from_bf16(g, W2, OUT2 * IN2);
    h3i_gpu_tensor *tb2 = h3i_gpu_tensor_from_bf16(g, b2, OUT2);
    h3i_gpu_tensor *ty2 = h3i_gpu_tensor_new_bf16(g, R2 * OUT2);
    CHECK(h3i_gpu_begin(g), "begin lin2");
    CHECK(h3i_gpu_linear_bf16(g, ty2, tx2, tW2, tb2, R2, IN2, OUT2), "linear mps op");
    CHECK(h3i_gpu_submit(g), "submit lin2: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty2, y2, R2 * OUT2);
    maxref = 0;
    maxerr = 0;
    for (int r = 0; r < R2; r++) {
        for (int o = 0; o < OUT2; o++) {
            double acc = 0;
            for (int i = 0; i < IN2; i++) {
                acc += (double)bf16_f(x2[r * IN2 + i]) * bf16_f(W2[o * IN2 + i]);
            }
            acc += bf16_f(b2[o]);
            double got = bf16_f(y2[r * OUT2 + o]);
            if (fabs(acc) > maxref) {
                maxref = fabs(acc);
            }
            if (fabs(got - acc) > maxerr) {
                maxerr = fabs(got - acc);
            }
        }
    }
    CHECK(maxerr <= 0.01 * maxref, "linear_bf16 MPS err %.4g (≤1%% of %.4g)", maxerr, maxref);

    /* F32 naive path */
    enum { R3 = 4, IN3 = 16, OUT3 = 8 };
    float xf[R3 * IN3], Wf[OUT3 * IN3], bf[OUT3], yf[R3 * OUT3];
    for (int i = 0; i < R3 * IN3; i++) xf[i] = (float)rnd();
    for (int i = 0; i < OUT3 * IN3; i++) Wf[i] = (float)rnd();
    for (int i = 0; i < OUT3; i++) bf[i] = (float)rnd();
    h3i_gpu_tensor *txf = h3i_gpu_tensor_from_f32(g, xf, R3 * IN3);
    h3i_gpu_tensor *tWf = h3i_gpu_tensor_from_f32(g, Wf, OUT3 * IN3);
    h3i_gpu_tensor *tbf = h3i_gpu_tensor_from_f32(g, bf, OUT3);
    h3i_gpu_tensor *tyf = h3i_gpu_tensor_new_f32(g, R3 * OUT3);
    CHECK(h3i_gpu_begin(g), "begin linf");
    CHECK(h3i_gpu_linear_f32(g, tyf, txf, tWf, tbf, R3, IN3, OUT3), "linear f32 op");
    CHECK(h3i_gpu_submit(g), "submit linf: %s", h3i_gpu_error(g));
    h3i_gpu_read_f32(tyf, yf, R3 * OUT3);
    maxref = 0;
    maxerr = 0;
    for (int r = 0; r < R3; r++) {
        for (int o = 0; o < OUT3; o++) {
            double acc = 0;
            for (int i = 0; i < IN3; i++) {
                acc += (double)xf[r * IN3 + i] * Wf[o * IN3 + i];
            }
            acc += bf[o];
            if (fabs(acc) > maxref) maxref = fabs(acc);
            if (fabs(yf[r * OUT3 + o] - acc) > maxerr) maxerr = fabs(yf[r * OUT3 + o] - acc);
        }
    }
    CHECK(maxerr <= 1e-4 * maxref, "linear_f32 err %.4g", maxerr);

    h3i_gpu_tensor_free(tx); h3i_gpu_tensor_free(tW); h3i_gpu_tensor_free(tb); h3i_gpu_tensor_free(ty);
    h3i_gpu_tensor_free(tx2); h3i_gpu_tensor_free(tW2); h3i_gpu_tensor_free(tb2); h3i_gpu_tensor_free(ty2);
    h3i_gpu_tensor_free(txf); h3i_gpu_tensor_free(tWf); h3i_gpu_tensor_free(tbf); h3i_gpu_tensor_free(tyf);
}

static void test_sdpa(h3i_gpu *g)
{
    enum { S = 64, H = 4, D = 64 };
    static uint16_t q[S * H * D], k[S * H * D], v[S * H * D], y[S * H * D];
    rand_bf16(q, S * H * D);
    rand_bf16(k, S * H * D);
    rand_bf16(v, S * H * D);
    h3i_gpu_tensor *tq = h3i_gpu_tensor_from_bf16(g, q, S * H * D);
    h3i_gpu_tensor *tk = h3i_gpu_tensor_from_bf16(g, k, S * H * D);
    h3i_gpu_tensor *tv = h3i_gpu_tensor_from_bf16(g, v, S * H * D);
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, S * H * D);
    float scale = 1.0f / sqrtf((float)D);
    CHECK(h3i_gpu_begin(g), "begin sdpa");
    CHECK(h3i_gpu_sdpa_bf16(g, ty, tq, tk, tv, S, H, D, scale), "sdpa op");
    CHECK(h3i_gpu_submit(g), "submit sdpa: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y, S * H * D);
    double maxerr = 0;
    for (int t = 0; t < S; t++) {
        for (int h = 0; h < H; h++) {
            double scores[S], mx = -1e30, sum = 0;
            for (int j = 0; j < S; j++) {
                double dot = 0;
                for (int d = 0; d < D; d++) {
                    dot += (double)bf16_f(q[(t * H + h) * D + d]) * bf16_f(k[(j * H + h) * D + d]);
                }
                scores[j] = dot * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            for (int j = 0; j < S; j++) {
                scores[j] = exp(scores[j] - mx);
                sum += scores[j];
            }
            for (int d = 0; d < D; d++) {
                double acc = 0;
                for (int j = 0; j < S; j++) {
                    acc += scores[j] / sum * bf16_f(v[(j * H + h) * D + d]);
                }
                double err = fabs(bf16_f(y[(t * H + h) * D + d]) - acc);
                if (err > maxerr) maxerr = err;
            }
        }
    }
    CHECK(maxerr < 0.02, "sdpa_bf16 max err %.4g (<0.02)", maxerr);
    h3i_gpu_tensor_free(tq); h3i_gpu_tensor_free(tk); h3i_gpu_tensor_free(tv); h3i_gpu_tensor_free(ty);
}

static void test_rope_kernels(h3i_gpu *g)
{
    /* rope_text: bit-exact vs separately-rounded F32 CPU math */
    enum { R = 8, QH = 8, KVH = 2, D = 128 };
    static uint16_t q[R * QH * D], k[R * KVH * D], qy[R * QH * D], ky[R * KVH * D];
    static float cs[R * (D / 2)], sn[R * (D / 2)];
    rand_bf16(q, R * QH * D);
    rand_bf16(k, R * KVH * D);
    for (int i = 0; i < R * (D / 2); i++) {
        double a = rnd() * 3.0;
        cs[i] = bf16_f(f_bf16((float)cos(a)));
        sn[i] = bf16_f(f_bf16((float)sin(a)));
    }
    memcpy(qy, q, sizeof q);
    memcpy(ky, k, sizeof k);
    h3i_gpu_tensor *tq = h3i_gpu_tensor_from_bf16(g, qy, R * QH * D);
    h3i_gpu_tensor *tk = h3i_gpu_tensor_from_bf16(g, ky, R * KVH * D);
    h3i_gpu_tensor *tc = h3i_gpu_tensor_from_f32(g, cs, R * (D / 2));
    h3i_gpu_tensor *ts = h3i_gpu_tensor_from_f32(g, sn, R * (D / 2));
    CHECK(h3i_gpu_begin(g), "begin rope");
    CHECK(h3i_gpu_rope_text_bf16(g, tq, tk, tc, ts, R, QH, KVH, D), "rope op");
    CHECK(h3i_gpu_submit(g), "submit rope: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(tq, qy, R * QH * D);
    h3i_gpu_read_bf16(tk, ky, R * KVH * D);
    int exact = 1;
    for (int r = 0; r < R && exact; r++) {
        for (int h = 0; h < QH; h++) {
            for (int d = 0; d < D / 2; d++) {
                float c = cs[r * (D / 2) + d], s = sn[r * (D / 2) + d];
                float a = bf16_f(q[(r * QH + h) * D + d]);
                float b = bf16_f(q[(r * QH + h) * D + d + D / 2]);
                if (qy[(r * QH + h) * D + d] != f_bf16(a * c - b * s) ||
                    qy[(r * QH + h) * D + d + D / 2] != f_bf16(b * c + a * s)) {
                    exact = 0;
                }
            }
        }
        for (int h = 0; h < KVH && exact; h++) {
            for (int d = 0; d < D / 2; d++) {
                float c = cs[r * (D / 2) + d], s = sn[r * (D / 2) + d];
                float a = bf16_f(k[(r * KVH + h) * D + d]);
                float b = bf16_f(k[(r * KVH + h) * D + d + D / 2]);
                if (ky[(r * KVH + h) * D + d] != f_bf16(a * c - b * s) ||
                    ky[(r * KVH + h) * D + d + D / 2] != f_bf16(b * c + a * s)) {
                    exact = 0;
                }
            }
        }
    }
    CHECK(exact, "rope_text_bf16 bit-exact");
    h3i_gpu_tensor_free(tq); h3i_gpu_tensor_free(tk); h3i_gpu_tensor_free(tc); h3i_gpu_tensor_free(ts);

    /* qkv_rope grouped, rope_half 48, within 2 BF16 steps of CPU F32 math */
    enum { R2 = 8, H2 = 56, D2 = 128, RH = 48 };
    static uint16_t qkv[R2 * 3 * H2 * D2], qq[R2 * H2 * D2], kk[R2 * H2 * D2], vv[R2 * H2 * D2];
    static uint16_t cb[R2 * 2 * RH], sb[R2 * 2 * RH], qn[D2], kn[D2];
    rand_bf16(qkv, R2 * 3 * H2 * D2);
    rand_bf16(cb, R2 * 2 * RH);
    rand_bf16(sb, R2 * 2 * RH);
    rand_bf16(qn, D2);
    rand_bf16(kn, D2);
    h3i_gpu_tensor *tqkv = h3i_gpu_tensor_from_bf16(g, qkv, R2 * 3 * H2 * D2);
    h3i_gpu_tensor *tqo = h3i_gpu_tensor_new_bf16(g, R2 * H2 * D2);
    h3i_gpu_tensor *tko = h3i_gpu_tensor_new_bf16(g, R2 * H2 * D2);
    h3i_gpu_tensor *tvo = h3i_gpu_tensor_new_bf16(g, R2 * H2 * D2);
    h3i_gpu_tensor *tcb = h3i_gpu_tensor_from_bf16(g, cb, R2 * 2 * RH);
    h3i_gpu_tensor *tsb = h3i_gpu_tensor_from_bf16(g, sb, R2 * 2 * RH);
    h3i_gpu_tensor *tqn = h3i_gpu_tensor_from_bf16(g, qn, D2);
    h3i_gpu_tensor *tkn = h3i_gpu_tensor_from_bf16(g, kn, D2);
    CHECK(h3i_gpu_begin(g), "begin qkv");
    CHECK(h3i_gpu_qkv_rope_bf16(g, tqo, tko, tvo, tqkv, tcb, tsb, tqn, tkn,
                                R2, H2, D2, RH, 1e-5f, 1), "qkv rope op");
    CHECK(h3i_gpu_submit(g), "submit qkv: %s", h3i_gpu_error(g));
    h3i_gpu_read_bf16(tqo, qq, R2 * H2 * D2);
    h3i_gpu_read_bf16(tko, kk, R2 * H2 * D2);
    h3i_gpu_read_bf16(tvo, vv, R2 * H2 * D2);
    int worst = 0;
    for (int r = 0; r < R2; r++) {
        for (int h = 0; h < H2; h++) {
            const uint16_t *src = qkv + (size_t)r * (3 * H2 * D2) + (size_t)h * (3 * D2);
            float qv[D2], kv[D2];
            float ssq = 0, ssk = 0;
            for (int d = 0; d < D2; d++) {
                qv[d] = bf16_f(src[d]);
                kv[d] = bf16_f(src[D2 + d]);
                ssq = fmaf(qv[d], qv[d], ssq);
                ssk = fmaf(kv[d], kv[d], ssk);
            }
            float invq = rsqrtf(ssq / D2 + 1e-5f);
            float invk = rsqrtf(ssk / D2 + 1e-5f);
            for (int d = 0; d < D2; d++) {
                qv[d] = qv[d] * invq * bf16_f(qn[d]);
                kv[d] = kv[d] * invk * bf16_f(kn[d]);
            }
            size_t ob = ((size_t)r * H2 + h) * D2;
            for (int d = 0; d < D2; d++) {
                float qo = qv[d], ko = kv[d];
                if (d < RH) {
                    float c = bf16_f(cb[(size_t)r * 2 * RH + d]);
                    float s = bf16_f(sb[(size_t)r * 2 * RH + d]);
                    qo = qv[d] * c - qv[d + RH] * s;
                    ko = kv[d] * c - kv[d + RH] * s;
                    float qh2 = qv[d + RH] * c + qv[d] * s;
                    float kh2 = kv[d + RH] * c + kv[d] * s;
                    int s1 = bf16_steps(qq[ob + d + RH], f_bf16(qh2));
                    int s2 = bf16_steps(kk[ob + d + RH], f_bf16(kh2));
                    if (s1 > worst) worst = s1;
                    if (s2 > worst) worst = s2;
                } else if (d < 2 * RH) {
                    continue;
                }
                int s1 = bf16_steps(qq[ob + d], f_bf16(qo));
                int s2 = bf16_steps(kk[ob + d], f_bf16(ko));
                if (s1 > worst) worst = s1;
                if (s2 > worst) worst = s2;
                if (vv[ob + d] != src[2 * D2 + d]) {
                    worst = 1000000;
                }
            }
        }
    }
    CHECK(worst <= 2, "qkv_rope_bf16 grouped worst %d steps (≤2)", worst);
    h3i_gpu_tensor_free(tqkv); h3i_gpu_tensor_free(tqo); h3i_gpu_tensor_free(tko);
    h3i_gpu_tensor_free(tvo); h3i_gpu_tensor_free(tcb); h3i_gpu_tensor_free(tsb);
    h3i_gpu_tensor_free(tqn); h3i_gpu_tensor_free(tkn);
}

static void test_vae_helpers(h3i_gpu *g)
{
    /* reflect pad: exact */
    enum { H = 5, W = 7, C = 3, P = 1 };
    float in[H * W * C], out[3 * (H + 2 * P) * (W + 2 * P) * C];
    for (int i = 0; i < H * W * C; i++) {
        in[i] = (float)i;
    }
    h3i_gpu_tensor *ti = h3i_gpu_tensor_from_f32(g, in, H * W * C);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_f32(g, 3 * (H + 2 * P) * (W + 2 * P) * C);
    CHECK(h3i_gpu_begin(g), "begin pad");
    CHECK(h3i_gpu_vae_encoder_pad_f32(g, to, ti, H, W, C, P), "pad op");
    CHECK(h3i_gpu_submit(g), "submit pad: %s", h3i_gpu_error(g));
    h3i_gpu_read_f32(to, out, 3 * (H + 2 * P) * (W + 2 * P) * C);
    int ok = 1;
    int OH = H + 2 * P, OW = W + 2 * P;
    for (int f = 0; f < 3 && ok; f++) {
        for (int y = 0; y < OH && ok; y++) {
            for (int x = 0; x < OW && ok; x++) {
                for (int c = 0; c < C; c++) {
                    float want = 0;
                    if (f == 2) {
                        int sy = y - P, sx = x - P;
                        if (sy < 0) sy = -sy;
                        if (sy >= H) sy = 2 * H - sy - 2;
                        if (sx < 0) sx = -sx;
                        if (sx >= W) sx = 2 * W - sx - 2;
                        want = in[(sy * W + sx) * C + c];
                    }
                    if (out[((size_t)f * OH + y) * OW * C + (size_t)x * C + c] != want) {
                        ok = 0;
                    }
                }
            }
        }
    }
    CHECK(ok, "vae_encoder_pad reflect exact");

    /* conv3d small case vs double reference */
    enum { N = 1, D = 3, HH = 6, WW = 6, CI = 2, CO = 3, K = 3 };
    float cin[N * D * HH * WW * CI], cw[CO * CI * K * K * K], cb[CO];
    float cout[N * 1 * (HH - K + 1) * (WW - K + 1) * CO];
    for (size_t i = 0; i < sizeof cin / sizeof cin[0]; i++) cin[i] = (float)rnd();
    for (size_t i = 0; i < sizeof cw / sizeof cw[0]; i++) cw[i] = (float)rnd();
    for (int i = 0; i < CO; i++) cb[i] = (float)rnd();
    h3i_gpu_tensor *tci = h3i_gpu_tensor_from_f32(g, cin, N * D * HH * WW * CI);
    h3i_gpu_tensor *tcw = h3i_gpu_tensor_from_f32(g, cw, CO * CI * K * K * K);
    h3i_gpu_tensor *tcb = h3i_gpu_tensor_from_f32(g, cb, CO);
    size_t on = N * 1 * (HH - K + 1) * (WW - K + 1) * CO;
    h3i_gpu_tensor *tco = h3i_gpu_tensor_new_f32(g, on);
    CHECK(h3i_gpu_begin(g), "begin conv");
    CHECK(h3i_gpu_conv3d_f32(g, tco, tci, tcw, tcb, N, D, HH, WW, CI, CO, K, K, K, 1, 1, 1),
          "conv3d op");
    CHECK(h3i_gpu_submit(g), "submit conv: %s", h3i_gpu_error(g));
    h3i_gpu_read_f32(tco, cout, on);
    double maxerr = 0, maxref = 0;
    int OH2 = HH - K + 1, OW2 = WW - K + 1;
    for (int oy = 0; oy < OH2; oy++) {
        for (int ox = 0; ox < OW2; ox++) {
            for (int o = 0; o < CO; o++) {
                double acc = 0;
                for (int tz = 0; tz < K; tz++) {
                    for (int ty = 0; ty < K; ty++) {
                        for (int tx = 0; tx < K; tx++) {
                            for (int i = 0; i < CI; i++) {
                                acc += (double)cin[(((size_t)tz * HH + oy + ty) * WW + ox + tx) * CI + i]
                                       * cw[((o * CI + i) * K * K + tz * K + ty) * K + tx];
                            }
                        }
                    }
                }
                acc += cb[o];
                double got = cout[((size_t)oy * OW2 + ox) * CO + o];
                if (fabs(acc) > maxref) maxref = fabs(acc);
                if (fabs(got - acc) > maxerr) maxerr = fabs(got - acc);
            }
        }
    }
    CHECK(maxerr <= 1e-4 * maxref, "conv3d_f32 err %.4g", maxerr);
    h3i_gpu_tensor_free(ti); h3i_gpu_tensor_free(to);
    h3i_gpu_tensor_free(tci); h3i_gpu_tensor_free(tcw); h3i_gpu_tensor_free(tcb); h3i_gpu_tensor_free(tco);
}

int main(void)
{
    srand(7);
    h3i_gpu *g = test_gpu_create();
    if (g == NULL) {
        printf("ok: 0 checks, 0 failures\n");
        return 0;
    }
    test_norms(g);
    test_adaln_gate(g);
    test_swiglu(g);
    test_linear(g);
    test_sdpa(g);
    test_rope_kernels(g);
    test_vae_helpers(g);
    h3i_gpu_free(g);
    TEST_RESULT();
}

#else /* !__APPLE__ */

int main(void)
{
    printf("skip: GPU ops tests require macOS/Metal\n");
    printf("ok: 0 checks, 0 failures\n");
    return 0;
}

#endif
