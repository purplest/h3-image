/* A9 gate: race-free causal GQA vs CPU double reference at 43 rows (>32-key
 * race case) and 8200 rows (>4096-key chunked online softmax), worst ≤1 BF16
 * step, bit-repeatable over runs. Requires macOS/Metal. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"

#ifdef __APPLE__

#include "bf16_util.h"
#include "gpu_test.h"

static double rnd(void)
{
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

/* Double-precision causal GQA reference, layout [seq][heads][hd]. */
static void cpu_causal(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                       uint16_t *out, size_t seq, size_t qh, size_t kvh, size_t hd,
                       double scale)
{
    size_t group = qh / kvh;
    double *scores = malloc(seq * sizeof(double));
    for (size_t t = 0; t < seq; t++) {
        for (size_t h = 0; h < qh; h++) {
            size_t kv = h / group;
            double mx = -1e300;
            for (size_t j = 0; j <= t; j++) {
                double dot = 0;
                for (size_t d = 0; d < hd; d++) {
                    dot += (double)bf16_f(q[(t * qh + h) * hd + d]) *
                           bf16_f(k[(j * kvh + kv) * hd + d]);
                }
                scores[j] = dot * scale;
                if (scores[j] > mx) {
                    mx = scores[j];
                }
            }
            double sum = 0;
            for (size_t j = 0; j <= t; j++) {
                scores[j] = exp(scores[j] - mx);
                sum += scores[j];
            }
            for (size_t d = 0; d < hd; d++) {
                double acc = 0;
                for (size_t j = 0; j <= t; j++) {
                    acc += scores[j] / sum * bf16_f(v[(j * kvh + kv) * hd + d]);
                }
                out[(t * qh + h) * hd + d] = f_bf16((float)acc);
            }
        }
    }
    free(scores);
}

static void run_case(h3i_gpu *g, size_t seq, size_t qh, size_t kvh, size_t hd,
                     const char *name)
{
    size_t nq = seq * qh * hd, nk = seq * kvh * hd;
    uint16_t *q = malloc(nq * 2), *k = malloc(nk * 2), *v = malloc(nk * 2);
    uint16_t *y1 = malloc(nq * 2), *y2 = malloc(nq * 2), *ref = malloc(nq * 2);
    for (size_t i = 0; i < nq; i++) q[i] = f_bf16((float)rnd());
    for (size_t i = 0; i < nk; i++) { k[i] = f_bf16((float)rnd()); v[i] = f_bf16((float)rnd()); }

    h3i_gpu_tensor *tq = h3i_gpu_tensor_from_bf16(g, q, nq);
    h3i_gpu_tensor *tk = h3i_gpu_tensor_from_bf16(g, k, nk);
    h3i_gpu_tensor *tv = h3i_gpu_tensor_from_bf16(g, v, nk);
    h3i_gpu_tensor *ty = h3i_gpu_tensor_new_bf16(g, nq);
    float scale = (float)(1.0 / sqrt((double)hd));

    CHECK(h3i_gpu_begin(g), "%s begin", name);
    CHECK(h3i_gpu_gqa_causal_bf16(g, ty, tq, tk, tv, seq, qh, kvh, hd, scale),
          "%s op: %s", name, h3i_gpu_error(g));
    CHECK(h3i_gpu_submit(g), "%s submit: %s", name, h3i_gpu_error(g));
    h3i_gpu_read_bf16(ty, y1, nq);

    /* repeatability */
    CHECK(h3i_gpu_begin(g), "%s begin 2", name);
    CHECK(h3i_gpu_gqa_causal_bf16(g, ty, tq, tk, tv, seq, qh, kvh, hd, scale), "%s op 2", name);
    CHECK(h3i_gpu_submit(g), "%s submit 2", name);
    h3i_gpu_read_bf16(ty, y2, nq);
    CHECK(memcmp(y1, y2, nq * 2) == 0, "%s bit-repeatable", name);

    cpu_causal(q, k, v, ref, seq, qh, kvh, hd, 1.0 / sqrt((double)hd));
    int worst = 0;
    size_t worst_i = 0;
    for (size_t i = 0; i < nq; i++) {
        int s = bf16_steps(y1[i], ref[i]);
        if (s > worst) {
            worst = s;
            worst_i = i;
        }
    }
    CHECK(worst <= 1, "%s worst %d BF16 steps at %zu (≤1)", name, worst, worst_i);

    h3i_gpu_tensor_free(tq); h3i_gpu_tensor_free(tk);
    h3i_gpu_tensor_free(tv); h3i_gpu_tensor_free(ty);
    free(q); free(k); free(v); free(y1); free(y2); free(ref);
}

int main(void)
{
    srand(11);
    h3i_gpu *g = test_gpu_create();
    if (g == NULL) {
        printf("ok: 0 checks, 0 failures\n");
        return 0;
    }
    run_case(g, 43, 8, 2, 128, "seq43");
    run_case(g, 8200, 1, 1, 128, "seq8200");
    h3i_gpu_free(g);
    TEST_RESULT();
}

#else

int main(void)
{
    printf("skip: attention tests require macOS/Metal\n");
    printf("ok: 0 checks, 0 failures\n");
    return 0;
}

#endif
