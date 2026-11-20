/* test_modulation — SPEC §6.5 (Phase 5 gate, A5/A6):
 * - row maps: exact formula on a synthetic layout (no weights)
 * - PDD heads: GGUF base+offset reconstruction vs official absolute heads,
 *   rel L2 < 1e-5 (A6; measured ~3e-10)
 * - pruned AdaLN tables vs the official time-MLP on block 0 + final:
 *   worst per-row rel L2 < 0.05; applying SiLU to the interpolated
 *   embeddings (the classic mistake) must be > 4x worse (A5)
 * Weight paths: $H3I_DIT_GGUF, $H3I_PDD_HEADS, $H3I_OFFICIAL_CHECK or the
 * spec defaults under ~/models. Sections skip when their files are absent.
 * The official check fixture is built by tools/fetch_tensors.py. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app/error.h"
#include "builders.h"
#include "check.h"
#include "gpu_test.h"
#include "io/host.h"
#include "model/layout.h"
#include "model/modulation.h"
#include "weights/gguf.h"
#include "weights/source.h"

static void test_maps(void) {
    char err[512];
    int ref_lh[1] = {4}, ref_lw[1] = {4};
    h3i_layout layout = {0};
    CHECK(h3i_layout_build(4, NULL, 1, ref_lh, ref_lw, 8, 8, &layout, err,
                           sizeof err),
          "layout: %s", err);
    CHECK(layout.sequence == 4 + 4 + 16, "sequence %zu", layout.sequence);
    uint32_t row_maps[6 * 24], final_maps[6 * 16];
    CHECK(h3i_dit_prepare_maps(&layout, 6, row_maps, final_maps, err,
                               sizeof err),
          "maps: %s", err);
    for (int s = 0; s < 6; s++) {
        CHECK(row_maps[s * 24 + 0] == (uint32_t)s * 3 + 1, "text row map");
        CHECK(row_maps[s * 24 + 4] == 6 * 3 + 0, "reference row map");
        CHECK(row_maps[s * 24 + 7] == 18, "last reference row map");
        CHECK(row_maps[s * 24 + 8] == (uint32_t)s * 3, "target row map");
        for (int r = 0; r < 16; r++) {
            CHECK(final_maps[s * 16 + r] == (uint32_t)s, "final map");
        }
    }
    h3i_layout_free(&layout);
}

static const char *env_or_default(const char *env, const char *suffix) {
    const char *value = getenv(env);
    if (value && *value) return value;
    static char path[4096];
    const char *home = getenv("HOME");
    snprintf(path, sizeof path, "%s/models/%s", home ? home : "", suffix);
    return path;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

static void test_heads(void) {
    char err[512];
    const char *gguf_path = env_or_default(
        "H3I_DIT_GGUF", "MiniMax-H3-Image-GGUF/minimax_h3_image_pruned-Q4_K_M.gguf");
    const char *heads_path = env_or_default(
        "H3I_PDD_HEADS", "MiniMax-H3-Image/pdd_fused_heads.safetensors");
    if (!file_exists(gguf_path) || !file_exists(heads_path)) {
        printf("skip: PDD heads comparison (needs the DiT GGUF + pdd_fused_heads)\n");
        return;
    }
    h3i_source gguf = {0}, official = {0};
    CHECK(h3i_source_add(&gguf, gguf_path, err, sizeof err), "open GGUF: %s", err);
    CHECK(h3i_source_add(&official, heads_path, err, sizeof err),
          "open heads: %s", err);
    h3i_heads a = {0}, b = {0};
    CHECK(h3i_heads_load(&gguf, &a, err, sizeof err), "GGUF heads: %s", err);
    CHECK(h3i_heads_load(&official, &b, err, sizeof err),
          "official heads: %s", err);
    if (a.weight && b.weight) {
        double rw = rel_l2(a.weight, b.weight, (size_t)6 * 96 * 5376);
        double rb = rel_l2(a.bias, b.bias, (size_t)6 * 96);
        CHECK(rw < 1e-5, "heads weight rel L2 %g (want < 1e-5)", rw);
        CHECK(rb < 1e-5, "heads bias rel L2 %g (want < 1e-5)", rb);
        printf("heads rel L2: weight %.3g bias %.3g\n", rw, rb);
    }
    h3i_heads_free(&a);
    h3i_heads_free(&b);
    h3i_source_free(&gguf);
    h3i_source_free(&official);
}

#ifdef __APPLE__

static float *read_table(const h3i_gpu_tensor *t, size_t elements) {
    uint16_t *bits = malloc(elements * sizeof(uint16_t));
    float *values = malloc(elements * sizeof(float));
    if (!bits || !values ||
        !h3i_gpu_tensor_read_bf16(t, bits, elements)) {
        free(bits);
        free(values);
        return NULL;
    }
    for (size_t i = 0; i < elements; i++) values[i] = h3i_bf16_to_f32(bits[i]);
    free(bits);
    return values;
}

static double worst_row_rel(const float *a, const float *b, uint32_t rows,
                            uint32_t outputs) {
    double worst = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        double rel = rel_l2(a + (size_t)r * outputs, b + (size_t)r * outputs,
                            outputs);
        if (rel > worst) worst = rel;
    }
    return worst;
}

static void test_tables(void) {
    char err[512];
    const char *gguf_path = env_or_default(
        "H3I_DIT_GGUF", "MiniMax-H3-Image-GGUF/minimax_h3_image_pruned-Q4_K_M.gguf");
    const char *check_path = env_or_default(
        "H3I_OFFICIAL_CHECK", "MiniMax-H3-Image/check/official_block0.safetensors");
    if (!file_exists(gguf_path) || !file_exists(check_path)) {
        printf("skip: AdaLN comparison (needs the DiT GGUF + check/official_block0.safetensors)\n");
        return;
    }
    h3i_gpu *gpu = test_gpu_create();
    if (!gpu) return;
    h3i_source gguf = {0}, check = {0};
    CHECK(h3i_source_add(&gguf, gguf_path, err, sizeof err), "open GGUF: %s", err);
    CHECK(h3i_source_add(&check, check_path, err, sizeof err),
          "open check: %s", err);
    h3i_schedule schedule;
    CHECK(h3i_schedule_build(NULL, &schedule, err, sizeof err), "schedule");

    h3i_modulation *pruned =
        h3i_modulation_build(gpu, &gguf, &schedule, 1, err, sizeof err);
    CHECK(pruned != NULL, "pruned build: %s", err);
    if (!pruned) goto cleanup;
    CHECK(h3i_modulation_is_pruned(pruned) == 1, "pruned detection");
    CHECK(h3i_modulation_time_rows(pruned) == 7, "time rows");

    /* Official block 0 + final via the exposed building blocks (the full
     * 50-block official build would need every adaln projection). */
    h3i_gpu_tensor *act = h3i_time_act(gpu, &check, schedule.t, 7, err,
                                       sizeof err);
    CHECK(act != NULL, "official time MLP: %s", err);
    h3i_gpu_tensor *ob0 = NULL, *ofin = NULL;
    CHECK(act && h3i_adaln_project(gpu, &check, act, 7,
                                   "blocks.0.adaln_proj.linear.weight",
                                   "blocks.0.adaln_proj.linear.bias",
                                   H3I_MOD_BLOCK_OUTPUT, &ob0, err, sizeof err),
          "official block0: %s", err);
    CHECK(act && h3i_adaln_project(gpu, &check, act, 7,
                                   "final_layer.adaln_proj.linear.weight",
                                   "final_layer.adaln_proj.linear.bias",
                                   H3I_MOD_FINAL_OUTPUT, &ofin, err, sizeof err),
          "official final: %s", err);
    if (!ob0 || !ofin) goto cleanup_mod;

    size_t n0 = (size_t)7 * H3I_MOD_BLOCK_OUTPUT;
    size_t nf = (size_t)7 * H3I_MOD_FINAL_OUTPUT;
    float *p0 = read_table(h3i_modulation_block(pruned, 0), n0);
    float *pf = read_table(h3i_modulation_final(pruned), nf);
    float *o0 = read_table(ob0, n0);
    float *of = read_table(ofin, nf);
    CHECK(p0 && pf && o0 && of, "table readback");
    if (p0 && pf && o0 && of) {
        double w0 = worst_row_rel(p0, o0, 7, H3I_MOD_BLOCK_OUTPUT);
        double wf = worst_row_rel(pf, of, 7, H3I_MOD_FINAL_OUTPUT);
        CHECK(w0 < 0.05, "pruned block0 worst rel %g (want < 0.05)", w0);
        CHECK(wf < 0.05, "pruned final worst rel %g (want < 0.05)", wf);
        printf("pruned vs official worst rel L2: block0 %.4f final %.4f\n",
               w0, wf);

        /* SiLU control: recompute block 0 host-side from the GGUF with SiLU
         * wrongly applied to the interpolated embeddings (SPEC A5). */
        const h3i_tensor_rec *tt = h3i_source_find(&gguf, "adaln_t_table");
        CHECK(tt && tt->ndim == 2, "adaln_t_table present");
        if (tt && tt->ndim == 2) {
            uint64_t G = tt->shape[0], D = tt->shape[1];
            float *table = h3i_source_read_f32(&gguf, "adaln_t_table", G * D,
                                               err, sizeof err);
            float *w = h3i_source_read_f32(
                &gguf, "blocks.0.adaln_proj.linear.weight",
                (uint64_t)H3I_MOD_BLOCK_OUTPUT * D, err, sizeof err);
            float *bias = h3i_source_read_f32(
                &gguf, "blocks.0.adaln_proj.linear.bias",
                H3I_MOD_BLOCK_OUTPUT, err, sizeof err);
            float *ctrl = malloc(n0 * sizeof(float));
            if (table && w && bias && ctrl) {
                for (uint32_t r = 0; r < 7; r++) {
                    float tc = schedule.t[r];
                    if (tc < 0.0f) tc = 0.0f;
                    if (tc > 1.0f) tc = 1.0f;
                    float pos = tc * (float)(G - 1);
                    float idx = floorf(pos);
                    if (idx > (float)(G - 2)) idx = (float)(G - 2);
                    float fr = pos - idx;
                    float e[64];
                    for (uint64_t k = 0; k < D; k++) {
                        float lo = table[(size_t)idx * D + k];
                        float hi = table[((size_t)idx + 1) * D + k];
                        float v = lo + (hi - lo) * fr;
                        e[k] = v / (1.0f + expf(-v)); /* the WRONG SiLU */
                    }
                    for (uint32_t o = 0; o < H3I_MOD_BLOCK_OUTPUT; o++) {
                        float sum = 0.0f;
                        for (uint64_t k = 0; k < D; k++) {
                            sum += w[(size_t)o * D + k] * e[k];
                        }
                        ctrl[(size_t)r * H3I_MOD_BLOCK_OUTPUT + o] =
                            h3i_bf16_to_f32(h3i_bf16_bits(sum + bias[o]));
                    }
                }
                double wc = worst_row_rel(ctrl, o0, 7, H3I_MOD_BLOCK_OUTPUT);
                CHECK(wc > 4.0 * w0,
                      "SiLU control %g not > 4x pruned %g", wc, w0);
                printf("SiLU control worst rel L2: %.4f (%.1fx worse)\n", wc,
                       w0 > 0 ? wc / w0 : 0.0);
            }
            free(table);
            free(w);
            free(bias);
            free(ctrl);
        }
    }
    free(p0);
    free(pf);
    free(o0);
    free(of);

cleanup_mod:
    h3i_gpu_tensor_free(ob0);
    h3i_gpu_tensor_free(ofin);
    h3i_gpu_tensor_free(act);
    h3i_modulation_free(pruned);
cleanup:
    h3i_source_free(&gguf);
    h3i_source_free(&check);
    h3i_gpu_free(gpu);
}

#endif /* __APPLE__ */

int main(void) {
    test_mkdir("build");
    test_maps();
    test_heads();
#ifdef __APPLE__
    test_tables();
#endif
    TEST_RESULT();
}
