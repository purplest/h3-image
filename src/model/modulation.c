/* AdaLN modulation (SPEC §3.7) and PDD heads (SPEC §3.11).
 * Official: Timesteps(256, flip_sin_to_cos, downscale 0) F32 libm features →
 * proj_in (F32) → SiLU (F32) → proj_out (F32) → SiLU (F32) → BF16 cast →
 * per-block BF16 projections with F32 accumulation, one block resident at
 * a time. Pruned: adaln_t_table interpolation on the host, sequential F32
 * dots, NO SiLU. Host math separately rounded (FP contraction OFF). */
#include "model/modulation.h"

#include "app/error.h"
#include "io/host.h"
#include "weights/gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#define TIME_INPUT 256
#define BLOCK_OUTPUT H3I_MOD_BLOCK_OUTPUT
#define FINAL_OUTPUT H3I_MOD_FINAL_OUTPUT
#define TIME_DIM H3I_DIT_TIME_DIM

struct h3i_modulation {
    h3i_gpu *gpu;
    int pruned;
    uint32_t time_rows;
    h3i_gpu_tensor *blocks[H3I_DIT_BLOCKS];
    h3i_gpu_tensor *final;
};

/* ---- shared helpers ---- */

static int op(h3i_gpu *gpu, int ok, char *error, size_t error_size,
              const char *what) {
    if (ok) return 1;
    h3i_fail(error, error_size, "%s: %s", what, h3i_gpu_error(gpu));
    return 0;
}

static void time_features(const float *times, uint32_t rows, float *features) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < TIME_INPUT / 2; i++) {
            float freq = expf(-logf(10000.0f) * (float)i /
                              (float)(TIME_INPUT / 2));
            float angle = times[row] * freq;
            features[(size_t)row * TIME_INPUT + i] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + i] =
                sinf(angle);
        }
    }
}

/* ---- official path (GPU) ---- */

static h3i_gpu_tensor *official_act(h3i_gpu *gpu, const h3i_source *source,
                                    uint32_t rows, const float *features,
                                    char *error, size_t error_size);

/* act = bf16(silu(temb)) — SiLU in F32 BEFORE the BF16 cast (SPEC §3.7;
 * the donor video path cast first, which double-rounds). */
h3i_gpu_tensor *h3i_time_act(h3i_gpu *gpu, const h3i_source *source,
                             const float *times, uint32_t rows,
                             char *error, size_t error_size) {
    float *features = malloc((size_t)rows * TIME_INPUT * sizeof(float));
    if (!features) {
        h3i_fail(error, error_size, "out of memory allocating timestep features");
        return NULL;
    }
    time_features(times, rows, features);
    h3i_gpu_tensor *act = official_act(gpu, source, rows, features,
                                       error, error_size);
    free(features);
    return act;
}

static h3i_gpu_tensor *official_act(h3i_gpu *gpu, const h3i_source *source,
                                    uint32_t rows, const float *features,
                                    char *error, size_t error_size) {
    h3i_gpu_tensor *input = NULL, *in_w = NULL, *in_b = NULL;
    h3i_gpu_tensor *out_w = NULL, *out_b = NULL;
    h3i_gpu_tensor *hidden = NULL, *act32 = NULL, *temb = NULL;
    h3i_gpu_tensor *act = NULL;
    input = h3i_gpu_tensor_from_f32(gpu, features, (size_t)rows * TIME_INPUT);
    hidden = h3i_gpu_tensor_new_f32(gpu, (size_t)rows * H3I_DIT_HIDDEN);
    act32 = h3i_gpu_tensor_new_f32(gpu, (size_t)rows * H3I_DIT_HIDDEN);
    temb = h3i_gpu_tensor_new_f32(gpu, (size_t)rows * TIME_DIM);
    act = h3i_gpu_tensor_new_bf16(gpu, (size_t)rows * TIME_DIM);
    if (!input || !hidden || !act32 || !temb || !act) {
        h3i_fail(error, error_size, "cannot allocate timestep tensors: %s",
                 h3i_gpu_error(gpu));
        goto cleanup;
    }
    if (!op(gpu, h3i_gpu_begin(gpu), error, error_size, "begin time MLP"))
        goto cleanup;
    in_w = h3i_source_load(source, gpu, "time_embedder.proj_in.weight",
                           H3I_TYPE_F32, (uint64_t)H3I_DIT_HIDDEN * TIME_INPUT,
                           error, error_size);
    in_b = h3i_source_load(source, gpu, "time_embedder.proj_in.bias",
                           H3I_TYPE_F32, H3I_DIT_HIDDEN, error, error_size);
    if (!in_w || !in_b) goto cleanup;
    if (!op(gpu, h3i_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                    TIME_INPUT, H3I_DIT_HIDDEN),
            error, error_size, "time proj_in") ||
        !op(gpu, h3i_gpu_silu_f32(gpu, act32, hidden,
                                  rows * H3I_DIT_HIDDEN),
            error, error_size, "time SiLU"))
        goto cleanup;
    out_w = h3i_source_load(source, gpu, "time_embedder.proj_out.weight",
                            H3I_TYPE_F32, (uint64_t)TIME_DIM * H3I_DIT_HIDDEN,
                            error, error_size);
    out_b = h3i_source_load(source, gpu, "time_embedder.proj_out.bias",
                            H3I_TYPE_F32, TIME_DIM, error, error_size);
    if (!out_w || !out_b) goto cleanup;
    if (!op(gpu, h3i_gpu_linear_f32(gpu, temb, act32, out_w, out_b, rows,
                                    H3I_DIT_HIDDEN, TIME_DIM),
            error, error_size, "time proj_out") ||
        !op(gpu, h3i_gpu_silu_f32(gpu, act32, temb, rows * TIME_DIM),
            error, error_size, "AdaLN SiLU") ||
        !op(gpu, h3i_gpu_cast_f32_to_bf16(gpu, act, act32, rows * TIME_DIM),
            error, error_size, "AdaLN BF16 cast") ||
        !op(gpu, h3i_gpu_submit(gpu), error, error_size, "submit time MLP"))
        goto cleanup;
    h3i_gpu_tensor_free(input);
    h3i_gpu_tensor_free(in_w);
    h3i_gpu_tensor_free(in_b);
    h3i_gpu_tensor_free(out_w);
    h3i_gpu_tensor_free(out_b);
    h3i_gpu_tensor_free(hidden);
    h3i_gpu_tensor_free(act32);
    h3i_gpu_tensor_free(temb);
    return act;

cleanup:
    h3i_gpu_tensor_free(input);
    h3i_gpu_tensor_free(in_w);
    h3i_gpu_tensor_free(in_b);
    h3i_gpu_tensor_free(out_w);
    h3i_gpu_tensor_free(out_b);
    h3i_gpu_tensor_free(hidden);
    h3i_gpu_tensor_free(act32);
    h3i_gpu_tensor_free(temb);
    h3i_gpu_tensor_free(act);
    return NULL;
}

int h3i_adaln_project(h3i_gpu *gpu, const h3i_source *source,
                               const h3i_gpu_tensor *act, uint32_t rows,
                               const char *weight_name, const char *bias_name,
                               uint32_t outputs, h3i_gpu_tensor **table_out,
                               char *error, size_t error_size) {
    h3i_gpu_tensor *weight = NULL, *bias = NULL, *table = NULL;
    table = h3i_gpu_tensor_new_bf16(gpu, (size_t)rows * outputs);
    if (!table) {
        h3i_fail(error, error_size, "cannot allocate AdaLN table: %s",
                 h3i_gpu_error(gpu));
        goto cleanup;
    }
    if (!op(gpu, h3i_gpu_begin(gpu), error, error_size, "begin AdaLN"))
        goto cleanup;
    weight = h3i_source_load(source, gpu, weight_name, H3I_TYPE_BF16,
                             (uint64_t)outputs * TIME_DIM, error, error_size);
    bias = h3i_source_load(source, gpu, bias_name, H3I_TYPE_BF16, outputs,
                           error, error_size);
    if (!weight || !bias) goto cleanup;
    if (!op(gpu, h3i_gpu_linear_bf16(gpu, table, act, weight, bias, rows,
                                     TIME_DIM, outputs),
            error, error_size, "AdaLN projection") ||
        !op(gpu, h3i_gpu_submit(gpu), error, error_size, "submit AdaLN"))
        goto cleanup;
    h3i_gpu_tensor_free(weight);
    h3i_gpu_tensor_free(bias);
    *table_out = table;
    return 1;

cleanup:
    h3i_gpu_tensor_free(weight);
    h3i_gpu_tensor_free(bias);
    h3i_gpu_tensor_free(table);
    return 0;
}

/* ---- pruned path (host) ---- */

static int pruned_tables(h3i_modulation *m, const h3i_source *source,
                         const float *times, char *error, size_t error_size) {
    const h3i_tensor_rec *rec = h3i_source_find(source, "adaln_t_table");
    if (!rec || rec->ndim != 2 || rec->shape[0] < 2 || rec->shape[1] < 1 ||
        rec->shape[1] > 64) {
        h3i_fail(error, error_size, "adaln_t_table has an invalid shape");
        return 0;
    }
    uint64_t G = rec->shape[0], D = rec->shape[1];
    float *table = h3i_source_read_f32(source, "adaln_t_table", G * D,
                                       error, error_size);
    if (!table) return 0;
    /* Interpolated embeddings e[row][D], separately rounded F32. */
    float *e = malloc((size_t)m->time_rows * D * sizeof(float));
    uint16_t *bits = malloc((size_t)m->time_rows * BLOCK_OUTPUT *
                            sizeof(uint16_t));
    if (!e || !bits) {
        free(table);
        free(e);
        free(bits);
        h3i_fail(error, error_size, "out of memory building AdaLN tables");
        return 0;
    }
    for (uint32_t r = 0; r < m->time_rows; r++) {
        float tc = times[r];
        if (tc < 0.0f) tc = 0.0f;
        if (tc > 1.0f) tc = 1.0f;
        float pos = tc * (float)(G - 1);
        float idx = floorf(pos);
        if (idx > (float)(G - 2)) idx = (float)(G - 2);
        float fr = pos - idx;
        const float *lo = table + (size_t)idx * D;
        const float *hi = lo + D;
        for (uint64_t k = 0; k < D; k++) {
            e[(size_t)r * D + k] = lo[k] + (hi[k] - lo[k]) * fr;
        }
    }
    for (unsigned b = 0; b < H3I_DIT_BLOCKS; b++) {
        char name[160];
        snprintf(name, sizeof name, "blocks.%u.adaln_proj.linear.weight", b);
        float *w = h3i_source_read_f32(source, name,
                                       (uint64_t)BLOCK_OUTPUT * D,
                                       error, error_size);
        snprintf(name, sizeof name, "blocks.%u.adaln_proj.linear.bias", b);
        float *bias = h3i_source_read_f32(source, name, BLOCK_OUTPUT,
                                          error, error_size);
        if (!w || !bias) {
            free(w);
            free(bias);
            goto fail;
        }
        for (uint32_t r = 0; r < m->time_rows; r++) {
            const float *er = e + (size_t)r * D;
            uint16_t *out = bits + (size_t)r * BLOCK_OUTPUT;
            for (uint32_t o = 0; o < BLOCK_OUTPUT; o++) {
                const float *wr = w + (size_t)o * D;
                float sum = 0.0f;
                for (uint64_t k = 0; k < D; k++) {
                    sum += wr[k] * er[k];
                }
                out[o] = h3i_bf16_bits(sum + bias[o]);
            }
        }
        free(w);
        free(bias);
        m->blocks[b] = h3i_gpu_tensor_from_bf16(
            m->gpu, bits, (size_t)m->time_rows * BLOCK_OUTPUT);
        if (!m->blocks[b]) {
            h3i_fail(error, error_size, "%s", h3i_gpu_error(m->gpu));
            goto fail;
        }
    }
    {
        float *w = h3i_source_read_f32(source,
                                       "final_layer.adaln_proj.linear.weight",
                                       (uint64_t)FINAL_OUTPUT * D,
                                       error, error_size);
        float *bias = h3i_source_read_f32(source,
                                          "final_layer.adaln_proj.linear.bias",
                                          FINAL_OUTPUT, error, error_size);
        if (!w || !bias) {
            free(w);
            free(bias);
            goto fail;
        }
        for (uint32_t r = 0; r < m->time_rows; r++) {
            const float *er = e + (size_t)r * D;
            uint16_t *out = bits + (size_t)r * FINAL_OUTPUT;
            for (uint32_t o = 0; o < FINAL_OUTPUT; o++) {
                const float *wr = w + (size_t)o * D;
                float sum = 0.0f;
                for (uint64_t k = 0; k < D; k++) {
                    sum += wr[k] * er[k];
                }
                out[o] = h3i_bf16_bits(sum + bias[o]);
            }
        }
        free(w);
        free(bias);
        m->final = h3i_gpu_tensor_from_bf16(
            m->gpu, bits, (size_t)m->time_rows * FINAL_OUTPUT);
        if (!m->final) {
            h3i_fail(error, error_size, "%s", h3i_gpu_error(m->gpu));
            goto fail;
        }
    }
    free(table);
    free(e);
    free(bits);
    return 1;

fail:
    free(table);
    free(e);
    free(bits);
    return 0;
}

/* ---- build ---- */

h3i_modulation *h3i_modulation_build(h3i_gpu *gpu, const h3i_source *source,
                                     const h3i_schedule *schedule,
                                     int has_references,
                                     char *error, size_t error_size) {
    if (!gpu || !source || !schedule) {
        h3i_fail(error, error_size, "invalid modulation request");
        return NULL;
    }
    int pruned = h3i_source_find(source, "adaln_t_table") != NULL;
    if (!pruned &&
        !h3i_source_find(source, "time_embedder.proj_in.weight")) {
        h3i_fail(error, error_size,
                 "DiT weights have neither adaln_t_table nor time_embedder");
        return NULL;
    }
    h3i_modulation *m = calloc(1, sizeof(*m));
    if (!m) {
        h3i_fail(error, error_size, "out of memory creating modulation");
        return NULL;
    }
    m->gpu = gpu;
    m->pruned = pruned;
    m->time_rows = has_references ? 7 : 6;
    float times[7];
    for (uint32_t r = 0; r < m->time_rows; r++) times[r] = schedule->t[r];

    if (pruned) {
        if (!pruned_tables(m, source, times, error, error_size)) goto fail;
        return m;
    }
    h3i_gpu_tensor *act = h3i_time_act(gpu, source, times, m->time_rows,
                                       error, error_size);
    if (!act) goto fail;
    for (unsigned b = 0; b < H3I_DIT_BLOCKS; b++) {
        char weight_name[128], bias_name[128];
        snprintf(weight_name, sizeof weight_name,
                 "blocks.%u.adaln_proj.linear.weight", b);
        snprintf(bias_name, sizeof bias_name,
                 "blocks.%u.adaln_proj.linear.bias", b);
        if (!h3i_adaln_project(gpu, source, act, m->time_rows, weight_name,
                                 bias_name, BLOCK_OUTPUT, &m->blocks[b],
                                 error, error_size)) {
            h3i_gpu_tensor_free(act);
            goto fail;
        }
    }
    if (!h3i_adaln_project(gpu, source, act, m->time_rows,
                             "final_layer.adaln_proj.linear.weight",
                             "final_layer.adaln_proj.linear.bias",
                             FINAL_OUTPUT, &m->final, error, error_size)) {
        h3i_gpu_tensor_free(act);
        goto fail;
    }
    h3i_gpu_tensor_free(act);
    return m;

fail:
    h3i_modulation_free(m);
    return NULL;
}

void h3i_modulation_free(h3i_modulation *m) {
    if (!m) return;
    for (unsigned b = 0; b < H3I_DIT_BLOCKS; b++) {
        h3i_gpu_tensor_free(m->blocks[b]);
    }
    h3i_gpu_tensor_free(m->final);
    free(m);
}

int h3i_modulation_is_pruned(const h3i_modulation *m) {
    return m ? m->pruned : 0;
}

uint32_t h3i_modulation_time_rows(const h3i_modulation *m) {
    return m ? m->time_rows : 0;
}

const h3i_gpu_tensor *h3i_modulation_block(const h3i_modulation *m, int block) {
    if (!m || block < 0 || block >= (int)H3I_DIT_BLOCKS) return NULL;
    return m->blocks[block];
}

const h3i_gpu_tensor *h3i_modulation_final(const h3i_modulation *m) {
    return m ? m->final : NULL;
}

/* ---- row maps ---- */

int h3i_dit_prepare_maps(const h3i_layout *layout, int steps,
                         uint32_t *row_maps, uint32_t *final_maps,
                         char *error, size_t error_size) {
    if (!layout || steps <= 0 || !row_maps || !final_maps) {
        h3i_fail(error, error_size, "invalid row map request");
        return 0;
    }
    for (int s = 0; s < steps; s++) {
        uint32_t *map = row_maps + (size_t)s * layout->sequence;
        for (size_t row = 0; row < layout->sequence; row++) {
            int is_ref = row >= layout->text_rows &&
                         row < layout->text_rows + layout->reference_rows;
            uint32_t time_row = is_ref ? 6u : (uint32_t)s;
            map[row] = time_row * 3u + layout->tags[row];
        }
        uint32_t *fmap = final_maps + (size_t)s * layout->target_rows;
        for (size_t r = 0; r < layout->target_rows; r++) {
            fmap[r] = (uint32_t)s;
        }
    }
    return 1;
}

/* ---- PDD heads ---- */

int h3i_heads_load(const h3i_source *source, h3i_heads *out,
                   char *error, size_t error_size) {
    if (!source || !out) {
        h3i_fail(error, error_size, "invalid heads request");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    const h3i_tensor_rec *w = h3i_source_find(source, "video_out.weight");
    const char *wname = "video_out.weight";
    const char *bname = "video_out.bias";
    uint64_t heads = 0;
    int offset_form = 0;
    if (w && w->ndim == 3 && w->shape[1] == 96 && w->shape[2] == 5376) {
        heads = w->shape[0];
    } else {
        w = h3i_source_find(source, "final_layer.video_out.weight");
        wname = "final_layer.video_out.weight";
        bname = "final_layer.video_out.bias";
        if (w && w->ndim == 2 && w->shape[1] == 5376 && w->shape[0] > 96 &&
            w->shape[0] % 96 == 0) {
            heads = w->shape[0] / 96;
            offset_form = 1;
        } else {
            h3i_fail(error, error_size,
                     "no PDD output heads: expected pdd_fused_heads.safetensors "
                     "or a 6-step final_layer.video_out");
            return 0;
        }
    }
    if (heads != 6) {
        h3i_fail(error, error_size, "checkpoint has %llu PDD heads, expected 6",
                 (unsigned long long)heads);
        return 0;
    }
    out->weight = h3i_source_read_f32(source, wname,
                                      (uint64_t)6 * 96 * 5376,
                                      error, error_size);
    out->bias = h3i_source_read_f32(source, bname, (uint64_t)6 * 96,
                                    error, error_size);
    if (!out->weight || !out->bias) {
        h3i_heads_free(out);
        return 0;
    }
    if (offset_form) {
        /* heads 1..5 are offsets from head 0; sum once in F32. */
        const size_t head_elems = (size_t)96 * 5376;
        for (int i = 1; i < 6; i++) {
            float *wi = out->weight + (size_t)i * head_elems;
            for (size_t j = 0; j < head_elems; j++) {
                wi[j] += out->weight[j];
            }
            float *bi = out->bias + (size_t)i * 96;
            for (int j = 0; j < 96; j++) {
                bi[j] += out->bias[j];
            }
        }
    }
    return 1;
}

void h3i_heads_free(h3i_heads *heads) {
    if (!heads) return;
    free(heads->weight);
    free(heads->bias);
    heads->weight = NULL;
    heads->bias = NULL;
}
