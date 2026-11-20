/* VAE decoder — 2D ViT (SPEC §4.5). All F32; host math separately rounded.
 * Per tile: un-normalize latent cells → post_quant_conv → x_embedder →
 * [image rows | 4 registers | zero row] → 24 blocks (RMSNorm, grouped QKV +
 * weightless q/k RMS + 2D RoPE, SDPA scale 1/8, per-channel scale1/scale2
 * residuals, SwiGLU FFN) → LayerNorm → proj_out 768 → 16×16 RGB patches.
 * Tiles stitch in normalized pixel space (raw above/left neighbours), then
 * ImageNet un-normalize. */
#include "model/vae.h"

#include "app/error.h"
#include "io/host.h"
#include "model/layout.h"
#include "weights/gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#define V_HIDDEN 1024
#define V_LAYERS 24
#define V_HEADS 16
#define V_HEAD_DIM 64
#define V_FFN 4096
#define V_REGISTERS 4
#define V_OUT_PATCH 768
#define V_EPS 1e-5f
#define V_MAX_CELLS 16                 /* 256 px tile / 16 */
#define V_MAX_N (V_MAX_CELLS * V_MAX_CELLS)
#define V_MAX_SEQ (V_MAX_N + V_REGISTERS + 1)

static const float default_mean[H3I_VAE_LATENT_CHANNELS] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f,
    -0.5090325474739075f, -0.2727581858634949f, -1.3675414323806763f,
    -0.2553254961967468f, -0.26907554268836975f, -0.5376840829849243f,
    -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f,
    0.25928452610969543f, -0.30133944749832153f, 0.211341992020607f,
    -1.1206848621368408f, 0.3581933379173279f, -0.04225143790245056f,
    0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f,
};

static const float default_std[H3I_VAE_LATENT_CHANNELS] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f,
    1.7549455165863037f, 1.5636216402053833f, 2.194143533706665f,
    0.9653137922286987f, 1.0569885969161987f, 0.841948926448822f,
    0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f,
    0.6936293244361877f, 2.961095094680786f, 2.7694199085235596f,
    3.0496184825897217f, 2.1088054180145264f, 3.276226282119751f,
    3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f,
};

static const float img_mean[3] = {0.485f, 0.456f, 0.406f};
static const float img_std[3] = {0.229f, 0.224f, 0.225f};

/* ---- config (focused JSON array extraction) ---- */

static int parse_array24(const char *json, const char *key, float *out,
                         char *error, size_t error_size) {
    char needle[64];
    snprintf(needle, sizeof needle, "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) {
        h3i_fail(error, error_size, "vae config is missing %s", key);
        return 0;
    }
    p = strchr(p + strlen(needle), '[');
    if (!p) {
        h3i_fail(error, error_size, "vae config %s is not an array", key);
        return 0;
    }
    p++;
    for (int i = 0; i < H3I_VAE_LATENT_CHANNELS; i++) {
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p || !isfinite(v)) {
            h3i_fail(error, error_size, "vae config %s has an invalid number",
                     key);
            return 0;
        }
        out[i] = (float)v;
        p = end;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (i < H3I_VAE_LATENT_CHANNELS - 1) {
            if (*p != ',') {
                h3i_fail(error, error_size, "vae config %s needs 24 values",
                         key);
                return 0;
            }
            p++;
        }
    }
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != ']') {
        h3i_fail(error, error_size, "vae config %s needs exactly 24 values",
                 key);
        return 0;
    }
    return 1;
}

int h3i_vae_config_load(const char *config_path, float *mean, float *std,
                        char *error, size_t error_size) {
    memcpy(mean, default_mean, sizeof(default_mean));
    memcpy(std, default_std, sizeof(default_std));
    if (!config_path) return 1;
    FILE *f = fopen(config_path, "rb");
    if (!f) return 1; /* absent file → built-in defaults */
    char *json = NULL;
    long size;
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0 || size > 1048576) {
        fclose(f);
        h3i_fail(error, error_size, "cannot read vae config %s", config_path);
        return 0;
    }
    json = malloc((size_t)size + 1);
    if (!json || fread(json, 1, (size_t)size, f) != (size_t)size) {
        free(json);
        fclose(f);
        h3i_fail(error, error_size, "cannot read vae config %s", config_path);
        return 0;
    }
    fclose(f);
    json[size] = '\0';
    int ok = parse_array24(json, "latents_mean", mean, error, error_size) &&
             parse_array24(json, "latents_std", std, error, error_size);
    free(json);
    return ok;
}

/* ---- decoder ---- */

typedef struct {
    h3i_gpu_tensor *norm1, *norm2, *scale1, *scale2;
    h3i_gpu_tensor *qkv_w, *qkv_b, *out_w, *out_b;
    h3i_gpu_tensor *w1_w, *w1_b, *w2_w, *w2_b;
} vae_block;

struct h3i_vae_decoder {
    h3i_gpu *gpu;
    float mean[H3I_VAE_LATENT_CHANNELS];
    float std[H3I_VAE_LATENT_CHANNELS];
    h3i_gpu_tensor *post_quant_w, *post_quant_b;
    h3i_gpu_tensor *x_embed_w, *x_embed_b;
    h3i_gpu_tensor *registers;
    h3i_gpu_tensor *norm_out_w, *norm_out_b;
    h3i_gpu_tensor *proj_out_w, *proj_out_b;
    vae_block blocks[V_LAYERS];
    /* scratch, sized for the largest tile (256 px → 256 cells + 5) */
    h3i_gpu_tensor *rows, *p, *seq, *n, *qkv, *q, *k, *v, *a, *ao;
    h3i_gpu_tensor *f, *g, *yo, *o, *y, *rope_cos, *rope_sin;
    /* every allocated tensor, for uniform cleanup */
    h3i_gpu_tensor **owned;
    size_t owned_count, owned_cap;
};

static h3i_gpu_tensor *own(h3i_vae_decoder *d, h3i_gpu_tensor *t) {
    if (!t) return NULL;
    if (d->owned_count == d->owned_cap) {
        size_t cap = d->owned_cap ? d->owned_cap * 2 : 64;
        h3i_gpu_tensor **grown = realloc(d->owned, cap * sizeof(*grown));
        if (!grown) {
            h3i_gpu_tensor_free(t);
            return NULL;
        }
        d->owned = grown;
        d->owned_cap = cap;
    }
    d->owned[d->owned_count++] = t;
    return t;
}

static h3i_gpu_tensor *ld(h3i_vae_decoder *d, const h3i_source *s,
                          const char *name, uint64_t elements,
                          char *error, size_t error_size) {
    h3i_gpu_tensor *t = h3i_source_load(s, d->gpu, name, H3I_TYPE_F32,
                                        elements, error, error_size);
    return own(d, t);
}

static h3i_gpu_tensor *scratch(h3i_vae_decoder *d, size_t elements,
                               char *error, size_t error_size) {
    h3i_gpu_tensor *t = h3i_gpu_tensor_new_f32(d->gpu, elements);
    if (!t) h3i_fail(error, error_size, "%s", h3i_gpu_error(d->gpu));
    return own(d, t);
}

h3i_vae_decoder *h3i_vae_decoder_load(h3i_gpu *gpu, const h3i_source *source,
                                      const char *config_path,
                                      char *error, size_t error_size) {
    if (!gpu || !source) {
        h3i_fail(error, error_size, "invalid VAE load request");
        return NULL;
    }
    h3i_vae_decoder *d = calloc(1, sizeof(*d));
    if (!d) {
        h3i_fail(error, error_size, "out of memory loading the VAE");
        return NULL;
    }
    d->gpu = gpu;
    if (!h3i_vae_config_load(config_path, d->mean, d->std, error, error_size)) {
        goto fail;
    }
    if (!h3i_gpu_begin(gpu)) {
        h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
        goto fail;
    }
#define LD(field, name, elements)                                              \
    do {                                                                       \
        d->field = ld(d, source, name, elements, error, error_size);           \
        if (!d->field) goto fail_submit;                                       \
    } while (0)
    LD(post_quant_w, "post_quant_conv.weight", 24 * 24);
    LD(post_quant_b, "post_quant_conv.bias", 24);
    LD(x_embed_w, "decoder.x_embedder.weight", 1024 * 24);
    LD(x_embed_b, "decoder.x_embedder.bias", 1024);
    LD(registers, "decoder.register_tokens", 4 * 1024);
    LD(norm_out_w, "decoder.norm_out.weight", 1024);
    LD(norm_out_b, "decoder.norm_out.bias", 1024);
    LD(proj_out_w, "decoder.proj_out.weight", 768 * 1024);
    LD(proj_out_b, "decoder.proj_out.bias", 768);
    for (int b = 0; b < V_LAYERS; b++) {
        char name[160];
        vae_block *blk = &d->blocks[b];
#define BLK(field, suffix, elements)                                           \
        do {                                                                   \
            snprintf(name, sizeof name,                                        \
                     "decoder.transformer_blocks.%d.%s", b, suffix);           \
            blk->field = ld(d, source, name, elements, error, error_size);     \
            if (!blk->field) goto fail_submit;                                 \
        } while (0)
        BLK(norm1, "norm1.weight", 1024);
        BLK(norm2, "norm2.weight", 1024);
        BLK(scale1, "scale1", 1024);
        BLK(scale2, "scale2", 1024);
        BLK(qkv_w, "attn.to_qkv.weight", 3072 * 1024);
        BLK(qkv_b, "attn.to_qkv.bias", 3072);
        BLK(out_w, "attn.to_out.weight", 1024 * 1024);
        BLK(out_b, "attn.to_out.bias", 1024);
        BLK(w1_w, "ff.w1.weight", 8192 * 1024);
        BLK(w1_b, "ff.w1.bias", 8192);
        BLK(w2_w, "ff.w2.weight", 1024 * 4096);
        BLK(w2_b, "ff.w2.bias", 1024);
#undef BLK
    }
    /* scratch */
    d->rows = scratch(d, V_MAX_N * 24, error, error_size);
    d->p = scratch(d, V_MAX_N * 24, error, error_size);
    d->seq = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->n = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->qkv = scratch(d, V_MAX_SEQ * 3 * V_HIDDEN, error, error_size);
    d->q = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->k = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->v = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->a = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->ao = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->f = scratch(d, V_MAX_SEQ * 2 * V_FFN, error, error_size);
    d->g = scratch(d, V_MAX_SEQ * V_FFN, error, error_size);
    d->yo = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->o = scratch(d, V_MAX_SEQ * V_HIDDEN, error, error_size);
    d->y = scratch(d, V_MAX_SEQ * V_OUT_PATCH, error, error_size);
    d->rope_cos = scratch(d, V_MAX_SEQ * 16, error, error_size);
    d->rope_sin = scratch(d, V_MAX_SEQ * 16, error, error_size);
    if (!d->rows || !d->p || !d->seq || !d->n || !d->qkv || !d->q || !d->k ||
        !d->v || !d->a || !d->ao || !d->f || !d->g || !d->yo || !d->o ||
        !d->y || !d->rope_cos || !d->rope_sin) {
        goto fail_submit;
    }
    if (!h3i_gpu_submit(gpu)) {
        h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
        goto fail;
    }
#undef LD
    return d;

fail_submit:
    h3i_gpu_submit(gpu); /* drain whatever was encoded; ignore its status */
fail:
    h3i_vae_decoder_free(d);
    return NULL;
}

void h3i_vae_decoder_free(h3i_vae_decoder *d) {
    if (!d) return;
    for (size_t i = 0; i < d->owned_count; i++) {
        h3i_gpu_tensor_free(d->owned[i]);
    }
    free(d->owned);
    free(d);
}

/* ---- decode ---- */

#define OP(call)                                                               \
    do {                                                                       \
        if (!(call)) {                                                         \
            h3i_fail(error, error_size, "%s", h3i_gpu_error(d->gpu));          \
            goto tile_fail;                                                    \
        }                                                                      \
    } while (0)

int h3i_vae_decoder_decode(h3i_vae_decoder *d, const float *latent,
                           int lh, int lw, float **rgb,
                           char *error, size_t error_size) {
    if (rgb) *rgb = NULL;
    if (!d || !latent || !rgb || lh <= 0 || lw <= 0 || (lh & 1) || (lw & 1)) {
        h3i_fail(error, error_size, "invalid VAE decode request");
        return 0;
    }
    int H = lh * 16;
    int W = lw * 16;
    h3i_tile_axis plan_y, plan_x;
    if (!h3i_tile_axis_plan(H, 256, 64, &plan_y) ||
        !h3i_tile_axis_plan(W, 256, 64, &plan_x)) {
        h3i_fail(error, error_size, "cannot plan VAE tiles for %dx%d", W, H);
        return 0;
    }
    float *out = calloc((size_t)H * W * 3, sizeof(float));
    float **prev = calloc((size_t)plan_x.count, sizeof(float *));
    float **cur = calloc((size_t)plan_x.count, sizeof(float *));
    float *rows_host = malloc(V_MAX_N * 24 * sizeof(float));
    float *cos_host = malloc(V_MAX_SEQ * 16 * sizeof(float));
    float *sin_host = malloc(V_MAX_SEQ * 16 * sizeof(float));
    float *y_host = malloc(V_MAX_N * V_OUT_PATCH * sizeof(float));
    if (!out || !prev || !cur || !rows_host || !cos_host || !sin_host ||
        !y_host) {
        h3i_fail(error, error_size, "out of memory decoding the image");
        goto fail;
    }
    for (int ty = 0; ty < plan_y.count; ty++) {
        for (int tx = 0; tx < plan_x.count; tx++) {
            int y0 = plan_y.starts[ty] / 16, x0 = plan_x.starts[tx] / 16;
            int th = plan_y.length / 16, tw = plan_x.length / 16;
            int N = th * tw, seq = N + V_REGISTERS + 1;
            int ph = th * 16, pw = tw * 16;
            float *raw = malloc((size_t)ph * pw * 3 * sizeof(float));
            if (!raw) {
                h3i_fail(error, error_size, "out of memory decoding the image");
                goto tile_fail;
            }
            /* un-normalize latent cells → rows (no FMA) */
            for (int h = 0; h < th; h++) {
                for (int w = 0; w < tw; w++) {
                    float *row = rows_host + (size_t)(h * tw + w) * 24;
                    for (int c = 0; c < 24; c++) {
                        row[c] = latent[((size_t)c * lh + (y0 + h)) * lw +
                                        (x0 + w)] * d->std[c] + d->mean[c];
                    }
                }
            }
            /* RoPE tables: 16 per row (axis·8+f), registers/zero cos=1 sin=0 */
            for (int h = 0; h < th; h++) {
                for (int w = 0; w < tw; w++) {
                    int n = h * tw + w;
                    float ax_h = 2.0f * (((float)h + 0.5f) / (float)th) - 1.0f;
                    float ax_w = 2.0f * (((float)w + 0.5f) / (float)tw) - 1.0f;
                    for (int f = 0; f < 8; f++) {
                        float inv = 1.0f / powf(100.0f, (float)f * 0.125f);
                        float ang_h = 2.0f * 3.14159265358979323846f * ax_h * inv;
                        float ang_w = 2.0f * 3.14159265358979323846f * ax_w * inv;
                        cos_host[n * 16 + f] = cosf(ang_h);
                        sin_host[n * 16 + f] = sinf(ang_h);
                        cos_host[n * 16 + 8 + f] = cosf(ang_w);
                        sin_host[n * 16 + 8 + f] = sinf(ang_w);
                    }
                }
            }
            for (int n = N; n < seq; n++) {
                for (int f = 0; f < 16; f++) {
                    cos_host[n * 16 + f] = 1.0f;
                    sin_host[n * 16 + f] = 0.0f;
                }
            }
            /* seq = [x (N rows) | registers (4) | zero row]; the block chain
             * overwrites seq in place, so reassemble every tile. The zero row
             * is a CPU write into the shared buffer, done before encoding. */
            {
                float zeros[V_HIDDEN] = {0};
                if (!h3i_gpu_tensor_write_f32_range(
                        d->seq, (size_t)(N + V_REGISTERS) * V_HIDDEN, zeros,
                        V_HIDDEN)) {
                    h3i_fail(error, error_size, "cannot zero the VAE token");
                    goto tile_fail;
                }
            }
            if (!h3i_gpu_tensor_write_f32(d->rows, rows_host, (size_t)N * 24) ||
                !h3i_gpu_tensor_write_f32(d->rope_cos, cos_host,
                                          (size_t)seq * 16) ||
                !h3i_gpu_tensor_write_f32(d->rope_sin, sin_host,
                                          (size_t)seq * 16)) {
                h3i_fail(error, error_size, "cannot upload VAE tile inputs");
                goto tile_fail;
            }
            OP(h3i_gpu_begin(d->gpu));
            OP(h3i_gpu_linear_f32(d->gpu, d->p, d->rows, d->post_quant_w,
                                  d->post_quant_b, (uint32_t)N, 24, 24));
            OP(h3i_gpu_linear_f32(d->gpu, d->seq, d->p, d->x_embed_w,
                                  d->x_embed_b, (uint32_t)N, 24, V_HIDDEN));
            OP(h3i_gpu_copy_f32(d->gpu, d->seq, (size_t)N * V_HIDDEN,
                                d->registers, 0, V_REGISTERS * V_HIDDEN));
            for (int b = 0; b < V_LAYERS; b++) {
                vae_block *blk = &d->blocks[b];
                OP(h3i_gpu_rms_norm_f32(d->gpu, d->n, d->seq, blk->norm1,
                                        (uint32_t)seq, V_HIDDEN, V_EPS));
                OP(h3i_gpu_linear_f32(d->gpu, d->qkv, d->n, blk->qkv_w,
                                      blk->qkv_b, (uint32_t)seq, V_HIDDEN,
                                      3 * V_HIDDEN));
                OP(h3i_gpu_video_qkv_rope_f32(d->gpu, d->q, d->k, d->v,
                                              d->qkv, d->rope_cos, d->rope_sin,
                                              (uint32_t)seq, V_HEADS,
                                              V_HEAD_DIM, 16, V_EPS));
                OP(h3i_gpu_sdpa_f32(d->gpu, d->a, d->q, d->k, d->v,
                                    (uint32_t)seq, V_HEADS, V_HEAD_DIM,
                                    0.125f));
                OP(h3i_gpu_linear_f32(d->gpu, d->ao, d->a, blk->out_w,
                                      blk->out_b, (uint32_t)seq, V_HIDDEN,
                                      V_HIDDEN));
                OP(h3i_gpu_scale_add_f32(d->gpu, d->seq, d->seq, d->ao,
                                         blk->scale1, (uint32_t)seq,
                                         V_HIDDEN));
                OP(h3i_gpu_rms_norm_f32(d->gpu, d->n, d->seq, blk->norm2,
                                        (uint32_t)seq, V_HIDDEN, V_EPS));
                OP(h3i_gpu_linear_f32(d->gpu, d->f, d->n, blk->w1_w,
                                      blk->w1_b, (uint32_t)seq, V_HIDDEN,
                                      2 * V_FFN));
                OP(h3i_gpu_swiglu_f32(d->gpu, d->g, d->f, (uint32_t)seq,
                                      V_FFN));
                OP(h3i_gpu_linear_f32(d->gpu, d->yo, d->g, blk->w2_w,
                                      blk->w2_b, (uint32_t)seq, V_FFN,
                                      V_HIDDEN));
                OP(h3i_gpu_scale_add_f32(d->gpu, d->seq, d->seq, d->yo,
                                         blk->scale2, (uint32_t)seq,
                                         V_HIDDEN));
            }
            OP(h3i_gpu_layer_norm_f32(d->gpu, d->o, d->seq, d->norm_out_w,
                                      d->norm_out_b, (uint32_t)seq, V_HIDDEN,
                                      V_EPS));
            OP(h3i_gpu_linear_f32(d->gpu, d->y, d->o, d->proj_out_w,
                                  d->proj_out_b, (uint32_t)seq, V_HIDDEN,
                                  V_OUT_PATCH));
            OP(h3i_gpu_submit(d->gpu));
            if (!h3i_gpu_tensor_read_f32_range(d->y, 0, y_host,
                                               (size_t)N * V_OUT_PATCH)) {
                h3i_fail(error, error_size, "cannot read the VAE tile output");
                goto tile_fail;
            }
            /* unpatchify: pixel(h·16+py, w·16+px, c) =
             * y[h·tw+w][c·256 + py·16 + px] */
            for (int h = 0; h < th; h++) {
                for (int w = 0; w < tw; w++) {
                    const float *yv = y_host + (size_t)(h * tw + w) * V_OUT_PATCH;
                    for (int c = 0; c < 3; c++) {
                        for (int py = 0; py < 16; py++) {
                            for (int px = 0; px < 16; px++) {
                                raw[((size_t)(h * 16 + py) * pw +
                                     (w * 16 + px)) * 3 + c] =
                                    yv[c * 256 + py * 16 + px];
                            }
                        }
                    }
                }
            }
            /* stitch in normalized pixel space against RAW neighbours */
            int oy = ty > 0 ? plan_y.overlaps[ty - 1] : 0;
            int ox = tx > 0 ? plan_x.overlaps[tx - 1] : 0;
            int keep_h = ty < plan_y.count - 1 ? ph - plan_y.overlaps[ty] : ph;
            int keep_w = tx < plan_x.count - 1 ? pw - plan_x.overlaps[tx] : pw;
            const float *above = prev[tx];
            const float *left = tx > 0 ? cur[tx - 1] : NULL;
            for (int y = 0; y < keep_h; y++) {
                for (int x = 0; x < keep_w; x++) {
                    for (int c = 0; c < 3; c++) {
                        size_t idx = ((size_t)y * pw + x) * 3 + c;
                        float v = raw[idx];
                        if (ty > 0 && y < oy) {
                            float a = (float)y / (float)oy;
                            size_t aidx = ((size_t)(ph - oy + y) * pw + x) * 3 + c;
                            v = above[aidx] * (1.0f - a) + v * a;
                        }
                        if (tx > 0 && x < ox) {
                            float a = (float)x / (float)ox;
                            size_t lidx = ((size_t)y * pw + (pw - ox + x)) * 3 + c;
                            v = left[lidx] * (1.0f - a) + v * a;
                        }
                        out[((size_t)(plan_y.starts[ty] + y) * W +
                             (plan_x.starts[tx] + x)) * 3 + c] = v;
                    }
                }
            }
            cur[tx] = raw;
            continue;

        tile_fail:
            free(raw);
            goto fail;
        }
        /* finished row ty: its raw tiles become the next row's "above" */
        for (int tx = 0; tx < plan_x.count; tx++) free(prev[tx]);
        float **swap = prev;
        prev = cur;
        cur = swap;
        memset(cur, 0, (size_t)plan_x.count * sizeof(float *));
    }
    /* ImageNet un-normalize (no FMA); unit_to_u8 clamps downstream */
    for (size_t i = 0; i < (size_t)H * W; i++) {
        for (int c = 0; c < 3; c++) {
            out[i * 3 + c] = out[i * 3 + c] * img_std[c] + img_mean[c];
        }
    }
    for (int tx = 0; tx < plan_x.count; tx++) {
        free(prev[tx]);
        free(cur[tx]);
    }
    free(prev);
    free(cur);
    free(rows_host);
    free(cos_host);
    free(sin_host);
    free(y_host);
    *rgb = out;
    return 1;

fail:
    for (int tx = 0; tx < plan_x.count; tx++) {
        free(prev[tx]);
        free(cur[tx]);
    }
    free(prev);
    free(cur);
    free(out);
    free(rows_host);
    free(cos_host);
    free(sin_host);
    free(y_host);
    return 0;
}

#undef OP
