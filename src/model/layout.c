/* Canvas, patchify, layout, schedule, RoPE tables, tile plans (SPEC part 3).
 * Every operation is separately rounded (FP contraction OFF); double where
 * the spec says double, float where it says float, in the exact order shown. */
#include "model/layout.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "app/error.h"
#include "io/host.h"

int h3i_canvas_resolve(int width, int height, long long min_area,
                       h3i_canvas *out, char *error, size_t error_size)
{
    if (width <= 0 || height <= 0) {
        h3i_fail(error, error_size, "width and height must be positive");
        return 0;
    }
    if (min_area < 1024 || min_area > H3I_MAX_AREA) {
        h3i_fail(error, error_size,
                 "minimum generation area must be in [1024, %lld]", H3I_MAX_AREA);
        return 0;
    }
    double scale = sqrt((double)min_area / ((double)width * (double)height));
    if (scale < 1.0) {
        scale = 1.0;
    }
    double h = nearbyint((double)height * scale / 32.0) * 32.0;
    double w = nearbyint((double)width * scale / 32.0) * 32.0;
    if (h < 32.0) {
        h = 32.0;
    }
    if (w < 32.0) {
        w = 32.0;
    }
    if (h * w > (double)H3I_MAX_AREA * 1.25) {
        h3i_fail(error, error_size, "canvas %.0fx%.0f exceeds the supported area", w, h);
        return 0;
    }
    out->width = width;
    out->height = height;
    out->canvas_w = (int)w;
    out->canvas_h = (int)h;
    out->latent_w = (int)w / H3I_VAE_RATIO;
    out->latent_h = (int)h / H3I_VAE_RATIO;
    return 1;
}

void h3i_patchify(const float *latent, int lh, int lw, float *rows)
{
    size_t out = 0;
    for (int y = 0; y < lh; y += 2) {
        for (int x = 0; x < lw; x += 2) {
            for (int c = 0; c < H3I_LATENT_CHANNELS; c++) {
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        rows[out++] = latent[((size_t)c * (size_t)lh + (size_t)(y + dy))
                                             * (size_t)lw + (size_t)(x + dx)];
                    }
                }
            }
        }
    }
}

void h3i_unpatchify(const float *rows, int lh, int lw, float *latent)
{
    size_t in = 0;
    for (int y = 0; y < lh; y += 2) {
        for (int x = 0; x < lw; x += 2) {
            for (int c = 0; c < H3I_LATENT_CHANNELS; c++) {
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        latent[((size_t)c * (size_t)lh + (size_t)(y + dy))
                               * (size_t)lw + (size_t)(x + dx)] = rows[in++];
                    }
                }
            }
        }
    }
}

/* numpy.linspace(left, left+ratio, dim//2, endpoint=False) * 32, evaluated
 * exactly in this order (SPEC §3.4). */
static void grid_axis(int dim, double sqrt_area, double *out)
{
    double ratio = (double)dim / sqrt_area;
    double left = (1.0 - ratio) / 2.0;
    int n = dim / 2;
    double step = ((left + ratio) - left) / (double)n;
    for (int i = 0; i < n; i++) {
        out[i] = (((double)i * step) + left) * 32.0;
    }
}

int h3i_layout_build(size_t text_rows, const uint8_t *text_tags,
                     int ref_count, const int *ref_lh, const int *ref_lw,
                     int lh, int lw, h3i_layout *out, char *error, size_t error_size)
{
    memset(out, 0, sizeof *out);
    size_t ref_rows = 0;
    for (int k = 0; k < ref_count; k++) {
        ref_rows += (size_t)(ref_lh[k] / 2) * (size_t)(ref_lw[k] / 2);
    }
    size_t target_rows = (size_t)(lh / 2) * (size_t)(lw / 2);
    size_t seq = text_rows + ref_rows + target_rows;

    out->text_rows = text_rows;
    out->reference_rows = ref_rows;
    out->target_rows = target_rows;
    out->sequence = seq;
    out->target_start = text_rows + ref_rows;
    out->positions = malloc(seq * 3 * sizeof(double));
    out->tags = malloc(seq);
    if ((out->positions == NULL || out->tags == NULL) && seq > 0) {
        h3i_layout_free(out);
        h3i_fail(error, error_size, "out of memory building the layout");
        return 0;
    }

    size_t row = 0;
    for (size_t i = 0; i < text_rows; i++) {
        out->positions[row * 3 + 0] = (double)i;
        out->positions[row * 3 + 1] = 0.0;
        out->positions[row * 3 + 2] = 0.0;
        out->tags[row] = text_tags ? text_tags[i] : 1;
        row++;
    }

    /* Image frames: time starts at T; reference k uses T+k, target T+ref_count. */
    int frames = ref_count + 1;
    for (int f = 0; f < frames; f++) {
        int fh = (f < ref_count) ? ref_lh[f] : lh;
        int fw = (f < ref_count) ? ref_lw[f] : lw;
        double time = (double)(text_rows + (size_t)f);
        double sqrt_area = sqrt((double)fh * (double)fw);
        double H[4096], W[4096];
        if (fh / 2 > 4096 || fw / 2 > 4096) {
            h3i_layout_free(out);
            h3i_fail(error, error_size, "latent grid too large");
            return 0;
        }
        grid_axis(fh, sqrt_area, H);
        grid_axis(fw, sqrt_area, W);
        for (int y = 0; y < fh / 2; y++) {
            for (int x = 0; x < fw / 2; x++) {
                out->positions[row * 3 + 0] = time;
                out->positions[row * 3 + 1] = H[y];
                out->positions[row * 3 + 2] = W[x];
                out->tags[row] = 0;
                row++;
            }
        }
    }
    return 1;
}

void h3i_layout_free(h3i_layout *l)
{
    free(l->positions);
    free(l->tags);
    l->positions = NULL;
    l->tags = NULL;
}

static const double default_sigmas[7] = {
    1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
    0.5327739119529724, 0.23200887441635132, 0.0,
};

int h3i_schedule_build(const double *cfg, h3i_schedule *out, char *error, size_t error_size)
{
    const double *s = cfg ? cfg : default_sigmas;
    for (int i = 0; i < 7; i++) {
        if (!(s[i] <= 1.0)) {
            h3i_fail(error, error_size, "invalid pdd_config sigmas");
            return 0;
        }
    }
    for (int i = 0; i < 6; i++) {
        if (!(s[i] > s[i + 1])) {
            h3i_fail(error, error_size, "invalid pdd_config sigmas");
            return 0;
        }
    }
    if (s[6] != 0.0) {
        h3i_fail(error, error_size, "invalid pdd_config sigmas");
        return 0;
    }
    for (int i = 0; i < H3I_STEPS; i++) {
        out->t[i] = (float)(1.0 - s[i]);   /* subtraction in double, then cast */
        out->sigma[i] = 1.0f - out->t[i];  /* float32 subtraction */
    }
    out->sigma[6] = 0.0f;
    out->t[6] = 0.999f;
    return 1;
}

void h3i_rope_tables(const double *positions, size_t rows, const float *inv_freq,
                     uint16_t *cos_bits, uint16_t *sin_bits)
{
    for (size_t row = 0; row < rows; row++) {
        for (int axis = 0; axis < 3; axis++) {
            float a = (float)positions[row * 3 + (size_t)axis];
            for (int f = 0; f < 16; f++) {
                float angle = a * inv_freq[f];
                size_t idx = row * 48 + (size_t)axis * 16 + (size_t)f;
                cos_bits[idx] = h3i_bf16_bits(cosf(angle));
                sin_bits[idx] = h3i_bf16_bits(sinf(angle));
            }
        }
    }
}

int h3i_tile_axis_plan(int extent, int tile, int min_overlap, h3i_tile_axis *out)
{
    memset(out, 0, sizeof *out);
    if (extent <= tile) {
        out->count = 1;
        out->length = extent;
        out->starts[0] = 0;
        return 1;
    }
    int count = (extent + tile - 1) / tile;
    while (tile * count - min_overlap * (count - 1) < extent) {
        count++;
    }
    if (count > H3I_MAX_TILES) {
        return 0;
    }
    for (int i = 0; i < count - 1; i++) {
        out->overlaps[i] = min_overlap;
    }
    int rem = tile * count - min_overlap * (count - 1) - extent;
    for (int u = 0; u < rem / 16; u++) {
        out->overlaps[u % (count - 1)] += 16;
    }
    out->starts[0] = 0;
    for (int i = 1; i < count; i++) {
        out->starts[i] = out->starts[i - 1] + tile - out->overlaps[i - 1];
    }
    out->count = count;
    out->length = tile;
    return 1;
}

uint8_t h3i_unit_to_u8(float v)
{
    if (!(v > 0.0f)) { /* also catches NaN */
        return 0;
    }
    if (v >= 1.0f) {
        return 255;
    }
    float m = v * 255.0f;
    if (m > 255.0f) {
        m = 255.0f;
    }
    return (uint8_t)truncf(m);
}
