/* Canvas, patchify, packed sequence layout, schedule, RoPE tables, tile
 * plans, uint8 conversion (SPEC §3.2–3.6, §4.5). All host math: double where
 * the spec says double, separately-rounded ops (FP contraction OFF). */
#ifndef H3I_MODEL_LAYOUT_H
#define H3I_MODEL_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#define H3I_LATENT_CHANNELS 24
#define H3I_VIDEO_PATCH 96
#define H3I_STEPS 6
#define H3I_CANVAS_MULTIPLE 32
#define H3I_VAE_RATIO 16
#define H3I_DEFAULT_MIN_AREA 4194304LL   /* 2048² */
#define H3I_MAX_AREA 16777216LL          /* 4096² */

typedef struct {
    int width, height;          /* requested */
    int canvas_w, canvas_h;     /* multiples of 32 */
    int latent_w, latent_h;     /* canvas / 16, always even */
} h3i_canvas;

int h3i_canvas_resolve(int width, int height, long long min_area,
                       h3i_canvas *out, char *error, size_t error_size);

/* Latent [24][lh][lw] F32 ↔ rows [(lh/2)*(lw/2)][96] (SPEC §3.3). */
void h3i_patchify(const float *latent, int lh, int lw, float *rows);
void h3i_unpatchify(const float *rows, int lh, int lw, float *latent);

typedef struct {
    size_t text_rows;
    size_t reference_rows;
    size_t target_rows;
    size_t sequence;
    size_t target_start;        /* = text_rows + reference_rows */
    double *positions;          /* [sequence][3] */
    uint8_t *tags;              /* [sequence]: 0 image, 1 text */
} h3i_layout;

/* Sequence = [text | ref 1..k | target] (SPEC §3.4). text_tags may be NULL
 * (all 1). ref_lh/ref_lw arrays have ref_count entries (latent sides). */
int h3i_layout_build(size_t text_rows, const uint8_t *text_tags,
                     int ref_count, const int *ref_lh, const int *ref_lw,
                     int lh, int lw, h3i_layout *out, char *error, size_t error_size);
void h3i_layout_free(h3i_layout *l);

typedef struct {
    float t[7];                 /* AdaLN timesteps; t[6] = 0.999f (references) */
    float sigma[7];             /* sigma[6] = 0 */
} h3i_schedule;

/* cfg: 7 sigmas from pdd_config, or NULL for the released schedule. */
int h3i_schedule_build(const double *cfg, h3i_schedule *out, char *error, size_t error_size);

/* 3-axis RoPE cos/sin tables, BF16 bits, [rows][48] (SPEC §3.5).
 * positions [rows][3] double; inv_freq[16] F32 from the checkpoint. */
void h3i_rope_tables(const double *positions, size_t rows, const float *inv_freq,
                     uint16_t *cos_bits, uint16_t *sin_bits);

#define H3I_MAX_TILES 128

typedef struct {
    int count;
    int length;                 /* tile side (extent when count == 1) */
    int starts[H3I_MAX_TILES];
    int overlaps[H3I_MAX_TILES]; /* overlaps[i] between tile i and i+1 */
} h3i_tile_axis;

/* extent multiple of 16; tile 256, min_overlap 64 (SPEC §4.5). */
int h3i_tile_axis_plan(int extent, int tile, int min_overlap, h3i_tile_axis *out);

/* Normalized [0,1] float → uint8: truncation, NaN → 0 (SPEC §4.5). */
uint8_t h3i_unit_to_u8(float v);

#endif /* H3I_MODEL_LAYOUT_H */
