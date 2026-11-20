#ifndef H3I_VIDEO_HOST_H
#define H3I_VIDEO_HOST_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define H3I_CANVAS_MULTIPLE 32
#define H3I_MAX_PIXELS (768 * 1344)
#define H3I_FPS 24
#define H3I_AUDIO_LATENT_FPS 40
#define H3I_VAE_SPATIAL_RATIO 16
#define H3I_VIDEO_SIGMA_SHIFT 12.0
#define H3I_AUDIO_SIGMA_SHIFT 3.0
#define H3I_MAX_STEPS 1000

typedef struct {
    int frame_count;
    int video_t;
    int audio_t;
} h3i_temporal_shape;

typedef struct {
    double t;
    double h;
    double w;
} h3i_position;

typedef enum {
    H3I_SEG_TEXT,
    H3I_SEG_COND,
    H3I_SEG_REF_IMAGE,
    H3I_SEG_REF_AUDIO,
    H3I_SEG_AUDIO,
    H3I_SEG_VIDEO
} h3i_segment_kind;

typedef struct {
    size_t start;
    size_t stop;
    h3i_segment_kind kind;
} h3i_segment;

typedef enum {
    H3I_VIDEO_LAYOUT_REF_IMAGE,
    H3I_VIDEO_LAYOUT_REF_AUDIO,
    H3I_VIDEO_LAYOUT_REF_VIDEO
} h3i_video_layout_ref_kind;

typedef struct {
    h3i_video_layout_ref_kind kind;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
} h3i_video_layout_ref;

typedef struct {
    int text_len;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    int frame_count;
    const int *keyframes;
    size_t keyframe_count;
    const h3i_video_layout_ref *references;
    size_t reference_count;
} h3i_video_layout_spec;

typedef struct {
    size_t seq_len;
    h3i_segment *segments;
    size_t segment_count;
    h3i_position *positions;
    size_t img_cond_rows;
    size_t img_target_rows;
    size_t audio_cond_rows;
    size_t audio_target_rows;
    int signature[5];
} h3i_video_layout;

typedef struct {
    int steps;
    float video[H3I_MAX_STEPS + 1];
    float audio[H3I_MAX_STEPS + 1];
} h3i_video_sigma_schedule;

int h3i_align_frame_count(int requested);
int h3i_video_latent_t(int frame_count);
int h3i_video_encoder_latent_t(int frame_count);
h3i_temporal_shape h3i_temporal(int requested_frames);
void h3i_latent_canvas(int width, int height, int *latent_w, int *latent_h);
int h3i_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h);
/* Ref2VA image sizing is down-only and aspect preserving. max_short_edge=0
 * matches the output pixel area; a positive value selects a short-edge cap. */
int h3i_reference_image_canvas(int width, int height,
                              int target_width, int target_height,
                              int max_short_edge,
                              int *adapted_w, int *adapted_h);
/* Ref2VA video references use the normal target-style canvas, except that a
 * smaller source is never enlarged. */
int h3i_reference_video_canvas(int width, int height,
                              int *adapted_w, int *adapted_h);

double h3i_time_shift_sigma(double sigma, double from_shift, double to_shift);
double h3i_time_shift_slope(double sigma, double from_shift, double to_shift);
int h3i_video_schedule_build(int steps, h3i_video_sigma_schedule *schedule);
/* Released linear base grid: evaluations model forwards plus terminal zero. */
int h3i_video_serving_schedule_build(int evaluations, h3i_video_sigma_schedule *schedule);

int h3i_video_layout_build(const h3i_video_layout_spec *spec, h3i_video_layout *layout,
                    char *error, size_t error_size);
void h3i_video_layout_free(h3i_video_layout *layout);
const char *h3i_segment_name(h3i_segment_kind kind);

int h3i_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, size_t count,
                const float *sigmas, int step, int total_steps);

#endif
