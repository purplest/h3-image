#ifndef H3I_DIT_H
#define H3I_DIT_H

#include "gpu/gpu.h"
#include "io/video_host.h"
#include "model/text_encoder.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3i_dit h3i_dit;

typedef void (*h3i_dit_progress)(const char *phase, int completed, int total,
                                void *opaque);

typedef int (*h3i_dit_preview)(int completed_steps, int total_steps,
                              const float *video_latent,
                              size_t video_elements, void *opaque);

/* Load a text-only FL2VA transformer. Text refinement and AdaLN precomputation
 * happen before the persistent core is loaded. SSD streaming retains only the
 * small block norms and two alternating BF16 matrix slots. */
h3i_dit *h3i_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3i_text_embedding *text,
                         const h3i_video_layout *layout,
                         const h3i_video_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         h3i_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Load FL2VA/Ref2VA packing. Condition inputs are already patchified F32 row
 * sources: visual rows have width 96 and audio rows width 32. Their element
 * counts must exactly match layout.img_cond_rows/audio_cond_rows. */
h3i_dit *h3i_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3i_text_embedding *text,
                         const h3i_video_layout *layout,
                         const h3i_video_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3i_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);
void h3i_dit_free(h3i_dit *dit);

/* Reset mutable sampler state and replace seed-dependent condition rows before
 * reusing an otherwise identical prepared transformer. */
int h3i_dit_reset_run(h3i_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size);

size_t h3i_dit_video_elements(const h3i_dit *dit);
size_t h3i_dit_audio_elements(const h3i_dit *dit);

/* One raw data-ward velocity evaluation. Input/output video layout is
 * [24,T,H,W], audio is [32,2,T], all F32 on the host boundary. */
int h3i_dit_forward(h3i_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size);

/* Run every configured RES multistep update in place. Solver state remains
 * F32 between transformer evaluations, matching the released MLX path. */
int h3i_dit_denoise(h3i_dit *dit, float *video_latent, float *audio_latent,
                   h3i_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size);

/* Current serving sampler: independent video/audio shifted Euler grids. */
int h3i_dit_denoise_euler(h3i_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3i_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Preview variant. The callback runs after every Euler transition with the
 * current channel-major F32 video latent. It is deliberately opt-in because it
 * introduces a synchronization point after each step. */
int h3i_dit_denoise_euler_preview(
                         h3i_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3i_dit_progress progress, void *progress_opaque,
                         h3i_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size);

/* Build the velocity-evaluation mask used by the serving sampler. Returns the
 * evaluation count, or -1 for invalid arguments. Public internally so the
 * quality-tuned aggressive schedule remains pinned by cheap host tests. */
int h3i_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count);

int h3i_dit_get_gpu_stats(const h3i_dit *dit, h3i_gpu_stats *stats);

/* Exact row-order conversions, public internally so they can be pinned by
 * cheap tests independently of the 62 GiB checkpoint. */
int h3i_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements);
int h3i_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements);
int h3i_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements);
int h3i_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements);

#endif
