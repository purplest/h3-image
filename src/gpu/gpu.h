/* Derived from h3.c commit 8974cc0 (MIT, Salvatore Sanfilippo):
 * prefixes renamed h3_/H3_ -> h3i_/H3I_ and extended with the
 * h3-image additions (GGUF dequant convert, NE handoff). See SPEC.md
 * provenance. Do not hand-edit the derived portion; port from h3.c. */
#ifndef H3I_GPU_H
#define H3I_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct h3i_gpu h3i_gpu;
typedef struct h3i_gpu_tensor h3i_gpu_tensor;

typedef enum {
    H3I_GPU_F32 = 0,
    H3I_GPU_BF16,
    H3I_GPU_I8,
    H3I_GPU_U32
} h3i_gpu_dtype;

typedef struct {
    uint64_t allocated_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    uint64_t tensor_allocations;
    uint64_t direct_dispatches;
    uint64_t mps_linear_dispatches;
    uint64_t mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches;
    uint64_t blit_copies;
    uint64_t submissions;
    double command_encode_seconds;
    double command_wait_seconds;
    /* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
     * so command_wait_seconds is the complete turnaround measurement. */
    double gpu_seconds;
} h3i_gpu_stats;

h3i_gpu *h3i_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size);
void h3i_gpu_free(h3i_gpu *gpu);
int h3i_gpu_is_m5(const h3i_gpu *gpu);
int h3i_gpu_has_nax_mlp(const h3i_gpu *gpu);
int h3i_gpu_has_int8_mlp(const h3i_gpu *gpu);

h3i_gpu_tensor *h3i_gpu_tensor_new_f32(h3i_gpu *gpu, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_new_bf16(h3i_gpu *gpu, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_new_i8(h3i_gpu *gpu, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_from_f32(h3i_gpu *gpu, const float *values,
                                      size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_from_bf16(h3i_gpu *gpu, const uint16_t *values,
                                       size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_from_u32(h3i_gpu *gpu, const uint32_t *values,
                                      size_t elements);
/* Allocate shared Metal storage and pread BF16 payload directly into it. */
h3i_gpu_tensor *h3i_gpu_tensor_load_bf16(h3i_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_load_f32(h3i_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Fill an existing shared BF16 buffer from a file. The tensor and its
 * accounting are unchanged, so this may run on an I/O thread while another
 * tensor is in flight on the GPU. */
int h3i_gpu_tensor_read_file_bf16(h3i_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* As above, but ask Darwin to avoid retaining a second copy in the file cache.
 * Intended for large sequential weight streams whose destination is the only
 * useful resident copy. */
int h3i_gpu_tensor_stream_file_bf16(h3i_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size);
void h3i_gpu_tensor_free(h3i_gpu_tensor *tensor);
size_t h3i_gpu_tensor_elements(const h3i_gpu_tensor *tensor);
h3i_gpu_dtype h3i_gpu_tensor_dtype(const h3i_gpu_tensor *tensor);
int h3i_gpu_tensor_read_f32(const h3i_gpu_tensor *tensor, float *values,
                           size_t elements);
int h3i_gpu_tensor_read_f32_range(const h3i_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements);
int h3i_gpu_tensor_read_bf16(const h3i_gpu_tensor *tensor, uint16_t *values,
                            size_t elements);
int h3i_gpu_tensor_write_f32(h3i_gpu_tensor *tensor, const float *values,
                            size_t elements);
int h3i_gpu_tensor_write_f32_range(h3i_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements);
int h3i_gpu_tensor_write_bf16(h3i_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements);
int h3i_gpu_tensor_write_bf16_range(h3i_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements);

int h3i_gpu_begin(h3i_gpu *gpu);
/* Commit the current command buffer without waiting, then continue encoding on
 * the same ordered queue. h3i_gpu_submit() waits and validates the whole chain. */
int h3i_gpu_continue(h3i_gpu *gpu);
int h3i_gpu_submit(h3i_gpu *gpu);
const char *h3i_gpu_error(const h3i_gpu *gpu);
int h3i_gpu_get_stats(const h3i_gpu *gpu, h3i_gpu_stats *stats);
/* Optional benchmark labels. With H3I_PROFILE set, marks and context teardown
 * print wall time alongside command-buffer GPU time and allocation counters. */
void h3i_gpu_profile_set_label(h3i_gpu *gpu, const char *label);
void h3i_gpu_profile_mark(h3i_gpu *gpu, const char *phase);

int h3i_gpu_linear_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input, const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim);
int h3i_gpu_patch_linear_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                             const h3i_gpu_tensor *input,
                             const h3i_gpu_tensor *weight,
                             const h3i_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3i_gpu_patch_linear_bf16_offset(
                             h3i_gpu *gpu, h3i_gpu_tensor *output,
                             size_t output_offset,
                             const h3i_gpu_tensor *input, size_t input_offset,
                             const h3i_gpu_tensor *weight,
                             const h3i_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3i_gpu_patch_linear_bf16_map(
                             h3i_gpu *gpu, h3i_gpu_tensor *output,
                             const h3i_gpu_tensor *input,
                             const h3i_gpu_tensor *weight,
                             const h3i_gpu_tensor *bias,
                             const h3i_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3i_gpu_silu_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *input, uint32_t elements);
int h3i_gpu_cast_f32_to_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                            const h3i_gpu_tensor *input, uint32_t elements);
int h3i_gpu_cast_bf16_to_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                            const h3i_gpu_tensor *input, uint32_t elements);
int h3i_gpu_copy_bf16(h3i_gpu *gpu, h3i_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3i_gpu_tensor *source, size_t source_offset,
                     size_t elements);
int h3i_gpu_copy_f32(h3i_gpu *gpu, h3i_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3i_gpu_tensor *source, size_t source_offset,
                    size_t elements);
int h3i_gpu_rms_norm_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                        const h3i_gpu_tensor *input,
                        const h3i_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);
int h3i_gpu_adaln_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *input,
                     const h3i_gpu_tensor *norm_weight,
                     const h3i_gpu_tensor *modulation,
                     const h3i_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3i_gpu_gate_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *residual,
                    const h3i_gpu_tensor *branch,
                    const h3i_gpu_tensor *modulation,
                    const h3i_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3i_gpu_qkv_rope_f32(h3i_gpu *gpu, h3i_gpu_tensor *query,
                        h3i_gpu_tensor *key, h3i_gpu_tensor *value,
                        const h3i_gpu_tensor *qkv,
                        const h3i_gpu_tensor *q_norm,
                        const h3i_gpu_tensor *k_norm,
                        const h3i_gpu_tensor *rope_cos,
                        const h3i_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon);
int h3i_gpu_sdpa_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *query, const h3i_gpu_tensor *key,
                    const h3i_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale);
int h3i_gpu_swiglu_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *fused, uint32_t rows,
                      uint32_t width);
int h3i_gpu_scale_add_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                         const h3i_gpu_tensor *residual,
                         const h3i_gpu_tensor *branch,
                         const h3i_gpu_tensor *scale, uint32_t rows,
                         uint32_t width);
int h3i_gpu_layer_norm_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                          const h3i_gpu_tensor *input,
                          const h3i_gpu_tensor *weight,
                          const h3i_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);
int h3i_gpu_video_qkv_rope_f32(h3i_gpu *gpu, h3i_gpu_tensor *query,
                              h3i_gpu_tensor *key, h3i_gpu_tensor *value,
                              const h3i_gpu_tensor *qkv,
                              const h3i_gpu_tensor *rope_cos,
                              const h3i_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);

/* H3 AudioVAE uses time-major [batch,length,channels] activations and stores
 * Conv1d/ConvTranspose1d weights in PyTorch OIK/IOK order respectively. */
int h3i_gpu_conv1d_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input,
                      const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation);
int h3i_gpu_conv1d_stride_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input,
                      const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation);
int h3i_gpu_conv_transpose1d_f32(
                      h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input,
                      const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding);
int h3i_gpu_weight_norm_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *vector,
                           const h3i_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner);
int h3i_gpu_add_scaled_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                          const h3i_gpu_tensor *left,
                          const h3i_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements);
int h3i_gpu_alias_free_snake_f32(
                          h3i_gpu *gpu, h3i_gpu_tensor *output,
                          const h3i_gpu_tensor *input,
                          const h3i_gpu_tensor *alpha_log,
                          const h3i_gpu_tensor *beta_log,
                          const h3i_gpu_tensor *upsample_filter,
                          const h3i_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels);
int h3i_gpu_snake1d_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                       const h3i_gpu_tensor *input,
                       const h3i_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels);
int h3i_gpu_audio_qkv_split_f32(h3i_gpu *gpu,
                       h3i_gpu_tensor *query, h3i_gpu_tensor *key,
                       h3i_gpu_tensor *value, const h3i_gpu_tensor *qkv,
                       const h3i_gpu_tensor *q_bias,
                       const h3i_gpu_tensor *k_bias,
                       const h3i_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim);
int h3i_gpu_sdpa_causal_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                       const h3i_gpu_tensor *query,
                       const h3i_gpu_tensor *key,
                       const h3i_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale);
int h3i_gpu_audio_attention_pool_f32(h3i_gpu *gpu,
                       h3i_gpu_tensor *output,
                       const h3i_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim);
int h3i_gpu_geglu_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *gate,
                     const h3i_gpu_tensor *linear, uint32_t elements);
int h3i_gpu_clip_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum);

/* Visual-VAE encoder tensors use channels-last [B,T,H,W,C] storage. Spatial
 * padding reflects pixels while temporal front padding is zero-filled. */
int h3i_gpu_vae_encoder_pad_f32(
                    h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after);
int h3i_gpu_conv3d_f32(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input,
                      const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width);
int h3i_gpu_vae_encoder_group_norm_silu_f32(
                      h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input,
                      const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);

/* Portable BF16 storage path. Arithmetic accumulates in F32 and rounds at
 * operation boundaries, matching the released checkpoint's compute dtype. */
int h3i_gpu_linear_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                       const h3i_gpu_tensor *input,
                       const h3i_gpu_tensor *weight,
                       const h3i_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim);
int h3i_gpu_mlp_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *input,
                    const h3i_gpu_tensor *fc1_weight,
                    const h3i_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim);
/* Experimental M5 Metal 4 paired FC1/SwiGLU plus direct FC2 path. Available
 * only when the context was created with H3I_NAX=mlp. */
int h3i_gpu_mlp_nax_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                        h3i_gpu_tensor *activated,
                        const h3i_gpu_tensor *input,
                        const h3i_gpu_tensor *fc1_weight,
                        const h3i_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* Experimental M5 Metal 4 int8 MLP. Weights use one F32 scale per output
 * channel; activations are quantized dynamically with one F32 scale per row. */
int h3i_gpu_quantize_weight_int8(h3i_gpu *gpu, h3i_gpu_tensor *output,
                                h3i_gpu_tensor *scales,
                                const h3i_gpu_tensor *input, uint32_t rows,
                                uint32_t columns);
int h3i_gpu_linear_int8_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                            h3i_gpu_tensor *quantized_input,
                            h3i_gpu_tensor *input_scales,
                            const h3i_gpu_tensor *input,
                            const h3i_gpu_tensor *weight,
                            const h3i_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Consume SDPA's native [head,row,dimension] BF16 layout without a full
 * BF16 transpose, gathering directly into the projection's row-major int8. */
int h3i_gpu_linear_int8_head_major_bf16(
                            h3i_gpu *gpu, h3i_gpu_tensor *output,
                            h3i_gpu_tensor *quantized_input,
                            h3i_gpu_tensor *input_scales,
                            const h3i_gpu_tensor *input,
                            const h3i_gpu_tensor *weight,
                            const h3i_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim);
int h3i_gpu_mlp_int8_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                         h3i_gpu_tensor *activated,
                         h3i_gpu_tensor *quantized_activation,
                         h3i_gpu_tensor *activation_scales,
                         const h3i_gpu_tensor *input,
                         const h3i_gpu_tensor *fc1_weight,
                         const h3i_gpu_tensor *fc1_scales,
                         const h3i_gpu_tensor *fc2_weight,
                         const h3i_gpu_tensor *fc2_scales,
                         const h3i_gpu_tensor *fc1_bf16,
                         const h3i_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized);
int h3i_gpu_silu_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *input, uint32_t elements);
int h3i_gpu_rms_norm_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                         const h3i_gpu_tensor *input,
                         const h3i_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);
int h3i_gpu_layer_norm_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *input,
                           const h3i_gpu_tensor *weight,
                           const h3i_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon);
int h3i_gpu_gelu_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *input, uint32_t elements,
                     int approximate);
int h3i_gpu_vision_qkv_rope_bf16(
                     h3i_gpu *gpu, h3i_gpu_tensor *query,
                     h3i_gpu_tensor *key, h3i_gpu_tensor *value,
                     const h3i_gpu_tensor *qkv,
                     const h3i_gpu_tensor *rope_cos,
                     const h3i_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half);
int h3i_gpu_adaln_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input,
                      const h3i_gpu_tensor *norm_weight,
                      const h3i_gpu_tensor *modulation,
                      const h3i_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3i_gpu_adaln_bf16_offset(h3i_gpu *gpu, h3i_gpu_tensor *output,
                      const h3i_gpu_tensor *input, size_t input_offset,
                      const h3i_gpu_tensor *norm_weight,
                      const h3i_gpu_tensor *modulation,
                      const h3i_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3i_gpu_adaln_linear_bf16(
                      h3i_gpu *gpu, h3i_gpu_tensor *output,
                      h3i_gpu_tensor *inverse,
                      const h3i_gpu_tensor *input, size_t input_offset,
                      const h3i_gpu_tensor *norm_weight,
                      const h3i_gpu_tensor *modulation,
                      const h3i_gpu_tensor *row_map,
                      const h3i_gpu_tensor *weight,
                      const h3i_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon);
int h3i_gpu_gate_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *residual,
                     const h3i_gpu_tensor *branch,
                     const h3i_gpu_tensor *modulation,
                     const h3i_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3i_gpu_gate_adaln_bf16(
                     h3i_gpu *gpu, h3i_gpu_tensor *gated_residual,
                     h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *residual,
                     const h3i_gpu_tensor *branch,
                     const h3i_gpu_tensor *norm_weight,
                     const h3i_gpu_tensor *gate_modulation,
                     const h3i_gpu_tensor *norm_modulation,
                     const h3i_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon);
int h3i_gpu_gate_adaln_quantize_int8(
                     h3i_gpu *gpu, h3i_gpu_tensor *gated_residual,
                     h3i_gpu_tensor *quantized_output,
                     h3i_gpu_tensor *quantized_scales,
                     const h3i_gpu_tensor *residual,
                     const h3i_gpu_tensor *branch,
                     const h3i_gpu_tensor *norm_weight,
                     const h3i_gpu_tensor *gate_modulation,
                     const h3i_gpu_tensor *norm_modulation,
                     const h3i_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3i_gpu_qkv_rope_bf16_h3c(h3i_gpu *gpu, h3i_gpu_tensor *query,
                         h3i_gpu_tensor *key, h3i_gpu_tensor *value,
                         const h3i_gpu_tensor *qkv,
                         const h3i_gpu_tensor *q_norm,
                         const h3i_gpu_tensor *k_norm,
                         const h3i_gpu_tensor *rope_cos,
                         const h3i_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon);
/* H3 checkpoint QKV rows are [head, q/k/v, dimension], unlike the
 * conventional [q/k/v, head, dimension] layout accepted above. */
int h3i_gpu_grouped_qkv_rope_bf16_h3c(h3i_gpu *gpu, h3i_gpu_tensor *query,
                                 h3i_gpu_tensor *key, h3i_gpu_tensor *value,
                                 const h3i_gpu_tensor *qkv,
                                 const h3i_gpu_tensor *q_norm,
                                 const h3i_gpu_tensor *k_norm,
                                 const h3i_gpu_tensor *rope_cos,
                                 const h3i_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
/* Project grouped H3 QKV and apply its exact Q/K norm/RoPE boundary. Metal 4
 * may route projections directly into the attention layout; other devices
 * retain the ordinary two calls. */
int h3i_gpu_grouped_qkv_linear_rope_bf16(
                                 h3i_gpu *gpu,
                                 h3i_gpu_tensor *query,
                                 h3i_gpu_tensor *key,
                                 h3i_gpu_tensor *value,
                                 h3i_gpu_tensor *qkv,
                                 const h3i_gpu_tensor *input,
                                 const h3i_gpu_tensor *weight,
                                 const h3i_gpu_tensor *q_norm,
                                 const h3i_gpu_tensor *k_norm,
                                 const h3i_gpu_tensor *rope_cos,
                                 const h3i_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon);
int h3i_gpu_grouped_qkv_linear_rope_int8(
                                 h3i_gpu *gpu,
                                 h3i_gpu_tensor *query,
                                 h3i_gpu_tensor *key,
                                 h3i_gpu_tensor *value,
                                 h3i_gpu_tensor *quantized_input,
                                 h3i_gpu_tensor *input_scales,
                                 const h3i_gpu_tensor *input,
                                 const h3i_gpu_tensor *weight,
                                 const h3i_gpu_tensor *weight_scales,
                                 const h3i_gpu_tensor *q_norm,
                                 const h3i_gpu_tensor *k_norm,
                                 const h3i_gpu_tensor *rope_cos,
                                 const h3i_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
int h3i_gpu_sdpa_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *query, const h3i_gpu_tensor *key,
                     const h3i_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Preserve SDPA's native [head,row,dimension] output for an immediately
 * following layout-aware projection. */
int h3i_gpu_sdpa_bf16_head_major_output(
                     h3i_gpu *gpu, h3i_gpu_tensor *output,
                     const h3i_gpu_tensor *query, const h3i_gpu_tensor *key,
                     const h3i_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
int h3i_gpu_swiglu_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                       const h3i_gpu_tensor *fused, uint32_t rows,
                       uint32_t width);
int h3i_gpu_embedding_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                          const h3i_gpu_tensor *weight,
                          const h3i_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width);
int h3i_gpu_text_qk_rope_bf16(h3i_gpu *gpu,
                             h3i_gpu_tensor *query_output,
                             h3i_gpu_tensor *key_output,
                             const h3i_gpu_tensor *query_input,
                             const h3i_gpu_tensor *key_input,
                             const h3i_gpu_tensor *q_norm,
                             const h3i_gpu_tensor *k_norm,
                             const h3i_gpu_tensor *rope_cos,
                             const h3i_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon);
int h3i_gpu_head_rms_norm_bf16(h3i_gpu *gpu, h3i_gpu_tensor *tensor,
                              const h3i_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);
int h3i_gpu_rope_text_bf16(h3i_gpu *gpu, h3i_gpu_tensor *query,
                          h3i_gpu_tensor *key,
                          const h3i_gpu_tensor *rope_cos_f32,
                          const h3i_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim);
int h3i_gpu_gqa_causal_bf16_h3c(h3i_gpu *gpu, h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *query,
                           const h3i_gpu_tensor *key,
                           const h3i_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale);
int h3i_gpu_add_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *left, const h3i_gpu_tensor *right,
                    uint32_t elements);
int h3i_gpu_sub_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                    const h3i_gpu_tensor *left, const h3i_gpu_tensor *right,
                    uint32_t elements);
int h3i_gpu_token_pool_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *input,
                           size_t input_offset,
                           h3i_gpu_tensor *original,
                           size_t original_offset,
                           h3i_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3i_gpu_tensor *baseline_indices,
                           const h3i_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width);
int h3i_gpu_token_pool_adaln_bf16(
                           h3i_gpu *gpu, h3i_gpu_tensor *residual,
                           h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *input, size_t input_offset,
                           h3i_gpu_tensor *original, size_t original_offset,
                           h3i_gpu_tensor *baseline, size_t baseline_offset,
                           const h3i_gpu_tensor *baseline_indices,
                           const h3i_gpu_tensor *pairs,
                           const h3i_gpu_tensor *norm_weight,
                           const h3i_gpu_tensor *modulation,
                           const h3i_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
int h3i_gpu_token_expand_delta_bf16(
                           h3i_gpu *gpu, h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *original,
                           size_t original_offset,
                           const h3i_gpu_tensor *reduced,
                           const h3i_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3i_gpu_tensor *baseline_indices,
                           const h3i_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale);
int h3i_gpu_token_expand_adaln_bf16(
                           h3i_gpu *gpu, h3i_gpu_tensor *residual,
                           h3i_gpu_tensor *output,
                           const h3i_gpu_tensor *original,
                           size_t original_offset,
                           const h3i_gpu_tensor *reduced,
                           const h3i_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3i_gpu_tensor *baseline_indices,
                           const h3i_gpu_tensor *parents,
                           const h3i_gpu_tensor *norm_weight,
                           const h3i_gpu_tensor *modulation,
                           const h3i_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
/* Apply one Euler step to an F32 sample range from BF16 velocity caches:
 * sample += delta * (last + ratio * (last - previous)). */
int h3i_gpu_euler_bf16(h3i_gpu *gpu, h3i_gpu_tensor *sample,
                      size_t sample_offset, const h3i_gpu_tensor *last,
                      const h3i_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio);
int h3i_gpu_silu_mul_bf16(h3i_gpu *gpu, h3i_gpu_tensor *output,
                         const h3i_gpu_tensor *gate,
                         const h3i_gpu_tensor *up, uint32_t elements);

#endif

/* ---- h3-image additions (not in h3.c) ---- */

typedef struct {
    char name[128];
    char architecture[64];
    int apple_gpu_family;
    int metal4;
    uint64_t unified_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
} h3i_gpu_info;

int h3i_gpu_get_info(h3i_gpu *gpu, h3i_gpu_info *out);
uint64_t h3i_gpu_recommended_working_set(h3i_gpu *gpu);
void *h3i_gpu_tensor_contents(h3i_gpu_tensor *tensor);
h3i_gpu_tensor *h3i_gpu_tensor_new_u32(h3i_gpu *gpu, size_t elements);

/* GGML dequantization: raw I8 -> BF16/F32 (SPEC §5.3). */
int h3i_gpu_convert_raw(h3i_gpu *gpu, h3i_gpu_tensor *out, size_t out_off,
                        const h3i_gpu_tensor *raw, size_t raw_off, int type,
                        size_t elements);

/* Neural Engine handoff (SPEC §5.4); out is I8 storage holding FP16. */
int h3i_gpu_bf16_to_f16(h3i_gpu *gpu, h3i_gpu_tensor *out,
                        const h3i_gpu_tensor *in, size_t n, float scale);
int h3i_gpu_f16_to_bf16_checked(h3i_gpu *gpu, h3i_gpu_tensor *out,
                                const h3i_gpu_tensor *in, size_t n, float scale,
                                h3i_gpu_tensor *flag, uint32_t bits);
int h3i_gpu_transpose_bf16_f16(h3i_gpu *gpu, h3i_gpu_tensor *out,
                               const h3i_gpu_tensor *in, size_t row_off,
                               size_t rows, size_t cols, float scale);

/* Spec-exact QKV RoPE (grouped flag) and chunked causal GQA; the reference
 * MLX-matched variants are available as *_h3c. */
int h3i_gpu_qkv_rope_bf16(h3i_gpu *gpu, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                          h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                          const h3i_gpu_tensor *cosb, const h3i_gpu_tensor *sinb,
                          const h3i_gpu_tensor *q_norm, const h3i_gpu_tensor *k_norm,
                          size_t rows, size_t heads, size_t hd, size_t rope_half,
                          float eps, int grouped);
int h3i_gpu_grouped_qkv_rope_bf16(h3i_gpu *gpu, h3i_gpu_tensor *q,
                                  h3i_gpu_tensor *k, h3i_gpu_tensor *v,
                                  const h3i_gpu_tensor *qkv,
                                  const h3i_gpu_tensor *q_norm,
                                  const h3i_gpu_tensor *k_norm,
                                  const h3i_gpu_tensor *rope_cos,
                                  const h3i_gpu_tensor *rope_sin,
                                  size_t rows, size_t heads, size_t hd,
                                  size_t rope_half, float eps);
int h3i_gpu_gqa_causal_bf16(h3i_gpu *gpu, h3i_gpu_tensor *out,
                            const h3i_gpu_tensor *q, const h3i_gpu_tensor *k,
                            const h3i_gpu_tensor *v, size_t seq, size_t qheads,
                            size_t kvheads, size_t hd, float scale);
