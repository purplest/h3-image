/* GPU runtime abstraction (SPEC §1.3, §6.3).
 *
 * Pure-C public API; the implementation is Objective-C/Metal (context.m,
 * tensor.m, ops_h3i.m) and only builds on Apple platforms. Tensors live in
 * shared (CPU+GPU visible) storage. Work is encoded into one open command
 * buffer on a single serial queue:
 *
 *   begin → op calls (encode) → [continue → op calls]* → submit (wait all)
 *
 * Every op validates dtypes/sizes, sets the context's last error and returns
 * 0 on failure. BF16 ops accumulate in F32 and round once at the output. */
#ifndef H3I_GPU_GPU_H
#define H3I_GPU_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3I_DTYPE_F32 = 0,   /* 4 bytes */
    H3I_DTYPE_BF16 = 1,  /* 2 bytes */
    H3I_DTYPE_I8 = 2,    /* 1 byte, raw storage */
    H3I_DTYPE_U32 = 3,   /* 4 bytes */
} h3i_dtype;

typedef struct h3i_gpu h3i_gpu;
typedef struct h3i_gpu_tensor h3i_gpu_tensor;

/* ---- context -------------------------------------------------------- */

/* Compiles the combined shader source at shader_path with safe math mode
 * (no fast math) and builds every pipeline eagerly. */
h3i_gpu *h3i_gpu_create(const char *shader_path, char *error, size_t error_size);
void h3i_gpu_free(h3i_gpu *g);
const char *h3i_gpu_error(h3i_gpu *g);

typedef struct {
    char name[128];
    char architecture[64];
    int apple_gpu_family;
    int metal4;
    uint64_t unified_memory;          /* physical bytes */
    uint64_t recommended_working_set; /* Metal budget bytes */
    uint64_t max_buffer_length;
} h3i_gpu_info;

int h3i_gpu_get_info(h3i_gpu *g, h3i_gpu_info *out);
uint64_t h3i_gpu_recommended_working_set(h3i_gpu *g);

/* ---- command model (SPEC §6.3.2) ------------------------------------ */

int h3i_gpu_begin(h3i_gpu *g);     /* fails if a buffer is open or in flight */
int h3i_gpu_continue(h3i_gpu *g);  /* commit without waiting, open a new one */
int h3i_gpu_submit(h3i_gpu *g);    /* commit and wait for all in-flight work */

/* ---- tensors (SPEC §6.3.1, §6.3.3) ---------------------------------- */

h3i_gpu_tensor *h3i_gpu_tensor_new(h3i_gpu *g, h3i_dtype dt, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_new_f32(h3i_gpu *g, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_new_bf16(h3i_gpu *g, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_new_i8(h3i_gpu *g, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_new_u32(h3i_gpu *g, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_from_f32(h3i_gpu *g, const float *host, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_from_bf16(h3i_gpu *g, const uint16_t *host, size_t elements);
h3i_gpu_tensor *h3i_gpu_tensor_from_u32(h3i_gpu *g, const uint32_t *host, size_t elements);
void h3i_gpu_tensor_free(h3i_gpu_tensor *t);

void *h3i_gpu_tensor_contents(h3i_gpu_tensor *t);
size_t h3i_gpu_tensor_elements(const h3i_gpu_tensor *t);
size_t h3i_gpu_tensor_bytes(const h3i_gpu_tensor *t);
h3i_dtype h3i_gpu_tensor_dtype(const h3i_gpu_tensor *t);

/* Copy out (GPU results are visible after submit). */
int h3i_gpu_read_f32(h3i_gpu_tensor *t, float *out, size_t elements);
int h3i_gpu_read_bf16(h3i_gpu_tensor *t, uint16_t *out, size_t elements);

/* pread directly into a fresh tensor's buffer. */
h3i_gpu_tensor *h3i_gpu_load_bf16(h3i_gpu *g, const char *path, uint64_t offset,
                                  size_t elements, char *error, size_t error_size);
h3i_gpu_tensor *h3i_gpu_load_f32(h3i_gpu *g, const char *path, uint64_t offset,
                                 size_t elements, char *error, size_t error_size);

/* pread bytes into an existing tensor at tensor_offset (bytes). uncached sets
 * F_NOCACHE. Thread-safe with respect to GPU work on other tensors. */
int h3i_gpu_read_file_bytes(h3i_gpu *g, h3i_gpu_tensor *t, uint64_t tensor_offset,
                            const char *path, uint64_t file_offset, uint64_t bytes,
                            int uncached, char *error, size_t error_size);

/* ---- operations (SPEC §6.3.5) ---------------------------------------
 * Shapes row-major "rows×width"; weights [out, in]; y = x·Wᵀ + b.
 * b may be NULL where the spec has no bias. All return 1 ok / 0 failure. */

int h3i_gpu_linear_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                       size_t rows, size_t in_w, size_t out_w);

/* Patch embed: in/W/b F32 → out BF16, with element offsets. */
int h3i_gpu_patch_linear_bf16_offset(h3i_gpu *g, h3i_gpu_tensor *out, size_t out_off,
                                     const h3i_gpu_tensor *in, size_t in_off,
                                     const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                                     size_t rows, size_t in_w, size_t out_w);

int h3i_gpu_linear_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                        const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                        size_t rows, size_t in_w, size_t out_w);

/* Fused SwiGLU MLP: fc1 [2*hidden, in] rows [gate|up], fc2 [out, hidden]. */
int h3i_gpu_mlp_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                     const h3i_gpu_tensor *fc1, const h3i_gpu_tensor *fc2,
                     size_t rows, size_t in_w, size_t hidden, size_t out_w);

int h3i_gpu_silu_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in, size_t n);
int h3i_gpu_cast_f32_to_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in, size_t n);
int h3i_gpu_cast_bf16_to_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in, size_t n);

int h3i_gpu_copy_bf16(h3i_gpu *g, h3i_gpu_tensor *dst, size_t doff,
                      const h3i_gpu_tensor *src, size_t soff, size_t n);
int h3i_gpu_copy_f32(h3i_gpu *g, h3i_gpu_tensor *dst, size_t doff,
                     const h3i_gpu_tensor *src, size_t soff, size_t n);

int h3i_gpu_rms_norm_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                         const h3i_gpu_tensor *w, size_t rows, size_t width, float eps);
int h3i_gpu_rms_norm_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                          const h3i_gpu_tensor *w, size_t rows, size_t width, float eps);
int h3i_gpu_layer_norm_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                           const h3i_gpu_tensor *w, const h3i_gpu_tensor *b,
                           size_t rows, size_t width, float eps);
int h3i_gpu_layer_norm_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                            const h3i_gpu_tensor *w, const h3i_gpu_tensor *b,
                            size_t rows, size_t width, float eps);

/* Full non-causal attention, layout [seq][heads][hd] (SPEC §4.0 SDPA). */
int h3i_gpu_sdpa_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                     const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                     size_t seq, size_t heads, size_t hd, float scale);
int h3i_gpu_sdpa_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                      const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                      size_t seq, size_t heads, size_t hd, float scale);

/* fused [rows][2*width] → silu(first half)·second half. */
int h3i_gpu_swiglu_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *fused,
                       size_t rows, size_t width);
int h3i_gpu_swiglu_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *fused,
                        size_t rows, size_t width);

int h3i_gpu_scale_add_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *res,
                          const h3i_gpu_tensor *branch, const h3i_gpu_tensor *scale,
                          size_t rows, size_t width);
int h3i_gpu_add_scaled_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *l,
                           const h3i_gpu_tensor *r, float ls, float rs, size_t n);

/* QKV split + per-head q/k RMS norm (weights optional) + RoPE.
 * qkv layout: grouped=0 → [q|k|v][head][dim]; grouped=1 → [head][q|k|v][dim].
 * cos/sin are BF16 tables [rows][2*rope_half] (axis-major within the row);
 * rope_half=0 disables rotation. Outputs [rows][heads][hd] BF16 (F32 variant
 * for the VAE decoder, whose tables are F32 and q/k norm has no weights). */
int h3i_gpu_qkv_rope_bf16(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                          h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                          const h3i_gpu_tensor *cosb, const h3i_gpu_tensor *sinb,
                          const h3i_gpu_tensor *q_norm, const h3i_gpu_tensor *k_norm,
                          size_t rows, size_t heads, size_t hd, size_t rope_half,
                          float eps, int grouped);
int h3i_gpu_video_qkv_rope_f32(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                               h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                               const h3i_gpu_tensor *cos_t, const h3i_gpu_tensor *sin_t,
                               size_t rows, size_t heads, size_t hd, size_t rope_half,
                               float eps);
int h3i_gpu_vision_qkv_rope_bf16(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                                 h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                                 const h3i_gpu_tensor *cosb, const h3i_gpu_tensor *sinb,
                                 size_t rows, size_t heads, size_t hd, size_t rope_half);

/* In-place per-head RMS norm (text encoder q/k), BF16. */
int h3i_gpu_head_rms_norm_bf16(h3i_gpu *g, h3i_gpu_tensor *io, const h3i_gpu_tensor *w,
                               size_t rows, size_t heads, size_t hd, float eps);

/* In-place rotate-half RoPE on q and k, GQA (k has kvheads), all hd dims
 * rotated in (d, d+hd/2) pairs. cos/sin F32 tensors holding BF16-rounded
 * values, [rows][hd/2]. */
int h3i_gpu_rope_text_bf16(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                           const h3i_gpu_tensor *cos_t, const h3i_gpu_tensor *sin_t,
                           size_t rows, size_t qheads, size_t kvheads, size_t hd);

/* AdaLN: out = bf16(norm(x)·(1+scale[row_map[r]]) + shift[row_map[r]]).
 * mod table [slots][width] per modulation row; row_map U32 [rows]. */
int h3i_gpu_adaln_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *normw, const h3i_gpu_tensor *mod,
                       const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                       size_t slots, size_t shift_slot, size_t scale_slot, float eps);
int h3i_gpu_adaln_bf16_offset(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                              size_t in_off, const h3i_gpu_tensor *normw,
                              const h3i_gpu_tensor *mod, const h3i_gpu_tensor *row_map,
                              size_t rows, size_t width, size_t slots,
                              size_t shift_slot, size_t scale_slot, float eps);

/* Gate: out = bf16(res + branch·gate[row_map[r]]). */
int h3i_gpu_gate_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *res,
                      const h3i_gpu_tensor *branch, const h3i_gpu_tensor *mod,
                      const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                      size_t slots, size_t gate_slot);

/* Fused gate then AdaLN of the gated result; writes both outputs. */
int h3i_gpu_gate_adaln_bf16(h3i_gpu *g, h3i_gpu_tensor *out_res, h3i_gpu_tensor *out_adaln,
                            const h3i_gpu_tensor *res, const h3i_gpu_tensor *branch,
                            const h3i_gpu_tensor *normw, const h3i_gpu_tensor *mod,
                            const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                            size_t slots, size_t gate_slot, size_t shift_slot,
                            size_t scale_slot, float eps);

int h3i_gpu_gelu_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                      size_t n, int approximate);
int h3i_gpu_add_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *l,
                     const h3i_gpu_tensor *r, size_t n);
int h3i_gpu_silu_mul_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *gate,
                          const h3i_gpu_tensor *up, size_t n);

/* VAE encoder helpers: NDHWC F32. pad: spatial reflect, temporal front zero
 * (1 frame → 3 frames, input at the last). */
int h3i_gpu_vae_encoder_pad_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                                size_t height, size_t width, size_t channels, size_t pad);
/* MPSGraph Conv3D, NDHWC activations, OIDHW weights, explicit padding none. */
int h3i_gpu_conv3d_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                       size_t n, size_t d, size_t h, size_t w, size_t c, size_t o,
                       size_t kd, size_t kh, size_t kw,
                       size_t sd, size_t sh, size_t sw);
int h3i_gpu_vae_encoder_group_norm_silu_f32(h3i_gpu *g, h3i_gpu_tensor *out,
                                            const h3i_gpu_tensor *in,
                                            const h3i_gpu_tensor *w, const h3i_gpu_tensor *b,
                                            size_t n, size_t h, size_t width, size_t c,
                                            size_t groups, float eps);

/* GGML dequantization: raw I8 → BF16/F32 (SPEC §5.3). elements must be
 * block-aligned; one thread per chunk. */
int h3i_gpu_convert_raw(h3i_gpu *g, h3i_gpu_tensor *out, size_t out_off,
                        const h3i_gpu_tensor *raw, size_t raw_off, int type,
                        size_t elements);

/* Race-free causal GQA attention (SPEC §6.3.5): exact softmax ≤4096 keys,
 * chunked F32 online softmax beyond. qh % kvh == 0, hd ≤ 128. */
int h3i_gpu_gqa_causal_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                            const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                            size_t seq, size_t qheads, size_t kvheads, size_t hd,
                            float scale);

/* Neural Engine handoff (SPEC §5.4). out is an I8 tensor holding FP16. */
int h3i_gpu_bf16_to_f16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                        size_t n, float scale);
int h3i_gpu_f16_to_bf16_checked(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                                size_t n, float scale, h3i_gpu_tensor *flag, uint32_t bits);
int h3i_gpu_transpose_bf16_f16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                               size_t row_off, size_t rows, size_t cols, float scale);

/* ---- profiling / stats ---------------------------------------------- */

void h3i_gpu_profile_set_label(h3i_gpu *g, const char *label);

typedef struct {
    uint64_t allocated_bytes, live_bytes, peak_bytes;
    uint64_t allocations, direct_dispatches, mps_linear, mps_conv, mps_sdpa;
    uint64_t blits, submissions;
    double encode_seconds, wait_seconds, gpu_seconds;
} h3i_gpu_stats;

void h3i_gpu_get_stats(h3i_gpu *g, h3i_gpu_stats *out);

#endif /* H3I_GPU_GPU_H */
