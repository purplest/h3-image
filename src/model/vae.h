/* VAE decoder — 2D ViT (SPEC §4.5). All F32; weights F16/F32 converted to
 * F32 at load. Tiled 256px / min-overlap 64 decode with host stitching in
 * normalized pixel space, then ImageNet un-normalize. */
#ifndef H3I_MODEL_VAE_H
#define H3I_MODEL_VAE_H

#include "gpu/gpu.h"
#include "weights/source.h"

#include <stddef.h>

#define H3I_VAE_LATENT_CHANNELS 24
#define H3I_VAE_RATIO 16

typedef struct h3i_vae_decoder h3i_vae_decoder;

/* latents_mean/latents_std: built-in defaults (SPEC §4.5); when config_path
 * is non-NULL and the file exists, its "latents_mean"/"latents_std" arrays
 * (24 numbers each) override them. NULL/absent file → defaults. */
int h3i_vae_config_load(const char *config_path, float *mean, float *std,
                        char *error, size_t error_size);

/* Load every decoder tensor as F32 (dequantizing/converting as needed).
 * Encodes conversions into its own command buffer and submits. */
h3i_vae_decoder *h3i_vae_decoder_load(h3i_gpu *gpu, const h3i_source *source,
                                      const char *config_path,
                                      char *error, size_t error_size);
void h3i_vae_decoder_free(h3i_vae_decoder *d);

/* normalized latent [24][lh][lw] F32 → un-normalized RGB F32 [lh*16][lw*16][3]
 * (malloc'd, caller frees). lh/lw must be positive multiples of 2. Values
 * may slightly exceed [0,1]; h3i_unit_to_u8 clamps (SPEC §4.5). */
int h3i_vae_decoder_decode(h3i_vae_decoder *d, const float *latent,
                           int lh, int lw, float **rgb,
                           char *error, size_t error_size);

#endif /* H3I_MODEL_VAE_H */
