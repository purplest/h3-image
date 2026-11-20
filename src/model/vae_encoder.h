#ifndef H3I_VIDEO_ENCODER_H
#define H3I_VIDEO_ENCODER_H

#include "gpu/gpu.h"

#include <stddef.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3i_gpu_stats gpu_stats;
} h3i_video_latent;

typedef void (*h3i_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. */
int h3i_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3i_video_encoder_progress progress, void *progress_opaque,
                        h3i_video_latent *output,
                        char *error, size_t error_size);
void h3i_video_latent_free(h3i_video_latent *latent);

#endif
