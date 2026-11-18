/* Resolved component paths and shader lookup (SPEC §2.6, §6.1.2, §6.3.4). */
#ifndef H3I_APP_PATHS_H
#define H3I_APP_PATHS_H

#include <stddef.h>

#define H3I_PATH_MAX 4096

typedef struct {
    char dit[H3I_PATH_MAX];
    char dit_heads[H3I_PATH_MAX];
    char text_encoder[H3I_PATH_MAX];
    char vae[H3I_PATH_MAX];
    char vae_config[H3I_PATH_MAX];
    char tokenizer[H3I_PATH_MAX];
    char pdd_config[H3I_PATH_MAX];
} h3i_paths;

/* Copy src into dst, truncating to dst_size - 1 bytes, always NUL-terminated. */
void h3i_path_copy(char *dst, size_t dst_size, const char *src);

/* Combined Metal shader source lookup (SPEC §6.3.4):
 * - $H3I_SHADERS if set non-empty: realpath, must be a regular file, else NULL
 *   (no fallback).
 * - otherwise relative to the realpath'd executable's directory, first regular
 *   file of: build/h3i_shaders.metal, h3i_shaders.metal, ../build/h3i_shaders.metal.
 * Returns buf on success, NULL when not found. */
const char *h3i_shader_path(char *buf, size_t buf_size);

#endif /* H3I_APP_PATHS_H */
