/* Image I/O (SPEC §4.8): ImageIO decode with EXIF orientation and
 * alpha-over-black compositing; sRGB-tagged 8-bit RGB PNG write. */
#ifndef H3I_IO_IMAGE_H
#define H3I_IO_IMAGE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int width;
    int height;
    uint8_t *rgb; /* RGB24, row-major top-down */
} h3i_image;

/* Decode frame 0 upright: EXIF orientation applied (default 1, outside
 * 1..8 → 1), drawn in the image's own RGB color space (non-RGB → sRGB),
 * alpha dropped by compositing onto a zero-cleared buffer (transparent →
 * black). Caller releases with h3i_image_free. */
int h3i_image_read(const char *path, h3i_image *out,
                   char *error, size_t error_size);

/* PNG: 8-bit RGB, no alpha, tagged sRGB, row-major top-down. */
int h3i_image_write_png(const char *path, const uint8_t *rgb,
                        int width, int height, char *error, size_t error_size);

void h3i_image_free(h3i_image *image);

#endif /* H3I_IO_IMAGE_H */
