#ifndef H3I_TERMINAL_H
#define H3I_TERMINAL_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3I_TERM_NONE = 0,
    H3I_TERM_KITTY,
    H3I_TERM_ITERM2
} h3i_terminal_protocol;

h3i_terminal_protocol h3i_terminal_detect(void);
const char *h3i_terminal_protocol_name(h3i_terminal_protocol protocol);

/* Terminal display dimensions default to 2x for macOS Retina displays. */
int h3i_terminal_set_zoom(int zoom);
int h3i_terminal_display_dimensions(int width, int height,
                                   int *display_width, int *display_height);

int h3i_terminal_display_rgb24(h3i_terminal_protocol protocol,
                              const uint8_t *rgb, int width, int height,
                              int stride, char *error, size_t error_size);

#endif
