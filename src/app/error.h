/* h3-image error convention (SPEC §1.3, §6.2):
 * every fallible function takes (char *error, size_t error_size), writes one
 * printf-formatted message on failure via h3i_fail, and returns 0/NULL.
 * h3i_fail is a no-op when error is NULL. Messages are always NUL-terminated
 * and truncated to error_size - 1 bytes. The CLI prints "h3-image: <message>"
 * and exits 1 (runtime) or 2 (usage). */
#ifndef H3I_APP_ERROR_H
#define H3I_APP_ERROR_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#if defined(__GNUC__) || defined(__clang__)
#define H3I_PRINTF_LIKE(fmt, args) __attribute__((format(printf, fmt, args)))
#else
#define H3I_PRINTF_LIKE(fmt, args)
#endif

static inline void h3i_fail(char *error, size_t error_size, const char *fmt, ...)
    H3I_PRINTF_LIKE(3, 4);

static inline void h3i_fail(char *error, size_t error_size, const char *fmt, ...)
{
    if (error == NULL || error_size == 0) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

#endif /* H3I_APP_ERROR_H */
