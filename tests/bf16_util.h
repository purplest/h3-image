/* BF16 comparison helpers for GPU tests: distances in BF16 ULP steps. */
#ifndef H3I_TESTS_BF16_UTIL_H
#define H3I_TESTS_BF16_UTIL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float bf16_f(uint16_t h)
{
    uint32_t b = (uint32_t)h << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static inline uint16_t f_bf16(float x)
{
    uint32_t b;
    memcpy(&b, &x, 4);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

static inline int bf16_is_nan(uint16_t h)
{
    return (h & 0x7F80u) == 0x7F80u && (h & 0x007Fu) != 0;
}

/* ULP distance between two BF16 values (bit distance for same sign; via
 * zero for opposite signs). NaN vs non-NaN is a large distance. */
static inline int bf16_steps(uint16_t a, uint16_t b)
{
    int na = bf16_is_nan(a), nb = bf16_is_nan(b);
    if (na || nb) {
        return (na && nb) ? 0 : 1000000;
    }
    if ((a >> 15) == (b >> 15)) {
        return abs((int)a - (int)b);
    }
    return (int)(a & 0x7FFFu) + (int)(b & 0x7FFFu);
}

#endif /* H3I_TESTS_BF16_UTIL_H */
