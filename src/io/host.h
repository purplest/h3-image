/* Host float helpers and seeded RNG (SPEC §3.13, §4.0, §5.1).
 * All host math is compiled with FP contraction OFF: every a*b+c is two
 * separately-rounded F32 ops unless an explicit fma() is written. */
#ifndef H3I_IO_HOST_H
#define H3I_IO_HOST_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* F32 → BF16 round-to-nearest-even (SPEC §5.1). u32 wraparound is intended;
 * no NaN special-casing (a NaN may become Inf). */
static inline uint16_t h3i_bf16_bits(float x)
{
    uint32_t b;
    memcpy(&b, &x, 4);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

/* BF16 (as u16) → F32: exact, just a shift. */
static inline float h3i_bf16_to_f32(uint16_t h)
{
    uint32_t b = (uint32_t)h << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

/* Round F32 to BF16 precision, returned as F32. */
static inline float h3i_bf16(float x)
{
    return h3i_bf16_to_f32(h3i_bf16_bits(x));
}

/* IEEE half RNE rounding, returned as F32 (SPEC §4.0 h3i_round_f16). */
float h3i_round_f16(float x);

/* PCG32 (XSH-RR) + Box–Muller with cached spare (SPEC §3.13). */
typedef struct {
    uint64_t state;
    uint64_t inc;
    int has_spare;
    float spare;
} h3i_rng;

void h3i_rng_seed(h3i_rng *r, uint64_t seed);
uint32_t h3i_rng_next_u32(h3i_rng *r);
float h3i_rng_normal(h3i_rng *r);
void h3i_rng_fill(h3i_rng *r, float *out, size_t n);

/* Euler velocity update (SPEC §3.12): x += (sigma - sigma_next) * v,
 * two separately-rounded F32 ops. */
int h3i_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next);

/* Resize interleaved RGB24 with vImage high-quality resampling (SPEC §4.4,
 * §4.5; Apple-only implementation, ports substitute an equivalent
 * Lanczos-class resampler). The caller owns *output; identity geometry
 * still returns an independent copy. */
int h3i_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output);

#endif /* H3I_IO_HOST_H */
