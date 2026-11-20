/* Host float helpers and seeded RNG (SPEC §3.13, §4.0). */
#include "io/host.h"

#include <math.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

/* SPEC §3.12/§4.0: separately-rounded host float math (no FMA contraction). */
#pragma STDC FP_CONTRACT OFF

float h3i_round_f16(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, 4);
    uint32_t sign = bits & 0x80000000u;
    uint32_t mag = bits & 0x7FFFFFFFu;

    if (mag >= 0x7F800000u) { /* inf / NaN pass through */
        return x;
    }
    if (mag >= 0x477FF000u) { /* rounds to ±inf (>= 65520) */
        uint32_t b = sign | 0x7F800000u;
        float f;
        memcpy(&f, &b, 4);
        return f;
    }
    if (mag < 0x33000000u) { /* < 2^-25 rounds to ±0 */
        float f;
        memcpy(&f, &sign, 4);
        return f;
    }
    if (mag < 0x38800000u) { /* half subnormal: nearbyint(|x|/2^-24)*2^-24 */
        float ax;
        memcpy(&ax, &mag, 4);
        float scaled = nearbyintf(ax * 0x1p24f) * 0x1p-24f;
        return sign ? -scaled : scaled;
    }
    mag += 0x0FFFu + ((mag >> 13) & 1u);
    mag &= 0xFFFFE000u;
    uint32_t b = sign | mag;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

uint32_t h3i_rng_next_u32(h3i_rng *r)
{
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xs = (uint32_t)(((old >> 18) ^ old) >> 27);
    uint32_t rot = (uint32_t)(old >> 59);
    return (xs >> rot) | (xs << ((-rot) & 31));
}

void h3i_rng_seed(h3i_rng *r, uint64_t seed)
{
    r->state = 0;
    r->inc = (seed << 1) | 1;
    r->has_spare = 0;
    r->spare = 0.0f;
    (void)h3i_rng_next_u32(r);
    r->state += seed ^ 0x9e3779b97f4a7c15ULL;
    (void)h3i_rng_next_u32(r);
}

float h3i_rng_normal(h3i_rng *r)
{
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    double u1 = ((double)h3i_rng_next_u32(r) + 1.0) / 4294967297.0;
    double u2 = ((double)h3i_rng_next_u32(r) + 0.5) / 4294967296.0;
    double rad = sqrt(-2.0 * log(u1));
    double a = (2.0 * 3.14159265358979323846) * u2;
    r->spare = (float)(rad * sin(a));
    r->has_spare = 1;
    return (float)(rad * cos(a));
}

void h3i_rng_fill(h3i_rng *r, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[i] = h3i_rng_normal(r);
    }
}

int h3i_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next) {
    if (!sample || !velocity || !isfinite(sigma) || !isfinite(sigma_next) ||
        !(sigma > sigma_next) || sigma_next < 0.0f) return 0;
    float delta = sigma - sigma_next;
    for (size_t index = 0; index < count; index++)
        sample[index] += delta * velocity[index];
    return 1;
}

#ifdef __APPLE__
int h3i_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output) {
    if (output) *output = NULL;
    if (!input || !output || frames < 1 || input_width < 1 ||
        input_height < 1 || output_width < 1 || output_height < 1)
        return 0;
    size_t input_area = (size_t)input_width * (size_t)input_height;
    size_t output_area = (size_t)output_width * (size_t)output_height;
    if (input_area > SIZE_MAX / 3 || output_area > SIZE_MAX / 3 ||
        (size_t)frames > SIZE_MAX / (output_area * 3)) return 0;
    size_t output_bytes = (size_t)frames * output_area * 3;
    uint8_t *pixels = malloc(output_bytes);
    if (!pixels) return 0;
    if (input_width == output_width && input_height == output_height) {
        if ((size_t)frames > SIZE_MAX / (input_area * 3)) {
            free(pixels);
            return 0;
        }
        memcpy(pixels, input, (size_t)frames * input_area * 3);
        *output = pixels;
        return 1;
    }
    if (input_area > SIZE_MAX / 4 || output_area > SIZE_MAX / 4) {
        free(pixels);
        return 0;
    }
    uint8_t *source_argb = malloc(input_area * 4);
    uint8_t *output_argb = malloc(output_area * 4);
    if (!source_argb || !output_argb) {
        free(source_argb); free(output_argb); free(pixels);
        return 0;
    }
    size_t input_frame_bytes = input_area * 3;
    size_t output_frame_bytes = output_area * 3;
    vImage_Buffer source_buffer = {
        source_argb, (vImagePixelCount)input_height,
        (vImagePixelCount)input_width, (size_t)input_width * 4
    };
    vImage_Buffer output_buffer = {
        output_argb, (vImagePixelCount)output_height,
        (vImagePixelCount)output_width, (size_t)output_width * 4
    };
    for (int frame = 0; frame < frames; frame++) {
        const uint8_t *source_frame = input + (size_t)frame * input_frame_bytes;
        uint8_t *output_frame = pixels + (size_t)frame * output_frame_bytes;
        for (size_t pixel = 0; pixel < input_area; pixel++) {
            source_argb[4 * pixel] = 255;
            source_argb[4 * pixel + 1] = source_frame[3 * pixel];
            source_argb[4 * pixel + 2] = source_frame[3 * pixel + 1];
            source_argb[4 * pixel + 3] = source_frame[3 * pixel + 2];
        }
        if (vImageScale_ARGB8888(
                &source_buffer, &output_buffer, NULL,
                kvImageHighQualityResampling | kvImageEdgeExtend) !=
            kvImageNoError) {
            free(source_argb); free(output_argb); free(pixels);
            return 0;
        }
        for (size_t pixel = 0; pixel < output_area; pixel++) {
            output_frame[3 * pixel] = output_argb[4 * pixel + 1];
            output_frame[3 * pixel + 1] = output_argb[4 * pixel + 2];
            output_frame[3 * pixel + 2] = output_argb[4 * pixel + 3];
        }
    }
    free(source_argb); free(output_argb);
    *output = pixels;
    return 1;
}
#endif /* __APPLE__ */
