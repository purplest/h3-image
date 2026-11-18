/* Host float helpers and seeded RNG (SPEC §3.13, §4.0). */
#include "io/host.h"

#include <math.h>

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
