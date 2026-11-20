/* Phase 0b harness proof: run migrated host math on Linux without Metal.
 * Known-answer tests for the SPEC.md §3.13 RNG (PCG32 + Box-Muller, spare
 * carry across fill calls) and the §3.12 Euler update (separately rounded
 * F32, no FMA). Expected values are generated from the spec formulas by
 * gen_expected.py on this machine (same libm as the binary).
 * Not part of the macOS test suite; built only by `make linux-check`. */
#include "io/host.h"

#include <stdio.h>
#include <string.h>

#include "host_smoke_expected.h"

static int checks;
static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        checks++;                                                              \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int main(void) {
    /* PCG32 raw stream, two seeds. */
    h3i_rng rng;
    h3i_rng_seed(&rng, 1);
    for (size_t i = 0; i < sizeof(expected_u32_seed1) / sizeof(uint32_t); i++) {
        uint32_t got = h3i_rng_u32(&rng);
        CHECK(got == expected_u32_seed1[i], "seed1 u32[%zu]: got %u want %u", i,
              got, expected_u32_seed1[i]);
    }
    h3i_rng_seed(&rng, 42);
    for (size_t i = 0; i < sizeof(expected_u32_seed42) / sizeof(uint32_t); i++) {
        uint32_t got = h3i_rng_u32(&rng);
        CHECK(got == expected_u32_seed42[i], "seed42 u32[%zu]: got %u want %u",
              i, got, expected_u32_seed42[i]);
    }

    /* Box-Muller normals, bit-exact, spare consumed on every second draw. */
    h3i_rng_seed(&rng, 1);
    for (size_t i = 0; i < sizeof(expected_normal_seed1) / sizeof(uint32_t); i++) {
        float got = h3i_rng_normal(&rng);
        CHECK(float_bits(got) == expected_normal_seed1[i],
              "seed1 normal[%zu]: got %08x want %08x", i, float_bits(got),
              expected_normal_seed1[i]);
    }
    h3i_rng_seed(&rng, 42);
    for (size_t i = 0; i < sizeof(expected_normal_seed42) / sizeof(uint32_t); i++) {
        float got = h3i_rng_normal(&rng);
        CHECK(float_bits(got) == expected_normal_seed42[i],
              "seed42 normal[%zu]: got %08x want %08x", i, float_bits(got),
              expected_normal_seed42[i]);
    }

    /* Spare carries across fill calls: fill(3)+fill(2) == fill(5). */
    float split[5];
    h3i_rng_seed(&rng, 7);
    h3i_rng_fill_normal(&rng, split, 3);
    h3i_rng_fill_normal(&rng, split + 3, 2);
    for (size_t i = 0; i < 5; i++) {
        CHECK(float_bits(split[i]) == expected_fill5_seed7[i],
              "fill carry[%zu]: got %08x want %08x", i, float_bits(split[i]),
              expected_fill5_seed7[i]);
    }

    /* Euler update: x += (sigma - sigma_next) * v, two rounded ops. */
    float sample[EULER_COUNT];
    memcpy(sample, euler_sample, sizeof(sample));
    int ok = h3i_euler_velocity_step(sample, euler_velocity, EULER_COUNT,
                                     euler_sigma, euler_sigma_next);
    CHECK(ok == 1, "euler step returned %d", ok);
    for (size_t i = 0; i < EULER_COUNT; i++) {
        CHECK(float_bits(sample[i]) == float_bits(euler_expected[i]),
              "euler[%zu]: got %08x want %08x", i, float_bits(sample[i]),
              float_bits(euler_expected[i]));
    }
    /* Invalid schedules are rejected. */
    CHECK(h3i_euler_velocity_step(sample, euler_velocity, EULER_COUNT,
                                  euler_sigma_next, euler_sigma) == 0,
          "euler accepted sigma < sigma_next");

    printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "ok", checks,
           failures);
    return failures != 0;
}
