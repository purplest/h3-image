/* Neural Engine handoff kernels, bit-exact vs CPU (SPEC §5.4, §5.9):
 * bf16→f16, f16→bf16 round trip, overflow flag bits, transpose. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"

#ifdef __APPLE__

#include "bf16_util.h"
#include "gpu_test.h"
#include "io/host.h"
#include "weights/gguf.h"

static double rnd(void)
{
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

static uint32_t bits_of_f(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

static void test_bf16_to_f16(h3i_gpu *g)
{
    enum { N = 4096 };
    uint16_t in[N], out[N];
    for (int i = 0; i < N; i++) {
        /* values spanning half's range, incl near-overflow */
        double v = rnd() * 70000.0;
        in[i] = f_bf16((float)v);
    }
    float scale = 0.25f; /* exact power of two */
    h3i_gpu_tensor *ti = h3i_gpu_tensor_from_bf16(g, in, N);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_i8(g, N * 2);
    CHECK(h3i_gpu_begin(g), "begin b2h");
    CHECK(h3i_gpu_bf16_to_f16(g, to, ti, N, scale), "bf16_to_f16 op");
    CHECK(h3i_gpu_submit(g), "submit b2h: %s", h3i_gpu_error(g));
    memcpy(out, h3i_gpu_tensor_contents(to), N * 2);
    int ok = 1, overflow_ok = 1;
    for (int i = 0; i < N; i++) {
        float want = h3i_round_f16(bf16_f(in[i]) * scale);
        float got = h3i_half_to_f32(out[i]);
        if (isnan(want) || isnan(got)) {
            if (!(isnan(want) && isnan(got))) ok = 0;
        } else if (bits_of_f(got) != bits_of_f(want)) {
            ok = 0;
        }
        if (bf16_f(in[i]) * scale > 65520.0f && out[i] != 0x7C00) {
            overflow_ok = 0;
        }
    }
    CHECK(ok, "bf16_to_f16 bit-exact vs CPU RNE");
    CHECK(overflow_ok, "overflow → 0x7C00");
    h3i_gpu_tensor_free(ti);
    h3i_gpu_tensor_free(to);
}

static void test_checked(h3i_gpu *g)
{
    enum { N = 1024 };
    uint16_t in[N], out[N];
    for (int i = 0; i < N; i++) {
        in[i] = (uint16_t)(rand() & 0x7BFF); /* finite half patterns */
    }
    /* directed non-finite inputs */
    in[0] = 0x7C00;  /* +inf */
    in[1] = 0xFC00;  /* -inf */
    in[2] = 0x7E00;  /* NaN */
    float scale = 4.0f;
    h3i_gpu_tensor *ti = h3i_gpu_tensor_new_i8(g, N * 2);
    memcpy(h3i_gpu_tensor_contents(ti), in, N * 2);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_bf16(g, N);
    h3i_gpu_tensor *tf = h3i_gpu_tensor_new_u32(g, 1);

    /* inf → flag bit 2 */
    CHECK(h3i_gpu_begin(g), "begin chk");
    CHECK(h3i_gpu_f16_to_bf16_checked(g, to, ti, N, scale, tf, 2), "checked op");
    CHECK(h3i_gpu_submit(g), "submit chk: %s", h3i_gpu_error(g));
    uint32_t flag = *(uint32_t *)h3i_gpu_tensor_contents(tf);
    CHECK((flag & 2u) != 0, "inf sets flag bit 2 (flag=%u)", flag);
    memcpy(out, h3i_gpu_tensor_contents(to), N * 2);
    int ok = 1;
    for (int i = 3; i < N; i++) {
        float v = h3i_half_to_f32(in[i]);
        if (out[i] != f_bf16(v * scale)) {
            ok = 0;
        }
    }
    CHECK(ok, "f16_to_bf16_checked finite values exact");

    /* NaN → flag bit 4 */
    for (int i = 0; i < N; i++) {
        in[i] = 0x7E00;
    }
    memcpy(h3i_gpu_tensor_contents(ti), in, N * 2);
    *(uint32_t *)h3i_gpu_tensor_contents(tf) = 0;
    CHECK(h3i_gpu_begin(g), "begin chk2");
    CHECK(h3i_gpu_f16_to_bf16_checked(g, to, ti, N, scale, tf, 4), "checked op 2");
    CHECK(h3i_gpu_submit(g), "submit chk2");
    flag = *(uint32_t *)h3i_gpu_tensor_contents(tf);
    CHECK((flag & 4u) != 0, "NaN sets flag bit 4 (flag=%u)", flag);

    /* round trip: bf16 → f16 → bf16 exact for half-normal-range values */
    for (int i = 0; i < N; i++) {
        double mag = pow(2.0, -14.0 + (rnd() * 0.5 + 0.5) * 29.0); /* [2^-14, 2^15) */
        in[i] = f_bf16((float)(rnd() > 0 ? mag : -mag));
    }
    h3i_gpu_tensor *tb = h3i_gpu_tensor_from_bf16(g, in, N);
    h3i_gpu_tensor *th = h3i_gpu_tensor_new_i8(g, N * 2);
    CHECK(h3i_gpu_begin(g), "begin rt");
    CHECK(h3i_gpu_bf16_to_f16(g, th, tb, N, 1.0f), "rt b2h");
    CHECK(h3i_gpu_f16_to_bf16_checked(g, to, th, N, 1.0f, tf, 1), "rt h2b");
    CHECK(h3i_gpu_submit(g), "submit rt: %s", h3i_gpu_error(g));
    memcpy(out, h3i_gpu_tensor_contents(to), N * 2);
    CHECK(memcmp(in, out, N * 2) == 0, "bf16→f16→bf16 round trip exact");

    h3i_gpu_tensor_free(ti); h3i_gpu_tensor_free(to); h3i_gpu_tensor_free(tf);
    h3i_gpu_tensor_free(tb); h3i_gpu_tensor_free(th);
}

static void test_transpose(h3i_gpu *g)
{
    enum { ROWS = 100, COLS = 70, ROFF = 5 };
    static uint16_t in[(ROWS + ROFF) * COLS], out[ROWS * COLS];
    for (size_t i = 0; i < (ROWS + ROFF) * COLS; i++) {
        in[i] = f_bf16((float)rnd());
    }
    float scale = 0.125f;
    h3i_gpu_tensor *ti = h3i_gpu_tensor_from_bf16(g, in, (ROWS + ROFF) * COLS);
    h3i_gpu_tensor *to = h3i_gpu_tensor_new_i8(g, ROWS * COLS * 2);
    CHECK(h3i_gpu_begin(g), "begin tr");
    CHECK(h3i_gpu_transpose_bf16_f16(g, to, ti, ROFF, ROWS, COLS, scale), "transpose op");
    CHECK(h3i_gpu_submit(g), "submit tr: %s", h3i_gpu_error(g));
    memcpy(out, h3i_gpu_tensor_contents(to), ROWS * COLS * 2);
    int ok = 1;
    for (int r = 0; r < ROWS && ok; r++) {
        for (int c = 0; c < COLS; c++) {
            float want = h3i_round_f16(bf16_f(in[(ROFF + r) * COLS + c]) * scale);
            float got = h3i_half_to_f32(out[(size_t)c * ROWS + r]);
            if (bits_of_f(got) != bits_of_f(want)) {
                ok = 0;
                break;
            }
        }
    }
    CHECK(ok, "transpose_bf16_f16 bit-exact");
    h3i_gpu_tensor_free(ti);
    h3i_gpu_tensor_free(to);
}

int main(void)
{
    srand(23);
    h3i_gpu *g = test_gpu_create();
    if (g == NULL) {
        printf("ok: 0 checks, 0 failures\n");
        return 0;
    }
    test_bf16_to_f16(g);
    test_checked(g);
    test_transpose(g);
    h3i_gpu_free(g);
    TEST_RESULT();
}

#else

int main(void)
{
    printf("skip: handoff tests require macOS/Metal\n");
    printf("ok: 0 checks, 0 failures\n");
    return 0;
}

#endif
