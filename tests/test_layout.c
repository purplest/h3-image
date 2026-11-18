/* A2 gate: canvas, schedule, patchify, layout, RoPE, tile plans, uint8,
 * RNG and rounding — bit-exact against independent Python vectors
 * (tests/gen_vectors.py → tests/vectors.h). No weights, no GPU. */
#include <math.h>
#include <string.h>

#include "app/error.h"
#include "io/host.h"
#include "model/layout.h"
#include "check.h"
#include "vectors.h"

static float f32_from(uint32_t b)
{
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static uint32_t bits_of(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

static void test_canvas(void)
{
    h3i_canvas c;
    char err[256];

    CHECK(h3i_canvas_resolve(1024, 1024, 1024, &c, err, sizeof err), "canvas 1024 min 1024");
    CHECK(c.canvas_w == 1024 && c.canvas_h == 1024, "canvas size %dx%d", c.canvas_w, c.canvas_h);
    CHECK(c.latent_w == 64 && c.latent_h == 64, "latent %dx%d", c.latent_w, c.latent_h);

    CHECK(h3i_canvas_resolve(500, 300, 1024, &c, err, sizeof err), "canvas 500x300");
    CHECK(c.canvas_w == 512 && c.canvas_h == 288, "500x300 → %dx%d", c.canvas_w, c.canvas_h);

    CHECK(h3i_canvas_resolve(1024, 1024, 4194304, &c, err, sizeof err), "canvas default area");
    CHECK(c.canvas_w == 2048 && c.canvas_h == 2048, "upscale → %dx%d", c.canvas_w, c.canvas_h);

    CHECK(h3i_canvas_resolve(48, 48, 1024, &c, err, sizeof err), "canvas tie");
    CHECK(c.canvas_w == 64 && c.canvas_h == 64, "1.5 ties-to-even → %dx%d", c.canvas_w, c.canvas_h);

    CHECK(h3i_canvas_resolve(16, 16, 1024, &c, err, sizeof err), "canvas min clamp");
    CHECK(c.canvas_w == 32 && c.canvas_h == 32, "0.5 → 0 → 32: %dx%d", c.canvas_w, c.canvas_h);

    CHECK(h3i_canvas_resolve(3000, 2000, 1024, &c, err, sizeof err), "canvas large request");
    CHECK(c.canvas_w == 3008 && c.canvas_h == 1984, "62.5 ties-to-even → %dx%d", c.canvas_w, c.canvas_h);

    CHECK(!h3i_canvas_resolve(16384, 16384, 1024, &c, err, sizeof err), "canvas too large rejected");
    CHECK(strcmp(err, "canvas 16384x16384 exceeds the supported area") == 0, "msg: %s", err);

    CHECK(!h3i_canvas_resolve(0, 100, 1024, &c, err, sizeof err), "zero width rejected");
    CHECK(strcmp(err, "width and height must be positive") == 0, "msg: %s", err);

    CHECK(!h3i_canvas_resolve(100, 100, 1023, &c, err, sizeof err), "min-area low rejected");
    CHECK(strcmp(err, "minimum generation area must be in [1024, 16777216]") == 0, "msg: %s", err);
    CHECK(!h3i_canvas_resolve(100, 100, 16777217, &c, err, sizeof err), "min-area high rejected");
}

static void test_schedule(void)
{
    h3i_schedule s;
    char err[256];
    CHECK(h3i_schedule_build(NULL, &s, err, sizeof err), "schedule build");
    for (int i = 0; i < 6; i++) {
        CHECK(bits_of(s.t[i]) == vec_schedule_t[i], "t[%d] = %08x want %08x",
              i, bits_of(s.t[i]), vec_schedule_t[i]);
        CHECK(bits_of(s.sigma[i]) == vec_schedule_sigma[i], "sigma[%d] = %08x want %08x",
              i, bits_of(s.sigma[i]), vec_schedule_sigma[i]);
    }
    CHECK(bits_of(s.sigma[6]) == 0, "sigma[6] = 0");
    CHECK(bits_of(s.t[6]) == bits_of(0.999f), "t[6] = 0.999f");

    double bad[7] = {1.0, 0.9, 0.9, 0.7, 0.5, 0.2, 0.0};
    CHECK(!h3i_schedule_build(bad, &s, err, sizeof err), "non-decreasing sigmas rejected");
    double bad2[7] = {1.0, 0.9, 0.8, 0.7, 0.5, 0.2, 0.1};
    CHECK(!h3i_schedule_build(bad2, &s, err, sizeof err), "last sigma != 0 rejected");
}

static void test_patchify(void)
{
    enum { LH = 4, LW = 6, N = (LH / 2) * (LW / 2) };
    float latent[24 * LH * LW], rows[N * 96], back[24 * LH * LW];
    for (int c = 0; c < 24; c++) {
        for (int y = 0; y < LH; y++) {
            for (int x = 0; x < LW; x++) {
                latent[(c * LH + y) * LW + x] = (float)(c * 1000 + y * 10 + x);
            }
        }
    }
    h3i_patchify(latent, LH, LW, rows);
    int ok = 1;
    for (int y = 0; y < LH; y += 2) {
        for (int x = 0; x < LW; x += 2) {
            size_t r = (size_t)(y / 2) * (LW / 2) + (size_t)(x / 2);
            for (int c = 0; c < 24 && ok; c++) {
                for (int dy = 0; dy < 2 && ok; dy++) {
                    for (int dx = 0; dx < 2 && ok; dx++) {
                        float want = (float)(c * 1000 + (y + dy) * 10 + (x + dx));
                        if (rows[r * 96 + (size_t)c * 4 + (size_t)dy * 2 + (size_t)dx] != want) {
                            ok = 0;
                        }
                    }
                }
            }
        }
    }
    CHECK(ok, "patchify row/value order");
    h3i_unpatchify(rows, LH, LW, back);
    CHECK(memcmp(latent, back, sizeof latent) == 0, "unpatchify is the exact inverse");
}

static void test_layout(void)
{
    uint8_t tags[3] = {1, 0, 1};
    int ref_lh[1] = {4}, ref_lw[1] = {4};
    h3i_layout l;
    char err[256];
    CHECK(h3i_layout_build(3, tags, 1, ref_lh, ref_lw, 4, 6, &l, err, sizeof err),
          "layout build: %s", err);
    CHECK(l.sequence == 13 && l.text_rows == 3 && l.reference_rows == 4 &&
          l.target_rows == 6 && l.target_start == 7,
          "layout counts: seq %zu ref %zu tgt %zu", l.sequence, l.reference_rows, l.target_rows);
    int pos_ok = 1;
    for (size_t i = 0; i < 39; i++) {
        if (l.positions[i] != vec_layout_pos[i]) {
            pos_ok = 0;
        }
    }
    CHECK(pos_ok, "layout positions bit-exact vs vectors");
    uint8_t want_tags[13] = {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(memcmp(l.tags, want_tags, 13) == 0, "layout tags");
    h3i_layout_free(&l);

    /* default text tags = 1 */
    CHECK(h3i_layout_build(2, NULL, 0, NULL, NULL, 2, 2, &l, err, sizeof err), "layout no refs");
    CHECK(l.sequence == 3 && l.tags[0] == 1 && l.tags[1] == 1 && l.tags[2] == 0,
          "default tags");
    h3i_layout_free(&l);
}

static void test_rope(void)
{
    double pos[9] = {1, 2, 3, 0.5, -2.25, 100, 31.999, 0, 0.001};
    float inv[16];
    for (int i = 0; i < 16; i++) {
        inv[i] = f32_from(vec_rope_inv[i]);
    }
    uint16_t cosb[144], sinb[144];
    h3i_rope_tables(pos, 3, inv, cosb, sinb);
    int ok = 1;
    for (int i = 0; i < 144; i++) {
        if (cosb[i] != (uint16_t)vec_rope_cos[i] || sinb[i] != (uint16_t)vec_rope_sin[i]) {
            ok = 0;
        }
    }
    CHECK(ok, "rope cos/sin BF16 tables bit-exact vs vectors");
}

static void test_tiles(void)
{
    h3i_tile_axis a;
    CHECK(h3i_tile_axis_plan(256, 256, 64, &a), "tile 256");
    CHECK(a.count == 1 && a.length == 256 && a.starts[0] == 0, "tile 256 single");
    CHECK(h3i_tile_axis_plan(200, 256, 64, &a), "tile 200");
    CHECK(a.count == 1 && a.length == 200, "tile 200 length = extent");

    CHECK(h3i_tile_axis_plan(512, 256, 64, &a), "tile 512");
    CHECK(a.count == 3 && a.overlaps[0] == 128 && a.overlaps[1] == 128 &&
          a.starts[0] == 0 && a.starts[1] == 128 && a.starts[2] == 256 && a.length == 256,
          "tile 512 plan: count %d ov %d/%d starts %d/%d/%d",
          a.count, a.overlaps[0], a.overlaps[1], a.starts[0], a.starts[1], a.starts[2]);

    CHECK(h3i_tile_axis_plan(1024, 256, 64, &a), "tile 1024");
    CHECK(a.count == 5 && a.starts[4] == 768 && a.overlaps[3] == 64, "tile 1024 plan");

    CHECK(h3i_tile_axis_plan(304, 256, 64, &a), "tile 304");
    /* count 2 → one overlap slot; all 9 remainder increments land on it */
    CHECK(a.count == 2 && a.overlaps[0] == 208 && a.starts[1] == 48,
          "tile 304 plan: ov %d start %d", a.overlaps[0], a.starts[1]);

    CHECK(h3i_tile_axis_plan(4096, 256, 64, &a), "tile 4096");
    CHECK(a.count == 21 && a.starts[20] == 3840, "tile 4096 plan: count %d", a.count);
}

static void test_u8(void)
{
    CHECK(h3i_unit_to_u8(0.5f) == 127, "0.5 → 127 (truncation)");
    CHECK(h3i_unit_to_u8(0.0f) == 0, "0 → 0");
    CHECK(h3i_unit_to_u8(1.0f) == 255, "1 → 255");
    CHECK(h3i_unit_to_u8(NAN) == 0, "NaN → 0");
    CHECK(h3i_unit_to_u8(-0.1f) == 0, "negative → 0");
    CHECK(h3i_unit_to_u8(1.5f) == 255, ">1 → 255");
    CHECK(h3i_unit_to_u8(0.999f) == 254, "0.999 → 254");
}

static void test_rng(void)
{
    h3i_rng r;
    h3i_rng_seed(&r, 42);
    for (int i = 0; i < 8; i++) {
        uint32_t v = h3i_rng_next_u32(&r);
        CHECK(v == vec_pcg_u32[i], "pcg u32[%d] = %08x want %08x", i, v, vec_pcg_u32[i]);
    }
    h3i_rng_seed(&r, 42);
    float n[6];
    h3i_rng_fill(&r, n, 6);
    for (int i = 0; i < 6; i++) {
        CHECK(bits_of(n[i]) == vec_normals[i], "normal[%d] = %08x want %08x",
              i, bits_of(n[i]), vec_normals[i]);
    }
    /* spare carries across fill calls */
    h3i_rng_seed(&r, 42);
    float a[3], b[3];
    h3i_rng_fill(&r, a, 3);
    h3i_rng_fill(&r, b, 3);
    CHECK(memcmp(a, n, 12) == 0 && memcmp(b, n + 3, 12) == 0, "spare carries across fills");
}

static void test_rounding(void)
{
    for (size_t i = 0; i < sizeof vec_f16_in / sizeof vec_f16_in[0]; i++) {
        float x = f32_from(vec_f16_in[i]);
        float y = h3i_round_f16(x);
        CHECK(bits_of(y) == vec_f16_out[i], "f16 round[%zu]: %08x → %08x want %08x",
              i, vec_f16_in[i], bits_of(y), vec_f16_out[i]);
    }
    CHECK(isnan(h3i_round_f16(NAN)), "f16 NaN passes");
    CHECK(isinf(h3i_round_f16(INFINITY)) && h3i_round_f16(INFINITY) > 0, "f16 inf passes");

    for (size_t i = 0; i < sizeof vec_bf16_in / sizeof vec_bf16_in[0]; i++) {
        float x = f32_from(vec_bf16_in[i]);
        uint16_t b = h3i_bf16_bits(x);
        CHECK(b == (uint16_t)vec_bf16_out[i], "bf16 round[%zu]: %08x → %04x want %04x",
              i, vec_bf16_in[i], b, vec_bf16_out[i]);
    }
    /* bf16 → f32 is exact */
    CHECK(bits_of(h3i_bf16_to_f32(0x3F80)) == bits_of(1.0f), "bf16 1.0");
    CHECK(h3i_bf16(1.0f) == 1.0f, "bf16 round of exact value");
}

int main(void)
{
    test_canvas();
    test_schedule();
    test_patchify();
    test_layout();
    test_rope();
    test_tiles();
    test_u8();
    test_rng();
    test_rounding();
    TEST_RESULT();
}
