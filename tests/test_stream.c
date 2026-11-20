/* test_stream — SPEC §6.5 (Phase 2 gate): synthetic weights, streamed ==
 * resident bit-identical per layer (in-order, shuffled, repeated cyclic
 * passes), F32→BF16 conversion vs the CPU dequant reference, double slots
 * keep the previous layer, unsupported/missing names fail with the spec
 * strings. Requires Metal; skips without it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app/error.h"
#include "bf16_util.h"
#include "builders.h"
#include "check.h"
#include "gpu_test.h"
#include "weights/gguf.h"
#include "weights/source.h"
#include "weights/stream.h"

#ifdef __APPLE__

#define LAYERS 5
#define POS 3

static const size_t q8_elements[LAYERS] = {32, 64, 32, 96, 32};
static const size_t bf16_elements[LAYERS] = {64, 32, 64, 32, 64};
static const size_t f32_elements[LAYERS] = {32, 32, 64, 32, 32};

static size_t pos_elements(size_t layer, size_t pos) {
    switch (pos) {
    case 0: return q8_elements[layer];
    case 1: return bf16_elements[layer];
    default: return f32_elements[layer];
    }
}

static int pos_type(size_t pos) {
    switch (pos) {
    case 0: return H3I_TYPE_Q8_0;
    case 1: return H3I_TYPE_BF16;
    default: return H3I_TYPE_F32;
    }
}

/* Deterministic block payloads. */
static void fill_payload(size_t layer, size_t pos, uint8_t *data, size_t bytes) {
    if (pos == 0) {
        for (size_t b = 0; b < bytes / 34; b++) {
            uint8_t *block = data + b * 34;
            block[0] = 0x00; block[1] = 0x3C; /* f16 1.0 */
            for (int j = 0; j < 32; j++) {
                block[2 + j] = (uint8_t)(j * 7 + layer * 13 + b * 3 + 1);
            }
        }
    } else if (pos == 1) {
        for (size_t j = 0; j < bytes / 2; j++) {
            uint16_t v = (uint16_t)(0x3F00u + ((j + layer) & 0x3Fu));
            data[j * 2] = (uint8_t)(v & 0xFF);
            data[j * 2 + 1] = (uint8_t)(v >> 8);
        }
    } else {
        for (size_t j = 0; j < bytes / 4; j++) {
            float v = (float)(j + 1) * 0.25f + (float)layer * 0.5f;
            memcpy(data + j * 4, &v, 4);
        }
    }
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

int main(void) {
    h3i_gpu *gpu = test_gpu_create();
    if (!gpu) {
        printf("ok: 0 checks, 0 failures\n");
        return 0;
    }
    char error[512];
    const char *dir = "build/test_stream_tmp";
    const char *path = "build/test_stream_tmp/weights.gguf";
    test_mkdir("build");
    test_mkdir(dir);

    /* Build one synthetic GGUF: LAYERS x POS tensors, mixed types/sizes. */
    tb_tensor ts[LAYERS * POS];
    uint8_t *payloads[LAYERS * POS];
    char names_buf[LAYERS * POS][64];
    const char *names[LAYERS * POS];
    uint64_t offset = 0;
    for (size_t layer = 0; layer < LAYERS; layer++) {
        for (size_t pos = 0; pos < POS; pos++) {
            size_t idx = layer * POS + pos;
            size_t elements = pos_elements(layer, pos);
            h3i_type_info info = h3i_type_info_get(pos_type(pos));
            size_t bytes = elements / (size_t)info.block_values *
                           (size_t)info.block_bytes;
            payloads[idx] = malloc(bytes);
            CHECK(payloads[idx] != NULL, "payload alloc");
            fill_payload(layer, pos, payloads[idx], bytes);
            snprintf(names_buf[idx], sizeof(names_buf[idx]), "blocks.%zu.w%zu",
                     layer, pos);
            names[idx] = names_buf[idx];
            ts[idx] = (tb_tensor){
                .name = names_buf[idx], .ndim = 1, .ne = {elements},
                .type = (uint32_t)pos_type(pos), .offset = offset,
                .data = payloads[idx], .data_bytes = bytes,
            };
            offset = align_up(offset + bytes, 32);
        }
    }
    CHECK(build_gguf(path, 3, 32, ts, LAYERS * POS) >= 0, "build_gguf");

    h3i_source source = {0};
    CHECK(h3i_source_add(&source, path, error, sizeof error), "source: %s",
          error);

    /* CPU expected: dequantize each tensor and round to BF16. */
    uint16_t *expected[LAYERS * POS];
    for (size_t idx = 0; idx < LAYERS * POS; idx++) {
        const h3i_tensor_rec *t = h3i_source_find(&source, names[idx]);
        CHECK(t != NULL, "find %s", names[idx]);
        uint8_t *raw = malloc((size_t)t->bytes);
        float *f32 = malloc((size_t)t->elements * sizeof(float));
        expected[idx] = malloc((size_t)t->elements * sizeof(uint16_t));
        CHECK(raw && f32 && expected[idx], "expected alloc");
        CHECK(h3i_source_read_raw(t, raw, 0, 0, error, sizeof error),
              "read_raw: %s", error);
        h3i_dequant_row(t->type, raw, t->elements, f32);
        for (uint64_t j = 0; j < t->elements; j++) {
            expected[idx][j] = f_bf16(f32[j]);
        }
        free(raw);
        free(f32);
    }

    h3i_stream *resident = h3i_stream_create(gpu, &source, names, LAYERS, POS,
                                             1, 1, error, sizeof error);
    h3i_stream *streamed = h3i_stream_create(gpu, &source, names, LAYERS, POS,
                                             0, 1, error, sizeof error);
    CHECK(resident && streamed, "stream create: %s", error);
    CHECK(h3i_stream_resident(resident) == 1, "resident flag");
    CHECK(h3i_stream_resident(streamed) == 0, "streamed flag");
    CHECK(h3i_stream_layers(streamed) == LAYERS, "layers");

    /* One layer through both streams in a single submitted+waited buffer
     * (the ordering contract), then bit-compare both outputs and the CPU
     * reference. */
    h3i_gpu_tensor *out_r[POS], *out_s[POS];
    uint16_t *got = malloc(96 * sizeof(uint16_t));
    for (size_t pass = 0; pass < 4; pass++) {
        static const size_t orders[4][LAYERS] = {
            {0, 1, 2, 3, 4},        /* in order */
            {3, 1, 4, 0, 2},        /* shuffled */
            {0, 1, 2, 3, 4},        /* cyclic repeat */
            {0, 1, 2, 3, 4},
        };
        for (size_t step = 0; step < LAYERS; step++) {
            size_t layer = orders[pass][step];
            CHECK(h3i_gpu_begin(gpu), "begin");
            CHECK(h3i_stream_acquire(resident, layer, out_r, error,
                                     sizeof error),
                  "resident acquire %zu: %s", layer, error);
            CHECK(h3i_stream_acquire(streamed, layer, out_s, error,
                                     sizeof error),
                  "streamed acquire %zu: %s", layer, error);
            CHECK(h3i_gpu_submit(gpu), "submit: %s", h3i_gpu_error(gpu));
            for (size_t pos = 0; pos < POS; pos++) {
                size_t elements = pos_elements(layer, pos);
                size_t idx = layer * POS + pos;
                CHECK(h3i_gpu_tensor_read_bf16(out_r[pos], got, elements),
                      "read resident");
                CHECK(memcmp(got, expected[idx], elements * 2) == 0,
                      "pass %zu layer %zu pos %zu: resident != CPU reference",
                      pass, layer, pos);
                CHECK(h3i_gpu_tensor_read_bf16(out_s[pos], got, elements),
                      "read streamed");
                CHECK(memcmp(got, expected[idx], elements * 2) == 0,
                      "pass %zu layer %zu pos %zu: streamed != CPU reference",
                      pass, layer, pos);
            }
        }
    }
    CHECK(h3i_stream_wait_seconds(streamed) >= 0.0, "wait seconds");
    free(got);

    /* Double slots: layer 1 decodes into the odd set, layer 0's even slots
     * survive (the Neural Engine coexistence contract). */
    h3i_stream *doubled = h3i_stream_create(gpu, &source, names, LAYERS, POS,
                                            0, 0, error, sizeof error);
    CHECK(doubled != NULL, "doubled create: %s", error);
    CHECK(h3i_stream_double_slots(doubled, error, sizeof error),
          "double_slots: %s", error);
    h3i_gpu_tensor *out0[POS], *out1[POS];
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_stream_acquire(doubled, 0, out0, error, sizeof error) &&
              h3i_gpu_submit(gpu),
          "doubled layer 0: %s", error);
    CHECK(h3i_gpu_begin(gpu) &&
              h3i_stream_acquire(doubled, 1, out1, error, sizeof error) &&
              h3i_gpu_submit(gpu),
          "doubled layer 1: %s", error);
    uint16_t *kept = malloc(96 * sizeof(uint16_t));
    for (size_t pos = 0; pos < POS; pos++) {
        size_t n0 = pos_elements(0, pos), n1 = pos_elements(1, pos);
        CHECK(h3i_gpu_tensor_read_bf16(out0[pos], kept, n0), "read kept");
        CHECK(memcmp(kept, expected[pos], n0 * 2) == 0,
              "double slots: layer 0 pos %zu overwritten by layer 1", pos);
        CHECK(h3i_gpu_tensor_read_bf16(out1[pos], kept, n1), "read odd");
        CHECK(memcmp(kept, expected[POS + pos], n1 * 2) == 0,
              "double slots: layer 1 pos %zu wrong", pos);
    }
    free(kept);
    h3i_stream_free(doubled);

    /* Spec error strings: unsupported storage type and missing name. */
    const char *bad_path = "build/test_stream_tmp/bad.gguf";
    uint8_t bad_data[64] = {0};
    tb_tensor bad = {.name = "bad.w", .ndim = 1, .ne = {32}, .type = 31,
                     .offset = 0, .data = bad_data, .data_bytes = 64};
    CHECK(build_gguf(bad_path, 3, 32, &bad, 1) >= 0, "build bad gguf");
    h3i_source bad_source = {0};
    CHECK(h3i_source_add(&bad_source, bad_path, error, sizeof error),
          "bad source: %s", error);
    const char *bad_names[1] = {"bad.w"};
    h3i_stream *bad_stream = h3i_stream_create(gpu, &bad_source, bad_names, 1,
                                               1, 0, 0, error, sizeof error);
    CHECK(bad_stream == NULL, "unsupported type accepted");
    CHECK(strcmp(error, "weight bad.w has an unsupported storage type") == 0,
          "unsupported message: '%s'", error);
    const char *missing_names[1] = {"nope.w"};
    bad_stream = h3i_stream_create(gpu, &source, missing_names, 1, 1, 0, 0,
                                   error, sizeof error);
    CHECK(bad_stream == NULL, "missing name accepted");
    CHECK(strcmp(error, "weight nope.w is missing") == 0,
          "missing message: '%s'", error);
    h3i_source_free(&bad_source);

    h3i_stream_free(resident);
    h3i_stream_free(streamed);
    h3i_source_free(&source);
    for (size_t idx = 0; idx < LAYERS * POS; idx++) {
        free(payloads[idx]);
        free(expected[idx]);
    }
    h3i_gpu_free(gpu);
    TEST_RESULT();
}

#else /* !__APPLE__ */

int main(void) {
    printf("skip: stream tests need Metal\n");
    printf("ok: 0 checks, 0 failures\n");
    return 0;
}

#endif /* __APPLE__ */
