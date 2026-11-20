/* test_vae — SPEC §6.5 (Phase 4 gate): VAE decoder mechanics + config +
 * PNG round trip. With weights: decode a synthetic latent through the
 * tiled 2D ViT (finite, exact dimensions, bit-repeatable), single-tile
 * edge case, PNG write/read round trip, and F16-vs-F32 weight PSNR > 40 dB
 * when both files are available (A10 part). The coherent-image and
 * round-trip-PSNR parts of A10 land with real latents (Phases 8/10).
 *
 * Weight paths: $H3I_VAE (safetensors file), $H3I_VAE_F32 (optional official
 * F32 VAE), $H3I_VAE_CONFIG (optional config.json); defaults under
 * ~/models/MiniMax-H3-Image-GGUF/vae/. Skips without weights/GPU. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app/error.h"
#include "builders.h"
#include "check.h"
#include "gpu_test.h"
#include "io/image.h"
#include "model/layout.h"
#include "model/vae.h"
#include "weights/gguf.h"
#include "weights/source.h"

static void test_config(void) {
    char err[512];
    float mean[24], std[24];
    CHECK(h3i_vae_config_load(NULL, mean, std, err, sizeof err), "defaults");
    CHECK(fabsf(mean[0] - 0.858090341091156f) < 1e-6f, "default mean[0]");
    CHECK(fabsf(std[23] - 2.6127843856811523f) < 1e-6f, "default std[23]");
    CHECK(h3i_vae_config_load("/nonexistent/vae/config.json", mean, std, err,
                              sizeof err),
          "absent config falls back to defaults");

    const char *path = "build/test_vae_config.json";
    FILE *f = fopen(path, "w");
    CHECK(f != NULL, "write temp config");
    if (f) {
        fputs("{\"latents_mean\": [", f);
        for (int i = 0; i < 24; i++) fprintf(f, "%s%d.5", i ? "," : "", i);
        fputs("], \"latents_std\": [", f);
        for (int i = 0; i < 24; i++) fprintf(f, "%s1.25", i ? "," : "");
        fputs("]}\n", f);
        fclose(f);
        CHECK(h3i_vae_config_load(path, mean, std, err, sizeof err),
              "override config: %s", err);
        CHECK(mean[7] == 7.5f && std[7] == 1.25f, "override values");
    }
    f = fopen(path, "w");
    if (f) {
        fputs("{\"latents_mean\": [1,2,3]}\n", f);
        fclose(f);
        CHECK(!h3i_vae_config_load(path, mean, std, err, sizeof err),
              "short array accepted");
    }
}

#ifdef __APPLE__

static void fill_latent(float *latent, int lh, int lw) {
    for (int c = 0; c < 24; c++) {
        for (int y = 0; y < lh; y++) {
            for (int x = 0; x < lw; x++) {
                latent[((size_t)c * lh + y) * lw + x] =
                    0.5f * sinf(0.7f * (float)c + 0.11f * (float)y +
                                0.13f * (float)x);
            }
        }
    }
}

static int all_finite(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

static double psnr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= (double)n;
    if (mse == 0.0) return INFINITY;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static const char *vae_path(void) {
    const char *env = getenv("H3I_VAE");
    if (env && *env) return env;
    static char path[4096];
    const char *home = getenv("HOME");
    snprintf(path, sizeof path,
             "%s/models/MiniMax-H3-Image-GGUF/vae/minimax_h3_image_vae_fp16.safetensors",
             home ? home : "");
    return path;
}

static const char *vae_config_path(void) {
    const char *env = getenv("H3I_VAE_CONFIG");
    if (env && *env) return env;
    static char path[4096];
    const char *home = getenv("HOME");
    snprintf(path, sizeof path, "%s/models/MiniMax-H3-Image-GGUF/vae/config.json",
             home ? home : "");
    return path;
}

static void test_decode(void) {
    char err[512];
    FILE *probe = fopen(vae_path(), "rb");
    if (!probe) {
        printf("skip: VAE weights (%s absent)\n", vae_path());
        return;
    }
    fclose(probe);
    h3i_gpu *gpu = test_gpu_create();
    if (!gpu) return;

    h3i_source source = {0};
    CHECK(h3i_source_add(&source, vae_path(), err, sizeof err),
          "open VAE: %s", err);
    h3i_vae_decoder *dec =
        h3i_vae_decoder_load(gpu, &source, vae_config_path(), err, sizeof err);
    CHECK(dec != NULL, "load VAE: %s", err);
    if (!dec) {
        h3i_source_free(&source);
        h3i_gpu_free(gpu);
        return;
    }

    /* 32x32 latent = 512 px: multiple tiles per axis, exercises stitching. */
    enum { LH = 32, LW = 32 };
    float *latent = malloc(24 * LH * LW * sizeof(float));
    float *rgb = NULL, *rgb2 = NULL;
    CHECK(latent != NULL, "latent alloc");
    fill_latent(latent, LH, LW);
    CHECK(h3i_vae_decoder_decode(dec, latent, LH, LW, &rgb, err, sizeof err),
          "decode 512: %s", err);
    if (rgb) {
        CHECK(all_finite(rgb, (size_t)LH * 16 * LW * 16 * 3), "decode finite");
        CHECK(h3i_vae_decoder_decode(dec, latent, LH, LW, &rgb2, err,
                                     sizeof err),
              "decode repeat: %s", err);
        if (rgb2) {
            CHECK(memcmp(rgb, rgb2, (size_t)LH * 16 * LW * 16 * 3 * 4) == 0,
                  "decode not bit-repeatable");
            free(rgb2);
        }
        /* PNG round trip through the image I/O module. */
        size_t pixels = (size_t)LH * 16 * LW * 16;
        uint8_t *u8 = malloc(pixels * 3);
        for (size_t i = 0; i < pixels * 3; i++) u8[i] = h3i_unit_to_u8(rgb[i]);
        CHECK(h3i_image_write_png("build/test_vae_512.png", u8, LW * 16,
                                  LH * 16, err, sizeof err),
              "write PNG: %s", err);
        h3i_image back = {0};
        CHECK(h3i_image_read("build/test_vae_512.png", &back, err, sizeof err),
              "read PNG: %s", err);
        CHECK(back.width == LW * 16 && back.height == LH * 16, "PNG dims");
        if (back.rgb) {
            CHECK(memcmp(back.rgb, u8, pixels * 3) == 0,
                  "PNG round trip not lossless");
            h3i_image_free(&back);
        }
        free(u8);
        free(rgb);
    }

    /* Single-tile edge case: 2x2 latent = 32 px. */
    float small[24 * 2 * 2];
    fill_latent(small, 2, 2);
    float *srgb = NULL;
    CHECK(h3i_vae_decoder_decode(dec, small, 2, 2, &srgb, err, sizeof err),
          "decode 32: %s", err);
    if (srgb) {
        CHECK(all_finite(srgb, 32 * 32 * 3), "small decode finite");
        free(srgb);
    }

    /* F16 vs F32 weights (A10 part): same latent, PSNR > 40 dB. */
    const char *f32_path = getenv("H3I_VAE_F32");
    if (f32_path && *f32_path) {
        h3i_source src32 = {0};
        CHECK(h3i_source_add(&src32, f32_path, err, sizeof err),
              "open F32 VAE: %s", err);
        h3i_vae_decoder *dec32 =
            h3i_vae_decoder_load(gpu, &src32, vae_config_path(), err,
                                 sizeof err);
        CHECK(dec32 != NULL, "load F32 VAE: %s", err);
        if (dec32) {
            /* Diagnostic: compare the weights themselves (CPU dequant path,
             * independent of the GPU decoder). A low PSNR with near-identical
             * weights would implicate the decoder; differing weights implicate
             * the files (dtype or checkpoint version). */
            /* Full weight audit: every tensor present in both files. A low
             * PSNR with uniformly ~2e-4 rel (F16 rounding) and no non-finite
             * values would implicate the decoder; a standout tensor or inf
             * (F16 overflow > 65504) implicates the files. */
            const char *dta = "?";
            const char *dtb = "?";
            double worst = 0.0;
            const char *worst_name = "(none)";
            size_t compared = 0, missing = 0, nfa = 0, nfb = 0;
            for (size_t i = 0; i < source.count; i++) {
                const h3i_tensor_rec *ra = &source.recs[i];
                const h3i_tensor_rec *rb = h3i_source_find(&src32, ra->name);
                if (i == 0) {
                    dta = h3i_type_name(ra->type);
                    dtb = rb ? h3i_type_name(rb->type) : "?";
                }
                if (!rb || !ra->supported || !rb->supported ||
                    ra->elements != rb->elements) {
                    missing++;
                    continue;
                }
                float *wa = h3i_source_read_f32(&source, ra->name, ra->elements,
                                                err, sizeof err);
                float *wb = h3i_source_read_f32(&src32, ra->name, rb->elements,
                                                err, sizeof err);
                if (wa && wb) {
                    double num = 0.0, den = 0.0;
                    for (uint64_t j = 0; j < ra->elements; j++) {
                        if (!isfinite(wa[j])) nfa++;
                        if (!isfinite(wb[j])) nfb++;
                        double d = (double)wa[j] - (double)wb[j];
                        num += d * d;
                        den += (double)wb[j] * (double)wb[j];
                    }
                    double rel = sqrt(num / (den > 0.0 ? den : 1.0));
                    if (rel > worst) {
                        worst = rel;
                        worst_name = ra->name;
                    }
                    compared++;
                }
                free(wa);
                free(wb);
            }
            printf("vae audit: %zu tensors, worst rel %.4g (%s), nonfinite "
                   "gguf %zu official %zu, unmatched %zu\n",
                   compared, worst, worst_name, nfa, nfb, missing);
            float *a = NULL, *b = NULL;
            CHECK(h3i_vae_decoder_decode(dec, latent, LH, LW, &a, err,
                                         sizeof err), "decode F16");
            CHECK(h3i_vae_decoder_decode(dec32, latent, LH, LW, &b, err,
                                         sizeof err), "decode F32");
            if (a && b) {
                size_t n = (size_t)LH * 16 * LW * 16 * 3;
                uint8_t *ua = malloc(n), *ub = malloc(n);
                for (size_t i = 0; i < n; i++) {
                    ua[i] = h3i_unit_to_u8(a[i]);
                    ub[i] = h3i_unit_to_u8(b[i]);
                }
                double psnr = psnr_u8(ua, ub, n);
                CHECK(psnr > 40.0,
                      "F16 vs F32 PSNR %.1f dB (want > 40); gguf %s vs official "
                      "%s; audit %zu tensors worst rel %.4g (%s) nonfinite %zu/%zu "
                      "unmatched %zu",
                      psnr, dta, dtb, compared, worst, worst_name, nfa, nfb,
                      missing);
                free(ua);
                free(ub);
            }
            free(a);
            free(b);
            h3i_vae_decoder_free(dec32);
        }
        h3i_source_free(&src32);
    } else {
        printf("skip: F16-vs-F32 VAE comparison (set H3I_VAE_F32)\n");
    }

    free(latent);
    h3i_vae_decoder_free(dec);
    h3i_source_free(&source);
    h3i_gpu_free(gpu);
}

#endif /* __APPLE__ */

int main(void) {
    test_mkdir("build");
    test_config();
#ifdef __APPLE__
    test_decode();
#endif
    TEST_RESULT();
}
