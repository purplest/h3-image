/* Vendored ggml dequantization reference — the A1 bit-exactness oracle.
 *
 * Extracted from ggml (MIT licence, Copyright (c) 2023-2024 The ggml authors),
 * commit d9fe7bff3d2bf4d077e25c654486d02c321734aa:
 *   src/ggml-common.h   (block structs, QK constants)
 *   src/ggml-quants.c   (get_scale_min_k4, dequantize_row_{q8_0,q2_K,q3_K,q4_K,q5_K,q6_K})
 * Function bodies are verbatim; GGML_RESTRICT → restrict, and the
 * GGML_FP16_TO_FP32 table lookup is replaced by an IEEE-exact bit shim
 * (identical values for every half pattern).
 *
 * Compiled with -ffp-contract=fast (+ -mfma on x86-64) so `dl*q - ml`
 * contracts to FMA exactly as ggml does on Apple Silicon. */
#ifndef H3I_TESTS_GGML_REF_H
#define H3I_TESTS_GGML_REF_H

#include <stdint.h>

#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    ggml_half d;
    ggml_half dmin;
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    ggml_half d;
} block_q3_K;

typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
} block_q4_K;

typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    ggml_half d;
} block_q6_K;

void dequantize_row_q8_0(const block_q8_0 *restrict x, float *restrict y, int64_t k);
void dequantize_row_q2_K(const block_q2_K *restrict x, float *restrict y, int64_t k);
void dequantize_row_q3_K(const block_q3_K *restrict x, float *restrict y, int64_t k);
void dequantize_row_q4_K(const block_q4_K *restrict x, float *restrict y, int64_t k);
void dequantize_row_q5_K(const block_q5_K *restrict x, float *restrict y, int64_t k);
void dequantize_row_q6_K(const block_q6_K *restrict x, float *restrict y, int64_t k);

#endif /* H3I_TESTS_GGML_REF_H */
