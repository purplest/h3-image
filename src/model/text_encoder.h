#ifndef H3I_TEXT_ENCODER_H
#define H3I_TEXT_ENCODER_H

#include "gpu/gpu.h"

#include <stddef.h>
#include <stdint.h>

#define H3I_TEXT_HIDDEN_SIZE 5120u

typedef struct {
    size_t tokens;
    size_t width;
    uint16_t *values;
    h3i_gpu_stats gpu_stats;
    /* Optional DiT modality tag per presentation row: 1 for language and 0
     * for the Qwen vision span including its boundary tokens. */
    uint8_t *tags;
} h3i_text_embedding;

typedef void (*h3i_text_progress)(int completed_layers, int total_layers,
                                 void *opaque);

typedef struct {
    size_t start;
    size_t tokens;
    const uint16_t *embeddings;
    const uint16_t *deepstack[3];
} h3i_text_vision_span;

/* Run the released first 50 Qwen3-VL language layers. The caller owns the
 * returned BF16 values and releases them with h3i_text_embedding_free(). */
int h3i_text_encode_bf16(const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3i_text_progress progress, void *progress_opaque,
                        h3i_text_embedding *output,
                        char *error, size_t error_size);

/* Run the same 50 decoder layers with Qwen3-VL presentation spans. The base
 * token embedding at every span is replaced by vision embeddings; deepstack
 * rows are added after language layers 0, 1, and 2. position_ids is axis-major
 * [3,tokens], and tags carries the DiT modality tag for every text row. */
int h3i_text_encode_multimodal_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3i_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        h3i_text_progress progress, void *progress_opaque,
                        h3i_text_embedding *output,
                        char *error, size_t error_size);

/* Prefix form used by parity tooling to localize a multimodal mismatch. */
int h3i_text_encode_multimodal_layers_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3i_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3i_text_progress progress, void *progress_opaque,
                        h3i_text_embedding *output,
                        char *error, size_t error_size);
void h3i_text_embedding_free(h3i_text_embedding *embedding);

#endif
