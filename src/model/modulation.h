/* AdaLN modulation tables + row maps + PDD output heads
 * (SPEC §3.7, §3.11). Two weight forms, detected by tensor presence:
 * official time-MLP (GPU projections) and pruned adaln_t_table (host
 * interpolation, no SiLU). Tables are BF16 [time_rows][outputs]. */
#ifndef H3I_MODEL_MODULATION_H
#define H3I_MODEL_MODULATION_H

#include "gpu/gpu.h"
#include "model/layout.h"
#include "weights/source.h"

#include <stddef.h>
#include <stdint.h>

#ifndef H3I_DIT_HIDDEN
#define H3I_DIT_HIDDEN 5376u
#endif
#ifndef H3I_DIT_TIME_DIM
#define H3I_DIT_TIME_DIM 2688u
#endif
#ifndef H3I_DIT_BLOCKS
#define H3I_DIT_BLOCKS 50u
#endif
#define H3I_MOD_BLOCK_OUTPUT (3u * 6u * H3I_DIT_HIDDEN)  /* 96768 */
#define H3I_MOD_FINAL_OUTPUT (2u * H3I_DIT_HIDDEN)       /* 10752 */

typedef struct h3i_modulation h3i_modulation;

/* schedule: the 6-step PDD schedule (t[6] = 0.999f is the reference row).
 * has_references builds 7 time rows, else 6. Detects pruned vs official;
 * error "DiT weights have neither adaln_t_table nor time_embedder" when
 * neither is present. */
h3i_modulation *h3i_modulation_build(h3i_gpu *gpu, const h3i_source *source,
                                     const h3i_schedule *schedule,
                                     int has_references,
                                     char *error, size_t error_size);
void h3i_modulation_free(h3i_modulation *m);
int h3i_modulation_is_pruned(const h3i_modulation *m);
uint32_t h3i_modulation_time_rows(const h3i_modulation *m);
/* [time_rows][96768] BF16, laid out [modality 3][slot 6][5376]. */
const h3i_gpu_tensor *h3i_modulation_block(const h3i_modulation *m, int block);
/* [time_rows][10752] BF16, laid out [slot 2][5376]. */
const h3i_gpu_tensor *h3i_modulation_final(const h3i_modulation *m);

/* Building blocks of the official path, exposed for composition/tests:
 * act = bf16(silu(time-MLP(features(times)))) [rows][2688] BF16, and one
 * AdaLN projection act·Wᵀ+b → [rows][outputs] BF16 (caller frees). */
h3i_gpu_tensor *h3i_time_act(h3i_gpu *gpu, const h3i_source *source,
                             const float *times, uint32_t rows,
                             char *error, size_t error_size);
int h3i_adaln_project(h3i_gpu *gpu, const h3i_source *source,
                      const h3i_gpu_tensor *act, uint32_t rows,
                      const char *weight_name, const char *bias_name,
                      uint32_t outputs, h3i_gpu_tensor **table,
                      char *error, size_t error_size);

/* Row maps (SPEC §3.7): row_maps[s][row] = time_row*3 + tag[row], where
 * time_row = 6 for reference rows else s; final_maps[s][r] = s (target rows
 * only). row_maps: [steps][sequence]; final_maps: [steps][target_rows]. */
int h3i_dit_prepare_maps(const h3i_layout *layout, int steps,
                         uint32_t *row_maps, uint32_t *final_maps,
                         char *error, size_t error_size);

/* PDD output heads (SPEC §3.11): 6 absolute heads [6][96][5376] F32 plus
 * bias [6][96]. Official pdd_fused_heads stores them absolute; GGUF
 * final_layer.video_out stores head 0 as base and heads 1..5 as offsets,
 * summed once in F32 here. */
typedef struct {
    float *weight;
    float *bias;
} h3i_heads;

int h3i_heads_load(const h3i_source *source, h3i_heads *out,
                   char *error, size_t error_size);
void h3i_heads_free(h3i_heads *heads);

#endif /* H3I_MODEL_MODULATION_H */
