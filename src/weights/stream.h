/* Layer streamer (SPEC §2.7): per-layer weights, resident or read from SSD
 * into two staging sets with one read-ahead thread, then decoded to BF16
 * slots. Streamed and resident runs must be bit-identical (SPEC §1.4.7). */
#ifndef H3I_WEIGHTS_STREAM_H
#define H3I_WEIGHTS_STREAM_H

#include "gpu/gpu.h"
#include "weights/source.h"

#include <stddef.h>

typedef struct h3i_stream h3i_stream;

/* names: layers*per_layer tensor names, layer-major. Every tensor must be
 * present and supported. The source must outlive the stream. resident: keep
 * every weight in GPU memory (BF16 tensors zero-copy); cyclic: read-ahead
 * wraps last→0 (the wrap is skipped when it would target the in-use set). */
h3i_stream *h3i_stream_create(h3i_gpu *gpu, const h3i_source *source,
                              const char *const *names, size_t layers,
                              size_t per_layer, int resident, int cyclic,
                              char *error, size_t error_size);
void h3i_stream_free(h3i_stream *s);

/* Add a second BF16 slot set used for odd layers (Neural Engine path), so
 * two consecutive layers' decoded weights coexist. Call before acquire. */
int h3i_stream_double_slots(h3i_stream *s, char *error, size_t error_size);

/* Decode one layer into out[per_layer], encoding GPU work into the caller's
 * OPEN command buffer (h3i_gpu_begin).
 *
 * ORDERING CONTRACT: before acquire(L+1) the caller must have COMPLETED
 * (submitted and waited) the GPU work that consumed layer L's decode slots
 * and staging set: the prefetch for L+2 overwrites set L&1 and the decode of
 * L+1 overwrites the shared slots. Streamed users submit+wait per layer;
 * resident users may keep encoding (h3i_gpu_continue).
 *
 * out[i] aliases an internal slot (or a resident BF16 buffer); it stays
 * valid until an acquire that reuses that slot. */
int h3i_stream_acquire(h3i_stream *s, size_t layer, h3i_gpu_tensor **out,
                       char *error, size_t error_size);

/* Join any pending read-ahead (call before freeing other GPU memory). */
void h3i_stream_quiesce(h3i_stream *s);

/* SSD read time not hidden by GPU work (SPEC §6.1.7 profile lines). */
double h3i_stream_wait_seconds(const h3i_stream *s);
int h3i_stream_resident(const h3i_stream *s);
size_t h3i_stream_layers(const h3i_stream *s);

#endif /* H3I_WEIGHTS_STREAM_H */
