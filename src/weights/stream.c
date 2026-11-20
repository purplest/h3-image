/* Layer streamer implementation (SPEC §2.7). Mac-only: decodes through the
 * Metal convert kernels; excluded from the Linux portable core. */
#include "weights/stream.h"

#include "app/error.h"
#include "weights/gguf.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct h3i_stream {
    h3i_gpu *gpu;
    size_t layers;
    size_t per_layer;
    int resident;
    int cyclic;
    int no_prefetch;

    const h3i_tensor_rec **recs;   /* layers*per_layer, layer-major */
    size_t *max_elements;          /* per position */
    size_t *max_bytes;             /* per position */
    h3i_gpu_tensor **slots;        /* per position BF16 decode slots (or NULL) */
    h3i_gpu_tensor **odd_slots;    /* optional second set (double_slots) */
    h3i_gpu_tensor **staging[2];   /* streaming: raw byte sets */
    h3i_gpu_tensor **resident_bufs; /* resident: layers*per_layer BF16 or I8 */
    int loaded[2];                 /* streaming: layer held in each set, -1 none */

    /* Single pending read-ahead job. */
    pthread_t thread;
    int pending;
    size_t job_layer;
    int job_set;
    int job_result;
    double job_seconds;
    char job_error[512];

    double wait_seconds;
};

static double stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int read_layer(h3i_stream *s, size_t layer, int set,
                      char *error, size_t error_size) {
    for (size_t i = 0; i < s->per_layer; i++) {
        const h3i_tensor_rec *t = s->recs[layer * s->per_layer + i];
        if (!h3i_source_read_raw(t, h3i_gpu_tensor_contents(s->staging[set][i]),
                                 0, 1, error, error_size)) {
            return 0;
        }
    }
    return 1;
}

static void *read_thread(void *opaque) {
    h3i_stream *s = opaque;
    double start = stream_now();
    s->job_result = read_layer(s, s->job_layer, s->job_set,
                               s->job_error, sizeof(s->job_error));
    s->job_seconds = stream_now() - start;
    return NULL;
}

/* Join the pending read-ahead; account its time and publish its set. */
static int join_pending(h3i_stream *s, char *error, size_t error_size) {
    if (!s->pending) return 1;
    pthread_join(s->thread, NULL);
    s->pending = 0;
    s->wait_seconds += s->job_seconds;
    if (!s->job_result) {
        h3i_fail(error, error_size, "%s", s->job_error);
        return 0;
    }
    s->loaded[s->job_set] = (int)s->job_layer;
    return 1;
}

void h3i_stream_quiesce(h3i_stream *s) {
    if (s) join_pending(s, NULL, 0);
}

h3i_stream *h3i_stream_create(h3i_gpu *gpu, const h3i_source *source,
                              const char *const *names, size_t layers,
                              size_t per_layer, int resident, int cyclic,
                              char *error, size_t error_size) {
    if (!gpu || !source || !names || !layers || !per_layer) {
        h3i_fail(error, error_size, "invalid stream request");
        return NULL;
    }
    h3i_stream *s = calloc(1, sizeof(*s));
    if (!s) {
        h3i_fail(error, error_size, "out of memory creating the weight stream");
        return NULL;
    }
    s->gpu = gpu;
    s->layers = layers;
    s->per_layer = per_layer;
    s->resident = resident;
    s->cyclic = cyclic;
    s->no_prefetch = getenv("H3I_NO_PREFETCH") != NULL;
    s->loaded[0] = s->loaded[1] = -1;
    s->recs = calloc(layers * per_layer, sizeof(*s->recs));
    s->max_elements = calloc(per_layer, sizeof(*s->max_elements));
    s->max_bytes = calloc(per_layer, sizeof(*s->max_bytes));
    s->slots = calloc(per_layer, sizeof(*s->slots));
    s->staging[0] = calloc(per_layer, sizeof(*s->staging[0]));
    s->staging[1] = calloc(per_layer, sizeof(*s->staging[1]));
    if (resident) {
        s->resident_bufs = calloc(layers * per_layer,
                                  sizeof(*s->resident_bufs));
    }
    if (!s->recs || !s->max_elements || !s->max_bytes || !s->slots ||
        !s->staging[0] || !s->staging[1] ||
        (resident && !s->resident_bufs)) {
        h3i_fail(error, error_size, "out of memory creating the weight stream");
        goto fail;
    }
    for (size_t layer = 0; layer < layers; layer++) {
        for (size_t i = 0; i < per_layer; i++) {
            const char *name = names[layer * per_layer + i];
            const h3i_tensor_rec *t = h3i_source_find(source, name);
            if (!t) {
                h3i_fail(error, error_size, "weight %s is missing", name);
                goto fail;
            }
            if (!t->supported) {
                h3i_fail(error, error_size,
                         "weight %s has an unsupported storage type", name);
                goto fail;
            }
            if (t->elements > SIZE_MAX || t->bytes > SIZE_MAX) {
                h3i_fail(error, error_size, "weight %s is too large", name);
                goto fail;
            }
            s->recs[layer * per_layer + i] = t;
            if ((size_t)t->elements > s->max_elements[i]) {
                s->max_elements[i] = (size_t)t->elements;
            }
            if (t->bytes > s->max_bytes[i]) s->max_bytes[i] = t->bytes;
        }
    }
    /* BF16 decode slots: always when streaming; when resident, only for
     * positions where some layer's tensor is not BF16. */
    for (size_t i = 0; i < per_layer; i++) {
        int need_slot = !resident;
        if (resident) {
            for (size_t layer = 0; layer < layers; layer++) {
                if (s->recs[layer * per_layer + i]->type != H3I_TYPE_BF16) {
                    need_slot = 1;
                    break;
                }
            }
        }
        if (need_slot) {
            s->slots[i] = h3i_gpu_tensor_new_bf16(gpu, s->max_elements[i]);
            if (!s->slots[i]) {
                h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
                goto fail;
            }
        }
        if (!resident) {
            for (int set = 0; set < 2; set++) {
                s->staging[set][i] = h3i_gpu_tensor_new_i8(gpu, s->max_bytes[i]);
                if (!s->staging[set][i]) {
                    h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
                    goto fail;
                }
            }
        }
    }
    if (resident) {
        for (size_t idx = 0; idx < layers * per_layer; idx++) {
            const h3i_tensor_rec *t = s->recs[idx];
            h3i_gpu_tensor *buffer = t->type == H3I_TYPE_BF16
                ? h3i_gpu_tensor_new_bf16(gpu, (size_t)t->elements)
                : h3i_gpu_tensor_new_i8(gpu, (size_t)t->bytes);
            if (!buffer) {
                h3i_fail(error, error_size, "%s", h3i_gpu_error(gpu));
                goto fail;
            }
            s->resident_bufs[idx] = buffer;
            if (!h3i_source_read_raw(t, h3i_gpu_tensor_contents(buffer), 0, 1,
                                     error, error_size)) {
                goto fail;
            }
        }
    }
    return s;

fail:
    h3i_stream_free(s);
    return NULL;
}

int h3i_stream_double_slots(h3i_stream *s, char *error, size_t error_size) {
    if (!s || s->odd_slots) {
        h3i_fail(error, error_size, "invalid double-slot request");
        return 0;
    }
    s->odd_slots = calloc(s->per_layer, sizeof(*s->odd_slots));
    if (!s->odd_slots) {
        h3i_fail(error, error_size, "out of memory adding stream slots");
        return 0;
    }
    for (size_t i = 0; i < s->per_layer; i++) {
        if (!s->slots[i]) continue; /* resident BF16 position: zero-copy */
        s->odd_slots[i] = h3i_gpu_tensor_new_bf16(s->gpu, s->max_elements[i]);
        if (!s->odd_slots[i]) {
            h3i_fail(error, error_size, "%s", h3i_gpu_error(s->gpu));
            return 0;
        }
    }
    return 1;
}

int h3i_stream_acquire(h3i_stream *s, size_t layer, h3i_gpu_tensor **out,
                       char *error, size_t error_size) {
    if (!s || !out || layer >= s->layers) {
        h3i_fail(error, error_size, "invalid stream acquire");
        return 0;
    }
    int set = (int)(layer & 1);
    if (!s->resident) {
        if (!join_pending(s, error, error_size)) return 0;
        if (s->loaded[set] != (int)layer) {
            double start = stream_now();
            int ok = read_layer(s, layer, set, error, error_size);
            s->wait_seconds += stream_now() - start;
            if (!ok) return 0;
            s->loaded[set] = (int)layer;
        }
        size_t next = layer + 1 < s->layers ? layer + 1
                      : (s->cyclic ? 0 : SIZE_MAX);
        if (next != SIZE_MAX && (int)(next & 1) != set && !s->no_prefetch) {
            s->loaded[next & 1] = -1;
            s->job_layer = next;
            s->job_set = (int)(next & 1);
            s->job_result = 0;
            s->job_error[0] = '\0';
            s->pending = 1;
            if (pthread_create(&s->thread, NULL, read_thread, s) != 0) {
                /* Thread creation failure: read synchronously. */
                s->pending = 0;
                double start = stream_now();
                int ok = read_layer(s, next, s->job_set, error, error_size);
                s->wait_seconds += stream_now() - start;
                if (!ok) return 0;
                s->loaded[s->job_set] = (int)next;
            }
        }
    }
    for (size_t i = 0; i < s->per_layer; i++) {
        const h3i_tensor_rec *t = s->recs[layer * s->per_layer + i];
        if (s->resident && t->type == H3I_TYPE_BF16) {
            out[i] = s->resident_bufs[layer * s->per_layer + i];
            continue;
        }
        h3i_gpu_tensor *slot = (s->odd_slots && (layer & 1))
            ? s->odd_slots[i] : s->slots[i];
        h3i_gpu_tensor *raw = s->resident
            ? s->resident_bufs[layer * s->per_layer + i]
            : s->staging[set][i];
        if (!h3i_gpu_convert_raw(s->gpu, slot, 0, raw, 0, t->type,
                                 (size_t)t->elements)) {
            h3i_fail(error, error_size, "%s", h3i_gpu_error(s->gpu));
            return 0;
        }
        out[i] = slot;
    }
    return 1;
}

double h3i_stream_wait_seconds(const h3i_stream *s) {
    return s ? s->wait_seconds : 0.0;
}

int h3i_stream_resident(const h3i_stream *s) { return s ? s->resident : 0; }

size_t h3i_stream_layers(const h3i_stream *s) { return s ? s->layers : 0; }

void h3i_stream_free(h3i_stream *s) {
    if (!s) return;
    h3i_stream_quiesce(s);
    for (size_t i = 0; i < s->per_layer; i++) {
        h3i_gpu_tensor_free(s->slots ? s->slots[i] : NULL);
        h3i_gpu_tensor_free(s->odd_slots ? s->odd_slots[i] : NULL);
        for (int set = 0; set < 2; set++) {
            h3i_gpu_tensor_free(s->staging[set] ? s->staging[set][i] : NULL);
        }
    }
    if (s->resident_bufs) {
        for (size_t idx = 0; idx < s->layers * s->per_layer; idx++) {
            h3i_gpu_tensor_free(s->resident_bufs[idx]);
        }
    }
    free(s->resident_bufs);
    free(s->staging[0]);
    free(s->staging[1]);
    free(s->odd_slots);
    free(s->slots);
    free(s->max_bytes);
    free(s->max_elements);
    free(s->recs);
    free(s);
}
