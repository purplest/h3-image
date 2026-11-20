/* Shared helpers for GPU tests (Apple-only; other platforms skip). */
#ifndef H3I_TESTS_GPU_TEST_H
#define H3I_TESTS_GPU_TEST_H

#ifdef __APPLE__

#include <stdio.h>
#include <stdlib.h>

#include "app/paths.h"
#include "gpu/gpu.h"

static h3i_gpu *test_gpu_create(void)
{
    char path[4096];
    if (h3i_shader_path(path, sizeof path) == NULL) {
        fprintf(stderr, "skip: build/h3i_shaders.metal not found (run make)\n");
        return NULL;
    }
    char err[512];
    h3i_gpu *g = h3i_gpu_create(path, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "skip: GPU unavailable: %s\n", err);
    }
    return g;
}

#endif /* __APPLE__ */
#endif /* H3I_TESTS_GPU_TEST_H */
