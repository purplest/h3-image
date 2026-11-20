/* inspect — SPEC §6.6: list the tensors of a weight source.
 * Usage: inspect PATH [substring]
 * PATH is a .gguf/.safetensors file or a directory of them. Exit 2 usage,
 * 1 open error, 0 success. */
#include "weights/gguf.h"
#include "weights/source.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: inspect PATH [substring]\n");
        return 2;
    }
    h3i_source source = {0};
    char error[512];
    if (!h3i_source_add(&source, argv[1], error, sizeof error)) {
        fprintf(stderr, "inspect: %s\n", error);
        return 1;
    }
    const char *substring = argc > 2 ? argv[2] : NULL;
    for (size_t i = 0; i < source.count; i++) {
        const h3i_tensor_rec *t = &source.recs[i];
        if (substring && !strstr(t->name, substring)) continue;
        printf("%s  %s  [", t->name, h3i_type_name(t->type));
        for (int d = 0; d < t->ndim; d++) {
            printf("%s%llu", d ? ", " : "",
                   (unsigned long long)t->shape[d]);
        }
        printf("]  %llu bytes\n", (unsigned long long)t->bytes);
    }
    h3i_source_free(&source);
    return 0;
}
