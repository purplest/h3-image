/* Model directory auto-detection (SPEC §2.6): classify files by probe
 * tensor names, pick the lexicographically smallest match per kind. */
#include "weights/model.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "app/error.h"
#include "weights/source.h"

typedef enum {
    KIND_DIT,
    KIND_TEXT,
    KIND_VAE,
} kind;

static const char *probe_name(kind k)
{
    switch (k) {
    case KIND_DIT: return "blocks.0.attn.qkv_proj.weight";
    case KIND_TEXT: return "model.language_model.layers.0.self_attn.q_proj.weight";
    case KIND_VAE: return "decoder.x_embedder.weight";
    }
    return "";
}

static int is_dir(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

static int is_file(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

static void join(char *out, size_t n, const char *dir, const char *name)
{
    snprintf(out, n, "%s/%s", dir, name);
}

/* Open the path as a source silently; failure ⇒ not this kind. */
static int classifies(const char *path, kind k)
{
    h3i_source s = {0};
    char scratch[256];
    if (!h3i_source_add(&s, path, scratch, sizeof scratch)) {
        return 0;
    }
    int hit = h3i_source_find(&s, probe_name(k)) != NULL;
    h3i_source_free(&s);
    return hit;
}

static int ends_with(const char *s, const char *suffix)
{
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static int cmp_names(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Lexicographically smallest regular file in dir (not starting with '.',
 * suffix .gguf/.safetensors) that classifies as kind. Full path into out. */
static int find_kind(const char *dir, kind k, char *out, size_t out_size)
{
    DIR *d = opendir(dir);
    if (d == NULL) {
        return 0;
    }
    char **names = NULL;
    size_t n = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') {
            continue;
        }
        if (!ends_with(e->d_name, ".gguf") && !ends_with(e->d_name, ".safetensors")) {
            continue;
        }
        char full[4096];
        join(full, sizeof full, dir, e->d_name);
        if (!is_file(full)) {
            continue;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 16;
            char **nn = realloc(names, cap * sizeof *nn);
            if (nn == NULL) {
                break;
            }
            names = nn;
        }
        names[n] = strdup(e->d_name);
        if (names[n] == NULL) {
            break;
        }
        n++;
    }
    closedir(d);
    qsort(names, n, sizeof *names, cmp_names);
    int found = 0;
    for (size_t i = 0; i < n; i++) {
        char full[4096];
        join(full, sizeof full, dir, names[i]);
        if (classifies(full, k)) {
            h3i_path_copy(out, out_size, full);
            found = 1;
            break;
        }
    }
    for (size_t i = 0; i < n; i++) {
        free(names[i]);
    }
    free(names);
    return found;
}

static void set_empty(char *field, const char *value)
{
    if (field[0] == '\0' && value != NULL && value[0] != '\0') {
        h3i_path_copy(field, H3I_PATH_MAX, value);
    }
}

int h3i_model_resolve(const char *dir, h3i_paths *paths, char *error, size_t error_size)
{
    char p[H3I_PATH_MAX];
    char v[H3I_PATH_MAX];

    if (dir != NULL && dir[0] != '\0') {
        if (!is_dir(dir)) {
            h3i_fail(error, error_size, "model directory %s does not exist", dir);
            return 0;
        }

        /* dit: DIR/transformer if a directory classifying as DiT, else scan */
        if (paths->dit[0] == '\0') {
            join(p, sizeof p, dir, "transformer");
            if (is_dir(p) && classifies(p, KIND_DIT)) {
                set_empty(paths->dit, p);
            } else if (find_kind(dir, KIND_DIT, v, sizeof v)) {
                set_empty(paths->dit, v);
            }
        }

        /* dit_heads */
        join(p, sizeof p, dir, "pdd_fused_heads.safetensors");
        if (is_file(p)) {
            set_empty(paths->dit_heads, p);
        }

        /* text_encoder */
        if (paths->text_encoder[0] == '\0') {
            join(p, sizeof p, dir, "text_encoder");
            if (is_dir(p) && classifies(p, KIND_TEXT)) {
                set_empty(paths->text_encoder, p);
            } else if (find_kind(dir, KIND_TEXT, v, sizeof v)) {
                set_empty(paths->text_encoder, v);
            }
        }

        /* vae: prefer DIR/vae, else scan DIR */
        if (paths->vae[0] == '\0') {
            join(p, sizeof p, dir, "vae");
            if (is_dir(p) && find_kind(p, KIND_VAE, v, sizeof v)) {
                set_empty(paths->vae, v);
            } else if (find_kind(dir, KIND_VAE, v, sizeof v)) {
                set_empty(paths->vae, v);
            }
        }

        /* vae_config */
        join(p, sizeof p, dir, "vae/config.json");
        if (is_file(p)) {
            set_empty(paths->vae_config, p);
        }

        /* tokenizer: first existing of three candidates */
        if (paths->tokenizer[0] == '\0') {
            static const char *const candidates[] = {
                "tokenizer/tokenizer.json",
                "tokenizer.json",
                "text_encoder/tokenizer.json",
            };
            for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
                join(p, sizeof p, dir, candidates[i]);
                if (is_file(p)) {
                    set_empty(paths->tokenizer, p);
                    break;
                }
            }
        }

        /* pdd_config */
        join(p, sizeof p, dir, "pdd_config.json");
        if (is_file(p)) {
            set_empty(paths->pdd_config, p);
        }
    }

    const char *where = (dir != NULL && dir[0] != '\0') ? dir : "the command line";
    if (paths->dit[0] == '\0') {
        h3i_fail(error, error_size,
                 "no diffusion transformer found in %s (use --diffusion-model)", where);
        return 0;
    }
    if (paths->text_encoder[0] == '\0') {
        h3i_fail(error, error_size,
                 "no Qwen3-VL text encoder found in %s (use --text-encoder)", where);
        return 0;
    }
    if (paths->vae[0] == '\0') {
        h3i_fail(error, error_size, "no VAE found in %s (use --vae)", where);
        return 0;
    }
    if (paths->tokenizer[0] == '\0') {
        h3i_fail(error, error_size,
                 "no tokenizer.json found in %s (use --tokenizer)", where);
        return 0;
    }
    return 1;
}
