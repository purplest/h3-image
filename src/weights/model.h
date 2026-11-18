/* Model directory auto-detection (SPEC §2.6). */
#ifndef H3I_WEIGHTS_MODEL_H
#define H3I_WEIGHTS_MODEL_H

#include <stddef.h>

#include "app/paths.h"

/* Fill the still-empty fields of paths from dir (explicit CLI paths win).
 * dir may be NULL when all components are given explicitly. Returns 1 on
 * success, 0 with error filled. */
int h3i_model_resolve(const char *dir, h3i_paths *paths, char *error, size_t error_size);

#endif /* H3I_WEIGHTS_MODEL_H */
