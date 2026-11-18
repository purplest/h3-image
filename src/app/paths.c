/* Path utilities and combined-shader lookup (SPEC §6.3.4). */
#include "app/paths.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

void h3i_path_copy(char *dst, size_t dst_size, const char *src)
{
    if (dst == NULL || dst_size == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= dst_size) {
        n = dst_size - 1;
    }
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static int is_regular_file(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

/* Directory of the running executable, symlinks resolved. */
static const char *exe_dir(char *buf, size_t buf_size)
{
#ifdef __APPLE__
    uint32_t size = (uint32_t)buf_size;
    if (_NSGetExecutablePath(buf, &size) != 0) {
        return NULL;
    }
#else
    ssize_t len = readlink("/proc/self/exe", buf, buf_size - 1);
    if (len <= 0) {
        return NULL;
    }
    buf[len] = '\0';
#endif
    char resolved[H3I_PATH_MAX];
    if (realpath(buf, resolved) == NULL) {
        return NULL;
    }
    char *slash = strrchr(resolved, '/');
    if (slash == NULL) {
        return NULL;
    }
    *slash = '\0';
    h3i_path_copy(buf, buf_size, resolved);
    return buf;
}

const char *h3i_shader_path(char *buf, size_t buf_size)
{
    if (buf == NULL || buf_size == 0) {
        return NULL;
    }

    const char *env = getenv("H3I_SHADERS");
    if (env != NULL && env[0] != '\0') {
        char resolved[H3I_PATH_MAX];
        if (realpath(env, resolved) == NULL || !is_regular_file(resolved)) {
            return NULL; /* explicit override: no fallback (SPEC §6.3.4) */
        }
        h3i_path_copy(buf, buf_size, resolved);
        return buf;
    }

    char dir[H3I_PATH_MAX];
    if (exe_dir(dir, sizeof dir) == NULL) {
        return NULL;
    }
    static const char *const candidates[] = {
        "%s/build/h3i_shaders.metal",
        "%s/h3i_shaders.metal",
        "%s/../build/h3i_shaders.metal",
    };
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        snprintf(buf, buf_size, candidates[i], dir);
        if (is_regular_file(buf)) {
            return buf;
        }
    }
    return NULL;
}
