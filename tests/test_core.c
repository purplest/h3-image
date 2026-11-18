/* M0 gate: app utilities — error formatting, monotonic clock, path copy,
 * shader lookup. Runs on Linux and macOS (no weights, no GPU). */
#include <stdlib.h>
#include <string.h>

#include "app/clock.h"
#include "app/error.h"
#include "app/paths.h"
#include "check.h"

int main(void)
{
    char buf[64];

    h3i_fail(buf, sizeof buf, "tensor %s has %d elements", "x", 42);
    CHECK(strcmp(buf, "tensor x has 42 elements") == 0, "format: got '%s'", buf);

    char small[8];
    h3i_fail(small, sizeof small, "a very long message that must be truncated");
    CHECK(strlen(small) == 7, "truncation length: %zu", strlen(small));
    CHECK(strcmp(small, "a very ") == 0, "truncation content: '%s'", small);

    h3i_fail(NULL, 0, "no-op %d", 1); /* must not crash */
    CHECK(1, "null error buffer is a no-op");

    double t0 = h3i_now();
    double t1 = h3i_now();
    CHECK(t0 > 0.0, "clock positive: %f", t0);
    CHECK(t1 >= t0, "clock monotonic: %f < %f", t1, t0);

    char dst[8];
    h3i_path_copy(dst, sizeof dst, "abcdefghij");
    CHECK(strcmp(dst, "abcdefg") == 0, "path copy truncation: '%s'", dst);
    h3i_path_copy(dst, sizeof dst, "ok");
    CHECK(strcmp(dst, "ok") == 0, "path copy short: '%s'", dst);
    h3i_path_copy(dst, sizeof dst, NULL);
    CHECK(dst[0] == '\0', "path copy NULL gives empty string");

    /* Shader lookup: H3I_SHADERS wins; a missing override fails with no
     * fallback. Run from the repo root so SPEC.md is a regular file. */
    char shader[H3I_PATH_MAX];
    setenv("H3I_SHADERS", "SPEC.md", 1);
    const char *p = h3i_shader_path(shader, sizeof shader);
    CHECK(p != NULL, "H3I_SHADERS regular file accepted");
    if (p != NULL) {
        CHECK(strstr(p, "SPEC.md") != NULL, "H3I_SHADERS path: '%s'", p);
    }
    setenv("H3I_SHADERS", "/nonexistent-h3i-shaders-xyz", 1);
    CHECK(h3i_shader_path(shader, sizeof shader) == NULL,
          "missing H3I_SHADERS gives NULL, no fallback");
    unsetenv("H3I_SHADERS");

    TEST_RESULT();
}
