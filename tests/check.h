/* Shared test harness (SPEC §6.5): CHECK prints "FAIL file:line: msg";
 * every test ends with "%s: %d checks, %d failures\n" (ok/FAIL) and exits
 * nonzero when any check failed. */
#ifndef H3I_TESTS_CHECK_H
#define H3I_TESTS_CHECK_H

#include <stdio.h>

static int h3i_checks = 0;
static int h3i_failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        h3i_checks++;                                                          \
        if (!(cond)) {                                                         \
            h3i_failures++;                                                    \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

#define TEST_RESULT()                                                          \
    do {                                                                       \
        printf("%s: %d checks, %d failures\n",                                 \
               h3i_failures ? "FAIL" : "ok", h3i_checks, h3i_failures);        \
        return h3i_failures ? 1 : 0;                                           \
    } while (0)

#endif /* H3I_TESTS_CHECK_H */
