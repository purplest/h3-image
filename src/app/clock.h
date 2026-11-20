/* Monotonic wall clock in seconds (SPEC §6: progress and profiling timers).
 * CLOCK_MONOTONIC is available on macOS 10.12+ and Linux. */
#ifndef H3I_APP_CLOCK_H
#define H3I_APP_CLOCK_H

#include <time.h>

static inline double h3i_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif /* H3I_APP_CLOCK_H */
