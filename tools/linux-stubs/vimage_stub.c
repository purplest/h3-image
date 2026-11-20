/* Link-only stub: harness tests exercise host math, never resizing.
 * Returning an error makes accidental use fail loudly. */
#include <Accelerate/Accelerate.h>

vImage_Error vImageScale_ARGB8888(const vImage_Buffer *src,
                                const vImage_Buffer *dest,
                                void *tempBuffer, vImage_Flags flags) {
    (void)src;
    (void)dest;
    (void)tempBuffer;
    (void)flags;
    return -1; /* not kvImageNoError */
}
