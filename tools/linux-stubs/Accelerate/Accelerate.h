/* Linux harness stub for <Accelerate/Accelerate.h> (PLAN.md Phase 0b).
 * Declares only the vImage surface src/io/host.c uses, so the pure-C TUs
 * syntax-check and host-math tests link on Linux. Never used on macOS:
 * the real framework provides these. Values match vImage's documented
 * constants; the stub implementation refuses to resize (returns an error)
 * so accidental use in a harness test fails loudly instead of silently. */
#ifndef H3I_LINUX_STUB_ACCELERATE_H
#define H3I_LINUX_STUB_ACCELERATE_H

#include <stddef.h>

typedef unsigned long vImagePixelCount;
typedef long vImage_Error;
typedef unsigned int vImage_Flags;

typedef struct {
    void *data;
    vImagePixelCount height;
    vImagePixelCount width;
    size_t rowBytes;
} vImage_Buffer;

#define kvImageNoError 0
#define kvImageEdgeExtend 0x8
#define kvImageHighQualityResampling 0x4000

vImage_Error vImageScale_ARGB8888(const vImage_Buffer *src,
                                const vImage_Buffer *dest,
                                void *tempBuffer, vImage_Flags flags);

#endif
