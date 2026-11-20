/* Minimal CoreGraphics stubs for the Linux ObjC syntax gate.
 * Declarations only; never compiled into any real build. */
#ifndef H3I_STUB_COREGRAPHICS_H
#define H3I_STUB_COREGRAPHICS_H

#import <Foundation/Foundation.h>

typedef double CGFloat;
typedef struct { CGFloat x, y, width, height; } CGRect;
static inline CGRect CGRectMake(CGFloat x, CGFloat y, CGFloat w, CGFloat h) {
    CGRect r = {x, y, w, h};
    return r;
}

typedef struct CGImage *CGImageRef;
typedef struct CGColorSpace *CGColorSpaceRef;
typedef struct CGContext *CGContextRef;
typedef int CGColorSpaceModel;
typedef unsigned int CGBitmapInfo;

enum { kCGColorSpaceModelRGB = 1 };
enum { kCGImageAlphaNoneSkipLast = 5, kCGBitmapByteOrder32Big = 4 << 12 };
extern const CFStringRef kCGColorSpaceSRGB;

size_t CGImageGetWidth(CGImageRef image);
size_t CGImageGetHeight(CGImageRef image);
CGColorSpaceRef CGImageGetColorSpace(CGImageRef image);
void CGImageRelease(CGImageRef image);

CGColorSpaceRef CGColorSpaceCreateWithName(CFStringRef name);
CGColorSpaceRef CGColorSpaceRetain(CGColorSpaceRef space);
void CGColorSpaceRelease(CGColorSpaceRef space);
CGColorSpaceModel CGColorSpaceGetModel(CGColorSpaceRef space);

CGContextRef CGBitmapContextCreate(void *data, size_t width, size_t height,
                                   size_t bitsPerComponent,
                                   size_t bytesPerRow, CGColorSpaceRef space,
                                   CGBitmapInfo bitmapInfo);
CGImageRef CGBitmapContextCreateImage(CGContextRef context);
void CGContextRelease(CGContextRef context);
void CGContextDrawImage(CGContextRef context, CGRect rect, CGImageRef image);

#endif /* H3I_STUB_COREGRAPHICS_H */
