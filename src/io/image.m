/* Image I/O implementation (SPEC §4.8). Apple-only: ImageIO/CoreGraphics. */
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include "io/image.h"

#include "app/error.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

void h3i_image_free(h3i_image *image) {
    if (!image) return;
    free(image->rgb);
    image->rgb = NULL;
    image->width = image->height = 0;
}

int h3i_image_read(const char *path, h3i_image *out,
                   char *error, size_t error_size) {
    if (!path || !out) {
        h3i_fail(error, error_size, "invalid image read request");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        CGImageSourceRef source =
            CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) {
            h3i_fail(error, error_size, "cannot read image %s", path);
            return 0;
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        if (!image) {
            CFRelease(source);
            h3i_fail(error, error_size, "cannot decode image %s", path);
            return 0;
        }
        int orientation = 1;
        CFDictionaryRef properties =
            CGImageSourceCopyPropertiesAtIndex(source, 0, NULL);
        if (properties) {
            CFTypeRef value =
                CFDictionaryGetValue(properties, kCGImagePropertyOrientation);
            if (value && CFGetTypeID(value) == CFNumberGetTypeID()) {
                CFNumberGetValue((CFNumberRef)value, kCFNumberIntType,
                                 &orientation);
            }
            CFRelease(properties);
        }
        CFRelease(source);
        if (orientation < 1 || orientation > 8) orientation = 1;

        size_t w = CGImageGetWidth(image);
        size_t h = CGImageGetHeight(image);
        if (!w || !h || w > INT_MAX || h > INT_MAX) {
            CGImageRelease(image);
            h3i_fail(error, error_size, "image %s has invalid dimensions", path);
            return 0;
        }
        /* The image's own RGB color space; non-RGB models convert to sRGB. */
        CGColorSpaceRef original = CGImageGetColorSpace(image);
        CGColorSpaceRef space;
        if (original && CGColorSpaceGetModel(original) == kCGColorSpaceModelRGB) {
            space = CGColorSpaceRetain(original);
        } else {
            space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        }
        if (!space) {
            CGImageRelease(image);
            h3i_fail(error, error_size, "cannot create a color space for %s",
                     path);
            return 0;
        }
        /* Zero-cleared RGBX buffer: NoneSkipLast composites alpha over black. */
        size_t stride = w * 4;
        uint8_t *buffer = calloc(h, stride);
        CGContextRef context = buffer
            ? CGBitmapContextCreate(buffer, w, h, 8, stride, space,
                                    kCGImageAlphaNoneSkipLast |
                                        kCGBitmapByteOrder32Big)
            : NULL;
        CGColorSpaceRelease(space);
        if (!context) {
            free(buffer);
            CGImageRelease(image);
            h3i_fail(error, error_size, "cannot create a bitmap context for %s",
                     path);
            return 0;
        }
        CGContextDrawImage(context, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h),
                           image);
        CGImageRelease(image);
        CGContextRelease(context);

        int ow = orientation >= 5 ? (int)h : (int)w;
        int oh = orientation >= 5 ? (int)w : (int)h;
        uint8_t *rgb = malloc((size_t)ow * (size_t)oh * 3);
        if (!rgb) {
            free(buffer);
            h3i_fail(error, error_size, "out of memory reading %s", path);
            return 0;
        }
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                size_t sx, sy;
                switch (orientation) {
                case 2: sx = w - 1 - (size_t)x; sy = (size_t)y; break;
                case 3: sx = w - 1 - (size_t)x; sy = h - 1 - (size_t)y; break;
                case 4: sx = (size_t)x; sy = h - 1 - (size_t)y; break;
                case 5: sx = (size_t)y; sy = (size_t)x; break;
                case 6: sx = (size_t)y; sy = h - 1 - (size_t)x; break;
                case 7: sx = w - 1 - (size_t)y; sy = h - 1 - (size_t)x; break;
                case 8: sx = w - 1 - (size_t)y; sy = (size_t)x; break;
                default: sx = (size_t)x; sy = (size_t)y; break;
                }
                const uint8_t *p = buffer + sy * stride + sx * 4;
                uint8_t *q = rgb + ((size_t)y * (size_t)ow + (size_t)x) * 3;
                q[0] = p[0];
                q[1] = p[1];
                q[2] = p[2];
            }
        }
        free(buffer);
        out->width = ow;
        out->height = oh;
        out->rgb = rgb;
        return 1;
    }
}

int h3i_image_write_png(const char *path, const uint8_t *rgb,
                        int width, int height, char *error, size_t error_size) {
    if (!path || !rgb || width <= 0 || height <= 0) {
        h3i_fail(error, error_size, "invalid PNG write request");
        return 0;
    }
    @autoreleasepool {
        CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        if (!space) {
            h3i_fail(error, error_size, "cannot create the sRGB color space");
            return 0;
        }
        size_t stride = (size_t)width * 4;
        uint8_t *buffer = malloc((size_t)height * stride);
        if (!buffer) {
            CGColorSpaceRelease(space);
            h3i_fail(error, error_size, "out of memory writing %s", path);
            return 0;
        }
        for (size_t i = 0; i < (size_t)width * (size_t)height; i++) {
            buffer[i * 4 + 0] = rgb[i * 3 + 0];
            buffer[i * 4 + 1] = rgb[i * 3 + 1];
            buffer[i * 4 + 2] = rgb[i * 3 + 2];
            buffer[i * 4 + 3] = 255;
        }
        CGContextRef context =
            CGBitmapContextCreate(buffer, (size_t)width, (size_t)height, 8,
                                  stride, space,
                                  kCGImageAlphaNoneSkipLast |
                                      kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(space);
        CGImageRef image = context ? CGBitmapContextCreateImage(context) : NULL;
        if (context) CGContextRelease(context);
        free(buffer);
        if (!image) {
            h3i_fail(error, error_size, "cannot build the PNG image for %s",
                     path);
            return 0;
        }
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, (__bridge CFStringRef)UTTypePNG.identifier,
            1, NULL);
        int ok = 0;
        if (destination) {
            CGImageDestinationAddImage(destination, image, NULL);
            ok = CGImageDestinationFinalize(destination) ? 1 : 0;
            CFRelease(destination);
        }
        CGImageRelease(image);
        if (!ok) {
            h3i_fail(error, error_size, "cannot write PNG %s", path);
            return 0;
        }
        return 1;
    }
}
