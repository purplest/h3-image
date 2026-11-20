/* Minimal ImageIO stubs for the Linux ObjC syntax gate. */
#ifndef H3I_STUB_IMAGEIO_H
#define H3I_STUB_IMAGEIO_H

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

typedef struct CGImageSource *CGImageSourceRef;
typedef struct CGImageDestination *CGImageDestinationRef;

extern const CFStringRef kCGImagePropertyOrientation;

CGImageSourceRef CGImageSourceCreateWithURL(CFURLRef url,
                                            CFDictionaryRef options);
CGImageRef CGImageSourceCreateImageAtIndex(CGImageSourceRef source,
                                           size_t index,
                                           CFDictionaryRef options);
CFDictionaryRef CGImageSourceCopyPropertiesAtIndex(CGImageSourceRef source,
                                                   size_t index,
                                                   CFDictionaryRef options);

CGImageDestinationRef CGImageDestinationCreateWithURL(CFURLRef url,
                                                      CFStringRef type,
                                                      size_t count,
                                                      CFDictionaryRef options);
void CGImageDestinationAddImage(CGImageDestinationRef destination,
                                CGImageRef image, CFDictionaryRef properties);
Boolean CGImageDestinationFinalize(CGImageDestinationRef destination);

#endif /* H3I_STUB_IMAGEIO_H */
