/* Minimal Metal stubs for the Linux ObjC syntax gate (make objc-syntax).
 * Declarations only; never compiled into any real build. */
#ifndef H3I_STUB_METAL_H
#define H3I_STUB_METAL_H

#import <Foundation/Foundation.h>

typedef struct {
    NSUInteger width, height, depth;
} MTLSize;

static inline MTLSize MTLSizeMake(NSUInteger w, NSUInteger h, NSUInteger d)
{
    MTLSize s = {w, h, d};
    return s;
}

enum { MTLResourceStorageModeShared = 0 };
enum { MTLCommandBufferStatusError = 4 };
enum { MTLMathModeSafe = 1 };
enum {
    MTLGPUFamilyApple6 = 6,
    MTLGPUFamilyApple7 = 7,
    MTLGPUFamilyApple8 = 8,
    MTLGPUFamilyApple9 = 9,
    MTLGPUFamilyMetal4 = 10,
};

@protocol MTLBuffer <NSObject>
@property (readonly) void *contents;
@property (readonly) NSUInteger length;
@end

@protocol MTLFunction <NSObject>
@end

@protocol MTLLibrary <NSObject>
- (id<MTLFunction>)newFunctionWithName:(NSString *)name;
@end

@protocol MTLComputePipelineState <NSObject>
@property (readonly) NSUInteger maxTotalThreadsPerThreadgroup;
@property (readonly) NSUInteger staticThreadgroupMemoryLength;
@end

@protocol MTLDeviceArchitecture <NSObject>
@property (readonly) NSString *name;
@end

@protocol MTLCommandEncoder <NSObject>
- (void)endEncoding;
@end

@protocol MTLComputeCommandEncoder <MTLCommandEncoder>
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pso;
- (void)setBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)off atIndex:(NSUInteger)idx;
- (void)setBytes:(const void *)bytes length:(NSUInteger)len atIndex:(NSUInteger)idx;
- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
@end

@protocol MTLBlitCommandEncoder <MTLCommandEncoder>
- (void)copyFromBuffer:(id<MTLBuffer>)src
          sourceOffset:(NSUInteger)so
              toBuffer:(id<MTLBuffer>)dst
     destinationOffset:(NSUInteger)doff
                  size:(NSUInteger)size;
@end

@protocol MTLCommandBuffer <NSObject>
- (void)commit;
- (void)waitUntilCompleted;
@property (readonly) NSUInteger status;
@property (readonly) NSError *error;
@property (readonly) double GPUStartTime;
@property (readonly) double GPUEndTime;
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (id<MTLBlitCommandEncoder>)blitCommandEncoder;
@end

@protocol MTLCommandQueue <NSObject>
- (id<MTLCommandBuffer>)commandBuffer;
@end

@interface MTLCompileOptions : NSObject
@property (assign) NSUInteger mathMode;
@property (assign) BOOL fastMathEnabled;
@end

@protocol MTLDevice <NSObject>
@property (readonly) NSString *name;
@property (readonly) NSUInteger recommendedMaxWorkingSetSize;
@property (readonly) NSUInteger maxBufferLength;
@property (readonly) NSUInteger maxThreadgroupMemoryLength;
@property (readonly) id<MTLDeviceArchitecture> architecture;
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)len options:(NSUInteger)opts;
- (id<MTLLibrary>)newLibraryWithSource:(NSString *)src
                               options:(MTLCompileOptions *)opts
                                 error:(NSError **)err;
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)fn
                                                             error:(NSError **)err;
- (BOOL)supportsFamily:(NSUInteger)family;
@end

id<MTLDevice> MTLCreateSystemDefaultDevice(void);

#endif /* H3I_STUB_METAL_H */
