/* Internal GPU runtime structures — Objective-C only (.m files).
 * Public C structs wrap bridged ObjC objects so ARC never sees ObjC pointers
 * inside C structs. */
#ifndef H3I_GPU_GPU_INTERNAL_H
#define H3I_GPU_GPU_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "gpu/gpu.h"

struct h3i_gpu {
    void *obj; /* H3IGPU * (bridged retained) */
};

struct h3i_gpu_tensor {
    void *obj; /* H3ITensor * (bridged retained) */
};

@class H3IGPU;

@interface H3ITensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) h3i_dtype dtype;
@property(nonatomic, assign) size_t elements;
@property(nonatomic, assign) size_t bytes;
@property(nonatomic, unsafe_unretained) H3IGPU *owner;
@end

/* Cached MPSGraph: placeholders + result tensor, re-encoded per call with
 * fresh MPSGraphTensorData feeds (SPEC §6.3.1 graph cache). */
@interface H3IGraphEntry : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) NSMutableDictionary<NSString *, MPSGraphTensor *> *feeds;
@property(nonatomic, strong) MPSGraphTensor *result;
@end

@interface H3IGPU : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) id<MTLCommandBuffer> current;
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *inflight;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3IGraphEntry *> *graphCache;
@property(nonatomic, assign) h3i_gpu_stats stats;
@property(nonatomic, copy) NSString *label;
/* clang counts only explicit parameters for ObjC method format attributes
 * (verified on macOS and GNUstep runtimes): fmt = 1, varargs = 2. */
- (void)setError:(const char *)fmt, ... __attribute__((format(printf, 1, 2)));
- (const char *)lastError;
- (id<MTLComputePipelineState>)pipeline:(NSString *)name;
- (BOOL)encodeOp:(NSString *)pipelineName
       gridWidth:(NSUInteger)gw gridHeight:(NSUInteger)gh gridDepth:(NSUInteger)gd
      groupWidth:(NSUInteger)tw groupHeight:(NSUInteger)th groupDepth:(NSUInteger)td
          buffers:(NSArray<id<MTLBuffer>> *)buffers
          offsets:(NSArray<NSNumber *> *)offsets
           params:(const void *)params paramsSize:(size_t)psize;
- (void)noteAlloc:(size_t)bytes;
- (void)noteFree:(size_t)bytes;
- (h3i_gpu_stats *)statsPtr;
/* MPSGraph cache helpers (implemented in context.m, used by ops_h3i.m). */
- (H3IGraphEntry *)graphEntry:(NSString *)key build:(H3IGraphEntry *(^)(void))builder;
- (MPSGraphTensorData *)dataForBuffer:(id<MTLBuffer>)buf
                                shape:(NSArray<NSNumber *> *)shape
                                dtype:(h3i_dtype)dt;
- (BOOL)encodeGraph:(H3IGraphEntry *)e
            sources:(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)src
            results:(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)dst;
@end

static inline H3IGPU *gpu_obj(h3i_gpu *g)
{
    return (__bridge H3IGPU *)g->obj;
}

static inline H3ITensor *tensor_obj(h3i_gpu_tensor *t)
{
    return (__bridge H3ITensor *)t->obj;
}

static inline size_t dtype_size(h3i_dtype dt)
{
    switch (dt) {
    case H3I_DTYPE_F32: return 4;
    case H3I_DTYPE_BF16: return 2;
    case H3I_DTYPE_I8: return 1;
    case H3I_DTYPE_U32: return 4;
    }
    return 0;
}

#endif /* H3I_GPU_GPU_INTERNAL_H */
