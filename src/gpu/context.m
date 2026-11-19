/* GPU context: device, queue, runtime shader compile (safe math), eager
 * pipelines, command batching, MPSGraph cache, stats (SPEC §6.3). */
#import "gpu_internal.h"

#include <Availability.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* SDK-version guards: @available checks the runtime OS, but the symbols
 * themselves must exist in the SDK we compile against. */
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define H3I_SDK_MACOS15 1
#else
#define H3I_SDK_MACOS15 0
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
#define H3I_SDK_MACOS13 1
#else
#define H3I_SDK_MACOS13 0
#endif

#include "app/clock.h"
#include "app/error.h"

/* Every pipeline built eagerly at create (SPEC §6.3.4). */
static NSString *const kPipelineNames[] = {
    @"h3i_linear_f32",
    @"h3i_linear_f32_tiled",
    @"h3i_linear_f32_tiled_bf16",
    @"h3i_silu_f32",
    @"h3i_cast_f32_to_bf16",
    @"h3i_cast_bf16_to_f32",
    @"h3i_rms_norm_f32",
    @"h3i_scale_add_f32",
    @"h3i_layer_norm_f32",
    @"h3i_video_qkv_rope_f32",
    @"h3i_swiglu_f32",
    @"h3i_linear_bf16",
    @"h3i_rms_norm_bf16",
    @"h3i_adaln_bf16",
    @"h3i_gate_bf16",
    @"h3i_gate_adaln_bf16_exact_simd",
    @"h3i_qkv_rope_bf16",
    @"h3i_qkv_rope_bf16_coop",
    @"h3i_swiglu_bf16",
    @"h3i_layer_norm_bf16",
    @"h3i_gelu_bf16",
    @"h3i_vision_qkv_rope_bf16",
    @"h3i_head_rms_norm_bf16",
    @"h3i_rope_text_bf16",
    @"h3i_add_bf16",
    @"h3i_silu_mul_bf16",
    @"h3i_add_scaled_f32",
    @"h3i_vae_encoder_pad_f32",
    @"h3i_conv3d_f32",
    @"h3i_vae_encoder_group_norm_silu_f32",
    @"h3i_convert_f32_bf16", @"h3i_convert_f32_f32",
    @"h3i_convert_f16_bf16", @"h3i_convert_f16_f32",
    @"h3i_convert_bf16_bf16", @"h3i_convert_bf16_f32",
    @"h3i_convert_q8_0_bf16", @"h3i_convert_q8_0_f32",
    @"h3i_convert_q2_k_bf16", @"h3i_convert_q2_k_f32",
    @"h3i_convert_q3_k_bf16", @"h3i_convert_q3_k_f32",
    @"h3i_convert_q4_k_bf16", @"h3i_convert_q4_k_f32",
    @"h3i_convert_q5_k_bf16", @"h3i_convert_q5_k_f32",
    @"h3i_convert_q6_k_bf16", @"h3i_convert_q6_k_f32",
    @"h3i_gqa_causal_bf16",
    @"h3i_bf16_to_f16",
    @"h3i_f16_to_bf16_checked",
    @"h3i_transpose_bf16_f16",
};

@implementation H3IGraphEntry
@end

@interface H3IGPU ()
@property(nonatomic, assign) double createWall;
@end

@implementation H3IGPU {
    char _lastError[512];
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        _pipelines = [NSMutableDictionary new];
        _inflight = [NSMutableArray new];
        _graphCache = [NSMutableDictionary new];
        _lastError[0] = '\0';
        _label = @"";
        _createWall = h3i_now();
        memset(&_stats, 0, sizeof _stats);
    }
    return self;
}

- (void)setError:(const char *)fmt, ...
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(_lastError, sizeof _lastError, fmt, ap);
    va_end(ap);
}

- (const char *)lastError
{
    return _lastError[0] ? _lastError : "unknown Metal error";
}

- (id<MTLComputePipelineState>)pipeline:(NSString *)name
{
    return self.pipelines[name];
}

- (void)noteAlloc:(size_t)bytes
{
    _stats.allocated_bytes += bytes;
    _stats.live_bytes += bytes;
    if (_stats.live_bytes > _stats.peak_bytes) {
        _stats.peak_bytes = _stats.live_bytes;
    }
    _stats.allocations++;
}

- (void)noteFree:(size_t)bytes
{
    _stats.live_bytes -= bytes;
}

- (h3i_gpu_stats *)statsPtr
{
    return &_stats;
}

- (BOOL)encodeOp:(NSString *)pipelineName
       gridWidth:(NSUInteger)gw gridHeight:(NSUInteger)gh gridDepth:(NSUInteger)gd
      groupWidth:(NSUInteger)tw groupHeight:(NSUInteger)th groupDepth:(NSUInteger)td
          buffers:(NSArray<id<MTLBuffer>> *)buffers
          offsets:(NSArray<NSNumber *> *)offsets
           params:(const void *)params paramsSize:(size_t)psize
{
    if (self.current == nil) {
        [self setError:"h3i_gpu_begin() was not called"];
        return NO;
    }
    id<MTLComputePipelineState> pso = self.pipelines[pipelineName];
    if (pso == nil) {
        [self setError:"missing Metal pipeline %@", pipelineName];
        return NO;
    }
    id<MTLComputeCommandEncoder> enc = [self.current computeCommandEncoder];
    if (enc == nil) {
        [self setError:"cannot create compute encoder"];
        return NO;
    }
    [enc setComputePipelineState:pso];
    for (NSUInteger i = 0; i < buffers.count; i++) {
        [enc setBuffer:buffers[i] offset:offsets[i].unsignedIntegerValue atIndex:i];
    }
    if (params != NULL && psize > 0) {
        [enc setBytes:params length:psize atIndex:buffers.count];
    }
    [enc dispatchThreadgroups:MTLSizeMake(gw, gh, gd)
        threadsPerThreadgroup:MTLSizeMake(tw, th, td)];
    [enc endEncoding];
    _stats.direct_dispatches++;
    return YES;
}

/* ---- MPSGraph helpers ------------------------------------------------ */

- (H3IGraphEntry *)graphEntry:(NSString *)key build:(H3IGraphEntry *(^)(void))builder
{
    H3IGraphEntry *e = self.graphCache[key];
    if (e == nil) {
        e = builder();
        if (e != nil) {
            self.graphCache[key] = e;
        }
    }
    return e;
}

- (MPSGraphTensorData *)dataForBuffer:(id<MTLBuffer>)buf
                                shape:(NSArray<NSNumber *> *)shape
                                dtype:(h3i_dtype)dt
{
    MPSDataType mdt = (dt == H3I_DTYPE_BF16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:mdt];
}

- (BOOL)encodeGraph:(H3IGraphEntry *)e
            sources:(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)src
            results:(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)dst
{
    if (self.current == nil) {
        [self setError:"h3i_gpu_begin() was not called"];
        return NO;
    }
    [e.graph encodeToCommandBuffer:self.current sourceTensors:src resultsDictionary:dst];
    return YES;
}

@end

/* ---- public C API ----------------------------------------------------- */

h3i_gpu *h3i_gpu_create(const char *shader_path, char *error, size_t error_size)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        h3i_fail(error, error_size, "cannot initialize Metal");
        return NULL;
    }

    NSString *path = [NSString stringWithUTF8String:shader_path];
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:path
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
    if (source == nil) {
        h3i_fail(error, error_size, "cannot compile %s: %s", shader_path,
                 err.localizedDescription.UTF8String ?: "unreadable");
        return NULL;
    }

    MTLCompileOptions *opts = [MTLCompileOptions new];
#if H3I_SDK_MACOS15
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe; /* IEEE, no fast math: bit-exactness */
    } else {
        opts.fastMathEnabled = NO;
    }
#else
    opts.fastMathEnabled = NO;
#endif
    id<MTLLibrary> library = [device newLibraryWithSource:source options:opts error:&err];
    if (library == nil) {
        h3i_fail(error, error_size, "cannot compile %s: %s", shader_path,
                 err.localizedDescription.UTF8String ?: "unknown error");
        return NULL;
    }

    H3IGPU *g = [H3IGPU new];
    g.device = device;
    g.queue = [device newCommandQueue];
    g.library = library;

    for (size_t i = 0; i < sizeof kPipelineNames / sizeof kPipelineNames[0]; i++) {
        NSString *name = kPipelineNames[i];
        id<MTLFunction> fn = [library newFunctionWithName:name];
        if (fn == nil) {
            h3i_fail(error, error_size, "cannot build %s: function missing",
                     name.UTF8String);
            return NULL;
        }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
        if (pso == nil) {
            h3i_fail(error, error_size, "cannot build %s: %s", name.UTF8String,
                     err.localizedDescription.UTF8String ?: "unknown error");
            return NULL;
        }
        g.pipelines[name] = pso;
    }

    const char *dbg = getenv("H3I_DEBUG_GPU_MEMORY");
    if (dbg != NULL && dbg[0] != '\0') {
        fprintf(stderr, "h3: Metal live allocation at GPU startup: %.3f GiB\n",
                (double)g.stats.live_bytes / 1073741824.0);
    }

    h3i_gpu *out = malloc(sizeof *out);
    if (out == NULL) {
        h3i_fail(error, error_size, "out of memory");
        return NULL;
    }
    out->obj = (void *)CFBridgingRetain(g);
    return out;
}

void h3i_gpu_free(h3i_gpu *g)
{
    if (g == NULL) {
        return;
    }
    H3IGPU *gg = gpu_obj(g);

    const char *dbg = getenv("H3I_DEBUG_GPU_MEMORY");
    if (dbg != NULL && dbg[0] != '\0') {
        fprintf(stderr, "h3: Metal live allocation at GPU teardown: %.3f GiB\n",
                (double)gg.stats.live_bytes / 1073741824.0);
    }
    const char *prof = getenv("H3I_GPU_PROFILE");
    if (prof != NULL && prof[0] != '\0' && strcmp(prof, "0") != 0) {
        h3i_gpu_stats s = gg.stats;
        fprintf(stderr,
                "h3 profile: %-24s %-14s wall=%.2f encode=%.2f wait=%.2f root-gpu=%.2f "
                "peak=%.2fGiB alloc=%.2fGiB submissions=%llu direct=%llu linear=%llu "
                "conv=%llu attention=%llu\n",
                gg.label.UTF8String, "gpu", h3i_now() - gg.createWall,
                s.encode_seconds, s.wait_seconds, s.gpu_seconds,
                (double)s.peak_bytes / 1073741824.0,
                (double)s.allocated_bytes / 1073741824.0,
                (unsigned long long)s.submissions,
                (unsigned long long)s.direct_dispatches,
                (unsigned long long)s.mps_linear,
                (unsigned long long)s.mps_conv,
                (unsigned long long)s.mps_sdpa);
    }
    CFBridgingRelease(g->obj);
    free(g);
}

const char *h3i_gpu_error(h3i_gpu *g)
{
    return gpu_obj(g).lastError;
}

int h3i_gpu_get_info(h3i_gpu *g, h3i_gpu_info *out)
{
    H3IGPU *gg = gpu_obj(g);
    id<MTLDevice> d = gg.device;
    memset(out, 0, sizeof *out);
    snprintf(out->name, sizeof out->name, "%s", d.name.UTF8String ?: "unknown");
    snprintf(out->architecture, sizeof out->architecture, "Apple GPU");
    out->apple_gpu_family = 0;
#if H3I_SDK_MACOS13
    if (@available(macOS 13.0, *)) {
        snprintf(out->architecture, sizeof out->architecture, "%s",
                 d.architecture.name.UTF8String ?: "unknown");
        if ([d supportsFamily:MTLGPUFamilyApple9]) out->apple_gpu_family = 9;
        else if ([d supportsFamily:MTLGPUFamilyApple8]) out->apple_gpu_family = 8;
        else if ([d supportsFamily:MTLGPUFamilyApple7]) out->apple_gpu_family = 7;
        else if ([d supportsFamily:MTLGPUFamilyApple6]) out->apple_gpu_family = 6;
    }
#endif
#if H3I_SDK_MACOS15
    if (@available(macOS 15.0, *)) {
        out->metal4 = [d supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
    }
#endif
    out->unified_memory = NSProcessInfo.processInfo.physicalMemory;
    out->recommended_working_set = d.recommendedMaxWorkingSetSize;
    out->max_buffer_length = d.maxBufferLength;
    return 1;
}

uint64_t h3i_gpu_recommended_working_set(h3i_gpu *g)
{
    return gpu_obj(g).device.recommendedMaxWorkingSetSize;
}

/* ---- command model (SPEC §6.3.2) -------------------------------------- */

int h3i_gpu_begin(h3i_gpu *g)
{
    H3IGPU *gg = gpu_obj(g);
    if (gg.current != nil || gg.inflight.count > 0) {
        [gg setError:"h3i_gpu_begin() called while work is open or in flight"];
        return 0;
    }
    [gg setError:""];
    gg.current = [gg.queue commandBuffer];
    return 1;
}

int h3i_gpu_continue(h3i_gpu *g)
{
    H3IGPU *gg = gpu_obj(g);
    if (gg.current == nil) {
        [gg setError:"h3i_gpu_begin() was not called"];
        return 0;
    }
    [gg.current commit];
    [gg.inflight addObject:gg.current];
    gg.stats.submissions++;
    gg.current = [gg.queue commandBuffer];
    return 1;
}

int h3i_gpu_submit(h3i_gpu *g)
{
    H3IGPU *gg = gpu_obj(g);
    if (gg.current == nil) {
        [gg setError:"h3i_gpu_begin() was not called"];
        return 0;
    }
    double t0 = h3i_now();
    [gg.current commit];
    [gg.inflight addObject:gg.current];
    gg.current = nil;
    gg.stats.submissions++;
    int ok = 1;
    for (id<MTLCommandBuffer> cb in gg.inflight) {
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            [gg setError:"Metal command failed: %@",
             cb.error.localizedDescription ?: @"unknown"];
            ok = 0;
        }
        if (cb.GPUEndTime > cb.GPUStartTime) {
            gg.stats.gpu_seconds += cb.GPUEndTime - cb.GPUStartTime;
        }
    }
    [gg.inflight removeAllObjects];
    gg.stats.wait_seconds += h3i_now() - t0;
    return ok;
}

void h3i_gpu_profile_set_label(h3i_gpu *g, const char *label)
{
    gpu_obj(g).label = [NSString stringWithUTF8String:label ?: ""];
}

void h3i_gpu_get_stats(h3i_gpu *g, h3i_gpu_stats *out)
{
    *out = gpu_obj(g).stats;
}
