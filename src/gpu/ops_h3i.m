/* GPU operation wrappers (SPEC §6.3.5): validation, dispatch shapes,
 * MPSGraph routing for large linear / MLP / SDPA. Param structs mirror
 * shaders.metal / kernels.metal exactly (Metal uint == uint32_t).
 *
 * API-sensitive spots (MPSGraph selectors) are marked MPSGRAPH-API; they are
 * the first things to check if the Mac build reports unknown selectors. */
#import "gpu_internal.h"

#include <string.h>

#include "weights/gguf.h"

/* ---- param structs (keep in sync with the .metal files) ----------------- */

typedef struct { uint32_t rows, in_w, out_w, has_bias; } LinearParams;
typedef struct { uint32_t rows, width; float eps; } NormParams;
typedef struct { uint32_t n; } ElemParams;
typedef struct { uint32_t rows, width; } SwigluParams;
typedef struct { uint32_t rows, width; } ScaleAddParams;
typedef struct { uint32_t n; float ls, rs; } AddScaledParams;
typedef struct { uint32_t rows, heads, hd, rope_half; float eps; uint32_t grouped; } QkvRopeParams;
typedef struct { uint32_t rows, heads, hd; float eps; } HeadNormParams;
typedef struct { uint32_t rows, qheads, kvheads, hd; } RopeTextParams;
typedef struct { uint32_t rows, width, slots, shift_slot, scale_slot; float eps; uint32_t in_off; } AdalnParams;
typedef struct { uint32_t rows, width, slots, gate_slot; } GateParams;
typedef struct { uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot; float eps; } GateAdalnParams;
typedef struct { uint32_t n, approximate; } GeluParams;
typedef struct { uint32_t out_off, raw_off, blocks; } ConvertParams;
typedef struct { uint32_t seq, qheads, kvheads, hd; float scale; } GqaParams;
typedef struct { uint32_t count; float scale; } HandoffParams;
typedef struct { uint32_t count; float scale; uint32_t bits; } CheckedParams;
typedef struct { uint32_t rows, columns, row_offset; float scale; } TransposeParams;
typedef struct { uint32_t height, width, channels, pad; } PadParams;
typedef struct { uint32_t n, d, h, w, c, o, od, oh, ow, kd, kh, kw, sd, sh, sw, has_bias; } Conv3dParams;
typedef struct { uint32_t n, h, w, c, groups; float eps; } GnParams;

/* ---- helpers ------------------------------------------------------------- */

static id<MTLBuffer> buf(const h3i_gpu_tensor *t)
{
    return tensor_obj((h3i_gpu_tensor *)t).buffer;
}

static NSNumber *boff(size_t elem_off, h3i_dtype dt)
{
    return @(elem_off * dtype_size(dt));
}

static int need(h3i_gpu *g, const h3i_gpu_tensor *t, h3i_dtype dt, size_t elems,
                const char *name)
{
    H3IGPU *gg = gpu_obj(g);
    if (t == NULL) {
        [gg setError:@"%s tensor is absent or too small", name];
        return 0;
    }
    H3ITensor *tt = tensor_obj((h3i_gpu_tensor *)t);
    if (tt.dtype != dt) {
        const char *want = dt == H3I_DTYPE_BF16 ? "BF16" :
                           dt == H3I_DTYPE_F32 ? "F32" :
                           dt == H3I_DTYPE_I8 ? "int8" : "U32";
        [gg setError:@"%s tensor is not %s", name, want];
        return 0;
    }
    if (tt.elements < elems) {
        [gg setError:@"%s tensor is absent or too small", name];
        return 0;
    }
    return 1;
}

static BOOL disp1d(h3i_gpu *g, NSString *name, NSArray<id<MTLBuffer>> *bufs,
                   NSArray<NSNumber *> *offs, const void *params, size_t ps, size_t count)
{
    H3IGPU *gg = gpu_obj(g);
    id<MTLComputePipelineState> pso = [gg pipeline:name];
    NSUInteger tw = MIN((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = (NSUInteger)((count + tw - 1) / tw);
    return [gg encodeOp:name gridWidth:groups gridHeight:1 gridDepth:1
             groupWidth:tw groupHeight:1 groupDepth:1
                buffers:bufs offsets:offs params:params paramsSize:ps];
}

static BOOL disp2d(h3i_gpu *g, NSString *name, NSArray<id<MTLBuffer>> *bufs,
                   NSArray<NSNumber *> *offs, const void *params, size_t ps,
                   size_t w, size_t h)
{
    H3IGPU *gg = gpu_obj(g);
    id<MTLComputePipelineState> pso = [gg pipeline:name];
    NSUInteger maxt = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger th = MIN((NSUInteger)16, maxt / 16);
    if (th == 0) {
        th = 1;
    }
    NSUInteger gw = (NSUInteger)((w + 15) / 16);
    NSUInteger gh = (NSUInteger)((h + th - 1) / th);
    return [gg encodeOp:name gridWidth:gw gridHeight:gh gridDepth:1
             groupWidth:16 groupHeight:th groupDepth:1
                buffers:bufs offsets:offs params:params paramsSize:ps];
}

/* One threadgroup per row; largest power of two ≤ min(256, max). */
static BOOL disp_rows(h3i_gpu *g, NSString *name, NSArray<id<MTLBuffer>> *bufs,
                      NSArray<NSNumber *> *offs, const void *params, size_t ps, size_t rows)
{
    H3IGPU *gg = gpu_obj(g);
    id<MTLComputePipelineState> pso = [gg pipeline:name];
    NSUInteger cap = MIN((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
    NSUInteger tw = 1;
    while (tw * 2 <= cap) {
        tw *= 2;
    }
    return [gg encodeOp:name gridWidth:(NSUInteger)rows gridHeight:1 gridDepth:1
             groupWidth:tw groupHeight:1 groupDepth:1
                buffers:bufs offsets:offs params:params paramsSize:ps];
}

/* ---- MPSGraph linear / MLP / SDPA ---------------------------------------- */

static MPSDataType mdt(h3i_dtype dt)
{
    return dt == H3I_DTYPE_BF16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
}

static BOOL mps_linear(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                       size_t rows, size_t in_w, size_t out_w, h3i_dtype dt)
{
    H3IGPU *gg = gpu_obj(g);
    NSString *key = [NSString stringWithFormat:@"lin_%d_%zu_%zu_%zu_%d",
                     (int)dt, rows, in_w, out_w, b != NULL];
    H3IGraphEntry *e = [gg graphEntry:key build:^H3IGraphEntry *{
        H3IGraphEntry *ge = [H3IGraphEntry new];
        ge.graph = [MPSGraph new];
        ge.feeds = [NSMutableDictionary new];
        MPSGraphTensor *x = [ge.graph placeholderWithShape:@[@(rows), @(in_w)]
                                                  dataType:mdt(dt) name:nil];
        MPSGraphTensor *w = [ge.graph placeholderWithShape:@[@(out_w), @(in_w)]
                                                  dataType:mdt(dt) name:nil];
        /* MPSGRAPH-API */
        MPSGraphTensor *y = [ge.graph matrixMultiplicationWithPrimaryTensor:x
                                                            secondaryTensor:w
                                                              transposeLeft:NO
                                                             transposeRight:YES
                                                                       name:nil];
        if (b != NULL) {
            MPSGraphTensor *bb = [ge.graph placeholderWithShape:@[@(out_w)]
                                                       dataType:mdt(dt) name:nil];
            y = [ge.graph additionWithPrimaryTensor:y secondaryTensor:bb name:nil];
            ge.feeds[@"b"] = bb;
        }
        ge.feeds[@"x"] = x;
        ge.feeds[@"w"] = w;
        ge.result = y;
        return ge;
    }];
    if (e == nil) {
        [gg setError:"cannot build MPSGraph linear"];
        return NO;
    }
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *src = [NSMutableDictionary new];
    src[e.feeds[@"x"]] = [gg dataForBuffer:buf(in) shape:@[@(rows), @(in_w)] dtype:dt];
    src[e.feeds[@"w"]] = [gg dataForBuffer:buf(W) shape:@[@(out_w), @(in_w)] dtype:dt];
    if (b != NULL) {
        src[e.feeds[@"b"]] = [gg dataForBuffer:buf(b) shape:@[@(out_w)] dtype:dt];
    }
    MPSGraphTensorData *dst = [gg dataForBuffer:buf(out) shape:@[@(rows), @(out_w)] dtype:dt];
    if (![gg encodeGraph:e sources:src results:@{e.result: dst}]) {
        return NO;
    }
    gg.statsPtr->mps_linear++;
    return YES;
}

int h3i_gpu_linear_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                       size_t rows, size_t in_w, size_t out_w)
{
    if (!need(g, out, H3I_DTYPE_F32, rows * out_w, "out") ||
        !need(g, in, H3I_DTYPE_F32, rows * in_w, "in") ||
        !need(g, W, H3I_DTYPE_F32, out_w * in_w, "W") ||
        (b && !need(g, b, H3I_DTYPE_F32, out_w, "b"))) {
        return 0;
    }
    if (rows >= 16 && out_w == 5376 && (in_w == 32 || in_w == 96)) {
        LinearParams p = {(uint32_t)rows, (uint32_t)in_w, (uint32_t)out_w, b != NULL};
        NSArray *bufs = b ? @[buf(out), buf(in), buf(W), buf(b)]
                          : @[buf(out), buf(in), buf(W), buf(W)];
        NSArray *offs = @[@0, @0, @0, @0];
        return disp2d(g, @"h3i_linear_f32_tiled", bufs, offs, &p, sizeof p, out_w, rows);
    }
    if (rows >= 32 && in_w >= 256 && out_w >= 256) {
        return mps_linear(g, out, in, W, b, rows, in_w, out_w, H3I_DTYPE_F32);
    }
    LinearParams p = {(uint32_t)rows, (uint32_t)in_w, (uint32_t)out_w, b != NULL};
    NSArray *bufs = b ? @[buf(out), buf(in), buf(W), buf(b)]
                      : @[buf(out), buf(in), buf(W), buf(W)];
    NSArray *offs = @[@0, @0, @0, @0];
    return disp2d(g, @"h3i_linear_f32", bufs, offs, &p, sizeof p, out_w, rows);
}

int h3i_gpu_patch_linear_bf16_offset(h3i_gpu *g, h3i_gpu_tensor *out, size_t out_off,
                                     const h3i_gpu_tensor *in, size_t in_off,
                                     const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                                     size_t rows, size_t in_w, size_t out_w)
{
    if (out_w != 5376 || (in_w != 32 && in_w != 96)) {
        [gpu_obj(g) setError:"patch_linear_bf16_offset supports out=5376, in=32|96 only"];
        return 0;
    }
    if (!need(g, out, H3I_DTYPE_BF16, out_off + rows * out_w, "out") ||
        !need(g, in, H3I_DTYPE_F32, in_off + rows * in_w, "in") ||
        !need(g, W, H3I_DTYPE_F32, out_w * in_w, "W") ||
        !need(g, b, H3I_DTYPE_F32, out_w, "b")) {
        return 0;
    }
    LinearParams p = {(uint32_t)rows, (uint32_t)in_w, (uint32_t)out_w, 1};
    NSArray *bufs = @[buf(out), buf(in), buf(W), buf(b)];
    NSArray *offs = @[boff(out_off, H3I_DTYPE_BF16), boff(in_off, H3I_DTYPE_F32), @0, @0];
    return disp2d(g, @"h3i_linear_f32_tiled_bf16", bufs, offs, &p, sizeof p, out_w, rows);
}

int h3i_gpu_linear_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                        const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                        size_t rows, size_t in_w, size_t out_w)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * out_w, "out") ||
        !need(g, in, H3I_DTYPE_BF16, rows * in_w, "in") ||
        !need(g, W, H3I_DTYPE_BF16, out_w * in_w, "W") ||
        (b && !need(g, b, H3I_DTYPE_BF16, out_w, "b"))) {
        return 0;
    }
    if (rows >= 32 && in_w >= 256 && out_w >= 256) {
        return mps_linear(g, out, in, W, b, rows, in_w, out_w, H3I_DTYPE_BF16);
    }
    LinearParams p = {(uint32_t)rows, (uint32_t)in_w, (uint32_t)out_w, b != NULL};
    NSArray *bufs = b ? @[buf(out), buf(in), buf(W), buf(b)]
                      : @[buf(out), buf(in), buf(W), buf(W)];
    NSArray *offs = @[@0, @0, @0, @0];
    return disp2d(g, @"h3i_linear_bf16", bufs, offs, &p, sizeof p, out_w, rows);
}

int h3i_gpu_mlp_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                     const h3i_gpu_tensor *fc1, const h3i_gpu_tensor *fc2,
                     size_t rows, size_t in_w, size_t hidden, size_t out_w)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * out_w, "out") ||
        !need(g, in, H3I_DTYPE_BF16, rows * in_w, "in") ||
        !need(g, fc1, H3I_DTYPE_BF16, 2 * hidden * in_w, "fc1") ||
        !need(g, fc2, H3I_DTYPE_BF16, out_w * hidden, "fc2")) {
        return 0;
    }
    H3IGPU *gg = gpu_obj(g);
    NSString *key = [NSString stringWithFormat:@"mlp_%zu_%zu_%zu_%zu",
                     rows, in_w, hidden, out_w];
    H3IGraphEntry *e = [gg graphEntry:key build:^H3IGraphEntry *{
        H3IGraphEntry *ge = [H3IGraphEntry new];
        ge.graph = [MPSGraph new];
        ge.feeds = [NSMutableDictionary new];
        MPSGraphTensor *x = [ge.graph placeholderWithShape:@[@(rows), @(in_w)]
                                                  dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *w1 = [ge.graph placeholderWithShape:@[@(2 * hidden), @(in_w)]
                                                   dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *w2 = [ge.graph placeholderWithShape:@[@(out_w), @(hidden)]
                                                   dataType:MPSDataTypeBFloat16 name:nil];
        /* MPSGRAPH-API */
        MPSGraphTensor *fused = [ge.graph matrixMultiplicationWithPrimaryTensor:x
                                                                secondaryTensor:w1
                                                                  transposeLeft:NO
                                                                 transposeRight:YES
                                                                           name:nil];
        MPSGraphTensor *gate = [ge.graph sliceTensor:fused
                                              starts:@[@0, @0]
                                                ends:@[@(rows), @(hidden)]
                                             strides:@[@1, @1] name:nil];
        MPSGraphTensor *up = [ge.graph sliceTensor:fused
                                            starts:@[@0, @(hidden)]
                                              ends:@[@(rows), @(2 * hidden)]
                                           strides:@[@1, @1] name:nil];
        MPSGraphTensor *sig = [ge.graph sigmoidWithTensor:gate name:nil];
        MPSGraphTensor *act = [ge.graph multiplicationWithPrimaryTensor:gate
                                                        secondaryTensor:sig name:nil];
        MPSGraphTensor *prod = [ge.graph multiplicationWithPrimaryTensor:act
                                                         secondaryTensor:up name:nil];
        ge.result = [ge.graph matrixMultiplicationWithPrimaryTensor:prod
                                                    secondaryTensor:w2
                                                      transposeLeft:NO
                                                     transposeRight:YES
                                                               name:nil];
        ge.feeds[@"x"] = x;
        ge.feeds[@"w1"] = w1;
        ge.feeds[@"w2"] = w2;
        return ge;
    }];
    if (e == nil) {
        [gg setError:"cannot build MPSGraph MLP"];
        return NO;
    }
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *src = [NSMutableDictionary new];
    src[e.feeds[@"x"]] = [gg dataForBuffer:buf(in) shape:@[@(rows), @(in_w)] dtype:H3I_DTYPE_BF16];
    src[e.feeds[@"w1"]] = [gg dataForBuffer:buf(fc1) shape:@[@(2 * hidden), @(in_w)] dtype:H3I_DTYPE_BF16];
    src[e.feeds[@"w2"]] = [gg dataForBuffer:buf(fc2) shape:@[@(out_w), @(hidden)] dtype:H3I_DTYPE_BF16];
    MPSGraphTensorData *dst = [gg dataForBuffer:buf(out) shape:@[@(rows), @(out_w)] dtype:H3I_DTYPE_BF16];
    if (![gg encodeGraph:e sources:src results:@{e.result: dst}]) {
        return NO;
    }
    gg.statsPtr->mps_linear += 2;
    return YES;
}

/* SDPA via graph ops: transpose to heads-major, matmul → scale → softmax →
 * matmul, transpose back. (First-pass substitute for the fused MPSGraph
 * scaledDotProductAttention op; same math, upgrade during tuning.) */
static BOOL mps_sdpa(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                     const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                     size_t seq, size_t heads, size_t hd, float scale, h3i_dtype dt)
{
    H3IGPU *gg = gpu_obj(g);
    NSString *key = [NSString stringWithFormat:@"sdpa_%d_%zu_%zu_%zu",
                     (int)dt, seq, heads, hd];
    H3IGraphEntry *e = [gg graphEntry:key build:^H3IGraphEntry *{
        H3IGraphEntry *ge = [H3IGraphEntry new];
        ge.graph = [MPSGraph new];
        ge.feeds = [NSMutableDictionary new];
        NSArray<NSNumber *> *shape = @[@(seq), @(heads), @(hd)];
        MPSGraphTensor *qt = [ge.graph placeholderWithShape:shape dataType:mdt(dt) name:nil];
        MPSGraphTensor *kt = [ge.graph placeholderWithShape:shape dataType:mdt(dt) name:nil];
        MPSGraphTensor *vt = [ge.graph placeholderWithShape:shape dataType:mdt(dt) name:nil];
        /* MPSGRAPH-API */
        MPSGraphTensor *qh = [ge.graph transposeTensor:qt dimension:0 withDimension:1 name:nil];
        MPSGraphTensor *kh = [ge.graph transposeTensor:kt dimension:0 withDimension:1 name:nil];
        MPSGraphTensor *vh = [ge.graph transposeTensor:vt dimension:0 withDimension:1 name:nil];
        MPSGraphTensor *scores = [ge.graph matrixMultiplicationWithPrimaryTensor:qh
                                                                 secondaryTensor:kh
                                                                   transposeLeft:NO
                                                                  transposeRight:YES
                                                                            name:nil];
        MPSGraphTensor *sc = [ge.graph constantWithScalar:(double)scale
                                                    shape:@[@1, @1, @1]
                                                 dataType:mdt(dt) name:nil];
        MPSGraphTensor *scaled = [ge.graph multiplicationWithPrimaryTensor:scores
                                                           secondaryTensor:sc name:nil];
        MPSGraphTensor *probs = [ge.graph softMaxWithTensor:scaled axis:-1 name:nil];
        MPSGraphTensor *oh = [ge.graph matrixMultiplicationWithPrimaryTensor:probs
                                                             secondaryTensor:vh
                                                               transposeLeft:NO
                                                              transposeRight:NO
                                                                        name:nil];
        ge.result = [ge.graph transposeTensor:oh dimension:0 withDimension:1 name:nil];
        ge.feeds[@"q"] = qt;
        ge.feeds[@"k"] = kt;
        ge.feeds[@"v"] = vt;
        return ge;
    }];
    if (e == nil) {
        [gg setError:"native MPSGraph SDPA is unavailable"];
        return NO;
    }
    NSArray<NSNumber *> *shape = @[@(seq), @(heads), @(hd)];
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *src = [NSMutableDictionary new];
    src[e.feeds[@"q"]] = [gg dataForBuffer:buf(q) shape:shape dtype:dt];
    src[e.feeds[@"k"]] = [gg dataForBuffer:buf(k) shape:shape dtype:dt];
    src[e.feeds[@"v"]] = [gg dataForBuffer:buf(v) shape:shape dtype:dt];
    MPSGraphTensorData *dst = [gg dataForBuffer:buf(out) shape:shape dtype:dt];
    if (![gg encodeGraph:e sources:src results:@{e.result: dst}]) {
        return NO;
    }
    gg.statsPtr->mps_sdpa++;
    return YES;
}

int h3i_gpu_sdpa_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                     const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                     size_t seq, size_t heads, size_t hd, float scale)
{
    size_t n = seq * heads * hd;
    if (!need(g, out, H3I_DTYPE_F32, n, "out") || !need(g, q, H3I_DTYPE_F32, n, "q") ||
        !need(g, k, H3I_DTYPE_F32, n, "k") || !need(g, v, H3I_DTYPE_F32, n, "v")) {
        return 0;
    }
    return mps_sdpa(g, out, q, k, v, seq, heads, hd, scale, H3I_DTYPE_F32);
}

int h3i_gpu_sdpa_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                      const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                      size_t seq, size_t heads, size_t hd, float scale)
{
    size_t n = seq * heads * hd;
    if (!need(g, out, H3I_DTYPE_BF16, n, "out") || !need(g, q, H3I_DTYPE_BF16, n, "q") ||
        !need(g, k, H3I_DTYPE_BF16, n, "k") || !need(g, v, H3I_DTYPE_BF16, n, "v")) {
        return 0;
    }
    return mps_sdpa(g, out, q, k, v, seq, heads, hd, scale, H3I_DTYPE_BF16);
}

/* ---- elementwise / norms -------------------------------------------------- */

#define SIMPLE1D(fname, kname, dt_out, dt_in, PARAMS)                          \
    int fname(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in, size_t n) \
    {                                                                          \
        if (!need(g, out, dt_out, n, "out") || !need(g, in, dt_in, n, "in")) { \
            return 0;                                                          \
        }                                                                      \
        PARAMS p = {(uint32_t)n};                                              \
        return disp1d(g, kname, @[buf(out), buf(in)], @[@0, @0], &p, sizeof p, n); \
    }

SIMPLE1D(h3i_gpu_silu_f32, @"h3i_silu_f32", H3I_DTYPE_F32, H3I_DTYPE_F32, ElemParams)
SIMPLE1D(h3i_gpu_cast_f32_to_bf16, @"h3i_cast_f32_to_bf16", H3I_DTYPE_BF16, H3I_DTYPE_F32, ElemParams)
SIMPLE1D(h3i_gpu_cast_bf16_to_f32, @"h3i_cast_bf16_to_f32", H3I_DTYPE_F32, H3I_DTYPE_BF16, ElemParams)

int h3i_gpu_copy_bf16(h3i_gpu *g, h3i_gpu_tensor *dst, size_t doff,
                      const h3i_gpu_tensor *src, size_t soff, size_t n)
{
    if (!need(g, dst, H3I_DTYPE_BF16, doff + n, "dst") ||
        !need(g, src, H3I_DTYPE_BF16, soff + n, "src")) {
        return 0;
    }
    H3IGPU *gg = gpu_obj(g);
    if (gg.current == nil) {
        [gg setError:"h3i_gpu_begin() was not called"];
        return 0;
    }
    id<MTLBlitCommandEncoder> blit = [gg.current blitCommandEncoder];
    [blit copyFromBuffer:buf(src) sourceOffset:soff * 2
                toBuffer:buf(dst) destinationOffset:doff * 2 size:n * 2];
    [blit endEncoding];
    gg.statsPtr->blits++;
    return 1;
}

int h3i_gpu_copy_f32(h3i_gpu *g, h3i_gpu_tensor *dst, size_t doff,
                     const h3i_gpu_tensor *src, size_t soff, size_t n)
{
    if (!need(g, dst, H3I_DTYPE_F32, doff + n, "dst") ||
        !need(g, src, H3I_DTYPE_F32, soff + n, "src")) {
        return 0;
    }
    H3IGPU *gg = gpu_obj(g);
    if (gg.current == nil) {
        [gg setError:"h3i_gpu_begin() was not called"];
        return 0;
    }
    id<MTLBlitCommandEncoder> blit = [gg.current blitCommandEncoder];
    [blit copyFromBuffer:buf(src) sourceOffset:soff * 4
                toBuffer:buf(dst) destinationOffset:doff * 4 size:n * 4];
    [blit endEncoding];
    gg.statsPtr->blits++;
    return 1;
}

int h3i_gpu_rms_norm_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                         const h3i_gpu_tensor *w, size_t rows, size_t width, float eps)
{
    if (!need(g, out, H3I_DTYPE_F32, rows * width, "out") ||
        !need(g, in, H3I_DTYPE_F32, rows * width, "in") ||
        !need(g, w, H3I_DTYPE_F32, width, "w")) {
        return 0;
    }
    NormParams p = {(uint32_t)rows, (uint32_t)width, eps};
    return disp_rows(g, @"h3i_rms_norm_f32", @[buf(out), buf(in), buf(w)],
                     @[@0, @0, @0], &p, sizeof p, rows);
}

int h3i_gpu_rms_norm_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                          const h3i_gpu_tensor *w, size_t rows, size_t width, float eps)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * width, "out") ||
        !need(g, in, H3I_DTYPE_BF16, rows * width, "in") ||
        !need(g, w, H3I_DTYPE_BF16, width, "w")) {
        return 0;
    }
    NormParams p = {(uint32_t)rows, (uint32_t)width, eps};
    return disp_rows(g, @"h3i_rms_norm_bf16", @[buf(out), buf(in), buf(w)],
                     @[@0, @0, @0], &p, sizeof p, rows);
}

int h3i_gpu_layer_norm_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                           const h3i_gpu_tensor *w, const h3i_gpu_tensor *b,
                           size_t rows, size_t width, float eps)
{
    if (!need(g, out, H3I_DTYPE_F32, rows * width, "out") ||
        !need(g, in, H3I_DTYPE_F32, rows * width, "in") ||
        !need(g, w, H3I_DTYPE_F32, width, "w") ||
        !need(g, b, H3I_DTYPE_F32, width, "b")) {
        return 0;
    }
    NormParams p = {(uint32_t)rows, (uint32_t)width, eps};
    return disp_rows(g, @"h3i_layer_norm_f32", @[buf(out), buf(in), buf(w), buf(b)],
                     @[@0, @0, @0, @0], &p, sizeof p, rows);
}

int h3i_gpu_layer_norm_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                            const h3i_gpu_tensor *w, const h3i_gpu_tensor *b,
                            size_t rows, size_t width, float eps)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * width, "out") ||
        !need(g, in, H3I_DTYPE_BF16, rows * width, "in") ||
        !need(g, w, H3I_DTYPE_BF16, width, "w") ||
        !need(g, b, H3I_DTYPE_BF16, width, "b")) {
        return 0;
    }
    NormParams p = {(uint32_t)rows, (uint32_t)width, eps};
    return disp_rows(g, @"h3i_layer_norm_bf16", @[buf(out), buf(in), buf(w), buf(b)],
                     @[@0, @0, @0, @0], &p, sizeof p, rows);
}

int h3i_gpu_swiglu_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *fused,
                       size_t rows, size_t width)
{
    if (!need(g, out, H3I_DTYPE_F32, rows * width, "out") ||
        !need(g, fused, H3I_DTYPE_F32, rows * 2 * width, "fused")) {
        return 0;
    }
    SwigluParams p = {(uint32_t)rows, (uint32_t)width};
    return disp2d(g, @"h3i_swiglu_f32", @[buf(out), buf(fused)], @[@0, @0],
                  &p, sizeof p, width, rows);
}

int h3i_gpu_swiglu_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *fused,
                        size_t rows, size_t width)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * width, "out") ||
        !need(g, fused, H3I_DTYPE_BF16, rows * 2 * width, "fused")) {
        return 0;
    }
    SwigluParams p = {(uint32_t)rows, (uint32_t)width};
    return disp2d(g, @"h3i_swiglu_bf16", @[buf(out), buf(fused)], @[@0, @0],
                  &p, sizeof p, width, rows);
}

int h3i_gpu_scale_add_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *res,
                          const h3i_gpu_tensor *branch, const h3i_gpu_tensor *scale,
                          size_t rows, size_t width)
{
    if (!need(g, out, H3I_DTYPE_F32, rows * width, "out") ||
        !need(g, res, H3I_DTYPE_F32, rows * width, "res") ||
        !need(g, branch, H3I_DTYPE_F32, rows * width, "branch") ||
        !need(g, scale, H3I_DTYPE_F32, width, "scale")) {
        return 0;
    }
    ScaleAddParams p = {(uint32_t)rows, (uint32_t)width};
    return disp2d(g, @"h3i_scale_add_f32", @[buf(out), buf(res), buf(branch), buf(scale)],
                  @[@0, @0, @0, @0], &p, sizeof p, width, rows);
}

int h3i_gpu_add_scaled_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *l,
                           const h3i_gpu_tensor *r, float ls, float rs, size_t n)
{
    if (!need(g, out, H3I_DTYPE_F32, n, "out") || !need(g, l, H3I_DTYPE_F32, n, "l") ||
        !need(g, r, H3I_DTYPE_F32, n, "r")) {
        return 0;
    }
    AddScaledParams p = {(uint32_t)n, ls, rs};
    return disp1d(g, @"h3i_add_scaled_f32", @[buf(out), buf(l), buf(r)],
                  @[@0, @0, @0], &p, sizeof p, n);
}

int h3i_gpu_gelu_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                      size_t n, int approximate)
{
    if (!need(g, out, H3I_DTYPE_BF16, n, "out") || !need(g, in, H3I_DTYPE_BF16, n, "in")) {
        return 0;
    }
    GeluParams p = {(uint32_t)n, (uint32_t)approximate};
    return disp1d(g, @"h3i_gelu_bf16", @[buf(out), buf(in)], @[@0, @0], &p, sizeof p, n);
}

int h3i_gpu_add_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *l,
                     const h3i_gpu_tensor *r, size_t n)
{
    if (!need(g, out, H3I_DTYPE_BF16, n, "out") || !need(g, l, H3I_DTYPE_BF16, n, "l") ||
        !need(g, r, H3I_DTYPE_BF16, n, "r")) {
        return 0;
    }
    ElemParams p = {(uint32_t)n};
    return disp1d(g, @"h3i_add_bf16", @[buf(out), buf(l), buf(r)],
                  @[@0, @0, @0], &p, sizeof p, n);
}

int h3i_gpu_silu_mul_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *gate,
                          const h3i_gpu_tensor *up, size_t n)
{
    if (!need(g, out, H3I_DTYPE_BF16, n, "out") ||
        !need(g, gate, H3I_DTYPE_BF16, n, "gate") ||
        !need(g, up, H3I_DTYPE_BF16, n, "up")) {
        return 0;
    }
    ElemParams p = {(uint32_t)n};
    return disp1d(g, @"h3i_silu_mul_bf16", @[buf(out), buf(gate), buf(up)],
                  @[@0, @0, @0], &p, sizeof p, n);
}

/* ---- QKV / RoPE / AdaLN / gate -------------------------------------------- */

int h3i_gpu_qkv_rope_bf16(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                          h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                          const h3i_gpu_tensor *cosb, const h3i_gpu_tensor *sinb,
                          const h3i_gpu_tensor *q_norm, const h3i_gpu_tensor *k_norm,
                          size_t rows, size_t heads, size_t hd, size_t rope_half,
                          float eps, int grouped)
{
    size_t n = rows * heads * hd;
    if (!need(g, q, H3I_DTYPE_BF16, n, "q") || !need(g, k, H3I_DTYPE_BF16, n, "k") ||
        !need(g, v, H3I_DTYPE_BF16, n, "v") ||
        !need(g, qkv, H3I_DTYPE_BF16, rows * 3 * heads * hd, "qkv") ||
        !need(g, q_norm, H3I_DTYPE_BF16, hd, "q_norm") ||
        !need(g, k_norm, H3I_DTYPE_BF16, hd, "k_norm") ||
        (rope_half > 0 &&
         (!need(g, cosb, H3I_DTYPE_BF16, rows * 2 * rope_half, "cos") ||
          !need(g, sinb, H3I_DTYPE_BF16, rows * 2 * rope_half, "sin")))) {
        return 0;
    }
    QkvRopeParams p = {(uint32_t)rows, (uint32_t)heads, (uint32_t)hd,
                       (uint32_t)rope_half, eps, (uint32_t)grouped};
    NSArray *bufs = @[buf(q), buf(k), buf(v), buf(qkv),
                      rope_half ? buf(cosb) : buf(qkv),
                      rope_half ? buf(sinb) : buf(qkv),
                      buf(q_norm), buf(k_norm)];
    NSArray *offs = @[@0, @0, @0, @0, @0, @0, @0, @0];
    return disp2d(g, @"h3i_qkv_rope_bf16", bufs, offs, &p, sizeof p, heads, rows);
}

int h3i_gpu_video_qkv_rope_f32(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                               h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                               const h3i_gpu_tensor *cos_t, const h3i_gpu_tensor *sin_t,
                               size_t rows, size_t heads, size_t hd, size_t rope_half,
                               float eps)
{
    size_t n = rows * heads * hd;
    if (!need(g, q, H3I_DTYPE_F32, n, "q") || !need(g, k, H3I_DTYPE_F32, n, "k") ||
        !need(g, v, H3I_DTYPE_F32, n, "v") ||
        !need(g, qkv, H3I_DTYPE_F32, rows * 3 * heads * hd, "qkv") ||
        !need(g, cos_t, H3I_DTYPE_F32, rows * 2 * rope_half, "cos") ||
        !need(g, sin_t, H3I_DTYPE_F32, rows * 2 * rope_half, "sin")) {
        return 0;
    }
    QkvRopeParams p = {(uint32_t)rows, (uint32_t)heads, (uint32_t)hd,
                       (uint32_t)rope_half, eps, 1};
    return disp2d(g, @"h3i_video_qkv_rope_f32",
                  @[buf(q), buf(k), buf(v), buf(qkv), buf(cos_t), buf(sin_t)],
                  @[@0, @0, @0, @0, @0, @0], &p, sizeof p, heads, rows);
}

int h3i_gpu_vision_qkv_rope_bf16(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                                 h3i_gpu_tensor *v, const h3i_gpu_tensor *qkv,
                                 const h3i_gpu_tensor *cosb, const h3i_gpu_tensor *sinb,
                                 size_t rows, size_t heads, size_t hd, size_t rope_half)
{
    size_t n = rows * heads * hd;
    if (!need(g, q, H3I_DTYPE_BF16, n, "q") || !need(g, k, H3I_DTYPE_BF16, n, "k") ||
        !need(g, v, H3I_DTYPE_BF16, n, "v") ||
        !need(g, qkv, H3I_DTYPE_BF16, rows * 3 * heads * hd, "qkv") ||
        !need(g, cosb, H3I_DTYPE_BF16, rows * 2 * rope_half, "cos") ||
        !need(g, sinb, H3I_DTYPE_BF16, rows * 2 * rope_half, "sin")) {
        return 0;
    }
    QkvRopeParams p = {(uint32_t)rows, (uint32_t)heads, (uint32_t)hd,
                       (uint32_t)rope_half, 0.0f, 0};
    return disp2d(g, @"h3i_vision_qkv_rope_bf16",
                  @[buf(q), buf(k), buf(v), buf(qkv), buf(cosb), buf(sinb)],
                  @[@0, @0, @0, @0, @0, @0], &p, sizeof p, heads, rows);
}

int h3i_gpu_head_rms_norm_bf16(h3i_gpu *g, h3i_gpu_tensor *io, const h3i_gpu_tensor *w,
                               size_t rows, size_t heads, size_t hd, float eps)
{
    if (!need(g, io, H3I_DTYPE_BF16, rows * heads * hd, "io") ||
        !need(g, w, H3I_DTYPE_BF16, hd, "w")) {
        return 0;
    }
    HeadNormParams p = {(uint32_t)rows, (uint32_t)heads, (uint32_t)hd, eps};
    return disp2d(g, @"h3i_head_rms_norm_bf16", @[buf(io), buf(w)], @[@0, @0],
                  &p, sizeof p, heads, rows);
}

int h3i_gpu_rope_text_bf16(h3i_gpu *g, h3i_gpu_tensor *q, h3i_gpu_tensor *k,
                           const h3i_gpu_tensor *cos_t, const h3i_gpu_tensor *sin_t,
                           size_t rows, size_t qheads, size_t kvheads, size_t hd)
{
    if (!need(g, q, H3I_DTYPE_BF16, rows * qheads * hd, "q") ||
        !need(g, k, H3I_DTYPE_BF16, rows * kvheads * hd, "k") ||
        !need(g, cos_t, H3I_DTYPE_F32, rows * (hd / 2), "cos") ||
        !need(g, sin_t, H3I_DTYPE_F32, rows * (hd / 2), "sin")) {
        return 0;
    }
    RopeTextParams p = {(uint32_t)rows, (uint32_t)qheads, (uint32_t)kvheads, (uint32_t)hd};
    return disp2d(g, @"h3i_rope_text_bf16", @[buf(q), buf(k), buf(cos_t), buf(sin_t)],
                  @[@0, @0, @0, @0], &p, sizeof p, qheads, rows);
}

static int adaln_impl(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                      size_t in_off, const h3i_gpu_tensor *normw, const h3i_gpu_tensor *mod,
                      const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                      size_t slots, size_t shift_slot, size_t scale_slot, float eps)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * width, "out") ||
        !need(g, in, H3I_DTYPE_BF16, in_off + rows * width, "in") ||
        !need(g, normw, H3I_DTYPE_BF16, width, "normw") ||
        !need(g, row_map, H3I_DTYPE_U32, rows, "row_map")) {
        return 0;
    }
    /* mod table size depends on the max map value; validated by the caller
     * (modulation build). Require at least one full modulation row. */
    if (!need(g, mod, H3I_DTYPE_BF16, slots * width, "mod")) {
        return 0;
    }
    AdalnParams p = {(uint32_t)rows, (uint32_t)width, (uint32_t)slots,
                     (uint32_t)shift_slot, (uint32_t)scale_slot, eps, (uint32_t)in_off};
    return disp_rows(g, @"h3i_adaln_bf16",
                     @[buf(out), buf(in), buf(normw), buf(mod), buf(row_map)],
                     @[@0, boff(in_off, H3I_DTYPE_BF16), @0, @0, @0], &p, sizeof p, rows);
}

int h3i_gpu_adaln_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *normw, const h3i_gpu_tensor *mod,
                       const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                       size_t slots, size_t shift_slot, size_t scale_slot, float eps)
{
    return adaln_impl(g, out, in, 0, normw, mod, row_map, rows, width, slots,
                      shift_slot, scale_slot, eps);
}

int h3i_gpu_adaln_bf16_offset(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                              size_t in_off, const h3i_gpu_tensor *normw,
                              const h3i_gpu_tensor *mod, const h3i_gpu_tensor *row_map,
                              size_t rows, size_t width, size_t slots,
                              size_t shift_slot, size_t scale_slot, float eps)
{
    return adaln_impl(g, out, in, in_off, normw, mod, row_map, rows, width, slots,
                      shift_slot, scale_slot, eps);
}

int h3i_gpu_gate_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *res,
                      const h3i_gpu_tensor *branch, const h3i_gpu_tensor *mod,
                      const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                      size_t slots, size_t gate_slot)
{
    if (!need(g, out, H3I_DTYPE_BF16, rows * width, "out") ||
        !need(g, res, H3I_DTYPE_BF16, rows * width, "res") ||
        !need(g, branch, H3I_DTYPE_BF16, rows * width, "branch") ||
        !need(g, mod, H3I_DTYPE_BF16, slots * width, "mod") ||
        !need(g, row_map, H3I_DTYPE_U32, rows, "row_map")) {
        return 0;
    }
    GateParams p = {(uint32_t)rows, (uint32_t)width, (uint32_t)slots, (uint32_t)gate_slot};
    return disp2d(g, @"h3i_gate_bf16",
                  @[buf(out), buf(res), buf(branch), buf(mod), buf(row_map)],
                  @[@0, @0, @0, @0, @0], &p, sizeof p, width, rows);
}

int h3i_gpu_gate_adaln_bf16(h3i_gpu *g, h3i_gpu_tensor *out_res, h3i_gpu_tensor *out_adaln,
                            const h3i_gpu_tensor *res, const h3i_gpu_tensor *branch,
                            const h3i_gpu_tensor *normw, const h3i_gpu_tensor *mod,
                            const h3i_gpu_tensor *row_map, size_t rows, size_t width,
                            size_t slots, size_t gate_slot, size_t shift_slot,
                            size_t scale_slot, float eps)
{
    if (!need(g, out_res, H3I_DTYPE_BF16, rows * width, "out_res") ||
        !need(g, out_adaln, H3I_DTYPE_BF16, rows * width, "out_adaln") ||
        !need(g, res, H3I_DTYPE_BF16, rows * width, "res") ||
        !need(g, branch, H3I_DTYPE_BF16, rows * width, "branch") ||
        !need(g, normw, H3I_DTYPE_BF16, width, "normw") ||
        !need(g, mod, H3I_DTYPE_BF16, slots * width, "mod") ||
        !need(g, row_map, H3I_DTYPE_U32, rows, "row_map")) {
        return 0;
    }
    GateAdalnParams p = {(uint32_t)rows, (uint32_t)width, (uint32_t)slots,
                         (uint32_t)gate_slot, (uint32_t)shift_slot,
                         (uint32_t)scale_slot, eps};
    return disp_rows(g, @"h3i_gate_adaln_bf16_exact_simd",
                     @[buf(out_res), buf(out_adaln), buf(res), buf(branch),
                       buf(normw), buf(mod), buf(row_map)],
                     @[@0, @0, @0, @0, @0, @0, @0], &p, sizeof p, rows);
}

/* ---- VAE encoder ---------------------------------------------------------- */

int h3i_gpu_vae_encoder_pad_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                                size_t height, size_t width, size_t channels, size_t pad)
{
    size_t OH = height + 2 * pad, OW = width + 2 * pad;
    if (!need(g, out, H3I_DTYPE_F32, 3 * OH * OW * channels, "out") ||
        !need(g, in, H3I_DTYPE_F32, height * width * channels, "in")) {
        return 0;
    }
    PadParams p = {(uint32_t)height, (uint32_t)width, (uint32_t)channels, (uint32_t)pad};
    H3IGPU *gg = gpu_obj(g);
    NSUInteger gw = (NSUInteger)((OW + 7) / 8);
    NSUInteger gh = (NSUInteger)((OH + 3) / 4);
    return [gg encodeOp:@"h3i_vae_encoder_pad_f32"
              gridWidth:gw gridHeight:gh gridDepth:3
             groupWidth:8 groupHeight:4 groupDepth:1
                buffers:@[buf(out), buf(in)] offsets:@[@0, @0]
                 params:&p paramsSize:sizeof p];
}

int h3i_gpu_conv3d_f32(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                       const h3i_gpu_tensor *W, const h3i_gpu_tensor *b,
                       size_t n, size_t d, size_t h, size_t w, size_t c, size_t o,
                       size_t kd, size_t kh, size_t kw,
                       size_t sd, size_t sh, size_t sw)
{
    size_t od = (d - kd) / sd + 1;
    size_t oh = (h - kh) / sh + 1;
    size_t ow = (w - kw) / sw + 1;
    if (!need(g, out, H3I_DTYPE_F32, n * od * oh * ow * o, "out") ||
        !need(g, in, H3I_DTYPE_F32, n * d * h * w * c, "in") ||
        !need(g, W, H3I_DTYPE_F32, o * c * kd * kh * kw, "W") ||
        (b && !need(g, b, H3I_DTYPE_F32, o, "b"))) {
        return 0;
    }
    Conv3dParams p = {(uint32_t)n, (uint32_t)d, (uint32_t)h, (uint32_t)w,
                      (uint32_t)c, (uint32_t)o, (uint32_t)od, (uint32_t)oh,
                      (uint32_t)ow, (uint32_t)kd, (uint32_t)kh, (uint32_t)kw,
                      (uint32_t)sd, (uint32_t)sh, (uint32_t)sw, b != NULL};
    H3IGPU *gg = gpu_obj(g);
    NSArray *bufs = b ? @[buf(out), buf(in), buf(W), buf(b)]
                      : @[buf(out), buf(in), buf(W), buf(W)];
    NSUInteger gx = (NSUInteger)((ow * o + 15) / 16);
    return [gg encodeOp:@"h3i_conv3d_f32"
              gridWidth:gx gridHeight:(NSUInteger)oh gridDepth:(NSUInteger)(od * n)
             groupWidth:16 groupHeight:1 groupDepth:1
                buffers:bufs offsets:@[@0, @0, @0, @0]
                 params:&p paramsSize:sizeof p];
}

int h3i_gpu_vae_encoder_group_norm_silu_f32(h3i_gpu *g, h3i_gpu_tensor *out,
                                            const h3i_gpu_tensor *in,
                                            const h3i_gpu_tensor *w, const h3i_gpu_tensor *b,
                                            size_t n, size_t h, size_t width, size_t c,
                                            size_t groups, float eps)
{
    size_t count = n * h * width * c;
    if (!need(g, out, H3I_DTYPE_F32, count, "out") ||
        !need(g, in, H3I_DTYPE_F32, count, "in") ||
        !need(g, w, H3I_DTYPE_F32, c, "w") || !need(g, b, H3I_DTYPE_F32, c, "b") ||
        c % groups != 0) {
        return 0;
    }
    GnParams p = {(uint32_t)n, (uint32_t)h, (uint32_t)width, (uint32_t)c,
                  (uint32_t)groups, eps};
    return disp_rows(g, @"h3i_vae_encoder_group_norm_silu_f32",
                     @[buf(out), buf(in), buf(w), buf(b)],
                     @[@0, @0, @0, @0], &p, sizeof p, n * groups);
}

/* ---- dequantization dispatch (SPEC §5.3) ---------------------------------- */

int h3i_gpu_convert_raw(h3i_gpu *g, h3i_gpu_tensor *out, size_t out_off,
                        const h3i_gpu_tensor *raw, size_t raw_off, int type,
                        size_t elements)
{
    H3IGPU *gg = gpu_obj(g);
    h3i_type_info ti = h3i_type_info_get(type);
    if (ti.block_values == 0) {
        [gg setError:"convert_raw: unsupported tensor type %d", type];
        return 0;
    }
    if (elements % (size_t)ti.block_values != 0) {
        [gg setError:"convert_raw: elements not block-aligned"];
        return 0;
    }
    size_t blocks = elements / (size_t)ti.block_values;
    if (raw == NULL || tensor_obj((h3i_gpu_tensor *)raw).dtype != H3I_DTYPE_I8 ||
        raw_off + blocks * (size_t)ti.block_bytes > tensor_obj((h3i_gpu_tensor *)raw).bytes) {
        [gg setError:@"raw tensor is absent or too small"];
        return 0;
    }
    h3i_dtype odt = out ? tensor_obj(out).dtype : H3I_DTYPE_F32;
    if (out == NULL || (odt != H3I_DTYPE_BF16 && odt != H3I_DTYPE_F32) ||
        out_off + elements > tensor_obj(out).elements) {
        [gg setError:@"out tensor is absent or too small"];
        return 0;
    }
    if (blocks * (size_t)ti.chunks > UINT32_MAX) {
        [gg setError:"convert_raw: too many chunks"];
        return 0;
    }
    const char *tn = NULL;
    switch (type) {
    case H3I_TYPE_F32: tn = "f32"; break;
    case H3I_TYPE_F16: tn = "f16"; break;
    case H3I_TYPE_BF16: tn = "bf16"; break;
    case H3I_TYPE_Q8_0: tn = "q8_0"; break;
    case H3I_TYPE_Q2_K: tn = "q2_k"; break;
    case H3I_TYPE_Q3_K: tn = "q3_k"; break;
    case H3I_TYPE_Q4_K: tn = "q4_k"; break;
    case H3I_TYPE_Q5_K: tn = "q5_k"; break;
    case H3I_TYPE_Q6_K: tn = "q6_k"; break;
    }
    NSString *name = [NSString stringWithFormat:@"h3i_convert_%s_%s", tn,
                      odt == H3I_DTYPE_BF16 ? "bf16" : "f32"];
    ConvertParams p = {(uint32_t)out_off, (uint32_t)raw_off, (uint32_t)blocks};
    return disp1d(g, name, @[buf(out), buf(raw)], @[@0, @0], &p, sizeof p,
                  blocks * (size_t)ti.chunks);
}

/* ---- causal GQA attention -------------------------------------------------- */

int h3i_gpu_gqa_causal_bf16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *q,
                            const h3i_gpu_tensor *k, const h3i_gpu_tensor *v,
                            size_t seq, size_t qheads, size_t kvheads, size_t hd,
                            float scale)
{
    H3IGPU *gg = gpu_obj(g);
    if (kvheads == 0 || qheads % kvheads != 0 || hd > 128 || hd == 0) {
        [gg setError:"gqa_causal_bf16: invalid head geometry"];
        return 0;
    }
    if (!need(g, out, H3I_DTYPE_BF16, seq * qheads * hd, "out") ||
        !need(g, q, H3I_DTYPE_BF16, seq * qheads * hd, "q") ||
        !need(g, k, H3I_DTYPE_BF16, seq * kvheads * hd, "k") ||
        !need(g, v, H3I_DTYPE_BF16, seq * kvheads * hd, "v")) {
        return 0;
    }
    id<MTLComputePipelineState> pso = [gg pipeline:@"h3i_gqa_causal_bf16"];
    if (pso.maxTotalThreadsPerThreadgroup < 128 ||
        gg.device.maxThreadgroupMemoryLength < 4096 * 4 + 128 * 4) {
        [gg setError:"device cannot run the chunked causal attention kernel"];
        return 0;
    }
    GqaParams p = {(uint32_t)seq, (uint32_t)qheads, (uint32_t)kvheads, (uint32_t)hd, scale};
    return [gg encodeOp:@"h3i_gqa_causal_bf16"
              gridWidth:(NSUInteger)seq gridHeight:(NSUInteger)qheads gridDepth:1
             groupWidth:128 groupHeight:1 groupDepth:1
                buffers:@[buf(out), buf(q), buf(k), buf(v)]
                offsets:@[@0, @0, @0, @0] params:&p paramsSize:sizeof p];
}

/* ---- Neural Engine handoff -------------------------------------------------- */

int h3i_gpu_bf16_to_f16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                        size_t n, float scale)
{
    if (!need(g, in, H3I_DTYPE_BF16, n, "in") ||
        !need(g, out, H3I_DTYPE_I8, 2 * n, "out")) {
        return 0;
    }
    HandoffParams p = {(uint32_t)n, scale};
    return disp1d(g, @"h3i_bf16_to_f16", @[buf(out), buf(in)], @[@0, @0], &p, sizeof p, n);
}

int h3i_gpu_f16_to_bf16_checked(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                                size_t n, float scale, h3i_gpu_tensor *flag, uint32_t bits)
{
    if (!need(g, in, H3I_DTYPE_I8, 2 * n, "in") ||
        !need(g, out, H3I_DTYPE_BF16, n, "out") ||
        !need(g, flag, H3I_DTYPE_U32, 1, "flag")) {
        return 0;
    }
    CheckedParams p = {(uint32_t)n, scale, bits};
    return disp1d(g, @"h3i_f16_to_bf16_checked", @[buf(out), buf(in), buf(flag)],
                  @[@0, @0, @0], &p, sizeof p, n);
}

int h3i_gpu_transpose_bf16_f16(h3i_gpu *g, h3i_gpu_tensor *out, const h3i_gpu_tensor *in,
                               size_t row_off, size_t rows, size_t cols, float scale)
{
    if (!need(g, in, H3I_DTYPE_BF16, (row_off + rows) * cols, "in") ||
        !need(g, out, H3I_DTYPE_I8, 2 * rows * cols, "out")) {
        return 0;
    }
    H3IGPU *gg = gpu_obj(g);
    id<MTLComputePipelineState> pso = [gg pipeline:@"h3i_transpose_bf16_f16"];
    if (pso.maxTotalThreadsPerThreadgroup < 256) {
        [gg setError:"device cannot run the transpose kernel"];
        return 0;
    }
    TransposeParams p = {(uint32_t)rows, (uint32_t)cols, (uint32_t)row_off, scale};
    NSUInteger gx = (NSUInteger)((cols + 31) / 32);
    NSUInteger gy = (NSUInteger)((rows + 31) / 32);
    return [gg encodeOp:@"h3i_transpose_bf16_f16"
              gridWidth:gx gridHeight:gy gridDepth:1
             groupWidth:32 groupHeight:8 groupDepth:1
                buffers:@[buf(out), buf(in)] offsets:@[@0, @0]
                 params:&p paramsSize:sizeof p];
}
