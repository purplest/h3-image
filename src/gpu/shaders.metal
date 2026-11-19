/* h3-image basic GPU kernels (SPEC §3.9, §4.0, §6.3.4).
 * Concatenated before kernels.metal into build/h3i_shaders.metal; helpers
 * defined here are visible there. Compiled with safe math mode (no fast
 * math). BF16 is stored as ushort; conversions use the spec's exact RNE bit
 * trick. Safe math mode (no fast math) keeps operations separately rounded
 * where the spec requires it; fma() is used only where the spec writes fma.
 * (MSL has no "precise" qualifier — do not add one.) */
#include <metal_stdlib>
using namespace metal;

/* ---- shared helpers ---------------------------------------------------- */

static inline float bf16_to_f32(ushort h)
{
    return as_type<float>(uint(h) << 16);
}

static inline ushort f32_to_bf16(float x)
{
    uint b = as_type<uint>(x);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return ushort(b >> 16);
}

static inline float silu_f(float x)
{
    float e = exp(-x);
    float d = 1.0f + e;
    return x / d;
}

/* ---- parameter structs (mirrored in ops_h3i.m) ------------------------- */

struct LinearParams {
    uint rows;
    uint in_w;
    uint out_w;
    uint has_bias;
};

struct NormParams {
    uint rows;
    uint width;
    float eps;
};

struct ElemParams {
    uint n;
};

struct SwigluParams {
    uint rows;
    uint width;
};

struct ScaleAddParams {
    uint rows;
    uint width;
};

struct AddScaledParams {
    uint n;
    float ls;
    float rs;
};

struct QkvRopeParams {
    uint rows;
    uint heads;
    uint hd;
    uint rope_half;
    float eps;
    uint grouped;
};

struct HeadNormParams {
    uint rows;
    uint heads;
    uint hd;
    float eps;
};

struct RopeTextParams {
    uint rows;
    uint qheads;
    uint kvheads;
    uint hd;
};

struct AdalnParams {
    uint rows;
    uint width;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float eps;
    uint in_off;
};

struct GateParams {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
};

struct GateAdalnParams {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
    uint shift_slot;
    uint scale_slot;
    float eps;
};

struct GeluParams {
    uint n;
    uint approximate;
};

/* ---- linear ------------------------------------------------------------- */

/* Naive F32 linear: one thread per output, sequential fma accumulation in
 * ascending k order (the tiled variants keep the same order). */
kernel void h3i_linear_f32(device float *out [[buffer(0)]],
                           const device float *in [[buffer(1)]],
                           const device float *W [[buffer(2)]],
                           const device float *b [[buffer(3)]],
                           constant LinearParams &p [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y, o = gid.x;
    if (r >= p.rows || o >= p.out_w) {
        return;
    }
    float acc = 0.0f;
    for (uint i = 0; i < p.in_w; i++) {
        acc = fma(in[r * p.in_w + i], W[o * p.in_w + i], acc);
    }
    if (p.has_bias) {
        acc = acc + b[o];
    }
    out[r * p.out_w + o] = acc;
}

/* 16×16 tiled F32 linear (rows≥16, out=5376, in∈{32,96}): same per-output
 * ascending-k fma order as the naive kernel, so results are bit-identical. */
kernel void h3i_linear_f32_tiled(device float *out [[buffer(0)]],
                                 const device float *in [[buffer(1)]],
                                 const device float *W [[buffer(2)]],
                                 const device float *b [[buffer(3)]],
                                 constant LinearParams &p [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]],
                                 uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float xt[16][16];
    threadgroup float wt[16][16];
    uint r = gid.y, o = gid.x;
    float acc = 0.0f;
    for (uint k0 = 0; k0 < p.in_w; k0 += 16) {
        uint kk = k0 + tid.x;
        xt[tid.y][tid.x] = (r < p.rows && kk < p.in_w) ? in[r * p.in_w + kk] : 0.0f;
        wt[tid.y][tid.x] = (o < p.out_w && kk < p.in_w) ? W[o * p.in_w + kk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16 && k0 + k < p.in_w; k++) {
            acc = fma(xt[tid.y][k], wt[tid.x][k], acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r < p.rows && o < p.out_w) {
        if (p.has_bias) {
            acc = acc + b[o];
        }
        out[r * p.out_w + o] = acc;
    }
}

/* Patch embed: F32 in/W/b → BF16 out, tiled, same accumulation order. */
kernel void h3i_linear_f32_tiled_bf16(device ushort *out [[buffer(0)]],
                                      const device float *in [[buffer(1)]],
                                      const device float *W [[buffer(2)]],
                                      const device float *b [[buffer(3)]],
                                      constant LinearParams &p [[buffer(4)]],
                                      uint2 gid [[thread_position_in_grid]],
                                      uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float xt[16][16];
    threadgroup float wt[16][16];
    uint r = gid.y, o = gid.x;
    float acc = 0.0f;
    for (uint k0 = 0; k0 < p.in_w; k0 += 16) {
        uint kk = k0 + tid.x;
        xt[tid.y][tid.x] = (r < p.rows && kk < p.in_w) ? in[r * p.in_w + kk] : 0.0f;
        wt[tid.y][tid.x] = (o < p.out_w && kk < p.in_w) ? W[o * p.in_w + kk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16 && k0 + k < p.in_w; k++) {
            acc = fma(xt[tid.y][k], wt[tid.x][k], acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r < p.rows && o < p.out_w) {
        if (p.has_bias) {
            acc = acc + b[o];
        }
        out[r * p.out_w + o] = f32_to_bf16(acc);
    }
}

/* Small-shape BF16 linear: one thread per output, F32 fma accumulation,
 * one BF16 rounding at the end. */
kernel void h3i_linear_bf16(device ushort *out [[buffer(0)]],
                            const device ushort *in [[buffer(1)]],
                            const device ushort *W [[buffer(2)]],
                            const device ushort *b [[buffer(3)]],
                            constant LinearParams &p [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y, o = gid.x;
    if (r >= p.rows || o >= p.out_w) {
        return;
    }
    float acc = 0.0f;
    for (uint i = 0; i < p.in_w; i++) {
        acc = fma(bf16_to_f32(in[r * p.in_w + i]), bf16_to_f32(W[o * p.in_w + i]), acc);
    }
    if (p.has_bias) {
        acc = acc + bf16_to_f32(b[o]);
    }
    out[r * p.out_w + o] = f32_to_bf16(acc);
}

/* ---- elementwise -------------------------------------------------------- */

kernel void h3i_silu_f32(device float *out [[buffer(0)]],
                         const device float *in [[buffer(1)]],
                         constant ElemParams &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        out[gid] = silu_f(in[gid]);
    }
}

kernel void h3i_cast_f32_to_bf16(device ushort *out [[buffer(0)]],
                                 const device float *in [[buffer(1)]],
                                 constant ElemParams &p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        out[gid] = f32_to_bf16(in[gid]);
    }
}

kernel void h3i_cast_bf16_to_f32(device float *out [[buffer(0)]],
                                 const device ushort *in [[buffer(1)]],
                                 constant ElemParams &p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        out[gid] = bf16_to_f32(in[gid]);
    }
}

kernel void h3i_add_bf16(device ushort *out [[buffer(0)]],
                         const device ushort *l [[buffer(1)]],
                         const device ushort *r [[buffer(2)]],
                         constant ElemParams &p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        float a = bf16_to_f32(l[gid]);
        float b = bf16_to_f32(r[gid]);
        out[gid] = f32_to_bf16(a + b);
    }
}

kernel void h3i_silu_mul_bf16(device ushort *out [[buffer(0)]],
                              const device ushort *gate [[buffer(1)]],
                              const device ushort *up [[buffer(2)]],
                              constant ElemParams &p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        float g = bf16_to_f32(gate[gid]);
        float u = bf16_to_f32(up[gid]);
        float s = silu_f(g);
        out[gid] = f32_to_bf16(s * u);
    }
}

kernel void h3i_add_scaled_f32(device float *out [[buffer(0)]],
                               const device float *l [[buffer(1)]],
                               const device float *r [[buffer(2)]],
                               constant AddScaledParams &p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) {
        float a = p.ls * l[gid];
        float b = p.rs * r[gid];
        out[gid] = a + b;
    }
}

kernel void h3i_gelu_bf16(device ushort *out [[buffer(0)]],
                          const device ushort *in [[buffer(1)]],
                          constant GeluParams &p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    float x = bf16_to_f32(in[gid]);
    float y;
    if (p.approximate) {
        /* GELU-tanh (SPEC §4.0) */
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        if (inner <= -10.0f) {
            y = 0.0f;
        } else if (inner >= 10.0f) {
            y = x;
        } else {
            float t = tanh(inner);
            y = 0.5f * x * (1.0f + t);
        }
    } else {
        /* GELU-erf with Abramowitz–Stegun 7.1.26 (SPEC §4.0) */
        if (x <= -10.0f) {
            y = 0.0f;
        } else if (x >= 10.0f) {
            y = x;
        } else {
            float ax = abs(x);
            float tt = 1.0f / (1.0f + 0.3275911f * ax);
            float poly = ((((1.061405429f * tt - 1.453152027f) * tt
                                    + 1.421413741f) * tt - 0.284496736f) * tt
                                  + 0.254829592f) * tt;
            float e = exp(-ax * ax);
            float erf = 1.0f - poly * e;
            if (x < 0.0f) {
                erf = -erf;
            }
            y = 0.5f * x * (1.0f + erf);
        }
    }
    out[gid] = f32_to_bf16(y);
}

/* ---- reductions: one threadgroup per row, 256-way tree ------------------ */

#define ROW_THREADS 256

static inline float row_tree_sum(float partial, uint tid, threadgroup float *scratch)
{
    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ROW_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

kernel void h3i_rms_norm_f32(device float *out [[buffer(0)]],
                             const device float *in [[buffer(1)]],
                             const device float *w [[buffer(2)]],
                             constant NormParams &p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[ROW_THREADS];
    if (row >= p.rows) {
        return;
    }
    const device float *x = in + row * p.width;
    float ss = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        ss = fma(x[i], x[i], ss);
    }
    ss = row_tree_sum(ss, tid, scratch);
    float inv = rsqrt(ss / (float)p.width + p.eps);
    device float *y = out + row * p.width;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float a = x[i] * inv;
        y[i] = a * w[i];
    }
}

kernel void h3i_rms_norm_bf16(device ushort *out [[buffer(0)]],
                              const device ushort *in [[buffer(1)]],
                              const device ushort *w [[buffer(2)]],
                              constant NormParams &p [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[ROW_THREADS];
    if (row >= p.rows) {
        return;
    }
    const device ushort *x = in + row * p.width;
    float ss = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float v = bf16_to_f32(x[i]);
        ss = fma(v, v, ss);
    }
    ss = row_tree_sum(ss, tid, scratch);
    float inv = rsqrt(ss / (float)p.width + p.eps);
    device ushort *y = out + row * p.width;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float a = bf16_to_f32(x[i]) * inv;
        float b = a * bf16_to_f32(w[i]);
        y[i] = f32_to_bf16(b);
    }
}

kernel void h3i_layer_norm_f32(device float *out [[buffer(0)]],
                               const device float *in [[buffer(1)]],
                               const device float *w [[buffer(2)]],
                               const device float *b [[buffer(3)]],
                               constant NormParams &p [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[ROW_THREADS];
    if (row >= p.rows) {
        return;
    }
    const device float *x = in + row * p.width;
    float s = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        s += x[i];
    }
    s = row_tree_sum(s, tid, scratch);
    float mean = s / (float)p.width;
    float v = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float d = x[i] - mean;
        v = fma(d, d, v);
    }
    v = row_tree_sum(v, tid, scratch);
    float inv = rsqrt(v / (float)p.width + p.eps);
    device float *y = out + row * p.width;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float d = x[i] - mean;
        float n = d * inv;
        float a = n * w[i];
        y[i] = a + b[i];
    }
}

kernel void h3i_layer_norm_bf16(device ushort *out [[buffer(0)]],
                                const device ushort *in [[buffer(1)]],
                                const device ushort *w [[buffer(2)]],
                                const device ushort *b [[buffer(3)]],
                                constant NormParams &p [[buffer(4)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[ROW_THREADS];
    if (row >= p.rows) {
        return;
    }
    const device ushort *x = in + row * p.width;
    float s = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        s += bf16_to_f32(x[i]);
    }
    s = row_tree_sum(s, tid, scratch);
    float mean = s / (float)p.width;
    float v = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float d = bf16_to_f32(x[i]) - mean;
        v = fma(d, d, v);
    }
    v = row_tree_sum(v, tid, scratch);
    float inv = rsqrt(v / (float)p.width + p.eps);
    device ushort *y = out + row * p.width;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float d = bf16_to_f32(x[i]) - mean;
        float n = d * inv;
        /* BF16 LayerNorm uses fma(norm, w, b) (SPEC §4.0). */
        float r = fma(n, bf16_to_f32(w[i]), bf16_to_f32(b[i]));
        y[i] = f32_to_bf16(r);
    }
}

/* ---- SwiGLU / scale-add -------------------------------------------------- */

kernel void h3i_swiglu_f32(device float *out [[buffer(0)]],
                           const device float *fused [[buffer(1)]],
                           constant SwigluParams &p [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y, c = gid.x;
    if (r >= p.rows || c >= p.width) {
        return;
    }
    float g = fused[r * (2 * p.width) + c];
    float u = fused[r * (2 * p.width) + p.width + c];
    float s = silu_f(g);
    out[r * p.width + c] = s * u;
}

kernel void h3i_swiglu_bf16(device ushort *out [[buffer(0)]],
                            const device ushort *fused [[buffer(1)]],
                            constant SwigluParams &p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y, c = gid.x;
    if (r >= p.rows || c >= p.width) {
        return;
    }
    float g = bf16_to_f32(fused[r * (2 * p.width) + c]);
    float u = bf16_to_f32(fused[r * (2 * p.width) + p.width + c]);
    float s = silu_f(g);
    out[r * p.width + c] = f32_to_bf16(s * u);
}

kernel void h3i_scale_add_f32(device float *out [[buffer(0)]],
                              const device float *res [[buffer(1)]],
                              const device float *branch [[buffer(2)]],
                              const device float *scale [[buffer(3)]],
                              constant ScaleAddParams &p [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y, c = gid.x;
    if (r >= p.rows || c >= p.width) {
        return;
    }
    size_t i = (size_t)r * p.width + c;
    float a = scale[c] * branch[i];
    out[i] = res[i] + a;
}

/* ---- QKV split + per-head norm + RoPE ------------------------------------ */

/* DiT: one thread per (row, head). ss = sequential fma over hd; q̂ kept F32;
 * rotate-half pairs (d, d+rope_half); dims beyond 2*rope_half unrotated.
 * grouped=1: qkv row = [head][q|k|v][hd]; grouped=0: [q|k|v][head][hd]. */
kernel void h3i_qkv_rope_bf16(device ushort *q [[buffer(0)]],
                              device ushort *k [[buffer(1)]],
                              device ushort *v [[buffer(2)]],
                              const device ushort *qkv [[buffer(3)]],
                              const device ushort *cosb [[buffer(4)]],
                              const device ushort *sinb [[buffer(5)]],
                              const device ushort *q_norm [[buffer(6)]],
                              const device ushort *k_norm [[buffer(7)]],
                              constant QkvRopeParams &p [[buffer(8)]],
                              uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y, h = gid.x;
    if (row >= p.rows || h >= p.heads) {
        return;
    }
    uint inner = p.heads * p.hd;
    const device ushort *src = qkv + (size_t)row * (3 * inner);
    uint base;
    if (p.grouped) {
        base = h * (3 * p.hd);
    } else {
        base = h * p.hd;
    }
    uint qoff = base;
    uint koff = p.grouped ? base + p.hd : inner + base;
    uint voff = p.grouped ? base + 2 * p.hd : 2 * inner + base;

    float qv[128], kv[128];
    float ssq = 0.0f, ssk = 0.0f;
    for (uint d = 0; d < p.hd; d++) {
        qv[d] = bf16_to_f32(src[qoff + d]);
        kv[d] = bf16_to_f32(src[koff + d]);
        ssq = fma(qv[d], qv[d], ssq);
        ssk = fma(kv[d], kv[d], ssk);
    }
    float invq = rsqrt(ssq / (float)p.hd + p.eps);
    float invk = rsqrt(ssk / (float)p.hd + p.eps);
    for (uint d = 0; d < p.hd; d++) {
        qv[d] = qv[d] * invq * bf16_to_f32(q_norm[d]);
        kv[d] = kv[d] * invk * bf16_to_f32(k_norm[d]);
    }
    size_t obase = ((size_t)row * p.heads + h) * p.hd;
    uint rh = p.rope_half;
    for (uint d = 0; d < p.hd; d++) {
        float qo = qv[d], ko = kv[d];
        if (rh > 0 && d < rh) {
            float c = bf16_to_f32(cosb[(size_t)row * (2 * rh) + d]);
            float s = bf16_to_f32(sinb[(size_t)row * (2 * rh) + d]);
            float qa = qv[d] * c;
            float qb = qv[d + rh] * s;
            float ka = kv[d] * c;
            float kb = kv[d + rh] * s;
            qo = qa - qb;
            ko = ka - kb;
            float qc = qv[d + rh] * c;
            float qd = qv[d] * s;
            float kc = kv[d + rh] * c;
            float kd = kv[d] * s;
            qv[d + rh] = qc + qd; /* written below via pair handling */
            kv[d + rh] = kc + kd;
            /* store the rotated low dim now; high dim stored when d loop
             * reaches it? No — high dims must not be re-rotated. Handle by
             * writing both here and skipping d >= rh && d < 2rh below. */
            q[obase + d] = f32_to_bf16(qo);
            k[obase + d] = f32_to_bf16(ko);
            q[obase + d + rh] = f32_to_bf16(qv[d + rh]);
            k[obase + d + rh] = f32_to_bf16(kv[d + rh]);
            v[obase + d] = src[voff + d];
            v[obase + d + rh] = src[voff + d + rh];
            continue;
        }
        if (rh > 0 && d < 2 * rh) {
            continue; /* already written as the pair partner */
        }
        q[obase + d] = f32_to_bf16(qo);
        k[obase + d] = f32_to_bf16(ko);
        v[obase + d] = src[voff + d];
    }
}

/* Cooperative variant: one threadgroup per row, one thread per head.
 * Identical per-head math, identical results. */
kernel void h3i_qkv_rope_bf16_coop(device ushort *q [[buffer(0)]],
                                   device ushort *k [[buffer(1)]],
                                   device ushort *v [[buffer(2)]],
                                   const device ushort *qkv [[buffer(3)]],
                                   const device ushort *cosb [[buffer(4)]],
                                   const device ushort *sinb [[buffer(5)]],
                                   const device ushort *q_norm [[buffer(6)]],
                                   const device ushort *k_norm [[buffer(7)]],
                                   constant QkvRopeParams &p [[buffer(8)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint h [[thread_position_in_threadgroup]])
{
    if (row >= p.rows || h >= p.heads) {
        return;
    }
    uint inner = p.heads * p.hd;
    const device ushort *src = qkv + (size_t)row * (3 * inner);
    uint base = p.grouped ? h * (3 * p.hd) : h * p.hd;
    uint qoff = base;
    uint koff = p.grouped ? base + p.hd : inner + base;
    uint voff = p.grouped ? base + 2 * p.hd : 2 * inner + base;

    float qv[128], kv[128];
    float ssq = 0.0f, ssk = 0.0f;
    for (uint d = 0; d < p.hd; d++) {
        qv[d] = bf16_to_f32(src[qoff + d]);
        kv[d] = bf16_to_f32(src[koff + d]);
        ssq = fma(qv[d], qv[d], ssq);
        ssk = fma(kv[d], kv[d], ssk);
    }
    float invq = rsqrt(ssq / (float)p.hd + p.eps);
    float invk = rsqrt(ssk / (float)p.hd + p.eps);
    for (uint d = 0; d < p.hd; d++) {
        qv[d] = qv[d] * invq * bf16_to_f32(q_norm[d]);
        kv[d] = kv[d] * invk * bf16_to_f32(k_norm[d]);
    }
    size_t obase = ((size_t)row * p.heads + h) * p.hd;
    uint rh = p.rope_half;
    for (uint d = 0; d < p.hd; d++) {
        if (rh > 0 && d < rh) {
            float c = bf16_to_f32(cosb[(size_t)row * (2 * rh) + d]);
            float s = bf16_to_f32(sinb[(size_t)row * (2 * rh) + d]);
            float qlo = qv[d] * c - qv[d + rh] * s;
            float qhi = qv[d + rh] * c + qv[d] * s;
            float klo = kv[d] * c - kv[d + rh] * s;
            float khi = kv[d + rh] * c + kv[d] * s;
            q[obase + d] = f32_to_bf16(qlo);
            q[obase + d + rh] = f32_to_bf16(qhi);
            k[obase + d] = f32_to_bf16(klo);
            k[obase + d + rh] = f32_to_bf16(khi);
            v[obase + d] = src[voff + d];
            v[obase + d + rh] = src[voff + d + rh];
        } else if (rh > 0 && d < 2 * rh) {
            continue;
        } else {
            q[obase + d] = f32_to_bf16(qv[d]);
            k[obase + d] = f32_to_bf16(kv[d]);
            v[obase + d] = src[voff + d];
        }
    }
}

/* VAE decoder: grouped [head][q|k|v][64], per-head RMS with NO weight,
 * RoPE on dims 0..31 in (d, d+16) pairs, table index d mod 16, F32 tables,
 * dims 32..63 unrotated. All F32. */
kernel void h3i_video_qkv_rope_f32(device float *q [[buffer(0)]],
                                   device float *k [[buffer(1)]],
                                   device float *v [[buffer(2)]],
                                   const device float *qkv [[buffer(3)]],
                                   const device float *cos_t [[buffer(4)]],
                                   const device float *sin_t [[buffer(5)]],
                                   constant QkvRopeParams &p [[buffer(6)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y, h = gid.x;
    if (row >= p.rows || h >= p.heads) {
        return;
    }
    const device float *src = qkv + ((size_t)row * p.heads + h) * (3 * p.hd);
    float qv[64], kv[64];
    float ssq = 0.0f, ssk = 0.0f;
    for (uint d = 0; d < p.hd; d++) {
        qv[d] = src[d];
        kv[d] = src[p.hd + d];
        ssq = fma(qv[d], qv[d], ssq);
        ssk = fma(kv[d], kv[d], ssk);
    }
    float invq = rsqrt(ssq / (float)p.hd + p.eps);
    float invk = rsqrt(ssk / (float)p.hd + p.eps);
    for (uint d = 0; d < p.hd; d++) {
        qv[d] = qv[d] * invq;
        kv[d] = kv[d] * invk;
    }
    size_t obase = ((size_t)row * p.heads + h) * p.hd;
    uint rh = p.rope_half; /* 16 */
    for (uint d = 0; d < p.hd; d++) {
        float qo = qv[d], ko = kv[d];
        if (d < 2 * rh) {
            uint ti = d % rh;
            float c = cos_t[(size_t)row * (2 * rh) + ti];
            float s = sin_t[(size_t)row * (2 * rh) + ti];
            if (d < rh) {
                float a = qv[d] * c;
                float b = qv[d + rh] * s;
                qo = a - b;
                float ka = kv[d] * c;
                float kb = kv[d + rh] * s;
                ko = ka - kb;
            } else {
                float a = qv[d] * c;
                float b = qv[d - rh] * s;
                qo = a + b;
                float ka = kv[d] * c;
                float kb = kv[d - rh] * s;
                ko = ka + kb;
            }
        }
        q[obase + d] = qo;
        k[obase + d] = ko;
        v[obase + d] = src[2 * p.hd + d];
    }
}

/* Vision tower: conventional [q|k|v][head][72], NO q/k norm, rotate-half
 * over all 72 dims with half=36, table index d mod 36. */
kernel void h3i_vision_qkv_rope_bf16(device ushort *q [[buffer(0)]],
                                     device ushort *k [[buffer(1)]],
                                     device ushort *v [[buffer(2)]],
                                     const device ushort *qkv [[buffer(3)]],
                                     const device ushort *cosb [[buffer(4)]],
                                     const device ushort *sinb [[buffer(5)]],
                                     constant QkvRopeParams &p [[buffer(6)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y, h = gid.x;
    if (row >= p.rows || h >= p.heads) {
        return;
    }
    uint inner = p.heads * p.hd;
    const device ushort *src = qkv + (size_t)row * (3 * inner);
    size_t obase = ((size_t)row * p.heads + h) * p.hd;
    uint rh = p.rope_half; /* 36 */
    for (uint d = 0; d < p.hd; d++) {
        float qd = bf16_to_f32(src[h * p.hd + d]);
        float kd = bf16_to_f32(src[inner + h * p.hd + d]);
        uint ti = d % rh;
        float c = bf16_to_f32(cosb[(size_t)row * (2 * rh) + ti]);
        float s = bf16_to_f32(sinb[(size_t)row * (2 * rh) + ti]);
        float qo, ko;
        if (d < rh) {
            float qp = bf16_to_f32(src[h * p.hd + d + rh]);
            float kp = bf16_to_f32(src[inner + h * p.hd + d + rh]);
            qo = qd * c - qp * s;
            ko = kd * c - kp * s;
        } else {
            float qp = bf16_to_f32(src[h * p.hd + d - rh]);
            float kp = bf16_to_f32(src[inner + h * p.hd + d - rh]);
            qo = qd * c + qp * s;
            ko = kd * c + kp * s;
        }
        q[obase + d] = f32_to_bf16(qo);
        k[obase + d] = f32_to_bf16(ko);
        v[obase + d] = src[2 * inner + h * p.hd + d];
    }
}

/* In-place per-head RMS norm with weight (text encoder q/k). */
kernel void h3i_head_rms_norm_bf16(device ushort *io [[buffer(0)]],
                                   const device ushort *w [[buffer(1)]],
                                   constant HeadNormParams &p [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y, h = gid.x;
    if (row >= p.rows || h >= p.heads) {
        return;
    }
    device ushort *x = io + ((size_t)row * p.heads + h) * p.hd;
    float ss = 0.0f;
    for (uint d = 0; d < p.hd; d++) {
        float v = bf16_to_f32(x[d]);
        ss = fma(v, v, ss);
    }
    float inv = rsqrt(ss / (float)p.hd + p.eps);
    for (uint d = 0; d < p.hd; d++) {
        float a = bf16_to_f32(x[d]) * inv;
        float b = a * bf16_to_f32(w[d]);
        x[d] = f32_to_bf16(b);
    }
}

/* In-place rotate-half RoPE over all hd dims, pairs (d, d+hd/2), GQA.
 * cos/sin are F32 tensors holding BF16-rounded values, [rows][hd/2]. */
kernel void h3i_rope_text_bf16(device ushort *q [[buffer(0)]],
                               device ushort *k [[buffer(1)]],
                               const device float *cos_t [[buffer(2)]],
                               const device float *sin_t [[buffer(3)]],
                               constant RopeTextParams &p [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y, h = gid.x;
    if (row >= p.rows || h >= p.qheads) {
        return;
    }
    uint half = p.hd / 2;
    const device float *c = cos_t + (size_t)row * half;
    const device float *s = sin_t + (size_t)row * half;
    device ushort *qq = q + ((size_t)row * p.qheads + h) * p.hd;
    for (uint d = 0; d < half; d++) {
        float a = bf16_to_f32(qq[d]);
        float b = bf16_to_f32(qq[d + half]);
        float a2 = a * c[d] - b * s[d];
        float b2 = b * c[d] + a * s[d];
        qq[d] = f32_to_bf16(a2);
        qq[d + half] = f32_to_bf16(b2);
    }
    if (h < p.kvheads) {
        device ushort *kk = k + ((size_t)row * p.kvheads + h) * p.hd;
        for (uint d = 0; d < half; d++) {
            float a = bf16_to_f32(kk[d]);
            float b = bf16_to_f32(kk[d + half]);
            float a2 = a * c[d] - b * s[d];
            float b2 = b * c[d] + a * s[d];
            kk[d] = f32_to_bf16(a2);
            kk[d + half] = f32_to_bf16(b2);
        }
    }
}

/* ---- AdaLN / gate --------------------------------------------------------- */

kernel void h3i_adaln_bf16(device ushort *out [[buffer(0)]],
                           const device ushort *in [[buffer(1)]],
                           const device ushort *normw [[buffer(2)]],
                           const device ushort *mod [[buffer(3)]],
                           const device uint *row_map [[buffer(4)]],
                           constant AdalnParams &p [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[ROW_THREADS];
    if (row >= p.rows) {
        return;
    }
    const device ushort *x = in + (size_t)p.in_off + (size_t)row * p.width;
    float ss = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float v = bf16_to_f32(x[i]);
        ss = fma(v, v, ss);
    }
    ss = row_tree_sum(ss, tid, scratch);
    float inv = rsqrt(ss / (float)p.width + p.eps);
    size_t mbase = (size_t)row_map[row] * p.slots * p.width;
    const device ushort *shift = mod + mbase + p.shift_slot * p.width;
    const device ushort *scale = mod + mbase + p.scale_slot * p.width;
    device ushort *y = out + (size_t)row * p.width;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float n = (bf16_to_f32(x[i]) * inv) * bf16_to_f32(normw[i]);
        float sc = 1.0f + bf16_to_f32(scale[i]);
        float a = n * sc;
        y[i] = f32_to_bf16(a + bf16_to_f32(shift[i]));
    }
}

kernel void h3i_gate_bf16(device ushort *out [[buffer(0)]],
                          const device ushort *res [[buffer(1)]],
                          const device ushort *branch [[buffer(2)]],
                          const device ushort *mod [[buffer(3)]],
                          const device uint *row_map [[buffer(4)]],
                          constant GateParams &p [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y, c = gid.x;
    if (row >= p.rows || c >= p.width) {
        return;
    }
    size_t i = (size_t)row * p.width + c;
    size_t gbase = (size_t)row_map[row] * p.slots * p.width + p.gate_slot * p.width;
    float g = bf16_to_f32(mod[gbase + c]);
    float a = bf16_to_f32(branch[i]) * g;
    out[i] = f32_to_bf16(bf16_to_f32(res[i]) + a);
}

/* Fused gate then AdaLN of the rounded gated result (SPEC §3.10: norm2
 * normalizes the rounded hidden). Writes both outputs. */
kernel void h3i_gate_adaln_bf16_exact_simd(device ushort *out_res [[buffer(0)]],
                                           device ushort *out_adaln [[buffer(1)]],
                                           const device ushort *res [[buffer(2)]],
                                           const device ushort *branch [[buffer(3)]],
                                           const device ushort *normw [[buffer(4)]],
                                           const device ushort *mod [[buffer(5)]],
                                           const device uint *row_map [[buffer(6)]],
                                           constant GateAdalnParams &p [[buffer(7)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[ROW_THREADS];
    if (row >= p.rows) {
        return;
    }
    size_t base = (size_t)row * p.width;
    size_t mbase = (size_t)row_map[row] * p.slots * p.width;
    const device ushort *gate = mod + mbase + p.gate_slot * p.width;
    /* phase 1: gated residual, BF16-rounded */
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float g = bf16_to_f32(gate[i]);
        float a = bf16_to_f32(branch[base + i]) * g;
        out_res[base + i] = f32_to_bf16(bf16_to_f32(res[base + i]) + a);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* phase 2: AdaLN over the rounded values */
    float ss = 0.0f;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float v = bf16_to_f32(out_res[base + i]);
        ss = fma(v, v, ss);
    }
    ss = row_tree_sum(ss, tid, scratch);
    float inv = rsqrt(ss / (float)p.width + p.eps);
    const device ushort *shift = mod + mbase + p.shift_slot * p.width;
    const device ushort *scale = mod + mbase + p.scale_slot * p.width;
    for (uint i = tid; i < p.width; i += ROW_THREADS) {
        float n = (bf16_to_f32(out_res[base + i]) * inv) * bf16_to_f32(normw[i]);
        float sc = 1.0f + bf16_to_f32(scale[i]);
        float a = n * sc;
        out_adaln[base + i] = f32_to_bf16(a + bf16_to_f32(shift[i]));
    }
}
