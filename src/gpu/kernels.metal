/* h3-image addition kernels (SPEC §5.2-5.4): GGUF dequant convert and
 * Neural Engine handoff. Helpers below are unprefixed; the derived
 * shaders.metal uses h3i_-prefixed equivalents. */
#include <metal_stdlib>
using namespace metal;

static inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }
static inline ushort f32_to_bf16(float x) {
    uint b = as_type<uint>(x);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return ushort(b >> 16);
}

/* h3-image specific kernels (SPEC §5.2–5.4, §6.3.5).
 * Concatenated after shaders.metal; uses its helpers (bf16_to_f32,
 * f32_to_bf16, silu_f, row_tree_sum). Dequantization here is written to be
 * bit-identical with the CPU reference in weights/gguf.c: fma where the
 * spec writes fma, everything else separately-rounded F32. */

/* ---- dequantization (SPEC §5) ------------------------------------------ */

/* h3-image specific kernels (SPEC §5.2–5.4, §6.3.5).
 * Concatenated after shaders.metal; uses its helpers (bf16_to_f32,
 * f32_to_bf16, silu_f, row_tree_sum). Dequantization here is written to be
 * bit-identical with the CPU reference in weights/gguf.c: fma where the
 * spec writes fma, everything else separately-rounded F32. */

/* ---- dequantization (SPEC §5) ------------------------------------------ */

enum {
    T_F32 = 0, T_F16 = 1, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_BF16 = 30,
};

struct ConvertParams {
    uint out_off;
    uint raw_off;
    uint blocks;
};

static inline ushort le16(const device uchar *p)
{
    return ushort(uint(p[0]) | (uint(p[1]) << 8));
}

static inline uint le32(const device uchar *p)
{
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

static inline float half_f(ushort h)
{
    return float(as_type<half>(h));
}

static inline int s8(uchar b)
{
    return int(b) - (b >= 128 ? 256 : 0);
}

static inline void scale_min_k4(uint j, const device uchar *s, thread int *sc, thread int *mn)
{
    if (j < 4) {
        *sc = int(s[j] & 63);
        *mn = int(s[j + 4] & 63);
    } else {
        *sc = int(s[j + 4] & 0xF) | (int(s[j - 4] >> 6) << 4);
        *mn = int(s[j + 4] >> 4) | (int(s[j] >> 6) << 4);
    }
}

/* Decode one GPU chunk of one block into y[0..chunk_values). */
static void decode_chunk(int type, const device uchar *b, uint c, thread float *y)
{
    switch (type) {
    case T_F32:
        y[0] = as_type<float>(le32(b));
        return;
    case T_F16:
        y[0] = half_f(le16(b));
        return;
    case T_BF16:
        y[0] = bf16_to_f32(le16(b));
        return;
    case T_Q8_0: {
        float d = half_f(le16(b));
        for (uint j = 0; j < 32; j++) {
            float q = float(s8(b[2 + j]));
            y[j] = q * d;
        }
        return;
    }
    case T_Q2_K: {
        const device uchar *scales = b;
        const device uchar *qs = b + 16;
        float d = half_f(le16(b + 80));
        float dmin = half_f(le16(b + 82));
        const device uchar *q = qs + 32 * (c / 8);
        uint shift = 2 * ((c % 8) / 2);
        uint second = 16 * (c % 2);
        float dl = d * float(scales[c] & 0xF);
        float ml = dmin * float(scales[c] >> 4);
        for (uint l = 0; l < 16; l++) {
            y[l] = fma(dl, float((q[l + second] >> shift) & 3), -ml);
        }
        return;
    }
    case T_Q3_K: {
        const device uchar *hmask = b;
        const device uchar *qs = b + 32;
        uint aux0 = le32(b + 96), aux1 = le32(b + 100), tmp = le32(b + 104);
        uint k1 = 0x03030303u, k2 = 0x0F0F0F0Fu;
        uint w[4];
        w[0] = (aux0 & k2) | (((tmp >> 0) & k1) << 4);
        w[1] = (aux1 & k2) | (((tmp >> 2) & k1) << 4);
        w[2] = ((aux0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
        w[3] = ((aux1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
        float d = half_f(le16(b + 108));
        int scale = s8(uchar((w[c / 4] >> (8 * (c % 4))) & 0xFF));
        float dl = d * float(scale - 32);
        uint n = c / 8, j = (c % 8) / 2;
        uint second = 16 * (c % 2), shift = 2 * j;
        const device uchar *q = qs + 32 * n;
        uchar m = uchar(1u << (4 * n + j));
        for (uint l = 0; l < 16; l++) {
            uint i = l + second;
            int v = int((q[i] >> shift) & 3) - ((hmask[i] & m) ? 0 : 4);
            float r = dl * float(v);
            y[l] = r;
        }
        return;
    }
    case T_Q4_K: {
        float d = half_f(le16(b));
        float dmin = half_f(le16(b + 2));
        const device uchar *scales = b + 4;
        const device uchar *qs = b + 16;
        int sc, mn;
        scale_min_k4(c, scales, &sc, &mn);
        float d1 = d * float(sc);
        float m1 = dmin * float(mn);
        const device uchar *q = qs + 32 * (c / 2);
        for (uint l = 0; l < 32; l++) {
            uint nib = (c % 2) ? uint(q[l] >> 4) : uint(q[l] & 0xF);
            y[l] = fma(d1, float(nib), -m1);
        }
        return;
    }
    case T_Q5_K: {
        float d = half_f(le16(b));
        float dmin = half_f(le16(b + 2));
        const device uchar *scales = b + 4;
        const device uchar *qh = b + 16;
        const device uchar *qs = b + 48;
        int sc, mn;
        scale_min_k4(c, scales, &sc, &mn);
        float d1 = d * float(sc);
        float m1 = dmin * float(mn);
        uint j = c / 2;
        const device uchar *ql = qs + 32 * j;
        uchar u = uchar(1u << (2 * j + c % 2));
        for (uint l = 0; l < 32; l++) {
            uint low = (c % 2) ? uint(ql[l] >> 4) : uint(ql[l] & 0xF);
            uint hi = (qh[l] & u) ? 16u : 0u;
            y[l] = fma(d1, float(low + hi), -m1);
        }
        return;
    }
    case T_Q6_K: {
        const device uchar *ql = b;
        const device uchar *qh = b + 128;
        const device uchar *scales = b + 192;
        float d = half_f(le16(b + 208));
        uint n = c / 4, k = c % 4;
        const device uchar *L = ql + 64 * n;
        const device uchar *H = qh + 32 * n;
        const device uchar *S = scales + 8 * n;
        for (uint l = 0; l < 32; l++) {
            int q;
            switch (k) {
            case 0: q = int(L[l] & 0xF) | (int((H[l] >> 0) & 3) << 4); break;
            case 1: q = int(L[l + 32] & 0xF) | (int((H[l] >> 2) & 3) << 4); break;
            case 2: q = int(L[l] >> 4) | (int((H[l] >> 4) & 3) << 4); break;
            default: q = int(L[l + 32] >> 4) | (int((H[l] >> 6) & 3) << 4); break;
            }
            q -= 32;
            float a = d * float(s8(S[l / 16 + 2 * k]));
            y[l] = a * float(q);
        }
        return;
    }
    default:
        return;
    }
}

static inline float id_f(float x)
{
    return x;
}

#define DEF_CONVERT(NAME, TYPE, BB, CPB, CHUNK, OUTT, CONV)                    \
    kernel void NAME(device OUTT *out [[buffer(0)]],                           \
                     const device uchar *raw [[buffer(1)]],                   \
                     constant ConvertParams &p [[buffer(2)]],                 \
                     uint gid [[thread_position_in_grid]])                    \
    {                                                                          \
        uint block = gid / CPB;                                                \
        if (block >= p.blocks) {                                               \
            return;                                                            \
        }                                                                      \
        uint chunk = gid % CPB;                                                \
        float y[CHUNK];                                                        \
        decode_chunk(TYPE, raw + p.raw_off + block * BB, chunk, y);            \
        for (uint i = 0; i < CHUNK; i++) {                                     \
            out[p.out_off + gid * CHUNK + i] = CONV(y[i]);                     \
        }                                                                      \
    }

DEF_CONVERT(h3i_convert_f32_bf16,  T_F32,  4,   1,  1,  ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_f32_f32,   T_F32,  4,   1,  1,  float,  id_f)
DEF_CONVERT(h3i_convert_f16_bf16,  T_F16,  2,   1,  1,  ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_f16_f32,   T_F16,  2,   1,  1,  float,  id_f)
DEF_CONVERT(h3i_convert_bf16_bf16, T_BF16, 2,   1,  1,  ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_bf16_f32,  T_BF16, 2,   1,  1,  float,  id_f)
DEF_CONVERT(h3i_convert_q8_0_bf16, T_Q8_0, 34,  1,  32, ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_q8_0_f32,  T_Q8_0, 34,  1,  32, float,  id_f)
DEF_CONVERT(h3i_convert_q2_k_bf16, T_Q2_K, 84,  16, 16, ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_q2_k_f32,  T_Q2_K, 84,  16, 16, float,  id_f)
DEF_CONVERT(h3i_convert_q3_k_bf16, T_Q3_K, 110, 16, 16, ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_q3_k_f32,  T_Q3_K, 110, 16, 16, float,  id_f)
DEF_CONVERT(h3i_convert_q4_k_bf16, T_Q4_K, 144, 8,  32, ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_q4_k_f32,  T_Q4_K, 144, 8,  32, float,  id_f)
DEF_CONVERT(h3i_convert_q5_k_bf16, T_Q5_K, 176, 8,  32, ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_q5_k_f32,  T_Q5_K, 176, 8,  32, float,  id_f)
DEF_CONVERT(h3i_convert_q6_k_bf16, T_Q6_K, 210, 8,  32, ushort, f32_to_bf16)
DEF_CONVERT(h3i_convert_q6_k_f32,  T_Q6_K, 210, 8,  32, float,  id_f)

/* ---- race-free causal GQA attention (SPEC §6.3.5) ------------------------
 * One threadgroup per (query row, query head), 128 threads.
 * Phase 1: strided dots → shared scores (≤4096 keys).
 * Phase 2: max via a dedicated reduction array with full barriers — the
 *          upstream race (thread 0 overwriting the shared max before other
 *          SIMD groups read it) cannot occur here.
 * Phase 3: exp + sum, same pattern.
 * Phase 4: thread tid owns output dim tid; sequential fma over keys.
 * >4096 keys: chunked F32 online softmax (running max/sum/output rescale). */

struct GqaParams {
    uint seq;
    uint qheads;
    uint kvheads;
    uint hd;
    float scale;
};

#define GQA_THREADS 128
#define GQA_CHUNK 4096

static inline float gqa_tree_max(float v, uint tid, threadgroup float *red)
{
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = GQA_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float gqa_tree_sum(float v, uint tid, threadgroup float *red)
{
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = GQA_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

kernel void h3i_gqa_causal_bf16(device ushort *out [[buffer(0)]],
                                const device ushort *q [[buffer(1)]],
                                const device ushort *k [[buffer(2)]],
                                const device ushort *v [[buffer(3)]],
                                constant GqaParams &p [[buffer(4)]],
                                uint2 tgid [[threadgroup_position_in_grid]],
                                uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup float scores[GQA_CHUNK];
    threadgroup float red[GQA_THREADS];
    uint tid = tid2.x;
    uint t = tgid.x, h = tgid.y;
    if (t >= p.seq || h >= p.qheads || tid >= GQA_THREADS) {
        return;
    }
    uint kvh = h / (p.qheads / p.kvheads);
    uint nkeys = t + 1; /* causal */
    const device ushort *qt = q + ((size_t)t * p.qheads + h) * p.hd;
    size_t obase = ((size_t)t * p.qheads + h) * p.hd;

    if (nkeys <= GQA_CHUNK) {
        for (uint j = tid; j < nkeys; j += GQA_THREADS) {
            const device ushort *kj = k + ((size_t)j * p.kvheads + kvh) * p.hd;
            float dot = 0.0f;
            for (uint d = 0; d < p.hd; d++) {
                dot = fma(bf16_to_f32(qt[d]), bf16_to_f32(kj[d]), dot);
            }
            scores[j] = dot * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m = -INFINITY;
        for (uint j = tid; j < nkeys; j += GQA_THREADS) {
            m = max(m, scores[j]);
        }
        float mx = gqa_tree_max(m, tid, red);

        float s = 0.0f;
        for (uint j = tid; j < nkeys; j += GQA_THREADS) {
            float e = exp(scores[j] - mx);
            scores[j] = e;
            s += e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = gqa_tree_sum(s, tid, red);
        float inv = 1.0f / total;

        if (tid < p.hd) {
            float acc = 0.0f;
            for (uint j = 0; j < nkeys; j++) {
                const device ushort *vj = v + ((size_t)j * p.kvheads + kvh) * p.hd;
                float pj = scores[j] * inv;
                acc = fma(pj, bf16_to_f32(vj[tid]), acc);
            }
            out[obase + tid] = f32_to_bf16(acc);
        }
    } else {
        float m_run = -INFINITY;
        float l_run = 0.0f;
        float acc = 0.0f;
        for (uint c0 = 0; c0 < nkeys; c0 += GQA_CHUNK) {
            uint cn = min(uint(GQA_CHUNK), nkeys - c0);
            for (uint j = tid; j < cn; j += GQA_THREADS) {
                uint jj = c0 + j;
                const device ushort *kj = k + ((size_t)jj * p.kvheads + kvh) * p.hd;
                float dot = 0.0f;
                for (uint d = 0; d < p.hd; d++) {
                    dot = fma(bf16_to_f32(qt[d]), bf16_to_f32(kj[d]), dot);
                }
                scores[j] = dot * p.scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float cm = -INFINITY;
            for (uint j = tid; j < cn; j += GQA_THREADS) {
                cm = max(cm, scores[j]);
            }
            float cmx = gqa_tree_max(cm, tid, red);
            float new_m = max(m_run, cmx);
            float rescale = exp(m_run - new_m);

            float s = 0.0f;
            for (uint j = tid; j < cn; j += GQA_THREADS) {
                float e = exp(scores[j] - new_m);
                scores[j] = e;
                s += e;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float csum = gqa_tree_sum(s, tid, red);
            l_run = l_run * rescale + csum;

            if (tid < p.hd) {
                acc = acc * rescale;
                for (uint j = 0; j < cn; j++) {
                    const device ushort *vj = v + ((size_t)(c0 + j) * p.kvheads + kvh) * p.hd;
                    acc = fma(scores[j], bf16_to_f32(vj[tid]), acc);
                }
            }
            m_run = new_m;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < p.hd) {
            out[obase + tid] = f32_to_bf16(acc / l_run);
        }
    }
}

/* ---- Neural Engine handoff (SPEC §5.4) ---------------------------------- */

struct HandoffParams {
    uint count;
    float scale;
};

struct CheckedParams {
    uint count;
    float scale;
    uint bits;
};

struct TransposeParams {
    uint rows;
    uint columns;
    uint row_offset;
    float scale;
};

kernel void h3i_bf16_to_f16(device ushort *out [[buffer(0)]],
                            const device ushort *in [[buffer(1)]],
                            constant HandoffParams &p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    if (gid < p.count) {
        float v = bf16_to_f32(in[gid]) * p.scale;
        out[gid] = as_type<ushort>(half(v)); /* Metal float→half is RNE */
    }
}

kernel void h3i_f16_to_bf16_checked(device ushort *out [[buffer(0)]],
                                    const device ushort *in [[buffer(1)]],
                                    device atomic_uint *flag [[buffer(2)]],
                                    constant CheckedParams &p [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.count) {
        return;
    }
    float v = float(as_type<half>(in[gid]));
    if (!isfinite(v)) {
        atomic_fetch_or_explicit(flag, p.bits, memory_order_relaxed);
    }
    float s = v * p.scale;
    out[gid] = f32_to_bf16(s);
}

/* 32×32 tiles through threadgroup half[32][33], threadgroup 32×8. */
kernel void h3i_transpose_bf16_f16(device ushort *out [[buffer(0)]],
                                   const device ushort *in [[buffer(1)]],
                                   constant TransposeParams &p [[buffer(2)]],
                                   uint2 tgid [[threadgroup_position_in_grid]],
                                   uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup half tile[32][33];
    uint c0 = tgid.x * 32, r0 = tgid.y * 32;
    for (uint i = 0; i < 4; i++) {
        uint r = r0 + tid.y + i * 8;
        uint c = c0 + tid.x;
        if (r < p.rows && c < p.columns) {
            float v = bf16_to_f32(in[(size_t)(p.row_offset + r) * p.columns + c]) * p.scale;
            tile[tid.y + i * 8][tid.x] = half(v);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; i++) {
        uint c = c0 + tid.y + i * 8;
        uint r = r0 + tid.x;
        if (c < p.columns && r < p.rows) {
            out[(size_t)c * p.rows + r] = as_type<ushort>(tile[tid.x][tid.y + i * 8]);
        }
    }
}


struct QkvRopeParams {
    uint rows;
    uint heads;
    uint hd;
    uint rope_half;
    float eps;
    uint grouped;
};
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
