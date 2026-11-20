#!/usr/bin/env python3
"""Local cross-checks between the ObjC ops and the Metal addition kernels.

Covers the error classes that reached macOS before local gates existed:
 1. MSL reserved keywords used as identifiers in our addition kernels.
 2. Functions called in kernels.metal but defined nowhere (typo drift).
 3. Buffer-index bindings: op setBuffer/setBytes indices vs kernel
    [[buffer(n)]] parameter order, for every addition op.
 4. Param struct layout: ObjC typedef vs MSL struct (field order/types).

Usage: python3 tools/lint_bindings.py
"""
import re
import sys

METAL_RAW = open('src/gpu/kernels.metal').read()
METAL = re.sub(r'/\*.*?\*/', '', METAL_RAW, flags=re.S)
METAL = re.sub(r'//[^\n]*', '', METAL)
GPU = open('src/gpu/gpu.m').read()

RESERVED = {'half', 'sample', 'sampler', 'filter', 'texture', 'kernel',
            'device', 'constant', 'threadgroup', 'thread', 'atomic_float',
            'atomic_int', 'atomic_uint', 'precision', 'precise', 'invariant'}

fail = []

# 1. reserved identifiers (declarations like "float half =" or "uint half;")
for m in re.finditer(r'\b(?:float|uint|int|ushort|uchar|half|bool)\s+('
                     + '|'.join(RESERVED) + r')\s*[=;\[]', METAL):
    fail.append(f'reserved identifier: {m.group(1)}')

# 2. called vs defined functions in kernels.metal
defined = set(re.findall(r'(?:static inline |static )?\w[\w<>, ]*\s+(\w+)\s*\(', METAL))
defined |= set(re.findall(r'kernel void (\w+)', METAL))
builtins = {'fma', 'exp', 'log', 'sqrt', 'rsqrt', 'sin', 'cos', 'tanh', 'abs',
            'min', 'max', 'clamp', 'mix', 'floor', 'ceil', 'trunc', 'round',
            'pow', 'exp2', 'log2', 'isnan', 'isinf', 'isfinite', 'select',
            'as_type', 'float', 'uint', 'int', 'ushort', 'uchar', 'half',
            'bool', 'atomic_fetch_or_explicit', 'atomic_load_explicit',
            'atomic_store_explicit', 'threadgroup_barrier', 'simd_sum',
            'simd_max', 'simd_prefix_exclusive_sum', 'length', 'distance',
            'normalize', 'dot', 'cross', 'reflect', 'refract', 'smoothstep',
            'step', 'sign', 'fract', 'modf', 'frexp', 'ldexp', 'copysign',
            'fmin', 'fmax', 'fabs', 'memcpy', 'memset'}
calls = set(re.findall(r'\b([a-z_]\w*)\s*\(', METAL))
for c in sorted(calls - defined - builtins):
    if c in {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'buffer'}:
        continue
    fail.append(f'kernels.metal calls undefined function: {c}')

# 3. buffer bindings: addition ops vs kernels
OPS = {
    'h3i_gpu_convert_raw': ('h3i_convert_', ['out', 'raw', 'params']),
    'h3i_gpu_bf16_to_f16': ('h3i_bf16_to_f16', ['out', 'in', 'params']),
    'h3i_gpu_f16_to_bf16_checked': ('h3i_f16_to_bf16_checked', ['out', 'in', 'flag', 'params']),
    'h3i_gpu_transpose_bf16_f16': ('h3i_transpose_bf16_f16', ['out', 'in', 'params']),
    'h3i_gpu_qkv_rope_bf16': ('h3i_qkv_rope_bf16', ['q', 'k', 'v', 'qkv', 'cos', 'sin', 'q_norm', 'k_norm', 'params']),
    'h3i_gpu_gqa_causal_bf16': ('h3i_gqa_causal_bf16', ['out', 'q', 'k', 'v', 'params']),
}
for op, (kern, order) in OPS.items():
    body = GPU[GPU.index('int ' + op + '('):]
    body = body[:body.index('\n}\n')]
    idx = re.findall(r'atIndex:(\d+)', body)
    want = [str(i) for i in range(len(order))]
    if idx != want:
        fail.append(f'{op}: binding indices {idx} != {want}')
    # kernel side: buffer indices in declaration order
    if kern == 'h3i_convert_':
        kbody = METAL[METAL.index('#define DEF_CONVERT'):METAL.index('DEF_CONVERT(h3i_convert')]
        kidx = re.findall(r'\[\[buffer\((\d+)\)\]\]', kbody)
        # macro body: out 0, raw 1, params 2
        if kidx[:3] != ['0', '1', '2']:
            fail.append(f'DEF_CONVERT buffer order {kidx[:3]}')
    else:
        ks = METAL.index('kernel void ' + kern + '(')
        ke = METAL.index(')', METAL.index('[[buffer(', ks))
        kdecl = METAL[ks:METAL.index('{', ks)]
        kidx = re.findall(r'\[\[buffer\((\d+)\)\]\]', kdecl)
        if kidx != want:
            fail.append(f'{kern}: kernel buffer order {kidx} != {want}')

# 4. struct layouts ObjC vs MSL
PAIRS = [
    ('QkvRopeParams', r'typedef struct \{ uint32_t rows, heads, hd, rope_half; float eps; uint32_t grouped; \} QkvRopeParams;',
     r'struct QkvRopeParams \{\s*uint rows;\s*uint heads;\s*uint hd;\s*uint rope_half;\s*float eps;\s*uint grouped;\s*\};'),
    ('GqaParams', r'typedef struct \{ uint32_t seq, qheads, kvheads, hd; float scale; \} GqaParams;',
     r'struct GqaParams \{\s*uint seq;\s*uint qheads;\s*uint kvheads;\s*uint hd;\s*float scale;\s*\};'),
]
for name, objc_re, msl_re in PAIRS:
    if not re.search(objc_re, GPU):
        fail.append(f'{name}: ObjC typedef not found in expected layout')
    if not re.search(msl_re, METAL):
        fail.append(f'{name}: MSL struct not found in expected layout')
for name, fields in [('h3i_convert_args', 'out_off, raw_off, blocks'),
                     ('h3i_handoff_args', 'count; float scale'),
                     ('h3i_checked_args', 'count; float scale; uint32_t bits'),
                     ('h3i_transpose_args', 'rows, columns, row_offset; float scale')]:
    if f'typedef struct {{ uint32_t {fields}; }} {name};' not in GPU:
        fail.append(f'{name}: ObjC typedef layout drifted')

if fail:
    print('\n'.join(fail))
    sys.exit(1)
print('lint-bindings: OK (reserved words, helpers, buffer bindings, struct layouts)')
