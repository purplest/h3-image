#!/usr/bin/env python3
"""Local Metal Shading Language lint (no Metal compiler needed on Linux).

Checks the rule that bit us on macOS: kernel stage-input attributes
(threadgroup_position_in_grid, thread_position_in_threadgroup, ...) must be
all scalar or all vectors of the same size; per-component attributes
(.x/.y/.z) count as scalar. Buffer/constant/threadgroup resource bindings
are not stage inputs and are ignored.

Preprocessor-aware: regions guarded by H3I_METAL_HAS_TENSOR (M5 tensor-op
kernels, disabled on non-M5 machines) are skipped, matching what the Metal
compiler actually sees on a typical Mac.

Usage: python3 tools/lint_metal.py [files...]   (default: src/gpu/*.metal)
"""
import glob
import re
import sys

STAGE = (r'(?:threadgroup_position_in_grid|thread_position_in_grid|'
         r'thread_position_in_threadgroup|threads_per_threadgroup|'
         r'thread_index_in_threadgroup|threadgroup_position_in_threadgroup)')
DECL = re.compile(r'\b(uint|ushort|uchar|int|float|half)(\d?)\s+\w+\s*\[\['
                  + STAGE + r'(\.[xyzw])?\]\]')
KERNEL = re.compile(r'kernel void (\w+)\((.*?)\)\s*\{', re.S)


def active_source(text):
    """Drop lines inside #ifdef H3I_METAL_HAS_TENSOR / #if defined(...) blocks."""
    out = []
    stack = []  # each entry: bool active
    for line in text.split('\n'):
        st = line.strip()
        if st.startswith('#ifdef') or st.startswith('#ifndef'):
            macro = st.split()[1] if len(st.split()) > 1 else ''
            defined = macro == 'H3I_METAL_HAS_TENSOR'
            stack.append(not defined if st.startswith('#ifdef') else defined)
            continue
        if st.startswith('#if'):
            stack.append('H3I_METAL_HAS_TENSOR' not in st)
            continue
        if st.startswith('#else'):
            if stack:
                stack[-1] = not stack[-1]
            continue
        if st.startswith('#endif'):
            if stack:
                stack.pop()
            continue
        if all(stack):
            out.append(line)
    return '\n'.join(out)


def main(paths):
    bad = []
    for p in paths:
        src = active_source(open(p).read())
        for m in KERNEL.finditer(src):
            name, params = m.group(1), m.group(2)
            sizes = set()
            for d in DECL.finditer(params):
                if d.group(3):
                    bad.append(f'{p}: {name}: component stage attribute '
                               f'[[...{d.group(3)}]] is not valid MSL; use vectors')
                    continue
                vec = int(d.group(2)) if d.group(2) else 1
                sizes.add(vec)
            if len(sizes) > 1:
                bad.append(f'{p}: {name}: mixed stage-input sizes {sorted(sizes)}')
    if bad:
        print('\n'.join(bad))
        return 1
    print('metal-lint: OK (stage inputs uniform in all compiled kernels)')
    return 0


if __name__ == '__main__':
    args = sys.argv[1:] or sorted(glob.glob('src/gpu/*.metal'))
    sys.exit(main(args))
