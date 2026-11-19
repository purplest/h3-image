#!/bin/sh
# Make-free build + test run for macOS.
# Avoids GNU make 3.81 pattern-rule quirks entirely: every clang invocation
# has exactly one input file; the build dir is recreated from scratch so no
# stale dependency files can interfere.
#
# Usage:  sh tools/mac_check.sh            (build + run all current tests)
#         sh tools/mac_check.sh --bench    (also build + run bench_gpu)
set -eu
cd "$(dirname "$0")/.."

CC=${CC:-clang}
CFLAGS="-std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -ffp-contract=off -D_DARWIN_C_SOURCE -Isrc"
OBJCFLAGS="$CFLAGS -fobjc-arc"
LIBS="-framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework ImageIO -framework CoreGraphics -framework UniformTypeIdentifiers -framework CoreML -licucore -lm"

rm -rf build
mkdir -p build/obj build/tests build/tools

OBJS=""
for f in $(find src -name '*.c' ! -name main.c | sort); do
    o=build/obj/$(echo "$f" | sed 's|/|_|g').o
    $CC $CFLAGS -c "$f" -o "$o"
    OBJS="$OBJS $o"
done
for f in $(find src -name '*.m' | sort); do
    o=build/obj/$(echo "$f" | sed 's|/|_|g').o
    $CC $OBJCFLAGS -c "$f" -o "$o"
    OBJS="$OBJS $o"
done
ar rcs build/libh3image.a $OBJS
cat src/gpu/shaders.metal src/gpu/kernels.metal > build/h3i_shaders.metal

# ggml oracle (A1): contraction fast, matching ggml on Apple Silicon
$CC $CFLAGS -ffp-contract=fast -Wno-conversion -c tests/ggml_ref.c -o build/tests/ggml_ref.c.o

TESTS="core layout weights gguf gpu_ops handoff attention"

for t in $TESTS; do
    echo "== build test_$t"
    $CC $CFLAGS -c "tests/test_$t.c" -o "build/tests/test_$t.c.o"
    extra=""
    if [ "$t" = "gguf" ]; then
        extra=build/tests/ggml_ref.c.o
    fi
    $CC $CFLAGS -o "build/test_$t" "build/tests/test_$t.c.o" build/libh3image.a $extra $LIBS
done

fail=0
for t in $TESTS; do
    echo "== run test_$t"
    if ./build/test_$t; then
        :
    else
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED"
fi

if [ "${1:-}" = "--bench" ]; then
    $CC $CFLAGS -c tools/bench_gpu.c -o build/tools/bench_gpu.c.o
    $CC $CFLAGS -o build/bench_gpu build/tools/bench_gpu.c.o build/libh3image.a $LIBS
    ./build/bench_gpu 1024
fi

exit $fail
