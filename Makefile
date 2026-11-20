# h3-image — MiniMax-H3-Image inference for Apple Silicon (rebuilt from h3.c).
# Behavior/numerics: SPEC.md (local). Build contract: SPEC part 6 §4.
#
# macOS target:  make            → libh3image.a + shaders (h3-image once main.c lands)
#                make test       → every build/test_* then tests/test_cli.sh
#                make test-quick → the quick subset that exists
#                make check      → build + run every test, keep going, summarize
#                make tools      → build/<name> for each tools/*.c
# Linux target:  make core-tests → portable C core + CPU unit tests (A1/A2 host half)
#                make linux-check→ full-warning syntax gate + host-math smoke
#                make objc-syntax / test-syntax / metal-lint → static gates
#
# CC can be overridden (e.g. make core-tests CC=gcc).

CC = clang
AR = ar

# FP contraction OFF everywhere: the spec's separately-rounded F32 ops must
# not fuse; explicit fma() calls still emit FMA (SPEC §1.4, §4.0, §5).
CFLAGS = -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
         -Wno-sign-conversion -ffp-contract=off -D_DARWIN_C_SOURCE -Isrc
OBJCFLAGS = $(CFLAGS) -fobjc-arc
FRAMEWORKS = -framework Foundation -framework Metal \
             -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
             -framework Accelerate -framework ImageIO -framework CoreGraphics \
             -framework UniformTypeIdentifiers -framework CoreML
LDLIBS = $(FRAMEWORKS) -licucore -lm

# --- macOS full build -------------------------------------------------------

C_SRCS := $(shell find src -name '*.c' 2>/dev/null | sort)
M_SRCS := $(shell find src -name '*.m' 2>/dev/null | sort)
LIB_C_SRCS := $(filter-out src/app/main.c,$(C_SRCS))
LIB_OBJS := $(patsubst %,build/%.o,$(LIB_C_SRCS) $(M_SRCS))
MAIN_OBJ := build/src/app/main.c.o
SHADERS := build/h3i_shaders.metal

TEST_SRCS := $(sort $(wildcard tests/test_*.c))
TEST_BINS := $(patsubst tests/%.c,build/%,$(TEST_SRCS))
TOOL_SRCS := $(sort $(wildcard tools/*.c))
TOOL_BINS := $(patsubst tools/%.c,build/%,$(TOOL_SRCS))

QUICK_ALL := test_layout test_weights test_stream test_gpu_ops test_handoff test_ane
QUICK := $(foreach t,$(QUICK_ALL),$(if $(wildcard tests/$(t).c),$(t),))

.PHONY: all lib test test-quick check-images tools clean core-tests linux-check \
        objc-syntax test-syntax tests-build check bench metal-lint
.DEFAULT_GOAL := all

# Until src/app/main.c lands (Phase 8), `all` builds the library and shaders.
ifeq ($(wildcard src/app/main.c),)
all: build/libh3image.a $(SHADERS)
else
all: h3-image $(SHADERS)
endif

lib: build/libh3image.a $(SHADERS)

h3-image: $(MAIN_OBJ) build/libh3image.a
	$(CC) $(CFLAGS) -o $@ $(MAIN_OBJ) build/libh3image.a $(LDLIBS)

build/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.m.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(OBJCFLAGS) -c $< -o $@

# Archive is deleted and recreated each time (SPEC §6.4).
build/libh3image.a: $(LIB_OBJS)
	@mkdir -p build
	rm -f $@
	$(AR) rcs $@ $(LIB_OBJS)

$(SHADERS): src/gpu/shaders.metal src/gpu/kernels.metal
	@mkdir -p build
	cat src/gpu/shaders.metal src/gpu/kernels.metal > $@

# Two-step test build: compile to an object, then link objects + archive
# (make 3.81 safe). The ggml oracle (A1) is compiled with contraction FAST,
# same as the Linux core build: `dl*q - ml` must fuse exactly as ggml does.
build/tests/%.c.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/tests/ggml_ref.c.o: tests/ggml_ref.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -ffp-contract=fast -Wno-conversion -c $< -o $@

# Link recipes name their inputs explicitly (never $^): stale .d files from
# earlier build schemes can append extra prerequisites to these targets, and
# a leaked .c source turns the link into "cannot specify -o when generating
# multiple output files".
$(filter-out build/test_gguf,$(TEST_BINS)): build/test_%: build/tests/test_%.c.o build/libh3image.a
	$(CC) $(CFLAGS) -o $@ $< build/libh3image.a $(LDLIBS)

build/test_gguf: build/tests/test_gguf.c.o build/tests/ggml_ref.c.o build/libh3image.a
	$(CC) $(CFLAGS) -o $@ build/tests/test_gguf.c.o build/tests/ggml_ref.c.o \
		build/libh3image.a $(LDLIBS)

build/tools/%.c.o: tools/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(TOOL_BINS): build/%: build/tools/%.c.o build/libh3image.a
	$(CC) $(CFLAGS) -o $@ $< build/libh3image.a $(LDLIBS)

# The combined shader file is a runtime prerequisite of every GPU test and
# tool: build it here too, not only via `all` (make check after make clean
# must not run tests against a missing build/h3i_shaders.metal).
tests-build: $(TEST_BINS) $(TOOL_BINS) $(SHADERS)

tools: $(TOOL_BINS)

test: all $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done
	@if test -f tests/test_cli.sh; then echo "== test_cli.sh"; sh tests/test_cli.sh || exit 1; fi

test-quick: all $(addprefix build/,$(QUICK))
	@for t in $(QUICK); do echo "== $$t"; ./build/$$t || exit 1; done
	@if test -f tests/test_cli.sh; then echo "== test_cli.sh"; sh tests/test_cli.sh || exit 1; fi

check-images: all
	@if test -f tests/check_images.sh; then sh tests/check_images.sh; \
	else echo "skip: tests/check_images.sh is not present"; fi

# One-command run: build + run every test, keep going after failures,
# exit nonzero if any failed. On Linux: the portable core suite.
ifeq ($(shell uname -s),Darwin)
check: tests-build
	@fail=0; for t in $(TEST_BINS); do \
		echo "== $$t"; ./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi; \
	exit $$fail

bench: build/bench_gpu $(SHADERS)
	./build/bench_gpu 1024
else
check: core-tests

bench:
	@echo "bench requires macOS/Metal"
endif

clean:
	rm -rf build h3-image

# --- Linux portable core ----------------------------------------------------
# Grows with the phases: parsers/dequant/source/detection (Phase 2),
# layout/host math (Phase 3), modulation host math + heads (Phase 5).
# _XOPEN_SOURCE=700 implies POSIX 2008 and exposes XSI functions (realpath).

CORE_CFLAGS = -std=c11 -O2 -g -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
              -Wconversion -Wno-sign-conversion -ffp-contract=off \
              -D_XOPEN_SOURCE=700 -Isrc
CORE_LDLIBS = -lm
ifeq ($(shell uname -m),x86_64)
# FMA hardware lets explicit fma() compile to an instruction; the ggml
# oracle needs it for contraction parity.
CORE_CFLAGS += -mfma
endif

CORE_SRCS = src/app/paths.c src/io/host.c src/weights/gguf.c src/weights/safetensors.c \
            src/weights/source.c src/weights/model.c src/model/layout.c
CORE_TEST_NAMES = test_core test_gguf test_weights test_layout
CORE_LIB_OBJS = $(patsubst %,build/core/%.o,$(CORE_SRCS))
CORE_TEST_BINS = $(addprefix build/core/,$(CORE_TEST_NAMES))

build/core/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CORE_CFLAGS) -c $< -o $@

build/core/tests/ggml_ref.c.o: tests/ggml_ref.c
	@mkdir -p $(dir $@)
	$(CC) $(CORE_CFLAGS) -ffp-contract=fast -Wno-conversion -c $< -o $@

build/core/test_gguf: build/core/tests/test_gguf.c.o build/core/tests/ggml_ref.c.o $(CORE_LIB_OBJS)
	$(CC) $(CORE_CFLAGS) -o $@ build/core/tests/test_gguf.c.o \
		build/core/tests/ggml_ref.c.o $(CORE_LIB_OBJS) $(CORE_LDLIBS)

$(filter-out build/core/test_gguf,$(CORE_TEST_BINS)): build/core/test_%: build/core/tests/test_%.c.o $(CORE_LIB_OBJS)
	$(CC) $(CORE_CFLAGS) -o $@ $< $(CORE_LIB_OBJS) $(CORE_LDLIBS)

core-tests: $(CORE_TEST_BINS)
	@for t in $(CORE_TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done

# Linux host-test harness (Phase 0b): full-warning syntax gate over every
# pure-C TU, test and tool, plus the host-math known-answer smoke.
LINUX_CFLAGS := -std=c11 -O3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
	-Wno-sign-conversion -D_GNU_SOURCE -D_DARWIN_C_SOURCE \
	-Isrc -Itools/linux-stubs
linux-check:
	$(CC) -fsyntax-only $(LINUX_CFLAGS) $(LIB_C_SRCS) $(TEST_SRCS) $(TOOL_SRCS)
	$(CC) -fsyntax-only $(LINUX_CFLAGS) -D__APPLE__ src/io/host.c
	@mkdir -p build/linux
	$(CC) $(LINUX_CFLAGS) src/io/host.c tools/linux-stubs/vimage_stub.c \
		tools/linux-stubs/host_smoke.c -o build/linux/host_smoke -lm
	./build/linux/host_smoke

# ObjC syntax gate against stub Apple headers (catches declaration/type/
# selector bugs without a Mac; gpu.m is reference-derived and skipped).
objc-syntax:
	@for f in $(M_SRCS); do \
		case $$f in src/gpu/gpu.m) echo "== objc-syntax $$f (reference-derived, skipped)"; continue;; esac; \
		echo "== objc-syntax $$f"; \
		$(CC) -fsyntax-only -x objective-c -fobjc-arc -fblocks \
		  -fobjc-runtime=gnustep-2.0 \
		  -std=c11 -D_XOPEN_SOURCE=700 -Itests/macos-stubs -Isrc -DF_NOCACHE=48 \
		  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Wno-unused-parameter $$f || exit 1; \
	done

# Syntax-check the Apple-only branches of tests/tools on Linux: -D__APPLE__
# selects the GPU code paths, which only need the pure-C gpu.h declarations.
test-syntax:
	@for f in $(TEST_SRCS) $(TOOL_SRCS); do \
		echo "== test-syntax $$f"; \
		$(CC) -fsyntax-only -std=c11 -D_XOPEN_SOURCE=700 -D__APPLE__ \
		  -Isrc -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Wno-unused-parameter $$f || exit 1; \
	done

metal-lint:
	python3 tools/lint_metal.py
	python3 tools/lint_bindings.py

# Keep pattern-built objects (make would otherwise delete them as intermediate).
.SECONDARY:

-include $(shell find build -name '*.d' 2>/dev/null)
