# h3-image — rebuild of MiniMax-H3-Image on Apple Silicon (see SPEC.md).
#
# macOS target (SPEC §6.4):  make            → h3-image + build/h3i_shaders.metal
#                            make test       → every build/test_* then tests/test_cli.sh
#                            make test-quick → layout weights stream gpu_ops handoff ane + cli
#                            make tools      → build/<name> for each tools/*.c
#                            make check-images
#
# Linux target (PLAN §1):    make core-tests → portable C core + CPU unit tests.
#                            Apple-only sources are excluded; guarded code uses __APPLE__.
#
# CC can be overridden on the command line (e.g. make core-tests CC=gcc).

CC = clang
AR = ar

# FP contraction OFF everywhere: the spec's separately-rounded F32 ops must
# not fuse; explicit fma() calls still emit FMA (SPEC §1.4, §4.0, §5).
CFLAGS = -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
         -Wno-sign-conversion -ffp-contract=off -D_DARWIN_C_SOURCE -Isrc
OBJCFLAGS = $(CFLAGS) -fobjc-arc
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders \
             -framework MetalPerformanceShadersGraph -framework Accelerate \
             -framework ImageIO -framework CoreGraphics -framework UniformTypeIdentifiers \
             -framework CoreML
LDLIBS = $(FRAMEWORKS) -licucore -lm

# --- macOS full build -------------------------------------------------------

C_SRCS := $(shell find src -name '*.c' 2>/dev/null)
M_SRCS := $(shell find src -name '*.m' 2>/dev/null)
LIB_C_SRCS := $(filter-out src/app/main.c,$(C_SRCS))
LIB_OBJS := $(patsubst %,build/%.o,$(LIB_C_SRCS) $(M_SRCS))
MAIN_OBJ := build/src/app/main.c.o

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,build/%,$(TEST_SRCS))
TOOL_SRCS := $(wildcard tools/*.c)
TOOL_BINS := $(patsubst tools/%.c,build/%,$(TOOL_SRCS))

.PHONY: all lib test test-quick check-images tools clean core-tests objc-syntax test-syntax tests-build

all: h3-image build/h3i_shaders.metal

# Library + shaders only: lets the Mac side build and run tests before the
# CLI (src/app/main.c, milestone M8) exists.
lib: build/libh3image.a build/h3i_shaders.metal

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

h3-image: $(MAIN_OBJ) build/libh3image.a
	$(CC) $(CFLAGS) -o $@ $(MAIN_OBJ) build/libh3image.a $(LDLIBS)

build/h3i_shaders.metal: src/gpu/shaders.metal src/gpu/kernels.metal
	@mkdir -p build
	cat src/gpu/shaders.metal src/gpu/kernels.metal > $@

# Static pattern rules: unambiguous on GNU make 3.81 (macOS) and 4.x.
$(TEST_BINS): build/test_%: tests/test_%.c build/libh3image.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# ggml oracle (A1): contraction fast, same as the Linux core build.
build/tests/ggml_ref.c.o: tests/ggml_ref.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -ffp-contract=fast -Wno-conversion -c $< -o $@

build/test_gguf: build/tests/ggml_ref.c.o

$(TOOL_BINS): build/%: tools/%.c build/libh3image.a
	$(CC) $(CFLAGS) -o $@ $< build/libh3image.a $(LDLIBS)

# Build every current test binary and tool in one command.
tests-build: $(TEST_BINS) $(TOOL_BINS)

test: all $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done
	@echo "== test_cli.sh"; ./tests/test_cli.sh

test-quick: all $(addprefix build/,test_layout test_weights test_stream test_gpu_ops test_handoff test_ane)
	@for t in test_layout test_weights test_stream test_gpu_ops test_handoff test_ane; do \
		echo "== $$t"; ./build/$$t || exit 1; done
	@echo "== test_cli.sh"; ./tests/test_cli.sh

check-images: all
	./tests/check_images.sh

tools: $(TOOL_BINS)

clean:
	rm -rf build h3-image

# --- Linux portable core ----------------------------------------------------
# Grows with milestones (PLAN §4): parsers/dequant (M2), layout/host (M3),
# modulation host math + heads (M5), tokenizer (M7), ane/spec.c writer (M11).

# _XOPEN_SOURCE=700 implies POSIX 2008 and exposes XSI functions (realpath).
CORE_CFLAGS = -std=c11 -O2 -g -MMD -MP -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
              -Wno-sign-conversion -ffp-contract=off -D_XOPEN_SOURCE=700 -Isrc
CORE_LDLIBS = -lm
ifeq ($(shell uname -m),x86_64)
# FMA hardware lets explicit fma() compile to an instruction (still correctly
# rounded without it); the ggml oracle needs it for contraction parity.
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

# The vendored ggml oracle is compiled with contraction FAST so `dl*q - ml`
# fuses exactly as ggml does on Apple Silicon (tests/ggml_ref.h). Upstream
# code is verbatim, so its conversion warnings are suppressed.
build/core/tests/ggml_ref.c.o: tests/ggml_ref.c
	@mkdir -p $(dir $@)
	$(CC) $(CORE_CFLAGS) -ffp-contract=fast -Wno-conversion -c $< -o $@

build/core/test_gguf: build/core/tests/ggml_ref.c.o

build/core/test_%: build/core/tests/test_%.c.o $(CORE_LIB_OBJS)
	$(CC) $(CORE_CFLAGS) -o $@ $^ $(CORE_LDLIBS)

core-tests: $(CORE_TEST_BINS)
	@for t in $(CORE_TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done

# Linux syntax gate for the ObjC sources: clang -fsyntax-only against stub
# Apple headers (tests/macos-stubs). Catches declaration/type/selector
# consistency bugs in OUR code without a Mac. Apple API correctness and
# Metal shader compilation still require the macOS build.
objc-syntax:
	@for f in $(M_SRCS); do \
		echo "== objc-syntax $$f"; \
		$(CC) -fsyntax-only -x objective-c -fobjc-arc -fblocks \
		  -fobjc-runtime=gnustep-2.0 \
		  -std=c11 -D_XOPEN_SOURCE=700 -Itests/macos-stubs -Isrc -DF_NOCACHE=48 \
		  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Wno-unused-parameter $$f || exit 1; \
	done

# Syntax-check the Apple-only branches of tests/tools on Linux: -D__APPLE__
# selects the GPU code paths, which only need the pure-C gpu.h declarations
# (no linking). Catches libm/API mistakes in test code before the Mac build.
test-syntax:
	@for f in $(TEST_SRCS) $(TOOL_SRCS); do \
		echo "== test-syntax $$f"; \
		$(CC) -fsyntax-only -std=c11 -D_XOPEN_SOURCE=700 -D__APPLE__ \
		  -Isrc -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Wno-unused-parameter $$f || exit 1; \
	done

# Keep pattern-built objects (make would otherwise delete them as intermediate).
.SECONDARY:

-include $(shell find build -name '*.d' 2>/dev/null)
