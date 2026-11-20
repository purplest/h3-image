# h3-image — MiniMax-H3-Image inference for Apple Silicon (rebuilt from h3.c).
# Behavior/numerics: SPEC.md. Build contract: SPEC.md part 6 §4.
# Requires macOS + clang (Metal, MPSGraph, CoreML, ImageIO, ICU).

CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE -Isrc
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate -framework ImageIO -framework CoreGraphics \
	-framework UniformTypeIdentifiers -framework CoreML
LDLIBS := $(FRAMEWORKS) -licucore -lm

LIB_C := $(filter-out src/app/main.c,$(shell find src -name '*.c' 2>/dev/null | sort))
LIB_M := $(shell find src -name '*.m' 2>/dev/null | sort)
LIB_OBJ := $(addprefix build/,$(LIB_C:.c=.o) $(LIB_M:.m=.o))
SHADERS := build/h3i_shaders.metal

TEST_SRC := $(sort $(wildcard tests/test_*.c))
TEST_BINS := $(patsubst tests/%.c,build/%,$(TEST_SRC))
TOOL_SRC := $(sort $(wildcard tools/*.c))
TOOL_BINS := $(patsubst tools/%.c,build/%,$(TOOL_SRC))

QUICK_ALL := test_layout test_weights test_stream test_gpu_ops test_handoff test_ane
QUICK := $(foreach t,$(QUICK_ALL),$(if $(wildcard tests/$(t).c),$(t),))

.PHONY: all test test-quick check-images tools clean linux-check
.DEFAULT_GOAL := all

# Linux host-test harness (PLAN.md Phase 0b): syntax-check every pure-C TU
# (Accelerate stub) and run host-math known-answer tests without Metal.
# Development convenience only — the macOS build never uses these flags.
# Mirror the macOS warning set exactly so conversion issues surface here
# first. (Tests are additionally kept clean under -Wdouble-promotion.)
LINUX_CFLAGS := -std=c11 -O3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
	-Wno-sign-conversion -D_GNU_SOURCE -D_DARWIN_C_SOURCE \
	-Isrc -Itools/linux-stubs
linux-check:
	$(CC) -fsyntax-only $(LINUX_CFLAGS) $(LIB_C) $(TEST_SRC) $(TOOL_SRC)
	@mkdir -p build/linux
	$(CC) $(LINUX_CFLAGS) src/io/host.c tools/linux-stubs/vimage_stub.c \
		tools/linux-stubs/host_smoke.c -o build/linux/host_smoke -lm
	./build/linux/host_smoke

# Until src/app/main.c lands (Phase 8), `all` builds the library and shaders.
ifeq ($(wildcard src/app/main.c),)
all: build/libh3image.a $(SHADERS)
else
all: h3-image $(SHADERS)
endif

h3-image: build/src/app/main.o build/libh3image.a
	$(CC) -o $@ $^ $(LDLIBS)

build/libh3image.a: $(LIB_OBJ)
	@mkdir -p $(dir $@)
	rm -f $@
	$(AR) rcs $@ $^

$(SHADERS): src/gpu/shaders.metal src/gpu/kernels.metal
	@mkdir -p $(dir $@)
	cat $^ > $@

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(OBJCFLAGS) -c $< -o $@

build/test_%: tests/test_%.c build/libh3image.a
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< build/libh3image.a -o $@ $(LDLIBS)

build/%: tools/%.c build/libh3image.a
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< build/libh3image.a -o $@ $(LDLIBS)

tools: $(TOOL_BINS)

test: all $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done
	@if test -f tests/test_cli.sh; then echo "== test_cli.sh"; sh tests/test_cli.sh || exit 1; fi

test-quick: all $(addprefix build/,$(QUICK))
	@for t in $(QUICK); do echo "== $$t"; ./build/$$t || exit 1; done
	@if test -f tests/test_cli.sh; then echo "== test_cli.sh"; sh tests/test_cli.sh || exit 1; fi

check-images: h3-image
	@if test -f tests/check_images.sh; then sh tests/check_images.sh; \
	else echo "skip: tests/check_images.sh is not present"; fi

clean:
	rm -rf build h3-image

-include $(shell find build -name '*.d' 2>/dev/null)
