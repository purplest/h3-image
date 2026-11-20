# h3-image build plan

Rebuild of **h3-image** (MiniMax-H3-Image text-to-image + editing inference for Apple Silicon)
from **SPEC.md** (authoritative) and the **h3.c** donor codebase (MIT, Salvatore Sanfilippo,
pinned commit `8974cc0`, at `~/Github/h3.c`).

Status: **Phase 0 + 0b complete (Linux gates green).** 27 donor files migrated to `src/`
with `h3i_` naming; Makefile, shader concat, and Linux host-test harness in place.
`make linux-check`: 10/10 pure-C TUs syntax-clean, host-math smoke **45 checks, 0 failures**
(PCG32/Box-Muller bit-exact vs SPEC §3.13, spare carry, separately-rounded Euler).
Two spec-driven fixes applied during migration: `#include "h3.h"` → `"h3i.h"` (rename missed
the underscore-less form), and `#pragma STDC FP_CONTRACT OFF` in `io/host.c` + `model/vae_encoder.c`
(clang contracts `a*b+c` into FMA on arm64 at -O3; SPEC §3.12/§4.0 forbid it — verified with
an `-mfma` probe). **Pending: Mac-side `make` gate** (Metal compile of libh3image.a + shaders).
Next: Phase 1 (GPU runtime audit + test_gpu_ops).

**Phase 1 progress (Linux side done, Mac gate pending):** audited `gpu.m` against SPEC §6.3 —
safe-math compile (`MTLMathModeSafe`), begin/continue/submit semantics, exact error strings,
shared-storage tensors + `F_NOCACHE`, dispatch shapes (1d/2d/3d/rows) and MPSGraph thresholds
(rows≥32 ∧ in≥256 ∧ out≥256) all match; fused SwiGLU MLP and native SDPA confirmed; GQA is the
custom kernel by default (MPSGraph only via opt-in `H3I_MPS_GQA`). Added the one missing spec
op: `h3i_gpu_recommended_working_set`. Wrote `tests/test_gpu_ops.c` (command model, tensor I/O,
add/gate exact, silu-mul/swiglu ≤1, rms/layer ≤2, adaln ≤3, linear both dispatch paths ≤1%,
SDPA rel L2 <0.02) and `tools/bench_gpu.c` (QKV/MLP/SDPA at 1 MP/4 MP; decode bench deferred to
Phase 2). Confirmed for Phase 7: donor GQA allocates seq×4 B threadgroup memory and **fails
above ~8192 rows** — the spec's chunked online softmax (min(seq,4096)×4 B) must be added.
Linux: syntax-clean incl. new test/tool; host smoke still 45/0. **Mac gate:** `make && make test`
+ `./build/bench_gpu`. ** CLOSED — user confirmed all pass on Mac (after two test-reference
fixes: explicit conversion casts for Apple clang, and AdaLN single-rounding reference).**

**Phase 2 progress (Linux side done, Mac gate pending):** discovered `main` is a previous run
of this same rebuild (M2 weights-CPU and M3 layout already A1/A2-verified there). Adopted its
spec-faithful modules instead of rewriting: `weights/{gguf,safetensors,source,model}.{c,h}`,
`app/{paths.c,paths.h,clock.h,error.h}`, GPU additions (`gpu.h/gpu.m` convert_raw + get_info +
tensor_contents + new_u32 + `*_h3c` wrappers; `kernels.metal` with all 18 convert kernels,
NE handoff, **chunked-online-softmax GQA** and spec-exact QKV RoPE — Phase 7 kernel work
arrives early), test infra (`check.h`, `builders.h`, `bf16_util.h`, `gpu_test.h`, `ggml_ref.c`
oracle, `vectors.h`, `test_core/test_gguf/test_weights`, macos-stubs, lint tools).
New code: `weights/stream.{c,h}` (SPEC §2.7 verbatim: two staging sets, read-ahead thread,
ordering contract, double slots, quiesce), `tests/test_stream.c` (streamed==resident bit-exact,
shuffled/cyclic passes, F32 conversion vs CPU, double-slot coexistence, spec error strings),
`tools/inspect.c`, `h3i_source_load` (§4.4), `h3i_type_name`, donor weight-store API rewritten
as a shim over the source (dit.c/text_encoder.c adapted, ~15 sites). Fixed an adopted bug:
gpu.h additions sat **outside the include guard** (double-inclusion typedef redefinition).
Makefile merged with main's machinery: global `-ffp-contract=off`, two-step test builds,
ggml oracle compiled contraction-fast, `core-tests`/`objc-syntax`/`test-syntax`/`metal-lint`.
**Linux gates:** core-tests **154 checks 0 failures** (test_core 12, test_gguf 54 incl. A1 CPU
bit-exact vs ggml oracle all 9 types, test_weights 88), linux-check 0 warnings + smoke 45/0,
test-syntax/objc-syntax/metal-lint clean, all 14 spec error strings verified present.
**Mac gate:** `make && make check` (test_gpu_ops, test_gguf GPU==CPU + real-weight counts if
weights present, test_weights, test_stream) + `make tools && ./build/inspect <weights>`.
**CLOSED — ALL TESTS PASSED on Mac.** A1 complete (CPU bit-exact vs ggml oracle + GPU
bit-exact vs CPU, all 9 types). Gate debugging: stale Phase-1 .d files polluted $^ link
commands (explicit link inputs now); tests-build didn't build the combined shaders (fixed);
test_gguf GPU half had a heap overflow (71 floats read into a 64-float buffer) corrupting
the reference data — GPU kernels were correct (fixed, af2d019).

---

## 1. Goal and scope

One CLI program `h3-image` (version `0.1.0`):

- prompt → PNG (text-to-image), prompt + 1–16 reference images → PNG (editing);
- two weight formats, one program, same math: GGUF set (~24 GiB, Q4_K_M DiT / Q2_K_M Qwen3-VL /
  F16 VAE) and official BF16 release (~114 GiB); any mix of components works;
- DiT streams from SSD block-by-block (~3 GiB peak at 2048²);
- deterministic: same seed + inputs + options + machine → byte-identical PNG;
- optional `--neural-engine` FP16 accelerator path (deferred, Phase 11).

Out of scope (SPEC §1.1): prompt expansion, training, CFG/negative prompts, PyTorch-identical
RNG, Python at run time. Video/audio generation (the donor's purpose) is **not** carried over.

## 2. Assumptions and ground rules

| # | Assumption / rule |
|---|---|
| A1 | **Target is macOS / Apple Silicon (Metal + MPSGraph + CoreML + ImageIO + ICU)**, same as donor and spec reference. This Linux dev box writes and syntax-checks code; compiling, running and every GPU test gate happen on the user's Mac. If a Linux-runnable port (CUDA/Vulkan/CPU) is wanted instead, this plan changes fundamentally — confirm before Phase 0. |
| A2 | SPEC.md is the source of truth for behavior, numerics, strings and exit codes. Where spec and donor differ, spec wins (it documents the already-validated reference implementation). Spec part 7 "known issues" are reproduced as-is unless marked otherwise; the two "bug?" items are flagged for decision, not silently fixed. |
| A3 | Donor is pinned at `8974cc0`. No pulling newer h3.c changes mid-rebuild. |
| A4 | Public names `h3i_` / macros+env `H3I_`; every fallible function returns 0/NULL and fills `(char *error, size_t error_size)`; CLI prints `h3-image: <message>`, exit 1 runtime / 2 usage. |
| A5 | Each phase ends with its test gate (SPEC §1.7 build order). No phase starts before the previous gate passes on the Mac (host-only gates may pass on the Linux harness first). |
| A6 | Commits per phase on branch `fresh`, only when the user asks. |

## 3. What the code dig established

**The architectures really are shared.** HF `MiniMaxAI/MiniMax-H3-Image` is tagged
`base_model:finetune:MiniMaxAI/MiniMax-H3`; `transformer/config.json` matches the donor DiT
exactly: hidden 5376, 50 blocks + 2 refiner, 56×128 heads, FFN 14336, time 256→5376→2688,
AdaLN 96768 = 3×6×5376 / final 10752, `rope.inv_freq` len 16, eps 1e-5, `audio_enabled: false`.
`pdd_config.json` sigmas, VAE mean/std + 256/64 tiling + ViT decoder kwargs, Qwen3-VL text/vision
configs (mRoPE interleaved [24,20,20], theta 5e6, deepstack [8,16,24], token ids
151652/151653/151655) all match SPEC.md verbatim. The spec is trustworthy.

**Donor reuse map** (verified include graph; no kept file references a dropped module):

| Donor file | → Destination | Disposition |
|---|---|---|
| `h3_gpu.h` / `h3_gpu.m` | `src/gpu/gpu.h` / `gpu.m` | migrate; audit vs SPEC §6.3 (runtime is the spec's `gpu` module ancestor) |
| `h3_metal.h` / `.m` | `src/gpu/metal.h` / `.m` | migrate (device probe for `--info`) |
| `h3_shaders.metal` (205 KB) | `src/gpu/shaders.metal` | migrate **whole** (incl. unused int8/audio kernels — spec concatenates this file verbatim); rename `h3_*` kernels → `h3i_*` |
| `h3_host.h` / `.c` | `src/io/host.h` / `.c` | migrate: PCG32+Box-Muller RNG, vImage high-quality resize (ARGB8888, `kvImageHighQualityResampling\|kvImageEdgeExtend` — exactly SPEC §4.4/4.5), Euler step; canvas/layout/schedule parts get replaced by image-specific versions in Phase 3 |
| `h3_tokenizer.h` / `.m` | `src/io/tokenizer.h` / `.m` | migrate (ICU BPE); audit vs SPEC §4.1 in Phase 7 |
| `h3_terminal.h` / `.c` | `src/io/terminal.h` / `.c` | migrate (`--show`; spawns external ffmpeg helper, no lib dependency) |
| `h3_safetensors.h` / `.c` | `src/weights/safetensors.h` / `.c` | migrate; harden to SPEC §2.3 in Phase 2 |
| `h3_weights.h` / `.c` | `src/weights/weights.h` / `.c` | migrate as interim store; **superseded** by `source.c` in Phase 2 |
| `h3_text_encoder.h` / `.c` | `src/model/text_encoder.h` / `.c` | migrate; rewire to weight source + GGUF + always-streamed in Phase 7 |
| `h3_vision_encoder.h` / `.c` | `src/model/vision.h` / `.c` | migrate; adapt to still images in Phase 10 |
| `h3_video_encoder.h` / `.c` | `src/model/vae_encoder.h` / `.c` | migrate (causal 3D CNN VAE encoder — SPEC §8.5 says image encoder uses the same kernels); single-frame reduction in Phase 10 |
| `h3_multimodal.h` / `.c` | `src/model/multimodal.h` / `.c` | migrate (`<Picture n>` presentation ancestor); becomes `app/edit.c` basis in Phase 10 |
| `h3_dit.h` / `.c` (142 KB) | `src/model/dit.h` / `.c` | migrate; **heaviest adaptation** (Phase 6): strip audio + int8/NAX/token-reduction fast paths, rewire to source+streamer, add pruned AdaLN / QKV-layout detection / PDD heads / 6-step Euler |
| `h3_dit_schedule.h` / `.c` | `src/model/dit_schedule.h` / `.c` | migrate (official time-MLP AdaLN precompute); becomes `modulation.c` basis in Phase 5 |
| `h3.h` | `src/h3i.h` | **trimmed rewrite**: only `H3I_VERSION`, `h3i_device_info`, `h3i_component_info` (the only types kept files use) |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | same | copy (notices: ccv BSD-3 for Morton/int8 kernels kept in shaders.metal) |

**Dropped** (video/audio-only; donor repo remains on disk as reference): `h3.c` (video
orchestration), `h3_internal.h`, `h3_cli.*`, `main.c`, `linenoise.*`, `h3_audio_vae.*`,
`h3_ffmpeg.*`, `h3_video_vae.*` (3D-CNN video **decoder** — image decoder is a new 2D ViT),
donor `tests/*` (video fixtures). Note: donor decodes images via spawned ffmpeg
(`h3_ffmpeg_read_image_f32`); the spec requires ImageIO/CoreGraphics instead → `io/image.m`
is **new code**, not a migration.

**Genuinely new code** (no donor ancestor):

| Module | Spec | Content |
|---|---|---|
| `weights/gguf.c` | §2.2 | GGUF v2/v3 header parser, prefix-retry strategy, exact error strings |
| `weights/source.c` | §2.4 | multi-file flat source, GGUF dim reversal, one-way aliases, `read_f32` / `load` / `read_raw` |
| CPU dequant (in `gguf.c`) + `kernels.metal` `h3i_convert_*` | §2.5 | all 9 ggml types, bit-exact CPU==GPU, explicit `fma`, BF16 RNE |
| `weights/model.c` | §2.6 | directory auto-detection (classifier probes, resolution order) |
| `weights/stream.c` | §2.7 | generic layer streamer (donor has streaming baked into `h3_dit.c`; this is the refactor): 2 staging sets, read-ahead thread, slots, `double_slots`, quiesce |
| `model/layout.c` | §3.2–3.4, §4.5 | canvas resolve, patchify 2×2, packed sequence + normalized positions, tile plan, `unit_to_u8` |
| `model/modulation.c` | §3.7 | official + **pruned** AdaLN tables, row maps |
| PDD heads load | §3.11 | absolute vs base+offset heads |
| `model/vae.c` | §4.5 | **2D ViT decoder** (24 layers, registers, 2D RoPE θ=100, scale1/scale2, tiled stitch) |
| `io/image.m` | §4.8 | ImageIO decode + EXIF orientation + alpha-over-black; sRGB PNG write |
| `app/pipeline.c`, `app/main.c`, `app/paths.c`, `app/edit.c` | §6.1–6.2, §4.4 | orchestration, exact CLI, shader path resolution, edit preparation |
| `ane/*`, `model/dit_ane.c`, handoff kernels | part 5 | optional NE path (Phase 11) |
| `tests/*`, `tools/*` | §6.5–6.7 | spec test suite + tools |

**Key numeric/behavior deltas vs donor** (the adaptation work):

1. No audio anywhere; sequence = `[text | refs | target]`, 3-axis RoPE over 96 of 128 dims
   (donor video uses same machinery — verify axis mapping), `rope.inv_freq` from checkpoint.
2. Sampler: 6 fixed PDD Euler steps, per-step output heads, F32 velocity, no CFG / RES / reuse.
3. Two AdaLN forms: official time-MLP (donor has it) **and** pruned `adaln_t_table` interpolation
   (new; no SiLU; sequential F32 dot).
4. QKV row order: official grouped `[head][q|k|v][128]` (donor default) vs GGUF chunked
   `[q|k|v][head][128]` (new detection + `H3I_QKV_LAYOUT`).
5. GGUF quantized weights everywhere (donor is safetensors-BF16-only): dequant to BF16 slots,
   Q2_K text encoder, per-row embedding gather, always-streamed TE.
6. VAE decoder is a 2D ViT (new), not the donor's 3D CNN; VAE **encoder** is the donor's causal
   3D CNN reduced to one frame (last temporal tap).
7. Image I/O via ImageIO (new), canvas rules (min-area scaling, 32-px multiples, 1.25×4096²
   reject), final resize back to requested size.
8. Causal GQA kernel: donor @8974cc0 already tree-reduces the softmax max with barriers (the
   race SPEC §1.4.4 warns about appears fixed upstream), but it has **no >4096-key chunked
   online softmax** and pre-scales Q with BF16 rounding (MLX order). Phase 7 audits both against
   A9 and adds the chunking.

## 4. Target repository layout

```
SPEC.md  PLAN.md  README.md  LICENSE  THIRD_PARTY_NOTICES.md  Makefile  .gitignore
src/
  h3i.h            trimmed shared types
  app/             main.c pipeline.c paths.c edit.c error.h progress.h clock.h   (Phases 8, 10)
  gpu/             gpu.h gpu.m metal.h metal.m shaders.metal kernels.metal
  weights/         gguf.c safetensors.c source.c model.c stream.c weights.c(interim)
  model/           layout.c modulation.c dit.c dit_debug.c dit_schedule.c→modulation
                   text_encoder.c vision.c vae.c vae_encoder.c multimodal.c
  io/              host.c tokenizer.m image.m terminal.c
  ane/             ane.m spec.c                                                   (Phase 11)
tests/             test_*.c test_cli.sh check_images.sh data/
tools/             inspect.c bench_gpu.c profile_dit.c gpu_determinism.c te_repeat.c
                   download.sh modelscope_get.py fetch_tensors.py bench.sh ane_models.py
build/             (generated: libh3image.a, h3i_shaders.metal, objects, test binaries)
```

## 5. Phases

Each phase lists: work → files, exit gate → tests/acceptance ids from SPEC §1.5.
"Linux" = verifiable on this box; "Mac" = needs the user's Mac.

### Phase 0 — Migration + build skeleton  *(script staged, runs on approval)*

- Execute `/tmp/h3i-migrate/migrate.sh`: 27 files copied per §3 map with mechanical
  `h3_→h3i_` / `H3_→H3I_` rename (word-boundary sed; strings like `h3: …` and provenance
  comments intentionally untouched) and include rewrites to `-Isrc` paths.
- Write `src/h3i.h` (trimmed), `src/gpu/kernels.metal` (comment-only placeholder until Phase 2),
  `Makefile` per SPEC §6.4 (clang, `-std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow
  -Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE -Isrc`, `-fobjc-arc` for ObjC; frameworks
  Foundation/Metal/MPS/MPSGraph/Accelerate/ImageIO/CoreGraphics/UniformTypeIdentifiers/CoreML,
  `-licucore -lm`; `build/libh3image.a` deleted+recreated each build; objects `build/<path>.o`;
  `build/h3i_shaders.metal = cat shaders.metal kernels.metal`; targets `all test test-quick
  check-images tools clean`). Until `src/app/main.c` exists, `all` = lib + shaders (Makefile
  conditional flips automatically when main.c lands).
- `.gitignore` (build/, h3-image, outputs/, *.o, *.d), `README.md` (status + build instructions),
  copy LICENSE/notices.
- **Gate:** zero residual `\bh3_`/`\bH3_` in `src/`; `clang -fsyntax-only` clean on the 9 pure-C
  TUs (Linux); `make -n` sane; shader concatenation produces `build/h3i_shaders.metal` (Linux);
  **Mac:** `make` builds `libh3image.a` + shaders with no errors.

### Phase 0b — Linux host-test harness (cross-cutting, small)

- `tools/linux-stubs/`: minimal `<Accelerate/Accelerate.h>` stub (vImage decls used by host.c)
  so `io/host.c` syntax-checks; a harness make target (`make linux-check`) compiling the pure-C
  host/weights/layout code with a stubbed resize, enabling **running** host-only tests here:
  `test_layout`, `test_weights` parser/resolve/read_f32 parts, CPU dequant vs a literal ggml
  reference. GPU sections of tests compile out via `#ifdef __APPLE__` guards (test files only —
  never library code).
- **Gate:** harness runs green on Linux for whatever host tests exist at each phase; full gates
  still run on Mac. Keeps the bit-exact host math (the spec's main bug source) under test
  without a Mac round-trip.

### Phase 1 — GPU runtime verification (SPEC §6.3; build order 1)

- Audit migrated `gpu.m` against spec: begin/continue/submit semantics, shared-storage tensors,
  eager pipeline dictionary, dispatch shapes (1d/2d/3d/rows), MPSGraph caches + dispatch
  thresholds (rows≥32 ∧ in≥256 ∧ out≥256 → MPSGraph, else custom 16×16), stats, error strings.
- Confirm every pipeline in SPEC §3.4's list exists post-rename; note ops the spec lists that
  the donor lacks (none expected beyond convert/handoff — those land in Phases 2/11).
- `tests/test_gpu_ops.c` (tolerances: rms/layer ≤2 BF16 steps, AdaLN ≤3, SwiGLU/SiLU-mul ≤1,
  gate & add exact, linear max err ≤1% of max|ref|, SDPA <0.02); `tools/bench_gpu.c`.
- **Gate:** test_gpu_ops + bench on Mac. Linux: syntax only.

### Phase 2 — Weights: GGUF, source, dequant, detection, streamer (SPEC part 2; A1)

- `weights/gguf.{c,h}`: header-only parser, 1 MiB prefix ×4 retry (512 MiB cap), v2/v3,
  metadata skip rules (alignment only, depth ≤8), tensor info validation, exact error strings.
- Harden `weights/safetensors.c` to SPEC §2.3 (256 MiB header cap, one JSON object + trailing
  whitespace, `__metadata__` skip, dtype/shape/offsets rules, exact byte-count match, JSON
  integer/string escape details).
- `weights/source.{c,h}`: flat records; add file/dir (strcmp-sorted, non-recursive); GGUF shape
  reversal; first-match lookup + one-way prefix aliases (`model.language_model.`→`model.`,
  `model.visual.`→`visual.`, 511-byte truncation); `read_f32`, `load(F32|BF16)` (raw copy or
  int8 staging + convert), `read_raw` (pread/EINTR loop, `F_NOCACHE`, per-call open/close).
- Dequant: CPU reference (9 types, `fma(a,q,−m)` where written, F32 order per SPEC §5.2, BF16
  RNE §5.1) + GPU kernels `h3i_convert_<type>_<bf16|f32>` in `kernels.metal` +
  `h3i_gpu_convert_raw` (chunk grid, validation rules §5.3).
- `weights/model.{c,h}`: classifier probes + `find_kind` + resolution order + final checks
  (exact strings, SPEC §2.6); DiT source assembly with `pdd_fused_heads.safetensors` append rule.
- `weights/stream.{c,h}`: SPEC §2.7 verbatim (allocation rules, acquire/prefetch/join, ordering
  contract, `double_slots`, `quiesce`, `H3I_NO_PREFETCH`, wait_seconds accounting).
- Tests: `test_weights` (malformed headers, fake official/GGUF trees), `test_gguf` (GPU==CPU
  bit-exact all types; CPU vs literal ggml port rel ≤1e-6; real files: DiT 528 / TE 902 tensors),
  `test_stream` (streamed==resident, in-order/shuffled/repeated, F32 conversion, double slots).
  `tools/inspect.c`.
- **Gate: A1.** Linux harness: parsers + CPU dequant + read_f32 run here; GPU/stream gates on Mac.

### Phase 3 — Layout + host math (SPEC §3.2–3.6, §4.5 tile plan; A2)

- `model/layout.{c,h}`: `h3i_canvas_resolve` (double math, nearbyint ties-even, 32-px multiples,
  1.25×MAX_AREA rejection), patchify/unpatchify (2×2 raster, channel-major within patch),
  `h3i_layout_build` (text/refs/target, `grid()` with exact separately-rounded op order, tags),
  `h3i_tile_axis_plan` (256/64, remainder distribution), `h3i_unit_to_u8` (truncation, NaN→0).
- `io/host.c`: audit RNG against SPEC §3.13 (PCG32 constants, seed sequence, Box-Muller
  `(u32+1)/2^32+1` / `(u32+0.5)/2^32`, spare carry); replace video schedule with
  `h3i_schedule_build` (6 released sigmas, `pdd_config.json` override validation,
  `t=float32(1−σ)` double-subtract, `σ=1.0f−t`); Euler update two separately-rounded F32 ops;
  keep vImage resize as-is.
- `tests/test_layout.c`: bit-exact canvas/schedule/RoPE-position/tile/patchify/unit_to_u8 checks.
- **Gate: A2** — fully runnable on the Linux harness, then Mac.

### Phase 4 — VAE decoder + image I/O (SPEC §4.5, §4.8; A10)

- `model/vae.{c,h}` (new): F32 throughout; post_quant_conv 1×1, x_embedder, 4 registers + zero
  row, 24 blocks (RMSNorm 1e-5, fused qkv `[head][q|k|v][64]`, weightless q/k RMS, 2D RoPE
  θ=100 over dims 0..31, SDPA scale 1/8, scale1/scale2 residual scales, SwiGLU w1→8192),
  LayerNorm out, proj_out 768 → 16×16 RGB patches; un-normalize `z·std+mean` (no FMA);
  tile plan + raw-neighbour above-then-left stitch in normalized pixel space; ImageNet
  un-normalize + uint8; `vae/config.json` mean/std override with built-in defaults.
- `io/image.{m,h}` (new): ImageIO frame-0 decode, EXIF orientation 1–8 table, RGBX in image's
  own color space (non-RGB → sRGB), alpha NoneSkipLast over zeroed buffer; PNG write 8-bit RGB
  sRGB top-down; resize via migrated host vImage helper.
- `tests/test_vae.c` + `tests/data/vae_latent_512.f32` (fixture generated on Mac from official
  weights): decode stored latent, F16-vs-F32 >40 dB, encode→decode round trip >25 dB (encoder
  part activates in Phase 10), writes `build/vae_*.png`.
- **Gate:** coherent PNG from a stored latent; A10 decode parts. Mac only.

### Phase 5 — Modulation + PDD heads (SPEC §3.7, §3.11; A5, A6)

- `model/modulation.{c,h}` (from `dit_schedule.c`): official path (Timesteps(256,
  flip_sin_to_cos, downscale 0) F32 libm, proj_in/out F32 fma, SiLU→BF16 act, per-block BF16
  projection F32 accum) + pruned path (`adaln_t_table` [G,D], clamp/floor/lerp separately
  rounded, sequential F32 dot + bias, **no SiLU**); format detection + exact errors; table
  layout `[time][modality][slot][5376]`, final `[slot2][5376]`; `h3i_dit_prepare_maps`.
- Heads: `h3i_heads_load` — absolute `[6,96,5376]` vs GGUF base+offset (`W_i+=W_0` in F32 once),
  S==6 enforcement, exact error strings.
- `tools/fetch_tensors.py` (HTTP range-fetch → `check/official_block0.safetensors` etc.).
- `tests/test_modulation.c`: heads rel <1e-5; pruned-vs-official worst <0.05; wrong-SiLU control
  >4× worse.
- **Gate: A5, A6.** Mac (needs check fixtures); pruned interpolation math also unit-tested on Linux.

### Phase 6 — DiT (SPEC §3.8–3.10, §3.12, §3.14–3.15; A3, A4, A7)

- Adapt `model/dit.c`: strip audio branches, int8/NAX/Morton fast paths, token reduction,
  core reuse, preview hooks — keep the portable BF16 path (MPSGraph linear/MLP/SDPA + custom
  kernels exactly per SPEC §3.9–3.10 dispatch rules); rewire big matrices to
  `source`+`stream` (50×4, cyclic, `--weights auto|resident|stream`, auto threshold
  0.5×recommendedMaxWorkingSetSize, `H3I_WEIGHTS`); QKV layout detection (`adaln_t_table` ⇒
  chunked; `H3I_QKV_LAYOUT` override) for blocks **and** refiner; text refiner (2 blocks, no
  AdaLN/RoPE, fc1 rounded BF16 before SwiGLU); patch embed F32→BF16 offset kernel; per-step
  forward with row maps; F32 head velocity; 6-step Euler; **submit+wait per block when streamed**
  (ordering contract), continue-encoding when resident; `H3I_CAPTURE*` / `H3I_RANGE` debug
  (`model/dit_debug.c`); `H3I_PROFILE` SSD-wait line.
- Tests: `test_dit_weights` (chunked Q4_K vs official block0 rel L2 <0.2, wrong layout >1.0),
  `test_dit_block` (<0.25 per step, swapped-layout controls >4×), `test_dit` (8 text rows +
  16×16 latent: streamed==resident **bit-identical**, finite, rms∈(0.3,5), cos(v,noise)<−0.3).
- `tools/profile_dit.c`.
- **Gate: A3, A4, A7.** Mac.

### Phase 7 — Tokenizer + text encoder (SPEC §4.1–4.2; A8, A9)

- Audit `io/tokenizer.m` vs SPEC §4.1 (NFC, hard-coded Qwen2 pre-tokenizer rules, added-token
  alternatives sorted by UTF-16 length desc, merges rank/leftmost/all-occurrences, piece cache,
  `pad_empty`, no chat template).
- Rewire `model/text_encoder.c` to `source`: official names with GGUF aliases; 50×11
  **always streamed**, non-cyclic; embedding rows gathered on host (per-row pread + dequant +
  BF16 RNE, never uploaded); q/k per-head RMS eps 1e-6; rotate-half RoPE all 128 dims,
  mRoPE interleaved sections [24,20,20] θ=5e6, BF16-rounded tables; deepstack adds after layers
  0–2; **no final norm**; 65536 hard cap; causal GQA kernel: verify race-freedom, add >4096-key
  chunked F32 online softmax, threadgroup scores capped at 4096, keep donor's MLX-order Q
  pre-scale unless A9 disagrees; `H3I_PROFILE` SSD-wait line.
- Tests: `test_attention` (43 and 8200 rows vs CPU, ≤1 BF16 step, bit-repeatable),
  `test_text` (finite, rms∈(0.05,1000), prefix rows rel L2 <0.05, repeatability >32 tokens).
  `tools/gpu_determinism.c`, `tools/te_repeat.c`.
- **Gate: A8, A9.** Mac.

### Phase 8 — Pipeline + CLI → first image (SPEC §6.1–6.2; A11, A14)

- `app/pipeline.{c,h}`: `h3i_generate` with exact stage order/memory lifetimes (T1→T2→D→V;
  one component resident at a time), one RNG with spec draw order, progress phases/strings,
  monotonic timings, `H3I_DUMP_DIR` (text.bf16 / latent.f32), error propagation.
- `app/main.c` + `app/paths.c` + `error.h`/`progress.h`/`clock.h`: parsing model (left-to-right,
  next-argv values, no `=` form), every option/validation in exact order 1–13, byte-exact usage
  text, banner, summary line format, exit codes 0/1/2, `--info`, `--profile`, `--show`,
  `--neural-engine` HOME handling (flag parsed; path inactive until Phase 11), env vars §1.8,
  shader resolution (`H3I_SHADERS` strict, else exe-relative build/h3i_shaders.metal candidates).
- `tests/test_cli.sh` (all 17 cases, no weights), `tests/check_images.sh` (record/full/quick;
  fox512 + gguf_gpu cases first), `tools/download.sh` + `tools/modelscope_get.py` (weight
  acquisition stages gguf / official-small / official-transformer / official-text-encoder /
  official-edit), `tools/bench.sh`.
- **Gate: A11 (512², 1024², 500×300 resize), A14.** First coherent fox PNGs, same-seed
  byte-identical. Mac. `make test-quick` green.

### Phase 9 — Official BF16 end-to-end (A12)

- Exercise Phases 2–8 with the official release: grouped QKV, official modulation, absolute
  heads via `pdd_fused_heads.safetensors` append rule, streamed BF16→BF16 copy-kernel path,
  F32 VAE. Fix format-specific fallout; no new numerics.
- **Gate: A12** — 1024² coherent, same scene/layout as GGUF same seed; `check_images bf16_gpu`.

### Phase 10 — Editing (SPEC §4.3–4.4, §4.7; A11 edit)

- `model/vision.c`: still-image adaptation (temporal slot duplicated), merge-window patch order,
  bilinear pos embed (double coords, BF16 rounding of every term), 2D RoPE 36 freqs θ=10000,
  27 blocks GELU-tanh, final merger (per-patch LN) vs deepstack mergers (merged-row LN) at
  {8,16,24}, GELU-erf (A&S 7.1.26) in mergers, out 5120.
- `model/vae_encoder.c`: single-frame reduction of donor causal 3D CNN (last temporal tap),
  reflect pad rules, GN32+SiLU eps 1e-6, downsample pad 0-before/1-after, quant_conv 48→48,
  256/64 pixel tiling + latent-cell stitch (`pa=a·(1−w); pb=v·w; v=pa+pb`).
- `app/edit.c` (from `multimodal.c`): reference load/sizing (`bound_short_edge`, VAE 1024–2048,
  vision exactly 1024, %32 checks), presentation (`Picture i: ` + 151652 + M×151655 + 151653,
  tags), `get_rope_index` axis-major positions (vision_start/end treated as text), VAE reference
  latents (f16 RNE sampling, logvar clamp [−30,20], even crop, normalize), `h3i_scale_noise`
  (t=0.999f, separately rounded), RNG draw order, E1–E3 pipeline stages, CLI `-i` (≤16,
  first-reference default size).
- Tests: `test_vision` (SHA-256 fixture `tests/data/vision_512.sha256`), `test_edit` (sizing,
  presentation tokens/tags/positions, f16 rounding, reference latents), `check_images edit`
  (official `examples/reference1.png` + `revised_prompt.edit.txt`), A10 round-trip part.
- **Gate:** official edit example coherent; A10/A11 complete. Mac.
- Decision point: SPEC part 7 "vision sizing (bug?)" — reproduce as-is (default) or fix.

### Phase 11 — Optional Neural Engine path (SPEC part 5; A13) — deferred by default

- `ane/spec.{c,h}`: hand-written proto3 writer (exact field numbers/order, sign-extended
  varints), `.mlpackage` layout + byte-exact Manifest.json; four MIL programs (qkv/out/mlpin/down,
  CHUNK=1792 split-sums, FP16, weights as runtime inputs).
- `ane/ane.{m,h}`: compile/cache (`<dir>/v1/r<ROWS>/`), CPUAndNeuralEngine, validation,
  zero-copy MLMultiArray + outputBackings, copy-back fallback.
- Handoff kernels (`kernels.metal`): `h3i_bf16_to_f16`, `h3i_f16_to_bf16_checked` (atomic flag),
  `h3i_transpose_bf16_f16` (32×33 tiles); `[in,out]` FP16 weight layout with up-divisor.
- `model/dit_ane.c`: divisors (initial QKV 0 / OUT 2 / MLP 4; b39,b45→8, b43→6; step 2, max 14,
  then permanent GPU), per-block GPU↔NE schedule with prefetch overlap, `stream_double_slots`,
  `--profile` lines.
- Tests: `test_handoff` (bit-exact), `test_ane` (R=64, MLComputePlan all-NE, BLAS rel <0.3% /
  mlpin <3%, cache reuse, ane_rows). `tools/ane_models.py` reference builder.
- **Gate: A13** (PSNR ≥ ~30 dB vs GPU, deterministic, no unhandled overflow). macOS 15+.

**Phase 3 progress (Linux side done, Mac gate pending):** adopted main's M3-verified
`model/layout.{c,h}` (canvas, patchify, packed sequence, PDD schedule, RoPE tables, tile
plan, unit_to_u8) + image `io/host.{c,h}` (round_f16, PCG32/Box-Muller) + `test_layout` +
`vectors.h`/`gen_vectors.py`. Donor video host math split into `io/video_host.{c,h}`
(interim): colliding symbols renamed `h3i_video_layout*` / `h3i_video_schedule_build` /
`h3i_video_sigma_schedule`, dit/dit_schedule adapted; RNG/resize/euler moved to the new
host.c (resize Apple-guarded). host_smoke passes the same spec-derived expected bits
against the adopted RNG → swap proven bit-identical. User commit f4f0db6 added env-var
real-weight paths (H3I_DIT_GGUF/H3I_TE_GGUF) + any-quant acceptance to test_gguf — kept.
**Linux gates: core-tests 267/0 (test_layout 113 = A2), linux-check 0 warnings incl.
__APPLE__ host.c path, smoke 45/0, test-syntax/objc-syntax clean. Mac gate: make check.**

## 6. Verification strategy

| Layer | Where | How |
|---|---|---|
| Syntax / includes / rename integrity | Linux (this box) | `clang -fsyntax-only -Isrc` on all pure-C TUs (Accelerate stub for host.c); grep gates for residual `h3_`/`H3_`; `make -n` |
| Host math: layout, canvas, schedule, RNG, tile plans, parsers, CPU dequant | Linux harness (Phase 0b) | actually **run** test_layout / test_weights(host parts) / dequant-vs-ggml here |
| GPU kernels, MPSGraph ops, streaming, full models, images | Mac | `make test`, `make test-quick`, `tests/check_images.sh` per phase gate |
| End-to-end determinism | Mac | same-seed byte-identical PNGs (A11/A12), streamed==resident (A7) |

Weights for gates: `tools/download.sh gguf` (~40 GiB) first; `official-small` (5 GiB) for
edit example + configs; `official-transformer`/`official-text-encoder` (65+53 GiB) only for
Phase 9; `tools/fetch_tensors.py` for small check fixtures. All weight-dependent tests print
`skip:` and pass when files are absent (spec convention), so the suite is runnable before
downloads complete.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No Metal on the dev box; every GPU gate needs a Mac round-trip | Phase 0b Linux harness covers the bug-prone host math; small phases with exact spec strings; user runs `make test-quick` at each gate |
| Bit-exactness traps (FMA contraction, BF16 RNE, op order) | Host math compiled `FP_CONTRACT OFF` (spec §4.0); dequant uses explicit `fma` only where written; A1/A2 tests before any model code; never enable fast-math in shader compile |
| QKV layout / AdaLN form mis-detection | Detection by tensor presence per spec (not file type); wrong-layout control tests (A3/A4/A5) make regressions loud |
| MPSGraph vs custom-kernel dispatch thresholds change numerics | Keep donor thresholds exactly (rows≥32 ∧ in≥256 ∧ out≥256); test_gpu_ops pins both paths |
| Causal attention race / long-prompt chunking | Phase 7 audit + test_attention at 43 and 8200 rows (A9) before first image |
| Streamed≠resident drift | Ordering contract enforced (submit+wait per streamed block); A7 bit-identical gate |
| Weight availability / disk (24–114 GiB) | download.sh stages with free-space checks; tests skip gracefully; GGUF set first |
| Donor drift | Pinned `8974cc0`; donor repo untouched |
| Spec "known issues" ambiguity | Reproduce behavior as written; the two "bug?" items surfaced as explicit decision points (Phases 8/10) |
| Rename collisions / missed includes | Mechanical word-boundary sed + grep gate + syntax check in Phase 0 |

## 8. Decisions needed (defaults stated; correct me before Phase 0)

1. **Target platform** — macOS/Apple Silicon per A1? (Default: yes; a Linux GPU port would be a
   different plan.)
2. **Build/test machine** — which Mac runs the gates, and are weights already at `~/models`
   there? (Default: user's Mac; `download.sh gguf` + `official-small` when needed.)
3. **Reference source** — is the original h3-image implementation available to consult, or is
   this a pure rebuild from SPEC.md + donor? (Default: pure rebuild; the spec is complete enough.)
4. **ANE scope** — include Phase 11 in v1 or defer? (Default: defer; GPU path first.)
5. **Commit cadence** — one commit per phase gate on `fresh`? (Default: yes, on your go.)

## 9. Immediate next action (on approval)

Execute Phase 0 (staged script + Makefile + skeleton), run the Linux verification gates,
report results, then proceed to Phase 0b/1.
