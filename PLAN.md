# h3-image — implementation plan

Status: **in progress** (decisions resolved, see §9). Source of truth: `SPEC.md` (committed
alongside this plan). Where this plan and SPEC.md disagree, SPEC.md wins; where SPEC.md and
the original reference code disagree, the spec's Part 7 notes apply.

**Milestone status** (updated 2026-11-19):
- M0 scaffolding — ✅ committed `15e4a3f` (12 checks)
- M2 weights, CPU half — ✅ committed `d03763e` (A1 CPU bit-exact vs vendored ggml oracle;
  GPU convert half lands with M1)
- M3 layout + host math — ✅ committed `d03763e` (A2 bit-exact vs independent Python vectors;
  267 checks total, 0 failures on Linux)
- M1 GPU runtime — next; first Mac sync point covers M1+M2+M3
  (`test_gpu_ops`, `test_gguf` GPU half, `test_weights`, `test_layout`, real-weight checks)
- M4–M12 — pending

Target: a single C CLI (`h3-image`, version `0.1.0`) for **Apple Silicon, macOS 15+**,
rebuilding MiniMax-H3-Image text-to-image and editing exactly as specified: GGUF + official
BF16 weights (any mix), SSD weight streaming (~3 GiB peak), deterministic byte-identical PNGs
per seed, and the optional `--neural-engine` FP16 Core ML path with overflow handling.

Completion = acceptance criteria **A1–A14** (SPEC §1.5). Out of scope: prompt-expansion model,
training, CFG/negative prompts, Python at run time, PyTorch-identical RNG.

---

## 1. Environment reality and build/test workflow

This plan is being written on an **x86_64 Linux** box (Ubuntu 24.04, no compiler yet).
The target is macOS/Apple Silicon. Consequences:

- Metal / MPSGraph / Core ML / ANE / ImageIO / vImage code **cannot be compiled or run here**.
- A large part of the codebase is portable C and **can** be developed and unit-tested here:
  GGUF/safetensors parsers, CPU dequantization, weight source + model detection, canvas/layout/
  tile math, RNG, rounding helpers, tokenizer (ICU), modulation host math, head rebuilding,
  edit host math, and the ANE protobuf writer (`ane/spec.c` is pure C).

**One codebase, two build targets:**

| Target | What builds | Where |
|---|---|---|
| `make` (macOS) | full program per SPEC §6.4: clang C11 + ObjC (ARC), all frameworks, `libicucore`, `build/h3i_shaders.metal`, `libh3image.a`, tests, tools | the Mac |
| `make core-tests` (Linux) | portable subset + CPU unit tests (parsers, dequant CPU, layout, RNG, tokenizer, modulation/heads host math, ANE spec writer); Apple-only files excluded via the Makefile and `#ifdef __APPLE__` | this box (install clang/gcc + libicu-dev first) |

**Mac loop options (decision D1, §9):**
1. **SSH from this box to the Mac** (preferred if available): I run `make`, tests and generations
   remotely; tightest iteration.
2. **Git sync**: I commit here; you pull on the Mac, run `make test`, paste failures back.
3. **Move the session to the Mac**: sync this repo there and run Claude Code natively.

Git: repo initialized on `main`; one commit per milestone at its gate (plus smaller WIP commits);
commit messages end with the Claude Code co-author line. Tag `v0.1.0` when A1–A14 pass.
Remote (GitHub or other): to be decided; none configured yet.

## 2. Repository layout (SPEC §1.3 module map)

```
SPEC.md  PLAN.md  README.md  Makefile  .gitignore
src/
  app/      main.c pipeline.c edit.c paths.c error.h progress.h clock.h
  model/    dit.c dit_ane.c dit_debug.c dit_internal.h modulation.c layout.c
            text_encoder.c vision.c vae.c vae_encoder.c
  weights/  gguf.c safetensors.c source.c stream.c model.c
  ane/      ane.m spec.c
  io/       tokenizer.c host.c image.m terminal.m
  gpu/      context.m tensor.m ops_h3i.m gpu.h shaders.metal kernels.metal
tests/      test_*.c  test_cli.sh  check_images.sh  data/
tools/      inspect.c bench_gpu.c profile_dit.c gpu_determinism.c te_repeat.c
            fetch_tensors.py ane_models.py modelscope_get.py download.sh bench.sh
outputs/    (gitignored: generated images, benchmarks, check-image runs)
```

Dependencies point downward only: `app → model → weights → gpu`, with `io` and `ane` alongside.
Weights live outside the repo (`~/models/MiniMax-H3-Image-GGUF`, `~/models/MiniMax-H3-Image`).

## 3. Cross-cutting conventions (binding, from SPEC)

- **Language**: C11 everywhere; Objective-C only at Apple API boundaries (Metal device/context,
  MPSGraph, Core ML, ImageIO, vImage), compiled with `-fobjc-arc`. Public names `h3i_`, env `H3I_`.
- **Errors**: every fallible function takes `(char *error, size_t error_size)`, writes one
  printf-formatted message via `h3i_fail`, returns 0/NULL. CLI prints `h3-image: <message>`,
  exit 1 runtime / 2 usage. All message strings byte-exact per SPEC §6.
- **Numerics**: `bf16()` = RNE bit trick (`bits += 0x7FFF + ((bits>>16)&1)`), no NaN special
  case; `f16()` per §4.0. F32 ops separately rounded — host math compiled `FP_CONTRACT OFF`,
  Metal compiled **safe-math** (no fast-math). `fma` only where the spec writes fma
  (dequant `a*q − m`, F32 linear accumulation). BF16 op boundary: read BF16, compute F32,
  round once at output.
- **Determinism**: PCG32 (XSH-RR) + Box–Muller exactly per §3.13, spare carries across fills;
  one RNG per request; draw order = [edit posterior noise per ref] → target noise →
  [edit t=0.999 noise per ref].
- **Memory lifetime**: only one of {text encoder, VAE encoder, DiT, VAE decoder} resident at a
  time; each freed at stage end (§6.2.1). Streamer ordering contract (§2.7): when streamed,
  submit+wait per block before acquiring the next.
- **Attention**: DiT / refiner / vision / VAE use MPSGraph SDPA (non-causal). Text encoder uses
  the **custom race-free causal GQA kernel**: exact softmax ≤ 4096 keys, chunked F32 online
  softmax beyond; never a generic SDPA substitute (§1.4 rule 4).
- **QKV layout**: chunked for pruned/GGUF, grouped for official; detected by `adaln_t_table`
  presence; `H3I_QKV_LAYOUT` overrides. GGUF dims reversed at indexing so all consumers see
  PyTorch `[out, in]`.
- **Kernels**: we do not have h3.c; all ~50 Metal pipelines (§6.3.4) are written from the spec's
  exact formulas (§3.9, §4.0, §5.2, §6.3.5). MPSGraph handles large linear/MLP/SDPA/conv3d with
  the spec's routing thresholds; custom kernels for everything else.
- **Tokenizer NFC/whitespace**: ICU C API on both platforms (`libicucore` on macOS, `libicu-dev`
  on Linux) — same NFC results as Foundation's `precomposedStringWithCanonicalMapping`.
- **Licence**: MIT for our code (upstream lineage h3.c MIT, ccv BSD-3, ggml MIT — attribution in
  README). Weights: MiniMax-H3-Image Community License (note its regional exclusions). Optionally
  vendor ggml's dequant functions (MIT) as the A1 test oracle.
- **No Python at run time**; Python only in `tools/` (downloads, tensor fetch, ANE reference).

## 4. Milestones (SPEC §1.7 build order — each gate must pass before the next starts)

### M0 — Scaffolding
- git repo (done at plan time), `.gitignore`, README stub.
- Makefile: macOS target exactly per §6.4 (flags, frameworks, archive, shader concatenation,
  `all/test/test-quick/check-images/tools/clean`); Linux `core-tests` target for the portable subset.
- Install Linux toolchain (clang or gcc, libicu-dev) — needs sudo.
- `src` tree, `app/error.h`, `app/clock.h` (monotonic), `app/paths.c` skeleton.
- **Gate**: skeleton compiles; `core-tests` runs on Linux; macOS build verified once D1 is set.

### M1 — GPU runtime and operations (§6.3)
- `context.m`: device, one serial queue, runtime shader compile (safe math), eager pipelines,
  MPSGraph caches keyed by shape/dtype, `begin/continue/submit` semantics, in-flight list,
  stats, `H3I_GPU_PROFILE`, `H3I_DEBUG_GPU_MEMORY`.
- `tensor.m`: shared-storage tensors (F32/BF16/I8/U32), host ptr, from/read/load helpers,
  `read_file_bytes` with `F_NOCACHE`, thread-safe.
- `shaders.metal` + `kernels.metal`: all custom kernels, exact math and dispatch shapes.
- `ops_h3i.m`: full op table (§6.3.5) incl. MPSGraph routing (rows≥32, in≥256, out≥256),
  tiled F32 patch-linear special case (out=5376, in∈{32,96}), validation + exact error strings.
- `tools/bench_gpu.c` microbenchmarks.
- **Gate**: `test_gpu_ops` within tolerances; ~3 TFLOP/s matmul target on M3-class GPU. **Mac only.**

### M2 — Weight formats and dequantization (§2; A1)
- `gguf.c`: v2/v3, 1 MiB→4× prefix re-parse strategy, metadata skip rules (alignment only),
  tensor-info validation, exact error strings (§2.2–2.4).
- `safetensors.c`: hand-written JSON parser with the spec's rejection order, escape handling
  (no surrogate combining), overflow checks (§3).
- `source.c`: flat records, insertion-order lookup, one-way prefix aliases, `read_f32`,
  `load` (raw copy or GPU convert), `read_raw` (pread/EINTR/uncached) (§4).
- CPU dequant reference for all 9 types (§5.2, FMA where written) + GPU convert kernels and
  dispatch (§5.3); §5.1 BF16 rounding shared.
- `model.c`: directory classifier (three probe tensors), `find_kind` lexicographic, resolution
  order, final checks, DiT source assembly + `dit_heads` append rule (§6).
- `tools/inspect.c`, `tools/download.sh` + `modelscope_get.py` (so weights can be fetched early).
- **Gate**: `test_weights`, `test_gguf` — CPU↔GPU bit-identical for every type, vs ggml oracle
  (A1). CPU half testable on Linux; GPU half on Mac.

### M3 — Layout and host math (§3.2–3.6, §3.13, §4.0; A2)
- `layout.c`: `canvas_resolve` (double math, ties-to-even, 1.25×MAX_AREA rejection),
  patchify/unpatchify (exact row/value order), `layout_build` (grid() with separately-rounded
  ops, tags, time axes), `tile_axis_plan`, `unit_to_u8` (truncation, NaN→0).
- `host.c`: PCG32 + Box–Muller, `bf16()`/`f16()`, `schedule_build` (double→float casts exactly),
  RoPE table helpers, resize wrapper (vImage high-quality + edge extend on Mac).
- **Gate**: `test_layout` bit-exact. **Fully Linux-testable.**

### M4 — VAE decoder and image I/O (§4.5, §4.8; A10 decode half)
- `vae.c`: F32 24-layer ViT, 4 registers + zero token, per-tile RoPE (100^·0.125 freqs),
  per-head q/k RMS (no weight), scale1/scale2 residuals, SwiGLU, tile plan + raw-neighbour
  stitch blend, un-normalize with config/built-in mean/std, uint8 conversion.
- `image.m`: ImageIO read (frame 0, EXIF orientation table, own color space, alpha over black),
  PNG write (8-bit RGB, sRGB, top-down).
- **Gate**: `test_vae` decode checks — F16 vs F32 ≥ 70 dB (needs VAE weights; fixture substitute
  in §7). **Mac only.**

### M5 — Modulation and output heads (§3.7, §3.11; A5, A6)
- `modulation.c`: official path (Timesteps 256 flip_sin_to_cos, F32 time MLP with fma accum,
  SiLU in F32 **then** BF16 cast, BF16 adaln matmul) and pruned path (F32 table lerp, sequential
  mul+add sum, **no SiLU**); row maps (`time_row*3 + tag`, refs at time row 6).
- Heads: absolute `[6,96,5376]` vs GGUF base+offset rebuild in F32; S=6 validation; exact errors.
- `tools/fetch_tensors.py` (range-fetch `official_block0.safetensors` and other check tensors).
- **Gate**: `test_modulation` — heads rel < 1e-5 (A6), pruned vs official ≤ 0.2 % (A5),
  wrong-SiLU control 4× worse. Host math Linux-testable with weights; full check on Mac.

### M6 — DiT, weight providers, streaming (§3.8–3.10, §3.12, §2.7; A3, A4, A7)
- `stream.c`: layer-major tensors, two staging sets, one read-ahead thread (`uncached=1`),
  resident zero-copy BF16, decode slots, `double_slots`, cyclic wrap rules, `quiesce`,
  `H3I_NO_PREFETCH`, wait-seconds accounting (§7).
- `dit.c`: format detection (adaln_t_table / time_embedder / error), QKV layout detection,
  `condition_proj` + 2 refiner blocks (no AdaLN, no RoPE), 50 blocks (AdaLN → QKV split →
  q/k norm → 3-axis RoPE (inv_freq from checkpoint) → SDPA → gate → MLP → gate), final AdaLN +
  F32 head, Euler update (two rounded ops, no fma), resident/stream auto rule
  (Σ bytes > 0.5 × recommended working set), `H3I_WEIGHTS`, per-block submit+wait when streamed.
- `dit_debug.c`: `H3I_CAPTURE*`, `H3I_RANGE`.
- **Gate**: `test_dit_weights` (A3: chunked QKV rel L2 < 0.2 vs official block 0, wrong layout
  > 1.0), `test_dit_block` (A4: ≈3 %), `test_dit` + `test_stream` (A7: streamed == resident
  bit-identical). **Mac + GGUF weights** (+ official block-0 check file).

### M7 — Tokenizer and text encoder (§4.1–4.2; A8, A9)
- `tokenizer.c`: BPE load/validation (NFC, no unk, added-token flags), byte encoder,
  hand-coded Qwen2 pre-tokenizer (6 rules, ICU whitespace), merge-rank loop with piece cache,
  added-token scanning, caps (4096 t2i truncate+warn; per-piece edit).
- `text_encoder.c`: host embedding row gather (dequant→bf16), 50 layers **always streamed**
  (50×11), RMSNorm eps 1e-6, per-head q/k norm, rotate-half mRoPE (theta 5e6, BF16 tables,
  interleaved sections 24/20/20 for edits), custom `gqa_causal_bf16` kernel (race-free max,
  ≤4096 exact softmax, chunked online beyond), SwiGLU, deepstack add hook (wired in M10),
  **no final norm**, `H3I_PROFILE` SSD line.
- **Gate**: tokenizer unit tests (**Linux**); `test_attention` at 43 and 8200 tokens (A9);
  `test_text` finite/causal/repeatable (A8). **Mac + TE GGUF.**

### M8 — Pipeline and CLI (§6.1–6.2; A11 partial, A14)
- `pipeline.c`: stage order T1/T2 (t2i) or E1/E2/E3 (edit) → D → V, exact memory lifetimes,
  RNG draw order, progress phases/strings, timings, `H3I_DUMP_DIR`, request validation.
- `main.c`: left-to-right exact-match parsing, every option/validation/exit code/message,
  `--info`, `--profile`, `--show` (`terminal.m`), banner/usage/summary byte-exact, env vars.
- **Gate**: `tests/test_cli.sh` all cases (A14); first images — fox 512² (`--min-area 262144`)
  and 1024², coherent, same seed → byte-identical PNG (A11 partial). **Mac + GGUF.**

### M9 — Official BF16 end to end (A12)
- Verify mixed-format assembly (official transformer shards + `pdd_fused_heads.safetensors`
  append rule, official TE 64-layer file using first 50, grouped QKV, time_embedder modulation).
- **Gate**: `bf16_gpu` 1024² coherent, same scene/layout as the GGUF run with the same seed (A12).
  Needs `official-transformer` + `official-text-encoder` downloads (~118 GiB).

### M10 — Editing (§4.3–4.4, §4.7; edit example, test_vision, test_edit)
- `vision.c`: 27 blocks, merge-window patch order (channel→temporal→py→px, duplicated frame),
  bilinear pos embed (BF16 rounding per term), 2D RoPE (36 freqs, half 36), GELU-tanh blocks,
  final merger (per-patch LN) vs deepstack mergers (merged-row LN) at L8/16/24, GELU-erf
  (Abramowitz–Stegun 7.1.26), outputs merged + 3 deepstack [M,5120].
- `edit.c`: reference load/sizing (`bound_short_edge`, VAE 1024–2048, vision 1024), presentation
  (`Picture i: ` labels, vision_start/pad/end ids, tags), mRoPE `get_rope_index`, reference
  latents (posterior f16 rounding, even crop, normalize), `scale_noise` t=0.999, draw order.
- `vae_encoder.c`: causal 3D CNN single-frame reduction (last temporal tap), reflect padding,
  GroupNorm32+SiLU, MPSGraph conv3d (NDHWC/OIDHW), downsample pad asymmetry, tiling + latent
  stitch, moments [48][lh][lw].
- **Gate**: official edit example (`-i examples/reference1.png --prompt-file
  examples/revised_prompt.edit.txt`) coherent; `test_edit` host math (partly Linux);
  `test_vision` (fixture substitute §7). **Mac + TE GGUF (vision) + VAE + examples.**

### M11 — Neural Engine path (§5; A13)
- `ane/spec.c`: hand-written proto3 `.mlpackage` writer — exact field numbers/order (§5.3),
  Manifest.json exact bytes, four MIL programs (qkv/out/mlpin/down, CHUNK=1792 split sums,
  sequential FP16 adds, CoreML8 opset, spec version 9). **Pure C — byte-testable on Linux**
  against `tools/ane_models.py` (coremltools) output.
- `ane.m`: scratch package → `compileModelAtURL` → cache `<DIR>/v1/r<ROWS>/<name>.mlmodelc`,
  race-tolerant move, CPUAndNeuralEngine load + shape/dtype validation, no-copy
  `MLMultiArray` wraps, `outputBackings`, copy-back fallback counter, `ane_rows` (ceil/64·64).
- Handoff kernels: `bf16_to_f16`, `f16_to_bf16_checked` (atomic flag), `transpose_bf16_f16`
  (32×33 threadgroup tiles); power-of-two scales only.
- `dit_ane.c`: initial exponents (QKV 0, OUT 2, MLP 4; b39/b45→8, b43→6), growth +2 max 14
  then permanent GPU fallback, flag bits 1/2/4, per-block GPU↔NE schedule with next-block
  prefetch overlap (§5.6), double slots, `w16_up_exponent[set]` rescale semantics, zeroed
  padding rows, `--profile` NE report (§5.8).
- **Gate**: `test_handoff` bit-exact; `test_ane` (all non-const ops on NE via MLComputePlan,
  rel L2 < 0.3 % / < 3 % mlpin, repeatability, cache reuse, rows alignment); `gguf_ane`
  PSNR ≥ ~30 dB vs GPU path, deterministic (A13). **Mac, macOS 15+.**

### M12 — Hardening and release (throughout)
- Remaining tools: `profile_dit`, `gpu_determinism`, `te_repeat`, `bench.sh`, `ane_models.py`.
- `tests/check_images.sh` (record/full/quick) with our recorded baselines (§7).
- README (build, downloads, usage, licence/provenance), performance tuning vs §1.6 targets,
  full `make test` + `check-images full`, tag `v0.1.0`.

## 5. Acceptance matrix

| A# | Check | Test(s) | Platform | Weights needed |
|---|---|---|---|---|
| A1 | dequant bit-exact | test_gguf | CPU: Linux+Mac; GPU: Mac | synthetic (+real blocks optional) |
| A2 | canvas/positions/patchify/tiles | test_layout | Linux+Mac | none |
| A3 | GGUF vs official block-0 matrices | test_dit_weights | Mac | GGUF DiT + official_block0 |
| A4 | DiT block 0 BF16 vs Q4_K | test_dit_block | Mac | both |
| A5 | pruned vs official AdaLN | test_modulation | host: Linux; full: Mac | both |
| A6 | heads base+offset vs absolute | test_modulation | Linux+Mac | GGUF + pdd_fused_heads |
| A7 | streamed == resident | test_dit, test_stream | Mac (stream: synthetic) | GGUF DiT |
| A8 | text encoder finite/causal/repeatable | test_text | Mac | TE GGUF + tokenizer |
| A9 | causal attention 43 / 8200 | test_attention | Mac | none |
| A10 | VAE F16 vs F32, round trip | test_vae | Mac | VAE (+ example for encode) |
| A11 | end-to-end images + determinism | check_images.sh, manual | Mac | GGUF (+ official for edit example) |
| A12 | official BF16 end to end | check_images bf16_gpu | Mac | official transformer + TE |
| A13 | NE path | test_ane, test_handoff, gguf_ane | Mac (macOS 15+) | GGUF |
| A14 | CLI | test_cli.sh | Mac (parse-only subset Linux) | none |

## 6. Missing fixtures and substitutes (honesty note)

SPEC references test artifacts that exist only in the reference repo and are **not derivable
from the spec text**:

| Artifact | Substitute |
|---|---|
| `tests/data/vae_latent_512.f32` (saved normalized latent) | Regenerate with our own VAE encoder (M10) from `examples/reference1.png` at 512²; until then M4 gate uses a synthetic latent for finiteness + the F16-vs-F32 ≥70 dB check (which needs only VAE weights). |
| `tests/data/vision_512.sha256` (reference vision output hash) | Replace with repeatability + finiteness + tokens/grid checks. Exact SHA only reproducible if you can supply the reference fixture. |
| `outputs/check-images/*.png` (reference baselines) | Record our own baselines (`check_images.sh record`); determinism gate = byte-identical reruns of our build. Cross-implementation byte equality with the reference is not claimable without its outputs. |
| `check/official_block0.safetensors` | Reproducible: `tools/fetch_tensors.py` HTTP range-fetch from HF. |

`examples/reference1.png` and `examples/revised_prompt.edit.txt` are downloadable
(`official-small` stage), so the edit example itself is fully reproducible.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No Mac in the loop → Metal/ANE unverifiable from here | Decision D1; portable core + CPU tests on Linux keep M2/M3/M5/M7-tokenizer/M11-writer moving regardless |
| Dequant not bit-exact (historically the main bug source) | CPU reference first, FMA exactly where written, GPU cross-check per type, ggml oracle vendored for A1 |
| Causal attention race (>32 keys) / coarse softmax | Dedicated kernel per §6.3.5; test_attention at 43 and 8200 tokens; never reuse generic SDPA |
| NE FP16 overflow (inf at 2^15; residual ~2×10⁷) | Per-op power-of-two divisors, checked conversion + atomic flag, GPU fallback, exponent persistence (§5.5); test_ane + PSNR gate |
| macOS < 15 on target | NE path needs CoreML8/outputBackings (macOS 15+); GPU-only path still deliverable, A13 deferred |
| vImage resize not bit-exact across implementations | Spec accepts close match; A11 determinism is per-machine, unaffected |
| Weight downloads (24 GiB GGUF / ~118 GiB official) + disk | download.sh free-space checks; GGUF first; official stages only when M9/M10 need them |
| Vision sizing edge case (Part 7 "bug?") | Reproduce as specified by default; flag to you before "fixing" |
| MPSGraph SDPA unavailable on old OS | Exact error string per spec; macOS 15+ target avoids it |

## 8. Performance targets (measured on M3 Air, informational)

512² ≈ 36 s; 1024² ≈ 2.4 min GPU / 70 s NE; 2048² ≈ 8.5 min GPU / 4.6 min NE; peak ≈ 3 GiB
(GPU) / 6.5 GiB (NE). Matmul ≈ 3 TFLOP/s. Streaming hides SSD reads (0.1–0.3 s unhidden).
Tune in M12 with `bench_gpu` / `profile_dit` / `bench.sh`; targets, not gates.

## 9. Decisions (resolved 2026-11-18)

- **D1 — Mac workflow**: **Git sync.** Development and commits happen on the Linux box;
  the user pulls on the Mac, runs `make` / `make test` and pastes results back. The user
  offered to run any weight-dependent test on request. Mac-side gates are batched into
  sync points (first: after M1–M3) to keep the async loop efficient; Linux-verifiable
  work proceeds while a Mac gate is pending.
- **D2 — Weights**: **both** GGUF set and official BF16 release are already on the Mac.
  All acceptance tests A1–A14 are runnable; no downloads needed (tools/download.sh still
  built for completeness). Exact paths on the Mac to be confirmed at first sync point
  (defaults: `~/models/MiniMax-H3-Image-GGUF`, `~/models/MiniMax-H3-Image`).
- **D3 — Target Mac**: **macOS 15+** — full plan including the NE path (M11, A13) is viable.
- **D4 — Approval**: **approved** — start M0, then M1/M2 per the build order.
