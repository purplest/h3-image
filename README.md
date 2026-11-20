# h3-image

Native **MiniMax-H3-Image** inference for Apple Silicon — text-to-image and image editing,
no Python, no network at run time. Rebuilt from the local `SPEC.md` specification (kept
out of the repo) on the substrate of
[h3.c](https://github.com/antirez/h3.c) (MIT, Salvatore Sanfilippo, pinned commit `8974cc0`),
the MiniMax-H3 video/audio engine that shares the same DiT architecture.

**Status: Phase 0 (migration) complete.** The shared substrate is vendored under `src/`
with `h3i_` naming; the image-specific modules (GGUF weights, dequantization, streaming,
2D ViT VAE decoder, PDD sampler, pipeline, CLI) land per [PLAN.md](PLAN.md).

## Build (macOS, Apple Silicon)

```sh
make            # build/libh3image.a + build/h3i_shaders.metal (until src/app/main.c lands)
make test       # full test suite (weight-dependent tests skip when files are absent)
make test-quick # layout/weights/stream/gpu-ops/handoff/ane + CLI
make tools      # inspect, benches, profilers
```

Requires clang and the system frameworks (Metal, MetalPerformanceShadersGraph, Accelerate,
ImageIO, CoreGraphics, CoreML) plus ICU (`-licucore`, part of macOS). Shaders are compiled
at run time in safe-math mode from `build/h3i_shaders.metal` (override with `H3I_SHADERS`).

## Layout

```
src/gpu/      Metal runtime, device probe, shaders (h3.c kernel set + h3-image kernels)
src/weights/  safetensors (GGUF, source, detection, streamer land in Phase 2)
src/model/    DiT, modulation, text encoder, vision tower, VAE encoder/decoder
src/io/       host math (RNG, resize), tokenizer, terminal preview (image I/O in Phase 4)
src/app/      CLI + pipeline (Phase 8)
tests/ tools/ per SPEC.md part 6
```

## Licences

Code: MIT (see [LICENSE](LICENSE)); adapted ccv kernels: BSD-3
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Model weights: MiniMax-H3-Image
Community License (region and commercial-use restrictions apply — see the model repo).
