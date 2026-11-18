# h3-image

Local text-to-image and image editing with **MiniMax-H3-Image** on Apple Silicon —
a from-spec rebuild in C / Objective-C / Metal, with an optional Neural Engine
(FP16 Core ML) accelerator path. No Python and no network at run time.

**Status:** rebuild in progress. See `SPEC.md` (the complete behavioural
specification) and `PLAN.md` (milestones, acceptance gates A1–A14, workflow).

## Build (macOS 15+, Apple Silicon)

```sh
make            # h3-image + build/h3i_shaders.metal
make test       # full suite (weight-dependent tests skip when files are absent)
make test-quick
make tools
```

## Portable core tests (Linux or macOS)

```sh
make core-tests   # parsers, dequant CPU reference, layout, RNG, tokenizer, …
```

## Usage (target CLI, SPEC §6.1)

```sh
h3-image -m MODEL_DIR -p "a red fox in the snow" -o fox.png
h3-image -m MODEL_DIR -p "make it night <Picture 1>" -i ref.png -o edit.png
h3-image -m MODEL_DIR -p "…" --neural-engine -o fast.png
h3-image --info
```

## Weights

GGUF set (~24 GiB, 16–32 GB machines) or the official BF16 release (~114 GiB);
any mix works. `tools/download.sh` fetches them. Weights are under the
MiniMax-H3-Image Community License (regional exclusions and commercial
thresholds apply — see SPEC §1.8).

## Licence

Code: MIT. Lineage: h3.c (MIT, Salvatore Sanfilippo), ccv (BSD-3), ggml (MIT);
diffusers / stable-diffusion.cpp used as math references only (SPEC §1.8).
