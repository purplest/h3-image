#!/bin/sh
# h3-image one-command verification.
#
#   sh tools/verify.sh [OFFICIAL_DIR] [GGUF_DIR]
#
# Defaults: ~/models/MiniMax-H3-Image (official BF16) and
# ~/models/MiniMax-H3-Image-gguf (GGUF set; the spec-default -GGUF spelling
# is accepted too).
#
# macOS: builds lib + tests + tools, extracts the official block-0 check
# fixture from the LOCAL transformer shards (no download), exports the
# weight paths the tests read, and runs the full suite (make check).
# Linux: runs the portable core suite + syntax gates.
cd "$(dirname "$0")/.." || exit 1

if [ "$(uname -s)" != "Darwin" ]; then
    make linux-check && make core-tests
    exit $?
fi

OFFICIAL="${1:-$HOME/models/MiniMax-H3-Image}"
GGUF="${2:-$HOME/models/MiniMax-H3-Image-gguf}"
[ -d "$GGUF" ] || GGUF="$HOME/models/MiniMax-H3-Image-GGUF"

echo "== official: $OFFICIAL"
echo "== gguf:     $GGUF"

make lib tests-build tools || exit 1

# --- locate weight files (globs accept any quant variant) ---
DIT=$(ls "$GGUF"/minimax_h3_image_pruned-*.gguf 2>/dev/null | head -1)
TE=$(ls "$GGUF"/qwen3vl_32b_minimax_h3-*.gguf 2>/dev/null | head -1)
VAE16=$(ls "$GGUF"/vae/*.safetensors 2>/dev/null | head -1)
VCFG="$GGUF/vae/config.json"
[ -f "$VCFG" ] || VCFG="$OFFICIAL/vae/config.json"
PDD="$OFFICIAL/pdd_fused_heads.safetensors"
VAE32=$(ls "$OFFICIAL"/vae/*.safetensors 2>/dev/null | head -1)
CHECK="$OFFICIAL/check/official_block0.safetensors"

# --- block-0 fixture from local official shards (seconds, no network) ---
if [ ! -f "$CHECK" ] && [ -d "$OFFICIAL/transformer" ]; then
    echo "== extracting official block-0 fixture from local shards"
    python3 tools/extract_tensors.py "$OFFICIAL/transformer" "$CHECK" \
        time_embedder.proj_in.weight time_embedder.proj_in.bias \
        time_embedder.proj_out.weight time_embedder.proj_out.bias \
        blocks.0.adaln_proj.linear.weight blocks.0.adaln_proj.linear.bias \
        final_layer.adaln_proj.linear.weight final_layer.adaln_proj.linear.bias \
        || exit 1
fi

# --- export the paths the tests read (only what exists; rest skips) ---
[ -n "$DIT" ]   && export H3I_DIT_GGUF="$DIT"
[ -n "$TE" ]    && export H3I_TE_GGUF="$TE"
[ -n "$VAE16" ] && export H3I_VAE="$VAE16"
[ -f "$VCFG" ]  && export H3I_VAE_CONFIG="$VCFG"
[ -f "$PDD" ]   && export H3I_PDD_HEADS="$PDD"
[ -f "$CHECK" ] && export H3I_OFFICIAL_CHECK="$CHECK"
[ -n "$VAE32" ] && export H3I_VAE_F32="$VAE32"

echo "== test weight paths:"
for v in H3I_DIT_GGUF H3I_TE_GGUF H3I_VAE H3I_VAE_CONFIG H3I_PDD_HEADS \
         H3I_OFFICIAL_CHECK H3I_VAE_F32; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && echo "   $v=$val"
done

make check
