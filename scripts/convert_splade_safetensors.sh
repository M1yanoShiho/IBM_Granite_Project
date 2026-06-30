#!/bin/bash
# One-time, on the BluePebble LOGIN node: convert the SPLADE weights to safetensors.
#
# Why: transformers refuses torch.load on the HPC's torch 2.5.1 (< 2.6) for CVE-2025-32434,
# and naver/splade-cocondenser-ensembledistil ships ONLY pytorch_model.bin (no safetensors).
# RAW torch.load is still allowed by torch 2.5.1 itself (only the transformers loader blocks
# it), so we read the .bin here and re-save it as model.safetensors into a standalone dir;
# the run then loads that via use_safetensors=True (no torch.load).
#
#   source scripts/env.sh && bash scripts/convert_splade_safetensors.sh
#
# Produces $SPLADE_MODEL_ID (default /user/work/$USER/splade_st), which run_splade_hybrid.slurm
# points at by default.
set -euo pipefail

MODEL="${SPLADE_HF_MODEL:-naver/splade-cocondenser-ensembledistil}"
DST="${SPLADE_MODEL_ID:-/user/work/$USER/splade_st}"

hf download "$MODEL"
SRC=$(dirname "$(find "$HF_HOME" -path "*${MODEL##*/}*" -name config.json | head -1)")
echo "source snapshot: $SRC"
echo "destination:     $DST"

mkdir -p "$DST"
cp -L "$SRC"/*.json "$DST"/ 2>/dev/null || true   # config + tokenizer config
cp -L "$SRC"/*.txt "$DST"/ 2>/dev/null || true    # vocab.txt

SRC="$SRC" DST="$DST" python - <<'PY'
import os
import torch
from safetensors.torch import save_file

src, dst = os.environ["SRC"], os.environ["DST"]
# Raw torch.load (torch 2.5.1 allows it; transformers' from_pretrained is what blocks it).
state = torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
# .clone() breaks tied-weight storage sharing (BERT MLM ties decoder <-> embeddings),
# which safetensors.save_file otherwise rejects. transformers re-ties on load, so it is correct.
state = {k: v.clone().contiguous() for k, v in state.items()}
out = os.path.join(dst, "model.safetensors")
save_file(state, out)
print("wrote", out)
PY

echo "Done. SPLADE_MODEL_ID=$DST"
