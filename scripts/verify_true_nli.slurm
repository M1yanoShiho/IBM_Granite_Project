#!/bin/bash
# One-off: prove TRUE loads from the converted safetensors AND produces sane scores.
# Header-only verification in the converter checked keys/shapes/dtypes, not VALUES --
# a corrupted weight would load cleanly and score plausibly wrong, which is the exact
# failure shape this project keeps paying for. So score a pair whose answer is known.
#SBATCH --job-name=true-check
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load languages/python/3.12.3
source /user/work/$USER/venv/bin/activate
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export PYTHONUNBUFFERED=1 PYTHONPATH=src
cd /user/work/$USER/IBM_Granite_Project

python - <<'PY'
from evidence_rag.generator.nli import build_nli_model

model = build_nli_model("true")
print("loaded:", type(model).__name__, flush=True)

# Entailed / contradicted / irrelevant. If the converted weights were damaged these
# would not separate -- a model that scores everything ~0.5 loads fine and ruins a run.
cases = [
    ("Marie Curie won the Nobel Prize in Physics in 1903.",
     "Marie Curie won a Nobel Prize.", "ENTAILED  -> expect HIGH"),
    ("Marie Curie won the Nobel Prize in Physics in 1903.",
     "Marie Curie never received a Nobel Prize.", "CONTRADICTED -> expect LOW"),
    ("Marie Curie won the Nobel Prize in Physics in 1903.",
     "The Pacific Ocean is the largest ocean.", "IRRELEVANT -> expect LOW"),
]
scores = []
for premise, hypothesis, label in cases:
    s = model.score(premise, hypothesis)
    scores.append(s)
    print(f"  {s:.4f}  {label}", flush=True)

if not (scores[0] > 0.5 > max(scores[1], scores[2])):
    raise SystemExit(
        f"SEPARATION FAILED {scores} -- the entailed pair must outscore both others "
        "across 0.5. Converted weights are suspect; do NOT resubmit the G6 run."
    )
print("\nTRUE loads and separates correctly. Safe to resubmit G6 generation.")
PY
