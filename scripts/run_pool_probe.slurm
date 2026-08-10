#!/bin/bash
# Pool-shaped probe for the R013 relation model: one claim against a real top-20 window.
#
# Everything R013 has been judged on is one passage against one claim -- the OOF, the surface
# baseline, the single-side check, and Gate 0B's 0B-2 probe as well. S6 is why that is not
# enough: a method that looked good on a two-passage probe put false conflicts up 38.3 points
# on a real twenty-passage pool. This measures the same axis before anything is claimed about
# the selector.
#
# LOGIN NODE first (seconds, no GPU) -- every path here must exist, and an absent one costs a
# queue wait from inside the job:
#   for p in runs/r013/seed-13/fold-0 runs/niah-injected/manifest.json \
#            runs/niah-injected/provenance.jsonl runs/e2-gate-off/candidate_sets.jsonl; do
#     [ -e "$p" ] && echo "OK   $p" || echo "MISSING  $p"; done
#
# Submit from the project root. Positional args, in order:
#   CHECKPOINT  OUTPUT_JSON
#   mkdir -p logs results/r013
#   sbatch scripts/run_pool_probe.slurm runs/r013/seed-13/fold-0 results/r013/pool-probe-seed13.json
#
# AFTERWARDS: results/ is gitignored, so `git add -f` the JSON or the reading exists only here.
#SBATCH --job-name=pool-probe
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# Inference only, so bf16 is irrelevant and any card would do -- but rtx_3090 is the
# uncontested one and 18330608 has just shown it turns around in about an hour.
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CHECKPOINT="${1:?usage: sbatch run_pool_probe.slurm CHECKPOINT OUTPUT_JSON}"
OUTPUT="${2:?usage: sbatch run_pool_probe.slurm CHECKPOINT OUTPUT_JSON}"
if [ "$#" -gt 2 ]; then
  echo "ERROR: $# positional args; expected exactly 2 (CHECKPOINT OUTPUT_JSON)." >&2
  exit 2
fi

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true HF_HUB_DISABLE_TELEMETRY=1
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

# Compute nodes have no git -- 18330608 printed "[tree] unknown +uncommitted" because both
# fallbacks fired at once, not because the tree was dirty. So .git/HEAD is read directly: it is
# plain text, and the sha is the part that matters. Dirty state is not checkable without git and
# is therefore not claimed.
python - <<'PY'
from pathlib import Path
head = Path(".git/HEAD").read_text(encoding="utf-8").strip()
if head.startswith("ref: "):
    ref = Path(".git") / head[5:]
    head = ref.read_text(encoding="utf-8").strip() if ref.is_file() else f"unresolved {head[5:]}"
print(f"[tree] {head}")
PY

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no device')"

echo "=== pool probe: $CHECKPOINT -> $OUTPUT ==="
python scripts/pool_relation_probe.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-manifest runs/niah-injected/manifest.json \
  --provenance runs/niah-injected/provenance.jsonl \
  --candidates runs/e2-gate-off/candidate_sets.jsonl \
  --output "$OUTPUT"
