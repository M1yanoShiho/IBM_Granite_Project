#!/bin/bash
# 2x2 prompt/model probe for the twin missed-conflict (systematic-debugging).
# Extracts needle+cf under baseline / attribute / decoupled strategies and reports missed-conflict per
# strategy, for whatever model MODEL selects. Run once per model for the 2x2 (3B vs 8B) x (single-stage
# attribute vs decoupled). --limit subsamples for a fast pilot (~500 is decisive for the direction).
#
# One-time before the 8B run: hf download ibm-granite/granite-4.1-8b   (on the login node)
# gres is gpu:1 (the MIG slice was unavailable for the E1/E2 runs); a full GPU schedules and 3B/8B fit.
# Submit from the project root (args: MANIFEST PROVENANCE OUTPUT [MODEL] [LIMIT]):
#   mkdir -p logs results
#   # 3B pilot (500):
#   sbatch scripts/run_missed_conflict_probe.slurm runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e3-2x2-3b.json ibm-granite/granite-4.1-3b 500
#   # 8B pilot (500):
#   sbatch scripts/run_missed_conflict_probe.slurm runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e3-2x2-8b.json ibm-granite/granite-4.1-8b 500
#SBATCH --job-name=missed-conflict-probe
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MANIFEST="${1:?usage: sbatch run_missed_conflict_probe.slurm MANIFEST PROVENANCE OUTPUT [MODEL] [LIMIT]}"
PROVENANCE="${2:?usage: sbatch run_missed_conflict_probe.slurm MANIFEST PROVENANCE OUTPUT [MODEL] [LIMIT]}"
OUTPUT="${3:?usage: sbatch run_missed_conflict_probe.slurm MANIFEST PROVENANCE OUTPUT [MODEL] [LIMIT]}"
MODEL="${4:-ibm-granite/granite-4.1-3b}"
LIMIT="${5:-0}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID="$MODEL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"
echo "=== 2x2 twin missed-conflict probe | model=$MODEL limit=$LIMIT ==="

if [ "$LIMIT" -gt 0 ]; then
  python -m evidence_rag.evaluation.missed_conflict_probe_cli \
    --manifest "$MANIFEST" --provenance "$PROVENANCE" --output "$OUTPUT" --limit "$LIMIT"
else
  python -m evidence_rag.evaluation.missed_conflict_probe_cli \
    --manifest "$MANIFEST" --provenance "$PROVENANCE" --output "$OUTPUT"
fi

echo "=== probe report (missed_conflict_rate per strategy = the number) ==="
cat "$OUTPUT"
