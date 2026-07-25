#!/bin/bash
# Phase 3: prompt probe for the twin missed-conflict (systematic-debugging).
# Extracts the needle and counterfactual passages under baseline vs targeted prompts and reports
# missed-conflict / needle-gold / cf-replacement per prompt. ~2 x 3 LLM calls per injected query
# (~9k, ~15-20 min on 3B). Decides: a targeted prompt that lowers missed-conflict gets Graph 2.0's
# conflict-detection benefit without NLI; if none does, Graph 2.0 is justified.
#
# gres is gpu:1 (the MIG slice was unavailable for the E1/E2 runs); a full GPU schedules and 3B fits.
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_missed_conflict_probe.slurm \
#     runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e3-missed-conflict-probe.json
#SBATCH --job-name=missed-conflict-probe
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MANIFEST="${1:?usage: sbatch run_missed_conflict_probe.slurm MANIFEST PROVENANCE OUTPUT}"
PROVENANCE="${2:?usage: sbatch run_missed_conflict_probe.slurm MANIFEST PROVENANCE OUTPUT}"
OUTPUT="${3:?usage: sbatch run_missed_conflict_probe.slurm MANIFEST PROVENANCE OUTPUT}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== Phase 3 twin missed-conflict prompt probe ==="
python -m evidence_rag.evaluation.missed_conflict_probe_cli \
  --manifest "$MANIFEST" \
  --provenance "$PROVENANCE" \
  --output "$OUTPUT"

echo "=== probe report (missed_conflict_rate per prompt = the number) ==="
cat "$OUTPUT"
