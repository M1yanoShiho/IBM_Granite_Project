#!/bin/bash
# E1: edge/cluster component eval over the injected NIAH pools
# (spec docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md).
# Reuses E2's candidate_sets.jsonl (pools carry text); one Granite extraction pass over each
# injected query's top_n window -> missed/false-conflict + needle-gold-recovery + selection-bias.
#
# One-time on the login node before submitting (E2 must have produced the pool dump):
#   ls runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl
# Submit from the project root (args: CANDIDATES MANIFEST PROVENANCE OUTPUT [MODEL] [EXTRACTION] [LIMIT]):
#   mkdir -p logs results && sbatch scripts/run_cluster_eval.slurm \
#     runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e1-cluster-eval/cluster_eval_report.json
# S5 cascade check (8B + decoupled, subsampled; needs --gres=gpu:a100:1 for 8B):
#   sbatch --gres=gpu:a100:1 scripts/run_cluster_eval.slurm \
#     runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e1-cascade-8b-decoupled.json \
#     ibm-granite/granite-4.1-8b decoupled 300
#SBATCH --job-name=cluster-eval
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
USAGE="usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT [MODEL] [EXTRACTION] [LIMIT]"
CANDIDATES="${1:?$USAGE}"
MANIFEST="${2:?$USAGE}"
PROVENANCE="${3:?$USAGE}"
OUTPUT="${4:?$USAGE}"
MODEL="${5:-ibm-granite/granite-4.1-3b}"
EXTRACTION="${6:-single}"
LIMIT="${7:-0}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID="$MODEL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== E1 edge/cluster component eval | model=$MODEL extraction=$EXTRACTION limit=$LIMIT ==="
if [ "$LIMIT" -gt 0 ]; then
  python -m evidence_rag.evaluation.cluster_eval_cli \
    --candidates "$CANDIDATES" --manifest "$MANIFEST" --provenance "$PROVENANCE" \
    --output "$OUTPUT" --extraction "$EXTRACTION" --limit "$LIMIT"
else
  python -m evidence_rag.evaluation.cluster_eval_cli \
    --candidates "$CANDIDATES" --manifest "$MANIFEST" --provenance "$PROVENANCE" \
    --output "$OUTPUT" --extraction "$EXTRACTION"
fi

echo "=== E1 report ==="
cat "$OUTPUT"
