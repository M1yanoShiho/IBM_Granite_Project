#!/bin/bash
# E1: edge/cluster component eval over the injected NIAH pools
# (spec docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md).
# Reuses E2's candidate_sets.jsonl (pools carry text); one Granite extraction pass over each
# injected query's top_n window -> missed/false-conflict + needle-gold-recovery + selection-bias.
#
# One-time on the login node before submitting (E2 must have produced the pool dump):
#   ls runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_cluster_eval.slurm \
#     runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e1-cluster-eval/cluster_eval_report.json
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
CANDIDATES="${1:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"
MANIFEST="${2:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"
PROVENANCE="${3:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"
OUTPUT="${4:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== E1 edge/cluster component eval ==="
python -m evidence_rag.evaluation.cluster_eval_cli \
  --candidates "$CANDIDATES" \
  --manifest "$MANIFEST" \
  --provenance "$PROVENANCE" \
  --output "$OUTPUT"

echo "=== E1 report ==="
cat "$OUTPUT"
