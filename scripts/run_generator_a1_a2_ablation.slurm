#!/bin/bash
# G-A12: historical A1 prompt x A2 splitter 2x2 ablation.
# BEFORE: docs/hpc-run-log.md, section G-A12. Calibration data only.
# Submit from the repository root:
#   mkdir -p logs results/generator-a1-a2-ablation
#   sbatch scripts/run_generator_a1_a2_ablation.slurm results/generator-a1-a2-ablation
#
# Login-node prerequisites (do not run inference there):
#   module load languages/python/3.11.15
#   source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache
#   hf download ibm-granite/granite-4.1-3b
#   # ensure ALCE/ASQA was already extracted under TRIAGE_DATA_DIR
#
# All four arms run in this one job. Never combine arms from separate jobs.
#SBATCH --job-name=g-a12-ablation
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
OUTDIR="${1:?usage: sbatch run_generator_a1_a2_ablation.slurm OUTPUT_DIR}"

module load languages/python/3.11.15
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTDIR"
python -c "import sys, torch; print('Python:', sys.version); print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"
python scripts/generator_a1_a2_ablation.py \
  --limit 50 \
  --top-k 5 \
  --seed 13 \
  --model-id ibm-granite/granite-4.1-3b \
  --output-dir "$OUTDIR"

test -s "$OUTDIR/manifest.json"
test -s "$OUTDIR/raw_responses.jsonl"
test -s "$OUTDIR/outputs_by_arm.jsonl"
test -s "$OUTDIR/annotation_blinded.csv"
test -s "$OUTDIR/annotation_key.json"
test -s "$OUTDIR/automatic_summary.json"
echo "G-A12 raw run complete: $OUTDIR"
