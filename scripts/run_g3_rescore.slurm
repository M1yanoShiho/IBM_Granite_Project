#!/bin/bash
# G3 remediation Tasks 1 + 2: sentence-level re-scoring, and the per-sentence
# citation baseline. (local/guides/G3-remediation-guide.md)
#
# Task 1 re-scores the EXISTING G3 run with ALCE's sentence-level citation
# metrics -- no chain is re-run, the arms are reconstructed from the G3 and
# diagnosis dumps (reconstruction validated offline: counts 373/218/125 match G3
# exactly, verified-full answers rebuild byte-identical 125/125, and no sentence
# is left unattributed). Only the MiniCheck judging needs a GPU.
#
# Task 2 generates the per-sentence-citation baseline variant: Granite only, no
# TRUE, 400 generations. Bundled here so both land in one job.
#
# Judge is MiniCheck. TRUE is never used -- it selected the verified arms'
# citations, so scoring with it would be circular.
#
# NLTK (ALCE's sentence splitter) and its punkt data are installed in the venv at
# /user/work/$USER/nltk_data.
#
# Submit from the project root:
#   mkdir -p logs results/g3-remediation && sbatch scripts/run_g3_rescore.slurm
#SBATCH --job-name=g3-rescore
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export NLTK_DATA=/user/work/$USER/nltk_data
export PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

OUT=results/g3-remediation
mkdir -p "$OUT"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== Task 2: per-sentence citation baseline (Granite only) ==="
python scripts/g3_baseline_persentence.py \
  --limit 400 --top-k 5 --seed 13 \
  --dump "$OUT/baseline-persentence.jsonl"

echo "=== Task 1: sentence-level re-scoring of all arms (judge MiniCheck) ==="
python scripts/g3_sentence_rescore.py \
  --diagnosis results/g3-diagnosis/diagnosis.jsonl \
  --baseline-subsample results/g3/human-subsample.jsonl \
  --baseline-persentence "$OUT/baseline-persentence.jsonl" \
  --seed 13 \
  --report docs/generator/g3-sentence-level-rescore.md \
  --dump "$OUT/per-example-scores.jsonl"

echo "=== report ==="
cat docs/generator/g3-sentence-level-rescore.md
