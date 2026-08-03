#!/bin/bash
# Route B Task 3: validate the LLM checklist analyzer before it drives anything.
# (local/guides/route-B-llm-checklist-guide.md; gate pre-registered in the G4
# ledger entry, commit 2f4ca16, BEFORE any number was produced.)
#
# Granite builds one checklist per question from the question text alone; MiniCheck
# then matches each generated requirement against ASQA's gold disambiguations.
# qa_pairs are evaluation-side only and never reach Generator input.
#
#   mkdir -p logs results && sbatch scripts/run_checklist_validation.slurm
#SBATCH --job-name=checklist-val
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

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python scripts/checklist_validation.py \
  --limit 400 --top-k 5 --seed 13 \
  --report docs/generator/checklist-validation.md \
  --dump results/checklist-validation.jsonl
