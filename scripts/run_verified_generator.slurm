#!/bin/bash
# G2 Half 2: B4 completeness + B5 full chain, first real-data run.
# (local/guides/G2-guide.md; runner scripts/verified_generator_calibration.py)
#
# Co-residency decision (guide Task 2): the B5 VerifiedGenerator chain re-invokes
# Granite (completeness, evidence-recheck) AFTER the TRUE attribution pass, so the
# two models genuinely interleave within a single query and cannot be split into
# "generate then verify" stages without reimplementing the orchestrator. We do NOT
# hope it fits -- we hold both on ONE full a100 and PROVE it: Granite-4.1-3b
# (~6GB bf16) + TRUE t5_xxl (~21GB bf16) ~= 28GB, and the runner prints
# torch.cuda.max_memory_allocated after the first B5 query and at the end.
#
# This is why a full a100, not a 3g.40gb MIG slice: the double-load OOM this repo
# hit before (scripts/run_selector_gate.slurm) was two models on the 40GB slice;
# a full a100 has the headroom to hold both cleanly.
#
# Models + ASQA are already cached on /user/work from G1 (job 18200000):
#   hf_cache/hub: ibm-granite/granite-4.1-3b, google/t5_xxl_true_nli_mixture
#   triage_data:  asqa_eval_gtr_top100.json
# so no prefetch step is needed. If the cache is cold, run the G1 prefetch first.
#
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_verified_generator.slurm \
#     docs/generator/verified-generator-results-hpc.md results/verified-generator/cases.jsonl
#SBATCH --job-name=verified-generator
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
REPORT="${1:?usage: sbatch run_verified_generator.slurm REPORT DUMP}"
DUMP="${2:?usage: sbatch run_verified_generator.slurm REPORT DUMP}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# cache_dir= expects the hub-level dir (.../models--org--repo), not the HF_HOME
# root -- see the G1 fix in scripts/run_verifier_triage.slurm.
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export PYTHONUNBUFFERED=1
# Generator draft/completeness/recheck run on Granite-3b; the verifier is TRUE.
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export NLI_BACKEND=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== G2: B4 completeness + B5 full chain (Granite-3b + TRUE) ==="
python scripts/verified_generator_calibration.py \
  --stage all \
  --limit 60 \
  --top-k 5 \
  --seed 13 \
  --output "$REPORT" \
  --dump "$DUMP"

echo "=== report ==="
cat "$REPORT"
