#!/bin/bash
# G3 follow-up diagnosis: is the completeness loop broken, or just its firing rule?
# (local/guides/G3-followup-diagnosis-guide.md; runner scripts/g3_diagnosis.py)
#
# Re-runs ONLY the verified-full arm on the identical G3 case set with per-gap /
# citation-origin / abstention-reason instrumentation, plus persistence of the
# TRUE per-pair probabilities the production path discards. Greedy decoding
# (temperature=0.0 -> do_sample=False) makes this a faithful reproduction of the
# G3 verified-full behaviour, not a new experiment -- the runner prints an
# answered-count sanity check against G3's 125/378.
#
# Why a GPU job at all, when the guide says Tasks 1-3 are re-analysis: the G3
# dumps are aggregate (query_id + 4 metrics) plus final answers/citations. They
# carry no gaps, no citation provenance, no abstention reason, and no TRUE
# probabilities, so the diagnosis cannot be computed from them. This ONE job
# serves Tasks 1, 2, 3 and 5 together (the guide asks that Task 5's GPU work be
# bundled rather than queued separately).
#
# Models are already cached from G1/G2/G3: granite-4.1-3b, t5_xxl_true_nli_mixture,
# MiniCheck-Flan-T5-Large. Granite+TRUE co-reside for generation (~28GB, proven in
# G2), then MiniCheck loads for judging.
#
# Submit from the project root:
#   mkdir -p logs results/g3-diagnosis && sbatch scripts/run_g3_diagnosis.slurm \
#     docs/generator/g3-followup-diagnosis.md results/g3-diagnosis/diagnosis.jsonl
#SBATCH --job-name=g3-diagnosis
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
REPORT="${1:?usage: sbatch run_g3_diagnosis.slurm REPORT DUMP}"
DUMP="${2:?usage: sbatch run_g3_diagnosis.slurm REPORT DUMP}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== G3 diagnosis: instrumented verified-full, judge MiniCheck ==="
python scripts/g3_diagnosis.py \
  --limit 400 \
  --top-k 5 \
  --seed 13 \
  --dump "$DUMP" \
  --report "$REPORT"

echo "=== report ==="
cat "$REPORT"
