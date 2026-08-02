#!/bin/bash
# G3 baseline comparison: verified vs generation-time citation.
# (local/guides/G3-baseline-comparison-guide.md; runner scripts/g3_baseline_comparison.py)
#
# Three arms on the same ASQA queries, same Granite, same selected evidence:
#   baseline / verify-only / verified-full. Judge = MiniCheck (NOT TRUE).
#
# Models: Granite-4.1-3b + TRUE co-reside during generation (~28GB, proven to fit
# on the 3g.40gb MIG slice in G2); MiniCheck loads afterwards for scoring.
#
# One-time on the login node before submitting -- MiniCheck was a LOCAL CPU arm in
# G1 and is NOT in the HPC cache yet, so it must be fetched (Granite-3b + TRUE are
# already cached from G1/G2):
#   export HF_HOME=/user/work/$USER/hf_cache
#   hf download lytang/MiniCheck-Flan-T5-Large
#   # sanity: confirm the ASQA case yield is >= 300 before the full run
#   module load languages/python/3.12.3; source /user/work/$USER/venv/bin/activate
#   export TRIAGE_DATA_DIR=/user/work/$USER/triage_data PYTHONPATH=src
#   python - <<'PY'
#   import sys, random; sys.path.insert(0,'scripts')
#   import g3_baseline_comparison as g3
#   print(len(g3.build_cases(g3.vt.ensure_asqa(), 400, random.Random(13), 5)))
#   PY
#
# Submit from the project root:
#   mkdir -p logs results/g3 && sbatch scripts/run_g3_baseline.slurm \
#     docs/generator/g3-baseline-comparison-hpc.md results/g3
#SBATCH --job-name=g3-baseline
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
REPORT="${1:?usage: sbatch run_g3_baseline.slurm REPORT OUTPUT_DIR}"
OUTDIR="${2:?usage: sbatch run_g3_baseline.slurm REPORT OUTPUT_DIR}"

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

echo "=== G3: baseline / verify-only / verified-full (judge MiniCheck) ==="
python scripts/g3_baseline_comparison.py \
  --limit 400 \
  --top-k 5 \
  --seed 13 \
  --output-dir "$OUTDIR" \
  --report "$REPORT"

echo "=== paired significance (both-answered subset auto: None values dropped) ==="
for METRIC in citation_precision citation_recall coverage answer_correctness; do
  echo "--- $METRIC : verified-full vs baseline ---"
  python -m evidence_rag.evaluation.paired_metric_cli \
    --on-report "$OUTDIR/verified-full-report.json" \
    --off-report "$OUTDIR/baseline-report.json" \
    --metric "$METRIC" --seed 13
  echo "--- $METRIC : verified-full vs verify-only (completeness contribution) ---"
  python -m evidence_rag.evaluation.paired_metric_cli \
    --on-report "$OUTDIR/verified-full-report.json" \
    --off-report "$OUTDIR/verify-only-report.json" \
    --metric "$METRIC" --seed 13
done

echo "=== report ==="
cat "$REPORT"
