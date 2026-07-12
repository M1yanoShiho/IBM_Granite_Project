#!/bin/bash
# Validity suite (V1+V2+V3) over the frozen 300q NIAH task — ONE job that (a) produces
# the WS-0 dumps (dense + q2d full top-100 rankings via run_niah --dump-runs; the
# corroboration runs WITH raw answer strings via tune_corroboration --dump-runs), then
# (b) runs the three zero-GPU validity analyses on them in-place:
#   V1 any-gold re-score      (eval.anygold_rescore)   — metric-critique sensitivity
#   V2 injection removal      (eval.injection_analysis) — burial: our construction vs natural
#   V3 natural conflict rate  (eval.injection_analysis) — does the wild disagree at all?
# Consistency checks expected in this log: dense needle-found@10 ~= 0.493, q2d ~= 0.563,
# alpha*=0.6 blend ~= 0.61 (the certified numbers; deviations mean something is wrong).
# Pre-registration: docs/hpc-run-log.md "Validity suite" entry.
#
# Submit from the project root AFTER `git pull`:
#   mkdir -p logs results && sbatch scripts/run_validity_suite.slurm
#   # args: [TASK] [TAG]   (defaults: results/niah_nq300_frozen.json, nq300v)
#SBATCH --job-name=niah-validity
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# 3B extraction + q2d transform; the sole rtx_3090 drains forever -> A100 40GB MIG slice.
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:-results/niah_nq300_frozen.json}"
TAG="${2:-nq300v}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
# NOTE: no GRANITE_MODEL_ID override — default 3B is the certified extractor config.

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "===== [1/4] run_niah dense+q2d with --dump-runs (WS-0 item 3) ====="
python -m eval.run_niah \
    --task "$TASK" \
    --retrievers granite_dense q2d_granite \
    --k 10 --index-type hnsw \
    --out "results/niah_eval_${TAG}.csv" \
    --per-query-out "results/niah_eval_${TAG}_per_query.csv" \
    --cache-dir /user/work/$USER/index_cache \
    --dump-runs "results/niah_${TAG}_runs.json"
cat "results/niah_eval_${TAG}.csv"

echo "===== [2/4] tune_corroboration with answers-bearing --dump-runs ====="
python -m eval.tune_corroboration \
    --task "$TASK" --first-stage q2d_granite --k 10 --alpha-step 0.1 \
    --out "results/corroboration_alpha_curve_${TAG}.csv" \
    --per-query-out "results/corroboration_perq_${TAG}.csv" \
    --dump-runs "results/corroboration_runs_${TAG}.json"

echo "===== [3/4] V1 any-gold re-score ====="
python -m eval.anygold_rescore \
    --task "$TASK" \
    --runs "results/niah_${TAG}_runs_granite_dense.json" "results/niah_${TAG}_runs_q2d_granite.json" \
    --tune-dump "results/corroboration_runs_${TAG}.json" --alpha 0.6 \
    --k 10 --out-prefix "results/anygold_${TAG}"

echo "===== [4/4] V2 injection removal + V3 natural conflict rate ====="
python -m eval.injection_analysis \
    --task "$TASK" \
    --runs "results/niah_${TAG}_runs_granite_dense.json" "results/niah_${TAG}_runs_q2d_granite.json" \
    --tune-dump "results/corroboration_runs_${TAG}.json" \
    --k 10 --out-prefix "results/injection_${TAG}"

echo "===== significance: any-gold deltas (vs dense; vs q2d pool) ====="
python -m eval.significance --per-query-csv "results/anygold_${TAG}_anygold_per_query.csv" --reference granite_dense
python -m eval.significance --per-query-csv "results/anygold_${TAG}_anygold_per_query.csv" --reference q2d_pool
echo "===== DONE (tag=${TAG}) — record: git add -f results/*${TAG}* ====="
