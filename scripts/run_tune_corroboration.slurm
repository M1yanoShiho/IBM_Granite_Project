#!/bin/bash
# Offline lambda-sweep for the corroboration reranker: extract answers ONCE over the
# first-stage (q2d) pool, then sweep the relevance/corroboration blend as pure arithmetic
# and write the needle-found@k-vs-alpha curve. alpha=1.0 is pure q2d (a consistency check
# = it should match q2d's needle-found). The curve answers "does ANY alpha beat pure q2d?"
# WITHOUT trusting one blind default alpha.
#
#   # fast FIRST-READ on a 100-query subset (full 300q times out; extraction is the cost):
#   mkdir -p logs results && sbatch scripts/run_tune_corroboration.slurm results/niah_nq300_frozen.json nq100 0.1 100
#   # full sweep (no cap) once the first-read looks promising:
#   sbatch scripts/run_tune_corroboration.slurm results/niah_nq300_frozen.json nq300
#   # 8B extractor instead of the default 3B:
#   GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b sbatch scripts/run_tune_corroboration.slurm results/niah_nq300_frozen.json nq300_8b 0.1 100
#
# Prefetch on the LOGIN node (one-time): the Granite dense embedder + the generator LLM:
#   hf download ibm-granite/granite-embedding-english-r2
#   hf download ibm-granite/granite-4.1-3b        # (or -8b if you override GRANITE_MODEL_ID)
#SBATCH --job-name=niah-tunecorrob
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch scripts/run_tune_corroboration.slurm <task.json> [tag] [alpha-step] [max-queries]}"
TAG="${2:-nq}"
ALPHA_STEP="${3:-0.1}"
MAXQ="${4:-}"
EXTRA=()
[ -n "$MAXQ" ] && EXTRA=(--max-queries "$MAXQ")

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# The dense index is cached (--cache-dir), so q2d's first stage reuses the same index
# built for the run_niah / scale jobs (no re-embed).
python -m eval.tune_corroboration \
    --task "$TASK" \
    --first-stage q2d_granite \
    --k 10 --top-n 20 --alpha-step "$ALPHA_STEP" "${EXTRA[@]}" \
    --index-type hnsw \
    --cache-dir /user/work/$USER/index_cache \
    --out "results/corroboration_alpha_curve_${TAG}.csv" \
    --per-query-out "results/corroboration_perq_${TAG}.csv" \
    --dump-runs "results/corroboration_runs_${TAG}.json"

echo "=================== alpha curve (needle-found@10 vs alpha) ==================="
cat "results/corroboration_alpha_curve_${TAG}.csv"
echo "=================== significance: q2d_corroborate (alpha*) vs q2d (alpha=1) ==================="
python -m eval.significance --per-query-csv "results/corroboration_perq_${TAG}.csv" --reference q2d
echo "=================== DONE ==================="
echo "(runs dumped to results/corroboration_runs_${TAG}.json -> re-analyse OFFLINE with"
echo " python -m eval.tune_corroboration --from-runs results/corroboration_runs_${TAG}.json ...)"
