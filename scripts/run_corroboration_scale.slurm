#!/bin/bash
# Experiment C — does the corroboration gain (and the optimal blend alpha) hold as the
# haystack grows? For each corpus size, run the q2d -> corroboration extraction ONCE
# (load_niah_task --max-docs), dump the (relevance, corroboration) runs, then re-analyse
# each size OFFLINE with pure arithmetic:
#   * tune_corroboration writes the per-size alpha curve  -> "does alpha* drift with scale?"
#   * compare_rules writes blend/cascade/lexicographic + MRR at each size (reuses the dump)
#   * significance gives q2d_corroborate(alpha*) vs q2d   -> "does the gain survive scale?"
# The extraction (~top_n LLM calls/query) is the only GPU cost; every re-analysis is free.
#
#   mkdir -p logs results && sbatch scripts/run_corroboration_scale.slurm results/niah_nq300_frozen.json nq300scale
#   # fast first-read (100 queries, small sizes) before committing the full grid:
#   MAXQ=100 SIZES="10000 100000" sbatch scripts/run_corroboration_scale.slurm results/niah_nq300_frozen.json nq300scale_fast
#   # one size per job (if a single job would exceed the walltime):
#   SIZES="5000000" sbatch scripts/run_corroboration_scale.slurm results/niah_nq300_frozen.json nq300scale
#
# Prefetch on the LOGIN node (one-time) = same as run_tune_corroboration.slurm:
#   hf download ibm-granite/granite-embedding-english-r2
#   hf download ibm-granite/granite-4.1-3b
#SBATCH --job-name=niah-corrob-scale
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch scripts/run_corroboration_scale.slurm <task.json> [tag]}"
TAG="${2:-nq300scale}"
SIZES="${SIZES:-10000 100000 1000000 5000000}"
ALPHA_STEP="${ALPHA_STEP:-0.1}"
EPSILON="${EPSILON:-0.1}"
ALPHA="${ALPHA:-0.6}"
MAXQ="${MAXQ:-}"
EXTRA=()
[ -n "$MAXQ" ] && EXTRA=(--max-queries "$MAXQ")

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Smallest-first: each size dumps its runs immediately, so a later OOM keeps earlier work.
# The corpus fingerprint is in the index cache key (post-fix f63e905), so sharing
# --cache-dir across sizes is safe -- a different max_docs is a different fingerprint.
for N in $SIZES; do
  echo "=================== max_docs=$N : extract q2d -> corroboration ==================="
  python -m eval.tune_corroboration \
      --task "$TASK" --first-stage q2d_granite \
      --k 10 --top-n 20 --alpha-step "$ALPHA_STEP" "${EXTRA[@]}" \
      --max-docs "$N" --index-type hnsw \
      --cache-dir /user/work/$USER/index_cache \
      --out "results/corroboration_alpha_curve_${TAG}_${N}.csv" \
      --per-query-out "results/corroboration_perq_${TAG}_${N}.csv" \
      --dump-runs "results/corroboration_runs_${TAG}_${N}.json" \
    || { echo "size $N failed (likely OOM) — continuing"; continue; }

  echo "--- offline: blend vs cascade vs lexicographic (+MRR) at size $N ---"
  python -m eval.compare_rules \
      --from-runs "results/corroboration_runs_${TAG}_${N}.json" \
      --k 10 --alpha "$ALPHA" --epsilon "$EPSILON" --alpha-step "$ALPHA_STEP" \
      --out-dir results --tag "${TAG}_${N}"

  echo "--- significance: q2d_corroborate(alpha*) vs q2d at size $N ---"
  python -m eval.significance --per-query-csv "results/corroboration_perq_${TAG}_${N}.csv" --reference q2d
done

echo "=================== per-size alpha curve (does alpha* drift with scale?) ==================="
for N in $SIZES; do
  f="results/corroboration_alpha_curve_${TAG}_${N}.csv"
  [ -f "$f" ] && { echo "size=$N:"; cat "$f"; echo; }
done
echo "=================== DONE — commit the CSVs: git add -f results/*_${TAG}_*.csv ==================="
