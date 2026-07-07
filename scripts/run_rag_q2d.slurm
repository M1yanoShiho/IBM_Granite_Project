#!/bin/bash
# Closing the loop: does the retrieval raiser (Query2Doc, finding 13) translate into
# better GENERATED, gold-matched answers? Same RAG pipeline + generator + prompt; only
# the RETRIEVER varies: granite_dense (baseline) vs q2d_granite (the certified raiser).
# Metric = cover-EM / F1 vs gold answers, paired significance across the same queries.
#
# Scope note (be honest in the write-up): this runs on plain NQ (dpr-w100), NOT the NIAH
# counterfactual haystack (run_rag does not consume the NIAH task). So it shows "better
# retrieval -> better answers" on the natural QA set — a clean end-to-end 'so what' for
# finding 13. The stronger "does it survive counterfactual distractors" test needs a
# NIAH->run_rag adapter (a separate build).
#
# Both arms use DIFFERENT retriever names, so run_rag's per-query CSV (column-per-retriever)
# accumulates both columns in ONE file per metric -> feed eval.significance directly.
#
# Reuses the run_rag prefetch (8B generator + Granite dense embedder + dpr-w100 NQ). Submit
# from the project root AFTER `git pull`:
#   mkdir -p logs results && sbatch scripts/run_rag_q2d.slurm
#   # bigger subset (matches Table 3's 500): sbatch scripts/run_rag_q2d.slurm nq500 500
#SBATCH --job-name=granite-rag-q2d
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# 8B bf16 needs ~18-20GB (q2d also does an LLM query-transform per query). The only
# rtx_3090 (bp1-gpu030) is often drained; target the A100 40GB MIG slice (bp1-gpu035).
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TAG="${1:-nq300}"
MAXQ="${2:-300}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b   # instruct (no -base) for RAG + q2d transform
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

DATASET=nq
MAX_DOCS=1000000        # flat index OK to ~1M; same subset as the run_rag Table 3

# Same generator/prompt/top_k for both arms; only --retriever differs. Shared
# --per-query-out prefix -> one CSV per metric with a granite_dense + q2d_granite column.
for R in granite_dense q2d_granite; do
    echo "===== RAG retriever=$R dataset=$DATASET maxq=$MAXQ ====="
    python -m eval.run_rag --dataset "$DATASET" --retriever "$R" --pipeline vanilla \
        --max-queries "$MAXQ" --max-docs "$MAX_DOCS" \
        --out "results/rag_q2d_${TAG}_${R}_agg.csv" \
        --per-query-out "results/rag_q2d_${TAG}_cmp" \
        --predictions-out "results/rag_q2d_${TAG}_${R}_pred"
done

echo "===== significance: q2d_granite vs granite_dense (cover-EM) ====="
python -m eval.significance --per-query-csv "results/rag_q2d_${TAG}_cmp_answer_cover.csv" --reference granite_dense
echo "===== significance: q2d_granite vs granite_dense (F1) ====="
python -m eval.significance --per-query-csv "results/rag_q2d_${TAG}_cmp_answer_f1.csv" --reference granite_dense
echo "===== DONE (tag=${TAG}) ====="
