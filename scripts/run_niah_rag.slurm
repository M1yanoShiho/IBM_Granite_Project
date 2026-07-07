#!/bin/bash
# End-to-end system test: does better retrieval survive to a correct, gold-matched answer
# despite counterfactual distractors? RAG cover-EM / F1 over the NIAH haystack, across the
# retrieval stack (granite_dense -> q2d_granite -> q2d_corroborate). Same generator/prompt/
# top_k; only the retriever varies. One shared 8B client (transform + rerank + generate).
#
# Reuses the run_rag prefetch (8B generator + Granite dense embedder + dpr-w100 NQ). Submit
# from the project root AFTER `git pull`:
#   mkdir -p logs results && sbatch scripts/run_niah_rag.slurm
#   # a smaller/faster first read (q2d_corroborate is ~20 extractions/query):
#   sbatch scripts/run_niah_rag.slurm nq300 results/niah_nq300_frozen.json
#SBATCH --job-name=granite-niah-rag
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# 8B bf16 + q2d transform + corroboration reranking. The sole rtx_3090 (bp1-gpu030) is
# frequently drained -> target the A100 40GB MIG slice (bp1-gpu035).
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TAG="${1:-nq300}"
TASK="${2:-results/niah_nq300_frozen.json}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b   # instruct (no -base) for RAG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

python -m eval.run_niah_rag \
    --task "$TASK" \
    --retrievers granite_dense q2d_granite q2d_corroborate \
    --top-k 4 \
    --out "results/rag_niah_${TAG}_agg.csv" \
    --per-query-out "results/rag_niah_${TAG}_cmp" \
    --predictions-out "results/rag_niah_${TAG}_pred"

echo "===== aggregate (cover-EM / F1 per retriever) ====="
cat "results/rag_niah_${TAG}_agg.csv"
echo "===== significance: cover-EM vs granite_dense ====="
python -m eval.significance --per-query-csv "results/rag_niah_${TAG}_cmp_answer_cover.csv" --reference granite_dense
echo "===== significance: F1 vs granite_dense ====="
python -m eval.significance --per-query-csv "results/rag_niah_${TAG}_cmp_answer_f1.csv" --reference granite_dense
echo "===== DONE (tag=${TAG}) ====="
