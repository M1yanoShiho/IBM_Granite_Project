#!/bin/bash
# RAG evaluation on an NQ SUBSET — the "does better retrieval -> better answers"
# experiment. Three retrievers over the SAME corpus, scored on answer EM/F1 +
# qrels-based context precision + faithfulness (eval/rag_metrics.py):
#   bm25 (weak lexical anchor) -> gte_dense (strong fair peer) -> granite_dense (system)
# so we can show answer quality TRACKS retrieval quality, and compare Granite to a
# fair peer (gte), not just to the weak BM25 floor.
#
# NQ's corpus (dpr-w100) is ~21M passages — too big to index for a first run. We
# run a qrels-AWARE subset: the gold passages of the sampled queries are always
# kept, plus up to MAX_DOCS distractor passages. 500 queries / 500k docs is large
# enough that the dense-vs-BM25 gap is pronounced and n is ample for the paired
# significance test; scale up once it looks sane.
#
# ONE-TIME on the LOGIN node (compute nodes are offline). First VERIFY the dataset
# exposes answers + qrels (context-precision needs qrels), then pre-fetch it — this
# DOWNLOADS the dpr-w100 corpus (tens of GB):
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   python - <<'PY'
#   import ir_datasets
#   ds = ir_datasets.load("dpr-w100/natural-questions/dev")
#   print("has_qrels:", ds.has_qrels())
#   q = next(ds.queries_iter()); print("query fields:", q._fields, "| answers:", getattr(q, "answers", None))
#   print("doc fields:", next(ds.docs_iter())._fields)
#   PY
#   for m in ibm-granite/granite-4.1-8b thenlper/gte-base \
#            ibm-granite/granite-embedding-english-r2; do hf download "$m"; done
#
# Then submit from the project root:
#   mkdir -p logs && sbatch scripts/run_rag.slurm
#
# AFTER the job, get paired p-values between retriever columns (e.g. granite vs gte
# on answer-F1) by feeding the per-query CSVs to eval/significance.py:
#   results/rag_nq_subset_perq_answer_f1.csv  (also _answer_em / _context_precision / _faithfulness)
#SBATCH --job-name=granite-rag-nq-subset
#SBATCH --account=coms039904
#SBATCH --partition=gpu                 # 14-day cap; RAG loads the 8B generator
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1           # 24GB, fits 8B bf16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Compute nodes have no python by default — load the module the venv was built with.
module load languages/python/3.12.3

export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b   # instruct variant (no -base) for RAG
export PYTHONUNBUFFERED=1

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Subset size — scale up once the first run looks sane.
MAX_QUERIES=500
MAX_DOCS=500000
OUT=results/rag_nq_subset.csv
PERQ=results/rag_nq_subset_perq

# Same corpus, three retrievers, one aggregate table + merged per-query CSVs.
# --append accumulates the aggregate rows; --per-query-out merges each retriever's
# column into <PERQ>_<metric>.csv for the paired significance test.
for R in granite_dense gte_dense bm25; do
    echo "===== RAG retriever: $R ====="
    python -m eval.run_rag --dataset nq --retriever "$R" \
        --max-queries "$MAX_QUERIES" --max-docs "$MAX_DOCS" \
        --out "$OUT" --append --per-query-out "$PERQ"
done
