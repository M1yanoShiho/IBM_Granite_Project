#!/bin/bash
# RAG evaluation — "does better retrieval -> better answers", across TWO QA sets.
# Three retrievers over the SAME corpus, scored on answer EM / F1 / cover-EM +
# qrels-based context precision + faithfulness (eval/rag_metrics.py):
#   bm25 (weak lexical anchor) -> gte_dense (strong fair peer) -> granite_dense (system)
# so we show answer quality TRACKS retrieval quality AND compare Granite to a fair
# peer (gte). Two datasets test generalisation: NQ and TriviaQA — both served from
# the SAME dpr-w100 ~21M-passage corpus, so TriviaQA adds no new corpus download.
#
# Per dataset we run a qrels-AWARE subset: the gold passages of the sampled queries
# are always kept, plus up to MAX_DOCS distractor passages. flat index is fine to
# ~1M docs; the true 21M run needs HNSW (a separate step). Scale MAX_DOCS once sane.
#
# ONE-TIME on the LOGIN node (compute nodes are offline). VERIFY each set exposes
# answers + qrels, then pre-fetch — the dpr-w100 corpus is tens of GB (download once;
# both QA sets reuse it):
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   python - <<'PY'
#   import ir_datasets
#   for ds_id in ("dpr-w100/natural-questions/dev", "dpr-w100/trivia-qa/dev"):
#       ds = ir_datasets.load(ds_id); q = next(ds.queries_iter())
#       print(ds_id, "has_qrels:", ds.has_qrels(), "| answers:", getattr(q, "answers", None))
#   PY
#   for m in ibm-granite/granite-4.1-8b thenlper/gte-base \
#            ibm-granite/granite-embedding-english-r2; do hf download "$m"; done
#
# Then submit from the project root:
#   mkdir -p logs && sbatch scripts/run_rag.slurm
#
# AFTER the job, per dataset D in {nq, trivia}: aggregate results/rag_${D}_subset.csv;
# paired p-values via eval/significance.py on results/rag_${D}_subset_perq_<metric>.csv
# (answer_f1 / answer_cover / answer_em / context_precision / faithfulness).
#SBATCH --job-name=granite-rag-qa
#SBATCH --account=coms039904
#SBATCH --partition=gpu                 # 14-day cap; RAG loads the 8B generator
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1           # 24GB, fits 8B bf16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                  # 2 datasets x 3 retrievers x 500 generations
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

# Subset size — scale MAX_DOCS up once sane (flat index OK to ~1M; 21M needs HNSW).
MAX_QUERIES=500
MAX_DOCS=1000000

# Two QA sets (generalisation) x three retrievers, on the shared dpr-w100 corpus.
# Per dataset: aggregate table ($OUT, rm'd first so re-runs don't pile up rows) +
# merged per-query CSVs (--per-query-out, for eval.significance) + prediction
# dumps (--predictions-out, to eyeball what the model actually answered).
for D in nq trivia; do
    OUT=results/rag_${D}_subset.csv
    PERQ=results/rag_${D}_subset_perq
    PRED=results/rag_${D}_subset_pred
    rm -f "$OUT"
    for R in granite_dense gte_dense bm25; do
        echo "===== RAG dataset=$D retriever=$R ====="
        python -m eval.run_rag --dataset "$D" --retriever "$R" \
            --max-queries "$MAX_QUERIES" --max-docs "$MAX_DOCS" \
            --out "$OUT" --append --per-query-out "$PERQ" --predictions-out "$PRED"
    done
done
