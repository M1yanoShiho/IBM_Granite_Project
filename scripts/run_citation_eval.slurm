#!/bin/bash
# Citation-prompt A/B evaluation — "does model self-citation beat post-hoc attribution?"
#
# Same retriever (granite_dense) × same dataset (NQ) × same 100-query subset;
# ONLY the RAG prompt template changes (DEFAULT_RAG_PROMPT vs CITATION_RAG_PROMPT).
# The delta isolates the citation mechanism — post-hoc token-overlap attribution
# vs the model explicitly citing ``Evidence: [i]`` chunk numbers — so it is a
# clean single-variable experiment. The RAG quality metrics (EM/F1/context
# precision/faithfulness) are already computed inside each run via
# eval/rag_metrics.evaluate_rag (with ``cited_doc_ids`` supplied on the citation
# arm); eval/citation_eval.py additionally compares citation format-compliance
# between the two prediction dumps.
#
# GPU needed: Granite 8B generative model for 100 answers × 2 arms = 200 generations.
# This is a SHORT job (~30 min) compared to run_rag.slurm's 6-arm × 500q sweep.
#
# ONE-TIME on the LOGIN node (compute nodes are offline). VERIFY NQ exposes
# answers + qrels, then pre-fetch the dpr-w100 corpus + 8B generator (same as
# run_rag.slurm — reuse the same /user/work caches):
#   source scripts/env.sh
#   python - <<'PY'
#   import ir_datasets
#   ds = ir_datasets.load("dpr-w100/natural-questions/dev")
#   q = next(ds.queries_iter())
#   print("nq has_qrels:", ds.has_qrels(), "| answers:", getattr(q, "answers", None))
#   PY
#   for m in ibm-granite/granite-4.1-8b ibm-granite/granite-embedding-english-r2; do hf download "$m"; done
#
# Then submit from the project root:
#   mkdir -p logs && sbatch scripts/run_citation_eval.slurm
#
# AFTER the job, the format-compliance comparison JSON is in ``logs/<jobname>-<jobid>.out``.
# OPTIONAL: re-run eval/citation_eval.py on a login node or locally with
# ``--retrieved-context`` (per-query CSV) to enable full citation precision/recall/F1
# (see eval/citation_eval.py docstring).
#SBATCH --job-name=citation-eval
#SBATCH --account=coms039904
#SBATCH --partition=gpu                 # 14-day cap; needs 8B generator
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1           # 24GB, fits 8B bf16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00                  # 2 arms x 100 generations (short)
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

# Subset size — same 100q as the user-specified commands; flat index OK at ~1M.
MAX_QUERIES=100
MAX_DOCS=1000000
RETRIEVER=granite_dense         # the system under test; baseline peers in run_rag.slurm

mkdir -p data/nq results

# ----------------------------------------------------------------------
# Arm 1 — DEFAULT_RAG_PROMPT (post-hoc token-overlap attribution).
# ----------------------------------------------------------------------
echo "===== citation-eval: BASELINE (default prompt) retriever=$RETRIEVER ====="
python -m eval.run_rag --dataset nq --retriever "$RETRIEVER" \
    --max-queries "$MAX_QUERIES" --max-docs "$MAX_DOCS" \
    --predictions-out results/citation_baseline

# ----------------------------------------------------------------------
# Arm 2 — CITATION_RAG_PROMPT (model self-citation + Evidence: [i]).
# ----------------------------------------------------------------------
echo "===== citation-eval: CITATION-PROMPT retriever=$RETRIEVER ====="
python -m eval.run_rag --dataset nq --retriever "$RETRIEVER" \
    --max-queries "$MAX_QUERIES" --max-docs "$MAX_DOCS" \
    --predictions-out results/citation_prompt --citation-prompt

# ----------------------------------------------------------------------
# Dump qrels to TREC-format TSV for eval/citation_eval.py —
# qrels lives in ir_datasets (not a static file), so we materialise it once.
# ----------------------------------------------------------------------
echo "===== citation-eval: dumping qrels to data/nq/qrels.tsv ====="
python - <<'PY'
from pathlib import Path
from eval.benchmarks.loader import load_benchmark

bench = load_benchmark("nq", max_queries=100, max_docs=1_000_000)
out = Path("data/nq/qrels.tsv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for qid, rels in bench.qrels.items():
        for doc_id, rel in rels.items():
            f.write(f"{qid} 0 {doc_id} {int(rel)}\n")
print(f"wrote {out} ({sum(len(r) for r in bench.qrels.values())} judgements)")
PY

# ----------------------------------------------------------------------
# Comparison — citation format-compliance between baseline and citation arms.
# Full precision/recall/F1 needs --retrieved-context from a per-query CSV
# (uncomment the EXTRA= line below and re-submit if you want it).
# ----------------------------------------------------------------------
BASELINE_JSONL=results/citation_baseline_${RETRIEVER}.jsonl
CITATION_JSONL=results/citation_prompt_${RETRIEVER}.jsonl

echo "===== citation-eval: comparison ====="
echo "baseline: $BASELINE_JSONL"
echo "citation: $CITATION_JSONL"
echo "qrels:    data/nq/qrels.tsv"

# EXTRA="--retrieved-context results/citation_baseline_${RETRIEVER}.csv"   # optional: per-query CSV

python eval/citation_eval.py \
    --baseline "$BASELINE_JSONL" \
    --citation "$CITATION_JSONL" \
    --qrels data/nq/qrels.tsv
#   ${EXTRA:+--retrieved-context "$EXTRA"}   # uncomment to enable full citation scoring

echo "===== citation-eval: DONE ====="