#!/bin/bash
# Astute RAG measurement — does generation-stage source-aware consolidation
# (elicit -> per-source consolidate -> finalise; src/rag_pipeline.AstuteRAGPipeline)
# beat vanilla single-shot RAG on ANSWER quality, holding retriever + generator +
# prompt + top_k fixed? Only `--pipeline` varies, so the cover-EM / F1 delta
# isolates the consolidation step. Paired significance over the same query subset.
#
# Scope note (be honest in the write-up): this runs on plain NQ (dpr-w100), whose
# distractors are the natural near-miss passages the retriever pulls, NOT injected
# Source-A counterfactuals. So it measures "does consolidation help general RAG",
# a fair vanilla-vs-astute answer-quality test — not a clean counterfactual-
# robustness test (that would need the NIAH counterfactual corpus fed to run_rag,
# which is not wired yet).
#
# Reuses the run_rag prefetch (8B instruct generator + Granite dense embedder +
# dpr-w100 NQ). If you have run scripts/run_rag.slurm before, nothing new to
# download; otherwise do that script's one-time LOGIN-node prefetch first.
#
# Submit from the project root, AFTER `git pull` (needs eval.run_rag --pipeline astute):
#   mkdir -p logs results && sbatch scripts/run_astute_rag.slurm
#   # match Table 3's 500-query subset instead of the 300 first-read:
#   sbatch scripts/run_astute_rag.slurm nq500 500
#SBATCH --job-name=granite-astute-rag
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# GPU: 8B bf16 needs ~18-20GB. BluePebble's ONLY rtx_3090 (bp1-gpu030) is frequently
# drained for maintenance, so default to the A100 40GB MIG slice (bp1-gpu035) which has
# plentiful VRAM and more availability. Fallbacks: gpu:V100:1 (ONLY if 32GB), or drop to
# 3B (GRANITE_MODEL_ID=...granite-4.1-3b) on gpu:rtx_2080:1 (11GB) for instant scheduling
# -- the vanilla-vs-astute DELTA is valid at any model as long as both arms share it.
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00                  # astute = several generations/query; give headroom
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TAG="${1:-nq300}"
MAXQ="${2:-300}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b   # instruct (no -base) for RAG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

DATASET=nq
RETRIEVER=granite_dense
MAX_DOCS=1000000        # flat index OK to ~1M; same subset as the run_rag Table 3

# Same retriever/generator/prompt for both arms; only the pipeline flow differs.
# Separate --out / --per-query-out per arm (run_rag labels every column by the
# RETRIEVER name, which is granite_dense for both — so the arms must not share a file).
for P in vanilla astute; do
    echo "===== RAG pipeline=$P dataset=$DATASET retriever=$RETRIEVER maxq=$MAXQ ====="
    python -m eval.run_rag --dataset "$DATASET" --retriever "$RETRIEVER" \
        --pipeline "$P" \
        --max-queries "$MAXQ" --max-docs "$MAX_DOCS" \
        --out "results/astute_${TAG}_${P}_agg.csv" \
        --per-query-out "results/astute_${TAG}_${P}" \
        --predictions-out "results/astute_${TAG}_${P}_pred"
done

# Merge the two arms' per-query columns (both named "granite_dense", in separate
# files) into one 2-column table per metric so eval.significance can pair them.
echo "===== merge vanilla vs astute per-query (cover-EM, F1) ====="
python - "$TAG" <<'PY'
import csv, sys
from pathlib import Path
tag = sys.argv[1]
def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return {r[0]: r[1] for r in rows[1:]}          # qid -> score (the single retriever col)
for metric in ("answer_cover", "answer_f1"):
    vp = Path(f"results/astute_{tag}_vanilla_{metric}.csv")
    ap = Path(f"results/astute_{tag}_astute_{metric}.csv")
    if not (vp.exists() and ap.exists()):
        print(f"SKIP {metric}: missing {vp if not vp.exists() else ap}")
        continue
    van, ast = load(vp), load(ap)
    qids = sorted(set(van) & set(ast))
    out = Path(f"results/astute_{tag}_{metric}_cmp.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "vanilla", "astute"])
        for q in qids:
            w.writerow([q, van[q], ast[q]])
    print(f"wrote {out}  ({len(qids)} paired queries)")
PY

echo "===== aggregate cover-EM / F1 (vanilla vs astute) ====="
cat "results/astute_${TAG}_vanilla_agg.csv"
cat "results/astute_${TAG}_astute_agg.csv"
echo "===== significance: astute vs vanilla (cover-EM) ====="
python -m eval.significance --per-query-csv "results/astute_${TAG}_answer_cover_cmp.csv" --reference vanilla
echo "===== significance: astute vs vanilla (F1) ====="
python -m eval.significance --per-query-csv "results/astute_${TAG}_answer_f1_cmp.csv" --reference vanilla
echo "===== DONE (tag=${TAG}) ====="
