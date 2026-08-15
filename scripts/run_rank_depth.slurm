#!/bin/bash
# R18: rerun R16's arm with a much deeper candidate pool and read where the answer-bearing
# chunk actually ranks. R16 could only say "not in the top 50" for 83 of its 119 conditional
# misses; this job replaces that bound with a number, which is what decides whether raising
# `top_k` (or reranking a wider pool) is a viable fix or a hopeless one.
#
# CPU only. The arm is strong-bm25 + extractive: no GPU, no LLM, no embedder is ever
# constructed, so this runs on `compute` and must not queue behind GPU work. The env block
# is copied verbatim from run_pipeline_eval.slurm (same CLI entry point, same venv) with
# LLM_DEVICE flipped to cpu -- nothing in this arm reaches it, and leaving `cuda` on a node
# with no GPU is a landmine for whoever copies this script next.
#
# Memory: `pipeline` holds every run in memory before writing pipeline_runs.jsonl, so
# 2,000 queries x 1,000 candidates is the sizing constraint here, not the retrieval itself
# (~4-6 GB expected; 64G leaves room). Wall time is dominated by serialisation: R14's four
# arms at top_k=50 took 12:39, and BM25 scores the whole corpus regardless of depth.
#
# Submit from the project root:
#   mkdir -p logs results runs && sbatch scripts/run_rank_depth.slurm
#
# The forensics step at the end is the actual readout; the .out log carries it so the
# numbers are first-read from the job rather than reconstructed afterwards.
#SBATCH --job-name=rank-depth
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uz25020@bristol.ac.uk

set -euo pipefail

CONFIG="${1:-configs/experiments/chunk_nq_b-c60o10-topk1000.toml}"
RUN_DIR="${2:-runs/chunk-nq-b-c60o10-topk1000}"
DUMP="${3:-results/r18-nq-c60o10-topk1000-classes.jsonl}"
# Only labels the readout's ">N" line and its last cumulative bin; the pool depth itself
# comes from the config's `top_k`. Keep them in step if the config changes.
DEPTH="${4:-1000}"

# 3.12.3 and the venv from run_pipeline_eval.slurm. This arm never imports GraniteEmbedder,
# but `evidence_rag.cli.experiment` imports the composition module that declares it, and
# 3.11.15 is the mismatch that killed job 18325274 -- matching the script whose CLI this is
# costs nothing and removes the question.
module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export GRANITE_EMBEDDING_MODEL_ID=ibm-granite/granite-embedding-english-r2
export LLM_DEVICE=cpu
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

echo "=== $CONFIG : prepare -> pipeline ==="
python -m evidence_rag.cli.experiment --config "$CONFIG" prepare
python -m evidence_rag.cli.experiment --config "$CONFIG" pipeline

# Falsification A is structural and comes first: raising top_k cannot change selection, so
# the conditional-miss population must be exactly R16's 119. The forensics prints it on the
# `conditional misses:` line; a different number means the two runs are not comparable and
# every rank below it is void.
echo
echo "=== R18 readout: class split and rank depth (R16 anchor: 119 misses, 83/25/11/0) ==="
python scripts/answer_chunk_forensics.py \
  --run "$RUN_DIR" \
  --report "$RUN_DIR/evaluation_report.json" \
  --dump "$DUMP"

# The four rank quartiles the class counts alone cannot give. R16's readout stopped at
# min/median/max because 36 ranks did not support more; at top_k=1000 the distribution is
# the result, so it is printed rather than left to be recomputed from the dump.
echo
echo "=== rank distribution over answer-bearing candidates found in the deeper pool ==="
python - "$DUMP" "$DEPTH" <<'PY'
import json, sys, statistics
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
depth = int(sys.argv[2])
ranks = sorted(r["best_rank"] for r in rows if r.get("best_rank") is not None)
still_missing = sum(1 for r in rows if r["class"] == "retrieval")
print(f"cases with an answer-bearing candidate in the pool : {len(ranks)} of {len(rows)}")
print(f"still no answer-bearing candidate (rank > {depth})  : {still_missing}")
if ranks:
    # `median` is the midpoint of the sorted list, matching answer_chunk_forensics.py
    # exactly. statistics.median/quantiles interpolate and would print 20 where R16
    # published 23 for the same 36 ranks -- two readouts of one dataset disagreeing in
    # the log reads as a discrepancy long before anyone finds it is an estimator.
    median = ranks[len(ranks) // 2]
    qs = statistics.quantiles(ranks, n=4) if len(ranks) > 3 else None
    spread = f"  q1 {qs[0]:.0f}  q3 {qs[2]:.0f}" if qs else ""
    print(f"min {ranks[0]}  median {median}  max {ranks[-1]}{spread}")
    for bound in sorted({50, 100, 200, 500, depth}):
        n = sum(1 for rank in ranks if rank <= bound)
        print(f"  rank <= {bound:>5} : {n:>3} ({n / len(rows):.1%} of all {len(rows)} misses)")
PY
