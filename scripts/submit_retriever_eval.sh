#!/bin/bash
# One entry point to run a named retriever-eval SUITE — resolves the config list for you
# and either submits it to Slurm (GPU suites) or runs it inline on the login node (CPU
# suites). Wraps scripts/run_retriever_eval.slurm / evidence_rag.cli.experiment.
#
# Usage:
#   scripts/submit_retriever_eval.sh [--local|--print] <suite>
#
# Suites:
#   scifact | nq | 2wiki      base 8-variant matrix           (GPU  -> sbatch)
#   scifact-convex | nq-convex  Convex alpha sweep + endpoints (GPU  -> sbatch)   [补充2]
#   scifact-hyper             BM25 k1/b grid + RRF k sweep     (CPU  -> --local)  [补充4]
#
# Modes:
#   (default) submit with sbatch          --local run inline here    --print just show cmd
#
# 补充1 (significance) is post-hoc, not a job: use scripts/retriever_significance.sh <ds>.
# 补充3 (2wiki) needs the dataset first:
#   python -m evidence_rag.materializer.twowiki_cli --output runs/twowiki --query-limit 2000 --seed 42
set -euo pipefail

MODE="submit"
case "${1:-}" in
  --local) MODE="local"; shift ;;
  --print) MODE="print"; shift ;;
esac
SUITE="${1:?usage: scripts/submit_retriever_eval.sh [--local|--print] <suite>}"

C="configs/experiments"
BASE=(bm25 strong-bm25 hybrid-rrf hybrid-convex granite-dense query2doc hyde decompose)

# echo "$C/retr_<prefix>_<variant>.toml" for each variant
mk() { local p="$1"; shift; for v in "$@"; do printf '%s\n' "$C/retr_${p}_${v}.toml"; done; }

case "$SUITE" in
  scifact) mapfile -t FILES < <(mk scifact "${BASE[@]}") ;;
  nq)      mapfile -t FILES < <(mk nq "${BASE[@]}") ;;
  2wiki)   mapfile -t FILES < <(mk 2wiki "${BASE[@]}") ;;
  # Convex sweep: endpoints (pure sparse / pure dense) + alpha grid (a50 = hybrid-convex).
  scifact-convex)
    mapfile -t FILES < <(mk scifact strong-bm25 convex-a10 convex-a30 hybrid-convex convex-a70 convex-a90 granite-dense) ;;
  nq-convex)
    mapfile -t FILES < <(mk nq strong-bm25 convex-a30 hybrid-convex convex-a70 granite-dense) ;;
  # Hyperparameter sweep: pure-CPU sparse arms — prefer --local (no GPU queue).
  scifact-hyper)
    mapfile -t FILES < <(mk scifact bm25 strong-bm25 \
      bm25-k09b30 bm25-k09b75 bm25-k12b40 bm25-k12b75 bm25-k20b40 bm25-k20b75 \
      rrf-k10 rrf-k30 rrf-k100) ;;
  *) echo "unknown suite: $SUITE" >&2; exit 2 ;;
esac

# Fail early on a mistyped/absent config.
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "missing config: $f" >&2; exit 3; }
done

case "$MODE" in
  print)
    echo "suite '$SUITE' -> ${#FILES[@]} configs:"; printf '  %s\n' "${FILES[@]}"
    echo "sbatch scripts/run_retriever_eval.slurm ${FILES[*]}" ;;
  submit)
    mkdir -p logs runs
    sbatch scripts/run_retriever_eval.slurm "${FILES[@]}" ;;
  local)
    export PYTHONPATH="${PYTHONPATH:-src}"
    for f in "${FILES[@]}"; do
      echo "=== $f : prepare -> retriever ==="
      python -m evidence_rag.cli.experiment --config "$f" prepare
      python -m evidence_rag.cli.experiment --config "$f" retriever
    done
    echo "=== summary ==="
    python scripts/summarize_retriever_eval.py "${FILES[@]}" ;;
esac
