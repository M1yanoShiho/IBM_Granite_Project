#!/bin/bash
# Paired significance tests between retriever variants (CPU, offline, seconds).
#
# Wraps evidence_rag.evaluation.paired_metric_cli: for a fixed set of variant pairs it
# reads each arm's runs/retr-<DS>-<variant>/retriever_report.json, pairs per-case metrics
# by query_id, and prints the mean delta + randomization p-value + bootstrap CI. Run on
# the login node AFTER the retriever job for the dataset has finished.
#
# Usage:
#   scripts/retriever_significance.sh <dataset-tag>        # e.g. scifact | nq | 2wiki
# The pairs below are "ON OFF" (delta = ON - OFF); ON is the richer/candidate variant.
set -euo pipefail

DS="${1:?usage: scripts/retriever_significance.sh <dataset-tag>   e.g. scifact | nq | 2wiki}"
export PYTHONPATH="${PYTHONPATH:-src}"

# Comparisons worth reporting. Query-transform ablations answer "does rewriting pay off?".
PAIRS=(
  "query2doc strong-bm25"       # Query2Doc rewrite vs its own sparse base
  "hyde granite-dense"          # HyDE rewrite vs its own dense base
  "decompose strong-bm25"       # Decomposition vs its own sparse base
  "hybrid-rrf strong-bm25"      # Rank fusion vs best sparse
  "hybrid-convex granite-dense" # Score fusion vs dense
  "strong-bm25 bm25"            # Tuned sparse vs baseline sparse
  "decompose-orig decompose"    # Fusing the original query back in (R1 follow-up)
  "decompose-bestrank decompose"           # max-rank fusion vs RRF's sum (R2 follow-up)
  "decompose-orig-bestrank decompose-orig" # ...and on top of the original-query arm
  # The bar that matters: does the best decompose arm beat just using the sparse
  # retriever on its own? Everything above only measures recovery towards it.
  "decompose-orig-bestrank strong-bm25"
)
# MRR answers "is the top hit right", recall@10/20 and total recall answer "is the gold
# document anywhere the selector can still reach it" -- and with top_k=50 feeding a
# selector that keeps 5, the pool-depth metrics are the ones this architecture consumes.
METRICS=(
  retriever.core.document_mrr
  retriever.core.document_recall_at_10
  retriever.core.document_recall_at_20
  retriever.core.document_recall
)

report() { echo "runs/retr-${DS}-$1/retriever_report.json"; }

printf '%-26s %-34s %8s %8s %8s %10s %7s\n' "comparison (ON vs OFF)" "metric" "mean_on" "mean_off" "delta" "p_value" "n"
printf '%s\n' "-------------------------------------------------------------------------------------------------------------"
for pair in "${PAIRS[@]}"; do
  read -r ON OFF <<<"$pair"
  ON_REPORT="$(report "$ON")"; OFF_REPORT="$(report "$OFF")"
  if [[ ! -f "$ON_REPORT" || ! -f "$OFF_REPORT" ]]; then
    printf '%-26s %s\n' "${ON} vs ${OFF}" "(missing report; skipped)"
    continue
  fi
  for METRIC in "${METRICS[@]}"; do
    # Capture the CLI's JSON, then format it — do NOT pipe into a `python - <<heredoc`,
    # because the heredoc claims stdin and the piped JSON never reaches sys.stdin.
    JSON="$(python -m evidence_rag.evaluation.paired_metric_cli \
      --on-report "$ON_REPORT" --off-report "$OFF_REPORT" --metric "$METRIC")"
    python -c 'import json, sys
d = json.loads(sys.argv[3])
print("%-26s %-34s %8.4f %8.4f %+8.4f %10.4f %7d" % (
    sys.argv[1], sys.argv[2], d["mean_on"], d["mean_off"], d["delta"], d["p_value"], d["n_paired"]))' \
      "$ON vs $OFF" "$METRIC" "$JSON"
  done
done
