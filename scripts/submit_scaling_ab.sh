#!/bin/bash
# Prepare and submit the paired scaling A/B. Run this on the LOGIN node.
#
# The before arm needs the pre-hoist code checked out, but compute nodes have no git
# on PATH (job 18280216 died on exactly that), so the worktree is created here, where
# git exists, and the job only runs python. Re-running is safe: the worktree is
# recreated from scratch each time.
#
#   bash scripts/submit_scaling_ab.sh [before-ref]

set -euo pipefail

REPO="/user/work/$USER/IBM_Granite_Project"
cd "$REPO"

BEFORE_REF="${1:-159069c^}"
WORKTREE="$REPO/.scaling-ab-before"

git worktree remove --force "$WORKTREE" 2>/dev/null || true
git worktree prune
git worktree add --detach "$WORKTREE" "$BEFORE_REF"

mkdir -p logs results
# The job cannot resolve these itself, so they are captured here and echoed into the
# job log; without them the .out would not say which two commits were compared.
{
  echo "after_ref  : $(git rev-parse --short HEAD)"
  echo "before_ref : $BEFORE_REF -> $(git rev-parse --short "$BEFORE_REF")"
} > logs/scaling-ab-refs.txt
cat logs/scaling-ab-refs.txt

sbatch scripts/run_retriever_scaling_ab.slurm
