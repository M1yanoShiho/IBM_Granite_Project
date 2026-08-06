#!/bin/bash
# Paired A/B of the R5 hoist, both arms in ONE job on ONE node.
#
# R6 compared a fresh run against R5's numbers from a different job on a different
# node, and the confound showed: index build came out 10-16% *faster* after a change
# that makes the build do strictly more work (it now materialises the per-chunk
# Counters). That can only be the node, so R6's speedup carries roughly that much
# uncertainty. Running both arms back to back in one allocation removes it.
#
# The "before" arm is the real pre-hoist code, checked out into a git worktree at
# 159069c^ — not a transcription of it, so there is nothing to keep in sync. The
# worktree has no data/ (gitignored), hence the absolute --manifest path.
#
# Order is before, after, before again. If the two before arms disagree by more than
# the after arm's distance from them, the node drifted mid-job and the comparison is
# void — that is the point of the third run, and it costs 30 seconds.
#
# CPU only. Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_retriever_scaling_ab.slurm
#SBATCH --job-name=retriever-scaling-ab
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dh25275@bristol.ac.uk

set -euo pipefail

module load languages/python/3.11.15
source /user/work/$USER/venv/bin/activate

REPO="/user/work/$USER/IBM_Granite_Project"
cd "$REPO"

MANIFEST="${1:-$REPO/data/benchmarks/scifact/test/manifest.json}"
BEFORE_REF="${2:-159069c^}"
WORKTREE="$REPO/.scaling-ab-before"
SIZES="500 1000 2000 3000 4000 5183"

echo "node        : $(hostname)"
echo "after  ref  : $(git rev-parse --short HEAD)"
echo "before ref  : $BEFORE_REF -> $(git rev-parse --short "$BEFORE_REF")"
echo "manifest    : $MANIFEST"

cleanup() { git worktree remove --force "$WORKTREE" 2>/dev/null || true; }
trap cleanup EXIT
cleanup
git worktree add --detach "$WORKTREE" "$BEFORE_REF"

run_arm() {  # run_arm <label> <workdir> <output>
  echo "=== arm: $1 ($(git -C "$2" rev-parse --short HEAD)) ==="
  ( cd "$2" && PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/retriever_scaling.py \
      --manifest "$MANIFEST" --sizes $SIZES --queries 50 --output "$3" )
}

run_arm before-1 "$WORKTREE" "$REPO/results/retriever-scaling-ab-before-1.json"
run_arm after    "$REPO"     "$REPO/results/retriever-scaling-ab-after.json"
run_arm before-2 "$WORKTREE" "$REPO/results/retriever-scaling-ab-before-2.json"

echo "=== paired comparison (same node, same job) ==="
PYTHONPATH=src python scripts/compare_scaling_runs.py \
  --before "$REPO/results/retriever-scaling-ab-before-1.json" \
  --after "$REPO/results/retriever-scaling-ab-after.json" \
  --replicate "$REPO/results/retriever-scaling-ab-before-2.json"
