#!/bin/bash
# Paired A/B of the R5 hoist, both arms in ONE job on ONE node.
#
# R6 compared a fresh run against R5's numbers from a different job on a different
# node, and the confound showed: index build came out 10-16% *faster* after a change
# that makes the build do strictly more work (it now materialises the per-chunk
# Counters). That can only be the node, so R6's speedup carries roughly that much
# uncertainty. Running both arms back to back in one allocation removes it.
#
# Order is before, after, before again. If the two before arms disagree by more than
# the after arm's distance from them, the node drifted mid-job and the comparison is
# void — that is the point of the third run, and it costs 30 seconds.
#
# Do NOT sbatch this directly: the before arm needs a git worktree at the pre-hoist
# commit, and compute nodes have no git on PATH (job 18280216 died on that). Submit
# through the login-node wrapper, which creates the worktree first:
#
#   bash scripts/submit_scaling_ab.sh
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
WORKTREE="$REPO/.scaling-ab-before"
SIZES="500 1000 2000 3000 4000 5183"

if [ ! -d "$WORKTREE" ]; then
  echo "missing $WORKTREE -- submit via 'bash scripts/submit_scaling_ab.sh' instead" >&2
  exit 2
fi

echo "node       : $(hostname)"
cat logs/scaling-ab-refs.txt 2>/dev/null || echo "refs       : (not recorded)"
echo "manifest   : $MANIFEST"

run_arm() {  # run_arm <label> <workdir> <output>
  echo "=== arm: $1 ==="
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
