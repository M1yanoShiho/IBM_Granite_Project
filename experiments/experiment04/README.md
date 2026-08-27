# Experiment 04: held-out three-module evaluation

Experiment 04 evaluates three datasets, seven main-system arms, three Generator seeds, and
three single-module ablations with a frozen five-metric scorer. The final audit records 11,000
new answer generations and 1,100 reused Full seed-13 rows.

## Quick table verification

No dataset, model, GPU, or raw per-query output is required:

```bash
python experiments/experiment04/build_tables.py --output-dir build/experiment04
```

The command reads only `results/experiment04/final_results.json`, rebuilds six public table
files, and rejects any generated SHA-256 that differs from the frozen aggregate.

## Full execution chain

1. `scripts/experiment04_goal1.py`: materialize the sealed dataset manifests and validate the
   scorer without running a model.
2. `scripts/experiment04_goal2.py` and `experiment04_goal2_server_baselines.py`: validate the
   ten-arm development wiring and server baselines.
3. `scripts/experiment04_goal3.py`: prepare, generate, freeze, score, and summarize the seven
   formal arms on HotpotQA, MuSiQue-answerable, and RGB-noise. The two portable Slurm launchers
   use explicit environment variables and all three Generator adapters.
4. `scripts/experiment04_goal4.py`: run the frozen Dense-Retriever, Top-10, and Direct-Generator
   substitutions while reusing Full seed13.
5. `scripts/experiment04_goal5.py`: independently recompute aggregates, paired component-cluster
   bootstrap intervals, final hashes, and audit status from a complete external run directory.

The ten frozen arm configurations are under `configs/experiments/experiment04/`. Full
re-execution additionally needs the sealed input/runtime bundles, scorer-only sidecars, model
snapshots, Selector checkpoint, and three Generator adapters. Their original run records are
recoverable from `research-archive-2026-08-25`; public asset locations are handled separately
from the Git repository.

## Frozen conclusion

The technical protocol passed, but Ours-RAR superiority was `NOT_SUPPORTED`. Direct Granite
outperformed GR-C seed13 on HotpotQA and MuSiQue RAR; the RGB direction was inconclusive.
Dense-Retriever and Top-10 substitutions produced no observed RAR difference in the frozen
samples. See `results/experiment04/final_results.json` for the exact registered decisions.
