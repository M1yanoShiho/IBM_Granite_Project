# Research reproduction

This directory is the public entry point for the dissertation experiments. It exposes the
final Selector and Generator training protocols and the two end-to-end evaluations without
duplicating their tested implementations.

There are two distinct reproduction levels:

1. **Result verification (CPU, no model weights):** rebuild every public Experiment 04/05
   table from the small audited JSON files under `results/`.
2. **Full re-execution (GPU and external assets):** rerun training, generation, model-judged
   scoring, bootstrap inference, and integrity checks from the frozen data/model bundles.

The second level requires datasets, model snapshots, trained adapters, and raw run bundles
that are intentionally not stored in Git. Their identities are frozen by SHA-256. The
archive reference `research-archive-2026-08-25` preserves the complete development records;
the external-asset manifest is completed in release Goal 5.

## Entry points

| Area | Public entry | Frozen outputs |
|---|---|---|
| Selector | [`selector/README.md`](selector/README.md) | `results/selector/` |
| Generator | [`generator/README.md`](generator/README.md) | `generator/frozen_provenance.json` |
| Experiment 04 | [`experiment04/README.md`](experiment04/README.md) | `results/experiment04/` |
| Experiment 05 | [`experiment05/README.md`](experiment05/README.md) | `results/experiment05/` |

The cross-reference from dissertation claims to code, inputs, configurations, and results is
in [`REPRODUCIBILITY_MAP.md`](../REPRODUCIBILITY_MAP.md).

## Scientific boundary

`FINAL PASS` means an experiment completed its frozen technical protocol. It does not mean a
superiority claim passed. Experiment 04 did not support an Ours-RAR superiority claim;
Experiment 05 Claim A and Claim B are both `NOT SUPPORTED`. The Selector passed its dedicated
evidence-filtering gate but failed the separate blind final-answer gate.
