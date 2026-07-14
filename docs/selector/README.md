# Selector module

## Module contract

Input: Query plus CandidateSet from Retriever.

Output: selected evidence IDs, selection scores, and selection ranks.

The Pipeline resolves those IDs to the original Retriever evidence before Generator use.
The interface does not include sufficiency, missing-fact, or conflict status.

## Current implementation

`src/evidence_rag/selector/top_k.py` is only a replaceable baseline.

## Selector V2 status

`TRAINING_PLAN.md` and `EXPERIMENT_TRACKER.md` were migrated from the old project so work
can continue after the reset.

They are planning inputs, not an approved command to start training. Before execution,
the Selector team must re-check:

1. whether the task and labels still match the new project definition;
2. whether old NIAH, artificial-needle, and counterfactual construction should be used;
3. the training, development, and test datasets;
4. the baseline and success metrics;
5. the team-accessible artifact location;
6. that FinanceBench is excluded from Selector work by team agreement.

Training remains not started until those decisions are reviewed.
