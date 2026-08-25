# Experiment 05 — Independent Integrity Audit

**Audit date:** 2026-08-24  
**Independent reviewer:** separate `gpt-5.5` agent, read-only  
**Overall integrity verdict:** `WARN`  
**Core execution verdict:** `PASS / FINAL PASS`  
**Scientific claims:** Claim A `NOT SUPPORTED`; Claim B `NOT SUPPORTED`

The independent reviewer found no evidence of generated/fake ground truth, model-output self-normalization,
phantom row counts, formal-gold leakage, missing bootstrap replicates, or a mismatch between the registered
Claim A/B decision rule and the reported labels. The overall label remains `WARN` because the review found
documentation drift at audit time and an unused appendix-diagnostics helper. The documentation drift was
corrected immediately after the finding; the unused helper does not affect any main score or final artifact.

## A–F integrity checks

| Check | Verdict | Evidence |
|---|---|---|
| A. Ground-truth provenance | PASS | KILT aliases/provenance are derived from dataset `output`; ASQA facts from `qa_pairs`. Runtime schemas exclude gold fields. Formal/exposure overlap recomputed as zero for all datasets. |
| B. Score normalization | PASS | Query metrics use registered raw denominators; aggregation is macro for RFC/VRFC/CP/CR/RR and claim-micro for UCR. All 12,000 metric rows stay in `[0,1]`; undefined UCR occurs only for zero-claim/non-substantive rows. |
| C. Result existence | WARN | Remote results are complete and hash-valid: 30 score bundles × 400, zero scorer errors, 12,000 outputs and 12,000 scores. Local status prose was stale when reviewed; README/TRACKER/plan-review banner have since been corrected. |
| D. Dead code | WARN | The live scorer and final compiler paths are connected. `compute_appendix_diagnostics` is tested but the final compiler computes appendix diagnostics separately; this is maintenance debt, not a result-integrity defect. |
| E. Scope | PASS | Registered scope is exactly 3 datasets × 10 arms × 400. Goal 3 froze 8,400 and Goal 4 froze 3,600 before scoring. The report correctly limits the retrieval claim to BM25 Top-1000 candidate-dense scoring/RRF. |
| F. Evaluation type | PASS | `real_gt + model-judged semantic/citation scoring`: real dataset reference facts with MiniCheck as scorer-only entailment judge; neither synthetic proxy nor human evaluation. |

## Independent mechanical recheck

- Generation inventory: `30` files, each `400` rows, total `12,000`.
- Scoring inventory: `30` arm directories, each `400` query scores and `400` claim traces, total `12,000`.
- Ordered query IDs match generation → query score → claim trace for every arm.
- Score/trace/aggregate schemas and dataset/arm identities match; `scorer_errors=0` in every arm.
- Bootstrap: three datasets, six metrics, `10,000` resamples, seed `13`, all metrics `ESTIMABLE`, invalid UCR replicates `0`.
- Table 1: `15` rows (3 datasets × 5 systems). Table 2: `12` rows (3 datasets × 4 configurations).
- All nine artifacts recorded by `final_audit.json` match recomputed SHA-256.
- Final Slurm job `18686047`: `COMPLETED`, exit `0:0`, elapsed `00:00:16`.

## Claim impact

- Claim A: `NOT SUPPORTED`; no dataset passed superiority and the RFC/RR harm gate failed.
- Claim B: `NOT SUPPORTED`; no dataset passed superiority, while the CP/CR harm gate passed.
- Technical `FINAL PASS` therefore means the registered experiment completed correctly; it does not mean
  either scientific superiority claim was supported.

## Non-blocking maintenance actions

1. Keep the final status and v4 BM25 Top-1000 scope wording synchronized across paper-facing documents.
2. If a single paper-facing selection manifest is required, archive or regenerate a compact formal-ID manifest
   from the frozen Goal 1 bundle; do not alter the selected IDs.
3. In a later maintenance change, either route final appendix generation through
   `compute_appendix_diagnostics` or mark/remove that helper to avoid dead-code ambiguity.

These actions do not reopen Goal 5 or change the frozen results.
