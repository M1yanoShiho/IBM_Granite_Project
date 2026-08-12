# Independent reviewer output

**Reviewer:** GPT-5.6 Terra, xhigh reasoning fallback  
**Mode:** read-only  
**Overall verdict:** the formal R005 bundle supports `COMPLETE / FAIL` and the fail-closed stop. No P0 integrity breach was found. Historical claims that two `--verify-only` runs and three post-run audits passed were not themselves preserved as independent attestations, so those historical-event claims are P1/WARN rather than independently proven.

## Checks

- **A. Ground-truth provenance — WARN.** NIAH harm is a deterministically generated, provenance-verified synthetic counterfactual proxy, not open-world misinformation ground truth. 2Wiki supplies only official supporting `protect=1`; all harm and unjudged labels are masked. R005 states these limitations honestly.
- **B. Score normalization — PASS.** Decisive accuracy is raw `score >= 0.5` against frozen labels, using `correct / active-label total`; no prediction-derived denominator was found. BCE masks and active-label normalization are explicit.
- **C. Result existence — WARN.** The formal server artifacts reproduce 30 epochs, 360 steps, loss trends, 16/16 directions, all five class accuracies, 640 train-fit-only score rows, zero downstream rows and the NOT_EVALUATED marker. The historical verifier/auditor event claims lack persisted outputs.
- **D. Dead code / invocation — WARN.** The decisive training gate and FAIL branch were invoked. Policy metrics and safe-corner functions were correctly not invoked after gate failure; they are therefore not result evidence.
- **E. Scope — PASS.** Documents consistently describe seed 13, 16 NIAH + 16 2Wiki train-fit queries, 30 epochs and no modelval/delete evaluation. No selector-gain claim was found.
- **F. Evaluation classification — PASS.** NIAH is correctly a deterministic synthetic-proxy train-fit overfit sanity; 2Wiki is official protect-positive only; selector effect/modelval/CRC/sealed/heldout are not evaluated.

## Artifact reconciliation

- Training trace supports `30/30`, `360`, active losses, head changes, zero masked 2Wiki harm gradient and `16/16` NIAH pair directions.
- Stored class results are `32/32`, `16/16`, `71/79`, `12/16`, `14/16`.
- Candidate scores contain 640 rows; four downstream JSONL files are empty; quantile policy status is `NOT_EVALUATED_TRAINING_GATE_FAIL`.
- All 14 checksum entries verify, including the checkpoint SHA. The formal root has exactly 16 regular files and no symlinks.
- Model input is restricted to question and candidate text. No sealed/heldout path is pinned, and actual R005 never entered modelval.

## Issues

- **P0:** none.
- **P1-1:** Unsupported historical verifier/audit event counts. Preserve immutable current/future stdout or JSON attestations with command, commit, timestamp, exit code, verifier identity and hashes outside the exact formal root; otherwise describe the old events only as previously reported.
- **P2-1:** Never shorten NIAH harm to natural-world misinformation ground truth.
- **P2-2:** 2Wiki `32/32` is positive-class sensitivity, not full protect-head binary accuracy.
- **P2-3:** Do not treat unexecuted policy/safe-corner code as an evaluated R005 effect result.

No project files were edited by the reviewer.
