# Experiment 04 Goal 3 — main-system results

**Execution verdict:** `PASS`  
**Scientific verdict:** `Ours is not superior on the primary RAR metric`

## Validity and coverage

The fresh formal attempt `attempt-20260822d` completed as three one-job seven-arm dataset
bundles: HotpotQA job `18674672`, MuSiQue job `18674673`, and RGB job `18674674`. All three
jobs exited `0:0`. Every arm covers all frozen ordered IDs, giving exactly `7,700` outputs across
21 dataset-arm cells. Every arm has zero runtime errors. Ours includes all three independently
trained Generator seeds (13, 42, 73), with no seed selection.

Each dataset generation manifest was frozen and hashed before its scorer-only sidecar was read.
All three generation manifests and all three score manifests are `PASS`. No cross-job or
cross-attempt outputs were mixed, and no held-out question, answer, support label, or generated
answer was printed or manually inspected during execution.

The earlier prompt-overflow attempt remains invalid and excluded. The repaired formal attempt
kept the frozen 2,304-token limit and used the same rank-preserving maximal whole-evidence prefix
for all seven arms.

## Table 1 result

Ours improves citation F1 but loses substantial answer quality, so that improvement does not
translate into RAR. On HotpotQA, Ours obtains `0.25 ± 0.00` RAR versus `2.25–4.00` for the four
baselines. On MuSiQue answerable, Ours is `0.00 ± 0.00` versus `0.25–1.00`. On RGB noise, Ours
is `0.00 ± 0.00`, tied with Dense and Provence and below Hybrid and Granite Rerank.

The paired RAR analysis uses the per-query mean over all three Ours seeds minus each baseline,
with 10,000 component-cluster percentile bootstrap resamples (seed 13):

| Dataset | Baseline | ΔRAR (pp) | 95% CI (pp) |
|---|---|---:|---:|
| HotpotQA | Dense RAG | -2.75 | [-4.50, -1.00] |
| HotpotQA | Hybrid RAG | -3.25 | [-5.25, -1.50] |
| HotpotQA | Granite Rerank RAG | -3.75 | [-5.75, -1.75] |
| HotpotQA | Provence RAG | -2.00 | [-3.75, -0.50] |
| MuSiQue | Dense RAG | -1.00 | [-2.00, -0.25] |
| MuSiQue | Hybrid RAG | -1.00 | [-2.00, -0.25] |
| MuSiQue | Granite Rerank RAG | -1.00 | [-2.00, -0.25] |
| MuSiQue | Provence RAG | -0.25 | [-0.75, 0.00] |
| RGB noise | Dense RAG | 0.00 | [0.00, 0.00] |
| RGB noise | Hybrid RAG | -0.67 | [-1.67, 0.00] |
| RGB noise | Granite Rerank RAG | -0.33 | [-1.00, 0.00] |
| RGB noise | Provence RAG | 0.00 | [0.00, 0.00] |

Thus Goal 3 passes as a valid, complete experiment, while the superiority hypothesis is not
supported. No method, checkpoint, or threshold is changed in response to these held-out results.

## Audit evidence

- `results/TABLE1.md` and `results/table1.json`: report-ready and machine-readable Table 1.
- `results/per_query_metrics.csv`: 7,700 paired rows with no question or answer text.
- `results/bootstrap_ci.json`: 12 paired RAR comparisons.
- `results/goal3_audit.json`: final traceability and artifact hashes.
- `artifacts/goal3_formal_manifests/`: the six frozen dataset generation/score manifests.
- `artifacts/goal3_pass_manifest.json`: consolidated machine-readable PASS evidence.

The final targeted suite is `19 passed`; targeted Ruff and Slurm shell syntax pass. Mypy passed
across 137 source files after the prompt-budget repair. The local full regression suite excluding
the already recorded stale Experiment 03 state assertion also passed. Repository-wide Ruff has
26 pre-existing unrelated Experiment 03/script findings, which were not modified.
