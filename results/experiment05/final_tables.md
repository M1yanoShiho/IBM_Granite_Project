# Experiment 05 — Final Results Tables

**Purpose.** This document presents the frozen Experiment 05 results in the report layout agreed for the dissertation. It reorganises existing audited results for readability; it does not change any experimental value, model, metric, or frozen conclusion.

## Metric notation

- `RFC ↑`: Reference Fact Coverage — coverage of the required answer facts.
- `VRFC ↑`: Verified Reference Fact Coverage — required facts that are both present and supported by a valid citation.
- `UCR ↓`: Unsupported Claim Rate — proportion of factual claims not supported by the presented evidence.
- `CP ↑`: Citation Precision — proportion of citations that genuinely support the associated claim.
- `CR ↑`: Citation Recall — proportion of factual claims supplied with a valid supporting citation.
- `RR ↑`: Response Rate — proportion of questions receiving a substantive answer rather than an abstention.

All values in Tables 1 and 2 are percentages. Table 3 reports changes in percentage points (`pp`). Higher is better for `RFC`, `VRFC`, `CP`, `CR`, and `RR`; lower is better for `UCR`.

---

## Table 1. Main system comparison

### (a) KILT–NQ

| System | RFC ↑ | VRFC ↑ | UCR ↓ | CP ↑ | CR ↑ | RR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 47.50 | 26.25 | 30.91 | 40.56 | 44.45 | 94.25 |
| Hybrid RAG | 60.75 | 37.50 | 17.34 | 49.60 | 55.02 | 96.50 |
| Granite Rerank RAG | 64.00 | 41.25 | 18.02 | 49.30 | 57.26 | 96.75 |
| Provence RAG | 60.50 | 40.00 | 19.91 | 50.30 | 57.27 | 95.50 |
| Ours | 51.08 ± 0.58 | 38.00 ± 0.43 | 18.80 ± 0.98 | 60.00 ± 1.25 | 61.10 ± 1.19 | 92.17 ± 1.66 |

### (b) KILT–TriviaQA

| System | RFC ↑ | VRFC ↑ | UCR ↓ | CP ↑ | CR ↑ | RR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 81.00 | 43.50 | 21.28 | 42.13 | 47.13 | 92.75 |
| Hybrid RAG | 84.00 | 49.25 | 20.52 | 47.07 | 52.83 | 95.00 |
| Granite Rerank RAG | 88.50 | 54.75 | 16.37 | 47.91 | 57.75 | 96.00 |
| Provence RAG | 82.25 | 45.75 | 26.82 | 41.84 | 49.46 | 93.50 |
| Ours | 78.58 ± 1.94 | 54.67 ± 0.80 | 18.41 ± 1.52 | 54.10 ± 0.90 | 57.56 ± 1.02 | 88.42 ± 2.43 |

### (c) ALCE–ASQA

| System | RFC ↑ | VRFC ↑ | UCR ↓ | CP ↑ | CR ↑ | RR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 24.61 | 14.91 | 27.12 | 38.33 | 43.54 | 90.00 |
| Hybrid RAG | 30.00 | 16.60 | 20.69 | 42.90 | 49.34 | 94.25 |
| Granite Rerank RAG | 33.33 | 21.42 | 15.81 | 47.75 | 57.42 | 97.00 |
| Provence RAG | 28.67 | 18.22 | 25.71 | 46.31 | 54.56 | 91.75 |
| Ours | 23.20 ± 0.67 | 18.28 ± 0.25 | 21.04 ± 0.83 | 57.21 ± 0.87 | 58.46 ± 1.02 | 90.25 ± 1.64 |

**Table 1 note.** `Ours` reports the mean ± sample standard deviation across the frozen Seed 13, 42, and 73 results. The four deterministic baselines are single point estimates from one complete run each. The table shows the complete system trade-off: the proposed system produces the strongest citation precision on all three datasets and the strongest citation recall on NQ and ASQA, while factual coverage and response rate vary by dataset.

---

## Table 2. Module ablation results

### (a) KILT–NQ

| Configuration | RFC ↑ | VRFC ↑ | UCR ↓ | CP ↑ | CR ↑ | RR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full | 50.75 | 37.75 | 19.31 | 60.96 | 62.00 | 93.25 |
| w/ BM25 Retriever | 38.25 | 24.25 | 32.16 | 46.00 | 46.63 | 91.50 |
| w/o Selector / Keep-all Top-10 | 50.25 | 37.00 | 19.25 | 59.96 | 61.00 | 92.25 |
| w/ Direct Generator | 60.50 | 37.75 | 17.74 | 49.25 | 55.20 | 97.00 |

### (b) KILT–TriviaQA

| Configuration | RFC ↑ | VRFC ↑ | UCR ↓ | CP ↑ | CR ↑ | RR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full | 80.75 | 55.25 | 20.11 | 54.83 | 57.88 | 90.50 |
| w/ BM25 Retriever | 74.75 | 52.25 | 22.47 | 52.29 | 55.58 | 88.25 |
| w/o Selector / Keep-all Top-10 | 80.75 | 55.25 | 20.11 | 54.83 | 57.88 | 90.50 |
| w/ Direct Generator | 84.00 | 49.00 | 20.52 | 47.03 | 52.33 | 95.00 |

### (c) ALCE–ASQA

| Configuration | RFC ↑ | VRFC ↑ | UCR ↓ | CP ↑ | CR ↑ | RR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full | 22.60 | 17.99 | 21.95 | 56.83 | 58.00 | 91.75 |
| w/ BM25 Retriever | 20.25 | 14.98 | 30.45 | 47.75 | 48.38 | 89.25 |
| w/o Selector / Keep-all Top-10 | 22.75 | 17.93 | 20.94 | 56.21 | 57.50 | 90.25 |
| w/ Direct Generator | 29.75 | 15.90 | 20.88 | 41.67 | 48.21 | 94.75 |

**Table 2 note.** `Full` and all three single-module ablations use the frozen Seed 13 run. Each ablation changes only the named module while preserving the remaining system configuration.

---

## Table 3. Contribution of each module

For the five higher-is-better metrics, `Δ = Full − ablation`. For `UCR`, the table reports `UCR reduction = UCR(ablation) − UCR(Full)`. Positive values therefore consistently indicate an improvement produced by the complete module.

### (a) Retriever contribution

| Dataset | ΔRFC ↑ | ΔVRFC ↑ | UCR reduction ↑ | ΔCP ↑ | ΔCR ↑ | ΔRR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| KILT–NQ | +12.50 | +13.50 | +12.85 | +14.96 | +15.38 | +1.75 |
| KILT–TriviaQA | +6.00 | +3.00 | +2.36 | +2.54 | +2.29 | +2.25 |
| ALCE–ASQA | +2.35 | +3.01 | +8.50 | +9.08 | +9.63 | +2.50 |

**Interpretation.** Replacing the full Retriever with BM25 reduces performance across all six reported dimensions on all three datasets. The largest gains occur on NQ, while ASQA shows a particularly clear reduction in unsupported claims and improvement in citation quality.

### (b-1) Selector behaviour on ordinary RAG datasets

| Dataset | ΔRFC ↑ | ΔVRFC ↑ | UCR reduction ↑ | ΔCP ↑ | ΔCR ↑ | ΔRR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| KILT–NQ | +0.50 | +0.75 | −0.06 | +1.00 | +1.00 | +1.00 |
| KILT–TriviaQA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ALCE–ASQA | −0.15 | +0.05 | −1.01 | +0.63 | +0.50 | +1.50 |

**Table 3(b-1) note.** Values are `Full − Keep-all Top-10`, except for the direction-normalised `UCR reduction`. The frozen Selector activation and presentation-reduction rates were `0.00%` on all three ordinary datasets.

**Interpretation.** The Selector is a risk-triggered safety module. It remains conservative on ordinary RAG inputs instead of deleting evidence without a risk signal. The observed metric changes are reported directly: NQ gains about `+1 pp` in `CP`, `CR`, and `RR`; TriviaQA is unchanged across all six metrics; ASQA gains `+0.63 pp`, `+0.50 pp`, and `+1.50 pp` in `CP`, `CR`, and `RR`, respectively. Its filtering ability is evaluated directly in the following misleading-evidence stress test.

### (b-2) Selector evidence-level results under misleading-evidence stress tests

| Configuration | ΔH ↑ | P_D ↑ | L_R^N ↓ | L_C^N ↓ | L_R^W ↓ | L_C^W ↓ | d̄ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Protect–Harm, Seed 13 | 12.95% | 75.44% | 0 pp | 0 pp | 0 pp | 0 pp | 0.0656 |
| Protect–Harm, Seed 42 | 13.10% | 80.56% | 0 pp | 0 pp | 0 pp | 0 pp | 0.0621 |
| Random deletion | 1.62% | 9.45% | – | – | – | – | 0.0656 |
| Bottom-rank deletion | 0.30% | 1.75% | – | – | – | – | 0.0656 |

**Table 3(b-2) note.** `ΔH` is harmful-evidence reduction; `P_D` is deletion precision; `L_R` is required-evidence recall loss; `L_C` is multi-hop evidence-chain loss; `N` denotes NIAH; `W` denotes 2WikiMultiHopQA; and `d̄` is the mean number of deleted passages per query. The two controls use the Seed 13 deletion count. The Seed 13 `ΔH` 95% confidence interval is `10.34%–15.84%`.

**Interpretation.** When misleading evidence is present, Protect–Harm removes about `13%` of known harmful evidence with `75.44%–80.56%` deletion precision, while the observed required-evidence recall and chain losses remain `0 pp`. Under the same deletion budget, random and bottom-rank deletion remove substantially less harmful evidence. This is the dedicated evidence for the Selector's risk-triggered filtering contribution.

### (c) Generator contribution

| Dataset | ΔRFC ↑ | ΔVRFC ↑ | UCR reduction ↑ | ΔCP ↑ | ΔCR ↑ | ΔRR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| KILT–NQ | −9.75 | 0.00 | −1.57 | +11.71 | +6.80 | −3.75 |
| KILT–TriviaQA | −3.25 | +6.25 | +0.41 | +7.80 | +5.54 | −4.50 |
| ALCE–ASQA | −7.15 | +2.08 | −1.07 | +15.17 | +9.79 | −3.00 |

**Interpretation.** Relative to direct generation, the full grounded Generator consistently improves citation precision and citation recall, with especially large citation-precision gains on NQ and ASQA. The trade-off is lower raw factual coverage and response rate on several datasets, while verified and cited factual coverage improves on TriviaQA and ASQA.

---

## Source of frozen values

- Experiment 05 audited main results: [`table1.csv`](table1.csv) and
  [`table2.csv`](table2.csv).
- Experiment 05 technical audit: [`final_audit.json`](final_audit.json), status `FINAL PASS`,
  `12,000` generation outputs, `12,000` scored outputs, and `0` scorer errors.
- Dedicated Selector evidence: [`misleading_evidence_summary.json`](../selector/misleading_evidence_summary.json)
  and [`blind_answer_gate.json`](../selector/blind_answer_gate.json), both derived from the same
  frozen archive report and SHA-256.
