# G410 Cross-Data Qualification

**Status:** `G410_CROSS_DATA_RESPONSIBILITY_PASS`

This is G410 2Wiki internal cross-data qualification only. It is not GQ freeze, not G420, and not held-out.

## Family Deltas vs Fixed G0

| Context | correct+cited | answer | coverage | citation precision | citation recall |
|---|---:|---:|---:|---:|---:|
| all_2wiki | 39.65% | 6.95% | 2.53% | 31.37% | 35.47% |
| topk | 34.04% | -1.75% | -1.75% | 26.32% | 31.23% |
| support_only | 38.60% | 16.14% | 2.81% | 23.33% | 27.19% |
| support_first | 36.14% | -3.86% | -2.81% | 28.95% | 33.68% |
| support_middle | 46.67% | 14.39% | 7.72% | 36.49% | 40.76% |
| support_last | 42.81% | 9.82% | 6.67% | 41.75% | 44.50% |

## Gate

```json
{
  "status": "G410_CROSS_DATA_RESPONSIBILITY_PASS",
  "technical_pass": true,
  "technical_failures": [],
  "responsibility_pass": true,
  "responsibility_checks": {
    "twowiki_correct_and_cited_delta_ge_minus_2pp": true,
    "answer_delta_ge_minus_3pp": true,
    "coverage_delta_ge_minus_2pp": true,
    "citation_precision_delta_ge_minus_3pp": true,
    "citation_recall_delta_ge_minus_3pp": true,
    "at_least_2_of_3_seeds_correct_and_cited_nonnegative": true
  },
  "tripwires": [],
  "positive_signal": true,
  "strong_claim": "PENDING_G420_STATISTICS",
  "citation_regression_guard": "PENDING_G420_REVEALED_ASQA_QAMPARI_BOUNDARY_CONFIRMATION"
}
```
