# G410 Cross-Data Qualification

**Status:** `G410_CROSS_DATA_RESPONSIBILITY_PASS`

This is G410 2Wiki internal cross-data qualification only. It is not GQ freeze, not G420, and not held-out.

## Family Deltas vs Fixed G0

| Context | correct+cited | answer | coverage | citation precision | citation recall |
|---|---:|---:|---:|---:|---:|
| all_2wiki | 100.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| support_first | 100.00% | 0.00% | 0.00% | 100.00% | 100.00% |

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
