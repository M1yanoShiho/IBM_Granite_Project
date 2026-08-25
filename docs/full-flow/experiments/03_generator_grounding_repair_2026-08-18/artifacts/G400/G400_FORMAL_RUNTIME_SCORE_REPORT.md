# G400 Locked NIAH Qualification

**Status:** `G400_NIAH_RESPONSIBILITY_PASS`

This is G400 only. It is not GQ freeze, not G410 cross-data qualification, and not held-out.

## Family Deltas vs Fixed G0

| Context | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|
| K_topk | 6.13% | 6.27% | 7.13% | 1.27% | 1.98% | 0.00% |
| S_legacy_selected | 10.70% | 8.87% | 7.95% | 5.12% | 5.28% | 0.00% |
| O_support_only | 8.10% | 8.56% | 5.20% | 2.75% | 2.98% | 0.00% |
| OB_support_benign | 5.35% | 8.26% | 5.35% | 0.54% | 1.15% | 0.00% |
| OH_support_harmful | 11.93% | 11.77% | 7.19% | 8.03% | 8.56% | 0.00% |
| OP_support_last | 10.86% | 14.83% | 6.12% | 2.83% | 3.29% | 0.00% |
| train_unsupported_safety | 0.00% | 0.00% | -29.95% | 0.00% | 0.00% | -29.95% |

## Aggregate

### K_topk

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 739 | 47.90% | 64.01% | 87.96% | 66.11% | 65.33% | 0.00% |
| GRC13 | 739 | 53.18% | 69.01% | 95.26% | 67.19% | 67.07% | 0.00% |
| GRC42 | 739 | 54.53% | 70.91% | 95.81% | 68.00% | 67.88% | 0.00% |
| GRC73 | 739 | 54.40% | 70.91% | 94.18% | 66.98% | 66.98% | 0.00% |

### S_legacy_selected

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 218 | 45.87% | 58.72% | 88.07% | 68.20% | 68.04% | 0.00% |
| GRC13 | 218 | 54.13% | 66.51% | 96.33% | 71.79% | 71.79% | 0.00% |
| GRC42 | 218 | 57.80% | 68.81% | 97.25% | 73.62% | 73.62% | 0.00% |
| GRC73 | 218 | 57.80% | 67.43% | 94.50% | 74.54% | 74.54% | 0.00% |

### O_support_only

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 218 | 54.13% | 68.81% | 90.37% | 70.87% | 70.64% | 0.00% |
| GRC13 | 218 | 59.63% | 75.23% | 95.87% | 72.71% | 72.71% | 0.00% |
| GRC42 | 218 | 62.39% | 77.06% | 96.33% | 74.54% | 74.54% | 0.00% |
| GRC73 | 218 | 64.68% | 79.82% | 94.50% | 73.62% | 73.62% | 0.00% |

### OB_support_benign

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 218 | 56.42% | 67.43% | 89.91% | 73.85% | 73.24% | 0.00% |
| GRC13 | 218 | 60.09% | 74.31% | 96.33% | 73.17% | 73.17% | 0.00% |
| GRC42 | 218 | 60.55% | 74.31% | 95.87% | 75.00% | 75.00% | 0.00% |
| GRC73 | 218 | 64.68% | 78.44% | 93.58% | 75.00% | 75.00% | 0.00% |

### OH_support_harmful

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 218 | 50.92% | 65.14% | 88.53% | 66.97% | 66.44% | 0.00% |
| GRC13 | 218 | 61.93% | 76.15% | 96.33% | 73.62% | 73.62% | 0.00% |
| GRC42 | 218 | 63.30% | 77.06% | 96.33% | 76.38% | 76.38% | 0.00% |
| GRC73 | 218 | 63.30% | 77.52% | 94.50% | 75.00% | 75.00% | 0.00% |

### OP_support_last

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 218 | 50.46% | 63.30% | 89.45% | 69.95% | 69.27% | 0.00% |
| GRC13 | 218 | 60.55% | 77.06% | 95.41% | 73.17% | 72.94% | 0.00% |
| GRC42 | 218 | 61.47% | 77.98% | 96.33% | 73.17% | 72.94% | 0.00% |
| GRC73 | 218 | 61.93% | 79.36% | 94.95% | 72.02% | 71.79% | 0.00% |

### train_unsupported_safety

| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 1044 | 0.00% | 0.00% | 30.08% | 0.00% | 0.00% | 30.08% |
| GRC13 | 1044 | 0.00% | 0.00% | 0.10% | 0.00% | 0.00% | 0.10% |
| GRC42 | 1044 | 0.00% | 0.00% | 0.29% | 0.00% | 0.00% | 0.29% |
| GRC73 | 1044 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Gate

```json
{
  "status": "G400_NIAH_RESPONSIBILITY_PASS",
  "technical_pass": true,
  "technical_failures": [],
  "responsibility_pass": true,
  "responsibility_checks": {
    "niah_correct_and_cited_delta_gt_0": true,
    "answer_delta_ge_minus_2pp": true,
    "coverage_delta_ge_minus_2pp": true,
    "citation_precision_delta_ge_minus_3pp": true,
    "citation_recall_delta_ge_minus_3pp": true,
    "unsupported_assertion_delta_le_plus_2pp": true,
    "at_least_2_of_3_seeds_correct_and_cited_nonnegative": true,
    "no_seed_citation_precision_or_recall_drop_gt_5pp": true
  },
  "tripwires": [],
  "positive_signal": true,
  "strong_claim": "PENDING_G420_STATISTICS",
  "cross_data_checks": "PENDING_G410"
}
```
