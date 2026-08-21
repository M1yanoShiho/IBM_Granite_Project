# G420 Generator Gate

**Status:** `G420_NEW_GENERATOR_QUALIFIED_G430_READY`
**Teacher recommendation:** `FREEZE_NEW_GRC_IN_G430`
**Strong claim:** `ESTABLISHED`

G420 combines the completed G400 NIAH qualification and G410 2Wiki cross-data qualification. It does not run generation, train Selector, create utility labels, or read held-out data.

## Primary Bootstrap Summary

| Family | Metric | Point | 95% CI | Queries | Components |
|---|---|---:|---:|---:|---:|
| G400 K_topk | correct_and_cited | 6.13pp | [2.63pp, 9.90pp] | 739 | 534 |
| G400 K_topk | answer_match | 6.27pp | [2.86pp, 9.92pp] | 739 | 534 |
| G400 K_topk | coverage | 7.13pp | [4.73pp, 9.66pp] | 739 | 534 |
| G400 K_topk | minicheck_citation_precision | 1.27pp | [-1.99pp, 4.62pp] | 739 | 534 |
| G400 K_topk | minicheck_citation_recall | 1.98pp | [-1.26pp, 5.30pp] | 739 | 534 |
| G410 all_2wiki | correct_and_cited | 39.65pp | [29.97pp, 49.35pp] | 95 | 93 |
| G410 all_2wiki | answer_match | 6.95pp | [-2.34pp, 16.15pp] | 95 | 93 |
| G410 all_2wiki | coverage | 2.53pp | [-1.72pp, 6.74pp] | 95 | 93 |
| G410 all_2wiki | minicheck_citation_precision | 31.37pp | [24.18pp, 38.48pp] | 95 | 93 |
| G410 all_2wiki | minicheck_citation_recall | 35.47pp | [28.06pp, 42.81pp] | 95 | 93 |

## Gate

```json
{
  "technical_pass": true,
  "responsibility_pass": true,
  "no_tripwires": true,
  "formal_complete": true,
  "g400_status": "G400_NIAH_RESPONSIBILITY_PASS",
  "g410_status": "G410_CROSS_DATA_RESPONSIBILITY_PASS",
  "strong_claim_checks": {
    "niah_primary_correct_and_cited_ci_low_gt_0": true,
    "niah_answer_ci_low_ge_minus_2pp": true,
    "niah_coverage_ci_low_ge_minus_2pp": true,
    "niah_citation_precision_ci_low_ge_minus_3pp": true,
    "niah_citation_recall_ci_low_ge_minus_3pp": true,
    "twowiki_correct_and_cited_ci_low_gt_0": true
  },
  "asqa_qampari_revealed_guard": "BOUNDARY_CONFIRMED_NOT_RERUN_NO_NEW_CLAIM",
  "heldout_read": false,
  "utility_labels_started": false
}
```

## Boundary

ASQA/QAMPARI revealed citation guard is recorded as boundary-confirmed but not rerun in this stage. No new independent claim is made from those data here.
