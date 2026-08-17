# G230 Generator Development Gate

## Full Decision-Dev (TopK)

| Config | Answer | Coverage | Draft empty | Zero claims | Final empty | Errors |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 64.01% | 87.96% | 2.03% | 4.87% | 12.04% | 0 |
| GN | 63.73% | 84.71% | 6.77% | 9.07% | 15.29% | 0 |
| GC13 | 66.04% | 93.91% | 0.00% | 0.95% | 6.09% | 0 |
| GM13 | 69.42% | 95.81% | 0.00% | 0.95% | 4.19% | 0 |
| GC42 | 65.90% | 94.45% | 0.00% | 0.81% | 5.55% | 0 |
| GM42 | 70.37% | 96.21% | 0.00% | 0.81% | 3.79% | 0 |
| GC73 | 64.41% | 93.91% | 0.00% | 0.81% | 6.09% | 0 |
| GM73 | 67.52% | 96.08% | 0.00% | 0.68% | 3.92% | 0 |

## Pre-Citation Gate

```json
{
  "family_gate": {
    "GC": {
      "answer_point_above_g0_all_seeds": true,
      "answer_direction_by_seed": {
        "13": true,
        "42": true,
        "73": true
      },
      "coverage_ci_above_minus_1pp_all_seeds": true,
      "coverage_noninferior_by_seed": {
        "13": true,
        "42": true,
        "73": true
      },
      "major_failure_reduction_at_least_1pp_all_seeds": true,
      "failure_reduction_by_seed": {
        "13": true,
        "42": true,
        "73": true
      },
      "support_only_answer_and_empty_improve_all_seeds": true,
      "pass_before_citation": true
    },
    "GM": {
      "answer_point_above_g0_all_seeds": true,
      "answer_direction_by_seed": {
        "13": true,
        "42": true,
        "73": true
      },
      "coverage_ci_above_minus_1pp_all_seeds": true,
      "coverage_noninferior_by_seed": {
        "13": true,
        "42": true,
        "73": true
      },
      "major_failure_reduction_at_least_1pp_all_seeds": true,
      "failure_reduction_by_seed": {
        "13": true,
        "42": true,
        "73": true
      },
      "support_only_answer_and_empty_improve_all_seeds": true,
      "pass_before_citation": true
    }
  },
  "gm_beats_gc_stress_all_seeds": {
    "OB_support_benign": false,
    "OH_support_harmful": true,
    "OP_support_last": false
  },
  "stress_direction_by_seed": {
    "OB_support_benign": [
      false,
      true,
      false
    ],
    "OH_support_harmful": [
      true,
      true,
      true
    ],
    "OP_support_last": [
      false,
      true,
      true
    ]
  },
  "gm_mixed_context_robustness_pass": false,
  "candidate_before_citation": "GC",
  "citation_gate": "PENDING_INDEPENDENT_MINICHECK",
  "pre_citation_pass": true
}
```
