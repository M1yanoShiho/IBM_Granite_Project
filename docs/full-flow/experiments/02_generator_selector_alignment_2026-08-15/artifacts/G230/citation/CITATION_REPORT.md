# G230 Independent Citation Report

Judge: MiniCheck. Production TRUE is excluded from judging.

## Full Decision-Dev (TopK)

| Config | Answered | Citation precision | Precision (cited tasks) | Citation recall |
|---|---:|---:|---:|---:|
| G0 | 650 | 75.17% | 86.94% | 74.28% |
| GN | 626 | 73.28% | 84.95% | 72.38% |
| GC13 | 694 | 65.92% | 83.64% | 65.92% |
| GM13 | 708 | 67.73% | 85.02% | 67.68% |
| GC42 | 698 | 66.33% | 83.88% | 66.33% |
| GM42 | 711 | 66.39% | 83.84% | 66.39% |
| GC73 | 694 | 66.07% | 83.21% | 66.07% |
| GM73 | 710 | 65.70% | 83.01% | 65.70% |

## Citation Gate

```json
{
  "margin": -0.02,
  "noninferior_by_family": {
    "GC": {
      "13": {
        "citation_precision": false,
        "citation_recall": false
      },
      "42": {
        "citation_precision": false,
        "citation_recall": false
      },
      "73": {
        "citation_precision": false,
        "citation_recall": false
      }
    },
    "GM": {
      "13": {
        "citation_precision": false,
        "citation_recall": false
      },
      "42": {
        "citation_precision": false,
        "citation_recall": false
      },
      "73": {
        "citation_precision": false,
        "citation_recall": false
      }
    }
  },
  "pass_by_family": {
    "GC": false,
    "GM": false
  }
}
```

## Final Development Gate

```json
{
  "pre_citation_pass": true,
  "candidate_before_citation": "GC",
  "candidate_citation_pass": false,
  "selected_generator": null,
  "pass": false
}
```
