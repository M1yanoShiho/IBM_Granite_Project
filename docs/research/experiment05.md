# Experiment 05 Final Report

**Technical status:** `FINAL PASS`
**Claim A:** `NOT SUPPORTED`
**Claim B:** `NOT SUPPORTED`

This report evaluates complete-corpus BM25 retrieval followed by frozen Granite candidate scoring within BM25 Top-1000. It does not claim global standalone dense retrieval.

## Table 1 — kilt-nq

| system | RFC | VRFC | UCR | CP | CR | RR |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 0.4750 | 0.2625 | 0.3091 | 0.4056 | 0.4445 | 0.9425 |
| Hybrid RAG | 0.6075 | 0.3750 | 0.1734 | 0.4960 | 0.5502 | 0.9650 |
| Granite Rerank RAG | 0.6400 | 0.4125 | 0.1802 | 0.4930 | 0.5726 | 0.9675 |
| Provence RAG | 0.6050 | 0.4000 | 0.1991 | 0.5030 | 0.5727 | 0.9550 |
| Ours | 0.5108 | 0.3800 | 0.1880 | 0.6000 | 0.6110 | 0.9217 |

## Table 1 — kilt-tqa

| system | RFC | VRFC | UCR | CP | CR | RR |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 0.8100 | 0.4350 | 0.2128 | 0.4213 | 0.4713 | 0.9275 |
| Hybrid RAG | 0.8400 | 0.4925 | 0.2052 | 0.4707 | 0.5283 | 0.9500 |
| Granite Rerank RAG | 0.8850 | 0.5475 | 0.1637 | 0.4791 | 0.5775 | 0.9600 |
| Provence RAG | 0.8225 | 0.4575 | 0.2682 | 0.4184 | 0.4946 | 0.9350 |
| Ours | 0.7858 | 0.5467 | 0.1841 | 0.5410 | 0.5756 | 0.8842 |

## Table 1 — alce-asqa

| system | RFC | VRFC | UCR | CP | CR | RR |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 0.2461 | 0.1491 | 0.2712 | 0.3833 | 0.4354 | 0.9000 |
| Hybrid RAG | 0.3000 | 0.1660 | 0.2069 | 0.4290 | 0.4934 | 0.9425 |
| Granite Rerank RAG | 0.3333 | 0.2142 | 0.1581 | 0.4775 | 0.5742 | 0.9700 |
| Provence RAG | 0.2867 | 0.1822 | 0.2571 | 0.4631 | 0.5456 | 0.9175 |
| Ours | 0.2320 | 0.1828 | 0.2104 | 0.5721 | 0.5846 | 0.9025 |

## Table 2 — kilt-nq

| configuration | RFC | VRFC | UCR | CP | CR | RR |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.5075 | 0.3775 | 0.1931 | 0.6096 | 0.6200 | 0.9325 |
| w/ BM25 Retriever | 0.3825 | 0.2425 | 0.3216 | 0.4600 | 0.4662 | 0.9150 |
| w/o Selector / Keep-all Top10 | 0.5025 | 0.3700 | 0.1925 | 0.5996 | 0.6100 | 0.9225 |
| w/ Direct Generator | 0.6050 | 0.3775 | 0.1774 | 0.4925 | 0.5520 | 0.9700 |

## Table 2 — kilt-tqa

| configuration | RFC | VRFC | UCR | CP | CR | RR |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.8075 | 0.5525 | 0.2011 | 0.5483 | 0.5787 | 0.9050 |
| w/ BM25 Retriever | 0.7475 | 0.5225 | 0.2247 | 0.5229 | 0.5558 | 0.8825 |
| w/o Selector / Keep-all Top10 | 0.8075 | 0.5525 | 0.2011 | 0.5483 | 0.5787 | 0.9050 |
| w/ Direct Generator | 0.8400 | 0.4900 | 0.2052 | 0.4703 | 0.5233 | 0.9500 |

## Table 2 — alce-asqa

| configuration | RFC | VRFC | UCR | CP | CR | RR |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.2260 | 0.1799 | 0.2195 | 0.5683 | 0.5800 | 0.9175 |
| w/ BM25 Retriever | 0.2025 | 0.1498 | 0.3045 | 0.4775 | 0.4838 | 0.8925 |
| w/o Selector / Keep-all Top10 | 0.2275 | 0.1793 | 0.2094 | 0.5621 | 0.5750 | 0.9025 |
| w/ Direct Generator | 0.2975 | 0.1590 | 0.2088 | 0.4167 | 0.4821 | 0.9475 |

## Registered claim decision

```json
{
  "claim_a": {
    "label": "NOT SUPPORTED",
    "rfc_rr_harm_gates": false,
    "superiority_by_dataset": {
      "alce-asqa": false,
      "kilt-nq": false,
      "kilt-tqa": false
    },
    "ucr_noninferiority_by_dataset": {
      "alce-asqa": true,
      "kilt-nq": false,
      "kilt-tqa": true
    }
  },
  "claim_b": {
    "cp_cr_harm_gates": true,
    "label": "NOT SUPPORTED",
    "superiority_by_dataset": {
      "alce-asqa": false,
      "kilt-nq": false,
      "kilt-tqa": false
    },
    "vrfc_noninferiority_by_dataset": {
      "alce-asqa": true,
      "kilt-nq": true,
      "kilt-tqa": true
    }
  }
}
```

Technical PASS means the frozen protocol completed; it does not imply that either scientific claim was supported.
