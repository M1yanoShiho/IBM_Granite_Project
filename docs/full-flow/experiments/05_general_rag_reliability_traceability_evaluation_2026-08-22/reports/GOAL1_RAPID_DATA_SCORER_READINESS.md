# Experiment 05 Goal 1 rapid readiness

**Decision:** `PASS`  
**Next active goal:** `Goal 2`

Goal 1 passes under the user-authorized v4 rapid execution amendment. The two complete BM25
indices cover all 37,752,557 KILT and 21,015,324 DPR passages. The cancelled full-dense partial
artifacts are preserved for audit but are not part of any formal system.

All three datasets have 120 frozen development IDs and 400 frozen formal IDs. Development,
formal, and historical-exposure sets satisfy the required disjointness checks. Runtime bundles
contain no gold fields; scorer-only files are physically separate and mode-restricted. No formal
Retriever, Selector, Generator, TRUE, MiniCheck, generation, or scoring call occurred in Goal 1.

The frozen rapid retrieval contract is full-corpus BM25 Top-1000 followed by Granite candidate
scoring and RRF (`k=60`). It forbids gold/oracle injection and does not claim independent
full-corpus dense retrieval.

Scorer readiness remains PASS: locked FactMatch macro-F1 `0.9562`, evidence-support macro-F1
`0.9625`, claim extraction span recall/precision `1.0`, and all `80/80` deterministic contract
cases exact. The v4 focused suite reports `51 passed`; lint is PASS.
