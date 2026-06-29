# Retrieval results summary (source data for the report)

A running log of the **numbers and findings** from the retrieval evaluation, kept
so they can be cited when writing the report. This file is **data, not prose** —
write the report in your own words from these tables (Bristol: own-work rule).

Raw data: the per-run aggregate CSVs (`results/<dataset>_fair.csv`) and per-query
CSVs (`results/<dataset>_per_query.csv`) on the HPC — commit them so they are
version-controlled (`git add results/*.csv`).

## Setup

- **System under test:** `granite_dense` = IBM `granite-embedding-english-r2` (~150M) dense retriever.
- **Open baselines (same class, ~110M):** `gte_dense` (gte-base), `e5_dense` (e5-base-v2, with query/passage prefixes), `bge_dense` (bge-base-en-v1.5, with query instruction).
- **Other points:** `granite_small_dense` (granite-embedding-small-english-r2, ~47M), `st_dense` (all-MiniLM-L6-v2, 2021, 22M — a weak/old reference), `bm25` (rank_bm25 — a weak-ish lexical floor; note: below Anserini BM25).
- **Enhancements:** `hybrid_*` = RRF fusion (dense + BM25); `*_rerank` = + Granite cross-encoder (`granite-embedding-reranker-english-r2`).
- **Datasets (BEIR):** SciFact (5,183 docs / 300 q / ~1.1 rel-per-q), NFCorpus (3,633 / 323 / ~38), FiQA (57,638 / 648 / ~2.6).
- **Significance:** paired two-sided randomization (sign-flip) test + percentile bootstrap 95% CI on per-query nDCG@10 differences (`eval/significance.py`), 10k resamples. `*` = p < 0.05.

## Table 1 — nDCG@10 (all retrievers × datasets)

| retriever | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| **granite_dense** | **0.767** | **0.375** | **0.459** |
| granite_small_dense | 0.746 | 0.361 | 0.406 |
| gte_dense | 0.753 | 0.368 | 0.405 |
| bge_dense | 0.739 | 0.366 | 0.407 |
| e5_dense | 0.712 | 0.340 | 0.398 |
| st_dense (MiniLM) | 0.641 | 0.309 | 0.368 |
| bm25 | 0.636 | 0.302 | 0.217 |
| hybrid_granite_bm25 (RRF) | 0.719 | 0.360 | 0.371 |
| hybrid_granite_small_bm25 | 0.710 | 0.350 | 0.343 |
| granite_rerank | 0.751 | 0.353 | 0.444 |
| granite_small_rerank | 0.752 | 0.350 | 0.438 |
| hybrid_granite_bm25_rerank | 0.750 | 0.355 | 0.441 |

(Full precision/recall/nDCG@{1,3,5,10}/MRR + ms_per_query are in the CSVs.)

## Table 2 — significance vs `granite_dense` (nDCG@10; Δ = retriever − granite)

| contrast | SciFact Δ (p) | NFCorpus Δ (p) | FiQA Δ (p) |
|---|---|---|---|
| gte_dense | −0.013 (0.19) ns | −0.007 (0.29) ns | **−0.054 (0.0001)** |
| bge_dense | −0.028 (0.014) | −0.009 (0.21) ns | **−0.051 (0.0001)** |
| e5_dense | −0.055 (0.0002) | −0.035 (0.0001) | **−0.061 (0.0001)** |
| granite_small_dense | −0.021 (0.014) | −0.014 (0.033) | −0.052 (0.0001) |
| granite_rerank | −0.016 (0.19) ns | −0.023 (0.008) | −0.015 (0.097) ns |
| hybrid_granite_bm25 | −0.048 (0.0001) | −0.015 (0.050) | −0.087 (0.0001) |
| bm25 | −0.131 (0.0001) | −0.066 (0.0001) | −0.242 (0.0001) |

Also (ref = `gte_dense`): `granite_small_dense` vs gte is **not significant** on SciFact (p=0.48) or NFCorpus (p=0.33) → small Granite is statistically indistinguishable from gte-base at <½ the parameters.

## Findings (each tied to the numbers above)

1. **Granite's quality edge over open peers grows with dataset difficulty.** On the small, near-saturated SciFact/NFCorpus, granite_dense ≈ gte/bge (differences not significant). On the harder, larger **FiQA, granite_dense significantly beats gte/bge/e5 by ~5 nDCG points (all p<0.001)**. The "tie" was a small-saturated-dataset artifact. → strongest positive result.
2. **Efficiency:** `granite_small_dense` (~47M) is statistically indistinguishable from gte/bge-base (~110M) across datasets → peer-quality retrieval at <½ the parameters. (It does trail full granite, significantly, by ~2 pts on SciFact and ~5 on FiQA — the expected size/quality trade-off.)
3. **The old "0.767 ≫ baseline" headline was a strawman.** vs the original MiniLM (0.641) and rank_bm25 (0.636) baselines, granite looked dominant; vs fair modern peers (gte/bge/e5 at 0.71–0.75) the SciFact lead shrinks to ~1–3 pts (mostly not significant).
4. **RRF hybrid (dense+BM25) does NOT help — significantly worse on every dataset.** Equal-weight rank fusion regresses toward the much weaker BM25.
5. **Cross-encoder reranking does NOT help either** — neutral at best (granite_rerank ns on SciFact & FiQA), significantly worse on NFCorpus and for the small/hybrid variants. The strong first stage is already near-ceiling on these sets, and the reranker does not discriminate more sharply.
6. **Failure analysis (per-query, nDCG@10):** granite vs gte are **redundant** (Pearson r = 0.88 SciFact / 0.93 NFCorpus, balanced query wins). granite vs BM25 are **complementary on SciFact** (r = 0.63; BM25 wins 28/300 queries, some by a wide margin) but more redundant on NFCorpus (r = 0.80). The complementarity is **not exploitable** by global RRF fusion or reranking (findings 4–5). NB: a clean "BM25 wins on rare-entity queries" pattern did **not** survive inspection of the full top-20 (both win-sets are entity-heavy).
7. **ANN indexing (HNSW) — efficiency-axis optimization.** Built (`VectorIndexer` `index_type`); exact `flat` index is O(N)/query and tens of GB at millions of docs. *Recall-retention + speedup figure (flat vs HNSW on a large corpus, e.g. dbpedia-entity / MS MARCO) is PENDING the scale run* (`scripts/run_scale_demo.slurm`).

## Pending / not yet done

- ANN scale demo (recall-vs-latency on a millions-doc corpus).
- Failure-mode analysis write-up (per-query CSVs + `eval/failure_analysis.py` exist).
- RAG evaluation (answer quality per retriever; NIAH RAG-vs-long-context). `eval/run_rag.py`, `eval/rag_metrics.py`, `src/explainability/citations.py` are skeletons.
- A more lexical dataset (ArguAna/Touché) if the failure analysis needs more BM25-favourable material.
