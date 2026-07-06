# Retrieval results summary (source data for the report)

A running log of the **numbers and findings** from the retrieval evaluation, kept
so they can be cited when writing the report. This file is **data, not prose** —
write the report in your own words from these tables (Bristol: own-work rule).

Raw data on the HPC — the per-run aggregate CSVs (`results/<dataset>_fair.csv` /
`_convex.csv` / `_splade.csv`), per-query CSVs (`results/<dataset>_per_query.csv` and
the `_convex_` / `_splade_` variants), and the α-curves
(`results/convex_alpha_curve_*.csv` / `splade_alpha_curve_*.csv`). Commit them so they
are version-controlled: `git add -f results/*.csv` (the `results/` dir is gitignored).

## Setup

- **System under test:** `granite_dense` = IBM `granite-embedding-english-r2` (~150M) dense retriever.
- **Open baselines (same class, ~110M):** `gte_dense` (gte-base), `e5_dense` (e5-base-v2, with query/passage prefixes), `bge_dense` (bge-base-en-v1.5, with query instruction).
- **Other points:** `granite_small_dense` (granite-embedding-small-english-r2, ~47M), `st_dense` (all-MiniLM-L6-v2, 2021, 22M — a weak/old reference), `bm25` (rank_bm25 — a weak-ish lexical floor; note: below Anserini BM25).
- **Enhancements:** `hybrid_*` = RRF fusion (dense + BM25); `convex_hybrid_*` = convex combination of per-query min-max-normalised dense + lexical scores (α = dense weight, tuned on dev/train); `splade` = SPLADE learned-sparse retriever (`naver/splade-cocondenser-ensembledistil`, a research baseline — a stronger lexical arm than BM25); `*_rerank` = + Granite cross-encoder (`granite-embedding-reranker-english-r2`).
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
| splade | 0.690 | 0.348 | 0.356 |
| convex_hybrid_granite_bm25 (α-tuned) | 0.767 | 0.387 | 0.466 |
| convex_hybrid_granite_splade (α-tuned) | 0.772 | 0.392 | 0.466 |

(Full precision/recall/nDCG@{1,3,5,10}/MRR + ms_per_query are in the CSVs. The convex
hybrids are at their dev-tuned α: BM25 arm 0.80/0.75/0.90, SPLADE arm 0.65/0.60/0.65.)

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
4. **Hybrid retrieval: RRF fails, but convex fusion + a strong lexical arm give small, real gains over pure dense.** The full investigation (built as `ConvexHybridRetriever`, `eval/tune_alpha.py`, `SparseRetriever`; raw in `results/*_convex.csv` / `*_splade.csv`):
   - **(a) Equal-weight RRF regresses toward the weak BM25** — significantly worse than granite_dense on every set (Table 2: −0.048 / −0.015 / −0.087). Rank fusion discards the strong dense *scores*.
   - **(b) Convex combination** of per-query min-max-normalised scores (α = dense weight, tuned on dev/train) beats RRF everywhere (+0.048 / +0.027 / +0.095, all p=0.0001) and edges pure dense: with the **BM25** arm — tie SciFact (+0.001, ns), **+0.012 NFCorpus (p=0.001)**, **+0.008 FiQA (p=0.0015)**; tuned α 0.80 / 0.75 / 0.90. The α=1 curve point reproduces published granite_dense (0.767 / 0.375 / 0.459) — a consistency check that the dense arm is unchanged.
   - **(c) A stronger lexical arm (SPLADE)** is much better standalone than BM25 (nDCG@10 +0.054 / +0.046 / **+0.139**, all p≤0.0005; ~2× on FiQA), though SPLADE alone (0.690 / 0.348 / 0.356) still trails dense. It earns more fusion weight (tuned α drops to **0.65 / 0.60 / 0.65**) and makes a modestly better hybrid: convex+SPLADE vs dense — +0.006 SciFact (ns), **+0.017 NFCorpus (p=0.0006, the largest hybrid gain here)**, +0.008 FiQA (p=0.12, ns). At the *same* α the SPLADE arm beats the BM25 arm — **+0.014 FiQA (p=0.003)**, +0.013 SciFact (p=0.09), +0.006 NFCorpus (ns). Efficiency bonus: SPLADE's scipy-CSR sparse search is ~3.5× faster than pure-Python `rank_bm25` at scale (FiQA query latency 99 ms vs 342 ms), at a heavier one-off indexing cost (a BERT forward per doc).

   Convex-hybrid **2×2** — nDCG@10 Δ vs granite_dense, each hybrid at its own tuned α (* = p<0.05):

   |  | BM25 arm | SPLADE arm |
   |---|---|---|
   | RRF fusion | −0.048* / −0.015* / −0.087* | (not run — convex dominates RRF) |
   | Convex fusion | +0.001 / **+0.012*** / **+0.008*** | +0.006 / **+0.017*** / +0.008 |

   (SciFact / NFCorpus / FiQA.) → The original "hybrid fails" was a *fusion-method* artifact; correct fusion plus a strong lexical arm turns it into a small but real — on NFCorpus significant — improvement, while confirming the **dense Granite arm carries most of the quality** (hybrids add ≤ +0.017).
5. **Cross-encoder reranking does NOT help either** — neutral at best (granite_rerank ns on SciFact & FiQA), significantly worse on NFCorpus and for the small/hybrid variants. The strong first stage is already near-ceiling on these sets, and the reranker does not discriminate more sharply.
6. **Failure analysis (per-query, nDCG@10):** granite vs gte are **redundant** (Pearson r = 0.88 SciFact / 0.93 NFCorpus, balanced query wins). granite vs BM25 are **complementary on SciFact** (r = 0.63; BM25 wins 28/300 queries, some by a wide margin) but more redundant on NFCorpus (r = 0.80). Global *rank* fusion (RRF) and reranking cannot exploit this complementarity (finding 5); score-level **convex** fusion, α-tuned, extracts a small but significant part of it — more so with the stronger SPLADE arm (finding 4). NB: a clean "BM25 wins on rare-entity queries" pattern did **not** survive inspection of the full top-20 (both win-sets are entity-heavy).
7. **ANN indexing (HNSW) — efficiency-axis optimization.** Built (`VectorIndexer` `index_type`); exact `flat` index is O(N)/query and tens of GB at millions of docs. *Recall-retention + speedup figure (flat vs HNSW on a large corpus, e.g. dbpedia-entity / MS MARCO) is PENDING the scale run* (`scripts/run_scale_demo.slurm`).

## Table 3 — RAG answer quality (NQ + TriviaQA): does better retrieval → better answers?

Controlled retrieve-then-generate experiment: same generator (`granite-4.1-8b` instruct), same corpus, same (concise-answer) prompt — **only the retriever varies**, so any difference in answer quality is attributable to retrieval. Two answer-bearing QA sets, both on the shared `dpr-w100` Wikipedia corpus (generalisation): NQ (`dpr-w100/natural-questions/dev`) and TriviaQA (`dpr-w100/trivia-qa/dev`). Per set: first 500 queries; qrels-aware subset (gold passages always kept + distractors), **~1.0M docs**; `top_k`=4 chunks to the generator. Metrics: **cover-EM** (answer recall — a gold answer is contained in the answer; the de-facto metric for *generative* open-domain QA), normalised **EM** / token-**F1**, **context_precision** (precision@k of retrieved docs vs qrels), **faithfulness** (answer-token coverage of context). Significance = paired randomization test (`eval/significance.py`), 500 paired queries.

**NQ:**

| retriever | cover-EM | F1 | EM | context_precision | faithfulness |
|---|---|---|---|---|---|
| **granite_dense** | **0.584** | 0.517 | 0.376 | 0.337 | 0.891 |
| gte_dense | 0.554 | 0.483 | 0.344 | 0.335 | 0.897 |
| bm25 | 0.418 | 0.361 | 0.266 | 0.202 | 0.760 |

**TriviaQA:**

| retriever | cover-EM | F1 | EM | context_precision | faithfulness |
|---|---|---|---|---|---|
| **granite_dense** | **0.698** | 0.650 | 0.586 | 0.358 | 0.797 |
| gte_dense | 0.670 | 0.635 | 0.570 | 0.376 | 0.807 |
| bm25 | 0.608 | 0.566 | 0.498 | 0.345 | 0.719 |

Significance on cover-EM (Δ = retriever − reference; EM and F1 are same-direction and same-significance):
- vs **bm25**: NQ — granite **+0.166** (p=0.0001), gte +0.136 (p=0.0001); TriviaQA — granite **+0.090** (p=0.0001), gte +0.062 (p=0.0013). Dense significantly beats lexical on **both** sets.
- granite vs **gte**: NQ +0.030 (p=0.069, **ns**), TriviaQA +0.028 (p=0.072, **ns**) — a small, consistent, but non-significant lean to Granite.

8. **Retrieval quality drives RAG answer quality — and it generalises.** Only the retriever changes, so the gap is causal: dense retrieval yields significantly higher answer correctness than BM25 on **both** QA sets (cover-EM +0.09 to +0.17, all p≤0.0013; EM/F1 agree). The downstream payoff the retrieval-only nDCG numbers cannot show on their own.
9. **Granite matches the strongest open peer end-to-end.** granite ≈ gte on every metric and both sets — a consistent ~+0.03 cover-EM edge to Granite that does **not** reach significance (p≈0.07). Mirrors the retrieval audit (granite ≈ peers on these); both significantly beat BM25.
10. **The dense-vs-lexical gap is dataset-dependent.** Much larger on NQ (cover-EM +0.166 over BM25) than TriviaQA (+0.090): TriviaQA's entity/keyword-heavy questions make BM25 far more competitive (its cover-EM 0.608 vs 0.418 on NQ), shrinking — but not erasing — dense's significant lead. Confirms NQ is dense-favourable; the effect generalises, its magnitude is dataset-specific.
11. **Strict EM/F1 under-report verbose generative answers; a concise prompt fixes the metric.** With the default verbose prompt the reader quoted the context (gold "Linda Davis" → a whole paragraph), giving EM=0 / F1≈0.06 on NQ despite being correct. A concise-answer `DEFAULT_RAG_PROMPT` → crisp spans ("Linda Davis"), lifting NQ EM to 0.376 / F1 to 0.517; cover-EM de-inflates 0.66→0.58 (no longer rewarded for quoting). cover-EM is robust across both prompts and is the headline metric.

**Caveats (do not overclaim):** (a) granite vs gte is a *tie* (ns, p≈0.07 both sets), not a win — "Granite specifically wins" holds only on the harder FiQA *retrieval* set. (b) faithfulness is ~0.8–0.9 with little discrimination — a short correct answer's tokens are almost always in the context — so treat it as secondary, not a system-separating metric. (c) 500-query subset at ~1M docs, not the full 21M corpus (flat index; scale-to-21M with HNSW pending). (d) cover-EM can still over-credit an incidental gold mention; EM/F1 (now meaningful) are the stricter cross-checks and agree.

## NIAH — rare-needle retrieval at scale (the re-anchored headline)

Task: one **designated needle** per query (the smallest-doc_id gold that literally
contains the answer) hidden among injected **counterfactual** (Source A: a ~1-token
entity swap of the needle — relevant but factually wrong), **generative** (B) and
**mined** (C) distractors, over the `dpr-w100` NQ corpus. Metric = **needle-found@k**
(the designated needle in the top-k) + **MRR** of that needle. Frozen 300-query task
(`results/niah_nq300_frozen.json`); the diagnostic sub-run is 100 queries. Significance =
paired randomization on per-query needle-found. Construction + gate: `src/niah/`,
`eval/build_niah_task.py`; methodology record in `docs/niah-task-definition.md`.

### Table 4a — The bottleneck is ranking, not recall (diagnostic, n=100)

| dense needle-found@100 | @10 | → decomposition (fraction of queries) |
|---|---|---|
| **0.87** | 0.43 | found in top-10 **0.43** / buried in pool ranks 11–100 **0.44** / unreachable (not in top-100) **0.13** |

→ The needle is retrieved into the top-100 pool 87% of the time, but half of those are
buried below rank 10. **Ranking failure (0.44) ≈ 3.4× the recall failure (0.13).**

### Table 4b — What fixes it: query reformulation, not reranking (certified, n=300 frozen)

| retriever | needle-found@10 | MRR | Δ vs dense (p) |
|---|---|---|---|
| **granite_dense** | 0.493 | 0.265 | — |
| **q2d_granite** (Query2Doc) | **0.563** | **0.317** | **+0.070 (p=0.0002)** * |
| hyde_granite (HyDE) | 0.553 | 0.284 | +0.060 (p=0.0054) * |

Reranking null (n=100, same q2d/dense first stage): `granite_rerank` 0.47 (ns),
`granite_listrank` 0.47 (MRR 0.216 < dense 0.222, ns), `q2d_granite_rerank` 0.48 /
`hyde_granite_rerank` 0.48 (ns) — all below q2d-alone (0.50–0.51 at n=100). → **Both
query-transforms significantly beat dense; every reranker (pointwise cross-encoder,
listwise LLM, and reranking stacked on q2d) does not.**

### Table 4c — Scale degradation (n=300 frozen; the headline figure) — `results/niah_scale_curve.csv`

| background max_docs | granite_dense | q2d_granite | q2d edge |
|---|---|---|---|
| 10k  | 0.517 | 0.590 | +0.073 |
| 100k | 0.500 | 0.587 | +0.087 |
| 1M   | 0.467 | 0.533 | +0.066 |
| 5M   | 0.423 | 0.497 | +0.074 |

(Total corpus = background + ~45k always-kept golds → n_docs 55k / 145k / 1.04M / 5.04M.
MRR in the CSV. fp32 single-node build ceiling ≈ 5M; 21M needs the IVFPQ compressed index.)

### Findings (NIAH)

12. **The bottleneck is ranking, not recall.** The needle is almost always retrieved
    into the pool (R@100 = 0.87) but ~half the retrieved needles sit at ranks 11–100,
    buried by the near-duplicate counterfactual distractors; ranking failure is ~3.4×
    the recall failure (Table 4a).
13. **Query reformulation is the only lever that helps; reranking does not.** At n=300,
    Query2Doc **+0.070 (p=0.0002)** and HyDE +0.060 (p=0.0054) significantly beat dense,
    while cross-encoder, listwise-LLM, and q2d+rerank stacks give no significant gain
    (Table 4b). Mechanism: a Source-A counterfactual is *relevant* (on-topic,
    answer-shaped, only the entity is wrong), so a relevance-based reranker cannot
    separate it from the needle; a richer query representation lifts the true needle at
    the retrieval stage, before the tie needs breaking.
14. **Graceful degradation, scale-invariant raiser edge.** needle-found@10 falls ~9
    points as the haystack grows 10k→5M (dense 0.517→0.423, q2d 0.590→0.497), and q2d's
    advantage (~+0.075) holds at *every* scale (Table 4c) — the query-transform edge is
    not a small-corpus artifact.

15. **Corroboration Reranking (our novel method) is the first reranker that helps.** It ranks
    the pool by cross-source answer *corroboration* — extract each candidate's answer with the
    LLM, boost answers that other retrieved passages / the model's parametric knowledge agree
    with — not query relevance; a lone counterfactual is corroborated by nobody. On top of q2d
    it adds **+0.040 needle-found@10 (0.610 vs 0.570, blend α*=0.6, p=0.022)** at n=300 —
    where every *relevance* reranker was null (finding 13). Combined **dense 0.49 → q2d 0.57 →
    +corroboration 0.61**. α (relevance weight) tuned on the α-curve
    (`results/corroboration_alpha_curve_nq300cert.csv`; α=1.0 = pure q2d consistency check;
    α=0 pure corroboration = 0.50, < q2d ⇒ complementary, not standalone). **Caveats:** α is
    tuned on the same 300q as the p-test (mild optimism, no dev split); the +0.04 replicates
    across two independent extraction runs (+0.033 / +0.040 — 3B answer-extraction is
    stochastic); cost ≈ 20 LLM extractions/query.

**NIAH caveats (do not overclaim):** (a) single designated needle on a natural
multi-gold corpus → other "relevant non-target" golds remain in the haystack; the clean
fix is the deferred **synthetic-insert** variant. (b) Source-A counterfactuals are a
naive `str.replace`, so they carry logical seams — an internal-consistency detector
would exploit a *construction artifact* that would not generalise to fluent
misinformation; the in-progress **Corroboration Reranker** deliberately uses the
defensible *cross-source* signal instead. (c) the diagnostic decomposition is at n=100
and the raiser/scale at n=300 (different runs); an earlier apparent "collapse" to 0.18
was a stale-index **caching bug** (`_cache_key` omitted the corpus), since fixed — the
n=300 numbers here are post-fix.

## Pending / not yet done

- ANN scale demo (recall-vs-latency on a millions-doc corpus).
- Failure-mode analysis write-up (per-query CSVs + `eval/failure_analysis.py` exist).
- RAG evaluation: **DONE — Table 3** (concise prompt; NQ + TriviaQA; dense ≫ BM25 significant on both, granite ≈ gte). Remaining: scale to the full 21M corpus (needs HNSW in run_rag), and NIAH RAG-vs-long-context (still skeleton).
- A more lexical dataset (ArguAna/Touché) if the failure analysis needs more BM25-favourable material.
- **NIAH scale curve: DONE — Table 4c.** Remaining NIAH: (a) **Corroboration Reranker** α-curve (`eval/tune_corroboration.py`, `q2d_corroborate`) — does the cross-source signal beat q2d? *in flight (100q first-read)*; (b) **Astute** generation-stage measurement (`run_rag --pipeline astute` vs vanilla, against the counterfactual distractors) — built, unmeasured; (c) synthetic-insert task variant (deferred rigor upgrade); (d) IVFPQ run to 21M (recall-vs-compression).
