# Retriever — Progress Summary

Notice: R1, R2, etc. are task numbers

---

## R1 - Retriever capability extensions

The codebase previously had only a single baseline BM25 retriever. This task adds four new
capabilities, all selectable through configuration, with no code changes required to switch
between them:

| Capability | What it does |
|---|---|
| StrongBM25 | Sparse retriever, tuned parameters (k1=0.9, b=0.4) + stopword filtering |
| Hybrid retrieval | Fuses multiple retrievers — rank-based (RRF) or score-based (convex) |
| Query transformations | LLM rewrites or decomposes the query (Query2Doc, HyDE, Decompose) before retrieving |
| Configuration-driven selection | Any retriever, including combinations, chosen via one config file |

Evaluation reports **MRR**, **Recall**, and **Recall@{5,10,20}** (`src/evidence_rag/evaluation/scoring.py`),
so retriever variants can be compared precisely.

**Status:** all four implemented; unit tests exist per retriever (`tests/retriever/`);
config-driven construction verified end-to-end for `bm25`, `strong-bm25`, `hybrid`.

---

## R2 - Full benchmark: 8 retrievers × 3 datasets (SciFact, NQ, 2Wiki)

Run by MengW7, 2026-07-29–31 (commit `886cc8f`), full results in
[`docs/retriever/eval-results.md`](../eval-results.md) — includes base matrices for all three
datasets, a convex-α sweep, a BM25/RRF hyperparameter sweep, and paired significance tests
(randomization, p-values). This supersedes an earlier version of this report that cited a
pre-refactor, unverified SciFact CSV as if it were current-architecture output — it wasn't
(caught during a later review; see git history). The numbers below are real, current-architecture
output; SciFact table shown in full, NQ/2Wiki summarized (full tables in the linked doc).

**SciFact base matrix:**

| Retriever | MRR | R@10 | Recall |
|---|---|---|---|
| BM25 | 0.608 | 0.756 | 0.861 |
| StrongBM25 | 0.611 | 0.760 | 0.862 |
| Hybrid (RRF) | 0.707 | 0.863 | 0.947 |
| Hybrid (Convex) | 0.723 | 0.857 | 0.945 |
| GraniteDense | 0.718 | 0.863 | 0.940 |
| Query2Doc | 0.649 | 0.824 | 0.922 |
| HyDE | 0.700 | 0.862 | 0.961 |
| Decompose | 0.558 | 0.709 | 0.843 |

**Conclusions (corrected from the earlier version of this report)**

1. **StrongBM25 vs BM25 is not a reliable win.** Paired significance test: not significant on
   SciFact (Δ +0.0027 MRR, p=0.78) or NQ (Δ −0.0004 MRR, p=0.92); significant but small on 2Wiki
   (Δ +0.0146 MRR, p<0.0001). The earlier report claimed a clear win on all axes — that claim came
   from stale, pre-refactor data and should be discarded.
2. **Hybrid (RRF) reliably and substantially outperforms StrongBM25 on all three datasets**,
   significant on both MRR and Recall@10 every time (e.g. SciFact Δ +0.096 MRR p<0.0001; NQ
   Δ +0.071 MRR p<0.0001; 2Wiki Δ +0.025 MRR p<0.0001). This is the strongest, most consistent
   result in the whole matrix.
3. **Decompose significantly underperforms StrongBM25 on every dataset**, and catastrophically on
   2Wiki (Δ −0.388 MRR, p<0.0001 — 2Wiki is multi-hop, and splitting a multi-hop query into
   independent sub-queries appears to destroy the cross-hop signal). This is a genuine, measured
   failure mode, not a hypothesis — see "The open question" below.
4. **Query2Doc and HyDE are dataset-dependent**, not uniformly good or bad — e.g. Query2Doc beats
   StrongBM25 significantly on SciFact/NQ MRR but loses significantly on 2Wiki MRR (Δ −0.022,
   p<0.0001) while still winning on 2Wiki Recall@10 (Δ +0.011, p=0.0002).

**Decision: Hybrid (RRF) is the strongest general-purpose retriever measured so far and is the
recommended default when latency budget allows running two arms. Decompose should not be used on
multi-hop-style corpora without further work (see Next steps).**

---

## R3 - RAG PDF / Image ingestion pipeline

Retrieval quality depends on ingesting the corpus correctly. This frontend converts different
file formats into plain text, so the retriever only ever faces one uniform interface.

- **PDF** — parsed with Docling, keeping tables and sections intact rather than splitting purely
  by length.
- **Images** — described with Granite Vision, supplemented by OCR, so visual content and any
  embedded text are both captured.
- **Unified handling** — one entry point processes mixed folders of text/PDF/image files, with
  caching to avoid re-processing unchanged sources.

**OCR-smoke result (2026-08-04, job `18258588`, PASS — full entry in `docs/hpc-run-log.md`):**
the embedded-figure caption + in-figure OCR path was run end to end on real models (Docling +
Granite Vision, not test fakes). A sentinel string painted only inside a test figure
(`REVENUE 2024 42 PERCENT`) was correctly recovered in both the Vision caption and the OCR'd
`Text in image:` section — proving the new ingestion path actually works, not just that it runs
without crashing.

### Current limitations

| Limitation | Risk |
|---|---|
| Embedded PDF images ignored by default | Charts/figures inside documents contribute no content unless captioning is explicitly enabled |
| Image captions are lossy and may hallucinate | Treated as retrievable evidence regardless — visual detail and exact figures can be lost or misstated |
| Scanned PDFs depend entirely on OCR quality | Can degrade silently on low-quality scans or non-Latin scripts |
| Non-recursive directory scan; failed files skipped silently | Gaps in ingested corpus go unreported |

---

## The open question

The benchmark question from the previous version of this report is answered: Hybrid (RRF)
reliably beats StrongBM25 by a wide margin on all three datasets; Decompose reliably loses,
worst on multi-hop (2Wiki). The open question now is **why Decompose fails so badly on 2Wiki
specifically** — is splitting a multi-hop query into independent sub-queries fundamentally
incompatible with multi-hop retrieval (in which case Decompose should be gated off for
multi-hop-style corpora), or is it a fixable prompting/merging issue (e.g. the RRF fusion across
sub-queries losing the dependency between hops)? We don't know yet; this needs its own targeted
investigation rather than being lumped in with the base matrix.

A related, unresolved risk on the ingestion side: hallucinated image captions are indistinguishable
from real evidence once they enter the retriever's candidate pool. The OCR-smoke test above proves
the happy path works on a clean synthetic figure; it says nothing about hallucination rate on real,
messy documents. We don't yet know how often this happens on our actual corpus, or how much it
would inflate apparent retrieval quality on image-heavy documents versus just adding noise — this
needs measurement, not assumption, and it's a cross-module question since the Generator stage is
what ultimately trusts (or doesn't) the retrieved caption.

---

## Current work — addressing latest feedback (Bharat Arora, 2026-07-26)

- **Bringing actual numbers next time:** done, substantially — R2 above is a full 3-dataset ×
  8-variant matrix with significance tests, not a single paired comparison. The gap now is
  narrower and more specific: understanding *why* Decompose fails on 2Wiki, and reconciling
  the two retriever docs (done, in this update).
- **Hallucinated-caption risk:** OCR path now verified to work functionally (R3), but hallucination
  *rate* on real documents is still unmeasured. Still needs a sync with the Generator student,
  since the risk spans both modules.
- **Recursive scanning / silent file failures:** not yet fixed. Still urgent, ahead of any
  dataset scale-up.
- **Performance at larger corpus sizes:** not yet started.

---

## Next steps

1. **Investigate why Decompose fails on 2Wiki** — isolate whether the failure is in query
   splitting, sub-query retrieval, or RRF re-merging; decide whether Decompose should be gated
   off for multi-hop corpora or is fixable.
2. **Fix recursive directory scanning and silent file failures** — before dataset scale-up makes
   gaps harder to detect and diagnose.
3. **Measure hallucinated-caption rate on real (non-synthetic) documents** — the OCR-smoke PASS
   proves the mechanism works, not that captions are trustworthy at scale; requires the
   cross-module sync with Generator noted above.
4. **Test retrieval performance at larger corpus sizes** — check how latency and index build time
   scale well past our current benchmark scale (SciFact 300 docs, NQ/2Wiki 2000), before it
   becomes a blocker.
5. Configurable chunking, to better support structured documents (tables/sections) instead of
   fixed-length splits.
6. Broaden ingestion robustness: wider format coverage (docx/pptx/html via Docling), OCR quality
   signals instead of silent degradation, and an ingestion summary reporting skipped/failed files.
