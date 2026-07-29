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

Evaluation was also extended from a single recall metric to **nDCG**, **Recall@K**, and **MRR**,
so retriever variants can be compared more precisely.

**Status:** all four implemented; unit tests exist per retriever (`tests/retriever/`);
config-driven construction verified end-to-end for `bm25`, `strong-bm25`, `hybrid`. Passing
tests proves correctness, not performance — that distinction matters for R2 below.

---

## R2 - StrongBM25 vs BM25 benchmark (SciFact, paired comparison)

Same corpus, same query set, same evaluation code — only the retriever implementation changes.

| Retriever | nDCG@10 | Recall@10 | MRR | ms / query |
|---|---|---|---|---|
| BM25 (baseline) | 0.636 | 0.756 | 0.604 | 12.8 |
| StrongBM25 | 0.649 | 0.770 | 0.617 | 8.7 |

**Conclusions**

1. **StrongBM25 wins on every axis measured** — nDCG@10 +1.3pp, Recall@10 +1.4pp, MRR +1.3pp,
   and 32% lower latency, on the same paired queries.
2. **The tuned parameters are not ours** — k1=0.9, b=0.4 are the published Anserini/BEIR
   defaults, not values we grid-searched. We adopted an established baseline rather than tune
   our own, so the win is attributable to a known, reproducible configuration.
3. **Latency improvement is a side effect of the stopword-filtered analyzer** (fewer, shorter
   posting lists), not a deliberate speed optimization — worth noting since it wasn't the
   original goal.

**Gap, stated plainly:** Hybrid, Dense (Granite embeddings), and Decompose retrievers are built
and pass their unit tests, so we know they behave correctly — but none of them have been run
through this same paired benchmark. Only StrongBM25 has been proven against a baseline with real
numbers; the other three are implementation claims, not performance claims.

**Decision: StrongBM25 adopted as the current default sparse retriever. Benchmarking Hybrid,
Dense, and Decompose against it is the top-priority next task (see Next steps).**

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

### Current limitations

| Limitation | Risk |
|---|---|
| Embedded PDF images ignored by default | Charts/figures inside documents contribute no content unless captioning is explicitly enabled |
| Image captions are lossy and may hallucinate | Treated as retrievable evidence regardless — visual detail and exact figures can be lost or misstated |
| Scanned PDFs depend entirely on OCR quality | Can degrade silently on low-quality scans or non-Latin scripts |
| Non-recursive directory scan; failed files skipped silently | Gaps in ingested corpus go unreported |

---

## The open question

We can prove StrongBM25 outperforms baseline BM25. We cannot yet say whether Hybrid, Dense, or
Decompose outperform StrongBM25, underperform it, or land somewhere in between — passing unit
tests confirms the code runs correctly, but says nothing about retrieval quality. Until R2's
benchmark is repeated for these three, "we built four new retrievers" and "we improved retrieval"
are two different claims, and only the first one is currently backed by evidence.

A related, unresolved risk on the ingestion side: hallucinated image captions are indistinguishable
from real evidence once they enter the retriever's candidate pool. We don't yet know how often this
happens on our actual corpus, or how much it would inflate apparent retrieval quality on
image-heavy documents versus just adding noise — this needs measurement, not assumption, and it's
a cross-module question since the Generator stage is what ultimately trusts (or doesn't) the
retrieved caption.

---

## Current work — addressing latest feedback (Bharat Arora, 2026-07-26)

- **Bringing actual numbers next time:** R2 above is the first response — a real, paired
  benchmark instead of an implementation claim. Extending the same benchmark to Hybrid, Dense,
  and Decompose is in progress.
- **Hallucinated-caption risk:** acknowledged (see Current limitations / open question above),
  not yet mitigated. Scheduling a sync with the Generator student, since the risk spans both
  modules — retriever surfaces the caption, generator decides how much to trust it.
- **Recursive scanning / silent file failures:** reprioritized from "future work" to urgent,
  ahead of any dataset scale-up, per feedback.
- **Performance at larger corpus sizes:** not yet started; added below as new future work, given
  the enterprise-scale angle in the brief.

---

## Next steps

1. **Benchmark Hybrid, Dense, and Decompose against StrongBM25 on SciFact** — same paired setup as
   R2. This is the headline experiment and it has not been run.
2. **Fix recursive directory scanning and silent file failures** — before dataset scale-up makes
   gaps harder to detect and diagnose.
3. **De-risk hallucinated captions as evidence** — constrain caption prompts, rely on downstream
   corroboration so lossy/hallucinated captions carry less weight; requires the cross-module sync
   noted above.
4. **Test retrieval performance at larger corpus sizes** — check how latency and index build time
   scale well past our current benchmark scale, before it becomes a blocker.
5. Configurable chunking, to better support structured documents (tables/sections) instead of
   fixed-length splits.
6. Broaden ingestion robustness: wider format coverage (docx/pptx/html via Docling), OCR quality
   signals instead of silent degradation, and an ingestion summary reporting skipped/failed files.
