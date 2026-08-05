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
   2Wiki (Δ −0.388 MRR, p<0.0001). We have since diagnosed *why* and tested a fix — see R4.
4. **Query2Doc and HyDE are dataset-dependent**, not uniformly good or bad — e.g. Query2Doc beats
   StrongBM25 significantly on SciFact/NQ MRR but loses significantly on 2Wiki MRR (Δ −0.022,
   p<0.0001) while still winning on 2Wiki Recall@10 (Δ +0.011, p=0.0002).

**Decision: Hybrid (RRF) is the strongest general-purpose retriever measured so far and is the
recommended default when latency budget allows running two arms. Decompose should not be used on
multi-hop-style corpora (see R4 for why, and for how far a fix gets).**

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
| ~~Non-recursive directory scan; failed files skipped silently~~ | **Fixed** — see below |

**Ingestion robustness fix (2026-08-05).** The scan now recurses by default
(`recursive=False` opts out), every skip and failure is logged at warning level with a
per-run summary of how many files were ingested, and `on_error` finally governs *parsing*
as well as captioning.

Two things worth recording because they were worse than the limitation table said:

- A file that failed to parse did not "skip silently" — nothing caught it at all, so one
  corrupt PDF **aborted the entire ingest**, discarding every document already parsed. The
  documented `on_error="skip"` default only ever applied to image captioning.
- Making the scan recursive is not safe on its own: `document_id` was the bare file name,
  so two sub-directories each holding `report.pdf` would have produced **duplicate ids**
  and broken the corpus contract. Ids are now the path relative to the scan root
  (`reports/q1.pdf`), which for a flat directory is byte-identical to the old name — so
  existing corpora and the index signatures built on them are unaffected. There is a
  regression test pinning exactly that.

---

## R4 - Why Decompose collapses on multi-hop, and how far a fix gets

Two runs of my own on 2Wiki (n=2000, jobs `18259143` and `18265982`; ledger entries and raw
per-case data in `docs/hpc-run-log.md` / `results/r1-*`, `results/r2-*`). Both hypotheses were
pre-registered before the numbers were read.

**Step 1 — the collapse is a *ranking* failure, not a retrieval failure.** Re-running the two arms
reproduced MengW7's numbers to four decimals on all five metrics, then per-case pairing showed the
damage shrinks monotonically with depth: R@5 −20.5pp, R@10 −17.3pp, R@20 −9.6pp, top-50 recall
**−0.7pp**. Of the 1854/2000 cases where StrongBM25 ranked the gold document first, Decompose lost
it entirely in only **8** (0.4%) while merely demoting it in 1053 (56.8%) — typically from rank 1 to
rank 3–4. The candidate pool is essentially intact; only the ordering is worse. This refuted my own
pre-registered hypothesis, which predicted the gold would fall out of the pool.

**Mechanism.** RRF sums `1/(k+rank)` across arms. A multi-hop gold document answers exactly *one*
hop, so it ranks highly for one sub-query and is absent from the others, while a document that
ranks mediocrely across *all* sub-queries accumulates more total RRF mass and overtakes it. RRF
structurally rewards breadth across arms and penalises single-point precision — which is precisely
what multi-hop retrieval needs. Compounding it, `DecomposingRetriever` never fused the original
query at all (it was only a fallback for empty LLM output), discarding the ranking that puts gold
first in 92.7% of these cases.

**Step 2 — fixing the fusion works, significantly, but only partly.** I added an
`include_original` option (default off, so no recorded result changes) that fuses the unmodified
query as one more arm:

| Arm | MRR | R@5 | R@10 | R@20 | Recall |
|---|---|---|---|---|---|
| Decompose (baseline) | 0.5702 | 0.4716 | 0.5491 | 0.6506 | 0.7610 |
| **Decompose + original arm** | **0.7155** | 0.5727 | 0.6639 | 0.7371 | **0.7675** |
| StrongBM25 (upper reference) | 0.9580 | 0.6766 | 0.7222 | 0.7468 | 0.7678 |

MRR **+0.1453 (p=0.0000)**, R@10 **+0.1148 (p=0.0000)**, recall flat (+0.0065). But it recovers only
**37%** of the MRR gap, and the recovered fraction rises with depth — MRR 37% < R@5 49% <
R@10 66% < **R@20 90%**. So the original-query arm reliably drags gold back into the top 20 but
cannot win rank 1: the signature of one vote diluted among N sub-query votes. Weighting that arm
rather than adding it at equal weight is the obvious next step.

**Weighting is now implemented but not yet measured (2026-08-05).** Both fusion primitives take
per-arm `weights`, and `DecomposingRetriever` exposes `original_weight` (default `1.0`, so R2's arm
is reproduced exactly and no recorded result moves). Sweep points are in
`configs/experiments/retr_{scifact,2wiki}_decompose-orig-w{2,3,5}.toml`; each weight yields a
distinct index signature, so two sweep points cannot silently share one index.

**What the sweep can and cannot show.** RRF scores `w/(k+rank_original) + Σ 1/(k+rank_sub_i)`, so
as `w → ∞` the original arm dominates and the ranking converges on the base retriever — that is,
`decompose-orig(w→∞) ≡ strong-bm25`. On 2Wiki, where strong-bm25 (.9580) sits far above
decompose-orig at parity (.7155), "MRR rises with weight" is therefore close to structural and
proves little; it merely interpolates between two known endpoints. The question with real content
is whether any *finite* weight **exceeds** strong-bm25, which would mean the sub-query arms add
information on top of the full-query ranking. A monotone climb that never crosses it is the honest
negative: decomposition contributes nothing and the optimal weight is effectively infinite. SciFact
is the judging ground (decompose .5584 / strong-bm25 .6105 — real headroom); 2Wiki runs only as
corroboration. Pre-registered in `docs/hpc-run-log.md` under R4.

**Step 3 — a second fix, and the same question asked on a fairer dataset.** Two questions were left
open, and both were pre-registered before the numbers were read. Does the original-query arm help
generally, or was it repairing damage peculiar to 2Wiki, where BM25 already ranks gold first in 92.7%
of cases so *any* dilution of the full-query ranking must hurt? And does the diagnosed mechanism admit
a more direct remedy — if summing `1/(k+rank)` is what penalises single-hop specialists, taking the
maximum instead should suit them. That is a second option, `fusion="best-rank"`. Run as a 2×2 on
SciFact (decompose MRR 0.5584, so real headroom) and 2Wiki:

| Arm | SciFact MRR (n=300) | 2Wiki MRR (n=2000) |
|---|---|---|
| Decompose (baseline) | 0.5584 | 0.5702 |
| + original arm | 0.5824 | 0.7155 |
| + best-rank fusion | 0.5745 | 0.8059 |
| **+ both** | 0.5938 | **0.9100** |
| StrongBM25 (reference) | **0.6105** | **0.9580** |

- **The original-query arm generalises.** On SciFact it is significant on all four metrics
  (MRR +0.0240 p=0.0000; R@10 p=0.0382; R@20 p=0.0422; recall p=0.0387), and it is the *only* fix
  that reaches significance there. So it is a real improvement, not 2Wiki damage control.
- **Best-rank fusion does not generalise.** On 2Wiki it is the stronger of the two fixes
  (MRR +0.2357 vs +0.1453) and stacks with the other; on SciFact it is significant on nothing, alone
  (p=0.2455) or on top of the original arm (p=0.3671). The mechanism explains the split: taking the
  maximum protects a *dominant single-arm placement*, and 2Wiki has one while SciFact does not
  (StrongBM25's own MRR there is only 0.6105). **The critique of RRF's sum is real but scoped** to
  corpora whose full-query ranking is already strong — it is not a general improvement to fusion.

**What this means practically — stated plainly.** No configuration beats simply using StrongBM25. On
SciFact the best arm (0.5938) is *statistically indistinguishable* from it on all four metrics
(p = 0.15–0.88); on 2Wiki (0.9100) it remains significantly worse (MRR −0.0480, p=0.0000). And this
is after the self-inflicted damage has largely been repaired — 87.6% of the 2Wiki MRR gap and 67.9%
of SciFact's. **So the finding is stronger than "decomposition is broken": it is repaired, and still
not worth it**, since every query costs N extra LLM calls and N extra retrievals to draw level at
best. **Recommendation: do not use Decompose on either corpus. If it is used, `include_original` is
the one switch worth turning on everywhere; `fusion="best-rank"` only pays off where the full-query
ranking is already strong.**

Two things to hold against this report's own earlier claims. A mid-analysis reading that
decomposition "trades top-rank precision for pool coverage" — from SciFact recall 0.8683 vs 0.8624 —
**did not survive the paired test** (p=0.5811) and is withdrawn. And the NQ arm never ran: its
dataset (`runs/niah-base`) was not materialised, so generality was established on SciFact alone, not
on the two datasets the pre-registration promised.

---

## The open question

A still-unresolved risk on the ingestion side: hallucinated image captions are indistinguishable
from real evidence once they enter the retriever's candidate pool. The OCR-smoke test above proves
the happy path works on a clean synthetic figure; it says nothing about hallucination rate on real,
messy documents. We don't yet know how often this happens on our actual corpus, or how much it
would inflate apparent retrieval quality on image-heavy documents versus just adding noise — this
needs measurement, not assumption, and it's a cross-module question since the Generator stage is
what ultimately trusts (or doesn't) the retrieved caption.

---

## Current work — addressing latest feedback (Bharat Arora, 2026-07-26)

- **Bringing actual numbers next time:** done. R2 is a full 3-dataset × 8-variant matrix with
  significance tests; R4 adds two runs of my own that diagnose the one catastrophic result in that
  matrix and measure a fix, both pre-registered before the numbers were read.
- **Finding edge cases where retrieval fails:** done for the clearest one. Decompose on multi-hop is
  a measured, mechanistically explained failure with a quantified partial fix — not a hypothesis.
- **Hallucinated-caption risk:** OCR path verified to work functionally (R3), but hallucination
  *rate* on real documents is still unmeasured. Still needs a sync with the Generator student,
  since the risk spans both modules. Reading the OCR raw did surface a concrete related defect: the
  OCR engine misread `2023 TO 2024` as `2023 T0 2024` on a clean synthetic figure while the Vision
  caption read it correctly, so one document can carry two contradictory readings of the same
  figure. The smoke assertion has been tightened accordingly.
- **Recursive scanning / silent file failures:** **done** (2026-08-05, see R3). Recursion is
  now the default, failures are logged rather than swallowed, and the parse path honours
  `on_error` — which it previously ignored, so a single corrupt file used to abort a whole
  ingest. Ids moved to relative paths so recursion cannot silently collide them.
- **Performance at larger corpus sizes:** not yet started.

---

## Next steps

1. **Weight the original-query fusion arm instead of adding it at equal weight** — R4 shows the arm
   recovers the top-20 but not rank 1, consistent with one vote diluted among N. **Mechanism landed
   2026-08-05, result still pending**: run the `w{2,3,5}` sweep on SciFact (judging ground) with
   2Wiki as corroboration, and pair against the `original_weight=1.0` arm. The pre-registered
   question is **not** "does MRR rise" — that is near-structural, see above — but **whether any
   finite weight beats plain strong-bm25**. Falsifying outcomes: all three weights
   indistinguishable from `w=1` (the dilution account is wrong), or a monotone climb that never
   crosses strong-bm25 (decomposition adds nothing). Both are publishable negatives.
2. **Re-run the NQ arm** — the third dataset the R3 pre-registration promised and did not deliver,
   blocked only on materialising `runs/niah-base`. Generality currently rests on SciFact alone, so a
   second headroom-bearing dataset is what would actually settle it.
3. **Measure hallucinated-caption rate on real (non-synthetic) documents** — the OCR-smoke PASS
   proves the mechanism works, not that captions are trustworthy at scale; requires the
   cross-module sync with Generator noted above.
4. **Test retrieval performance at larger corpus sizes** — check how latency and index build time
   scale well past our current benchmark scale (SciFact 300 docs, NQ/2Wiki 2000), before it
   becomes a blocker.
5. Configurable chunking, to better support structured documents (tables/sections) instead of
   fixed-length splits.
6. Broaden ingestion format coverage (docx/pptx/html via Docling) and surface OCR quality signals
   instead of letting a poor scan degrade silently.
