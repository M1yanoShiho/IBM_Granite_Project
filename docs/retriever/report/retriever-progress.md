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

**Independent reproduction of the NQ arm, and one correction to conclusion 1 (2026-08-11,
job `18421897`, partial).** The NQ dataset R2 used was never committed, so it was re-materialised
from scratch (`base_cli --split dev --corpus-size 100000 --query-limit 2000 --seed 42`) and the
sparse arms re-run. `strong-bm25 vs bm25` comes out at MRR Δ **−0.0004, p=0.9181** against R2's
recorded Δ −0.0004, p=0.92 — the materializer is deterministic and rebuilt the same dataset, so
R2's NQ numbers are reproduced rather than merely trusted. This is the first time any R2 number
has been independently reproduced from a rebuilt dataset.

It also sharpens conclusion 1, which only ever cited MRR. On NQ StrongBM25 is not simply "no
better" than BM25 — it is **significantly worse on every recall metric**:

| Metric | StrongBM25 | BM25 | Δ | p |
|---|---|---|---|---|
| MRR | 0.8153 | 0.8157 | −0.0004 | 0.9181 |
| R@10 | 0.7476 | 0.7598 | **−0.0122** | 0.0024 |
| R@20 | 0.8391 | 0.8508 | **−0.0117** | 0.0006 |
| Recall | 0.9098 | 0.9166 | **−0.0068** | 0.0292 |

So "StrongBM25 vs BM25 is not a reliable win" understates it: on NQ the tuned parameters and
stopword filtering cost recall at every depth, and only the top-rank metric breaks even.

**The hybrid arm reproduces too (job `18421897`, COMPLETED 04:38:23).** `hybrid-rrf vs
strong-bm25` on NQ: MRR 0.8873 vs 0.8153 (**+0.0720**, p=0.0000), R@10 +0.0748, R@20 +0.0736,
recall +0.0611 — all p=0.0000 at n=2000. R2 recorded Δ +0.071 MRR p<0.0001 for this pair on NQ,
so conclusion 2 — the strongest and most consistent result in the whole matrix — is now
independently reproduced from a rebuilt dataset as well.

Two R2 findings therefore survive re-derivation from scratch, and the one correction is to
conclusion 1's scope rather than to its direction.

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

**Format coverage: DOCX, PPTX and HTML (2026-08-11).** `src/evidence_rag/loaders/office_loader.py`,
routed from the same directory dispatcher, so a mixed folder of txt/pdf/image/docx/pptx/html
goes through one entry point. Verified end to end on a mixed recursive directory: relative ids
(`sub/deck.pptx`), one `source_type` per real format rather than everything labelled `pdf`, and
no invented `page_number` — DOCX and HTML have no pages, and a PPTX slide is a section, not a
page.

The reason to do this *now* rather than as generic format coverage: these formats are where
document structure actually lives, and **the tree contained no structured corpus at all** —
SciFact and 2Wiki are plain prose, and the only PDF is a synthetic OCR smoke file. R7's
`section` chunker therefore could not be shown to help or hurt on anything. Hence the default
`office_mode = "markdown"`: export the whole file as one Markdown `Document` and leave the
splitting to the corpus chunker, so `name = "section"` and `name = "word"` can be compared on
the same documents. `office_mode = "chunks"` uses Docling's own HybridChunker instead (pair
with `name = "prechunked"`), matching the PDF loader's contract.

Two behaviours worth recording because they are deliberate rather than accidental: chunk ids
keep the source position, so a dropped empty chunk leaves a gap (`::c1`, `::c3`) rather than
being renumbered into something that was never the second chunk — same as the PDF loader; and
`.htm`/`.html` collapse to one `source_type`, so provenance does not depend on which spelling
a file used.

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

**Weighting is implemented (2026-08-05) and now measured (2026-08-10).** Both fusion primitives take
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

**Result (SciFact, n=300, 2026-08-10): the second falsifying outcome happened.** The climb is
monotone and never crosses strong-bm25 on any of the four metrics.

| Arm | MRR | Δ vs strong-bm25 | p | R@10 | Δ vs strong-bm25 | p |
|---|---|---|---|---|---|---|
| decompose | 0.5584 | −0.0521 | 0.0018 | 0.7094 | −0.0509 | 0.0132 |
| decompose-orig (w=1) | 0.5824 | −0.0281 | — | 0.7304 | −0.0300 | — |
| w=2 | 0.5873 | −0.0232 | 0.1301 | 0.7487 | −0.0117 | 0.5342 |
| w=3 | 0.5949 | −0.0156 | 0.2905 | 0.7587 | −0.0017 | 0.9600 |
| w=5 | 0.5986 | −0.0120 | 0.3742 | 0.7694 | +0.0090 | 0.5617 |
| StrongBM25 | **0.6105** | — | — | **0.7604** | — | — |

Two readings, and they point opposite ways — both were pre-registered, so both are reported.

- **The dilution account survives.** Weighting the original arm is not inert: against `w=1`, `w=3`
  and `w=5` are significant on MRR (+0.0125 p=0.0027; +0.0161 p=0.0239) and on R@10 (+0.0283
  p=0.0111; +0.0390 p=0.0008), and the MRR gap to strong-bm25 closes monotonically — 46% recovered
  at `w=1`, then 55%, 70%, **77%** at `w=5`. Total recall stays flat throughout (`w=5` −0.0042,
  p=0.6781), so what moves is the ordering, not the candidate pool — exactly what Step 1 diagnosed.
  The first falsifying outcome (all three weights indistinguishable from `w=1`) did **not** occur.
- **And it buys nothing that strong-bm25 does not already have.** No finite weight exceeds
  strong-bm25 significantly on any metric; every point estimate on MRR is still negative, and the
  two that turn positive (`w=5` R@10 +0.0090, R@20 +0.0153) are nowhere near significance
  (p=0.5617, p=0.1669). Since `w → ∞` *is* strong-bm25 by construction, the whole climb is
  interpolation towards the base retriever. **The optimal weight is effectively infinite.**

So the R4 recommendation stands and is now measured on its strongest configuration rather than
argued: the best weighted arm (0.5986) is statistically indistinguishable from simply using
strong-bm25 (0.6105) while costing N extra LLM calls and N extra retrievals per query. `w=5` also
edges past the previous best arm, `decompose-orig-bestrank` (0.5938) — which changes nothing,
because that one was already indistinguishable from strong-bm25 too (p=0.1523).

Reproduce with `bash scripts/retriever_significance.sh scifact` (the pairs are wired in).

**2Wiki corroborates, and more sharply (n=2000, job `18380681`).** Same shape, stronger verdict:

| Arm | MRR | Δ vs strong-bm25 | p |
|---|---|---|---|
| decompose-orig (w=1) | 0.7155 | −0.2425 | — |
| w=2 | 0.7912 | −0.1668 | 0.0000 |
| w=3 | 0.8186 | −0.1394 | 0.0000 |
| w=5 | 0.8538 | −0.1042 | 0.0000 |
| StrongBM25 | **0.9580** | — | — |

The climb is again monotone and again never crosses — but here every weight is **significantly
worse** (p=0.0000 throughout), where SciFact's shortfall was merely non-significant. Two datasets,
same direction, and on the one with the most headroom to lose the negative is unambiguous. Note
also that on 2Wiki `decompose-orig-bestrank` (0.9100) beats the best weight (0.8538), consistent
with Step 3: best-rank fusion pays off exactly where the full-query ranking is already strong.

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

## R5 - Retrieval cost at corpus scale, and a correction to my own measurement

The one item in Bharat's feedback with no data behind it. Measured on SciFact at six corpus
sizes, CPU only. Ledger entries R5/R6/R6b; full numbers in `docs/results-summary.md` §R5.

- **Cost is linear in corpus size, with no sub-linear region.** Chunk count ×10.33 gives latency
  ×10.47, and ms per 1k chunks holds inside 16.47–17.86 throughout. Index build is linear and
  negligible — 1.09s to build against 7.7s for fifty queries — so the cost is **per query, not
  per index**. Extrapolated, a million-chunk corpus is ~17 s/query and a million documents
  ~29 s/query. **This does not meet enterprise scale**, and now that is a measurement rather
  than a worry.
- **The cost splits in two**: per-query work repeated without regard to the query (a constant),
  and the absence of an inverted index (the linearity itself).
- **The constant is fixed: 5.27–5.40×, output bit-for-bit identical.** Four quantities depending
  only on the corpus or only on the query were being recomputed in the innermost loop — each
  chunk's term counts, the query's analysis (re-run once per chunk), each term's IDF, each
  chunk's length normalisation. At 8778 chunks, both arms on one node, a query goes from
  **138.4 ms to 26.3 ms**. The test transcribes the pre-rewrite loop as a reference
  implementation and asserts equality exactly, not approximately, because a moved score would
  break every recorded benchmark number.

**I reported this as 8.29–10.17× two days ago and that was wrong.** Its own data said so: index
build came out 10–16% *faster* after a change that makes the build do strictly more work, which
is impossible. The two arms had run on different nodes. Re-run in one allocation on one node,
with a second control arm as a noise floor (drift 2.0%, 214× below the effect), the answer is
5.27–5.40× and the build is correctly 0.89–0.96×.

**The lesson generalises past this experiment: the confound was far larger than the proxy that
revealed it.** Build time differed by 10–16% and I sized the node effect from that. Across nodes
the unoptimised arm varies by 6.9–13.0% — but the *optimised* arm varies by **41.9–73.5%**,
because hoisting moves the hot path from CPU-bound to memory-latency-bound work, which is an
order of magnitude more sensitive to the machine. Estimating a node effect from the part you did
not optimise understates it systematically. Paired comparisons have to run on one node; they
cannot be corrected afterwards.

Two further readings from the confounded run are withdrawn rather than quietly dropped: the
speedup does **not** fall with corpus size (on one node it is flat within ±1.2%), and the residual
super-linearity I inferred was cross-run noise (+5.0% on one node against the before arm's
+3.7%, not the +26.9% first reported).

### Current limitations

- **Memory cost is unmeasured, not zero.** A local estimate of ~2.9 KB/chunk for the new cache
  was contradicted twice by peak RSS, which showed no rise at all. Peak RSS is dominated by
  build-phase transients and is probably the wrong instrument, so the estimate is withdrawn
  rather than reported; settling it needs steady-state sampling.
- The measured range tops out at 5183 documents. Within it the ratio is flat to ~5%, but an
  order of magnitude beyond needs a larger corpus materialised first.
- These are properties of this Python implementation, not of BM25 as an algorithm.

---

## R9 - The inverted index, and what it actually buys

R5 ended by saying the remaining cost *is* the linearity — every query touched every chunk —
and that only an inverted index changes it. It is now built, and the result is more specific
than that sentence implied.

**Correctness first.** The scan now walks the query's posting lists instead of the corpus.
That must not move a score, or every recorded SciFact/NQ/2Wiki number stops being reproducible.
Two things secure it: contributions are accumulated **in query-term order** exactly as the old
inner loop did (float addition is not associative, so any other order changes the last bits),
and length normalisation is precomputed with the expression grouped exactly as it was. Checked
two ways — the bit-for-bit equivalence test against a self-contained transcription of the old
full scan, and the committed frozen candidate chain, which reproduces **byte for byte**.

The equivalence test's reference implementation was also made self-contained as part of this:
it used to read the retriever's own cached `tokens` and `term_frequencies`, and a reference
that shares structure with the thing it checks can only catch a subset of the ways that thing
can be wrong. It now re-analyses the chunks itself.

**What it buys, measured (synthetic corpora, local, CPU).** Cost moved from O(corpus) to
O(postings of the query's terms). Whether that is a win therefore depends entirely on how
selective the query is:

| Corpus | Query terms | Full scan | Inverted | Speed-up |
|---|---|---|---|---|
| 2000 chunks | common | 3.07 ms | 2.05 ms | 1.5× |
| 2000 chunks | rare | 1.81 ms | **0.01 ms** | ~180× |
| 8000 chunks | common | 13.89 ms | 10.19 ms | 1.4× |
| 8000 chunks | rare | 6.82 ms | **0.01 ms** | ~680× |

The rare-term row is flat in corpus size — 0.01 ms at both 2000 and 8000 chunks — which is the
asymptotic change R5 asked for. **The common-term row is still linear**, and that is not a
shortfall of the implementation but the mechanism: a common term's posting list *is* most of
the corpus, so "touch only the chunks containing the term" degenerates to "touch nearly
everything". Cost tracks posting coverage directly — 0.008 ms at df=1 against 4.445 ms at 69.5%
coverage on the same 8000-chunk corpus.

**So the honest correction to R5:** an inverted index does not remove the linearity. It makes
the cost proportional to what the query actually asks for, and the linearity survives exactly
to the extent that the query asks for common terms.

**A consequence worth acting on: the analyzer is now a cost decision, not only a quality one.**
Stopword filtering removes precisely the highest-coverage terms, so it should benefit far more
from postings than plain tokenisation. Predicted, then measured on prose-like text (~45%
stopwords, 8000 chunks, natural-language queries): BM25 **9.12 ms/query**, StrongBM25
**0.01 ms/query** — roughly **900×**.

That reframes R2's verdict on StrongBM25. R2 found it is not a reliable *quality* win (and the
NQ re-run above shows it is significantly worse on recall there). With an inverted index it is
a large *latency* win on natural-language queries, because it never scores the stopword
postings at all. Those are separate axes and should be recommended separately.

### Limitations

- Synthetic corpora only. The vocabulary is Pareto-distributed and the "prose" is generated,
  so the numbers size the effect and identify the mechanism; they are not SciFact or NQ
  figures. The R5 scaling harness on the cluster is where they should be confirmed.
- Memory is again unmeasured. The postings are a transpose of the per-chunk counters they
  replace rather than an addition to them, but Python's per-object overhead on
  `tuple[tuple[int, int], ...]` is real and unquantified — the same gap R5 left open.
- Nothing here changes index *build* time asymptotics; build was already linear and negligible
  against query cost (R5: 1.09 s to build against 7.7 s for fifty queries).

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
- **Performance at larger corpus sizes:** **done, with a fix landed** (2026-08-05/06, see R5).
  Cost is linear in corpus size with no sub-linear region, which puts a million-document corpus
  at roughly 29 s/query — measured, not guessed. The per-query constant has since been cut
  **5.27–5.40×** with bit-for-bit identical output. What remains is the asymptotics, which only
  an inverted index changes.

---

## R6 - Connecting to the shared pipeline with the retriever we actually recommend

The shared infrastructure's acceptance list has been satisfied since 2026-07-15, but the
reference pipeline it was signed off with runs plain BM25 — the v1 baseline, not the arm R2
recommends. So the end-to-end SciFact number the three groups quote came from the weakest
retriever we measured. That is now a config choice rather than a rebuild:
`configs/experiments/scifact_reference_hybrid-rrf.toml` is the same dataset, split, chunking
and IDs as `scifact_reference.toml` with only `[retriever]` changed (RRF over
strong-bm25 + granite-dense). The changed retriever changes the index signature, so it writes
to its own `runs/` directory and the runner refuses to score it against the baseline's index.

**Result (job `18387618`, full `beir/scifact/test`, 5183 documents / 300 queries).** Measured
against the BM25 baseline the shared infrastructure was signed off with (recorded in
`docs/SHARED_RAG_INFRASTRUCTURE_PLAN.md` §8, 2026-07-15):

| Metric | BM25 baseline | Hybrid (RRF) | Δ |
|---|---|---|---|
| retriever document recall | 0.7562 | **0.8711** | +0.1149 |
| system final document recall | 0.6962 | **0.8051** | +0.1089 |
| citation validity | 1.0 | 1.0 | — |
| answer match | unscored | unscored | SciFact has no answer text |

Supporting numbers from the same run: retriever MRR 0.7047 (R2 measured 0.707 for this arm on a
separate `top_k=50` run — an independent reproduction to three decimals), R@5 0.8051, selector
conditional document recall 0.9201, cited document precision 0.2647.

The point worth drawing out is the second row rather than the first. **Essentially all of the
+0.1149 retrieval gain survives to the system output (+0.1089)** — the selector does not eat it.
A retriever improvement that dies downstream would show up here as a large first row and a flat
second one; that is not what happened, so switching the reference pipeline's retriever is a real
end-to-end gain rather than a local one. Answer match stays unscored on both arms, so this says
nothing about generation quality — only about whether the right evidence reaches the generator.

Verified beforehand locally, CPU only, on the CI fixture via
`configs/experiments/reference_baseline_hybrid-rrf.toml` (two sparse arms, no GPU needed):
`all` produces the full six-stage artifact chain with provenance sidecars and
`index_implementation = "hybrid"` in the run manifest. So the fused path goes through the
shared index plugin boundary end to end, not just through the retriever's own tests. The
SciFact run itself needs a GPU and is a cluster submission.

Worth recording, because it contradicts §4.4 of the shared infrastructure plan: that document
still scopes the v1 index boundary to BM25 with "FAISS、StrongBM25、SPLADE、Hybrid 后续按实验
需要增加，不在 v1 范围内", while `composition.py` already routes `strong-bm25`,
`granite-dense` and `hybrid` through `prepare_retriever_index()`. The code is ahead of the
plan. That document is the infrastructure owner's to change, not ours — reported, not edited.

## R7 - Chunking is now a choice of chunker, not just of two numbers

`[chunker] chunk_size`/`overlap` were already configurable, but the chunker *type* was pinned
to `Literal["word"]`, so `PrechunkedChunker` — which exists precisely so structure-aware units
from ingestion (tables, cross-page sections) survive as one chunk each — was unreachable from
any experiment config. Passing it would have crashed anyway: `CorpusBuilder` was typed to
`WordChunker` and read `chunk_size`/`overlap` off it unconditionally.

- `[chunker] name` now accepts `"word"` (default, unchanged) or `"prechunked"`, resolved by
  `corpus.build_chunker()`.
- A window-less chunker records `chunk_size`/`overlap` as `null` in the corpus and run
  manifests rather than an invented number, and the config **rejects** setting either one
  under `name = "prechunked"`. Both matter for the same reason: an inert `chunk_size` would
  let two sweep points look distinct in their configs while producing the identical corpus.
- The word path is unchanged byte for byte. A `reference_baseline.toml` run produces the same
  `corpus_signature` before and after this change (verified by running it on both trees), so
  no recorded SciFact/NQ/2Wiki number moves.
- Full suite, ruff and strict mypy pass.

**Structure-aware chunking landed too (2026-08-11), as `name = "section"`.** `WordChunker`
cuts every `chunk_size` words regardless of what is at that offset, and two of the things it
cuts through matter for retrieval rather than tidiness: **half a table is not evidence** (the
rows that keep the header answer the question, the rows that lose it cannot be read at all),
and a section whose **heading landed in the previous chunk** cannot be matched on its own
subject. `SectionChunker` cuts on structure instead:

- a heading starts a chunk and stays *with* the section it titles;
- a table is never split, even when it alone exceeds `chunk_size` — an oversized table
  becomes one oversized chunk, deliberately, because header-less rows are worse;
- otherwise blocks accumulate up to `chunk_size`;
- a single paragraph longer than `chunk_size` falls back to the sliding window, carrying any
  pending heading into its first piece only.

On corpora with no markup (SciFact, 2Wiki) it degrades to paragraph-aware windowing — still
never cutting mid-paragraph unless the paragraph itself is too long. That is a real
behavioural difference, so it produces a different `corpus_signature` and cannot silently
share a word-chunked index. Verified end to end: `all` on the CI fixture records
`chunker_version = "section-v1"` with its own signature.

The first implementation emitted a **bare heading chunk** when the following table
overflowed — precisely the failure the class exists to prevent, caught by its own test before
commit. A pending heading now never triggers a split. There is a test asserting no chunk is
ever a lone heading, at five different `chunk_size`s, and one asserting `WordChunker` really
does split the same table, so the justification is measured rather than assumed.

Limitations, stated because they bound where this helps: Markdown only (ATX `#` headings,
`|` table rows — reStructuredText, HTML tables and setext headings read as prose); an
oversized table stays oversized (repeating the header row on each piece would be the fix, not
implemented); and structural chunks do not overlap, so a fact spanning a section boundary is
not duplicated into both.

**Not yet measured.** Whether this improves retrieval is an open question, not a claim — it
needs a corpus with real structure to show anything, and SciFact/2Wiki have none. The PDF
ingestion path (`pdf_mode = "pages"`, which emits Markdown) is where it should first be
compared against `word` on equal footing.

## R8 - The committed frozen baseline had stopped reproducing, and nothing noticed

Found while verifying R7, and worth its own entry because it is a cross-module defect: the
frozen chain committed at `tests/fixtures/reference_baseline_artifacts/` could no longer be
rebuilt by the code in the tree. Its `corpus_signature` was `7de07bd…`; current code produces
`f359854…` from the same fixture with the same 120/20 chunking. §4.5 of the shared
infrastructure plan makes that chain the frozen baseline the Selector and Generator groups
read, so a chain claiming a corpus nobody can rebuild is a chain nobody can verify.

**Cause, by bisect: `1a38f52` (multimodal loaders, 2026-07-16) — and it is an accident, not a
schema decision.** That commit explicitly set out to keep signatures byte-stable and added
`_omit_absent_metadata` so `Document`/`EvidenceCandidate` drop a `None` `metadata` on
serialisation. It works — for those two. But `corpus_signature` also hashes the *chunks*, and
`CorpusBuilder` hand-builds those dicts, unconditionally writing `"metadata": null`. `Chunk` is
a dataclass, not a Pydantic model, so the serializer could never have covered it. **The
guarantee held over exactly half of the hash it was written to protect.**

Note the provenance trap that made this hard to see: the chain's `run_manifest.json` records
`git_commit: ef9066…` with `git_dirty: True`. That commit predates the infrastructure layer
entirely — the fixture was generated from an uncommitted tree, so the recorded commit says
nothing about the code that produced it. The chain was committed later, in `224164e`.

**Fix: re-frozen at current code, not reverted.** Restoring byte-stability would change
`corpus_signature` for every current run — invalidating every persisted index on the cluster
and every signature recorded since 2026-07-16, including all of R2–R5. And including chunk
metadata in the corpus signature is *correct* on its own terms: two corpora differing only in
chunk metadata should not share a signature. The defect was the silence, not the value.

Re-freezing surfaced a second staleness nobody had reported: `candidate_sets.jsonl` gained a
whole `retriever` provenance block (implementation, version, parameter hash) at some point
after the freeze. The Selector group's frozen candidates were missing it.

A regression test now pins the property that was missing — `test_frozen_reference_artifacts.py`
asserts the committed chain *rebuilds*, not merely that it agrees with itself. Every other
assertion in that file compares the frozen artifacts against each other, which is why a chain
consistently carrying a signature no code could produce stayed green for three weeks.

## Next steps

1. ~~**Weight the original-query fusion arm instead of adding it at equal weight**~~ — **done and
   answered 2026-08-10 (see R4)**: monotone climb, 77% of the MRR gap recovered at `w=5`, and no
   finite weight exceeds strong-bm25 on any metric. The pre-registered negative. Original text kept
   below for the pre-registration record. R4 shows the arm
   recovers the top-20 but not rank 1, consistent with one vote diluted among N. **Mechanism landed
   2026-08-05, result still pending**: run the `w{2,3,5}` sweep on SciFact (judging ground) with
   2Wiki as corroboration, and pair against the `original_weight=1.0` arm. The pre-registered
   question is **not** "does MRR rise" — that is near-structural, see above — but **whether any
   finite weight beats plain strong-bm25**. Falsifying outcomes: all three weights
   indistinguishable from `w=1` (the dilution account is wrong), or a monotone climb that never
   crosses strong-bm25 (decomposition adds nothing). Both are publishable negatives.
   **Analysis side is wired (2026-08-10)**: `scripts/retriever_significance.sh` now carries both
   pair families — `w{2,3,5}` vs `decompose-orig` (does weight do anything) and `w{2,3,5}` vs
   `strong-bm25` (the pre-registered bar). Only the cluster runs are outstanding:
   ```
   sbatch scripts/run_retriever_eval.slurm \
     configs/experiments/retr_scifact_decompose-orig-w{2,3,5}.toml
   sbatch scripts/run_retriever_eval.slurm \
     configs/experiments/retr_2wiki_decompose-orig-w{2,3,5}.toml
   scripts/retriever_significance.sh scifact   # then, on the login node
   ```
2. **Re-run the NQ arm** — the third dataset the R3 pre-registration promised and did not deliver,
   blocked only on materialising `runs/niah-base`. Generality currently rests on SciFact alone, so a
   second headroom-bearing dataset is what would actually settle it. The blocker is one login-node
   command (dpr-w100 download, so it cannot run on a compute node):
   ```
   python -m evidence_rag.materializer.base_cli --split dev --output runs/niah-base \
     --corpus-size 100000 --query-limit 2000 --seed 42
   ```
   Every `configs/experiments/retr_nq_*.toml` already points at `runs/niah-base/manifest.json`,
   so the arms need no new configuration once it exists.
3. **Measure hallucinated-caption rate on real (non-synthetic) documents** — the OCR-smoke PASS
   proves the mechanism works, not that captions are trustworthy at scale; requires the
   cross-module sync with Generator noted above.
4. ~~**Build an inverted index**~~ — **built and measured 2026-08-11 (R9)**, with a correction
   to the premise: it does not remove the linearity, it makes cost proportional to the query's
   own postings, and the linearity survives for common terms. What remains is confirming the
   synthetic numbers on a real corpus an order of magnitude larger than SciFact, which still
   needs that corpus materialised.
5. ~~Configurable chunking~~ — **done (R7)**: chunker selection landed 2026-08-10, and the
   structure-aware `section` chunker on 2026-08-11. What remains is not implementation but
   evidence: compare `section` against `word` on a corpus that actually has structure (the
   Markdown-emitting PDF path), since neither SciFact nor 2Wiki can show a difference.
6. ~~Broaden ingestion format coverage (docx/pptx/html via Docling)~~ — **done 2026-08-11
   (see R3)**. What remains of this item is surfacing OCR quality signals instead of letting a
   poor scan degrade silently, which is untouched.
7. **Measure `section` against `word` on a genuinely structured corpus.** Now unblocked by the
   two items above — the chunker exists and there is finally a loader that can produce a corpus
   with headings and tables in it. Neither SciFact nor 2Wiki can show anything here, so this
   needs a real DOCX/HTML corpus materialised first. Until then R7 is a mechanism with a
   rationale, not a measured improvement, and should be described that way.
