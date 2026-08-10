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

Verified locally, CPU only, on the CI fixture via
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

Structure-aware chunking of *plain-text* corpora (splitting on sections/tables rather than a
word window) is still not implemented; what landed is the ability to select a chunker at all,
plus the one alternative implementation we already had.

**Unrelated defect found while verifying the above, not fixed here:** the frozen baseline
chain committed at `tests/fixtures/reference_baseline_artifacts/` no longer reproduces. Its
`corpus_signature` is `7de07bd…`; current `main` produces `f359854…` from the same fixture and
the same 120/20 chunking, and this is true on a clean tree as well as with the change above.
`dataset_signature`, `chunk_size` and `overlap` all still match, so something in document or
chunk serialisation moved since that fixture was frozen at `ef90665` and nothing in the test
suite asserts the committed signature still reproduces.

## Next steps

1. **Weight the original-query fusion arm instead of adding it at equal weight** — R4 shows the arm
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
4. **Build an inverted index** — R5 showed the remaining cost *is* the linearity, and only this
   changes it. Every query still touches every chunk; the constant has been cut as far as it goes.
   A corpus an order of magnitude larger than SciFact is needed alongside it, since the measured
   range tops out at 5183 documents and the extrapolation past that is an assumption.
5. ~~Configurable chunking~~ — **selecting a chunker landed 2026-08-10 (R7)**. What remains is a
   structure-aware chunker for plain-text corpora; `prechunked` only helps corpora whose units
   were already cut by the loader.
6. Broaden ingestion format coverage (docx/pptx/html via Docling) and surface OCR quality signals
   instead of letting a poor scan degrade silently.
