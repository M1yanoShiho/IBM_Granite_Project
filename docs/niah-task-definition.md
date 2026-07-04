# NIAH Task Definition (methodology record)

- Date: 2026-07-04
- Status: task construction DONE and HPC-validated (job 18000067). This is a
  reference/methodology record for the team to rewrite into the report in their own
  words (Bristol own-work rule; the repo is not the assessed report).
- Code: `src/niah/`, `eval/build_niah_task.py`, `eval/run_niah.py`. Spec:
  `docs/superpowers/specs/2026-07-02-niah-reorientation-design.md` (§5).

## 1. Problem & scope

The IBM brief is "Needle in a Haystack": find *rare*, relevant evidence in
*massive*, *misleading* corpora. We operationalise this as a **retrieval** task on
**public data** (no enterprise data — Meeting-1 constraint), simulating the needle
regime with a rare designated target, diverse misleading distractors, and a
corpus that scales to millions of documents.

First instance: **Natural Questions** dev on the DPR Wikipedia split (`dpr-w100`),
100 queries, ~215k-document background (via `load_benchmark(max_docs=200000)` +
the gold docs, which are always kept).

## 2. Task definition

### 2.1 Needle — a single designated target per query
Each query keeps its full gold set in `qrels` (NQ dev averages **8.29** rel>0 golds
per query — honest labels), but the task designates **one** of them as *the needle*
and the metric tracks only that one (`src/niah/`… `designate_needle`).

- Designation rule (unbiased): among golds that literally contain the gold answer
  string (so a Source-A counterfactual can be built), the one with the **smallest
  doc_id** — deterministic and **uncorrelated with retrieval difficulty**, so it does
  not bias the needle-found metric.
- Why single-target: (a) it is the standard NIAH design (RULER S-/MK-NIAH); (b) it
  is genuinely "rare" (one target vs the 215k haystack); (c) it gives the metric
  dynamic range (0/1 per query) so the scale curve has slope; (d) tracking one
  *known* doc is **robust to NQ's incomplete labels** (many relevant passages are
  unlabelled — D-MERIT — so multi-gold recall is biased; single-target is not).
- Known limitation: the other (undesignated) golds remain relevant docs in the
  haystack — "relevant non-target" — so a retriever ranking one of them above the
  needle counts as a miss. This is inherent to single-target eval on a natural
  multi-gold corpus; the clean fix is the deferred **synthetic-insert** variant
  (RULER/NeedleBench style, one planted needle, no other relevant docs).

### 2.2 Haystack — the corpus and how it scales
`load_benchmark(name, split, max_queries, max_docs)` returns the background + gold.
`max_docs` keeps the first-N distractor docs in `docs_iter()` order → **deterministic
and nested** (first 200k ⊂ first 1M ⊂ …), which is ideal for a degradation curve
(same needle, monotonically growing competition, no between-point sampling noise).
The Phase-1 sweep varies `max_docs` via `load_niah_task(max_docs=…)`.

### 2.3 Misleading hay — three distractor sources + two filters
(`src/niah/` counterfactual.py / generative.py / mining.py / filters.py; §5.1.)

- **Source A — counterfactual** (`make_counterfactual`): an LLM swaps the needle's
  answer entity for a type-consistent but factually wrong one (entity-substitution /
  knowledge-conflict, Longpre et al. 2021). A ~1-token edit → near-identical embedding
  (fools dense) + near-identical terms (fools sparse), but the answer is now wrong.
  **Gated by the answerability judge ONLY** — the positive-anchor margin is
  inapplicable (a good counterfactual scores *high*, like the needle); hardness is
  by construction.
- **Source B — generative** (`make_generative_distractor`): an LLM writes an on-topic,
  plausible passage that shares the query's terms but does NOT answer it (SyNeg-style
  multi-attribute prompting). Gated by the answerability judge.
- **Source C — mined** (`mine_topical`): the union of dense- and sparse-top-k passages
  from the corpus (already real docs). Full filter, but **Filter 2 loosened**
  (`require_both=False`): hard for *either* retriever, since a mined doc came from the
  top-k of at least one arm — this restores distractor yield/diversity.

**Filter 1 — false-negative / answerability** (protects eval validity; ~70% of
naive top negatives are actually relevant, NV-Retriever 2024): a distractor must
score below the needle by a `margin` (positive-anchor, NV-Retriever TopK-MarginPos)
**and** an LLM answerability judge must say it does not answer the query.
**Filter 2 — dual-retriever hardness** (`is_hard`): ranked highly by the retrievers.

Parameters used: `margin=0.05`, `rank_threshold=10`, `mine_k=5`, metric `k=10`, LLM
= `granite-4.1-3b` (generator + judge), sparse arm = SPLADE.

### 2.4 Metrics
- **needle-found@k** = fraction of queries whose designated needle is in the top-k
  (single-target hit-rate); **MRR** of the designated needle (`src/niah/hardness_gate.py`).
- **Hardness gate** (non-saturation): the task is only useful if a strong baseline
  does NOT already solve it — `passes_gate = mean needle-found < 0.95`. The gate is
  **conservative**: the baseline is indexed over the original corpus (which already
  holds the mined Source-C distractors); the injected A/B counterfactuals are not in
  that index but could only push the needle down, so "not saturated" here is a lower
  bound on the true difficulty.
- Significance: paired randomization + bootstrap on per-query differences
  (`eval/significance.py`), for cross-retriever gains.
- Low-prevalence framing: with the needle rare among 215k docs, precision is
  intrinsically low; report recall/hit-rate of the target (PR framing, not ROC).

## 3. Construction & evaluation pipeline
`build_niah_task.main()`: `load_benchmark` → build Granite dense + SPLADE →
`compute_retrieval_signals` (per-query ranks/scores, positive anchor, mined ids) →
`build_task` (A+B+C through the filters) → conservative gate → **`write_task_json`**
(a RECIPE: the load_benchmark args + every distractor's text — ~MBs, not the corpus).
`load_niah_task` rebuilds it deterministically (reload benchmark + re-inject
distractors). `run_niah.py` scores any retriever by needle-found@k/MRR + significance.

## 4. Validation (job 18000067, NQ dev, 215k docs)
- Hardness gate: **needle-found@10 = 0.440, MRR = 0.275, saturated = False,
  passes_gate = True** → hard and non-saturated, with dynamic range for the curve.
- Distractors: **counterfactual 70 / generative 66 / mined 93** (total 229 over 100
  queries) → three balanced types (not a monoculture).
- Persistence: recipe JSON **179 KB** (vs a 135 MB full-corpus dump).

## 5. Scale feasibility (probe: `scripts/run_scale_probe.slurm`)
Granite dense HNSW index build, single BluePebble GPU node (256 GB):

| corpus | build time | peak RAM |
|---|---|---|
| 1M | 1h27m | 36 GB |
| 5M | 6h52m | 173 GB |
| 21M | ~29h (extrap.) | ~730 GB (extrap.) |

→ **fp32 single-node ceiling ≈ 5M documents**; 21M is infeasible (memory- and
time-bound). Reaching 21M needs the **IVFPQ** product-quantized index
(`--index-type ivfpq`, ~32× smaller vectors), at some recall cost — that recall-vs-
compression trade is itself a result.

## 6. Limitations (do not overclaim)
- Public-data *simulation* of the needle regime, not real enterprise data.
- Single-target on a natural multi-gold corpus → "relevant non-target" golds remain
  (synthetic-insert variant is the deferred rigor upgrade).
- Conservative gate is a lower bound (injected A/B not in the baseline index).
- Contextual retrieval deferred: dpr-w100 is pre-chunked with no parent documents,
  so its premise (situate a chunk in its document) does not hold here.

## 7. References (verify bib details before citing)
- RULER — Hsieh et al., 2024 (S-/MK-/MV-NIAH task family).
- NeedleBench — 2024 (single-needle retrieval across information densities).
- D-MERIT — 2024 (partial-annotation bias in IR evaluation; NQ incomplete labels).
- NV-Retriever — Moreira et al., 2024 (positive-aware hard-negative mining; ~70%
  false-negative finding → Filter 1).
- SyNeg — 2024 (LLM-driven synthetic hard negatives → Source B).
- Entity-Based Knowledge Conflicts — Longpre et al., EMNLP 2021 (entity
  substitution → Source A).
- RGB — Chen et al., AAAI 2024 (noise / negative-rejection / counterfactual
  robustness framing).
- Precision–recall vs ROC in low-prevalence retrieval (rare-item metric framing).
