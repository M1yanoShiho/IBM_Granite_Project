# Gated dynamic-alpha Corroboration — offline design spec

- Date: 2026-07-06
- Owner: P6 (Weikai Mao)
- Status: design approved (brainstorm 2026-07-06); spec-review gate waived by user →
  writing-plans.
- Context: follow-up to the certified Corroboration Reranking result (finding 15:
  q2d_corroborate 0.610 vs q2d 0.570, +0.040, p=0.022, n=300, alpha*=0.6). Parent spec:
  `2026-07-05-corroboration-reranking-design.md`. Entirely OFFLINE — pure arithmetic on
  the dumped runs, no LLM, no GPU, no new HPC extraction.

## 1. Problem

Two open issues with finding 15:

1. **The caveat:** alpha* = 0.6 was selected on the same 300 queries the p-value was
   computed on (mild tuned-on-test optimism; replicated across two extractions, but the
   headline number is still not protocol-clean).
2. **The blend is global.** One alpha for every query. The corroboration signal is
   plausibly heterogeneous: where a real consensus exists it separates the needle from a
   lone counterfactual; where consensus is weak (max vote count 1 — two extractions
   agreeing by chance, or the parametric answer echoing one wrong passage) blending may
   *hurt* queries the first stage already had right. A per-query **gate** might keep the
   fixes and drop the breaks.

Note a structural fact that shapes the hypothesis space: `minmax_normalize` maps an
all-equal score set to all-1.0 (a constant), and a convex blend with a constant is
order-preserving. So queries with **zero** consensus (all vote counts equal, typically
all 0) are already unaffected by the global blend. The gate's only room to act is the
**weak-consensus region** (and the relevance-confidence region) — hence the tau grid
starts where tau=1 reproduces the global blend as a built-in sanity check.

## 2. Goal

One offline experiment, two deliverables:

1. **Re-certify finding 15 under an honest protocol** — dev/test split; alpha tuned on
   dev only, needle-found@10 + paired p reported on the held-out test half. This removes
   the caveat regardless of how gating turns out.
2. **Test gating** — does a per-query gated blend beat the global blend on the test
   half? A null ("global is enough") is a reportable sensitivity finding, not a failure.

## 3. Data & feasibility

Input = the runs dump `results/corroboration_runs_nq300cert.json` (job 18025180
artifact, currently on the HPC — scp it down, or run this module on the login node;
the code is identical either way and finishes in seconds on CPU). Schema (produced by
`eval.tune_corroboration.dump_runs`):

- `relevance_run`: qid → {doc_id: raw q2d first-stage score} (top-20 doc-level pool)
- `corroboration_run`: qid → {doc_id: raw cross-source vote count}
- `needles`: qid → designated needle doc_id

Both per-query gate signals are computable from this dump alone (§4), so the whole
experiment — split, sweep, selection, certification, flip analysis — is pure arithmetic.

## 4. Method

**Split.** Sort qids, shuffle with a fixed seed (`random.Random(seed)`, default 0),
first 150 = dev, remaining 150 = test. Deterministic, disjoint, union = all. Seed is
recorded in every output.

**Gate signals** (per query, from the dump):

- `max_votes(q)` = max of `corroboration_run[q]` values (0.0 if empty). "How strong is
  the strongest consensus in the pool."
- `margin(q)` = top1 − top2 of `minmax_normalize(relevance_run[q])` (0.0 if fewer than
  2 docs; the all-equal degenerate case maps to all-1.0 so margin = 0 and the gate
  fires — harmless, the blend then decides). "How confident is the first stage." Same
  non-degenerate confidence signal as `CorrectiveRAGPipeline`.

**Gate families** (a gate decides, per query, blend vs keep pure relevance):

- **votes**: blend iff `max_votes(q) >= tau`, tau ∈ {1, 2, 3, 4}. tau=1 must reproduce
  the global blend's metric exactly (§1) — asserted in tests.
- **margin**: blend iff `margin(q) < m`, m ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}
  (blend only where the first stage is unsure).
- **global** (baseline family): blend always — the existing method; also the alpha
  carrier for deliverable 1.

Blend = existing `fuse_one`/`convex_fuse` (min-max + convex, alpha = relevance weight);
gated-off queries rank by relevance alone (equivalent to alpha = 1 for that query).
Alpha grid = `eval.tune_alpha._grid(alpha_step)`, default step 0.1.

**Selection protocol (dev only).** Within each family pick max dev needle-found@10;
ties break to larger alpha (closer to pure relevance, matching `best_alpha`), then to
the stricter gate (larger tau / smaller m — fires less often). Across families pick max
dev score; ties break global > votes > margin (simpler wins).

**Certification (test only).** Exactly three arms, parameters frozen from dev:
`q2d` (alpha = 1), `global_corroborate` (dev alpha*), `gated_corroborate` (the best
**gated** config from dev, i.e. the winner among the votes/margin families — reported
even when the across-families winner is global, so "gated vs global" is always
measured; in that case it is a sensitivity note, not a claim, because dev already
preferred global). Per-query hits → CSV → existing `eval.significance` paired test:

- global vs q2d — **the de-caveated finding-15 re-certification**
- gated vs global — does gating add anything
- gated vs q2d — combined effect

**Flip analysis (full 300q, descriptive, no tuning).** At the certified alpha
(`--flip-alpha`, default 0.6): classify each query fixed (miss→hit at k=10), broken
(hit→miss), unchanged-hit, unchanged-miss; cross-tab counts by `max_votes` bucket
{0, 1, 2, 3, 4+} and `margin` bucket {<0.05, 0.05–0.1, 0.1–0.2, ≥0.2}. This is the
evidence for/against gating's premise and goes in the report either way. Descriptive
use of all 300 queries is fine — nothing is selected from it.

## 5. Module & integration

New module `eval/gate_corroboration.py` — single purpose: all offline analysis from a
runs dump. Reuses verbatim: `tune_corroboration.load_runs` / `per_query_hits` /
`needle_found_at_k`, `fusion.fuse_one`/`convex_fuse`/`minmax_normalize`,
`tune_alpha._grid`, `run_benchmark.write_per_query_csv`. `tune_corroboration.py` (the
module that produced the certified artifact) is not modified.

Pure units (each independently testable, no IO):

- `split_queries(qids, seed, dev_fraction=0.5) -> (dev_qids, test_qids)`
- `max_votes_signal(corroboration_run) -> Dict[qid, float]`
- `margin_signal(relevance_run) -> Dict[qid, float]`
- `gated_fuse(relevance_run, corroboration_run, alpha, gate: Dict[qid, bool]) -> Run`
- `sweep_gated(...) -> curves` (family × param × alpha → score, n_gated)
- `select_on_dev(curves) -> winner` (implements §4 tie-breaks)
- `flip_table(relevance_run, corroboration_run, needles, alpha, k) -> rows`

CLI: `python -m eval.gate_corroboration --from-runs <json> [--k 10] [--seed 0]
[--alpha-step 0.1] [--flip-alpha 0.6] [--out-dir results]`. Prints the
`eval.significance` command for the test CSV (same pattern as `tune_corroboration`).

## 6. Outputs

- `results/corroboration_flip_table.csv` — columns: signal, bucket, fixed, broken,
  unchanged_hit, unchanged_miss.
- `results/corroboration_gate_dev_curves.csv` — columns: family, param, alpha,
  needle_found@k, n_gated (the sensitivity artifact; whole curves, no cherry-picking).
  For the global family `param` is empty and `n_gated` = the dev size.
- `results/corroboration_gate_test_per_query.csv` — per-query hits for the three test
  arms (feeds `eval.significance`).
- stdout summary: seed, split sizes, dev-selected parameters per family, overall
  winner, test scores for the three arms, the significance command.

## 7. Testing (TDD)

- Signals: `max_votes` on normal/empty/all-zero runs; `margin` on spread scores,
  all-equal (→ 0.0), single-doc (→ 0.0).
- `gated_fuse`: gate-off query ranks identically to pure relevance; gate-on query
  matches `fuse_one`; mixed map.
- Sanity equivalences: votes gate tau=1 == global blend needle-found (the §1
  structural fact); alpha=1.0 == pure q2d for every family.
- `split_queries`: deterministic per seed, disjoint, covers all, 150/150 on 300.
- `select_on_dev`: tie-break order (alpha, then strictness, then family) exercised.
- `flip_table`: hand-built 3-query run with one fixed / one broken / one unchanged.
- End-to-end on a synthetic dump: a query whose lone counterfactual sits top-1 by
  relevance with a 2-vote consensus needle below → gated (tau=2) fixes it while a
  no-consensus query is left untouched; runs via the CLI entry with `--from-runs`.

## 8. Risks & honest limitations

- **Power:** n=150 per half. The re-certified delta may lose significance even if
  real; report whatever comes out (the n=300 tuned number stays in the report as the
  sensitivity curve, labelled as tuned).
- **Gating adds a hyperparameter dimension** (family + tau/m). Only test-half numbers
  are protocol-clean claims; dev curves are published as sensitivity, never as claims.
- **Conditionality:** conclusions are conditional on the q2d first stage, this dump's
  top-20 window, and the frozen 300q NQ task — one dataset, one first stage.
- **Dump immutability:** the dump is a certified-run artifact; this module only reads
  it. Any re-extraction would be a new experiment, not this one.

## 9. Not doing (YAGNI)

- Continuous per-query alpha (alpha_q = f(signal)) — too many degrees of freedom for
  n=150 dev; hard to explain in the report.
- Combined gates (votes AND/OR margin).
- Online wiring into `CorroborationReranker` / `run_niah` — only worth it if the
  offline evidence says gating wins; separate change if so.
- New HPC extraction of any kind.
