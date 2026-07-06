# Nested-CV + MRR re-evaluation of gated corroboration — offline design spec

- Date: 2026-07-06
- Owner: P6 (Weikai Mao)
- Status: design approved (brainstorm 2026-07-06) → writing-plans.
- Context: follow-up to `2026-07-06-gated-corroboration-design.md`. The single 150/150
  dev/test split found the gated arm at +0.040 needle-found@10 but **p=0.106 (n=150)** —
  suggestive, underpowered; the un-gated arm collapsed to +0.013 (p=0.68). This adds an
  **outer 5-fold cross-validated re-evaluation** that uses all 300 queries out-of-sample
  (no tuning-on-test) plus **MRR** as a secondary metric. Entirely OFFLINE from the
  existing dump — no LLM, no GPU, no new HPC.

## 1. Problem

The single split throws away half the data as "dev" and rests the whole verdict on one
seed. We want the most powerful *honest* estimate the existing dump allows: every query
scored by a model that never saw it, aggregated to n=300.

## 2. Method — outer K-fold with per-fold selection

Partition the 300 qids into **K=5** deterministic folds. For each outer fold `i`:

1. **Train** = the other K−1 folds (240 queries). Run the existing `sweep_gated` +
   `select_on_dev` on the *training* qids to pick the arms' configs:
   `q2d = (global, None, α=1.0)` (fixed, no selection), `global_corroborate = (global,
   None, best-global-α)`, `gated_corroborate = best gated config (votes/margin family)`.
2. **Test** = fold `i` (60 queries, never touched during selection). Apply each selected
   config to the test qids and record per-query **needle-found@10** *and* **MRR**.

Concatenate the K test folds → each arm has **300 out-of-fold** per-query scores, each
from a config selected without that query. Paired significance (`eval.significance`,
reference `q2d`) on the concatenated vectors.

**Why no inner CV.** A gate config has **no fitted parameters** — selection is
grid-search over hyperparameters only. An inner cross-validation loop to "select on
validation not train" is therefore degenerate: for a fitting-free config, its inner-CV
score equals its score on the whole training partition, so inner selection returns the
*identical* config as selecting on the training fold directly. The **outer** loop is
what removes selection-overfitting from the reported estimate (the test fold is never
used to choose the config). This is stated in the report as the justification for
outer-only CV — a deliberate, understood choice, not an omission.

## 3. Discipline (pre-specified, to avoid p-hacking)

- **Selection metric = needle-found@10 only.** MRR is computed on the *same* selected
  configs and reported alongside — never used to choose configs or folds.
- **Single K=5, one fixed seed** → one clean primary p-value per metric. Repeated-seed
  CV is out of scope (would turn one p-value into many).
- **Per-fold selected configs are printed.** If all folds pick the margin gate with
  similar params → the method is stable; if they scatter → honest evidence of fragility.
  Either way it is reported, not hidden.
- Report whatever the concatenated paired test returns. A null keeps corroboration as an
  honest, underpowered upside under the certified spine (finding 13 remains the headline).

## 4. Units (pure, independently testable)

New in `eval/gate_corroboration.py`:

- `kfold_folds(qids, k, seed) -> List[List[str]]` — sort, seeded shuffle, K near-equal
  contiguous chunks; deterministic, disjoint, union = all.
- `per_query_reciprocal_rank(fused_run, needles) -> Dict[qid, float]` — `1/rank` of the
  designated needle in the fused ranking (0.0 if the needle is absent from the pool).
  **Must use the same deterministic tie-break as `per_query_hits`** (score desc, then
  doc_id asc — the 027d3a3 fix), so @10 and MRR agree on ordering.
- `fuse_for_config(relevance_run, corroboration_run, qids, family, param, alpha) -> Run`
  — the gated fused run for one config on a qid subset (signals → `gate_mask` → restrict
  → `gated_fuse`). `evaluate_config` is refactored to delegate to it (identical output,
  removes duplication) so @10 and MRR score the *same* fused run.
- `nested_cv(relevance_run, corroboration_run, needles, k, seed, alpha_grid, k_cut)
  -> (hits_by_arm, rr_by_arm, per_fold_configs)` — the outer-fold loop above; returns two
  `{arm: {qid: score}}` maps (needle-found@k_cut and MRR) over all 300 qids, plus the
  list of per-fold selected configs.

Reused verbatim: `sweep_gated`, `select_on_dev`, `gate_mask`, `gated_fuse`,
`max_votes_signal`, `margin_signal`, `per_query_hits`, `_grid`, `write_per_query_csv`.

## 5. CLI & outputs

Extend the existing command with `--nested-cv` (flag) and `--folds` (int, default 5).
When set, after the existing single-split analysis, `main` also runs `nested_cv` and
writes:

- `results/corroboration_nested_cv_per_query.csv` — needle-found@k, columns
  `qid, q2d, global_corroborate, gated_corroborate` over all 300.
- `results/corroboration_nested_cv_mrr.csv` — same columns, MRR.

and prints the per-fold selected configs and the two significance commands:
`python -m eval.significance --per-query-csv <csv> --reference q2d` (for @k and for MRR).

## 6. Testing (TDD)

- `kfold_folds`: deterministic per seed; disjoint; union = all; 5 folds of 60 on 300;
  input-order-independent (sorted first).
- `per_query_reciprocal_rank`: needle at rank 1 → 1.0, rank 2 → 0.5, absent → 0.0;
  exact-tie ordering matches `per_query_hits` (needle tied with a distractor at doc_id
  order resolves the same way both metrics see it).
- `fuse_for_config`: equals `gated_fuse` on the restricted, masked inputs; and
  `per_query_hits(fuse_for_config(...))` equals the (refactored) `evaluate_config` — a
  regression guard that the refactor changed nothing.
- `nested_cv`: on a synthetic dump, the concatenated per-arm vectors cover exactly all
  qids (disjoint folds); `q2d`'s nested vector equals `per_query_hits` at α=1 over the
  full set (fixed config, no selection); a constructed fold where the training set makes
  the gate select the margin family and it fixes a held-out counterfactual query;
  `per_fold_configs` has one entry per fold.
- CLI/`main`: `--nested-cv` writes both CSVs with the right headers and 300 rows, prints
  per-fold configs and both significance commands (end-to-end on a synthetic dump).

## 7. Risks & limitations

- **Still one dataset / one first stage / this dump's top-20 pool.** CV improves the
  estimate's honesty and power, not external validity.
- **Fold-config instability** is possible (folds picking different gates); it is
  surfaced, not smoothed away.
- **Power is bounded by 300 queries.** If the +0.04 effect is real and stable across
  folds it should approach significance at n=300; if the paired test is still null,
  corroboration is reported as an honest underpowered upside (Track-3 narrative).

## 8. Not doing (YAGNI)

- Inner CV (degenerate for fitting-free selection — §2).
- Repeated-seed / stratified CV (would fragment the single p-value).
- Selecting on MRR, or any metric-shopping.
- Online wiring / new HPC extraction.
