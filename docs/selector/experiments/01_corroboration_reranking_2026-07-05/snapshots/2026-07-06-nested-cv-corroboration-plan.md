# Nested-CV + MRR for gated corroboration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline outer-5-fold cross-validated re-evaluation (+ MRR) of the gated corroboration arms to `eval/gate_corroboration.py`, giving an honest n=300 out-of-sample estimate with no tuning-on-test.

**Architecture:** Five new pure functions appended to the existing module (`per_query_reciprocal_rank`, `kfold_folds`, `fuse_for_config`, a delegating refactor of `evaluate_config`, and `nested_cv`), wired into `main` behind a `--nested-cv` flag. Reuses `sweep_gated` / `select_on_dev` / `gate_mask` / `gated_fuse` / `per_query_hits` / `_grid` / `write_per_query_csv` verbatim. Pure arithmetic on the runs dump — no LLM/GPU. Spec: `docs/superpowers/specs/2026-07-06-nested-cv-corroboration-design.md`.

**Tech Stack:** Python stdlib (random/csv/pathlib/argparse), pytest. No new dependencies.

---

## File structure

- Modify: `eval/gate_corroboration.py` — append the 5 units + extend `_parse_args`/`main`.
- Modify: `tests/test_gate_corroboration.py` — append tests; extend the top import block.
- Not modified: `eval/tune_corroboration.py` (import `per_query_hits`/`dump_runs` only), `src/retrieval/fusion.py`, `eval/run_benchmark.py` (`write_per_query_csv`, already lazily imported in `main`).

Run the module's tests with `python -m pytest tests/test_gate_corroboration.py -v`. The file currently has **32 passing tests**; imports are consolidated at the top of the test file (keep them there — add new names to the existing blocks, do not re-import mid-file).

**Type vocabulary:** a `Run` is `Dict[qid, Dict[doc_id, float]]`; `needles` is `Dict[qid, needle_doc_id]`; per-query score maps are `Dict[qid, float]`.

---

### Task 1: `per_query_reciprocal_rank` (MRR metric)

**Files:**
- Modify: `eval/gate_corroboration.py` (add function after `gated_fuse`)
- Modify: `tests/test_gate_corroboration.py` (extend imports + append tests)

- [ ] **Step 1: Write the failing tests**

In `tests/test_gate_corroboration.py`, add `per_query_reciprocal_rank` to the `from eval.gate_corroboration import (...)` block and add `per_query_hits` to the `from eval.tune_corroboration import ...` line (so it reads `from eval.tune_corroboration import dump_runs, per_query_hits`). Then append:

```python
def test_reciprocal_rank_needle_first_is_one():
    fused = {"q": {"n1": 0.9, "a": 0.5, "b": 0.1}}
    assert per_query_reciprocal_rank(fused, {"q": "n1"}) == {"q": 1.0}


def test_reciprocal_rank_needle_second_is_half():
    fused = {"q": {"a": 0.9, "n1": 0.5}}
    assert per_query_reciprocal_rank(fused, {"q": "n1"}) == {"q": 0.5}


def test_reciprocal_rank_needle_absent_is_zero():
    fused = {"q": {"a": 0.9, "b": 0.5}}          # needle not in the pool at all
    assert per_query_reciprocal_rank(fused, {"q": "n1"}) == {"q": 0.0}


def test_reciprocal_rank_tie_break_matches_per_query_hits():
    # exact tie must resolve the SAME deterministic way both metrics see it
    # (score desc, then doc_id asc): 'a' leads 'z', so needle 'z' is rank 2.
    fused = {"q": {"a": 1.0, "z": 1.0}}
    assert per_query_reciprocal_rank(fused, {"q": "z"}) == {"q": 0.5}
    assert per_query_hits(fused, {"q": "z"}, k=1) == {"q": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`per_query_reciprocal_rank`).

- [ ] **Step 3: Write minimal implementation**

In `eval/gate_corroboration.py`, add this function immediately after `gated_fuse` (before `split_queries`):

```python
def per_query_reciprocal_rank(
    fused_run: Run, needles: Dict[str, str]
) -> Dict[str, float]:
    """Per-query ``1/rank`` of the designated needle in ``fused_run`` (0.0 if the
    needle is absent from the pool). Uses the SAME deterministic tie-break as
    ``per_query_hits`` (score descending, then doc_id ascending) so needle-found@k
    and MRR agree on the ranking of tied documents."""
    out: Dict[str, float] = {}
    for qid, needle in needles.items():
        scores = fused_run.get(qid, {})
        ranked = sorted(sorted(scores), key=lambda d: scores[d], reverse=True)
        out[qid] = 1.0 / (ranked.index(needle) + 1) if needle in ranked else 0.0
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 36 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): per_query_reciprocal_rank (MRR) with per_query_hits-consistent tie-break"
```

---

### Task 2: `kfold_folds` (deterministic K-fold partition)

**Files:**
- Modify: `eval/gate_corroboration.py` (add function after `split_queries`)
- Modify: `tests/test_gate_corroboration.py` (extend imports + append tests)

- [ ] **Step 1: Write the failing tests**

Add `kfold_folds` to the `from eval.gate_corroboration import (...)` block, then append:

```python
def test_kfold_folds_partitions_deterministically():
    qids = [f"q{i}" for i in range(300)]
    folds1 = kfold_folds(qids, 5, seed=0)
    folds2 = kfold_folds(list(reversed(qids)), 5, seed=0)   # input order irrelevant
    assert folds1 == folds2
    assert [len(f) for f in folds1] == [60, 60, 60, 60, 60]
    flat = [q for f in folds1 for q in f]
    assert len(flat) == 300 and set(flat) == set(qids)      # disjoint (len) + union = all


def test_kfold_folds_seed_changes_partition():
    qids = [f"q{i}" for i in range(300)]
    assert kfold_folds(qids, 5, 0) != kfold_folds(qids, 5, 1)


def test_kfold_folds_uneven_sizes_are_near_equal():
    folds = kfold_folds([f"q{i}" for i in range(10)], 3, seed=0)
    assert sorted(len(f) for f in folds) == [3, 3, 4]
    flat = [q for f in folds for q in f]
    assert len(flat) == 10 and len(set(flat)) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`kfold_folds`).

- [ ] **Step 3: Write minimal implementation**

In `eval/gate_corroboration.py`, add this function immediately after `split_queries` (before `evaluate_config`):

```python
def kfold_folds(qids: List[str], k: int, seed: int = 0) -> List[List[str]]:
    """Partition ``qids`` into ``k`` near-equal disjoint folds, deterministically.

    Sort first (so the partition depends only on the qid set + seed, not input
    order), seeded-shuffle, then cut into ``k`` contiguous chunks with round-based
    boundaries (sizes differ by at most 1). Union of the folds = all qids.
    """
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    return [ordered[round(f * n / k):round((f + 1) * n / k)] for f in range(k)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 39 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): deterministic k-fold partition for nested-CV"
```

---

### Task 3: `fuse_for_config` + delegate `evaluate_config`

**Files:**
- Modify: `eval/gate_corroboration.py` (add `fuse_for_config` before `evaluate_config`; rewrite `evaluate_config` body)
- Modify: `tests/test_gate_corroboration.py` (extend imports + append tests)

This extracts the "build the gated fused run for one config on a qid subset" logic so both the @k metric and MRR score the *same* fused run. `evaluate_config`'s observable behavior is unchanged (regression-guarded).

- [ ] **Step 1: Write the failing tests**

Add `fuse_for_config` to the `from eval.gate_corroboration import (...)` block, then append:

```python
def test_fuse_for_config_equals_gated_fuse_on_masked_inputs():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 2.0}}
    fused = fuse_for_config(rel, cor, ["q1"], "global", None, 0.5)
    mask = gate_mask("global", None, max_votes_signal(cor), margin_signal(rel))
    assert fused == gated_fuse(rel, cor, 0.5, mask)


def test_evaluate_config_unchanged_after_refactor():
    # regression: same output as before the fuse_for_config extraction
    rel = {"q1": {"cf1": 0.99, "n1": 0.5}, "q2": {"cf2": 0.99, "n2": 0.5}}
    cor = {"q1": {"cf1": 0.0, "n1": 2.0}, "q2": {"cf2": 0.0, "n2": 0.0}}
    needles = {"q1": "n1", "q2": "n2"}
    hits = evaluate_config(rel, cor, needles, ["q1", "q2"], "votes", 2.0, 0.2, k=1)
    assert hits == {"q1": 1.0, "q2": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`fuse_for_config`).

- [ ] **Step 3: Write minimal implementation**

In `eval/gate_corroboration.py`, insert `fuse_for_config` immediately BEFORE the `evaluate_config` definition:

```python
def fuse_for_config(
    relevance_run: Run,
    corroboration_run: Run,
    qids: List[str],
    family: str,
    param: Optional[float],
    alpha: float,
) -> Run:
    """The gated fused run for ONE (family, param, alpha) config on a qid subset.

    Signals over the full runs -> gate mask -> restrict to ``qids`` -> ``gated_fuse``.
    Extracted so needle-found@k and MRR score the exact same fused ranking, and so
    the sweep/selection/nested-CV paths share one definition of "fuse a config".
    """
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    mask = gate_mask(family, param, votes, margins)
    rel = {q: relevance_run[q] for q in qids}
    cor = {q: corroboration_run.get(q, {}) for q in qids}
    return gated_fuse(rel, cor, alpha, mask)
```

Then REPLACE the body of `evaluate_config` so it delegates (keep the signature and docstring; only the body changes):

```python
def evaluate_config(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    family: str,
    param: Optional[float],
    alpha: float,
    k: int,
) -> Dict[str, float]:
    """Per-query needle-found hits for ONE (family, param, alpha) config, restricted
    to ``qids`` (the dev or test half). The single definition of "score a config" --
    the sweep, the selection, and the test arms all go through here."""
    fused = fuse_for_config(relevance_run, corroboration_run, qids, family, param, alpha)
    nee = {q: needles[q] for q in qids}
    return per_query_hits(fused, nee, k)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 41 passed (the existing sweep/select/end-to-end tests still pass — `evaluate_config` output is unchanged).

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "refactor(eval): extract fuse_for_config; evaluate_config delegates to it"
```

---

### Task 4: `nested_cv` (outer K-fold, per-fold selection)

**Files:**
- Modify: `eval/gate_corroboration.py` (add function after `select_on_dev`)
- Modify: `tests/test_gate_corroboration.py` (extend imports + add a helper + append tests)

- [ ] **Step 1: Write the failing tests**

Add `nested_cv` to the `from eval.gate_corroboration import (...)` block, then append this helper and tests:

```python
def _gated_dump(n):
    """n identical fixable queries: the lone counterfactual tops relevance, the
    needle holds a 2-vote consensus (q2d misses all; a low-alpha blend recovers)."""
    rel, cor, needles = {}, {}, {}
    for i in range(n):
        q = f"q{i}"
        rel[q] = {f"cf{i}": 0.99, f"n{i}": 0.5}
        cor[q] = {f"cf{i}": 0.0, f"n{i}": 2.0}
        needles[q] = f"n{i}"
    return rel, cor, needles


def test_nested_cv_scores_every_query_out_of_fold():
    rel, cor, needles = _gated_dump(12)
    hits, rr, cfgs = nested_cv(rel, cor, needles, k=1, seed=0, folds=3,
                               alpha_grid=[0.0, 0.2, 0.5, 1.0])
    for arm in ("q2d", "global_corroborate", "gated_corroborate"):
        assert set(hits[arm]) == set(needles)     # each query scored exactly once
        assert set(rr[arm]) == set(needles)
    assert len(cfgs) == 3                          # one selected-config set per fold


def test_nested_cv_q2d_equals_pure_first_stage_over_all():
    rel, cor, needles = _gated_dump(12)
    hits, rr, cfgs = nested_cv(rel, cor, needles, k=1, seed=0, folds=3,
                               alpha_grid=[0.0, 0.5, 1.0])
    # q2d is (global, None, alpha=1.0) on every fold -> equals pure first stage
    full = fuse_for_config(rel, cor, list(needles), "global", None, 1.0)
    assert hits["q2d"] == per_query_hits(full, needles, k=1)


def test_nested_cv_gate_recovers_needles_q2d_misses():
    rel, cor, needles = _gated_dump(12)
    hits, rr, cfgs = nested_cv(rel, cor, needles, k=1, seed=0, folds=3,
                               alpha_grid=[0.0, 0.2, 0.5, 1.0])
    assert sum(hits["q2d"].values()) == 0                    # counterfactual ranks first
    assert sum(hits["gated_corroborate"].values()) == 12     # all recovered out-of-fold
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`nested_cv`).

- [ ] **Step 3: Write minimal implementation**

In `eval/gate_corroboration.py`, add this function immediately after `select_on_dev` (before the `_VOTE_BUCKETS` block):

```python
def nested_cv(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    k: int,
    seed: int,
    folds: int,
    alpha_grid: List[float],
) -> "Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict[str, tuple]]]":
    """Outer K-fold, per-fold selection -> honest out-of-fold scores for the 3 arms.

    For each outer fold: select the arms' configs on the OTHER folds (train) via
    ``sweep_gated`` + ``select_on_dev`` (selection metric = needle-found@k), then apply
    each selected config to the held-out fold, recording needle-found@k AND MRR. Every
    query is thus scored by a config that never saw it. Returns
    ``(hits_by_arm, rr_by_arm, per_fold_configs)`` — the two score maps span all qids;
    ``per_fold_configs[i]`` is fold i's ``{arm: (family, param, alpha)}`` (transparency:
    stable selection vs fold-to-fold drift). No inner CV: the config is fitting-free, so
    inner selection reduces to selecting on the training partition (see the design spec).
    """
    arms = ("q2d", "global_corroborate", "gated_corroborate")
    hits_by_arm: Dict[str, Dict[str, float]] = {a: {} for a in arms}
    rr_by_arm: Dict[str, Dict[str, float]] = {a: {} for a in arms}
    per_fold_configs: List[Dict[str, tuple]] = []
    fold_lists = kfold_folds(list(needles), folds, seed)
    for i, test_qids in enumerate(fold_lists):
        train_qids = [q for j, fold in enumerate(fold_lists) if j != i for q in fold]
        rows = sweep_gated(relevance_run, corroboration_run, needles, train_qids, alpha_grid, k)
        best, _winner, best_gated = select_on_dev(rows)
        configs = {
            "q2d": ("global", None, 1.0),
            "global_corroborate": ("global", None, best["global"].alpha),
            "gated_corroborate": (best_gated.family, best_gated.param, best_gated.alpha),
        }
        per_fold_configs.append(configs)
        nee_test = {q: needles[q] for q in test_qids}
        for name, (fam, par, al) in configs.items():
            fused = fuse_for_config(relevance_run, corroboration_run, test_qids, fam, par, al)
            hits_by_arm[name].update(per_query_hits(fused, nee_test, k))
            rr_by_arm[name].update(per_query_reciprocal_rank(fused, nee_test))
    return hits_by_arm, rr_by_arm, per_fold_configs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 44 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): nested_cv - outer k-fold per-fold selection, out-of-fold hits + MRR"
```

---

### Task 5: CLI `--nested-cv`/`--folds` + `main` wiring

**Files:**
- Modify: `eval/gate_corroboration.py` (`_parse_args` + `main`)
- Modify: `tests/test_gate_corroboration.py` (append end-to-end test)

- [ ] **Step 1: Write the failing test**

Append (no new imports needed — `main`, `csv`, `_synthetic_dump` are already in the file):

```python
def test_main_nested_cv_writes_cv_artifacts(tmp_path, capsys):
    runs = _synthetic_dump(tmp_path)          # the 4-query dump helper from Task 9
    out = tmp_path / "out"

    main(["--from-runs", str(runs), "--k", "1", "--folds", "2",
          "--nested-cv", "--out-dir", str(out)])

    hits_csv = out / "corroboration_nested_cv_per_query.csv"
    mrr_csv = out / "corroboration_nested_cv_mrr.csv"
    assert hits_csv.exists() and mrr_csv.exists()
    with open(hits_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"qid", "q2d", "global_corroborate", "gated_corroborate"}
    assert len(rows) == 4                      # all 4 queries scored out-of-fold

    printed = capsys.readouterr().out
    assert "nested-CV" in printed
    assert str(mrr_csv) in printed and "--reference q2d" in printed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate_corroboration.py::test_main_nested_cv_writes_cv_artifacts -v`
Expected: FAIL — `--nested-cv` is an unrecognized argument (SystemExit), or the CSVs are not written.

- [ ] **Step 3: Write minimal implementation**

In `eval/gate_corroboration.py`, add two arguments to `_parse_args` (just before `return p.parse_args(argv)`):

```python
    p.add_argument("--nested-cv", action="store_true", dest="nested_cv",
                   help="Also run outer K-fold nested-CV (honest n=300 out-of-sample) "
                        "+ MRR, from the same dump.")
    p.add_argument("--folds", type=int, default=5,
                   help="Nested-CV outer folds (default: %(default)s).")
```

Then in `main`, append this block at the very END of `main` — after the single-split significance-command prints (lines that print `--reference q2d` / `--reference global_corroborate`), so the single-split report is complete before the nested-CV report begins:

```python
    if args.nested_cv:
        hits_cv, rr_cv, fold_cfgs = nested_cv(
            relevance_run, corroboration_run, needles,
            args.k, args.seed, args.folds, _grid(args.alpha_step),
        )
        hits_csv = args.out_dir / "corroboration_nested_cv_per_query.csv"
        mrr_csv = args.out_dir / "corroboration_nested_cv_mrr.csv"
        write_per_query_csv(hits_cv, hits_csv)
        write_per_query_csv(rr_cv, mrr_csv)
        print(f"nested-CV: folds={args.folds} seed={args.seed} "
              f"(fitting-free selection -> outer CV only, no inner loop)")
        for i, cfg in enumerate(fold_cfgs):
            print(f"  fold {i}: gated={cfg['gated_corroborate']} "
                  f"global_alpha={cfg['global_corroborate'][2]}")
        for arm in ("q2d", "global_corroborate", "gated_corroborate"):
            print(f"  cv {arm}: needle_found@{args.k}={_mean(hits_cv[arm]):.4f} "
                  f"MRR={_mean(rr_cv[arm]):.4f}")
        print(f"wrote {hits_csv} and {mrr_csv} -> significance:")
        print(f"  python -m eval.significance --per-query-csv {hits_csv} --reference q2d")
        print(f"  python -m eval.significance --per-query-csv {mrr_csv} --reference q2d")
```

(The existing single-split behavior is unchanged; `--nested-cv` defaults off, so the existing `test_main_end_to_end_writes_all_artifacts` still passes.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 45 passed.

- [ ] **Step 5: Run the FULL suite**

Run: `python -m pytest -q`
Expected: all green, no NEW failures (baseline was 460 passed + 1 xfailed → now 473 passed + 1 xfailed; the xfail is the pre-existing `inject_needle` placeholder).

- [ ] **Step 6: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): --nested-cv CLI - out-of-fold certification + MRR from the dump"
```

---

## After implementation (user-run, offline on the login node)

```bash
python -m eval.gate_corroboration \
    --from-runs results/corroboration_runs_nq300cert.json \
    --k 10 --seed 0 --alpha-step 0.1 --nested-cv --folds 5 --out-dir results

python -m eval.significance --per-query-csv results/corroboration_nested_cv_per_query.csv --reference q2d
python -m eval.significance --per-query-csv results/corroboration_nested_cv_mrr.csv --reference q2d
```

The `--reference q2d` p-values on the two CV CSVs are the honest n=300 out-of-sample verdict (needle-found@10 primary, MRR secondary). Paste into `docs/results-summary.md` alongside the corrected finding 15.
