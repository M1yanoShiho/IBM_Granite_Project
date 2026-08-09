# System-level held-out (HotpotQA / RGB / MuSiQue-Full) — handover

For whoever runs the **system-level** evaluation of the integrated three-module
pipeline. The Generator's own module-level held-out is QAMPARI; these three sets
were prepared but **deliberately not consumed**, so they remain untouched as the
system test.

Everything here is working code plus three findings that will each cost a run if
hit blind. All were found by a dry run, not by reasoning.

## What is ready to use

| file | what it does |
|---|---|
| `scripts/heldout_data.py` | loaders for all three sets, plus RGB's counterfactual sub-test, normalised to one record shape |
| `scripts/heldout_dryrun.py` | schema + record counts + a tiny generation slice. Computes no metric, imports no scorer, prints no answer text |
| `scripts/run_heldout_dryrun.slurm` | the above as a GPU job |
| `scripts/heldout_sample.py` | draws an auditable sample and writes a manifest with a SHA-256 per set |
| `configs/heldout-sample.json` | a drawn sample: hotpotqa 400, musique-full 800 (400 ids × 2), rgb 300, rgb-counterfactual 100 |

Record shape, identical across sets so the runner needs no per-dataset branching:

```python
{"query_id": str, "question": str, "gold_answers": [(alias, ...)],
 "passages": [{"title": str, "text": str}, ...]}
```

Verified working (job `18318986`): **0 schema problems on all three**, and both a
baseline and a verify-annotate arm produced valid results on every sampled
record.

| set | records | passages/record (median) |
|---|---|---|
| hotpotqa | 7405 | 10 |
| musique-full | 4834 | 20 |
| rgb | 300 | 40 |

## Three findings that will cost you a run

### 1. MuSiQue-Full's unanswerable half inverts the correctness metric

Counted, not assumed: the dev split is **4834 records = 2417 ids each appearing
twice**, once answerable and once not — and **both variants carry a non-empty gold
answer string**. On the unanswerable variant the supporting paragraphs have been
removed, so the evidence does not support that answer.

**A string-match correctness metric therefore rewards a system for asserting the
answer anyway, and penalises abstention.** It will inflate any system that always
guesses and depress any system that declines when evidence is absent — in either
direction the number misleads.

Handled in `heldout_data.py` by suffixing the id (`{id}#ans` / `{id}#unans`) —
the raw id is **not unique** and collides in every id-keyed structure, paired
tests included — and by carrying an `answerable` flag through.

Recommended reporting, which the Generator's own pre-registration adopts:

- report the two subsets **separately**, and the union;
- evaluate any headline criterion on the **answerable** subset;
- on the unanswerable subset treat string-match as a **negative** indicator —
  higher means more ungrounded assertion, not more correctness — and report
  abstention rate alongside.

### 2. The obvious MuSiQue source serves the wrong half

`dgslibisey/MuSiQue` and most mirrors carry only `musique_ans_*` — the
**answerable half**. Loading it silently substitutes an easier task than
"MuSiQue-Full" names, and nothing errors.

`heldout_data.py` reads `musique_full_v1.0_dev.jsonl` from `bdsaglam/musique`
explicitly, chosen for carrying that file.

### 3. HotpotQA's canonical host hangs rather than refuses

`http://curtis.ml.cmu.edu/datasets/hotpot/...` **times out** from the cluster.
It does not refuse, so it presents as a hang — the hardest failure to diagnose,
and it burns walltime silently.

`heldout_data.py` uses the HF mirror
(`hotpotqa/hotpot_qa`, `distractor/validation-00000-of-00001.parquet`). That
mirror stores `context` **column-wise** (`{"title": [...], "sentences": [[...]]}`)
rather than as a list of pairs; the loader transposes it back.

## Two process notes worth inheriting

**No reported comparison may span two jobs.** Measured on this codebase: with a
fixed seed, identical code, identical models and the *same node*, two runs agreed
on 400/400 answers when they executed at the same speed and on 356/400 when one
ran 2.2× faster. Greedy decoding is deterministic in arithmetic, not bitwise —
GPU reduction order varies and near-ties flip. Run every arm of a comparison
**inside one job**. Details in `docs/generator/frozen-results.md` §5.

**Guard the error rate, not just the exit code.** A run that swallows exceptions
and exits 0 will produce a well-formed report full of numbers computed from
almost nothing; this happened here (365 errors per arm, exit 0, files written).
`scripts/g5_verify_annotate.py` now takes `--max-error-rate` and writes **no
files** when an arm exceeds it. Worth copying.

## What was not done

These sets were **never scored**. The dry run read schema and record counts only.
No metric was computed on any of them, so they are intact as a held-out test.
