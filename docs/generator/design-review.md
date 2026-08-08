# Generator — design and code review

Written after G8, with calibration frozen. Nothing here is applied: the held-out
run must execute the same code the calibration numbers came from. Each item is
marked with when it could be acted on.

Findings are ordered by how much they would matter to someone reading or
inheriting this module, not by how easy they are to fix.

---

## 1. The main method is not reachable from the composition API

`composition.build_generator` accepts exactly one name:

```python
def build_generator(config: ModuleConfig) -> Generator:
    if config.name != "extractive":
        raise ValueError(f"unknown generator: {config.name}")
    return ExtractiveGenerator()
```

`VerifyAnnotateGenerator` — the thing this module is *about* — has **zero
references in `src/`**. It exists only in `scripts/g5_verify_annotate.py`. The
named builders (`build_granite_baseline`, `build_q2d_corroboration_granite`, …)
all wire `GraniteGenerator`, i.e. the baseline.

So the config-driven CLI documented in `CLAUDE.md` cannot run the method, and
`ExtractiveGenerator` — a 29-line passthrough placeholder — is the only generator
the pipeline factory can build.

This is the single largest structural gap. Everything measured is real, but it is
measured through a script-only path, and a reader following the documented
architecture would never reach it.

**Act:** after held-out. Register `verify-annotate` in `build_generator`, and add
a named builder. Low risk, no behaviour change to the measured path.

## 2. Three sentence splitters, three different rules

| implementation | abbreviations known | used by |
|---|---|---|
| `contracts.split_sentences` | ~50 (added in G8) | the contract validator |
| `claim_splitter._sentence_spans` | **10** | claim anchoring |
| `g3_sentence_rescore.sent_split` + `g5_score.annotated_sent_split` | ALCE's, plus a repair pass | scoring |

G8 unified the first and third because a disagreement between them destroyed an
answer. **The second was not touched and still disagrees.** `_sentence_spans`
does not know `inc`, `jan`, `u.s.`, `ltd`, or month names, so it will cut
`"Acme Inc. in Ohio"` into two sentences while the contract treats it as one.

The consequence here is milder than the G7 bug — a claim anchors to half a
sentence rather than an answer being rejected — but it is the same defect class,
still live, in the module that decides what a claim *is*.

**Act:** after held-out. One shared splitter, imported by all three. The rule
already exists in `contracts`.

## 3. A retired subsystem is still on the tree, unlabelled

`verifier.py`, `completeness.py`, `attribution.py`, `evidence_recheck.py`,
`repair.py` — 435 lines implementing the B1–B5 roles — are reachable **only**
through `verified.py`, which is used **only** as the `verify-only` ablation arm in
`scripts/`. The completeness loop they implement was retired as a runtime
mechanism rounds ago.

Keeping them is correct: the ablation is what makes the delete-vs-annotate
comparison meaningful, and deleting them would make the published comparison
unreproducible. But nothing in the package says so, so a reader cannot tell live
code from ablation scaffolding.

**Act:** now (documentation only, no code change). One line at the top of each:
*"Ablation-only: retired from the live path at G4; kept so `verify-only` remains
runnable."*

## 4. `CLAUDE.md` describes a design that no longer runs

The "Generator internal interface" section documents a generate → verify → patch
loop with roles A1/A2/B1–B5 handing off through `DraftAnswer` /
`VerificationReport`. On the live path:

- `VerificationReport`, `ClaimVerification`, `RequiredFactCoverage` are **not
  constructed at all** — `VerifyAnnotateGenerator` goes draft → route → assemble.
- `Verifier.verify` is never called.
- The completeness checker, described as "one LLM call per required fact", does
  not run; `required_facts` is empty by construction (`_empty_checklist`).

The description is an accurate account of the *previous* design. It is now the
most misleading document in the repository, precisely because it is the one a new
contributor is told to read first.

**Act:** now. Rewrite that section to describe routing, and keep the old text
under a "superseded" heading with the round that retired it — the history is
worth keeping, the false present tense is not.

---

## 5. Measured defect: ~20% of verifier calls are duplicates

`CitationRoutedVerifier.route` scans the declared-citation prefix, then the full
evidence list — and the declared items are members of that list, so they are
examined twice.

Measured on G8's 439 routed claims:

| | |
|---|---|
| declared indices per claim | mean 1.23 (0: 22, 1: 313, 2: 90, 3: 11, 4: 3) |
| redundant re-examinations | 538 |
| upper bound on `_supports` calls | 2733 |
| **duplicate share** | **19.7%** |

The comment I left in that loop says the duplication is acceptable because "the
memo-free NLI call makes [it] cheap enough". **That was wrong.** TRUE is a ~45GB
fp32 cross-encoder and these calls are the dominant cost of the whole job; the
early break on the first clean match reduces the realised figure below 19.7% but
does not change that the work is pure waste when it happens.

**Act:** after held-out. A dict keyed on `(evidence_id, claim_id)` inside `route`
removes it entirely, with no behavioural change — the same pair returns the same
verdict by construction. Worth roughly 15–20% of generation wall time.

## 6. The entity layer costs what it costs, in every configuration

With `entity_gate=False` the layer changes no routing decision, but
`_supports` still calls `entity_checker.check` on **every entailed pair**, because
the observe-only log needs it. That is right for research and wrong for
deployment: a production instance pays for spaCy NER on every entailed
claim/evidence pair to populate a counter nobody reads.

**Act:** after held-out. Three states rather than two — `gate` / `observe` /
`off` — defaulting to `observe` for experiments and `off` for the composition
builder. The review flag needs `observe`, so `off` is only for a deployment that
also drops the flag.

## 7. Inconsistent cursor discipline in claim anchoring

`_locate_span` has two paths. The exact path searches `answer_text.find(source_text, cursor)`
— forward-only. The fallback path searches **all** sentences and deliberately
ignores the cursor, which its docstring explains and justifies. But both then set
`cursor = span.end`.

So a fallback anchor can advance the cursor past a region that a *later* claim's
verbatim `source_text` occupies, after which the exact path fails on that claim
and it silently drops to fallback. The outcome is usually the same span, so this
has not produced a visible defect — but the two paths hold different beliefs
about what `cursor` means, which is how the next defect gets in.

**Act:** after held-out. Either make both forward-only or make both global; the
fallback's reasoning suggests global.

## 8. Naming that no longer tells the truth

- `contracts.strip_unverified_annotation` is now an alias for `strip_annotations`
  and removes the review label too. The name says otherwise. Kept during G8 to
  avoid touching call sites mid-freeze.
- `entity_check` emits an entity type called `name` alongside `person` /
  `organization` / `location` / `product`, meaning "proper noun, type unknown".
  Readable only from the docstring.
- Thirteen exported symbols have no external reference at all —
  `annotation_rate`, `format_amount`, `extract_dates_and_numbers`,
  `normalize_name`, four `Protocol`s in `verified.py`, and others. Some are
  deliberate seams; none are marked as such.

**Act:** after held-out, mechanical.

---

## 9. Risks specific to held-out, which no calibration round could surface

- **`max_new_tokens = 256`.** ASQA answers ran ~15 words, nowhere near the cap.
  RGB's evidence is twice as long (990 words in top-5 against ASQA's 500) and its
  questions can require integrating several documents. A truncated answer is
  indistinguishable in the output from a short one — it will not raise an error,
  it will just score badly. **Worth logging the token count against the cap** on
  the held-out run, which is diagnostics rather than a system change.
- **NLI truncation is not a risk**, contrary to what the passage lengths suggest:
  the premise is a single evidence item (~207 words on RGB, ~280 tokens) against a
  2048-token window. Checked rather than assumed.
- **The claim splitter is the only single point of total failure.** Its two JSON
  failures per 400 kill the query in *every* verify arm, because it sits upstream
  of all of them. On a new distribution that rate is unknown, and the
  `--max-error-rate 0.10` guard is the only thing standing between a bad rate and
  a plausible-looking report.

---

## What held up

A fair review should say what did not need fixing.

- **The contract boundary did its job.** Every cross-module change went through
  `contracts`, and `tests/architecture/test_boundaries.py` caught drift.
  `GenerationResult`'s validator rejected an answer the generator should not have
  produced — the G7 loss was a *correct* rejection of a miscounted answer, and it
  surfaced a real defect rather than hiding one.
- **Injectable models throughout.** `TextGenerator`, `NLIModel`, `EntityExtractor`
  and `EntityChecker` are Protocols with deterministic fakes in tests, so 1214
  tests run with no weights and no GPU. This is why calibration could iterate at
  all.
- **The routing redesign is genuinely simple.** `route()` returns one dataclass,
  computes both the live and the counterfactual verdict in one pass, and the
  observe-only self-check (`gate_would_drop` == `dropped_entity_conflict` on the
  control arm) is verifiable from the output. That design is what turned "~51
  claims wrongly destroyed" into a measured 66.
- **Scoring conventions are stated and applied identically to every arm**, with
  the baseline deliberately getting the *more generous* flat convention. The
  reported gains are therefore conservative.

---

## Priority

| # | item | when | effort |
|---|---|---|---|
| 4 | `CLAUDE.md` describes a retired design | **now** | small |
| 3 | label the ablation-only subsystem | **now** | small |
| 1 | wire `VerifyAnnotateGenerator` into composition | after held-out | small |
| 5 | memoise `_supports` (~20% of verifier calls) | after held-out | small |
| 2 | one sentence splitter, not three | after held-out | medium |
| 6 | three-state entity gate | after held-out | small |
| 9 | log answer-token headroom on held-out | with held-out | small |
| 7, 8 | cursor discipline, naming, dead exports | later | mechanical |

Items 4 and 3 are documentation only and touch no code the held-out run
executes, so they are safe to do under the freeze. Everything else waits.
