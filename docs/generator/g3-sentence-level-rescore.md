# G3 remediation Task 1 — sentence-level re-scoring

Re-scoring of the **existing** G3 run with ALCE's sentence-level citation
metrics. No chain was re-run; all arms are reconstructed from the G3 and
diagnosis dumps. Judge: **MiniCheck** (TRUE excluded — it selected the verified
arms' citations, so scoring with it would be circular). Job 18240434, seed 13.

**Hold-out respected:** ALCE/ASQA only. HotpotQA, RGB, MuSiQue-Full never loaded.

## Headline

**G3's negative result on `verified-full` was substantially a metric artifact.**
Against the baseline it moves from a significant loss to a null:

| comparison (paired, both-answered) | answer-level (G3) | sentence-level (ALCE) |
|---|---|---|
| verified-full vs baseline | **−0.107, p=0.034** (a loss) | **+0.011, p=0.80** (a null) |
| verified-full vs verify-only | **−0.228, p<0.001** | **−0.104, p<0.001** (still a loss) |

So roughly **half** of the completeness loop's apparent damage was measurement;
the other half is a real defect — the recheck fragments — which Task 4 addresses.
`verify-only` continues to beat the baseline, and by *more* under the matched
citation convention.

## The ALCE definitions as implemented (report item 2)

Ported from `compute_autoais` in princeton-nlp/ALCE `eval.py`, verified against
the source rather than reconstructed from memory. Per example:

```
entail = 0; entail_prec = 0; total_citations = 0
for each sentence:
    ref = citations of this sentence
    if len(ref) == 0:               joint_entail = 0
    elif any ref out of range:      joint_entail = 0
    else:
        total_citations += len(ref)
        joint_entail = NLI(concat(cited docs), sentence)
    entail += joint_entail
    if joint_entail and len(ref) > 1:
        for each cited doc d:
            if   NLI(d, sentence):                     entail_prec += 1
            elif NLI(concat(ref minus d), sentence):   pass          # overcite, +0
            else:                                      entail_prec += 1
    else:
        entail_prec += joint_entail
ais_scores.append(entail / len(sents))
ais_scores_prec.append(entail_prec / total_citations if total_citations else 0)

citation_rec  = 100 * mean(ais_scores)
citation_prec = 100 * mean(ais_scores_prec)
```

Load-bearing details, each pinned by a test in `tests/generator/test_alce_metrics.py`:

- **Recall's denominator is every sentence**, including uncited ones, which score
  0. Padding an answer with unsupported prose costs recall — the property that
  makes this metric length-fair instead of silently diluting every citation.
- **Precision's denominator is the example's total citations**, and an example
  that produced no citations at all scores **0**, not skipped.
- **The redundancy ablation runs only when a sentence has more than one
  citation.** A lone citation is precise iff it entails its sentence.
- **Both metrics are macro-averaged over examples**, not pooled.

Two deliberate deviations, both forced by our setting:

- The entailment function is **MiniCheck**, not ALCE's TRUE/AutoAIS. TRUE is the
  production verifier and selected the verified arms' citations.
- Documents are raw chunk text; ALCE's `_format_document` prepends a title and
  our `EvidenceCandidate` has no title field.

Sentence splitter: **`nltk.sent_tokenize`**, as ALCE uses, applied identically to
every arm.

## Reconstruction checks

The arms are rebuilt from dumps, so the reconstruction is validated before any
number is read off it:

| check | result |
|---|---|
| baseline examples (G3 reported 373 answered) | **373** |
| verify-only examples (G3 reported 218) | **218** |
| verified-full examples (G3 reported 125) | **125** |
| verified-full answers rebuilt byte-identical to the recorded `final_answer` | **125/125** |
| sentences left with no citation attached | **0** (both verified arms) |

`verify-only`'s 218 requires including the 13 cases that errored *later*, in
completeness/recheck — a stage `verify-only` never reaches. Sentences of the
repaired answer attribute to the best-overlapping trusted claim, because claims
are rewritten away from their source span and a fixed threshold stranded 22% of
them.

## Answer-level vs sentence-level (report item 1)

| arm | n | answer-level prec | sentence-level prec | Δ | answer-level rec | sentence-level rec | Δ |
|---|---|---|---|---|---|---|---|
| baseline | 373 | 0.602 | 0.613 | +0.011 | 0.646 | 0.647 | +0.001 |
| verify-only | 218 | 0.762 | 0.779 | +0.017 | 0.862 | 0.853 | −0.009 |
| **verified-full** | 125 | 0.575 | **0.717** | **+0.142** | 0.736 | 0.736 | +0.000 |
| baseline-persentence | 389 | — | 0.584 | — | — | 0.632 | — |

**The artifact is specific to the arm it should be specific to.** Baseline and
verify-only barely move (+0.011, +0.017); `verified-full` gains **+0.142**. That
is exactly the arm whose answers were lengthened by appended recheck fragments,
and the near-zero movement elsewhere is the control showing sentence-level
scoring is not simply inflating everything.

## Paired comparisons at sentence level

Paired randomization, 10 000 iterations, bootstrap CI, both-answered subsets.

| comparison | metric | delta | p | 95% CI | n |
|---|---|---|---|---|---|
| verify-only vs baseline | precision | **+0.087** | **0.009** | [+0.021, +0.154] | 215 |
| verify-only vs baseline | recall | **+0.116** | **0.0006** | [+0.051, +0.181] | 215 |
| verified-full vs baseline | precision | +0.011 | 0.80 | [−0.074, +0.097] | 123 |
| verified-full vs baseline | recall | −0.016 | 0.73 | [−0.103, +0.071] | 123 |
| verified-full vs verify-only | precision | **−0.104** | **<0.0001** | [−0.153, −0.059] | 94 |
| verified-full vs verify-only | recall | **−0.156** | **<0.0001** | [−0.213, −0.100] | 94 |
| **verify-only vs baseline-persentence** | precision | **+0.133** | **<0.0001** | [+0.067, +0.198] | 215 |
| **verify-only vs baseline-persentence** | recall | **+0.141** | **0.0001** | [+0.074, +0.207] | 215 |
| verified-full vs baseline-persentence | precision | +0.089 | 0.069 | [−0.005, +0.181] | 122 |
| verified-full vs baseline-persentence | recall | +0.032 | 0.52 | [−0.062, +0.129] | 122 |

## Both baseline citation conventions (report item 3)

| convention | n answered | sentence-level prec | sentence-level rec |
|---|---|---|---|
| (a) generous fallback — every citation applies to every sentence | 373 | 0.613 | 0.647 |
| (b) matched protocol — model cites per sentence (Task 2) | 389 | 0.584 | 0.632 |

The generous fallback is, as intended, **lenient toward the arm we are trying to
beat**: handing baseline every citation on every sentence scores it 0.613, while
asking the model to actually cite per sentence scores 0.584. `verify-only`'s
advantage is therefore *larger* under the matched protocol (+0.133) than under
the generous one (+0.087) — so the conservative number to quote is the generous
one, and it still wins.

The per-sentence prompt also slightly raised baseline coverage (389 vs 373
answered), so it is not a crippled variant.

## Diagnostics

| arm | sentences | citations | multi-cite sentences | jointly supported | overcite |
|---|---|---|---|---|---|
| baseline | 384 | 490 | 75 | 57 | 45 |
| verify-only | 240 | 431 | 124 | 114 | 49 |
| verified-full | 255 | 355 | 66 | 63 | 28 |
| baseline-persentence | 461 | 641 | 156 | 125 | 47 |

## Reading

1. **The G3 headline needs restating.** `verified-full` does not lose to the
   baseline on citation precision; the −0.107 was a length artifact. It is a null
   (+0.011, p=0.80). The pre-registered criterion asked whether verified precision
   *exceeds* baseline — it does not, so the full system's claim stays unsupported,
   but "unsupported" now means indistinguishable rather than worse.
2. **`verify-only` is the real result and it survives the fairer metric**, with a
   larger margin under the matched convention (+0.133 precision, +0.141 recall,
   both p<0.001). This is the configuration the method should be reported on.
3. **The completeness loop is still genuinely harmful, and now for a legible
   reason.** Half its apparent damage was measurement; the residual −0.104 is
   real, and the mechanism is already identified — recheck emits bare entity
   fragments ("1992", "Bhola Paswan Shastri"), median 17 characters, only 75 of
   255 ending in punctuation. Under sentence-level scoring these become their own
   sentences and fail on their own merits. Task 4's partial answering is what
   addresses this.
4. **No citation number computed at answer level in this project should be
   quoted again.** They are length-confounded; recompute with `alce_metrics.py`.
