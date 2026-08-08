# G7 — removing the entity gate

Jobs `18295681` (generation, 03:23:50) + `18295682` (scoring, 00:20:29), five arms
in one job, both exit 0. Judge **MiniCheck**; TRUE is the production verifier and
never judges. Pre-registered in `docs/hpc-run-log.md` §G7 before any arm ran.

Change: entailment alone decides citation, no drop path. The entity check still
runs on every claim in every arm with its verdict logged and ignored.

## Five arms, three axes

Primary figure is **ALCE-convention** citation precision, under which an example
that cited nothing scores 0. Cited-sample precision is alongside.

| arm | coverage | correctness (STR-EM) | cite prec (ALCE) | cite prec (cited) | cite recall | answered |
|---|---|---|---|---|---|---|
| baseline | 0.925 | 0.266 | 0.597 | 0.597 (370) | 0.635 | 370/400 |
| verify-only | 0.641 | 0.206 | 0.862 | 0.862 (253) | 0.862 | 253/395 |
| verify-annotate-capped | 0.635 | 0.202 | 0.890 | 0.890 (251) | 0.870 | 251/395 |
| verify-annotate-open (control) | 0.784 | 0.223 | 0.723 | 0.890 (251) | 0.707 | 309/394 |
| **verify-annotate-nogate** | **0.911** | **0.268** | 0.718 | 0.848 (304) | 0.703 | 359/394 |

**The control arm reproduced G6 exactly: 394/394 answers byte-identical**, and
baseline and verify-only reproduced their G6 numbers to the digit, including
their error counts. The routing refactor that computes both verdicts in one pass
changed nothing — that inference is sound and is what the comparison was for.

> ### Correction — what this does NOT establish
>
> The original wording continued "cross-run comparability … is verified rather
> than assumed here". **That inference was wrong and is withdrawn.**
>
> G8 re-ran the identical five arms and found the **baseline** arm — which
> contains no changes whatsoever — reproducing G7 on only **356/400 (0.890)**.
> Two candidate explanations were checked against the job logs:
>
> * *G7 reused cached generations.* **Ruled out.** G7 logged a full generation
>   pass at **6.06 s/arm/case** against G6's **6.03**, with per-arm answered and
>   error counts produced afresh; the runner has no caching path.
> * *Execution conditions happened to match.* **Supported.** G6 and G7 ran within
>   0.5% of each other per arm-case; G8 ran the same code on the same node
>   (`bp1-gpu035`) at **2.70 s/arm/case**, 2.2× faster, and diverged on 11% of
>   queries. Decoding is greedy (`temperature=0.0`), so the arithmetic is
>   deterministic but the GPU reduction order is not, and near-ties flip.
>
> The honest claim is therefore: **cross-run identity was observed between two
> runs that happened to execute alike. It is conditional on execution conditions,
> not guaranteed, and must never be relied on.** The rule that follows is applied
> throughout this repository: **no reported comparison may span two jobs.**
>
> Concretely, `verify-annotate-capped` — an arm untouched between the two rounds
> — reads citation precision 0.890 here and 0.911 in G8. That 0.021 gap is pure
> cross-job drift and is not a result.

Every comparison reported below is between arms **within this single job**, which
is why the all-arms-in-one-job discipline exists.

## The pre-registered failure criterion is breached

> Gate removal fails if citation precision on cited sentences falls below
> verify-only's 0.862.

**nogate cited-sample precision is 0.8476. That is below 0.862, by 1.4 points.**
On the literal criterion, gate removal fails. That is stated first because it was
pre-registered, and the decomposition below does not repeal it.

### What the 1.4 points are made of

| population | n | cited-sample precision |
|---|---|---|
| nogate, all cited examples | 304 | 0.8476 |
| nogate, on queries verify-only also answers | 253 | **0.8841** |
| verify-only, those same queries | 253 | 0.8617 |
| nogate, on the 51 queries only it answers | 51 | **0.6667** |

On shared work nogate is **better** than verify-only (+0.022 paired, p = 0.238).
The arm-level shortfall is composition: nogate answers 51 queries verify-only
refuses, and those are harder.

**The criterion I wrote compares one arm's mean to another arm's mean when the two
arms answer different query sets.** That is a defect in the criterion's
specification, not a finding, and I am naming it as mine rather than using it to
wave the breach away. The paired form of the same comparison — the form the rest
of this project uses — does not breach.

### But the gate was not firing at random

| population | n | cited-sample precision |
|---|---|---|
| nogate examples containing a claim the gate would have destroyed | 58 | **0.6782** |
| all other nogate examples | 246 | **0.8875** |

A 21-point gap. The claims the gate wanted to destroy really do end up in answers
whose citations MiniCheck accepts markedly less often. The human audit's 0.850
wrong-destruction rate and this 0.678 automatic precision are both true and are
not in conflict: the audit asked "does this passage support this claim", the
judge asks "does the cited passage entail the sentence as written". The gate was
a poor instrument for the first question and a better-than-chance one for the
second.

**The content is still worth keeping.** 0.667 precision on the marginal cohort is
well above the baseline's 0.597 average. The system's *worst* new content is cited
more precisely than the baseline's typical content.

## What the gate was costing — nogate vs open

Paired randomization, 10 000 iterations, bootstrap CI.

| axis | delta | p | 95% CI | n |
|---|---|---|---|---|
| coverage | **+0.127** | 0.0 | [0.096, 0.160] | 394 |
| correctness | **+0.044** | 0.0 | [0.030, 0.061] | 394 |
| citation precision | −0.005 | 0.698 | [−0.025, 0.014] | 251 |
| citation recall | +0.004 | 0.722 | [−0.013, 0.021] | 309 |

**The gate cost 12.7 points of coverage and 4.4 points of correctness and bought
nothing measurable on either citation axis.** nogate answers 50 queries open
does not, and open answers none that nogate does not — the change is strictly
additive. The predicted precision cost was "about one point"; the paired result
is 0.5 points and not distinguishable from zero.

## Against the baselines

| comparison | coverage | correctness | cite precision | cite recall |
|---|---|---|---|---|
| **nogate vs baseline** | −0.018 (**p=0.297**) | +0.001 (**p=0.888**) | **+0.191 (p=0.0)** | **+0.071 (p=0.009)** |
| nogate vs verify-only | +0.269 (p=0.0) | +0.061 (p=0.0) | +0.022 (p=0.238) | +0.001 (p=0.970) |
| open vs baseline | −0.145 (p=0.0) | −0.043 (p=0.0) | +0.199 (p=0.0) | +0.063 (p=0.026) |

**This is the result.** Against the generation-time-citation baseline, the method
is now **statistically indistinguishable on coverage (p=0.30) and on correctness
(p=0.89)**, while carrying **+0.19 citation precision and +0.07 citation recall**.
Every previous round had to concede a large coverage or correctness deficit as the
price of citation quality. That price is now gone.

Against the published delete method it dominates on every axis.

## Observe-only: what the gate would have destroyed

The gate ran on all 434 claims in this arm with its verdict recorded and ignored.

| | |
|---|---|
| claims the gate would have destroyed | **60** |
| ... of which received a citation instead | **60** (100%) |
| ... of which fell through to annotation | 0 |
| claims cited either way, but the gate would have picked other evidence | 14 |
| claims routed / verified / annotated / dropped | 434 / 359 / 75 / **0** |

**Self-check passes exactly**: on the control arm, where the gate *is* routing,
`gate_would_drop` = 60 = `dropped_entity_conflict`. The observe-only log records
what the gate really does, so the 60 is measured, not extrapolated. The G6
estimate was "~51 wrongly destroyed of ~60"; the population is confirmed at 60 and
every one of them is now cited.

Two claims that the gate had left *annotated* are now verified — entailing
evidence whose entity mismatch was an absence rather than a conflict.
Declared-citation survival rises from 223/399 (0.559) to **283/399 (0.709)** for
the same reason.

## Sentence composition

| arm | verified & cited | annotated unverified | uncited, unlabelled | total |
|---|---|---|---|---|
| baseline | 427 | 0 | 0 | 427 |
| verify-only | 275 | 0 | 0 | 275 |
| verify-annotate-capped | 313 | 14 | 1 | 328 |
| verify-annotate-open | 313 | 75 | 3 | 391 |
| **verify-annotate-nogate** | **379** | 73 | 3 | 455 |

+66 cited sentences over the control, no drop path, and annotation reach holds
(74 annotated claims reach an answer, against 76 in the control and 15 under the
cap).

## The entity layer is threat-model dependent, not failed

Both numbers belong in the report:

- **G1 adversarial entity-substitution slice: 0.963 verifier-alone → 1.000 with
  the layer engaged.** It does what it was built for.
- **Benign data: 0.850 wrong destruction** (human, two blind rounds), and the
  cohort it targets carries 0.678 citation precision against 0.888 elsewhere
  (automatic, this run).

"A defence against adversarial entity substitution that is expensive on benign
data" is a trade-off result. The gate is retained in the code behind
`entity_gate=True` and runs observe-only by default, so the setting is where the
threat model is declared rather than something that has to be rebuilt.

## A contract defect this run exposed

One query in each of open and nogate was lost to a `GenerationResult`
`ValidationError`: *uncited answer must mark every sentence with '[unverified]'*.

`count_sentences` splits on `[.!?]`, so a claim containing an abbreviation-style
internal period — `won on Jan. 11, 1970`, `works at Acme Inc. in Ohio` — counts as
two sentences carrying one marker, and the validator rejects a fully-annotated
answer. Verified directly against the function, not inferred from the message.

It is a false rejection that destroys a whole answer, so it is a real defect. It
is **not fixed in this round**: it costs 1 query of 400, and it hits open and
nogate identically, so it cannot bias nogate-vs-open. Fixing it requires a re-run
to keep code and reported numbers aligned, which is not worth 1/400. First item
for the next round.
