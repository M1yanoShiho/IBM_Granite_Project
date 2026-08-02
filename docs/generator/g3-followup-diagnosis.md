# G3 follow-up — diagnosing the completeness loop

G3 localised the harm to the completeness/recheck loop. Before closing the loop,
this pass tests the cheap alternative explanation: G2 measured B4 own-fact
coverage 0.786, implying a **~21% false-gap rate**, which alone would produce both
observed harms. Same shape as the `entity_check` false-veto, which turned out to
be one over-aggressive rule.

Instrumented re-run of the `verified-full` arm on the identical G3 case set (job
**18240137**, seed 13, 400 cases, 22 chain errors). Judge: **MiniCheck**
throughout, never TRUE. `src/evidence_rag/` untouched — all instrumentation is
script-side wrappers (`scripts/g3_diagnosis.py`).

**Faithful reproduction, not a new experiment.** Greedy decoding
(`temperature=0.0` → `do_sample=False`) is deterministic, and the run reproduces
G3 exactly: 400 − 22 errors = 378 completed, 253 abstentions, **125 answered** —
identical to G3's 125/378. The findings below therefore apply directly to the
published G3 numbers.

**Hold-out respected:** ALCE/ASQA only. HotpotQA, RGB, MuSiQue-Full never loaded.

## Headline: the false-gap hypothesis is refuted, and the two harms have different causes

| harm | cause | decision-table row |
|---|---|---|
| coverage collapse (0.331 vs 0.552) | genuine gaps + recheck succeeds only 38% + all-or-nothing abstention | row 2 — redesign, but the *abstention policy*, not the gap detector |
| citation precision drop (−0.228) | answer-length dilution in the answer-level metric; recheck citations not significantly worse | row 3 — measurement limitation, pending human confirmation |
| false gaps (the hypothesis under test) | **6.6%, not 21%** | row 1 **refuted** |

## Task 1 — false-gap rate on real data

- gaps raised by the completeness checker: **654** (mean 1.64 per query; 359 of
  378 completed queries had at least one)
- of those, MiniCheck says the draft **already stated** the fact (false gap):
  **43/654 = 0.066**

G2's 0.786 own-fact coverage implied ~0.214. On real data the rate is **0.066**,
and even correcting generously for MiniCheck's own 0.620 recall (43 / 0.620) the
upper estimate is **0.106** — still half of the hypothesised figure.

**The firing rule is not badly miscalibrated.** G2's 0.786 came from synthetic
gold statements; it did not transfer. The completeness checker is mostly right
when it says a fact is missing.

**Limitation, stated plainly:** using a model to judge whether a gap was false is
a proxy, not ground truth. MiniCheck misses ~38% of genuine support, so this is
*provisional* — Task 4's human packet is the ground truth for it. The
recall-corrected 0.106 is the honest upper bound.

## Task 2 — citation support split by origin

| citation origin | MiniCheck-supported |
|---|---|
| draft claims that survived verification | 97/180 = **0.539** |
| fragments added by recheck | 27/62 = **0.435** |

Delta 0.103, **permutation p = 0.184 (n=20000)** — *not significant*. Per the
guide's own test ("if the recheck population is not markedly worse, the loop is
not the precision culprit and the diagnosis moves elsewhere"), it moves elsewhere.

### Where it moves: answer-length dilution

The answer-level metric scores each cited chunk against the **whole answer**. The
same draft-origin citations score very differently depending on what they are
scored against:

| the same draft-origin citations, judged against | MiniCheck-supported |
|---|---|
| the individual **claim** they were selected for | 192/221 = **0.869** |
| the **whole final answer** (draft + recheck fragments) | 97/180 = **0.539** |

Recheck appends fragments to 105 of 125 answered cases, roughly doubling answer
length (median 60 → 97 chars, mean 73 → 137). Every citation is then required to
entail a longer, multi-topic answer — including good draft citations that never
claimed to support the appended material. Precision collapses for *all*
citations, not just recheck's.

The arms differ systematically on exactly the dimensions this metric is sensitive
to:

| arm | answer length (median) | citations per answered query | pooled answer-level support |
|---|---|---|---|
| baseline | 54 chars | 1.24 | 0.571 |
| verified-full (final) | 97 chars | 1.94 | 0.512 |

So G3's citation-precision comparison is **confounded by answer length and
citation count**. The metric was applied identically to both arms, but it is not
neutral: an arm that says more, and cites more, is penalised per citation even
when each citation is individually apt. (ALCE uses per-sentence attribution
precisely to avoid this; G3 used answer-level because the arms attach citations at
answer granularity.) **G3's −0.228 and −0.107 should be read as partly a
measurement artifact until the human numbers settle it.**

## Task 3 — abstentions attributable to false gaps

- abstentions (non-error): **253** of 400 cases
- abstained via the completeness path (missing required fact): **251/253 = 0.992**
- of those, **every** triggering gap was false: **9/251 = 0.036**
- of those, **at least one** triggering gap was false: **21/251 = 0.084**

So ~96% of the abstention flood is triggered by **genuine** gaps, not false ones.
The mechanism is the recheck success rate:

- recheck finds the fact in the selected evidence on genuine gaps: **234/611 = 0.383**
- (on false gaps, where the fact is already in the draft: 21/43 = 0.488)

With a mean of 1.80 required facts per query, gaps raised on 359 of 378 queries,
recheck resolving only ~38% of them, and an **all-or-nothing** policy (any
unresolved required fact ⇒ empty answer), a 63% abstention rate is the arithmetic
consequence. The component behaves as designed; the *policy* is what collapses
coverage.

## Task 5 — verify-only coverage/precision curve (TRUE threshold sweep)

Pure re-aggregation of persisted per-pair TRUE probabilities; every (evidence,
claim) pair carries a MiniCheck verdict, so no threshold needed a re-judge.

| TRUE threshold | coverage | citation precision (claim-level) | cited pairs |
|---|---|---|---|
| 0.05 | 0.579 | 0.815 | 438 |
| 0.10 | 0.569 | 0.831 | 420 |
| 0.20 | 0.556 | 0.846 | 402 |
| 0.30 | 0.550 | 0.861 | 388 |
| 0.40 | 0.545 | 0.868 | 378 |
| **0.50** (production) | **0.542** | **0.873** | 371 |
| 0.60 | 0.529 | 0.880 | 359 |
| 0.70 | 0.516 | 0.891 | 350 |
| 0.80 | 0.500 | 0.905 | 337 |
| 0.90 | 0.474 | 0.921 | 315 |

The trade-off is smooth and monotone with no cliff, and the production 0.50 point
is unremarkable within it — it is not a favourable point selected after the fact.
Across the whole usable range coverage moves only 0.579 → 0.474 while precision
moves 0.815 → 0.921.

**Caveat that blocks the guide's intended plot:** this curve is **claim-level**,
because that is what the persisted pairs support. The baseline cannot be placed on
these axes — baseline answers are never split into claims, so a claim-level
baseline precision is undefined. Putting both on one plot requires re-judging each
threshold's repaired answer at answer level (one MiniCheck pass per threshold);
that is a short job, worth bundling with any future GPU work rather than queued
alone.

## What to do (not pre-committed; this is what the numbers say)

1. **Keep the gap detector.** It is right ~93% of the time; retuning it addresses
   6.6% of gaps and would not have moved G3.
2. **Replace the all-or-nothing abstention policy.** Abstaining because *any* one
   required fact is unresolved, when recheck resolves only 38%, is what produced
   the coverage collapse. Partial answers with the resolved facts, or abstention
   only when the *primary* fact is missing, is the change indicated by the data.
3. **Do not conclude that recheck citations are bad** on the current evidence —
   the difference is not significant (p=0.18) and the apparent precision damage is
   substantially answer-length dilution. Task 4's human adjudication decides this.
4. **Re-report G3's citation numbers with the granularity caveat.** The
   answer-level metric penalises longer, more-cited answers; claim-level scoring
   of the same citations gives 0.869.
