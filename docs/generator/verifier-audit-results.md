# Verifier audit — manual adjudication of dropped claims (Generator Part B)

Human adjudication of the claims the B5 chain dropped, to decide whether the
verifier is *working* (the draft really was unsupported) or *broken* (the
verifier is deleting true statements). Built with `scripts/audit_export.py`,
scored with `scripts/audit_score.py`; the adjudicator is a person, independent of
the model stack — a model-adjudicated result would defeat the purpose.

**Source run:** G2 job 18213561 (`ibm-granite/granite-4.1-3b` generator + TRUE
verifier, ASQA calibration, seed 13). Per-claim audit dump
`local/raw-results/verified-generator-audit-claims-18213561.jsonl` (77 faithful
claims). Packet/key `local/audit/audit-packet.md` / `audit-key.json`.

**Hold-out respected:** ALCE/ASQA only. HotpotQA, RGB, MuSiQue-Full never loaded.

## Method

Blind, stratified, with controls (audit-export-spec):

- **Blind** — the packet shows only the claim, the question, and *all* selected
  evidence chunks in a shuffled (non-score) order. No verdict, stratum, or
  supporting-evidence marker is visible, so the adjudicator is not anchored.
- **Stratified** — dropped claims fail for two different reasons with different
  fixes, so they are sampled and scored separately:
  - **Stratum A** — NLI returned no entailment.
  - **Stratum B** — NLI entailed but `entity_check` vetoed it.
- **Controls** — claims the system marked *supported*, shuffled in
  indistinguishably, to measure in-chain citation precision directly rather than
  extrapolating G1's 0.966 calibration figure.

40-item packet, seed 13: 15 A + 15 B + 10 supported controls.

## Population and the claim-vs-pair resolution

| stratum | claims |
|---|---|
| A — NLI no entailment | 27 |
| B — entailed, entity-vetoed | 24 |
| supported (controls available) | 26 |

The run-level "entity mismatches caught = 54" counts **(claim, evidence) pairs**,
not claims: those 54 pairs fall on **24 claims** (stratum B). The dump's
`entity_veto_pairs` per claim is what resolves this — read 54 as pairs, 24 as
claims.

## Scores (per stratum, never pooled)

| stratum | n | agreement (human vs system) | **false-veto rate** | unclear |
|---|---|---|---|---|
| **A — NLI no entailment** | 15 | 11/12 = **0.917** | 1/15 = **0.067** | 3/15 = 0.200 |
| **B — entity-vetoed** | 15 | 2/11 = **0.182** | **9/15 = 0.600** | 4/15 = 0.267 |
| controls (system = supported) | 10 | in-chain citation precision **9/10 = 0.900** | — | 0/10 |

`false-veto` = a claim the system dropped that the human marks **supported** (the
"method is broken" signal). Agreement is over the non-unclear adjudicated items.

## Reading

### The NLI layer is sound; the entity-check layer is the broken one

The two dropped strata have opposite quality, which is exactly why they must not
be pooled:

- **Stratum A (TRUE/NLI found nothing entailing) is trustworthy** — 92% human
  agreement, one false veto in fifteen. When NLI says "the evidence does not
  entail this," it is almost always right (e.g. item-10 "Sir Garfield Sobers …"
  against evidence that is about Gooch; item-16 "Samsung … in 2007" with no 2007
  anywhere).
- **Stratum B (entity_check vetoed an entailment) is mostly wrong — 60% false
  veto** (9/15 overall, 9/11 of the non-unclear cases). The entity layer is
  rejecting claims the evidence plainly supports.
- **In-chain citation precision is 0.90, measured** — when the system says
  supported, the human agrees 9 times in 10. Slightly below G1's extrapolated
  0.966, and now a real in-chain number rather than a calibration one.

### Why entity_check false-vetoes — reproduced, not guessed

Re-running `EntityConsistencyChecker` on the false-veto claims against their own
evidence shows the mechanism is **brittle proper-noun matching**, not the
"missing date/number" rule:

- **item-18** "The battle of King's Mountain resulted in a victory for the
  Patriots." — **every one of the five chunks** flags `name 'king mountain'
  CONFLICT` and `name 'patriots' CONFLICT`, although every chunk literally
  contains "Kings Mountain" and "Patriots". The claim's `King's Mountain`
  normalises (possessive stripped) to `king mountain`, the evidence's `Kings
  Mountain` to `kings mountain`; `Patriots` vs `Patriot militia`. None matches
  exactly, and because the chunk carries a *related same-role name*, the mismatch
  is scored as a **conflict** rather than a match.
- **item-17** "Tour de France … 1903 … L'Auto" — the verbatim-supporting chunk
  still flags `date '1903' CONFLICT` and `name "l'auto" CONFLICT`.
- **item-21** "… established in August 1978" — chunks that legitimately do not
  restate 1978 (they cover the commission's later 1992–2003 history) mark the
  claim's `1978` as conflicting with their other years. Support then fails
  because the entailing chunk and a clean entity-pass never coincide on the *same*
  chunk.

The design intent (CLAUDE.md: "proper nouns only fail when the evidence names
something conflicting") is right; the gap is that the *conflict* test is too
loose — benign surface variants (possessive, singular/plural, partial vs full
name) don't match, and the mere presence of any other same-role value is then
read as a conflict. The genuinely-correct vetoes (item-10 Sobers≠Gooch, item-24
Victoria/Eleanor) are real entity swaps, so the fix must sharpen precision without
losing those.

### A second, independent issue: the splitter over-segments (adjudicator finding)

The human adjudicator flagged, unprompted, that the draft is split into claims
that are not the answer, which then read `unclear` / `not supported` on their own:

- **meta / provenance claims** — item-02 "The information about the festival dates
  is sourced from the 2015 event details"; item-23 "The word 'Marfa' is a proper
  noun representing a city in Texas."
- **entity-less fragments** — item-08 "The movie aired on NBC in 1973" (no film
  title, which the question needs); item-16 does not answer the OS question.
- **vague / redundant** — item-30 "Adipose tissue exists in multiple locations"
  (without listing them, while item-27 already lists them).

Effect: (a) it inflates the B5 unsupported (0.662) and abstention (0.582) rates —
some "dropped" claims are bad fragments that *should* be dropped, not verifier
strictness; (b) every extra claim adds a batch of entity_check pairs, amplifying
the false-veto count; (c) it confounds the audit itself (the adjudicator noted
item-02 "should be ignored or combined with item-12"). This is a ClaimSplitter
quality problem, distinct from the entity-check false vetoes.

## Remedies (two causes, two fixes — both inside Generator role B)

1. **entity_check precision (the measured main defect, 60% false veto).**
   Sharpen proper-noun matching so benign variants match — possessive/plural
   normalisation and partial-vs-full containment ("Patriots" ≈ "Patriot
   militia", "King's Mountain" ≈ "Kings Mountain") — and stop scoring the mere
   presence of *other* same-role values as a conflict when the claim's own value
   is present in the chunk. Preserve the true-swap catches (Sobers≠Gooch).
2. **ClaimSplitter over-segmentation.** Restrict claims to atomic facts that
   answer the question; suppress meta/provenance, entity-less, and vague/redundant
   claims. This also shrinks the entity_check pair count, so it partly relieves
   defect 1.

## Caveats

- Single adjudicator, n=15 per dropped stratum / 10 controls — directional, not a
  tight estimate. The false-veto signal (0.60 vs 0.07) is large enough to act on;
  the exact rate is not the point.
- ASQA calibration only; 2Wiki multi-hop residue is visible in the ~20–27%
  unclear rates but not separately characterised here.
- The mechanism in "Why entity_check false-vetoes" was reproduced by re-running
  the production `EntityConsistencyChecker` on the audited claim×evidence pairs;
  the scores above are from the human packet.
