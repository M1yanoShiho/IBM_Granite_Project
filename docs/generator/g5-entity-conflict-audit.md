# Entity-conflict audit — the fix changed the population completely and the error rate not at all

Blind human adjudication of the entity-conflict drops, run twice: once before the
spaCy/genuine-conflict fix and once after. Protocol both times: verdict hidden,
no highlighting, no score ordering, adjudicator sees claim + one evidence passage
+ the question.

Under verify-and-annotate everything unsupported is kept and labelled, so an
entity conflict is the **sole remaining reason a claim is deleted outright**.
That makes `entity_check`'s precision decisive in a way it was not when other
deletion paths shared the load.

## Result — round 2 (post-fix), 20 items of 66, seed 17

| human verdict | count | share | consequence |
|---|---|---|---|
| **supported** | **14** | **0.700** | **false veto — correct content destroyed** |
| conflicting | 3 | 0.150 | the drop was right |
| unrelated | 3 | 0.150 | should have been **annotated**, not dropped |

**False-veto rate 0.700, 95% Wilson CI [0.481, 0.855].**
**Wrongly destroyed (supported + unrelated) 17/20 = 0.850, CI [0.640, 0.948].**

### Against round 1

| | round 1 (pre-fix, n=20 of 72) | round 2 (post-fix, n=20 of 66) |
|---|---|---|
| supported (false veto) | 14 (0.700) | **14 (0.700)** |
| conflicting (correct) | 4 (0.200) | 3 (0.150) |
| unrelated (should annotate) | 2 (0.100) | 3 (0.150) |
| trigger types | `name` 75, number 9, date 5 | org 29, location 18, person 13, product 12, number 4, date 3 |
| reason | value conflict 87 / absent 2 | value conflict 78 / absent 1 |

**The fix worked on exactly what it targeted and bought nothing.** Every trigger
is now a real NER type — `name:some`, `name:season`, `name:unemployment` are gone
from the population entirely, which is what the spaCy switch was for. Routing
`absent` to annotate rather than drop left only 1 absent case in 78. And the
false-veto rate is **unchanged to the item**: 14/20 both times.

Two blind rounds, disjoint mechanisms, same answer. This is not sampling noise
around a real improvement; it is evidence that the failure was never mainly about
which spans the extractor picked up.

## Why: the check fires on a comparison it is not equipped to make

The round-1 diagnosis was extraction quality. Round 2 rules that out and points
at the comparison itself. Reading the adjudicator's reasons on the three drops
that were *correct*:

- **E06** — "we fought against Italy" vs evidence naming Germany, Italy, and
  Japan. Conflicting because the claim is **incomplete**, not because an entity
  was swapped.
- **E17** — "Washington's minimum age is 18" vs evidence saying all 50 states
  require 18 but "most states permit under 18 with parental permission".
  Conflicting because of an unresolved **quantifier scope**.
- **E18** — "permitted women as bishops on 14 July 2014" vs evidence saying the
  Synod *approved* on that date and implementation followed in November.
  Conflicting on **approved vs implemented** — a temporal-semantic distinction.

None of the three is an entity mismatch. The entity checker got the right verdict
on all three for reasons it does not model. Meanwhile the 14 false vetoes are
cases where the evidence plainly states the claim and the checker vetoed on a
surface difference in some role.

So the layer is close to **never right for the right reason**: 0.150 correct, and
that 0.150 is coincidental.

A related pattern in the *supported* group is worth recording because it is a
different module's problem: E01, E07, E09, E12, E16 and E20 are all adjudicated
supported but flagged by the adjudicator as **partial or ambiguous answers** —
"the claim is not complete to the question, which should include Maureen O'Hara
as well" (E12, with E07 supplying the other half). That is a claim-splitting and
checklist-coverage issue, not an entity issue, and it does not affect this
audit's verdict.

## What this does to the G6 numbers

The G6 figures were computed with these drops in place: 60 claims dropped of 434
routed (13.8%). At a 0.850 wrongly-destroyed rate that is roughly **51 claims
deleted that should have survived**, most of them entailed and therefore
destined to become **cited** sentences.

The direction of the bias should not be guessed. Returning those claims would add
cited sentences (moving precision either way), raise recall's denominator, and
raise coverage and correctness. It needs measuring, not inferring.

## Recommendation

Route entity conflict to a **distinct annotation** ("evidence conflicts with this
claim") instead of deletion. The system would then destroy nothing at all, which
is the coherent end state of annotate-not-delete: the contract already guarantees
every ungrounded sentence is labelled, and there is no longer a defensible reason
for one path to bypass that guarantee at a measured 85% error rate.

Two rounds of blind adjudication now support it and the last calibration round is
spent. **Not implemented here** — it would confound the contract lift measured in
G6, which was pre-registered as a one-change round. It is the first item for the
next round, or a strong future-work item with an audit behind it.
