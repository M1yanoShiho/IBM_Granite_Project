# Entity-conflict audit — the only content-destroying path is 80% wrong

Blind human adjudication of the entity-conflict drops from the G5
verify-and-annotate run. 20 items sampled from a population of **72**, seed 13,
protocol as before: verdict hidden, no highlighting, no score ordering.

Under verify-and-annotate everything unsupported is kept and labelled, so an
entity conflict is the **sole remaining reason a claim is deleted outright**.
That makes `entity_check`'s precision decisive in a way it was not when other
deletion paths shared the load.

## Result

| human verdict | count | share | consequence |
|---|---|---|---|
| **supported** | **14** | **0.700** | **false veto — correct content destroyed** |
| conflicting | 4 | 0.200 | the drop was right |
| unrelated | 2 | 0.100 | should have been **annotated**, not dropped |

**80% of drops were wrong**: 70% destroyed content the evidence actually
supports, and a further 10% had no conflict to justify deletion and belonged in
the annotate path.

Population breakdown (all 72 drops): triggering entity type **name 75**, number
9, date 5 (89 triggers, some claims firing more than one); by reason, **value
conflict 87 vs absent-from-evidence 2**; 57 drops fired on a single mismatch.

## Two distinct mechanisms produce the false vetoes

**1. Common nouns extracted as proper names.** The rule-based extractor
recognises names by capitalisation, so a sentence-initial common noun becomes a
"name" and then vetoes on it. Actual triggers from the audited sample:

```
name:some            name:season         name:small
name:substitutions   name:unemployment   name:household
```

This is the residual risk `REQUIRE_PRESENCE`'s own docstring describes and
declares "bounded by the fact that this check only ever runs on pairs NLI already
labelled entailment". **The audit shows it is not bounded in practice** — it is
the dominant failure mode.

**2. Compatible alternatives read as conflicts.** When the evidence carries
several values in the same role, any difference is scored as a conflict even
where the values are compatible. From the adjudicator's reasons:

- E07 — "West Germany win the world cup two times, but that is not conflicting,
  they are compatible."
- E06 — "There are several names for the bait car, but that is not conflicting,
  they are compatible."
- E05 — "The claim successfully caught the final release date. There are other
  dates but not the real one."

A third, narrower case appeared once: E03, where the adjudicator notes the check
fired on a *name* when "the focus of entity check should be time instead of
name" — the check has no notion of which entity the question is actually about.

## What this does to the G5 numbers

The G5 citation-precision figures were computed with these drops in place. If
~80% of them are wrong, roughly 58 claims were removed from the pipeline that
should not have been, most of which were entailed and would have become **cited**
sentences. Every G5 metric is therefore measured on a pipeline that is deleting
correct, citable content at a rate the audit puts at four in five drops.

The direction of the bias is not obvious and should not be guessed: returning
those claims would add cited sentences (affecting precision either way), raise
recall's denominator, and raise correctness. It needs to be measured, not
inferred.

## Status

Reported, not acted on. Fixing `entity_check` is bug-fix class by the standing
test — the defect is stateable without reference to any metric ("the extractor
treats sentence-initial common nouns as proper names and vetoes on them"; "several
compatible values in the same role are scored as a conflict") and it was
established by human adjudication, not by a number moving. But it is a
substantial change this close to freeze, so the call is the reviewer's.
