# Analyzer dry run on the calibration questions

Required before rebuilding the completeness harness: `RuleBasedQueryAnalyzer` is
rule-based with an enterprise-oriented vocabulary, ASQA is open-domain Wikipedia,
and what it produces there had never been looked at. Run over the same 400
questions, same construction and seed as the G3 run. No models, no gold fields.

Reproduce with `scripts/analyzer_dry_run.py --limit 400 --seed 13`; raw output in
`results/analyzer-dry-run.jsonl`.

## Distribution

| measure | result |
|---|---|
| `required_facts` per query — mean | **1.00** |
| queries producing exactly 1 fact | **400/400 (100%)** |
| queries producing an empty checklist | 0 (0%) |
| facts sourced from the `METRICS` vocabulary | **1/400 (0.2%)** |
| facts sourced from the **focus fallback** | **399/400 (99.8%)** |
| `constraints` per query — mean | 0.10 |

## Why: the two lines that decide it

```python
required = tuple(term for term in terms if term.lower() in METRICS)
if not required:
    required = (focus,)          # focus = " ".join(content_terms[:6])
```

`METRICS` is `{revenue, growth, profit, margin, income, cash, assets,
liabilities, termination, notice, clause, rate, percentage}`. On open-domain
Wikipedia questions it essentially never fires, so the checklist is always the
**focus fallback**: the question's first six non-stopword tokens joined by
spaces, with original casing.

## What the items actually look like

```
Q: Where can adipose tissue be found in the body?
   required_facts: ['can adipose tissue found body']
Q: What is the os of samsung smart tv?
   required_facts: ['os samsung smart tv']
Q: Who does sansa marry on game of thrones?
   required_facts: ['sansa marry game thrones']
```

These are not facts. They are keyword bags restating the question, so the
completeness prompt — "Decide whether the answer already states the required
fact" — is being asked a malformed question on every query.

Two further defects are measurable:

| defect | rate | example |
|---|---|---|
| auxiliary/verb junk retained in the "fact" | **66/400 (16.5%)** | `has scored highest number runs test` |
| discriminative tail dropped by the 6-term cap | **42/400 (10.5%)** | see below |

The truncation is the more damaging of the two, because it can remove the part of
the question that carries its meaning:

```
Q: Who has won the most trophies man utd or liverpool?
   fact   : has won most trophies man utd          DROPPED: ['liverpool']
Q: As blues moved into chicago's south side what style of blues developed?
   fact   : blues moved into chicago south side    DROPPED: ['style', 'developed']
Q: Who has scored the highest number of runs in test cricket?
   fact   : has scored highest number runs test    DROPPED: ['cricket']
```

In the first, one of the two entities being compared is gone; in the second, the
entire question intent ("what style … developed") is gone.

## Which outcome

Against the three outcomes set out for this task:

- **Not outcome 1 ("sensible facts").** Definitively not — the items are keyword
  bags, not checkable facts, and 10.5% are semantically mangled.
- **Not cleanly outcome 2 ("mostly empty or trivial") either.** The checklist is
  never empty: it produces exactly one item on 100% of queries, so the
  completeness step fires on every single query rather than rarely.
- **Closest to outcome 3 ("noisy or junk terms")** on the content of the items:
  junk tokens on 16.5%, meaning-destroying truncation on 10.5%.

**The genuinely open question, which this dry run cannot answer:** whether the
LLM completeness checker treats a keyword bag as *covered* whenever the draft is
on-topic — in which case the mechanism degenerates into a crude relevance check
that rarely produces gaps (outcome 2 in effect) — or produces erratic gaps on the
mangled ones (outcome 3 in effect). Settling that needs the checker to be run on
real drafts, which is GPU work and is exactly what this task exists to gate.

Reported without proceeding, as instructed.
