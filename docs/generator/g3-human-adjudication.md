# G3 human adjudication — results, and a harness defect it exposed

Human adjudication of the 70-item blind packet (`local/audit/g3-packet.md`, seed
13), scored with `scripts/g3_packet_score.py`. The adjudicator is independent of
the model stack; nothing here was model-judged.

**The headline is not in the scores.** The free-text reasons exposed a defect in
how this harness builds `required_facts`, which invalidates every completeness
result and — importantly — makes the naive reading of Section C a
misattribution to another team.

## Scores as measured

| section | result |
|---|---|
| **C** — is the required fact present in the selected evidence? | **absent 13/20 (0.65)**, partial 6/20, present 1/20 |
| **A** — does this evidence support this statement? | human-supported: baseline 6/8, verified-draft 6/7, verified-recheck 8/12 |
| **B** — was the required fact already answered in the draft? | stratified false-gap estimate **0.178** vs automatic 0.073 |

## The defect: `required_facts` are not what the answer must contain

This harness built `QueryChecklist.required_facts` from ASQA
`annotations[].knowledge[].content`, keeping entries that overlap the gold long
answer. Inspecting the data shows what that field actually is:

- `qa_pairs` — ASQA's **disambiguated sub-questions with short answers**
  ("Who has the highest goals in *men's international* football?" → `Ali Daei`).
  This *is* ASQA's completeness target: a complete long answer must cover every
  disambiguation.
- `annotations[].knowledge` — **background Wikipedia snippets** (`wikipage` +
  `content`) used as provenance. Not the answer, and frequently empty.

Measured over the 400 cases actually used:

| check | result |
|---|---|
| gold answer-sets whose text appears in **any** required_fact | **406/1311 = 31.0%** |
| cases whose required_facts cover **at least one** gold answer | **215/400 = 53.8%** |
| required_facts per case vs ASQA's real target (`qa_pairs`) per case | **1.80 vs 3.28** |

The adjudicator reached the same conclusion independently, unprompted:

- C01 (question: *speed limit through the Hindhead tunnel*; required fact: the
  tunnel is part of a 4-mile bypass) — "The required fact does not have the answer
  to the question, neither."
- C08 — "the evidences indicate that Lynn Cartwright played the older Dottie …
  **which is the answer to the question while the required fact is not**."
- C18, C19, C20 — "The required fact is not the answer to the question, neither."
- B04 — "the required fact is really bad for the question."
- B13 — "the draft answer seems to be **the correct answer to the question rather
  than the required fact**."
- **15 of 20** Section B reasons are variants of: "Even though the answer does not
  state the required fact, it had answered the question **in a more proper way**."

There is a **second, independent defect** in the same construction: these facts
are *gold-derived*. They come from the annotator's knowledge list and were then
filtered by overlap with the gold long answer. `QueryChecklist` is an **input** to
the Generator, and the production path builds it with `RuleBasedQueryAnalyzer`
from the **question text alone**. So the verified arms were fed oracle-derived
guidance the real system never has. It mostly *hurt* them — spurious requirements
drove spurious gaps and abstentions — but it is methodologically wrong either way.

## Section C: the cross-team implication, corrected

The scorer prints "mostly ABSENT → Selector recall finding". **That reading is
wrong and must not be forwarded to the Selector team.**

The facts are absent from the selected evidence because they are *background
sentences that the question never asked for*. Retrieval fetched passages for the
question; it was then judged on whether they contain an annotator's background
note about a different aspect. Of course they mostly do not. Handing that over as
a recall finding would misattribute this harness's data-construction bug to
another module.

The genuine Section C signal is narrower and still useful: when the required fact
*was* answerable, the evidence usually did contain the answer to the **question**
even where it lacked the "fact" — the adjudicator points this out repeatedly
(C08, C12, C14, C16, C20: "evi-3 may contain the answer to the question"). That is
evidence *for* retrieval, not against it.

## Section A: comparatively healthy, with two real findings

| origin | human-supported | agreement w/ MiniCheck claim-level | agreement w/ answer-level |
|---|---|---|---|
| baseline | 6/8 (0.750) | n/a | 6/8 (0.750) |
| verified-draft | 6/7 (0.857) | **6/7 (0.857)** | 6/7 (0.857) |
| verified-recheck | 8/12 (0.667) | n/a | **5/12 (0.417)** |

1. **The dilution question is not settled by this sample.** For verified-draft,
   MiniCheck's claim-level and answer-level verdicts agree with the human equally
   (6/7 each), so the packet cannot separate them here. n=7 is too small; Task 1's
   +0.142 arm-specific delta remains the stronger evidence.
2. **MiniCheck under-credits recheck citations.** Humans call 0.667 of them
   supported while MiniCheck's answer-level verdict agrees with the human on only
   0.417 — i.e. part of the recheck penalty is judge limitation, not citation
   quality. This is the amendment's "judge limitation" row, and it further weakens
   the case that recheck citations are intrinsically bad.
3. Recheck items draw a recurring complaint — "**Incomplete statement**" (A11,
   A14, A15, A20) — which independently confirms the bare-fragment defect the
   diagnosis found (median 17 characters).

**One artifact I own:** A04's reason notes the statement "might be wrongly split
due to the dot in 'St. Louis'". Correct — the *packet builder* fell back to a
regex sentence splitter because NLTK was unavailable locally, while the *scoring*
run used `nltk.sent_tokenize` on the cluster. The scored metrics are unaffected;
that one packet item was mis-split.

## What this invalidates, and what survives

**Invalidated — must be rebuilt before being reported:**

- B4 completeness results, including G2's 0.786 own-fact coverage and the
  false-gap rate at every revision (6.6% automatic, 0.178 human-corrected). All of
  them answer "did the draft state this background sentence", which is not the
  question anyone cares about.
- The abstention analysis and its attribution: gaps were raised against a
  specification that was not the answer, so "96% of abstentions came from genuine
  gaps" describes genuine *background* gaps.
- `verified-full` coverage (0.331) as a measure of anything about completeness.
- Section C's cross-team implication, as set out above.

**Survives — independent of `required_facts`:**

- G1's six-arm verifier selection.
- Task 1's ALCE artifact finding, including the per-arm delta table. Citation
  metrics never read `required_facts`.
- **`verify-only` vs baseline** — the headline. `verify-only` disables
  completeness and recheck entirely, so it never sees a required fact. Its
  +0.087/+0.133 citation-precision win and +0.116/+0.141 recall win stand.
- The earlier blind audit's claim-level in-chain citation precision (0.900).
- STR-EM answer correctness, which reads `qa_pairs` short answers — the correct
  field all along.

So the method's central result is untouched; what is invalidated is everything
that depended on the completeness specification.

## What the fix should be

Two changes, both bug-fix class (an invalid construction demonstrated by human
adjudication, not a tuning move):

1. **Stop deriving the checklist from gold.** Build `QueryChecklist` with the
   production `RuleBasedQueryAnalyzer` from the question text, as the pipeline
   does. This removes the oracle input.
2. **Evaluate completeness against ASQA's own target.** Completeness is "did the
   answer cover the disambiguated interpretations", i.e. STR-EM coverage over
   `qa_pairs` — which this harness already computes as `answer_correctness`. The
   `qa_pairs` short answers must stay on the evaluation side only; putting them in
   `required_facts` would feed gold answers into the Generator's input.

Together these mean completeness stops being measured by a proxy that was neither
the answer nor available to the system, and starts being measured by the dataset's
own definition.
