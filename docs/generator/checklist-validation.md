# Checklist validation — LLM analyzer against ASQA disambiguations

Seed 13, 400 questions. The analyzer saw the question text only. `qa_pairs` are evaluation-side; they never entered Generator input.

## Matching rule

A generated requirement R corresponds to a gold disambiguation G when **MiniCheck** judges `premise = G's sub-question, hypothesis = R` to be entailment. MiniCheck's recall is 0.620, so it misses genuine correspondences and the measured precision is a **floor**.

## Gated numbers

| quantity | value |
|---|---|
| **checklist precision** (gated, >= 0.60) | **0.556** (625/1125) |
| checklist recall (reported, not gated) | 0.715 (937/1311) |
| requirements matching >1 gold reading (over-generic) | 426/1125 (0.379) |

## Distribution

requirements per query — mean **2.81**

| requirements | queries |
|---|---|
| 0 | 8 (2.0%) |
| 1 | 71 (17.8%) |
| 2 | 32 (8.0%) |
| 3 | 176 (44.0%) |
| 4 | 103 (25.8%) |
| 5 | 10 (2.5%) |

| gold question type | empty checklists |
|---|---|
| unambiguous (gold readings <=1) | 0/0 |
| ambiguous (gold readings >1) | 8/400 (0.020) |

An empty rate near zero on **unambiguous** questions would mean the analyzer is manufacturing obligations; a high empty rate on **ambiguous** ones means it is missing real readings.

## Analyzer prompt, verbatim

```text
A question can have more than one reasonable interpretation. List what a complete answer is obliged to cover.

Rules:
- Write each item as an obligation on the answer, never as a fact. Say what the answer must specify; do not say what the answer is. You do not know the answer and must not guess it.
- Only list an item when the question genuinely has that separate reading.
- If the question has a single reasonable interpretation, return an empty list.

Return JSON only, in this shape: {{"requirements": ["The answer must specify ..."]}}

Question: {question}
```

## Samples

### Ambiguous (multiple gold readings)

- **Q:** Where can adipose tissue be found in the body?
  - gold readings: 2
  - req: The answer must specify the anatomical locations where adipose tissue is located, such as subcutaneous fat, visceral fat surrounding organs, and intramuscular fat within muscle tissue.
- **Q:** When was the commission fir sc/st established?
  - gold readings: 2
  - req: The answer must specify the exact date or year when the National Commission for Scheduled Castes and Scheduled Tribes was established.
  - req: The answer must indicate the authority or legislation under which the commission was created.
  - req: The answer must provide any relevant background information about the establishment of the commission, such as the purpose or the legal framework.
- **Q:** Who made it through the divisional finals world of dance?
  - gold readings: 3
  - req: The answer must specify which competitors or teams made it through the divisional finals of the World of Dance.
  - req: The answer must clarify the order or ranking of those competitors within their respective divisions.
  - req: The answer must indicate if there were multiple divisions and specify the winners of each division.
  - req: The answer must provide any relevant details about the performances or criteria that led to their advancement.
- **Q:** Who beat villanova last year in the ncaa tournament?
  - gold readings: 6
  - req: The answer must specify the team that defeated Virginia Tech (Villanova) in the most recent NCAA Tournament.
  - req: The answer must indicate the year of the tournament in which this victory occurred.
  - req: The answer must provide the full official name of the defeating team, if it is a well-known or commonly referenced team.
- **Q:** What is the os of samsung smart tv?
  - gold readings: 4
  - req: The answer must specify the operating system used in Samsung Smart TVs.
  - req: The answer must clarify whether it refers to the software platform (e.g., Tizen, WebOS) and any version specifics.
  - req: The answer must indicate if there are variations of the operating system across different Samsung TV models or generations.
- **Q:** When does the new super troopers movie come out?
  - gold readings: 3
  - req: The answer must specify the release date of the new Super Troopers movie.
  - req: The answer must indicate whether this refers to a theatrical release, a streaming platform, or both.
  - req: The answer must clarify if there are any regional differences in release dates.
  - req: The answer must mention if the movie is a sequel, prequel, or part of a series.

### Unambiguous (one gold reading)



---

## Gate outcome: **FAILED**

The gate pre-registered in the G4 ledger entry (commit `2f4ca16`, written before
any number existed) required **checklist precision ≥ 0.60**. Measured: **0.556**.

Per the pre-registration, the consequence is fixed in advance: the analyzer is
**not fit to drive the experiment** — fall back to the Route A result and report
the rule-based domain limit as the finding.

### The headline understates the failure

Precision counts a requirement as correct if it matches *at least one* gold
reading. Splitting by how many it matches:

| requirement matches | count | share |
|---|---|---|
| **0 gold readings — invented** | 500 | **0.444** |
| exactly 1 — a specific, useful obligation | 199 | **0.177** |
| **more than 1 — generic catch-all** | 426 | **0.379** |

Over-generic requirements inflate precision without carrying information: "The
answer must indicate the year of the tournament" matches all six Villanova
readings precisely because it is vague enough to match anything. Discounting
those, only **17.7%** of generated requirements correspond to one specific
reading. Nearly half are invented outright.

### What the analyzer actually does

ASQA's disambiguations run along **one concrete axis per question**, usually a
scope or time qualifier:

```
Q: Who beat villanova last year in the ncaa tournament?
   gold: ... in 2017 / ... in 2015 / ... in 2014 / ...
Q: What is the os of samsung smart tv?
   gold: newer sets / former OS / older sets / newer sets
Q: Who has scored the highest number of runs in test cricket?
   gold: in a career / in a single match / in a series / in a calendar year
```

The analyzer instead writes a **generic answer-quality rubric**:

```
- The answer must indicate the authority or legislation under which the commission was created.
- The answer must provide any relevant background information about the establishment.
- The answer must clarify the order or ranking of those competitors.
- The answer must mention if the movie is a sequel, prequel, or part of a series.
```

These are plausible-sounding obligations that were never the question's actual
ambiguity. Two further defects appear in the samples, both of the category the
prompt explicitly forbade:

- **Hallucinated entity:** "The answer must specify the team that defeated
  *Virginia Tech (Villanova)*" — Virginia Tech is invented.
- **Answer-guessing:** "whether it refers to the software platform *(e.g., Tizen,
  WebOS)*" and "such as *subcutaneous fat, visceral fat surrounding organs*" — the
  analyzer is supplying answer content it was told it cannot know.

### One diagnostic could not be run

The pre-registration flagged that a near-zero empty rate would be a warning sign
of manufacturing obligations for unambiguous questions. **That cannot be tested
here: ASQA contains only ambiguous questions** — all 400 have more than one gold
reading (mean 3.28). The empty rate of 2.0% is therefore not by itself evidence
of over-firing, and the over-generic rate (37.9%) is the available proxy instead.

### Status

Automatic criterion failed. The human spot check (`local/audit/checklist-spotcheck.md`,
20 blind items, seed 13) is still delivered, because the pre-registration states
that MiniCheck's 0.620 recall makes the automatic precision a floor and that the
human judgement governs on significant disagreement. The qualitative evidence
above points the same way as the automatic number, but that is for the reviewer to
confirm, not for this report to assume.

Stopped for review, as the guide requires.
