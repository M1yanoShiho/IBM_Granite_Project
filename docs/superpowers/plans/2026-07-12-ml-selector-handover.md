# ML Selector Handover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scattered ML Selector V1/V2 material with one concise handover entry, one historical experiment record, one V2 plan, one V2 tracker, and a Git-tracked compact evidence package.

**Architecture:** `docs/ml-selector/README.md` is the only navigation entry. V1 interpretation, V2 protocol, and V2 run state live in separate Markdown files; small evidence remains in `docs/data/ml_selector_validation/`, while private-server caches are documented but not copied.

**Tech Stack:** Markdown, Git, existing JSON/CSV experiment artifacts, shell-based link and repository checks.

## Global Constraints

- V1 is labelled an exploratory diagnostic experiment, not a final evaluation.
- FinanceBench is labelled `EXPOSED_DIAGNOSTIC_ONLY` and cannot be used for V2 tuning, selection, stopping, or final blind evaluation.
- Do not copy the 986 MB private-server result tree into the repository.
- Keep only four handover documents under `docs/ml-selector/`.
- Do not add credentials, full JSONL caches, checkpoints, or generated HTML.
- Preserve the three existing result figures and the compact evidence under `docs/data/ml_selector_validation/`.

---

### Task 1: Create the V1 record and handover entry

**Files:**
- Create: `docs/ml-selector/README.md`
- Create: `docs/ml-selector/V1_EXPERIMENT_RECORD.md`
- Read: `docs/ml-selector-experiment-plan.md`
- Read: `docs/ml-selector-experiment-tracker.md`
- Read: `docs/W5 ML selector/current_findings_top10_evidence_reranking.md`
- Read: `docs/data/ml_selector_validation/**/*.json`

**Interfaces:**
- Consumes: existing V1 plan, completed Tracker, current corrected conclusion, result JSON, audit JSON, and result figures.
- Produces: the canonical human entry point and the canonical V1 historical record.

- [ ] **Step 1: Extract the evidence-backed V1 facts**

Run:

```bash
find docs/data/ml_selector_validation -type f -name '*.json' -maxdepth 4 -print
python -m json.tool docs/data/ml_selector_validation/pilot/results.json >/dev/null
python -m json.tool docs/data/ml_selector_validation/ramdocs/results.json >/dev/null
python -m json.tool docs/data/ml_selector_validation/financebench/results.json >/dev/null
python -m json.tool docs/data/ml_selector_validation/contractnli/results.json >/dev/null
```

Expected: all retained JSON files parse successfully and the four main result files exist.

- [ ] **Step 2: Write `V1_EXPERIMENT_RECORD.md`**

Include these exact sections: status and warning; research question; system pipeline; datasets and split approach; work completed by milestone; Gate results; evidence-backed findings; FinanceBench exposure; what V1 did not prove; reusable code/data; private-server artifact disposition; lessons carried into V2.

The record must state that the source tree was `/scratch/fl25387/IBM_Granite_Project/results/ml_selector/` on private host `it097952`, was approximately 986 MB, was not migrated, and is not a dependency for continuing V2.

- [ ] **Step 3: Write `README.md`**

Include: current status; a four-link reading order; a five-minute handover summary; V1/V2 boundary; FinanceBench warning; compact evidence links; first action for the successor.

- [ ] **Step 4: Verify V1 claims against evidence**

Run:

```bash
rg -n "Gate 0|Gate 1|Gate 2|Gate 3|FinanceBench|EXPOSED_DIAGNOSTIC_ONLY|986 MB|it097952" docs/ml-selector/README.md docs/ml-selector/V1_EXPERIMENT_RECORD.md
```

Expected: both files consistently describe V1 as diagnostic, identify the failed/not-run Gates, and carry the FinanceBench restriction.

### Task 2: Create the canonical V2 plan and tracker

**Files:**
- Create: `docs/ml-selector/V2_EXPERIMENT_PLAN.md`
- Create: `docs/ml-selector/V2_EXPERIMENT_TRACKER.md`
- Read: `docs/W5 ML selector/refine-logs/EXPERIMENT_PLAN.md`
- Read: `docs/W5 ML selector/refine-logs/EXPERIMENT_TRACKER.md`
- Read: `docs/W5 ML selector/refine-logs/DESIGN_REVIEW.md`

**Interfaces:**
- Consumes: reviewed Graph-Assisted Evidence Selector 2.0 protocol, run table, and design-review corrections.
- Produces: the only V2 protocol and the only mutable V2 run-state document.

- [ ] **Step 1: Build `V2_EXPERIMENT_PLAN.md` from the reviewed plan**

Preserve the research claims, three-relation graph design, zero-new-human-annotation rule, dataset roles, Gates, controls, statistics, run order, and reproducibility requirements. Add a top-level protocol warning that FinanceBench is `EXPOSED_DIAGNOSTIC_ONLY` and excluded from V2 model development and final blind evaluation.

- [ ] **Step 2: Merge the final design review into the plan**

Add a concise “Design review status” section recording the final PASS and the enforced corrections: three core relations; automatic provenance Gates; OOF full pipeline; NLI-without-Graph and shuffled-Graph controls; RAMDocs mined passages treated as unjudged; no FinanceBench result in the V2 claim.

- [ ] **Step 3: Build `V2_EXPERIMENT_TRACKER.md`**

Retain Run IDs R000–R080. Add columns for Git commit, seed, host/job, output package, Gate/metric, and interpretation. Keep every not-yet-run row at `TODO`; do not imply V2 execution has started.

- [ ] **Step 4: Verify protocol consistency**

Run:

```bash
rg -n "EXPOSED_DIAGNOSTIC_ONLY|zero-new-human-annotation|CLAIM_SUPPORTS|CLAIM_REFUTES|SAME_SOURCE|PASS|R000|R080" docs/ml-selector/V2_EXPERIMENT_PLAN.md docs/ml-selector/V2_EXPERIMENT_TRACKER.md
```

Expected: the V2 plan contains the frozen restrictions and review result; the Tracker spans R000–R080 and contains no completed V2 runs.

### Task 3: Remove redundant files and update project navigation

**Files:**
- Delete: `docs/W5 ML selector/`
- Delete: `docs/ml-selector-experiment-plan.md`
- Delete: `docs/ml-selector-experiment-tracker.md`
- Delete: `docs/ml_selector_validation_results.html`
- Modify: `docs/results-summary.md`

**Interfaces:**
- Consumes: the four canonical handover files created in Tasks 1–2.
- Produces: one non-duplicated documentation path with no stale V1 plan or generated report competing as a current source.

- [ ] **Step 1: Update `docs/results-summary.md`**

Add a short ML Selector handover note linking to `ml-selector/README.md`. State that the V1 evidence is diagnostic and FinanceBench is exposed; do not rewrite the existing retrieval/corroboration result tables.

- [ ] **Step 2: Remove old tracked V1 entry files**

Delete the old plan, Tracker, and generated HTML only after their unique information is represented in the V1 record.

- [ ] **Step 3: Remove the untracked W5 structure**

Delete all historical Markdown, HTML renderings, duplicate dated aliases, old manifest, and standalone design review under `docs/W5 ML selector/` after the canonical V2 files exist.

- [ ] **Step 4: Confirm no stale references remain**

Run:

```bash
rg -n "ml-selector-experiment-plan|ml-selector-experiment-tracker|ml_selector_validation_results|W5 ML selector|current_findings_top10" . --hidden --glob '!.git/**' --glob '!docs/superpowers/**'
```

Expected: no matches.

### Task 4: Validate the handover package and commit

**Files:**
- Verify: `docs/ml-selector/*.md`
- Verify: `docs/data/ml_selector_validation/**`
- Verify: `docs/assets/ml_selector_validation/**`
- Verify: `.gitignore`

**Interfaces:**
- Consumes: all preceding documentation and cleanup changes.
- Produces: a reviewable, linked, compact Git handover commit.

- [ ] **Step 1: Validate document count and evidence files**

Run:

```bash
find docs/ml-selector -maxdepth 1 -type f -name '*.md' | sort
find docs/data/ml_selector_validation -type f | sort
find docs/assets/ml_selector_validation -type f | sort
```

Expected: exactly four Markdown handover files; compact JSON evidence and three figures remain.

- [ ] **Step 2: Check Markdown links**

Run a local link checker over all relative Markdown links in `docs/ml-selector/*.md` and fail if any target does not exist.

Expected: zero broken local links.

- [ ] **Step 3: Check for large or sensitive additions**

Run:

```bash
find docs/ml-selector docs/data/ml_selector_validation -type f -size +10M -print
git diff --check
git status --short
```

Expected: no new file above 10 MB, no whitespace error, and only the planned create/delete/modify set appears.

- [ ] **Step 4: Review the final diff**

Run:

```bash
git diff --stat
git diff -- docs/results-summary.md docs/ml-selector
```

Expected: the diff implements the approved four-document structure and consistently marks FinanceBench as exposed.

- [ ] **Step 5: Commit the implementation**

```bash
git add docs/ml-selector docs/results-summary.md docs/data/ml_selector_validation docs/assets/ml_selector_validation docs/ml-selector-experiment-plan.md docs/ml-selector-experiment-tracker.md docs/ml_selector_validation_results.html
git commit -m "docs: consolidate ML selector handover"
```

Expected: one implementation commit following the already committed design and plan documents.

