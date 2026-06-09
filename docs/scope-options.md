# Scope Options: Evaluation vs. System

A reference for the first meeting with IBM. The project brief asks for **both**
*building* retrieval/RAG pipelines and *evaluating* them. These two emphases
imply very different repository structures and deliverables, so we need to
agree on a primary direction before committing engineering effort.

> **Current state:** the repo is, in effect, an **evaluation harness**
> (`eval/niah_runner.py` + `eval/metrics.py` + `notebooks/` heatmaps +
> trimmed haystacks). `app/` and `src/rag_pipeline.py` are currently
> supporting components. This is a safe foundation for *either* direction.

---

## Route A — Emphasis on Evaluation (current trajectory)

**Goal:** a rigorous, reproducible study of IBM Granite's long-context
retrieval performance, with actionable deployment guidance.

**Change required: small — mostly hardening what already exists.**

| Add | Location | Why |
| --- | --- | --- |
| `results/` directory | repo root | Store CSVs / figures separately from code (the runner already writes here, but the dir isn't created yet). |
| `configs/` + YAML files | repo root | Externalise experiment parameters (lengths, depths, models) → reproducibility, our core selling point. |
| Statistical rigor | `eval/` | Multiple runs, mean ± confidence intervals, fixed random seeds. |
| More visualisations | `notebooks/` | Comparison plots, distractor-misleading curves, etc. |
| Experiment logging | `results/logs/` | Record the config and timestamp of each run. |

**Deliverables:** evaluation report + heatmaps + comparison tables + a
lightweight Streamlit demo to showcase findings.

**Pros:** ~80% already in place; plays to a Data Science / Data Visualisation
skill set; produces a clean, defensible academic result.
**Cons:** lighter on the "build a usable system" half of the brief.

---

## Route B — Emphasis on System (a usable retrieval product)

**Goal:** an intelligent retrieval system that ingests real enterprise-scale
documents and returns precise, trustworthy answers.

**Change required: large — the eval-centric layout is demoted to a support role.**

The current haystack is a toy document trimmed to a fixed token count; a real
system must handle a real document corpus. New components:

```
ingestion/            # NEW: real document ingestion
  loaders.py            (PDF / Word / HTML parsing, not trimmed .txt)
  chunker.py
  indexer.py            (persistent vector store, not rebuilt-in-memory FAISS)

service/              # NEW: serving layer
  api.py                (FastAPI: upload documents, query endpoint)
  retriever.py          (persistent retrieval)
  generator.py

app/                  # UPGRADE: from demo to real frontend
  main.py               (document management, upload, results display)

src/explainability/   # NEW: the "trust" the brief explicitly asks for
  citations.py          (attribute answers to source: which page / chunk)
```

**Key differences from Route A:**
- Vector store must be **persistent** (saved to disk, incrementally
  updatable) — not rebuilt on every query.
- Must handle **real document formats** (PDF / Word), not just trimmed `.txt`.
- Must include **explainability** — show which chunk / page an answer came
  from (the "trust in retrieved outputs" named in the brief).
- **Evaluation is demoted** to a quality gate / regression test rather than
  the main deliverable.

**Deliverables:** working retrieval service + frontend + explainability +
a small internal evaluation module for quality assurance.

**Pros:** directly matches the "build intelligent retrieval systems" framing;
higher perceived product value to IBM.
**Cons:** much larger engineering surface; harder to finish to a high standard
in the available time; risk of a shallow result if over-scoped.

---

## Route C — "Both" (highest risk)

If IBM wants both, the danger is **delivering two half-finished things.**

**Recommendation: insist on a primary vs. secondary split.** For example:
- *Evaluation-primary:* "an evaluation study, plus a demo that showcases the
  findings," or
- *System-primary:* "a retrieval system with a built-in lightweight evaluation
  module for quality assurance."

A single project can only have one centre of gravity.

---

## Recommendation for the Meeting

1. **Do not restructure the repo before the meeting.** The current
   evaluation-centric layout is safe for both routes:
   - If evaluation wins → harden it (Route A).
   - If system wins → the evaluation parts become the system's quality-assurance
     module (no wasted work).
2. **Ask IBM to commit to a primary direction** (see
   `meeting-1-questions.md`, Q1).
3. **Clarify the priority of "explainability & trust"** — it is light in the
   current proposal but named explicitly in the brief, and it is a major driver
   of Route B effort.

Deciding direction first is worth more than designing additional experiments.
