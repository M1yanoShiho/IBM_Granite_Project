# Meeting 1 — Questions to Settle with IBM

The goal of this meeting is to **lock the project's centre of gravity** before
committing engineering effort. The brief is deliberately broad ("design and
evaluate AI-driven systems"), and a few decisions below fork the entire
repository structure, evaluation plan, and time plan. Questions are ordered by
how much they block downstream work, not by importance to the final report.

Legend: **P0** = blocks the next two weeks of work · **P1** = shapes the
deliverable but has a safe default · **P2** = logistics, can be resolved offline.

---

## P0 — Decisions that gate everything

### Q1. What's the headline — and we want to elevate RAG to sit alongside retrieval (A+B).

The brief is **broader than our original proposal scoped it.** The Approach
lists, *co-equally*, "semantic search **and** retrieval-augmented generation
(RAG) pipelines," evaluated with "precision/recall on benchmark datasets," over
"enterprise-scale data." Our first proposal narrowed this to *retrieval-primary,
RAG/NIAH secondary* for tractability — that was **our choice, not a direct
readout of the brief.**

**Our updated position:** to stay faithful to the brief, we want to **promote RAG
from a secondary item to a co-headline — i.e. A+B (retrieval *and* RAG)** —
rather than keeping it demoted. We'd like IBM to confirm this is the right reading
before we commit the engineering effort (the RAG generation layer needs GPU).

Three candidate headlines:

- **A — semantic *retrieval* quality.** Granite dense retriever vs BM25 vs an
  open-source dense baseline; precision/recall/nDCG/MRR on benchmark datasets.
  Clean, CPU-friendly. (The foundation either way.)
- **B — retrieval + *RAG*.** Add the generation layer (retrieve-then-generate),
  scored on answer quality + faithfulness. The brief names RAG **co-equally**
  with semantic search, so this has direct textual support. Needs a GPU.
- **C — long-context "needle" (NIAH).** Inject a fact at a controlled depth in a
  long document and test whether the model finds it (Lost-in-the-Middle). This is
  the *team's added diagnostic*; the brief's "needle" actually means general
  information discovery, not the synthetic-injection benchmark. We keep this as a
  secondary diagnostic, not a headline. **Finding:** Granite 4.1 8B showed **no
  degradation at 32k context** in our test (an earlier "failure" was a config
  artifact on our side), so RAG's justification rests on **cost and documents
  longer than the context window**, not on the model breaking down.

**What we need:**
- IBM to **confirm A+B (retrieval + RAG) as the headline**, in line with the
  brief naming the two co-equally. If IBM would rather we keep RAG secondary and
  stay retrieval-only (A), we need to hear that explicitly.
- **Does IBM want the long-context needle test (C) as a graded deliverable, or is
  keeping it a secondary diagnostic fine?** Our recommendation is **secondary** —
  Granite 4.1 8B showed no degradation at 32k (see option C above), so the needle
  test isn't surfacing a real failure. **But if you want C elevated,** we'd need
  to test much longer contexts (64k/128k+) to find a genuine break point, which
  costs more GPU and time — we'd want that traded against the RAG work, not added
  on top.

Our headline recommendation, read straight off the brief, is **A+B**.

### Q2. Is the deliverable a *system*, an *evaluation study*, or both — and which is primary?

The brief asks to "design **and** evaluate." `docs/scope-options.md` lays out
Route A (evaluation-emphasis), Route B (system-emphasis), and Route C (both).
A single project can only have one centre of gravity; trying to do both equally
risks two half-finished things.

**Our current plan:** a **Granite-powered retrieval-plus-RAG system** (the
deliverable) **plus** a rigorous evaluation — benchmark IR metrics for the
retrieval layer and answer-quality/faithfulness for the RAG layer (the evidence)
— i.e. system-primary with evaluation as a first-class, not bolted-on, component.
Per Q1, the "system" now **includes the RAG generation layer** (the A+B reading),
not retrieval alone.
**What we need:** IBM to confirm system-primary, or tell us they want the
evaluation study to be the headline (which would demote the demo/UI work).

*Note:* the **scope of the "system" depends on Q1** — retrieval-only (A) is a
smaller system than retrieval + RAG generation (A+B, our recommendation). Settle
Q1 first.

### Q3. watsonx.ai access — accounts, project, and student credit/quota? — ✅ RESOLVED

> **Answered by IBM (Bharat, email):** there is **no watsonx.ai API access** for
> university projects. The project runs the **open-source Granite models
> (Apache 2.0), self-hosted from Hugging Face**, locally and on the university
> HPC. No API keys, quotas, or region endpoints apply.
>
> Consequences already actioned: `src/llm_client.py` pivoted to local
> `transformers` inference; `WATSONX_*` env vars removed from `.env.example`;
> `langchain-ibm` dropped from `requirements.txt`. The serving sub-question in Q4
> (watsonx API vs. self-host) is therefore moot — **self-host is the only path.**
>
> Remaining compute ask: **GPU nodes on the HPC** for the generation layer
> (Granite 4.1 8B/30B) and long-context experiments; the embedding/retrieval +
> BM25 + metrics pipeline is CPU-friendly and can start without HPC.

---

## P1 — Shapes the deliverable (we have a safe default)

### Q4. Granite's role: confirm `granite-embedding` for the dense retriever.

We have committed in the proposal to using **Granite embeddings to power the
dense retriever itself** (not just Granite-as-generator), because otherwise
"Granite-powered *retrieval* system" and the Granite-vs-baseline hypothesis are
not really being tested.

**What we need:**
- Which `granite-embedding` model to use (e.g.
  `granite-embedding-278m-multilingual`, `granite-embedding-30m-english`)?
- Its **max input length** / embedding dimension — this sets our chunk size, so
  we are **blocked on this detail** before finalising ingestion.
- Which **Granite generative model** for the RAG layer — Bharat suggested
  Granite 4.1 (3B for dev, 8B/30B on the HPC; 8B base offers a ~512K context).
  Confirm the recommended size given our HPC GPU budget.
- ~~Serving question (watsonx API vs. self-host)~~ — **moot:** Q3 confirmed
  there is no watsonx API, so everything is self-hosted from Hugging Face.

### Q5. Which data — academic benchmarks, enterprise documents, or both?

The brief pulls two ways: it says **"enterprise-scale data"** (legal discovery,
fraud, healthcare, compliance) *and* **"precision/recall on benchmark datasets."**
These pull apart — clean, ready-made precision/recall comes from academic IR
benchmarks; enterprise data is more faithful to the brief but ships with no
relevance labels.

**A+B raises the bar on data.** With RAG now a co-headline (Q1), the dataset has
to serve **two** scoring needs at once:
- **Retrieval** scoring needs **relevance labels (qrels)**.
- **RAG answer-quality** scoring needs **gold free-text answers** — which qrels
  do not provide. (We've added an optional `answers` field to `BenchmarkData`
  for this.)

This matters because **SciFact has qrels but no gold answers** — fine for
retrieval, not enough to score RAG. So A+B pushes us toward answer-bearing sets.

- **A — standard academic IR** (BEIR/SciFact → MS MARCO/NQ). SciFact has qrels
  only; **Natural Questions / MS MARCO QA also carry gold answers**, so they can
  score both retrieval *and* RAG. CPU-friendly for retrieval. Clean but not
  "enterprise."
- **B — enterprise datasets** (e.g. **CUAD** legal-contract clauses, **DocFinQA**
  long SEC financial QA). Much closer to the brief's enterprise framing; DocFinQA
  carries answers (good for RAG), but they are QA/extraction sets — we'd have to
  **construct corpus/queries/relevance ourselves** for retrieval metrics, and the
  long docs need a GPU.
- **C — A for rigorous metrics + B as an enterprise case study.**

**What we need:** which the comparison should be **reported on**, knowing A+B
needs **gold answers**, not just qrels. The brief's "enterprise-scale data" leans
**B**; its "precision/recall on benchmark datasets" line leans **A** — only IBM
can resolve which they actually want. We start on SciFact regardless (fast
pipeline bring-up for retrieval); the headline-dataset decision is the ask.

### Q6. Baseline dense retriever — which open-source model?

The hypothesis is "Granite > BM25 **and** > an open-source dense retriever."

**Our current plan:** BM25 (`rank_bm25`) + a `sentence-transformers` dense
retriever (DPR-style) as the two baselines. **What we need:** IBM okay with
this, or do they want a specific competitor model named.

### Q7. How much does "explainability & trust" matter for the grade/deliverable?

The brief names *"explainability and trust in retrieved outputs"* explicitly,
but it is light in the current proposal (a citations layer attributing answers
to source chunks). It is also a major driver of system-side effort.

**What we need:** is source-attribution/citations sufficient, or does IBM expect
something deeper (confidence calibration, faithfulness scoring, etc.)?

---

## P2 — Logistics (can resolve offline)

### Q8. Final deliverable format and audience.

- Report + code + live demo? Or report + reproducible benchmark tables only?
- Is there an IBM-side stakeholder/demo at the end we should design the UI for?

### Q9. Cadence and contact.

- Meeting frequency, preferred channel, and who is the day-to-day technical
  contact vs. the academic supervisor.

### Q10. IP / licensing and data handling.

- Any constraints on open-sourcing the repo, the choice of license, or handling
  of any IBM-provided data.

---

## One-line ask for the meeting

> "We plan a **Granite-embedding–powered retrieval system with a RAG layer on
> top** — read straight off the brief, which names semantic search and RAG
> co-equally. Retrieval is evaluated on **small BEIR benchmarks** against **BM25
> + an open-source dense retriever** (precision/recall/nDCG/MRR); RAG is scored
> on answer quality + faithfulness, with **citations** for trust; the
> **long-context needle test** stays a secondary diagnostic. Does **A+B
> (retrieval + RAG)** match what you want as the *primary* deliverable, and can
> you confirm the Granite model choices in Q4?"
