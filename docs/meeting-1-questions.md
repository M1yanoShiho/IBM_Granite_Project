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

### Q1. What does "needle in a haystack" mean here — general retrieval, or deep-buried rare facts?

The project name and the brief pull in two different directions, and they imply
different datasets and experiments:

- **Interpretation A — precise *semantic retrieval*.** Find the relevant
  passage even when it shares few keywords with the query. Measured on standard
  IR benchmarks (BEIR / MS MARCO / NQ) with precision/recall. This matches the
  brief's line *"precision/recall metrics on benchmark datasets."*
- **Interpretation B — *rare fact buried deep* in a long document.** A single
  injected "needle" at a controlled depth in a long context (the NIAH /
  *Lost-in-the-Middle* setup). This matches the literal project name but does
  **not** produce standard precision/recall (only injection-accuracy).

**Our current plan:** A is primary (standard benchmarks + baselines), B is a
secondary diagnostic. We think this is the safest reading of the brief.
**What we need:** confirmation that IBM agrees A is the headline, or a steer
toward B if they specifically mean long-context retrieval.

### Q2. Is the deliverable a *system*, an *evaluation study*, or both — and which is primary?

The brief asks to "design **and** evaluate." `docs/scope-options.md` lays out
Route A (evaluation-emphasis), Route B (system-emphasis), and Route C (both).
A single project can only have one centre of gravity; trying to do both equally
risks two half-finished things.

**Our current plan:** a **Granite-powered retrieval system** (the deliverable)
**plus** a rigorous benchmark evaluation against baselines (the evidence) —
i.e. system-primary with evaluation as a first-class, not bolted-on, component.
**What we need:** IBM to confirm system-primary, or tell us they want the
evaluation study to be the headline (which would demote the demo/UI work).

### Q3. watsonx.ai access — accounts, project, and student credit/quota?

Everything downstream needs working Granite access.

- Will IBM provision a **watsonx.ai project** for the team, or do we self-serve
  on a trial/lite plan?
- What is the **token/compute quota** per student?
- Any preferred **region endpoint** (`WATSONX_URL`)?

**Compute note:** we expect to have access to the **university HPC cluster**.
Because the `granite-embedding` models are open and small, we can **self-host
the embedding/indexing on the HPC**, so watsonx quota would mainly bound the
**RAG generation** calls (one per query) rather than corpus indexing (one per
passage). This substantially de-risks the quota question — but see Q4 on whether
IBM wants embeddings served via the watsonx API regardless.

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
- Its **max input length** / embedding dimension — this sets our chunk size and
  bounds the secondary long-context test.
- Which **Granite generative model** to use for the RAG layer.
- **Serving question (HPC):** since these models are open, do you want the
  embeddings served via the **watsonx API** (to showcase the platform), or are
  you happy for us to **self-host `granite-embedding` on the university HPC** for
  the heavy indexing and reserve watsonx for the generation layer? Either way
  it's the *same Granite weights* — the result is comparable — but it changes the
  quota story in Q3.

### Q5. Benchmark dataset scope — which datasets, how large?

With **university HPC** for self-hosted indexing (see Q3), the old
compute-budget ceiling is largely lifted: full **MS MARCO / NQ** become
feasible, not just toy subsets.

**Our current plan:** start on **small BEIR subsets** (scifact / nfcorpus / fiqa)
to get the pipeline end-to-end quickly, then **scale to MS MARCO / NQ on the HPC**
for the headline results. **What we need:** which dataset(s) IBM most wants the
comparison reported on — a standard academic benchmark, or an enterprise-style
corpus closer to their real use case?

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

> "We plan a **Granite-embedding–powered retrieval system**, evaluated on
> **small BEIR benchmarks** against **BM25 + an open-source dense retriever**
> using precision/recall/nDCG/MRR, with **citations** for trust and a
> **long-context needle test** as a secondary diagnostic. Does that match what
> you want as the *primary* deliverable, and can you confirm Q3/Q4 access?"
