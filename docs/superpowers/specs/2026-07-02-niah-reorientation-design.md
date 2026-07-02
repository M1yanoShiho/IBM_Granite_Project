# Design — NIAH reorientation: from embedder bake-off to a needle-in-a-haystack retrieval system

- Date: 2026-07-02
- Branch context: `week4_SPLADE`
- Status: design approved (user confirmed) → spec self-review → writing-plans
- Owner: P6 (Weikai Mao), driving a project-level re-aim; **pending supervisor (Bharat) alignment** (Phase 0)
- Re-aims (keeps, does not discard): [convex-hybrid](2026-06-29-convex-hybrid-fusion-design.md), [SPLADE](2026-06-30-splade-sparse-arm-design.md) — those components are kept and pointed at the NIAH regime (see §4)

## 1. Background / problem — the drift

The IBM brief is **"Needle in a Haystack"**: build a system that finds *rare*, relevant information in *massive*, *misleading* data ("terabytes of ... hay"; legal discovery / fraud / intelligence / compliance). The project instead matured into a rigorous but **drifted embedder bake-off** (Granite vs gte/bge/e5 on SciFact 5k / NFCorpus 3.6k / FiQA 57k, and a ~1M NQ/TriviaQA subset). Its central result is a *defensible null*: Granite ≈ modern peers, significant only on the harder FiQA (~5 nDCG pts). The brief's own keywords — **massive, rare, misleading** — are exactly the parts left as "pending / skeleton" (21M HNSW scale run; NIAH-vs-long-context).

Two conflations, both settled with the user:

- **Deliverable vs benchmark.** The deliverable is a **system** (Granite retrieve→rerank→generate + citations, over a scalable index). "NIAH" is the **evaluation lens** that proves the system finds needles — *not* the deliverable (§3).
- **"NIAH" the problem vs "NIAH" the benchmark.** The brief's title = the *problem class* ("find rare items in massive misleading data"). The specific long-context "insert-a-fact" NIAH benchmark is *one supporting eval*, not the whole project.

This spec re-aims the project to the mandate: the rigorous comparison is **kept but demoted** to "why Granite, justified"; the **scale / rarity / distractor / trust** axes are **promoted** to the headline.

## 2. Goal & the rewritten headline

**Goal.** Re-aim the existing components around the needle-in-a-haystack problem, with the **retrieval core (scale + rare-needle recall) as the star**, delivered as a *compact Granite system* evidenced by NIAH-style evaluation.

**Title (report / IBM):**
> Finding the Needle — efficient, citable retrieval-augmented discovery of rare evidence in large, noisy corpora with IBM Granite.

**Thesis / research question:**
> How reliably can a compact IBM Granite retrieve → rerank → generate system surface *rare*, relevant evidence as the haystack scales and fills with *misleading* near-duplicates — and how much of that reliability comes *efficiently* and with *citation-level trust*, rather than from brute-force long-context reading?

**Claims to be evidenced** (nothing claimed without a phase deliverable behind it):

| # | Claim | Axis |
|---|---|---|
| C1 | Needle recall/precision degrades **gracefully** as corpus size grows (the degradation curve is the headline figure). | Scale — **star** |
| C2 | The lexical (SPLADE/BM25) arm and reranking earn their keep **specifically** under rarity + misleading distractors — flipping the prior "hybrid/rerank don't help" nulls into a *when-they-help* finding. | Rarity / distractors |
| C3 | Beyond a **crossover** corpus size, retrieval beats stuffing the haystack into Granite long-context at equal compute. | RAG vs long-context |
| C4 | Found needles are **attributable** (citations); the system re-searches / abstains when not confident. | Trust |
| C5 | All delivered by a **compact** Granite stack (granite-small 47M competitive) — the enterprise value proposition. | Efficiency |

Scale wording ("large" / "millions" / "21M") is **provisional** until the Phase-0 probe (§6.0).

## 3. Deliverable vs evaluation (the system, and how NIAH validates it)

Deliverable = the **system** (product). Evaluation (NIAH stress tests, precision/recall) = the **evidence**. The existing eval harness (`run_benchmark` / `run_rag` / `significance`) is the *measurement*, not the product.

| System capability (build — deliverable) | NIAH stress test (evaluate — evidence) |
|---|---|
| Scalable index (HNSW) | recall vs corpus size 5k → … → ceiling (degradation curve) |
| Rare-term needle (hybrid/SPLADE arm) | recall when the needle is a rare exact term |
| Distractor robustness (reranker) | recall after injecting misleading near-duplicates |
| Trust / attribution (citations) | is the found needle correctly cited |
| Adaptivity (corrective re-retrieval) | gain from re-search when confidence is low |
| Retrieve-vs-read decision | NIAH long-context crossover |

## 4. What changes (re-aim of existing work — nothing discarded)

- **Demote → foundation.** Embedder bake-off + rigor machinery ⇒ "why Granite, rigorously" (a design/justification section). The *difficulty-dependence* finding motivates why the hard NIAH regime is the interesting one.
- **Re-aim.** Hybrid/SPLADE, reranking (`Reranker` + `LLMListwiseReranker`), corrective RAG, HyDE/Query2Doc ⇒ evaluated **under NIAH stress** (rarity/distractors/scale), where they can show signal — not on saturated BEIR.
- **Promote.** HNSW/scale (was "pending") ⇒ **core**; NIAH-vs-long-context (was skeleton) ⇒ supporting; citations ⇒ the **trust** pillar.
- **New.** The NIAH task construction (§5); the scale-feasibility probe (§6.0); packaging the eval scripts into a **demo-able system** (Phase 4).

## 5. NIAH task construction (the make-or-break design — Phase 0)

The project's validity rests on how **needle / haystack / misleading hay** are operationalised on **public data** (Meeting-1 lock). Phase 0 fixes, with the supervisor:

- **Needle.** What is the rare relevant item — queries with ≤1 gold doc (SciFact-like), a synthetically inserted fact (long-context NIAH), and/or a rare-entity / rare-term query subset.
- **Haystack.** The corpus and how it scales (subsets of `dpr-w100` up to the probed ceiling) + the distractor pool.
- **Misleading hay.** How hard negatives / near-duplicates are sourced or generated (BM25/dense hard negatives; near-duplicate perturbations; topically-adjacent distractors) and injected at a controlled ratio.
- **Metrics.** recall@k / precision@k / nDCG on the needle; "needle-found" rate; `context_precision`; citation-correctness; abstention correctness. Reuse `eval/significance.py`.

**Phase-0 output:** a written task definition + a small builder (query/distractor construction) with a **hardness gate** — a check that baselines do *not* saturate the constructed task (else it measures nothing).

## 6. Work plan (phases; backward from 2026-09-03, ~9 weeks)

Phases can partly parallelize across the team; each lists its key output. Detailed task breakdown → writing-plans.

- **6.0 Phase 0 — Align, probe, define (Week 1).** (a) supervisor alignment on the re-anchor (changes the Meeting-1 headline); (b) scale-feasibility probe — HNSW build time + memory at 1M / 5M / 21M → the honest scale ceiling; (c) the §5 NIAH task definition + builder + hardness gate.
- **6.1 Phase 1 — Scale axis / star (Weeks 2–3).** Scalable index at the ceiling → recall/precision **degradation curve** vs corpus size + rare-needle (low-prevalence) recall. Headline figure.
- **6.2 Phase 2 — Rarity + distractors (Weeks 4–5).** Inject misleading near-duplicates; measure **reranking** + **lexical/SPLADE** under distractor pressure. Most likely flips the earlier nulls.
- **6.3 Phase 3 — Trust + adaptivity + RAG-vs-long-context (Weeks 6–7).** Citation/attribution on found needles; **calibrate corrective re-retrieval** (fix the degenerate confidence threshold, §9); retrieve-vs-stuff **crossover** study.
- **6.4 Phase 4 — System packaging + efficiency (Week 8).** Eval scripts → deliverable **system** (app + demo); efficiency numbers (compact Granite, latency/memory at scale).
- **6.5 Phase 5 — Report + buffer (Week 9).** Write-up, figures, slippage buffer.

## 7. Constraints / locked decisions

| Constraint | Value |
|---|---|
| Deadline | 2026-09-03 (~9 weeks) |
| Star capability | scale + rare-needle recall |
| Scale ceiling | **TBD — Phase-0 probe** (scale claim provisional until then) |
| Delivered system | stays **IBM Granite** (Apache-2.0); SPLADE etc. are components/baselines |
| Data | **public only** (Meeting-1 lock) |
| Report | own-work rule — this spec/code is scaffolding; the report is written by the team |
| Rigor | significance testing kept throughout (paired randomization + bootstrap) |

## 8. Success criteria (definition of done)

- Each claim C1–C5 has a corresponding **evaluated** result, with significance where applicable; provisional/negative results reported honestly.
- The NIAH task is **demonstrably non-saturated** (baselines separate) — the Phase-0 hardness gate; otherwise the project measures nothing.
- A packaged, **demo-able Granite needle-finding system** exists (not just eval scripts).
- The report **leads with the NIAH headline**; the embedder comparison appears as justification, not thesis.

## 9. Risks / caveats

- **Task-definition risk (highest).** A weak "needle / misleading hay" definition saturates the task and measures nothing — mitigated by the Phase-0 hardness gate.
- **Scale feasibility.** 21M may be infeasible on the allocation → cap honestly at the probed ceiling; don't overclaim "enterprise-scale".
- **Confidence-gate degeneracy.** `CorrectiveRAGPipeline._confidence = (top1−top2)/top1` with threshold 0.5 likely fires on nearly every dense query (cosine top-1/top-2 sit in a narrow band) → gate degenerates to "always rewrite". Phase 3 must recalibrate/validate it or it is dead code.
- **Supervisor gating.** The re-anchor changes the Meeting-1 headline; Phase 0 aligns it before heavy investment.
- **Time.** 9 weeks / 7 people is comfortable, but **Phase 0 is on the critical path** — everything else depends on the task definition.

## 10. Out of scope / no longer claimed

- "Granite outperforms modern peers" as a **headline** (only FiQA retrieval, only vs older peers) — demoted to a bounded sub-finding.
- Graph / agentic RAG, multilingual, multimodal — **out** (upside, not the bet) unless a phase finishes early.
- The full **21M** claim until the Phase-0 probe supports it.
