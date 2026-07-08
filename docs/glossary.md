# Glossary — core terms (source material for the report)

A plain-language reference for every core term in the project, so anyone (teammate,
supervisor, examiner) who did **not** build a given piece can still read the
`results-summary.md` and the report. Each entry: what it is (with the full spelling),
why we use it here, and a source.

**Citation hygiene (Bristol integrity).** Sources are author / year / venue. Papers marked
**✓** were read in full this session (arXiv IDs given). For every other citation, **open the
abstract page and verify the exact arXiv/DOI before it enters the report** — the project rule
(work-plan WS-13). Never copy a citation you have not opened; "author, year" here is a
pointer to verify, not a finished reference.

---

## 1. Problem & task

| Term | What it is | Why we use it | Source |
|---|---|---|---|
| **NIAH — the problem** | "Needle in a Haystack": finding rare, relevant evidence buried in massive, misleading data. | Our **thesis** — the deliverable is a Granite system that finds the needle at scale. | IBM brief; enterprise-search framing |
| **NIAH — the benchmark** | The long-context needle probe: hide one fact deep in a long document, test whether it is still retrieved (a.k.a. *Lost in the Middle*). | **One evaluation lens, not the whole project** — separating this from the problem answers "how does NIAH fit the mainline". | Liu et al., *Lost in the Middle*, TACL 2023 |
| **Needle** | The single designated gold passage that literally contains the answer (the smallest-doc-id gold). | Target of needle-found@k; one clean target avoids multi-gold ambiguity. | our task design (`docs/niah-task-definition.md`) |
| **Haystack** | The background corpus + injected distractors the needle hides in. | The scalable index the system must search. | — |
| **Distractor** | A non-answer passage competing with the needle; three sources — counterfactual (A), generative (B), mined (C). | Controls task hardness; the counterfactual is the adversary the novel method targets. | our task design |
| **Counterfactual distractor / Source-A** | The needle's own text with the answer entity swapped to a same-type **wrong** entity — relevant but factually wrong. | The hardest distractor: a relevance reranker cannot tell it from the needle, because it *is* relevant. | Longpre et al., EMNLP 2021 (entity substitution) — arXiv:2109.05052 **✓** |
| **Knowledge conflict** | When the retrieved passage contradicts the model's learned (parametric) knowledge. | Explains why generation-stage fixes struggle; motivates cross-source signals. | Longpre et al., EMNLP 2021 **✓** |

## 2. Retrieval methods & system

| Term | What it is | Why we use it | Source |
|---|---|---|---|
| **Dense retrieval / bi-encoder** | Embed query and passage separately, rank by vector similarity. | Our system-under-test = the Granite dense embedder. | Karpukhin et al., DPR, EMNLP 2020 |
| **BM25** | Classic bag-of-words lexical ranking (term frequency × rarity). | Lexical baseline and the lexical arm of the hybrid. | Robertson & Zaragoza, 2009 |
| **SPLADE** (SParse Lexical AnD Expansion) | A *learned sparse* retriever — a BERT model predicts weighted term expansions per passage. | A stronger lexical arm than BM25 (~3.5× faster than rank_bm25 at scale). | Formal et al., SIGIR 2021 |
| **Cross-encoder reranker** | Scores a (query, passage) pair *jointly* with full attention — slower but sharper than a bi-encoder. | Second-stage reranking baseline (Granite reranker). | Nogueira & Cho, 2019 |
| **RRF** (Reciprocal Rank Fusion) | Fuse two rankings by summing 1/(k+rank). | The hybrid we first tried — it **failed** (regresses toward the weak arm). | Cormack et al., SIGIR 2009 |
| **Convex fusion / convex hybrid** | Blend two arms' **min-max-normalised scores**: α·dense + (1−α)·lexical. | Fixes RRF's failure; the weight is tuned offline. | Bruch et al., 2023 |
| **α (alpha)** | The convex-blend weight (the dense / relevance weight). | One tunable knob: α=1 → pure dense, α=0 → pure other arm. | — |
| **HNSW / IVFPQ** | Approximate-nearest-neighbour indexes (graph / compressed) for fast search over millions of docs. | Scale past the exact "flat" index (O(N)/query, tens of GB). | Malkov & Yashunin 2018 (HNSW); Jégou et al. 2011 (PQ) |
| **Query2Doc (q2d)** | Ask an LLM to draft a pseudo-answer, append it to the query, then retrieve. | Our best-certified lever (**+0.070** needle-found@10); enriches the *query*. | Wang et al., EMNLP 2023 |
| **HyDE** (Hypothetical Document Embeddings) | Like q2d — generate a hypothetical answer doc and embed *that* to retrieve. | Second query-transform (+0.060); a cross-check for q2d. | Gao et al., ACL 2023 |
| **Listwise reranking / RankGPT** | Show an LLM a whole window of candidates, ask for a re-ordering. | A reranking baseline; null on our task. | Sun et al., EMNLP 2023 |

## 3. The novel method & its neighbours

| Term | What it is | Why we use it | Source |
|---|---|---|---|
| **Corroboration Reranking** | **(our method)** Rank the pool by cross-source answer *agreement*: extract each passage's answer, boost answers other passages / the model's own knowledge also give; a lone counterfactual is corroborated by nobody → demoted. | The one reranking approach that beats the null (**+0.037** @10, p≈0.03) — the separation relevance-reranking cannot make. | ours; roots in self-consistency + Astute |
| **Self-Consistency** | Sample several reasoning paths, take the majority-vote answer. | The conceptual root of corroboration (agreement = correctness), moved from *samples* to *sources*. | Wang et al., ICLR 2023 — arXiv:2203.11171 **✓** |
| **Cascade / lexicographic tie-break** | **(our Exp-A rules)** Harder alternatives to the blend: rank by votes first (cascade), or let votes break only near-relevance ties (lexicographic). | Tested whether a harder combination rule beats the convex blend — it does not (Table 4d). | ours |
| **Astute RAG** | Generation-stage source-aware consolidation (elicit → consolidate → finalise). | We measured it — it **hurt** on NQ (finding 16); a reported negative. | Astute RAG, 2024 (verify) |
| **MADAM-RAG / RAMDocs** | Multi-agent debate over conflicting evidence + a dataset mixing ambiguity, misinformation, noise. | Closest neighbour/competitor to corroboration (generation-stage debate vs our rank-stage vote). | Wang et al., COLM 2025 — arXiv:2504.13079 **✓** |
| **Adaptive-RAG** | Route each query by complexity (no / single / multi-step retrieval) via a classifier. | Basis for our cost-gated cascade — spend the LLM only on hard queries. | Jeong et al., NAACL 2024 — arXiv:2403.14403 **✓** |
| **RGB benchmark** | Tests 4 RAG abilities: noise robustness, negative rejection, information integration, counterfactual robustness. | Vocabulary + external evidence that generation-stage counterfactual handling is genuinely hard. | Chen et al., AAAI 2024 — arXiv:2309.01431 **✓** |

## 4. Metrics

| Term | What it is | Why we use it | Source |
|---|---|---|---|
| **needle-found@k** | **(our primary metric)** Is the designated needle in the top-k? (fraction over queries). | Single-target recall — the headline NIAH number (k=10). | ours (a variant of recall@k) |
| **MRR** (Mean Reciprocal Rank) | Average of 1/(rank of the needle). | Rank-sensitive companion to found@k — shows whether a method lifts the needle *up*, not just across the top-k line. | standard IR |
| **nDCG@k** (normalised Discounted Cumulative Gain) | Rank-aware quality; hits near the top count more. | Retrieval quality on the BEIR bake-off. | Järvelin & Kekäläinen, 2002 |
| **Precision@k / Recall@k** | Of the top-k, how many are relevant / of all relevant, how many made the top-k. | Standard retrieval reporting. | standard IR |
| **cover-EM** (cover Exact Match) | Answer recall: is a gold answer *contained* in the generated answer? | The de-facto metric for generative open-domain QA (robust to verbose answers). | common QA eval |
| **EM / F1** (Exact Match / token-F1) | Strict normalised string match / token overlap vs gold. | Stricter cross-checks on answer quality (SQuAD-style). | Rajpurkar et al., SQuAD, EMNLP 2016 |
| **Context precision / Faithfulness** | How much retrieved context was relevant / whether the answer is grounded in the context. | RAG quality (secondary — faithfulness barely discriminates here). | Es et al., RAGAS, 2023 |
| **Memorization Ratio** | How often the model returns its memorised answer instead of the (conflicting) context. | Explains why a bigger extractor doesn't help — larger models memorise more. | Longpre et al., EMNLP 2021 **✓** |

## 5. Evaluation methodology (the rigor)

| Term | What it is | Why we use it | Source |
|---|---|---|---|
| **Paired randomization test** (sign-flip permutation) | Significance test on per-query score *differences*. | Every claim's p-value: is an edge real or noise? | Smucker, Allan & Carterette, CIKM 2007 |
| **Bootstrap CI** | Percentile confidence interval on the mean per-query difference. | The visual companion to the p-value. | Efron (standard) |
| **Nested cross-validation** | Select hyper-parameters per fold, score every query *out-of-fold*. | Removes "tuning-on-test" optimism — how corroboration was honestly certified. | standard ML |
| **Pre-registration** | Fix the primary metric/analysis *before* seeing results. | Prevents cherry-picking; e.g. the MRR-null was pre-registered. | standard practice |
| **Ranking vs recall headroom** | **(our framing)** Split the remaining error: needle in the pool but buried (ranking) vs not in the pool at all (recall). | Diagnoses that the bottleneck is **ranking (~0.26)** not recall (~0.13) → where to invest. | ours (WS-5 oracle decomposition) |

## 6. Datasets

| Term | What it is | Source |
|---|---|---|
| **BEIR** (SciFact, NFCorpus, FiQA) | A standard heterogeneous IR benchmark suite. | Thakur et al., NeurIPS 2021 Datasets & Benchmarks |
| **dpr-w100 / Natural Questions / TriviaQA** | Wikipedia 100-word passage corpus + two open-domain QA sets (our NIAH + RAG data). | Karpukhin 2020 (dpr-w100); Kwiatkowski 2019 (NQ); Joshi 2017 (TriviaQA) |

---

*Entries marked **✓** were read in full this session. Every other citation is a pointer
(author / year / venue) that MUST be opened and verified before it enters the report —
project citation-hygiene rule (work-plan WS-13).*
