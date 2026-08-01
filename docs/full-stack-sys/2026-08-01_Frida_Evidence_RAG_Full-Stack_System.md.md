# Evidence RAG — Local Demo

## 1. Tech Stack

| Layer | Technology |
| :--- | :--- |
| Backend | Python 3.11 + FastAPI + SSE streaming |
| Frontend | Next.js 14 (static export) + Tailwind CSS + TypeScript |
| Package manager | uv (Python) / npm (Frontend) |
| Models | Granite 3B (HF Transformers), Ollama (GGUF), MiniCheck (NLI) |
| Storage | localStorage (chat history), file uploads to /tmp |
| Deploy | Single command: `python -m evidence_rag.ui.server` serves both API and static frontend |

## 2. Features

### Settings Panel

Click the top-right settings button to switch pipeline components in real time.

- **Retriever (8 options)** — BM25, Strong BM25, Granite Dense, Hybrid RRF, Hybrid Convex, Query2Doc, HyDE, Decompose

- **Selector (4 options)** — Top-K, Corroboration, Gated Corroboration, Gated Coverage

- **Generator (2 options)** — Granite 3B, Ollama

- **Verifier (7 options)** — Off, MiniCheck, DeBERTa Base, DeBERTa Large, TRUE 11B, Granite 3B Judge, Granite 8B Judge

Switching settings rebuilds the pipeline and clears the current conversation.

### Chat History

Left sidebar lists all past conversations. Conversations persist in localStorage across page refreshes.

- New Chat button creates a fresh conversation

- Click any conversation to restore its full message history + evidence context

- Hover and click trash icon to delete

- Sidebar can be collapsed to reclaim screen space

### Evidence Panel

Right sidebar shows all 20 candidate passages retrieved by the current query.

- Each passage card shows retrieval rank, source, text preview, and BM25 score

- Green checkmark = selected by Selector into the top-10 evidence set

- Blue number badge = cited by the Generator in the final answer

- Clicking `[1]` `[2]` citation markers in the answer auto-scrolls the panel to highlight the corresponding passage

- Panel can be collapsed

- Evidence context is saved in history and restored when switching conversations

### File Upload

Upload `.txt`, `.md`, or `.pdf` files to extend the search corpus on the fly.

- Paperclip button next to the chat input

- Uploaded files appear as tags above the input, deletable with X

- The BM25 index is rebuilt to include uploaded documents alongside the base corpus

### Stop Generation

During generation, the send button becomes a red stop button. Clicking it aborts the SSE stream. The partial answer generated so far is retained.

## 3. Local Limitations

Some Settings options require GPU, large models, or specific data not available on a local Mac.

### Retriever

| Option | Supported | Reason |
| :--- | :--- | :--- |
| BM25 | Yes | Pure CPU |
| Strong BM25 | Yes | Pure CPU |
| Granite Dense | Yes | Requires granite-embedding-r2 (~500MB), works locally |
| Hybrid RRF | Yes | BM25 + Dense, rank fusion, works locally |
| Hybrid Convex | Yes | BM25 + Dense, score fusion, works locally |
| Query2Doc | Yes | Uses LLM; reuses the Generator model |
| HyDE | Yes | Uses LLM; reuses the Generator model |
| Decompose | Yes | Uses LLM; reuses the Generator model |

### Selector

| Option | Supported | Reason |
| :--- | :--- | :--- |
| Top-K | Yes | Pure ranking |
| Corroboration | Yes | Uses LLM for answer extraction |
| Gated Corroboration | Runs, cannot evaluate | Gate logic works on any corpus, but NIAH provenance data (harm labels, counterfactuals) is required to measure harm/recall impact |
| Gated Coverage | Runs, cannot evaluate | Same as above |

### Generator

| Option | Supported | Reason |
| :--- | :--- | :--- |
| Granite 3B | Yes | ~6GB, runs on M2 with MPS |
| Ollama | Yes | GGUF-quantized, 5-10x faster than HF. Run `ollama pull` first |

### Verifier

| Option | Supported | Reason |
| :--- | :--- | :--- |
| Off | Yes | No model needed |
| MiniCheck | Yes | CPU, lightweight |
| DeBERTa Base | Yes | CPU, ~180MB download |
| DeBERTa Large | Yes | CPU, ~400MB download |
| TRUE (11B) | No | Requires ~21GB GPU VRAM |
| Granite 3B Judge | Yes | Reuses Generator model; G1 found FP rate .067 |
| Granite 8B Judge | No | 8B model too large for M2; G1 found FP rate .240 |

## 4. Launch
```bash

cd web && npm run build && cd ..

EVIDENCE_RAG_CORPUS=runs/reference-baseline uv run python -m evidence_rag.ui.server

```

Open `http://127.0.0.1:8000`.

Set `EVIDENCE_RAG_CORPUS` to any run directory containing `index/corpus_snapshot.json` or `documents.jsonl`.

## 5. Test Datasets and Questions

### CI Fixture (`runs/reference-baseline`)

3 synthetic documents about a fictional company. Used for pipeline smoke testing.

| Attribute | Value |
| :--- | :--- |
| Prepare | `uv run evidence-rag-experiment --config configs/experiments/reference_baseline.toml prepare` |
| Size | 3 documents, 3 chunks, 2 queries |
| Domain | synthetic / small |

Sample questions:

| Query ID | Question |
| :--- | :--- |
| q-solar | What share of factory electricity do the solar panels supply? |
| q-water | How much did Northstar reduce annual water use? |

### SciFact (`runs/scifact-reference`)

Scientific claim verification. Each query is a biomedical claim; the retriever searches 5,183 paper abstracts.

| Attribute | Value |
| :--- | :--- |
| Prepare (raw) | `uv run evidence-rag-materialize-benchmark scifact --split test --output data/benchmarks/scifact/test` |
| Prepare (index) | `uv run evidence-rag-experiment --config configs/experiments/scifact_reference.toml prepare` |
| Size | 5,183 documents, 300 queries |
| Domain | biomedical literature |

Sample questions:

| Query ID | Claim |
| :--- | :--- |
| 1 | 0-dimensional biomaterials show inductive properties. |
| 100 | All hematopoietic stem cells segregate their chromosomes randomly. |
| 1012 | Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume. |
| 1014 | Rapamycin decreases the concentration of triacylglycerols in fruit flies. |
| 1019 | Rapid phosphotransfer rates govern fidelity in two component systems. |

### NQ — Natural Questions (`runs/nq-reference`)

Open-domain QA from real Google queries (DPR-W100 corpus, subsampled to 100K docs).

| Attribute | Value |
| :--- | :--- |
| Size | 100,000 documents, 2,000 queries |
| Domain | Wikipedia, general knowledge |

Sample questions:

| Example question |
| :--- |
| What is the capital of France? |
| Who wrote the Harry Potter series? |
| When was the iPhone first released? |

### 2Wiki (`runs/2wiki-reference`)

Multi-hop reasoning from two Wikipedia articles (subsampled to 100K docs).

| Attribute | Value |
| :--- | :--- |
| Size | 100,000 documents, 2,000 queries |
| Domain | Wikipedia, multi-hop |

Sample questions:

| Example question |
| :--- |
| Who was the director of the film that won Best Picture in 1994? |
| Which country does the river that flows through Paris originate in? |

### File Upload

Upload `.txt`, `.md`, or `.pdf` via the paperclip button. The file is parsed, chunked, and merged into the active corpus on the fly — no pre-processing needed.

## 6. Liveplay

### Tested Datasets

| Dataset | Queries | Answer Length | Status |
| :--- | :--- | :--- | :--- |
| CI Fixture | 2 | 1 sentence | Runs |
| SciFact | 300 | 2-4 sentences | Runs |
| File Upload | Unlimited | Depends on document | Runs |
| NQ / 2Wiki | 2,000 each | 1-3 sentences | Not tested (corpus too large for M2) |

### Latency Estimates (M2, 16GB RAM)

Retrieval is instant (<50ms). Generation dominates.

| Settings | Cold Start | Per Query |
| :--- | :--- | :--- |
| BM25 + TopK + Granite 3B | 30-60s (model load) | 20-60s |
| BM25 + TopK + Ollama | 5-10s | 3-10s |
| BM25 + TopK + Granite 3B + MiniCheck | 30-60s | 40-90s |
| Granite Dense / Hybrid | +15s first load | +2-5s |
| Query2Doc / HyDE / Decompose | — (reuses LLM) | +10-30s (extra LLM calls) |
| Corroboration Selector | — | +5-15s (answer extraction) |

Cold start = first request after server launch. Subsequent queries reuse the loaded model.

### Device Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| RAM | 16 GB | 24 GB+ |
| Disk | 15 GB free | 30 GB+ |
| GPU | MPS (Apple Silicon) | M2 or better |
| Node.js | 18+ | 20+ |
| Python | 3.11 | 3.11 |

Granite 3B uses ~6 GB. With MiniCheck, peak memory is ~8 GB. TRUE 11B (~21 GB) will not fit.

### Known Performance Issues

**Model loading stalls the first request.**

The pipeline is built lazily on the first chat message. Granite 3B triggers a 30-60s download + load. Subsequent queries reuse the cached model. Recommendation: send a dummy query after server launch to warm up.

**Uploading files rebuilds the pipeline.**

Each upload invalidates the current pipeline. The next query reloads the model (30-60s). Recommendation: upload all files before asking questions.

**VerifiedGenerator is not streaming.**

With any Verifier enabled, the answer appears all at once after draft, verify, and repair complete. The "generating" status bar stays for 40-90s with no intermediate output.

**Dense retrievers load an extra model.**

Switching to Granite Dense or Hybrid loads `granite-embedding-r2` (~500MB, 10-15s first time). It stays resident alongside the Generator.

**TRUE 11B and Granite 8B will not run on M2.**

Both exceed available unified memory. Use MiniCheck or DeBERTa for local verification.

**Switching Generator reloads the model.**

Toggling between Granite and Ollama invalidates the pipeline and incurs a cold-start penalty.

**Large corpora may swap.**

SciFact is ~50 MB. NQ/2Wiki at 100K-subsampled are ~200 MB. Full corpora (10-20 GB) may cause swapping on 16 GB machines.

### Questions the System Cannot Answer

The system retrieves evidence only from the active corpus (base dataset + uploaded files). It has no internet access, no general knowledge beyond what the LLM memorized, and no access to local files unless uploaded.

| Scenario | Behavior | Fix |
| :--- | :--- | :--- |
| Topic not in corpus (e.g. "capital of France" on SciFact) | "I don't know" or uncited hallucination | Switch corpus or upload relevant documents |
| Local .md file on disk, not uploaded | Not indexed; empty result | Upload via the paperclip button |
| Custom question outside dataset domain | No relevant passages found | Upload your own documents |
| Multi-hop on single-document corpus | Partial answer at best | Use 2Wiki or upload documents covering both facts |
| Non-English on English corpus | BM25 tokenization mismatch | Use a multilingual corpus |

**CI Fixture limitation.**

Three documents about a fictional company. Any question outside these three topics produces empty results. This is by design for pipeline smoke testing.

**SciFact limitation.**

All 5,183 documents are biomedical paper abstracts. General-knowledge questions find no evidence. Ask biomedical claims or upload your own documents.

**File upload limitation.**

Only `.txt`, `.md`, and `.pdf` are supported. PDF parsing needs `docling`. Files are chunked at 120 words with 20-word overlap. Files >10 MB may cause memory pressure during indexing.
