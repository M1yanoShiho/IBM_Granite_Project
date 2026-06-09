# Needle in a Haystack — Precise Information Discovery with IBM Granite

An engineering + research project that **designs and evaluates a Granite-powered
retrieval system** for finding rare, relevant information in large document
collections ("finding the needle in the haystack"). The system combines
semantic (dense) search and retrieval-augmented generation (RAG), and is
benchmarked against classical and neural baselines on **standard IR datasets**
using **precision/recall** metrics.

## Motivation

Enterprises are drowning in data; the hard part is surfacing the *precise* piece
of information needed at the right moment (legal discovery, compliance,
healthcare, fraud detection). This project builds an intelligent retrieval
system on IBM Granite and rigorously measures whether it finds relevant
information more accurately than established baselines — with explainable,
source-attributed answers.

## Goals

**Primary (the deliverable system + its evaluation)**

- Build a Granite-powered **semantic search + RAG** retrieval system.
- Evaluate it on **standard IR benchmarks** (BEIR / MS MARCO / Natural
  Questions) with **precision@k, recall@k, nDCG, MRR**.
- **Compare against baselines**: classical BM25 and an open-source dense
  retriever, to show where Granite wins.
- Provide **explainability**: attribute each answer to its source chunks.
- Ship a **Streamlit demo** (upload documents, ask questions, see cited answers).

**Secondary (optional stress test)**

- A **long-context "needle" stress test** (the NIAH / *Lost-in-the-Middle*
  analysis) probing how retrieval degrades when relevant text is buried deep in
  a long document. Kept as a complementary diagnostic, not the core evaluation.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── data/                          # corpora / documents (and needle data for the stress test)
├── src/
│   ├── llm_client.py              # IBM Granite + baseline LLM wrapper
│   ├── ingestion/                 # corpus -> chunks -> persistent vector index
│   │   ├── loaders.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── retrieval/                 # the delivered retrieval system + baseline
│   │   ├── embedder.py            # Granite / sentence-transformers embeddings
│   │   ├── retriever.py           # dense (semantic) retriever
│   │   └── bm25_baseline.py       # classical BM25 baseline
│   ├── rag_pipeline.py            # retrieve-then-generate (RAG) layer
│   └── explainability/            # answer provenance / citations ("trust")
│       └── citations.py
├── eval/                          # PRIMARY: benchmark evaluation
│   ├── benchmarks/loader.py       # load BEIR / MS MARCO / NQ (corpus, queries, qrels)
│   ├── ir_metrics.py              # precision@k, recall@k, nDCG, MRR
│   ├── run_benchmark.py           # system vs. baselines on standard datasets
│   ├── niah_runner.py             # SECONDARY: long-context needle stress test
│   └── metrics.py                 # SECONDARY: NIAH accuracy / RAGAS scoring
├── src/data_processing.py         # SECONDARY: NIAH data prep (needle/haystack assembly) for the stress test
├── notebooks/                     # results analysis & figures
├── app/main.py                    # Streamlit demo
└── tests/                         # unit tests
```

## Setup

### 1. Prerequisites

- Python 3.10+
- An IBM watsonx.ai account and project (for Granite API access)
- (Optional) A Hugging Face account for baseline models

### 2. Create a virtual environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
# then edit .env with your real credentials
```

Required variables (see `.env.example`):

- `WATSONX_API_KEY` — IBM watsonx.ai API key
- `WATSONX_PROJECT_ID` — watsonx.ai project ID
- `WATSONX_URL` — watsonx.ai region endpoint
- `HUGGINGFACE_API_KEY` — (optional) for baseline models

## Usage

**Primary — benchmark the retrieval system vs. baselines:**

```bash
python -m eval.run_benchmark --dataset scifact
```

**Launch the demo UI:**

```bash
streamlit run app/main.py
```

**Secondary — long-context needle stress test:**

```bash
python -m eval.niah_runner
```

**Explore and plot results:**

```bash
jupyter notebook notebooks/
```

## Evaluation Metrics

**Primary — retrieval quality on standard benchmarks (vs. BM25 + dense baseline):**

| Metric        | Description                                                          |
| ------------- | ------------------------------------------------------------------- |
| Precision@k   | Of the top-k retrieved passages, how many are relevant.             |
| Recall@k      | Of all relevant passages, how many appear in the top-k.             |
| nDCG@k        | Rank-aware quality — relevant hits near the top count more.         |
| MRR           | Mean reciprocal rank of the first relevant hit.                     |

**Secondary — RAG answer quality (the [RAGAS](https://docs.ragas.io/) framework):**

| Metric            | Description                                                     |
| ----------------- | -------------------------------------------------------------- |
| Context Precision | How much of the retrieved context was actually relevant.       |
| Faithfulness      | Is the answer grounded in the retrieved context (no hallucination). |

**Secondary — long-context stress test:** retrieval accuracy across document
length × needle depth (the *Lost-in-the-Middle* diagnostic).

## Team

| Name   | Role | Contact |
| ------ | ---- | ------- |
| _TBD_  | _TBD_ | _TBD_  |

> Update this table with team members, roles, and contact information.

## License

_TBD — add a license before publishing._
