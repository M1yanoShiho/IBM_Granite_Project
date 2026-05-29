# Solving the Needle in a Haystack Problem with IBM Granite

An academic research and engineering project evaluating Large Language Models —
specifically **IBM Granite** (via the watsonx.ai API) alongside open-source
baselines — on their ability to retrieve specific facts ("needles") embedded
deep within long-context documents ("haystacks").

## Motivation

As context windows grow, a critical question is whether models can *actually
use* the full context. The "Needle In A Haystack" (NIAH) test injects a known
fact at a controlled depth inside a long document and asks the model to retrieve
it. Plotting accuracy across **document length** × **needle depth** reveals the
well-documented **"Lost in the Middle"** U-shape: models recall facts near the
start and end of a context far better than facts buried in the middle.

This project builds an **automated evaluation pipeline** to quantify that effect
for IBM Granite, plus a **lightweight Streamlit UI** for interactive demos.

## Goals

- Build a reproducible NIAH evaluation harness.
- Benchmark IBM Granite against open-source baseline models.
- Measure retrieval **accuracy**, **context precision**, and **faithfulness**.
- Visualize results as a "Lost in the Middle" heatmap.
- Ship a simple demo app for stakeholders.

## Project Structure

```
.
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                  # Haystack source documents
│   └── needles/              # Synthetic facts to inject
├── src/
│   ├── data_processing.py    # Load, chunk, and inject needles at set depths
│   ├── llm_client.py         # Wrapper for IBM Granite + baseline model APIs
│   └── rag_pipeline.py       # Basic retrieval-augmented generation setup
├── eval/
│   ├── niah_runner.py        # Main NIAH test loop (length × depth grid)
│   └── metrics.py            # Accuracy, context precision, faithfulness
├── notebooks/
│   └── 01_visualize_heatmap.ipynb   # "Lost in the Middle" heatmap plotting
└── app/
    └── main.py               # Streamlit demo (search box + upload)
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

Run the full NIAH evaluation sweep:

```bash
python -m eval.niah_runner
```

Launch the demo UI:

```bash
streamlit run app/main.py
```

Explore and plot results:

```bash
jupyter notebook notebooks/01_visualize_heatmap.ipynb
```

## Evaluation Metrics

| Metric             | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| Accuracy           | Did the model retrieve the injected needle correctly?             |
| Context Precision  | How much of the retrieved context was actually relevant?          |
| Faithfulness       | Is the answer grounded in the provided context (no hallucination)?|

Metrics are computed with custom logic and the [RAGAS](https://docs.ragas.io/)
framework.

## Team

| Name   | Role | Contact |
| ------ | ---- | ------- |
| _TBD_  | _TBD_ | _TBD_  |

> Update this table with team members, roles, and contact information.

## License

_TBD — add a license before publishing._
