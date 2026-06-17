"""Streamlit demo for the IBM Granite retrieval + RAG system.

A lightweight UI to demonstrate the retrieve-then-generate pipeline (the A+B
deliverable) with IBM Granite. The user can:

- Upload (or paste) a document corpus.
- Type a question (search box).
- Run the query and view the model's answer **plus the source chunks it cited**
  (the ``RAGPipeline`` → ``RAGResult`` flow, with citations for trust).
- Toggle a benchmark comparison chart to see Granite vs. baselines.

Run from the project root:

    streamlit run app/main.py

Currently runs on **mock data** — the retrieval + generation are simulated so
the UI skeleton can be designed and tested before the real model pipeline is
wired in.  Swap ``mock_retrieve`` / ``mock_generate`` for the real
``Retriever`` + ``LLMClient`` once they are ready.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Mock data types (mirror src/retrieval/base.py:RetrievedChunk)
# ---------------------------------------------------------------------------


@dataclass
class MockChunk:
    """Stand-in for ``RetrievedChunk`` so the demo runs without src/ imports."""

    doc_id: str
    text: str
    score: float


# ---------------------------------------------------------------------------
# Mock corpus — a small fake-document bank the mock retriever draws from
# ---------------------------------------------------------------------------

MOCK_CORPUS: List[MockChunk] = [
    MockChunk("doc-1", "The Q4 financial report shows revenue increased by 18% "
               "year-over-year, driven primarily by the new cloud service "
               "line. Operating margins improved to 24%.", 0.0),
    MockChunk("doc-2", "Cloud infrastructure costs decreased 12% after the "
               "migration to containerised workloads on Kubernetes. The team "
               "plans further optimisation in Q1.", 0.0),
    MockChunk("doc-3", "Employee satisfaction survey results: 87% of staff "
               "report being satisfied or very satisfied. The top request was "
               "for flexible working hours and improved remote-work tooling.", 0.0),
    MockChunk("doc-4", "Project Nightingale update — the secret launch code is "
               "ECHO-42. This code must not be shared outside the core team. "
               "The launch window is scheduled for March 15.", 0.0),
    MockChunk("doc-5", "Security audit findings: three medium-severity "
               "vulnerabilities were identified in the authentication service. "
               "Patches are scheduled for the next sprint. No critical issues "
               "were found in the core infrastructure.", 0.0),
    MockChunk("doc-6", "The machine learning pipeline achieved 94.3% accuracy "
               "on the held-out test set. The biggest improvement came from "
               "switching the embedding model to the Granite 8B dense encoder, "
               "which outperformed the previous Sentence-BERT baseline by "
               "6.8 percentage points.", 0.0),
    MockChunk("doc-7", "Customer support ticket volume decreased 30% after "
               "deploying the new self-service knowledge base. The most common "
               "remaining tickets relate to password resets and account "
               "recovery — accounting for 45% of all inbound requests.", 0.0),
    MockChunk("doc-8", "The annual company retreat will be held in Lisbon, "
               "Portugal from June 10–14. Teams should submit their workshop "
               "proposals by May 1. Travel will be arranged through the "
               "approved corporate travel agency.", 0.0),
]

# ---------------------------------------------------------------------------
# Mock retrieval + generation
# ---------------------------------------------------------------------------


def _keyword_overlap(query: str, text: str) -> int:
    """Crude keyword-overlap score — counts shared lowercase words."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    return len(q_words & t_words)


def mock_retrieve(query: str, top_k: int = 4) -> List[MockChunk]:
    """Simulate retrieval by scoring chunks against the query via keyword overlap.

    In production this would be replaced by ``Retriever.retrieve(query)``.
    """
    scored = []
    for chunk in MOCK_CORPUS:
        overlap = _keyword_overlap(query, chunk.text)
        if overlap > 0:
            # Add some noise so scores aren't identical.
            score = round(min(overlap / max(len(query.split()), 1), 1.0)
                          + random.uniform(-0.05, 0.05), 3)
            scored.append(MockChunk(chunk.doc_id, chunk.text, max(0.0, min(1.0, score))))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def mock_generate(question: str, chunks: List[MockChunk]) -> str:
    """Simulate LLM generation with a canned answer that references the chunks.

    In production this would be replaced by ``LLMClient.generate(prompt)``.
    """
    if not chunks:
        return ("I could not find relevant information in the provided "
                "documents to answer this question. Please try rephrasing or "
                "providing a different haystack.")

    # Pick a canned answer style based on question keywords.
    q_lower = question.lower()
    if any(w in q_lower for w in ("secret", "code", "launch", "nightingale")):
        return (
            "Based on the retrieved documents, the secret launch code is "
            "**ECHO-42**. Project Nightingale's launch window is scheduled "
            "for **March 15**. This information was found in an internal "
            "project update marked as confidential.  \n\n"
            "⚠️ _This information must not be shared outside the core team._"
        )
    if any(w in q_lower for w in ("revenue", "financial", "q4", "margin", "profit")):
        return (
            "According to the Q4 financial report, revenue grew **18% "
            "year-over-year**, with operating margins improving to **24%**. "
            "The primary growth driver was the new cloud service line. "
            "Cloud infrastructure costs also decreased 12% following the "
            "Kubernetes migration — a positive signal for continued margin "
            "expansion in Q1."
        )
    if any(w in q_lower for w in ("security", "audit", "vulnerability")):
        return (
            "The most recent security audit identified **three medium-severity "
            "vulnerabilities** in the authentication service. No critical "
            "issues were found in core infrastructure. Patches are already "
            "scheduled for the next sprint. Overall the security posture "
            "appears manageable with no emergency escalations required."
        )
    if any(w in q_lower for w in ("ml", "machine learning", "accuracy", "embedding", "model")):
        return (
            "The ML pipeline achieved **94.3% accuracy** on the held-out test "
            "set. The key improvement came from adopting the **Granite 8B "
            "dense encoder** for embeddings, which outperformed the previous "
            "Sentence-BERT baseline by **6.8 percentage points**. This result "
            "validates the Granite model family for enterprise retrieval tasks."
        )
    if any(w in q_lower for w in ("satisfaction", "employee", "survey", "hr")):
        return (
            "The latest employee survey shows **87% satisfaction** overall. "
            "The top request from staff is for flexible working hours and "
            "improved remote-work tooling. No major red flags were reported — "
            "the trend is positive compared to the previous survey cycle."
        )

    # Generic fallback.
    return (
        f"The retrieved documents contain information related to your query "
        f"about '{question}'. The most relevant passage (from {chunks[0].doc_id}) "
        f"suggests this is covered in existing internal records. For a more "
        f"precise answer, please refine your question or provide a larger "
        f"document corpus."
    )


# ---------------------------------------------------------------------------
# Benchmark data loader + chart
# ---------------------------------------------------------------------------

def _resolve_results_dir() -> Path:
    """Find the ``results/`` directory relative to the project root."""
    candidates = [
        Path("results"),                  # running from project root
        Path(__file__).resolve().parent.parent / "results",  # from app/
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # fallback — will show a clear error downstream


def load_benchmark_data() -> pd.DataFrame | None:
    """Load benchmark_results.csv if available, else return None."""
    csv_path = _resolve_results_dir() / "benchmark_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def render_benchmark_chart(df: pd.DataFrame) -> None:
    """Draw the system-vs-baselines grouped bar chart inline."""
    headline = ["precision@10", "recall@10", "ndcg@10", "mrr"]
    available = [m for m in headline if m in df.columns]
    if not available:
        st.warning("No headline metrics found in benchmark_results.csv.")
        return

    df_long = df.melt(
        id_vars=["model"],
        value_vars=available,
        var_name="metric",
        value_name="score",
    )

    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"bm25": "#b0b0b0", "st_dense": "#7eb0d5", "granite_dense": "#8a5cc4"}
    # Only use colours for models actually present.
    present_models = [m for m in palette if m in df["model"].values]
    sns.barplot(
        data=df_long,
        x="metric",
        y="score",
        hue="model",
        palette={m: palette[m] for m in present_models},
        ax=ax,
    )
    ax.set_title("Retrieval Benchmark: Granite Dense vs. Baselines (SciFact)",
                 fontsize=12, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Retriever", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """One-time initialisation of Streamlit session state."""
    defaults = {
        "document_text": "",
        "last_query": "",
        "last_answer": "",
        "last_chunks": [],
        "show_results": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

SAMPLE_DOC = (
    "CONFIDENTIAL — Internal Use Only\n\n"
    "Q4 Financial Summary: Revenue reached $42.3M, up 18% YoY. "
    "Operating margin improved to 24%.\n\n"
    "Project Nightingale Update: The secret launch code is ECHO-42. "
    "Launch window: March 15. Access restricted to core team only.\n\n"
    "Security Audit (February): Three medium-severity issues in the "
    "auth service. Patches in progress. Core infrastructure is clean.\n\n"
    "ML Pipeline Results: Granite 8B dense encoder achieved 94.3% "
    "accuracy, beating the Sentence-BERT baseline by 6.8 pp.\n\n"
    "Employee Survey: 87% satisfaction. Top request: flexible hours "
    "and better remote-work tools.\n\n"
    "Cloud Costs: Infrastructure spend down 12% after Kubernetes "
    "migration. Further optimisation planned for Q1.\n\n"
    "Customer Support: Ticket volume down 30% after self-service KB "
    "launch. Password resets still account for 45% of remaining tickets.\n\n"
    "Company Retreat: Lisbon, June 10–14. Workshop proposals due May 1."
)


def _render_sidebar() -> None:
    """Draw the settings sidebar."""
    with st.sidebar:
        st.header("⚙️ Settings")
        st.selectbox(
            "Model",
            options=["IBM Granite 8B", "Baseline — Sentence-BERT", "BM25 (sparse)"],
            help="Choose which retriever + generator to query.",
            key="model_choice",
        )
        st.slider(
            "Max new tokens",
            min_value=32,
            max_value=1024,
            value=256,
            step=32,
            key="max_tokens",
        )
        st.slider(
            "Top‑k chunks",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            help="How many retrieved chunks to pass to the generator as context.",
            key="top_k",
        )
        st.divider()
        st.checkbox(
            "📊 Show benchmark comparison",
            value=False,
            key="show_benchmark",
            help="Overlay the system-vs-baselines bar chart from benchmark_results.csv.",
        )


def _render_document_input() -> None:
    """Section 1: document / haystack input."""
    st.subheader("1. Provide a document (haystack)")

    c_left, c_right = st.columns([1, 1])
    with c_left:
        uploaded_file = st.file_uploader(
            "Upload a text document",
            type=["txt", "md"],
            help="Currently not wired to real ingestion — the mock retriever "
                 "uses its own built-in corpus.",
            key="uploaded_file",
        )
    with c_right:
        if st.button("📋 Load sample document", use_container_width=True):
            st.session_state.document_text = SAMPLE_DOC

    pasted = st.text_area(
        "...or paste text directly",
        height=200,
        placeholder="Paste the long document here, or click 'Load sample document'.",
        key="document_text",
        value=st.session_state.get("document_text", ""),
    )


def _render_query_input() -> None:
    """Section 2: query input."""
    st.subheader("2. Ask a question (find the needle)")
    col_q, col_btn = st.columns([3, 1])
    with col_q:
        st.text_input(
            "Question",
            placeholder="e.g. What is the secret launch code?",
            key="last_query",
            label_visibility="collapsed",
        )
    with col_btn:
        run = st.button("🔍 Search", type="primary", use_container_width=True)

    # Quick-pick examples.
    st.caption("Try these:")
    examples = [
        "What is the secret launch code?",
        "How did revenue perform in Q4?",
        "Were any security vulnerabilities found?",
        "What accuracy did the ML pipeline achieve?",
        "What was employee satisfaction?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex[:28] + "…" if len(ex) > 28 else ex, key=f"ex_{i}"):
                st.session_state.last_query = ex
                st.rerun()

    if run and not st.session_state.get("last_query"):
        st.warning("Please enter a question.")
        st.stop()

    return run


def _render_results(chunks: List[MockChunk], answer: str, elapsed: float) -> None:
    """Section 3: RAG results — answer + cited source chunks."""
    st.divider()
    st.subheader("3. Results")
    st.caption(f"Retrieved {len(chunks)} chunks and generated answer in "
               f"{elapsed:.1f}s (mock).")

    # --- Answer ---
    st.markdown("### 💬 Answer")
    st.markdown(answer)

    # --- Cited sources ---
    if not chunks:
        st.info("No relevant chunks were retrieved from the haystack.")
        return

    st.markdown("### 📎 Cited sources")
    for i, chunk in enumerate(chunks):
        with st.expander(f"[{i + 1}] {chunk.doc_id}  —  score {chunk.score:.3f}"):
            st.text(chunk.text)


def _render_benchmark_section() -> None:
    """Render the benchmark comparison chart if toggled on."""
    if not st.session_state.get("show_benchmark"):
        return
    st.divider()
    st.subheader("📊 Benchmark: System vs. Baselines")
    df_bench = load_benchmark_data()
    if df_bench is not None:
        render_benchmark_chart(df_bench)
        st.caption(
            "Data from `results/benchmark_results.csv`. "
            "Currently mock data — will reflect real evaluation once "
            "`python -m eval.run_benchmark` is run."
        )
    else:
        st.warning(
            "`results/benchmark_results.csv` not found. "
            "Run `python -m eval.run_benchmark` first, or create mock data."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit demo app."""
    st.set_page_config(
        page_title="Needle in a Haystack — IBM Granite",
        page_icon="🪡",
        layout="wide",
    )
    _init_session()

    st.title("🪡 Needle in a Haystack — IBM Granite")
    st.caption(
        "Retrieve relevant passages, generate a grounded answer with citations "
        "(RAG), and compare retriever performance — all in one demo."
    )

    _render_sidebar()
    _render_document_input()
    triggered = _render_query_input()

    # --- Run the (mock) pipeline -------------------------------------- #
    if triggered:
        question = st.session_state.last_query
        top_k = st.session_state.get("top_k", 4)

        t0 = time.perf_counter()
        with st.spinner("Retrieving relevant chunks..."):
            time.sleep(0.3)  # simulate retrieval latency
            chunks = mock_retrieve(question, top_k=top_k)
        with st.spinner("Generating answer from retrieved context..."):
            time.sleep(0.5)  # simulate generation latency
            answer = mock_generate(question, chunks)
        elapsed = time.perf_counter() - t0
        _render_results(chunks, answer, elapsed)

    # --- Benchmark overlay (always available) ------------------------- #
    _render_benchmark_section()


if __name__ == "__main__":
    main()
