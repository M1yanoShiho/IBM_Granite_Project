"""Streamlit demo for the Needle in a Haystack project.

A lightweight UI to demonstrate fact retrieval from long-context documents
with IBM Granite. The user can:

- Upload (or paste) a haystack document.
- Type a question (search box).
- Run the query and view the model's answer.

Run from the project root:

    streamlit run app/main.py

This is boilerplate — the model wiring is left as TODOs.
"""

from __future__ import annotations

import streamlit as st

# from src.llm_client import LLMClient


def main() -> None:
    """Render the Streamlit demo app."""
    st.set_page_config(
        page_title="Needle in a Haystack — IBM Granite",
        page_icon="🪡",
        layout="wide",
    )

    st.title("🪡 Needle in a Haystack — IBM Granite")
    st.caption(
        "Retrieve specific facts buried deep within long-context documents."
    )

    # --- Sidebar: configuration --------------------------------------- #
    with st.sidebar:
        st.header("Settings")
        st.selectbox(
            "Model",
            options=["IBM Granite", "Baseline (open-source)"],
            help="Choose which model to query.",
        )
        st.slider(
            "Max new tokens",
            min_value=32,
            max_value=1024,
            value=256,
            step=32,
        )

    # --- Document input ----------------------------------------------- #
    st.subheader("1. Provide a document (haystack)")
    uploaded_file = st.file_uploader(
        "Upload a text document",
        type=["txt", "md"],
        help="Placeholder: document parsing not yet implemented.",
    )
    pasted_text = st.text_area(
        "...or paste text directly",
        height=200,
        placeholder="Paste the long document here.",
    )

    # --- Query input -------------------------------------------------- #
    st.subheader("2. Ask a question (find the needle)")
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the secret code mentioned in the document?",
    )
    run = st.button("🔍 Search", type="primary")

    # --- Results ------------------------------------------------------ #
    if run:
        if not question:
            st.warning("Please enter a question.")
            return
        if not (uploaded_file or pasted_text):
            st.warning("Please upload or paste a document first.")
            return

        with st.spinner("Querying the model..."):
            # TODO: read document, build prompt, call LLMClient.generate(),
            #       and display the answer.
            st.info("Model integration not implemented yet (boilerplate).")


if __name__ == "__main__":
    main()
