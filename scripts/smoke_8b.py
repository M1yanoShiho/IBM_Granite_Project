"""Smoke test: verify a Granite model loads and generates on a GPU node.

This is the first real run of ``src/llm_client.py`` against a downloaded model —
it confirms the generation layer works before we wire it into the RAG eval.

Run via Slurm (NEVER on the login node):

    sbatch scripts/smoke_8b.slurm

Override the model with the GRANITE_MODEL_ID env var, e.g.

    GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b   # small, for a first check
    GRANITE_MODEL_ID=ibm-granite/<8B-INSTRUCT>    # the RAG model (confirm repo on HF)
"""

from __future__ import annotations

import os

import torch

from src.llm_client import LLMClient


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({props.total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: no GPU visible — are you running this on a GPU node?")

    print(f"GRANITE_MODEL_ID = {os.getenv('GRANITE_MODEL_ID', '(LLMClient default)')}")

    client = LLMClient()  # reads GRANITE_MODEL_ID, else the default in llm_client.py
    print(repr(client))

    prompt = "In one sentence, what is retrieval-augmented generation?"
    print("Prompt:", prompt)
    print("Answer:", client.generate(prompt))
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
