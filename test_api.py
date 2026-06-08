"""Baseline LLM infrastructure smoke test (connectivity / plumbing check).

The purpose of this script is narrow: prove that our LLM plumbing works
end-to-end -- credentials load, the Hugging Face client authenticates, a request
is routed to a live inference provider, and a response comes back. It is meant to
run *before* we obtain IBM watsonx.ai (Granite) access, so the team can validate
connectivity early.

Model choice note:
    IBM Granite generation models are **not** available on Hugging Face's free
    ``hf-inference`` provider (Granite chat/instruct models are served only via
    paid third-party providers such as ``featherless-ai``, and the embedding
    Granite models don't do text generation). Since this is purely an
    infrastructure check, we use a small model that is live on the free
    ``hf-inference`` provider. The *real* Granite inference for the project will
    run through watsonx.ai (``langchain_ibm.WatsonxLLM``) in ``src/llm_client.py``.

What it does:
    1. Loads the ``HUGGINGFACE_API_KEY`` from the local ``.env`` file.
    2. Initializes a free ``hf-inference`` chat model with a low temperature
       (0.1) for near-deterministic output.
    3. Sends a simple prompt and prints the model's response to the terminal.

Usage:
    python test_api.py

Prerequisites:
    - Dependencies installed (see ``requirements.txt``, incl. ``langchain-huggingface``).
    - A valid Hugging Face access token set in ``.env`` as ``HUGGINGFACE_API_KEY``.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Model and prompt configuration.
# A small chat model that is live on the FREE ``hf-inference`` provider, used
# here only to prove connectivity. Swap to a Granite model once a hosting
# provider (featherless-ai or watsonx.ai) is available.
MODEL_ID = "katanemo/Arch-Router-1.5B"
HF_PROVIDER = "hf-inference"
PROMPT = "What is the capital of UK?"


def main() -> int:
    """Run the baseline Hugging Face connectivity test.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a configuration or runtime error.
    """
    # 1. Load environment variables from the local .env file.
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")

    if not api_key or api_key.startswith("your-"):
        print(
            "ERROR: HUGGINGFACE_API_KEY is not set (or still a placeholder).\n"
            "       Add your real token to the .env file, e.g.:\n"
            "           HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx\n",
            file=sys.stderr,
        )
        return 1

    # 2. Initialize the model via the Hugging Face Inference Endpoint.
    #    These models are served as the ``conversational`` (chat) task, so we
    #    wrap the raw endpoint in ChatHuggingFace, which calls chat_completion.
    print(f"Initializing model: {MODEL_ID} (provider: {HF_PROVIDER}) ...")
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="conversational",
        provider=HF_PROVIDER,
        temperature=0.1,
        max_new_tokens=128,
        huggingfacehub_api_token=api_key,
    )
    chat = ChatHuggingFace(llm=endpoint)

    # 3. Send the prompt and print the response.
    print(f"\nPrompt: {PROMPT}")
    print("Querying the model (this may take a moment)...\n")

    try:
        response = chat.invoke([HumanMessage(content=PROMPT)])
    except Exception as exc:  # noqa: BLE001 - surface any API error to the user
        # Some HF errors carry an empty str(), so include the type, repr, and a
        # full traceback to make the real cause visible.
        import traceback

        print(
            f"ERROR: model call failed: [{type(exc).__name__}] {exc!r}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1

    print("=" * 60)
    print("Model response:")
    print("=" * 60)
    print(response.content.strip())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
