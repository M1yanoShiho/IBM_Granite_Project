"""Login-node preflight for the G5/G7 generation job.

Exists because a G6 attempt burned 76 GPU-minutes to discover that TRUE's weights
were not in the offline cache. `run_gate0b.slurm` has had a login-node model
pre-download since its first version; this job did not, and that is the direct
reason the root cause surfaced on a GPU instead of in a login shell.

Checks, in the order that fails cheapest first:

1. both model repos resolve offline, with the weight FILES actually present --
   `HF_HUB_OFFLINE=1` turns a missing repo into an OSError at first use, deep
   inside a 400-case loop;
2. the verifier loads AND answers a known entailment pair correctly. A checkpoint
   that converted badly loads fine, runs fine, and returns plausible-looking
   scores, so presence is not sufficient -- the pair is the check.

Run it on the login node before `sbatch`, or as the first step of the job.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ENTAILING = (
    "Marie Curie was awarded the Nobel Prize in Physics in 1903.",
    "Marie Curie won a Nobel Prize.",
)
CONTRADICTING = (
    "Marie Curie was awarded the Nobel Prize in Physics in 1903.",
    "Marie Curie never received any Nobel Prize.",
)


def _fail(message: str) -> int:
    print(f"[preflight] FAIL: {message}", flush=True)
    return 1


def check_weights_present(repo_id: str) -> str | None:
    """The repo directory exists in the cache and holds loadable weight files.

    ``.bin`` counts as loadable only under torch >= 2.6: transformers refuses to
    ``torch.load`` a pickle below that (CVE-2025-32434), and this repo ships only
    ``.bin``. That version boundary is what separated a working account from a
    failing one across the two G6 attempts, so it is checked rather than assumed.
    """
    import torch

    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    folder = cache / f"models--{repo_id.replace('/', '--')}"
    if not folder.is_dir():
        return f"{repo_id}: not in the offline cache at all ({folder})"
    files = [p for p in folder.rglob("*") if not p.name.startswith(".")]
    safetensors = [p for p in files if p.suffix == ".safetensors"]
    pickles = [p for p in files if p.suffix == ".bin"]
    if safetensors:
        return None
    if not pickles:
        return f"{repo_id}: cached but carries no weight files ({folder})"
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 6):
        return (
            f"{repo_id}: only .bin weights are cached and torch is "
            f"{torch.__version__} (<2.6), so transformers will refuse to load them "
            "(CVE-2025-32434). Upgrade torch, or convert to safetensors with "
            "scripts/convert_bin_to_safetensors.py."
        )
    return None


def check_verifier_answers() -> str | None:
    """Load the verifier and make it judge a pair whose answer is known.

    Presence is not enough. A checkpoint whose tied embeddings were dropped rather
    than cloned loads cleanly, runs at full speed, and returns scores that look
    entirely reasonable -- it is simply wrong. Only a known pair catches that.
    """
    from evidence_rag.generator.nli import build_nli_model

    nli = build_nli_model("true")
    entailed = nli.classify(premise=ENTAILING[0], hypothesis=ENTAILING[1])
    contradicted = nli.classify(premise=CONTRADICTING[0], hypothesis=CONTRADICTING[1])
    print(f"[preflight] TRUE: entailing -> {entailed}, contradicting -> {contradicted}")
    if entailed != "entailment":
        return f"verifier called a known entailment {entailed!r}"
    if contradicted == "entailment":
        return "verifier called a known contradiction 'entailment'"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-load", action="store_true", help="cache checks only")
    args = parser.parse_args()

    repos = [
        os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-4.1-3b"),
        os.getenv("NLI_MODEL_ID_TRUE", "google/t5_xxl_true_nli_mixture"),
    ]
    for repo in repos:
        problem = check_weights_present(repo)
        if problem:
            return _fail(problem)
        print(f"[preflight] cached: {repo}")

    if args.skip_load:
        print("[preflight] OK (cache only)", flush=True)
        return 0

    problem = check_verifier_answers()
    if problem:
        return _fail(problem)
    print("[preflight] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
