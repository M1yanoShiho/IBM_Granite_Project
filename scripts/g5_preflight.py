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
    """The repo directory exists in the cache and holds at least one weight file."""
    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    folder = cache / f"models--{repo_id.replace('/', '--')}"
    if not folder.is_dir():
        return f"{repo_id}: not in the offline cache at all ({folder})"
    weights = [
        p
        for p in folder.rglob("*")
        if p.suffix in {".safetensors", ".bin"} and not p.name.startswith(".")
    ]
    if not weights:
        return f"{repo_id}: cached but carries no weight files ({folder})"
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
