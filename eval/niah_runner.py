"""Main runner for the Needle In A Haystack (NIAH) evaluation.

Sweeps a grid of **context lengths** × **needle depths**, queries the model for
each combination, scores the response, and collects the results into a tidy
table for downstream visualization (see
``notebooks/01_visualize_heatmap.ipynb``).

Run as a module from the project root:

    python -m eval.niah_runner

The implementation is left as a documented skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from src.data_processing import HaystackSample, Needle, build_sample, load_haystack, load_needles
from src.llm_client import LLMClient
from eval.metrics import score_accuracy


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class NIAHConfig:
    """Configuration for a NIAH evaluation sweep.

    Attributes
    ----------
    context_lengths:
        Target context lengths (in tokens) to evaluate.
    depth_percents:
        Needle injection depths (as percentages, 0–100) to evaluate.
    haystack_path:
        Path to the raw haystack document.
    needles_path:
        Path to the needle definitions.
    results_path:
        Where to write the results table (CSV).
    """

    context_lengths: List[int] = field(
        default_factory=lambda: [1000, 2000, 4000, 8000, 16000]
    )
    depth_percents: List[float] = field(
        default_factory=lambda: [0, 10, 25, 50, 75, 90, 100]
    )
    haystack_path: Path = Path("data/raw/haystack.txt")
    needles_path: Path = Path("data/needles/needles.json")
    results_path: Path = Path("results/niah_results.csv")


# --------------------------------------------------------------------------- #
# Core loop
# --------------------------------------------------------------------------- #
def run_single(client: LLMClient, sample: HaystackSample) -> dict:
    """Evaluate a single (length, depth) sample.

    Builds the prompt from the sample, queries the model, scores the answer,
    and returns one result row.

    Parameters
    ----------
    client:
        The model client to query.
    sample:
        The assembled haystack sample.

    Returns
    -------
    dict
        A result row with keys such as ``context_length``, ``depth_percent``,
        ``answer``, and ``score``.
    """
    raise NotImplementedError(
        "TODO: format prompt, call client.generate, score with eval.metrics."
    )


def run_sweep(config: NIAHConfig, client: LLMClient) -> pd.DataFrame:
    """Run the full length × depth evaluation grid.

    Parameters
    ----------
    config:
        The sweep configuration.
    client:
        The model client to evaluate.

    Returns
    -------
    pandas.DataFrame
        One row per (context_length, depth_percent) combination.
    """
    raise NotImplementedError(
        "TODO: loop over lengths and depths, build samples, call run_single."
    )


def main() -> None:
    """Entry point: load data, run the sweep, and persist results."""
    config = NIAHConfig()
    client = LLMClient()  # self-hosted Granite (model id from GRANITE_MODEL_ID)

    results = run_sweep(config, client)

    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.results_path, index=False)
    print(f"Saved results to {config.results_path}")


if __name__ == "__main__":
    main()
