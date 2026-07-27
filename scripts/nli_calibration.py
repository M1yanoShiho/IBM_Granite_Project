"""Generator Part B -- NLI calibration (Half 1.5, local / CPU).

Measurement only. Answers three questions with a decision table:
  1. nli-deberta-v3-base vs -large?
  2. sentence-level premise (max over sentences) vs full-abstract premise?
  3. is the ``contradicted`` signal reliable enough to report, or diagnostic-only?

It does NOT modify nli.py / attribution.py / entity_check.py -- it imports
``DebertaNLIModel`` and calls it. No tuning, no thresholds, no production config
change. This pass produces numbers, not decisions.

Data: the canonical SciFact release (claims with SUPPORT / CONTRADICT rationale
sentences + full abstracts + cited-but-non-evidence abstracts for hard neutrals).
ir-datasets only ships BEIR/scifact, which is binary-relevance and has dropped the
SUPPORT/CONTRADICT labels and rationale-sentence indices this task needs, so we
pull the original release tarball with stdlib urllib (no new dependency). Set
``SCIFACT_DATA_DIR`` to choose the cache location (defaults next to
``MODEL_CACHE_DIR`` when set).

Model weights honor ``MODEL_CACHE_DIR`` (read inside nli.py). CPU only; no CUDA
assumption. Downloads models + a dataset -- lives in scripts/, never CI.

Run:
    python scripts/nli_calibration.py            # base + large, ~50 pairs/label
    python scripts/nli_calibration.py --n 30     # smaller/faster
    python scripts/nli_calibration.py --models base   # base only

Writes docs/generator/nli-calibration-results.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from evidence_rag.generator.nli import NLILabel, DebertaNLIModel

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

MODEL_IDS = {
    "base": "cross-encoder/nli-deberta-v3-base",
    "large": "cross-encoder/nli-deberta-v3-large",
}

GOLD_LABELS: tuple[NLILabel, ...] = ("entailment", "contradiction", "neutral")
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "docs" / "generator" / "nli-calibration-results.md"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def _data_dir() -> Path:
    override = os.getenv("SCIFACT_DATA_DIR")
    if override:
        return Path(override)
    cache_root = os.getenv("MODEL_CACHE_DIR")
    base = Path(cache_root) if cache_root else Path.home() / ".cache"
    return base / "scifact"


def ensure_scifact() -> Path:
    """Download + extract the SciFact release once; return the ``data/`` dir."""
    root = _data_dir()
    data = root / "data"
    if (data / "corpus.jsonl").exists():
        return data
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "scifact_data.tar.gz"
    if not tgz.exists():
        print(f"Downloading SciFact release -> {tgz}")
        try:
            urllib.request.urlretrieve(SCIFACT_URL, tgz)
        except Exception as exc:  # noqa: BLE001 - surface, don't fall back to fakes
            raise SystemExit(
                f"FAILED to download SciFact ({type(exc).__name__}: {exc}).\n"
                "This calibration needs real labelled text; fix the network/proxy "
                "or set SCIFACT_DATA_DIR to a pre-downloaded copy, then rerun."
            ) from exc
    with tarfile.open(tgz) as t:
        t.extractall(root)  # noqa: S202 - trusted first-party archive
    return data


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass
class Pair:
    gold: NLILabel
    claim: str
    sentence_premises: tuple[str, ...]  # sentence-level cell (max over these)
    abstract_premise: str  # full-abstract cell (single classify)


def build_pairs(data: Path, n_per_label: int, seed: int = 13) -> list[Pair]:
    """Build entailment / contradiction / hard-neutral pairs at both granularities.

    - entailment / contradiction: claim + gold rationale sentences (sentence cell)
      and claim + the full abstract that contains them (abstract cell). Same pair,
      only premise length changes -- a direct dilution measurement.
    - hard-neutral: claim + an abstract the claim CITES but that is NOT evidence
      for it (same topic, shares entities, says nothing decisive). Topically
      adjacent by construction, which is what makes the contradiction-FP number
      meaningful.
    """
    corpus = {str(d["doc_id"]): d for d in _read_jsonl(data / "corpus.jsonl")}
    claims: list[dict] = []
    for name in ("claims_train.jsonl", "claims_dev.jsonl"):
        claims.extend(_read_jsonl(data / name))

    entail: list[Pair] = []
    contra: list[Pair] = []
    neutral: list[Pair] = []

    for c in claims:
        claim_text = c["claim"]
        evidence: dict = c.get("evidence") or {}
        evidence_doc_ids = set(evidence.keys())

        for doc_id, groups in evidence.items():
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            abstract: list[str] = doc["abstract"]
            abstract_text = " ".join(abstract)
            for group in groups:
                sent_idx = [i for i in group["sentences"] if 0 <= i < len(abstract)]
                if not sent_idx:
                    continue
                rationale = tuple(abstract[i] for i in sent_idx)
                pair = Pair(
                    gold="entailment" if group["label"] == "SUPPORT" else "contradiction",
                    claim=claim_text,
                    sentence_premises=rationale,
                    abstract_premise=abstract_text,
                )
                (entail if pair.gold == "entailment" else contra).append(pair)

        # Hard neutral: cited docs that are NOT evidence for this claim.
        for doc_id in c.get("cited_doc_ids", []):
            if str(doc_id) in evidence_doc_ids:
                continue
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            abstract = doc["abstract"]
            neutral.append(
                Pair(
                    gold="neutral",
                    claim=claim_text,
                    sentence_premises=tuple(abstract),
                    abstract_premise=" ".join(abstract),
                )
            )

    rng = random.Random(seed)
    for pool in (entail, contra, neutral):
        rng.shuffle(pool)

    picked = entail[:n_per_label] + contra[:n_per_label] + neutral[:n_per_label]
    print(
        f"Pairs available -> entailment:{len(entail)} contradiction:{len(contra)} "
        f"neutral:{len(neutral)}; using {n_per_label}/label"
    )
    return picked


# --------------------------------------------------------------------------- #
# Prediction + aggregation
# --------------------------------------------------------------------------- #

def aggregate_sentences(labels: list[NLILabel]) -> NLILabel:
    """Max over sentences with precedence entailment > contradiction > neutral.

    Rationale: for attribution, any sentence that entails the claim makes the
    chunk supporting; a contradiction only stands if nothing entailed. This is the
    precedence the production code would need, stated explicitly per the guide.
    """
    if "entailment" in labels:
        return "entailment"
    if "contradiction" in labels:
        return "contradiction"
    return "neutral"


@dataclass
class CellResult:
    matrix: Counter = field(default_factory=Counter)  # (gold, predicted) -> count
    n: int = 0
    seconds: float = 0.0

    def add(self, gold: NLILabel, predicted: NLILabel) -> None:
        self.matrix[(gold, predicted)] += 1
        self.n += 1

    def entailment_recall(self) -> float:
        total = sum(self.matrix[("entailment", p)] for p in GOLD_LABELS)
        return self.matrix[("entailment", "entailment")] / total if total else float("nan")

    def contradiction_fp_rate(self) -> float:
        total = sum(self.matrix[("neutral", p)] for p in GOLD_LABELS)
        return self.matrix[("neutral", "contradiction")] / total if total else float("nan")

    def per_pair_ms(self) -> float:
        return (self.seconds / self.n * 1000.0) if self.n else float("nan")


def run_cell(model: DebertaNLIModel, pairs: list[Pair], *, sentence_level: bool) -> CellResult:
    result = CellResult()
    for pair in pairs:
        start = time.perf_counter()
        if sentence_level:
            labels = [
                model.classify(premise=prem, hypothesis=pair.claim)
                for prem in pair.sentence_premises
            ]
            predicted = aggregate_sentences(labels)
        else:
            predicted = model.classify(premise=pair.abstract_premise, hypothesis=pair.claim)
        result.seconds += time.perf_counter() - start
        result.add(pair.gold, predicted)
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def matrix_table(result: CellResult) -> str:
    header = f"| gold \\ pred | {' | '.join(GOLD_LABELS)} |"
    sep = "|" + "---|" * (len(GOLD_LABELS) + 1)
    rows = [header, sep]
    for gold in GOLD_LABELS:
        cells = " | ".join(str(result.matrix[(gold, p)]) for p in GOLD_LABELS)
        rows.append(f"| **{gold}** | {cells} |")
    return "\n".join(rows)


def id2label_report(model: DebertaNLIModel) -> str:
    model._ensure_loaded()  # noqa: SLF001 - inspection only, for the normalization note
    return json.dumps({int(k): v for k, v in model._model.config.id2label.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="pairs per gold label")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "large"],
        choices=list(MODEL_IDS),
    )
    args = parser.parse_args()

    data = ensure_scifact()
    pairs = build_pairs(data, args.n)

    lines: list[str] = [
        "# NLI calibration results (Generator Part B, Half 1.5)",
        "",
        "Measurement only -- no production code changed. Data: canonical SciFact "
        "release (SUPPORT/CONTRADICT rationale sentences + full abstracts; hard "
        "neutrals = claim x a cited-but-non-evidence abstract). Premise = evidence, "
        "hypothesis = claim. Sentence-cell aggregation: max over sentences, "
        "precedence entailment > contradiction > neutral. CPU.",
        "",
        f"Pairs per gold label: {args.n}.",
        "",
    ]

    id2label_notes: dict[str, str] = {}
    cells: dict[tuple[str, str], CellResult] = {}

    for model_key in args.models:
        model_id = MODEL_IDS[model_key]
        print(f"\n=== {model_key} ({model_id}) ===")
        model = DebertaNLIModel(model_id=model_id, device="cpu")
        try:
            id2label_notes[model_key] = id2label_report(model)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FAILED to load {model_id}: {type(exc).__name__}: {exc}") from exc
        print(f"id2label: {id2label_notes[model_key]}")

        for sentence_level in (True, False):
            cell_name = "sentence" if sentence_level else "abstract"
            print(f"  running cell: {model_key} x {cell_name} ...")
            result = run_cell(model, pairs, sentence_level=sentence_level)
            cells[(model_key, cell_name)] = result
            print(
                f"    entailment recall={result.entailment_recall():.3f}  "
                f"contradiction FP={result.contradiction_fp_rate():.3f}  "
                f"{result.per_pair_ms():.1f} ms/pair"
            )

    # --- markdown report --------------------------------------------------- #
    lines.append("## id2label per checkpoint")
    lines.append("")
    for key, mapping in id2label_notes.items():
        lines.append(f"- `{MODEL_IDS[key]}`: `{mapping}`")
    lines.append("")

    lines.append("## Headline decision table")
    lines.append("")
    lines.append("| cell | entailment recall | contradiction FP rate | ms/pair |")
    lines.append("|---|---|---|---|")
    for (model_key, cell_name), result in cells.items():
        lines.append(
            f"| {model_key} / {cell_name} | {result.entailment_recall():.3f} | "
            f"{result.contradiction_fp_rate():.3f} | {result.per_pair_ms():.1f} |"
        )
    lines.append("")

    lines.append("## Confusion matrices (gold rows x predicted cols)")
    lines.append("")
    for (model_key, cell_name), result in cells.items():
        lines.append(f"### {model_key} / {cell_name} (n={result.n})")
        lines.append("")
        lines.append(matrix_table(result))
        lines.append("")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
