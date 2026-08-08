"""Held-out dataset loaders: HotpotQA, RGB, MuSiQue-Full.

These three sets are the project's closed test sets. They were never loaded
during any calibration round, and this module exists only so that the one
permitted run does not fail on a loader bug.

Each loader returns the same shape the ASQA runner already consumes, so the
generation script needs no dataset-specific branching:

    {"query_id": str, "question": str, "gold_answers": [(str, ...)],
     "passages": [{"title": str, "text": str}, ...]}

``passages`` is the candidate pool in dataset order, gold passages included --
the Generator is given *selected* evidence and never retrieves, so held-out
measures generation given evidence, exactly as calibration did.

**Deliberately absent: any scoring, any aggregation, any print of answer text.**
Confirming that a field exists is not observing a result.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
MUSIQUE_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/data/validation-00000-of-00001.parquet"
)
RGB_URL = (
    "https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json"
)

DATASETS = ("hotpotqa", "musique-full", "rgb")


def data_dir() -> Path:
    explicit = os.getenv("HELDOUT_DATA_DIR")
    if explicit:
        target = Path(explicit)
    else:
        cache = os.getenv("MODEL_CACHE_DIR")
        target = Path(cache) / "heldout" if cache else Path.home() / ".cache" / "heldout"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    print(f"[data] downloading {url} -> {destination}", flush=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 - a download failure must be loud
        partial.unlink(missing_ok=True)
        raise SystemExit(f"could not download {url}: {exc}") from exc
    partial.replace(destination)
    return destination


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def load_hotpotqa() -> list[dict[str, Any]]:
    """HotpotQA dev, distractor setting: 10 paragraphs, 2 gold + 8 distractors."""
    path = _download(HOTPOT_URL, data_dir() / "hotpot_dev_distractor_v1.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for row in raw:
        passages = [
            {"title": _clean(title), "text": _clean("".join(sentences))}
            for title, sentences in row.get("context", [])
            if "".join(sentences).strip()
        ]
        answer = _clean(row.get("answer", ""))
        if not passages or not answer:
            continue
        out.append(
            {
                "query_id": str(row["_id"]),
                "question": _clean(row["question"]),
                "gold_answers": [(answer,)],
                "passages": passages,
            }
        )
    return out


def load_musique_full() -> list[dict[str, Any]]:
    """MuSiQue validation, full setting: 20 paragraphs, answerable and not.

    The unanswerable half is kept -- dropping it would turn the held-out set into
    a different, easier task than the one named.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit("reading the MuSiQue parquet needs pandas + pyarrow") from exc

    path = _download(MUSIQUE_URL, data_dir() / "musique_validation.parquet")
    frame = pd.read_parquet(path)
    out: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        paragraphs = row.get("paragraphs")
        passages = [
            {
                "title": _clean(p.get("title", "")),
                "text": _clean(p.get("paragraph_text", "")),
            }
            for p in (paragraphs if paragraphs is not None else [])
            if _clean(p.get("paragraph_text", ""))
        ]
        answer = _clean(row.get("answer", "") or "")
        if not passages or not answer:
            continue
        aliases = [answer]
        for alias in row.get("answer_aliases") or []:
            if _clean(alias):
                aliases.append(_clean(alias))
        out.append(
            {
                "query_id": str(row["id"]),
                "question": _clean(row["question"]),
                "gold_answers": [tuple(dict.fromkeys(aliases))],
                "passages": passages,
            }
        )
    return out


def load_rgb() -> list[dict[str, Any]]:
    """RGB (Retrieval-Augmented Generation Benchmark), English split.

    Each item carries a query, a list of acceptable answers, and a passage pool
    that deliberately mixes supporting and noise documents.
    """
    path = _download(RGB_URL, data_dir() / "rgb_en.json")
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:  # the repo ships JSON Lines
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        docs = row.get("positive", []) or []
        docs = list(docs) + list(row.get("negative", []) or [])
        passages = [
            {"title": "", "text": _clean(d if isinstance(d, str) else " ".join(d))}
            for d in docs
            if _clean(d if isinstance(d, str) else " ".join(d))
        ]
        answers = row.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        flat: list[str] = []
        for a in answers:
            if isinstance(a, list):
                flat.extend(_clean(x) for x in a if _clean(x))
            elif _clean(a):
                flat.append(_clean(a))
        if not passages or not flat:
            continue
        out.append(
            {
                "query_id": str(row.get("id", index)),
                "question": _clean(row.get("query", "")),
                "gold_answers": [tuple(dict.fromkeys(flat))],
                "passages": passages,
            }
        )
    return out


LOADERS = {
    "hotpotqa": load_hotpotqa,
    "musique-full": load_musique_full,
    "rgb": load_rgb,
}


def load(name: str) -> list[dict[str, Any]]:
    if name not in LOADERS:
        raise SystemExit(f"unknown held-out dataset {name!r}; expected one of {DATASETS}")
    return LOADERS[name]()
