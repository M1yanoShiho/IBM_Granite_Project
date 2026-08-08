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

HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
"""The original CMU host (`curtis.ml.cmu.edu`) is unreachable from the cluster --
the connection times out rather than refusing, so it looks like a hang. The HF
mirror carries the same dev distractor split."""

MUSIQUE_URL = (
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_full_v1.0_dev.jsonl"
)
"""**Full, not Ans.** The commonly-mirrored `musique_ans_*` files contain only the
answerable half. Substituting them would quietly turn the held-out set into a
different and easier task than the one this project committed to, so the repo is
chosen for carrying `musique_full_v1.0_dev.jsonl` specifically."""

RGB_BASE = "https://raw.githubusercontent.com/chen700564/RGB/master/data/"
RGB_URL = RGB_BASE + "en.json"
RGB_FACT_URL = RGB_BASE + "en_fact.json"
"""RGB ships its four sub-tests as four files, not as one file with a label:
``en.json`` 300 (noise robustness / negative rejection), ``en_int.json`` 100
(information integration), ``en_fact.json`` 100 (counterfactual robustness),
``en_refine.json`` 300. So "RGB, 300 records" is one sub-test at full size, not
75 records of each -- verified by fetching all four and counting."""

DATASETS = ("hotpotqa", "musique-full", "rgb")
"""The three pre-registered held-out sets."""

SECONDARY = ("rgb-counterfactual",)
"""Named secondary analyses, pre-registered separately so they are never folded
into a headline number."""

ALL_SETS = DATASETS + SECONDARY


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
    """HotpotQA dev, distractor setting: 10 paragraphs, 2 gold + 8 distractors.

    The HF mirror stores ``context`` column-wise (``{"title": [...],
    "sentences": [[...], ...]}``) rather than as a list of pairs, so it is
    transposed back here.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit("reading the HotpotQA parquet needs pandas + pyarrow") from exc

    path = _download(HOTPOT_URL, data_dir() / "hotpot_dev_distractor.parquet")
    frame = pd.read_parquet(path)
    out: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        context = row["context"]
        titles = list(context["title"])
        bodies = list(context["sentences"])
        passages = [
            {"title": _clean(title), "text": _clean("".join(sentences))}
            for title, sentences in zip(titles, bodies, strict=False)
            if "".join(sentences).strip()
        ]
        answer = _clean(row.get("answer", ""))
        if not passages or not answer:
            continue
        out.append(
            {
                "query_id": str(row["id"]),
                "question": _clean(row["question"]),
                "gold_answers": [(answer,)],
                "passages": passages,
            }
        )
    return out


def load_musique_full() -> list[dict[str, Any]]:
    """MuSiQue dev, **full** setting: 20 paragraphs, answerable and unanswerable.

    The unanswerable half is kept. Dropping it -- which is what loading
    ``musique_ans`` instead would do -- would turn the held-out set into a
    different and easier task than the one this project committed to.

    Structure, verified rather than assumed: 4834 dev records = 2417 ids each
    appearing **twice**, once answerable and once not, and **both variants carry a
    non-empty ``answer`` string**. Two consequences:

    * The raw ``id`` is not unique, so it is suffixed. Using it as-is would collide
      in every id-keyed structure downstream, including the paired tests.
    * On an unanswerable item the evidence does *not* support that answer, because
      the supporting paragraphs were removed to construct it. String-match
      correctness would therefore **reward a system for producing it anyway** and
      penalise the correct behaviour of abstaining or labelling it unverified --
      exactly backwards for what this project measures. ``answerable`` is carried
      through so the two subsets are reported separately; see the pre-registration
      addendum.
    """
    path = _download(MUSIQUE_URL, data_dir() / "musique_full_v1.0_dev.jsonl")
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        passages = [
            {
                "title": _clean(p.get("title", "")),
                "text": _clean(p.get("paragraph_text", "")),
            }
            for p in row.get("paragraphs", [])
            if _clean(p.get("paragraph_text", ""))
        ]
        answer = _clean(row.get("answer", "") or "")
        if not passages or not answer:
            continue
        aliases = [answer]
        for alias in row.get("answer_aliases") or []:
            if _clean(alias):
                aliases.append(_clean(alias))
        answerable = bool(row.get("answerable", True))
        out.append(
            {
                "query_id": f"{row['id']}#{'ans' if answerable else 'unans'}",
                "question": _clean(row["question"]),
                "gold_answers": [tuple(dict.fromkeys(aliases))],
                "passages": passages,
                "answerable": answerable,
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


def load_rgb_counterfactual() -> list[dict[str, Any]]:
    """RGB counterfactual robustness (`en_fact.json`, 100 items).

    The only **real** adversarial data available to this project: G1's
    entity-substitution slice was synthetic. Each item carries a true ``answer``,
    a ``fakeanswer``, three ``positive`` documents stating the truth and three
    ``positive_wrong`` documents stating the falsehood.

    The pool is built as **true documents first, then wrong ones**, so a top-k
    cut leaves both present. That is deliberate and is the whole point: it
    creates a pool carrying two competing values in the same role, which is
    exactly the condition the entity layer claims to detect. A pool of only-wrong
    or only-true documents would test nothing about it.

    ``fake_answers`` is carried alongside ``gold_answers`` so the analysis can
    separate "asserted the truth" from "asserted the planted falsehood". Neither
    is shown to the Generator.
    """
    path = _download(RGB_FACT_URL, data_dir() / "rgb_en_fact.json")
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        true_docs = [_clean(d) for d in (row.get("positive") or []) if _clean(d)]
        wrong_docs = [_clean(d) for d in (row.get("positive_wrong") or []) if _clean(d)]
        passages = [{"title": "", "text": t} for t in true_docs + wrong_docs]
        answer = _clean(row.get("answer", ""))
        fake = _clean(row.get("fakeanswer", ""))
        if not passages or not answer:
            continue
        out.append(
            {
                "query_id": f"fact-{row.get('id', index)}",
                "question": _clean(row.get("query", "")),
                "gold_answers": [(answer,)],
                "fake_answers": [fake] if fake else [],
                "passages": passages,
                "n_true_passages": len(true_docs),
                "n_wrong_passages": len(wrong_docs),
            }
        )
    return out


LOADERS = {
    "hotpotqa": load_hotpotqa,
    "musique-full": load_musique_full,
    "rgb": load_rgb,
    "rgb-counterfactual": load_rgb_counterfactual,
}


def load(name: str) -> list[dict[str, Any]]:
    if name not in LOADERS:
        raise SystemExit(f"unknown held-out dataset {name!r}; expected one of {DATASETS}")
    return LOADERS[name]()
