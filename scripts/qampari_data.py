"""QAMPARI — the Generator's module-level held-out set.

ALCE's sibling to ASQA. `qampari_eval_gtr_top100.json` is the analogue of the
`asqa_eval_gtr_top100.json` used for calibration: same retriever family (GTR),
same top-100 pool, same top-5 cut, so the comparison tests the method rather than
evidence quality.

Pre-registered in `docs/generator/heldout-preregistration.md` before this file
ever read the data. Correctness is **answer recall by containment over gold alias
sets**, uncapped and capped at 5 -- deliberately not ALCE's official QAMPARI F1,
which parses the output as a comma-separated entity list and would mis-read prose
from every arm.

Lives in `scripts/` because the system is frozen: nothing in `src/` changes.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verifier_triage import ALCE_URL, _download, data_dir  # noqa: E402

QAMPARI_MEMBER = "ALCE-data/qampari_eval_gtr_top100.json"


def ensure_qampari() -> list[dict[str, Any]]:
    """The QAMPARI eval file, extracted from the ALCE archive already on disk.

    The archive has been present since the G1 triage, but this member has never
    been extracted -- only ASQA's was.
    """
    root = data_dir()
    extracted = root / "qampari_eval_gtr_top100.json"
    if not extracted.exists():
        tar_path = _download(ALCE_URL, root / "ALCE-data.tar")
        print("[data] extracting QAMPARI eval file", flush=True)
        with tarfile.open(tar_path) as archive:
            member = archive.extractfile(QAMPARI_MEMBER)
            if member is None:
                raise SystemExit(f"{QAMPARI_MEMBER} missing from the ALCE tar")
            extracted.write_bytes(member.read())
    with extracted.open(encoding="utf-8") as handle:
        return json.load(handle)


def _clean(text: object) -> str:
    return " ".join(str(text).split())


def load_qampari(top_k: int = 5) -> list[dict[str, Any]]:
    """Records in the shape the held-out runner consumes.

    ``gold_answers`` is a tuple per gold answer, each holding that answer's
    aliases -- the same shape ASQA's ``qa_pairs`` short answers took, so the
    correctness function is literally the same one.
    """
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(ensure_qampari()):
        docs = sample.get("docs") or []
        passages = [
            {"title": _clean(d.get("title", "")), "text": _clean(d.get("text", ""))}
            for d in docs[:top_k]
            if _clean(d.get("text", ""))
        ]
        gold: list[tuple[str, ...]] = []
        for answer in sample.get("answers") or []:
            aliases = tuple(
                dict.fromkeys(_clean(a) for a in (answer if isinstance(answer, list) else [answer]))
            )
            aliases = tuple(a for a in aliases if a)
            if aliases:
                gold.append(aliases)
        question = _clean(sample.get("question", ""))
        if not passages or not gold or not question:
            continue
        rows.append(
            {
                "query_id": str(sample.get("id", index)),
                "question": question,
                "gold_answers": gold,
                "passages": passages,
            }
        )
    return rows


def answer_recall(answer: str, gold_answers: list[tuple[str, ...]], cap: int | None = None) -> float:
    """Share of gold answers whose alias set the answer contains.

    The pre-registered correctness metric. ``cap`` mirrors ALCE's ``rec_top5``:
    QAMPARI questions can carry dozens of gold answers, and an uncapped recall
    makes those questions unwinnable for any answer of reasonable length.
    """
    if not gold_answers:
        return 0.0
    haystack = answer.lower()
    hits = sum(1 for aliases in gold_answers if any(a.lower() in haystack for a in aliases))
    if cap is None:
        return hits / len(gold_answers)
    return min(cap, hits) / min(cap, len(gold_answers))
