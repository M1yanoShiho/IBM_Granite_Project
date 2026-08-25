"""Measure the sentence-splitter change in isolation, on fixed inputs.

The splitter delta is invisible against G9-minus-G8, because generation churns
11% of answers between jobs under fixed seed and code. So it is measured
directly: run the old rule and the new rule over the *same* recorded text and
diff the boundaries. Deterministic, CPU-only, no model weights.

    python scripts/g9_splitter_delta.py --answers local/results/g8

What is measured: `claim_splitter._sentence_spans`, which is the fallback anchor
in `_locate_span` -- it decides which answer region a paraphrased claim is
attached to. The old implementation knew 10 abbreviations; the new one is the
contract's rule and knows ~50.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_rag.contracts.models import sentence_spans as new_spans  # noqa: E402

_OLD_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc"})


def old_spans(text: str) -> list[tuple[int, int]]:
    """`claim_splitter._sentence_spans` exactly as it stood before G9."""
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".!?":
            continue
        if character == ".":
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if previous.isdigit() and following.isdigit():
                continue
            if previous.isupper() and following.isupper():
                continue
            if re.search(r"(?:\b[A-Z]\.)+[A-Z]$", text[start:index]):
                continue
            word_match = re.search(r"([A-Za-z]+)$", text[start:index])
            if word_match is not None and word_match.group(1).casefold() in _OLD_ABBREVIATIONS:
                continue
        end = index + 1
        if text[start:end].strip():
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True, help="dir of arm jsonl files")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    texts: list[tuple[str, str, str]] = []
    for path in sorted(args.answers.glob("*.jsonl")):
        arm = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["answer"].strip():
                texts.append((arm, record["query_id"], record["answer"]))
    if not texts:
        raise SystemExit(f"no answers found under {args.answers}")

    changed: list[tuple[str, str, str, int, int]] = []
    old_total = new_total = 0
    for arm, query_id, text in texts:
        before = [text[a:b].strip() for a, b in old_spans(text)]
        after = [text[a:b] for a, b in new_spans(text)]
        old_total += len(before)
        new_total += len(after)
        if before != after:
            changed.append((arm, query_id, text, len(before), len(after)))

    print(f"answers examined: {len(texts)}")
    print(f"answers whose sentence boundaries change: {len(changed)} "
          f"({len(changed) / len(texts):.3%})")
    print(f"sentences: old {old_total} -> new {new_total}  (delta {new_total - old_total})")
    by_arm: dict[str, int] = {}
    for arm, *_ in changed:
        by_arm[arm] = by_arm.get(arm, 0) + 1
    print(f"by arm: {dict(sorted(by_arm.items()))}")

    print(f"\n--- first {args.examples} changed, old || new ---")
    for arm, query_id, text, n_old, n_new in changed[: args.examples]:
        print(f"\n[{arm} {query_id}] {n_old} -> {n_new} sentences")
        print(f"  old: {[text[a:b].strip() for a, b in old_spans(text)]}")
        print(f"  new: {[text[a:b] for a, b in new_spans(text)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
