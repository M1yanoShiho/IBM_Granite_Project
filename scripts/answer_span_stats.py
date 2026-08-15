"""Gold answer length, in words, per dataset -- a premise check, not a result.

R15's second inference assumed NQ answers span more words than 2Wiki's, which would make
the answer string likelier to straddle a chunk boundary. Running this before the main
metric killed that premise: NQ medians 2.0 words (mean 2.21, p90 4) and 2Wiki medians the
same 2.0 (mean 2.35, p90 4), so 2Wiki's answers are if anything the longer ones. The
threshold R15 then found sits on the *source passage* length instead -- dpr-w100 ships
fixed 100-word passages, so a chunk size below 100 splits every one of them.

Kept as a script rather than a one-off because the same premise gets assumed whenever a
chunking result is carried to a new dataset, and it is cheap to check.

    python scripts/answer_span_stats.py runs/A/gold_cases.jsonl runs/B/gold_cases.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# `GoldCase.reference_answers` is the field the loaders write; the others are accepted so
# older bundles and hand-built fixtures still report instead of failing.
ANSWER_FIELDS = ("reference_answers", "answers", "reference_answer", "answer")


def _answers(record: dict[str, Any]) -> list[str]:
    for field in ANSWER_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [item for item in value if isinstance(item, str)]
    return []


def _word_counts(path: Path) -> tuple[list[int], int, int]:
    """Return (shortest answer length per case, records seen, records with no answer)."""
    lengths: list[int] = []
    n_records = n_empty = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_records += 1
            answers = _answers(json.loads(line))
            if not answers:
                n_empty += 1
                continue
            # The shortest accepted answer is the one a substring match is likeliest to
            # find, so it is the relevant span for a cut-by-chunking question.
            lengths.append(min(len(answer.split()) for answer in answers))
    return lengths, n_records, n_empty


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold_cases", nargs="+", type=Path)
    arguments = parser.parse_args(argv)

    for path in arguments.gold_cases:
        lengths, n_records, n_empty = _word_counts(path)
        print(f"\n{path}")
        print(f"  records {n_records}  with-answer {len(lengths)}  without {n_empty}")
        if not lengths:
            with path.open(encoding="utf-8") as handle:
                keys = sorted(json.loads(handle.readline()))
            print(f"  !! no answer field found; available keys: {keys}")
            continue
        lengths.sort()
        print(
            f"  words/answer  median {statistics.median(lengths)}  "
            f"mean {statistics.mean(lengths):.2f}  "
            f"p90 {lengths[int(len(lengths) * 0.9)]}  max {max(lengths)}"
        )
        print(f"  share >1 word: {sum(1 for x in lengths if x > 1) / len(lengths):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
