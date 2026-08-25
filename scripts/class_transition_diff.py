"""R18: cross-tabulate two `answer_chunk_forensics --dump` files case by case.

R18 reran R16's arm with the candidate pool widened from 50 to 1000 and the class shares
moved a long way -- `retrieval` 83 -> 41, `sibling` 25 -> 64. Read as counts those two
movements look like one thing, and they are not. A case leaves `retrieval` because the
deeper pool finally contains an answer-bearing chunk; a case leaves `other` because that
deeper pool contains a *second* one whose document was selected, which flips the class
without anything new being found. The published share cannot distinguish them, so this
prints the transition matrix that can.

It also reports what the flips did to the reported rank, because `classify_run` returns the
sibling's rank rather than the shallowest answer-bearing one: an `other -> sibling` flip
moves a case's `best_rank` deeper even though the answer was visible at the old rank all
along. That is the artefact R18's anchor check caught, and this quantifies it.

    python scripts/class_transition_diff.py \
        --before results/r16-nq-c60o10-classes.jsonl \
        --after  results/r18-nq-c60o10-topk1000-classes.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

CLASSES = ("sibling", "other", "retrieval", "absent")


def read_dump(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row["query_id"]
            if query_id in rows:
                raise SystemExit(f"duplicate query_id in {path}: {query_id}")
            rows[query_id] = row
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    arguments = parser.parse_args(argv)

    before = read_dump(arguments.before)
    after = read_dump(arguments.after)

    # The populations must be identical, for the same structural reason R18 pre-registered:
    # a deeper pool cannot change which queries are conditional misses. A difference here
    # means the two dumps are not two readings of one experiment, and no transition below
    # would mean anything.
    if set(before) != set(after):
        only_before, only_after = set(before) - set(after), set(after) - set(before)
        raise SystemExit(
            f"populations differ: {len(only_before)} only in --before, "
            f"{len(only_after)} only in --after; the dumps are not comparable"
        )

    transitions = Counter(
        (before[query_id]["class"], after[query_id]["class"]) for query_id in before
    )

    print(f"population: {len(before)} cases, identical in both dumps")
    print()
    header = "  ".join(f"{label:>10}" for label in CLASSES)
    # Built outside the f-string: a backslash inside one is a syntax error on 3.11, which is
    # what CI runs. Locally on 3.12 it parses, so ruff is the only thing that catches this.
    corner = "before \\ after"
    print(f"{corner:<16}{header}{'':>8}total")
    for source in CLASSES:
        row = [transitions.get((source, target), 0) for target in CLASSES]
        cells = "  ".join(f"{n:>10}" for n in row)
        print(f"{source:<16}{cells}{sum(row):>13}")
    totals = [sum(transitions.get((s, t), 0) for s in CLASSES) for t in CLASSES]
    print(f"{'total':<16}{'  '.join(f'{n:>10}' for n in totals)}{sum(totals):>13}")

    moved = {(s, t): n for (s, t), n in transitions.items() if s != t}
    print()
    if not moved:
        print("no case changed class")
    for (source, target), n in sorted(moved.items(), key=lambda item: -item[1]):
        print(f"  {source:>10} -> {target:<10} {n:>4}")

    # The two ways into `sibling`, which the share alone conflates.
    newly_found = transitions.get(("retrieval", "sibling"), 0)
    reclassified = transitions.get(("other", "sibling"), 0)
    stayed = transitions.get(("sibling", "sibling"), 0)
    print()
    print("how `sibling` grew:")
    print(f"  already sibling before      : {stayed}")
    print(f"  newly found (was retrieval) : {newly_found}")
    print(f"  reclassified (was other)    : {reclassified}   <- nothing new was found here")

    # What the flips did to the reported rank. `other -> sibling` can only push it deeper,
    # because the sibling that decides the class sits below the stranger that used to.
    deepened = [
        (query_id, before[query_id]["best_rank"], after[query_id]["best_rank"])
        for query_id in before
        if before[query_id]["class"] == "other"
        and after[query_id]["class"] == "sibling"
        and before[query_id].get("best_rank") is not None
        and after[query_id].get("best_rank") is not None
    ]
    if deepened:
        print()
        print("reported rank moved deeper without the answer moving (other -> sibling):")
        for query_id, old, new in sorted(deepened, key=lambda item: item[1]):  # type: ignore[arg-type,return-value]
            print(f"  {query_id:>8}  rank {old} -> {new}")
        crossed = sum(1 for _, old, new in deepened if old <= 50 < new)  # type: ignore[operator]
        print(f"  of these, crossed the old pool boundary (<=50 -> >50): {crossed}")

    # Present only once the --after dump comes from the R18 fix; older dumps lack the field.
    have_shallow = [row for row in after.values() if row.get("shallowest_rank") is not None]
    if have_shallow:
        gaps = [
            int(row["best_rank"]) - int(row["shallowest_rank"])  # type: ignore[arg-type]
            for row in have_shallow
            if row.get("best_rank") is not None
        ]
        overstated = [gap for gap in gaps if gap > 0]
        print()
        print("class rank vs shallowest answer-bearing rank, in --after:")
        print(f"  cases where the class rank is deeper : {len(overstated)} of {len(gaps)}")
        if overstated:
            print(f"  overstatement: max {max(overstated)}, median {sorted(overstated)[len(overstated) // 2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
