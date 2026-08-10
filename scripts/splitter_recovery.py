"""How many of QAMPARI's 18 splitter failures does the fallback recover?

Deterministic, CPU-only, and it touches no evaluation: the drafts are already
recorded. The failures are identified from the run log, their drafts are taken
from the baseline arm (which never uses the splitter and therefore kept every
query), and the fallback is run over that recorded text.

    PYTHONPATH=src python scripts/splitter_recovery.py \
        --log local/results/qampari/qampari-run-18326078.out \
        --answers local/results/qampari/baseline.jsonl
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

from evidence_rag.generator.claim_splitter import ClaimSplitter  # noqa: E402

ERROR_LINE = re.compile(r"^\[error\] (\S+) (\S+?):")
"""``[error] <arm> <query_id>: <message>``. The id is not numeric on QAMPARI
(``675__wikidata_simple__dev``), so it is matched as a non-greedy token rather
than as digits."""


class AlwaysFails:
    """Stands in for the malformed response the real splitter got.

    The exact bytes Granite emitted are not recorded -- only that they failed to
    parse. Any unparseable payload exercises the same fallback, so this measures
    the floor's coverage of those queries, not a reconstruction of the failure.
    """

    def generate(self, prompt: str) -> str:
        return "not json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--verify-arm", type=Path, required=True)
    parser.add_argument("--show", type=int, default=4)
    args = parser.parse_args()

    def ids(path: Path) -> dict[str, str]:
        return {
            json.loads(line)["query_id"]: json.loads(line)["answer"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    drafts = ids(args.answers)
    survived = set(ids(args.verify_arm))
    # The complete failure set: the baseline never uses the splitter, so a query
    # it has and a verify arm lacks is exactly a splitter failure. The run log
    # only prints the first five per arm, so it names a subset.
    failed = set(drafts) - survived

    logged: set[str] = set()
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ERROR_LINE.match(line)
        if match:
            logged.add(match.group(2))

    print(f"failures, from the arm files : {len(failed)}")
    print(f"  ... of which the log names : {len(logged & failed)} "
          f"(the handler prints only the first five per arm)")
    if logged - failed:
        print(f"  ... logged but not missing : {sorted(logged - failed)}")
    print(f"  with a recorded draft to split: "
          f"{sum(1 for q in failed if drafts.get(q, '').strip())}")

    splitter = ClaimSplitter(llm=AlwaysFails())
    recovered = 0
    empty_draft = 0
    claim_counts: list[int] = []
    shown = 0
    for query_id in sorted(failed):
        draft = drafts.get(query_id, "")
        if not draft.strip():
            empty_draft += 1
            continue
        claims = splitter.split(draft)
        if claims:
            recovered += 1
            claim_counts.append(len(claims))
        if shown < args.show and claims:
            shown += 1
            print(f"\n  [{query_id}] -> {len(claims)} degraded claims")
            for claim in claims[:3]:
                print(f"      {claim.text[:110]!r}")

    print("\n=== recovery ===")
    print(f"  recovered (produce usable claims) : {recovered}/{len(failed)}")
    print(f"  no draft recorded to work from    : {empty_draft}")
    if claim_counts:
        ordered = sorted(claim_counts)
        print(
            f"  claims per recovered query        : min={min(ordered)} "
            f"median={ordered[len(ordered) // 2]} max={max(ordered)}"
        )
    print(
        "\n  NOTE: every recovered claim is a whole sentence, so atomicity is lost\n"
        "  and each is marked degraded=True. The point is that the query survives\n"
        "  and stays comparable across arms instead of being dropped from the\n"
        "  verify arms while the baseline keeps it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
