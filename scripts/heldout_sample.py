"""Draw the held-out samples and write an auditable manifest.

Run and commit this **before** the held-out job. The manifest fixes exactly which
records are evaluated, so the sample cannot be re-drawn after seeing anything.

    PYTHONPATH=src python scripts/heldout_sample.py --out configs/heldout-sample.json

Design of each draw:

* **hotpotqa** -- 400 query ids, uniform, matching the calibration sample size.
* **musique-full** -- 400 **ids**, both variants of each, so the answerable and
  unanswerable subsets are 400 apiece and the pairing survives. Sampling 400
  *records* instead would break the pair structure that MuSiQue-Full exists to
  provide: the same question with and without its support.
* **rgb** -- all 300. The set is already smaller than the calibration sample.
* **rgb-counterfactual** -- all 100.

No metric is computed here and no answer text is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heldout_data import ALL_SETS, load  # noqa: E402

TARGET = {"hotpotqa": 400, "musique-full": 400, "rgb": None, "rgb-counterfactual": None}
"""``None`` means take the whole set. The MuSiQue figure counts IDS, not records."""


def draw(name: str, rows: list[dict], rng: random.Random) -> list[str]:
    target = TARGET[name]
    if name == "musique-full":
        # Sample IDS and keep both variants, so answerable and unanswerable are
        # matched on the same questions rather than being two unrelated samples.
        by_id: dict[str, list[str]] = {}
        for row in rows:
            base = str(row["query_id"]).rsplit("#", 1)[0]
            by_id.setdefault(base, []).append(str(row["query_id"]))
        complete = sorted(base for base, variants in by_id.items() if len(variants) == 2)
        chosen = sorted(rng.sample(complete, min(target or len(complete), len(complete))))
        return sorted(qid for base in chosen for qid in by_id[base])
    ids = sorted(str(row["query_id"]) for row in rows)
    if target is None or target >= len(ids):
        return ids
    return sorted(rng.sample(ids, target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    manifest: dict[str, object] = {"seed": args.seed, "datasets": {}}
    for name in ALL_SETS:
        rows = load(name)
        rng = random.Random(f"{args.seed}:{name}")  # per-set stream: one set cannot shift another
        ids = draw(name, rows, rng)
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        entry = {
            "population": len(rows),
            "sampled_records": len(ids),
            "sha256": digest,
            "query_ids": ids,
        }
        if name == "musique-full":
            entry["sampled_ids"] = len({i.rsplit("#", 1)[0] for i in ids})
        manifest["datasets"][name] = entry  # type: ignore[index]
        print(
            f"{name:20s} population={len(rows):5d} -> sampled={len(ids):5d}  sha256={digest[:16]}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
