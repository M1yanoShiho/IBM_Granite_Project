"""R012f — cross-backbone agreement over Gate 0B dumps. Login-node safe, no GPU, no model load.

    export PYTHONPATH=src
    python scripts/r012f_backbone_agreement.py results/gate0b/dump-template.jsonl [more.jsonl ...]

Each dump may contain several arms; rows are grouped by their own `model_id`, so one file with
three arms and three files with one arm each give the same answer. External-tier rows (null
`kind`) are dropped: this analysis is about the 0B-2 probe.

Reads only files that already exist. Produces no Gate 0B reading — see the module docstring of
`evidence_rag.relations.backbone_agreement` for why a union arm here is a diagnostic and not a
pre-registered arm.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidence_rag.relations.backbone_agreement import (  # noqa: E402
    GOLD_KIND,
    TWIN_KINDS,
    compare_backbones,
)


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") is None:
                continue  # external tier (0B-1), not this probe
            by_arm[str(row["model_id"])].append(row)
    if not by_arm:
        print("no task-tier rows found — are these external-only dumps?")
        return 1

    print(f"kinds: gold={GOLD_KIND!r} twin={TWIN_KINDS}\narms: {len(by_arm)}\n")
    result = compare_backbones(by_arm)

    print("=== per arm (must reproduce the sweep's own numbers) ===")
    for name, arm in sorted(result["arms"].items()):
        print(
            f"{name}\n    gold_supports_recall {arm['gold_supports_recall']:.4f}"
            f" (n={arm['n_gold']})    twin_not_supported {arm['twin_not_supported_accuracy']:.4f}"
            f" (n={arm['n_twin']})"
        )

    print("\n=== pairwise agreement (the actual RADAR question) ===")
    for (left, right), pair in result["pairwise"].items():
        print(
            f"{left}\n  vs {right}\n"
            f"    n_common {pair['n_common']}  (only-left {pair['n_only_in_a']},"
            f" only-right {pair['n_only_in_b']})\n"
            f"    raw_agreement {pair['raw_agreement']:.4f}   cohen_kappa"
            f" {pair['cohen_kappa']:.4f}\n"
            f"    gold recovered only by left {pair['gold_only_a']}, only by right"
            f" {pair['gold_only_b']}  =>  {'CROSSING' if pair['crossing'] else 'NESTED'}"
        )

    print("\n=== union rule, diagnostic only (SUPPORTS if either arm says SUPPORTS) ===")
    for (left, right), rule in result["union"].items():
        print(
            f"{left} + {right}\n"
            f"    gold_supports_recall {rule['gold_supports_recall']:.4f}"
            f"   twin_not_supported {rule['twin_not_supported_accuracy']:.4f}"
        )
    print(
        "\nGate 0B thresholds for reference: gold_supports_recall >= .85 AND"
        " twin_not_supported_accuracy >= .70.\nA union that clears both is NOT a pass — it is an"
        " un-pre-registered arm and would need an amendment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
