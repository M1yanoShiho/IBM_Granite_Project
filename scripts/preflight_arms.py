"""Refuse a multi-arm run before it spends anything, if any arm's stored manifest is stale.

    export PYTHONPATH=src
    python scripts/preflight_arms.py configs/experiments/a.toml configs/experiments/b.toml

`ExperimentWorkflow.prepare` validates an arm's stored `run_manifest.json` against the index
files on disk -- but only when that arm's turn comes. In a sequential multi-arm job the last
arm's check runs after every earlier arm has finished computing. Job 18235971 died exactly
there: arm 2's prepare rejected `runs/e2-gate-off` over an upstream-hash mismatch that was
fully decidable at second zero, and by then arm 1 had spent 5h19m on the GPU.

Nothing about that check needs a GPU, a corpus, or arm 1's output. This runs it for every arm
up front, reports all of them rather than stopping at the first, and exits non-zero if any
would be rejected. Cost is a few file hashes on a login node.

It does NOT check what only exists after `prepare` -- dataset signatures, corpus compatibility,
selector wiring. It checks the one failure this job class has actually hit.

Verdicts:
  OK            recorded hashes match the index on disk; prepare will accept this arm.
  FRESH         no run_manifest.json; prepare will write one. Nothing to verify.
  INDEX ABSENT  run_manifest.json exists but its index/ does not. prepare will rebuild the
                index and compare the rebuilt file against a hash recorded for a file that no
                longer exists. It matches only if the current code serialises identically to
                the code that prepared the arm, which is not knowable from here -- so this
                fails, deliberately. If a clean rebuild is what you meant, delete
                run_manifest.json (and its .metadata.json) too: the sidecar is what pins the
                deleted file.
  MISMATCH      recorded hashes differ from the index on disk; prepare WILL reject this arm.
                Before re-running anything, check whether the index actually changed --
                scripts/verify_legacy_index_identity.py settles that for BM25 arms, and a
                schema reshape does not require re-running the arm at all.
"""

import argparse
import json
import sys
import tomllib
from hashlib import sha256
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidence_rag.evaluation.experiment import INDEX_ARTIFACTS  # noqa: E402
from evidence_rag.infrastructure.artifacts import metadata_filename  # noqa: E402

RUN_MANIFEST = "run_manifest.json"


def _output_directory(config: Path) -> Path:
    """Resolve `[output] directory` the way the workflow does -- relative to the config file."""

    config = config.resolve()
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    return (config.parent / raw["output"]["directory"]).resolve()


def _verdict(directory: Path) -> tuple[str, list[str]]:
    if not (directory / RUN_MANIFEST).is_file():
        return "FRESH", []

    sidecar = directory / metadata_filename(RUN_MANIFEST)
    if not sidecar.is_file():
        return "MISMATCH", [f"{RUN_MANIFEST} has no {sidecar.name}; the store cannot read it"]

    recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("upstream_artifact_hashes", {})
    details: list[str] = []
    absent = False
    mismatched = False
    for filename in sorted(INDEX_ARTIFACTS):
        path = directory / filename
        if not path.is_file():
            absent = True
            details.append(f"{filename}: absent on disk, recorded {recorded.get(filename, '-')}")
            continue
        found = sha256(path.read_bytes()).hexdigest()
        if found != recorded.get(filename):
            mismatched = True
            details.append(f"{filename}: on disk {found}, recorded {recorded.get(filename, '-')}")
        else:
            details.append(f"{filename}: same ({found[:16]}...)")

    if mismatched:
        return "MISMATCH", details
    if absent:
        return "INDEX ABSENT", details
    return "OK", details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", type=Path)
    arguments = parser.parse_args(argv)

    failed: list[str] = []
    for config in arguments.configs:
        directory = _output_directory(config)
        verdict, details = _verdict(directory)
        print(f"[{verdict:^12}] {config}  ->  {directory}")
        for line in details:
            print(f"               {line}")
        if verdict in {"MISMATCH", "INDEX ABSENT"}:
            failed.append(f"{config} ({verdict})")

    if failed:
        # Every arm is reported above before this fires: the point of a preflight is to learn
        # everything that is broken in one login-node minute, not one arm per submission.
        print(f"\nPREFLIGHT FAILED -- {len(failed)} arm(s) would be rejected by prepare:")
        for entry in failed:
            print(f"  {entry}")
        print("Do not submit. See this file's docstring for what each verdict means.")
        return 1

    print(f"\nPREFLIGHT OK -- {len(arguments.configs)} arm(s); no stored manifest is stale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
