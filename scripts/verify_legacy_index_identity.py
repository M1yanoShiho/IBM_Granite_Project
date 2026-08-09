"""Prove a pre-d5f7908 run directory and its rebuilt index describe the SAME index.

    python scripts/verify_legacy_index_identity.py runs/e2-gate-off

`d5f7908` moved BM25's `k1`/`b` from flat `IndexManifest` fields into a nested `parameters`
dict. `_index_signature` hashes that payload, so the same corpus with the same k1/b started
producing a different `index_signature` -- and every run prepared before the refactor now fails
`ArtifactStore.read_manifest` with an opaque

    upstream artifact hashes mismatch ... expected {...} found {...}

because `run_manifest.json.metadata.json` records the sha256 of the OLD manifest FILE, which was
deleted when the index was rebuilt. That error says the bytes differ. It cannot say whether the
INDEX differs, and those are not the same question -- the difference may be a schema reshape or
a genuinely different retriever. Answering it by argument is how a run gets compared against a
pool it was never built on.

This answers it with a number. The old file is gone, but its content is fully determined: the
pre-refactor schema is known, the semantic values survive in the rebuilt manifest, and
`index_signature` survives in `run_manifest.json`. Reconstruct those bytes, hash them, and
compare against what the sidecar recorded. A match pins every field of the deleted file at once
-- if any semantic value had also changed, the reconstruction would hash to something else.

PASS means the two encodings describe one index, so artifacts either side of the refactor are
comparable and the arms need not be re-run. FAIL means something beyond the reshape moved and
the comparison is not safe. Neither verdict licenses reusing the directory through
`read_manifest`: that guard stays broken by design, because the file it hashes no longer exists.

Scope: BM25 only. The pre-refactor schema spelled its parameters as the literal fields `k1` and
`b`; no other retriever existed to have its own spelling, and guessing one would defeat the
point. Login-node safe -- reads two small JSON files and never loads the corpus snapshot.

Worked example (`runs/e2-gate-off`, verified on bp1 2026-08-09):
    reconstructed  947844363ceee8fddf25c0323305fa32ac3333a7f70cf4ec1a6d23b9f244f5b0
    sidecar        947844363ceee8fddf25c0323305fa32ac3333a7f70cf4ec1a6d23b9f244f5b0
"""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

MANIFEST = "index/index_manifest.json"
SNAPSHOT = "index/corpus_snapshot.json"

# `_canonical_json` in evidence_rag.retriever.indexing, and the trailing newline `write_index`
# appends. Duplicated rather than imported on purpose: importing the current serialiser to check
# the current serialiser's output would make this tautological. These are the bytes as they were
# written in July 2026, pinned here.
_DUMP: dict[str, Any] = {
    "ensure_ascii": True,
    "separators": (",", ":"),
    "sort_keys": True,
    "allow_nan": False,
}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, **_DUMP) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SystemExit(f"unable to read {path}: {error}") from error
    except ValueError as error:
        raise SystemExit(f"invalid JSON in {path}: {error}") from error


def _legacy_bytes(current: dict[str, Any], legacy_signature: str) -> bytes:
    """The pre-d5f7908 manifest: same values, `parameters` spelled flat."""

    parameters = current["parameters"]
    unexpected = set(parameters) - {"k1", "b"}
    if unexpected:
        raise SystemExit(
            f"pre-refactor schema had only k1/b as flat fields; cannot place {sorted(unexpected)}"
        )
    return _canonical(
        {
            "schema_version": current["schema_version"],
            "implementation": current["implementation"],
            "implementation_version": current["implementation_version"],
            "corpus_signature": current["corpus_signature"],
            "k1": parameters["k1"],
            "b": parameters["b"],
            "snapshot_filename": current["snapshot_filename"],
            # From run_manifest.json -- the one field the rebuilt file cannot supply, and the
            # whole reason the reconstruction is a test rather than a restatement.
            "index_signature": legacy_signature,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args(argv)

    root = arguments.run_directory
    run_manifest = _read_json(root / "run_manifest.json")
    sidecar = _read_json(root / "run_manifest.json.metadata.json")
    current = _read_json(root / MANIFEST)

    if current["implementation"] != "bm25":
        raise SystemExit(f"BM25 only; this directory is {current['implementation']!r}")

    recorded = sidecar.get("upstream_artifact_hashes") or {}
    if MANIFEST not in recorded:
        raise SystemExit(f"{root} records no upstream hash for {MANIFEST}; nothing to verify")

    lines: list[str] = []
    verdicts: list[bool] = []

    # The corpus snapshot is checked first and separately. If its bytes moved, the manifest
    # reconstruction is meaningless -- a matching manifest over a different corpus would be a
    # far worse outcome than an honest mismatch.
    snapshot_disk = sha256((root / SNAPSHOT).read_bytes()).hexdigest()
    snapshot_ok = snapshot_disk == recorded.get(SNAPSHOT)
    verdicts.append(snapshot_ok)
    lines.append(f"corpus snapshot  {'same bytes' if snapshot_ok else 'DIFFERS'}  {snapshot_disk}")

    for field, current_key in (
        ("implementation", "implementation"),
        ("implementation_version", "implementation_version"),
        ("corpus_signature", "corpus_signature"),
    ):
        old = run_manifest.get(f"index_{field}", run_manifest.get(field))
        same = old == current[current_key]
        verdicts.append(same)
        lines.append(f"{field:<22} {'same' if same else 'DIFFERS'}  {current[current_key]}")

    legacy_signature = run_manifest["index_signature"]
    lines.append(f"index_signature  legacy {legacy_signature}")
    lines.append(f"index_signature  rebuilt {current['index_signature']}")

    digest = sha256(_legacy_bytes(current, legacy_signature)).hexdigest()
    reconstructed_ok = digest == recorded[MANIFEST]
    verdicts.append(reconstructed_ok)
    lines.append(f"reconstructed    {digest}")
    lines.append(f"sidecar recorded {recorded[MANIFEST]}")

    passed = all(verdicts)
    print("\n".join(lines))
    print(
        f"\n{'PASS' if passed else 'FAIL'}: {root} -- "
        + (
            "the deleted manifest was the same index under the pre-d5f7908 schema; "
            "artifacts either side of the refactor are comparable."
            if passed
            else "reconstruction does not match; something beyond the k1/b reshape moved. "
            "Do NOT compare artifacts across this boundary."
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
