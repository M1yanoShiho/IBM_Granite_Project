"""List, download, and verify artifacts declared by the public release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "ARTIFACT_MANIFEST.json"


class ArtifactError(ValueError):
    """Raised when an artifact request is incomplete or unsafe to execute."""


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not read artifact manifest: {path}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "evidence-rag.artifacts.v1":
        raise ArtifactError("unsupported artifact manifest schema")
    assets = raw.get("assets")
    if not isinstance(assets, list):
        raise ArtifactError("artifact manifest has no assets list")
    return raw


def _asset(manifest: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    for raw in manifest["assets"]:
        if isinstance(raw, Mapping) and raw.get("asset_id") == asset_id:
            return raw
    raise ArtifactError(f"unknown asset: {asset_id}")


def _file(asset: Mapping[str, Any], file_name: str | None) -> Mapping[str, Any]:
    files = asset.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError(f"asset {asset.get('asset_id')} has no downloadable files")
    if file_name is None:
        if len(files) != 1:
            raise ArtifactError("multi-file asset requires --file-name")
        selected = files[0]
    else:
        selected = next(
            (
                entry
                for entry in files
                if isinstance(entry, Mapping) and entry.get("name") == file_name
            ),
            None,
        )
        if selected is None:
            raise ArtifactError(f"asset has no file named {file_name}")
    if not isinstance(selected, Mapping):
        raise ArtifactError("invalid artifact file record")
    digest = selected.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ArtifactError("artifact file has no valid SHA-256")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verification(asset_id: str, record: Mapping[str, Any], path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ArtifactError(f"artifact file does not exist: {path}")
    observed_bytes = path.stat().st_size
    observed_sha256 = _sha256(path)
    expected_bytes = record.get("bytes")
    bytes_match = expected_bytes is None or observed_bytes == expected_bytes
    sha256_match = observed_sha256 == record["sha256"]
    return {
        "asset_id": asset_id,
        "file": record.get("name"),
        "path": str(path),
        "status": "PASS" if bytes_match and sha256_match else "FAIL",
        "bytes": observed_bytes,
        "expected_bytes": expected_bytes,
        "bytes_match": bytes_match,
        "sha256": observed_sha256,
        "expected_sha256": record["sha256"],
        "sha256_match": sha256_match,
    }


def _download(
    *, asset_id: str, asset: Mapping[str, Any], record: Mapping[str, Any], output: Path
) -> dict[str, object]:
    if output.exists():
        raise ArtifactError(f"refusing to overwrite existing output: {output}")
    url = record.get("download_url") or asset.get("download_url")
    if not isinstance(url, str) or not url:
        raise ArtifactError(f"asset {asset_id} is not publicly downloadable")
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".part", dir=output.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        verification = _verification(asset_id, record, temporary)
        verification["path"] = str(output)
        verification["download_url"] = url
        if verification["status"] != "PASS":
            return verification
        os.replace(temporary, output)
        return verification
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registered assets and availability")

    verify = commands.add_parser("verify", help="verify one local file")
    verify.add_argument("--asset", required=True)
    verify.add_argument("--file-name")
    verify.add_argument("--file", required=True, type=Path)

    download = commands.add_parser("download", help="download and verify one public file")
    download.add_argument("--asset", required=True)
    download.add_argument("--file-name")
    download.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = _load_manifest(arguments.manifest)
        if arguments.command == "list":
            result: object = [
                {
                    "asset_id": entry.get("asset_id"),
                    "kind": entry.get("kind"),
                    "availability": entry.get("availability"),
                    "version": entry.get("version") or entry.get("revision"),
                }
                for entry in manifest["assets"]
                if isinstance(entry, Mapping)
            ]
            exit_code = 0
        else:
            asset = _asset(manifest, arguments.asset)
            record = _file(asset, arguments.file_name)
            if arguments.command == "verify":
                result = _verification(arguments.asset, record, arguments.file)
            else:
                result = _download(
                    asset_id=arguments.asset,
                    asset=asset,
                    record=record,
                    output=arguments.output,
                )
            exit_code = 0 if result["status"] == "PASS" else 1
    except ArtifactError as error:
        print(json.dumps({"status": "ERROR", "message": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
