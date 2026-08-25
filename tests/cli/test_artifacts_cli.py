import hashlib
import json
from pathlib import Path

from evidence_rag.cli.artifacts import main


def _write_manifest(tmp_path: Path, source: Path) -> Path:
    payload = source.read_bytes()
    manifest = {
        "schema_version": "evidence-rag.artifacts.v1",
        "assets": [
            {
                "asset_id": "fixture",
                "availability": "upstream-public",
                "files": [
                    {
                        "name": "fixture.bin",
                        "bytes": len(payload),
                        "size_status": "verified",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "download_url": source.as_uri(),
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_verify_reports_matching_size_and_sha(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen artifact\n")
    manifest = _write_manifest(tmp_path, source)

    exit_code = main(
        (
            "--manifest",
            str(manifest),
            "verify",
            "--asset",
            "fixture",
            "--file",
            str(source),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert exit_code == 0
    assert output["status"] == "PASS"
    assert output["bytes_match"] is True
    assert output["sha256_match"] is True


def test_verify_fails_closed_on_modified_file(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    manifest = _write_manifest(tmp_path, source)
    source.write_bytes(b"modified")

    exit_code = main(
        (
            "--manifest",
            str(manifest),
            "verify",
            "--asset",
            "fixture",
            "--file",
            str(source),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert exit_code == 1
    assert output["status"] == "FAIL"
    assert output["sha256_match"] is False


def test_download_is_atomic_and_verifies_before_publish(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"download fixture")
    manifest = _write_manifest(tmp_path, source)
    destination = tmp_path / "downloads" / "fixture.bin"

    exit_code = main(
        (
            "--manifest",
            str(manifest),
            "download",
            "--asset",
            "fixture",
            "--output",
            str(destination),
        )
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert exit_code == 0
    assert output["status"] == "PASS"
    assert destination.read_bytes() == b"download fixture"
    assert not destination.with_suffix(".bin.part").exists()
