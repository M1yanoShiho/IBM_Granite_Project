from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "experiments/generator/frozen_provenance.json"


def test_generator_provenance_freezes_recipe_and_all_three_adapters() -> None:
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "evidence-rag.generator-provenance.v1"
    assert manifest["recipe"]["id"] == "gr-c"
    assert manifest["recipe"]["selection"] == "FROZEN_BEFORE_FORMAL_EVALUATION"
    assert set(manifest["adapters"]) == {"13", "42", "73"}
    assert manifest["adapters"]["13"]["weights_sha256"] == (
        "492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707"
    )
    assert manifest["adapters"]["42"]["weights_sha256"] == (
        "96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c"
    )
    assert manifest["adapters"]["73"]["weights_sha256"] == (
        "d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c"
    )
    assert all(item["reload_status"] == "PASS" for item in manifest["adapters"].values())


def test_generator_provenance_has_recoverable_sources_and_no_personal_path() -> None:
    text = PROVENANCE.read_text(encoding="utf-8")
    manifest = json.loads(text)

    assert manifest["source"]["archive_ref"] == "research-archive-2026-08-25"
    assert manifest["source"]["recipe_manifest_sha256"] == (
        "a25c273c5d8cd11af7fe4c002ad110bdbc3592d5d148f891cac24c9afe66ef32"
    )
    assert manifest["source"]["three_seed_manifest_sha256"] == (
        "e7101db9c5bbe9e061230095b364c6d658a4766c1657d01689d1d00f5a79813f"
    )
    assert "/scratch" + "/" not in text
    assert "/user" + "/work/" not in text
