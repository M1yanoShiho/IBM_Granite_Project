import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ARTIFACT_MANIFEST.json"

DERIVED_HASHES = {
    "selector-seed13": "86622bd9ab6391c9eb560133b01b0cf3744c3706638ff8b0bd38925b84bf72bf",
    "grc-adapter-seed13": "492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707",
    "grc-adapter-seed42": "96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c",
    "grc-adapter-seed73": "d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c",
}

BASE_MODELS = {
    "granite-embedding-r2",
    "granite-reranker-r2",
    "granite-4.1-3b",
    "nli-deberta-v3-base",
    "provence-reranker",
    "true-nli-t5-xxl",
    "minicheck-flan-t5-large",
}


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_registers_final_models_with_frozen_identity() -> None:
    raw = _manifest()
    assets = {asset["asset_id"]: asset for asset in raw["assets"]}  # type: ignore[index]

    assert raw["schema_version"] == "evidence-rag.artifacts.v1"
    assert BASE_MODELS <= assets.keys()
    assert DERIVED_HASHES.keys() <= assets.keys()

    for asset_id, digest in DERIVED_HASHES.items():
        asset = assets[asset_id]
        assert asset["availability"] == "restricted-not-published"
        assert asset["download_url"] is None
        assert asset["files"][0]["sha256"] == digest

    selector_file = assets["selector-seed13"]["files"][0]
    assert selector_file["bytes"] == 737_731_768
    assert selector_file["size_status"] == "verified"

    for seed in (13, 42, 73):
        adapter = assets[f"grc-adapter-seed{seed}"]
        assert adapter["version"] == f"gr-c-v1-seed{seed}"
        assert adapter["files"][0]["bytes"] == 62_332_992
        assert adapter["files"][0]["size_status"] == "verified-shared-preserved-copy"
        assert adapter["files"][1]["bytes"] == 1_274
        assert adapter["files"][1]["size_status"] == "verified-shared-preserved-copy"


def test_manifest_distinguishes_upstream_licenses_and_redistribution_boundaries() -> None:
    assets = {asset["asset_id"]: asset for asset in _manifest()["assets"]}  # type: ignore[index]

    for asset_id in BASE_MODELS:
        asset = assets[asset_id]
        assert asset["revision"]
        assert len(asset["config_sha256"]) == 64
        assert asset["source_url"].startswith("https://")
        assert asset["license"]["identifier"]
        assert all(file["bytes"] > 0 for file in asset["files"])
        assert all(len(file["sha256"]) == 64 for file in asset["files"])

    assert assets["provence-reranker"]["redistribution"] == "link-only-restricted"
    assert assets["rgb-noise-source"]["license"]["identifier"] == "CC-BY-NC-SA-4.0"
    assert assets["kilt-nq-source"]["redistribution"] == "not-redistributed-mixed-terms"
    assert assets["alce-asqa-source"]["redistribution"] == "not-redistributed-mixed-terms"
    for asset_id in (
        "hotpotqa-distractor-source",
        "musique-full-source",
        "rgb-noise-source",
    ):
        asset = assets[asset_id]
        assert len(asset["revision"]) == 40
        assert asset["revision"] in asset["files"][0]["download_url"]


def test_manifest_matches_the_canonical_seed13_runtime_identity() -> None:
    assets = {asset["asset_id"]: asset for asset in _manifest()["assets"]}  # type: ignore[index]
    runtime = json.loads((ROOT / "configs/models/final_seed13.json").read_text(encoding="utf-8"))

    assert runtime["retriever"]["revision"] == assets["granite-embedding-r2"]["revision"]
    assert runtime["retriever"]["config_sha256"] == assets["granite-embedding-r2"][
        "config_sha256"
    ]
    assert runtime["selector"]["revision"] == assets["nli-deberta-v3-base"]["revision"]
    assert runtime["selector"]["base_config_sha256"] == assets["nli-deberta-v3-base"][
        "config_sha256"
    ]
    assert runtime["selector"]["checkpoint_sha256"] == assets["selector-seed13"]["files"][
        0
    ]["sha256"]
    assert runtime["generator"]["base_revision"] == assets["granite-4.1-3b"]["revision"]
    assert runtime["generator"]["base_config_sha256"] == assets["granite-4.1-3b"][
        "config_sha256"
    ]
    assert runtime["generator"]["adapter_weights_sha256"] == assets[
        "grc-adapter-seed13"
    ]["files"][0]["sha256"]
    assert runtime["generator"]["verifier_revision"] == assets["true-nli-t5-xxl"][
        "revision"
    ]


def test_manifest_and_models_doc_publish_no_personal_or_hpc_paths() -> None:
    text = MANIFEST.read_text(encoding="utf-8") + (ROOT / "docs/models-and-data.md").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "/" + "scratch/",
        "/user/" + "work/",
        "/" + "Users/",
        "fl" + "25387",
        "ba" + "25966",
        "bp1-" + "login",
    )

    assert all(value not in text for value in forbidden)
