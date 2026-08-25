from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY_MAP.md",
    ROOT / "docs/README.md",
    ROOT / "docs/architecture.md",
    ROOT / "docs/setup.md",
    ROOT / "docs/reproduction.md",
    ROOT / "docs/results.md",
    ROOT / "docs/limitations.md",
    ROOT / "docs/models-and-data.md",
    ROOT / "docs/frontend-integration.md",
    ROOT / "docs/three-module-runtime-handoff.md",
    ROOT / "docs/model-cards/selector.md",
    ROOT / "docs/model-cards/generator.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "AUTHORS.md",
)

LINK_CHECK_DOCS = (
    *PUBLIC_DOCS,
    *sorted((ROOT / "docs/research").glob("*.md")),
    *sorted((ROOT / "experiments").glob("**/README.md")),
    *sorted((ROOT / "results").glob("**/*.md")),
)

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_emoji(text: str) -> bool:
    return any(
        0x1F000 <= ord(character) <= 0x1FAFF
        or 0x2600 <= ord(character) <= 0x27BF
        or ord(character) == 0xFE0F
        for character in text
    )


def test_public_release_documents_exist() -> None:
    expected = (*PUBLIC_DOCS, ROOT / "CITATION.cff", ROOT / "LICENSE")

    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)


def test_readme_is_a_complete_canonical_entrypoint() -> None:
    text = _text(ROOT / "README.md")

    for required in (
        "Hybrid Retriever",
        "trained NLI Selector",
        "grounded GR-C Generator",
        "python -m pip install -e '.[dev,api,data-prep]'",
        "evidence-rag-smoke",
        "evidence-rag-serve",
        "docs/setup.md",
        "docs/reproduction.md",
        "docs/results.md",
        "docs/limitations.md",
        "CITATION.cff",
    ):
        assert required in text

    assert "Query2Doc + GraniteDenseRetriever" not in text
    assert "CorroborationSelector" not in text


def test_readme_mermaid_is_accessible_and_theme_neutral() -> None:
    text = _text(ROOT / "README.md")

    assert "```mermaid\nflowchart LR" in text
    assert "accTitle:" in text
    assert "accDescr:" in text
    assert "classDef" in text
    assert "%%{init" not in text
    assert "\nstyle " not in text
    assert not _contains_emoji(text)


def test_results_and_limitations_preserve_frozen_claim_boundaries() -> None:
    text = _text(ROOT / "docs/results.md") + "\n" + _text(ROOT / "docs/limitations.md")

    for required in (
        "misleading-evidence",
        "evidence-level",
        "blind answer gate",
        "KEEP_TOPK10",
        "ordinary Experiment 05",
        "mostly inactive",
        "NOT SUPPORTED",
        "FINAL PASS",
        "protocol execution",
    ):
        assert required in text


def test_citation_metadata_is_machine_parseable() -> None:
    citation = json.loads(_text(ROOT / "CITATION.cff"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["title"] == "Evidence RAG"
    assert citation["version"] == "1.0.0-dissertation"
    assert citation["authors"]
    assert citation["repository-code"] == "https://github.com/M1yanoShiho/IBM_Granite_Project"


def test_package_metadata_points_to_public_release_files() -> None:
    metadata = tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]

    assert metadata["version"] == "1.0.0"
    assert metadata["readme"] == "README.md"
    assert metadata["authors"]
    assert metadata["license"]["file"] == "LICENSE"
    assert metadata["license-files"] == ["LICENSE"]
    assert metadata["urls"]["Repository"] == (
        "https://github.com/M1yanoShiho/IBM_Granite_Project"
    )


def test_public_document_relative_links_resolve() -> None:
    missing: list[str] = []
    for document in LINK_CHECK_DOCS:
        for raw_target in LINK_PATTERN.findall(_text(document)):
            target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")

    assert not missing, "missing relative links:\n" + "\n".join(missing)


def test_public_markdown_has_one_h1_and_scannable_h2_headings() -> None:
    approved_h2_prefixes = (
        "⚙️",
        "🏗️",
        "📁",
        "📋",
        "📊",
        "📚",
        "📦",
        "📥",
        "📤",
        "📝",
        "🧠",
        "🧪",
        "🏷️",
        "🖥️",
        "🔄",
        "🔗",
        "🔧",
        "🔍",
        "🔒",
        "💾",
        "🌐",
        "🎯",
        "✅",
        "⚠️",
        "⚖️",
        "🤝",
        "👥",
        "✏️",
        "🚀",
    )
    for document in PUBLIC_DOCS:
        headings = _text(document).splitlines()
        assert sum(line.startswith("# ") for line in headings) == 1, document
        for heading in (line.removeprefix("## ") for line in headings if line.startswith("## ")):
            if document == ROOT / "README.md":
                assert heading
                continue
            assert heading.startswith(approved_h2_prefixes), f"{document}: {heading}"


def test_public_docs_publish_no_personal_or_hpc_paths() -> None:
    text = "\n".join(_text(path) for path in PUBLIC_DOCS)
    forbidden = (
        "/" + "scratch/",
        "/user/" + "work/",
        "/" + "Users/",
        "fl" + "25387",
        "ba" + "25966",
        "bp1-" + "login",
    )

    assert all(value not in text for value in forbidden)


def test_license_is_an_explicit_owner_selected_policy() -> None:
    license_text = _text(ROOT / "LICENSE")

    assert any(
        marker in license_text
        for marker in ("MIT License", "Apache License", "All Rights Reserved")
    )
