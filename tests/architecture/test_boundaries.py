import ast
from pathlib import Path

SOURCE = Path("src/evidence_rag")
COMPONENTS = ("retriever", "selector", "generator")


def imports(path: Path) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return tuple(found)


def test_components_only_import_their_own_package_and_allowed_shared_modules() -> None:
    for component in COMPONENTS:
        allowed = (
            f"evidence_rag.{component}",
            "evidence_rag.contracts",
        )
        if component == "retriever":
            allowed += ("evidence_rag.infrastructure",)
        for path in (SOURCE / component).rglob("*.py"):
            for module, level in imports(path):
                assert level == 0, f"{path} uses a relative import"
                if module.startswith("evidence_rag"):
                    assert module.startswith(allowed), f"{path} imports {module}"


def test_infrastructure_does_not_import_business_implementations() -> None:
    forbidden = tuple(
        f"evidence_rag.{component}"
        for component in ("retriever", "selector", "generator", "pipeline", "evaluation")
    )
    for path in (SOURCE / "infrastructure").rglob("*.py"):
        for module, level in imports(path):
            assert level == 0, f"{path} uses a relative import"
            assert not module.startswith(forbidden), f"{path} imports {module}"


def test_pipeline_imports_contracts_and_query_analysis_only() -> None:
    allowed = (
        "evidence_rag.contracts",
        "evidence_rag.query_analysis",
    )
    for path in (SOURCE / "pipeline").rglob("*.py"):
        for module, level in imports(path):
            assert level == 0, f"{path} uses a relative import"
            if module.startswith("evidence_rag"):
                assert module.startswith(allowed), (
                    f"{path} imports concrete module {module}"
                )
