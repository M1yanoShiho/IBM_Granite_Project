import pytest

from evidence_rag.composition import build_selector
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.gated import GatedCorroborationSelector, GatedCoverageSelector
from evidence_rag.selector.top_k import TopKSelector


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "NONE"


def test_top_k_still_builds() -> None:
    assert isinstance(build_selector(ModuleConfig(name="top-k")), TopKSelector)


def test_corroboration_builds_with_parameters() -> None:
    selector = build_selector(
        ModuleConfig(name="corroboration", parameters={"alpha": 0.4, "top_n": 10}),
        llm=FakeLLM(),
    )
    assert isinstance(selector, CorroborationSelector)
    assert selector.alpha == 0.4
    assert selector.top_n == 10


def test_gated_corroboration_builds_with_parameters() -> None:
    selector = build_selector(
        ModuleConfig(
            name="gated-corroboration",
            parameters={"alpha": 0.6, "margin": 3, "support_cap": 2, "top_n": 15},
        ),
        llm=FakeLLM(),
    )
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.margin == 3
    assert selector.support_cap == 2
    assert selector.top_n == 15


def test_gated_corroboration_defaults_match_spec() -> None:
    selector = build_selector(ModuleConfig(name="gated-corroboration"), llm=FakeLLM())
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.margin == 2
    assert selector.support_cap == 1


def test_gated_coverage_builds_with_parameters() -> None:
    selector = build_selector(
        ModuleConfig(
            name="gated-coverage-corroboration",
            parameters={"margin": 3, "support_cap": 2},
        ),
        llm=FakeLLM(),
    )
    assert isinstance(selector, GatedCoverageSelector)
    assert selector.margin == 3
    assert selector.support_cap == 2


def test_gated_equivalence_defaults_to_exact() -> None:
    selector = build_selector(ModuleConfig(name="gated-corroboration"), llm=FakeLLM())
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.equivalence == "exact"


def test_gated_equivalence_lenient_builds() -> None:
    selector = build_selector(
        ModuleConfig(name="gated-corroboration", parameters={"equivalence": "lenient"}),
        llm=FakeLLM(),
    )
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.equivalence == "lenient"


def test_gated_equivalence_invalid_rejected() -> None:
    with pytest.raises(ValueError, match="invalid selector parameter equivalence"):
        build_selector(
            ModuleConfig(name="gated-corroboration", parameters={"equivalence": "fuzzy"}),
            llm=FakeLLM(),
        )


def test_unknown_parameter_rejected() -> None:
    with pytest.raises(ValueError, match="unknown selector parameter"):
        build_selector(
            ModuleConfig(name="gated-corroboration", parameters={"tau": 0.3}),
            llm=FakeLLM(),
        )


def test_out_of_range_parameters_rejected() -> None:
    for parameters in ({"alpha": 1.5}, {"margin": 0}, {"support_cap": -1}, {"top_n": 0}):
        with pytest.raises(ValueError, match="invalid selector parameter"):
            build_selector(
                ModuleConfig(name="gated-corroboration", parameters=parameters),
                llm=FakeLLM(),
            )


def test_non_integer_margin_rejected() -> None:
    with pytest.raises(ValueError, match="invalid selector parameter"):
        build_selector(
            ModuleConfig(name="gated-corroboration", parameters={"margin": 1.5}),
            llm=FakeLLM(),
        )


def test_top_k_rejects_parameters() -> None:
    with pytest.raises(ValueError):
        build_selector(ModuleConfig(name="top-k", parameters={"alpha": 0.5}))


def test_unknown_selector_rejected() -> None:
    with pytest.raises(ValueError, match="unknown selector"):
        build_selector(ModuleConfig(name="mystery"))


def test_support_unit_rejects_an_unknown_value() -> None:
    config = ModuleConfig(name="gated-corroboration", parameters={"support_unit": "page"})
    with pytest.raises(ValueError, match="support_unit"):
        build_selector(config, llm=FakeLLM())


def test_support_unit_defaults_to_document() -> None:
    selector = build_selector(
        ModuleConfig(name="gated-corroboration", parameters={}), llm=FakeLLM()
    )
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.parent_by_document is None


def test_support_unit_parent_loads_the_sidecar_from_the_environment(
    tmp_path, monkeypatch
) -> None:
    import json

    index = tmp_path / "source_parent.jsonl"
    index.write_text(
        json.dumps({"document_id": "d1", "source_parent_id": "page a"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("SOURCE_PARENT_INDEX", str(index))
    selector = build_selector(
        ModuleConfig(name="gated-corroboration", parameters={"support_unit": "parent"}),
        llm=FakeLLM(),
    )
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.parent_by_document == {"d1": "page a"}


def test_support_unit_parent_fails_loudly_without_the_sidecar(monkeypatch) -> None:
    """Silently falling back to the document unit would run a whole experiment on the old
    vote counting with no metric able to reveal it."""
    monkeypatch.delenv("SOURCE_PARENT_INDEX", raising=False)
    with pytest.raises(ValueError, match="SOURCE_PARENT_INDEX"):
        build_selector(
            ModuleConfig(name="gated-corroboration", parameters={"support_unit": "parent"}),
            llm=FakeLLM(),
        )
