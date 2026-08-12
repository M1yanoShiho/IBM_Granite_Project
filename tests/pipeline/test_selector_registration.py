import pytest

from evidence_rag.composition import build_selector
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.selector.top_k import TopKSelector


def test_top_k_is_the_only_registered_selector() -> None:
    assert isinstance(build_selector(ModuleConfig(name="top-k")), TopKSelector)


def test_top_k_rejects_parameters() -> None:
    with pytest.raises(ValueError, match="does not accept parameters"):
        build_selector(ModuleConfig(name="top-k", parameters={"top_n": 10}))


@pytest.mark.parametrize(
    "retired_name",
    (
        "beam-three-class",
        "corroboration",
        "gated-corroboration",
        "gated-coverage-corroboration",
        "reliability-mis",
    ),
)
def test_retired_selectors_are_not_registered(retired_name: str) -> None:
    with pytest.raises(ValueError, match="unknown selector"):
        build_selector(ModuleConfig(name=retired_name))
