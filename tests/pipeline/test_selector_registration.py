import hashlib
from pathlib import Path

import pytest

import evidence_rag.composition as composition_module
from evidence_rag.composition import build_selector
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.selector.top_k import TopKSelector


class _SelectorScores:
    def __init__(self, protect_scores: list[float], harm_scores: list[float]) -> None:
        self.protect_scores = protect_scores
        self.harm_scores = harm_scores


class _FrozenSelectorModel:
    loaded = True

    def eval(self) -> "_FrozenSelectorModel":
        return self

    def __call__(
        self,
        *,
        question: list[str],
        candidate_text: list[str],
    ) -> _SelectorScores:
        assert self.loaded
        assert question == ["Which evidence is safe?"] * 10
        return _SelectorScores(
            protect_scores=[0.01 if "poison" in text else 0.99 for text in candidate_text],
            harm_scores=[0.99 if "poison" in text else 0.01 for text in candidate_text],
        )


def _ten_candidates() -> CandidateSet:
    return CandidateSet(
        query_id="q-selector",
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=f"ev-{rank}",
                document_id=f"doc-{rank}",
                chunk_id=f"chunk-{rank}",
                text="poison evidence" if rank == 10 else f"safe evidence {rank}",
                source_uri=f"fixture://doc-{rank}",
                retrieval_score=float(11 - rank),
                retrieval_rank=rank,
            )
            for rank in range(1, 11)
        ),
    )


def test_top_k_selector_is_still_registered() -> None:
    assert isinstance(build_selector(ModuleConfig(name="top-k")), TopKSelector)


def test_top_k_rejects_parameters() -> None:
    with pytest.raises(ValueError, match="does not accept parameters"):
        build_selector(ModuleConfig(name="top-k", parameters={"top_n": 10}))


def test_nli_risk_selector_scores_live_candidates_and_drops_harmful_evidence() -> None:
    selector = build_selector(
        ModuleConfig(
            name="nli-risk-controlled",
            parameters={"safe_threshold": 0.9212157130241394, "max_delete": 2},
        ),
        model=_FrozenSelectorModel(),
    )

    result = selector.select(
        Query(query_id="q-selector", text="Which evidence is safe?"),
        _ten_candidates(),
        max_selected=10,
    )

    assert tuple(item.evidence_id for item in result.items) == tuple(
        f"ev-{rank}" for rank in range(1, 10)
    )


def test_nli_risk_selector_loads_the_frozen_checkpoint_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"frozen-selector-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = _FrozenSelectorModel()
    model.loaded = False
    monkeypatch.setenv("TEST_SELECTOR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TEST_SELECTOR_MODEL", "fixture://nli-base")

    def load_model(
        model_snapshot: str,
        *,
        revision: str,
        identity_model_id: str,
        local_files_only: bool,
        device: str,
    ) -> _FrozenSelectorModel:
        assert model_snapshot == "fixture://nli-base"
        assert revision == "selector-revision"
        assert identity_model_id == "fixture/nli-base"
        assert local_files_only is True
        assert device == "cpu"
        return model

    def load_checkpoint(loaded_model: _FrozenSelectorModel, path: Path) -> object:
        assert loaded_model is model
        assert path == checkpoint
        loaded_model.loaded = True
        return object()

    monkeypatch.setattr(composition_module, "load_nli_dual_head_model", load_model)
    monkeypatch.setattr(composition_module, "load_dual_head_checkpoint", load_checkpoint)

    selector = build_selector(
        ModuleConfig(
            name="nli-risk-controlled",
            parameters={
                "model_snapshot": "${TEST_SELECTOR_MODEL}",
                "model_id": "fixture/nli-base",
                "revision": "selector-revision",
                "checkpoint_path": "${TEST_SELECTOR_CHECKPOINT}",
                "checkpoint_sha256": checkpoint_sha256,
                "safe_threshold": 0.9212157130241394,
                "max_delete": 2,
                "local_files_only": True,
                "device": "cpu",
            },
        )
    )

    result = selector.select(
        Query(query_id="q-selector", text="Which evidence is safe?"),
        _ten_candidates(),
        max_selected=10,
    )

    assert model.loaded is True
    assert result.items[-1].evidence_id == "ev-9"


def test_nli_risk_selector_reports_a_missing_runtime_path_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_SELECTOR_CHECKPOINT", raising=False)

    with pytest.raises(
        ValueError,
        match="Selector checkpoint requires environment variable.*MISSING_SELECTOR_CHECKPOINT",
    ):
        build_selector(
            ModuleConfig(
                name="nli-risk-controlled",
                parameters={
                    "model_snapshot": "fixture://nli-base",
                    "model_id": "fixture/nli-base",
                    "revision": "selector-revision",
                    "checkpoint_path": "${MISSING_SELECTOR_CHECKPOINT}",
                    "checkpoint_sha256": "0" * 64,
                },
            )
        )


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
