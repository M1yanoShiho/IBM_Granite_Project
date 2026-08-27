from __future__ import annotations

import json
import socket

import pytest

from evidence_rag.cli.smoke import build_cpu_smoke_pipeline, main, run_cpu_smoke
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.retriever.hybrid import HybridRetriever
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector


def test_cpu_smoke_uses_final_module_classes_and_excludes_harm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CPU smoke attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    pipeline, config = build_cpu_smoke_pipeline()

    assert isinstance(pipeline.retriever, HybridRetriever)
    assert isinstance(pipeline.selector, NliRiskControlledSelector)
    assert isinstance(pipeline.generator, VerifyAnnotateGenerator)

    run = run_cpu_smoke(pipeline=pipeline, config=config)

    assert len(run.candidates.candidates) == 10
    assert any("poison" in item.text for item in run.candidates.candidates)
    assert all("poison" not in item.text for item in run.selected.evidence)
    assert run.generation.answer == "IBM acquired Red Hat in 2019."
    assert set(run.generation.cited_evidence_ids) <= {
        item.evidence_id for item in run.selected.evidence
    }


def test_cpu_smoke_cli_prints_the_complete_pipeline_trace(capsys) -> None:
    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["query"]["query_id"] == "cpu-smoke-query"
    assert payload["candidates"]["candidates"]
    assert payload["selection"]["items"]
    assert payload["selected"]["evidence"]
    assert payload["generation"]["answer"] == "IBM acquired Red Hat in 2019."
