#!/usr/bin/env python3
"""Generate resumable Experiment 05 outputs from frozen prepared evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evidence_rag.contracts.models import Query  # noqa: E402
from evidence_rag.contracts.validation import validate_generation  # noqa: E402
from evidence_rag.evaluation.experiment05_data import read_runtime_bundle  # noqa: E402
from evidence_rag.evaluation.experiment05_generation import (  # noqa: E402
    DIRECT_ARMS,
    GROUNDED_ARMS,
    SystemOutput,
    answer_from_routings,
    build_system_output,
    prepare_prompt_evidence,
    prompt_template,
)
from evidence_rag.evaluation.experiment05_io import (  # noqa: E402
    append_canonical_jsonl,
    read_jsonl,
)
from evidence_rag.evaluation.experiment05_retrieval import canonical_sha256  # noqa: E402
from evidence_rag.evaluation.experiment05_runtime import PreparedQuery  # noqa: E402
from evidence_rag.generator.claim_splitter import ClaimSplitter  # noqa: E402
from evidence_rag.generator.draft import DraftAnswerGenerator, DraftGenerator  # noqa: E402
from evidence_rag.generator.entity_check import (  # noqa: E402
    EntityConsistencyChecker,
    SpacyEntityExtractor,
)
from evidence_rag.generator.granite import (  # noqa: E402
    GraniteGenerationConfig,
    GraniteLLMClient,
    InlineCitationGraniteGenerator,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.nli import TrueNLIModel  # noqa: E402
from evidence_rag.generator.verify_annotate import (  # noqa: E402
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
)
from evidence_rag.query_analysis import RuleBasedQueryAnalyzer  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _existing_count(path: Path, expected_ids: list[str], arm: str) -> int:
    rows = [] if not path.exists() else read_jsonl(path)
    parsed = [SystemOutput.model_validate(row) for row in rows]
    if [item.query_id for item in parsed] != expected_ids[: len(parsed)]:
        raise ValueError(f"{arm} output is not an ordered runtime prefix")
    if any(item.arm_id != arm for item in parsed):
        raise ValueError(f"{arm} output contains another arm identity")
    return len(parsed)


def _grounded_generator(
    client: PeftGraniteLLMClient,
    verifier: TrueNLIModel,
    *,
    adapter_name: str,
    template: str,
) -> VerifyAnnotateGenerator:
    draft = DraftAnswerGenerator(
        draft_generator=DraftGenerator(
            llm=NamedAdapterTextGenerator(client, adapter_name),
            prompt_template=template,
            trace_enabled=True,
        ),
        claim_splitter=ClaimSplitter(llm=client, trace_enabled=True),
        trace_enabled=True,
    )
    return VerifyAnnotateGenerator(
        draft_generator=draft,
        verifier=CitationRoutedVerifier(
            verifier,
            EntityConsistencyChecker(SpacyEntityExtractor()),
            entity_gate="observe",
        ),
        abstain_when_unverified=False,
        entity_gate="observe",
        trace_enabled=True,
    )


def _trace_value(generator: VerifyAnnotateGenerator, arm: str, query_id: str) -> dict[str, Any]:
    trace = generator.last_trace
    return {
        "schema_version": "experiment05.internal_generator_trace.v1",
        "arm_id": arm,
        "query_id": query_id,
        "trace": None if trace is None else trace.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "grounded"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--true-snapshot", type=Path)
    parser.add_argument("--adapter13", type=Path)
    parser.add_argument("--adapter42", type=Path)
    parser.add_argument("--adapter73", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    allowed = set(DIRECT_ARMS if args.mode == "direct" else GROUNDED_ARMS)
    if not args.arms or len(args.arms) != len(set(args.arms)) or not set(args.arms) <= allowed:
        raise ValueError(f"invalid {args.mode} arm list")
    runtime = list(read_runtime_bundle(args.runtime, dataset=args.dataset))
    prepared = [PreparedQuery.model_validate(row) for row in read_jsonl(args.prepared)]
    if args.limit is not None:
        runtime = runtime[: args.limit]
        prepared = prepared[: args.limit]
    expected_ids = [str(item["query_id"]) for item in runtime]
    if [item.query_id for item in prepared] != expected_ids:
        raise ValueError("prepared evidence does not cover the selected runtime sequence")

    template = prompt_template(args.dataset)
    prompt_fingerprint = canonical_sha256({"dataset": args.dataset, "template": template})
    config_fingerprint = canonical_sha256(
        {
            "max_input_tokens": 2304,
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "whole_evidence_prefix": True,
            "canonical_abstention": True,
        }
    )
    config = GraniteGenerationConfig(
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        max_input_tokens=2304,
    )
    analyzer = RuleBasedQueryAnalyzer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    direct_generator: InlineCitationGraniteGenerator | None = None
    grounded_generators: dict[str, VerifyAnnotateGenerator] = {}
    if args.mode == "direct":
        client: GraniteLLMClient | PeftGraniteLLMClient = GraniteLLMClient(
            model_id=str(args.model_snapshot),
            config=config,
            device="cuda",
            dtype="bfloat16",
        )
        direct_generator = InlineCitationGraniteGenerator(
            llm=client,
            prompt_template=template,
            require_declared_citations=True,
        )
        model_fingerprints = {
            "generator": "ibm-granite/granite-4.1-3b@c0650403e44e78ec0262dab1c90914c65b196c4e"
        }
    else:
        if not args.true_snapshot or not args.adapter13 or not args.adapter42 or not args.adapter73:
            raise ValueError("grounded mode requires TRUE and all three frozen adapters")
        expected_adapters = {
            "13": (
                args.adapter13,
                "492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707",
                "1918333d819e6007d3faf17d3497c4f4665f7c0d3cb032f510986cfdb9837942",
            ),
            "42": (
                args.adapter42,
                "96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c",
                "df6c8e4c39f3d7567849a4a32355aea4bb9176c784d089832fe14a8697a2d8f8",
            ),
            "73": (
                args.adapter73,
                "d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c",
                "f1659959d9ffe7348c370e9f01a497066710793c9754e9d06c5d8963aaf45dd1",
            ),
        }
        for seed, (path, weights_sha256, config_sha256) in expected_adapters.items():
            if (
                _file_sha256(path / "adapter_model.safetensors") != weights_sha256
                or _file_sha256(path / "adapter_config.json") != config_sha256
            ):
                raise ValueError(f"frozen GRC seed{seed} adapter hash differs")
        client = PeftGraniteLLMClient(
            model_id=str(args.model_snapshot),
            adapters={
                "grc13": str(args.adapter13),
                "grc42": str(args.adapter42),
                "grc73": str(args.adapter73),
            },
            config=config,
            device="auto",
            dtype="bfloat16",
        )
        verifier = TrueNLIModel(model_id=str(args.true_snapshot))
        needed_seeds = {
            "42" if arm == "ours_seed42" else "73" if arm == "ours_seed73" else "13"
            for arm in args.arms
        }
        grounded_generators = {
            seed: _grounded_generator(
                client,
                verifier,
                adapter_name=f"grc{seed}",
                template=template,
            )
            for seed in needed_seeds
        }
        model_fingerprints = {
            "generator": "ibm-granite/granite-4.1-3b@c0650403e44e78ec0262dab1c90914c65b196c4e",
            "verifier": "google/t5_xxl_true_nli_mixture@aa6cfe1dd4257853bfdd772992045f41bfc14988",
        }

    for arm in args.arms:
        arm_model_fingerprints = dict(model_fingerprints)
        if args.mode == "grounded":
            seed = "42" if arm == "ours_seed42" else "73" if arm == "ours_seed73" else "13"
            arm_model_fingerprints["grc_adapter"] = {
                "13": "sha256:492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707",
                "42": "sha256:96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c",
                "73": "sha256:d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c",
            }[seed]
        output_path = args.output_dir / "generations" / f"{arm}.jsonl"
        completed = _existing_count(output_path, expected_ids, arm)
        trace_path = args.output_dir / "internal_traces" / f"{arm}.jsonl"
        if args.mode == "grounded":
            trace_count = 0 if not trace_path.exists() else len(read_jsonl(trace_path))
            if trace_count != completed:
                raise ValueError(f"{arm} internal trace/output resume counts differ")
        for index in range(completed, len(prepared)):
            item = prepared[index]
            query = Query(query_id=item.query_id, text=str(runtime[index]["question"]))
            presented, selected_records, presented_records, prompt, prompt_tokens = (
                prepare_prompt_evidence(
                    prepared=item,
                    arm_id=arm,
                    query=query,
                    template=template,
                    llm=client,
                    artifact_root=args.artifact_root,
                )
            )
            answer = ""
            runtime_error: str | None = None
            trace_uri: str | None = None
            try:
                if presented.evidence:
                    checklist = analyzer.analyze(query)
                    if args.mode == "direct":
                        assert direct_generator is not None
                        result = direct_generator.generate(query, checklist, presented)
                        validate_generation(presented, result)
                        answer = (
                            direct_generator.last_raw_output
                            if direct_generator.last_declared_indices
                            else result.answer
                        )
                    else:
                        seed = "42" if arm == "ours_seed42" else "73" if arm == "ours_seed73" else "13"
                        generator = grounded_generators[seed]
                        result = generator.generate(query, checklist, presented)
                        validate_generation(presented, result)
                        answer = answer_from_routings(
                            generator.last_routings,
                            tuple(record.evidence_id for record in presented_records),
                        )
                        trace_uri = f"sealed://experiment05/internal-trace/{args.dataset}/{arm}/{query.query_id}"
                        append_canonical_jsonl(
                            trace_path, _trace_value(generator, arm, query.query_id)
                        )
                elif args.mode == "grounded":
                    generator = grounded_generators[
                        "42" if arm == "ours_seed42" else "73" if arm == "ours_seed73" else "13"
                    ]
                    append_canonical_jsonl(
                        trace_path,
                        {
                            "schema_version": "experiment05.internal_generator_trace.v1",
                            "arm_id": arm,
                            "query_id": query.query_id,
                            "trace": {"final_empty_reason": "no_presented_evidence"},
                        },
                    )
                    trace_uri = f"sealed://experiment05/internal-trace/{args.dataset}/{arm}/{query.query_id}"
            except Exception as error:  # noqa: BLE001 - runtime failures stay in denominator
                runtime_error = type(error).__name__
                if args.mode == "grounded":
                    append_canonical_jsonl(
                        trace_path,
                        {
                            "schema_version": "experiment05.internal_generator_trace.v1",
                            "arm_id": arm,
                            "query_id": query.query_id,
                            "trace": None,
                            "runtime_error": runtime_error,
                        },
                    )
                    trace_uri = f"sealed://experiment05/internal-trace/{args.dataset}/{arm}/{query.query_id}"
            output = build_system_output(
                prepared=item,
                arm_id=arm,
                answer=answer,
                runtime_error=runtime_error,
                selected_records=selected_records,
                presented_records=presented_records,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                model_fingerprints=arm_model_fingerprints,
                config_fingerprint=config_fingerprint,
                prompt_fingerprint=prompt_fingerprint,
                internal_trace_uri=trace_uri,
            )
            append_canonical_jsonl(output_path, output)
            count = index + 1
            if count % args.log_every == 0 or count == len(prepared):
                print(
                    json.dumps(
                        {"dataset": args.dataset, "arm": arm, "completed": count, "total": len(prepared)}
                    ),
                    flush=True,
                )

    manifest = {
        "schema_version": "experiment05.generation_run_manifest.v1",
        "mode": args.mode,
        "dataset": args.dataset,
        "arms": args.arms,
        "count_per_arm": len(expected_ids),
        "runtime_sha256": _file_sha256(args.runtime),
        "prepared_sha256": _file_sha256(args.prepared),
        "config_fingerprint": config_fingerprint,
        "prompt_fingerprint": prompt_fingerprint,
        "outputs": {
            arm: _file_sha256(args.output_dir / "generations" / f"{arm}.jsonl")
            for arm in args.arms
        },
        "status": "PASS",
    }
    run_key = canonical_sha256({"mode": args.mode, "arms": args.arms})[:12]
    manifest_path = args.output_dir / f"run_manifest.{args.mode}.{run_key}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "mode": args.mode, "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
