import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.experiment import ExperimentWorkflow, WorkflowSummary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a persisted evidence RAG experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "command",
        choices=("prepare", "retriever", "selector", "generator", "pipeline", "all"),
    )
    return parser


def _run(workflow: ExperimentWorkflow, command: str) -> WorkflowSummary:
    methods = {
        "prepare": workflow.prepare,
        "retriever": workflow.run_retriever,
        "selector": workflow.run_selector,
        "generator": workflow.run_generator,
        "pipeline": workflow.run_pipeline,
        "all": workflow.run_all,
    }
    return methods[command]()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workflow = ExperimentWorkflow.from_toml(arguments.config)
    summary = _run(workflow, arguments.command)
    print(
        json.dumps(
            {
                "command": summary.command,
                "output_directory": str(summary.output_directory),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
