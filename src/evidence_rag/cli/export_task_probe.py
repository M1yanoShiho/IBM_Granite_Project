"""CLI: materialise the Gate 0B-2 task probe from a NIAH manifest + mutation log.

Point this at the TRAIN split. Building the probe from dev or the sealed test would put the
acceptance set and the evaluation set on the same queries, which is the thing the whole
pre-registration protocol exists to prevent.

--hypothesis-form selects a rung of the §2.4 form ablation and defaults to the frozen template,
so an invocation that predates the flag still produces a byte-identical pairs file. The pairs
file itself carries no marker of which arm produced it, so the chosen form is echoed in the
stdout report and that line is the protocol record.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.relations.claims import HYPOTHESIS_FORMS, HypothesisForm
from evidence_rag.relations.qa2d import load_qa2d_cache
from evidence_rag.relations.task_probe import GoldAnswerSource, build_probe_pairs

# Not in HYPOTHESIS_FORMS: rung 3 is not a pure function of (question, answer), it is a lookup
# into a cache pre-generated on the login node by evidence_rag.cli.export_qa2d.
QA2D_FORM = "qa2d"
FORM_CHOICES = (*HYPOTHESIS_FORMS, QA2D_FORM)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the Gate 0B-2 task probe")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--hypothesis-form",
        choices=FORM_CHOICES,
        default="template",
        help="§2.4 ablation rung; the default is the frozen pre-registered arm",
    )
    parser.add_argument(
        "--gold-answer",
        choices=("canonical", "surface"),
        default="canonical",
        help="which gold answer string the gold claim carries; 'canonical' is the "
        "pre-registered arm, 'surface' is the R012d casing control",
    )
    parser.add_argument(
        "--qa2d-cache",
        type=Path,
        help="QA2D cache jsonl from evidence_rag.cli.export_qa2d; "
        f"required for --hypothesis-form {QA2D_FORM}",
    )
    return parser


def _resolve_form(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> HypothesisForm:
    """Refuse to fall back to the template when the QA2D arm is asked for and unavailable.

    A fallback would emit a file the operator believes is rung 3 while it is actually rung 0 —
    the one failure mode that leaves no trace in the pairs file, the metrics, or the report.
    """
    if arguments.hypothesis_form != QA2D_FORM:
        form: HypothesisForm = HYPOTHESIS_FORMS[arguments.hypothesis_form]
        return form
    if arguments.qa2d_cache is None:
        parser.error(f"--hypothesis-form {QA2D_FORM} requires --qa2d-cache PATH")
    if not arguments.qa2d_cache.is_file():
        parser.error(f"QA2D cache not found: {arguments.qa2d_cache}")
    return load_qa2d_cache(arguments.qa2d_cache)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    hypothesis_form = _resolve_form(parser, arguments)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    records = read_provenance(arguments.provenance)
    pairs = build_probe_pairs(
        records=records,
        question_by_query={query.query_id: query.text for query in bundle.queries},
        text_by_document={document.document_id: document.text for document in bundle.documents},
        hypothesis_form=hypothesis_form,
        gold_answer_source=cast(GoldAnswerSource, arguments.gold_answer),
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(
            json.dumps(
                {
                    "premise": pair.premise,
                    "hypothesis": pair.hypothesis,
                    "label": pair.label.value,
                    "group": pair.group,
                    "kind": pair.kind,
                    "query_id": pair.query_id,
                },
                sort_keys=True,
            )
            + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )
    report = {
        "gold_answer_source": arguments.gold_answer,
        "hypothesis_form": arguments.hypothesis_form,
        "n_records": len(records),
        "n_pairs": len(pairs),
        "n_skipped_records": len(records) - len(pairs) // 4,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
