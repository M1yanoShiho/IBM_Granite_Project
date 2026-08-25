#!/usr/bin/env python3
"""Minimal Goal 1 scorer fixtures and deterministic contract validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_rag.evaluation.experiment05_scorer import (
    CANONICAL_ABSTENTION,
    ExtractedClaim,
    ScorerClaimExtractor,
    canonical_fact,
    score_query_safely,
)


@dataclass(frozen=True)
class Fact:
    question: str
    answer: str
    wrong: str
    statement: str

    @property
    def wrong_statement(self) -> str:
        return self.statement.replace(self.answer, self.wrong, 1)


LOCKED_FACTS = (
    Fact("Which city is the capital of France?", "Paris", "Rome", "Paris is the capital of France"),
    Fact("Which is the largest planet in the Solar System?", "Jupiter", "Mars", "Jupiter is the largest planet in the Solar System"),
    Fact("Who wrote Hamlet?", "William Shakespeare", "Charles Dickens", "William Shakespeare wrote Hamlet"),
    Fact("What is the chemical symbol for gold?", "Au", "Ag", "Au is the chemical symbol for gold"),
    Fact("Which ocean lies east of the United States?", "Atlantic Ocean", "Pacific Ocean", "The Atlantic Ocean lies east of the United States"),
    Fact("In which year did humans first land on the Moon?", "1969", "1972", "Humans first landed on the Moon in 1969"),
    Fact("Who painted the Mona Lisa?", "Leonardo da Vinci", "Michelangelo", "Leonardo da Vinci painted the Mona Lisa"),
    Fact("What is the currency of Japan?", "yen", "won", "The yen is the currency of Japan"),
    Fact("What is Earth's tallest mountain above sea level?", "Mount Everest", "K2", "Mount Everest is Earth's tallest mountain above sea level"),
    Fact("Which element has atomic number 1?", "hydrogen", "helium", "Hydrogen has atomic number 1"),
    Fact("In which city is the Colosseum located?", "Rome", "Athens", "The Colosseum is located in Rome"),
    Fact("What is the official language of Brazil?", "Portuguese", "Spanish", "Portuguese is the official language of Brazil"),
    Fact("On which continent is the Sahara Desert?", "Africa", "Asia", "The Sahara Desert is in Africa"),
    Fact("Which company developed the iPhone?", "Apple", "Microsoft", "Apple developed the iPhone"),
    Fact("Which river flows through Egypt?", "the Nile", "the Amazon", "The Nile flows through Egypt"),
    Fact("Who composed the Fifth Symphony?", "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven composed the Fifth Symphony"),
    Fact("What is the smallest prime number?", "2", "3", "2 is the smallest prime number"),
    Fact("Which planet is called the Red Planet?", "Mars", "Venus", "Mars is called the Red Planet"),
    Fact("Which gas do plants absorb during photosynthesis?", "carbon dioxide", "oxygen", "Plants absorb carbon dioxide during photosynthesis"),
    Fact("In which country is Sydney located?", "Australia", "New Zealand", "Sydney is located in Australia"),
)

CALIBRATION_FACTS = (
    Fact("Which city is the capital of Germany?", "Berlin", "Munich", "Berlin is the capital of Germany"),
    Fact("What is the chemical formula for water?", "H2O", "CO2", "H2O is the chemical formula for water"),
    Fact("Who wrote Pride and Prejudice?", "Jane Austen", "Charlotte Bronte", "Jane Austen wrote Pride and Prejudice"),
    Fact("What is Earth's natural satellite?", "the Moon", "Phobos", "The Moon is Earth's natural satellite"),
    Fact("At sea level, at what Celsius temperature does water boil?", "100 degrees", "90 degrees", "Water boils at 100 degrees Celsius at sea level"),
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _fact_match_rows(prefix: str, facts: tuple[Fact, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact_index, fact in enumerate(facts):
        cases = (
            ("short_correct", fact.answer, True, "concise"),
            ("expanded_correct", fact.statement + ".", True, "concise"),
            ("paraphrased_correct", f"The answer is {fact.answer}; {fact.statement}.", True, "explanatory"),
            ("contextual_correct", f"A standard reference confirms that {fact.statement}.", True, "explanatory"),
            ("wrong_entity", fact.wrong_statement + ".", False, "concise"),
            ("negated", f"It is false that {fact.statement.lower()}.", False, "concise"),
            ("contrast_wrong", f"The answer is {fact.wrong}, not {fact.answer}.", False, "explanatory"),
            ("partial", "The question concerns a well-known fact, but this statement does not give its answer.", False, "explanatory"),
        )
        for case_index, (family, claim, label, style) in enumerate(cases):
            rows.append(
                {
                    "schema_version": "experiment05.fact-match-fixture.v1",
                    "fixture_id": f"{prefix}-fm-{fact_index:03d}-{case_index:02d}",
                    "family": family,
                    "style": style,
                    "question": fact.question,
                    "alias": fact.answer,
                    "claim": claim,
                    "label": label,
                }
            )
    return rows


def _support_rows(prefix: str, facts: tuple[Fact, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_styles = ("short_qa", "trivia_qa", "long_form")
    for fact_index, fact in enumerate(facts):
        claim = fact.statement + "."
        cases = (
            ("exact_support", claim, True, "concise"),
            ("reference_support", f"Reference note: {claim}", True, "concise"),
            ("expanded_support", f"Historical context varies, but {fact.statement.lower()}.", True, "explanatory"),
            ("summary_support", f"In summary, {claim} This is the relevant fact.", True, "explanatory"),
            ("wrong_entity", fact.wrong_statement + ".", False, "concise"),
            ("explicit_denial", f"It is false that {fact.statement.lower()}.", False, "concise"),
            ("unrelated", "This passage discusses weather records and contains no answer to the question.", False, "explanatory"),
            ("near_miss", f"A separate source instead identifies {fact.wrong}. No other conclusion is given.", False, "explanatory"),
        )
        for case_index, (family, passage, label, style) in enumerate(cases):
            rows.append(
                {
                    "schema_version": "experiment05.support-fixture.v1",
                    "fixture_id": f"{prefix}-es-{fact_index:03d}-{case_index:02d}",
                    "family": family,
                    "style": style,
                    "dataset_style": dataset_styles[(fact_index + case_index) % 3],
                    "claim": claim,
                    "passage": passage,
                    "label": label,
                }
            )
    return rows


def _claim_rows(prefix: str, facts: tuple[Fact, ...], count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        if count == 60 and index < 10:
            width, style = 1, "short_fact"
        elif count == 60 and index < 30:
            width, style = 3, "concise"
        elif count == 60 and index < 50:
            width, style = 4, "expanded"
        elif count == 60:
            width, style = 5, "long_form"
        else:
            width, style = (1 if index < 3 else 3), ("short_fact" if index < 3 else "expanded")
        selected = [facts[(index + offset) % len(facts)] for offset in range(width)]
        if width == 1:
            answer = selected[0].answer
            question = selected[0].question
            sources = (selected[0].answer,)
        else:
            sources = tuple(fact.statement + "." for fact in selected)
            answer = " ".join(sources)
            question = "State the requested established facts."
        gold: list[dict[str, Any]] = []
        cursor = 0
        for fact, source in zip(selected, sources, strict=True):
            start = answer.index(source, cursor)
            gold.append(
                {
                    "source_text": source,
                    "source_start": start,
                    "source_end": start + len(source),
                    "claim_text": fact.statement + ".",
                }
            )
            cursor = start + len(source)
        rows.append(
            {
                "schema_version": "experiment05.claim-extraction-fixture.v1",
                "fixture_id": f"{prefix}-ce-{index:03d}",
                "style": style,
                "question": question,
                "answer": answer,
                "gold_claims": gold,
            }
        )
    return rows


def _zero_expected(query_id: str, runtime_error: bool = False, scorer_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query_id": query_id,
        "metrics": {"rfc": 0.0, "vrfc": 0.0, "ucr": None, "cp": 0.0, "cr": 0.0, "rr": 0.0},
        "counts": {
            "reference_facts": 1,
            "matched_reference_facts": 0,
            "matched_and_validly_cited_reference_facts": 0,
            "claims": 0,
            "unsupported_claims": 0,
            "citation_links": 0,
            "supporting_citation_links": 0,
            "duplicate_citation_links": 0,
            "invalid_citation_links": 0,
        },
        "runtime_error": runtime_error,
        "scorer_error": scorer_error,
    }
    if scorer_error:
        result["scorer_error_type"] = "ValueError"
    return result


def _contract_rows(prefix: str, facts: tuple[Fact, ...], fact_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact_index, fact in enumerate(facts[:fact_limit]):
        query_id = f"{prefix}-q-{fact_index:03d}"
        claim = fact.statement + "."
        answer_claim = claim.rstrip(".")
        fact_pair = [f"Question: {fact.question}\nAnswer statement: {claim}", canonical_fact(fact.question, fact.answer)]
        evidence = claim
        base_sidecar = {
            "query_id": query_id,
            "reference_fact_groups": [{"fact_id": "fact-1", "fact_question": fact.question, "aliases": [fact.answer]}],
        }

        def output(
            answer: str,
            presented: Any,
            _query_id: str = query_id,
            **extra: Any,
        ) -> dict[str, Any]:
            return {
                "query_id": _query_id,
                "answer_text": answer,
                "abstained": False,
                "runtime_error": None,
                "presented_evidence_records": presented,
                **extra,
            }

        presented = [{"prompt_ordinal": 1, "evidence_id": "ev-1", "text": evidence}]
        perfect_counts = {
            "reference_facts": 1, "matched_reference_facts": 1,
            "matched_and_validly_cited_reference_facts": 1, "claims": 1,
            "unsupported_claims": 0, "citation_links": 1,
            "supporting_citation_links": 1, "duplicate_citation_links": 0,
            "invalid_citation_links": 0,
        }
        cases: list[tuple[str, dict[str, Any], list[dict[str, Any]], list[list[Any]], dict[str, Any]]] = []
        cases.append(("perfect", output(answer_claim + " [1].", presented), [{"claim_id": "claim-1", "text": claim, "sentence_index": 0}], [fact_pair + [True], [evidence, claim, True]], {"query_id": query_id, "metrics": {"rfc": 1.0, "vrfc": 1.0, "ucr": 0.0, "cp": 1.0, "cr": 1.0, "rr": 1.0}, "counts": perfect_counts, "runtime_error": False, "scorer_error": False}))
        unsupported_counts = dict(perfect_counts, matched_and_validly_cited_reference_facts=0, unsupported_claims=1, citation_links=0, supporting_citation_links=0)
        cases.append(("unsupported", output(claim, []), [{"claim_id": "claim-1", "text": claim, "sentence_index": 0}], [fact_pair + [True]], {"query_id": query_id, "metrics": {"rfc": 1.0, "vrfc": 0.0, "ucr": 1.0, "cp": 0.0, "cr": 0.0, "rr": 1.0}, "counts": unsupported_counts, "runtime_error": False, "scorer_error": False}))
        abstention = "" if fact_index % 2 else CANONICAL_ABSTENTION
        cases.append(("abstention", output(abstention, [], abstained=True), [], [], _zero_expected(query_id)))
        invalid_counts = dict(perfect_counts, matched_and_validly_cited_reference_facts=0, supporting_citation_links=0, invalid_citation_links=1)
        cases.append(("invalid_citation", output(answer_claim + " [9].", presented), [{"claim_id": "claim-1", "text": claim, "sentence_index": 0}], [fact_pair + [True], [evidence, claim, True]], {"query_id": query_id, "metrics": {"rfc": 1.0, "vrfc": 0.0, "ucr": 0.0, "cp": 0.0, "cr": 0.0, "rr": 1.0}, "counts": invalid_counts, "runtime_error": False, "scorer_error": False}))
        duplicate_citation_counts = dict(perfect_counts, duplicate_citation_links=1)
        cases.append(("duplicate_citation", output(answer_claim + " [1][1].", presented), [{"claim_id": "claim-1", "text": claim, "sentence_index": 0}], [fact_pair + [True], [evidence, claim, True]], {"query_id": query_id, "metrics": {"rfc": 1.0, "vrfc": 1.0, "ucr": 0.0, "cp": 1.0, "cr": 1.0, "rr": 1.0}, "counts": duplicate_citation_counts, "runtime_error": False, "scorer_error": False}))
        duplicate_claim_counts = dict(perfect_counts, duplicate_citation_links=1)
        cases.append(("duplicate_claim", output(f"{answer_claim} [1]. {answer_claim} [1].", presented), [{"claim_id": "claim-1", "text": claim, "sentence_index": 0}, {"claim_id": "claim-2", "text": claim, "sentence_index": 1}], [fact_pair + [True], [evidence, claim, True], [claim, claim, True]], {"query_id": query_id, "metrics": {"rfc": 1.0, "vrfc": 1.0, "ucr": 0.0, "cp": 1.0, "cr": 1.0, "rr": 1.0}, "counts": duplicate_claim_counts, "runtime_error": False, "scorer_error": False}))
        cases.append(("runtime_error", output("", [], runtime_error="generation_failed"), [], [], _zero_expected(query_id, runtime_error=True)))
        cases.append(("scorer_error", output(answer_claim + " [1].", "malformed"), [{"claim_id": "claim-1", "text": claim, "sentence_index": 0}], [], _zero_expected(query_id, scorer_error=True)))
        for case_index, (family, out, claims, verdicts, expected) in enumerate(cases):
            rows.append({"schema_version": "experiment05.contract-fixture.v1", "fixture_id": f"{prefix}-ct-{fact_index:03d}-{case_index:02d}", "family": family, "sidecar": base_sidecar, "output": out, "claims": claims, "verdicts": verdicts, "expected": expected})
    return rows


def write_fixtures(output_dir: Path) -> None:
    groups = {
        "calibration": {
            "fact_match.jsonl": _fact_match_rows("cal", CALIBRATION_FACTS),
            "evidence_support.jsonl": _support_rows("cal", CALIBRATION_FACTS),
            "claim_extraction.jsonl": _claim_rows("cal", CALIBRATION_FACTS, 12),
            "contract.jsonl": _contract_rows("cal", CALIBRATION_FACTS, 5),
        },
        "locked": {
            "fact_match.jsonl": _fact_match_rows("lock", LOCKED_FACTS),
            "evidence_support.jsonl": _support_rows("lock", LOCKED_FACTS),
            "claim_extraction.jsonl": _claim_rows("lock", LOCKED_FACTS, 60),
            "contract.jsonl": _contract_rows("lock", LOCKED_FACTS, 10),
        },
    }
    manifest_files: dict[str, Any] = {}
    for split, files in groups.items():
        for name, rows in files.items():
            path = output_dir / split / name
            _write_jsonl(path, rows)
            raw = path.read_bytes()
            manifest_files[f"{split}/{name}"] = {"count": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {"schema_version": "experiment05.scorer-fixtures-manifest.v1", "files": manifest_files}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


class FixtureJudge:
    def __init__(self, verdicts: list[list[Any]]) -> None:
        self.verdicts = {(str(premise), str(hypothesis)): bool(label) for premise, hypothesis, label in verdicts}

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.verdicts.get((premise, hypothesis), False)


def run_contract(fixtures_dir: Path, report_path: Path) -> None:
    rows = [json.loads(line) for line in (fixtures_dir / "locked" / "contract.jsonl").read_text().splitlines() if line]
    failed: list[str] = []
    for row in rows:
        claims = tuple(ExtractedClaim(**item) for item in row["claims"])
        actual = score_query_safely(row["sidecar"], row["output"], claims, FixtureJudge(row["verdicts"]))
        if actual != row["expected"]:
            failed.append(str(row["fixture_id"]))
    report = {"schema_version": "experiment05.contract-report.v1", "locked_cases": len(rows), "exact_passed": len(rows) - len(failed), "exact_rate": (len(rows) - len(failed)) / len(rows) if rows else 0.0, "failed_fixture_ids": failed, "pass": len(rows) == 80 and not failed}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["pass"]:
        raise SystemExit(1)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _binary_metrics(gold: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    tp = sum(expected and actual for expected, actual in zip(gold, predicted, strict=True))
    tn = sum(not expected and not actual for expected, actual in zip(gold, predicted, strict=True))
    fp = sum(not expected and actual for expected, actual in zip(gold, predicted, strict=True))
    fn = sum(expected and not actual for expected, actual in zip(gold, predicted, strict=True))
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    positive_precision = tp / (tp + fp) if tp + fp else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    positive_f1 = (
        2 * positive_precision * positive_recall / (positive_precision + positive_recall)
        if positive_precision + positive_recall
        else 0.0
    )
    negative_f1 = (
        2 * negative_precision * specificity / (negative_precision + specificity)
        if negative_precision + specificity
        else 0.0
    )
    return {
        "count": len(gold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "positive_recall": positive_recall,
        "specificity": specificity,
    }


def _style_fnr(rows: list[dict[str, Any]], predicted: list[bool]) -> dict[str, float]:
    result: dict[str, float] = {}
    for style in sorted({str(row["style"]) for row in rows}):
        indexes = [
            index
            for index, row in enumerate(rows)
            if row["style"] == style and bool(row["label"])
        ]
        result[style] = (
            sum(not predicted[index] for index in indexes) / len(indexes) if indexes else 0.0
        )
    return result


def _max_gap(values: dict[str, float]) -> float:
    return max(values.values()) - min(values.values()) if values else 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_model_validation(
    *,
    fixtures_dir: Path,
    split: str,
    minicheck_model: Path,
    granite_model: Path,
    device: str,
    threshold: float,
    report_path: Path,
) -> None:
    from evidence_rag.generator.granite import GraniteGenerationConfig, GraniteLLMClient
    from evidence_rag.generator.nli import MiniCheckNLIModel

    fact_rows = _read_jsonl(fixtures_dir / split / "fact_match.jsonl")
    support_rows = _read_jsonl(fixtures_dir / split / "evidence_support.jsonl")
    extraction_rows = _read_jsonl(fixtures_dir / split / "claim_extraction.jsonl")

    minicheck = MiniCheckNLIModel(model_id=str(minicheck_model), threshold=threshold)
    if device != "cpu":
        _tokenizer, model = minicheck._ensure_loaded()
        minicheck._model = model.to(device)

    fact_predictions = [
        minicheck.classify(
            premise=f"Question: {row['question']}\nAnswer statement: {row['claim']}",
            hypothesis=canonical_fact(str(row["question"]), str(row["alias"])),
        )
        == "entailment"
        for row in fact_rows
    ]
    support_predictions = [
        minicheck.classify(premise=str(row["passage"]), hypothesis=str(row["claim"]))
        == "entailment"
        for row in support_rows
    ]
    fact_metrics = _binary_metrics([bool(row["label"]) for row in fact_rows], fact_predictions)
    support_metrics = _binary_metrics(
        [bool(row["label"]) for row in support_rows], support_predictions
    )
    fact_fnr = _style_fnr(fact_rows, fact_predictions)
    support_fnr = _style_fnr(support_rows, support_predictions)

    granite = GraniteLLMClient(
        model_id=str(granite_model),
        config=GraniteGenerationConfig(max_new_tokens=384, temperature=0.0),
        device=device,
        dtype="bfloat16" if device != "cpu" else "float32",
    )
    extractor = ScorerClaimExtractor(granite)
    gold_count = 0
    covered_count = 0
    predicted_count = 0
    faithful_count = 0
    extraction_failures: list[str] = []
    short_total = 0
    short_substantive = 0
    for row in extraction_rows:
        try:
            extractor.extract(str(row["question"]), str(row["answer"]))
        except (ValueError, KeyError, TypeError):
            extraction_failures.append(str(row["fixture_id"]))
            extractor.last_records = ()
        records = extractor.last_records
        gold = list(row["gold_claims"])
        gold_count += len(gold)
        predicted_count += len(records)
        for expected in gold:
            left, right = int(expected["source_start"]), int(expected["source_end"])
            if any(
                max(left, record.source_start) < min(right, record.source_end)
                for record in records
            ):
                covered_count += 1
        for record in records:
            if any(
                minicheck.classify(
                    premise=(
                        f"Question: {row['question']}\n"
                        f"Answer statement: {expected['claim_text']}"
                    ),
                    hypothesis=record.claim.text,
                )
                == "entailment"
                for expected in gold
            ):
                faithful_count += 1
        if row["style"] == "short_fact":
            short_total += 1
            short_substantive += int(bool(records))

    try:
        insufficiency_claims = extractor.extract(
            "What is the requested fact?",
            "The provided evidence is insufficient to answer the question.",
        )
    except (ValueError, KeyError, TypeError):
        insufficiency_claims = ()
    extraction_metrics = {
        "answers": len(extraction_rows),
        "gold_claims": gold_count,
        "predicted_claims": predicted_count,
        "covered_gold_claims": covered_count,
        "faithful_predicted_claims": faithful_count,
        "span_coverage_recall": covered_count / gold_count if gold_count else 0.0,
        "faithful_claim_precision": faithful_count / predicted_count if predicted_count else 0.0,
        "parse_failure_count": len(extraction_failures),
        "parse_failure_ids": extraction_failures,
        "short_substantive_rate": short_substantive / short_total if short_total else 0.0,
        "insufficiency_rr_zero": not insufficiency_claims,
    }
    fairness = {
        "fact_match_positive_fnr_by_style": fact_fnr,
        "fact_match_fnr_gap": _max_gap(fact_fnr),
        "support_positive_fnr_by_style": support_fnr,
        "support_fnr_gap": _max_gap(support_fnr),
    }
    counts_ok = (
        split != "locked"
        or (
            len(fact_rows) == 160
            and len(support_rows) == 160
            and len(extraction_rows) == 60
            and gold_count >= 180
        )
    )
    passed = all(
        (
            counts_ok,
            float(fact_metrics["macro_f1"]) >= 0.90,
            float(fact_metrics["positive_recall"]) >= 0.90,
            float(fact_metrics["specificity"]) >= 0.90,
            float(support_metrics["macro_f1"]) >= 0.85,
            float(support_metrics["positive_recall"]) >= 0.85,
            float(support_metrics["specificity"]) >= 0.85,
            extraction_metrics["span_coverage_recall"] >= 0.90,
            extraction_metrics["faithful_claim_precision"] >= 0.95,
            fairness["fact_match_fnr_gap"] <= 0.05,
            fairness["support_fnr_gap"] <= 0.05,
            extraction_metrics["short_substantive_rate"] == 1.0,
            extraction_metrics["insufficiency_rr_zero"],
        )
    )
    report = {
        "schema_version": "experiment05.scorer-model-validation.v1",
        "split": split,
        "threshold": threshold,
        "fixture_manifest_sha256": _file_sha256(fixtures_dir / "manifest.json"),
        "models": {
            "minicheck": {
                "path": str(minicheck_model),
                "revision": minicheck_model.name,
                "config_sha256": _file_sha256(minicheck_model / "config.json"),
            },
            "granite": {
                "path": str(granite_model),
                "revision": granite_model.name,
                "config_sha256": _file_sha256(granite_model / "config.json"),
                "adapters_enabled": False,
            },
        },
        "fact_match": fact_metrics,
        "evidence_support": support_metrics,
        "claim_extraction": extraction_metrics,
        "fairness": fairness,
        "pass": passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixtures = subparsers.add_parser("fixtures")
    fixtures.add_argument("--output-dir", type=Path, required=True)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--fixtures-dir", type=Path, required=True)
    contract.add_argument("--report", type=Path, required=True)
    models = subparsers.add_parser("models")
    models.add_argument("--fixtures-dir", type=Path, required=True)
    models.add_argument("--split", choices=("calibration", "locked"), required=True)
    models.add_argument("--minicheck-model", type=Path, required=True)
    models.add_argument("--granite-model", type=Path, required=True)
    models.add_argument("--device", default="cuda:0")
    models.add_argument("--threshold", type=float, default=0.7)
    models.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixtures":
        write_fixtures(args.output_dir)
    elif args.command == "contract":
        run_contract(args.fixtures_dir, args.report)
    else:
        run_model_validation(
            fixtures_dir=args.fixtures_dir,
            split=args.split,
            minicheck_model=args.minicheck_model,
            granite_model=args.granite_model,
            device=args.device,
            threshold=args.threshold,
            report_path=args.report,
        )


if __name__ == "__main__":
    main()
