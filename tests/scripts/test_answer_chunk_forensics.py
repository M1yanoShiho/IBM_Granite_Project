"""R16's headline rests on this classification, so its failure modes are exercised here.

The entry's finding is that 69.7% of conditional misses have no answer-bearing chunk in the
top-50 pool at all, against the 21.0% that lose a sibling half to the selection cut. Those
shares are what moved the diagnosis from the selector to the retrieval side, and a
misclassification would be invisible on real data: every case would still land in some class,
the percentages would still sum, and the conclusion would simply be wrong.

The classification was checked against a fixture carrying one case of each class before R16
was read. That fixture lived in a temp directory, which is the same as not having checked it
at all a week later. This is that fixture, kept.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from answer_chunk_forensics import (  # noqa: E402
    classify_run,
    corpus_holds_answer,
    read_gold,
    read_population,
)


def candidate(rank: int, document: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": f"c-{document}-{rank}",
        "document_id": document,
        "evidence_id": f"ev-{document}-{rank}",
        "retrieval_rank": rank,
        "retrieval_score": 100.0 - rank,
        "text": text,
    }


def run_with(*, selected: list[dict[str, object]], extra: list[dict[str, object]]) -> dict:
    """A pipeline run whose selected set is the first ten candidates."""
    return {
        "query": {"query_id": "q"},
        "top_k": 50,
        "max_selected": 10,
        "candidates": {"candidates": selected + extra},
        "selected": {"evidence": selected},
    }


SELECTED = [candidate(rank, f"D{rank}", f"query terms {rank}") for rank in range(1, 11)]


def test_a_split_passage_losing_its_other_half_is_a_sibling() -> None:
    # D1 is already in the selected set, so an answer-bearing chunk from D1 below the cut is
    # the split-passage case: the half carrying the query terms won.
    run = run_with(selected=SELECTED, extra=[candidate(12, "D1", "the capital is Paris")])
    assert classify_run(run, ("paris",)) == ("sibling", 12)


def test_an_answer_in_an_unselected_document_below_the_cut_is_other() -> None:
    run = run_with(selected=SELECTED, extra=[candidate(15, "D99", "Berlin is elsewhere")])
    assert classify_run(run, ("berlin",)) == ("other", 15)


def test_no_answer_bearing_candidate_anywhere_is_a_retrieval_miss() -> None:
    run = run_with(selected=SELECTED, extra=[candidate(15, "D99", "nothing relevant")])
    assert classify_run(run, ("tokyo",)) == ("retrieval", None)


def test_a_sibling_outranks_a_closer_candidate_from_another_document() -> None:
    # The subtle one. `other` at rank 11 is nearer the cut than the sibling at rank 30, but the
    # sibling is the split passage this entry is about, so it decides the class -- and the rank
    # reported must be the sibling's, not the better-ranked stranger's.
    run = run_with(
        selected=SELECTED,
        extra=[candidate(11, "D99", "Paris is a stranger"), candidate(30, "D1", "Paris again")],
    )
    assert classify_run(run, ("paris",)) == ("sibling", 30)


def test_matching_uses_answer_match_s_own_normalisation() -> None:
    # The criteria only mean anything while a "match" here is a match there: casefolded, with
    # punctuation dropped. A private copy of this rule would quietly become a second metric.
    run = run_with(selected=SELECTED, extra=[candidate(12, "D1", "opened in NEW-YORK, 1923.")])
    assert classify_run(run, ("new york",)) == ("sibling", 12)


def test_an_answer_inside_a_selected_chunk_is_still_reported_faithfully() -> None:
    # Such a case should never reach the classifier -- answer_match would be 1 and the report
    # would exclude it -- but if the population is ever built wrongly, the class should show
    # the answer sitting above the cut rather than silently reading as a retrieval miss.
    selected = SELECTED[:-1] + [candidate(10, "D10", "the answer is Paris")]
    assert classify_run(run_with(selected=selected, extra=[]), ("paris",)) == ("sibling", 10)


def test_the_population_is_exactly_the_conditional_misses(tmp_path: Path) -> None:
    def case(query_id: str, conditional: float | None, answer: float) -> dict[str, object]:
        return {
            "query_id": query_id,
            "generator": {"generator.core.conditional_answer_match": {"value": conditional}},
            "system": {"system.core.answer_match": {"value": answer}},
        }

    report = tmp_path / "evaluation_report.json"
    report.write_text(
        json.dumps(
            {
                "per_case": [
                    case("miss", 0.0, 0.0),  # gold selected, answer not found
                    case("hit", 1.0, 1.0),  # answered
                    case("not-selected", None, 0.0),  # gold never reached the generator
                ]
            }
        ),
        encoding="utf-8",
    )
    assert read_population(report) == {"miss"}


def test_absent_is_decided_against_the_gold_documents_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus_snapshot.json"
    corpus.write_text(
        json.dumps(
            {
                "chunks": [
                    {"document_id": "D1", "text": "gold text without the span"},
                    {"document_id": "D2", "text": "Lima appears here, but D2 is not gold"},
                ]
            }
        ),
        encoding="utf-8",
    )
    wanted = {"q": (("lima",), frozenset({"D1"}))}
    # The answer exists in the corpus, but not in any gold document, so the case is absent:
    # counting it as present would hide a case that chunking cannot explain.
    assert corpus_holds_answer(corpus, wanted) == {"q"}

    wanted_reaching = {"q": (("lima",), frozenset({"D1", "D2"}))}
    assert corpus_holds_answer(corpus, wanted_reaching) == set()


def test_gold_answers_are_normalised_once_at_read_time(tmp_path: Path) -> None:
    gold = tmp_path / "gold_cases.jsonl"
    gold.write_text(
        json.dumps({"query_id": "q", "reference_answers": ["New York!"], "relevant_document_ids": ["D1"]})
        + "\n",
        encoding="utf-8",
    )
    answers, documents = read_gold(gold)["q"]
    assert answers == ("new york",)
    assert documents == frozenset({"D1"})
