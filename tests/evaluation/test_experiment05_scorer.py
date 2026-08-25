from __future__ import annotations

from collections.abc import Mapping

import pytest

from evidence_rag.evaluation.experiment05_scorer import (
    CANONICAL_ABSTENTION,
    ExtractedClaim,
    NLIEntailmentJudge,
    ScorerClaimExtractor,
    aggregate_query_scores,
    canonical_fact,
    compute_appendix_diagnostics,
    score_query,
    score_query_safely,
)


class PairJudge:
    """Exact external-judge double; assertions target the real scorer behavior."""

    def __init__(self, verdicts: Mapping[tuple[str, str], bool]) -> None:
        self.verdicts = dict(verdicts)

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.verdicts.get((premise, hypothesis), False)


class FakeNLI:
    def __init__(self, labels: Mapping[tuple[str, str], str]) -> None:
        self.labels = dict(labels)

    def classify(self, premise: str, hypothesis: str) -> str:
        return self.labels.get((premise, hypothesis), "neutral")


class FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _fact_premise(question: str, claim: str) -> str:
    return f"Question: {question}\nAnswer statement: {claim}"


def _sidecar(*, aliases: list[str] | None = None) -> dict[str, object]:
    return {
        "schema_version": "experiment05.scorer.v1",
        "dataset": "kilt-nq",
        "query_id": "q1",
        "reference_fact_groups": [
            {
                "fact_id": "fact-0001",
                "fact_question": "Which city is the capital of France?",
                "aliases": aliases or ["Paris"],
            }
        ],
        "gold_provenance": [],
    }


def _presented(text: str = "Paris is the capital of France.") -> list[dict[str, object]]:
    return [
        {
            "prompt_ordinal": 1,
            "evidence_id": "ev-1",
            "artifact_uri": "sha256://evidence/abc",
            "text_sha256": "abc",
            "token_count": 7,
            "text": text,
        }
    ]


def _output(answer: str, *, presented: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "query_id": "q1",
        "answer_text": answer,
        "abstained": False,
        "runtime_error": None,
        "presented_evidence_records": _presented() if presented is None else presented,
    }


def test_canonical_fact_normalizes_unicode_whitespace_and_alias_case_duplicates() -> None:
    assert canonical_fact("  Which city?  ", "Ｐａｒｉｓ\n") == (
        'For the question "Which city?", the answer is "Paris".'
    )


def test_supported_correct_claim_with_bound_citation_scores_all_six_dimensions() -> None:
    claim = "Paris is the capital of France."
    fact = canonical_fact("Which city is the capital of France?", "Paris")
    evidence = "Paris is the capital of France."
    judge = PairJudge(
        {
            (_fact_premise("Which city is the capital of France?", claim), fact): True,
            (evidence, claim): True,
        }
    )

    scored = score_query(
        _sidecar(),
        _output("Paris is the capital of France [1]."),
        (ExtractedClaim("claim-1", claim, sentence_index=0),),
        judge,
    )

    assert scored["metrics"] == {
        "rfc": 1.0,
        "vrfc": 1.0,
        "ucr": 0.0,
        "cp": 1.0,
        "cr": 1.0,
        "rr": 1.0,
    }
    assert scored["counts"] == {
        "reference_facts": 1,
        "matched_reference_facts": 1,
        "matched_and_validly_cited_reference_facts": 1,
        "claims": 1,
        "unsupported_claims": 0,
        "citation_links": 1,
        "supporting_citation_links": 1,
        "duplicate_citation_links": 0,
        "invalid_citation_links": 0,
    }


def test_vrfc_requires_the_same_fact_matching_claim_to_own_the_valid_citation() -> None:
    correct_claim = "Paris is the capital of France."
    other_claim = "The Eiffel Tower is in Paris."
    fact = canonical_fact("Which city is the capital of France?", "Paris")
    evidence = "The Eiffel Tower is in Paris."
    judge = PairJudge(
        {
            (_fact_premise("Which city is the capital of France?", correct_claim), fact): True,
            (evidence, other_claim): True,
        }
    )

    scored = score_query(
        _sidecar(),
        _output("Paris is the capital of France. The Eiffel Tower is in Paris [1].", presented=_presented(evidence)),
        (
            ExtractedClaim("claim-1", correct_claim, sentence_index=0),
            ExtractedClaim("claim-2", other_claim, sentence_index=1),
        ),
        judge,
    )

    assert scored["metrics"] == {
        "rfc": 1.0,
        "vrfc": 0.0,
        "ucr": 0.5,
        "cp": 1.0,
        "cr": 0.5,
        "rr": 1.0,
    }


def test_illegal_citation_is_kept_in_precision_denominator_and_cannot_support() -> None:
    claim = "Paris is the capital of France."
    fact = canonical_fact("Which city is the capital of France?", "Paris")
    evidence = "Paris is the capital of France."
    judge = PairJudge(
        {
            (_fact_premise("Which city is the capital of France?", claim), fact): True,
            (evidence, claim): True,
        }
    )

    scored = score_query(
        _sidecar(),
        _output("Paris is the capital of France [9]."),
        (ExtractedClaim("claim-1", claim, sentence_index=0),),
        judge,
    )

    assert scored["metrics"] == {
        "rfc": 1.0,
        "vrfc": 0.0,
        "ucr": 0.0,
        "cp": 0.0,
        "cr": 0.0,
        "rr": 1.0,
    }
    assert scored["counts"]["invalid_citation_links"] == 1


def test_duplicate_claims_and_duplicate_links_do_not_dilute_error_rates() -> None:
    claim = "Paris is the capital of France."
    fact = canonical_fact("Which city is the capital of France?", "Paris")
    evidence = "Paris is the capital of France."
    fact_premise = _fact_premise("Which city is the capital of France?", claim)
    judge = PairJudge(
        {
            (claim, claim): True,
            (fact_premise, fact): True,
            (evidence, claim): True,
        }
    )

    scored = score_query(
        _sidecar(),
        _output("Paris is the capital of France [1][1]. Paris is the capital of France [1]."),
        (
            ExtractedClaim("claim-1", claim, sentence_index=0),
            ExtractedClaim("claim-2", claim, sentence_index=1),
        ),
        judge,
    )

    assert scored["counts"]["claims"] == 1
    assert scored["counts"]["citation_links"] == 1
    assert scored["counts"]["duplicate_citation_links"] == 2
    assert scored["metrics"]["ucr"] == 0.0
    assert scored["metrics"]["cp"] == 1.0


@pytest.mark.parametrize("answer", ["", CANONICAL_ABSTENTION])
def test_empty_or_canonical_abstention_is_non_substantive(answer: str) -> None:
    scored = score_query(_sidecar(), _output(answer), (), PairJudge({}))

    assert scored["metrics"] == {
        "rfc": 0.0,
        "vrfc": 0.0,
        "ucr": None,
        "cp": 0.0,
        "cr": 0.0,
        "rr": 0.0,
    }


def test_dataset_aggregation_uses_macro_metrics_but_claim_micro_ucr() -> None:
    rows = [
        {
            "metrics": {"rfc": 1.0, "vrfc": 1.0, "ucr": 0.5, "cp": 1.0, "cr": 1.0, "rr": 1.0},
            "counts": {"claims": 2, "unsupported_claims": 1},
        },
        {
            "metrics": {"rfc": 0.0, "vrfc": 0.0, "ucr": 1.0, "cp": 0.0, "cr": 0.0, "rr": 1.0},
            "counts": {"claims": 1, "unsupported_claims": 1},
        },
        {
            "metrics": {"rfc": 0.0, "vrfc": 0.0, "ucr": None, "cp": 0.0, "cr": 0.0, "rr": 0.0},
            "counts": {"claims": 0, "unsupported_claims": 0},
        },
    ]

    aggregate = aggregate_query_scores(rows)

    assert aggregate["metrics"] == {
        "rfc": pytest.approx(1 / 3),
        "vrfc": pytest.approx(1 / 3),
        "ucr": pytest.approx(2 / 3),
        "cp": pytest.approx(1 / 3),
        "cr": pytest.approx(1 / 3),
        "rr": pytest.approx(2 / 3),
    }
    assert aggregate["n_total"] == 3
    assert aggregate["n_answered"] == 2
    assert aggregate["n_claims"] == 3
    assert aggregate["n_undefined_ucr"] == 1


def test_nli_adapter_exposes_only_entailment_as_true() -> None:
    judge = NLIEntailmentJudge(
        FakeNLI({("support", "claim"): "entailment", ("conflict", "claim"): "contradiction"})
    )

    assert judge.entails("support", "claim") is True
    assert judge.entails("conflict", "claim") is False
    assert judge.entails("unknown", "claim") is False


def test_scorer_claim_extractor_reads_question_and_raw_answer_and_drops_meta() -> None:
    generator = FakeGenerator(
        '{"claims":['
        '{"source_text":"Paris","text":"Paris is the capital of France."},'
        '{"source_text":"This answer uses the evidence.",'
        '"text":"This answer uses the evidence."}'
        ']}'
    )
    extractor = ScorerClaimExtractor(generator)

    claims = extractor.extract(
        "Which city is the capital of France?",
        "Paris. This answer uses the evidence.",
    )

    assert claims == (ExtractedClaim("claim-1", "Paris is the capital of France.", 0),)
    assert "Which city is the capital of France?" in generator.prompts[0]
    assert "Paris. This answer uses the evidence." in generator.prompts[0]

    records = extractor.last_records
    assert len(records) == 1
    assert records[0].source_text == "Paris"
    assert (records[0].source_start, records[0].source_end) == (0, 5)


def test_scorer_claim_extractor_accepts_a_short_substantive_fact_but_rejects_insufficiency() -> None:
    short = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"Paris","text":"Paris is the answer to the question."}]}'
        )
    )
    insufficiency = ScorerClaimExtractor(FakeGenerator('{"claims":[]}'))

    assert short.extract("Which city?", "Paris") == (
        ExtractedClaim("claim-1", "Paris is the answer to the question.", 0),
    )
    assert insufficiency.extract(
        "Which city?", "The provided evidence is insufficient to answer."
    ) == ()


def test_scorer_claim_extractor_repairs_unescaped_quotes_without_changing_text() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"William G. "Parson" Brownlow was elected.",'
            '"text":"William G. "Parson" Brownlow was elected."}]}'
        )
    )

    claims = extractor.extract(
        "Who was elected?", 'William G. "Parson" Brownlow was elected.'
    )

    assert claims == (
        ExtractedClaim(
            "claim-1", 'William G. "Parson" Brownlow was elected.', 0
        ),
    )
    assert extractor.last_records[0].source_text == (
        'William G. "Parson" Brownlow was elected.'
    )


def test_scorer_claim_extractor_removes_invalid_apostrophe_escape() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            r'{"claims":[{"source_text":"Michael sings \"Mary\".",'
            r'"text":"Michael sings \'Mary\'."}]}'
        )
    )

    claims = extractor.extract("Who sings it?", 'Michael sings "Mary".')

    assert claims == (ExtractedClaim("claim-1", "Michael sings 'Mary'.", 0),)


def test_scorer_claim_extractor_repairs_mismatched_array_closer() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"Paris","text":"Paris is in France."}}'
        )
    )

    claims = extractor.extract("Where is Paris?", "Paris")

    assert claims == (ExtractedClaim("claim-1", "Paris is in France.", 0),)


def test_scorer_claim_extractor_appends_missing_delimiters() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator('{"claims":[{"source_text":"Paris","text":"Paris is in France."')
    )

    claims = extractor.extract("Where is Paris?", "Paris")

    assert claims == (ExtractedClaim("claim-1", "Paris is in France.", 0),)


def test_scorer_claim_extractor_closes_source_string_and_preserves_source_comma() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"Paris [1],"text":"Paris is in France."}]}'
        )
    )

    claims = extractor.extract("Where is Paris?", "Paris [1], and Lyon [2].")

    assert claims == (ExtractedClaim("claim-1", "Paris is in France.", 0),)
    assert extractor.last_records[0].source_text == "Paris [1],"


def test_scorer_claim_extractor_closes_source_string_before_delimiter_comma() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"Paris,"text":"Paris is in France."}]}'
        )
    )

    claims = extractor.extract("Where is Paris?", "Paris")

    assert claims == (ExtractedClaim("claim-1", "Paris is in France.", 0),)
    assert extractor.last_records[0].source_text == "Paris"


def test_scorer_claim_extractor_inserts_missing_comma_between_claim_fields() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"Paris [1]," '
            '"text":"Paris is in France."}]}'
        )
    )

    claims = extractor.extract("Where is Paris?", "Paris [1], and Lyon [2].")

    assert claims == (ExtractedClaim("claim-1", "Paris is in France.", 0),)
    assert extractor.last_records[0].source_text == "Paris [1],"


def test_scorer_claim_extractor_skips_nonlocatable_claim_after_quote_repair() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":['
            '{"source_text":"First fact "" [1]","text":"First claim."},'
            '{"source_text":"Second fact "".","text":"Second claim."}'
            ']}'
        )
    )

    claims = extractor.extract("What happened?", 'First fact "". [1] Second fact "".')

    assert claims == (ExtractedClaim("claim-1", "Second claim.", 1),)
    assert extractor.last_records[0].source_text == 'Second fact "".'


def test_scorer_claim_extractor_repairs_doubled_quotes_before_a_source_comma() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"In 1858 Rowland Macy established a new '
            'store named ""R. H. Macy & Company"", where it stayed on the same site '
            'for nearly forty years, in New York City.","text":"Rowland Macy '
            'established R. H. Macy & Company in New York City."}]}"'
        )
    )

    claims = extractor.extract(
        "In which city was the store?",
        'In 1858 Rowland Macy established a new store named ""R. H. Macy & '
        'Company"", where it stayed on the same site for nearly forty years, in '
        "New York City. [unverified]",
    )

    assert claims == (
        ExtractedClaim(
            "claim-1",
            "Rowland Macy established R. H. Macy & Company in New York City.",
            0,
        ),
    )
    assert '""R. H. Macy & Company""' in extractor.last_records[0].source_text


def test_scorer_claim_extractor_quotes_a_bare_fixed_claim_key() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            r'''{"claims":[{"source_text":"The 1969 film \"They Shoot Horses, Don't They?\"",text:"They Shoot Horses, Don't They? is a 1969 film."}]}"'''
        )
    )

    claims = extractor.extract(
        "Which 1969 film?",
        'The 1969 film "They Shoot Horses, Don\'t They?", [1].',
    )

    assert claims == (
        ExtractedClaim(
            "claim-1", "They Shoot Horses, Don't They? is a 1969 film.", 0
        ),
    )


def test_scorer_claim_extractor_quotes_fixed_key_missing_only_its_opening_quote() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            r'''{"claims":[{"source_text":"Michael Billington of \"The Guardian\" wrote, \"The evening can be quickly summed up as 'great songs, daft book' [10].\"",text":"Michael Billington of The Guardian wrote the songs for the musical Top Hat."}]}"'''
        )
    )

    claims = extractor.extract(
        "Who wrote the songs for the musical Top Hat?",
        'Michael Billington of "The Guardian" wrote, “The evening can be quickly '
        "summed up as 'great songs, daft book' [10].",
    )

    # The structural repair must make the JSON readable without rewriting the
    # generated source string.  Its non-locatable rewrite is then skipped by the
    # existing scorer contract.
    assert claims == ()
    assert extractor.last_records == ()


def test_scorer_claim_extractor_uses_locatable_source_when_text_key_is_missing() -> None:
    extractor = ScorerClaimExtractor(
        FakeGenerator(
            '{"claims":[{"source_text":"Head cheese is alternatively called '
            '"saltisón"."},{"source_text":"Head cheese is alternatively called '
            '"saltisón"."}]}'
        )
    )

    claims = extractor.extract(
        "What is head cheese alternatively called?",
        'Head cheese is alternatively called "saltisón". [unverified]',
    )

    expected = 'Head cheese is alternatively called "saltisón".'
    assert claims == (
        ExtractedClaim("claim-1", expected, 0),
        ExtractedClaim("claim-2", expected, 0),
    )
    assert tuple(record.source_text for record in extractor.last_records) == (
        expected,
        expected,
    )


def test_appendix_diagnostics_use_exact_support_units_not_only_evidence_ids() -> None:
    diagnostics = compute_appendix_diagnostics(
        reference_fact_ids=("f1", "f2"),
        retrieved_support_units=(("f1", "passage-1", "full-text-hash"),),
        # The same canonical passage survives, but pruning removed the supporting
        # sentence, so its exact-text hash is different and the support unit is lost.
        selected_support_units=(("f1", "passage-1", "pruned-text-hash"),),
        selector_input_tokens=100,
        selected_tokens=60,
        presented_tokens=40,
    )

    assert diagnostics == {
        "er_at_10": 0.5,
        "selr": 1.0,
        "selection_reduction_rate": 0.4,
        "presented_reduction_rate": 0.6,
    }


def test_scorer_failure_stays_in_the_common_denominator_as_zero_score() -> None:
    malformed = _output("Paris [1].")
    malformed["presented_evidence_records"] = "not-a-list"

    scored = score_query_safely(
        _sidecar(),
        malformed,
        (ExtractedClaim("claim-1", "Paris is the capital of France.", 0),),
        PairJudge({}),
    )

    assert scored["metrics"] == {
        "rfc": 0.0,
        "vrfc": 0.0,
        "ucr": None,
        "cp": 0.0,
        "cr": 0.0,
        "rr": 0.0,
    }
    assert scored["scorer_error"] is True
    assert scored["scorer_error_type"] == "ValueError"


def test_supported_explanation_preserves_all_metrics_but_unsupported_fact_worsens_ucr() -> None:
    main_claim = "Paris is the capital of France."
    explanation = "Paris is located in France."
    fact = canonical_fact("Which city is the capital of France?", "Paris")
    fact_premise = _fact_premise("Which city is the capital of France?", main_claim)
    presented = [
        {
            "prompt_ordinal": 1,
            "evidence_id": "ev-1",
            "text": main_claim,
        },
        {
            "prompt_ordinal": 2,
            "evidence_id": "ev-2",
            "text": explanation,
        },
    ]
    judge = PairJudge(
        {
            (fact_premise, fact): True,
            (main_claim, main_claim): True,
            (explanation, explanation): True,
        }
    )
    short = score_query(
        _sidecar(),
        _output("Paris is the capital of France [1].", presented=presented),
        (ExtractedClaim("claim-1", main_claim, 0),),
        judge,
    )
    expanded = score_query(
        _sidecar(),
        _output(
            "Paris is the capital of France [1]. Paris is located in France [2].",
            presented=presented,
        ),
        (
            ExtractedClaim("claim-1", main_claim, 0),
            ExtractedClaim("claim-2", explanation, 1),
        ),
        judge,
    )
    unsupported = score_query(
        _sidecar(),
        _output(
            "Paris is the capital of France [1]. Paris has a population of ten million.",
            presented=presented,
        ),
        (
            ExtractedClaim("claim-1", main_claim, 0),
            ExtractedClaim("claim-2", "Paris has a population of ten million.", 1),
        ),
        judge,
    )

    assert expanded["metrics"] == short["metrics"]
    assert unsupported["metrics"]["ucr"] > short["metrics"]["ucr"]
