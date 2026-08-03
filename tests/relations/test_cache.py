import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidence_rag.relations.cache import CachedRelationPredictor, RelationCache
from evidence_rag.relations.models import RelationLabel, RelationPrediction
from evidence_rag.relations.predictor import text_hash


class FakePredictor:
    """Records every batch it is handed, so a test can assert what the cache saved."""

    def __init__(self, *, model_version: str = "nli-v1@1111111111111111") -> None:
        self.model_version = model_version
        self.batches: list[tuple[tuple[str, str], ...]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[RelationPrediction, ...]:
        self.batches.append(tuple(pairs))
        return tuple(
            _prediction(premise, hypothesis, model_version=self.model_version)
            for premise, hypothesis in pairs
        )


def _prediction(
    premise: str = "p",
    hypothesis: str = "h",
    *,
    label: RelationLabel = RelationLabel.SUPPORTS,
    confidence: float = 0.9,
    model_version: str = "nli-v1@1111111111111111",
) -> RelationPrediction:
    return RelationPrediction(
        label=label,
        confidence=confidence,
        model_version=model_version,
        premise_hash=text_hash(premise),
        hypothesis_hash=text_hash(hypothesis),
    )


def test_a_stored_prediction_is_returned_on_a_hit(tmp_path: Path) -> None:
    cache = RelationCache(tmp_path / "edges.jsonl")

    cache.put(_prediction())

    assert cache.get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="h") == _prediction()


def test_a_different_model_version_misses(tmp_path: Path) -> None:
    """Swapping the checkpoint must not serve the previous checkpoint's edges (spec S7)."""
    cache = RelationCache(tmp_path / "edges.jsonl")

    cache.put(_prediction(model_version="nli-v1@1111111111111111"))

    assert cache.get(model_version="nli-v2@2222222222222222", premise="p", hypothesis="h") is None


def test_a_different_premise_misses(tmp_path: Path) -> None:
    """This is the property that makes the f63e905 corpus-reuse hazard inapplicable: a different
    corpus yields different premise text, so it cannot hit an entry built from another corpus."""
    cache = RelationCache(tmp_path / "edges.jsonl")

    cache.put(_prediction(premise="Kennedy won the 1960 election."))

    assert cache.get(model_version="nli-v1@1111111111111111", premise="Nixon won.", hypothesis="h") is None


def test_a_different_hypothesis_misses(tmp_path: Path) -> None:
    """One passage is scored against every claim in the window, so the hypothesis is the
    component that varies fastest. Dropping it from the key would give every claim the first
    claim's edge."""
    cache = RelationCache(tmp_path / "edges.jsonl")

    cache.put(_prediction(hypothesis="The answer is Kennedy."))

    assert cache.get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="The answer is Nixon.") is None


def test_entries_survive_into_a_fresh_cache_instance(tmp_path: Path) -> None:
    """The whole point: a cache that only lives inside one process saves nothing, because the
    expensive runs are separate SLURM jobs."""
    path = tmp_path / "edges.jsonl"
    RelationCache(path).put(_prediction())

    reopened = RelationCache(path)

    assert reopened.get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="h") == _prediction()


def test_a_stored_unknown_at_zero_confidence_is_a_hit_not_a_miss(tmp_path: Path) -> None:
    """UNKNOWN is a predicted class, not an absent answer (spec S2.5), and it is the majority
    class. If the loader or the lookup treated a falsy payload as absent, the cache would return
    a miss for the most common edge in the graph and re-run the model on every one of them."""
    path = tmp_path / "edges.jsonl"
    abstention = _prediction(label=RelationLabel.UNKNOWN, confidence=0.0)
    RelationCache(path).put(abstention)

    reopened = RelationCache(path)

    assert reopened.get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="h") == abstention
    assert reopened.get(model_version="nli-v1@1111111111111111", premise="other", hypothesis="h") is None


def test_a_truncated_tail_line_is_dropped_and_the_rest_of_the_file_survives(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A SLURM job killed mid-append leaves a half-written last line. Every entry here is
    recomputable by definition, so the cheap, loud repair is to drop that record and re-run the
    model for it; refusing to open the file would throw away a whole warm cache over one byte."""
    path = tmp_path / "edges.jsonl"
    cache = RelationCache(path)
    cache.put(_prediction(premise="first"))
    cache.put(_prediction(premise="second"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"model_version": "nli-v1@1111111111111111", "premise_ha')

    with caplog.at_level("WARNING", logger="evidence_rag.relations"):
        reopened = RelationCache(path)

    assert reopened.get(model_version="nli-v1@1111111111111111", premise="first", hypothesis="h") is not None
    assert reopened.get(model_version="nli-v1@1111111111111111", premise="second", hypothesis="h") is not None
    assert "corrupt relation cache" in caplog.text
    assert reopened.stats().dropped_records == 1


def test_the_file_is_one_self_describing_jsonl_record_per_edge(tmp_path: Path) -> None:
    """The record carries its own key, which is what makes a corrupt entry a miss rather than a
    wrong answer: there is no separate key that could disagree with the payload."""
    path = tmp_path / "edges.jsonl"
    RelationCache(path).put(_prediction())

    (line,) = path.read_text(encoding="utf-8").splitlines()

    assert json.loads(line) == {
        "model_version": "nli-v1@1111111111111111",
        "premise_hash": text_hash("p"),
        "hypothesis_hash": text_hash("h"),
        "label": "SUPPORTS",
        "confidence": 0.9,
    }


def test_two_records_disagreeing_under_one_key_raise(tmp_path: Path) -> None:
    """A deterministic predictor cannot produce two labels for one key, so this is evidence of a
    real defect -- most likely one model_version string reused across two different checkpoints.
    Silently taking either record would serve one run's edges under the other run's provenance,
    which no metric downstream could detect. Unlike a truncated tail this is never expected in
    normal operation, so it gets a human rather than a repair."""
    path = tmp_path / "edges.jsonl"
    cache = RelationCache(path)
    cache.put(_prediction(label=RelationLabel.SUPPORTS))
    cache.put(_prediction(label=RelationLabel.REFUTES))

    with pytest.raises(ValueError, match="conflicting relation cache records"):
        RelationCache(path)


def test_an_identical_record_appended_twice_is_not_a_conflict(tmp_path: Path) -> None:
    """The file is append-only, so two jobs that both miss the same pair both append it. That is
    normal, and tightening the conflict check to fire on any repeated key would make every
    concurrent re-run unopenable."""
    path = tmp_path / "edges.jsonl"
    cache = RelationCache(path)
    cache.put(_prediction())
    cache.put(_prediction())

    assert RelationCache(path).get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="h") is not None


def test_hits_misses_and_drops_are_counted_for_the_per_query_dump(tmp_path: Path) -> None:
    """Spec S2.6 requires the edge-cache hit rate in the per-query dump (G-PQ). Without a
    counter, a cache silently degraded to 0% hits looks exactly like a slow model."""
    cache = RelationCache(tmp_path / "edges.jsonl")
    cache.put(_prediction())

    cache.get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="h")
    cache.get(model_version="nli-v1@1111111111111111", premise="p", hypothesis="h")
    cache.get(model_version="nli-v1@1111111111111111", premise="absent", hypothesis="h")

    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.dropped_records) == (2, 1, 0)
    assert stats.hit_rate == pytest.approx(2 / 3)


def test_a_partial_hit_returns_predictions_in_the_callers_pair_order(tmp_path: Path) -> None:
    """The caller matches result i to pair i positionally, so a partial hit that returns cached
    and freshly computed predictions in the wrong order attaches every edge to the wrong claim.
    Nothing downstream would notice: the labels are all individually plausible."""
    cache = RelationCache(tmp_path / "edges.jsonl")
    cache.put(_prediction("b", "h"))
    cached = CachedRelationPredictor(
        predictor=FakePredictor(), cache=cache, model_version="nli-v1@1111111111111111"
    )

    predictions = cached.predict([("a", "h"), ("b", "h"), ("c", "h")])

    assert [p.premise_hash for p in predictions] == [
        text_hash("a"),
        text_hash("b"),
        text_hash("c"),
    ]


def test_a_partial_hit_sends_only_the_missing_pairs_to_the_model(tmp_path: Path) -> None:
    """The saving is the point: forwards, not lookups, are what cost |C|^2 + 20|C| per query."""
    cache = RelationCache(tmp_path / "edges.jsonl")
    cache.put(_prediction("b", "h"))
    predictor = FakePredictor()
    cached = CachedRelationPredictor(predictor=predictor, cache=cache, model_version="nli-v1@1111111111111111")

    cached.predict([("a", "h"), ("b", "h"), ("c", "h")])

    assert predictor.batches == [(("a", "h"), ("c", "h"))]


def test_a_full_hit_never_touches_the_model(tmp_path: Path) -> None:
    cache = RelationCache(tmp_path / "edges.jsonl")
    cache.put(_prediction("a", "h"))
    predictor = FakePredictor()
    cached = CachedRelationPredictor(predictor=predictor, cache=cache, model_version="nli-v1@1111111111111111")

    cached.predict([("a", "h")])

    assert predictor.batches == []


def test_a_repeated_pair_in_one_batch_costs_one_forward(tmp_path: Path) -> None:
    """Claim-claim clustering compares a claim against every other claim, so the same pair
    reaches the predictor from both ends of the comparison."""
    predictor = FakePredictor()
    cached = CachedRelationPredictor(
        predictor=predictor,
        cache=RelationCache(tmp_path / "edges.jsonl"),
        model_version="nli-v1@1111111111111111",
    )

    predictions = cached.predict([("a", "h"), ("a", "h")])

    assert predictor.batches == [(("a", "h"),)]
    assert len(predictions) == 2


def test_an_empty_batch_returns_empty_and_never_calls_the_model(tmp_path: Path) -> None:
    predictor = FakePredictor()
    cached = CachedRelationPredictor(
        predictor=predictor,
        cache=RelationCache(tmp_path / "edges.jsonl"),
        model_version="nli-v1@1111111111111111",
    )

    assert cached.predict([]) == ()
    assert predictor.batches == []


class TruncatingPredictor:
    """Hashes the premise it actually scored, not the premise it was handed.

    This is the realistic shape of the bug, not a contrived liar: spec S2.6 truncates the premise
    to `passage_chars` and then again to the tokenizer limit. The moment that truncation moves
    inside the predictor, its recorded hashes stop describing the caller's text.
    """

    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[RelationPrediction, ...]:
        return tuple(
            _prediction(premise[:4], hypothesis, model_version="nli-v1@1111111111111111")
            for premise, hypothesis in pairs
        )


def test_a_prediction_whose_hashes_do_not_describe_its_inputs_is_rejected(tmp_path: Path) -> None:
    """`put` files a record under the record's own hashes. If those hashes describe a different
    string than the caller's pair, the entry is filed under some other pair's key and is later
    served as that pair's edge -- a wrong answer no downstream metric could attribute."""
    cached = CachedRelationPredictor(
        predictor=TruncatingPredictor(),
        cache=RelationCache(tmp_path / "edges.jsonl"),
        model_version="nli-v1@1111111111111111",
    )

    with pytest.raises(ValueError, match="premise_hash"):
        cached.predict([("a long premise that gets truncated", "h")])


def test_a_prediction_carrying_another_model_version_is_rejected(tmp_path: Path) -> None:
    """Lookups use the wrapper's model_version and `put` files under the prediction's. If they
    disagree, every write lands where no read looks: a permanent 0% hit rate that presents as
    the model simply being slow."""
    cached = CachedRelationPredictor(
        predictor=FakePredictor(model_version="nli-v2@2222222222222222"),
        cache=RelationCache(tmp_path / "edges.jsonl"),
        model_version="nli-v1@1111111111111111",
    )

    with pytest.raises(ValueError, match="model_version"):
        cached.predict([("a", "h")])


def test_a_hand_written_model_version_is_refused_by_the_cache(tmp_path: Path) -> None:
    """Defence in depth over the real fix in gate0b. The key's other two components are content
    hashes; model_version is not, so a hand-written label is the one way two checkpoints can
    still collide on a key. Three fine-tuning seeds sharing one label would silently void the
    three-seed clause, so the cache refuses the shape rather than trusting the caller."""
    cache = RelationCache(tmp_path / "edges.jsonl")
    with pytest.raises(ValueError, match="not fingerprinted"):
        cache.get(model_version="relation-builder-v1", premise="p", hypothesis="h")
    with pytest.raises(ValueError, match="not fingerprinted"):
        cache.put(_prediction(model_version="relation-builder-v1"))
