"""Relation-edge cache, keyed on (model_version, premise_hash, hypothesis_hash) (design §2.1).

The relation layer costs ~|C|² encoder forwards for claim–claim clustering plus ~20×|C| for
passage–claim prediction, per query (§2.4). Nothing about that is amortised inside one process:
the runs that matter are separate SLURM jobs. So this cache persists to disk, and the file is the
cache — an instance is just an index over it.

WHY THIS IS NOT THE f63e905 HAZARD
----------------------------------
Commit f63e905 fixed a retrieval-index cache that served one corpus's index for a different
corpus, invalidating a whole certification round. Someone will eventually ask whether this cache
can do the same. It cannot, and the reason is the shape of the key, not the care of the caller:

    that cache was keyed on CONFIGURATION -- a description of the inputs. Two different corpora
    could be described identically, so a stale entry was reachable.

    this cache is keyed on CONTENT -- sha256 of the premise and hypothesis themselves. A
    different corpus yields different premise text, hence different premise_hash, hence a MISS.
    There is no way to describe your way onto another corpus's entry.

The fix there was to add a corpus fingerprint. Here the fingerprint IS the key, for both texts.
Do not add a corpus/config component: it would be redundant with the content hashes, and it would
reintroduce exactly the description-based reachability that caused the incident.

WHERE IT CAN STILL GO STALE
---------------------------
`model_version` is the one component that is NOT content-addressed. It is a free-form string the
caller passes to `NLIRelationPredictor`, and nothing binds it to the actual weights. Swap the
checkpoint, retrain, or change the score_fn's label-index mapping while leaving the string alone,
and every lookup hits and returns the OLD model's edges. That is the f63e905 failure mode exactly,
surviving in the one dimension the content hashes do not cover. Mitigation is at the caller:
derive `model_version` from checkpoint identity (HF revision sha, or a hash of the weights), never
from a hand-written label. `_load` raising on conflicting records catches this only when two runs
happen to interleave; a warm cache just keeps hitting, so the version string is the real control.

The other stale path is transformation inside the predictor. The key describes the strings handed
to `predict()`, so any truncation or normalisation applied *inside* the predictor is invisible to
it -- and §2.6 does truncate premises. Changing `passage_chars` must therefore change the text
before it reaches this cache, or the same key will name a different model input.
`CachedRelationPredictor._check_provenance` enforces this by rejecting any prediction whose
recorded hashes do not describe the pair it was given.

Residual, accepted: `text_hash` keeps 16 hex chars (64 bits), so two distinct texts collide with
probability ~n²/2^65 -- about 3e-10 across 10^5 premises, and a false hit needs the premise AND
hypothesis to collide together. Not worth widening; worth knowing it is not literally zero.

ON-DISK FORMAT
--------------
Append-only JSONL, one self-describing record per edge, matching the JSONL convention already used
by `qa2d.py`. Append-only is the durability story: a job killed at hour five keeps every edge it
paid for, and no rewrite pass can lose the file. The cost is that a killed job can leave a partial
final line -- see `_load` for the deliberate split between corruption that is repaired and
corruption that stops the run.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_rag.relations.models import RelationLabel, RelationPrediction
from evidence_rag.relations.predictor import (
    RelationPredictor,
    require_fingerprinted_version,
    text_hash,
)

logger = logging.getLogger("evidence_rag.relations")

CacheKey = tuple[str, str, str]


def _key_of(prediction: RelationPrediction) -> CacheKey:
    """The key a record files itself under, read off the record's own provenance fields.

    Nothing else may compute a stored record's key. A record that carried a key separately from
    its payload could disagree with it, and a disagreeing entry is served as a wrong answer for
    some other pair rather than as a miss.
    """
    return (
        require_fingerprinted_version(prediction.model_version),
        prediction.premise_hash,
        prediction.hypothesis_hash,
    )


@dataclass(frozen=True)
class CacheStats:
    """Goes into the per-query dump (spec S2.6, G-PQ)."""

    hits: int
    misses: int
    dropped_records: int

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class RelationCache:
    """An index over one append-only JSONL file of predicted edges.

    The file is read once at construction, so a live instance does not see writes made by another
    process after that point. That is deliberate and harmless: a stale-by-omission index only
    misses, and a miss only costs a forward pass.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[CacheKey, RelationPrediction] = {}
        self.dropped_records = 0
        self.hits = 0
        self.misses = 0
        self._load()

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self.hits, misses=self.misses, dropped_records=self.dropped_records
        )

    def _load(self) -> None:
        """Two corruption classes, two deliberately different answers.

        A record that does not parse is DROPPED, counted and warned about. Every entry here is
        recomputable by definition -- that is what makes this safe, and it is the property the
        `qa2d.py` cache lacks, which is why a miss there is a hard error and a miss here is not.
        A half-written last line is the expected result of a SLURM kill mid-append, so refusing to
        open the file would throw away a whole warm cache to save one forward pass.

        Two records that DISAGREE under one key stop the run. A deterministic predictor cannot
        produce both, so this is evidence of a real defect (see the module docstring on
        `model_version` reuse), and picking a winner would serve one checkpoint's edges under
        another's provenance with nothing downstream able to tell. Never expected in normal
        operation, so it gets a human rather than a repair.
        """
        if not self.path.is_file():
            return
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prediction = RelationPrediction(
                    label=RelationLabel(record["label"]),
                    confidence=float(record["confidence"]),
                    model_version=str(record["model_version"]),
                    premise_hash=str(record["premise_hash"]),
                    hypothesis_hash=str(record["hypothesis_hash"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                self.dropped_records += 1
                logger.warning(
                    "Dropping corrupt relation cache record at %s:%d; the pair it held will be "
                    "recomputed",
                    self.path,
                    number,
                    exc_info=True,
                )
                continue
            key = _key_of(prediction)
            previous = self._entries.get(key)
            if previous is not None and previous != prediction:
                raise ValueError(
                    f"conflicting relation cache records for key {key!r} at {self.path}:{number}: "
                    f"{previous} then {prediction}. A deterministic predictor cannot produce both, "
                    f"so either the file was edited or one model_version names two checkpoints; "
                    f"fix the version string and delete the cache rather than picking a winner."
                )
            self._entries[key] = prediction

    def get(
        self, *, model_version: str, premise: str, hypothesis: str
    ) -> RelationPrediction | None:
        require_fingerprinted_version(model_version)
        found = self._entries.get((model_version, text_hash(premise), text_hash(hypothesis)))
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, prediction: RelationPrediction) -> None:
        self._entries[_key_of(prediction)] = prediction
        record = {
            "model_version": prediction.model_version,
            "premise_hash": prediction.premise_hash,
            "hypothesis_hash": prediction.hypothesis_hash,
            "label": prediction.label.value,
            "confidence": prediction.confidence,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class CachedRelationPredictor:
    """A `RelationPredictor` that consults `RelationCache` before calling the real one."""

    def __init__(
        self,
        *,
        predictor: RelationPredictor,
        cache: RelationCache,
        model_version: str,
    ) -> None:
        self.predictor = predictor
        self.cache = cache
        self.model_version = model_version

    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[RelationPrediction, ...]:
        found: dict[tuple[str, str], RelationPrediction] = {}
        missing: list[tuple[str, str]] = []
        for pair in pairs:
            if pair in found or pair in missing:
                continue
            cached = self.cache.get(
                model_version=self.model_version, premise=pair[0], hypothesis=pair[1]
            )
            if cached is None:
                missing.append(pair)
            else:
                found[pair] = cached
        if missing:
            for pair, prediction in zip(missing, self.predictor.predict(missing), strict=True):
                self._check_provenance(pair, prediction)
                self.cache.put(prediction)
                found[pair] = prediction
        return tuple(found[pair] for pair in pairs)

    def _check_provenance(
        self, pair: tuple[str, str], prediction: RelationPrediction
    ) -> None:
        """Refuse to store a prediction whose key does not describe the pair that produced it.

        `RelationCache.put` files a record under the record's own three fields, so an unchecked
        prediction can be filed under a key the wrapper will never look up (a silent 0% hit rate)
        or -- worse -- under a key that belongs to some genuinely different pair, whose edge it is
        then served as. This is the only place that can catch either, because after `put` the
        record is indistinguishable from an honest one.
        """
        premise, hypothesis = pair
        expected = {
            "model_version": (self.model_version, prediction.model_version),
            "premise_hash": (text_hash(premise), prediction.premise_hash),
            "hypothesis_hash": (text_hash(hypothesis), prediction.hypothesis_hash),
        }
        wrong = {
            field: values for field, values in expected.items() if values[0] != values[1]
        }
        if wrong:
            detail = ", ".join(
                f"{field}: expected {want!r}, got {got!r}" for field, (want, got) in wrong.items()
            )
            raise ValueError(
                f"predictor returned a prediction that does not describe its input pair "
                f"({detail}). The predictor must hash exactly the strings it was handed; apply "
                f"any truncation or normalisation before the cache, not inside the predictor."
            )
