"""Half 1.6 -- verifier triage over ONE shared pair set (measurement only).

Answers the four questions in local/generator-B-verifier-triage-guide-v2.md:

  1. does a grounding-specific verifier beat general NLI enough to fix recall?
  2. can a softmax threshold rescue recall at acceptable precision cost?
  3. after atomic decomposition, what fraction of claims stay unverifiable from
     any single chunk?
  4. does Granite-as-judge beat the best model-based verifier?

Nothing here mutates production code: `nli.py`, `attribution.py` and
`entity_check.py` are imported and called, never edited, and no threshold is
committed anywhere. The script writes numbers to docs/, the decision follows
separately.

HOLD-OUT: HotpotQA, RGB and MuSiQue-Full are the closed final test sets and are
deliberately never loaded here. Calibration data only: ALCE/ASQA (single-hop
grounding) + 2WikiMultihopQA (multi-hop decomposition) + a synthetic entity-swap
slice built on top of the ASQA pairs.

Datasets are pulled with stdlib urllib (ALCE tar) and pandas/pyarrow (2Wiki
parquet); no new dependency beyond the installed `benchmark`/`granite` extras.
Set `TRIAGE_DATA_DIR` (or `MODEL_CACHE_DIR`) to keep the ~450MB ALCE tar off the
home filesystem on a cluster -- see docs/hpc-run-log.md.

Arms A/B/C/D-minicheck run on CPU. Arm D-true (google/t5_xxl_true_nli_mixture,
42.5GB fp32) and arm E (Granite-as-judge: `granite3b` / `granite8b`, a
capacity control on the LLM-as-a-judge path) need a GPU node and are selected
with `--arms`; they are not run by default.

Usage (local, CPU arms):
    PYTHONPATH=src python scripts/verifier_triage.py --arms base,large,minicheck
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_rag.generator.entity_check import (  # noqa: E402
    EntityConsistencyChecker,
    RuleBasedEntityExtractor,
)
from evidence_rag.generator.nli import DebertaNLIModel, normalize_nli_label  # noqa: E402

ALCE_URL = "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar"
ALCE_MEMBER = "ALCE-data/asqa_eval_gtr_top100.json"
TWIKI_URL = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet"

RESULTS_PATH = REPO_ROOT / "docs" / "generator" / "verifier-triage-results.md"

Verdict = Literal["entailment", "contradiction", "neutral"]


# --------------------------------------------------------------------------
# data location
# --------------------------------------------------------------------------


def data_dir() -> Path:
    explicit = os.getenv("TRIAGE_DATA_DIR")
    if explicit:
        target = Path(explicit)
    else:
        cache = os.getenv("MODEL_CACHE_DIR")
        target = Path(cache) / "verifier-triage" if cache else Path.home() / ".cache" / "verifier-triage"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    print(f"[data] downloading {url} -> {destination}", flush=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 - a download failure must be loud
        partial.unlink(missing_ok=True)
        raise SystemExit(f"could not download {url}: {exc}") from exc
    partial.replace(destination)
    return destination


def ensure_asqa() -> list[dict[str, Any]]:
    """ALCE's ASQA eval file: gold long answers, human `knowledge` annotations,
    and the top-100 GTR passages per question (the hard-negative pool)."""
    root = data_dir()
    extracted = root / "asqa_eval_gtr_top100.json"
    if not extracted.exists():
        tar_path = _download(ALCE_URL, root / "ALCE-data.tar")
        print("[data] extracting ASQA eval file", flush=True)
        with tarfile.open(tar_path) as archive:
            member = archive.extractfile(ALCE_MEMBER)
            if member is None:
                raise SystemExit(f"{ALCE_MEMBER} missing from the ALCE tar")
            extracted.write_bytes(member.read())
    with extracted.open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_2wiki() -> Any:
    """2WikiMultihopQA dev split. NOTE: this is the script's one loading
    dependency that pyproject does not declare -- reading the parquet needs
    pandas + pyarrow. Both are present transitively in the current dev env; they
    are deliberately NOT added to an extra, since this pass is measurement only
    and pyproject is production config."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "reading the 2Wiki parquet needs pandas + pyarrow: pip install pandas pyarrow"
        ) from exc

    path = _download(TWIKI_URL, data_dir() / "2wiki_dev.parquet")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# shared text helpers
# --------------------------------------------------------------------------

STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been being by with as that "
    "this it its from has have had also his her their he she they there which who whom "
    "but not into than then when while about over under after before".split()
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def coverage(needle: str, haystack: str) -> float:
    """Fraction of the needle's content tokens that appear in the haystack."""
    needle_tokens = set(content_tokens(needle))
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & set(content_tokens(haystack))) / len(needle_tokens)


def longest_common_run(claim: str, premise: str) -> int:
    """Longest run of contiguous content tokens the claim and premise share.

    This is the near-verbatim guard the guide asks for: a claim that is a copied
    span of its own premise makes entailment trivially detectable and inflates
    recall into meaninglessness. Claims above the cap are dropped, and the
    surviving distribution is reported so the number can be judged.
    """
    left, right = content_tokens(claim), content_tokens(premise)
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for index in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        for position in range(1, len(right) + 1):
            if left[index - 1] == right[position - 1]:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


# --------------------------------------------------------------------------
# the shared pair set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One (premise, hypothesis) judgement every arm makes identically."""

    cell: str
    gold: Literal["entailment", "neutral"]
    premise: str
    hypothesis: str
    group: str
    """Which claim this pair belongs to -- lets the multi-hop cell aggregate
    'did ANY single chunk entail this claim' over its own gold passages."""
    note: str = ""


@dataclass(frozen=True)
class MultiHopItem:
    """One 2Wiki question, kept whole so decomposition can be measured on it."""

    item_id: str
    composed_claim: str
    atomic_claims: tuple[str, ...]
    gold_passages: tuple[str, ...]
    """Same order as ``atomic_claims``: passage i is the gold evidence for
    atomic claim i."""
    distractors: tuple[str, ...]


def build_asqa_pairs(
    data: list[dict[str, Any]],
    n_per_label: int,
    rng: random.Random,
    max_verbatim_run: int = 8,
) -> list[Pair]:
    """entailment = an answer sentence + the retrieved passage carrying the human
    `knowledge` sentence it was written from.
    hard neutral = the SAME sentence + another top-ranked passage retrieved for
    the SAME question that does not carry that knowledge.

    The negative pool is deliberately the question's own top-20 GTR results, so
    negatives are topically adjacent by construction -- a random-passage pool
    produces a falsely reassuring false-positive rate.
    """
    positives: list[Pair] = []
    negatives: list[Pair] = []
    order = list(range(len(data)))
    rng.shuffle(order)

    for index in order:
        if len(positives) >= n_per_label and len(negatives) >= n_per_label:
            break
        sample = data[index]
        answer_sentences = sentences(sample.get("answer", ""))
        docs = sample.get("docs", [])
        if not answer_sentences or len(docs) < 20:
            continue

        knowledge = [
            item["content"]
            for annotation in sample.get("annotations", [])
            for item in annotation.get("knowledge", [])
            if item.get("content")
        ]
        used_sentences: set[str] = set()
        for content in knowledge:
            # 1. the passage that actually carries this knowledge sentence
            scored = [(coverage(content, doc["text"]), position) for position, doc in enumerate(docs)]
            best_cover, best_doc = max(scored)
            if best_cover < 0.9:
                continue
            # 2. the answer sentence that knowledge was written into
            claim, claim_overlap = max(
                ((s, coverage(content, s)) for s in answer_sentences),
                key=lambda pair: pair[1],
            )
            if claim_overlap < 0.25 or claim in used_sentences:
                continue
            # the claim must also be recoverable FROM the passage, otherwise the
            # sentence carries facts this passage never states
            if coverage(claim, docs[best_doc]["text"]) < 0.6:
                continue
            # ...but it must not be a copied span of that passage
            if longest_common_run(claim, docs[best_doc]["text"]) >= max_verbatim_run:
                continue
            used_sentences.add(claim)

            if len(positives) < n_per_label:
                positives.append(
                    Pair(
                        cell="asqa",
                        gold="entailment",
                        premise=docs[best_doc]["text"],
                        hypothesis=claim,
                        group=f"{sample['sample_id']}::{len(positives)}",
                        note=f"knowledge_cover={best_cover:.2f}",
                    )
                )
            # 3. a same-question, topically adjacent passage that is NOT support
            if len(negatives) < n_per_label:
                candidates = [
                    doc
                    for position, doc in enumerate(docs[:20])
                    if position != best_doc
                    and coverage(content, doc["text"]) < 0.45
                    and coverage(claim, doc["text"]) < 0.45
                ]
                if candidates:
                    negatives.append(
                        Pair(
                            cell="asqa",
                            gold="neutral",
                            premise=rng.choice(candidates)["text"],
                            hypothesis=claim,
                            group=f"{sample['sample_id']}::neg{len(negatives)}",
                            note="same-question top-20 distractor",
                        )
                    )
            break  # at most one claim per question keeps topics independent

    return positives[:n_per_label] + negatives[:n_per_label]


def _triple_claim(subject: str, relation: str, obj: str) -> str:
    return f"The {relation} of {subject} is {obj}."


def _composed_claim(chain: list[list[str]]) -> str:
    """[(A, r1, B), (B, r2, C)] -> 'The r2 of the r1 of A is C.'"""
    return f"The {chain[1][1]} of the {chain[0][1]} of {chain[0][0]} is {chain[1][2]}."


def build_multihop_items(frame: Any, limit: int, rng: random.Random) -> list[MultiHopItem]:
    """2Wiki compositional 2-hop questions, with the gold evidence triples used
    as a DECOMPOSITION ORACLE (a perfect A2) so the verifier is measured without
    the claim splitter's own quality as a confound.

    Claims are built from the annotated triples by a fixed template -- no LLM
    paraphrase of the passages, which would make entailment trivially detectable.
    """
    rows = frame.to_dict("records")
    rng.shuffle(rows)
    items: list[MultiHopItem] = []
    for row in rows:
        if len(items) >= limit:
            break
        if row["type"] != "compositional":
            continue
        chain = json.loads(row["evidences"])
        facts = json.loads(row["supporting_facts"])
        context = {title: " ".join(body) for title, body in json.loads(row["context"])}
        if len(chain) != 2 or chain[0][2] != chain[1][0]:
            continue
        titles = [title for title, _index in facts]
        if len(titles) != 2 or any(title not in context for title in titles):
            continue
        # supporting_facts order follows the reasoning chain, same as `evidences`
        gold = tuple(context[title] for title in titles)
        # label guard: hop i's object must actually be stated in hop i's passage,
        # otherwise the "entailment" gold label is wrong and recall is understated
        if any(triple[2] not in passage for triple, passage in zip(chain, gold, strict=True)):
            continue
        atomic = tuple(_triple_claim(*triple) for triple in chain)
        distractors = tuple(
            body for title, body in context.items() if title not in set(titles)
        )
        if len(distractors) < 2:
            continue
        items.append(
            MultiHopItem(
                item_id=str(row["_id"]),
                composed_claim=_composed_claim(chain),
                atomic_claims=atomic,
                gold_passages=gold,
                distractors=tuple(rng.sample(distractors, 2)),
            )
        )
    return items


def multihop_pairs(items: list[MultiHopItem]) -> list[Pair]:
    pairs: list[Pair] = []
    for item in items:
        union = "\n".join(item.gold_passages)
        # the undecomposed multi-hop claim against each single gold chunk
        for position, passage in enumerate(item.gold_passages):
            pairs.append(
                Pair(
                    cell="2wiki-composed",
                    gold="entailment",
                    premise=passage,
                    hypothesis=item.composed_claim,
                    group=f"{item.item_id}::composed",
                    note=f"gold-hop-{position}",
                )
            )
        pairs.append(
            Pair(
                cell="2wiki-composed-union",
                gold="entailment",
                premise=union,
                hypothesis=item.composed_claim,
                group=f"{item.item_id}::composed-union",
                note="union diagnostic",
            )
        )
        # each atomic claim against every gold chunk (not just its own): the
        # question is "is SOME single chunk enough", which is what A2 promises
        for claim_position, claim in enumerate(item.atomic_claims):
            for passage_position, passage in enumerate(item.gold_passages):
                pairs.append(
                    Pair(
                        cell="2wiki-atomic",
                        gold="entailment",
                        premise=passage,
                        hypothesis=claim,
                        group=f"{item.item_id}::atomic{claim_position}",
                        note=f"gold-hop-{passage_position}",
                    )
                )
            pairs.append(
                Pair(
                    cell="2wiki-atomic-union",
                    gold="entailment",
                    premise=union,
                    hypothesis=claim,
                    group=f"{item.item_id}::atomic{claim_position}-union",
                    note="union diagnostic",
                )
            )
            for distractor_position, passage in enumerate(item.distractors):
                pairs.append(
                    Pair(
                        cell="2wiki-neutral",
                        gold="neutral",
                        premise=passage,
                        hypothesis=claim,
                        group=f"{item.item_id}::atomic{claim_position}-neg{distractor_position}",
                        note="same-question distractor paragraph",
                    )
                )
    return pairs


NUMBER_TOKEN_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def build_counterfactual_pairs(
    positives: list[Pair],
    rng: random.Random,
) -> list[tuple[Pair, str, str]]:
    """Entity-swap slice: take a verified-supported ASQA pair and swap one
    entity in the CLAIM for a same-type entity harvested from a different
    question. Premise is untouched, so the correct verdict flips to unsupported.

    RGB is held out, so this is synthetic -- adequate here because the question
    is detector sensitivity, not realism.
    """
    extractor = RuleBasedEntityExtractor()

    def swappable(entity: Any, claim: str) -> bool:
        """Keep the swap realistic. The rule-based extractor emits `name` for any
        capitalized span, so a sentence-initial common noun would otherwise be
        'swapped' into nonsense -- and nonsense is trivially easy to catch, which
        would flatter the detector."""
        surface = entity.text.strip(" .,;:'\"")
        if entity.entity_type in {"date", "number"}:
            return len(surface) > 1
        return (
            len(surface.split()) >= 2
            and len(surface) >= 6
            and not claim.startswith(entity.text)
        )

    pool: dict[str, list[str]] = {"name": [], "number": [], "date": []}
    per_pair: list[tuple[Pair, list[Any]]] = []
    for pair in positives:
        premise_values = {other.normalized for other in extractor.extract(pair.premise)}
        entities = [
            entity
            for entity in extractor.extract(pair.hypothesis)
            # only swap entities the premise actually states, so the swap is a
            # genuine contradiction of the evidence rather than a new fact
            if entity.normalized in premise_values and swappable(entity, pair.hypothesis)
        ]
        per_pair.append((pair, entities))
        for entity in entities:
            bucket = pool.get(entity.entity_type)
            if bucket is not None:
                bucket.append(entity.text.strip(" .,;:'\""))

    swapped: list[tuple[Pair, str, str]] = []
    for pair, entities in per_pair:
        usable = [entity for entity in entities if pool.get(entity.entity_type)]
        if not usable:
            continue
        victim = rng.choice(usable)
        # same type, and for names a similar shape, so the swap reads like a
        # counterfactual rather than a category error
        victim_words = len(victim.text.split())
        alternatives = [
            text
            for text in pool[victim.entity_type]
            if text.lower() != victim.text.lower()
            and text.lower() not in pair.premise.lower()
            and (victim.entity_type != "name" or abs(len(text.split()) - victim_words) <= 1)
        ]
        if not alternatives:
            continue
        replacement = rng.choice(alternatives)
        corrupted = pair.hypothesis.replace(victim.text, replacement, 1)
        if corrupted == pair.hypothesis:
            continue
        swapped.append(
            (
                Pair(
                    cell="counterfactual",
                    gold="neutral",
                    premise=pair.premise,
                    hypothesis=corrupted,
                    group=pair.group + "::swap",
                    note=f"{victim.entity_type}: {victim.text} -> {replacement}",
                ),
                victim.entity_type,
                f"{victim.text} -> {replacement}",
            )
        )
    return swapped


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    label: Verdict
    """The arm's own argmax verdict, mapped through production
    `normalize_nli_label` where the checkpoint has a label vocabulary."""
    p_entail: float
    """Probability mass the arm puts on 'this passage supports this claim' --
    what the arm C threshold sweep runs over."""


class DebertaArm:
    """Arms A/B -- general 3-way NLI, argmax. Loading goes through the production
    `DebertaNLIModel` so the arm cannot drift from what B1 actually runs."""

    kind = "nli3"

    def __init__(self, name: str, model_id: str, max_length: int = 512) -> None:
        self.name = name
        self.model_id = model_id
        self.max_length = max_length
        self._backend = DebertaNLIModel(model_id=model_id, device="cpu", max_length=max_length)
        self._entail_index: int | None = None

    def describe(self) -> str:
        tokenizer, model = self._backend._ensure_loaded()
        del tokenizer
        return f"{self.model_id} id2label={json.dumps(model.config.id2label, ensure_ascii=False)}"

    def score(self, premise: str, hypothesis: str) -> Score:
        import torch

        tokenizer, model = self._backend._ensure_loaded()
        if self._entail_index is None:
            self._entail_index = next(
                index
                for index, raw in model.config.id2label.items()
                if normalize_nli_label(str(raw)) == "entailment"
            )
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        predicted = int(logits.argmax().item())
        return Score(
            label=normalize_nli_label(str(model.config.id2label[predicted])),
            p_entail=float(probabilities[self._entail_index].item()),
        )


class MiniCheckArm:
    """Arm D (CPU) -- MiniCheck-Flan-T5-Large, a *grounding-specific* verifier
    (trained for 'does this document support this claim', not sentence-pair NLI).

    Scoring follows the checkpoint's own published inference code: the input is
    ``predict: {doc}</s>{claim}``, one decoder step, and the support probability
    is the softmax over the two decoder vocabulary ids the model was trained to
    emit (3 = unsupported, 209 = supported).

    Binary by construction: it has no contradiction class, which is itself a
    finding for the `contradicted` flag.
    """

    kind = "binary"
    UNSUPPORTED_ID = 3
    SUPPORTED_ID = 209

    def __init__(self, name: str = "minicheck", model_id: str = "lytang/MiniCheck-Flan-T5-Large") -> None:
        self.name = name
        self.model_id = model_id
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            import transformers

            cache_dir = os.getenv("MODEL_CACHE_DIR") or None
            token = os.getenv("HUGGINGFACE_API_KEY") or None
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=cache_dir, token=token
            )
            # this checkpoint ships only pytorch_model.bin; without the explicit
            # flag transformers goes looking for a safetensors conversion PR on
            # the hub and stalls there
            self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, cache_dir=cache_dir, token=token, use_safetensors=False
            )
            self._model.eval()
        return self._tokenizer, self._model

    def describe(self) -> str:
        return f"{self.model_id} (binary supported/unsupported, decoder ids 209/3)"

    def score(self, premise: str, hypothesis: str) -> Score:
        import torch

        tokenizer, model = self._ensure_loaded()
        text = "predict: " + tokenizer.eos_token.join([premise, hypothesis])
        inputs = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")
        decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)
        with torch.no_grad():
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits.squeeze(1)
        pair = logits[:, torch.tensor([self.UNSUPPORTED_ID, self.SUPPORTED_ID])]
        probability = float(torch.softmax(pair, dim=-1)[0, 1].item())
        return Score(label="entailment" if probability > 0.5 else "neutral", p_entail=probability)


class TrueT5Arm:
    """Arm D (GPU) -- google/t5_xxl_true_nli_mixture, the TRUE judge ALCE's own
    citation evaluation uses. 11B params / 42.5GB fp32: it does not fit the CPU
    box, so this arm only runs on the HPC GPU node (see scripts/run_verifier_triage.slurm).

    Published input format is ``premise: {p} hypothesis: {h}``; the model emits
    '1' (entailed) or '0' (not entailed) as a single decoder token.
    """

    kind = "binary"

    def __init__(self, name: str = "true", model_id: str = "google/t5_xxl_true_nli_mixture") -> None:
        self.name = name
        self.model_id = model_id
        self._tokenizer: Any = None
        self._model: Any = None
        self._ids: tuple[int, int] | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            import transformers

            cache_dir = os.getenv("MODEL_CACHE_DIR") or None
            token = os.getenv("HUGGINGFACE_API_KEY") or None
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=cache_dir, token=token
            )
            self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                token=token,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()
            self._ids = (
                int(self._tokenizer("0", add_special_tokens=False).input_ids[0]),
                int(self._tokenizer("1", add_special_tokens=False).input_ids[0]),
            )
        return self._tokenizer, self._model

    def describe(self) -> str:
        return f"{self.model_id} (TRUE mixture, binary 1/0)"

    def score(self, premise: str, hypothesis: str) -> Score:
        import torch

        tokenizer, model = self._ensure_loaded()
        assert self._ids is not None
        text = f"premise: {premise} hypothesis: {hypothesis}"
        inputs = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits.squeeze(1)
        pair = logits[0, torch.tensor(self._ids, device=logits.device)].float().cpu()
        probability = float(torch.softmax(pair, dim=-1)[1].item())
        return Score(label="entailment" if probability > 0.5 else "neutral", p_entail=probability)


GRANITE_JUDGE_PROMPT = (
    "You are checking whether a passage supports a claim.\n"
    "Answer with exactly one word: SUPPORTS, CONTRADICTS, or NEITHER.\n"
    "SUPPORTS  - the passage states or directly implies the claim.\n"
    "CONTRADICTS - the passage states something incompatible with the claim.\n"
    "NEITHER - the passage neither supports nor contradicts the claim.\n"
    "Judge only from the passage; do not use outside knowledge.\n\n"
    "Passage:\n{premise}\n\nClaim:\n{hypothesis}\n\nAnswer:"
)
"""Written once, used unchanged for every pair and never tuned per arm (guide:
'Do not tune it across arms to make it win')."""


class GraniteJudgeArm:
    """Arm E -- LLM-as-a-judge, HPC GPU only.

    The model id is pinned explicitly (mirrors how ``DebertaArm`` takes its own):
    two Granite arms in one run must NOT both resolve to whatever ``GRANITE_MODEL_ID``
    happens to be, or the grid would show two identically-configured arms under
    different labels (guide task 1).
    """

    kind = "nli3"

    def __init__(self, name: str, model_id: str) -> None:
        from evidence_rag.generator.granite import GraniteGenerationConfig, GraniteLLMClient

        self.name = name
        # the judge answers in one word; the production default of 256 new tokens
        # would multiply the arm's cost for nothing
        self._llm = GraniteLLMClient(
            model_id=model_id, config=GraniteGenerationConfig(max_new_tokens=8)
        )

    def describe(self) -> str:
        return f"{getattr(self._llm, 'model_id', 'granite')} (LLM-as-a-judge, fixed prompt)"

    def score(self, premise: str, hypothesis: str) -> Score:
        raw = self._llm.generate(
            GRANITE_JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
        ).strip().upper()
        if "SUPPORT" in raw:
            return Score(label="entailment", p_entail=1.0)
        if "CONTRADICT" in raw:
            return Score(label="contradiction", p_entail=0.0)
        return Score(label="neutral", p_entail=0.0)


DUMP_DESCRIPTIONS = {
    "base": 'cross-encoder/nli-deberta-v3-base id2label={"0": "contradiction", "1": "entailment", "2": "neutral"}',
    "large": 'cross-encoder/nli-deberta-v3-large id2label={"0": "contradiction", "1": "entailment", "2": "neutral"}',
    "minicheck": "lytang/MiniCheck-Flan-T5-Large (binary supported/unsupported, decoder ids 209/3)",
    "true": "google/t5_xxl_true_nli_mixture (TRUE mixture, binary 1/0)",
    "granite3b": "ibm-granite/granite-4.1-3b (LLM-as-a-judge, fixed prompt)",
    "granite8b": "ibm-granite/granite-4.1-8b (LLM-as-a-judge, fixed prompt)",
}
"""Descriptions for --from-dump, where no model is loaded to ask."""


def build_arm(name: str) -> Any:
    if name == "base":
        return DebertaArm("base", "cross-encoder/nli-deberta-v3-base")
    if name == "large":
        return DebertaArm("large", "cross-encoder/nli-deberta-v3-large")
    if name == "minicheck":
        return MiniCheckArm()
    if name == "true":
        return TrueT5Arm()
    # The bare `granite` name is retired: it resolved to GRANITE_MODEL_ID (ambient
    # state), so the two Granite arms below each pin their own id explicitly.
    if name == "granite3b":
        return GraniteJudgeArm("granite3b", "ibm-granite/granite-4.1-3b")
    if name == "granite8b":
        return GraniteJudgeArm("granite8b", "ibm-granite/granite-4.1-8b")
    raise SystemExit(
        f"unknown arm {name!r} (base|large|minicheck|true|granite3b|granite8b)"
    )


# --------------------------------------------------------------------------
# scoring + metrics
# --------------------------------------------------------------------------


@dataclass
class ArmRun:
    name: str
    description: str
    scores: list[Score] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def per_pair_ms(self) -> float:
        return 1000.0 * self.seconds / max(len(self.scores), 1)


def run_arm(arm: Any, pairs: list[Pair]) -> ArmRun:
    print(f"[arm {arm.name}] loading", flush=True)
    description = arm.describe()
    print(f"[arm {arm.name}] {description}", flush=True)
    started = time.perf_counter()
    scores: list[Score] = []
    for index, pair in enumerate(pairs, start=1):
        scores.append(arm.score(pair.premise, pair.hypothesis))
        if index % 100 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[arm {arm.name}] {index}/{len(pairs)} "
                f"({1000 * elapsed / index:.0f} ms/pair)",
                flush=True,
            )
    return ArmRun(
        name=arm.name,
        description=description,
        scores=scores,
        seconds=time.perf_counter() - started,
    )


def _free_gpu() -> None:
    """Return the just-dropped arm's model memory to the CUDA allocator so the
    next arm starts from a clean card (measurement-only helper, no prod impact)."""
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def group_any_entailed(
    pairs: list[Pair],
    scores: list[Score],
    cell: str,
    threshold: float | None = None,
) -> dict[str, bool]:
    """Per claim-group: did ANY of its gold chunks come out entailed?"""
    verdicts: dict[str, bool] = {}
    for pair, score in zip(pairs, scores, strict=True):
        if pair.cell != cell:
            continue
        hit = score.p_entail >= threshold if threshold is not None else score.label == "entailment"
        verdicts[pair.group] = verdicts.get(pair.group, False) or hit
    return verdicts


def cell_counts(pairs: list[Pair], scores: list[Score], cell: str) -> Counter:
    matrix: Counter = Counter()
    for pair, score in zip(pairs, scores, strict=True):
        if pair.cell == cell:
            matrix[(pair.gold, score.label)] += 1
    return matrix


def rate(matrix: Counter, gold: str, predicted: str) -> float:
    total = sum(count for (g, _p), count in matrix.items() if g == gold)
    if not total:
        return float("nan")
    return matrix[(gold, predicted)] / total


def sweep(pairs: list[Pair], scores: list[Score]) -> list[tuple[float, float, float, float]]:
    """Arm C: threshold on P(entail) instead of argmax. Free -- the forward pass
    already produced the distribution. Returns (threshold, ASQA recall, ASQA
    hard-neutral FP, derived precision-proxy)."""
    rows: list[tuple[float, float, float, float]] = []
    positives = [
        score for pair, score in zip(pairs, scores, strict=True)
        if pair.cell == "asqa" and pair.gold == "entailment"
    ]
    negatives = [
        score for pair, score in zip(pairs, scores, strict=True)
        if pair.cell == "asqa" and pair.gold == "neutral"
    ]
    if not positives or not negatives:
        return rows
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        recall = sum(1 for s in positives if s.p_entail >= threshold) / len(positives)
        false_positive = sum(1 for s in negatives if s.p_entail >= threshold) / len(negatives)
        precision = recall / (recall + false_positive) if (recall + false_positive) else float("nan")
        rows.append((threshold, recall, false_positive, precision))
    return rows


def citation_precision(recall: float, false_positive: float, selected: int, truly_supporting: int) -> float:
    """Expected correct/(correct+spurious) citations for one claim, given a
    selected set of `selected` chunks of which `truly_supporting` really support
    the claim. Stated assumption, not a measured quantity."""
    correct = recall * truly_supporting
    spurious = false_positive * (selected - truly_supporting)
    if correct + spurious == 0:
        return float("nan")
    return correct / (correct + spurious)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def format_number(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def write_report(
    path: Path,
    pairs: list[Pair],
    runs: list[ArmRun],
    items: list[MultiHopItem],
    counterfactual: list[tuple[Pair, str, str]],
    entity_rows: list[tuple[str, bool]],
    assumption: tuple[int, int],
    near_verbatim: float,
) -> None:
    selected, truly_supporting = assumption
    lines: list[str] = []
    add = lines.append
    add("# Verifier triage results (Generator Part B, Half 1.6)")
    add("")
    add(
        "Measurement only -- `nli.py`, `attribution.py`, `entity_check.py` and every "
        "production config are unchanged; no threshold is committed. Premise = "
        "evidence, hypothesis = claim, the direction B1 is built on."
    )
    add("")
    add(
        "**Hold-out respected:** HotpotQA, RGB and MuSiQue-Full are never loaded. "
        "Calibration data only -- ALCE/ASQA, 2WikiMultihopQA, and a synthetic "
        "entity-swap slice."
    )
    add("")
    counts = Counter((pair.cell, pair.gold) for pair in pairs)
    add("## Shared pair set")
    add("")
    add("Every arm judges exactly these pairs, in this order.")
    add("")
    add("| cell | gold | pairs |")
    add("|---|---|---|")
    for (cell, gold), count in sorted(counts.items()):
        add(f"| {cell} | {gold} | {count} |")
    add(f"| counterfactual (reported separately) | not-supported | {len(counterfactual)} |")
    add("")
    add(
        f"Near-verbatim guard: an ASQA claim is dropped if it shares 8 or more "
        f"contiguous content tokens with its own gold passage. Median longest shared "
        f"run in the surviving set: **{near_verbatim:.0f} tokens** against a median "
        f"claim length of ~14 content tokens -- reported because a claim copied out "
        f"of its premise makes entailment trivially detectable and inflates recall "
        f"into meaninglessness."
    )
    add("")
    add("## Arms")
    add("")
    for run in runs:
        add(f"- **{run.name}** -- {run.description}")
    add("")

    add("## Main grid")
    add("")
    add(
        "| arm | ASQA entail recall | ASQA hard-neutral -> entail | ASQA hard-neutral -> contra "
        "| 2Wiki atomic recall (any single chunk) | 2Wiki composed recall (single chunk) "
        "| 2Wiki composed recall (union) | residual multi-hop | union diag. (residual) "
        "| 2Wiki distractor -> entail | ms/pair |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        asqa = cell_counts(pairs, run.scores, "asqa")
        recall = rate(asqa, "entailment", "entailment")
        neutral_entail = rate(asqa, "neutral", "entailment")
        neutral_contra = rate(asqa, "neutral", "contradiction")

        atomic = group_any_entailed(pairs, run.scores, "2wiki-atomic")
        atomic_recall = (sum(atomic.values()) / len(atomic)) if atomic else float("nan")
        composed = group_any_entailed(pairs, run.scores, "2wiki-composed")
        composed_recall = (sum(composed.values()) / len(composed)) if composed else float("nan")
        composed_union = group_any_entailed(pairs, run.scores, "2wiki-composed-union")
        composed_union_recall = (
            sum(composed_union.values()) / len(composed_union) if composed_union else float("nan")
        )
        residual = 1.0 - atomic_recall if atomic else float("nan")

        union = group_any_entailed(pairs, run.scores, "2wiki-atomic-union")
        residual_groups = [
            group for group, hit in atomic.items() if not hit
        ]
        recovered = [
            union.get(f"{group}-union", False) for group in residual_groups
        ]
        union_recall = (sum(recovered) / len(recovered)) if recovered else float("nan")

        distractor = cell_counts(pairs, run.scores, "2wiki-neutral")
        distractor_entail = rate(distractor, "neutral", "entailment")

        add(
            f"| {run.name} | {format_number(recall)} | {format_number(neutral_entail)} "
            f"| {format_number(neutral_contra)} | {format_number(atomic_recall)} "
            f"| {format_number(composed_recall)} | {format_number(composed_union_recall)} "
            f"| {format_number(residual)} "
            f"| {format_number(union_recall)} | {format_number(distractor_entail)} "
            f"| {run.per_pair_ms:.1f} |"
        )
    add("")
    add(
        "`residual multi-hop` = atomic claims (from the gold decomposition) that no "
        "single gold chunk entails. `union diag.` = of those, the share the same "
        "verifier recovers when both gold chunks are concatenated -- diagnostic only, "
        "it separates 'decomposition did not reduce this claim' from 'the verifier "
        "cannot see support that is present'."
    )
    add("")

    add("## Arm C -- threshold sweep on P(entail) (ASQA cell)")
    add("")
    for run in runs:
        rows = sweep(pairs, run.scores)
        if not rows:
            continue
        add(f"### {run.name}")
        add("")
        add("| threshold | entail recall | hard-neutral FP | precision proxy |")
        add("|---|---|---|---|")
        for threshold, recall, false_positive, precision in rows:
            add(
                f"| {threshold:.2f} | {recall:.3f} | {false_positive:.3f} "
                f"| {format_number(precision)} |"
            )
        add("")

    add("## Derived citation precision estimate")
    add("")
    add(
        f"**Stated assumption:** the Selector hands the Generator {selected} chunks per "
        f"query and, for any one atomic claim, {truly_supporting} of them genuinely "
        f"support it. Expected correct-to-total citation ratio = "
        f"`recall*{truly_supporting} / (recall*{truly_supporting} + FP*{selected - truly_supporting})` "
        f"using each arm's ASQA numbers at its own operating point."
    )
    add("")
    add("| arm | operating point | recall | hard-neutral FP | expected citation precision |")
    add("|---|---|---|---|---|")
    for run in runs:
        asqa = cell_counts(pairs, run.scores, "asqa")
        recall = rate(asqa, "entailment", "entailment")
        false_positive = rate(asqa, "neutral", "entailment")
        add(
            f"| {run.name} | argmax / 0.5 | {format_number(recall)} | {format_number(false_positive)} "
            f"| {format_number(citation_precision(recall, false_positive, selected, truly_supporting))} |"
        )
    add("")

    add("## Counterfactual slice (entity swaps, reported separately)")
    add("")
    add(
        "Built by swapping one entity in an ASQA claim that its gold passage actually "
        "states, for a same-type entity from a different question. Correct verdict is "
        "always 'not supported'."
    )
    add("")
    add("| arm | caught by verifier alone | caught by verifier OR entity check |")
    add("|---|---|---|")
    entity_caught = [caught for _type, caught in entity_rows]
    for run in runs:
        subset = [
            (pair, score)
            for pair, score in zip(pairs, run.scores, strict=True)
            if pair.cell == "counterfactual"
        ]
        if not subset:
            continue
        alone = sum(1 for _pair, score in subset if score.label != "entailment") / len(subset)
        combined = sum(
            1
            for (_pair, score), caught in zip(subset, entity_caught, strict=True)
            if score.label != "entailment" or caught
        ) / len(subset)
        add(f"| {run.name} | {alone:.3f} | {combined:.3f} |")
    add("")
    by_type = Counter(entity_type for entity_type, _caught in entity_rows)
    caught_by_type = Counter(entity_type for entity_type, caught in entity_rows if caught)
    add("Entity-check catch rate on its own, by swapped entity type:")
    add("")
    add("| entity type | swaps | caught by entity_check |")
    add("|---|---|---|")
    for entity_type, total in sorted(by_type.items()):
        add(f"| {entity_type} | {total} | {caught_by_type[entity_type] / total:.3f} |")
    add("")
    add("Examples of the swaps made:")
    add("")
    for _pair, entity_type, description in counterfactual[:8]:
        add(f"- `{entity_type}` -- {description}")
    add("")

    add("## Granite judge prompt (arm E), verbatim")
    add("")
    add("```text")
    add(GRANITE_JUDGE_PROMPT)
    add("```")
    add("")
    add(
        f"Multi-hop cell built from {len(items)} 2Wiki compositional questions; the "
        "gold `evidences` triples are used as a decomposition oracle (a perfect A2) "
        "so the verifier is measured without the claim splitter's own quality as a "
        "confound."
    )
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {path}", flush=True)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150, help="ASQA pairs per gold label")
    parser.add_argument("--multihop", type=int, default=60, help="2Wiki questions")
    parser.add_argument(
        "--arms",
        default="base,large,minicheck",
        help="comma-separated: base,large,minicheck,true,granite3b,granite8b",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-verbatim-run",
        type=int,
        default=8,
        help="drop an ASQA claim sharing this many contiguous content tokens with its premise",
    )
    parser.add_argument("--selected-chunks", type=int, default=5)
    parser.add_argument("--truly-supporting", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--dump", type=Path, default=None, help="also dump raw scores as JSONL")
    parser.add_argument(
        "--from-dump",
        type=Path,
        default=None,
        help="rebuild the report from an existing --dump instead of re-scoring "
        "(the pair set is deterministic in --seed/--n/--multihop, so this is exact)",
    )
    parser.add_argument(
        "--ms-per-pair",
        default="",
        help="with --from-dump: comma-separated arm=ms values carried over from the scoring run",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("[data] loading ALCE/ASQA", flush=True)
    asqa_pairs = build_asqa_pairs(ensure_asqa(), args.n, rng, args.max_verbatim_run)
    positives = [pair for pair in asqa_pairs if pair.gold == "entailment"]
    print(f"[data] ASQA: {len(positives)} entailment / {len(asqa_pairs) - len(positives)} hard neutral")

    print("[data] loading 2WikiMultihopQA", flush=True)
    items = build_multihop_items(ensure_2wiki(), args.multihop, rng)
    print(f"[data] 2Wiki: {len(items)} compositional questions")

    counterfactual = build_counterfactual_pairs(positives, rng)
    print(f"[data] counterfactual: {len(counterfactual)} entity swaps")

    pairs = asqa_pairs + multihop_pairs(items) + [pair for pair, _t, _d in counterfactual]
    print(f"[data] {len(pairs)} pairs total per arm", flush=True)

    # entity-check verdicts on the counterfactual slice (production code, called)
    checker = EntityConsistencyChecker()
    entity_rows = [
        (entity_type, not checker.check(pair.hypothesis, pair.premise).consistent)
        for pair, entity_type, _description in counterfactual
    ]

    runs_of_tokens = sorted(longest_common_run(pair.hypothesis, pair.premise) for pair in positives)
    near_verbatim = float(runs_of_tokens[len(runs_of_tokens) // 2]) if runs_of_tokens else float("nan")

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    if args.from_dump:
        timings = dict(
            (part.split("=")[0], float(part.split("=")[1]))
            for part in args.ms_per_pair.split(",")
            if "=" in part
        )
        rows = [json.loads(line) for line in args.from_dump.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(pairs):
            raise SystemExit(
                f"dump has {len(rows)} rows but the pair set rebuilt to {len(pairs)} -- "
                "--seed/--n/--multihop must match the scoring run"
            )
        runs = [
            ArmRun(
                name=name,
                description=DUMP_DESCRIPTIONS.get(name, name),
                scores=[
                    Score(label=row["scores"][name]["label"], p_entail=row["scores"][name]["p_entail"])
                    for row in rows
                ],
                seconds=timings.get(name, 0.0) * len(rows) / 1000.0,
            )
            for name in names
        ]
    else:
        # Load one arm at a time and free it before the next loads, so two large
        # GPU models (e.g. TRUE-XXL ~21GB bf16 and granite-4.1-8b ~16GB) never
        # co-reside on one card (guide task 2). arm drops out of scope each pass;
        # gc + empty_cache returns its blocks to the allocator.
        runs = []
        for name in names:
            arm = build_arm(name)
            runs.append(run_arm(arm, pairs))
            arm = None
            _free_gpu()

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as handle:
            for index, pair in enumerate(pairs):
                handle.write(
                    json.dumps(
                        {
                            "cell": pair.cell,
                            "gold": pair.gold,
                            "group": pair.group,
                            "note": pair.note,
                            "hypothesis": pair.hypothesis,
                            "scores": {
                                run.name: {
                                    "label": run.scores[index].label,
                                    "p_entail": run.scores[index].p_entail,
                                }
                                for run in runs
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    write_report(
        args.output,
        pairs,
        runs,
        items,
        counterfactual,
        entity_rows,
        (args.selected_chunks, args.truly_supporting),
        near_verbatim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
