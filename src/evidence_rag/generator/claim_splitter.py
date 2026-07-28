import json

from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.json_parsing import parse_json_object
from evidence_rag.generator.models import Claim, ClaimSpan

SPLIT_PROMPT = (
    "Split the answer into atomic, self-contained factual claims. Resolve pronouns "
    "without adding information. Return JSON only in this shape: "
    '{{"claims":[{{"source_text":"exact contiguous text from the answer",'
    '"text":"self-contained claim"}}]}}.\n\nAnswer:\n{answer}'
)

FAITHFULNESS_PROMPT = (
    "Check whether each rewritten claim has exactly the same meaning as its source text. "
    "Return JSON only in this shape: "
    '{{"results":[{{"claim_id":"claim-1","faithful":true}}]}}.\n\nItems:\n{items}'
)


class ClaimSplitter:
    """Split a draft answer into atomic claims and independently check rewrites."""

    def __init__(self, llm: TextGenerator | None = None) -> None:
        self.llm = llm or GraniteLLMClient()

    def split(self, answer_text: str) -> tuple[Claim, ...]:
        if not answer_text.strip():
            return ()
        split_data = self._load_json(
            self.llm.generate(SPLIT_PROMPT.format(answer=answer_text))
        )
        raw_claims = split_data.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError("splitter output must contain a claims array")
        pending: list[tuple[str, str, str, ClaimSpan]] = []
        cursor = 0
        for index, item in enumerate(raw_claims, start=1):
            claim_id = f"claim-{index}"
            if not isinstance(item, dict):
                raise ValueError("each claim must contain nonempty source_text and text")
            source_text = item.get("source_text")
            claim_text = item.get("text")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError("each claim must contain nonempty source_text and text")
            if not isinstance(claim_text, str) or not claim_text.strip():
                raise ValueError("each claim must contain nonempty source_text and text")
            start = answer_text.find(source_text, cursor)
            if start < 0:
                raise ValueError(f"claim {claim_id!r} source_text is not in answer_text")
            span = ClaimSpan(start=start, end=start + len(source_text))
            pending.append((claim_id, source_text, claim_text, span))
            cursor = span.end

        if not pending:
            return ()

        items = json.dumps(
            [
                {"claim_id": claim_id, "source_text": source, "claim_text": text}
                for claim_id, source, text, _span in pending
            ],
            ensure_ascii=False,
        )
        check_data = self._load_json(
            self.llm.generate(FAITHFULNESS_PROMPT.format(items=items))
        )
        raw_results = check_data.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("faithfulness output must contain a results array")
        faithful_by_id = {
            item["claim_id"]: item["faithful"] for item in raw_results
        }
        expected_ids = {claim_id for claim_id, _source, _text, _span in pending}
        if set(faithful_by_id) != expected_ids or len(faithful_by_id) != len(
            raw_results
        ):
            raise ValueError("faithfulness results must cover exactly the split claims")
        return tuple(
            Claim(
                claim_id=claim_id,
                text=text,
                span=span,
                faithful_to_answer=faithful_by_id[claim_id],
            )
            for claim_id, _source, text, span in pending
        )

    @staticmethod
    def _load_json(raw: str) -> dict[str, object]:
        # tolerate the trailing prose a real Granite model appends after the JSON
        return parse_json_object(raw)
