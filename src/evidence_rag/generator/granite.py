import importlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from evidence_rag.contracts.models import (
    UNVERIFIED_ANNOTATION,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    split_sentences,
)

DEFAULT_GRANITE_MODEL_ID = "ibm-granite/granite-4.1-3b"

CITATION_RAG_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "For every claim in your answer, cite the supporting evidence by number.\n\n"
    "Use this exact format:\n"
    "Answer: <answer>\n"
    "Evidence: [number_1], [number_2], ...\n\n"
    "If the evidence does not contain the answer, use:\n"
    "Answer: I don't know\n"
    "Evidence: []\n\n"
    "Evidence:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

WORD = re.compile(r"[A-Za-z0-9]+")


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class GraniteGenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    max_input_tokens: int | None = None


class GraniteLLMClient:
    """Lazy local Hugging Face Granite client used by the Generator module."""

    def __init__(
        self,
        model_id: str | None = None,
        config: GraniteGenerationConfig | None = None,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        self.model_id = model_id or os.getenv("GRANITE_MODEL_ID") or DEFAULT_GRANITE_MODEL_ID
        self.config = config or GraniteGenerationConfig()
        self.device = device or os.getenv("LLM_DEVICE", "auto")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("unsupported Granite dtype")
        self.dtype = dtype
        self._tokenizer: Any
        self._model: Any
        self._tokenizer, self._model = self._load_model()

    def _load_model(self) -> tuple[Any, Any]:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "GraniteLLMClient requires the optional 'transformers' package."
            ) from exc

        token = os.getenv("HUGGINGFACE_API_KEY") or None
        cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
        )
        torch_dtype: Any = self.dtype
        if self.dtype != "auto":
            torch_dtype = getattr(importlib.import_module("torch"), self.dtype)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
            dtype=torch_dtype,
            device_map="auto" if self.device == "auto" else None,
        )
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def _encode_prompt(self, prompt: str) -> Any:
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids

    def input_token_count(self, prompt: str) -> int:
        """Return the exact chat-templated input length used by ``generate``."""

        return int(self._encode_prompt(prompt).shape[-1])

    def generate(self, prompt: str) -> str:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError("GraniteLLMClient requires the optional 'torch' package.") from exc

        input_ids = self._encode_prompt(prompt)

        if (
            self.config.max_input_tokens is not None
            and input_ids.shape[-1] > self.config.max_input_tokens
        ):
            raise ValueError(
                "prompt exceeds frozen max_input_tokens; Experiment 04 forbids silent truncation"
            )

        model_device = getattr(self._model, "device", None)
        if model_device is not None and hasattr(input_ids, "to"):
            input_ids = input_ids.to(model_device)

        generation_args: dict[str, int | float | bool] = {
            "max_new_tokens": self.config.max_new_tokens,
        }
        if self.config.temperature > 0:
            generation_args.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                }
            )
        else:
            generation_args["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(input_ids=input_ids, **generation_args)
        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


class PeftGraniteLLMClient(GraniteLLMClient):
    """One Granite instance with named LoRA adapters and an explicit base view.

    F006 adapts only key-fact extraction. ``generate_with_adapter`` enables the
    requested LoRA for that call, while ordinary ``generate`` explicitly
    disables every adapter so draft generation and claim splitting remain on
    the frozen base model.
    """

    def __init__(
        self,
        *,
        model_id: str,
        adapters: dict[str, str],
        config: GraniteGenerationConfig | None = None,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        if not adapters:
            raise ValueError("PeftGraniteLLMClient requires at least one adapter")
        self.adapter_paths = dict(adapters)
        super().__init__(model_id=model_id, config=config, device=device, dtype=dtype)

    def _load_model(self) -> tuple[Any, Any]:
        try:
            peft = importlib.import_module("peft")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - optional server dependency
            raise RuntimeError("PeftGraniteLLMClient requires transformers and peft") from exc

        token = os.getenv("HUGGINGFACE_API_KEY") or None
        cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
        )
        torch_dtype: Any = self.dtype
        if self.dtype != "auto":
            torch_dtype = getattr(importlib.import_module("torch"), self.dtype)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
            dtype=torch_dtype,
            device_map="auto" if self.device == "auto" else None,
        )
        if self.device != "auto":
            base_model = base_model.to(self.device)
        first_name, first_path = next(iter(self.adapter_paths.items()))
        model = peft.PeftModel.from_pretrained(
            base_model,
            first_path,
            adapter_name=first_name,
            is_trainable=False,
        )
        for name, path in list(self.adapter_paths.items())[1:]:
            model.load_adapter(path, adapter_name=name, is_trainable=False)
        model.eval()
        return tokenizer, model

    def _generate_current_model(self, prompt: str) -> str:
        return super().generate(prompt)

    def generate(self, prompt: str) -> str:
        """Generate with the frozen base model by explicitly disabling adapters."""

        with self._model.disable_adapter():
            return self._generate_current_model(prompt)

    def generate_with_adapter(self, prompt: str, adapter_name: str) -> str:
        if adapter_name not in self.adapter_paths:
            raise ValueError(f"unknown LoRA adapter: {adapter_name}")
        self._model.set_adapter(adapter_name)
        return self._generate_current_model(prompt)


class NamedAdapterTextGenerator:
    """TextGenerator view that activates one named LoRA for every call."""

    def __init__(self, client: PeftGraniteLLMClient, adapter_name: str) -> None:
        if adapter_name not in client.adapter_paths:
            raise ValueError(f"unknown LoRA adapter: {adapter_name}")
        self.client = client
        self.adapter_name = adapter_name

    def generate(self, prompt: str) -> str:
        return self.client.generate_with_adapter(prompt, self.adapter_name)


def parse_citation_output(raw: str) -> tuple[str, tuple[int, ...]]:
    text = raw.strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:") :].strip()
    parts = re.split(r"\n\s*Evidence:\s*", text, maxsplit=1)
    answer = parts[0].strip()
    evidence_text = parts[1] if len(parts) > 1 else ""
    seen: set[int] = set()
    indices: list[int] = []
    for match in re.findall(r"\[(\d+)\]", evidence_text):
        index = int(match)
        if index not in seen:
            seen.add(index)
            indices.append(index)
    return answer, tuple(indices)


def _is_unknown_answer(answer: str) -> bool:
    normalized = answer.strip().lower().strip(".!?\"' ")
    return normalized in {
        "",
        "unknown",
        "i don't know",
        "i do not know",
        "don't know",
        "do not know",
        "not in the evidence",
        "not contained in the evidence",
        "not in the context",
        "not contained in the context",
    }


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in WORD.finditer(text)}


def _fallback_citation_ids(answer: str, selected: SelectedEvidenceSet) -> tuple[str, ...]:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return ()
    scored = sorted(
        (
            (len(answer_tokens & _tokens(item.text)), item.evidence_id)
            for item in selected.evidence
        ),
        key=lambda item: (-item[0], item[1]),
    )
    overlapping = tuple(evidence_id for score, evidence_id in scored if score > 0)
    if overlapping:
        return overlapping
    return (selected.evidence[0].evidence_id,) if selected.evidence else ()


class GraniteGenerator:
    """Generator implementation backed by a local Granite language model."""

    def __init__(
        self,
        llm: TextGenerator | None = None,
        prompt_template: str = CITATION_RAG_PROMPT,
    ) -> None:
        self.llm = llm or GraniteLLMClient()
        self.prompt_template = prompt_template

    @staticmethod
    def _format_context(selected: SelectedEvidenceSet) -> str:
        return "\n".join(
            f"[{index}] ({item.evidence_id}) {item.text}"
            for index, item in enumerate(selected.evidence, start=1)
        )

    @staticmethod
    def _citation_ids(indices: Iterable[int], selected: SelectedEvidenceSet) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        evidence = selected.evidence
        for index in indices:
            if 1 <= index <= len(evidence):
                evidence_id = evidence[index - 1].evidence_id
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    ids.append(evidence_id)
        return tuple(ids)

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        if not selected.evidence:
            return GenerationResult(query_id=query.query_id, answer="", cited_evidence_ids=())

        prompt = self.prompt_template.format(
            context=self._format_context(selected),
            question=query.text,
        )
        raw = self.llm.generate(prompt)
        answer, citation_indices = parse_citation_output(raw)
        if _is_unknown_answer(answer):
            return GenerationResult(query_id=query.query_id, answer="", cited_evidence_ids=())

        cited_ids = self._citation_ids(citation_indices, selected)
        if not cited_ids:
            cited_ids = _fallback_citation_ids(answer, selected)
        return GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=cited_ids,
        )


INLINE_CITATION = re.compile(r"\[(\d+)\]")


class InlineCitationGraniteGenerator(GraniteGenerator):
    """Direct base Granite under the same inline-citation prompt as grounded GR-C."""

    def __init__(
        self,
        llm: TextGenerator | None = None,
        prompt_template: str = CITATION_RAG_PROMPT,
        *,
        require_declared_citations: bool = False,
    ) -> None:
        super().__init__(llm=llm, prompt_template=prompt_template)
        self.require_declared_citations = require_declared_citations
        self.last_raw_output = ""
        self.last_declared_indices: tuple[int, ...] = ()
        self.last_invalid_indices: tuple[int, ...] = ()

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        self.last_raw_output = ""
        self.last_declared_indices = ()
        self.last_invalid_indices = ()
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        if not selected.evidence:
            return GenerationResult(query_id=query.query_id, answer="", cited_evidence_ids=())
        prompt = self.prompt_template.format(
            context=self._format_context(selected),
            question=query.text,
            focus=checklist.focus,
            required_facts="; ".join(checklist.required_facts) or "none",
            constraints="; ".join(checklist.constraints) or "none",
        )
        raw = self.llm.generate(prompt).strip()
        if raw.casefold().startswith("answer:"):
            raw = raw[len("answer:") :].strip()
        indices = tuple(dict.fromkeys(int(value) for value in INLINE_CITATION.findall(raw)))
        self.last_raw_output = raw
        self.last_declared_indices = indices
        self.last_invalid_indices = tuple(
            index for index in indices if not 1 <= index <= len(selected.evidence)
        )
        answer = " ".join(INLINE_CITATION.sub("", raw).split())
        if _is_unknown_answer(answer):
            return GenerationResult(query_id=query.query_id, answer="", cited_evidence_ids=())
        cited_ids = self._citation_ids(indices, selected)
        if not cited_ids and not self.require_declared_citations:
            cited_ids = _fallback_citation_ids(answer, selected)
        if not cited_ids and self.require_declared_citations:
            answer = " ".join(
                f"{sentence} {UNVERIFIED_ANNOTATION}"
                for sentence in split_sentences(answer)
            )
        return GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=cited_ids,
        )
