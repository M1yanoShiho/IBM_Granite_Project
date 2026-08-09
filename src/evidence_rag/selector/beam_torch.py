"""Lazy PyTorch/Transformers backend for the three-class Beam Selector.

The project keeps Torch optional for CPU-only development.  Imports therefore happen only when a
real training or inference run constructs this backend.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.selector.beam_three_class import ClassProbabilities


def model_input_text(question: str, selected_passages: Sequence[str]) -> str:
    if not selected_passages:
        return f"Question: {question}"
    context = "\n".join(f"Selected evidence {index}: {text}" for index, text in enumerate(selected_passages, 1))
    return f"Question: {question}\n{context}"


class TorchBeamNetwork:
    """One shared DeBERTa encoder with separate first-hop and later-hop heads."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        cache_dir: str | None,
        seed: int,
    ) -> None:
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:  # pragma: no cover - server-only dependency
            raise RuntimeError("Beam Selector training requires torch and transformers") from error
        self.device = device
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(self.torch.backends, "cudnn"):
            self.torch.backends.cudnn.benchmark = False
            self.torch.backends.cudnn.deterministic = True
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir
        )
        self.encoder = transformers.AutoModel.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir
        ).to(device)
        hidden_size = int(self.encoder.config.hidden_size)
        first = self.torch.nn.Linear(hidden_size, 3)
        self.torch.nn.init.xavier_uniform_(first.weight)
        self.torch.nn.init.zeros_(first.bias)
        self.first_head = first.to(device)
        self.next_head = copy.deepcopy(first).to(device)
        self.dropout = self.torch.nn.Dropout(float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))).to(device)

    def parameters(self) -> Iterable[Any]:
        yield from self.encoder.parameters()
        yield from self.first_head.parameters()
        yield from self.next_head.parameters()

    def train(self) -> None:
        self.encoder.train()
        self.first_head.train()
        self.next_head.train()
        self.dropout.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.first_head.eval()
        self.next_head.eval()
        self.dropout.eval()

    def encode(
        self,
        *,
        questions: Sequence[str],
        selected_passages: Sequence[Sequence[str]],
        candidate_passages: Sequence[str],
        max_length: int,
    ) -> Mapping[str, Any]:
        if not (
            len(questions) == len(selected_passages) == len(candidate_passages)
        ):
            raise ValueError("Beam text batch lengths differ")
        first = [
            model_input_text(question, selected)
            for question, selected in zip(questions, selected_passages, strict=True)
        ]
        second = [f"Candidate evidence: {text}" for text in candidate_passages]
        raw = self.tokenizer(
            first,
            second,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if not isinstance(raw, Mapping):
            raise RuntimeError("Beam tokenizer returned a non-mapping batch")
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in raw.items()
        }

    def logits(self, encoded: Mapping[str, Any], hops: Sequence[int]) -> Any:
        output = self.encoder(**encoded)
        pooled = self.dropout(output.last_hidden_state[:, 0])
        first = self.first_head(pooled)
        later = self.next_head(pooled)
        hop_tensor = self.torch.tensor(hops, device=self.device, dtype=self.torch.long)
        return self.torch.where((hop_tensor == 0).unsqueeze(-1), first, later)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "first_head": self.first_head.state_dict(),
                "next_head": self.next_head.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = self.torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        self.first_head.load_state_dict(state["first_head"])
        self.next_head.load_state_dict(state["next_head"])


class TorchBeamTextScorer:
    def __init__(
        self,
        network: TorchBeamNetwork,
        *,
        max_length: int,
        batch_size: int,
    ) -> None:
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.network = network
        self.max_length = max_length
        self.batch_size = batch_size

    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        if not candidate_passages:
            return ()
        self.network.eval()
        output: list[ClassProbabilities] = []
        with self.network.torch.inference_mode():
            for start in range(0, len(candidate_passages), self.batch_size):
                passages = candidate_passages[start : start + self.batch_size]
                encoded = self.network.encode(
                    questions=[question] * len(passages),
                    selected_passages=[selected_passages] * len(passages),
                    candidate_passages=passages,
                    max_length=self.max_length,
                )
                logits = self.network.logits(encoded, [hop] * len(passages))
                rows = self.network.torch.softmax(logits, dim=-1).detach().cpu().tolist()
                output.extend(
                    ClassProbabilities(
                        irrelevant=float(row[0]),
                        required=float(row[1]),
                        harmful=float(row[2]),
                    )
                    for row in rows
                )
        return tuple(output)
