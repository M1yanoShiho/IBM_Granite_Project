"""A3 pre-flight: the §11.9 checks that must pass before R013 is submitted.

M0 §11.9 lists seven hard prerequisites and says "任一不过即停". A checklist in a
document is not a mechanism -- this project has written that sentence four times now
(§9.11, §10.8, §11.9a) -- so the checks live here and this script is what gets run.

Six of the seven are automated below. The seventh (passage-hash leakage axis, §11.7
item 1) needs the sealed-600 corpus and is a separate job; it is reported as SKIPPED
rather than silently dropped.

    python scripts/a3_preflight.py

Exit code is 0 only if every automated check passes. Anything else means R013 does
not get submitted yet.

WHY CHECK 1 CAN CANCEL THE AMENDMENT. A3 exists because microsoft/deberta-v3-base
ships only pytorch_model.bin, which transformers refuses to torch.load under
torch < 2.6 (CVE-2025-32434). If this cluster is on torch >= 2.6 that premise is
false, §11.1 says the amendment "应被撤回", and §3.8 should run on its original base.
The check therefore reports that outcome as a distinct verdict, not as a pass.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

# Run by hand on a login node, unlike the slurm scripts that export PYTHONPATH=src for
# themselves. Put the repo's src on the path so `python scripts/a3_preflight.py` works
# with no ceremony -- a pre-flight check that needs its own pre-flight gets skipped.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

BASE = "cross-encoder/nli-deberta-v3-base"
ORIGINAL_BASE = "microsoft/deberta-v3-base"

# The three-class order this checkpoint is expected to expose, from its published
# config on 2026-08-06. §11.9 requires it be MEASURED and registered by name: albert is
# (SUPPORTS, REFUTES, UNKNOWN) and DeBERTa-v3-large-mnli is (SUPPORTS, UNKNOWN,
# REFUTES), so this is a third distinct order and a positional guess would not raise --
# it would just relabel every edge.
EXPECTED_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}

RELATION_BY_NLI_NAME = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "UNKNOWN",
}


class Result:
    def __init__(self) -> None:
        self.failed: list[str] = []
        self.skipped: list[str] = []

    def report(self, name: str, ok: bool, detail: str) -> None:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}\n       {detail}", flush=True)
        if not ok:
            self.failed.append(name)

    def skip(self, name: str, detail: str) -> None:
        print(f"[SKIP] {name}\n       {detail}", flush=True)
        self.skipped.append(name)


def check_torch(result: Result) -> None:
    torch = importlib.import_module("torch")
    version = str(torch.__version__)
    major, minor = (int(part) for part in version.split(".")[:2])
    under_pin = (major, minor) < (2, 6)
    if under_pin:
        result.report(
            "1. cluster torch < 2.6",
            True,
            f"torch {version}. A3's premise holds: a .bin-only checkpoint cannot be "
            f"loaded here, so {ORIGINAL_BASE} is genuinely unusable.",
        )
        return
    result.report(
        "1. cluster torch < 2.6",
        False,
        f"torch {version} is >= 2.6. **A3's core premise is FALSE on this cluster** -- "
        f"{ORIGINAL_BASE} can be torch.load-ed after all. Per M0 §11.1 the amendment "
        "should be WITHDRAWN and §3.8 run on its original base. Do not 'fix' this by "
        "proceeding; take it back to the protocol.",
    )


def check_original_base_has_no_safetensors(result: Result) -> None:
    """§11.9 item 2 -- verify ON THE CLUSTER, not from a local API query."""
    try:
        hub = importlib.import_module("huggingface_hub")
        files = hub.list_repo_files(ORIGINAL_BASE)
    except Exception as error:  # noqa: BLE001 - any failure here is inconclusive
        result.skip(
            "2. original base ships no safetensors",
            f"could not list repo files ({type(error).__name__}: {error}). The cluster "
            "runs with HF_HUB_OFFLINE=1, so this needs a login node with network. "
            "INCONCLUSIVE is not a pass -- §11.9 requires this confirmed here.",
        )
        return
    safetensors = [name for name in files if name.endswith(".safetensors")]
    result.report(
        "2. original base ships no safetensors",
        not safetensors,
        f"{ORIGINAL_BASE} weight files: "
        f"{sorted(name for name in files if name.endswith(('.safetensors', '.bin')))}",
    )


def check_loads(result: Result) -> Any:
    transformers = importlib.import_module("transformers")
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            BASE, use_safetensors=True
        )
    except Exception as error:  # noqa: BLE001 - the point is to report, not to raise
        result.report(
            "3. selected base loads with use_safetensors=True",
            False,
            f"{type(error).__name__}: {error}",
        )
        return None
    result.report(
        "3. selected base loads with use_safetensors=True",
        True,
        f"{BASE} loaded, {sum(p.numel() for p in model.parameters()):,} parameters.",
    )
    return model


def check_id2label(result: Result, model: Any) -> None:
    """§11.9 item 4 -- the one that fails silently if skipped."""
    measured = {int(key): str(value) for key, value in model.config.id2label.items()}
    matches = measured == EXPECTED_ID2LABEL
    order = tuple(RELATION_BY_NLI_NAME.get(measured[i], "?") for i in sorted(measured))
    detail = (
        f"measured id2label = {measured}\n"
        f"       -> LABEL_ORDER entry to register: \"{BASE}\": {order}\n"
        "       Map BY NAME. A positional guess does not raise -- it relabels every "
        "edge and every number downstream stays plausible."
    )
    if not matches:
        detail += (
            f"\n       MISMATCH vs the order recorded on 2026-08-06 "
            f"({EXPECTED_ID2LABEL}). The checkpoint changed, or the record was wrong. "
            "Resolve before registering; do not adopt either silently."
        )
    result.report("4. id2label measured and mapped by name", matches, detail)


def check_fingerprint(result: Result, model: Any) -> None:
    # Both helpers live in relations.predictor; cli.gate0b only imports them. Call them
    # where they are defined so this does not break if that import is ever dropped.
    cli = importlib.import_module("evidence_rag.cli.gate0b")
    predictor = importlib.import_module("evidence_rag.relations.predictor")
    fingerprint = predictor.weight_fingerprint(cli._weight_buffers(model))
    version = predictor.fingerprinted_version(BASE, fingerprint)
    result.report(
        "5. weight fingerprint recorded",
        bool(fingerprint),
        f"model_version = {version}\n"
        "       Record this. Two checkpoints can share an id; this is what lands on "
        "every edge and every dump row.",
    )


def check_cross_encoder_scaffold(result: Result) -> None:
    try:
        module = importlib.import_module("sentence_transformers.cross_encoder")
        module.CrossEncoder(BASE, num_labels=3)
    except Exception as error:  # noqa: BLE001
        result.report(
            "6. sentence-transformers CrossEncoder accepts the checkpoint",
            False,
            f"{type(error).__name__}: {error}\n"
            "       §3.8 pins the CrossEncoder three-class scaffold, so this is a "
            "recipe prerequisite, not a convenience.",
        )
        return
    result.report(
        "6. sentence-transformers CrossEncoder accepts the checkpoint",
        True,
        "CrossEncoder(num_labels=3) constructed.",
    )


def main() -> int:
    print(f"A3 pre-flight (M0 §11.9) -- base: {BASE}\n")
    result = Result()

    check_torch(result)
    check_original_base_has_no_safetensors(result)
    model = check_loads(result)
    if model is not None:
        check_id2label(result, model)
        check_fingerprint(result, model)
    else:
        result.skip("4. id2label", "base did not load")
        result.skip("5. weight fingerprint", "base did not load")
    check_cross_encoder_scaffold(result)

    result.skip(
        "7. passage-hash leakage axis measured",
        "Needs the sealed-600 corpus and is a separate job. §11.7 item 1 struck "
        "§3.8's 'empty by construction' because the selected base has seen SNLI+MNLI, "
        "so this axis now needs evidence. NOT automated here -- track it, do not "
        "assume it.",
    )

    print()
    if result.failed:
        print(f"BLOCKED -- {len(result.failed)} check(s) failed: {result.failed}")
        print("R013 must not be submitted. See M0 §11.9.")
        return 1
    print(f"All automated checks passed. Still open: {result.skipped}")
    print("R013 may be submitted once the skipped items are settled (M0 §11.9).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
