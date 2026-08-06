"""Convert a `.bin`-only HF checkpoint to safetensors, in place in the HF cache.

WHY THIS EXISTS. `google/t5_xxl_true_nli_mixture` -- the G module's production
verifier (`generator/nli.py`, DEFAULT_NLI_BACKEND = "true") -- publishes only
`pytorch_model-*.bin`. transformers >= ~4.51 refuses to `torch.load` a `.bin` when
torch < 2.6 (CVE-2025-32434), and this cluster is transformers 4.57.6 on torch
2.5.1. So the verifier cannot be loaded at all, and all three verification arms of
the G6 run failed 365/400 with OSError before this was even reachable.

THIS IS NOT A SECURITY BYPASS, and the distinction matters. The CVE is about
`torch.load` executing arbitrary code during unpickling. The fix transformers
applied is a blanket refusal; the mitigation the CVE itself asks for is
`weights_only=True`, which this script uses and which torch 2.5.1 supports. We
deserialise once, under the safe mode, and write a format that needs no unpickling
ever again.

WHY NOT THE ALTERNATIVES (M0 §11 records the same trade-off for a training base):
upgrading torch >= 2.6 would fix the load but collapse A3's premise and put R012's
in-place reproducibility (§9.10a) at risk; downgrading transformers below 4.51
risks Granite 4.1 support; switching the NLI backend breaks comparability with G5,
whose numbers were all measured with TRUE.

THE COST, STATED: the resulting weights are a LOCAL ARTEFACT, not an official
release. That is exactly the provenance objection M0 §11.1 limitation 2 raises
against doing this for a *training base*. The asymmetry that makes it acceptable
here has to be argued, not assumed: a training base's provenance flows into the
model being shipped, whereas the verifier is a measurement instrument whose
weights are pinned by hash either way. This script therefore writes a provenance
record with the sha256 of every input and output file, so the conversion is
auditable rather than remembered.

    python scripts/convert_bin_to_safetensors.py \\
        --snapshot /user/work/$USER/hf_cache/hub/models--google--t5_xxl_true_nli_mixture/snapshots/<rev> \\
        --provenance results/true_nli_safetensors_conversion.json

Shard-by-shard, so peak memory is one shard rather than the whole 11B model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BIN_INDEX = "pytorch_model.bin.index.json"
SAFE_INDEX = "model.safetensors.index.json"

# safetensors reports its OWN dtype names ("F32"), not torch's ("torch.float32"), so the
# two have to be mapped rather than string-matched. Spelled out instead of derived,
# because the check this feeds exists to catch a silent dtype demotion -- a comparison
# that quietly never matches, or quietly always matches, defeats it either way.
SAFETENSORS_DTYPE = {
    "torch.float64": "F64",
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int16": "I16",
    "torch.int8": "I8",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_shared(state: dict[str, Any]) -> tuple[dict[str, Any], list[list[str]]]:
    """Clone tensors that share storage, because safetensors refuses aliases.

    T5 ties `shared.weight` to the encoder and decoder embeddings and often to
    `lm_head.weight`, so a T5 state dict always trips this. Cloning keeps every key
    present, which matters: transformers re-ties them on load when the config says
    `tie_word_embeddings`, but a MISSING key would instead surface as a randomly
    initialised embedding and a model that runs and scores plausibly wrong.

    Returns the de-aliased dict and the groups that were found, so the provenance
    record shows what was cloned rather than leaving it to be rediscovered.
    """
    import torch

    by_storage: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
            key = (tensor.untyped_storage().data_ptr(), tensor.dtype)
            by_storage[key].append(name)

    groups = [sorted(names) for names in by_storage.values() if len(names) > 1]
    if not groups:
        return state, []

    resolved = dict(state)
    for names in groups:
        for name in names[1:]:
            resolved[name] = resolved[name].clone()
    return resolved, groups


def convert_shard(source: Path, target: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    # weights_only=True is the CVE's own mitigation, not a workaround of it: it
    # forbids the arbitrary-object unpickling that the vulnerability turns on.
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{source.name}: expected a non-empty state dict, got {type(state)!r}")

    resolved, shared_groups = _resolve_shared(state)
    contiguous = {name: tensor.contiguous() for name, tensor in resolved.items()}
    # "format": "pt" is not decorative -- transformers checks it and refuses the file
    # without it, which would leave the .bin as the only loadable route again.
    save_file(contiguous, str(target), metadata={"format": "pt"})

    written = _verify(target, contiguous)
    del state, resolved, contiguous
    return {
        "source": source.name,
        "target": target.name,
        "tensors": written,
        "shared_tensor_groups": shared_groups,
    }


def _verify(target: Path, expected: dict[str, Any]) -> int:
    """Re-open the written file and check every key, shape and dtype round-tripped.

    Header-only: `safe_open` reads metadata without materialising the weights, so
    this stays inside one shard's memory budget. A silent dtype demotion here would
    produce a model that loads, runs, and scores subtly differently -- the failure
    shape this project keeps paying for.
    """
    from safetensors import safe_open

    with safe_open(str(target), framework="pt") as handle:  # type: ignore[no-untyped-call]
        keys = set(handle.keys())
        if keys != set(expected):
            missing = sorted(set(expected) - keys)
            extra = sorted(keys - set(expected))
            raise ValueError(f"{target.name}: key mismatch. missing={missing} extra={extra}")
        for name in keys:
            slice_ = handle.get_slice(name)
            shape = tuple(slice_.get_shape())
            if shape != tuple(expected[name].shape):
                raise ValueError(
                    f"{target.name}: {name} shape {shape} != {tuple(expected[name].shape)}"
                )
            want = SAFETENSORS_DTYPE.get(str(expected[name].dtype))
            if want is None:
                raise ValueError(
                    f"{target.name}: {name} has dtype {expected[name].dtype}, which is not "
                    "in SAFETENSORS_DTYPE. Add it rather than skipping the check."
                )
            if slice_.get_dtype() != want:
                raise ValueError(
                    f"{target.name}: {name} dtype {slice_.get_dtype()} != {want} "
                    f"(from {expected[name].dtype})"
                )
    return len(expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="HF cache snapshot dir")
    parser.add_argument("--provenance", required=True, type=Path, help="conversion record json")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing safetensors; refuses without it so a half-finished "
        "run is never silently completed by a second one",
    )
    args = parser.parse_args(argv)

    snapshot: Path = args.snapshot
    if not (snapshot / BIN_INDEX).is_file():
        raise SystemExit(f"no {BIN_INDEX} in {snapshot} -- is this a sharded checkpoint snapshot?")

    index = json.loads((snapshot / BIN_INDEX).read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    out_index = snapshot / SAFE_INDEX
    if out_index.exists() and not args.force:
        raise SystemExit(f"{out_index} already exists -- pass --force to redo the conversion")

    print(f"[convert] {len(shards)} shard(s), {len(weight_map)} weights", flush=True)

    records: list[dict[str, Any]] = []
    new_map: dict[str, str] = {}
    for n, shard in enumerate(shards, start=1):
        source = snapshot / shard
        target = snapshot / shard.replace("pytorch_model", "model").replace(".bin", ".safetensors")
        print(f"[convert] {n}/{len(shards)} {shard} -> {target.name}", flush=True)
        source_sha = sha256_of(source)
        record = convert_shard(source, target)
        record["source_sha256"] = source_sha
        record["target_sha256"] = sha256_of(target)
        records.append(record)
        for name, where in weight_map.items():
            if where == shard:
                new_map[name] = target.name

    if set(new_map) != set(weight_map):
        raise ValueError("converted weight map does not cover the original index")

    out_index.write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": new_map}, indent=2),
        encoding="utf-8",
    )

    provenance = {
        "converted_at": datetime.now(UTC).isoformat(),
        "snapshot": str(snapshot),
        "reason": (
            "google/t5_xxl_true_nli_mixture ships only .bin; transformers 4.57.6 refuses "
            "torch.load under torch < 2.6 (CVE-2025-32434). Converted with "
            "weights_only=True, the CVE's own mitigation. Weights are a LOCAL ARTEFACT, "
            "not an official release -- see this file's hashes."
        ),
        "index_sha256": sha256_of(out_index),
        "shards": records,
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\n[convert] wrote {out_index.name} and {len(records)} safetensors shard(s)")
    print(f"[convert] provenance -> {args.provenance}")
    print("[convert] the .bin files are left in place; transformers prefers safetensors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
