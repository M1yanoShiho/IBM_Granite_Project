"""Loader for the RGB benchmark.

Data: RGB/data/ directory (already cloned at IBM_Granite_Project/RGB/)
Files:
  en.json        -> noise_robustness
  en_fact.json   -> counterfactual_robustness
  en_int.json    -> information_integration
  en_refine.json -> negative_rejection

Each line is a JSON object with keys:
  id, query, answer, positive (list of str), negative (list of str)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RGBSample:
    sample_id: str
    dimension: str
    question: str
    answer: List
    positive: List[str]
    negative: List[str]


@dataclass
class RGBDataset:
    dimension: str
    samples: List[RGBSample] = field(default_factory=list)


_DIMENSION_FILES = {
    "noise_robustness": "en.json",
    "counterfactual_robustness": "en_fact.json",
    "information_integration": "en_int.json",
    "negative_rejection": "en_refine.json",
}


def _load_jsonl(filepath: Path) -> List[Dict]:
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_rgb(data_dir: str, dimension: Optional[str] = None) -> List[RGBDataset]:
    data_path = Path(data_dir)
    dims = {dimension: _DIMENSION_FILES[dimension]} if dimension else _DIMENSION_FILES

    datasets = []
    for dim_name, filename in dims.items():
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"RGB file not found: {filepath}")

        records = _load_jsonl(filepath)
        samples = [
            RGBSample(
                sample_id=f"{dim_name}_{rec['id']}",
                dimension=dim_name,
                question=rec["query"],
                answer=rec.get("answer", []),
                positive=rec.get("positive", []),
                negative=rec.get("negative", []),
            )
            for rec in records
        ]
        datasets.append(RGBDataset(dimension=dim_name, samples=samples))

    return datasets
