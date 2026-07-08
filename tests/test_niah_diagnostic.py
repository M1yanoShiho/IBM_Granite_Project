"""Tests for eval.niah_diagnostic — the ranking-vs-recall decomposition (finding 12 / Table 4a).

Pure arithmetic on two run_niah aggregate CSVs (same task/retriever at k=10 and k=100):
buried = found@100 - found@10 (RANKING failure), unreachable = 1 - found@100 (RECALL failure).
The finding is *derived* from two numbers, not a stored column.
"""
from __future__ import annotations

import csv

import pytest

from eval.niah_diagnostic import decompose, main, read_needle_found


def _write_agg(path, retriever, k, value):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["retriever", "n_docs", f"needle_found@{k}", "mrr"])
        w.writerow([retriever, 245000, value, 0.3])


def test_decompose_splits_ranking_and_recall_failure():
    d = decompose(0.43, 0.87)
    assert d["buried"] == pytest.approx(0.44)
    assert d["unreachable"] == pytest.approx(0.13)
    assert d["ratio"] == pytest.approx(0.44 / 0.13)


def test_decompose_ratio_is_inf_when_no_recall_failure():
    d = decompose(0.5, 1.0)
    assert d["unreachable"] == pytest.approx(0.0)
    assert d["ratio"] == float("inf")


def test_read_needle_found_extracts_k10_and_value(tmp_path):
    p = tmp_path / "k10.csv"
    _write_agg(p, "granite_dense", 10, 0.493)
    k, v = read_needle_found(p, "granite_dense")
    assert k == 10
    assert v == pytest.approx(0.493)


def test_read_needle_found_handles_k100_column(tmp_path):
    p = tmp_path / "k100.csv"
    _write_agg(p, "granite_dense", 100, 0.87)
    k, v = read_needle_found(p, "granite_dense")
    assert k == 100
    assert v == pytest.approx(0.87)


def test_read_needle_found_raises_for_missing_retriever(tmp_path):
    p = tmp_path / "k10.csv"
    _write_agg(p, "granite_dense", 10, 0.49)
    with pytest.raises(ValueError):
        read_needle_found(p, "q2d_granite")


def test_main_writes_decomposition_csv(tmp_path):
    k10 = tmp_path / "k10.csv"
    _write_agg(k10, "granite_dense", 10, 0.43)
    k100 = tmp_path / "k100.csv"
    _write_agg(k100, "granite_dense", 100, 0.87)
    out = tmp_path / "decomp.csv"
    main([
        "--k10-csv", str(k10), "--k100-csv", str(k100),
        "--retriever", "granite_dense", "--out", str(out),
    ])
    assert out.exists()
    row = next(csv.DictReader(open(out, encoding="utf-8")))
    assert float(row["found@10"]) == pytest.approx(0.43)
    assert float(row["found@100"]) == pytest.approx(0.87)
    assert float(row["buried"]) == pytest.approx(0.44)
    assert float(row["unreachable"]) == pytest.approx(0.13)
