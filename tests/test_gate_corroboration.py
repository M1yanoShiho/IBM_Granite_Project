"""Tests for the offline gated dynamic-alpha corroboration analysis."""
from eval.gate_corroboration import margin_signal, max_votes_signal


def test_max_votes_signal_takes_the_per_query_max():
    corr = {"q1": {"a": 0.0, "b": 2.0, "c": 1.0}, "q2": {"a": 0.0, "b": 0.0}}
    assert max_votes_signal(corr) == {"q1": 2.0, "q2": 0.0}


def test_max_votes_signal_empty_query_scores_zero():
    assert max_votes_signal({"q1": {}}) == {"q1": 0.0}


def test_margin_signal_is_top1_minus_top2_after_minmax():
    # exactly-representable scores: minmax -> 1.0/0.5/0.0 -> margin 0.5 (no float dust)
    rel = {"q1": {"a": 1.0, "b": 0.5, "c": 0.0}}
    assert margin_signal(rel) == {"q1": 0.5}


def test_margin_signal_degenerate_cases_are_zero():
    # all-equal scores minmax to all-1.0 (margin 0); a single doc has no top2.
    rel = {"q1": {"a": 0.7, "b": 0.7}, "q2": {"a": 0.9}}
    assert margin_signal(rel) == {"q1": 0.0, "q2": 0.0}


import pytest

from eval.gate_corroboration import gate_mask, gated_fuse
from src.retrieval.fusion import fuse_one


def test_gate_mask_global_always_fires():
    votes, margins = {"q1": 0.0, "q2": 3.0}, {"q1": 0.5, "q2": 0.01}
    assert gate_mask("global", None, votes, margins) == {"q1": True, "q2": True}


def test_gate_mask_votes_fires_at_or_above_tau():
    votes, margins = {"q1": 1.0, "q2": 2.0, "q3": 0.0}, {}
    assert gate_mask("votes", 2.0, votes, margins) == {"q1": False, "q2": True, "q3": False}


def test_gate_mask_margin_fires_below_threshold():
    votes, margins = {}, {"q1": 0.04, "q2": 0.5}
    assert gate_mask("margin", 0.05, votes, margins) == {"q1": True, "q2": False}


def test_gate_mask_rejects_unknown_family():
    with pytest.raises(ValueError):
        gate_mask("nonsense", 1.0, {}, {})


def test_gated_fuse_gate_off_keeps_pure_relevance_order():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 5.0}}
    fused = gated_fuse(rel, cor, alpha=0.5, gate={"q1": False})
    assert fused["q1"] == rel["q1"]  # untouched: first-stage scores pass through


def test_gated_fuse_gate_on_matches_fuse_one():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 5.0}}
    fused = gated_fuse(rel, cor, alpha=0.5, gate={"q1": True})
    assert fused["q1"] == fuse_one(rel["q1"], cor["q1"], 0.5)


def test_gated_fuse_mixed_gate():
    rel = {"q1": {"a": 0.9, "b": 0.1}, "q2": {"a": 0.9, "b": 0.1}}
    cor = {"q1": {"a": 0.0, "b": 3.0}, "q2": {"a": 0.0, "b": 3.0}}
    fused = gated_fuse(rel, cor, alpha=0.0, gate={"q1": True, "q2": False})
    assert fused["q1"] == fuse_one(rel["q1"], cor["q1"], 0.0)
    assert fused["q2"] == rel["q2"]


from eval.gate_corroboration import split_queries


def test_split_queries_is_deterministic_disjoint_and_covers_all():
    qids = [f"q{i}" for i in range(300)]
    dev1, test1 = split_queries(qids, seed=0)
    dev2, test2 = split_queries(list(reversed(qids)), seed=0)  # input order irrelevant
    assert (dev1, test1) == (dev2, test2)
    assert len(dev1) == len(test1) == 150
    assert set(dev1).isdisjoint(test1)
    assert set(dev1) | set(test1) == set(qids)


def test_split_queries_seed_changes_the_split():
    qids = [f"q{i}" for i in range(300)]
    assert split_queries(qids, seed=0) != split_queries(qids, seed=1)


def test_split_queries_dev_fraction():
    dev, test = split_queries([f"q{i}" for i in range(10)], seed=0, dev_fraction=0.3)
    assert len(dev) == 3 and len(test) == 7


from eval.gate_corroboration import evaluate_config

# One fixable query: relevance ranks the counterfactual first; the needle holds a
# 2-vote consensus. And one zero-consensus query: all votes 0.
_REL = {"q1": {"cf1": 0.99, "n1": 0.5}, "q2": {"cf2": 0.99, "n2": 0.5}}
_COR = {"q1": {"cf1": 0.0, "n1": 2.0}, "q2": {"cf2": 0.0, "n2": 0.0}}
_NEEDLES = {"q1": "n1", "q2": "n2"}


def test_evaluate_config_alpha_one_is_pure_first_stage():
    hits = evaluate_config(_REL, _COR, _NEEDLES, ["q1", "q2"], "global", None, 1.0, k=1)
    assert hits == {"q1": 0.0, "q2": 0.0}  # counterfactual on top -> both missed


def test_evaluate_config_votes_gate_fixes_only_the_consensus_query():
    # tau=2: q1 (max votes 2) blends -> needle wins at alpha=0.2; q2 (no consensus)
    # stays pure relevance -> still missed.
    hits = evaluate_config(_REL, _COR, _NEEDLES, ["q1", "q2"], "votes", 2.0, 0.2, k=1)
    assert hits == {"q1": 1.0, "q2": 0.0}


def test_evaluate_config_restricts_to_the_given_qids():
    hits = evaluate_config(_REL, _COR, _NEEDLES, ["q1"], "global", None, 1.0, k=1)
    assert set(hits) == {"q1"}


from eval.gate_corroboration import (
    MARGIN_THRESHOLDS,
    VOTE_TAUS,
    CurveRow,
    sweep_gated,
)


def _rows_by(rows, family, param=None):
    return {r.alpha: r for r in rows if r.family == family and r.param == param}


def test_sweep_covers_all_families_and_params():
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [0.0, 0.5, 1.0], k=1)
    families = {(r.family, r.param) for r in rows}
    assert ("global", None) in families
    assert all(("votes", t) in families for t in VOTE_TAUS)
    assert all(("margin", m) in families for m in MARGIN_THRESHOLDS)
    assert len(rows) == (1 + len(VOTE_TAUS) + len(MARGIN_THRESHOLDS)) * 3


def test_sweep_votes_tau1_equals_global_blend():
    # Zero-consensus queries have all-equal votes -> constant after minmax -> the
    # global blend is order-preserving on them, so gating them off (tau=1) changes
    # nothing: identical needle-found at every alpha (the spec's structural fact).
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [0.0, 0.2, 0.5, 1.0], k=1)
    global_curve = {a: r.score for a, r in _rows_by(rows, "global").items()}
    tau1_curve = {a: r.score for a, r in _rows_by(rows, "votes", 1.0).items()}
    assert tau1_curve == global_curve


def test_sweep_alpha_one_scores_equal_pure_first_stage_everywhere():
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [1.0], k=1)
    assert {r.score for r in rows} == {0.0}  # every family at alpha=1 = q2d = both missed


def test_sweep_n_gated_counts_gated_queries():
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [0.5], k=1)
    assert _rows_by(rows, "global")[0.5].n_gated == 2
    assert _rows_by(rows, "votes", 2.0)[0.5].n_gated == 1  # only q1 has votes >= 2


from eval.gate_corroboration import select_on_dev


def test_select_prefers_higher_score_then_larger_alpha_then_stricter_gate():
    rows = [
        CurveRow("votes", 1.0, 0.2, 0.8, 10),
        CurveRow("votes", 1.0, 0.5, 0.8, 10),  # same score, larger alpha -> preferred
        CurveRow("votes", 2.0, 0.5, 0.8, 5),   # same score+alpha, stricter tau -> preferred
        CurveRow("votes", 3.0, 0.1, 0.7, 2),   # lower score -> ignored
        CurveRow("global", None, 0.6, 0.75, 20),
        CurveRow("margin", 0.05, 0.3, 0.8, 4),
        CurveRow("margin", 0.10, 0.3, 0.8, 8),  # same score+alpha, smaller m stricter -> 0.05 wins
    ]
    best, winner, best_gated = select_on_dev(rows)
    assert best["votes"] == CurveRow("votes", 2.0, 0.5, 0.8, 5)
    assert best["margin"] == CurveRow("margin", 0.05, 0.3, 0.8, 4)
    assert best["global"] == CurveRow("global", None, 0.6, 0.75, 20)
    # Across families: 0.8 beats global's 0.75; votes beats margin on the family order.
    assert winner == CurveRow("votes", 2.0, 0.5, 0.8, 5)
    assert best_gated == CurveRow("votes", 2.0, 0.5, 0.8, 5)


def test_select_across_family_tie_prefers_global_then_votes():
    rows = [
        CurveRow("global", None, 0.5, 0.8, 20),
        CurveRow("votes", 2.0, 0.5, 0.8, 5),
        CurveRow("margin", 0.05, 0.5, 0.8, 4),
    ]
    best, winner, best_gated = select_on_dev(rows)
    assert winner.family == "global"        # tie -> simpler wins
    assert best_gated.family == "votes"     # best GATED config still reported (spec 4)


from eval.gate_corroboration import flip_status, flip_table, margin_bucket, vote_bucket


def test_flip_status_four_way():
    assert flip_status(0.0, 1.0) == "fixed"
    assert flip_status(1.0, 0.0) == "broken"
    assert flip_status(1.0, 1.0) == "unchanged_hit"
    assert flip_status(0.0, 0.0) == "unchanged_miss"


def test_vote_bucket_edges():
    assert [vote_bucket(v) for v in (0.0, 1.0, 3.0, 4.0, 9.0)] == ["0", "1", "3", "4+", "4+"]


def test_margin_bucket_edges():
    assert [margin_bucket(m) for m in (0.0, 0.049, 0.05, 0.1, 0.19, 0.2, 0.9)] == [
        "<0.05", "<0.05", "0.05-0.1", "0.1-0.2", "0.1-0.2", ">=0.2", ">=0.2",
    ]


def test_flip_table_counts_one_fixed_one_broken_one_unchanged():
    # q1: blend fixes it (needle 2-vote consensus). q2: blend breaks it (needle on
    # top by relevance, counterfactual holds the consensus). q3: unchanged miss
    # (no consensus -> constant blend preserves order).
    rel = {
        "q1": {"cf1": 0.99, "n1": 0.5},
        "q2": {"n2": 0.99, "cf2": 0.5},
        "q3": {"cf3": 0.99, "n3": 0.5},
    }
    cor = {
        "q1": {"cf1": 0.0, "n1": 2.0},
        "q2": {"n2": 0.0, "cf2": 2.0},
        "q3": {"cf3": 0.0, "n3": 0.0},
    }
    needles = {"q1": "n1", "q2": "n2", "q3": "n3"}

    rows = flip_table(rel, cor, needles, alpha=0.2, k=1)

    votes_rows = {r["bucket"]: r for r in rows if r["signal"] == "max_votes"}
    assert votes_rows["2"]["fixed"] == 1 and votes_rows["2"]["broken"] == 1
    assert votes_rows["0"]["unchanged_miss"] == 1
    # every query lands in exactly one bucket per signal
    for signal in ("max_votes", "margin"):
        sig_rows = [r for r in rows if r["signal"] == signal]
        assert sum(
            r["fixed"] + r["broken"] + r["unchanged_hit"] + r["unchanged_miss"]
            for r in sig_rows
        ) == 3


import csv

from eval.gate_corroboration import _parse_args, write_dev_curves, write_flip_table


def test_write_dev_curves_csv(tmp_path):
    rows = [CurveRow("global", None, 0.6, 0.58, 150), CurveRow("votes", 2.0, 0.5, 0.6, 80)]
    path = tmp_path / "curves.csv"
    write_dev_curves(rows, path, k=10)
    with open(path, newline="") as f:
        got = list(csv.reader(f))
    assert got[0] == ["family", "param", "alpha", "needle_found@10", "n_gated"]
    assert got[1] == ["global", "", "0.6", "0.58", "150"]  # global param is empty (spec 6)
    assert got[2] == ["votes", "2.0", "0.5", "0.6", "80"]


def test_write_flip_table_csv(tmp_path):
    rows = [
        {"signal": "max_votes", "bucket": "0", "fixed": 0, "broken": 1,
         "unchanged_hit": 2, "unchanged_miss": 3},
    ]
    path = tmp_path / "flips.csv"
    write_flip_table(rows, path)
    with open(path, newline="") as f:
        got = list(csv.reader(f))
    assert got[0] == ["signal", "bucket", "fixed", "broken", "unchanged_hit", "unchanged_miss"]
    assert got[1] == ["max_votes", "0", "0", "1", "2", "3"]


def test_parse_args_defaults_and_required_from_runs():
    args = _parse_args(["--from-runs", "results/runs.json"])
    assert args.from_runs.name == "runs.json"
    assert args.k == 10 and args.seed == 0
    assert args.alpha_step == 0.1 and args.flip_alpha == 0.6
    assert args.out_dir.name == "results"


def test_parse_args_requires_from_runs():
    with pytest.raises(SystemExit):
        _parse_args([])


from eval.gate_corroboration import main
from eval.tune_corroboration import dump_runs


def _synthetic_dump(tmp_path):
    """4 identical fixable queries (any 2/2 split behaves the same): the lone
    counterfactual tops relevance; the needle holds a 2-vote consensus."""
    rel, cor, needles = {}, {}, {}
    for i in range(4):
        qid = f"q{i}"
        rel[qid] = {f"cf{i}": 0.99, f"n{i}": 0.5}
        cor[qid] = {f"cf{i}": 0.0, f"n{i}": 2.0}
        needles[qid] = f"n{i}"
    path = tmp_path / "runs.json"
    dump_runs(rel, cor, needles, path)
    return path


def test_main_end_to_end_writes_all_artifacts(tmp_path, capsys):
    runs = _synthetic_dump(tmp_path)
    out = tmp_path / "out"

    main(["--from-runs", str(runs), "--k", "1", "--out-dir", str(out)])

    # 1. all three artifacts exist
    per_query = out / "corroboration_gate_test_per_query.csv"
    assert (out / "corroboration_flip_table.csv").exists()
    assert (out / "corroboration_gate_dev_curves.csv").exists()
    assert per_query.exists()

    # 2. test arms: q2d misses everywhere (cf on top); blend + gated blend fix it.
    with open(per_query, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"qid", "q2d", "global_corroborate", "gated_corroborate"}
    assert len(rows) == 2  # the test half of 4 queries
    assert all(r["q2d"] == "0.0" for r in rows)
    assert all(r["global_corroborate"] == "1.0" for r in rows)
    assert all(r["gated_corroborate"] == "1.0" for r in rows)

    # 3. summary names the winner and prints both significance commands
    printed = capsys.readouterr().out
    assert "winner" in printed and "eval.significance" in printed
    assert "--reference q2d" in printed and "--reference global_corroborate" in printed
