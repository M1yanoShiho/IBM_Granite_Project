"""Paired Selector metrics and the pre-registered development-threshold decision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from evidence_rag.evaluation.paired_metric import PairedComparison, compare_paired


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Selector metric must be numeric")
    return float(value)


def _metric(value: object) -> float:
    return float(value) if isinstance(value, bool) else _number(value)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [_number(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else sum(values) / len(values)


def _arm(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    conditional_harm = [
        float(bool(row["harmful_selected"])) for row in rows if row.get("harmful_pool_hit") is True
    ]
    return {
        "harmful_in_context": (
            None if not conditional_harm else sum(conditional_harm) / len(conditional_harm)
        ),
        "harmful_in_context_n": len(conditional_harm),
        "required_evidence_recall": _mean(rows, "required_evidence_recall"),
        "evidence_precision": _mean(rows, "evidence_precision"),
        "selected_evidence_count": _mean(rows, "selected_evidence_count"),
    }


def _paired(
    beam_rows: Sequence[Mapping[str, object]],
    top_rows: Sequence[Mapping[str, object]],
    key: str,
    *,
    seed: int,
    iterations: int,
    harmful_only: bool = False,
) -> PairedComparison:
    top_by_query = {str(row["query_id"]): row for row in top_rows}
    on: dict[str, float | None] = {}
    off: dict[str, float | None] = {}
    for row in beam_rows:
        query_id = str(row["query_id"])
        top = top_by_query.get(query_id)
        if top is None:
            raise ValueError(f"TopK output is missing query {query_id}")
        if harmful_only and not (
            row.get("harmful_pool_hit") is True and top.get("harmful_pool_hit") is True
        ):
            on[query_id] = None
            off[query_id] = None
            continue
        beam_value = row.get(key)
        top_value = top.get(key)
        on[query_id] = None if beam_value is None else _metric(beam_value)
        off[query_id] = None if top_value is None else _metric(top_value)
    return compare_paired(on, off, seed=seed, iterations=iterations)


def _comparison(value: PairedComparison) -> dict[str, object]:
    return {
        "mean_beam": value.mean_on,
        "mean_top_k": value.mean_off,
        "delta_beam_minus_top_k": value.delta,
        "p_value": value.p_value,
        "ci_low": value.ci_low,
        "ci_high": value.ci_high,
        "n_paired": value.n_paired,
    }


def compare_selector_arms(
    top_rows: Sequence[Mapping[str, object]],
    beam_rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    iterations: int,
) -> dict[str, object]:
    metrics = {
        "required_recall": _paired(
            beam_rows, top_rows, "required_evidence_recall", seed=seed, iterations=iterations
        ),
        "evidence_precision": _paired(
            beam_rows, top_rows, "evidence_precision", seed=seed, iterations=iterations
        ),
        "selected_count": _paired(
            beam_rows, top_rows, "selected_evidence_count", seed=seed, iterations=iterations
        ),
    }
    if any(row.get("harmful_selected") is not None for row in beam_rows):
        metrics["harmful_in_context"] = _paired(
            beam_rows,
            top_rows,
            "harmful_selected",
            seed=seed,
            iterations=iterations,
            harmful_only=True,
        )
    return {
        "schema_version": "1.0",
        "top_k": _arm(top_rows),
        "beam_selector": _arm(beam_rows),
        "paired": {name: _comparison(value) for name, value in metrics.items()},
    }


def choose_threshold(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen credible-harm, then minimax-recall policy."""

    eligible: list[tuple[Mapping[str, object], float, float, float]] = []
    for row in rows:
        harm_delta = _number(row.get("harm_delta"))
        harm_ci_high = _number(row.get("harm_ci_high"))
        niah_loss = _number(row.get("niah_recall_loss"))
        twowiki_loss = _number(row.get("twowiki_recall_loss"))
        if (
            harm_delta <= -0.03
            and harm_ci_high < 0.0
            and niah_loss <= 0.05
            and twowiki_loss <= 0.05
        ):
            eligible.append((row, harm_delta, niah_loss, twowiki_loss))
    if not eligible:
        return {"status": "FAIL", "selected": None, "eligible_count": 0}
    clear = [item for item in eligible if item[2] <= 0.03 and item[3] <= 0.03]
    candidates = clear or eligible
    selected = min(
        candidates,
        key=lambda item: (
            max(item[2], item[3]),
            item[1],
            -_number(item[0].get("reject_threshold")),
            -_number(item[0].get("required_threshold")),
        ),
    )[0]
    return {
        "status": "CLEAR_PASS" if clear else "TRADEOFF",
        "selected": dict(selected),
        "eligible_count": len(eligible),
        "clear_count": len(clear),
    }
