def evidence_recall(
    predicted_ids: tuple[str, ...],
    required_ids: tuple[str, ...],
) -> float:
    if not required_ids:
        raise ValueError("required_ids must not be empty")
    return len(set(predicted_ids) & set(required_ids)) / len(set(required_ids))


def evidence_precision(
    predicted_ids: tuple[str, ...],
    relevant_ids: tuple[str, ...],
) -> float:
    if not predicted_ids:
        return 0.0
    return len(set(predicted_ids) & set(relevant_ids)) / len(set(predicted_ids))


def citation_validity(
    cited_ids: tuple[str, ...],
    selected_ids: tuple[str, ...],
) -> float:
    if not cited_ids:
        return 0.0
    allowed = set(selected_ids)
    return sum(evidence_id in allowed for evidence_id in cited_ids) / len(cited_ids)
