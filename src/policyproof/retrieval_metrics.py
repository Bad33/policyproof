"""Deterministic passage-ranking metric primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class RetrievalMetricError(ValueError):
    """Raised when ranking metric inputs are invalid."""


def _validate_inputs(
    ranked_passage_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    *,
    k: int,
) -> None:
    """Validate shared metric inputs."""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise RetrievalMetricError("k must be a positive integer.")

    if not relevance_grades:
        raise RetrievalMetricError(
            "Metrics require at least one relevant judgment."
        )

    for passage_id, grade in relevance_grades.items():
        if not isinstance(passage_id, str) or not passage_id:
            raise RetrievalMetricError(
                "Relevant passage IDs must be non-empty strings."
            )

        if (
            not isinstance(grade, int)
            or isinstance(grade, bool)
            or grade < 1
        ):
            raise RetrievalMetricError(
                "Relevance grades must be positive integers."
            )

    seen_ranked_ids: set[str] = set()

    for passage_id in ranked_passage_ids:
        if not isinstance(passage_id, str) or not passage_id:
            raise RetrievalMetricError(
                "Ranked passage IDs must be non-empty strings."
            )

        if passage_id in seen_ranked_ids:
            raise RetrievalMetricError(
                f"Duplicate ranked passage ID: {passage_id}"
            )

        seen_ranked_ids.add(passage_id)


def recall_at_k(
    ranked_passage_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return the fraction of all relevant passages retrieved within k."""

    _validate_inputs(
        ranked_passage_ids,
        relevance_grades,
        k=k,
    )

    retrieved_relevant = sum(
        passage_id in relevance_grades
        for passage_id in ranked_passage_ids[:k]
    )

    return retrieved_relevant / len(relevance_grades)


def reciprocal_rank_at_k(
    ranked_passage_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant passage within k."""

    _validate_inputs(
        ranked_passage_ids,
        relevance_grades,
        k=k,
    )

    for rank, passage_id in enumerate(
        ranked_passage_ids[:k],
        start=1,
    ):
        if passage_id in relevance_grades:
            return 1.0 / rank

    return 0.0


def direct_evidence_hit_at_k(
    ranked_passage_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    *,
    k: int,
) -> bool:
    """Return whether a grade-2 direct-evidence passage appears within k."""

    _validate_inputs(
        ranked_passage_ids,
        relevance_grades,
        k=k,
    )

    return any(
        relevance_grades.get(passage_id) == 2
        for passage_id in ranked_passage_ids[:k]
    )


def _discounted_cumulative_gain(
    grades: Sequence[int],
) -> float:
    """Calculate DCG using exponential graded gain."""

    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(
    ranked_passage_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return normalized graded discounted cumulative gain within k."""

    _validate_inputs(
        ranked_passage_ids,
        relevance_grades,
        k=k,
    )

    actual_grades = tuple(
        relevance_grades.get(passage_id, 0)
        for passage_id in ranked_passage_ids[:k]
    )
    ideal_grades = tuple(
        sorted(
            relevance_grades.values(),
            reverse=True,
        )[:k]
    )

    ideal_dcg = _discounted_cumulative_gain(ideal_grades)

    if ideal_dcg == 0:
        raise RetrievalMetricError(
            "Metrics require at least one relevant judgment."
        )

    return (
        _discounted_cumulative_gain(actual_grades)
        / ideal_dcg
    )
