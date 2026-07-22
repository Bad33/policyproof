from __future__ import annotations

import math

import pytest

from policyproof.retrieval_metrics import (
    direct_evidence_hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RELEVANCE_GRADES = {
    "direct-1": 2,
    "direct-2": 2,
    "supporting": 1,
}


def test_recall_at_k_counts_all_positive_relevance_grades() -> None:
    ranked_passage_ids = (
        "unjudged",
        "supporting",
        "direct-1",
        "direct-2",
    )

    assert recall_at_k(
        ranked_passage_ids,
        RELEVANCE_GRADES,
        k=1,
    ) == 0.0
    assert recall_at_k(
        ranked_passage_ids,
        RELEVANCE_GRADES,
        k=3,
    ) == pytest.approx(2 / 3)
    assert recall_at_k(
        ranked_passage_ids,
        RELEVANCE_GRADES,
        k=10,
    ) == 1.0


def test_reciprocal_rank_uses_first_positive_relevance_grade() -> None:
    ranked_passage_ids = (
        "unjudged",
        "supporting",
        "direct-1",
    )

    assert reciprocal_rank_at_k(
        ranked_passage_ids,
        RELEVANCE_GRADES,
        k=10,
    ) == 0.5

    assert reciprocal_rank_at_k(
        ("unjudged", "also-unjudged"),
        RELEVANCE_GRADES,
        k=10,
    ) == 0.0


def test_direct_evidence_hit_requires_grade_two() -> None:
    assert not direct_evidence_hit_at_k(
        ("supporting", "unjudged"),
        RELEVANCE_GRADES,
        k=10,
    )

    assert direct_evidence_hit_at_k(
        ("unjudged", "direct-2"),
        RELEVANCE_GRADES,
        k=10,
    )


def test_ndcg_at_k_uses_graded_gain_and_logarithmic_discount() -> None:
    ranked_passage_ids = (
        "unjudged",
        "supporting",
        "direct-1",
        "direct-2",
    )

    actual_dcg = (
        1 / math.log2(3)
        + 3 / math.log2(4)
    )
    ideal_dcg = (
        3 / math.log2(2)
        + 3 / math.log2(3)
        + 1 / math.log2(4)
    )

    assert ndcg_at_k(
        ranked_passage_ids,
        RELEVANCE_GRADES,
        k=3,
    ) == pytest.approx(actual_dcg / ideal_dcg)


@pytest.mark.parametrize(
    "metric",
    [
        recall_at_k,
        reciprocal_rank_at_k,
        direct_evidence_hit_at_k,
        ndcg_at_k,
    ],
)
def test_metrics_reject_queries_without_relevant_judgments(metric) -> None:
    with pytest.raises(
        ValueError,
        match="at least one relevant judgment",
    ):
        metric(
            ("passage-1",),
            {},
            k=10,
        )


@pytest.mark.parametrize(
    "metric",
    [
        recall_at_k,
        reciprocal_rank_at_k,
        direct_evidence_hit_at_k,
        ndcg_at_k,
    ],
)
def test_metrics_reject_nonpositive_cutoffs(metric) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        metric(
            ("direct-1",),
            RELEVANCE_GRADES,
            k=0,
        )
