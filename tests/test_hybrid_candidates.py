from __future__ import annotations

from dataclasses import fields

import pytest

from policyproof.bm25 import BM25Hit
from policyproof.dense import DenseHit
from policyproof.hybrid_candidates import (
    HybridCandidate,
    HybridCandidateError,
    build_hybrid_candidate_union,
)


def bm25_hit(
    passage_id: str,
    *,
    score: float,
    accepted_order: int,
) -> BM25Hit:
    return BM25Hit(
        passage_id=passage_id,
        score=score,
        accepted_order=accepted_order,
    )


def dense_hit(
    passage_id: str,
    *,
    score: float,
    accepted_order: int,
) -> DenseHit:
    return DenseHit(
        passage_id=passage_id,
        score=score,
        accepted_order=accepted_order,
    )


def test_union_deduplicates_and_preserves_source_ranks() -> None:
    bm25_hits = (
        bm25_hit(
            "passage-b",
            score=4.0,
            accepted_order=1,
        ),
        bm25_hit(
            "passage-a",
            score=3.0,
            accepted_order=0,
        ),
        bm25_hit(
            "passage-d",
            score=2.0,
            accepted_order=3,
        ),
    )
    dense_hits = (
        dense_hit(
            "passage-a",
            score=0.90,
            accepted_order=0,
        ),
        dense_hit(
            "passage-c",
            score=0.80,
            accepted_order=2,
        ),
        dense_hit(
            "passage-b",
            score=0.70,
            accepted_order=1,
        ),
    )

    candidates = build_hybrid_candidate_union(
        bm25_hits,
        dense_hits,
        input_depth=3,
    )

    assert candidates == (
        HybridCandidate(
            passage_id="passage-a",
            accepted_order=0,
            bm25_rank=2,
            dense_rank=1,
        ),
        HybridCandidate(
            passage_id="passage-b",
            accepted_order=1,
            bm25_rank=1,
            dense_rank=3,
        ),
        HybridCandidate(
            passage_id="passage-c",
            accepted_order=2,
            bm25_rank=None,
            dense_rank=2,
        ),
        HybridCandidate(
            passage_id="passage-d",
            accepted_order=3,
            bm25_rank=3,
            dense_rank=None,
        ),
    )


def test_candidate_order_is_corpus_order_not_fused_ranking() -> None:
    bm25_hits = (
        bm25_hit(
            "passage-z",
            score=10.0,
            accepted_order=9,
        ),
        bm25_hit(
            "passage-a",
            score=1.0,
            accepted_order=0,
        ),
    )
    dense_hits = (
        dense_hit(
            "passage-y",
            score=0.99,
            accepted_order=8,
        ),
        dense_hit(
            "passage-b",
            score=0.50,
            accepted_order=1,
        ),
    )

    candidates = build_hybrid_candidate_union(
        bm25_hits,
        dense_hits,
        input_depth=2,
    )

    assert [candidate.passage_id for candidate in candidates] == [
        "passage-a",
        "passage-b",
        "passage-y",
        "passage-z",
    ]


def test_candidate_contract_does_not_expose_fused_score() -> None:
    field_names = {
        field.name
        for field in fields(HybridCandidate)
    }

    assert field_names == {
        "passage_id",
        "accepted_order",
        "bm25_rank",
        "dense_rank",
    }


@pytest.mark.parametrize(
    ("input_depth", "message"),
    [
        (0, "positive integer"),
        (-1, "positive integer"),
        (True, "positive integer"),
        (1.5, "positive integer"),
    ],
)
def test_union_rejects_invalid_input_depth(
    input_depth,
    message: str,
) -> None:
    with pytest.raises(
        HybridCandidateError,
        match=message,
    ):
        build_hybrid_candidate_union(
            (
                bm25_hit(
                    "passage-a",
                    score=1.0,
                    accepted_order=0,
                ),
            ),
            (
                dense_hit(
                    "passage-a",
                    score=0.5,
                    accepted_order=0,
                ),
            ),
            input_depth=input_depth,
        )


@pytest.mark.parametrize(
    "source_name",
    [
        "BM25",
        "dense",
    ],
)
def test_union_requires_exact_input_depth(
    source_name: str,
) -> None:
    bm25_hits = (
        bm25_hit(
            "passage-a",
            score=1.0,
            accepted_order=0,
        ),
        bm25_hit(
            "passage-b",
            score=0.5,
            accepted_order=1,
        ),
    )
    dense_hits = (
        dense_hit(
            "passage-a",
            score=0.9,
            accepted_order=0,
        ),
        dense_hit(
            "passage-b",
            score=0.8,
            accepted_order=1,
        ),
    )

    if source_name == "BM25":
        bm25_hits = bm25_hits[:1]
    else:
        dense_hits = dense_hits[:1]

    with pytest.raises(
        HybridCandidateError,
        match=source_name,
    ):
        build_hybrid_candidate_union(
            bm25_hits,
            dense_hits,
            input_depth=2,
        )


@pytest.mark.parametrize(
    "source_name",
    [
        "BM25",
        "dense",
    ],
)
def test_union_rejects_duplicate_ids_within_source(
    source_name: str,
) -> None:
    bm25_hits = (
        bm25_hit(
            "passage-a",
            score=1.0,
            accepted_order=0,
        ),
        bm25_hit(
            "passage-b",
            score=0.5,
            accepted_order=1,
        ),
    )
    dense_hits = (
        dense_hit(
            "passage-a",
            score=0.9,
            accepted_order=0,
        ),
        dense_hit(
            "passage-b",
            score=0.8,
            accepted_order=1,
        ),
    )

    if source_name == "BM25":
        bm25_hits = (
            bm25_hits[0],
            bm25_hit(
                "passage-a",
                score=0.5,
                accepted_order=0,
            ),
        )
    else:
        dense_hits = (
            dense_hits[0],
            dense_hit(
                "passage-a",
                score=0.8,
                accepted_order=0,
            ),
        )

    with pytest.raises(
        HybridCandidateError,
        match=f"{source_name}.*Duplicate passage_id",
    ):
        build_hybrid_candidate_union(
            bm25_hits,
            dense_hits,
            input_depth=2,
        )


def test_union_rejects_cross_retriever_order_mismatch() -> None:
    bm25_hits = (
        bm25_hit(
            "passage-a",
            score=1.0,
            accepted_order=0,
        ),
    )
    dense_hits = (
        dense_hit(
            "passage-a",
            score=0.9,
            accepted_order=4,
        ),
    )

    with pytest.raises(
        HybridCandidateError,
        match="accepted_order mismatch",
    ):
        build_hybrid_candidate_union(
            bm25_hits,
            dense_hits,
            input_depth=1,
        )


@pytest.mark.parametrize(
    ("passage_id", "accepted_order", "message"),
    [
        ("", 0, "passage_id"),
        ("passage-a", -1, "accepted_order"),
        ("passage-a", True, "accepted_order"),
    ],
)
def test_union_rejects_invalid_hit_identity(
    passage_id: str,
    accepted_order,
    message: str,
) -> None:
    with pytest.raises(
        HybridCandidateError,
        match=message,
    ):
        build_hybrid_candidate_union(
            (
                bm25_hit(
                    passage_id,
                    score=1.0,
                    accepted_order=accepted_order,
                ),
            ),
            (
                dense_hit(
                    passage_id,
                    score=0.9,
                    accepted_order=accepted_order,
                ),
            ),
            input_depth=1,
        )
