from __future__ import annotations

from typing import Any

import pytest

import policyproof.hybrid_candidate_evaluation as evaluation_module
from policyproof.bm25 import BM25Hit
from policyproof.dense import DenseHit
from policyproof.hybrid_candidate_evaluation import (
    HybridCandidateEvaluationError,
    evaluate_hybrid_candidates,
)


def passage(
    passage_id: str,
    *,
    retrieval_text: str | None = None,
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "retrieval_text": retrieval_text or passage_id,
    }


def benchmark() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": "policyproof-retrieval-evaluation",
        "dataset_version": "0.1.1",
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": "a" * 64,
        "query_count": 3,
        "queries": [
            {
                "query_id": "answer-001",
                "question": "What is required?",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-a",
                        "relevance_grade": 2,
                    },
                    {
                        "passage_id": "passage-c",
                        "relevance_grade": 1,
                    },
                ],
            },
            {
                "query_id": "answer-002",
                "question": "What else is required?",
                "expected_behavior": "answer",
                "document_scope": ["document-b"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-d",
                        "relevance_grade": 2,
                    }
                ],
            },
            {
                "query_id": "abstain-001",
                "question": "What is the weather?",
                "expected_behavior": "abstain",
                "document_scope": [],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [],
            },
        ],
    }


def install_controlled_retrievers(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    bm25_index = object()
    dense_index = object()
    session = object()
    captured: dict[str, object] = {}

    bm25_rankings = {
        "What is required?": (
            BM25Hit(
                passage_id="passage-a",
                score=4.0,
                accepted_order=0,
            ),
            BM25Hit(
                passage_id="passage-b",
                score=3.0,
                accepted_order=1,
            ),
        ),
        "What else is required?": (
            BM25Hit(
                passage_id="passage-b",
                score=2.0,
                accepted_order=1,
            ),
            BM25Hit(
                passage_id="passage-c",
                score=1.0,
                accepted_order=2,
            ),
        ),
    }
    dense_rankings = {
        "What is required?": (
            DenseHit(
                passage_id="passage-c",
                score=0.9,
                accepted_order=2,
            ),
            DenseHit(
                passage_id="passage-b",
                score=0.8,
                accepted_order=1,
            ),
        ),
        "What else is required?": (
            DenseHit(
                passage_id="passage-d",
                score=0.95,
                accepted_order=3,
            ),
            DenseHit(
                passage_id="passage-a",
                score=0.7,
                accepted_order=0,
            ),
        ),
    }

    def fake_build_bm25_index(passages):
        captured["bm25_passages"] = passages
        return bm25_index

    def fake_build_dense_index(
        passages,
        *,
        session: object,
        batch_size: int,
    ):
        captured["dense_passages"] = passages
        captured["dense_session"] = session
        captured["batch_size"] = batch_size
        return dense_index

    def fake_rank_bm25(
        index: object,
        query: str,
        *,
        limit: int,
    ):
        assert index is bm25_index
        captured.setdefault("bm25_calls", []).append(
            (query, limit)
        )
        return bm25_rankings[query]

    def fake_rank_dense(
        index: object,
        query: str,
        *,
        session: object,
        limit: int,
    ):
        assert index is dense_index
        captured.setdefault("dense_calls", []).append(
            (query, limit, session)
        )
        return dense_rankings[query]

    monkeypatch.setattr(
        evaluation_module,
        "build_bm25_index",
        fake_build_bm25_index,
    )
    monkeypatch.setattr(
        evaluation_module,
        "build_dense_index",
        fake_build_dense_index,
    )
    monkeypatch.setattr(
        evaluation_module,
        "rank_bm25",
        fake_rank_bm25,
    )
    monkeypatch.setattr(
        evaluation_module,
        "rank_dense",
        fake_rank_dense,
    )

    captured["session"] = session
    return captured


def test_evaluation_builds_one_index_per_retriever_and_measures_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_controlled_retrievers(monkeypatch)
    passages = [
        passage("passage-a"),
        passage("passage-b"),
        passage("passage-c"),
        passage("passage-d"),
    ]

    result = evaluate_hybrid_candidates(
        passages,
        benchmark(),
        session=captured["session"],
        input_depth=2,
        dense_batch_size=16,
    )

    assert result.corpus_passage_count == 4
    assert result.answer_query_count == 2
    assert result.abstention_query_count == 1
    assert result.input_depth == 2
    assert result.dense_batch_size == 16
    assert result.mean_candidate_recall == 1.0
    assert result.direct_evidence_hit_rate == 1.0
    assert result.mean_candidate_count == 3.5

    assert captured["bm25_passages"] is passages
    assert captured["dense_passages"] is passages
    assert captured["dense_session"] is captured["session"]
    assert captured["batch_size"] == 16
    assert captured["bm25_calls"] == [
        ("What is required?", 2),
        ("What else is required?", 2),
    ]
    assert captured["dense_calls"] == [
        ("What is required?", 2, captured["session"]),
        ("What else is required?", 2, captured["session"]),
    ]


def test_evaluation_records_per_query_candidate_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_controlled_retrievers(monkeypatch)

    result = evaluate_hybrid_candidates(
        [
            passage("passage-a"),
            passage("passage-b"),
            passage("passage-c"),
            passage("passage-d"),
        ],
        benchmark(),
        session=captured["session"],
        input_depth=2,
    )

    first = result.query_results[0]

    assert first.query_id == "answer-001"
    assert first.candidate_passage_ids == (
        "passage-a",
        "passage-b",
        "passage-c",
    )
    assert first.candidate_count == 3
    assert first.candidate_recall == 1.0
    assert first.direct_evidence_hit is True
    assert first.retrieved_gold_passage_ids == (
        "passage-a",
        "passage-c",
    )
    assert first.missed_gold_passage_ids == ()
    assert first.bm25_only_gold_passage_ids == (
        "passage-a",
    )
    assert first.dense_only_gold_passage_ids == (
        "passage-c",
    )

    assert [
        (
            candidate.passage_id,
            candidate.bm25_rank,
            candidate.dense_rank,
        )
        for candidate in first.candidates
    ] == [
        ("passage-a", 1, None),
        ("passage-b", 2, 2),
        ("passage-c", None, 1),
    ]


def test_evaluation_does_not_rank_or_score_union_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_controlled_retrievers(monkeypatch)

    result = evaluate_hybrid_candidates(
        [
            passage("passage-a"),
            passage("passage-b"),
            passage("passage-c"),
            passage("passage-d"),
        ],
        benchmark(),
        session=captured["session"],
        input_depth=2,
    )

    candidate = result.query_results[0].candidates[0]

    assert not hasattr(candidate, "score")
    assert not hasattr(candidate, "rank")
    assert not hasattr(result.query_results[0], "ndcg_at_10")
    assert not hasattr(result.query_results[0], "reciprocal_rank_at_10")


def test_evaluation_excludes_abstentions_from_candidate_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_controlled_retrievers(monkeypatch)

    result = evaluate_hybrid_candidates(
        [
            passage("passage-a"),
            passage("passage-b"),
            passage("passage-c"),
            passage("passage-d"),
        ],
        benchmark(),
        session=captured["session"],
        input_depth=2,
    )

    assert [
        query_result.query_id
        for query_result in result.query_results
    ] == [
        "answer-001",
        "answer-002",
    ]


@pytest.mark.parametrize(
    "invalid_depth",
    [
        0,
        -1,
        True,
        1.5,
        5,
    ],
)
def test_evaluation_rejects_invalid_input_depth(
    invalid_depth,
) -> None:
    with pytest.raises(
        HybridCandidateEvaluationError,
        match="input_depth",
    ):
        evaluate_hybrid_candidates(
            [
                passage("passage-a"),
                passage("passage-b"),
                passage("passage-c"),
                passage("passage-d"),
            ],
            benchmark(),
            session=object(),
            input_depth=invalid_depth,
        )


@pytest.mark.parametrize(
    "invalid_batch_size",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_evaluation_rejects_invalid_dense_batch_size(
    invalid_batch_size,
) -> None:
    with pytest.raises(
        HybridCandidateEvaluationError,
        match="dense_batch_size",
    ):
        evaluate_hybrid_candidates(
            [
                passage("passage-a"),
                passage("passage-b"),
                passage("passage-c"),
                passage("passage-d"),
            ],
            benchmark(),
            session=object(),
            input_depth=2,
            dense_batch_size=invalid_batch_size,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda data: data.update({"queries": []}),
            "answerable",
        ),
        (
            lambda data: data["queries"][0].update(
                {"expected_behavior": "unsupported"}
            ),
            "expected_behavior",
        ),
        (
            lambda data: data["queries"][0].update(
                {"relevance_judgments": []}
            ),
            "relevance",
        ),
    ],
)
def test_evaluation_rejects_invalid_dataset_shape(
    mutator,
    message: str,
) -> None:
    dataset = benchmark()
    mutator(dataset)

    with pytest.raises(
        HybridCandidateEvaluationError,
        match=message,
    ):
        evaluate_hybrid_candidates(
            [
                passage("passage-a"),
                passage("passage-b"),
                passage("passage-c"),
                passage("passage-d"),
            ],
            dataset,
            session=object(),
            input_depth=2,
        )
