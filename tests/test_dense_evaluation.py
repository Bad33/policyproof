from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import policyproof.dense as dense_module
from policyproof.dense_evaluation import (
    DenseEvaluationError,
    evaluate_dense,
)


class ControlledEmbedder:
    def __init__(
        self,
        vectors_by_text: dict[str, Sequence[float]],
    ) -> None:
        self.vectors_by_text = vectors_by_text
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def __call__(
        self,
        session: object,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> np.ndarray:
        del session

        frozen_texts = tuple(texts)
        self.calls.append((frozen_texts, is_query))

        return np.asarray(
            [
                self.vectors_by_text[text]
                for text in frozen_texts
            ],
            dtype=np.float32,
        )


def passages() -> list[dict[str, str]]:
    return [
        {
            "passage_id": "passage-alpha",
            "retrieval_text": "alpha passage",
        },
        {
            "passage_id": "passage-beta",
            "retrieval_text": "beta passage",
        },
        {
            "passage_id": "passage-mixed",
            "retrieval_text": "mixed passage",
        },
    ]


def benchmark() -> dict[str, Any]:
    return {
        "queries": [
            {
                "query_id": "query-alpha",
                "question": "alpha query",
                "expected_behavior": "answer",
                "relevance_judgments": [
                    {
                        "passage_id": "passage-alpha",
                        "relevance_grade": 2,
                    },
                    {
                        "passage_id": "passage-mixed",
                        "relevance_grade": 1,
                    },
                ],
            },
            {
                "query_id": "query-beta",
                "question": "beta query",
                "expected_behavior": "answer",
                "relevance_judgments": [
                    {
                        "passage_id": "passage-beta",
                        "relevance_grade": 2,
                    }
                ],
            },
            {
                "query_id": "query-abstain",
                "question": "unsupported query",
                "expected_behavior": "abstain",
                "relevance_judgments": [],
            },
        ]
    }


def install_controlled_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> ControlledEmbedder:
    embedder = ControlledEmbedder(
        {
            "alpha passage": [1.0, 0.0],
            "beta passage": [0.0, 1.0],
            "mixed passage": [
                math.sqrt(0.5),
                math.sqrt(0.5),
            ],
            "alpha query": [1.0, 0.0],
            "beta query": [0.0, 1.0],
        }
    )
    monkeypatch.setattr(
        dense_module,
        "embed_dense_texts",
        embedder,
    )
    return embedder


def test_evaluate_dense_ranks_full_corpus_and_calculates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = install_controlled_embedder(monkeypatch)

    result = evaluate_dense(
        passages(),
        benchmark(),
        session=object(),
        batch_size=2,
    )

    assert result.corpus_passage_count == 3
    assert result.answer_query_count == 2
    assert result.abstention_query_count == 1
    assert result.batch_size == 2
    assert tuple(
        query_result.query_id
        for query_result in result.query_results
    ) == (
        "query-alpha",
        "query-beta",
    )

    alpha_result = result.query_results[0]

    assert alpha_result.ranked_passage_ids == (
        "passage-alpha",
        "passage-mixed",
        "passage-beta",
    )
    assert alpha_result.recall_at_1 == pytest.approx(0.5)
    assert alpha_result.recall_at_3 == pytest.approx(1.0)
    assert alpha_result.recall_at_5 == pytest.approx(1.0)
    assert alpha_result.recall_at_10 == pytest.approx(1.0)
    assert alpha_result.reciprocal_rank_at_10 == pytest.approx(1.0)
    assert alpha_result.direct_evidence_hit_at_10 is True
    assert alpha_result.ndcg_at_10 == pytest.approx(1.0)

    beta_result = result.query_results[1]

    assert beta_result.ranked_passage_ids == (
        "passage-beta",
        "passage-mixed",
        "passage-alpha",
    )
    assert beta_result.recall_at_1 == pytest.approx(1.0)
    assert beta_result.recall_at_3 == pytest.approx(1.0)
    assert beta_result.reciprocal_rank_at_10 == pytest.approx(1.0)
    assert beta_result.direct_evidence_hit_at_10 is True
    assert beta_result.ndcg_at_10 == pytest.approx(1.0)

    assert result.mean_recall_at_1 == pytest.approx(0.75)
    assert result.mean_recall_at_3 == pytest.approx(1.0)
    assert result.mean_recall_at_5 == pytest.approx(1.0)
    assert result.mean_recall_at_10 == pytest.approx(1.0)
    assert result.mrr_at_10 == pytest.approx(1.0)
    assert result.direct_evidence_hit_rate_at_10 == pytest.approx(1.0)
    assert result.mean_ndcg_at_10 == pytest.approx(1.0)

    assert embedder.calls == [
        (
            ("alpha passage", "beta passage"),
            False,
        ),
        (
            ("mixed passage",),
            False,
        ),
        (
            ("alpha query",),
            True,
        ),
        (
            ("beta query",),
            True,
        ),
    ]


def test_evaluate_dense_limits_rankings_to_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_passages = [
        {
            "passage_id": f"passage-{index:02d}",
            "retrieval_text": f"text-{index:02d}",
        }
        for index in range(12)
    ]
    vectors = {
        f"text-{index:02d}": [1.0, 0.0]
        for index in range(12)
    }
    vectors["query"] = [1.0, 0.0]

    monkeypatch.setattr(
        dense_module,
        "embed_dense_texts",
        ControlledEmbedder(vectors),
    )

    dataset = {
        "queries": [
            {
                "query_id": "query-1",
                "question": "query",
                "expected_behavior": "answer",
                "relevance_judgments": [
                    {
                        "passage_id": "passage-00",
                        "relevance_grade": 2,
                    }
                ],
            }
        ]
    }

    result = evaluate_dense(
        synthetic_passages,
        dataset,
        session=object(),
    )

    assert len(result.query_results[0].ranked_hits) == 10
    assert len(result.query_results[0].ranked_passage_ids) == 10


@pytest.mark.parametrize(
    "dataset",
    [
        None,
        {},
        {"queries": []},
        {"queries": "not-a-sequence"},
        {"queries": ["not-a-mapping"]},
    ],
)
def test_evaluate_dense_rejects_invalid_dataset_shape(
    dataset: Any,
) -> None:
    with pytest.raises(DenseEvaluationError):
        evaluate_dense(
            passages(),
            dataset,
            session=object(),
        )


def test_evaluate_dense_rejects_unsupported_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_controlled_embedder(monkeypatch)
    dataset = benchmark()
    dataset["queries"][0]["expected_behavior"] = "unknown"

    with pytest.raises(
        DenseEvaluationError,
        match="unsupported expected_behavior",
    ):
        evaluate_dense(
            passages(),
            dataset,
            session=object(),
        )


def test_evaluate_dense_requires_answerable_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_controlled_embedder(monkeypatch)
    dataset = {
        "queries": [
            {
                "query_id": "abstain-only",
                "question": "unsupported query",
                "expected_behavior": "abstain",
                "relevance_judgments": [],
            }
        ]
    }

    with pytest.raises(
        DenseEvaluationError,
        match="answerable",
    ):
        evaluate_dense(
            passages(),
            dataset,
            session=object(),
        )


@pytest.mark.parametrize(
    "judgments",
    [
        None,
        [],
        ["not-a-mapping"],
        [
            {
                "passage_id": "",
                "relevance_grade": 2,
            }
        ],
        [
            {
                "passage_id": "passage-alpha",
                "relevance_grade": 0,
            }
        ],
        [
            {
                "passage_id": "passage-alpha",
                "relevance_grade": 2,
            },
            {
                "passage_id": "passage-alpha",
                "relevance_grade": 1,
            },
        ],
    ],
)
def test_evaluate_dense_rejects_invalid_relevance_judgments(
    monkeypatch: pytest.MonkeyPatch,
    judgments: Any,
) -> None:
    install_controlled_embedder(monkeypatch)
    dataset = benchmark()
    dataset["queries"][0]["relevance_judgments"] = judgments

    with pytest.raises(DenseEvaluationError):
        evaluate_dense(
            passages(),
            dataset,
            session=object(),
        )
