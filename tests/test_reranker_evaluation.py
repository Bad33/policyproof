from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

import policyproof.reranker_evaluation as evaluation_module
from policyproof.reranker import RerankedCandidate, RerankerError
from policyproof.reranker_evaluation import (
    RerankerEvaluationError,
    RerankerEvaluationResult,
    RerankerQueryResult,
    evaluate_reranker,
)
from policyproof.retrieval_metrics import ndcg_at_k


class FakeNode:
    def __init__(
        self,
        name: str,
        node_type: str,
        shape: list[object],
    ) -> None:
        self.name = name
        self.type = node_type
        self.shape = shape


class QueueSession:
    def __init__(
        self,
        outputs: list[list[float]],
    ) -> None:
        self.outputs = [
            np.asarray(output, dtype=np.float32).reshape(-1, 1)
            for output in outputs
        ]
        self.call_count = 0

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[FakeNode]:
        return [
            FakeNode(
                "input_ids",
                "tensor(int64)",
                ["batch_size", "sequence_length"],
            ),
            FakeNode(
                "attention_mask",
                "tensor(int64)",
                ["batch_size", "sequence_length"],
            ),
            FakeNode(
                "token_type_ids",
                "tensor(int64)",
                ["batch_size", "sequence_length"],
            ),
        ]

    def get_outputs(self) -> list[FakeNode]:
        return [
            FakeNode(
                "logits",
                "tensor(float)",
                ["batch_size", 1],
            )
        ]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        del output_names, inputs

        output = self.outputs[self.call_count]
        self.call_count += 1
        return [output]


def passages() -> list[dict[str, str]]:
    return [
        {
            "passage_id": "passage-a",
            "document_id": "document-one",
            "retrieval_text": "Alpha direct evidence.",
        },
        {
            "passage_id": "passage-b",
            "document_id": "document-two",
            "retrieval_text": "Beta direct evidence.",
        },
        {
            "passage_id": "passage-c",
            "document_id": "document-three",
            "retrieval_text": "Alpha supporting evidence.",
        },
        {
            "passage_id": "passage-d",
            "document_id": "document-four",
            "retrieval_text": "Unrelated evidence.",
        },
    ]


def dataset() -> dict[str, Any]:
    return {
        "queries": [
            {
                "query_id": "answer-alpha",
                "question": "What is alpha evidence?",
                "expected_behavior": "answer",
                "document_scope": ["document-one"],
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
                "query_id": "answer-beta",
                "question": "What is beta evidence?",
                "expected_behavior": "answer",
                "document_scope": ["document-two"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-b",
                        "relevance_grade": 2,
                    }
                ],
            },
            {
                "query_id": "abstain-weather",
                "question": "What is tomorrow's weather?",
                "expected_behavior": "abstain",
                "document_scope": [],
                "relevance_judgments": [],
            },
        ]
    }


def candidate_artifact() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "result_id": "policyproof-hybrid-candidate-baseline",
        "result_version": "0.1.0",
        "candidate_generation": {
            "strategy": "deduplicated_union",
            "final_ranking": False,
        },
        "evaluation": {
            "answer_query_count": 2,
            "abstention_query_count": 1,
            "query_results": [
                {
                    "query_id": "answer-alpha",
                    "question": "What is alpha evidence?",
                    "candidate_passage_ids": [
                        "passage-a",
                        "passage-c",
                        "passage-d",
                    ],
                    "candidate_count": 3,
                    "candidates": [
                        {
                            "passage_id": "passage-a",
                            "accepted_order": 0,
                            "bm25_rank": 2,
                            "dense_rank": 1,
                        },
                        {
                            "passage_id": "passage-c",
                            "accepted_order": 2,
                            "bm25_rank": None,
                            "dense_rank": 2,
                        },
                        {
                            "passage_id": "passage-d",
                            "accepted_order": 3,
                            "bm25_rank": 1,
                            "dense_rank": None,
                        },
                    ],
                },
                {
                    "query_id": "answer-beta",
                    "question": "What is beta evidence?",
                    "candidate_passage_ids": [
                        "passage-b",
                        "passage-d",
                    ],
                    "candidate_count": 2,
                    "candidates": [
                        {
                            "passage_id": "passage-b",
                            "accepted_order": 1,
                            "bm25_rank": 1,
                            "dense_rank": 2,
                        },
                        {
                            "passage_id": "passage-d",
                            "accepted_order": 3,
                            "bm25_rank": 2,
                            "dense_rank": 1,
                        },
                    ],
                },
            ],
        },
    }


def test_evaluation_dataclass_contracts_are_explicit() -> None:
    assert {
        field.name
        for field in fields(RerankerQueryResult)
    } == {
        "query_id",
        "ranked_candidates",
        "ranked_passage_ids",
        "candidate_count",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "reciprocal_rank_at_10",
        "direct_evidence_hit_at_10",
        "ndcg_at_10",
    }

    assert {
        field.name
        for field in fields(RerankerEvaluationResult)
    } == {
        "corpus_passage_count",
        "answer_query_count",
        "abstention_query_count",
        "query_results",
        "mean_candidate_count",
        "mean_recall_at_1",
        "mean_recall_at_3",
        "mean_recall_at_5",
        "mean_recall_at_10",
        "mrr_at_10",
        "direct_evidence_hit_rate_at_10",
        "mean_ndcg_at_10",
    }


def test_evaluate_reranker_ranks_full_candidate_unions_and_metrics() -> None:
    session = QueueSession(
        [
            [0.2, 2.0, -1.0],
            [3.0, 1.0],
        ]
    )

    result = evaluate_reranker(
        passages(),
        dataset(),
        candidate_artifact(),
        session=session,
    )

    assert result.corpus_passage_count == 4
    assert result.answer_query_count == 2
    assert result.abstention_query_count == 1
    assert result.mean_candidate_count == pytest.approx(2.5)
    assert session.call_count == 2

    alpha = result.query_results[0]

    assert alpha.query_id == "answer-alpha"
    assert alpha.candidate_count == 3
    assert alpha.ranked_passage_ids == (
        "passage-c",
        "passage-a",
        "passage-d",
    )
    assert alpha.ranked_candidates == (
        RerankedCandidate(
            passage_id="passage-c",
            accepted_order=2,
            bm25_rank=None,
            dense_rank=2,
            reranker_score=2.0,
            reranker_rank=1,
        ),
        RerankedCandidate(
            passage_id="passage-a",
            accepted_order=0,
            bm25_rank=2,
            dense_rank=1,
            reranker_score=pytest.approx(0.2),
            reranker_rank=2,
        ),
        RerankedCandidate(
            passage_id="passage-d",
            accepted_order=3,
            bm25_rank=1,
            dense_rank=None,
            reranker_score=-1.0,
            reranker_rank=3,
        ),
    )
    assert alpha.recall_at_1 == pytest.approx(0.5)
    assert alpha.recall_at_3 == pytest.approx(1.0)
    assert alpha.recall_at_5 == pytest.approx(1.0)
    assert alpha.recall_at_10 == pytest.approx(1.0)
    assert alpha.reciprocal_rank_at_10 == pytest.approx(1.0)
    assert alpha.direct_evidence_hit_at_10 is True
    assert alpha.ndcg_at_10 == pytest.approx(
        ndcg_at_k(
            alpha.ranked_passage_ids,
            {
                "passage-a": 2,
                "passage-c": 1,
            },
            k=10,
        )
    )

    beta = result.query_results[1]

    assert beta.query_id == "answer-beta"
    assert beta.ranked_passage_ids == (
        "passage-b",
        "passage-d",
    )
    assert beta.recall_at_1 == pytest.approx(1.0)
    assert beta.reciprocal_rank_at_10 == pytest.approx(1.0)
    assert beta.direct_evidence_hit_at_10 is True
    assert beta.ndcg_at_10 == pytest.approx(1.0)

    assert result.mean_recall_at_1 == pytest.approx(0.75)
    assert result.mean_recall_at_3 == pytest.approx(1.0)
    assert result.mean_recall_at_5 == pytest.approx(1.0)
    assert result.mean_recall_at_10 == pytest.approx(1.0)
    assert result.mrr_at_10 == pytest.approx(1.0)
    assert result.direct_evidence_hit_rate_at_10 == pytest.approx(1.0)
    assert result.mean_ndcg_at_10 == pytest.approx(
        (alpha.ndcg_at_10 + 1.0) / 2
    )


def test_evaluation_does_not_filter_candidates_by_document_scope() -> None:
    result = evaluate_reranker(
        passages(),
        dataset(),
        candidate_artifact(),
        session=QueueSession(
            [
                [0.0, 5.0, 1.0],
                [2.0, 0.0],
            ]
        ),
    )

    # passage-c belongs to document-three, outside the benchmark scope.
    assert result.query_results[0].ranked_passage_ids[0] == "passage-c"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda artifact: artifact["evaluation"]["query_results"].pop(),
        lambda artifact: artifact["evaluation"]["query_results"].append(
            artifact["evaluation"]["query_results"][0]
        ),
    ],
)
def test_evaluation_requires_each_answer_query_exactly_once(
    mutator,
) -> None:
    artifact = candidate_artifact()
    mutator(artifact)

    with pytest.raises(
        RerankerEvaluationError,
        match="answer query|Duplicate",
    ):
        evaluate_reranker(
            passages(),
            dataset(),
            artifact,
            session=QueueSession([[0.0, 0.0, 0.0]]),
        )


def test_evaluation_rejects_candidate_question_mismatch() -> None:
    artifact = candidate_artifact()
    artifact["evaluation"]["query_results"][0]["question"] = (
        "Changed question"
    )

    with pytest.raises(
        RerankerEvaluationError,
        match="question",
    ):
        evaluate_reranker(
            passages(),
            dataset(),
            artifact,
            session=QueueSession([[0.0, 0.0, 0.0], [0.0, 0.0]]),
        )


def test_evaluation_rejects_candidate_id_list_mismatch() -> None:
    artifact = candidate_artifact()
    artifact["evaluation"]["query_results"][0][
        "candidate_passage_ids"
    ].reverse()

    with pytest.raises(
        RerankerEvaluationError,
        match="candidate_passage_ids",
    ):
        evaluate_reranker(
            passages(),
            dataset(),
            artifact,
            session=QueueSession([[0.0, 0.0, 0.0], [0.0, 0.0]]),
        )


def test_evaluation_rejects_already_ranked_candidate_input() -> None:
    artifact = candidate_artifact()
    artifact["candidate_generation"]["final_ranking"] = True

    with pytest.raises(
        RerankerEvaluationError,
        match="final_ranking",
    ):
        evaluate_reranker(
            passages(),
            dataset(),
            artifact,
            session=QueueSession([[0.0, 0.0, 0.0], [0.0, 0.0]]),
        )


def test_evaluation_wraps_reranker_failure_with_query_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_reranking(*args, **kwargs):
        raise RerankerError("controlled reranker failure")

    monkeypatch.setattr(
        evaluation_module,
        "rerank_candidates",
        fail_reranking,
    )

    with pytest.raises(
        RerankerEvaluationError,
        match="answer-alpha.*controlled reranker failure",
    ):
        evaluate_reranker(
            passages(),
            dataset(),
            candidate_artifact(),
            session=object(),
        )
