from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from policyproof.hybrid_candidates import HybridCandidate
from policyproof.reranker import (
    RerankedCandidate,
    RerankerError,
    rerank_candidates,
)


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


class FakeSession:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.asarray(logits, dtype=np.float32).reshape(-1, 1)
        self.received_inputs: dict[str, np.ndarray] | None = None
        self.received_output_names: list[str] | None = None

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
        self.received_output_names = output_names
        self.received_inputs = inputs
        return [self.logits]


def passages() -> list[dict[str, object]]:
    return [
        {
            "passage_id": "passage-a",
            "retrieval_text": "Alpha evidence.",
            "document_id": "document-one",
        },
        {
            "passage_id": "passage-b",
            "retrieval_text": "Beta evidence.",
            "document_id": "document-two",
        },
        {
            "passage_id": "passage-c",
            "retrieval_text": "Gamma evidence.",
            "document_id": "document-three",
        },
        {
            "passage_id": "passage-d",
            "retrieval_text": "Delta evidence.",
            "document_id": "document-four",
        },
    ]


def candidates() -> tuple[HybridCandidate, ...]:
    return (
        HybridCandidate(
            passage_id="passage-a",
            accepted_order=0,
            bm25_rank=2,
            dense_rank=1,
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
            bm25_rank=1,
            dense_rank=None,
        ),
    )


def test_reranked_candidate_contract_preserves_provenance() -> None:
    assert {
        field.name
        for field in fields(RerankedCandidate)
    } == {
        "passage_id",
        "accepted_order",
        "bm25_rank",
        "dense_rank",
        "reranker_score",
        "reranker_rank",
    }


def test_rerank_candidates_scores_only_supplied_candidate_union() -> None:
    session = FakeSession([0.25, 2.0, -1.0])

    results = rerank_candidates(
        passages(),
        "Which evidence is most relevant?",
        candidates(),
        session=session,
    )

    assert results == (
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
            reranker_score=0.25,
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

    assert session.received_inputs is not None
    assert session.received_inputs["input_ids"].shape[0] == 3

    # passage-b is in the corpus but not in the accepted candidate union.
    # Its text must never be sent to the reranker.
    assert len(results) == 3
    assert "passage-b" not in {
        result.passage_id
        for result in results
    }


def test_rerank_candidates_breaks_equal_logits_by_accepted_order_then_id() -> None:
    tied_candidates = (
        HybridCandidate(
            passage_id="passage-d",
            accepted_order=3,
            bm25_rank=1,
            dense_rank=None,
        ),
        HybridCandidate(
            passage_id="passage-c",
            accepted_order=2,
            bm25_rank=None,
            dense_rank=2,
        ),
        HybridCandidate(
            passage_id="passage-a",
            accepted_order=0,
            bm25_rank=2,
            dense_rank=1,
        ),
    )
    session = FakeSession([1.5, 1.5, 1.5])

    results = rerank_candidates(
        passages(),
        "controlled question",
        tied_candidates,
        session=session,
    )

    assert [
        result.passage_id
        for result in results
    ] == [
        "passage-a",
        "passage-c",
        "passage-d",
    ]
    assert [
        result.reranker_rank
        for result in results
    ] == [1, 2, 3]


def test_rerank_candidates_does_not_use_document_scope() -> None:
    session = FakeSession([0.0, 3.0, 1.0])

    results = rerank_candidates(
        passages(),
        "Question whose benchmark scope might name document-one",
        candidates(),
        session=session,
    )

    # A passage from document-three remains eligible and ranks first.
    assert results[0].passage_id == "passage-c"


@pytest.mark.parametrize(
    ("question", "candidate_records", "message"),
    [
        ("", candidates(), "question"),
        ("   ", candidates(), "question"),
        ("valid question", (), "candidate"),
        (
            "valid question",
            (
                HybridCandidate(
                    passage_id="missing",
                    accepted_order=0,
                    bm25_rank=1,
                    dense_rank=None,
                ),
            ),
            "missing",
        ),
        (
            "valid question",
            (
                HybridCandidate(
                    passage_id="passage-a",
                    accepted_order=2,
                    bm25_rank=1,
                    dense_rank=None,
                ),
            ),
            "accepted_order",
        ),
    ],
)
def test_rerank_candidates_rejects_invalid_inputs(
    question: str,
    candidate_records: tuple[HybridCandidate, ...],
    message: str,
) -> None:
    with pytest.raises(
        RerankerError,
        match=message,
    ):
        rerank_candidates(
            passages(),
            question,
            candidate_records,
            session=FakeSession(
                [0.0] * max(1, len(candidate_records))
            ),
        )


def test_rerank_candidates_rejects_duplicate_candidate_ids() -> None:
    duplicate_candidates = (
        HybridCandidate(
            passage_id="passage-a",
            accepted_order=0,
            bm25_rank=1,
            dense_rank=None,
        ),
        HybridCandidate(
            passage_id="passage-a",
            accepted_order=0,
            bm25_rank=None,
            dense_rank=1,
        ),
    )

    with pytest.raises(
        RerankerError,
        match="Duplicate",
    ):
        rerank_candidates(
            passages(),
            "valid question",
            duplicate_candidates,
            session=FakeSession([1.0, 0.5]),
        )
