from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import policyproof.dense as dense_module
from policyproof.dense import (
    DenseIndex,
    DenseRetrievalError,
    build_dense_index,
    rank_dense,
)


def passage(
    passage_id: str,
    retrieval_text: str,
) -> dict[str, str]:
    return {
        "passage_id": passage_id,
        "retrieval_text": retrieval_text,
    }


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


def install_embedder(
    monkeypatch: pytest.MonkeyPatch,
    vectors_by_text: dict[str, Sequence[float]],
) -> ControlledEmbedder:
    embedder = ControlledEmbedder(vectors_by_text)
    monkeypatch.setattr(
        dense_module,
        "embed_dense_texts",
        embedder,
    )
    return embedder


def test_build_dense_index_embeds_passages_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = install_embedder(
        monkeypatch,
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "gamma text": [math.sqrt(0.5), math.sqrt(0.5)],
        },
    )

    index = build_dense_index(
        [
            passage("passage-b", "alpha text"),
            passage("passage-a", "beta text"),
            passage("passage-c", "gamma text"),
        ],
        session=object(),
        batch_size=2,
    )

    assert index.passage_ids == (
        "passage-b",
        "passage-a",
        "passage-c",
    )
    assert index.accepted_orders == (0, 1, 2)
    assert index.embeddings.shape == (3, 2)
    assert index.embeddings.dtype == np.float32
    assert index.batch_size == 2

    assert embedder.calls == [
        (
            ("alpha text", "beta text"),
            False,
        ),
        (
            ("gamma text",),
            False,
        ),
    ]


def test_rank_dense_uses_query_embedding_and_dot_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = install_embedder(
        monkeypatch,
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "mixed text": [math.sqrt(0.5), math.sqrt(0.5)],
            "alpha query": [1.0, 0.0],
        },
    )

    index = build_dense_index(
        [
            passage("passage-alpha", "alpha text"),
            passage("passage-beta", "beta text"),
            passage("passage-mixed", "mixed text"),
        ],
        session=object(),
        batch_size=3,
    )

    hits = rank_dense(
        index,
        "alpha query",
        session=object(),
        limit=3,
    )

    assert tuple(hit.passage_id for hit in hits) == (
        "passage-alpha",
        "passage-mixed",
        "passage-beta",
    )
    assert tuple(hit.accepted_order for hit in hits) == (
        0,
        2,
        1,
    )
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(math.sqrt(0.5))
    assert hits[2].score == pytest.approx(0.0)

    assert embedder.calls[-1] == (
        ("alpha query",),
        True,
    )


def test_rank_dense_breaks_equal_score_ties_by_accepted_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_embedder(
        monkeypatch,
        {
            "first text": [1.0, 0.0],
            "second text": [1.0, 0.0],
            "same query": [1.0, 0.0],
        },
    )

    index = build_dense_index(
        [
            passage("passage-z", "first text"),
            passage("passage-a", "second text"),
        ],
        session=object(),
    )

    hits = rank_dense(
        index,
        "same query",
        session=object(),
        limit=2,
    )

    assert tuple(hit.passage_id for hit in hits) == (
        "passage-z",
        "passage-a",
    )


def test_rank_dense_respects_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_embedder(
        monkeypatch,
        {
            "first": [1.0, 0.0],
            "second": [0.0, 1.0],
            "query": [1.0, 0.0],
        },
    )

    index = build_dense_index(
        [
            passage("passage-1", "first"),
            passage("passage-2", "second"),
        ],
        session=object(),
    )

    hits = rank_dense(
        index,
        "query",
        session=object(),
        limit=1,
    )

    assert len(hits) == 1
    assert hits[0].passage_id == "passage-1"


@pytest.mark.parametrize(
    "passages",
    [
        [],
        [
            passage("duplicate", "first"),
            passage("duplicate", "second"),
        ],
        [
            passage("", "text"),
        ],
        [
            passage("passage-1", ""),
        ],
        [
            {
                "passage_id": "passage-1",
            },
        ],
        [
            "not-a-mapping",
        ],
    ],
)
def test_build_dense_index_rejects_invalid_passages(
    passages: list[Any],
) -> None:
    with pytest.raises(DenseRetrievalError):
        build_dense_index(
            passages,
            session=object(),
        )


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_build_dense_index_rejects_invalid_batch_size(
    batch_size: Any,
) -> None:
    with pytest.raises(
        DenseRetrievalError,
        match="batch_size",
    ):
        build_dense_index(
            [passage("passage-1", "text")],
            session=object(),
            batch_size=batch_size,
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        np.zeros((1,), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.full((1, 3), np.nan, dtype=np.float32),
        np.zeros((1, 0), dtype=np.float32),
    ],
)
def test_build_dense_index_rejects_invalid_embedding_batches(
    monkeypatch: pytest.MonkeyPatch,
    embeddings: np.ndarray,
) -> None:
    def invalid_embedder(
        session: object,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> np.ndarray:
        del session, texts, is_query
        return embeddings

    monkeypatch.setattr(
        dense_module,
        "embed_dense_texts",
        invalid_embedder,
    )

    with pytest.raises(DenseRetrievalError):
        build_dense_index(
            [passage("passage-1", "text")],
            session=object(),
        )


@pytest.mark.parametrize(
    ("query", "limit"),
    [
        ("", 1),
        ("   ", 1),
        (123, 1),
        ("query", 0),
        ("query", -1),
        ("query", True),
    ],
)
def test_rank_dense_rejects_invalid_query_or_limit(
    query: Any,
    limit: Any,
) -> None:
    index = DenseIndex(
        passage_ids=("passage-1",),
        accepted_orders=(0,),
        embeddings=np.asarray(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
        batch_size=32,
    )

    with pytest.raises(DenseRetrievalError):
        rank_dense(
            index,
            query,
            session=object(),
            limit=limit,
        )


def test_rank_dense_rejects_query_embedding_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_embedder(
        monkeypatch,
        {
            "query": [1.0, 0.0, 0.0],
        },
    )

    index = DenseIndex(
        passage_ids=("passage-1",),
        accepted_orders=(0,),
        embeddings=np.asarray(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
        batch_size=32,
    )

    with pytest.raises(
        DenseRetrievalError,
        match="dimension",
    ):
        rank_dense(
            index,
            "query",
            session=object(),
            limit=1,
        )
