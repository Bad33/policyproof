"""Deterministic corpus-wide dense passage retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from policyproof.dense_model import (
    DenseModelError,
    embed_dense_texts,
)

DEFAULT_BATCH_SIZE = 32


class DenseRetrievalError(ValueError):
    """Raised when dense retrieval input or model output is invalid."""


@dataclass(frozen=True)
class DenseIndex:
    """Immutable dense embeddings in accepted passage order."""

    passage_ids: tuple[str, ...]
    accepted_orders: tuple[int, ...]
    embeddings: np.ndarray
    batch_size: int


@dataclass(frozen=True)
class DenseHit:
    """One deterministically ranked dense-retrieval result."""

    passage_id: str
    score: float
    accepted_order: int


def _require_batch_size(batch_size: Any) -> int:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise DenseRetrievalError(
            "batch_size must be a positive integer."
        )

    return batch_size


def _validate_embedding_batch(
    embeddings: Any,
    *,
    expected_rows: int,
    expected_dimension: int | None,
) -> np.ndarray:
    values = np.asarray(embeddings)

    if values.ndim != 2:
        raise DenseRetrievalError(
            "Dense embedding batch must be a rank-2 array."
        )

    if values.shape[0] != expected_rows:
        raise DenseRetrievalError(
            "Dense embedding batch row count does not match "
            "the requested text count."
        )

    if values.shape[1] < 1:
        raise DenseRetrievalError(
            "Dense embedding dimension must be greater than zero."
        )

    if (
        expected_dimension is not None
        and values.shape[1] != expected_dimension
    ):
        raise DenseRetrievalError(
            "Dense embedding batch dimension does not match "
            f"the existing index dimension {expected_dimension}."
        )

    if not np.issubdtype(values.dtype, np.floating):
        raise DenseRetrievalError(
            "Dense embeddings must use a floating-point dtype."
        )

    values = np.asarray(values, dtype=np.float32)

    if not np.all(np.isfinite(values)):
        raise DenseRetrievalError(
            "Dense embeddings must contain only finite values."
        )

    norms = np.linalg.norm(
        values,
        axis=1,
    )

    if np.any(norms == 0):
        raise DenseRetrievalError(
            "Dense embeddings must not contain zero-length vectors."
        )

    if not np.allclose(
        norms,
        np.ones(expected_rows),
        atol=1e-5,
    ):
        raise DenseRetrievalError(
            "Dense embeddings must be L2-normalized."
        )

    return values


def _validate_passages(
    passages: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if (
        not isinstance(passages, Sequence)
        or isinstance(passages, (str, bytes))
        or not passages
    ):
        raise DenseRetrievalError(
            "At least one passage is required."
        )

    passage_ids: list[str] = []
    retrieval_texts: list[str] = []
    seen_passage_ids: set[str] = set()

    for accepted_order, passage in enumerate(passages):
        if not isinstance(passage, Mapping):
            raise DenseRetrievalError(
                f"Passage at accepted order {accepted_order} "
                "must be a mapping."
            )

        passage_id = passage.get("passage_id")
        retrieval_text = passage.get("retrieval_text")

        if not isinstance(passage_id, str) or not passage_id:
            raise DenseRetrievalError(
                f"Passage at accepted order {accepted_order} "
                "requires a non-empty passage_id."
            )

        if passage_id in seen_passage_ids:
            raise DenseRetrievalError(
                f"Duplicate passage_id: {passage_id}"
            )

        if (
            not isinstance(retrieval_text, str)
            or not retrieval_text.strip()
        ):
            raise DenseRetrievalError(
                f"{passage_id}: retrieval_text must be "
                "a non-empty string."
            )

        seen_passage_ids.add(passage_id)
        passage_ids.append(passage_id)
        retrieval_texts.append(retrieval_text)

    return tuple(passage_ids), tuple(retrieval_texts)


def build_dense_index(
    passages: Sequence[Mapping[str, Any]],
    *,
    session: object,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DenseIndex:
    """Embed all accepted passages in bounded deterministic batches."""

    validated_batch_size = _require_batch_size(batch_size)
    passage_ids, retrieval_texts = _validate_passages(passages)
    embedding_batches: list[np.ndarray] = []
    embedding_dimension: int | None = None

    for start in range(
        0,
        len(retrieval_texts),
        validated_batch_size,
    ):
        batch_texts = retrieval_texts[
            start : start + validated_batch_size
        ]

        try:
            raw_embeddings = embed_dense_texts(
                session,
                batch_texts,
                is_query=False,
            )
        except DenseModelError as error:
            raise DenseRetrievalError(str(error)) from error

        embeddings = _validate_embedding_batch(
            raw_embeddings,
            expected_rows=len(batch_texts),
            expected_dimension=embedding_dimension,
        )

        if embedding_dimension is None:
            embedding_dimension = embeddings.shape[1]

        embedding_batches.append(embeddings)

    combined_embeddings = np.ascontiguousarray(
        np.concatenate(
            embedding_batches,
            axis=0,
        ),
        dtype=np.float32,
    )

    return DenseIndex(
        passage_ids=passage_ids,
        accepted_orders=tuple(range(len(passage_ids))),
        embeddings=combined_embeddings,
        batch_size=validated_batch_size,
    )


def _validate_index(index: DenseIndex) -> np.ndarray:
    if not isinstance(index, DenseIndex):
        raise DenseRetrievalError(
            "index must be a DenseIndex."
        )

    passage_count = len(index.passage_ids)

    if passage_count < 1:
        raise DenseRetrievalError(
            "Dense index must contain at least one passage."
        )

    if len(index.accepted_orders) != passage_count:
        raise DenseRetrievalError(
            "Dense index accepted-order count does not match "
            "its passage count."
        )

    if len(set(index.passage_ids)) != passage_count:
        raise DenseRetrievalError(
            "Dense index contains duplicate passage IDs."
        )

    if index.accepted_orders != tuple(range(passage_count)):
        raise DenseRetrievalError(
            "Dense index accepted orders must preserve "
            "contiguous corpus order."
        )

    return _validate_embedding_batch(
        index.embeddings,
        expected_rows=passage_count,
        expected_dimension=None,
    )


def rank_dense(
    index: DenseIndex,
    query: str,
    *,
    session: object,
    limit: int,
) -> tuple[DenseHit, ...]:
    """Rank all indexed passages by normalized embedding dot product."""

    passage_embeddings = _validate_index(index)

    if not isinstance(query, str) or not query.strip():
        raise DenseRetrievalError(
            "query must be a non-empty string."
        )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):
        raise DenseRetrievalError(
            "limit must be a positive integer."
        )

    try:
        raw_query_embedding = embed_dense_texts(
            session,
            [query],
            is_query=True,
        )
    except DenseModelError as error:
        raise DenseRetrievalError(str(error)) from error

    query_embedding = _validate_embedding_batch(
        raw_query_embedding,
        expected_rows=1,
        expected_dimension=passage_embeddings.shape[1],
    )

    raw_scores = passage_embeddings @ query_embedding[0]
    hits: list[DenseHit] = []

    for position, score in enumerate(raw_scores):
        score_value = float(score)

        if not math.isfinite(score_value):
            raise DenseRetrievalError(
                "Dense similarity score must be finite."
            )

        hits.append(
            DenseHit(
                passage_id=index.passage_ids[position],
                score=score_value,
                accepted_order=index.accepted_orders[position],
            )
        )

    hits.sort(
        key=lambda hit: (
            -hit.score,
            hit.accepted_order,
            hit.passage_id,
        )
    )

    return tuple(hits[:limit])
