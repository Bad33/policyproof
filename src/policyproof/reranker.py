"""Deterministic cross-encoder ranking of accepted hybrid candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from policyproof.hybrid_candidates import HybridCandidate
from policyproof.reranker_model import (
    RerankerModelError,
    score_reranker_pairs,
)


class RerankerError(ValueError):
    """Raised when candidate reranking inputs or outputs are invalid."""


@dataclass(frozen=True)
class RerankedCandidate:
    """One reranked candidate retaining source-retriever provenance."""

    passage_id: str
    accepted_order: int
    bm25_rank: int | None
    dense_rank: int | None
    reranker_score: float
    reranker_rank: int


def _require_passages(
    passages: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, str]]:
    """Index validated corpus passages by ID and accepted order."""

    if (
        not isinstance(passages, Sequence)
        or isinstance(passages, (str, bytes))
        or not passages
    ):
        raise RerankerError(
            "passages must be a non-empty sequence."
        )

    passages_by_id: dict[str, tuple[int, str]] = {}

    for accepted_order, passage in enumerate(passages):
        if not isinstance(passage, Mapping):
            raise RerankerError(
                f"Passage at accepted_order {accepted_order} "
                "must be a mapping."
            )

        passage_id = passage.get("passage_id")
        retrieval_text = passage.get("retrieval_text")

        if not isinstance(passage_id, str) or not passage_id:
            raise RerankerError(
                f"Passage at accepted_order {accepted_order} "
                "requires a non-empty passage_id."
            )

        if not isinstance(retrieval_text, str) or not retrieval_text:
            raise RerankerError(
                f"Passage {passage_id!r} requires non-empty retrieval_text."
            )

        if passage_id in passages_by_id:
            raise RerankerError(
                f"Duplicate passage_id: {passage_id}"
            )

        passages_by_id[passage_id] = (
            accepted_order,
            retrieval_text,
        )

    return passages_by_id


def _require_source_rank(
    value: int | None,
    *,
    field_name: str,
    passage_id: str,
) -> None:
    if value is None:
        return

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise RerankerError(
            f"{passage_id}: {field_name} must be a positive integer "
            "or None."
        )


def _require_candidates(
    candidates: Sequence[HybridCandidate],
    *,
    passages_by_id: Mapping[str, tuple[int, str]],
) -> tuple[HybridCandidate, ...]:
    """Validate that candidates are an exact subset of the corpus."""

    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        raise RerankerError(
            "candidate records must be a non-empty sequence."
        )

    validated = tuple(candidates)
    seen_passage_ids: set[str] = set()
    seen_accepted_orders: set[int] = set()

    for position, candidate in enumerate(validated):
        if not isinstance(candidate, HybridCandidate):
            raise RerankerError(
                f"Candidate at position {position} has an invalid type."
            )

        passage_id = candidate.passage_id

        if not isinstance(passage_id, str) or not passage_id:
            raise RerankerError(
                f"Candidate at position {position} requires "
                "a non-empty passage_id."
            )

        if passage_id in seen_passage_ids:
            raise RerankerError(
                f"Duplicate candidate passage_id: {passage_id}"
            )

        corpus_record = passages_by_id.get(passage_id)

        if corpus_record is None:
            raise RerankerError(
                f"Candidate passage {passage_id!r} is missing "
                "from the corpus."
            )

        expected_order, _ = corpus_record

        if (
            not isinstance(candidate.accepted_order, int)
            or isinstance(candidate.accepted_order, bool)
            or candidate.accepted_order < 0
        ):
            raise RerankerError(
                f"{passage_id}: accepted_order must be "
                "a non-negative integer."
            )

        if candidate.accepted_order != expected_order:
            raise RerankerError(
                f"{passage_id}: accepted_order "
                f"{candidate.accepted_order} does not match corpus order "
                f"{expected_order}."
            )

        if candidate.accepted_order in seen_accepted_orders:
            raise RerankerError(
                f"Duplicate candidate accepted_order: "
                f"{candidate.accepted_order}"
            )

        _require_source_rank(
            candidate.bm25_rank,
            field_name="bm25_rank",
            passage_id=passage_id,
        )
        _require_source_rank(
            candidate.dense_rank,
            field_name="dense_rank",
            passage_id=passage_id,
        )

        if candidate.bm25_rank is None and candidate.dense_rank is None:
            raise RerankerError(
                f"{passage_id}: candidate must retain at least one "
                "source-retriever rank."
            )

        seen_passage_ids.add(passage_id)
        seen_accepted_orders.add(candidate.accepted_order)

    return validated


def rerank_candidates(
    passages: Sequence[Mapping[str, Any]],
    question: str,
    candidates: Sequence[HybridCandidate],
    *,
    session: object,
) -> tuple[RerankedCandidate, ...]:
    """Rank only the supplied hybrid candidate union by raw model logit."""

    if not isinstance(question, str) or not question.strip():
        raise RerankerError(
            "question must be a non-empty string."
        )

    passages_by_id = _require_passages(passages)
    validated_candidates = _require_candidates(
        candidates,
        passages_by_id=passages_by_id,
    )

    pairs = tuple(
        (
            question,
            passages_by_id[candidate.passage_id][1],
        )
        for candidate in validated_candidates
    )

    try:
        raw_scores = score_reranker_pairs(
            session,
            pairs,
        )
    except RerankerModelError as error:
        raise RerankerError(str(error)) from error

    if len(raw_scores) != len(validated_candidates):
        raise RerankerError(
            "Reranker score count does not match candidate count."
        )

    scored: list[tuple[HybridCandidate, float]] = []

    for candidate, raw_score in zip(
        validated_candidates,
        raw_scores,
        strict=True,
    ):
        score = float(raw_score)

        if not math.isfinite(score):
            raise RerankerError(
                f"{candidate.passage_id}: reranker score must be finite."
            )

        scored.append((candidate, score))

    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].accepted_order,
            item[0].passage_id,
        )
    )

    return tuple(
        RerankedCandidate(
            passage_id=candidate.passage_id,
            accepted_order=candidate.accepted_order,
            bm25_rank=candidate.bm25_rank,
            dense_rank=candidate.dense_rank,
            reranker_score=score,
            reranker_rank=rank,
        )
        for rank, (candidate, score) in enumerate(
            scored,
            start=1,
        )
    )
