"""Deterministic BM25 and dense candidate-union construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from policyproof.bm25 import BM25Hit
from policyproof.dense import DenseHit


class HybridCandidateError(ValueError):
    """Raised when hybrid candidate inputs violate the accepted contract."""


@dataclass(frozen=True)
class HybridCandidate:
    """One deduplicated candidate with its source-retriever ranks."""

    passage_id: str
    accepted_order: int
    bm25_rank: int | None
    dense_rank: int | None


def _require_input_depth(input_depth: int) -> int:
    if (
        not isinstance(input_depth, int)
        or isinstance(input_depth, bool)
        or input_depth < 1
    ):
        raise HybridCandidateError(
            "input_depth must be a positive integer."
        )

    return input_depth


def _validate_hits(
    hits: Sequence[BM25Hit] | Sequence[DenseHit],
    *,
    source_name: str,
    expected_type: type[BM25Hit] | type[DenseHit],
    input_depth: int,
) -> dict[str, tuple[int, int]]:
    if not isinstance(hits, Sequence):
        raise HybridCandidateError(
            f"{source_name} hits must be a sequence."
        )

    if len(hits) != input_depth:
        raise HybridCandidateError(
            f"{source_name} hits must contain exactly "
            f"{input_depth} records."
        )

    ranks_by_id: dict[str, tuple[int, int]] = {}
    passage_id_by_order: dict[int, str] = {}

    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, expected_type):
            raise HybridCandidateError(
                f"{source_name} hit at rank {rank} has "
                "an invalid record type."
            )

        passage_id = hit.passage_id
        accepted_order = hit.accepted_order

        if not isinstance(passage_id, str) or not passage_id:
            raise HybridCandidateError(
                f"{source_name} hit at rank {rank} requires "
                "a non-empty passage_id."
            )

        if (
            not isinstance(accepted_order, int)
            or isinstance(accepted_order, bool)
            or accepted_order < 0
        ):
            raise HybridCandidateError(
                f"{source_name} hit {passage_id!r} requires "
                "a non-negative integer accepted_order."
            )

        if passage_id in ranks_by_id:
            raise HybridCandidateError(
                f"{source_name}: Duplicate passage_id: {passage_id}"
            )

        prior_passage_id = passage_id_by_order.get(accepted_order)

        if prior_passage_id is not None:
            raise HybridCandidateError(
                f"{source_name}: accepted_order {accepted_order} "
                f"is shared by {prior_passage_id!r} and "
                f"{passage_id!r}."
            )

        ranks_by_id[passage_id] = (
            rank,
            accepted_order,
        )
        passage_id_by_order[accepted_order] = passage_id

    return ranks_by_id


def build_hybrid_candidate_union(
    bm25_hits: Sequence[BM25Hit],
    dense_hits: Sequence[DenseHit],
    *,
    input_depth: int,
) -> tuple[HybridCandidate, ...]:
    """Union equal-depth retriever outputs without producing a fused ranking."""

    validated_depth = _require_input_depth(input_depth)

    bm25_by_id = _validate_hits(
        bm25_hits,
        source_name="BM25",
        expected_type=BM25Hit,
        input_depth=validated_depth,
    )
    dense_by_id = _validate_hits(
        dense_hits,
        source_name="dense",
        expected_type=DenseHit,
        input_depth=validated_depth,
    )

    candidates: list[HybridCandidate] = []

    for passage_id in bm25_by_id.keys() | dense_by_id.keys():
        bm25_record = bm25_by_id.get(passage_id)
        dense_record = dense_by_id.get(passage_id)

        if bm25_record is not None and dense_record is not None:
            bm25_rank, bm25_order = bm25_record
            dense_rank, dense_order = dense_record

            if bm25_order != dense_order:
                raise HybridCandidateError(
                    f"{passage_id}: accepted_order mismatch "
                    f"between BM25 ({bm25_order}) and "
                    f"dense ({dense_order})."
                )

            accepted_order = bm25_order
        elif bm25_record is not None:
            bm25_rank, accepted_order = bm25_record
            dense_rank = None
        elif dense_record is not None:
            dense_rank, accepted_order = dense_record
            bm25_rank = None
        else:
            raise HybridCandidateError(
                "Candidate union encountered an unreachable empty record."
            )

        candidates.append(
            HybridCandidate(
                passage_id=passage_id,
                accepted_order=accepted_order,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.accepted_order,
            candidate.passage_id,
        )
    )

    return tuple(candidates)
