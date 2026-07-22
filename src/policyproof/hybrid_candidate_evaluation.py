"""Coverage evaluation for deterministic hybrid candidate unions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from policyproof.bm25 import (
    BM25Error,
    build_bm25_index,
    rank_bm25,
)
from policyproof.dense import (
    DenseRetrievalError,
    build_dense_index,
    rank_dense,
)
from policyproof.hybrid_candidates import (
    HybridCandidate,
    HybridCandidateError,
    build_hybrid_candidate_union,
)


class HybridCandidateEvaluationError(ValueError):
    """Raised when hybrid candidate coverage cannot be evaluated."""


@dataclass(frozen=True)
class HybridCandidateQueryResult:
    """Candidate-union coverage for one answerable query."""

    query_id: str
    candidates: tuple[HybridCandidate, ...]
    candidate_passage_ids: tuple[str, ...]
    candidate_count: int
    candidate_recall: float
    direct_evidence_hit: bool
    retrieved_gold_passage_ids: tuple[str, ...]
    missed_gold_passage_ids: tuple[str, ...]
    bm25_only_gold_passage_ids: tuple[str, ...]
    dense_only_gold_passage_ids: tuple[str, ...]


@dataclass(frozen=True)
class HybridCandidateEvaluationResult:
    """Aggregate candidate coverage before any reranking step."""

    corpus_passage_count: int
    answer_query_count: int
    abstention_query_count: int
    input_depth: int
    dense_batch_size: int
    query_results: tuple[HybridCandidateQueryResult, ...]
    mean_candidate_recall: float
    direct_evidence_hit_rate: float
    mean_candidate_count: float


@dataclass(frozen=True)
class _AnswerQuery:
    query_id: str
    question: str
    relevance_grades: tuple[tuple[str, int], ...]


def _require_positive_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise HybridCandidateEvaluationError(
            f"{field_name} must be a positive integer."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HybridCandidateEvaluationError(
            f"{field_name} must be a non-empty string."
        )

    return value


def _require_queries(
    dataset: Mapping[str, Any],
) -> tuple[tuple[_AnswerQuery, ...], int]:
    if not isinstance(dataset, Mapping):
        raise HybridCandidateEvaluationError(
            "dataset must be a mapping."
        )

    raw_queries = dataset.get("queries")

    if not isinstance(raw_queries, list):
        raise HybridCandidateEvaluationError(
            "dataset queries must be a list."
        )

    answer_queries: list[_AnswerQuery] = []
    abstention_query_count = 0
    seen_query_ids: set[str] = set()

    for position, raw_query in enumerate(raw_queries):
        if not isinstance(raw_query, Mapping):
            raise HybridCandidateEvaluationError(
                f"query at position {position} must be a mapping."
            )

        query_id = _require_nonempty_string(
            raw_query.get("query_id"),
            field_name=f"query[{position}].query_id",
        )
        question = _require_nonempty_string(
            raw_query.get("question"),
            field_name=f"{query_id}.question",
        )

        if query_id in seen_query_ids:
            raise HybridCandidateEvaluationError(
                f"Duplicate query_id: {query_id}"
            )

        seen_query_ids.add(query_id)
        expected_behavior = raw_query.get("expected_behavior")

        if expected_behavior not in {"answer", "abstain"}:
            raise HybridCandidateEvaluationError(
                f"{query_id}.expected_behavior must be "
                "'answer' or 'abstain'."
            )

        raw_judgments = raw_query.get("relevance_judgments")

        if not isinstance(raw_judgments, list):
            raise HybridCandidateEvaluationError(
                f"{query_id}.relevance_judgments must be a list."
            )

        if expected_behavior == "abstain":
            if raw_judgments:
                raise HybridCandidateEvaluationError(
                    f"{query_id}: abstention query relevance "
                    "judgments must be empty."
                )

            abstention_query_count += 1
            continue

        if not raw_judgments:
            raise HybridCandidateEvaluationError(
                f"{query_id}: answerable query requires "
                "at least one relevance judgment."
            )

        relevance_grades: list[tuple[str, int]] = []
        seen_passage_ids: set[str] = set()

        for judgment_position, raw_judgment in enumerate(
            raw_judgments
        ):
            if not isinstance(raw_judgment, Mapping):
                raise HybridCandidateEvaluationError(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}] must be a mapping."
                )

            passage_id = _require_nonempty_string(
                raw_judgment.get("passage_id"),
                field_name=(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}].passage_id"
                ),
            )
            relevance_grade = raw_judgment.get(
                "relevance_grade"
            )

            if (
                not isinstance(relevance_grade, int)
                or isinstance(relevance_grade, bool)
                or relevance_grade not in {1, 2}
            ):
                raise HybridCandidateEvaluationError(
                    f"{query_id}: relevance grade must be 1 or 2."
                )

            if passage_id in seen_passage_ids:
                raise HybridCandidateEvaluationError(
                    f"{query_id}: duplicate relevance passage_id "
                    f"{passage_id!r}."
                )

            seen_passage_ids.add(passage_id)
            relevance_grades.append(
                (
                    passage_id,
                    relevance_grade,
                )
            )

        answer_queries.append(
            _AnswerQuery(
                query_id=query_id,
                question=question,
                relevance_grades=tuple(relevance_grades),
            )
        )

    if not answer_queries:
        raise HybridCandidateEvaluationError(
            "dataset must contain at least one answerable query."
        )

    return (
        tuple(answer_queries),
        abstention_query_count,
    )


def evaluate_hybrid_candidates(
    passages: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    *,
    session: object,
    input_depth: int,
    dense_batch_size: int = 32,
) -> HybridCandidateEvaluationResult:
    """Measure reviewed-evidence coverage in a BM25+dense candidate union."""

    if not isinstance(passages, Sequence) or not passages:
        raise HybridCandidateEvaluationError(
            "passages must be a non-empty sequence."
        )

    validated_input_depth = _require_positive_integer(
        input_depth,
        field_name="input_depth",
    )
    validated_batch_size = _require_positive_integer(
        dense_batch_size,
        field_name="dense_batch_size",
    )
    passage_count = len(passages)

    if validated_input_depth > passage_count:
        raise HybridCandidateEvaluationError(
            "input_depth cannot exceed the corpus passage count."
        )

    answer_queries, abstention_query_count = _require_queries(
        dataset
    )

    try:
        bm25_index = build_bm25_index(passages)
        dense_index = build_dense_index(
            passages,
            session=session,
            batch_size=validated_batch_size,
        )
    except (
        BM25Error,
        DenseRetrievalError,
    ) as error:
        raise HybridCandidateEvaluationError(
            str(error)
        ) from error

    query_results: list[HybridCandidateQueryResult] = []

    for query in answer_queries:
        try:
            bm25_hits = rank_bm25(
                bm25_index,
                query.question,
                limit=validated_input_depth,
            )
            dense_hits = rank_dense(
                dense_index,
                query.question,
                session=session,
                limit=validated_input_depth,
            )
            candidates = build_hybrid_candidate_union(
                bm25_hits,
                dense_hits,
                input_depth=validated_input_depth,
            )
        except (
            BM25Error,
            DenseRetrievalError,
            HybridCandidateError,
        ) as error:
            raise HybridCandidateEvaluationError(
                f"{query.query_id}: {error}"
            ) from error

        candidate_ids = tuple(
            candidate.passage_id
            for candidate in candidates
        )
        candidate_id_set = set(candidate_ids)
        bm25_id_set = {
            hit.passage_id
            for hit in bm25_hits
        }
        dense_id_set = {
            hit.passage_id
            for hit in dense_hits
        }
        gold_ids = tuple(
            passage_id
            for passage_id, _ in query.relevance_grades
        )
        direct_ids = {
            passage_id
            for passage_id, grade in query.relevance_grades
            if grade == 2
        }

        retrieved_gold_ids = tuple(
            passage_id
            for passage_id in gold_ids
            if passage_id in candidate_id_set
        )
        missed_gold_ids = tuple(
            passage_id
            for passage_id in gold_ids
            if passage_id not in candidate_id_set
        )
        bm25_only_gold_ids = tuple(
            passage_id
            for passage_id in gold_ids
            if (
                passage_id in bm25_id_set
                and passage_id not in dense_id_set
            )
        )
        dense_only_gold_ids = tuple(
            passage_id
            for passage_id in gold_ids
            if (
                passage_id in dense_id_set
                and passage_id not in bm25_id_set
            )
        )

        query_results.append(
            HybridCandidateQueryResult(
                query_id=query.query_id,
                candidates=candidates,
                candidate_passage_ids=candidate_ids,
                candidate_count=len(candidates),
                candidate_recall=(
                    len(retrieved_gold_ids)
                    / len(gold_ids)
                ),
                direct_evidence_hit=bool(
                    direct_ids.intersection(candidate_id_set)
                ),
                retrieved_gold_passage_ids=(
                    retrieved_gold_ids
                ),
                missed_gold_passage_ids=missed_gold_ids,
                bm25_only_gold_passage_ids=(
                    bm25_only_gold_ids
                ),
                dense_only_gold_passage_ids=(
                    dense_only_gold_ids
                ),
            )
        )

    return HybridCandidateEvaluationResult(
        corpus_passage_count=passage_count,
        answer_query_count=len(answer_queries),
        abstention_query_count=abstention_query_count,
        input_depth=validated_input_depth,
        dense_batch_size=validated_batch_size,
        query_results=tuple(query_results),
        mean_candidate_recall=fmean(
            result.candidate_recall
            for result in query_results
        ),
        direct_evidence_hit_rate=fmean(
            float(result.direct_evidence_hit)
            for result in query_results
        ),
        mean_candidate_count=fmean(
            result.candidate_count
            for result in query_results
        ),
    )
