"""Deterministic evaluation of corpus-wide dense retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from policyproof.dense import (
    DenseHit,
    DenseRetrievalError,
    build_dense_index,
    rank_dense,
)
from policyproof.retrieval_metrics import (
    RetrievalMetricError,
    direct_evidence_hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class DenseEvaluationError(ValueError):
    """Raised when dense-retrieval evaluation input is invalid."""


@dataclass(frozen=True)
class DenseQueryResult:
    """Deterministic dense-ranking metrics for one answerable query."""

    query_id: str
    ranked_hits: tuple[DenseHit, ...]
    ranked_passage_ids: tuple[str, ...]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank_at_10: float
    direct_evidence_hit_at_10: bool
    ndcg_at_10: float


@dataclass(frozen=True)
class DenseEvaluationResult:
    """Aggregate dense metrics over answerable benchmark queries."""

    corpus_passage_count: int
    answer_query_count: int
    abstention_query_count: int
    query_results: tuple[DenseQueryResult, ...]
    batch_size: int
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mrr_at_10: float
    direct_evidence_hit_rate_at_10: float
    mean_ndcg_at_10: float


def _require_queries(
    dataset: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """Return benchmark queries after minimal shape validation."""

    queries = dataset.get("queries")

    if (
        not isinstance(queries, Sequence)
        or isinstance(queries, (str, bytes))
        or not queries
    ):
        raise DenseEvaluationError(
            "Dataset queries must be a non-empty sequence."
        )

    for position, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise DenseEvaluationError(
                f"Query at position {position} must be a mapping."
            )

    return queries


def _relevance_grades(
    query: Mapping[str, Any],
) -> dict[str, int]:
    """Return validated passage-level grades for one answer query."""

    query_id = query.get("query_id")
    judgments = query.get("relevance_judgments")

    if (
        not isinstance(judgments, Sequence)
        or isinstance(judgments, (str, bytes))
        or not judgments
    ):
        raise DenseEvaluationError(
            f"{query_id}: answer query requires relevance judgments."
        )

    grades: dict[str, int] = {}

    for judgment in judgments:
        if not isinstance(judgment, Mapping):
            raise DenseEvaluationError(
                f"{query_id}: relevance judgment must be a mapping."
            )

        passage_id = judgment.get("passage_id")
        grade = judgment.get("relevance_grade")

        if not isinstance(passage_id, str) or not passage_id:
            raise DenseEvaluationError(
                f"{query_id}: judgment requires a non-empty passage_id."
            )

        if passage_id in grades:
            raise DenseEvaluationError(
                f"{query_id}: duplicate judgment for {passage_id}."
            )

        if (
            not isinstance(grade, int)
            or isinstance(grade, bool)
            or grade < 1
        ):
            raise DenseEvaluationError(
                f"{query_id}: relevance grade must be a positive integer."
            )

        grades[passage_id] = grade

    return grades


def evaluate_dense(
    passages: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    *,
    session: object,
    batch_size: int = 32,
) -> DenseEvaluationResult:
    """Evaluate full-corpus dense ranking on answerable queries."""

    if not isinstance(dataset, Mapping):
        raise DenseEvaluationError("dataset must be a mapping.")

    queries = _require_queries(dataset)

    try:
        index = build_dense_index(
            passages,
            session=session,
            batch_size=batch_size,
        )
    except DenseRetrievalError as error:
        raise DenseEvaluationError(str(error)) from error

    query_results: list[DenseQueryResult] = []
    abstention_query_count = 0
    ranking_limit = min(10, len(index.passage_ids))

    for query in queries:
        query_id = query.get("query_id")
        question = query.get("question")
        expected_behavior = query.get("expected_behavior")

        if not isinstance(query_id, str) or not query_id:
            raise DenseEvaluationError(
                "Each query requires a non-empty query_id."
            )

        if not isinstance(question, str) or not question.strip():
            raise DenseEvaluationError(
                f"{query_id}: question must be a non-empty string."
            )

        if expected_behavior == "abstain":
            abstention_query_count += 1
            continue

        if expected_behavior != "answer":
            raise DenseEvaluationError(
                f"{query_id}: unsupported expected_behavior "
                f"{expected_behavior!r}."
            )

        relevance_grades = _relevance_grades(query)

        try:
            hits = rank_dense(
                index,
                question,
                session=session,
                limit=ranking_limit,
            )
        except DenseRetrievalError as error:
            raise DenseEvaluationError(str(error)) from error

        ranked_passage_ids = tuple(
            hit.passage_id
            for hit in hits
        )

        try:
            result = DenseQueryResult(
                query_id=query_id,
                ranked_hits=hits,
                ranked_passage_ids=ranked_passage_ids,
                recall_at_1=recall_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=1,
                ),
                recall_at_3=recall_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=3,
                ),
                recall_at_5=recall_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=5,
                ),
                recall_at_10=recall_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=10,
                ),
                reciprocal_rank_at_10=reciprocal_rank_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=10,
                ),
                direct_evidence_hit_at_10=direct_evidence_hit_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=10,
                ),
                ndcg_at_10=ndcg_at_k(
                    ranked_passage_ids,
                    relevance_grades,
                    k=10,
                ),
            )
        except RetrievalMetricError as error:
            raise DenseEvaluationError(str(error)) from error

        query_results.append(result)

    if not query_results:
        raise DenseEvaluationError(
            "Dataset must contain at least one answerable query."
        )

    frozen_results = tuple(query_results)

    return DenseEvaluationResult(
        corpus_passage_count=len(index.passage_ids),
        answer_query_count=len(frozen_results),
        abstention_query_count=abstention_query_count,
        query_results=frozen_results,
        batch_size=index.batch_size,
        mean_recall_at_1=fmean(
            result.recall_at_1
            for result in frozen_results
        ),
        mean_recall_at_3=fmean(
            result.recall_at_3
            for result in frozen_results
        ),
        mean_recall_at_5=fmean(
            result.recall_at_5
            for result in frozen_results
        ),
        mean_recall_at_10=fmean(
            result.recall_at_10
            for result in frozen_results
        ),
        mrr_at_10=fmean(
            result.reciprocal_rank_at_10
            for result in frozen_results
        ),
        direct_evidence_hit_rate_at_10=fmean(
            result.direct_evidence_hit_at_10
            for result in frozen_results
        ),
        mean_ndcg_at_10=fmean(
            result.ndcg_at_10
            for result in frozen_results
        ),
    )
