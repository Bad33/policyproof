"""Ranking evaluation for cross-encoded hybrid candidate unions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from policyproof.hybrid_candidates import HybridCandidate
from policyproof.reranker import (
    RerankedCandidate,
    RerankerError,
    rerank_candidates,
)
from policyproof.retrieval_metrics import (
    RetrievalMetricError,
    direct_evidence_hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class RerankerEvaluationError(ValueError):
    """Raised when reranker evaluation inputs or outputs are invalid."""


@dataclass(frozen=True)
class RerankerQueryResult:
    """Reranker metrics for one answerable benchmark query."""

    query_id: str
    ranked_candidates: tuple[RerankedCandidate, ...]
    ranked_passage_ids: tuple[str, ...]
    candidate_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank_at_10: float
    direct_evidence_hit_at_10: bool
    ndcg_at_10: float


@dataclass(frozen=True)
class RerankerEvaluationResult:
    """Aggregate reranker metrics over answerable benchmark queries."""

    corpus_passage_count: int
    answer_query_count: int
    abstention_query_count: int
    query_results: tuple[RerankerQueryResult, ...]
    mean_candidate_count: float
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mrr_at_10: float
    direct_evidence_hit_rate_at_10: float
    mean_ndcg_at_10: float


@dataclass(frozen=True)
class _AnswerQuery:
    query_id: str
    question: str
    relevance_grades: dict[str, int]


@dataclass(frozen=True)
class _CandidateQuery:
    query_id: str
    question: str
    candidates: tuple[HybridCandidate, ...]


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RerankerEvaluationError(
            f"{field_name} must be a mapping."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankerEvaluationError(
            f"{field_name} must be a non-empty string."
        )

    return value


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
        raise RerankerEvaluationError(
            f"{field_name} must be a positive integer."
        )

    return value


def _require_sequence(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or (not allow_empty and not value)
    ):
        qualifier = "" if allow_empty else "non-empty "
        raise RerankerEvaluationError(
            f"{field_name} must be a {qualifier}sequence."
        )

    return value


def _require_answer_queries(
    dataset: Mapping[str, Any],
) -> tuple[tuple[_AnswerQuery, ...], int]:
    raw_queries = _require_sequence(
        dataset.get("queries"),
        field_name="dataset.queries",
    )

    answer_queries: list[_AnswerQuery] = []
    abstention_query_count = 0
    seen_query_ids: set[str] = set()

    for position, raw_query in enumerate(raw_queries):
        query = _require_mapping(
            raw_query,
            field_name=f"dataset.queries[{position}]",
        )
        query_id = _require_nonempty_string(
            query.get("query_id"),
            field_name=f"dataset.queries[{position}].query_id",
        )
        question = _require_nonempty_string(
            query.get("question"),
            field_name=f"{query_id}.question",
        )

        if query_id in seen_query_ids:
            raise RerankerEvaluationError(
                f"Duplicate dataset query_id: {query_id}"
            )

        seen_query_ids.add(query_id)
        expected_behavior = query.get("expected_behavior")
        raw_judgments = _require_sequence(
            query.get("relevance_judgments"),
            field_name=f"{query_id}.relevance_judgments",
            allow_empty=True,
        )

        if expected_behavior == "abstain":
            if raw_judgments:
                raise RerankerEvaluationError(
                    f"{query_id}: abstention query relevance "
                    "judgments must be empty."
                )

            abstention_query_count += 1
            continue

        if expected_behavior != "answer":
            raise RerankerEvaluationError(
                f"{query_id}: unsupported expected_behavior "
                f"{expected_behavior!r}."
            )

        if not raw_judgments:
            raise RerankerEvaluationError(
                f"{query_id}: answer query requires "
                "relevance judgments."
            )

        relevance_grades: dict[str, int] = {}

        for judgment_position, raw_judgment in enumerate(
            raw_judgments
        ):
            judgment = _require_mapping(
                raw_judgment,
                field_name=(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}]"
                ),
            )
            passage_id = _require_nonempty_string(
                judgment.get("passage_id"),
                field_name=(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}].passage_id"
                ),
            )
            grade = judgment.get("relevance_grade")

            if passage_id in relevance_grades:
                raise RerankerEvaluationError(
                    f"{query_id}: duplicate judgment for "
                    f"{passage_id}."
                )

            if (
                not isinstance(grade, int)
                or isinstance(grade, bool)
                or grade < 1
            ):
                raise RerankerEvaluationError(
                    f"{query_id}: relevance grade must be "
                    "a positive integer."
                )

            relevance_grades[passage_id] = grade

        answer_queries.append(
            _AnswerQuery(
                query_id=query_id,
                question=question,
                relevance_grades=relevance_grades,
            )
        )

    if not answer_queries:
        raise RerankerEvaluationError(
            "Dataset must contain at least one answerable query."
        )

    return (
        tuple(answer_queries),
        abstention_query_count,
    )


def _require_optional_rank(
    value: Any,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    return _require_positive_integer(
        value,
        field_name=field_name,
    )


def _require_candidate_queries(
    candidate_artifact: Mapping[str, Any],
    *,
    answer_queries: Sequence[_AnswerQuery],
) -> dict[str, _CandidateQuery]:
    generation = _require_mapping(
        candidate_artifact.get("candidate_generation"),
        field_name="candidate_artifact.candidate_generation",
    )

    if generation.get("final_ranking") is not False:
        raise RerankerEvaluationError(
            "candidate_generation.final_ranking must be false."
        )

    evaluation = _require_mapping(
        candidate_artifact.get("evaluation"),
        field_name="candidate_artifact.evaluation",
    )
    raw_results = _require_sequence(
        evaluation.get("query_results"),
        field_name="candidate_artifact.evaluation.query_results",
    )

    answer_queries_by_id = {
        query.query_id: query
        for query in answer_queries
    }
    results_by_id: dict[str, _CandidateQuery] = {}

    for position, raw_result in enumerate(raw_results):
        result = _require_mapping(
            raw_result,
            field_name=(
                "candidate_artifact.evaluation."
                f"query_results[{position}]"
            ),
        )
        query_id = _require_nonempty_string(
            result.get("query_id"),
            field_name=(
                "candidate_artifact.evaluation."
                f"query_results[{position}].query_id"
            ),
        )

        if query_id in results_by_id:
            raise RerankerEvaluationError(
                f"Duplicate candidate query result: {query_id}"
            )

        answer_query = answer_queries_by_id.get(query_id)

        if answer_query is None:
            raise RerankerEvaluationError(
                f"{query_id}: candidate result is not an "
                "answer query."
            )

        question = _require_nonempty_string(
            result.get("question"),
            field_name=f"{query_id}.question",
        )

        if question != answer_query.question:
            raise RerankerEvaluationError(
                f"{query_id}: candidate question does not match "
                "the benchmark question."
            )

        raw_candidates = _require_sequence(
            result.get("candidates"),
            field_name=f"{query_id}.candidates",
        )
        raw_candidate_ids = _require_sequence(
            result.get("candidate_passage_ids"),
            field_name=f"{query_id}.candidate_passage_ids",
        )
        candidate_count = _require_positive_integer(
            result.get("candidate_count"),
            field_name=f"{query_id}.candidate_count",
        )

        candidates: list[HybridCandidate] = []

        for candidate_position, raw_candidate in enumerate(
            raw_candidates
        ):
            candidate = _require_mapping(
                raw_candidate,
                field_name=(
                    f"{query_id}.candidates"
                    f"[{candidate_position}]"
                ),
            )
            passage_id = _require_nonempty_string(
                candidate.get("passage_id"),
                field_name=(
                    f"{query_id}.candidates"
                    f"[{candidate_position}].passage_id"
                ),
            )
            accepted_order = candidate.get("accepted_order")

            if (
                not isinstance(accepted_order, int)
                or isinstance(accepted_order, bool)
                or accepted_order < 0
            ):
                raise RerankerEvaluationError(
                    f"{query_id}: accepted_order must be "
                    "a non-negative integer."
                )

            candidates.append(
                HybridCandidate(
                    passage_id=passage_id,
                    accepted_order=accepted_order,
                    bm25_rank=_require_optional_rank(
                        candidate.get("bm25_rank"),
                        field_name=(
                            f"{query_id}.{passage_id}.bm25_rank"
                        ),
                    ),
                    dense_rank=_require_optional_rank(
                        candidate.get("dense_rank"),
                        field_name=(
                            f"{query_id}.{passage_id}.dense_rank"
                        ),
                    ),
                )
            )

        candidate_ids = tuple(
            candidate.passage_id
            for candidate in candidates
        )

        if tuple(raw_candidate_ids) != candidate_ids:
            raise RerankerEvaluationError(
                f"{query_id}: candidate_passage_ids do not match "
                "the candidate records."
            )

        if candidate_count != len(candidates):
            raise RerankerEvaluationError(
                f"{query_id}: candidate_count does not match "
                "the candidate records."
            )

        results_by_id[query_id] = _CandidateQuery(
            query_id=query_id,
            question=question,
            candidates=tuple(candidates),
        )

    expected_ids = set(answer_queries_by_id)

    if set(results_by_id) != expected_ids:
        missing_ids = sorted(expected_ids - set(results_by_id))
        raise RerankerEvaluationError(
            "Candidate artifact must contain every answer query "
            f"exactly once; missing: {missing_ids}"
        )

    return results_by_id


def evaluate_reranker(
    passages: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    *,
    session: object,
) -> RerankerEvaluationResult:
    """Evaluate reranking over accepted hybrid candidate unions."""

    if (
        not isinstance(passages, Sequence)
        or isinstance(passages, (str, bytes))
        or not passages
    ):
        raise RerankerEvaluationError(
            "passages must be a non-empty sequence."
        )

    dataset = _require_mapping(
        dataset,
        field_name="dataset",
    )
    candidate_artifact = _require_mapping(
        candidate_artifact,
        field_name="candidate_artifact",
    )

    answer_queries, abstention_query_count = (
        _require_answer_queries(dataset)
    )
    candidate_queries = _require_candidate_queries(
        candidate_artifact,
        answer_queries=answer_queries,
    )

    query_results: list[RerankerQueryResult] = []

    for query in answer_queries:
        candidate_query = candidate_queries[query.query_id]

        try:
            ranked_candidates = rerank_candidates(
                passages,
                query.question,
                candidate_query.candidates,
                session=session,
            )
        except RerankerError as error:
            raise RerankerEvaluationError(
                f"{query.query_id}: {error}"
            ) from error

        ranked_passage_ids = tuple(
            candidate.passage_id
            for candidate in ranked_candidates
        )

        try:
            query_result = RerankerQueryResult(
                query_id=query.query_id,
                ranked_candidates=ranked_candidates,
                ranked_passage_ids=ranked_passage_ids,
                candidate_count=len(ranked_candidates),
                recall_at_1=recall_at_k(
                    ranked_passage_ids,
                    query.relevance_grades,
                    k=1,
                ),
                recall_at_3=recall_at_k(
                    ranked_passage_ids,
                    query.relevance_grades,
                    k=3,
                ),
                recall_at_5=recall_at_k(
                    ranked_passage_ids,
                    query.relevance_grades,
                    k=5,
                ),
                recall_at_10=recall_at_k(
                    ranked_passage_ids,
                    query.relevance_grades,
                    k=10,
                ),
                reciprocal_rank_at_10=reciprocal_rank_at_k(
                    ranked_passage_ids,
                    query.relevance_grades,
                    k=10,
                ),
                direct_evidence_hit_at_10=(
                    direct_evidence_hit_at_k(
                        ranked_passage_ids,
                        query.relevance_grades,
                        k=10,
                    )
                ),
                ndcg_at_10=ndcg_at_k(
                    ranked_passage_ids,
                    query.relevance_grades,
                    k=10,
                ),
            )
        except RetrievalMetricError as error:
            raise RerankerEvaluationError(
                f"{query.query_id}: {error}"
            ) from error

        query_results.append(query_result)

    frozen_results = tuple(query_results)

    return RerankerEvaluationResult(
        corpus_passage_count=len(passages),
        answer_query_count=len(frozen_results),
        abstention_query_count=abstention_query_count,
        query_results=frozen_results,
        mean_candidate_count=fmean(
            result.candidate_count
            for result in frozen_results
        ),
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
