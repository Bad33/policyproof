"""Versioned result artifacts for deterministic candidate reranking."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from policyproof.reranker_evaluation import (
    RerankerEvaluationResult,
)
from policyproof.reranker_model import RERANKER_MODEL_CONTRACT
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT
from policyproof.retrieval_units import (
    RetrievalUnitError,
    write_json_atomically,
)

RESULT_SCHEMA_VERSION = "1.0"
RESULT_ID = "policyproof-reranker-baseline"

_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RerankerResultError(ValueError):
    """Raised when a reranker result artifact is invalid."""


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RerankerResultError(
            f"{field_name} must be a mapping."
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
        raise RerankerResultError(
            f"{field_name} must be a {qualifier}sequence."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankerResultError(
            f"{field_name} must be a non-empty string."
        )

    return value


def _require_version(
    value: Any,
    *,
    field_name: str,
) -> str:
    version = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if _VERSION_PATTERN.fullmatch(version) is None:
        raise RerankerResultError(
            f"{field_name} must use semantic version form X.Y.Z."
        )

    return version


def _require_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    checksum = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if _SHA256_PATTERN.fullmatch(checksum) is None:
        raise RerankerResultError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return checksum


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
        raise RerankerResultError(
            f"{field_name} must be a positive integer."
        )

    return value


def _require_nonnegative_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise RerankerResultError(
            f"{field_name} must be a non-negative integer."
        )

    return value


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


def _require_metric(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise RerankerResultError(
            f"{field_name} must be a finite number from 0 to 1."
        )

    converted = float(value)

    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise RerankerResultError(
            f"{field_name} must be a finite number from 0 to 1."
        )

    return converted


def _require_nonnegative_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise RerankerResultError(
            f"{field_name} must be a finite non-negative number."
        )

    converted = float(value)

    if not math.isfinite(converted) or converted < 0:
        raise RerankerResultError(
            f"{field_name} must be a finite non-negative number."
        )

    return converted


def _require_score(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise RerankerResultError(
            f"{field_name} score must be a finite number."
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise RerankerResultError(
            f"{field_name} score must be a finite number."
        )

    return converted


def _dataset_queries(
    dataset: Mapping[str, Any],
) -> tuple[dict[str, str], int]:
    raw_queries = _require_sequence(
        dataset.get("queries"),
        field_name="dataset.queries",
    )

    questions_by_id: dict[str, str] = {}
    abstention_count = 0

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

        if query_id in questions_by_id:
            raise RerankerResultError(
                f"Duplicate dataset query_id: {query_id}"
            )

        expected_behavior = query.get("expected_behavior")

        if expected_behavior == "abstain":
            abstention_count += 1
        elif expected_behavior != "answer":
            raise RerankerResultError(
                f"{query_id}: unsupported expected_behavior "
                f"{expected_behavior!r}."
            )

        questions_by_id[query_id] = question

    return questions_by_id, abstention_count


def _candidate_results_by_id(
    candidate_artifact: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    evaluation = _require_mapping(
        candidate_artifact.get("evaluation"),
        field_name="candidate_artifact.evaluation",
    )
    raw_results = _require_sequence(
        evaluation.get("query_results"),
        field_name="candidate_artifact.evaluation.query_results",
    )

    results_by_id: dict[str, Mapping[str, Any]] = {}

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
            raise RerankerResultError(
                f"Duplicate candidate query result: {query_id}"
            )

        results_by_id[query_id] = result

    return results_by_id


def _validate_candidate_bindings(
    candidate_artifact: Mapping[str, Any],
    *,
    benchmark_sha256: str,
    passage_sha256: str,
    corpus_passage_count: int,
) -> tuple[str, str, str]:
    candidate_schema_version = _require_nonempty_string(
        candidate_artifact.get("schema_version"),
        field_name="candidate_artifact.schema_version",
    )
    candidate_result_id = _require_nonempty_string(
        candidate_artifact.get("result_id"),
        field_name="candidate_artifact.result_id",
    )
    candidate_result_version = _require_version(
        candidate_artifact.get("result_version"),
        field_name="candidate_artifact.result_version",
    )

    bindings = _require_mapping(
        candidate_artifact.get("bindings"),
        field_name="candidate_artifact.bindings",
    )
    benchmark_binding = _require_mapping(
        bindings.get("benchmark"),
        field_name="candidate_artifact.bindings.benchmark",
    )
    passage_binding = _require_mapping(
        bindings.get("passages"),
        field_name="candidate_artifact.bindings.passages",
    )

    bound_benchmark_sha256 = _require_sha256(
        benchmark_binding.get("sha256"),
        field_name=(
            "candidate_artifact.bindings.benchmark.sha256"
        ),
    )
    bound_passage_sha256 = _require_sha256(
        passage_binding.get("sha256"),
        field_name=(
            "candidate_artifact.bindings.passages.sha256"
        ),
    )
    bound_passage_count = _require_positive_integer(
        passage_binding.get("count"),
        field_name="candidate_artifact.bindings.passages.count",
    )

    if bound_benchmark_sha256 != benchmark_sha256:
        raise RerankerResultError(
            "Hybrid candidate benchmark binding does not match "
            "the supplied benchmark SHA-256."
        )

    if bound_passage_sha256 != passage_sha256:
        raise RerankerResultError(
            "Hybrid candidate passage binding does not match "
            "the supplied passage SHA-256."
        )

    if bound_passage_count != corpus_passage_count:
        raise RerankerResultError(
            "Hybrid candidate passage count does not match "
            "the reranker evaluation corpus count."
        )

    generation = _require_mapping(
        candidate_artifact.get("candidate_generation"),
        field_name="candidate_artifact.candidate_generation",
    )

    if generation.get("final_ranking") is not False:
        raise RerankerResultError(
            "Hybrid candidate artifact must not contain a final ranking."
        )

    return (
        candidate_result_id,
        candidate_schema_version,
        candidate_result_version,
    )


def build_reranker_result_artifact(
    evaluation: RerankerEvaluationResult,
    *,
    dataset: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    result_version: str,
    benchmark_sha256: str,
    passage_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    """Serialize reranker evaluation with immutable input bindings."""

    if not isinstance(evaluation, RerankerEvaluationResult):
        raise RerankerResultError(
            "evaluation must be a RerankerEvaluationResult."
        )

    dataset = _require_mapping(
        dataset,
        field_name="dataset",
    )
    candidate_artifact = _require_mapping(
        candidate_artifact,
        field_name="candidate_artifact",
    )
    result_version = _require_version(
        result_version,
        field_name="result_version",
    )
    benchmark_sha256 = _require_sha256(
        benchmark_sha256,
        field_name="benchmark_sha256",
    )
    passage_sha256 = _require_sha256(
        passage_sha256,
        field_name="passage_sha256",
    )
    candidate_sha256 = _require_sha256(
        candidate_sha256,
        field_name="candidate_sha256",
    )

    dataset_id = _require_nonempty_string(
        dataset.get("dataset_id"),
        field_name="dataset.dataset_id",
    )
    dataset_schema_version = _require_nonempty_string(
        dataset.get("schema_version"),
        field_name="dataset.schema_version",
    )
    dataset_version = _require_version(
        dataset.get("dataset_version"),
        field_name="dataset.dataset_version",
    )
    corpus_id = _require_nonempty_string(
        dataset.get("corpus_id"),
        field_name="dataset.corpus_id",
    )
    corpus_version = _require_version(
        dataset.get("corpus_version"),
        field_name="dataset.corpus_version",
    )
    passage_schema_version = _require_nonempty_string(
        dataset.get("passage_schema_version"),
        field_name="dataset.passage_schema_version",
    )
    declared_passage_sha256 = _require_sha256(
        dataset.get("passage_artifact_sha256"),
        field_name="dataset.passage_artifact_sha256",
    )

    if declared_passage_sha256 != passage_sha256:
        raise RerankerResultError(
            "Dataset passage artifact SHA-256 does not match "
            "the supplied passage SHA-256."
        )

    questions_by_id, dataset_abstention_count = _dataset_queries(
        dataset
    )
    candidate_results = _candidate_results_by_id(
        candidate_artifact
    )

    (
        candidate_result_id,
        candidate_schema_version,
        candidate_result_version,
    ) = _validate_candidate_bindings(
        candidate_artifact,
        benchmark_sha256=benchmark_sha256,
        passage_sha256=passage_sha256,
        corpus_passage_count=evaluation.corpus_passage_count,
    )

    if evaluation.abstention_query_count != dataset_abstention_count:
        raise RerankerResultError(
            "Evaluation abstention query count does not match "
            "the dataset."
        )

    if evaluation.answer_query_count != len(evaluation.query_results):
        raise RerankerResultError(
            "Evaluation answer query count does not match "
            "query results."
        )

    serialized_query_results: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()

    for result in evaluation.query_results:
        query_id = _require_nonempty_string(
            result.query_id,
            field_name="query_result.query_id",
        )

        if query_id in seen_query_ids:
            raise RerankerResultError(
                f"Duplicate evaluation query result: {query_id}"
            )

        question = questions_by_id.get(query_id)
        candidate_result = candidate_results.get(query_id)

        if question is None:
            raise RerankerResultError(
                f"{query_id}: evaluation query is absent "
                "from the dataset."
            )

        if candidate_result is None:
            raise RerankerResultError(
                f"{query_id}: evaluation query is absent "
                "from the hybrid candidate artifact."
            )

        candidate_question = _require_nonempty_string(
            candidate_result.get("question"),
            field_name=f"{query_id}.candidate_question",
        )

        if candidate_question != question:
            raise RerankerResultError(
                f"{query_id}: candidate question does not match "
                "the dataset question."
            )

        raw_candidate_ids = _require_sequence(
            candidate_result.get("candidate_passage_ids"),
            field_name=f"{query_id}.candidate_passage_ids",
        )
        candidate_ids = tuple(raw_candidate_ids)
        candidate_count = _require_positive_integer(
            candidate_result.get("candidate_count"),
            field_name=f"{query_id}.candidate_count",
        )

        if candidate_count != len(candidate_ids):
            raise RerankerResultError(
                f"{query_id}: candidate count does not match "
                "candidate passage IDs."
            )

        if result.candidate_count != candidate_count:
            raise RerankerResultError(
                f"{query_id}: reranker candidate count does not "
                "match the hybrid candidate artifact."
            )

        ranked_results: list[dict[str, Any]] = []

        for rank, candidate in enumerate(
            result.ranked_candidates,
            start=1,
        ):
            if candidate.reranker_rank != rank:
                raise RerankerResultError(
                    f"{query_id}: reranker rank does not match "
                    "serialized order."
                )

            passage_id = _require_nonempty_string(
                candidate.passage_id,
                field_name=f"{query_id}.ranked_passage_id",
            )
            score = _require_score(
                candidate.reranker_score,
                field_name=f"{query_id}.{passage_id}",
            )

            ranked_results.append(
                {
                    "rank": rank,
                    "passage_id": passage_id,
                    "score": score,
                    "accepted_order": _require_nonnegative_integer(
                        candidate.accepted_order,
                        field_name=(
                            f"{query_id}.{passage_id}.accepted_order"
                        ),
                    ),
                    "bm25_rank": _require_optional_rank(
                        candidate.bm25_rank,
                        field_name=(
                            f"{query_id}.{passage_id}.bm25_rank"
                        ),
                    ),
                    "dense_rank": _require_optional_rank(
                        candidate.dense_rank,
                        field_name=(
                            f"{query_id}.{passage_id}.dense_rank"
                        ),
                    ),
                }
            )

        ranked_ids = tuple(
            ranked_result["passage_id"]
            for ranked_result in ranked_results
        )

        if ranked_ids != result.ranked_passage_ids:
            raise RerankerResultError(
                f"{query_id}: ranked candidates do not match "
                "ranked_passage_ids."
            )

        if len(ranked_ids) != candidate_count:
            raise RerankerResultError(
                f"{query_id}: ranked result count does not match "
                "candidate count."
            )

        if len(set(ranked_ids)) != len(ranked_ids):
            raise RerankerResultError(
                f"{query_id}: ranked passage IDs must be unique."
            )

        if set(ranked_ids) != set(candidate_ids):
            raise RerankerResultError(
                f"{query_id}: reranked passage IDs do not match "
                "the hybrid candidate union."
            )

        serialized_query_results.append(
            {
                "query_id": query_id,
                "question": question,
                "candidate_count": candidate_count,
                "ranked_passage_ids": list(ranked_ids),
                "ranked_results": ranked_results,
                "metrics": {
                    "recall_at_1": _require_metric(
                        result.recall_at_1,
                        field_name=f"{query_id}.recall_at_1",
                    ),
                    "recall_at_3": _require_metric(
                        result.recall_at_3,
                        field_name=f"{query_id}.recall_at_3",
                    ),
                    "recall_at_5": _require_metric(
                        result.recall_at_5,
                        field_name=f"{query_id}.recall_at_5",
                    ),
                    "recall_at_10": _require_metric(
                        result.recall_at_10,
                        field_name=f"{query_id}.recall_at_10",
                    ),
                    "reciprocal_rank_at_10": _require_metric(
                        result.reciprocal_rank_at_10,
                        field_name=(
                            f"{query_id}.reciprocal_rank_at_10"
                        ),
                    ),
                    "direct_evidence_hit_at_10": (
                        result.direct_evidence_hit_at_10
                    ),
                    "ndcg_at_10": _require_metric(
                        result.ndcg_at_10,
                        field_name=f"{query_id}.ndcg_at_10",
                    ),
                },
            }
        )
        seen_query_ids.add(query_id)

    if set(candidate_results) != seen_query_ids:
        raise RerankerResultError(
            "Reranker query results must contain every hybrid "
            "candidate query exactly once."
        )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_id": RESULT_ID,
        "result_version": result_version,
        "bindings": {
            "benchmark": {
                "dataset_id": dataset_id,
                "schema_version": dataset_schema_version,
                "dataset_version": dataset_version,
                "sha256": benchmark_sha256,
            },
            "corpus": {
                "corpus_id": corpus_id,
                "corpus_version": corpus_version,
            },
            "passages": {
                "schema_version": passage_schema_version,
                "sha256": passage_sha256,
                "count": evaluation.corpus_passage_count,
            },
            "hybrid_candidates": {
                "result_id": candidate_result_id,
                "schema_version": candidate_schema_version,
                "result_version": candidate_result_version,
                "sha256": candidate_sha256,
            },
            "model": {
                "model_id": RERANKER_MODEL_CONTRACT.model_id,
                "revision": RERANKER_MODEL_CONTRACT.model_revision,
                "filename": RERANKER_MODEL_CONTRACT.model_filename,
                "size_bytes": RERANKER_MODEL_CONTRACT.model_size_bytes,
                "sha256": RERANKER_MODEL_CONTRACT.model_sha256,
                "license_id": RERANKER_MODEL_CONTRACT.license_id,
            },
            "tokenizer": {
                "source_model": TOKENIZER_CONTRACT.vocab_source_model,
                "source_revision": (
                    TOKENIZER_CONTRACT.vocab_source_revision
                ),
                "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
                "vocab_size": TOKENIZER_CONTRACT.vocab_size,
            },
        },
        "reranker": {
            "implementation": "onnx_ms_marco_minilm_l6_v2",
            "candidate_scope": "hybrid_candidate_union",
            "final_ranking": True,
            "runtime": {
                "library": RERANKER_MODEL_CONTRACT.runtime_library,
                "version": RERANKER_MODEL_CONTRACT.runtime_version,
                "array_library": (
                    RERANKER_MODEL_CONTRACT.array_library
                ),
                "array_version": (
                    RERANKER_MODEL_CONTRACT.array_version
                ),
                "execution_provider": (
                    RERANKER_MODEL_CONTRACT.execution_provider
                ),
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
                "execution_mode": "sequential",
                "graph_optimization": "ORT_ENABLE_ALL",
                "deterministic_compute": True,
            },
            "pair_encoding": {
                "template": [
                    "[CLS]",
                    "query",
                    "[SEP]",
                    "passage",
                    "[SEP]",
                ],
                "query_token_type_id": 0,
                "passage_token_type_id": 1,
                "query_instruction": (
                    RERANKER_MODEL_CONTRACT.query_instruction or None
                ),
                "passage_instruction": (
                    RERANKER_MODEL_CONTRACT.passage_instruction or None
                ),
                "maximum_sequence_length": (
                    RERANKER_MODEL_CONTRACT.max_sequence_length
                ),
                "overlength_behavior": (
                    RERANKER_MODEL_CONTRACT.truncation
                ),
            },
            "scoring": {
                "output_name": (
                    RERANKER_MODEL_CONTRACT.output_name
                ),
                "output_dimension": (
                    RERANKER_MODEL_CONTRACT.output_dimension
                ),
                "interpretation": (
                    RERANKER_MODEL_CONTRACT.output_interpretation
                ),
                "ranking_order": (
                    RERANKER_MODEL_CONTRACT.ranking_order
                ),
            },
            "tie_break_order": [
                "raw_logit_descending",
                "accepted_passage_order",
                "passage_id",
            ],
        },
        "evaluation": {
            "answer_query_count": evaluation.answer_query_count,
            "abstention_query_count": (
                evaluation.abstention_query_count
            ),
            "abstention_queries_in_ranking_metrics": False,
            "cutoffs": [1, 3, 5, 10],
            "aggregate_metrics": {
                "mean_candidate_count": (
                    _require_nonnegative_number(
                        evaluation.mean_candidate_count,
                        field_name="mean_candidate_count",
                    )
                ),
                "mean_recall_at_1": _require_metric(
                    evaluation.mean_recall_at_1,
                    field_name="mean_recall_at_1",
                ),
                "mean_recall_at_3": _require_metric(
                    evaluation.mean_recall_at_3,
                    field_name="mean_recall_at_3",
                ),
                "mean_recall_at_5": _require_metric(
                    evaluation.mean_recall_at_5,
                    field_name="mean_recall_at_5",
                ),
                "mean_recall_at_10": _require_metric(
                    evaluation.mean_recall_at_10,
                    field_name="mean_recall_at_10",
                ),
                "mrr_at_10": _require_metric(
                    evaluation.mrr_at_10,
                    field_name="mrr_at_10",
                ),
                "direct_evidence_hit_rate_at_10": (
                    _require_metric(
                        evaluation.direct_evidence_hit_rate_at_10,
                        field_name=(
                            "direct_evidence_hit_rate_at_10"
                        ),
                    )
                ),
                "mean_ndcg_at_10": _require_metric(
                    evaluation.mean_ndcg_at_10,
                    field_name="mean_ndcg_at_10",
                ),
            },
            "query_results": serialized_query_results,
        },
    }


def write_reranker_result_artifact(
    artifact: Mapping[str, Any],
    path: Any,
) -> None:
    """Write one formatted result atomically without overwriting."""

    artifact = _require_mapping(
        artifact,
        field_name="artifact",
    )

    try:
        write_json_atomically(
            path,
            artifact,
        )
    except RetrievalUnitError as error:
        raise RerankerResultError(str(error)) from error
