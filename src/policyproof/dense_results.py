"""Versioned result artifacts for deterministic dense retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from policyproof.dense_evaluation import DenseEvaluationResult
from policyproof.dense_model import DENSE_MODEL_CONTRACT
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT
from policyproof.retrieval_units import (
    RetrievalUnitError,
    write_json_atomically,
)

RESULT_SCHEMA_VERSION = "1.0"
RESULT_ID = "policyproof-dense-baseline"
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DenseResultError(ValueError):
    """Raised when a dense result artifact cannot be built or written."""


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DenseResultError(
            f"{field_name} must be a mapping."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise DenseResultError(
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

    if not VERSION_PATTERN.fullmatch(version):
        raise DenseResultError(
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

    if not SHA256_PATTERN.fullmatch(checksum):
        raise DenseResultError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return checksum


def _require_finite_metric(
    value: float,
    *,
    field_name: str,
) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise DenseResultError(
            f"{field_name} must be a finite float."
        )

    if value < 0 or value > 1:
        raise DenseResultError(
            f"{field_name} must be between zero and one."
        )

    return value


def _require_similarity_score(
    value: float,
    *,
    field_name: str,
) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise DenseResultError(
            f"{field_name} score must be a finite float."
        )

    if value < -1 or value > 1:
        raise DenseResultError(
            f"{field_name} score must be between -1 and 1."
        )

    return value


def build_dense_result_artifact(
    evaluation: DenseEvaluationResult,
    *,
    dataset: Mapping[str, Any],
    result_version: str,
    benchmark_sha256: str,
    passage_sha256: str,
) -> dict[str, Any]:
    """Serialize dense evaluation with immutable model and input bindings."""

    if not isinstance(evaluation, DenseEvaluationResult):
        raise DenseResultError(
            "evaluation must be a DenseEvaluationResult."
        )

    dataset = _require_mapping(
        dataset,
        field_name="dataset",
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
        raise DenseResultError(
            "Dataset passage artifact SHA-256 does not match "
            "the supplied passage artifact SHA-256."
        )

    queries = dataset.get("queries")

    if not isinstance(queries, list):
        raise DenseResultError(
            "dataset.queries must be a list."
        )

    questions_by_id: dict[str, str] = {}

    for query in queries:
        query = _require_mapping(
            query,
            field_name="dataset query",
        )
        query_id = _require_nonempty_string(
            query.get("query_id"),
            field_name="query.query_id",
        )
        question = _require_nonempty_string(
            query.get("question"),
            field_name=f"{query_id}.question",
        )

        if query_id in questions_by_id:
            raise DenseResultError(
                f"Duplicate dataset query_id: {query_id}"
            )

        questions_by_id[query_id] = question

    query_results: list[dict[str, Any]] = []

    for result in evaluation.query_results:
        if result.query_id not in questions_by_id:
            raise DenseResultError(
                f"Evaluation query {result.query_id!r} "
                "is absent from the dataset."
            )

        ranked_results: list[dict[str, Any]] = []

        for rank, hit in enumerate(
            result.ranked_hits,
            start=1,
        ):
            score = _require_similarity_score(
                hit.score,
                field_name=f"{result.query_id}.ranked_result",
            )

            if hit.accepted_order < 0:
                raise DenseResultError(
                    f"{result.query_id}: accepted_order must be "
                    "non-negative."
                )

            ranked_results.append(
                {
                    "rank": rank,
                    "passage_id": hit.passage_id,
                    "score": score,
                    "accepted_order": hit.accepted_order,
                }
            )

        if tuple(
            ranked_result["passage_id"]
            for ranked_result in ranked_results
        ) != result.ranked_passage_ids:
            raise DenseResultError(
                f"{result.query_id}: ranked hit IDs do not match "
                "ranked_passage_ids."
            )

        query_results.append(
            {
                "query_id": result.query_id,
                "question": questions_by_id[result.query_id],
                "ranked_passage_ids": list(
                    result.ranked_passage_ids
                ),
                "ranked_results": ranked_results,
                "metrics": {
                    "recall_at_1": _require_finite_metric(
                        result.recall_at_1,
                        field_name=(
                            f"{result.query_id}.recall_at_1"
                        ),
                    ),
                    "recall_at_3": _require_finite_metric(
                        result.recall_at_3,
                        field_name=(
                            f"{result.query_id}.recall_at_3"
                        ),
                    ),
                    "recall_at_5": _require_finite_metric(
                        result.recall_at_5,
                        field_name=(
                            f"{result.query_id}.recall_at_5"
                        ),
                    ),
                    "recall_at_10": _require_finite_metric(
                        result.recall_at_10,
                        field_name=(
                            f"{result.query_id}.recall_at_10"
                        ),
                    ),
                    "reciprocal_rank_at_10": (
                        _require_finite_metric(
                            result.reciprocal_rank_at_10,
                            field_name=(
                                f"{result.query_id}."
                                "reciprocal_rank_at_10"
                            ),
                        )
                    ),
                    "direct_evidence_hit_at_10": (
                        result.direct_evidence_hit_at_10
                    ),
                    "ndcg_at_10": _require_finite_metric(
                        result.ndcg_at_10,
                        field_name=(
                            f"{result.query_id}.ndcg_at_10"
                        ),
                    ),
                },
            }
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
            "model": {
                "model_id": DENSE_MODEL_CONTRACT.model_id,
                "revision": DENSE_MODEL_CONTRACT.model_revision,
                "filename": DENSE_MODEL_CONTRACT.model_filename,
                "size_bytes": DENSE_MODEL_CONTRACT.model_size_bytes,
                "sha256": DENSE_MODEL_CONTRACT.model_sha256,
                "license_id": DENSE_MODEL_CONTRACT.license_id,
            },
            "tokenizer": {
                "source_model": (
                    TOKENIZER_CONTRACT.vocab_source_model
                ),
                "source_revision": (
                    TOKENIZER_CONTRACT.vocab_source_revision
                ),
                "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
                "vocab_size": TOKENIZER_CONTRACT.vocab_size,
            },
        },
        "retriever": {
            "implementation": "onnx_bge_small_en_v1_5",
            "candidate_scope": "all_passages",
            "batch_size": evaluation.batch_size,
            "runtime": {
                "library": DENSE_MODEL_CONTRACT.runtime_library,
                "version": DENSE_MODEL_CONTRACT.runtime_version,
                "array_library": (
                    DENSE_MODEL_CONTRACT.array_library
                ),
                "array_version": (
                    DENSE_MODEL_CONTRACT.array_version
                ),
                "execution_provider": (
                    DENSE_MODEL_CONTRACT.execution_provider
                ),
            },
            "embedding": {
                "dimension": (
                    DENSE_MODEL_CONTRACT.embedding_dimension
                ),
                "pooling": DENSE_MODEL_CONTRACT.pooling,
                "normalization": (
                    DENSE_MODEL_CONTRACT.normalization
                ),
                "similarity": "normalized_dot_product",
                "query_instruction": (
                    DENSE_MODEL_CONTRACT.query_instruction
                ),
                "passage_instruction": None,
                "maximum_sequence_length": (
                    DENSE_MODEL_CONTRACT.max_sequence_length
                ),
                "truncation": False,
            },
            "tie_break_order": [
                "score_descending",
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
                "mean_recall_at_1": _require_finite_metric(
                    evaluation.mean_recall_at_1,
                    field_name="mean_recall_at_1",
                ),
                "mean_recall_at_3": _require_finite_metric(
                    evaluation.mean_recall_at_3,
                    field_name="mean_recall_at_3",
                ),
                "mean_recall_at_5": _require_finite_metric(
                    evaluation.mean_recall_at_5,
                    field_name="mean_recall_at_5",
                ),
                "mean_recall_at_10": _require_finite_metric(
                    evaluation.mean_recall_at_10,
                    field_name="mean_recall_at_10",
                ),
                "mrr_at_10": _require_finite_metric(
                    evaluation.mrr_at_10,
                    field_name="mrr_at_10",
                ),
                "direct_evidence_hit_rate_at_10": (
                    _require_finite_metric(
                        evaluation.direct_evidence_hit_rate_at_10,
                        field_name=(
                            "direct_evidence_hit_rate_at_10"
                        ),
                    )
                ),
                "mean_ndcg_at_10": _require_finite_metric(
                    evaluation.mean_ndcg_at_10,
                    field_name="mean_ndcg_at_10",
                ),
            },
            "query_results": query_results,
        },
    }


def write_dense_result_artifact(
    artifact: Mapping[str, Any],
    path: Any,
) -> None:
    """Write one dense result artifact atomically without overwriting."""

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
        raise DenseResultError(str(error)) from error
