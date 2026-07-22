"""Versioned result artifacts for hybrid candidate-union coverage."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyproof.bm25 import BM25Parameters
from policyproof.dense_model import DENSE_MODEL_CONTRACT
from policyproof.hybrid_candidate_evaluation import (
    HybridCandidateEvaluationResult,
    HybridCandidateQueryResult,
)
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT

RESULT_SCHEMA_VERSION = "1.0"
RESULT_ID = "policyproof-hybrid-candidate-baseline"


class HybridCandidateResultError(ValueError):
    """Raised when a hybrid candidate result artifact is invalid."""


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HybridCandidateResultError(
            f"{field_name} must be a non-empty string."
        )

    return value


def _require_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HybridCandidateResultError(
            f"{field_name} must be a lowercase SHA-256 digest."
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
        raise HybridCandidateResultError(
            f"{field_name} must be a positive integer."
        )

    return value


def _require_rate(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise HybridCandidateResultError(
            f"{field_name} must be a finite number from 0 to 1."
        )

    converted = float(value)

    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise HybridCandidateResultError(
            f"{field_name} must be a finite number from 0 to 1."
        )

    return converted


def _require_nonnegative_finite_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise HybridCandidateResultError(
            f"{field_name} must be a finite non-negative number."
        )

    converted = float(value)

    if not math.isfinite(converted) or converted < 0:
        raise HybridCandidateResultError(
            f"{field_name} must be a finite non-negative number."
        )

    return converted


def _require_dataset(
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    if not isinstance(dataset, Mapping):
        raise HybridCandidateResultError(
            "dataset must be a mapping."
        )

    required_strings = (
        "schema_version",
        "dataset_id",
        "dataset_version",
        "corpus_id",
        "corpus_version",
        "passage_schema_version",
        "passage_artifact_sha256",
    )

    for field_name in required_strings:
        _require_nonempty_string(
            dataset.get(field_name),
            field_name=f"dataset.{field_name}",
        )

    _require_sha256(
        dataset.get("passage_artifact_sha256"),
        field_name="dataset.passage_artifact_sha256",
    )

    queries = dataset.get("queries")

    if not isinstance(queries, list):
        raise HybridCandidateResultError(
            "dataset.queries must be a list."
        )

    queries_by_id: dict[str, Mapping[str, Any]] = {}
    abstention_count = 0

    for position, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise HybridCandidateResultError(
                f"dataset query at position {position} "
                "must be a mapping."
            )

        query_id = _require_nonempty_string(
            query.get("query_id"),
            field_name=f"dataset.queries[{position}].query_id",
        )
        _require_nonempty_string(
            query.get("question"),
            field_name=f"{query_id}.question",
        )

        if query_id in queries_by_id:
            raise HybridCandidateResultError(
                f"Duplicate dataset query_id: {query_id}"
            )

        expected_behavior = query.get("expected_behavior")

        if expected_behavior not in {"answer", "abstain"}:
            raise HybridCandidateResultError(
                f"{query_id}.expected_behavior must be "
                "'answer' or 'abstain'."
            )

        if expected_behavior == "abstain":
            abstention_count += 1

        queries_by_id[query_id] = query

    return queries_by_id, abstention_count


def _serialize_query_result(
    result: HybridCandidateQueryResult,
    *,
    dataset_query: Mapping[str, Any],
) -> dict[str, Any]:
    question = _require_nonempty_string(
        dataset_query.get("question"),
        field_name=f"{result.query_id}.question",
    )

    if dataset_query.get("expected_behavior") != "answer":
        raise HybridCandidateResultError(
            f"{result.query_id}: candidate result must bind "
            "to an answerable query."
        )

    candidate_count = _require_positive_integer(
        result.candidate_count,
        field_name=f"{result.query_id}.candidate_count",
    )

    if candidate_count != len(result.candidates):
        raise HybridCandidateResultError(
            f"{result.query_id}: candidate_count does not match "
            "the candidate records."
        )

    if len(result.candidate_passage_ids) != candidate_count:
        raise HybridCandidateResultError(
            f"{result.query_id}: candidate_passage_ids length "
            "does not match candidate_count."
        )

    serialized_candidates: list[dict[str, Any]] = []
    serialized_candidate_ids: list[str] = []
    seen_passage_ids: set[str] = set()
    previous_order: tuple[int, str] | None = None

    for position, candidate in enumerate(result.candidates):
        passage_id = _require_nonempty_string(
            candidate.passage_id,
            field_name=(
                f"{result.query_id}.candidates[{position}].passage_id"
            ),
        )
        accepted_order = candidate.accepted_order

        if (
            not isinstance(accepted_order, int)
            or isinstance(accepted_order, bool)
            or accepted_order < 0
        ):
            raise HybridCandidateResultError(
                f"{result.query_id}: candidate accepted_order "
                "must be a non-negative integer."
            )

        if passage_id in seen_passage_ids:
            raise HybridCandidateResultError(
                f"{result.query_id}: duplicate candidate "
                f"passage_id {passage_id!r}."
            )

        current_order = (accepted_order, passage_id)

        if previous_order is not None and current_order <= previous_order:
            raise HybridCandidateResultError(
                f"{result.query_id}: candidates must be ordered by "
                "accepted passage order and passage ID."
            )

        for rank_name, rank_value in (
            ("bm25_rank", candidate.bm25_rank),
            ("dense_rank", candidate.dense_rank),
        ):
            if rank_value is not None and (
                not isinstance(rank_value, int)
                or isinstance(rank_value, bool)
                or rank_value < 1
            ):
                raise HybridCandidateResultError(
                    f"{result.query_id}: {rank_name} must be "
                    "a positive integer or null."
                )

        if candidate.bm25_rank is None and candidate.dense_rank is None:
            raise HybridCandidateResultError(
                f"{result.query_id}: each candidate must come from "
                "at least one source retriever."
            )

        seen_passage_ids.add(passage_id)
        serialized_candidate_ids.append(passage_id)
        previous_order = current_order
        serialized_candidates.append(
            {
                "passage_id": passage_id,
                "accepted_order": accepted_order,
                "bm25_rank": candidate.bm25_rank,
                "dense_rank": candidate.dense_rank,
            }
        )

    if tuple(serialized_candidate_ids) != result.candidate_passage_ids:
        raise HybridCandidateResultError(
            f"{result.query_id}: candidate_passage_ids do not "
            "match candidate records."
        )

    candidate_recall = _require_rate(
        result.candidate_recall,
        field_name=f"{result.query_id}.candidate_recall",
    )

    if not isinstance(result.direct_evidence_hit, bool):
        raise HybridCandidateResultError(
            f"{result.query_id}.direct_evidence_hit must be boolean."
        )

    coverage_id_fields = {
        "retrieved_gold_passage_ids": result.retrieved_gold_passage_ids,
        "missed_gold_passage_ids": result.missed_gold_passage_ids,
        "bm25_only_gold_passage_ids": (
            result.bm25_only_gold_passage_ids
        ),
        "dense_only_gold_passage_ids": (
            result.dense_only_gold_passage_ids
        ),
    }

    serialized_coverage_ids: dict[str, list[str]] = {}

    for field_name, passage_ids in coverage_id_fields.items():
        if not isinstance(passage_ids, tuple):
            raise HybridCandidateResultError(
                f"{result.query_id}.{field_name} must be a tuple."
            )

        validated_ids: list[str] = []
        seen_ids: set[str] = set()

        for position, passage_id in enumerate(passage_ids):
            validated_id = _require_nonempty_string(
                passage_id,
                field_name=(
                    f"{result.query_id}.{field_name}[{position}]"
                ),
            )

            if validated_id in seen_ids:
                raise HybridCandidateResultError(
                    f"{result.query_id}.{field_name} contains "
                    f"duplicate passage_id {validated_id!r}."
                )

            seen_ids.add(validated_id)
            validated_ids.append(validated_id)

        serialized_coverage_ids[field_name] = validated_ids

    candidate_id_set = set(serialized_candidate_ids)
    retrieved_id_set = set(
        serialized_coverage_ids["retrieved_gold_passage_ids"]
    )
    missed_id_set = set(
        serialized_coverage_ids["missed_gold_passage_ids"]
    )

    if not retrieved_id_set.issubset(candidate_id_set):
        raise HybridCandidateResultError(
            f"{result.query_id}: retrieved gold passages must be "
            "present in the candidate union."
        )

    if missed_id_set.intersection(candidate_id_set):
        raise HybridCandidateResultError(
            f"{result.query_id}: missed gold passages cannot be "
            "present in the candidate union."
        )

    if retrieved_id_set.intersection(missed_id_set):
        raise HybridCandidateResultError(
            f"{result.query_id}: retrieved and missed gold passages "
            "must be disjoint."
        )

    return {
        "query_id": result.query_id,
        "question": question,
        "candidate_passage_ids": serialized_candidate_ids,
        "candidate_count": candidate_count,
        "candidates": serialized_candidates,
        "coverage": {
            "candidate_recall": candidate_recall,
            "direct_evidence_hit": result.direct_evidence_hit,
            **serialized_coverage_ids,
        },
    }


def build_hybrid_candidate_result_artifact(
    evaluation: HybridCandidateEvaluationResult,
    *,
    dataset: Mapping[str, Any],
    result_version: str,
    benchmark_sha256: str,
    passage_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic coverage artifact for hybrid candidates."""

    if not isinstance(evaluation, HybridCandidateEvaluationResult):
        raise HybridCandidateResultError(
            "evaluation must be a HybridCandidateEvaluationResult."
        )

    validated_result_version = _require_nonempty_string(
        result_version,
        field_name="result_version",
    )
    validated_benchmark_sha256 = _require_sha256(
        benchmark_sha256,
        field_name="benchmark SHA-256",
    )
    validated_passage_sha256 = _require_sha256(
        passage_sha256,
        field_name="passage artifact SHA-256",
    )
    queries_by_id, dataset_abstention_count = _require_dataset(
        dataset
    )

    dataset_passage_sha256 = dataset["passage_artifact_sha256"]

    if dataset_passage_sha256 != validated_passage_sha256:
        raise HybridCandidateResultError(
            "passage artifact SHA-256 does not match the benchmark binding."
        )

    corpus_passage_count = _require_positive_integer(
        evaluation.corpus_passage_count,
        field_name="corpus_passage_count",
    )
    answer_query_count = _require_positive_integer(
        evaluation.answer_query_count,
        field_name="answer_query_count",
    )

    if (
        not isinstance(evaluation.abstention_query_count, int)
        or isinstance(evaluation.abstention_query_count, bool)
        or evaluation.abstention_query_count < 0
    ):
        raise HybridCandidateResultError(
            "abstention_query_count must be a non-negative integer."
        )

    if evaluation.abstention_query_count != dataset_abstention_count:
        raise HybridCandidateResultError(
            "abstention_query_count does not match the benchmark."
        )

    input_depth = _require_positive_integer(
        evaluation.input_depth,
        field_name="input_depth",
    )
    dense_batch_size = _require_positive_integer(
        evaluation.dense_batch_size,
        field_name="dense_batch_size",
    )

    if input_depth > corpus_passage_count:
        raise HybridCandidateResultError(
            "input_depth cannot exceed corpus_passage_count."
        )

    if answer_query_count != len(evaluation.query_results):
        raise HybridCandidateResultError(
            "answer_query_count does not match query_results."
        )

    serialized_query_results: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()

    for result in evaluation.query_results:
        query_id = _require_nonempty_string(
            result.query_id,
            field_name="query_result.query_id",
        )

        if query_id in seen_query_ids:
            raise HybridCandidateResultError(
                f"Duplicate query result: {query_id}"
            )

        dataset_query = queries_by_id.get(query_id)

        if dataset_query is None:
            raise HybridCandidateResultError(
                f"{query_id}: query result is not present "
                "in the benchmark."
            )

        seen_query_ids.add(query_id)
        serialized_query_results.append(
            _serialize_query_result(
                result,
                dataset_query=dataset_query,
            )
        )

    benchmark_answer_ids = {
        query_id
        for query_id, query in queries_by_id.items()
        if query.get("expected_behavior") == "answer"
    }

    if seen_query_ids != benchmark_answer_ids:
        raise HybridCandidateResultError(
            "query_results must contain every answerable benchmark "
            "query exactly once."
        )

    mean_candidate_recall = _require_rate(
        evaluation.mean_candidate_recall,
        field_name="mean_candidate_recall",
    )
    direct_evidence_hit_rate = _require_rate(
        evaluation.direct_evidence_hit_rate,
        field_name="direct_evidence_hit_rate",
    )
    mean_candidate_count = _require_nonnegative_finite_number(
        evaluation.mean_candidate_count,
        field_name="mean_candidate_count",
    )

    calculated_mean_recall = sum(
        result.candidate_recall
        for result in evaluation.query_results
    ) / answer_query_count
    calculated_direct_hit_rate = sum(
        int(result.direct_evidence_hit)
        for result in evaluation.query_results
    ) / answer_query_count
    calculated_mean_count = sum(
        result.candidate_count
        for result in evaluation.query_results
    ) / answer_query_count

    if mean_candidate_recall != calculated_mean_recall:
        raise HybridCandidateResultError(
            "mean_candidate_recall does not match query results."
        )

    if direct_evidence_hit_rate != calculated_direct_hit_rate:
        raise HybridCandidateResultError(
            "direct_evidence_hit_rate does not match query results."
        )

    if mean_candidate_count != calculated_mean_count:
        raise HybridCandidateResultError(
            "mean_candidate_count does not match query results."
        )

    bm25_parameters = BM25Parameters()

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_id": RESULT_ID,
        "result_version": validated_result_version,
        "bindings": {
            "benchmark": {
                "dataset_id": dataset["dataset_id"],
                "schema_version": dataset["schema_version"],
                "dataset_version": dataset["dataset_version"],
                "sha256": validated_benchmark_sha256,
            },
            "corpus": {
                "corpus_id": dataset["corpus_id"],
                "corpus_version": dataset["corpus_version"],
            },
            "passages": {
                "schema_version": dataset["passage_schema_version"],
                "sha256": validated_passage_sha256,
                "count": corpus_passage_count,
            },
            "dense_model": {
                "model_id": DENSE_MODEL_CONTRACT.model_id,
                "revision": DENSE_MODEL_CONTRACT.model_revision,
                "filename": DENSE_MODEL_CONTRACT.model_filename,
                "size_bytes": DENSE_MODEL_CONTRACT.model_size_bytes,
                "sha256": DENSE_MODEL_CONTRACT.model_sha256,
                "license_id": DENSE_MODEL_CONTRACT.license_id,
            },
            "dense_tokenizer": {
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
        "candidate_generation": {
            "strategy": "deduplicated_union",
            "final_ranking": False,
            "input_depth_per_retriever": input_depth,
            "output_order": [
                "accepted_passage_order",
                "passage_id",
            ],
            "source_retrievers": {
                "bm25": {
                    "implementation": "plain_python_bm25",
                    "candidate_scope": "all_passages",
                    "tokenizer": {
                        "normalization": "NFKC",
                        "lowercase": True,
                        "term_pattern": "[a-z0-9]+",
                        "query_term_frequency": "ignored",
                        "passage_term_frequency": "retained",
                    },
                    "parameters": {
                        "k1": bm25_parameters.k1,
                        "b": bm25_parameters.b,
                        "idf": (
                            "log(1 + (N - df + 0.5) / (df + 0.5))"
                        ),
                    },
                    "tie_break_order": [
                        "score_descending",
                        "accepted_passage_order",
                        "passage_id",
                    ],
                },
                "dense": {
                    "implementation": "onnx_bge_small_en_v1_5",
                    "candidate_scope": "all_passages",
                    "batch_size": dense_batch_size,
                    "runtime": {
                        "library": (
                            DENSE_MODEL_CONTRACT.runtime_library
                        ),
                        "version": (
                            DENSE_MODEL_CONTRACT.runtime_version
                        ),
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
            },
        },
        "evaluation": {
            "answer_query_count": answer_query_count,
            "abstention_query_count": (
                evaluation.abstention_query_count
            ),
            "abstention_queries_in_candidate_metrics": False,
            "aggregate_metrics": {
                "mean_candidate_recall": mean_candidate_recall,
                "direct_evidence_hit_rate": (
                    direct_evidence_hit_rate
                ),
                "mean_candidate_count": mean_candidate_count,
            },
            "query_results": serialized_query_results,
        },
    }


def write_hybrid_candidate_result_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically publish a formatted result without overwriting."""

    if not isinstance(artifact, Mapping):
        raise HybridCandidateResultError(
            "artifact must be a mapping."
        )

    validated_output_path = Path(output_path)

    if validated_output_path.exists():
        raise HybridCandidateResultError(
            f"Output already exists: {validated_output_path}"
        )

    validated_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    serialized = (
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    temporary_path: Path | None = None

    try:
        file_descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{validated_output_path.name}.",
            suffix=".tmp",
            dir=validated_output_path.parent,
        )
        temporary_path = Path(raw_temporary_path)

        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(
                temporary_path,
                validated_output_path,
            )
        except FileExistsError as error:
            raise HybridCandidateResultError(
                f"Output already exists: {validated_output_path}"
            ) from error

        temporary_path.unlink()
        temporary_path = None
    except HybridCandidateResultError:
        raise
    except OSError as error:
        raise HybridCandidateResultError(
            f"Unable to publish result artifact: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
