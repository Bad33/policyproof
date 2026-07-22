from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyproof.dense_model import DENSE_MODEL_CONTRACT
from policyproof.hybrid_candidate_evaluation import (
    HybridCandidateEvaluationResult,
    HybridCandidateQueryResult,
)
from policyproof.hybrid_candidate_results import (
    HybridCandidateResultError,
    build_hybrid_candidate_result_artifact,
    write_hybrid_candidate_result_artifact,
)
from policyproof.hybrid_candidates import HybridCandidate
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT

BENCHMARK_SHA256 = "a" * 64
PASSAGE_SHA256 = "b" * 64


def sample_dataset() -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": "policyproof-retrieval-evaluation",
        "dataset_version": "0.1.1",
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_SHA256,
        "query_count": 2,
        "queries": [
            {
                "query_id": "answer-001",
                "question": "What is required?",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-a",
                        "relevance_grade": 2,
                    },
                    {
                        "passage_id": "passage-c",
                        "relevance_grade": 1,
                    },
                ],
            },
            {
                "query_id": "abstain-001",
                "question": "What is the weather?",
                "expected_behavior": "abstain",
                "document_scope": [],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [],
            },
        ],
    }


def sample_evaluation() -> HybridCandidateEvaluationResult:
    query_result = HybridCandidateQueryResult(
        query_id="answer-001",
        candidates=(
            HybridCandidate(
                passage_id="passage-a",
                accepted_order=0,
                bm25_rank=1,
                dense_rank=None,
            ),
            HybridCandidate(
                passage_id="passage-b",
                accepted_order=1,
                bm25_rank=2,
                dense_rank=2,
            ),
            HybridCandidate(
                passage_id="passage-c",
                accepted_order=2,
                bm25_rank=None,
                dense_rank=1,
            ),
        ),
        candidate_passage_ids=(
            "passage-a",
            "passage-b",
            "passage-c",
        ),
        candidate_count=3,
        candidate_recall=1.0,
        direct_evidence_hit=True,
        retrieved_gold_passage_ids=(
            "passage-a",
            "passage-c",
        ),
        missed_gold_passage_ids=(),
        bm25_only_gold_passage_ids=("passage-a",),
        dense_only_gold_passage_ids=("passage-c",),
    )

    return HybridCandidateEvaluationResult(
        corpus_passage_count=40,
        answer_query_count=1,
        abstention_query_count=1,
        input_depth=20,
        dense_batch_size=32,
        query_results=(query_result,),
        mean_candidate_recall=1.0,
        direct_evidence_hit_rate=1.0,
        mean_candidate_count=3.0,
    )


def sample_artifact() -> dict:
    return build_hybrid_candidate_result_artifact(
        sample_evaluation(),
        dataset=sample_dataset(),
        result_version="0.1.0",
        benchmark_sha256=BENCHMARK_SHA256,
        passage_sha256=PASSAGE_SHA256,
    )


def test_result_artifact_binds_inputs_and_generator_contracts() -> None:
    artifact = sample_artifact()

    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == (
        "policyproof-hybrid-candidate-baseline"
    )
    assert artifact["result_version"] == "0.1.0"

    assert artifact["bindings"] == {
        "benchmark": {
            "dataset_id": "policyproof-retrieval-evaluation",
            "schema_version": "1.0",
            "dataset_version": "0.1.1",
            "sha256": BENCHMARK_SHA256,
        },
        "corpus": {
            "corpus_id": "policyproof-initial-corpus",
            "corpus_version": "0.1.0",
        },
        "passages": {
            "schema_version": "1.1",
            "sha256": PASSAGE_SHA256,
            "count": 40,
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
            "source_model": TOKENIZER_CONTRACT.vocab_source_model,
            "source_revision": (
                TOKENIZER_CONTRACT.vocab_source_revision
            ),
            "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
            "vocab_size": TOKENIZER_CONTRACT.vocab_size,
        },
    }

    assert artifact["candidate_generation"] == {
        "strategy": "deduplicated_union",
        "final_ranking": False,
        "input_depth_per_retriever": 20,
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
                    "k1": 1.2,
                    "b": 0.75,
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
                "batch_size": 32,
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
        },
    }


def test_result_artifact_serializes_coverage_not_ranking_metrics() -> None:
    evaluation = sample_artifact()["evaluation"]

    assert evaluation["answer_query_count"] == 1
    assert evaluation["abstention_query_count"] == 1
    assert evaluation["abstention_queries_in_candidate_metrics"] is False
    assert evaluation["aggregate_metrics"] == {
        "mean_candidate_recall": 1.0,
        "direct_evidence_hit_rate": 1.0,
        "mean_candidate_count": 3.0,
    }

    query_result = evaluation["query_results"][0]

    assert query_result == {
        "query_id": "answer-001",
        "question": "What is required?",
        "candidate_passage_ids": [
            "passage-a",
            "passage-b",
            "passage-c",
        ],
        "candidate_count": 3,
        "candidates": [
            {
                "passage_id": "passage-a",
                "accepted_order": 0,
                "bm25_rank": 1,
                "dense_rank": None,
            },
            {
                "passage_id": "passage-b",
                "accepted_order": 1,
                "bm25_rank": 2,
                "dense_rank": 2,
            },
            {
                "passage_id": "passage-c",
                "accepted_order": 2,
                "bm25_rank": None,
                "dense_rank": 1,
            },
        ],
        "coverage": {
            "candidate_recall": 1.0,
            "direct_evidence_hit": True,
            "retrieved_gold_passage_ids": [
                "passage-a",
                "passage-c",
            ],
            "missed_gold_passage_ids": [],
            "bm25_only_gold_passage_ids": [
                "passage-a",
            ],
            "dense_only_gold_passage_ids": [
                "passage-c",
            ],
        },
    }

    serialized = json.dumps(query_result)

    for forbidden in (
        "score",
        "reciprocal_rank",
        "ndcg",
        "document_scope",
        "evaluation_tags",
        "relevance_judgments",
    ):
        assert forbidden not in serialized


def test_result_artifact_rejects_passage_hash_mismatch() -> None:
    with pytest.raises(
        HybridCandidateResultError,
        match="passage artifact SHA-256",
    ):
        build_hybrid_candidate_result_artifact(
            sample_evaluation(),
            dataset=sample_dataset(),
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256="c" * 64,
        )


@pytest.mark.parametrize(
    "metric_value",
    [
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
    ],
)
def test_result_artifact_rejects_invalid_rate_metric(
    metric_value: float,
) -> None:
    evaluation = sample_evaluation()
    invalid_evaluation = HybridCandidateEvaluationResult(
        corpus_passage_count=evaluation.corpus_passage_count,
        answer_query_count=evaluation.answer_query_count,
        abstention_query_count=(
            evaluation.abstention_query_count
        ),
        input_depth=evaluation.input_depth,
        dense_batch_size=evaluation.dense_batch_size,
        query_results=evaluation.query_results,
        mean_candidate_recall=metric_value,
        direct_evidence_hit_rate=(
            evaluation.direct_evidence_hit_rate
        ),
        mean_candidate_count=evaluation.mean_candidate_count,
    )

    with pytest.raises(
        HybridCandidateResultError,
        match="mean_candidate_recall",
    ):
        build_hybrid_candidate_result_artifact(
            invalid_evaluation,
            dataset=sample_dataset(),
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256=PASSAGE_SHA256,
        )


def test_result_writer_is_formatted_atomic_and_non_overwriting(
    tmp_path: Path,
) -> None:
    artifact = sample_artifact()
    output_path = (
        tmp_path / "hybrid-candidate-baseline-v0.1.0.json"
    )

    write_hybrid_candidate_result_artifact(
        artifact,
        output_path,
    )

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == artifact
    assert output_path.read_text(encoding="utf-8").endswith("\n")

    original_bytes = output_path.read_bytes()

    with pytest.raises(
        HybridCandidateResultError,
        match="Output already exists",
    ):
        write_hybrid_candidate_result_artifact(
            artifact,
            output_path,
        )

    assert output_path.read_bytes() == original_bytes
