from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyproof.dense import DenseHit
from policyproof.dense_evaluation import (
    DenseEvaluationResult,
    DenseQueryResult,
)
from policyproof.dense_model import DENSE_MODEL_CONTRACT
from policyproof.dense_results import (
    DenseResultError,
    build_dense_result_artifact,
    write_dense_result_artifact,
)
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
                "question": "What governs AI risk?",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-a",
                        "relevance_grade": 2,
                    }
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


def sample_evaluation() -> DenseEvaluationResult:
    query_result = DenseQueryResult(
        query_id="answer-001",
        ranked_hits=(
            DenseHit(
                passage_id="passage-a",
                score=0.75,
                accepted_order=0,
            ),
            DenseHit(
                passage_id="passage-b",
                score=-0.25,
                accepted_order=1,
            ),
        ),
        ranked_passage_ids=(
            "passage-a",
            "passage-b",
        ),
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        reciprocal_rank_at_10=1.0,
        direct_evidence_hit_at_10=True,
        ndcg_at_10=1.0,
    )

    return DenseEvaluationResult(
        corpus_passage_count=2,
        answer_query_count=1,
        abstention_query_count=1,
        query_results=(query_result,),
        batch_size=32,
        mean_recall_at_1=1.0,
        mean_recall_at_3=1.0,
        mean_recall_at_5=1.0,
        mean_recall_at_10=1.0,
        mrr_at_10=1.0,
        direct_evidence_hit_rate_at_10=1.0,
        mean_ndcg_at_10=1.0,
    )


def sample_artifact() -> dict:
    return build_dense_result_artifact(
        sample_evaluation(),
        dataset=sample_dataset(),
        result_version="0.1.0",
        benchmark_sha256=BENCHMARK_SHA256,
        passage_sha256=PASSAGE_SHA256,
    )


def test_result_artifact_binds_inputs_model_and_retrieval_contract() -> None:
    artifact = sample_artifact()

    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == "policyproof-dense-baseline"
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
            "count": 2,
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
            "source_model": TOKENIZER_CONTRACT.vocab_source_model,
            "source_revision": (
                TOKENIZER_CONTRACT.vocab_source_revision
            ),
            "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
            "vocab_size": TOKENIZER_CONTRACT.vocab_size,
        },
    }

    assert artifact["retriever"] == {
        "implementation": "onnx_bge_small_en_v1_5",
        "candidate_scope": "all_passages",
        "batch_size": 32,
        "runtime": {
            "library": DENSE_MODEL_CONTRACT.runtime_library,
            "version": DENSE_MODEL_CONTRACT.runtime_version,
            "array_library": DENSE_MODEL_CONTRACT.array_library,
            "array_version": DENSE_MODEL_CONTRACT.array_version,
            "execution_provider": (
                DENSE_MODEL_CONTRACT.execution_provider
            ),
        },
        "embedding": {
            "dimension": DENSE_MODEL_CONTRACT.embedding_dimension,
            "pooling": DENSE_MODEL_CONTRACT.pooling,
            "normalization": DENSE_MODEL_CONTRACT.normalization,
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
    }


def test_result_artifact_serializes_rankings_and_metrics() -> None:
    artifact = sample_artifact()
    evaluation = artifact["evaluation"]

    assert evaluation["answer_query_count"] == 1
    assert evaluation["abstention_query_count"] == 1
    assert evaluation["abstention_queries_in_ranking_metrics"] is False
    assert evaluation["cutoffs"] == [1, 3, 5, 10]
    assert evaluation["aggregate_metrics"] == {
        "mean_recall_at_1": 1.0,
        "mean_recall_at_3": 1.0,
        "mean_recall_at_5": 1.0,
        "mean_recall_at_10": 1.0,
        "mrr_at_10": 1.0,
        "direct_evidence_hit_rate_at_10": 1.0,
        "mean_ndcg_at_10": 1.0,
    }

    query_result = evaluation["query_results"][0]

    assert query_result == {
        "query_id": "answer-001",
        "question": "What governs AI risk?",
        "ranked_passage_ids": [
            "passage-a",
            "passage-b",
        ],
        "ranked_results": [
            {
                "rank": 1,
                "passage_id": "passage-a",
                "score": 0.75,
                "accepted_order": 0,
            },
            {
                "rank": 2,
                "passage_id": "passage-b",
                "score": -0.25,
                "accepted_order": 1,
            },
        ],
        "metrics": {
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "reciprocal_rank_at_10": 1.0,
            "direct_evidence_hit_at_10": True,
            "ndcg_at_10": 1.0,
        },
    }

    assert "document_scope" not in query_result
    assert "evaluation_tags" not in query_result
    assert "relevance_judgments" not in query_result


def test_result_artifact_rejects_passage_hash_mismatch() -> None:
    with pytest.raises(
        DenseResultError,
        match="passage artifact SHA-256",
    ):
        build_dense_result_artifact(
            sample_evaluation(),
            dataset=sample_dataset(),
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256="c" * 64,
        )


@pytest.mark.parametrize(
    "score",
    [
        float("nan"),
        float("inf"),
        -1.01,
        1.01,
    ],
)
def test_result_artifact_rejects_invalid_similarity_score(
    score: float,
) -> None:
    evaluation = sample_evaluation()
    invalid_query_result = DenseQueryResult(
        query_id=evaluation.query_results[0].query_id,
        ranked_hits=(
            DenseHit(
                passage_id="passage-a",
                score=score,
                accepted_order=0,
            ),
        ),
        ranked_passage_ids=("passage-a",),
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        reciprocal_rank_at_10=1.0,
        direct_evidence_hit_at_10=True,
        ndcg_at_10=1.0,
    )
    invalid_evaluation = DenseEvaluationResult(
        corpus_passage_count=1,
        answer_query_count=1,
        abstention_query_count=1,
        query_results=(invalid_query_result,),
        batch_size=32,
        mean_recall_at_1=1.0,
        mean_recall_at_3=1.0,
        mean_recall_at_5=1.0,
        mean_recall_at_10=1.0,
        mrr_at_10=1.0,
        direct_evidence_hit_rate_at_10=1.0,
        mean_ndcg_at_10=1.0,
    )

    with pytest.raises(
        DenseResultError,
        match="score",
    ):
        build_dense_result_artifact(
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
    output_path = tmp_path / "dense-baseline-v0.1.0.json"

    write_dense_result_artifact(
        artifact,
        output_path,
    )

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == artifact
    assert output_path.read_text(encoding="utf-8").endswith("\n")

    original_bytes = output_path.read_bytes()

    with pytest.raises(
        DenseResultError,
        match="Output already exists",
    ):
        write_dense_result_artifact(
            artifact,
            output_path,
        )

    assert output_path.read_bytes() == original_bytes
