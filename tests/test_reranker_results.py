from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from policyproof.reranker import RerankedCandidate
from policyproof.reranker_evaluation import (
    RerankerEvaluationResult,
    RerankerQueryResult,
)
from policyproof.reranker_model import RERANKER_MODEL_CONTRACT
from policyproof.reranker_results import (
    RerankerResultError,
    build_reranker_result_artifact,
    write_reranker_result_artifact,
)
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT

BENCHMARK_SHA256 = "a" * 64
PASSAGE_SHA256 = "b" * 64
CANDIDATE_SHA256 = "c" * 64


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
                "question": "What evidence applies?",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-a",
                        "relevance_grade": 2,
                    },
                    {
                        "passage_id": "passage-b",
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


def sample_candidate_artifact() -> dict:
    return {
        "schema_version": "1.0",
        "result_id": "policyproof-hybrid-candidate-baseline",
        "result_version": "0.1.0",
        "bindings": {
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
                "count": 3,
            },
        },
        "candidate_generation": {
            "strategy": "deduplicated_union",
            "final_ranking": False,
            "input_depth_per_retriever": 20,
            "output_order": [
                "accepted_passage_order",
                "passage_id",
            ],
        },
        "evaluation": {
            "answer_query_count": 1,
            "abstention_query_count": 1,
            "query_results": [
                {
                    "query_id": "answer-001",
                    "question": "What evidence applies?",
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
                            "bm25_rank": 2,
                            "dense_rank": 1,
                        },
                        {
                            "passage_id": "passage-b",
                            "accepted_order": 1,
                            "bm25_rank": None,
                            "dense_rank": 2,
                        },
                        {
                            "passage_id": "passage-c",
                            "accepted_order": 2,
                            "bm25_rank": 1,
                            "dense_rank": None,
                        },
                    ],
                }
            ],
        },
    }


def sample_evaluation() -> RerankerEvaluationResult:
    query_result = RerankerQueryResult(
        query_id="answer-001",
        ranked_candidates=(
            RerankedCandidate(
                passage_id="passage-b",
                accepted_order=1,
                bm25_rank=None,
                dense_rank=2,
                reranker_score=2.25,
                reranker_rank=1,
            ),
            RerankedCandidate(
                passage_id="passage-a",
                accepted_order=0,
                bm25_rank=2,
                dense_rank=1,
                reranker_score=0.75,
                reranker_rank=2,
            ),
            RerankedCandidate(
                passage_id="passage-c",
                accepted_order=2,
                bm25_rank=1,
                dense_rank=None,
                reranker_score=-1.5,
                reranker_rank=3,
            ),
        ),
        ranked_passage_ids=(
            "passage-b",
            "passage-a",
            "passage-c",
        ),
        candidate_count=3,
        recall_at_1=0.5,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        reciprocal_rank_at_10=1.0,
        direct_evidence_hit_at_10=True,
        ndcg_at_10=0.7967075809905066,
    )

    return RerankerEvaluationResult(
        corpus_passage_count=3,
        answer_query_count=1,
        abstention_query_count=1,
        query_results=(query_result,),
        mean_candidate_count=3.0,
        mean_recall_at_1=0.5,
        mean_recall_at_3=1.0,
        mean_recall_at_5=1.0,
        mean_recall_at_10=1.0,
        mrr_at_10=1.0,
        direct_evidence_hit_rate_at_10=1.0,
        mean_ndcg_at_10=0.7967075809905066,
    )


def sample_artifact() -> dict:
    return build_reranker_result_artifact(
        sample_evaluation(),
        dataset=sample_dataset(),
        candidate_artifact=sample_candidate_artifact(),
        result_version="0.1.0",
        benchmark_sha256=BENCHMARK_SHA256,
        passage_sha256=PASSAGE_SHA256,
        candidate_sha256=CANDIDATE_SHA256,
    )


def test_result_artifact_binds_all_immutable_inputs() -> None:
    artifact = sample_artifact()

    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == "policyproof-reranker-baseline"
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
            "count": 3,
        },
        "hybrid_candidates": {
            "result_id": "policyproof-hybrid-candidate-baseline",
            "schema_version": "1.0",
            "result_version": "0.1.0",
            "sha256": CANDIDATE_SHA256,
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
            "source_revision": TOKENIZER_CONTRACT.vocab_source_revision,
            "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
            "vocab_size": TOKENIZER_CONTRACT.vocab_size,
        },
    }


def test_result_artifact_records_exact_reranking_contract() -> None:
    artifact = sample_artifact()

    assert artifact["reranker"] == {
        "implementation": "onnx_ms_marco_minilm_l6_v2",
        "candidate_scope": "hybrid_candidate_union",
        "final_ranking": True,
        "runtime": {
            "library": RERANKER_MODEL_CONTRACT.runtime_library,
            "version": RERANKER_MODEL_CONTRACT.runtime_version,
            "array_library": RERANKER_MODEL_CONTRACT.array_library,
            "array_version": RERANKER_MODEL_CONTRACT.array_version,
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
            "query_instruction": None,
            "passage_instruction": None,
            "maximum_sequence_length": 512,
            "overlength_behavior": "reject",
        },
        "scoring": {
            "output_name": "logits",
            "output_dimension": 1,
            "interpretation": "raw_logit",
            "ranking_order": "descending",
        },
        "tie_break_order": [
            "raw_logit_descending",
            "accepted_passage_order",
            "passage_id",
        ],
    }


def test_result_artifact_serializes_rankings_metrics_and_provenance() -> None:
    evaluation = sample_artifact()["evaluation"]

    assert evaluation["answer_query_count"] == 1
    assert evaluation["abstention_query_count"] == 1
    assert evaluation["abstention_queries_in_ranking_metrics"] is False
    assert evaluation["cutoffs"] == [1, 3, 5, 10]
    assert evaluation["aggregate_metrics"] == {
        "mean_candidate_count": 3.0,
        "mean_recall_at_1": 0.5,
        "mean_recall_at_3": 1.0,
        "mean_recall_at_5": 1.0,
        "mean_recall_at_10": 1.0,
        "mrr_at_10": 1.0,
        "direct_evidence_hit_rate_at_10": 1.0,
        "mean_ndcg_at_10": 0.7967075809905066,
    }

    query_result = evaluation["query_results"][0]

    assert query_result == {
        "query_id": "answer-001",
        "question": "What evidence applies?",
        "candidate_count": 3,
        "ranked_passage_ids": [
            "passage-b",
            "passage-a",
            "passage-c",
        ],
        "ranked_results": [
            {
                "rank": 1,
                "passage_id": "passage-b",
                "score": 2.25,
                "accepted_order": 1,
                "bm25_rank": None,
                "dense_rank": 2,
            },
            {
                "rank": 2,
                "passage_id": "passage-a",
                "score": 0.75,
                "accepted_order": 0,
                "bm25_rank": 2,
                "dense_rank": 1,
            },
            {
                "rank": 3,
                "passage_id": "passage-c",
                "score": -1.5,
                "accepted_order": 2,
                "bm25_rank": 1,
                "dense_rank": None,
            },
        ],
        "metrics": {
            "recall_at_1": 0.5,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "reciprocal_rank_at_10": 1.0,
            "direct_evidence_hit_at_10": True,
            "ndcg_at_10": 0.7967075809905066,
        },
    }

    assert "document_scope" not in query_result
    assert "relevance_judgments" not in query_result
    assert "evaluation_tags" not in query_result


def test_result_artifact_rejects_passage_binding_mismatch() -> None:
    with pytest.raises(
        RerankerResultError,
        match="passage",
    ):
        build_reranker_result_artifact(
            sample_evaluation(),
            dataset=sample_dataset(),
            candidate_artifact=sample_candidate_artifact(),
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256="d" * 64,
            candidate_sha256=CANDIDATE_SHA256,
        )


def test_result_artifact_rejects_candidate_benchmark_binding_mismatch() -> None:
    candidates = sample_candidate_artifact()
    candidates["bindings"]["benchmark"]["sha256"] = "d" * 64

    with pytest.raises(
        RerankerResultError,
        match="benchmark",
    ):
        build_reranker_result_artifact(
            sample_evaluation(),
            dataset=sample_dataset(),
            candidate_artifact=candidates,
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256=PASSAGE_SHA256,
            candidate_sha256=CANDIDATE_SHA256,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_result_artifact_rejects_nonfinite_raw_logits(
    invalid_score: float,
) -> None:
    evaluation = sample_evaluation()
    query_result = evaluation.query_results[0]
    first = replace(
        query_result.ranked_candidates[0],
        reranker_score=invalid_score,
    )
    invalid_query = replace(
        query_result,
        ranked_candidates=(
            first,
            *query_result.ranked_candidates[1:],
        ),
    )
    invalid_evaluation = replace(
        evaluation,
        query_results=(invalid_query,),
    )

    with pytest.raises(
        RerankerResultError,
        match="score",
    ):
        build_reranker_result_artifact(
            invalid_evaluation,
            dataset=sample_dataset(),
            candidate_artifact=sample_candidate_artifact(),
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256=PASSAGE_SHA256,
            candidate_sha256=CANDIDATE_SHA256,
        )


def test_result_artifact_rejects_inconsistent_reranker_rank() -> None:
    evaluation = sample_evaluation()
    query_result = evaluation.query_results[0]
    first = replace(
        query_result.ranked_candidates[0],
        reranker_rank=2,
    )
    invalid_query = replace(
        query_result,
        ranked_candidates=(
            first,
            *query_result.ranked_candidates[1:],
        ),
    )

    with pytest.raises(
        RerankerResultError,
        match="rank",
    ):
        build_reranker_result_artifact(
            replace(
                evaluation,
                query_results=(invalid_query,),
            ),
            dataset=sample_dataset(),
            candidate_artifact=sample_candidate_artifact(),
            result_version="0.1.0",
            benchmark_sha256=BENCHMARK_SHA256,
            passage_sha256=PASSAGE_SHA256,
            candidate_sha256=CANDIDATE_SHA256,
        )


def test_result_writer_is_formatted_atomic_and_non_overwriting(
    tmp_path: Path,
) -> None:
    artifact = sample_artifact()
    output_path = tmp_path / "reranker-baseline-v0.1.0.json"

    write_reranker_result_artifact(
        artifact,
        output_path,
    )

    expected = (
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    assert output_path.read_text(encoding="utf-8") == expected

    with pytest.raises(
        RerankerResultError,
        match="already exists",
    ):
        write_reranker_result_artifact(
            artifact,
            output_path,
        )

    assert output_path.read_text(encoding="utf-8") == expected
