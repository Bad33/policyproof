from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyproof.bm25 import BM25Parameters
from policyproof.bm25_evaluation import evaluate_bm25
from policyproof.bm25_results import (
    BM25ResultError,
    build_bm25_result_artifact,
    write_bm25_result_artifact,
)

BENCHMARK_SHA256 = "a" * 64
PASSAGE_SHA256 = "b" * 64


def sample_passages() -> list[dict[str, str]]:
    return [
        {
            "passage_id": "passage-outside-scope",
            "document_id": "document-b",
            "retrieval_text": "alpha",
        },
        {
            "passage_id": "passage-direct",
            "document_id": "document-a",
            "retrieval_text": "alpha",
        },
    ]


def sample_dataset() -> dict[str, object]:
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
                "question": "alpha",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-direct",
                        "relevance_grade": 2,
                        "rationale": "Synthetic direct evidence.",
                    }
                ],
            },
            {
                "query_id": "abstain-001",
                "question": "unsupported request",
                "expected_behavior": "abstain",
                "document_scope": ["document-a", "document-b"],
                "evaluation_tags": ["abstention"],
                "relevance_judgments": [],
            },
        ],
    }


def sample_artifact() -> dict[str, object]:
    passages = sample_passages()
    dataset = sample_dataset()
    evaluation = evaluate_bm25(
        passages,
        dataset,
    )

    return build_bm25_result_artifact(
        evaluation,
        dataset=dataset,
        result_version="0.1.0",
        benchmark_sha256=BENCHMARK_SHA256,
        passage_sha256=PASSAGE_SHA256,
    )


def test_result_artifact_binds_inputs_and_retrieval_contract() -> None:
    artifact = sample_artifact()

    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == "policyproof-bm25-baseline"
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
    }

    assert artifact["retriever"] == {
        "implementation": "plain_python_bm25",
        "candidate_scope": "all_passages",
        "parameters": {
            "k1": 1.2,
            "b": 0.75,
        },
        "tokenizer": {
            "normalization": "NFKC",
            "lowercase": True,
            "term_pattern": "[a-z0-9]+",
            "punctuation_behavior": "term_boundary",
            "query_term_frequency": "ignored",
            "query_term_order": "first_seen",
        },
        "tie_break_order": [
            "score_descending",
            "accepted_passage_order",
            "passage_id",
        ],
    }


def test_result_artifact_persists_metrics_without_gold_scope_filtering() -> None:
    artifact = sample_artifact()
    evaluation = artifact["evaluation"]

    assert evaluation["answer_query_count"] == 1
    assert evaluation["abstention_query_count"] == 1
    assert not evaluation["abstention_queries_in_ranking_metrics"]
    assert evaluation["cutoffs"] == [1, 3, 5, 10]

    assert evaluation["aggregate_metrics"] == {
        "mean_recall_at_1": 0.0,
        "mean_recall_at_3": 1.0,
        "mean_recall_at_5": 1.0,
        "mean_recall_at_10": 1.0,
        "mrr_at_10": 0.5,
        "direct_evidence_hit_rate_at_10": 1.0,
        "mean_ndcg_at_10": pytest.approx(1 / 1.584962500721156),
    }

    query_result = evaluation["query_results"][0]

    assert query_result["query_id"] == "answer-001"
    assert query_result["question"] == "alpha"
    assert query_result["ranked_passage_ids"] == [
        "passage-outside-scope",
        "passage-direct",
    ]
    assert "document_scope" not in query_result
    assert "relevance_judgments" not in query_result


def test_result_writer_is_formatted_atomic_and_non_overwriting(
    tmp_path: Path,
) -> None:
    artifact = sample_artifact()
    output_path = tmp_path / "bm25-baseline-v0.1.0.json"

    write_bm25_result_artifact(
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
        BM25ResultError,
        match="already exists",
    ):
        write_bm25_result_artifact(
            artifact,
            output_path,
        )

    assert output_path.read_text(encoding="utf-8") == expected



def test_result_artifact_persists_actual_bm25_parameters() -> None:
    passages = sample_passages()
    dataset = sample_dataset()
    parameters = BM25Parameters(
        k1=1.6,
        b=0.4,
    )
    evaluation = evaluate_bm25(
        passages,
        dataset,
        parameters=parameters,
    )

    artifact = build_bm25_result_artifact(
        evaluation,
        dataset=dataset,
        result_version="0.1.0",
        benchmark_sha256=BENCHMARK_SHA256,
        passage_sha256=PASSAGE_SHA256,
    )

    assert artifact["retriever"]["parameters"] == {
        "k1": 1.6,
        "b": 0.4,
    }



def test_result_artifact_persists_ranked_scores_and_order() -> None:
    artifact = sample_artifact()
    query_result = artifact["evaluation"]["query_results"][0]

    assert query_result["ranked_results"] == [
        {
            "rank": 1,
            "passage_id": "passage-outside-scope",
            "score": pytest.approx(
                query_result["ranked_results"][0]["score"]
            ),
            "accepted_order": 0,
        },
        {
            "rank": 2,
            "passage_id": "passage-direct",
            "score": pytest.approx(
                query_result["ranked_results"][1]["score"]
            ),
            "accepted_order": 1,
        },
    ]

    assert query_result["ranked_results"][0]["score"] > 0.0
    assert query_result["ranked_results"][1]["score"] > 0.0
    assert (
        query_result["ranked_results"][0]["score"]
        == query_result["ranked_results"][1]["score"]
    )
