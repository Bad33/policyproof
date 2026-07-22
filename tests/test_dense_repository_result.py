from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from policyproof.dense_model import DENSE_MODEL_CONTRACT
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "data/results/dense-baseline-v0.1.0.json"

RESULT_SHA256 = (
    "fd6477ba09c8d9a4a3d36eeeaa2455882a90fd26fb0a9f82f3363c16189f6c5d"
)
BENCHMARK_SHA256 = (
    "42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c"
)
PASSAGE_SHA256 = (
    "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_result() -> dict:
    return json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )


def test_published_dense_result_is_byte_stable() -> None:
    assert RESULT_PATH.is_file()
    assert sha256_file(RESULT_PATH) == RESULT_SHA256
    assert RESULT_PATH.read_bytes().endswith(b"\n")


def test_published_dense_result_binds_exact_inputs_and_model() -> None:
    artifact = load_result()

    assert list(artifact) == [
        "schema_version",
        "result_id",
        "result_version",
        "bindings",
        "retriever",
        "evaluation",
    ]
    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == "policyproof-dense-baseline"
    assert artifact["result_version"] == "0.1.0"

    bindings = artifact["bindings"]

    assert bindings["benchmark"] == {
        "dataset_id": "policyproof-retrieval-evaluation",
        "schema_version": "1.0",
        "dataset_version": "0.1.1",
        "sha256": BENCHMARK_SHA256,
    }
    assert bindings["corpus"] == {
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
    }
    assert bindings["passages"] == {
        "schema_version": "1.1",
        "sha256": PASSAGE_SHA256,
        "count": 707,
    }
    assert bindings["model"] == {
        "model_id": DENSE_MODEL_CONTRACT.model_id,
        "revision": DENSE_MODEL_CONTRACT.model_revision,
        "filename": DENSE_MODEL_CONTRACT.model_filename,
        "size_bytes": DENSE_MODEL_CONTRACT.model_size_bytes,
        "sha256": DENSE_MODEL_CONTRACT.model_sha256,
        "license_id": DENSE_MODEL_CONTRACT.license_id,
    }
    assert bindings["tokenizer"] == {
        "source_model": TOKENIZER_CONTRACT.vocab_source_model,
        "source_revision": (
            TOKENIZER_CONTRACT.vocab_source_revision
        ),
        "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
        "vocab_size": TOKENIZER_CONTRACT.vocab_size,
    }


def test_published_dense_result_records_exact_retrieval_contract() -> None:
    retriever = load_result()["retriever"]

    assert retriever == {
        "implementation": "onnx_bge_small_en_v1_5",
        "candidate_scope": "all_passages",
        "batch_size": 32,
        "runtime": {
            "library": "onnxruntime",
            "version": "1.27.0",
            "array_library": "numpy",
            "array_version": "2.5.1",
            "execution_provider": "CPUExecutionProvider",
        },
        "embedding": {
            "dimension": 384,
            "pooling": "cls",
            "normalization": "l2",
            "similarity": "normalized_dot_product",
            "query_instruction": (
                "Represent this sentence for searching relevant passages: "
            ),
            "passage_instruction": None,
            "maximum_sequence_length": 512,
            "truncation": False,
        },
        "tie_break_order": [
            "score_descending",
            "accepted_passage_order",
            "passage_id",
        ],
    }


def test_published_dense_metrics_match_accepted_baseline() -> None:
    evaluation = load_result()["evaluation"]

    assert evaluation["answer_query_count"] == 16
    assert evaluation["abstention_query_count"] == 4
    assert evaluation["abstention_queries_in_ranking_metrics"] is False
    assert evaluation["cutoffs"] == [1, 3, 5, 10]

    assert evaluation["aggregate_metrics"] == {
        "mean_recall_at_1": 0.4583333333333333,
        "mean_recall_at_3": 0.8802083333333334,
        "mean_recall_at_5": 0.8802083333333334,
        "mean_recall_at_10": 0.96875,
        "mrr_at_10": 0.90625,
        "direct_evidence_hit_rate_at_10": 1.0,
        "mean_ndcg_at_10": 0.8866302400292934,
    }


def test_published_dense_query_results_are_safe_and_complete() -> None:
    query_results = load_result()["evaluation"]["query_results"]

    assert len(query_results) == 16
    assert [result["query_id"] for result in query_results] == [
        "rmf-001",
        "rmf-002",
        "rmf-003",
        "rmf-004",
        "genai-001",
        "genai-002",
        "genai-003",
        "genai-004",
        "eu-001",
        "eu-002",
        "eu-003",
        "eu-004",
        "gpt4o-001",
        "gpt4o-002",
        "gpt4o-003",
        "gpt4o-004",
    ]

    for result in query_results:
        assert list(result) == [
            "query_id",
            "question",
            "ranked_passage_ids",
            "ranked_results",
            "metrics",
        ]
        assert "document_scope" not in result
        assert "evaluation_tags" not in result
        assert "relevance_judgments" not in result

        assert len(result["ranked_passage_ids"]) == 10
        assert len(result["ranked_results"]) == 10
        assert len(set(result["ranked_passage_ids"])) == 10

        assert [
            ranked["passage_id"]
            for ranked in result["ranked_results"]
        ] == result["ranked_passage_ids"]

        previous_score = math.inf

        for expected_rank, ranked in enumerate(
            result["ranked_results"],
            start=1,
        ):
            assert ranked["rank"] == expected_rank
            assert math.isfinite(ranked["score"])
            assert -1 <= ranked["score"] <= 1
            assert ranked["score"] <= previous_score
            assert ranked["accepted_order"] >= 0
            previous_score = ranked["score"]

        metrics = result["metrics"]

        for metric_name in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "reciprocal_rank_at_10",
            "ndcg_at_10",
        ):
            assert 0 <= metrics[metric_name] <= 1

        assert isinstance(
            metrics["direct_evidence_hit_at_10"],
            bool,
        )


def test_only_two_dense_queries_have_partial_recall_at_ten() -> None:
    query_results = load_result()["evaluation"]["query_results"]

    partial = {
        result["query_id"]: result["metrics"]["recall_at_10"]
        for result in query_results
        if result["metrics"]["recall_at_10"] < 1
    }

    assert partial == {
        "rmf-002": 0.75,
        "eu-003": 0.75,
    }
