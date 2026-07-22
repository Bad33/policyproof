from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from policyproof.dense_model import DENSE_MODEL_CONTRACT
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "data/results/hybrid-candidate-baseline-v0.1.0.json"
)

RESULT_SHA256 = (
    "94b98eda3795280ef31aa0dfaa49a44d912c23d77e50a75f33a4f2f26e1fe0d4"
)
BENCHMARK_SHA256 = (
    "42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c"
)
PASSAGE_SHA256 = (
    "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"
)

QUERY_IDS = [
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

EXPECTED_CANDIDATE_COUNTS = {
    "rmf-001": 28,
    "rmf-002": 34,
    "rmf-003": 35,
    "rmf-004": 30,
    "genai-001": 34,
    "genai-002": 34,
    "genai-003": 32,
    "genai-004": 34,
    "eu-001": 33,
    "eu-002": 34,
    "eu-003": 29,
    "eu-004": 31,
    "gpt4o-001": 26,
    "gpt4o-002": 27,
    "gpt4o-003": 33,
    "gpt4o-004": 27,
}


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


def test_published_hybrid_candidate_result_is_byte_stable() -> None:
    assert RESULT_PATH.is_file()
    assert RESULT_PATH.stat().st_size == 195077
    assert sha256_file(RESULT_PATH) == RESULT_SHA256
    assert RESULT_PATH.read_bytes().endswith(b"\n")


def test_published_result_binds_exact_inputs_and_dense_contract() -> None:
    artifact = load_result()

    assert list(artifact) == [
        "schema_version",
        "result_id",
        "result_version",
        "bindings",
        "candidate_generation",
        "evaluation",
    ]
    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == (
        "policyproof-hybrid-candidate-baseline"
    )
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
    assert bindings["dense_model"] == {
        "model_id": DENSE_MODEL_CONTRACT.model_id,
        "revision": DENSE_MODEL_CONTRACT.model_revision,
        "filename": DENSE_MODEL_CONTRACT.model_filename,
        "size_bytes": DENSE_MODEL_CONTRACT.model_size_bytes,
        "sha256": DENSE_MODEL_CONTRACT.model_sha256,
        "license_id": DENSE_MODEL_CONTRACT.license_id,
    }
    assert bindings["dense_tokenizer"] == {
        "source_model": TOKENIZER_CONTRACT.vocab_source_model,
        "source_revision": (
            TOKENIZER_CONTRACT.vocab_source_revision
        ),
        "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
        "vocab_size": TOKENIZER_CONTRACT.vocab_size,
    }


def test_published_result_records_candidate_union_not_final_ranking() -> None:
    generation = load_result()["candidate_generation"]

    assert generation["strategy"] == "deduplicated_union"
    assert generation["final_ranking"] is False
    assert generation["input_depth_per_retriever"] == 20
    assert generation["output_order"] == [
        "accepted_passage_order",
        "passage_id",
    ]

    bm25 = generation["source_retrievers"]["bm25"]
    dense = generation["source_retrievers"]["dense"]

    assert bm25["candidate_scope"] == "all_passages"
    assert bm25["parameters"] == {
        "k1": 1.2,
        "b": 0.75,
        "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
    }
    assert dense["candidate_scope"] == "all_passages"
    assert dense["batch_size"] == 32
    assert dense["runtime"]["execution_provider"] == (
        "CPUExecutionProvider"
    )
    assert dense["embedding"]["similarity"] == (
        "normalized_dot_product"
    )
    assert dense["embedding"]["truncation"] is False


def test_published_candidate_coverage_matches_accepted_measurement() -> None:
    evaluation = load_result()["evaluation"]

    assert evaluation["answer_query_count"] == 16
    assert evaluation["abstention_query_count"] == 4
    assert evaluation[
        "abstention_queries_in_candidate_metrics"
    ] is False
    assert evaluation["aggregate_metrics"] == {
        "mean_candidate_recall": 1.0,
        "direct_evidence_hit_rate": 1.0,
        "mean_candidate_count": 31.3125,
    }

    query_results = evaluation["query_results"]

    assert [result["query_id"] for result in query_results] == (
        QUERY_IDS
    )
    assert {
        result["query_id"]: result["candidate_count"]
        for result in query_results
    } == EXPECTED_CANDIDATE_COUNTS


def test_every_query_has_complete_reviewed_evidence_coverage() -> None:
    query_results = load_result()["evaluation"]["query_results"]

    for result in query_results:
        coverage = result["coverage"]

        assert coverage["candidate_recall"] == 1.0
        assert coverage["direct_evidence_hit"] is True
        assert coverage["missed_gold_passage_ids"] == []
        assert coverage["retrieved_gold_passage_ids"]


def test_candidate_records_preserve_source_rank_provenance() -> None:
    query_results = load_result()["evaluation"]["query_results"]

    for result in query_results:
        candidates = result["candidates"]

        assert result["candidate_count"] == len(candidates)
        assert result["candidate_passage_ids"] == [
            candidate["passage_id"]
            for candidate in candidates
        ]
        assert len(set(result["candidate_passage_ids"])) == len(
            candidates
        )

        previous_order: tuple[int, str] | None = None

        for candidate in candidates:
            assert list(candidate) == [
                "passage_id",
                "accepted_order",
                "bm25_rank",
                "dense_rank",
            ]

            current_order = (
                candidate["accepted_order"],
                candidate["passage_id"],
            )

            if previous_order is not None:
                assert current_order > previous_order

            previous_order = current_order

            bm25_rank = candidate["bm25_rank"]
            dense_rank = candidate["dense_rank"]

            assert bm25_rank is not None or dense_rank is not None

            if bm25_rank is not None:
                assert 1 <= bm25_rank <= 20

            if dense_rank is not None:
                assert 1 <= dense_rank <= 20


def test_candidate_result_contains_no_ranking_or_gold_scope_leakage() -> None:
    serialized = RESULT_PATH.read_text(encoding="utf-8")

    for forbidden in (
        '"document_scope"',
        '"evaluation_tags"',
        '"relevance_judgments"',
        '"fused_score"',
        '"final_rank"',
        '"reciprocal_rank"',
        '"ndcg"',
    ):
        assert forbidden not in serialized

    artifact = load_result()

    for query_result in artifact["evaluation"]["query_results"]:
        assert list(query_result) == [
            "query_id",
            "question",
            "candidate_passage_ids",
            "candidate_count",
            "candidates",
            "coverage",
        ]

        for metric_name in (
            "candidate_recall",
            "direct_evidence_hit",
        ):
            value = query_result["coverage"][metric_name]

            if isinstance(value, float):
                assert math.isfinite(value)
