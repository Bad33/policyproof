from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from policyproof.reranker_model import RERANKER_MODEL_CONTRACT
from policyproof.retrieval_tokenizer import TOKENIZER_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT / "data/results/reranker-baseline-v0.1.0.json"
)
CANDIDATE_PATH = (
    ROOT
    / "data/results/hybrid-candidate-baseline-v0.1.0.json"
)

RESULT_SIZE_BYTES = 222_296
RESULT_SHA256 = (
    "3c7e49f121422d3200822cdb328b349de28a2de3fcbd510f11ce43dc56611d8e"
)
BENCHMARK_SHA256 = (
    "42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c"
)
PASSAGE_SHA256 = (
    "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"
)
CANDIDATE_SHA256 = (
    "94b98eda3795280ef31aa0dfaa49a44d912c23d77e50a75f33a4f2f26e1fe0d4"
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


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")

    return value


def load_result() -> dict:
    return load_json(RESULT_PATH)


def test_published_reranker_result_is_byte_stable() -> None:
    assert RESULT_PATH.is_file()
    assert RESULT_PATH.stat().st_size == RESULT_SIZE_BYTES
    assert sha256_file(RESULT_PATH) == RESULT_SHA256
    assert RESULT_PATH.read_bytes().endswith(b"\n")


def test_published_reranker_result_binds_exact_inputs() -> None:
    artifact = load_result()

    assert list(artifact) == [
        "schema_version",
        "result_id",
        "result_version",
        "bindings",
        "reranker",
        "evaluation",
    ]
    assert artifact["schema_version"] == "1.0"
    assert artifact["result_id"] == (
        "policyproof-reranker-baseline"
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
    assert bindings["hybrid_candidates"] == {
        "result_id": "policyproof-hybrid-candidate-baseline",
        "schema_version": "1.0",
        "result_version": "0.1.0",
        "sha256": CANDIDATE_SHA256,
    }
    assert bindings["model"] == {
        "model_id": RERANKER_MODEL_CONTRACT.model_id,
        "revision": RERANKER_MODEL_CONTRACT.model_revision,
        "filename": RERANKER_MODEL_CONTRACT.model_filename,
        "size_bytes": RERANKER_MODEL_CONTRACT.model_size_bytes,
        "sha256": RERANKER_MODEL_CONTRACT.model_sha256,
        "license_id": RERANKER_MODEL_CONTRACT.license_id,
    }
    assert bindings["tokenizer"] == {
        "source_model": TOKENIZER_CONTRACT.vocab_source_model,
        "source_revision": (
            TOKENIZER_CONTRACT.vocab_source_revision
        ),
        "vocab_sha256": TOKENIZER_CONTRACT.vocab_sha256,
        "vocab_size": TOKENIZER_CONTRACT.vocab_size,
    }


def test_published_result_records_exact_reranker_contract() -> None:
    assert load_result()["reranker"] == {
        "implementation": "onnx_ms_marco_minilm_l6_v2",
        "candidate_scope": "hybrid_candidate_union",
        "final_ranking": True,
        "runtime": {
            "library": "onnxruntime",
            "version": "1.27.0",
            "array_library": "numpy",
            "array_version": "2.5.1",
            "execution_provider": "CPUExecutionProvider",
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


def test_published_reranker_metrics_match_accepted_baseline() -> None:
    evaluation = load_result()["evaluation"]

    assert evaluation["answer_query_count"] == 16
    assert evaluation["abstention_query_count"] == 4
    assert evaluation[
        "abstention_queries_in_ranking_metrics"
    ] is False
    assert evaluation["cutoffs"] == [1, 3, 5, 10]
    assert evaluation["aggregate_metrics"] == {
        "mean_candidate_count": 31.3125,
        "mean_recall_at_1": 0.4375,
        "mean_recall_at_3": 0.65625,
        "mean_recall_at_5": 0.828125,
        "mean_recall_at_10": 0.9270833333333334,
        "mrr_at_10": 0.825,
        "direct_evidence_hit_rate_at_10": 1.0,
        "mean_ndcg_at_10": 0.7892557931490553,
    }


def test_published_query_results_are_safe_and_complete() -> None:
    query_results = load_result()["evaluation"]["query_results"]

    assert [result["query_id"] for result in query_results] == (
        QUERY_IDS
    )
    assert {
        result["query_id"]: result["candidate_count"]
        for result in query_results
    } == EXPECTED_CANDIDATE_COUNTS

    for result in query_results:
        assert list(result) == [
            "query_id",
            "question",
            "candidate_count",
            "ranked_passage_ids",
            "ranked_results",
            "metrics",
        ]
        assert "document_scope" not in result
        assert "evaluation_tags" not in result
        assert "relevance_judgments" not in result

        ranked_ids = result["ranked_passage_ids"]
        ranked_results = result["ranked_results"]

        assert result["candidate_count"] == len(ranked_ids)
        assert result["candidate_count"] == len(ranked_results)
        assert len(set(ranked_ids)) == len(ranked_ids)
        assert ranked_ids == [
            ranked["passage_id"]
            for ranked in ranked_results
        ]

        previous_key: tuple[float, int, str] | None = None

        for expected_rank, ranked in enumerate(
            ranked_results,
            start=1,
        ):
            assert list(ranked) == [
                "rank",
                "passage_id",
                "score",
                "accepted_order",
                "bm25_rank",
                "dense_rank",
            ]
            assert ranked["rank"] == expected_rank
            assert math.isfinite(ranked["score"])
            assert ranked["accepted_order"] >= 0

            bm25_rank = ranked["bm25_rank"]
            dense_rank = ranked["dense_rank"]

            assert bm25_rank is not None or dense_rank is not None

            if bm25_rank is not None:
                assert 1 <= bm25_rank <= 20

            if dense_rank is not None:
                assert 1 <= dense_rank <= 20

            key = (
                -ranked["score"],
                ranked["accepted_order"],
                ranked["passage_id"],
            )

            if previous_key is not None:
                assert previous_key <= key

            previous_key = key

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


def test_reranker_preserves_exact_hybrid_candidate_unions() -> None:
    result_queries = {
        result["query_id"]: result
        for result in load_result()["evaluation"]["query_results"]
    }
    candidate_queries = {
        result["query_id"]: result
        for result in load_json(CANDIDATE_PATH)[
            "evaluation"
        ]["query_results"]
    }

    assert set(result_queries) == set(candidate_queries)

    for query_id, result in result_queries.items():
        candidate = candidate_queries[query_id]

        assert set(result["ranked_passage_ids"]) == set(
            candidate["candidate_passage_ids"]
        )

        source_ranks = {
            record["passage_id"]: (
                record["accepted_order"],
                record["bm25_rank"],
                record["dense_rank"],
            )
            for record in candidate["candidates"]
        }

        assert {
            record["passage_id"]: (
                record["accepted_order"],
                record["bm25_rank"],
                record["dense_rank"],
            )
            for record in result["ranked_results"]
        } == source_ranks


def test_partial_recall_cases_remain_visible() -> None:
    partial = {
        result["query_id"]: result["metrics"]["recall_at_10"]
        for result in load_result()["evaluation"]["query_results"]
        if result["metrics"]["recall_at_10"] < 1
    }

    assert partial == {
        "rmf-002": 0.5,
        "rmf-004": 2 / 3,
        "genai-003": 2 / 3,
    }


def test_result_contains_no_gold_scope_or_model_binary() -> None:
    serialized = RESULT_PATH.read_text(encoding="utf-8")

    for forbidden in (
        '"document_scope"',
        '"evaluation_tags"',
        '"relevance_judgments"',
        '"model_bytes"',
        '"model_data"',
    ):
        assert forbidden not in serialized
