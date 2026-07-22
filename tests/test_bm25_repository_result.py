from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from policyproof.bm25_baseline import run_bm25_baseline
from policyproof.retrieval_units import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
PASSAGES_PATH = ROOT / "data/processed/retrieval-passages.jsonl"
BENCHMARK_PATH = (
    ROOT / "data/evaluation/retrieval-evaluation-v0.1.1.json"
)
RESULT_PATH = ROOT / "data/results/bm25-baseline-v0.1.0.json"

RESULT_SHA256 = (
    "5609b146b0901fc84851789d3b6c2799ec6aad0545e33b9c80afaa29c9d80003"
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")

    return value


def test_repository_bm25_result_retains_accepted_contract() -> None:
    artifact = load_json(RESULT_PATH)
    benchmark = load_json(BENCHMARK_PATH)
    passages = load_jsonl(
        PASSAGES_PATH,
        record_name="retrieval passage",
    )

    assert sha256_file(RESULT_PATH) == RESULT_SHA256
    assert sha256_file(BENCHMARK_PATH) == BENCHMARK_SHA256
    assert sha256_file(PASSAGES_PATH) == PASSAGE_SHA256

    assert list(artifact) == [
        "schema_version",
        "result_id",
        "result_version",
        "bindings",
        "retriever",
        "evaluation",
    ]
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
            "count": 707,
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

    evaluation = artifact["evaluation"]

    assert evaluation["answer_query_count"] == 16
    assert evaluation["abstention_query_count"] == 4
    assert not evaluation["abstention_queries_in_ranking_metrics"]
    assert evaluation["cutoffs"] == [1, 3, 5, 10]

    assert evaluation["aggregate_metrics"] == {
        "mean_recall_at_1": pytest.approx(
            0.3125,
            abs=1e-12,
        ),
        "mean_recall_at_3": pytest.approx(
            0.6041666666666666,
            abs=1e-12,
        ),
        "mean_recall_at_5": pytest.approx(
            0.6822916666666666,
            abs=1e-12,
        ),
        "mean_recall_at_10": pytest.approx(
            0.7760416666666666,
            abs=1e-12,
        ),
        "mrr_at_10": pytest.approx(
            0.7433035714285714,
            abs=1e-12,
        ),
        "direct_evidence_hit_rate_at_10": pytest.approx(
            0.9375,
            abs=1e-12,
        ),
        "mean_ndcg_at_10": pytest.approx(
            0.6555156356384406,
            abs=1e-12,
        ),
    }

    expected_query_ids = [
        query["query_id"]
        for query in benchmark["queries"]
        if query["expected_behavior"] == "answer"
    ]
    query_results = evaluation["query_results"]

    assert [
        query_result["query_id"]
        for query_result in query_results
    ] == expected_query_ids

    passage_order = {
        passage["passage_id"]: position
        for position, passage in enumerate(passages)
    }

    for query_result in query_results:
        ranked_results = query_result["ranked_results"]

        assert len(ranked_results) == 10
        assert query_result["ranked_passage_ids"] == [
            ranked_result["passage_id"]
            for ranked_result in ranked_results
        ]
        assert [
            ranked_result["rank"]
            for ranked_result in ranked_results
        ] == list(range(1, 11))
        assert len(
            {
                ranked_result["passage_id"]
                for ranked_result in ranked_results
            }
        ) == 10

        for ranked_result in ranked_results:
            assert ranked_result["accepted_order"] == (
                passage_order[ranked_result["passage_id"]]
            )
            assert ranked_result["score"] >= 0.0

        assert ranked_results == sorted(
            ranked_results,
            key=lambda ranked_result: (
                -ranked_result["score"],
                ranked_result["accepted_order"],
                ranked_result["passage_id"],
            ),
        )


def test_repository_bm25_result_regenerates_byte_identically(
    tmp_path: Path,
) -> None:
    regenerated_path = tmp_path / "bm25-baseline-v0.1.0.json"

    run_bm25_baseline(
        manifest_path=MANIFEST_PATH,
        passages_path=PASSAGES_PATH,
        benchmark_path=BENCHMARK_PATH,
        output_path=regenerated_path,
        result_version="0.1.0",
    )

    assert regenerated_path.read_bytes() == RESULT_PATH.read_bytes()
    assert sha256_file(regenerated_path) == RESULT_SHA256
