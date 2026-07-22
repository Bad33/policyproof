from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from policyproof.bm25_evaluation import evaluate_bm25
from policyproof.retrieval_evaluation import (
    validate_retrieval_evaluation_dataset,
)
from policyproof.retrieval_units import load_jsonl


def test_evaluation_ranks_full_corpus_and_excludes_abstention_queries() -> None:
    passages = [
        {
            "passage_id": "outside-scope",
            "document_id": "document-b",
            "retrieval_text": "alpha",
        },
        {
            "passage_id": "direct-evidence",
            "document_id": "document-a",
            "retrieval_text": "alpha",
        },
    ]
    dataset = {
        "queries": [
            {
                "query_id": "answer-001",
                "question": "alpha",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "direct-evidence",
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
        ]
    }

    result = evaluate_bm25(
        passages,
        dataset,
    )

    assert result.corpus_passage_count == 2
    assert result.answer_query_count == 1
    assert result.abstention_query_count == 1
    assert len(result.query_results) == 1

    query_result = result.query_results[0]

    assert query_result.query_id == "answer-001"
    assert query_result.ranked_passage_ids == (
        "outside-scope",
        "direct-evidence",
    )
    assert query_result.recall_at_1 == 0.0
    assert query_result.recall_at_3 == 1.0
    assert query_result.recall_at_5 == 1.0
    assert query_result.recall_at_10 == 1.0
    assert query_result.reciprocal_rank_at_10 == 0.5
    assert query_result.direct_evidence_hit_at_10
    assert query_result.ndcg_at_10 == pytest.approx(
        1 / math.log2(3)
    )

    assert result.mean_recall_at_1 == 0.0
    assert result.mean_recall_at_3 == 1.0
    assert result.mean_recall_at_5 == 1.0
    assert result.mean_recall_at_10 == 1.0
    assert result.mrr_at_10 == 0.5
    assert result.direct_evidence_hit_rate_at_10 == 1.0
    assert result.mean_ndcg_at_10 == pytest.approx(
        1 / math.log2(3)
    )



ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
PASSAGES_PATH = ROOT / "data/processed/retrieval-passages.jsonl"
BENCHMARK_PATH = (
    ROOT / "data/evaluation/retrieval-evaluation-v0.1.1.json"
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


def test_repository_bm25_baseline_matches_accepted_metrics() -> None:
    manifest = load_json(MANIFEST_PATH)
    benchmark = load_json(BENCHMARK_PATH)
    passages = load_jsonl(
        PASSAGES_PATH,
        record_name="retrieval passage",
    )

    assert sha256_file(BENCHMARK_PATH) == BENCHMARK_SHA256
    assert sha256_file(PASSAGES_PATH) == PASSAGE_SHA256

    validate_retrieval_evaluation_dataset(
        benchmark,
        manifest=manifest,
        passage_records=passages,
        passage_artifact_sha256=PASSAGE_SHA256,
    )

    result = evaluate_bm25(
        passages,
        benchmark,
    )

    assert result.corpus_passage_count == 707
    assert result.answer_query_count == 16
    assert result.abstention_query_count == 4

    assert result.mean_recall_at_1 == pytest.approx(
        0.3125,
        abs=1e-12,
    )
    assert result.mean_recall_at_3 == pytest.approx(
        0.6041666666666666,
        abs=1e-12,
    )
    assert result.mean_recall_at_5 == pytest.approx(
        0.6822916666666666,
        abs=1e-12,
    )
    assert result.mean_recall_at_10 == pytest.approx(
        0.7760416666666666,
        abs=1e-12,
    )
    assert result.mrr_at_10 == pytest.approx(
        0.7433035714285714,
        abs=1e-12,
    )
    assert result.direct_evidence_hit_rate_at_10 == pytest.approx(
        0.9375,
        abs=1e-12,
    )
    assert result.mean_ndcg_at_10 == pytest.approx(
        0.6555156356384406,
        abs=1e-12,
    )

    assert tuple(
        query_result.query_id
        for query_result in result.query_results
    ) == tuple(
        query["query_id"]
        for query in benchmark["queries"]
        if query["expected_behavior"] == "answer"
    )



def test_query_result_preserves_ranked_scores_and_order() -> None:
    passages = [
        {
            "passage_id": "passage-stronger",
            "document_id": "document-a",
            "retrieval_text": "alpha alpha",
        },
        {
            "passage_id": "passage-weaker",
            "document_id": "document-a",
            "retrieval_text": "alpha beta beta beta",
        },
        {
            "passage_id": "passage-zero",
            "document_id": "document-b",
            "retrieval_text": "gamma",
        },
    ]
    dataset = {
        "queries": [
            {
                "query_id": "answer-001",
                "question": "alpha",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["synthetic"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-stronger",
                        "relevance_grade": 2,
                        "rationale": "Synthetic direct evidence.",
                    }
                ],
            }
        ]
    }

    result = evaluate_bm25(
        passages,
        dataset,
    )
    query_result = result.query_results[0]

    assert tuple(
        hit.passage_id
        for hit in query_result.ranked_hits
    ) == query_result.ranked_passage_ids
    assert tuple(
        hit.accepted_order
        for hit in query_result.ranked_hits
    ) == (0, 1, 2)
    assert query_result.ranked_hits[0].score > (
        query_result.ranked_hits[1].score
    )
    assert query_result.ranked_hits[1].score > 0.0
    assert query_result.ranked_hits[2].score == 0.0
