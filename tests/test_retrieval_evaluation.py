from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from policyproof.retrieval_evaluation import (
    RetrievalEvaluationError,
    validate_retrieval_evaluation_dataset,
)

PASSAGE_ARTIFACT_SHA256 = "a" * 64


def manifest() -> dict:
    return {
        "schema_version": "1.0",
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "documents": [
            {"document_id": "document-a"},
            {"document_id": "document-b"},
        ],
    }


def passage_records() -> tuple[dict, ...]:
    return (
        {
            "schema_version": "1.1",
            "passage_id": "source-a:passage-001",
            "document_id": "document-a",
        },
        {
            "schema_version": "1.1",
            "passage_id": "source-a:passage-002",
            "document_id": "document-a",
        },
        {
            "schema_version": "1.1",
            "passage_id": "source-b:passage-001",
            "document_id": "document-b",
        },
    )


def valid_dataset() -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": "policyproof-retrieval-evaluation",
        "dataset_version": "0.1.0",
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "query_count": 2,
        "queries": [
            {
                "query_id": "rq-001",
                "question": "What does document A say about risk?",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": [
                    "single_document",
                    "graded_relevance",
                ],
                "relevance_judgments": [
                    {
                        "passage_id": "source-a:passage-001",
                        "relevance_grade": 2,
                        "rationale": (
                            "This passage directly answers the question."
                        ),
                    },
                    {
                        "passage_id": "source-a:passage-002",
                        "relevance_grade": 1,
                        "rationale": (
                            "This passage provides supporting context."
                        ),
                    },
                ],
            },
            {
                "query_id": "rq-002",
                "question": (
                    "Does the corpus determine whether my company complies?"
                ),
                "expected_behavior": "abstain",
                "document_scope": [
                    "document-a",
                    "document-b",
                ],
                "evaluation_tags": [
                    "organization_specific",
                    "legal_advice_boundary",
                ],
                "relevance_judgments": [],
            },
        ],
    }


def validate(dataset: dict) -> None:
    validate_retrieval_evaluation_dataset(
        dataset,
        manifest=manifest(),
        passage_records=passage_records(),
        passage_artifact_sha256=PASSAGE_ARTIFACT_SHA256,
    )


def test_valid_dataset_passes_without_mutation() -> None:
    dataset = valid_dataset()
    original = deepcopy(dataset)

    validate(dataset)

    assert dataset == original


def test_rejects_query_count_mismatch() -> None:
    dataset = valid_dataset()
    dataset["query_count"] = 3

    with pytest.raises(
        RetrievalEvaluationError,
        match="query_count",
    ):
        validate(dataset)


def test_rejects_duplicate_query_id() -> None:
    dataset = valid_dataset()
    dataset["queries"][1]["query_id"] = "rq-001"

    with pytest.raises(
        RetrievalEvaluationError,
        match="Duplicate query_id",
    ):
        validate(dataset)


def test_rejects_unknown_document_scope() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["document_scope"] = ["unknown-document"]

    with pytest.raises(
        RetrievalEvaluationError,
        match="unknown document",
    ):
        validate(dataset)


def test_rejects_unknown_passage_id() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["relevance_judgments"][0][
        "passage_id"
    ] = "unknown-passage"

    with pytest.raises(
        RetrievalEvaluationError,
        match="unknown passage",
    ):
        validate(dataset)


def test_rejects_judgment_outside_document_scope() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["relevance_judgments"][0][
        "passage_id"
    ] = "source-b:passage-001"

    with pytest.raises(
        RetrievalEvaluationError,
        match="outside document_scope",
    ):
        validate(dataset)


def test_answer_query_requires_relevance_judgment() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["relevance_judgments"] = []

    with pytest.raises(
        RetrievalEvaluationError,
        match="at least one relevance judgment",
    ):
        validate(dataset)


def test_answer_query_requires_grade_two_judgment() -> None:
    dataset = valid_dataset()

    for judgment in dataset["queries"][0]["relevance_judgments"]:
        judgment["relevance_grade"] = 1

    with pytest.raises(
        RetrievalEvaluationError,
        match="grade 2",
    ):
        validate(dataset)


def test_abstain_query_rejects_relevance_judgments() -> None:
    dataset = valid_dataset()
    dataset["queries"][1]["relevance_judgments"] = [
        {
            "passage_id": "source-a:passage-001",
            "relevance_grade": 2,
            "rationale": "This should not be present.",
        }
    ]

    with pytest.raises(
        RetrievalEvaluationError,
        match="must not contain relevance judgments",
    ):
        validate(dataset)


def test_rejects_duplicate_passage_judgment() -> None:
    dataset = valid_dataset()
    duplicate = deepcopy(
        dataset["queries"][0]["relevance_judgments"][0]
    )
    dataset["queries"][0]["relevance_judgments"].append(
        duplicate
    )

    with pytest.raises(
        RetrievalEvaluationError,
        match="Duplicate passage judgment",
    ):
        validate(dataset)


@pytest.mark.parametrize(
    "grade",
    (
        0,
        3,
        True,
        "2",
    ),
)
def test_rejects_invalid_relevance_grade(
    grade: object,
) -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["relevance_judgments"][0][
        "relevance_grade"
    ] = grade

    with pytest.raises(
        RetrievalEvaluationError,
        match="relevance_grade",
    ):
        validate(dataset)


def test_rejects_empty_judgment_rationale() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["relevance_judgments"][0][
        "rationale"
    ] = "   "

    with pytest.raises(
        RetrievalEvaluationError,
        match="rationale",
    ):
        validate(dataset)


def test_rejects_passage_artifact_hash_mismatch() -> None:
    dataset = valid_dataset()
    dataset["passage_artifact_sha256"] = "b" * 64

    with pytest.raises(
        RetrievalEvaluationError,
        match="passage artifact SHA-256",
    ):
        validate(dataset)


def test_rejects_passage_schema_version_mismatch() -> None:
    dataset = valid_dataset()
    dataset["passage_schema_version"] = "1.0"

    with pytest.raises(
        RetrievalEvaluationError,
        match="passage_schema_version",
    ):
        validate(dataset)


def test_rejects_corpus_version_mismatch() -> None:
    dataset = valid_dataset()
    dataset["corpus_version"] = "9.9.9"

    with pytest.raises(
        RetrievalEvaluationError,
        match="corpus_version",
    ):
        validate(dataset)

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_DATASET_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "retrieval-evaluation-v0.1.0.json"
)
REPOSITORY_MANIFEST_PATH = ROOT / "data" / "source_manifest.json"
REPOSITORY_PASSAGES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "retrieval-passages.jsonl"
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(64 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_jsonl(path: Path) -> tuple[dict, ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )


def test_repository_retrieval_evaluation_dataset_validates() -> None:
    dataset = json.loads(
        REPOSITORY_DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )
    repository_manifest = json.loads(
        REPOSITORY_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    repository_passages = load_jsonl(
        REPOSITORY_PASSAGES_PATH
    )
    passage_hash = sha256_file(
        REPOSITORY_PASSAGES_PATH
    )

    validate_retrieval_evaluation_dataset(
        dataset,
        manifest=repository_manifest,
        passage_records=repository_passages,
        passage_artifact_sha256=passage_hash,
    )

    answer_queries = [
        query
        for query in dataset["queries"]
        if query["expected_behavior"] == "answer"
    ]
    abstain_queries = [
        query
        for query in dataset["queries"]
        if query["expected_behavior"] == "abstain"
    ]

    assert dataset["query_count"] == 20
    assert len(answer_queries) == 16
    assert len(abstain_queries) == 4

    answer_document_counts = {
        document_id: sum(
            query["document_scope"] == [document_id]
            for query in answer_queries
        )
        for document_id in (
            "nist-ai-rmf-1.0",
            "nist-ai-600-1-genai-profile",
            "eu-ai-act-2024-1689",
            "openai-gpt-4o-system-card-2024-08-08",
        )
    }

    assert answer_document_counts == {
        "nist-ai-rmf-1.0": 4,
        "nist-ai-600-1-genai-profile": 4,
        "eu-ai-act-2024-1689": 4,
        "openai-gpt-4o-system-card-2024-08-08": 4,
    }

    grade_counts = Counter(
        judgment["relevance_grade"]
        for query in answer_queries
        for judgment in query["relevance_judgments"]
    )

    assert grade_counts == {
        1: 4,
        2: 31,
    }

    judgments_by_query = {
        query["query_id"]: {
            judgment["passage_id"]
            for judgment in query["relevance_judgments"]
        }
        for query in answer_queries
    }

    assert len(judgments_by_query["rmf-002"]) == 4
    assert len(judgments_by_query["eu-003"]) == 4

def test_rejects_unknown_top_level_field() -> None:
    dataset = valid_dataset()
    dataset["unexpected"] = "value"

    with pytest.raises(
        RetrievalEvaluationError,
        match="unknown dataset fields",
    ):
        validate(dataset)


def test_rejects_unknown_query_field() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["unexpected"] = "value"

    with pytest.raises(
        RetrievalEvaluationError,
        match="unknown query fields",
    ):
        validate(dataset)


def test_rejects_unknown_judgment_field() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["relevance_judgments"][0][
        "unexpected"
    ] = "value"

    with pytest.raises(
        RetrievalEvaluationError,
        match="unknown relevance judgment fields",
    ):
        validate(dataset)


def test_rejects_unexpected_dataset_id() -> None:
    dataset = valid_dataset()
    dataset["dataset_id"] = "different-dataset"

    with pytest.raises(
        RetrievalEvaluationError,
        match="dataset_id",
    ):
        validate(dataset)


def test_rejects_empty_query_list() -> None:
    dataset = valid_dataset()
    dataset["query_count"] = 0
    dataset["queries"] = []

    with pytest.raises(
        RetrievalEvaluationError,
        match="at least one query",
    ):
        validate(dataset)


def test_rejects_empty_evaluation_tags() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["evaluation_tags"] = []

    with pytest.raises(
        RetrievalEvaluationError,
        match="at least one evaluation tag",
    ):
        validate(dataset)


def test_rejects_duplicate_document_scope() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["document_scope"] = [
        "document-a",
        "document-a",
    ]

    with pytest.raises(
        RetrievalEvaluationError,
        match="duplicate document_scope",
    ):
        validate(dataset)


def test_rejects_duplicate_evaluation_tag() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["evaluation_tags"] = [
        "single_document",
        "single_document",
    ]

    with pytest.raises(
        RetrievalEvaluationError,
        match="duplicate evaluation tag",
    ):
        validate(dataset)


def test_rejects_invalid_expected_behavior() -> None:
    dataset = valid_dataset()
    dataset["queries"][0]["expected_behavior"] = "generate"

    with pytest.raises(
        RetrievalEvaluationError,
        match="expected_behavior",
    ):
        validate(dataset)


def test_rejects_corpus_id_mismatch() -> None:
    dataset = valid_dataset()
    dataset["corpus_id"] = "different-corpus"

    with pytest.raises(
        RetrievalEvaluationError,
        match="corpus_id",
    ):
        validate(dataset)
