from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from policyproof.evidence_sufficiency_evaluation import (
    EVIDENCE_SUFFICIENCY_DATASET_ID,
    EvidenceSufficiencyEvaluationError,
    validate_evidence_sufficiency_dataset,
)

PASSAGE_SHA256 = "a" * 64
RETRIEVAL_DATASET_SHA256 = "b" * 64


def manifest() -> dict[str, Any]:
    return {
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
    }


def passages() -> list[dict[str, Any]]:
    return [
        {
            "passage_id": "passage-a",
            "document_id": "document-a",
        },
        {
            "passage_id": "passage-b",
            "document_id": "document-a",
        },
        {
            "passage_id": "passage-c",
            "document_id": "document-a",
        },
    ]


def retrieval_dataset() -> dict[str, Any]:
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
                "question": "What does the source say about risk?",
                "expected_behavior": "answer",
                "document_scope": ["document-a"],
                "evaluation_tags": ["single_document"],
                "relevance_judgments": [
                    {
                        "passage_id": "passage-a",
                        "relevance_grade": 2,
                        "rationale": "Direct evidence.",
                    },
                    {
                        "passage_id": "passage-b",
                        "relevance_grade": 1,
                        "rationale": "Useful supporting context.",
                    },
                ],
            },
            {
                "query_id": "abstain-001",
                "question": "Which current product should I buy?",
                "expected_behavior": "abstain",
                "document_scope": ["document-a"],
                "evaluation_tags": [
                    "abstention",
                    "outside_corpus",
                ],
                "relevance_judgments": [],
            },
        ],
    }


def dataset() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": EVIDENCE_SUFFICIENCY_DATASET_ID,
        "dataset_version": "0.1.0",
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_SHA256,
        "source_retrieval_dataset_id": (
            "policyproof-retrieval-evaluation"
        ),
        "source_retrieval_dataset_version": "0.1.1",
        "source_retrieval_dataset_sha256": (
            RETRIEVAL_DATASET_SHA256
        ),
        "case_count": 2,
        "cases": [
            {
                "case_id": "answer-001-reference",
                "query_id": "answer-001",
                "question": "What does the source say about risk?",
                "evidence_passage_ids": ["passage-a"],
                "expected_evidence_status": "sufficient",
                "expected_response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "rationale": (
                    "The evidence directly answers the question."
                ),
                "evaluation_tags": [
                    "reference_evidence",
                    "answerable",
                ],
            },
            {
                "case_id": "abstain-001-topical-evidence",
                "query_id": "abstain-001",
                "question": "Which current product should I buy?",
                "evidence_passage_ids": ["passage-b"],
                "expected_evidence_status": "insufficient",
                "expected_response_action": "abstain",
                "reason_codes": [
                    "outside_controlled_corpus",
                    "high_stakes_recommendation",
                ],
                "missing_information": [
                    (
                        "Current independently reviewed product "
                        "comparisons are not present in the corpus."
                    )
                ],
                "rationale": (
                    "Topically related evidence cannot support a "
                    "current high-stakes purchase recommendation."
                ),
                "evaluation_tags": [
                    "topical_but_insufficient",
                    "required_abstention",
                ],
            },
        ],
    }


def validate(value: dict[str, Any]) -> None:
    validate_evidence_sufficiency_dataset(
        value,
        retrieval_dataset=retrieval_dataset(),
        manifest=manifest(),
        passages=passages(),
        passage_artifact_sha256=PASSAGE_SHA256,
        retrieval_dataset_sha256=RETRIEVAL_DATASET_SHA256,
    )


def test_valid_dataset_passes_without_mutation() -> None:
    value = dataset()
    original = deepcopy(value)

    validate(value)

    assert value == original


def test_rejects_unsupported_schema_version() -> None:
    value = dataset()
    value["schema_version"] = "2.0"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="schema_version",
    ):
        validate(value)


def test_rejects_unexpected_dataset_id() -> None:
    value = dataset()
    value["dataset_id"] = "other-dataset"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="dataset_id",
    ):
        validate(value)


def test_rejects_invalid_dataset_version() -> None:
    value = dataset()
    value["dataset_version"] = "version-one"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="dataset_version",
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("corpus_id", "other-corpus"),
        ("corpus_version", "9.9.9"),
        ("passage_schema_version", "9.9"),
        ("passage_artifact_sha256", "c" * 64),
        (
            "source_retrieval_dataset_id",
            "other-retrieval-dataset",
        ),
        ("source_retrieval_dataset_version", "9.9.9"),
        ("source_retrieval_dataset_sha256", "d" * 64),
    ],
)
def test_rejects_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = dataset()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match=field_name,
    ):
        validate(value)


def test_rejects_invalid_sha256_format() -> None:
    value = dataset()
    value["source_retrieval_dataset_sha256"] = "not-a-sha"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="source_retrieval_dataset_sha256",
    ):
        validate(value)


def test_rejects_empty_case_collection() -> None:
    value = dataset()
    value["cases"] = []
    value["case_count"] = 0

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="cases",
    ):
        validate(value)


def test_rejects_case_count_mismatch() -> None:
    value = dataset()
    value["case_count"] = 3

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="case_count",
    ):
        validate(value)


def test_rejects_duplicate_case_id() -> None:
    value = dataset()
    value["cases"][1]["case_id"] = (
        value["cases"][0]["case_id"]
    )

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="Duplicate case_id",
    ):
        validate(value)


def test_rejects_unknown_query_id() -> None:
    value = dataset()
    value["cases"][0]["query_id"] = "unknown-query"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="query_id",
    ):
        validate(value)


def test_rejects_question_mismatch() -> None:
    value = dataset()
    value["cases"][0]["question"] = "Changed question"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="question",
    ):
        validate(value)


def test_rejects_unknown_evidence_passage_id() -> None:
    value = dataset()
    value["cases"][0]["evidence_passage_ids"] = [
        "unknown-passage"
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="evidence_passage_ids",
    ):
        validate(value)


def test_rejects_duplicate_evidence_passage_id() -> None:
    value = dataset()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-a",
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="duplicate evidence_passage_ids",
    ):
        validate(value)


def test_rejects_invalid_evidence_status() -> None:
    value = dataset()
    value["cases"][0]["expected_evidence_status"] = "unknown"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="expected_evidence_status",
    ):
        validate(value)


def test_rejects_invalid_response_action() -> None:
    value = dataset()
    value["cases"][0]["expected_response_action"] = "generate"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="expected_response_action",
    ):
        validate(value)


def test_sufficient_case_requires_nonempty_evidence() -> None:
    value = dataset()
    value["cases"][0]["evidence_passage_ids"] = []

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="sufficient.*evidence",
    ):
        validate(value)


def test_sufficient_case_requires_answer_action() -> None:
    value = dataset()
    value["cases"][0]["expected_response_action"] = "abstain"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="sufficient.*answer",
    ):
        validate(value)


def test_sufficient_case_rejects_reason_codes() -> None:
    value = dataset()
    value["cases"][0]["reason_codes"] = [
        "incomplete_evidence_set"
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="sufficient.*reason_codes",
    ):
        validate(value)


def test_sufficient_case_rejects_missing_information() -> None:
    value = dataset()
    value["cases"][0]["missing_information"] = [
        "Nothing is actually missing."
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="sufficient.*missing_information",
    ):
        validate(value)


def test_insufficient_case_requires_abstain_action() -> None:
    value = dataset()
    value["cases"][1]["expected_response_action"] = "answer"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="insufficient.*abstain",
    ):
        validate(value)


def test_insufficient_case_requires_reason_code() -> None:
    value = dataset()
    value["cases"][1]["reason_codes"] = []

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="insufficient.*reason_codes",
    ):
        validate(value)


def test_insufficient_case_requires_missing_information() -> None:
    value = dataset()
    value["cases"][1]["missing_information"] = []

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="insufficient.*missing_information",
    ):
        validate(value)


def test_rejects_unknown_reason_code() -> None:
    value = dataset()
    value["cases"][1]["reason_codes"] = [
        "unknown-reason"
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="reason_codes",
    ):
        validate(value)


def test_rejects_duplicate_reason_code() -> None:
    value = dataset()
    value["cases"][1]["reason_codes"] = [
        "outside_controlled_corpus",
        "outside_controlled_corpus",
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="duplicate reason_codes",
    ):
        validate(value)


def test_rejects_empty_missing_information_item() -> None:
    value = dataset()
    value["cases"][1]["missing_information"] = [""]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="missing_information",
    ):
        validate(value)


def test_rejects_duplicate_missing_information_item() -> None:
    value = dataset()
    item = value["cases"][1]["missing_information"][0]
    value["cases"][1]["missing_information"] = [item, item]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="duplicate missing_information",
    ):
        validate(value)


def test_rejects_empty_rationale() -> None:
    value = dataset()
    value["cases"][0]["rationale"] = ""

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="rationale",
    ):
        validate(value)


def test_rejects_empty_evaluation_tags() -> None:
    value = dataset()
    value["cases"][0]["evaluation_tags"] = []

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="evaluation_tags",
    ):
        validate(value)


def test_rejects_duplicate_evaluation_tag() -> None:
    value = dataset()
    value["cases"][0]["evaluation_tags"] = [
        "answerable",
        "answerable",
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="duplicate evaluation_tags",
    ):
        validate(value)


def test_rejects_unknown_top_level_field() -> None:
    value = dataset()
    value["unexpected"] = True

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="unknown dataset fields",
    ):
        validate(value)


def test_rejects_unknown_case_field() -> None:
    value = dataset()
    value["cases"][0]["unexpected"] = True

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="unknown case fields",
    ):
        validate(value)


def test_source_abstention_query_cannot_be_sufficient() -> None:
    value = dataset()
    case = value["cases"][1]
    case["evidence_passage_ids"] = ["passage-b"]
    case["expected_evidence_status"] = "sufficient"
    case["expected_response_action"] = "answer"
    case["reason_codes"] = []
    case["missing_information"] = []

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="abstention query.*insufficient",
    ):
        validate(value)


def test_sufficient_case_requires_reviewed_direct_evidence() -> None:
    value = dataset()
    value["cases"][0]["evidence_passage_ids"] = ["passage-b"]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="sufficient.*grade 2",
    ):
        validate(value)


def test_sufficient_case_rejects_unjudged_extra_evidence() -> None:
    value = dataset()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-c",
    ]

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="sufficient.*reviewed evidence",
    ):
        validate(value)


def test_rejects_source_judgment_for_unknown_passage() -> None:
    value = dataset()
    source_retrieval = retrieval_dataset()
    source_retrieval["queries"][0][
        "relevance_judgments"
    ][1]["passage_id"] = "unknown-reviewed-passage"

    with pytest.raises(
        EvidenceSufficiencyEvaluationError,
        match="reviewed passage_id.*accepted passage",
    ):
        validate_evidence_sufficiency_dataset(
            value,
            retrieval_dataset=source_retrieval,
            manifest=manifest(),
            passages=passages(),
            passage_artifact_sha256=PASSAGE_SHA256,
            retrieval_dataset_sha256=(
                RETRIEVAL_DATASET_SHA256
            ),
        )

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_EVIDENCE_DATASET_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "evidence-sufficiency-evaluation-v0.1.0.json"
)
REPOSITORY_RETRIEVAL_DATASET_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "retrieval-evaluation-v0.1.1.json"
)
REPOSITORY_MANIFEST_PATH = (
    ROOT / "data" / "source_manifest.json"
)
REPOSITORY_PASSAGES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "retrieval-passages.jsonl"
)
REPOSITORY_EVIDENCE_DATASET_SHA256 = (
    "9ecd30e4ff829561b50d56bf4f1d3d44"
    "c79dcb043ec15661175842597d733a6a"
)
REPOSITORY_RETRIEVAL_DATASET_SHA256 = (
    "42e7e0e1a824b1c48973bb2163aca766"
    "4d53161632fcd699068931cd9fe80a7c"
)


def repository_sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(64 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def repository_load_jsonl(
    path: Path,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )


def test_published_evidence_dataset_is_byte_stable() -> None:
    assert (
        repository_sha256_file(
            REPOSITORY_EVIDENCE_DATASET_PATH
        )
        == REPOSITORY_EVIDENCE_DATASET_SHA256
    )


def test_repository_evidence_dataset_validates() -> None:
    evidence_dataset = json.loads(
        REPOSITORY_EVIDENCE_DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )
    retrieval_dataset = json.loads(
        REPOSITORY_RETRIEVAL_DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )
    manifest_value = json.loads(
        REPOSITORY_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    passage_records = repository_load_jsonl(
        REPOSITORY_PASSAGES_PATH
    )
    passage_sha256 = repository_sha256_file(
        REPOSITORY_PASSAGES_PATH
    )
    retrieval_sha256 = repository_sha256_file(
        REPOSITORY_RETRIEVAL_DATASET_PATH
    )

    assert (
        retrieval_sha256
        == REPOSITORY_RETRIEVAL_DATASET_SHA256
    )

    validate_evidence_sufficiency_dataset(
        evidence_dataset,
        retrieval_dataset=retrieval_dataset,
        manifest=manifest_value,
        passages=passage_records,
        passage_artifact_sha256=passage_sha256,
        retrieval_dataset_sha256=retrieval_sha256,
    )

    assert evidence_dataset["dataset_version"] == "0.1.0"
    assert evidence_dataset["case_count"] == 39

    status_counts = Counter(
        case["expected_evidence_status"]
        for case in evidence_dataset["cases"]
    )
    action_counts = Counter(
        case["expected_response_action"]
        for case in evidence_dataset["cases"]
    )
    reason_counts = Counter(
        reason_code
        for case in evidence_dataset["cases"]
        for reason_code in case["reason_codes"]
    )

    assert status_counts == {
        "sufficient": 16,
        "insufficient": 23,
    }
    assert action_counts == {
        "answer": 16,
        "abstain": 23,
    }
    assert reason_counts == {
        "incomplete_evidence_set": 19,
        "organization_specific_conclusion": 3,
        "outside_controlled_corpus": 3,
        "current_information_required": 3,
        "legal_advice_boundary": 2,
        "high_stakes_recommendation": 1,
        "unsupported_comparison": 1,
    }

    cases_by_query: dict[str, list[dict[str, Any]]] = {}

    for case in evidence_dataset["cases"]:
        cases_by_query.setdefault(
            case["query_id"],
            [],
        ).append(case)

    assert set(cases_by_query) == {
        query["query_id"]
        for query in retrieval_dataset["queries"]
    }

    retrieval_behavior = {
        query["query_id"]: query["expected_behavior"]
        for query in retrieval_dataset["queries"]
    }

    for query_id, cases in cases_by_query.items():
        sufficient_count = sum(
            case["expected_evidence_status"]
            == "sufficient"
            for case in cases
        )

        if retrieval_behavior[query_id] == "answer":
            assert sufficient_count == 1
        else:
            assert sufficient_count == 0
            assert all(
                "dense_top5" in case["evaluation_tags"]
                for case in cases
            )

    query_evidence_keys = [
        (
            case["query_id"],
            tuple(sorted(case["evidence_passage_ids"])),
        )
        for case in evidence_dataset["cases"]
    ]

    assert len(query_evidence_keys) == len(
        set(query_evidence_keys)
    )
