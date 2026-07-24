from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    ANNOTATION_ANALYSIS_METADATA_ID,
    EvidenceSufficiencyAnnotationError,
    validate_annotation_analysis_metadata,
)

ANNOTATION_BATCH_SHA256 = "a" * 64


def annotation_batch() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "batch_version": "0.1.0",
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": "b" * 64,
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": "c" * 64,
        "case_count": 2,
        "cases": [
            {
                "case_id": "pilot-001-reference",
                "query_id": "pilot-001",
                "question": "What risk is identified?",
                "evidence": [
                    {
                        "passage_id": "passage-a",
                        "document_id": "document-a",
                        "label": "Section A",
                        "citation_text": "Risk A is identified.",
                    }
                ],
            },
            {
                "case_id": "pilot-002-incomplete",
                "query_id": "pilot-002",
                "question": (
                    "What risk and mitigation are identified?"
                ),
                "evidence": [
                    {
                        "passage_id": "passage-b",
                        "document_id": "document-b",
                        "label": "Section B",
                        "citation_text": "Risk B is identified.",
                    },
                    {
                        "passage_id": "passage-c",
                        "document_id": "document-b",
                        "label": "Section C",
                        "citation_text": (
                            "A related passage does not provide "
                            "the requested mitigation."
                        ),
                    },
                ],
            },
        ],
    }


def analysis_metadata() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "metadata_id": ANNOTATION_ANALYSIS_METADATA_ID,
        "metadata_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.1.0",
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "case_count": 2,
        "cases": [
            {
                "case_id": "pilot-001-reference",
                "question_structure_codes": [
                    "direct_factual_lookup",
                ],
                "evidence_structure_codes": [
                    "one_complete_passage",
                ],
            },
            {
                "case_id": "pilot-002-incomplete",
                "question_structure_codes": [
                    "risk_and_mitigation",
                    "multi_part_question",
                ],
                "evidence_structure_codes": [
                    "incomplete_evidence_set",
                    "topically_related_distractors",
                ],
            },
        ],
    }


def validate(value: dict[str, Any]) -> None:
    validate_annotation_analysis_metadata(
        value,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
    )


def test_valid_analysis_metadata_passes_without_mutation() -> None:
    value = analysis_metadata()
    original = deepcopy(value)

    validate(value)

    assert value == original


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("annotation_batch_id", "other-batch"),
        ("annotation_batch_version", "9.9.9"),
        ("annotation_batch_sha256", "d" * 64),
    ],
)
def test_analysis_metadata_rejects_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = analysis_metadata()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=field_name,
    ):
        validate(value)


def test_analysis_metadata_requires_every_batch_case_once() -> None:
    value = analysis_metadata()
    value["cases"].pop()
    value["case_count"] = 1

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="cases must cover every batch case exactly once",
    ):
        validate(value)


def test_analysis_metadata_rejects_duplicate_case() -> None:
    value = analysis_metadata()
    value["cases"][1]["case_id"] = (
        value["cases"][0]["case_id"]
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="duplicate metadata case_id",
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        (
            "question_structure_codes",
            ["unknown_question_structure"],
        ),
        (
            "evidence_structure_codes",
            ["unknown_evidence_structure"],
        ),
    ],
)
def test_analysis_metadata_rejects_unknown_structure_code(
    field_name: str,
    replacement: list[str],
) -> None:
    value = analysis_metadata()
    value["cases"][0][field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=field_name,
    ):
        validate(value)


@pytest.mark.parametrize(
    "field_name",
    [
        "question_structure_codes",
        "evidence_structure_codes",
    ],
)
def test_analysis_metadata_requires_nonempty_structure_codes(
    field_name: str,
) -> None:
    value = analysis_metadata()
    value["cases"][0][field_name] = []

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=rf"{field_name}.*nonempty",
    ):
        validate(value)


def test_analysis_metadata_rejects_duplicate_structure_code() -> None:
    value = analysis_metadata()
    value["cases"][0]["question_structure_codes"] = [
        "direct_factual_lookup",
        "direct_factual_lookup",
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="duplicate question_structure_codes",
    ):
        validate(value)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "expected_evidence_status",
        "expected_response_action",
        "reason_codes",
        "missing_information",
        "adjudicated_label",
        "policy_prediction",
        "model_score",
    ],
)
def test_analysis_metadata_rejects_label_or_policy_fields(
    forbidden_field: str,
) -> None:
    value = analysis_metadata()
    value["cases"][0][forbidden_field] = "must-not-be-present"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=rf"unknown analysis metadata case fields.*{forbidden_field}",
    ):
        validate(value)
