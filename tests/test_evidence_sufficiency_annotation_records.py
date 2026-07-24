from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    ANNOTATION_RECORD_SET_ID,
    EvidenceSufficiencyAnnotationError,
    validate_annotation_record_set,
)

ANNOTATION_BATCH_SHA256 = "a" * 64
PASSAGE_ARTIFACT_SHA256 = "b" * 64
ANNOTATION_GUIDE_SHA256 = "c" * 64


def annotation_batch() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "batch_version": "0.1.0",
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": ANNOTATION_GUIDE_SHA256,
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
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
                        "citation_text": "The evidence identifies risk A.",
                    }
                ],
            },
            {
                "case_id": "pilot-002-incomplete",
                "query_id": "pilot-002",
                "question": "What risk and mitigation are identified?",
                "evidence": [
                    {
                        "passage_id": "passage-b",
                        "document_id": "document-b",
                        "label": "Section B",
                        "citation_text": "The evidence identifies risk B.",
                    }
                ],
            },
        ],
    }


def annotation_record_set() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_set_id": ANNOTATION_RECORD_SET_ID,
        "record_set_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.1.0",
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": ANNOTATION_GUIDE_SHA256,
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "annotator_id": "annotator-alpha",
        "annotation_count": 2,
        "annotations": [
            {
                "annotation_id": (
                    "annotator-alpha:pilot-001-reference"
                ),
                "annotator_id": "annotator-alpha",
                "annotation_guide_version": "0.1.0",
                "case_id": "pilot-001-reference",
                "evidence_status": "sufficient",
                "response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "rationale": (
                    "The supplied evidence directly supports the "
                    "complete question."
                ),
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-07-22T19:00:00Z",
            },
            {
                "annotation_id": (
                    "annotator-alpha:pilot-002-incomplete"
                ),
                "annotator_id": "annotator-alpha",
                "annotation_guide_version": "0.1.0",
                "case_id": "pilot-002-incomplete",
                "evidence_status": "insufficient",
                "response_action": "abstain",
                "reason_codes": [
                    "incomplete_evidence_set",
                ],
                "missing_information": [
                    "The supplied evidence does not provide the requested mitigation."
                ],
                "rationale": (
                    "The evidence supports the risk but not the "
                    "requested mitigation."
                ),
                "uncertainty": True,
                "adjudication_note": (
                    "Confirm whether the question requires a named mitigation."
                ),
                "annotation_timestamp": "2026-07-22T19:05:00Z",
            },
        ],
    }


def validate(value: dict[str, Any]) -> None:
    validate_annotation_record_set(
        value,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
    )


def test_valid_annotation_record_set_passes_without_mutation() -> None:
    value = annotation_record_set()
    original = deepcopy(value)

    validate(value)

    assert value == original


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("annotation_batch_version", "9.9.9"),
        ("annotation_batch_sha256", "d" * 64),
        ("annotation_guide_version", "9.9.9"),
        ("annotation_guide_sha256", "e" * 64),
        ("passage_artifact_sha256", "f" * 64),
    ],
)
def test_annotation_record_set_rejects_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = annotation_record_set()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=field_name,
    ):
        validate(value)


def test_annotation_record_set_requires_every_batch_case_once() -> None:
    value = annotation_record_set()
    value["annotations"].pop()
    value["annotation_count"] = 1

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="annotations must cover every batch case exactly once",
    ):
        validate(value)


def test_annotation_record_set_rejects_duplicate_case_annotation() -> None:
    value = annotation_record_set()
    value["annotations"][1]["case_id"] = (
        value["annotations"][0]["case_id"]
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="duplicate annotated case_id",
    ):
        validate(value)


def test_annotation_record_must_match_submission_annotator() -> None:
    value = annotation_record_set()
    value["annotations"][0]["annotator_id"] = "annotator-beta"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="annotator_id.*submission annotator",
    ):
        validate(value)


def test_annotation_record_requires_boolean_uncertainty() -> None:
    value = annotation_record_set()
    value["annotations"][0]["uncertainty"] = "false"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="uncertainty.*boolean",
    ):
        validate(value)


def test_sufficient_annotation_rejects_reason_codes() -> None:
    value = annotation_record_set()
    value["annotations"][0]["reason_codes"] = [
        "incomplete_evidence_set",
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="sufficient.*reason_codes",
    ):
        validate(value)


def test_insufficient_annotation_requires_missing_information() -> None:
    value = annotation_record_set()
    value["annotations"][1]["missing_information"] = []

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="insufficient.*missing_information",
    ):
        validate(value)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "adjudicated_evidence_status",
        "adjudicated_response_action",
        "other_annotator_annotation",
        "agreement_result",
        "policy_prediction",
        "model_score",
    ],
)
def test_raw_annotation_rejects_post_annotation_or_external_fields(
    forbidden_field: str,
) -> None:
    value = annotation_record_set()
    value["annotations"][0][forbidden_field] = (
        "must-not-be-present"
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=rf"unknown annotation record fields.*{forbidden_field}",
    ):
        validate(value)


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        "2026-07-22 19:00:00Z",
        "2026-07-22T19:00:00",
        "2026-07-22T19:00:00+00:00",
        "2026-13-40T25:61:61Z",
        "not-a-timestamp",
    ],
)
def test_annotation_timestamp_requires_canonical_utc_format(
    invalid_timestamp: str,
) -> None:
    value = annotation_record_set()
    value["annotations"][0]["annotation_timestamp"] = (
        invalid_timestamp
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="annotation_timestamp.*RFC 3339 UTC",
    ):
        validate(value)
