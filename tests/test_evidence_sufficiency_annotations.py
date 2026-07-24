from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    ANNOTATION_BATCH_ID,
    EvidenceSufficiencyAnnotationError,
    validate_annotation_batch,
)

PASSAGE_ARTIFACT_SHA256 = "a" * 64
ANNOTATION_GUIDE_VERSION = "0.1.0"
ANNOTATION_GUIDE_SHA256 = "b" * 64


def manifest() -> dict[str, Any]:
    return {
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
    }


def passages() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "1.1",
            "passage_id": "passage-a",
            "document_id": "document-a",
            "label": "Section A",
            "citation_text": "Accepted citation text for passage A.",
            "retrieval_text": "Hidden retrieval text for passage A.",
            "logical_source_key": "internal-source-a",
        },
        {
            "schema_version": "1.1",
            "passage_id": "passage-b",
            "document_id": "document-b",
            "label": "Section B",
            "citation_text": "Accepted citation text for passage B.",
            "retrieval_text": "Hidden retrieval text for passage B.",
            "logical_source_key": "internal-source-b",
        },
    ]


def annotation_batch() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": ANNOTATION_BATCH_ID,
        "batch_version": "0.1.0",
        "annotation_guide_version": ANNOTATION_GUIDE_VERSION,
        "annotation_guide_sha256": ANNOTATION_GUIDE_SHA256,
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "case_count": 1,
        "cases": [
            {
                "case_id": "pilot-001-reference",
                "query_id": "pilot-001",
                "question": (
                    "What risk does the supplied evidence identify?"
                ),
                "evidence": [
                    {
                        "passage_id": "passage-b",
                        "document_id": "document-b",
                        "label": "Section B",
                        "citation_text": (
                            "Accepted citation text for passage B."
                        ),
                    },
                    {
                        "passage_id": "passage-a",
                        "document_id": "document-a",
                        "label": "Section A",
                        "citation_text": (
                            "Accepted citation text for passage A."
                        ),
                    },
                ],
            }
        ],
    }


def validate(value: dict[str, Any]) -> None:
    validate_annotation_batch(
        value,
        manifest=manifest(),
        passages=passages(),
        passage_artifact_sha256=PASSAGE_ARTIFACT_SHA256,
        annotation_guide_version=ANNOTATION_GUIDE_VERSION,
        annotation_guide_sha256=ANNOTATION_GUIDE_SHA256,
    )


def test_valid_annotation_batch_passes_without_mutation() -> None:
    value = annotation_batch()
    original = deepcopy(value)

    validate(value)

    assert value == original
    assert [
        item["passage_id"]
        for item in value["cases"][0]["evidence"]
    ] == [
        "passage-b",
        "passage-a",
    ]


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "expected_evidence_status",
        "expected_response_action",
        "reason_codes",
        "missing_information",
        "rationale",
        "uncertainty",
        "evaluation_tags",
        "relevance_judgments",
        "policy_prediction",
        "model_score",
    ],
)
def test_annotation_batch_rejects_hidden_label_or_evaluation_fields(
    forbidden_field: str,
) -> None:
    value = annotation_batch()
    value["cases"][0][forbidden_field] = "must-not-be-visible"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=rf"must not expose.*{forbidden_field}",
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("corpus_id", "other-corpus"),
        ("corpus_version", "9.9.9"),
        ("passage_schema_version", "9.9"),
        ("passage_artifact_sha256", "c" * 64),
        ("annotation_guide_version", "9.9.9"),
        ("annotation_guide_sha256", "d" * 64),
    ],
)
def test_annotation_batch_rejects_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = annotation_batch()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=field_name,
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("citation_text", "Changed citation text."),
        ("document_id", "changed-document"),
        ("label", "Changed label"),
    ],
)
def test_annotation_batch_rejects_evidence_snapshot_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = annotation_batch()
    value["cases"][0]["evidence"][0][field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=rf"{field_name}.*accepted passage",
    ):
        validate(value)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "retrieval_text",
        "logical_source_key",
        "passage_token_count",
        "source_slices",
    ],
)
def test_annotation_evidence_rejects_internal_passage_fields(
    forbidden_field: str,
) -> None:
    value = annotation_batch()
    value["cases"][0]["evidence"][0][forbidden_field] = (
        "must-not-be-visible"
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=rf"unknown annotation evidence fields.*{forbidden_field}",
    ):
        validate(value)


def test_annotation_batch_rejects_unknown_passage_id() -> None:
    value = annotation_batch()
    value["cases"][0]["evidence"][0]["passage_id"] = (
        "unknown-passage"
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="evidence.*unknown passage_id",
    ):
        validate(value)


def test_annotation_batch_rejects_duplicate_evidence_passage_id() -> None:
    value = annotation_batch()
    value["cases"][0]["evidence"][1] = deepcopy(
        value["cases"][0]["evidence"][0]
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="duplicate evidence passage_id",
    ):
        validate(value)
