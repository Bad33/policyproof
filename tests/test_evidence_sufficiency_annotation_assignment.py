from __future__ import annotations

import json
from copy import deepcopy

import pytest

from policyproof.evidence_sufficiency_annotation_round import (
    ANNOTATION_ASSIGNMENT_ID,
    build_annotation_assignment_package,
    build_annotation_round_manifest,
    validate_annotation_assignment_package,
)
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
)

BATCH_SHA256 = "a" * 64
GUIDE_SHA256 = "b" * 64
PASSAGE_SHA256 = "c" * 64
ROUND_SHA256 = "d" * 64


def _annotation_batch() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "batch_version": "0.2.0",
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": GUIDE_SHA256,
        "passage_artifact_sha256": PASSAGE_SHA256,
        "case_count": 3,
        "cases": [
            {
                "case_id": "case-001",
                "query_id": "query-001",
                "question": "Question one?",
                "evidence": [
                    {
                        "passage_id": "passage-001",
                        "document_id": "document-001",
                        "label": "Example section",
                        "citation_text": "Evidence one.",
                    }
                ],
            },
            {
                "case_id": "case-002",
                "query_id": "query-002",
                "question": "Question two?",
                "evidence": [
                    {
                        "passage_id": "passage-002",
                        "document_id": "document-002",
                        "label": "Example section",
                        "citation_text": "Evidence two.",
                    }
                ],
            },
            {
                "case_id": "case-003",
                "query_id": "query-003",
                "question": "Question three?",
                "evidence": [
                    {
                        "passage_id": "passage-003",
                        "document_id": "document-003",
                        "label": "Example section",
                        "citation_text": "Evidence three.",
                    }
                ],
            },
        ],
    }


def _round_manifest() -> dict[str, object]:
    return build_annotation_round_manifest(
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "annotator-001",
            "annotator-002",
        ),
        adjudicator_id="adjudicator-001",
        assignment_case_orders={
            "annotator-001": (
                "case-001",
                "case-002",
                "case-003",
            ),
            "annotator-002": (
                "case-003",
                "case-002",
                "case-001",
            ),
        },
    )


def _assignment_package() -> dict[str, object]:
    return build_annotation_assignment_package(
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=_round_manifest(),
        round_manifest_sha256=ROUND_SHA256,
        annotator_id="annotator-002",
        assignment_version="0.1.0",
    )


def test_builds_isolated_counterbalanced_assignment() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    original_batch = deepcopy(batch)
    original_manifest = deepcopy(manifest)

    package = build_annotation_assignment_package(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        annotator_id="annotator-002",
        assignment_version="0.1.0",
    )

    assert batch == original_batch
    assert manifest == original_manifest
    assert package["assignment_id"] == ANNOTATION_ASSIGNMENT_ID
    assert package["assignment_version"] == "0.1.0"
    assert package["annotator_id"] == "annotator-002"
    assert package["case_count"] == 3

    cases = package["cases"]
    assert isinstance(cases, list)
    assert [
        case["case_id"]
        for case in cases
        if isinstance(case, dict)
    ] == [
        "case-003",
        "case-002",
        "case-001",
    ]


def test_assignment_does_not_expose_other_roles() -> None:
    package = _assignment_package()

    assert "primary_annotator_ids" not in package
    assert "adjudicator_id" not in package
    assert "assignments" not in package

    serialized = json.dumps(package)
    assert "annotator-001" not in serialized
    assert "adjudicator-001" not in serialized


def test_valid_assignment_passes_without_mutation() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    package = _assignment_package()

    original_batch = deepcopy(batch)
    original_manifest = deepcopy(manifest)
    original_package = deepcopy(package)

    validate_annotation_assignment_package(
        package,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
    )

    assert batch == original_batch
    assert manifest == original_manifest
    assert package == original_package


def test_builder_rejects_unassigned_annotator() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="annotator is not assigned in this round",
    ):
        build_annotation_assignment_package(
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            annotator_id="annotator-999",
            assignment_version="0.1.0",
        )


def test_validator_rejects_wrong_case_order() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    package = _assignment_package()

    cases = package["cases"]
    assert isinstance(cases, list)
    cases.reverse()

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="cases must match the assigned case order",
    ):
        validate_annotation_assignment_package(
            package,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=manifest,
            round_manifest_sha256=ROUND_SHA256,
        )


def test_validator_rejects_private_identity_fields() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    package = _assignment_package()
    package["annotator_real_name"] = "Private Person"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="unsupported fields",
    ):
        validate_annotation_assignment_package(
            package,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=manifest,
            round_manifest_sha256=ROUND_SHA256,
        )
