from __future__ import annotations

from copy import deepcopy

import pytest

from policyproof.evidence_sufficiency_annotation_round import (
    ANNOTATION_ROUND_ID,
    FULL_OVERLAP_ASSIGNMENT_POLICY,
    build_annotation_round_manifest,
    validate_annotation_round_manifest,
)
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
)

BATCH_SHA256 = "a" * 64
GUIDE_SHA256 = "b" * 64
PASSAGE_SHA256 = "c" * 64


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
                "evidence": [],
            },
            {
                "case_id": "case-002",
                "query_id": "query-002",
                "question": "Question two?",
                "evidence": [],
            },
            {
                "case_id": "case-003",
                "query_id": "query-003",
                "question": "Question three?",
                "evidence": [],
            },
        ],
    }


def _build_manifest() -> dict[str, object]:
    return build_annotation_round_manifest(
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "annotator-001",
            "annotator-002",
        ),
        adjudicator_id="adjudicator-001",
    )


def test_builds_full_overlap_round_without_mutating_batch() -> None:
    batch = _annotation_batch()
    original_batch = deepcopy(batch)

    manifest = build_annotation_round_manifest(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "annotator-001",
            "annotator-002",
        ),
        adjudicator_id="adjudicator-001",
    )

    assert batch == original_batch
    assert manifest["round_id"] == ANNOTATION_ROUND_ID
    assert manifest["round_version"] == "0.1.0"
    assert (
        manifest["assignment_policy"]
        == FULL_OVERLAP_ASSIGNMENT_POLICY
    )
    assert manifest["primary_annotator_count"] == 2
    assert manifest["assignment_count"] == 2
    assert manifest["case_count"] == 3

    assert manifest["primary_annotator_ids"] == [
        "annotator-001",
        "annotator-002",
    ]
    assert manifest["adjudicator_id"] == "adjudicator-001"

    expected_case_ids = [
        "case-001",
        "case-002",
        "case-003",
    ]
    assert manifest["assignments"] == [
        {
            "annotator_id": "annotator-001",
            "case_ids": expected_case_ids,
        },
        {
            "annotator_id": "annotator-002",
            "case_ids": expected_case_ids,
        },
    ]


def test_valid_round_manifest_passes_without_mutation() -> None:
    batch = _annotation_batch()
    manifest = _build_manifest()
    original_batch = deepcopy(batch)
    original_manifest = deepcopy(manifest)

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
    )

    assert batch == original_batch
    assert manifest == original_manifest


def test_round_requires_two_distinct_primary_annotators() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="at least two distinct primary annotators",
    ):
        build_annotation_round_manifest(
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_version="0.1.0",
            primary_annotator_ids=("annotator-001",),
            adjudicator_id="adjudicator-001",
        )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="at least two distinct primary annotators",
    ):
        build_annotation_round_manifest(
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_version="0.1.0",
            primary_annotator_ids=(
                "annotator-001",
                "annotator-001",
            ),
            adjudicator_id="adjudicator-001",
        )


def test_adjudicator_must_be_distinct_from_annotators() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="adjudicator must be distinct",
    ):
        build_annotation_round_manifest(
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_version="0.1.0",
            primary_annotator_ids=(
                "annotator-001",
                "annotator-002",
            ),
            adjudicator_id="annotator-001",
        )


def test_validator_rejects_incomplete_overlap() -> None:
    batch = _annotation_batch()
    manifest = _build_manifest()
    assignments = manifest["assignments"]
    assert isinstance(assignments, list)
    first_assignment = assignments[0]
    assert isinstance(first_assignment, dict)
    first_assignment["case_ids"] = [
        "case-001",
        "case-002",
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="every primary annotator must receive every batch case",
    ):
        validate_annotation_round_manifest(
            manifest,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
        )


def test_validator_rejects_identity_fields() -> None:
    batch = _annotation_batch()
    manifest = _build_manifest()
    manifest["annotator_names"] = [
        "Real Person One",
        "Real Person Two",
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="unsupported fields",
    ):
        validate_annotation_round_manifest(
            manifest,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
        )



def test_builds_counterbalanced_full_overlap_assignments() -> None:
    batch = _annotation_batch()

    manifest = build_annotation_round_manifest(
        annotation_batch=batch,
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

    assert manifest["assignments"] == [
        {
            "annotator_id": "annotator-001",
            "case_ids": [
                "case-001",
                "case-002",
                "case-003",
            ],
        },
        {
            "annotator_id": "annotator-002",
            "case_ids": [
                "case-003",
                "case-002",
                "case-001",
            ],
        },
    ]

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
    )


def test_counterbalanced_order_must_be_complete_permutation() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            "assignment case order must contain every "
            "batch case exactly once"
        ),
    ):
        build_annotation_round_manifest(
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
                    "case-003",
                    "case-001",
                ),
            },
        )


def test_validator_accepts_different_complete_case_orders() -> None:
    batch = _annotation_batch()
    manifest = _build_manifest()

    assignments = manifest["assignments"]
    assert isinstance(assignments, list)

    second_assignment = assignments[1]
    assert isinstance(second_assignment, dict)
    second_assignment["case_ids"] = [
        "case-003",
        "case-002",
        "case-001",
    ]

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
    )
