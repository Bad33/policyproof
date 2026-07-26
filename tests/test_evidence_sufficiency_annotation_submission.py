from __future__ import annotations

import json
from copy import deepcopy

import pytest

from policyproof.evidence_sufficiency_annotation_round import (
    build_annotation_assignment_package,
    build_annotation_round_manifest,
)
from policyproof.evidence_sufficiency_annotation_submission import (
    ANNOTATION_INDEPENDENCE_ATTESTATION_ID,
    ANNOTATION_SUBMISSION_RECEIPT_ID,
    REQUIRED_INDEPENDENCE_STATEMENTS,
    build_annotation_independence_attestation,
    build_annotation_submission_receipt,
    independence_attestation_failures,
    validate_annotation_independence_attestation,
    validate_annotation_round_completion,
    validate_annotation_submission,
    validate_annotation_submission_receipt,
)
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
)

BATCH_SHA256 = "a" * 64
GUIDE_SHA256 = "b" * 64
PASSAGE_SHA256 = "c" * 64
ROUND_SHA256 = "d" * 64
ASSIGNMENT_SHA256 = "e" * 64
RECORD_SET_SHA256 = "f" * 64


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
        "case_count": 2,
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
            ),
            "annotator-002": (
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


def _record_set() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "record_set_id": (
            "policyproof-evidence-sufficiency-annotation-record-set"
        ),
        "record_set_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.2.0",
        "annotation_batch_sha256": BATCH_SHA256,
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": GUIDE_SHA256,
        "passage_artifact_sha256": PASSAGE_SHA256,
        "annotator_id": "annotator-002",
        "annotation_count": 2,
        "annotations": [
            {
                "annotation_id": "annotator-002:case-002",
                "annotator_id": "annotator-002",
                "annotation_guide_version": "0.1.0",
                "case_id": "case-002",
                "evidence_status": "sufficient",
                "response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "rationale": "The evidence answers the question.",
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-08-01T14:00:00Z",
            },
            {
                "annotation_id": "annotator-002:case-001",
                "annotator_id": "annotator-002",
                "annotation_guide_version": "0.1.0",
                "case_id": "case-001",
                "evidence_status": "sufficient",
                "response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "rationale": "The evidence answers the question.",
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-08-01T14:05:00Z",
            },
        ],
    }


def _receipt() -> dict[str, object]:
    return build_annotation_submission_receipt(
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=_round_manifest(),
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=_assignment_package(),
        assignment_package_sha256=ASSIGNMENT_SHA256,
        record_set=_record_set(),
        record_set_sha256=RECORD_SET_SHA256,
        receipt_version="0.1.0",
        intake_operator_id="operator-001",
        received_timestamp="2026-08-01T15:00:00Z",
    )


def test_valid_submission_passes_without_mutation() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    assignment = _assignment_package()
    record_set = _record_set()

    original_batch = deepcopy(batch)
    original_manifest = deepcopy(manifest)
    original_assignment = deepcopy(assignment)
    original_record_set = deepcopy(record_set)

    validate_annotation_submission(
        record_set,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment,
        assignment_package_sha256=ASSIGNMENT_SHA256,
    )

    assert batch == original_batch
    assert manifest == original_manifest
    assert assignment == original_assignment
    assert record_set == original_record_set


def test_submission_annotator_must_match_assignment() -> None:
    record_set = _record_set()
    record_set["annotator_id"] = "annotator-001"

    annotations = record_set["annotations"]
    assert isinstance(annotations, list)
    for annotation in annotations:
        assert isinstance(annotation, dict)
        annotation["annotator_id"] = "annotator-001"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="submission annotator does not match assignment",
    ):
        validate_annotation_submission(
            record_set,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=_assignment_package(),
            assignment_package_sha256=ASSIGNMENT_SHA256,
        )


def test_submission_preserves_assigned_case_order() -> None:
    record_set = _record_set()
    annotations = record_set["annotations"]
    assert isinstance(annotations, list)
    annotations.reverse()

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="annotations must preserve assigned case order",
    ):
        validate_annotation_submission(
            record_set,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=_assignment_package(),
            assignment_package_sha256=ASSIGNMENT_SHA256,
        )


def test_builds_metadata_only_submission_receipt() -> None:
    receipt = _receipt()

    assert (
        receipt["receipt_id"]
        == ANNOTATION_SUBMISSION_RECEIPT_ID
    )
    assert receipt["receipt_version"] == "0.1.0"
    assert receipt["annotator_id"] == "annotator-002"
    assert receipt["intake_operator_id"] == "operator-001"
    assert receipt["record_set_sha256"] == RECORD_SET_SHA256

    serialized = json.dumps(receipt)
    assert "evidence_status" not in serialized
    assert "sufficient" not in serialized
    assert "rationale" not in serialized
    assert "case-001" not in serialized


def test_valid_receipt_passes_without_mutation() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    assignment = _assignment_package()
    record_set = _record_set()
    receipt = _receipt()

    originals = tuple(
        deepcopy(value)
        for value in (
            batch,
            manifest,
            assignment,
            record_set,
            receipt,
        )
    )

    validate_annotation_submission_receipt(
        receipt,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment,
        assignment_package_sha256=ASSIGNMENT_SHA256,
        record_set=record_set,
        record_set_sha256=RECORD_SET_SHA256,
    )

    assert (
        batch,
        manifest,
        assignment,
        record_set,
        receipt,
    ) == originals


def test_receipt_rejects_record_set_hash_mismatch() -> None:
    receipt = _receipt()
    receipt["record_set_sha256"] = "0" * 64

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="record_set_sha256 does not match",
    ):
        validate_annotation_submission_receipt(
            receipt,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=_assignment_package(),
            assignment_package_sha256=ASSIGNMENT_SHA256,
            record_set=_record_set(),
            record_set_sha256=RECORD_SET_SHA256,
        )


def test_receipt_requires_canonical_utc_timestamp() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="received_timestamp must use canonical UTC format",
    ):
        build_annotation_submission_receipt(
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=_assignment_package(),
            assignment_package_sha256=ASSIGNMENT_SHA256,
            record_set=_record_set(),
            record_set_sha256=RECORD_SET_SHA256,
            receipt_version="0.1.0",
            intake_operator_id="operator-001",
            received_timestamp="2026-08-01 15:00:00",
        )



def _independence_statements() -> dict[str, bool]:
    return {
        "completed_without_collaboration": True,
        "did_not_view_other_annotations": True,
        "did_not_view_construction_or_silver_labels": True,
        "did_not_view_split_assignments": True,
        "did_not_view_retrieval_or_model_scores": True,
        "used_only_assigned_materials": True,
    }


def _attestation(
    *,
    statements: dict[str, bool] | None = None,
) -> dict[str, object]:
    return build_annotation_independence_attestation(
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=_round_manifest(),
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=_assignment_package(),
        assignment_package_sha256=ASSIGNMENT_SHA256,
        record_set=_record_set(),
        record_set_sha256=RECORD_SET_SHA256,
        attestation_version="0.1.0",
        statements=(
            _independence_statements()
            if statements is None
            else statements
        ),
        attested_timestamp="2026-08-01T14:10:00Z",
    )


def test_builds_record_set_bound_independence_attestation() -> None:
    attestation = _attestation()

    assert (
        attestation["attestation_id"]
        == ANNOTATION_INDEPENDENCE_ATTESTATION_ID
    )
    assert attestation["attestation_version"] == "0.1.0"
    assert attestation["annotator_id"] == "annotator-002"
    assert attestation["record_set_sha256"] == RECORD_SET_SHA256
    assert (
        set(attestation["statements"])
        == REQUIRED_INDEPENDENCE_STATEMENTS
    )
    assert independence_attestation_failures(attestation) == ()


def test_attestation_contains_no_case_or_label_content() -> None:
    attestation = _attestation()
    serialized = json.dumps(attestation)

    assert "case-001" not in serialized
    assert "case-002" not in serialized
    assert "evidence_status" not in serialized
    assert "sufficient" not in serialized
    assert "rationale" not in serialized


def test_valid_attestation_passes_without_mutation() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    assignment = _assignment_package()
    record_set = _record_set()
    attestation = _attestation()

    originals = tuple(
        deepcopy(value)
        for value in (
            batch,
            manifest,
            assignment,
            record_set,
            attestation,
        )
    )

    validate_annotation_independence_attestation(
        attestation,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment,
        assignment_package_sha256=ASSIGNMENT_SHA256,
        record_set=record_set,
        record_set_sha256=RECORD_SET_SHA256,
    )

    assert (
        batch,
        manifest,
        assignment,
        record_set,
        attestation,
    ) == originals


def test_independence_gate_reports_failed_statements() -> None:
    statements = _independence_statements()
    statements["did_not_view_other_annotations"] = False

    attestation = _attestation(statements=statements)

    assert independence_attestation_failures(attestation) == (
        "did_not_view_other_annotations",
    )


def test_attestation_statement_requires_boolean() -> None:
    attestation = _attestation()
    statements = attestation["statements"]
    assert isinstance(statements, dict)
    statements["completed_without_collaboration"] = "yes"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            "completed_without_collaboration "
            "must be a boolean"
        ),
    ):
        validate_annotation_independence_attestation(
            attestation,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=_assignment_package(),
            assignment_package_sha256=ASSIGNMENT_SHA256,
            record_set=_record_set(),
            record_set_sha256=RECORD_SET_SHA256,
        )


def test_attestation_rejects_private_identity_fields() -> None:
    attestation = _attestation()
    attestation["annotator_real_name"] = "Private Person"

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="unsupported fields",
    ):
        validate_annotation_independence_attestation(
            attestation,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=_assignment_package(),
            assignment_package_sha256=ASSIGNMENT_SHA256,
            record_set=_record_set(),
            record_set_sha256=RECORD_SET_SHA256,
        )


def _record_set_for(
    annotator_id: str,
) -> dict[str, object]:
    record_set = deepcopy(_record_set())
    record_set["annotator_id"] = annotator_id

    annotations = record_set["annotations"]
    assert isinstance(annotations, list)

    for annotation in annotations:
        assert isinstance(annotation, dict)
        case_id = annotation["case_id"]
        annotation["annotator_id"] = annotator_id
        annotation["annotation_id"] = (
            f"{annotator_id}:{case_id}"
        )

    if annotator_id == "annotator-001":
        annotations.reverse()

    return record_set


def _round_submission_bundle(
    annotator_id: str,
) -> dict[str, object]:
    batch = _annotation_batch()
    manifest = _round_manifest()

    assignment_package = build_annotation_assignment_package(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        annotator_id=annotator_id,
        assignment_version="0.1.0",
    )
    record_set = _record_set_for(annotator_id)

    if annotator_id == "annotator-001":
        assignment_sha256 = "1" * 64
        record_set_sha256 = "3" * 64
        received_timestamp = "2026-08-01T15:00:00Z"
        attested_timestamp = "2026-08-01T14:10:00Z"
    else:
        assignment_sha256 = "2" * 64
        record_set_sha256 = "4" * 64
        received_timestamp = "2026-08-01T16:00:00Z"
        attested_timestamp = "2026-08-01T15:10:00Z"

    receipt = build_annotation_submission_receipt(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment_package,
        assignment_package_sha256=assignment_sha256,
        record_set=record_set,
        record_set_sha256=record_set_sha256,
        receipt_version="0.1.0",
        intake_operator_id="operator-001",
        received_timestamp=received_timestamp,
    )
    attestation = build_annotation_independence_attestation(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment_package,
        assignment_package_sha256=assignment_sha256,
        record_set=record_set,
        record_set_sha256=record_set_sha256,
        attestation_version="0.1.0",
        statements=_independence_statements(),
        attested_timestamp=attested_timestamp,
    )

    return {
        "assignment_package": assignment_package,
        "assignment_package_sha256": assignment_sha256,
        "record_set": record_set,
        "record_set_sha256": record_set_sha256,
        "receipt": receipt,
        "attestation": attestation,
    }


def test_complete_round_accepts_all_valid_submissions() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    bundles = [
        _round_submission_bundle("annotator-001"),
        _round_submission_bundle("annotator-002"),
    ]

    original_batch = deepcopy(batch)
    original_manifest = deepcopy(manifest)
    original_bundles = deepcopy(bundles)

    validate_annotation_round_completion(
        bundles,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
    )

    assert batch == original_batch
    assert manifest == original_manifest
    assert bundles == original_bundles


def test_complete_round_requires_every_assigned_annotator() -> None:
    bundles = [
        _round_submission_bundle("annotator-001"),
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            "submission bundles must cover every primary "
            "annotator exactly once"
        ),
    ):
        validate_annotation_round_completion(
            bundles,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
        )


def test_complete_round_rejects_duplicate_annotator() -> None:
    bundles = [
        _round_submission_bundle("annotator-001"),
        _round_submission_bundle("annotator-001"),
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="duplicate submission annotator",
    ):
        validate_annotation_round_completion(
            bundles,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
        )


def test_complete_round_requires_affirmative_independence() -> None:
    bundles = [
        _round_submission_bundle("annotator-001"),
        _round_submission_bundle("annotator-002"),
    ]

    attestation = bundles[1]["attestation"]
    assert isinstance(attestation, dict)
    statements = attestation["statements"]
    assert isinstance(statements, dict)
    statements["did_not_view_other_annotations"] = False

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            "independence attestation failed.*"
            "did_not_view_other_annotations"
        ),
    ):
        validate_annotation_round_completion(
            bundles,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
        )


def test_complete_round_requires_distinct_record_sets() -> None:
    bundles = [
        _round_submission_bundle("annotator-001"),
        _round_submission_bundle("annotator-002"),
    ]
    bundles[1]["record_set_sha256"] = bundles[0][
        "record_set_sha256"
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="record_set_sha256 values must be distinct",
    ):
        validate_annotation_round_completion(
            bundles,
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=_round_manifest(),
            round_manifest_sha256=ROUND_SHA256,
        )
