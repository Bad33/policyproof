from __future__ import annotations

from copy import deepcopy

import pytest

from policyproof.evidence_sufficiency_annotation_round import (
    build_annotation_assignment_package,
    build_annotation_round_manifest,
)
from policyproof.evidence_sufficiency_annotation_submission import (
    build_annotation_record_set_template,
    validate_annotation_submission,
)
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
    validate_annotation_record_set,
)

BATCH_SHA256 = "1" * 64
ROUND_SHA256 = "2" * 64
ASSIGNMENT_SHA256 = "3" * 64
GUIDE_SHA256 = "4" * 64
PASSAGE_SHA256 = "5" * 64


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
                        "label": "Section one",
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
                        "label": "Section two",
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


def _template() -> dict[str, object]:
    return build_annotation_record_set_template(
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=_round_manifest(),
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=_assignment_package(),
        assignment_package_sha256=ASSIGNMENT_SHA256,
        record_set_version="0.1.0",
    )


def test_template_preserves_assignment_order_and_bindings() -> None:
    template = _template()

    assert template == {
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
                "evidence_status": None,
                "response_action": None,
                "reason_codes": [],
                "missing_information": [],
                "rationale": None,
                "uncertainty": None,
                "adjudication_note": None,
                "annotation_timestamp": None,
            },
            {
                "annotation_id": "annotator-002:case-001",
                "annotator_id": "annotator-002",
                "annotation_guide_version": "0.1.0",
                "case_id": "case-001",
                "evidence_status": None,
                "response_action": None,
                "reason_codes": [],
                "missing_information": [],
                "rationale": None,
                "uncertainty": None,
                "adjudication_note": None,
                "annotation_timestamp": None,
            },
        ],
    }


def test_template_contains_no_case_content_or_hidden_labels() -> None:
    template = _template()
    serialized = repr(template)

    forbidden = (
        "Question one?",
        "Question two?",
        "Evidence one.",
        "Evidence two.",
        "expected_evidence_status",
        "expected_response_action",
        "silver_label",
        "model_score",
        "policy_prediction",
    )

    for value in forbidden:
        assert value not in serialized


def test_template_builder_does_not_mutate_inputs() -> None:
    batch = _annotation_batch()
    manifest = _round_manifest()
    assignment = _assignment_package()

    original_batch = deepcopy(batch)
    original_manifest = deepcopy(manifest)
    original_assignment = deepcopy(assignment)

    build_annotation_record_set_template(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment,
        assignment_package_sha256=ASSIGNMENT_SHA256,
        record_set_version="0.1.0",
    )

    assert batch == original_batch
    assert manifest == original_manifest
    assert assignment == original_assignment


def test_unfilled_template_is_not_a_valid_submission() -> None:
    with pytest.raises(EvidenceSufficiencyAnnotationError):
        validate_annotation_record_set(
            _template(),
            annotation_batch=_annotation_batch(),
            annotation_batch_sha256=BATCH_SHA256,
        )


def test_completed_template_becomes_a_valid_submission() -> None:
    record_set = _template()
    annotations = record_set["annotations"]
    assert isinstance(annotations, list)

    timestamps = (
        "2026-08-01T14:00:00Z",
        "2026-08-01T14:05:00Z",
    )

    for annotation, timestamp in zip(
        annotations,
        timestamps,
        strict=True,
    ):
        assert isinstance(annotation, dict)
        annotation["evidence_status"] = "sufficient"
        annotation["response_action"] = "answer"
        annotation["rationale"] = (
            "The assigned evidence supports the complete answer."
        )
        annotation["uncertainty"] = False
        annotation["annotation_timestamp"] = timestamp

    validate_annotation_submission(
        record_set,
        annotation_batch=_annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=_round_manifest(),
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=_assignment_package(),
        assignment_package_sha256=ASSIGNMENT_SHA256,
    )
