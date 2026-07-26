from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from policyproof.evidence_sufficiency_annotation_round import (
    build_annotation_assignment_package,
    build_annotation_round_manifest,
    validate_annotation_assignment_package,
    validate_annotation_round_manifest,
)
from policyproof.evidence_sufficiency_annotation_submission import (
    build_annotation_record_set_template,
    validate_annotation_submission,
)
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
)

BATCH_PATH = Path(
    "data/evaluation/"
    "evidence-sufficiency-annotation-batch-v0.2.0.json"
)
BATCH_SHA256 = (
    "1bb6a7bed55a43f59a79ff4861c81c3d"
    "36ffa5ed78af1bf12292bceb927bf93c"
)
ROUND_SHA256 = "d" * 64

FORBIDDEN_KEYS = {
    "construction_label",
    "construction_status",
    "expected_status",
    "expected_label",
    "gold_label",
    "silver_label",
    "sufficiency_status",
    "is_sufficient",
    "split",
    "split_name",
    "split_assignment",
    "retrieval_score",
    "retrieval_rank",
    "model_score",
    "model_probability",
    "candidate_score",
    "label_provenance",
}


def _load_batch() -> dict[str, object]:
    value = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()

    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))

    return keys


def test_real_batch_supports_counterbalanced_full_overlap() -> None:
    raw_bytes = BATCH_PATH.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == BATCH_SHA256

    batch = _load_batch()
    original_batch = deepcopy(batch)

    cases = batch["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 160

    case_ids = [
        case["case_id"]
        for case in cases
        if isinstance(case, dict)
    ]
    assert len(case_ids) == 160
    assert len(set(case_ids)) == 160

    manifest = build_annotation_round_manifest(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "test-annotator-a",
            "test-annotator-b",
        ),
        adjudicator_id="test-adjudicator",
        assignment_case_orders={
            "test-annotator-a": tuple(case_ids),
            "test-annotator-b": tuple(reversed(case_ids)),
        },
    )

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
    )

    assert batch == original_batch
    assert manifest["case_count"] == 160
    assert manifest["assignment_count"] == 2

    assignments = manifest["assignments"]
    assert isinstance(assignments, list)
    assert assignments == [
        {
            "annotator_id": "test-annotator-a",
            "case_ids": case_ids,
        },
        {
            "annotator_id": "test-annotator-b",
            "case_ids": list(reversed(case_ids)),
        },
    ]


def test_real_assignment_packages_remain_isolated_and_blinded() -> None:
    batch = _load_batch()
    cases = batch["cases"]
    assert isinstance(cases, list)

    case_ids = [
        case["case_id"]
        for case in cases
        if isinstance(case, dict)
    ]

    manifest = build_annotation_round_manifest(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "test-annotator-a",
            "test-annotator-b",
        ),
        adjudicator_id="test-adjudicator",
        assignment_case_orders={
            "test-annotator-a": tuple(case_ids),
            "test-annotator-b": tuple(reversed(case_ids)),
        },
    )

    for annotator_id, excluded_ids in (
        (
            "test-annotator-a",
            ("test-annotator-b", "test-adjudicator"),
        ),
        (
            "test-annotator-b",
            ("test-annotator-a", "test-adjudicator"),
        ),
    ):
        package = build_annotation_assignment_package(
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=manifest,
            round_manifest_sha256=ROUND_SHA256,
            annotator_id=annotator_id,
            assignment_version="0.1.0",
        )

        validate_annotation_assignment_package(
            package,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=manifest,
            round_manifest_sha256=ROUND_SHA256,
        )

        assert package["annotator_id"] == annotator_id
        assert package["case_count"] == 160
        assert not (_all_keys(package) & FORBIDDEN_KEYS)

        serialized = json.dumps(package)
        for excluded_id in excluded_ids:
            assert excluded_id not in serialized

        package_cases = package["cases"]
        assert isinstance(package_cases, list)

        expected_cases = (
            cases
            if annotator_id == "test-annotator-a"
            else list(reversed(cases))
        )
        assert package_cases == expected_cases


def test_real_record_set_template_is_ordered_blinded_and_unfilled() -> None:
    raw_bytes = BATCH_PATH.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == BATCH_SHA256

    batch = _load_batch()
    original_batch = deepcopy(batch)

    cases = batch["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 160

    case_ids = [
        case["case_id"]
        for case in cases
        if isinstance(case, dict)
    ]
    reversed_case_ids = list(reversed(case_ids))

    manifest = build_annotation_round_manifest(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "test-annotator-a",
            "test-annotator-b",
        ),
        adjudicator_id="test-adjudicator",
        assignment_case_orders={
            "test-annotator-a": tuple(case_ids),
            "test-annotator-b": tuple(reversed_case_ids),
        },
    )
    assignment = build_annotation_assignment_package(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        annotator_id="test-annotator-b",
        assignment_version="0.1.0",
    )
    assignment_sha256 = "e" * 64

    template = build_annotation_record_set_template(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        assignment_package=assignment,
        assignment_package_sha256=assignment_sha256,
        record_set_version="0.1.0",
    )

    assert batch == original_batch
    assert template["annotator_id"] == "test-annotator-b"
    assert template["annotation_count"] == 160

    annotations = template["annotations"]
    assert isinstance(annotations, list)
    assert len(annotations) == 160

    assert [
        annotation["case_id"]
        for annotation in annotations
        if isinstance(annotation, dict)
    ] == reversed_case_ids

    expected_template_fields = {
        "schema_version",
        "record_set_id",
        "record_set_version",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "annotation_guide_version",
        "annotation_guide_sha256",
        "passage_artifact_sha256",
        "annotator_id",
        "annotation_count",
        "annotations",
    }
    expected_annotation_fields = {
        "annotation_id",
        "annotator_id",
        "annotation_guide_version",
        "case_id",
        "evidence_status",
        "response_action",
        "reason_codes",
        "missing_information",
        "rationale",
        "uncertainty",
        "adjudication_note",
        "annotation_timestamp",
    }

    assert set(template) == expected_template_fields

    for annotation in annotations:
        assert isinstance(annotation, dict)
        assert set(annotation) == expected_annotation_fields
        assert annotation["evidence_status"] is None
        assert annotation["response_action"] is None
        assert annotation["reason_codes"] == []
        assert annotation["missing_information"] == []
        assert annotation["rationale"] is None
        assert annotation["uncertainty"] is None
        assert annotation["adjudication_note"] is None
        assert annotation["annotation_timestamp"] is None

    template_keys = _all_keys(template)
    assert not (template_keys & FORBIDDEN_KEYS)
    assert not (
        template_keys
        & {
            "question",
            "evidence",
            "passage_id",
            "document_id",
            "label",
            "citation_text",
        }
    )

    with pytest.raises(EvidenceSufficiencyAnnotationError):
        validate_annotation_submission(
            template,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
            round_manifest=manifest,
            round_manifest_sha256=ROUND_SHA256,
            assignment_package=assignment,
            assignment_package_sha256=assignment_sha256,
        )

    assert hashlib.sha256(BATCH_PATH.read_bytes()).hexdigest() == (
        BATCH_SHA256
    )
