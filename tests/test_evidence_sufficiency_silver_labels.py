from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from policyproof.evidence_sufficiency_silver_labels import (
    COMPLETE_DERIVATION_RULE,
    INCOMPLETE_DERIVATION_RULE,
    INCOMPLETE_MISSING_INFORMATION,
    EvidenceSufficiencySilverLabelError,
    validate_evidence_sufficiency_silver_labels,
    write_silver_label_json_artifact,
)

CONSTRUCTION_SHA256 = "a" * 64
BATCH_SHA256 = "b" * 64


def construction() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "construction_id": ("policyproof-evidence-sufficiency-case-construction"),
        "construction_version": "0.2.0",
        "case_count": 3,
        "cases": [
            {
                "case_id": "case-a",
                "evidence_structure_codes": ["one_complete_passage"],
            },
            {
                "case_id": "case-b",
                "evidence_structure_codes": [
                    "strict_subset_of_complete_evidence",
                    "incomplete_evidence_set",
                ],
                "complete_reference_case_id": "case-a",
            },
            {
                "case_id": "case-c",
                "evidence_structure_codes": [
                    "one_complete_passage",
                    "topically_related_distractors",
                ],
            },
        ],
    }


def annotation_batch() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": ("policyproof-evidence-sufficiency-annotation-batch"),
        "batch_version": "0.2.0",
        "case_count": 3,
        "cases": [
            {"case_id": "case-a"},
            {"case_id": "case-b"},
            {"case_id": "case-c"},
        ],
    }


def label_set() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "label_set_id": ("policyproof-evidence-sufficiency-silver-label-set"),
        "label_set_version": "0.1.0",
        "label_provenance": "construction_derived",
        "construction_id": ("policyproof-evidence-sufficiency-case-construction"),
        "construction_version": "0.2.0",
        "construction_sha256": CONSTRUCTION_SHA256,
        "annotation_batch_id": ("policyproof-evidence-sufficiency-annotation-batch"),
        "annotation_batch_version": "0.2.0",
        "annotation_batch_sha256": BATCH_SHA256,
        "case_count": 3,
        "labels": [
            {
                "case_id": "case-a",
                "evidence_status": "sufficient",
                "response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "derivation_rule": COMPLETE_DERIVATION_RULE,
            },
            {
                "case_id": "case-b",
                "evidence_status": "insufficient",
                "response_action": "abstain",
                "reason_codes": ["incomplete_evidence_set"],
                "missing_information": [INCOMPLETE_MISSING_INFORMATION],
                "derivation_rule": INCOMPLETE_DERIVATION_RULE,
            },
            {
                "case_id": "case-c",
                "evidence_status": "sufficient",
                "response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "derivation_rule": COMPLETE_DERIVATION_RULE,
            },
        ],
    }


def validate(value: dict[str, Any]) -> None:
    validate_evidence_sufficiency_silver_labels(
        value,
        construction=construction(),
        construction_sha256=CONSTRUCTION_SHA256,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=BATCH_SHA256,
    )


def test_valid_silver_labels_pass_without_mutation() -> None:
    value = label_set()
    original = deepcopy(value)

    validate(value)

    assert value == original


@pytest.mark.parametrize(
    "field_name",
    [
        "unexpected",
        "split",
        "human_annotator_id",
    ],
)
def test_rejects_unknown_top_level_fields(
    field_name: str,
) -> None:
    value = label_set()
    value[field_name] = "not-allowed"

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match="unknown silver label set fields",
    ):
        validate(value)


@pytest.mark.parametrize(
    "field_name",
    [
        "rationale",
        "uncertainty",
        "adjudication_note",
        "split",
    ],
)
def test_rejects_unknown_label_fields(
    field_name: str,
) -> None:
    value = label_set()
    value["labels"][0][field_name] = "not-allowed"

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match="unknown silver label fields",
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("label_provenance", "human_annotation"),
        ("construction_sha256", "c" * 64),
        ("annotation_batch_sha256", "d" * 64),
        ("construction_version", "9.9.9"),
        ("annotation_batch_version", "9.9.9"),
    ],
)
def test_rejects_binding_or_provenance_changes(
    field_name: str,
    replacement: str,
) -> None:
    value = label_set()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match=field_name,
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("evidence_status", "sufficient"),
        ("response_action", "answer"),
        ("reason_codes", []),
        ("missing_information", []),
        ("derivation_rule", COMPLETE_DERIVATION_RULE),
    ],
)
def test_rejects_incorrect_incomplete_label(
    field_name: str,
    replacement: Any,
) -> None:
    value = label_set()
    value["labels"][1][field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match="does not match",
    ):
        validate(value)


def test_rejects_label_order_mismatch() -> None:
    value = label_set()
    value["labels"][0], value["labels"][1] = (
        value["labels"][1],
        value["labels"][0],
    )

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match="order",
    ):
        validate(value)


def test_rejects_batch_construction_case_mismatch() -> None:
    batch = annotation_batch()
    batch["cases"][1]["case_id"] = "other-case"

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match="case order differ",
    ):
        validate_evidence_sufficiency_silver_labels(
            label_set(),
            construction=construction(),
            construction_sha256=CONSTRUCTION_SHA256,
            annotation_batch=batch,
            annotation_batch_sha256=BATCH_SHA256,
        )


def test_writer_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "labels.json"
    write_silver_label_json_artifact(
        label_set(),
        output_file,
    )

    with pytest.raises(
        EvidenceSufficiencySilverLabelError,
        match="exists",
    ):
        write_silver_label_json_artifact(
            label_set(),
            output_file,
        )
