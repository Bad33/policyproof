# Validation for construction-derived evidence-sufficiency silver labels.

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
    write_annotation_json_artifact,
)

SILVER_LABEL_SET_ID = "policyproof-evidence-sufficiency-silver-label-set"
SILVER_LABEL_SCHEMA_VERSION = "1.0"
SILVER_LABEL_PROVENANCE = "construction_derived"

SUFFICIENT_STATUS = "sufficient"
INSUFFICIENT_STATUS = "insufficient"
ANSWER_ACTION = "answer"
ABSTAIN_ACTION = "abstain"

COMPLETE_DERIVATION_RULE = "canonical_complete_or_complete_with_distractor"
INCOMPLETE_DERIVATION_RULE = "incomplete_strict_subset"
INCOMPLETE_REASON_CODE = "incomplete_evidence_set"
INCOMPLETE_MISSING_INFORMATION = (
    "At least one passage from the canonical complete-reference evidence set is absent."
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

LABEL_SET_FIELDS = frozenset(
    {
        "schema_version",
        "label_set_id",
        "label_set_version",
        "label_provenance",
        "construction_id",
        "construction_version",
        "construction_sha256",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "case_count",
        "labels",
    }
)

LABEL_FIELDS = frozenset(
    {
        "case_id",
        "evidence_status",
        "response_action",
        "reason_codes",
        "missing_information",
        "derivation_rule",
    }
)


class EvidenceSufficiencySilverLabelError(ValueError):
    """Raised when construction-derived silver labels are invalid."""


def _mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencySilverLabelError(f"{field_name} must be an object.")

    return value


def _sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceSufficiencySilverLabelError(f"{field_name} must be an array.")

    return value


def _text(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencySilverLabelError(f"{field_name} must be a nonempty string.")

    return value


def _version(
    value: Any,
    *,
    field_name: str,
) -> str:
    text = _text(
        value,
        field_name=field_name,
    )

    if not VERSION_PATTERN.fullmatch(text):
        raise EvidenceSufficiencySilverLabelError(
            f"{field_name} must use semantic version form X.Y.Z."
        )

    return text


def _sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    text = _text(
        value,
        field_name=field_name,
    )

    if not SHA256_PATTERN.fullmatch(text):
        raise EvidenceSufficiencySilverLabelError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return text


def _nonnegative_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceSufficiencySilverLabelError(f"{field_name} must be a non-negative integer.")

    return value


def _reject_unknown(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    object_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)

    if unknown:
        raise EvidenceSufficiencySilverLabelError(f"unknown {object_name} fields: {unknown}.")


def _string_array(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    items = _sequence(
        value,
        field_name=field_name,
    )
    result: list[str] = []
    seen: set[str] = set()

    for position, item in enumerate(items):
        text = _text(
            item,
            field_name=f"{field_name}[{position}]",
        )

        if text in seen:
            raise EvidenceSufficiencySilverLabelError(f"duplicate {field_name}: {text}.")

        seen.add(text)
        result.append(text)

    return tuple(result)


def _artifact_cases(
    artifact: Mapping[str, Any],
    *,
    artifact_name: str,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    cases = _sequence(
        artifact.get("cases"),
        field_name=f"{artifact_name}.cases",
    )
    result: list[Mapping[str, Any]] = []
    case_ids: list[str] = []
    seen: set[str] = set()

    for position, raw_case in enumerate(cases):
        case = _mapping(
            raw_case,
            field_name=f"{artifact_name}.cases[{position}]",
        )
        case_id = _text(
            case.get("case_id"),
            field_name=(f"{artifact_name}.cases[{position}].case_id"),
        )

        if case_id in seen:
            raise EvidenceSufficiencySilverLabelError(
                f"duplicate {artifact_name} case_id: {case_id}."
            )

        seen.add(case_id)
        result.append(case)
        case_ids.append(case_id)

    declared_count = _nonnegative_integer(
        artifact.get("case_count"),
        field_name=f"{artifact_name}.case_count",
    )

    if declared_count != len(result):
        raise EvidenceSufficiencySilverLabelError(
            f"{artifact_name}.case_count does not match cases."
        )

    return result, case_ids


def validate_evidence_sufficiency_silver_labels(
    label_set: Mapping[str, Any],
    *,
    construction: Mapping[str, Any],
    construction_sha256: str,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
) -> None:
    """Validate one immutable construction-derived silver-label set."""

    value = _mapping(
        label_set,
        field_name="silver label set",
    )
    construction_value = _mapping(
        construction,
        field_name="construction",
    )
    batch_value = _mapping(
        annotation_batch,
        field_name="annotation_batch",
    )

    _reject_unknown(
        value,
        allowed=LABEL_SET_FIELDS,
        object_name="silver label set",
    )

    if (
        _text(
            value.get("schema_version"),
            field_name="schema_version",
        )
        != SILVER_LABEL_SCHEMA_VERSION
    ):
        raise EvidenceSufficiencySilverLabelError("schema_version must be 1.0.")

    if (
        _text(
            value.get("label_set_id"),
            field_name="label_set_id",
        )
        != SILVER_LABEL_SET_ID
    ):
        raise EvidenceSufficiencySilverLabelError("label_set_id is not supported.")

    _version(
        value.get("label_set_version"),
        field_name="label_set_version",
    )

    if (
        _text(
            value.get("label_provenance"),
            field_name="label_provenance",
        )
        != SILVER_LABEL_PROVENANCE
    ):
        raise EvidenceSufficiencySilverLabelError("label_provenance must be construction_derived.")

    expected_construction_id = _text(
        construction_value.get("construction_id"),
        field_name="construction.construction_id",
    )
    expected_construction_version = _version(
        construction_value.get("construction_version"),
        field_name="construction.construction_version",
    )
    accepted_construction_sha256 = _sha256(
        construction_sha256,
        field_name="construction_sha256 argument",
    )

    expected_batch_id = _text(
        batch_value.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )
    expected_batch_version = _version(
        batch_value.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    accepted_batch_sha256 = _sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256 argument",
    )

    expected_bindings = {
        "construction_id": expected_construction_id,
        "construction_version": expected_construction_version,
        "construction_sha256": accepted_construction_sha256,
        "annotation_batch_id": expected_batch_id,
        "annotation_batch_version": expected_batch_version,
        "annotation_batch_sha256": accepted_batch_sha256,
    }

    for field_name, expected in expected_bindings.items():
        actual = _text(
            value.get(field_name),
            field_name=field_name,
        )

        if actual != expected:
            raise EvidenceSufficiencySilverLabelError(
                f"{field_name} does not match the accepted binding."
            )

    construction_cases, construction_case_ids = _artifact_cases(
        construction_value,
        artifact_name="construction",
    )
    _, batch_case_ids = _artifact_cases(
        batch_value,
        artifact_name="annotation_batch",
    )

    if construction_case_ids != batch_case_ids:
        raise EvidenceSufficiencySilverLabelError(
            "construction and annotation batch case order differ."
        )

    labels = _sequence(
        value.get("labels"),
        field_name="labels",
    )
    case_count = _nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )

    if case_count != len(labels):
        raise EvidenceSufficiencySilverLabelError("case_count does not match labels.")

    if case_count != len(construction_cases):
        raise EvidenceSufficiencySilverLabelError(
            "silver labels do not cover every construction case."
        )

    seen_label_ids: set[str] = set()

    for position, (
        raw_label,
        construction_case,
        expected_case_id,
    ) in enumerate(
        zip(
            labels,
            construction_cases,
            construction_case_ids,
            strict=True,
        )
    ):
        label = _mapping(
            raw_label,
            field_name=f"labels[{position}]",
        )
        _reject_unknown(
            label,
            allowed=LABEL_FIELDS,
            object_name="silver label",
        )

        case_id = _text(
            label.get("case_id"),
            field_name=f"labels[{position}].case_id",
        )

        if case_id in seen_label_ids:
            raise EvidenceSufficiencySilverLabelError(f"duplicate silver-label case_id: {case_id}.")

        seen_label_ids.add(case_id)

        if case_id != expected_case_id:
            raise EvidenceSufficiencySilverLabelError(
                "silver-label order does not match construction order."
            )

        structures = set(
            _string_array(
                construction_case.get("evidence_structure_codes"),
                field_name=(f"construction {case_id}.evidence_structure_codes"),
            )
        )

        evidence_status = _text(
            label.get("evidence_status"),
            field_name=f"{case_id}.evidence_status",
        )
        response_action = _text(
            label.get("response_action"),
            field_name=f"{case_id}.response_action",
        )
        reason_codes = _string_array(
            label.get("reason_codes"),
            field_name=f"{case_id}.reason_codes",
        )
        missing_information = _string_array(
            label.get("missing_information"),
            field_name=f"{case_id}.missing_information",
        )
        derivation_rule = _text(
            label.get("derivation_rule"),
            field_name=f"{case_id}.derivation_rule",
        )

        if "incomplete_evidence_set" in structures:
            if "strict_subset_of_complete_evidence" not in structures:
                raise EvidenceSufficiencySilverLabelError(
                    f"{case_id}: incomplete construction must be a strict subset."
                )

            _text(
                construction_case.get("complete_reference_case_id"),
                field_name=(f"construction {case_id}.complete_reference_case_id"),
            )

            expected_label = (
                INSUFFICIENT_STATUS,
                ABSTAIN_ACTION,
                (INCOMPLETE_REASON_CODE,),
                (INCOMPLETE_MISSING_INFORMATION,),
                INCOMPLETE_DERIVATION_RULE,
            )
        else:
            if not (
                structures
                & {
                    "one_complete_passage",
                    "multiple_complementary_passages",
                }
            ):
                raise EvidenceSufficiencySilverLabelError(
                    f"{case_id}: complete construction structure is not declared."
                )

            expected_label = (
                SUFFICIENT_STATUS,
                ANSWER_ACTION,
                (),
                (),
                COMPLETE_DERIVATION_RULE,
            )

        actual_label = (
            evidence_status,
            response_action,
            reason_codes,
            missing_information,
            derivation_rule,
        )

        if actual_label != expected_label:
            raise EvidenceSufficiencySilverLabelError(
                f"{case_id}: silver label does not match its construction-derived rule."
            )


def write_silver_label_json_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Publish deterministic UTF-8 JSON without overwriting."""

    try:
        write_annotation_json_artifact(
            artifact,
            output_path,
        )
    except EvidenceSufficiencyAnnotationError as error:
        raise EvidenceSufficiencySilverLabelError(str(error)) from error
