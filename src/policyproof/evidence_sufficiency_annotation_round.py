"""Operational manifest for an independent human-annotation round."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from policyproof.evidence_sufficiency_annotations import (
    BATCH_FIELDS,
    CASE_FIELDS,
    EVIDENCE_FIELDS,
    EvidenceSufficiencyAnnotationError,
)

ANNOTATION_ROUND_ID = (
    "policyproof-evidence-sufficiency-annotation-round"
)
ANNOTATION_ROUND_SCHEMA_VERSION = "1.0"
ANNOTATION_ASSIGNMENT_ID = (
    "policyproof-evidence-sufficiency-annotation-assignment"
)
ANNOTATION_ASSIGNMENT_SCHEMA_VERSION = "1.0"
FULL_OVERLAP_ASSIGNMENT_POLICY = "full_overlap"

ANNOTATION_ROUND_FIELDS = frozenset(
    {
        "schema_version",
        "round_id",
        "round_version",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "annotation_guide_version",
        "annotation_guide_sha256",
        "passage_artifact_sha256",
        "assignment_policy",
        "primary_annotator_count",
        "primary_annotator_ids",
        "adjudicator_id",
        "case_count",
        "assignment_count",
        "assignments",
    }
)

ANNOTATION_ASSIGNMENT_FIELDS = frozenset(
    {
        "annotator_id",
        "case_ids",
    }
)


ANNOTATION_PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "assignment_id",
        "assignment_version",
        "annotation_round_id",
        "annotation_round_version",
        "annotation_round_sha256",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "annotation_guide_version",
        "annotation_guide_sha256",
        "passage_artifact_sha256",
        "annotator_id",
        "case_count",
        "cases",
    }
)

_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be an object."
        )
    return value


def _require_sequence(
    value: object,
    *,
    field_name: str,
) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be an array."
        )
    return value


def _require_nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a nonempty string."
        )
    return value


def _require_nonnegative_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a nonnegative integer."
        )
    return value


def _require_semantic_version(
    value: object,
    *,
    field_name: str,
) -> str:
    version = _require_nonempty_string(
        value,
        field_name=field_name,
    )
    if _SEMANTIC_VERSION_PATTERN.fullmatch(version) is None:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a semantic version."
        )
    return version


def _require_sha256(
    value: object,
    *,
    field_name: str,
) -> str:
    digest = _require_nonempty_string(
        value,
        field_name=field_name,
    )
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )
    return digest


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str],
    object_name: str,
) -> None:
    unsupported_fields = sorted(
        set(value) - allowed_fields
    )
    if unsupported_fields:
        raise EvidenceSufficiencyAnnotationError(
            f"{object_name} contains unsupported fields: "
            f"{unsupported_fields}."
        )


def _require_binding(
    value: Mapping[str, Any],
    *,
    field_name: str,
    expected_value: str,
) -> None:
    actual_value = _require_nonempty_string(
        value.get(field_name),
        field_name=field_name,
    )
    if actual_value != expected_value:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} does not match the annotation batch."
        )


def _batch_contract(
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
]:
    _reject_unknown_fields(
        annotation_batch,
        allowed_fields=BATCH_FIELDS,
        object_name="annotation_batch",
    )

    batch_id = _require_nonempty_string(
        annotation_batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )
    batch_version = _require_semantic_version(
        annotation_batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    guide_version = _require_semantic_version(
        annotation_batch.get("annotation_guide_version"),
        field_name="annotation_batch.annotation_guide_version",
    )
    guide_sha256 = _require_sha256(
        annotation_batch.get("annotation_guide_sha256"),
        field_name="annotation_batch.annotation_guide_sha256",
    )
    passage_sha256 = _require_sha256(
        annotation_batch.get("passage_artifact_sha256"),
        field_name="annotation_batch.passage_artifact_sha256",
    )

    cases = _require_sequence(
        annotation_batch.get("cases"),
        field_name="annotation_batch.cases",
    )
    declared_case_count = _require_nonnegative_integer(
        annotation_batch.get("case_count"),
        field_name="annotation_batch.case_count",
    )

    if declared_case_count != len(cases):
        raise EvidenceSufficiencyAnnotationError(
            "annotation_batch.case_count does not match its cases."
        )

    case_ids: list[str] = []
    seen_case_ids: set[str] = set()

    for position, raw_case in enumerate(cases):
        case = _require_mapping(
            raw_case,
            field_name=f"annotation_batch.cases[{position}]",
        )
        case_name = f"annotation_batch.cases[{position}]"
        _reject_unknown_fields(
            case,
            allowed_fields=CASE_FIELDS,
            object_name=case_name,
        )

        case_id = _require_nonempty_string(
            case.get("case_id"),
            field_name=f"{case_name}.case_id",
        )
        _require_nonempty_string(
            case.get("query_id"),
            field_name=f"{case_name}.query_id",
        )
        _require_nonempty_string(
            case.get("question"),
            field_name=f"{case_name}.question",
        )

        evidence = _require_sequence(
            case.get("evidence"),
            field_name=f"{case_name}.evidence",
        )

        for evidence_position, raw_evidence in enumerate(evidence):
            evidence_item = _require_mapping(
                raw_evidence,
                field_name=(
                    f"{case_name}.evidence[{evidence_position}]"
                ),
            )
            evidence_name = (
                f"{case_name}.evidence[{evidence_position}]"
            )
            _reject_unknown_fields(
                evidence_item,
                allowed_fields=EVIDENCE_FIELDS,
                object_name=evidence_name,
            )

            for field_name in (
                "passage_id",
                "document_id",
                "label",
                "citation_text",
            ):
                _require_nonempty_string(
                    evidence_item.get(field_name),
                    field_name=f"{evidence_name}.{field_name}",
                )

        if case_id in seen_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"annotation batch contains duplicate case_id: "
                f"{case_id}."
            )

        seen_case_ids.add(case_id)
        case_ids.append(case_id)

    return (
        batch_id,
        batch_version,
        batch_sha256,
        guide_version,
        guide_sha256,
        passage_sha256,
        tuple(case_ids),
    )


def _primary_annotator_ids(
    value: object,
) -> tuple[str, ...]:
    raw_ids = _require_sequence(
        value,
        field_name="primary_annotator_ids",
    )
    annotator_ids = tuple(
        _require_nonempty_string(
            raw_id,
            field_name=(
                f"primary_annotator_ids[{position}]"
            ),
        )
        for position, raw_id in enumerate(raw_ids)
    )

    if (
        len(annotator_ids) < 2
        or len(set(annotator_ids)) != len(annotator_ids)
    ):
        raise EvidenceSufficiencyAnnotationError(
            "annotation round requires at least two distinct "
            "primary annotators."
        )

    return annotator_ids


def _assignment_case_orders(
    value: object,
    *,
    annotator_ids: tuple[str, ...],
    batch_case_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {
            annotator_id: batch_case_ids
            for annotator_id in annotator_ids
        }

    orders = _require_mapping(
        value,
        field_name="assignment_case_orders",
    )
    provided_annotator_ids = set(orders)
    expected_annotator_ids = set(annotator_ids)

    if provided_annotator_ids != expected_annotator_ids:
        raise EvidenceSufficiencyAnnotationError(
            "assignment_case_orders must contain exactly one "
            "entry for each primary annotator."
        )

    expected_case_id_set = set(batch_case_ids)
    accepted_orders: dict[str, tuple[str, ...]] = {}

    for annotator_id in annotator_ids:
        raw_case_ids = _require_sequence(
            orders.get(annotator_id),
            field_name=(
                f"assignment_case_orders[{annotator_id!r}]"
            ),
        )
        case_ids = tuple(
            _require_nonempty_string(
                raw_case_id,
                field_name=(
                    f"assignment_case_orders[{annotator_id!r}]"
                    f"[{position}]"
                ),
            )
            for position, raw_case_id in enumerate(raw_case_ids)
        )

        if (
            len(case_ids) != len(batch_case_ids)
            or len(set(case_ids)) != len(case_ids)
            or set(case_ids) != expected_case_id_set
        ):
            raise EvidenceSufficiencyAnnotationError(
                "assignment case order must contain every "
                "batch case exactly once."
            )

        accepted_orders[annotator_id] = case_ids

    return accepted_orders


def _assigned_case_ids(
    round_manifest: Mapping[str, Any],
    *,
    annotator_id: str,
) -> tuple[str, ...]:
    assignments = _require_sequence(
        round_manifest.get("assignments"),
        field_name="round_manifest.assignments",
    )

    for position, raw_assignment in enumerate(assignments):
        assignment = _require_mapping(
            raw_assignment,
            field_name=(
                f"round_manifest.assignments[{position}]"
            ),
        )
        assignment_annotator_id = _require_nonempty_string(
            assignment.get("annotator_id"),
            field_name=(
                f"round_manifest.assignments"
                f"[{position}].annotator_id"
            ),
        )

        if assignment_annotator_id != annotator_id:
            continue

        raw_case_ids = _require_sequence(
            assignment.get("case_ids"),
            field_name=(
                f"round_manifest.assignments"
                f"[{position}].case_ids"
            ),
        )
        return tuple(
            _require_nonempty_string(
                raw_case_id,
                field_name=(
                    f"round_manifest.assignments"
                    f"[{position}].case_ids[{case_position}]"
                ),
            )
            for case_position, raw_case_id
            in enumerate(raw_case_ids)
        )

    raise EvidenceSufficiencyAnnotationError(
        "annotator is not assigned in this round."
    )


def _batch_cases_by_id(
    annotation_batch: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw_cases = _require_sequence(
        annotation_batch.get("cases"),
        field_name="annotation_batch.cases",
    )
    cases_by_id: dict[str, Mapping[str, Any]] = {}

    for position, raw_case in enumerate(raw_cases):
        case = _require_mapping(
            raw_case,
            field_name=f"annotation_batch.cases[{position}]",
        )
        case_id = _require_nonempty_string(
            case.get("case_id"),
            field_name=(
                f"annotation_batch.cases[{position}].case_id"
            ),
        )

        if case_id in cases_by_id:
            raise EvidenceSufficiencyAnnotationError(
                f"annotation batch contains duplicate case_id: "
                f"{case_id}."
            )

        cases_by_id[case_id] = case

    return cases_by_id


def build_annotation_assignment_package(
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    annotator_id: str,
    assignment_version: str,
) -> dict[str, Any]:
    """Build one isolated blinded package for a primary annotator."""

    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=annotation_batch_sha256,
    )

    accepted_annotator_id = _require_nonempty_string(
        annotator_id,
        field_name="annotator_id",
    )
    accepted_assignment_version = _require_semantic_version(
        assignment_version,
        field_name="assignment_version",
    )
    assigned_case_ids = _assigned_case_ids(
        manifest,
        annotator_id=accepted_annotator_id,
    )
    cases_by_id = _batch_cases_by_id(batch)

    package: dict[str, Any] = {
        "schema_version": ANNOTATION_ASSIGNMENT_SCHEMA_VERSION,
        "assignment_id": ANNOTATION_ASSIGNMENT_ID,
        "assignment_version": accepted_assignment_version,
        "annotation_round_id": _require_nonempty_string(
            manifest.get("round_id"),
            field_name="round_manifest.round_id",
        ),
        "annotation_round_version": _require_semantic_version(
            manifest.get("round_version"),
            field_name="round_manifest.round_version",
        ),
        "annotation_round_sha256": accepted_round_sha256,
        "annotation_batch_id": _require_nonempty_string(
            batch.get("batch_id"),
            field_name="annotation_batch.batch_id",
        ),
        "annotation_batch_version": _require_semantic_version(
            batch.get("batch_version"),
            field_name="annotation_batch.batch_version",
        ),
        "annotation_batch_sha256": _require_sha256(
            annotation_batch_sha256,
            field_name="annotation_batch_sha256",
        ),
        "annotation_guide_version": _require_semantic_version(
            batch.get("annotation_guide_version"),
            field_name=(
                "annotation_batch.annotation_guide_version"
            ),
        ),
        "annotation_guide_sha256": _require_sha256(
            batch.get("annotation_guide_sha256"),
            field_name=(
                "annotation_batch.annotation_guide_sha256"
            ),
        ),
        "passage_artifact_sha256": _require_sha256(
            batch.get("passage_artifact_sha256"),
            field_name=(
                "annotation_batch.passage_artifact_sha256"
            ),
        ),
        "annotator_id": accepted_annotator_id,
        "case_count": len(assigned_case_ids),
        "cases": [
            deepcopy(cases_by_id[case_id])
            for case_id in assigned_case_ids
        ],
    }

    validate_annotation_assignment_package(
        package,
        annotation_batch=batch,
        annotation_batch_sha256=annotation_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
    )
    return package


def validate_annotation_assignment_package(
    package: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
) -> None:
    """Validate one isolated blinded annotator assignment package."""

    value = _require_mapping(
        package,
        field_name="annotation assignment package",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=ANNOTATION_PACKAGE_FIELDS,
        object_name="annotation assignment package",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )
    if schema_version != ANNOTATION_ASSIGNMENT_SCHEMA_VERSION:
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    assignment_id = _require_nonempty_string(
        value.get("assignment_id"),
        field_name="assignment_id",
    )
    if assignment_id != ANNOTATION_ASSIGNMENT_ID:
        raise EvidenceSufficiencyAnnotationError(
            "assignment_id is not the PolicyProof "
            "annotation-assignment ID."
        )

    _require_semantic_version(
        value.get("assignment_version"),
        field_name="assignment_version",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
    )

    expected_bindings = (
        (
            "annotation_round_id",
            _require_nonempty_string(
                manifest.get("round_id"),
                field_name="round_manifest.round_id",
            ),
        ),
        (
            "annotation_round_version",
            _require_semantic_version(
                manifest.get("round_version"),
                field_name="round_manifest.round_version",
            ),
        ),
        ("annotation_round_sha256", accepted_round_sha256),
        (
            "annotation_batch_id",
            _require_nonempty_string(
                batch.get("batch_id"),
                field_name="annotation_batch.batch_id",
            ),
        ),
        (
            "annotation_batch_version",
            _require_semantic_version(
                batch.get("batch_version"),
                field_name="annotation_batch.batch_version",
            ),
        ),
        ("annotation_batch_sha256", accepted_batch_sha256),
        (
            "annotation_guide_version",
            _require_semantic_version(
                batch.get("annotation_guide_version"),
                field_name=(
                    "annotation_batch.annotation_guide_version"
                ),
            ),
        ),
        (
            "annotation_guide_sha256",
            _require_sha256(
                batch.get("annotation_guide_sha256"),
                field_name=(
                    "annotation_batch.annotation_guide_sha256"
                ),
            ),
        ),
        (
            "passage_artifact_sha256",
            _require_sha256(
                batch.get("passage_artifact_sha256"),
                field_name=(
                    "annotation_batch.passage_artifact_sha256"
                ),
            ),
        ),
    )

    for field_name, expected_value in expected_bindings:
        _require_binding(
            value,
            field_name=field_name,
            expected_value=expected_value,
        )

    annotator_id = _require_nonempty_string(
        value.get("annotator_id"),
        field_name="annotator_id",
    )
    assigned_case_ids = _assigned_case_ids(
        manifest,
        annotator_id=annotator_id,
    )
    cases_by_id = _batch_cases_by_id(batch)
    expected_cases = [
        cases_by_id[case_id]
        for case_id in assigned_case_ids
    ]

    raw_cases = _require_sequence(
        value.get("cases"),
        field_name="cases",
    )
    case_count = _require_nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )

    if case_count != len(raw_cases):
        raise EvidenceSufficiencyAnnotationError(
            "case_count does not match cases."
        )

    actual_cases = list(raw_cases)
    if actual_cases != expected_cases:
        raise EvidenceSufficiencyAnnotationError(
            "cases must match the assigned case order and "
            "frozen batch snapshots."
        )


def build_annotation_round_manifest(
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_version: str,
    primary_annotator_ids: Sequence[str],
    adjudicator_id: str,
    assignment_case_orders: (
        Mapping[str, Sequence[str]] | None
    ) = None,
) -> dict[str, Any]:
    """Build a full-overlap human-annotation round manifest."""

    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    (
        batch_id,
        batch_version,
        batch_sha256,
        guide_version,
        guide_sha256,
        passage_sha256,
        case_ids,
    ) = _batch_contract(
        batch,
        annotation_batch_sha256,
    )

    accepted_round_version = _require_semantic_version(
        round_version,
        field_name="round_version",
    )
    annotator_ids = _primary_annotator_ids(
        primary_annotator_ids
    )
    accepted_adjudicator_id = _require_nonempty_string(
        adjudicator_id,
        field_name="adjudicator_id",
    )

    if accepted_adjudicator_id in annotator_ids:
        raise EvidenceSufficiencyAnnotationError(
            "adjudicator must be distinct from primary annotators."
        )

    accepted_case_orders = _assignment_case_orders(
        assignment_case_orders,
        annotator_ids=annotator_ids,
        batch_case_ids=case_ids,
    )

    manifest: dict[str, Any] = {
        "schema_version": ANNOTATION_ROUND_SCHEMA_VERSION,
        "round_id": ANNOTATION_ROUND_ID,
        "round_version": accepted_round_version,
        "annotation_batch_id": batch_id,
        "annotation_batch_version": batch_version,
        "annotation_batch_sha256": batch_sha256,
        "annotation_guide_version": guide_version,
        "annotation_guide_sha256": guide_sha256,
        "passage_artifact_sha256": passage_sha256,
        "assignment_policy": (
            FULL_OVERLAP_ASSIGNMENT_POLICY
        ),
        "primary_annotator_count": len(annotator_ids),
        "primary_annotator_ids": list(annotator_ids),
        "adjudicator_id": accepted_adjudicator_id,
        "case_count": len(case_ids),
        "assignment_count": len(annotator_ids),
        "assignments": [
            {
                "annotator_id": annotator_id,
                "case_ids": list(
                    accepted_case_orders[annotator_id]
                ),
            }
            for annotator_id in annotator_ids
        ],
    }

    validate_annotation_round_manifest(
        manifest,
        annotation_batch=batch,
        annotation_batch_sha256=batch_sha256,
    )
    return manifest


def validate_annotation_round_manifest(
    manifest: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
) -> None:
    """Validate a blinded full-overlap annotation-round contract."""

    value = _require_mapping(
        manifest,
        field_name="annotation round manifest",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=ANNOTATION_ROUND_FIELDS,
        object_name="annotation round manifest",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )
    if schema_version != ANNOTATION_ROUND_SCHEMA_VERSION:
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    round_id = _require_nonempty_string(
        value.get("round_id"),
        field_name="round_id",
    )
    if round_id != ANNOTATION_ROUND_ID:
        raise EvidenceSufficiencyAnnotationError(
            "round_id is not the PolicyProof annotation-round ID."
        )

    _require_semantic_version(
        value.get("round_version"),
        field_name="round_version",
    )

    (
        batch_id,
        batch_version,
        batch_sha256,
        guide_version,
        guide_sha256,
        passage_sha256,
        batch_case_ids,
    ) = _batch_contract(
        batch,
        annotation_batch_sha256,
    )

    for field_name, expected_value in (
        ("annotation_batch_id", batch_id),
        ("annotation_batch_version", batch_version),
        ("annotation_batch_sha256", batch_sha256),
        ("annotation_guide_version", guide_version),
        ("annotation_guide_sha256", guide_sha256),
        ("passage_artifact_sha256", passage_sha256),
    ):
        _require_binding(
            value,
            field_name=field_name,
            expected_value=expected_value,
        )

    assignment_policy = _require_nonempty_string(
        value.get("assignment_policy"),
        field_name="assignment_policy",
    )
    if assignment_policy != FULL_OVERLAP_ASSIGNMENT_POLICY:
        raise EvidenceSufficiencyAnnotationError(
            "assignment_policy must be full_overlap."
        )

    annotator_ids = _primary_annotator_ids(
        value.get("primary_annotator_ids")
    )
    adjudicator_id = _require_nonempty_string(
        value.get("adjudicator_id"),
        field_name="adjudicator_id",
    )

    if adjudicator_id in annotator_ids:
        raise EvidenceSufficiencyAnnotationError(
            "adjudicator must be distinct from primary annotators."
        )

    primary_annotator_count = _require_nonnegative_integer(
        value.get("primary_annotator_count"),
        field_name="primary_annotator_count",
    )
    if primary_annotator_count != len(annotator_ids):
        raise EvidenceSufficiencyAnnotationError(
            "primary_annotator_count does not match "
            "primary_annotator_ids."
        )

    case_count = _require_nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )
    if case_count != len(batch_case_ids):
        raise EvidenceSufficiencyAnnotationError(
            "case_count does not match the annotation batch."
        )

    assignments = _require_sequence(
        value.get("assignments"),
        field_name="assignments",
    )
    assignment_count = _require_nonnegative_integer(
        value.get("assignment_count"),
        field_name="assignment_count",
    )

    if assignment_count != len(assignments):
        raise EvidenceSufficiencyAnnotationError(
            "assignment_count does not match assignments."
        )

    if assignment_count != len(annotator_ids):
        raise EvidenceSufficiencyAnnotationError(
            "each primary annotator must have one assignment."
        )

    assignment_ids: list[str] = []

    for position, raw_assignment in enumerate(assignments):
        assignment = _require_mapping(
            raw_assignment,
            field_name=f"assignments[{position}]",
        )
        _reject_unknown_fields(
            assignment,
            allowed_fields=ANNOTATION_ASSIGNMENT_FIELDS,
            object_name=f"assignments[{position}]",
        )

        annotator_id = _require_nonempty_string(
            assignment.get("annotator_id"),
            field_name=f"assignments[{position}].annotator_id",
        )
        assignment_ids.append(annotator_id)

        raw_case_ids = _require_sequence(
            assignment.get("case_ids"),
            field_name=f"assignments[{position}].case_ids",
        )
        assigned_case_ids = tuple(
            _require_nonempty_string(
                raw_case_id,
                field_name=(
                    f"assignments[{position}].case_ids"
                    f"[{case_position}]"
                ),
            )
            for case_position, raw_case_id
            in enumerate(raw_case_ids)
        )

        if (
            len(assigned_case_ids) != len(batch_case_ids)
            or len(set(assigned_case_ids))
            != len(assigned_case_ids)
            or set(assigned_case_ids) != set(batch_case_ids)
        ):
            raise EvidenceSufficiencyAnnotationError(
                "every primary annotator must receive every "
                "batch case exactly once."
            )

    if (
        len(set(assignment_ids)) != len(assignment_ids)
        or set(assignment_ids) != set(annotator_ids)
    ):
        raise EvidenceSufficiencyAnnotationError(
            "assignments must cover each primary annotator "
            "exactly once."
        )
