"""Validation and receipts for immutable human-annotation submissions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from policyproof.evidence_sufficiency_annotation_round import (
    validate_annotation_assignment_package,
    validate_annotation_round_manifest,
)
from policyproof.evidence_sufficiency_annotations import (
    ANNOTATION_RECORD_SET_ID,
    ANNOTATION_RECORD_SET_SCHEMA_VERSION,
    EvidenceSufficiencyAnnotationError,
    validate_annotation_record_set,
)

ANNOTATION_SUBMISSION_RECEIPT_ID = (
    "policyproof-evidence-sufficiency-annotation-submission-receipt"
)
ANNOTATION_SUBMISSION_RECEIPT_SCHEMA_VERSION = "1.0"
ANNOTATION_INDEPENDENCE_ATTESTATION_ID = (
    "policyproof-evidence-sufficiency-annotation-independence-attestation"
)
ANNOTATION_INDEPENDENCE_ATTESTATION_SCHEMA_VERSION = "1.0"

_INDEPENDENCE_STATEMENT_ORDER = (
    "completed_without_collaboration",
    "did_not_view_other_annotations",
    "did_not_view_construction_or_silver_labels",
    "did_not_view_split_assignments",
    "did_not_view_retrieval_or_model_scores",
    "used_only_assigned_materials",
)
REQUIRED_INDEPENDENCE_STATEMENTS = frozenset(
    _INDEPENDENCE_STATEMENT_ORDER
)

ANNOTATION_SUBMISSION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "receipt_version",
        "annotation_round_id",
        "annotation_round_version",
        "annotation_round_sha256",
        "annotation_assignment_id",
        "annotation_assignment_version",
        "annotation_assignment_sha256",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "annotator_id",
        "record_set_id",
        "record_set_version",
        "record_set_sha256",
        "annotation_count",
        "intake_operator_id",
        "received_timestamp",
    }
)


ANNOTATION_INDEPENDENCE_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "attestation_id",
        "attestation_version",
        "annotation_round_id",
        "annotation_round_version",
        "annotation_round_sha256",
        "annotation_assignment_id",
        "annotation_assignment_version",
        "annotation_assignment_sha256",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "annotator_id",
        "record_set_id",
        "record_set_version",
        "record_set_sha256",
        "statements",
        "attested_timestamp",
    }
)


ANNOTATION_ROUND_COMPLETION_BUNDLE_FIELDS = frozenset(
    {
        "assignment_package",
        "assignment_package_sha256",
        "record_set",
        "record_set_sha256",
        "receipt",
        "attestation",
    }
)

_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


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


def _require_canonical_utc_timestamp(
    value: object,
    *,
    field_name: str,
) -> str:
    timestamp = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if _CANONICAL_UTC_PATTERN.fullmatch(timestamp) is None:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must use canonical UTC format "
            "YYYY-MM-DDTHH:MM:SSZ."
        )

    try:
        datetime.strptime(
            timestamp,
            "%Y-%m-%dT%H:%M:%SZ",
        )
    except ValueError as error:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must use canonical UTC format "
            "YYYY-MM-DDTHH:MM:SSZ."
        ) from error

    return timestamp


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
            f"{field_name} does not match the bound artifact."
        )


def _case_ids_from_assignment(
    assignment_package: Mapping[str, Any],
) -> tuple[str, ...]:
    cases = _require_sequence(
        assignment_package.get("cases"),
        field_name="assignment_package.cases",
    )
    case_ids: list[str] = []

    for position, raw_case in enumerate(cases):
        case = _require_mapping(
            raw_case,
            field_name=f"assignment_package.cases[{position}]",
        )
        case_ids.append(
            _require_nonempty_string(
                case.get("case_id"),
                field_name=(
                    f"assignment_package.cases"
                    f"[{position}].case_id"
                ),
            )
        )

    return tuple(case_ids)


def _case_ids_from_record_set(
    record_set: Mapping[str, Any],
) -> tuple[str, ...]:
    annotations = _require_sequence(
        record_set.get("annotations"),
        field_name="record_set.annotations",
    )
    case_ids: list[str] = []

    for position, raw_annotation in enumerate(annotations):
        annotation = _require_mapping(
            raw_annotation,
            field_name=f"record_set.annotations[{position}]",
        )
        case_ids.append(
            _require_nonempty_string(
                annotation.get("case_id"),
                field_name=(
                    f"record_set.annotations"
                    f"[{position}].case_id"
                ),
            )
        )

    return tuple(case_ids)


def _validated_independence_statements(
    value: object,
) -> dict[str, bool]:
    statements = _require_mapping(
        value,
        field_name="statements",
    )
    actual_fields = set(statements)

    if actual_fields != REQUIRED_INDEPENDENCE_STATEMENTS:
        missing = sorted(
            REQUIRED_INDEPENDENCE_STATEMENTS - actual_fields
        )
        unsupported = sorted(
            actual_fields - REQUIRED_INDEPENDENCE_STATEMENTS
        )
        raise EvidenceSufficiencyAnnotationError(
            "statements must contain exactly the required "
            "independence statements; "
            f"missing={missing}, unsupported={unsupported}."
        )

    accepted: dict[str, bool] = {}

    for statement_name in _INDEPENDENCE_STATEMENT_ORDER:
        statement_value = statements.get(statement_name)

        if not isinstance(statement_value, bool):
            raise EvidenceSufficiencyAnnotationError(
                f"{statement_name} must be a boolean."
            )

        accepted[statement_name] = statement_value

    return accepted


def independence_attestation_failures(
    attestation: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return failed independence statements in canonical order."""

    value = _require_mapping(
        attestation,
        field_name="independence attestation",
    )
    statements = _validated_independence_statements(
        value.get("statements")
    )

    return tuple(
        statement_name
        for statement_name in _INDEPENDENCE_STATEMENT_ORDER
        if not statements[statement_name]
    )


def build_annotation_independence_attestation(
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    assignment_package: Mapping[str, Any],
    assignment_package_sha256: str,
    record_set: Mapping[str, Any],
    record_set_sha256: str,
    attestation_version: str,
    statements: Mapping[str, bool],
    attested_timestamp: str,
) -> dict[str, Any]:
    """Build a record-set-bound independence attestation."""

    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    assignment = _require_mapping(
        assignment_package,
        field_name="assignment_package",
    )
    submission = _require_mapping(
        record_set,
        field_name="record_set",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )
    accepted_assignment_sha256 = _require_sha256(
        assignment_package_sha256,
        field_name="assignment_package_sha256",
    )
    accepted_record_set_sha256 = _require_sha256(
        record_set_sha256,
        field_name="record_set_sha256",
    )
    accepted_attestation_version = _require_semantic_version(
        attestation_version,
        field_name="attestation_version",
    )
    accepted_statements = _validated_independence_statements(
        statements
    )
    accepted_timestamp = _require_canonical_utc_timestamp(
        attested_timestamp,
        field_name="attested_timestamp",
    )

    validate_annotation_submission(
        submission,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
        assignment_package=assignment,
        assignment_package_sha256=accepted_assignment_sha256,
    )

    attestation: dict[str, Any] = {
        "schema_version": (
            ANNOTATION_INDEPENDENCE_ATTESTATION_SCHEMA_VERSION
        ),
        "attestation_id": ANNOTATION_INDEPENDENCE_ATTESTATION_ID,
        "attestation_version": accepted_attestation_version,
        "annotation_round_id": _require_nonempty_string(
            manifest.get("round_id"),
            field_name="round_manifest.round_id",
        ),
        "annotation_round_version": _require_semantic_version(
            manifest.get("round_version"),
            field_name="round_manifest.round_version",
        ),
        "annotation_round_sha256": accepted_round_sha256,
        "annotation_assignment_id": _require_nonempty_string(
            assignment.get("assignment_id"),
            field_name="assignment_package.assignment_id",
        ),
        "annotation_assignment_version": (
            _require_semantic_version(
                assignment.get("assignment_version"),
                field_name=(
                    "assignment_package.assignment_version"
                ),
            )
        ),
        "annotation_assignment_sha256": (
            accepted_assignment_sha256
        ),
        "annotation_batch_id": _require_nonempty_string(
            batch.get("batch_id"),
            field_name="annotation_batch.batch_id",
        ),
        "annotation_batch_version": _require_semantic_version(
            batch.get("batch_version"),
            field_name="annotation_batch.batch_version",
        ),
        "annotation_batch_sha256": accepted_batch_sha256,
        "annotator_id": _require_nonempty_string(
            submission.get("annotator_id"),
            field_name="record_set.annotator_id",
        ),
        "record_set_id": _require_nonempty_string(
            submission.get("record_set_id"),
            field_name="record_set.record_set_id",
        ),
        "record_set_version": _require_semantic_version(
            submission.get("record_set_version"),
            field_name="record_set.record_set_version",
        ),
        "record_set_sha256": accepted_record_set_sha256,
        "statements": accepted_statements,
        "attested_timestamp": accepted_timestamp,
    }

    validate_annotation_independence_attestation(
        attestation,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
        assignment_package=assignment,
        assignment_package_sha256=accepted_assignment_sha256,
        record_set=submission,
        record_set_sha256=accepted_record_set_sha256,
    )
    return attestation


def validate_annotation_independence_attestation(
    attestation: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    assignment_package: Mapping[str, Any],
    assignment_package_sha256: str,
    record_set: Mapping[str, Any],
    record_set_sha256: str,
) -> None:
    """Validate one truthful record-set-bound attestation."""

    value = _require_mapping(
        attestation,
        field_name="independence attestation",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    assignment = _require_mapping(
        assignment_package,
        field_name="assignment_package",
    )
    submission = _require_mapping(
        record_set,
        field_name="record_set",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=(
            ANNOTATION_INDEPENDENCE_ATTESTATION_FIELDS
        ),
        object_name="independence attestation",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )
    if (
        schema_version
        != ANNOTATION_INDEPENDENCE_ATTESTATION_SCHEMA_VERSION
    ):
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    attestation_id = _require_nonempty_string(
        value.get("attestation_id"),
        field_name="attestation_id",
    )
    if attestation_id != ANNOTATION_INDEPENDENCE_ATTESTATION_ID:
        raise EvidenceSufficiencyAnnotationError(
            "attestation_id is not the PolicyProof "
            "annotation-independence-attestation ID."
        )

    _require_semantic_version(
        value.get("attestation_version"),
        field_name="attestation_version",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )
    accepted_assignment_sha256 = _require_sha256(
        assignment_package_sha256,
        field_name="assignment_package_sha256",
    )
    accepted_record_set_sha256 = _require_sha256(
        record_set_sha256,
        field_name="record_set_sha256",
    )

    validate_annotation_submission(
        submission,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
        assignment_package=assignment,
        assignment_package_sha256=accepted_assignment_sha256,
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
            "annotation_assignment_id",
            _require_nonempty_string(
                assignment.get("assignment_id"),
                field_name="assignment_package.assignment_id",
            ),
        ),
        (
            "annotation_assignment_version",
            _require_semantic_version(
                assignment.get("assignment_version"),
                field_name=(
                    "assignment_package.assignment_version"
                ),
            ),
        ),
        (
            "annotation_assignment_sha256",
            accepted_assignment_sha256,
        ),
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
            "annotator_id",
            _require_nonempty_string(
                submission.get("annotator_id"),
                field_name="record_set.annotator_id",
            ),
        ),
        (
            "record_set_id",
            _require_nonempty_string(
                submission.get("record_set_id"),
                field_name="record_set.record_set_id",
            ),
        ),
        (
            "record_set_version",
            _require_semantic_version(
                submission.get("record_set_version"),
                field_name="record_set.record_set_version",
            ),
        ),
        ("record_set_sha256", accepted_record_set_sha256),
    )

    for field_name, expected_value in expected_bindings:
        _require_binding(
            value,
            field_name=field_name,
            expected_value=expected_value,
        )

    _validated_independence_statements(
        value.get("statements")
    )
    _require_canonical_utc_timestamp(
        value.get("attested_timestamp"),
        field_name="attested_timestamp",
    )


def validate_annotation_round_completion(
    submission_bundles: Sequence[Mapping[str, Any]],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
) -> None:
    """Validate readiness for pre-adjudication agreement analysis."""

    bundles = _require_sequence(
        submission_bundles,
        field_name="submission_bundles",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
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

    raw_primary_annotator_ids = _require_sequence(
        manifest.get("primary_annotator_ids"),
        field_name="round_manifest.primary_annotator_ids",
    )
    expected_annotator_ids = tuple(
        _require_nonempty_string(
            raw_annotator_id,
            field_name=(
                "round_manifest.primary_annotator_ids"
                f"[{position}]"
            ),
        )
        for position, raw_annotator_id
        in enumerate(raw_primary_annotator_ids)
    )

    prepared_bundles: list[
        tuple[
            str,
            Mapping[str, Any],
            str,
            Mapping[str, Any],
            str,
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ] = []
    seen_annotator_ids: set[str] = set()
    record_set_sha256s: list[str] = []

    for position, raw_bundle in enumerate(bundles):
        bundle = _require_mapping(
            raw_bundle,
            field_name=f"submission_bundles[{position}]",
        )
        _reject_unknown_fields(
            bundle,
            allowed_fields=(
                ANNOTATION_ROUND_COMPLETION_BUNDLE_FIELDS
            ),
            object_name=f"submission_bundles[{position}]",
        )

        assignment_package = _require_mapping(
            bundle.get("assignment_package"),
            field_name=(
                f"submission_bundles[{position}]"
                ".assignment_package"
            ),
        )
        assignment_package_sha256 = _require_sha256(
            bundle.get("assignment_package_sha256"),
            field_name=(
                f"submission_bundles[{position}]"
                ".assignment_package_sha256"
            ),
        )
        record_set = _require_mapping(
            bundle.get("record_set"),
            field_name=(
                f"submission_bundles[{position}].record_set"
            ),
        )
        record_set_sha256 = _require_sha256(
            bundle.get("record_set_sha256"),
            field_name=(
                f"submission_bundles[{position}]"
                ".record_set_sha256"
            ),
        )
        receipt = _require_mapping(
            bundle.get("receipt"),
            field_name=(
                f"submission_bundles[{position}].receipt"
            ),
        )
        attestation = _require_mapping(
            bundle.get("attestation"),
            field_name=(
                f"submission_bundles[{position}].attestation"
            ),
        )
        annotator_id = _require_nonempty_string(
            record_set.get("annotator_id"),
            field_name=(
                f"submission_bundles[{position}]"
                ".record_set.annotator_id"
            ),
        )

        if annotator_id in seen_annotator_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate submission annotator: {annotator_id}."
            )

        seen_annotator_ids.add(annotator_id)
        record_set_sha256s.append(record_set_sha256)
        prepared_bundles.append(
            (
                annotator_id,
                assignment_package,
                assignment_package_sha256,
                record_set,
                record_set_sha256,
                receipt,
                attestation,
            )
        )

    if seen_annotator_ids != set(expected_annotator_ids):
        raise EvidenceSufficiencyAnnotationError(
            "submission bundles must cover every primary "
            "annotator exactly once."
        )

    if len(set(record_set_sha256s)) != len(record_set_sha256s):
        raise EvidenceSufficiencyAnnotationError(
            "record_set_sha256 values must be distinct."
        )

    for (
        annotator_id,
        assignment_package,
        assignment_package_sha256,
        record_set,
        record_set_sha256,
        receipt,
        attestation,
    ) in prepared_bundles:
        validate_annotation_submission_receipt(
            receipt,
            annotation_batch=batch,
            annotation_batch_sha256=accepted_batch_sha256,
            round_manifest=manifest,
            round_manifest_sha256=accepted_round_sha256,
            assignment_package=assignment_package,
            assignment_package_sha256=(
                assignment_package_sha256
            ),
            record_set=record_set,
            record_set_sha256=record_set_sha256,
        )
        validate_annotation_independence_attestation(
            attestation,
            annotation_batch=batch,
            annotation_batch_sha256=accepted_batch_sha256,
            round_manifest=manifest,
            round_manifest_sha256=accepted_round_sha256,
            assignment_package=assignment_package,
            assignment_package_sha256=(
                assignment_package_sha256
            ),
            record_set=record_set,
            record_set_sha256=record_set_sha256,
        )

        failures = independence_attestation_failures(
            attestation
        )

        if failures:
            raise EvidenceSufficiencyAnnotationError(
                "independence attestation failed for "
                f"{annotator_id}: {', '.join(failures)}."
            )


def build_annotation_record_set_template(
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    assignment_package: Mapping[str, Any],
    assignment_package_sha256: str,
    record_set_version: str,
) -> dict[str, Any]:
    """Build an unfilled record-set template for one assignment."""

    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    assignment = _require_mapping(
        assignment_package,
        field_name="assignment_package",
    )
    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )
    _require_sha256(
        assignment_package_sha256,
        field_name="assignment_package_sha256",
    )
    accepted_record_set_version = _require_semantic_version(
        record_set_version,
        field_name="record_set_version",
    )

    validate_annotation_assignment_package(
        assignment,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
    )

    batch_id = _require_nonempty_string(
        batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )
    batch_version = _require_semantic_version(
        batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    guide_version = _require_semantic_version(
        batch.get("annotation_guide_version"),
        field_name="annotation_batch.annotation_guide_version",
    )
    guide_sha256 = _require_sha256(
        batch.get("annotation_guide_sha256"),
        field_name="annotation_batch.annotation_guide_sha256",
    )
    passage_sha256 = _require_sha256(
        batch.get("passage_artifact_sha256"),
        field_name="annotation_batch.passage_artifact_sha256",
    )
    annotator_id = _require_nonempty_string(
        assignment.get("annotator_id"),
        field_name="assignment_package.annotator_id",
    )
    assigned_case_ids = _case_ids_from_assignment(assignment)

    annotations = [
        {
            "annotation_id": f"{annotator_id}:{case_id}",
            "annotator_id": annotator_id,
            "annotation_guide_version": guide_version,
            "case_id": case_id,
            "evidence_status": None,
            "response_action": None,
            "reason_codes": [],
            "missing_information": [],
            "rationale": None,
            "uncertainty": None,
            "adjudication_note": None,
            "annotation_timestamp": None,
        }
        for case_id in assigned_case_ids
    ]

    return {
        "schema_version": ANNOTATION_RECORD_SET_SCHEMA_VERSION,
        "record_set_id": ANNOTATION_RECORD_SET_ID,
        "record_set_version": accepted_record_set_version,
        "annotation_batch_id": batch_id,
        "annotation_batch_version": batch_version,
        "annotation_batch_sha256": accepted_batch_sha256,
        "annotation_guide_version": guide_version,
        "annotation_guide_sha256": guide_sha256,
        "passage_artifact_sha256": passage_sha256,
        "annotator_id": annotator_id,
        "annotation_count": len(annotations),
        "annotations": annotations,
    }


def validate_annotation_submission(
    record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    assignment_package: Mapping[str, Any],
    assignment_package_sha256: str,
) -> None:
    """Validate a completed record set against its exact assignment."""

    submission = _require_mapping(
        record_set,
        field_name="record_set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    assignment = _require_mapping(
        assignment_package,
        field_name="assignment_package",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )
    _require_sha256(
        assignment_package_sha256,
        field_name="assignment_package_sha256",
    )

    validate_annotation_assignment_package(
        assignment,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
    )
    validate_annotation_record_set(
        submission,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
    )

    assignment_annotator_id = _require_nonempty_string(
        assignment.get("annotator_id"),
        field_name="assignment_package.annotator_id",
    )
    submission_annotator_id = _require_nonempty_string(
        submission.get("annotator_id"),
        field_name="record_set.annotator_id",
    )

    if submission_annotator_id != assignment_annotator_id:
        raise EvidenceSufficiencyAnnotationError(
            "submission annotator does not match assignment."
        )

    assigned_case_ids = _case_ids_from_assignment(
        assignment
    )
    submitted_case_ids = _case_ids_from_record_set(
        submission
    )

    if submitted_case_ids != assigned_case_ids:
        raise EvidenceSufficiencyAnnotationError(
            "annotations must preserve assigned case order."
        )


def build_annotation_submission_receipt(
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    assignment_package: Mapping[str, Any],
    assignment_package_sha256: str,
    record_set: Mapping[str, Any],
    record_set_sha256: str,
    receipt_version: str,
    intake_operator_id: str,
    received_timestamp: str,
) -> dict[str, Any]:
    """Build a metadata-only receipt for one validated submission."""

    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    assignment = _require_mapping(
        assignment_package,
        field_name="assignment_package",
    )
    submission = _require_mapping(
        record_set,
        field_name="record_set",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )
    accepted_assignment_sha256 = _require_sha256(
        assignment_package_sha256,
        field_name="assignment_package_sha256",
    )
    accepted_record_set_sha256 = _require_sha256(
        record_set_sha256,
        field_name="record_set_sha256",
    )
    accepted_receipt_version = _require_semantic_version(
        receipt_version,
        field_name="receipt_version",
    )
    accepted_operator_id = _require_nonempty_string(
        intake_operator_id,
        field_name="intake_operator_id",
    )
    accepted_timestamp = _require_canonical_utc_timestamp(
        received_timestamp,
        field_name="received_timestamp",
    )

    validate_annotation_submission(
        submission,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
        assignment_package=assignment,
        assignment_package_sha256=accepted_assignment_sha256,
    )

    receipt: dict[str, Any] = {
        "schema_version": (
            ANNOTATION_SUBMISSION_RECEIPT_SCHEMA_VERSION
        ),
        "receipt_id": ANNOTATION_SUBMISSION_RECEIPT_ID,
        "receipt_version": accepted_receipt_version,
        "annotation_round_id": _require_nonempty_string(
            manifest.get("round_id"),
            field_name="round_manifest.round_id",
        ),
        "annotation_round_version": _require_semantic_version(
            manifest.get("round_version"),
            field_name="round_manifest.round_version",
        ),
        "annotation_round_sha256": accepted_round_sha256,
        "annotation_assignment_id": _require_nonempty_string(
            assignment.get("assignment_id"),
            field_name="assignment_package.assignment_id",
        ),
        "annotation_assignment_version": (
            _require_semantic_version(
                assignment.get("assignment_version"),
                field_name=(
                    "assignment_package.assignment_version"
                ),
            )
        ),
        "annotation_assignment_sha256": (
            accepted_assignment_sha256
        ),
        "annotation_batch_id": _require_nonempty_string(
            batch.get("batch_id"),
            field_name="annotation_batch.batch_id",
        ),
        "annotation_batch_version": _require_semantic_version(
            batch.get("batch_version"),
            field_name="annotation_batch.batch_version",
        ),
        "annotation_batch_sha256": accepted_batch_sha256,
        "annotator_id": _require_nonempty_string(
            submission.get("annotator_id"),
            field_name="record_set.annotator_id",
        ),
        "record_set_id": _require_nonempty_string(
            submission.get("record_set_id"),
            field_name="record_set.record_set_id",
        ),
        "record_set_version": _require_semantic_version(
            submission.get("record_set_version"),
            field_name="record_set.record_set_version",
        ),
        "record_set_sha256": accepted_record_set_sha256,
        "annotation_count": _require_nonnegative_integer(
            submission.get("annotation_count"),
            field_name="record_set.annotation_count",
        ),
        "intake_operator_id": accepted_operator_id,
        "received_timestamp": accepted_timestamp,
    }

    validate_annotation_submission_receipt(
        receipt,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
        assignment_package=assignment,
        assignment_package_sha256=accepted_assignment_sha256,
        record_set=submission,
        record_set_sha256=accepted_record_set_sha256,
    )
    return receipt


def validate_annotation_submission_receipt(
    receipt: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    round_manifest: Mapping[str, Any],
    round_manifest_sha256: str,
    assignment_package: Mapping[str, Any],
    assignment_package_sha256: str,
    record_set: Mapping[str, Any],
    record_set_sha256: str,
) -> None:
    """Validate a metadata-only immutable-submission receipt."""

    value = _require_mapping(
        receipt,
        field_name="annotation submission receipt",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    manifest = _require_mapping(
        round_manifest,
        field_name="round_manifest",
    )
    assignment = _require_mapping(
        assignment_package,
        field_name="assignment_package",
    )
    submission = _require_mapping(
        record_set,
        field_name="record_set",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=ANNOTATION_SUBMISSION_RECEIPT_FIELDS,
        object_name="annotation submission receipt",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )
    if (
        schema_version
        != ANNOTATION_SUBMISSION_RECEIPT_SCHEMA_VERSION
    ):
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    receipt_id = _require_nonempty_string(
        value.get("receipt_id"),
        field_name="receipt_id",
    )
    if receipt_id != ANNOTATION_SUBMISSION_RECEIPT_ID:
        raise EvidenceSufficiencyAnnotationError(
            "receipt_id is not the PolicyProof "
            "annotation-submission-receipt ID."
        )

    _require_semantic_version(
        value.get("receipt_version"),
        field_name="receipt_version",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_round_sha256 = _require_sha256(
        round_manifest_sha256,
        field_name="round_manifest_sha256",
    )
    accepted_assignment_sha256 = _require_sha256(
        assignment_package_sha256,
        field_name="assignment_package_sha256",
    )
    accepted_record_set_sha256 = _require_sha256(
        record_set_sha256,
        field_name="record_set_sha256",
    )

    validate_annotation_submission(
        submission,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        round_manifest=manifest,
        round_manifest_sha256=accepted_round_sha256,
        assignment_package=assignment,
        assignment_package_sha256=accepted_assignment_sha256,
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
            "annotation_assignment_id",
            _require_nonempty_string(
                assignment.get("assignment_id"),
                field_name="assignment_package.assignment_id",
            ),
        ),
        (
            "annotation_assignment_version",
            _require_semantic_version(
                assignment.get("assignment_version"),
                field_name=(
                    "assignment_package.assignment_version"
                ),
            ),
        ),
        (
            "annotation_assignment_sha256",
            accepted_assignment_sha256,
        ),
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
            "annotator_id",
            _require_nonempty_string(
                submission.get("annotator_id"),
                field_name="record_set.annotator_id",
            ),
        ),
        (
            "record_set_id",
            _require_nonempty_string(
                submission.get("record_set_id"),
                field_name="record_set.record_set_id",
            ),
        ),
        (
            "record_set_version",
            _require_semantic_version(
                submission.get("record_set_version"),
                field_name="record_set.record_set_version",
            ),
        ),
        ("record_set_sha256", accepted_record_set_sha256),
    )

    for field_name, expected_value in expected_bindings:
        _require_binding(
            value,
            field_name=field_name,
            expected_value=expected_value,
        )

    annotation_count = _require_nonnegative_integer(
        value.get("annotation_count"),
        field_name="annotation_count",
    )
    expected_annotation_count = _require_nonnegative_integer(
        submission.get("annotation_count"),
        field_name="record_set.annotation_count",
    )

    if annotation_count != expected_annotation_count:
        raise EvidenceSufficiencyAnnotationError(
            "annotation_count does not match the record set."
        )

    _require_nonempty_string(
        value.get("intake_operator_id"),
        field_name="intake_operator_id",
    )
    _require_canonical_utc_timestamp(
        value.get("received_timestamp"),
        field_name="received_timestamp",
    )
