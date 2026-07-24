"""Validation for blinded evidence-sufficiency annotation batches."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ANNOTATION_BATCH_ID = "policyproof-evidence-sufficiency-annotation-batch"
ANNOTATION_BATCH_SCHEMA_VERSION = "1.0"
ANNOTATION_RECORD_SET_ID = (
    "policyproof-evidence-sufficiency-annotation-record-set"
)
ANNOTATION_RECORD_SET_SCHEMA_VERSION = "1.0"
ADJUDICATION_RECORD_SET_ID = (
    "policyproof-evidence-sufficiency-adjudication-record-set"
)
ADJUDICATION_RECORD_SET_SCHEMA_VERSION = "1.0"
ANNOTATION_ANALYSIS_METADATA_ID = (
    "policyproof-evidence-sufficiency-annotation-analysis-metadata"
)
ANNOTATION_ANALYSIS_METADATA_SCHEMA_VERSION = "1.0"
ANNOTATION_AGREEMENT_REPORT_ID = (
    "policyproof-evidence-sufficiency-annotation-agreement-report"
)
ANNOTATION_AGREEMENT_REPORT_SCHEMA_VERSION = "1.0"
ANNOTATION_STRUCTURE_DISAGREEMENT_REPORT_ID = (
    "policyproof-evidence-sufficiency-"
    "annotation-structure-disagreement-report"
)
ANNOTATION_STRUCTURE_DISAGREEMENT_REPORT_SCHEMA_VERSION = "1.0"

ALLOWED_QUESTION_STRUCTURE_CODES = frozenset(
    {
        "direct_factual_lookup",
        "definition",
        "factual_list",
        "risk_and_mitigation",
        "process_or_evaluation_method",
        "policy_interpretation",
        "legal_classification",
        "legal_obligations",
        "comparison",
        "multi_part_question",
    }
)

ALLOWED_EVIDENCE_STRUCTURE_CODES = frozenset(
    {
        "one_complete_passage",
        "multiple_complementary_passages",
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
        "topically_related_distractors",
        "multiple_documents",
    }
)

ANNOTATION_ANALYSIS_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "metadata_id",
        "metadata_version",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "case_count",
        "cases",
    }
)

ANNOTATION_ANALYSIS_METADATA_CASE_FIELDS = frozenset(
    {
        "case_id",
        "question_structure_codes",
        "evidence_structure_codes",
    }
)

ALLOWED_DISAGREEMENT_CATEGORIES = frozenset(
    {
        "question_decomposition_disagreement",
        "evidence_interpretation_disagreement",
        "implicit_inference_disagreement",
        "completeness_disagreement",
        "reason_code_disagreement",
        "missing_information_disagreement",
        "source_boundary_disagreement",
        "legal_scope_disagreement",
        "current_information_disagreement",
        "annotation_error",
        "guide_ambiguity",
        "source_extraction_defect",
    }
)

ADJUDICATION_RECORD_SET_FIELDS = frozenset(
    {
        "schema_version",
        "adjudication_set_id",
        "adjudication_set_version",
        "annotation_batch_id",
        "annotation_batch_version",
        "annotation_batch_sha256",
        "annotation_guide_version",
        "annotation_guide_sha256",
        "first_record_set_id",
        "first_record_set_version",
        "first_record_set_sha256",
        "second_record_set_id",
        "second_record_set_version",
        "second_record_set_sha256",
        "adjudicator_id",
        "record_count",
        "records",
    }
)

ADJUDICATION_RECORD_FIELDS = frozenset(
    {
        "adjudication_id",
        "case_id",
        "first_annotation_id",
        "second_annotation_id",
        "disagreement_categories",
        "final_evidence_status",
        "final_response_action",
        "final_reason_codes",
        "final_missing_information",
        "final_rationale",
        "adjudication_rationale",
        "guide_change_required",
        "guide_change_summary",
        "adjudication_timestamp",
    }
)

SUFFICIENT_STATUS = "sufficient"
INSUFFICIENT_STATUS = "insufficient"
ANSWER_ACTION = "answer"
ABSTAIN_ACTION = "abstain"

ALLOWED_EVIDENCE_STATUSES = frozenset(
    {
        SUFFICIENT_STATUS,
        INSUFFICIENT_STATUS,
    }
)

ALLOWED_RESPONSE_ACTIONS = frozenset(
    {
        ANSWER_ACTION,
        ABSTAIN_ACTION,
    }
)

ALLOWED_REASON_CODES = frozenset(
    {
        "outside_controlled_corpus",
        "current_information_required",
        "organization_specific_conclusion",
        "legal_advice_boundary",
        "high_stakes_recommendation",
        "unsupported_comparison",
        "incomplete_evidence_set",
        "conflicting_evidence",
    }
)

ANNOTATION_RECORD_SET_FIELDS = frozenset(
    {
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
)

ANNOTATION_RECORD_FIELDS = frozenset(
    {
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
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

BATCH_FIELDS = frozenset(
    {
        "schema_version",
        "batch_id",
        "batch_version",
        "annotation_guide_version",
        "annotation_guide_sha256",
        "corpus_id",
        "corpus_version",
        "passage_schema_version",
        "passage_artifact_sha256",
        "case_count",
        "cases",
    }
)

CASE_FIELDS = frozenset(
    {
        "case_id",
        "query_id",
        "question",
        "evidence",
    }
)

EVIDENCE_FIELDS = frozenset(
    {
        "passage_id",
        "document_id",
        "label",
        "citation_text",
    }
)

FORBIDDEN_CASE_FIELDS = frozenset(
    {
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
    }
)


class EvidenceSufficiencyAnnotationError(ValueError):
    """Raised when an annotation batch violates its blinded contract."""


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be an object."
        )

    return value


def _require_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be an array."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a nonempty string."
        )

    return value


def _require_version(
    value: Any,
    *,
    field_name: str,
) -> str:
    version = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if not VERSION_PATTERN.fullmatch(version):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must use semantic version form X.Y.Z."
        )

    return version


def _require_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    checksum = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if not SHA256_PATTERN.fullmatch(checksum):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return checksum


def _require_nonnegative_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be a non-negative integer."
        )

    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str],
    object_name: str,
) -> None:
    unknown_fields = sorted(set(value) - allowed_fields)

    if unknown_fields:
        raise EvidenceSufficiencyAnnotationError(
            f"unknown {object_name} fields: {unknown_fields}."
        )


def _require_binding(
    value: Mapping[str, Any],
    *,
    field_name: str,
    expected_value: str,
) -> str:
    actual_value = _require_nonempty_string(
        value.get(field_name),
        field_name=field_name,
    )

    if actual_value != expected_value:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} does not match the accepted binding."
        )

    return actual_value


def _accepted_passage_contract(
    passages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    values = _require_sequence(
        passages,
        field_name="passages",
    )
    result: dict[str, Mapping[str, Any]] = {}
    passage_schema_version: str | None = None

    if not values:
        raise EvidenceSufficiencyAnnotationError(
            "passages must be nonempty."
        )

    for position, raw_passage in enumerate(values):
        passage = _require_mapping(
            raw_passage,
            field_name=f"passages[{position}]",
        )
        passage_id = _require_nonempty_string(
            passage.get("passage_id"),
            field_name=f"passages[{position}].passage_id",
        )
        schema_version = _require_nonempty_string(
            passage.get("schema_version"),
            field_name=f"passages[{position}].schema_version",
        )

        for field_name in (
            "document_id",
            "label",
            "citation_text",
        ):
            _require_nonempty_string(
                passage.get(field_name),
                field_name=(
                    f"passages[{position}].{field_name}"
                ),
            )

        if passage_id in result:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate passage_id: {passage_id}."
            )

        if passage_schema_version is None:
            passage_schema_version = schema_version
        elif schema_version != passage_schema_version:
            raise EvidenceSufficiencyAnnotationError(
                "passages contain inconsistent schema_version values."
            )

        result[passage_id] = passage

    if passage_schema_version is None:
        raise EvidenceSufficiencyAnnotationError(
            "passages must be nonempty."
        )

    return result, passage_schema_version


def _validate_case(
    raw_case: Any,
    *,
    position: int,
    accepted_passages: Mapping[str, Mapping[str, Any]],
    seen_case_ids: set[str],
) -> None:
    case = _require_mapping(
        raw_case,
        field_name=f"cases[{position}]",
    )

    exposed_fields = sorted(
        set(case) & FORBIDDEN_CASE_FIELDS
    )

    if exposed_fields:
        raise EvidenceSufficiencyAnnotationError(
            "annotation batch cases must not expose hidden label or "
            f"evaluation fields: {exposed_fields}."
        )

    _reject_unknown_fields(
        case,
        allowed_fields=CASE_FIELDS,
        object_name="annotation case",
    )

    case_id = _require_nonempty_string(
        case.get("case_id"),
        field_name=f"cases[{position}].case_id",
    )

    if case_id in seen_case_ids:
        raise EvidenceSufficiencyAnnotationError(
            f"Duplicate case_id: {case_id}."
        )

    seen_case_ids.add(case_id)

    _require_nonempty_string(
        case.get("query_id"),
        field_name=f"{case_id}.query_id",
    )
    _require_nonempty_string(
        case.get("question"),
        field_name=f"{case_id}.question",
    )

    evidence_values = _require_sequence(
        case.get("evidence"),
        field_name=f"{case_id}.evidence",
    )

    if not evidence_values:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}.evidence must be nonempty."
        )

    seen_evidence_ids: set[str] = set()

    for evidence_position, raw_evidence in enumerate(
        evidence_values
    ):
        evidence = _require_mapping(
            raw_evidence,
            field_name=(
                f"{case_id}.evidence[{evidence_position}]"
            ),
        )
        _reject_unknown_fields(
            evidence,
            allowed_fields=EVIDENCE_FIELDS,
            object_name="annotation evidence",
        )

        passage_id = _require_nonempty_string(
            evidence.get("passage_id"),
            field_name=(
                f"{case_id}.evidence"
                f"[{evidence_position}].passage_id"
            ),
        )

        if passage_id in seen_evidence_ids:
            raise EvidenceSufficiencyAnnotationError(
                "duplicate evidence passage_id: "
                f"{passage_id}."
            )

        seen_evidence_ids.add(passage_id)

        if passage_id not in accepted_passages:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}.evidence[{evidence_position}] "
                "contains unknown passage_id: "
                f"{passage_id}."
            )

        accepted_passage = accepted_passages[passage_id]

        for field_name in (
            "document_id",
            "label",
            "citation_text",
        ):
            actual_value = _require_nonempty_string(
                evidence.get(field_name),
                field_name=(
                    f"{case_id}.evidence"
                    f"[{evidence_position}].{field_name}"
                ),
            )
            expected_value = _require_nonempty_string(
                accepted_passage.get(field_name),
                field_name=(
                    f"accepted passage {passage_id}."
                    f"{field_name}"
                ),
            )

            if actual_value != expected_value:
                raise EvidenceSufficiencyAnnotationError(
                    f"{field_name} does not match accepted passage "
                    f"{passage_id}."
                )


def validate_annotation_batch(
    batch: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
    passage_artifact_sha256: str,
    annotation_guide_version: str,
    annotation_guide_sha256: str,
) -> None:
    """Validate one immutable, label-blinded annotation batch."""

    value = _require_mapping(
        batch,
        field_name="annotation batch",
    )
    corpus_manifest = _require_mapping(
        manifest,
        field_name="manifest",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=BATCH_FIELDS,
        object_name="annotation batch",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != ANNOTATION_BATCH_SCHEMA_VERSION:
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    batch_id = _require_nonempty_string(
        value.get("batch_id"),
        field_name="batch_id",
    )

    if batch_id != ANNOTATION_BATCH_ID:
        raise EvidenceSufficiencyAnnotationError(
            "batch_id is not the evidence-sufficiency "
            "annotation-batch ID."
        )

    _require_version(
        value.get("batch_version"),
        field_name="batch_version",
    )

    accepted_guide_version = _require_version(
        annotation_guide_version,
        field_name="annotation_guide_version argument",
    )
    accepted_guide_sha256 = _require_sha256(
        annotation_guide_sha256,
        field_name="annotation_guide_sha256 argument",
    )
    accepted_passage_sha256 = _require_sha256(
        passage_artifact_sha256,
        field_name="passage_artifact_sha256 argument",
    )

    _require_binding(
        value,
        field_name="annotation_guide_version",
        expected_value=accepted_guide_version,
    )
    _require_binding(
        value,
        field_name="annotation_guide_sha256",
        expected_value=accepted_guide_sha256,
    )

    expected_corpus_id = _require_nonempty_string(
        corpus_manifest.get("corpus_id"),
        field_name="manifest.corpus_id",
    )
    expected_corpus_version = _require_nonempty_string(
        corpus_manifest.get("corpus_version"),
        field_name="manifest.corpus_version",
    )

    _require_binding(
        value,
        field_name="corpus_id",
        expected_value=expected_corpus_id,
    )
    _require_binding(
        value,
        field_name="corpus_version",
        expected_value=expected_corpus_version,
    )
    accepted_passages, expected_passage_schema_version = (
        _accepted_passage_contract(passages)
    )
    _require_binding(
        value,
        field_name="passage_schema_version",
        expected_value=expected_passage_schema_version,
    )
    _require_binding(
        value,
        field_name="passage_artifact_sha256",
        expected_value=accepted_passage_sha256,
    )

    cases = _require_sequence(
        value.get("cases"),
        field_name="cases",
    )

    if not cases:
        raise EvidenceSufficiencyAnnotationError(
            "cases must be nonempty."
        )

    case_count = _require_nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )

    if case_count != len(cases):
        raise EvidenceSufficiencyAnnotationError(
            "case_count does not match the number of cases."
        )

    seen_case_ids: set[str] = set()

    for position, raw_case in enumerate(cases):
        _validate_case(
            raw_case,
            position=position,
            accepted_passages=accepted_passages,
            seen_case_ids=seen_case_ids,
        )


def _require_unique_strings(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    sequence = _require_sequence(
        value,
        field_name=field_name,
    )

    if not sequence and not allow_empty:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must be nonempty."
        )

    result: list[str] = []
    seen: set[str] = set()

    for position, item in enumerate(sequence):
        text = _require_nonempty_string(
            item,
            field_name=f"{field_name}[{position}]",
        )

        if text in seen:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate {field_name}: {text}."
            )

        seen.add(text)
        result.append(text)

    return tuple(result)


def _require_optional_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    return _require_nonempty_string(
        value,
        field_name=field_name,
    )


def _validate_annotation_relationships(
    *,
    case_id: str,
    evidence_status: str,
    response_action: str,
    reason_codes: tuple[str, ...],
    missing_information: tuple[str, ...],
) -> None:
    if evidence_status == SUFFICIENT_STATUS:
        if response_action != ANSWER_ACTION:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}: sufficient annotations require "
                "answer action."
            )

        if reason_codes:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}: sufficient annotations require "
                "empty reason_codes."
            )

        if missing_information:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}: sufficient annotations require "
                "empty missing_information."
            )

        return

    if response_action != ABSTAIN_ACTION:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}: insufficient annotations require "
            "abstain action."
        )

    if not reason_codes:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}: insufficient annotations require "
            "reason_codes."
        )

    if not missing_information:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}: insufficient annotations require "
            "missing_information."
        )


def validate_annotation_record_set(
    record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
) -> None:
    """Validate one independent annotator's raw batch submission."""

    value = _require_mapping(
        record_set,
        field_name="annotation record set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=ANNOTATION_RECORD_SET_FIELDS,
        object_name="annotation record set",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != ANNOTATION_RECORD_SET_SCHEMA_VERSION:
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    record_set_id = _require_nonempty_string(
        value.get("record_set_id"),
        field_name="record_set_id",
    )

    if record_set_id != ANNOTATION_RECORD_SET_ID:
        raise EvidenceSufficiencyAnnotationError(
            "record_set_id is not the evidence-sufficiency "
            "annotation-record-set ID."
        )

    _require_version(
        value.get("record_set_version"),
        field_name="record_set_version",
    )

    expected_batch_id = _require_nonempty_string(
        batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )

    if expected_batch_id != ANNOTATION_BATCH_ID:
        raise EvidenceSufficiencyAnnotationError(
            "annotation_batch.batch_id is not supported."
        )

    expected_batch_version = _require_version(
        batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256 argument",
    )
    expected_guide_version = _require_version(
        batch.get("annotation_guide_version"),
        field_name="annotation_batch.annotation_guide_version",
    )
    expected_guide_sha256 = _require_sha256(
        batch.get("annotation_guide_sha256"),
        field_name="annotation_batch.annotation_guide_sha256",
    )
    expected_passage_sha256 = _require_sha256(
        batch.get("passage_artifact_sha256"),
        field_name="annotation_batch.passage_artifact_sha256",
    )

    _require_binding(
        value,
        field_name="annotation_batch_id",
        expected_value=expected_batch_id,
    )
    _require_binding(
        value,
        field_name="annotation_batch_version",
        expected_value=expected_batch_version,
    )
    _require_binding(
        value,
        field_name="annotation_batch_sha256",
        expected_value=accepted_batch_sha256,
    )
    _require_binding(
        value,
        field_name="annotation_guide_version",
        expected_value=expected_guide_version,
    )
    _require_binding(
        value,
        field_name="annotation_guide_sha256",
        expected_value=expected_guide_sha256,
    )
    _require_binding(
        value,
        field_name="passage_artifact_sha256",
        expected_value=expected_passage_sha256,
    )

    submission_annotator_id = _require_nonempty_string(
        value.get("annotator_id"),
        field_name="annotator_id",
    )

    batch_cases = _require_sequence(
        batch.get("cases"),
        field_name="annotation_batch.cases",
    )
    batch_case_ids: set[str] = set()

    for position, raw_case in enumerate(batch_cases):
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

        if case_id in batch_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"annotation_batch contains duplicate case_id: "
                f"{case_id}."
            )

        batch_case_ids.add(case_id)

    annotations = _require_sequence(
        value.get("annotations"),
        field_name="annotations",
    )
    annotation_count = _require_nonnegative_integer(
        value.get("annotation_count"),
        field_name="annotation_count",
    )

    if annotation_count != len(annotations):
        raise EvidenceSufficiencyAnnotationError(
            "annotation_count does not match the number of "
            "annotations."
        )

    seen_annotation_ids: set[str] = set()
    seen_case_ids: set[str] = set()

    for position, raw_annotation in enumerate(annotations):
        annotation = _require_mapping(
            raw_annotation,
            field_name=f"annotations[{position}]",
        )
        _reject_unknown_fields(
            annotation,
            allowed_fields=ANNOTATION_RECORD_FIELDS,
            object_name="annotation record",
        )

        annotation_id = _require_nonempty_string(
            annotation.get("annotation_id"),
            field_name=f"annotations[{position}].annotation_id",
        )

        if annotation_id in seen_annotation_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"Duplicate annotation_id: {annotation_id}."
            )

        seen_annotation_ids.add(annotation_id)

        annotation_annotator_id = _require_nonempty_string(
            annotation.get("annotator_id"),
            field_name=f"{annotation_id}.annotator_id",
        )

        if annotation_annotator_id != submission_annotator_id:
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.annotator_id does not match "
                "the submission annotator."
            )

        annotation_guide_version = _require_version(
            annotation.get("annotation_guide_version"),
            field_name=(
                f"{annotation_id}.annotation_guide_version"
            ),
        )

        if annotation_guide_version != expected_guide_version:
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.annotation_guide_version does "
                "not match the bound annotation guide."
            )

        case_id = _require_nonempty_string(
            annotation.get("case_id"),
            field_name=f"{annotation_id}.case_id",
        )

        if case_id in seen_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate annotated case_id: {case_id}."
            )

        if case_id not in batch_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.case_id is not present in the "
                f"annotation batch: {case_id}."
            )

        seen_case_ids.add(case_id)

        evidence_status = _require_nonempty_string(
            annotation.get("evidence_status"),
            field_name=f"{annotation_id}.evidence_status",
        )

        if evidence_status not in ALLOWED_EVIDENCE_STATUSES:
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.evidence_status must be "
                "'sufficient' or 'insufficient'."
            )

        response_action = _require_nonempty_string(
            annotation.get("response_action"),
            field_name=f"{annotation_id}.response_action",
        )

        if response_action not in ALLOWED_RESPONSE_ACTIONS:
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.response_action must be "
                "'answer' or 'abstain'."
            )

        reason_codes = _require_unique_strings(
            annotation.get("reason_codes"),
            field_name="reason_codes",
            allow_empty=True,
        )
        unknown_reason_codes = sorted(
            set(reason_codes) - ALLOWED_REASON_CODES
        )

        if unknown_reason_codes:
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.reason_codes contains unsupported "
                f"values: {unknown_reason_codes}."
            )

        missing_information = _require_unique_strings(
            annotation.get("missing_information"),
            field_name="missing_information",
            allow_empty=True,
        )

        _require_nonempty_string(
            annotation.get("rationale"),
            field_name=f"{annotation_id}.rationale",
        )

        uncertainty = annotation.get("uncertainty")

        if not isinstance(uncertainty, bool):
            raise EvidenceSufficiencyAnnotationError(
                f"{annotation_id}.uncertainty must be a boolean."
            )

        _require_optional_nonempty_string(
            annotation.get("adjudication_note"),
            field_name=f"{annotation_id}.adjudication_note",
        )
        _require_rfc3339_utc_timestamp(
            annotation.get("annotation_timestamp"),
            field_name=f"{annotation_id}.annotation_timestamp",
        )

        _validate_annotation_relationships(
            case_id=case_id,
            evidence_status=evidence_status,
            response_action=response_action,
            reason_codes=reason_codes,
            missing_information=missing_information,
        )

    if seen_case_ids != batch_case_ids:
        raise EvidenceSufficiencyAnnotationError(
            "annotations must cover every batch case exactly once."
        )


@dataclass(frozen=True)
class AnnotationCaseComparison:
    """Pre-adjudication comparison for one assigned case."""

    case_id: str
    disagreement_fields: tuple[str, ...]
    uncertainty_present: bool
    requires_adjudication: bool


@dataclass(frozen=True)
class AnnotationComparison:
    """Deterministic comparison of two independent record sets."""

    annotator_ids: tuple[str, str]
    record_set_sha256s: tuple[str, str]
    case_comparisons: tuple[AnnotationCaseComparison, ...]


def _annotations_by_case_id(
    record_set: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    annotations = _require_sequence(
        record_set.get("annotations"),
        field_name="annotations",
    )
    result: dict[str, Mapping[str, Any]] = {}

    for position, raw_annotation in enumerate(annotations):
        annotation = _require_mapping(
            raw_annotation,
            field_name=f"annotations[{position}]",
        )
        case_id = _require_nonempty_string(
            annotation.get("case_id"),
            field_name=f"annotations[{position}].case_id",
        )

        if case_id in result:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate annotated case_id: {case_id}."
            )

        result[case_id] = annotation

    return result


def _unordered_string_values(
    annotation: Mapping[str, Any],
    *,
    key: str,
    field_name: str,
) -> frozenset[str]:
    values = _require_unique_strings(
        annotation.get(key),
        field_name=field_name,
        allow_empty=True,
    )
    return frozenset(values)


def compare_annotation_record_sets(
    first_record_set: Mapping[str, Any],
    second_record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    first_record_set_sha256: str,
    second_record_set_sha256: str,
) -> AnnotationComparison:
    """Compare two independent annotations before adjudication."""

    first = _require_mapping(
        first_record_set,
        field_name="first_record_set",
    )
    second = _require_mapping(
        second_record_set,
        field_name="second_record_set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )

    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )
    accepted_first_sha256 = _require_sha256(
        first_record_set_sha256,
        field_name="first_record_set_sha256",
    )
    accepted_second_sha256 = _require_sha256(
        second_record_set_sha256,
        field_name="second_record_set_sha256",
    )

    validate_annotation_record_set(
        first,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
    )
    validate_annotation_record_set(
        second,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
    )

    first_annotator_id = _require_nonempty_string(
        first.get("annotator_id"),
        field_name="first_record_set.annotator_id",
    )
    second_annotator_id = _require_nonempty_string(
        second.get("annotator_id"),
        field_name="second_record_set.annotator_id",
    )

    if first_annotator_id == second_annotator_id:
        raise EvidenceSufficiencyAnnotationError(
            "annotation comparison requires distinct annotators."
        )

    first_by_case = _annotations_by_case_id(first)
    second_by_case = _annotations_by_case_id(second)

    batch_cases = _require_sequence(
        batch.get("cases"),
        field_name="annotation_batch.cases",
    )
    comparisons: list[AnnotationCaseComparison] = []

    for position, raw_case in enumerate(batch_cases):
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

        first_annotation = first_by_case[case_id]
        second_annotation = second_by_case[case_id]
        disagreement_fields: list[str] = []

        for field_name in (
            "evidence_status",
            "response_action",
        ):
            first_value = _require_nonempty_string(
                first_annotation.get(field_name),
                field_name=f"first.{case_id}.{field_name}",
            )
            second_value = _require_nonempty_string(
                second_annotation.get(field_name),
                field_name=f"second.{case_id}.{field_name}",
            )

            if first_value != second_value:
                disagreement_fields.append(field_name)

        for field_name in (
            "reason_codes",
            "missing_information",
        ):
            first_values = _unordered_string_values(
                first_annotation,
                key=field_name,
                field_name=f"first.{case_id}.{field_name}",
            )
            second_values = _unordered_string_values(
                second_annotation,
                key=field_name,
                field_name=f"second.{case_id}.{field_name}",
            )

            if first_values != second_values:
                disagreement_fields.append(field_name)

        first_uncertainty = first_annotation.get("uncertainty")
        second_uncertainty = second_annotation.get("uncertainty")

        if not isinstance(first_uncertainty, bool):
            raise EvidenceSufficiencyAnnotationError(
                f"first.{case_id}.uncertainty must be a boolean."
            )

        if not isinstance(second_uncertainty, bool):
            raise EvidenceSufficiencyAnnotationError(
                f"second.{case_id}.uncertainty must be a boolean."
            )

        uncertainty_present = (
            first_uncertainty or second_uncertainty
        )
        disagreements = tuple(disagreement_fields)

        comparisons.append(
            AnnotationCaseComparison(
                case_id=case_id,
                disagreement_fields=disagreements,
                uncertainty_present=uncertainty_present,
                requires_adjudication=(
                    bool(disagreements) or uncertainty_present
                ),
            )
        )

    return AnnotationComparison(
        annotator_ids=(
            first_annotator_id,
            second_annotator_id,
        ),
        record_set_sha256s=(
            accepted_first_sha256,
            accepted_second_sha256,
        ),
        case_comparisons=tuple(comparisons),
    )


def _validate_final_adjudication_relationships(
    *,
    case_id: str,
    evidence_status: str,
    response_action: str,
    reason_codes: tuple[str, ...],
    missing_information: tuple[str, ...],
) -> None:
    if evidence_status == SUFFICIENT_STATUS:
        if response_action != ANSWER_ACTION:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}: sufficient final labels require "
                "answer action."
            )

        if reason_codes:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}: sufficient final labels require "
                "empty final_reason_codes."
            )

        if missing_information:
            raise EvidenceSufficiencyAnnotationError(
                f"{case_id}: sufficient final labels require "
                "empty final_missing_information."
            )

        return

    if response_action != ABSTAIN_ACTION:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}: insufficient final labels require "
            "abstain action."
        )

    if not reason_codes:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}: insufficient final labels require "
            "final_reason_codes."
        )

    if not missing_information:
        raise EvidenceSufficiencyAnnotationError(
            f"{case_id}: insufficient final labels require "
            "final_missing_information."
        )


def validate_adjudication_record_set(
    adjudication_record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    first_record_set: Mapping[str, Any],
    first_record_set_sha256: str,
    second_record_set: Mapping[str, Any],
    second_record_set_sha256: str,
) -> None:
    """Validate written adjudication for all cases requiring review."""

    value = _require_mapping(
        adjudication_record_set,
        field_name="adjudication record set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    first = _require_mapping(
        first_record_set,
        field_name="first_record_set",
    )
    second = _require_mapping(
        second_record_set,
        field_name="second_record_set",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=ADJUDICATION_RECORD_SET_FIELDS,
        object_name="adjudication record set",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != ADJUDICATION_RECORD_SET_SCHEMA_VERSION:
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    adjudication_set_id = _require_nonempty_string(
        value.get("adjudication_set_id"),
        field_name="adjudication_set_id",
    )

    if adjudication_set_id != ADJUDICATION_RECORD_SET_ID:
        raise EvidenceSufficiencyAnnotationError(
            "adjudication_set_id is not the evidence-sufficiency "
            "adjudication-record-set ID."
        )

    _require_version(
        value.get("adjudication_set_version"),
        field_name="adjudication_set_version",
    )

    expected_batch_id = _require_nonempty_string(
        batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )
    expected_batch_version = _require_version(
        batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256 argument",
    )
    expected_guide_version = _require_version(
        batch.get("annotation_guide_version"),
        field_name="annotation_batch.annotation_guide_version",
    )
    expected_guide_sha256 = _require_sha256(
        batch.get("annotation_guide_sha256"),
        field_name="annotation_batch.annotation_guide_sha256",
    )

    expected_first_id = _require_nonempty_string(
        first.get("record_set_id"),
        field_name="first_record_set.record_set_id",
    )
    expected_first_version = _require_version(
        first.get("record_set_version"),
        field_name="first_record_set.record_set_version",
    )
    accepted_first_sha256 = _require_sha256(
        first_record_set_sha256,
        field_name="first_record_set_sha256 argument",
    )

    expected_second_id = _require_nonempty_string(
        second.get("record_set_id"),
        field_name="second_record_set.record_set_id",
    )
    expected_second_version = _require_version(
        second.get("record_set_version"),
        field_name="second_record_set.record_set_version",
    )
    accepted_second_sha256 = _require_sha256(
        second_record_set_sha256,
        field_name="second_record_set_sha256 argument",
    )

    if expected_first_id != ANNOTATION_RECORD_SET_ID:
        raise EvidenceSufficiencyAnnotationError(
            "first_record_set.record_set_id is not supported."
        )

    if expected_second_id != ANNOTATION_RECORD_SET_ID:
        raise EvidenceSufficiencyAnnotationError(
            "second_record_set.record_set_id is not supported."
        )

    for field_name, expected_value in (
        ("annotation_batch_id", expected_batch_id),
        ("annotation_batch_version", expected_batch_version),
        ("annotation_batch_sha256", accepted_batch_sha256),
        ("annotation_guide_version", expected_guide_version),
        ("annotation_guide_sha256", expected_guide_sha256),
        ("first_record_set_id", expected_first_id),
        ("first_record_set_version", expected_first_version),
        ("first_record_set_sha256", accepted_first_sha256),
        ("second_record_set_id", expected_second_id),
        ("second_record_set_version", expected_second_version),
        ("second_record_set_sha256", accepted_second_sha256),
    ):
        _require_binding(
            value,
            field_name=field_name,
            expected_value=expected_value,
        )

    _require_nonempty_string(
        value.get("adjudicator_id"),
        field_name="adjudicator_id",
    )

    comparison = compare_annotation_record_sets(
        first,
        second,
        annotation_batch=batch,
        annotation_batch_sha256=accepted_batch_sha256,
        first_record_set_sha256=accepted_first_sha256,
        second_record_set_sha256=accepted_second_sha256,
    )

    required_case_ids = {
        case_comparison.case_id
        for case_comparison in comparison.case_comparisons
        if case_comparison.requires_adjudication
    }
    comparisons_by_case_id = {
        case_comparison.case_id: case_comparison
        for case_comparison in comparison.case_comparisons
    }
    first_annotations = _annotations_by_case_id(first)
    second_annotations = _annotations_by_case_id(second)

    records = _require_sequence(
        value.get("records"),
        field_name="records",
    )
    record_count = _require_nonnegative_integer(
        value.get("record_count"),
        field_name="record_count",
    )

    if record_count != len(records):
        raise EvidenceSufficiencyAnnotationError(
            "record_count does not match the number of records."
        )

    seen_adjudication_ids: set[str] = set()
    seen_case_ids: set[str] = set()

    for position, raw_record in enumerate(records):
        record = _require_mapping(
            raw_record,
            field_name=f"records[{position}]",
        )
        _reject_unknown_fields(
            record,
            allowed_fields=ADJUDICATION_RECORD_FIELDS,
            object_name="adjudication record",
        )

        adjudication_id = _require_nonempty_string(
            record.get("adjudication_id"),
            field_name=f"records[{position}].adjudication_id",
        )

        if adjudication_id in seen_adjudication_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"Duplicate adjudication_id: {adjudication_id}."
            )

        seen_adjudication_ids.add(adjudication_id)

        case_id = _require_nonempty_string(
            record.get("case_id"),
            field_name=f"{adjudication_id}.case_id",
        )

        if case_id in seen_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate adjudicated case_id: {case_id}."
            )

        seen_case_ids.add(case_id)

        if (
            case_id not in first_annotations
            or case_id not in second_annotations
            or case_id not in comparisons_by_case_id
        ):
            raise EvidenceSufficiencyAnnotationError(
                f"{adjudication_id}.case_id is not present in "
                "both bound annotation record sets."
            )

        expected_first_annotation_id = _require_nonempty_string(
            first_annotations[case_id].get("annotation_id"),
            field_name=(
                f"first_record_set.{case_id}.annotation_id"
            ),
        )
        expected_second_annotation_id = _require_nonempty_string(
            second_annotations[case_id].get("annotation_id"),
            field_name=(
                f"second_record_set.{case_id}.annotation_id"
            ),
        )

        _require_binding(
            record,
            field_name="first_annotation_id",
            expected_value=expected_first_annotation_id,
        )
        _require_binding(
            record,
            field_name="second_annotation_id",
            expected_value=expected_second_annotation_id,
        )

        disagreement_categories = _require_unique_strings(
            record.get("disagreement_categories"),
            field_name="disagreement_categories",
            allow_empty=True,
        )
        unknown_categories = sorted(
            set(disagreement_categories)
            - ALLOWED_DISAGREEMENT_CATEGORIES
        )

        if unknown_categories:
            raise EvidenceSufficiencyAnnotationError(
                "disagreement_categories contains unsupported "
                f"values: {unknown_categories}."
            )

        case_comparison = comparisons_by_case_id[case_id]

        if (
            case_comparison.disagreement_fields
            and not disagreement_categories
        ):
            raise EvidenceSufficiencyAnnotationError(
                "disagreement_categories must be nonempty when "
                "annotation labels disagree."
            )

        final_evidence_status = _require_nonempty_string(
            record.get("final_evidence_status"),
            field_name=(
                f"{adjudication_id}.final_evidence_status"
            ),
        )

        if final_evidence_status not in ALLOWED_EVIDENCE_STATUSES:
            raise EvidenceSufficiencyAnnotationError(
                f"{adjudication_id}.final_evidence_status must be "
                "'sufficient' or 'insufficient'."
            )

        final_response_action = _require_nonempty_string(
            record.get("final_response_action"),
            field_name=(
                f"{adjudication_id}.final_response_action"
            ),
        )

        if final_response_action not in ALLOWED_RESPONSE_ACTIONS:
            raise EvidenceSufficiencyAnnotationError(
                f"{adjudication_id}.final_response_action must be "
                "'answer' or 'abstain'."
            )

        final_reason_codes = _require_unique_strings(
            record.get("final_reason_codes"),
            field_name="final_reason_codes",
            allow_empty=True,
        )
        unknown_reason_codes = sorted(
            set(final_reason_codes) - ALLOWED_REASON_CODES
        )

        if unknown_reason_codes:
            raise EvidenceSufficiencyAnnotationError(
                f"{adjudication_id}.final_reason_codes contains "
                f"unsupported values: {unknown_reason_codes}."
            )

        final_missing_information = _require_unique_strings(
            record.get("final_missing_information"),
            field_name="final_missing_information",
            allow_empty=True,
        )

        _require_nonempty_string(
            record.get("final_rationale"),
            field_name=f"{adjudication_id}.final_rationale",
        )
        _require_nonempty_string(
            record.get("adjudication_rationale"),
            field_name=(
                f"{adjudication_id}.adjudication_rationale"
            ),
        )

        guide_change_required = record.get(
            "guide_change_required"
        )

        if not isinstance(guide_change_required, bool):
            raise EvidenceSufficiencyAnnotationError(
                f"{adjudication_id}.guide_change_required must "
                "be a boolean."
            )

        guide_change_summary = _require_optional_nonempty_string(
            record.get("guide_change_summary"),
            field_name=(
                f"{adjudication_id}.guide_change_summary"
            ),
        )

        if guide_change_required and guide_change_summary is None:
            raise EvidenceSufficiencyAnnotationError(
                "guide_change_required requires a nonempty "
                "guide_change_summary."
            )

        if (
            not guide_change_required
            and guide_change_summary is not None
        ):
            raise EvidenceSufficiencyAnnotationError(
                "guide_change_summary must be null when "
                "guide_change_required is false."
            )

        _require_rfc3339_utc_timestamp(
            record.get("adjudication_timestamp"),
            field_name=(
                f"{adjudication_id}.adjudication_timestamp"
            ),
        )

        _validate_final_adjudication_relationships(
            case_id=case_id,
            evidence_status=final_evidence_status,
            response_action=final_response_action,
            reason_codes=final_reason_codes,
            missing_information=final_missing_information,
        )

    if seen_case_ids != required_case_ids:
        raise EvidenceSufficiencyAnnotationError(
            "records must cover exactly the cases requiring "
            "adjudication."
        )


@dataclass(frozen=True)
class ReasonCodeAgreement:
    """Directional agreement statistics for one reason code."""

    reason_code: str
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class AnnotationAgreementReport:
    """Pre-adjudication agreement between two independent annotators."""

    annotator_ids: tuple[str, str]
    record_set_sha256s: tuple[str, str]
    case_count: int
    status_agreement_count: int
    status_raw_agreement: float
    cohen_kappa: float | None
    reason_code_exact_match_count: int
    reason_code_exact_match_agreement: float
    mean_reason_code_jaccard: float
    per_code_agreement: tuple[ReasonCodeAgreement, ...]
    macro_reason_code_precision: float | None
    macro_reason_code_recall: float | None
    macro_reason_code_f1: float | None


def _cohen_kappa_binary(
    first_statuses: Sequence[str],
    second_statuses: Sequence[str],
) -> float | None:
    if len(first_statuses) != len(second_statuses):
        raise EvidenceSufficiencyAnnotationError(
            "status sequences must have equal length."
        )

    case_count = len(first_statuses)

    if case_count == 0:
        raise EvidenceSufficiencyAnnotationError(
            "agreement requires at least one case."
        )

    observed_agreement = sum(
        first_status == second_status
        for first_status, second_status
        in zip(first_statuses, second_statuses, strict=True)
    ) / case_count

    first_sufficient = (
        sum(
            status == SUFFICIENT_STATUS
            for status in first_statuses
        )
        / case_count
    )
    second_sufficient = (
        sum(
            status == SUFFICIENT_STATUS
            for status in second_statuses
        )
        / case_count
    )

    first_insufficient = 1.0 - first_sufficient
    second_insufficient = 1.0 - second_sufficient

    expected_agreement = (
        first_sufficient * second_sufficient
        + first_insufficient * second_insufficient
    )
    denominator = 1.0 - expected_agreement

    if denominator == 0.0:
        return None

    return (
        observed_agreement - expected_agreement
    ) / denominator


def _reason_code_jaccard(
    first_codes: frozenset[str],
    second_codes: frozenset[str],
) -> float:
    union = first_codes | second_codes

    if not union:
        return 1.0

    return len(first_codes & second_codes) / len(union)


def _safe_precision(
    true_positive: int,
    false_positive: int,
) -> float:
    denominator = true_positive + false_positive

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def _safe_recall(
    true_positive: int,
    false_negative: int,
) -> float:
    denominator = true_positive + false_negative

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def _safe_f1(
    precision: float,
    recall: float,
) -> float:
    denominator = precision + recall

    if denominator == 0.0:
        return 0.0

    return 2.0 * precision * recall / denominator


def calculate_annotation_agreement(
    first_record_set: Mapping[str, Any],
    second_record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    first_record_set_sha256: str,
    second_record_set_sha256: str,
) -> AnnotationAgreementReport:
    """Calculate pre-adjudication status and reason-code agreement."""

    comparison = compare_annotation_record_sets(
        first_record_set,
        second_record_set,
        annotation_batch=annotation_batch,
        annotation_batch_sha256=annotation_batch_sha256,
        first_record_set_sha256=first_record_set_sha256,
        second_record_set_sha256=second_record_set_sha256,
    )

    first = _require_mapping(
        first_record_set,
        field_name="first_record_set",
    )
    second = _require_mapping(
        second_record_set,
        field_name="second_record_set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )

    first_by_case = _annotations_by_case_id(first)
    second_by_case = _annotations_by_case_id(second)
    batch_cases = _require_sequence(
        batch.get("cases"),
        field_name="annotation_batch.cases",
    )

    first_statuses: list[str] = []
    second_statuses: list[str] = []
    reason_code_pairs: list[
        tuple[frozenset[str], frozenset[str]]
    ] = []

    for position, raw_case in enumerate(batch_cases):
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

        first_annotation = first_by_case[case_id]
        second_annotation = second_by_case[case_id]

        first_status = _require_nonempty_string(
            first_annotation.get("evidence_status"),
            field_name=f"first.{case_id}.evidence_status",
        )
        second_status = _require_nonempty_string(
            second_annotation.get("evidence_status"),
            field_name=f"second.{case_id}.evidence_status",
        )

        first_statuses.append(first_status)
        second_statuses.append(second_status)

        first_codes = _unordered_string_values(
            first_annotation,
            key="reason_codes",
            field_name=f"first.{case_id}.reason_codes",
        )
        second_codes = _unordered_string_values(
            second_annotation,
            key="reason_codes",
            field_name=f"second.{case_id}.reason_codes",
        )
        reason_code_pairs.append(
            (
                first_codes,
                second_codes,
            )
        )

    case_count = len(first_statuses)

    if case_count == 0:
        raise EvidenceSufficiencyAnnotationError(
            "agreement requires at least one case."
        )

    status_agreement_count = sum(
        first_status == second_status
        for first_status, second_status
        in zip(first_statuses, second_statuses, strict=True)
    )
    status_raw_agreement = (
        status_agreement_count / case_count
    )
    cohen_kappa = _cohen_kappa_binary(
        first_statuses,
        second_statuses,
    )

    reason_code_exact_match_count = sum(
        first_codes == second_codes
        for first_codes, second_codes in reason_code_pairs
    )
    reason_code_exact_match_agreement = (
        reason_code_exact_match_count / case_count
    )

    jaccard_values = [
        _reason_code_jaccard(
            first_codes,
            second_codes,
        )
        for first_codes, second_codes in reason_code_pairs
    ]
    mean_reason_code_jaccard = (
        sum(jaccard_values) / case_count
    )

    all_reason_codes = sorted(
        {
            reason_code
            for first_codes, second_codes in reason_code_pairs
            for reason_code in first_codes | second_codes
        }
    )
    per_code_agreement: list[ReasonCodeAgreement] = []

    for reason_code in all_reason_codes:
        true_positive = sum(
            reason_code in first_codes
            and reason_code in second_codes
            for first_codes, second_codes in reason_code_pairs
        )
        false_positive = sum(
            reason_code not in first_codes
            and reason_code in second_codes
            for first_codes, second_codes in reason_code_pairs
        )
        false_negative = sum(
            reason_code in first_codes
            and reason_code not in second_codes
            for first_codes, second_codes in reason_code_pairs
        )

        precision = _safe_precision(
            true_positive,
            false_positive,
        )
        recall = _safe_recall(
            true_positive,
            false_negative,
        )
        f1 = _safe_f1(
            precision,
            recall,
        )

        per_code_agreement.append(
            ReasonCodeAgreement(
                reason_code=reason_code,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    if per_code_agreement:
        code_count = len(per_code_agreement)
        macro_reason_code_precision = (
            sum(
                item.precision
                for item in per_code_agreement
            )
            / code_count
        )
        macro_reason_code_recall = (
            sum(
                item.recall
                for item in per_code_agreement
            )
            / code_count
        )
        macro_reason_code_f1 = (
            sum(
                item.f1
                for item in per_code_agreement
            )
            / code_count
        )
    else:
        macro_reason_code_precision = None
        macro_reason_code_recall = None
        macro_reason_code_f1 = None

    return AnnotationAgreementReport(
        annotator_ids=comparison.annotator_ids,
        record_set_sha256s=comparison.record_set_sha256s,
        case_count=case_count,
        status_agreement_count=status_agreement_count,
        status_raw_agreement=status_raw_agreement,
        cohen_kappa=cohen_kappa,
        reason_code_exact_match_count=(
            reason_code_exact_match_count
        ),
        reason_code_exact_match_agreement=(
            reason_code_exact_match_agreement
        ),
        mean_reason_code_jaccard=(
            mean_reason_code_jaccard
        ),
        per_code_agreement=tuple(per_code_agreement),
        macro_reason_code_precision=(
            macro_reason_code_precision
        ),
        macro_reason_code_recall=(
            macro_reason_code_recall
        ),
        macro_reason_code_f1=macro_reason_code_f1,
    )


def validate_annotation_analysis_metadata(
    metadata: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
) -> None:
    """Validate non-blinded structure metadata for agreement analysis."""

    value = _require_mapping(
        metadata,
        field_name="annotation analysis metadata",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=ANNOTATION_ANALYSIS_METADATA_FIELDS,
        object_name="annotation analysis metadata",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != ANNOTATION_ANALYSIS_METADATA_SCHEMA_VERSION:
        raise EvidenceSufficiencyAnnotationError(
            "schema_version must be 1.0."
        )

    metadata_id = _require_nonempty_string(
        value.get("metadata_id"),
        field_name="metadata_id",
    )

    if metadata_id != ANNOTATION_ANALYSIS_METADATA_ID:
        raise EvidenceSufficiencyAnnotationError(
            "metadata_id is not the evidence-sufficiency "
            "annotation-analysis-metadata ID."
        )

    _require_version(
        value.get("metadata_version"),
        field_name="metadata_version",
    )

    expected_batch_id = _require_nonempty_string(
        batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )

    if expected_batch_id != ANNOTATION_BATCH_ID:
        raise EvidenceSufficiencyAnnotationError(
            "annotation_batch.batch_id is not supported."
        )

    expected_batch_version = _require_version(
        batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256 argument",
    )

    _require_binding(
        value,
        field_name="annotation_batch_id",
        expected_value=expected_batch_id,
    )
    _require_binding(
        value,
        field_name="annotation_batch_version",
        expected_value=expected_batch_version,
    )
    _require_binding(
        value,
        field_name="annotation_batch_sha256",
        expected_value=accepted_batch_sha256,
    )

    batch_cases = _require_sequence(
        batch.get("cases"),
        field_name="annotation_batch.cases",
    )
    batch_case_ids: set[str] = set()

    for position, raw_case in enumerate(batch_cases):
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

        if case_id in batch_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                "annotation_batch contains duplicate case_id: "
                f"{case_id}."
            )

        batch_case_ids.add(case_id)

    metadata_cases = _require_sequence(
        value.get("cases"),
        field_name="cases",
    )
    case_count = _require_nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )

    if case_count != len(metadata_cases):
        raise EvidenceSufficiencyAnnotationError(
            "case_count does not match the number of cases."
        )

    seen_case_ids: set[str] = set()

    for position, raw_metadata_case in enumerate(metadata_cases):
        metadata_case = _require_mapping(
            raw_metadata_case,
            field_name=f"cases[{position}]",
        )
        _reject_unknown_fields(
            metadata_case,
            allowed_fields=(
                ANNOTATION_ANALYSIS_METADATA_CASE_FIELDS
            ),
            object_name="analysis metadata case",
        )

        case_id = _require_nonempty_string(
            metadata_case.get("case_id"),
            field_name=f"cases[{position}].case_id",
        )

        if case_id in seen_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate metadata case_id: {case_id}."
            )

        if case_id not in batch_case_ids:
            raise EvidenceSufficiencyAnnotationError(
                f"analysis metadata contains unknown case_id: "
                f"{case_id}."
            )

        seen_case_ids.add(case_id)

        question_structure_codes = _require_unique_strings(
            metadata_case.get("question_structure_codes"),
            field_name="question_structure_codes",
            allow_empty=False,
        )
        unknown_question_structure_codes = sorted(
            set(question_structure_codes)
            - ALLOWED_QUESTION_STRUCTURE_CODES
        )

        if unknown_question_structure_codes:
            raise EvidenceSufficiencyAnnotationError(
                "question_structure_codes contains unsupported "
                f"values: {unknown_question_structure_codes}."
            )

        evidence_structure_codes = _require_unique_strings(
            metadata_case.get("evidence_structure_codes"),
            field_name="evidence_structure_codes",
            allow_empty=False,
        )
        unknown_evidence_structure_codes = sorted(
            set(evidence_structure_codes)
            - ALLOWED_EVIDENCE_STRUCTURE_CODES
        )

        if unknown_evidence_structure_codes:
            raise EvidenceSufficiencyAnnotationError(
                "evidence_structure_codes contains unsupported "
                f"values: {unknown_evidence_structure_codes}."
            )

    if seen_case_ids != batch_case_ids:
        raise EvidenceSufficiencyAnnotationError(
            "cases must cover every batch case exactly once."
        )


@dataclass(frozen=True)
class StructureDisagreementCount:
    """Pre-adjudication counts for one analysis structure code."""

    structure_code: str
    case_count: int
    disagreement_case_count: int
    uncertainty_case_count: int
    requires_adjudication_count: int
    evidence_status_disagreement_count: int
    response_action_disagreement_count: int
    reason_codes_disagreement_count: int
    missing_information_disagreement_count: int


@dataclass(frozen=True)
class AnnotationStructureDisagreementReport:
    """Disagreement counts grouped by analysis-only structure codes."""

    annotator_ids: tuple[str, str]
    record_set_sha256s: tuple[str, str]
    analysis_metadata_sha256: str
    case_count: int
    question_structure_counts: tuple[
        StructureDisagreementCount,
        ...,
    ]
    evidence_structure_counts: tuple[
        StructureDisagreementCount,
        ...,
    ]


def _metadata_by_case_id(
    analysis_metadata: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    cases = _require_sequence(
        analysis_metadata.get("cases"),
        field_name="analysis_metadata.cases",
    )
    result: dict[str, Mapping[str, Any]] = {}

    for position, raw_case in enumerate(cases):
        case = _require_mapping(
            raw_case,
            field_name=f"analysis_metadata.cases[{position}]",
        )
        case_id = _require_nonempty_string(
            case.get("case_id"),
            field_name=(
                f"analysis_metadata.cases[{position}].case_id"
            ),
        )

        if case_id in result:
            raise EvidenceSufficiencyAnnotationError(
                f"duplicate metadata case_id: {case_id}."
            )

        result[case_id] = case

    return result


def _structure_counts(
    *,
    structure_field: str,
    metadata_by_case_id: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[AnnotationCaseComparison],
) -> tuple[StructureDisagreementCount, ...]:
    counters: dict[str, dict[str, int]] = {}

    for comparison in comparisons:
        metadata_case = metadata_by_case_id[comparison.case_id]
        structure_codes = _require_unique_strings(
            metadata_case.get(structure_field),
            field_name=structure_field,
            allow_empty=False,
        )
        disagreement_fields = set(
            comparison.disagreement_fields
        )

        for structure_code in structure_codes:
            counter = counters.setdefault(
                structure_code,
                {
                    "case_count": 0,
                    "disagreement_case_count": 0,
                    "uncertainty_case_count": 0,
                    "requires_adjudication_count": 0,
                    "evidence_status_disagreement_count": 0,
                    "response_action_disagreement_count": 0,
                    "reason_codes_disagreement_count": 0,
                    "missing_information_disagreement_count": 0,
                },
            )

            counter["case_count"] += 1

            if disagreement_fields:
                counter["disagreement_case_count"] += 1

            if comparison.uncertainty_present:
                counter["uncertainty_case_count"] += 1

            if comparison.requires_adjudication:
                counter["requires_adjudication_count"] += 1

            for field_name in (
                "evidence_status",
                "response_action",
                "reason_codes",
                "missing_information",
            ):
                if field_name in disagreement_fields:
                    counter[
                        f"{field_name}_disagreement_count"
                    ] += 1

    return tuple(
        StructureDisagreementCount(
            structure_code=structure_code,
            case_count=counter["case_count"],
            disagreement_case_count=counter[
                "disagreement_case_count"
            ],
            uncertainty_case_count=counter[
                "uncertainty_case_count"
            ],
            requires_adjudication_count=counter[
                "requires_adjudication_count"
            ],
            evidence_status_disagreement_count=counter[
                "evidence_status_disagreement_count"
            ],
            response_action_disagreement_count=counter[
                "response_action_disagreement_count"
            ],
            reason_codes_disagreement_count=counter[
                "reason_codes_disagreement_count"
            ],
            missing_information_disagreement_count=counter[
                "missing_information_disagreement_count"
            ],
        )
        for structure_code, counter in sorted(
            counters.items()
        )
    )


def calculate_structure_disagreement_counts(
    first_record_set: Mapping[str, Any],
    second_record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    first_record_set_sha256: str,
    second_record_set_sha256: str,
    analysis_metadata: Mapping[str, Any],
    analysis_metadata_sha256: str,
) -> AnnotationStructureDisagreementReport:
    """Count disagreements by question and evidence structure."""

    metadata = _require_mapping(
        analysis_metadata,
        field_name="analysis_metadata",
    )
    accepted_metadata_sha256 = _require_sha256(
        analysis_metadata_sha256,
        field_name="analysis_metadata_sha256",
    )

    validate_annotation_analysis_metadata(
        metadata,
        annotation_batch=annotation_batch,
        annotation_batch_sha256=annotation_batch_sha256,
    )

    comparison = compare_annotation_record_sets(
        first_record_set,
        second_record_set,
        annotation_batch=annotation_batch,
        annotation_batch_sha256=annotation_batch_sha256,
        first_record_set_sha256=first_record_set_sha256,
        second_record_set_sha256=second_record_set_sha256,
    )

    metadata_by_case_id = _metadata_by_case_id(metadata)
    case_count = len(comparison.case_comparisons)

    if case_count == 0:
        raise EvidenceSufficiencyAnnotationError(
            "structure disagreement analysis requires at least "
            "one case."
        )

    question_structure_counts = _structure_counts(
        structure_field="question_structure_codes",
        metadata_by_case_id=metadata_by_case_id,
        comparisons=comparison.case_comparisons,
    )
    evidence_structure_counts = _structure_counts(
        structure_field="evidence_structure_codes",
        metadata_by_case_id=metadata_by_case_id,
        comparisons=comparison.case_comparisons,
    )

    return AnnotationStructureDisagreementReport(
        annotator_ids=comparison.annotator_ids,
        record_set_sha256s=comparison.record_set_sha256s,
        analysis_metadata_sha256=accepted_metadata_sha256,
        case_count=case_count,
        question_structure_counts=question_structure_counts,
        evidence_structure_counts=evidence_structure_counts,
    )



def _require_rfc3339_utc_timestamp(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Require canonical second-precision RFC 3339 UTC text."""

    timestamp = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    try:
        parsed = datetime.strptime(
            timestamp,
            "%Y-%m-%dT%H:%M:%SZ",
        )
    except ValueError as error:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must use canonical RFC 3339 UTC "
            "format YYYY-MM-DDTHH:MM:SSZ."
        ) from error

    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
        raise EvidenceSufficiencyAnnotationError(
            f"{field_name} must use canonical RFC 3339 UTC "
            "format YYYY-MM-DDTHH:MM:SSZ."
        )

    return timestamp

def write_annotation_json_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically publish formatted JSON without overwriting."""

    if not isinstance(artifact, Mapping):
        raise EvidenceSufficiencyAnnotationError(
            "artifact must be a mapping."
        )

    if not isinstance(output_path, Path):
        raise EvidenceSufficiencyAnnotationError(
            "output_path must be a pathlib.Path."
        )

    if output_path.exists():
        raise EvidenceSufficiencyAnnotationError(
            f"Output already exists: {output_path}"
        )

    temporary_path: Path | None = None

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        serialized = (
            json.dumps(
                artifact,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

        file_descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(raw_temporary_path)

        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(
                temporary_path,
                output_path,
            )
        except FileExistsError as error:
            raise EvidenceSufficiencyAnnotationError(
                f"Output already exists: {output_path}"
            ) from error

        temporary_path.unlink()
        temporary_path = None
    except EvidenceSufficiencyAnnotationError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise EvidenceSufficiencyAnnotationError(
            "Unable to publish annotation artifact: "
            f"{error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )


def build_annotation_agreement_artifact(
    first_record_set: Mapping[str, Any],
    second_record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    first_record_set_sha256: str,
    second_record_set_sha256: str,
    report_version: str,
) -> dict[str, Any]:
    """Build a deterministic, fully bound agreement artifact."""

    first = _require_mapping(
        first_record_set,
        field_name="first_record_set",
    )
    second = _require_mapping(
        second_record_set,
        field_name="second_record_set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    validated_report_version = _require_version(
        report_version,
        field_name="report_version",
    )

    report = calculate_annotation_agreement(
        first,
        second,
        annotation_batch=batch,
        annotation_batch_sha256=annotation_batch_sha256,
        first_record_set_sha256=first_record_set_sha256,
        second_record_set_sha256=second_record_set_sha256,
    )

    batch_id = _require_nonempty_string(
        batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )
    batch_version = _require_version(
        batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )

    first_record_set_id = _require_nonempty_string(
        first.get("record_set_id"),
        field_name="first_record_set.record_set_id",
    )
    first_record_set_version = _require_version(
        first.get("record_set_version"),
        field_name="first_record_set.record_set_version",
    )
    accepted_first_sha256 = _require_sha256(
        first_record_set_sha256,
        field_name="first_record_set_sha256",
    )

    second_record_set_id = _require_nonempty_string(
        second.get("record_set_id"),
        field_name="second_record_set.record_set_id",
    )
    second_record_set_version = _require_version(
        second.get("record_set_version"),
        field_name="second_record_set.record_set_version",
    )
    accepted_second_sha256 = _require_sha256(
        second_record_set_sha256,
        field_name="second_record_set_sha256",
    )

    return {
        "schema_version": ANNOTATION_AGREEMENT_REPORT_SCHEMA_VERSION,
        "report_id": ANNOTATION_AGREEMENT_REPORT_ID,
        "report_version": validated_report_version,
        "annotation_batch_id": batch_id,
        "annotation_batch_version": batch_version,
        "annotation_batch_sha256": accepted_batch_sha256,
        "first_record_set_id": first_record_set_id,
        "first_record_set_version": first_record_set_version,
        "first_record_set_sha256": accepted_first_sha256,
        "second_record_set_id": second_record_set_id,
        "second_record_set_version": second_record_set_version,
        "second_record_set_sha256": accepted_second_sha256,
        "annotator_ids": list(report.annotator_ids),
        "case_count": report.case_count,
        "status_agreement": {
            "agreement_count": report.status_agreement_count,
            "raw_agreement": report.status_raw_agreement,
            "cohen_kappa": report.cohen_kappa,
        },
        "reason_code_agreement": {
            "exact_match_count": (
                report.reason_code_exact_match_count
            ),
            "exact_match_agreement": (
                report.reason_code_exact_match_agreement
            ),
            "mean_jaccard": report.mean_reason_code_jaccard,
            "macro_precision": (
                report.macro_reason_code_precision
            ),
            "macro_recall": report.macro_reason_code_recall,
            "macro_f1": report.macro_reason_code_f1,
            "per_code": [
                {
                    "reason_code": item.reason_code,
                    "true_positive": item.true_positive,
                    "false_positive": item.false_positive,
                    "false_negative": item.false_negative,
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                }
                for item in report.per_code_agreement
            ],
        },
    }


def _structure_disagreement_count_artifact(
    count: StructureDisagreementCount,
) -> dict[str, Any]:
    return {
        "structure_code": count.structure_code,
        "case_count": count.case_count,
        "disagreement_case_count": (
            count.disagreement_case_count
        ),
        "uncertainty_case_count": (
            count.uncertainty_case_count
        ),
        "requires_adjudication_count": (
            count.requires_adjudication_count
        ),
        "evidence_status_disagreement_count": (
            count.evidence_status_disagreement_count
        ),
        "response_action_disagreement_count": (
            count.response_action_disagreement_count
        ),
        "reason_codes_disagreement_count": (
            count.reason_codes_disagreement_count
        ),
        "missing_information_disagreement_count": (
            count.missing_information_disagreement_count
        ),
    }


def build_structure_disagreement_artifact(
    first_record_set: Mapping[str, Any],
    second_record_set: Mapping[str, Any],
    *,
    annotation_batch: Mapping[str, Any],
    annotation_batch_sha256: str,
    first_record_set_sha256: str,
    second_record_set_sha256: str,
    analysis_metadata: Mapping[str, Any],
    analysis_metadata_sha256: str,
    report_version: str,
) -> dict[str, Any]:
    """Build a deterministic, fully bound structure report."""

    first = _require_mapping(
        first_record_set,
        field_name="first_record_set",
    )
    second = _require_mapping(
        second_record_set,
        field_name="second_record_set",
    )
    batch = _require_mapping(
        annotation_batch,
        field_name="annotation_batch",
    )
    metadata = _require_mapping(
        analysis_metadata,
        field_name="analysis_metadata",
    )
    validated_report_version = _require_version(
        report_version,
        field_name="report_version",
    )

    report = calculate_structure_disagreement_counts(
        first,
        second,
        annotation_batch=batch,
        annotation_batch_sha256=annotation_batch_sha256,
        first_record_set_sha256=first_record_set_sha256,
        second_record_set_sha256=second_record_set_sha256,
        analysis_metadata=metadata,
        analysis_metadata_sha256=analysis_metadata_sha256,
    )

    batch_id = _require_nonempty_string(
        batch.get("batch_id"),
        field_name="annotation_batch.batch_id",
    )
    batch_version = _require_version(
        batch.get("batch_version"),
        field_name="annotation_batch.batch_version",
    )
    accepted_batch_sha256 = _require_sha256(
        annotation_batch_sha256,
        field_name="annotation_batch_sha256",
    )

    first_record_set_id = _require_nonempty_string(
        first.get("record_set_id"),
        field_name="first_record_set.record_set_id",
    )
    first_record_set_version = _require_version(
        first.get("record_set_version"),
        field_name="first_record_set.record_set_version",
    )
    accepted_first_sha256 = _require_sha256(
        first_record_set_sha256,
        field_name="first_record_set_sha256",
    )

    second_record_set_id = _require_nonempty_string(
        second.get("record_set_id"),
        field_name="second_record_set.record_set_id",
    )
    second_record_set_version = _require_version(
        second.get("record_set_version"),
        field_name="second_record_set.record_set_version",
    )
    accepted_second_sha256 = _require_sha256(
        second_record_set_sha256,
        field_name="second_record_set_sha256",
    )

    metadata_id = _require_nonempty_string(
        metadata.get("metadata_id"),
        field_name="analysis_metadata.metadata_id",
    )
    metadata_version = _require_version(
        metadata.get("metadata_version"),
        field_name="analysis_metadata.metadata_version",
    )
    accepted_metadata_sha256 = _require_sha256(
        analysis_metadata_sha256,
        field_name="analysis_metadata_sha256",
    )

    return {
        "schema_version": (
            ANNOTATION_STRUCTURE_DISAGREEMENT_REPORT_SCHEMA_VERSION
        ),
        "report_id": (
            ANNOTATION_STRUCTURE_DISAGREEMENT_REPORT_ID
        ),
        "report_version": validated_report_version,
        "annotation_batch_id": batch_id,
        "annotation_batch_version": batch_version,
        "annotation_batch_sha256": accepted_batch_sha256,
        "first_record_set_id": first_record_set_id,
        "first_record_set_version": first_record_set_version,
        "first_record_set_sha256": accepted_first_sha256,
        "second_record_set_id": second_record_set_id,
        "second_record_set_version": second_record_set_version,
        "second_record_set_sha256": accepted_second_sha256,
        "analysis_metadata_id": metadata_id,
        "analysis_metadata_version": metadata_version,
        "analysis_metadata_sha256": accepted_metadata_sha256,
        "annotator_ids": list(report.annotator_ids),
        "case_count": report.case_count,
        "question_structure_counts": [
            _structure_disagreement_count_artifact(count)
            for count in report.question_structure_counts
        ],
        "evidence_structure_counts": [
            _structure_disagreement_count_artifact(count)
            for count in report.evidence_structure_counts
        ],
    }
