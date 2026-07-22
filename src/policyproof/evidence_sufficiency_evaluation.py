"""Validation for versioned evidence-sufficiency evaluation datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

EVIDENCE_SUFFICIENCY_DATASET_ID = (
    "policyproof-evidence-sufficiency-evaluation"
)
EVIDENCE_SUFFICIENCY_SCHEMA_VERSION = "1.0"
RETRIEVAL_EVALUATION_DATASET_ID = (
    "policyproof-retrieval-evaluation"
)

SUFFICIENT_STATUS = "sufficient"
INSUFFICIENT_STATUS = "insufficient"
ALLOWED_EVIDENCE_STATUSES = frozenset(
    {
        SUFFICIENT_STATUS,
        INSUFFICIENT_STATUS,
    }
)

ANSWER_ACTION = "answer"
ABSTAIN_ACTION = "abstain"
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

DATASET_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_version",
        "corpus_id",
        "corpus_version",
        "passage_schema_version",
        "passage_artifact_sha256",
        "source_retrieval_dataset_id",
        "source_retrieval_dataset_version",
        "source_retrieval_dataset_sha256",
        "case_count",
        "cases",
    }
)

CASE_FIELDS = frozenset(
    {
        "case_id",
        "query_id",
        "question",
        "evidence_passage_ids",
        "expected_evidence_status",
        "expected_response_action",
        "reason_codes",
        "missing_information",
        "rationale",
        "evaluation_tags",
    }
)


class EvidenceSufficiencyEvaluationError(ValueError):
    """Raised when an evidence-sufficiency dataset is invalid."""


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencyEvaluationError(
            f"{field_name} must be an object."
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
        raise EvidenceSufficiencyEvaluationError(
            f"unknown {object_name} fields: {unknown_fields}."
        )


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencyEvaluationError(
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
        raise EvidenceSufficiencyEvaluationError(
            f"{field_name} must use semantic version form X.Y.Z."
        )

    return version


def _require_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    sha256 = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if not SHA256_PATTERN.fullmatch(sha256):
        raise EvidenceSufficiencyEvaluationError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return sha256


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
        raise EvidenceSufficiencyEvaluationError(
            f"{field_name} must be a non-negative integer."
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
        raise EvidenceSufficiencyEvaluationError(
            f"{field_name} must be an array."
        )

    return value


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
        raise EvidenceSufficiencyEvaluationError(
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
            raise EvidenceSufficiencyEvaluationError(
                f"duplicate {field_name}: {text}."
            )

        seen.add(text)
        result.append(text)

    return tuple(result)


def _require_binding(
    dataset: Mapping[str, Any],
    *,
    field_name: str,
    expected_value: str,
) -> str:
    value = _require_nonempty_string(
        dataset.get(field_name),
        field_name=field_name,
    )

    if value != expected_value:
        raise EvidenceSufficiencyEvaluationError(
            f"{field_name} does not match the accepted binding."
        )

    return value


@dataclass(frozen=True)
class _SourceQuery:
    """Validated source-query behavior and reviewed evidence."""

    question: str
    expected_behavior: str
    relevance_grades: Mapping[str, int]


def _query_contract(
    retrieval_dataset: Mapping[str, Any],
) -> dict[str, _SourceQuery]:
    query_values = _require_sequence(
        retrieval_dataset.get("queries"),
        field_name="retrieval_dataset.queries",
    )
    queries_by_id: dict[str, _SourceQuery] = {}

    for position, query_value in enumerate(query_values):
        query = _require_mapping(
            query_value,
            field_name=f"retrieval_dataset.queries[{position}]",
        )
        query_id = _require_nonempty_string(
            query.get("query_id"),
            field_name=(
                f"retrieval_dataset.queries[{position}].query_id"
            ),
        )
        question = _require_nonempty_string(
            query.get("question"),
            field_name=f"{query_id}.question",
        )
        expected_behavior = _require_nonempty_string(
            query.get("expected_behavior"),
            field_name=f"{query_id}.expected_behavior",
        )

        if expected_behavior not in {
            ANSWER_ACTION,
            ABSTAIN_ACTION,
        }:
            raise EvidenceSufficiencyEvaluationError(
                f"{query_id}.expected_behavior must be "
                "'answer' or 'abstain'."
            )

        judgment_values = _require_sequence(
            query.get("relevance_judgments"),
            field_name=f"{query_id}.relevance_judgments",
        )
        relevance_grades: dict[str, int] = {}

        for judgment_position, judgment_value in enumerate(
            judgment_values
        ):
            judgment = _require_mapping(
                judgment_value,
                field_name=(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}]"
                ),
            )
            passage_id = _require_nonempty_string(
                judgment.get("passage_id"),
                field_name=(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}].passage_id"
                ),
            )
            grade = judgment.get("relevance_grade")

            if (
                not isinstance(grade, int)
                or isinstance(grade, bool)
                or grade not in {1, 2}
            ):
                raise EvidenceSufficiencyEvaluationError(
                    f"{query_id}.relevance_judgments"
                    f"[{judgment_position}].relevance_grade "
                    "must be 1 or 2."
                )

            if passage_id in relevance_grades:
                raise EvidenceSufficiencyEvaluationError(
                    f"{query_id} contains duplicate reviewed "
                    f"passage_id: {passage_id}."
                )

            relevance_grades[passage_id] = grade

        if expected_behavior == ANSWER_ACTION:
            if not relevance_grades:
                raise EvidenceSufficiencyEvaluationError(
                    f"{query_id}: source answer query requires "
                    "reviewed evidence."
                )

            if 2 not in relevance_grades.values():
                raise EvidenceSufficiencyEvaluationError(
                    f"{query_id}: source answer query requires "
                    "reviewed grade 2 evidence."
                )
        elif relevance_grades:
            raise EvidenceSufficiencyEvaluationError(
                f"{query_id}: source abstention query must not "
                "contain reviewed evidence."
            )

        if query_id in queries_by_id:
            raise EvidenceSufficiencyEvaluationError(
                f"Duplicate retrieval query_id: {query_id}."
            )

        queries_by_id[query_id] = _SourceQuery(
            question=question,
            expected_behavior=expected_behavior,
            relevance_grades=relevance_grades,
        )

    if not queries_by_id:
        raise EvidenceSufficiencyEvaluationError(
            "retrieval_dataset.queries must be nonempty."
        )

    return queries_by_id


def _passage_ids(
    passages: Sequence[Mapping[str, Any]],
) -> set[str]:
    values = _require_sequence(
        passages,
        field_name="passages",
    )
    result: set[str] = set()

    if not values:
        raise EvidenceSufficiencyEvaluationError(
            "passages must be nonempty."
        )

    for position, passage_value in enumerate(values):
        passage = _require_mapping(
            passage_value,
            field_name=f"passages[{position}]",
        )
        passage_id = _require_nonempty_string(
            passage.get("passage_id"),
            field_name=f"passages[{position}].passage_id",
        )

        if passage_id in result:
            raise EvidenceSufficiencyEvaluationError(
                f"Duplicate passage_id: {passage_id}."
            )

        result.add(passage_id)

    return result


def _validate_case_relationships(
    *,
    case_id: str,
    evidence_status: str,
    response_action: str,
    evidence_passage_ids: tuple[str, ...],
    reason_codes: tuple[str, ...],
    missing_information: tuple[str, ...],
) -> None:
    if evidence_status == SUFFICIENT_STATUS:
        if not evidence_passage_ids:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}: sufficient cases require nonempty evidence."
            )

        if response_action != ANSWER_ACTION:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}: sufficient cases require answer action."
            )

        if reason_codes:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}: sufficient cases require empty reason_codes."
            )

        if missing_information:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}: sufficient cases require empty "
                "missing_information."
            )

        return

    if response_action != ABSTAIN_ACTION:
        raise EvidenceSufficiencyEvaluationError(
            f"{case_id}: insufficient cases require abstain action."
        )

    if not reason_codes:
        raise EvidenceSufficiencyEvaluationError(
            f"{case_id}: insufficient cases require reason_codes."
        )

    if not missing_information:
        raise EvidenceSufficiencyEvaluationError(
            f"{case_id}: insufficient cases require missing_information."
        )


def validate_evidence_sufficiency_dataset(
    dataset: Mapping[str, Any],
    *,
    retrieval_dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
    passage_artifact_sha256: str,
    retrieval_dataset_sha256: str,
) -> None:
    """Validate one immutable evidence-sufficiency dataset."""

    value = _require_mapping(
        dataset,
        field_name="dataset",
    )
    retrieval = _require_mapping(
        retrieval_dataset,
        field_name="retrieval_dataset",
    )
    corpus_manifest = _require_mapping(
        manifest,
        field_name="manifest",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=DATASET_FIELDS,
        object_name="dataset",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != EVIDENCE_SUFFICIENCY_SCHEMA_VERSION:
        raise EvidenceSufficiencyEvaluationError(
            "schema_version must be 1.0."
        )

    dataset_id = _require_nonempty_string(
        value.get("dataset_id"),
        field_name="dataset_id",
    )

    if dataset_id != EVIDENCE_SUFFICIENCY_DATASET_ID:
        raise EvidenceSufficiencyEvaluationError(
            "dataset_id is not the evidence-sufficiency dataset ID."
        )

    _require_version(
        value.get("dataset_version"),
        field_name="dataset_version",
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

    retrieval_corpus_id = _require_nonempty_string(
        retrieval.get("corpus_id"),
        field_name="retrieval_dataset.corpus_id",
    )
    retrieval_corpus_version = _require_nonempty_string(
        retrieval.get("corpus_version"),
        field_name="retrieval_dataset.corpus_version",
    )

    if retrieval_corpus_id != expected_corpus_id:
        raise EvidenceSufficiencyEvaluationError(
            "retrieval_dataset.corpus_id does not match manifest."
        )

    if retrieval_corpus_version != expected_corpus_version:
        raise EvidenceSufficiencyEvaluationError(
            "retrieval_dataset.corpus_version does not match manifest."
        )

    expected_passage_schema = _require_nonempty_string(
        retrieval.get("passage_schema_version"),
        field_name="retrieval_dataset.passage_schema_version",
    )
    _require_binding(
        value,
        field_name="passage_schema_version",
        expected_value=expected_passage_schema,
    )

    accepted_passage_sha256 = _require_sha256(
        passage_artifact_sha256,
        field_name="passage_artifact_sha256 argument",
    )
    retrieval_passage_sha256 = _require_sha256(
        retrieval.get("passage_artifact_sha256"),
        field_name="retrieval_dataset.passage_artifact_sha256",
    )

    if retrieval_passage_sha256 != accepted_passage_sha256:
        raise EvidenceSufficiencyEvaluationError(
            "retrieval_dataset.passage_artifact_sha256 does not match "
            "the accepted passage artifact."
        )

    dataset_passage_sha256 = _require_sha256(
        value.get("passage_artifact_sha256"),
        field_name="passage_artifact_sha256",
    )

    if dataset_passage_sha256 != accepted_passage_sha256:
        raise EvidenceSufficiencyEvaluationError(
            "passage_artifact_sha256 does not match the accepted binding."
        )

    expected_retrieval_id = _require_nonempty_string(
        retrieval.get("dataset_id"),
        field_name="retrieval_dataset.dataset_id",
    )

    if expected_retrieval_id != RETRIEVAL_EVALUATION_DATASET_ID:
        raise EvidenceSufficiencyEvaluationError(
            "retrieval_dataset.dataset_id is not supported."
        )

    _require_binding(
        value,
        field_name="source_retrieval_dataset_id",
        expected_value=expected_retrieval_id,
    )

    expected_retrieval_version = _require_version(
        retrieval.get("dataset_version"),
        field_name="retrieval_dataset.dataset_version",
    )
    _require_binding(
        value,
        field_name="source_retrieval_dataset_version",
        expected_value=expected_retrieval_version,
    )

    accepted_retrieval_sha256 = _require_sha256(
        retrieval_dataset_sha256,
        field_name="retrieval_dataset_sha256 argument",
    )
    dataset_retrieval_sha256 = _require_sha256(
        value.get("source_retrieval_dataset_sha256"),
        field_name="source_retrieval_dataset_sha256",
    )

    if dataset_retrieval_sha256 != accepted_retrieval_sha256:
        raise EvidenceSufficiencyEvaluationError(
            "source_retrieval_dataset_sha256 does not match the "
            "accepted binding."
        )

    queries_by_id = _query_contract(retrieval)
    known_passage_ids = _passage_ids(passages)

    for query_id, source_query in queries_by_id.items():
        for passage_id in source_query.relevance_grades:
            if passage_id not in known_passage_ids:
                raise EvidenceSufficiencyEvaluationError(
                    f"{query_id}: reviewed passage_id is not "
                    "present in the accepted passage artifact: "
                    f"{passage_id}."
                )

    cases = _require_sequence(
        value.get("cases"),
        field_name="cases",
    )

    if not cases:
        raise EvidenceSufficiencyEvaluationError(
            "cases must be nonempty."
        )

    case_count = _require_nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )

    if case_count != len(cases):
        raise EvidenceSufficiencyEvaluationError(
            "case_count does not match the number of cases."
        )

    seen_case_ids: set[str] = set()

    for position, case_value in enumerate(cases):
        case = _require_mapping(
            case_value,
            field_name=f"cases[{position}]",
        )
        _reject_unknown_fields(
            case,
            allowed_fields=CASE_FIELDS,
            object_name="case",
        )

        case_id = _require_nonempty_string(
            case.get("case_id"),
            field_name=f"cases[{position}].case_id",
        )

        if case_id in seen_case_ids:
            raise EvidenceSufficiencyEvaluationError(
                f"Duplicate case_id: {case_id}."
            )

        seen_case_ids.add(case_id)

        query_id = _require_nonempty_string(
            case.get("query_id"),
            field_name=f"{case_id}.query_id",
        )

        if query_id not in queries_by_id:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}.query_id is not present in the source "
                "retrieval dataset."
            )

        source_query = queries_by_id[query_id]
        question = _require_nonempty_string(
            case.get("question"),
            field_name=f"{case_id}.question",
        )

        if question != source_query.question:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}.question does not match source query question."
            )

        evidence_passage_ids = _require_unique_strings(
            case.get("evidence_passage_ids"),
            field_name="evidence_passage_ids",
            allow_empty=True,
        )

        for passage_id in evidence_passage_ids:
            if passage_id not in known_passage_ids:
                raise EvidenceSufficiencyEvaluationError(
                    f"{case_id}.evidence_passage_ids contains unknown "
                    f"passage_id: {passage_id}."
                )

        evidence_status = _require_nonempty_string(
            case.get("expected_evidence_status"),
            field_name=f"{case_id}.expected_evidence_status",
        )

        if evidence_status not in ALLOWED_EVIDENCE_STATUSES:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}.expected_evidence_status must be "
                "'sufficient' or 'insufficient'."
            )

        response_action = _require_nonempty_string(
            case.get("expected_response_action"),
            field_name=f"{case_id}.expected_response_action",
        )

        if response_action not in ALLOWED_RESPONSE_ACTIONS:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}.expected_response_action must be "
                "'answer' or 'abstain'."
            )

        reason_codes = _require_unique_strings(
            case.get("reason_codes"),
            field_name="reason_codes",
            allow_empty=True,
        )
        missing_information = _require_unique_strings(
            case.get("missing_information"),
            field_name="missing_information",
            allow_empty=True,
        )

        _validate_case_relationships(
            case_id=case_id,
            evidence_status=evidence_status,
            response_action=response_action,
            evidence_passage_ids=evidence_passage_ids,
            reason_codes=reason_codes,
            missing_information=missing_information,
        )

        if (
            source_query.expected_behavior == ABSTAIN_ACTION
            and evidence_status != INSUFFICIENT_STATUS
        ):
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}: source abstention query must remain "
                "insufficient."
            )

        if evidence_status == SUFFICIENT_STATUS:
            unreviewed_passage_ids = sorted(
                set(evidence_passage_ids)
                - set(source_query.relevance_grades)
            )

            if unreviewed_passage_ids:
                raise EvidenceSufficiencyEvaluationError(
                    f"{case_id}: sufficient cases require only "
                    "reviewed evidence; unreviewed passage IDs: "
                    f"{unreviewed_passage_ids}."
                )

            if not any(
                source_query.relevance_grades[passage_id] == 2
                for passage_id in evidence_passage_ids
            ):
                raise EvidenceSufficiencyEvaluationError(
                    f"{case_id}: sufficient cases require at least "
                    "one reviewed grade 2 passage."
                )

        unknown_reason_codes = sorted(
            set(reason_codes) - ALLOWED_REASON_CODES
        )

        if unknown_reason_codes:
            raise EvidenceSufficiencyEvaluationError(
                f"{case_id}.reason_codes contains unsupported values: "
                f"{unknown_reason_codes}."
            )

        _require_nonempty_string(
            case.get("rationale"),
            field_name=f"{case_id}.rationale",
        )
        _require_unique_strings(
            case.get("evaluation_tags"),
            field_name="evaluation_tags",
            allow_empty=False,
        )
