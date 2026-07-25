"""Validate label-blind evidence-sufficiency case construction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
    write_annotation_json_artifact,
)

CASE_CONSTRUCTION_ID = "policyproof-evidence-sufficiency-case-construction"
CASE_CONSTRUCTION_SCHEMA_VERSION = "1.0"

QUERY_INVENTORY_ID = "policyproof-evidence-sufficiency-query-inventory"

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

CONSTRUCTION_FIELDS = frozenset(
    {
        "schema_version",
        "construction_id",
        "construction_version",
        "query_inventory_id",
        "query_inventory_version",
        "query_inventory_sha256",
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
        "evidence_passage_ids",
        "question_structure_codes",
        "evidence_structure_codes",
        "complete_reference_case_id",
    }
)

QUERY_INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "inventory_id",
        "inventory_version",
        "corpus_id",
        "corpus_version",
        "passage_schema_version",
        "passage_artifact_sha256",
        "development_evidence_dataset_id",
        "development_evidence_dataset_version",
        "development_evidence_dataset_sha256",
        "query_count",
        "queries",
    }
)

QUERY_FIELDS = frozenset(
    {
        "query_id",
        "question",
        "document_scope",
        "question_structure_codes",
        "source_logical_source_keys",
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

VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvidenceSufficiencyCaseConstructionError(ValueError):
    """Raised when a case-construction artifact is invalid."""


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencyCaseConstructionError(f"{field_name} must be an object.")

    return value


def _require_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EvidenceSufficiencyCaseConstructionError(f"{field_name} must be an array.")

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencyCaseConstructionError(f"{field_name} must be a nonempty string.")

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

    if VERSION_PATTERN.fullmatch(version) is None:
        raise EvidenceSufficiencyCaseConstructionError(f"{field_name} must be a semantic version.")

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

    if SHA256_PATTERN.fullmatch(sha256) is None:
        raise EvidenceSufficiencyCaseConstructionError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )

    return sha256


def _require_nonnegative_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceSufficiencyCaseConstructionError(
            f"{field_name} must be a nonnegative integer."
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
        raise EvidenceSufficiencyCaseConstructionError(
            f"unknown {object_name} fields: {unknown_fields}."
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
        raise EvidenceSufficiencyCaseConstructionError(
            f"{field_name} does not match the accepted binding."
        )


def _require_unique_strings(
    value: Any,
    *,
    field_name: str,
    allowed_values: frozenset[str] | None = None,
    require_nonempty: bool = True,
) -> list[str]:
    values = _require_sequence(
        value,
        field_name=field_name,
    )

    if require_nonempty and not values:
        raise EvidenceSufficiencyCaseConstructionError(f"{field_name} must be nonempty.")

    result: list[str] = []
    seen: set[str] = set()

    for position, raw_item in enumerate(values):
        item = _require_nonempty_string(
            raw_item,
            field_name=f"{field_name}[{position}]",
        )

        if item in seen:
            raise EvidenceSufficiencyCaseConstructionError(f"duplicate {field_name}: {item}.")

        if allowed_values is not None and item not in allowed_values:
            raise EvidenceSufficiencyCaseConstructionError(
                f"{field_name} contains unsupported value: {item}."
            )

        seen.add(item)
        result.append(item)

    return result


def _manifest_document_ids(
    manifest: Mapping[str, Any],
) -> set[str]:
    documents = _require_sequence(
        manifest.get("documents"),
        field_name="manifest.documents",
    )

    if not documents:
        raise EvidenceSufficiencyCaseConstructionError("manifest.documents must be nonempty.")

    document_ids: set[str] = set()

    for position, raw_document in enumerate(documents):
        document = _require_mapping(
            raw_document,
            field_name=f"manifest.documents[{position}]",
        )
        document_id = _require_nonempty_string(
            document.get("document_id"),
            field_name=(f"manifest.documents[{position}].document_id"),
        )

        if document_id in document_ids:
            raise EvidenceSufficiencyCaseConstructionError(
                f"duplicate manifest document_id: {document_id}."
            )

        document_ids.add(document_id)

    return document_ids


def _accepted_passage_contract(
    passages: Sequence[Mapping[str, Any]],
    *,
    manifest_document_ids: set[str],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    values = _require_sequence(
        passages,
        field_name="passages",
    )

    if not values:
        raise EvidenceSufficiencyCaseConstructionError("passages must be nonempty.")

    passages_by_id: dict[str, Mapping[str, Any]] = {}
    passage_schema_version: str | None = None

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
        document_id = _require_nonempty_string(
            passage.get("document_id"),
            field_name=f"passages[{position}].document_id",
        )

        if document_id not in manifest_document_ids:
            raise EvidenceSufficiencyCaseConstructionError(
                f"passage {passage_id} has unknown document_id."
            )

        if passage_id in passages_by_id:
            raise EvidenceSufficiencyCaseConstructionError(f"duplicate passage_id: {passage_id}.")

        if passage_schema_version is None:
            passage_schema_version = schema_version
        elif schema_version != passage_schema_version:
            raise EvidenceSufficiencyCaseConstructionError(
                "passages contain inconsistent schema_version values."
            )

        passages_by_id[passage_id] = passage

    if passage_schema_version is None:
        raise EvidenceSufficiencyCaseConstructionError("passages must be nonempty.")

    return passages_by_id, passage_schema_version


def _query_contract(
    query_inventory: Mapping[str, Any],
    *,
    manifest_document_ids: set[str],
    corpus_id: str,
    corpus_version: str,
    passage_schema_version: str,
    passage_artifact_sha256: str,
) -> tuple[
    dict[str, Mapping[str, Any]],
    str,
]:
    inventory = _require_mapping(
        query_inventory,
        field_name="query inventory",
    )
    _reject_unknown_fields(
        inventory,
        allowed_fields=QUERY_INVENTORY_FIELDS,
        object_name="query inventory",
    )

    inventory_id = _require_nonempty_string(
        inventory.get("inventory_id"),
        field_name="query_inventory.inventory_id",
    )

    if inventory_id != QUERY_INVENTORY_ID:
        raise EvidenceSufficiencyCaseConstructionError(
            "query_inventory.inventory_id is not accepted."
        )

    inventory_version = _require_version(
        inventory.get("inventory_version"),
        field_name="query_inventory.inventory_version",
    )

    binding_fields = {
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "passage_schema_version": passage_schema_version,
        "passage_artifact_sha256": passage_artifact_sha256,
    }

    for field_name, expected_value in binding_fields.items():
        _require_binding(
            inventory,
            field_name=field_name,
            expected_value=expected_value,
        )

    raw_queries = _require_sequence(
        inventory.get("queries"),
        field_name="query_inventory.queries",
    )

    if not raw_queries:
        raise EvidenceSufficiencyCaseConstructionError("query_inventory.queries must be nonempty.")

    query_count = _require_nonnegative_integer(
        inventory.get("query_count"),
        field_name="query_inventory.query_count",
    )

    if query_count != len(raw_queries):
        raise EvidenceSufficiencyCaseConstructionError(
            "query_inventory.query_count does not match queries."
        )

    queries_by_id: dict[str, Mapping[str, Any]] = {}

    for position, raw_query in enumerate(raw_queries):
        query = _require_mapping(
            raw_query,
            field_name=f"query_inventory.queries[{position}]",
        )
        _reject_unknown_fields(
            query,
            allowed_fields=QUERY_FIELDS,
            object_name="query inventory query",
        )

        query_id = _require_nonempty_string(
            query.get("query_id"),
            field_name=(f"query_inventory.queries[{position}].query_id"),
        )

        if query_id in queries_by_id:
            raise EvidenceSufficiencyCaseConstructionError(
                f"duplicate query inventory query_id: {query_id}."
            )

        _require_nonempty_string(
            query.get("question"),
            field_name=f"{query_id}.question",
        )

        document_scope = _require_unique_strings(
            query.get("document_scope"),
            field_name=f"{query_id}.document_scope",
        )

        unknown_documents = sorted(set(document_scope) - manifest_document_ids)

        if unknown_documents:
            raise EvidenceSufficiencyCaseConstructionError(
                f"{query_id}.document_scope contains unknown documents: {unknown_documents}."
            )

        _require_unique_strings(
            query.get("question_structure_codes"),
            field_name=f"{query_id}.question_structure_codes",
            allowed_values=ALLOWED_QUESTION_STRUCTURE_CODES,
        )
        _require_unique_strings(
            query.get("source_logical_source_keys"),
            field_name=f"{query_id}.source_logical_source_keys",
            require_nonempty=False,
        )

        queries_by_id[query_id] = query

    return queries_by_id, inventory_version


def _validate_case(
    raw_case: Any,
    *,
    position: int,
    queries_by_id: Mapping[str, Mapping[str, Any]],
    passages_by_id: Mapping[str, Mapping[str, Any]],
    seen_case_ids: set[str],
) -> dict[str, Any]:
    case = _require_mapping(
        raw_case,
        field_name=f"cases[{position}]",
    )

    exposed_fields = sorted(set(case) & FORBIDDEN_CASE_FIELDS)

    if exposed_fields:
        raise EvidenceSufficiencyCaseConstructionError(
            "case construction must not expose hidden label or "
            f"evaluation fields: {exposed_fields}."
        )

    _reject_unknown_fields(
        case,
        allowed_fields=CASE_FIELDS,
        object_name="case construction case",
    )

    case_id = _require_nonempty_string(
        case.get("case_id"),
        field_name=f"cases[{position}].case_id",
    )

    if case_id in seen_case_ids:
        raise EvidenceSufficiencyCaseConstructionError(f"duplicate case_id: {case_id}.")

    seen_case_ids.add(case_id)

    query_id = _require_nonempty_string(
        case.get("query_id"),
        field_name=f"{case_id}.query_id",
    )

    if query_id not in queries_by_id:
        raise EvidenceSufficiencyCaseConstructionError(
            f"{case_id} contains unknown query_id: {query_id}."
        )

    accepted_query = queries_by_id[query_id]

    question = _require_nonempty_string(
        case.get("question"),
        field_name=f"{case_id}.question",
    )

    if question != accepted_query["question"]:
        raise EvidenceSufficiencyCaseConstructionError(
            f"{case_id}.question does not match query inventory."
        )

    question_structure_codes = _require_unique_strings(
        case.get("question_structure_codes"),
        field_name=f"{case_id}.question_structure_codes",
        allowed_values=ALLOWED_QUESTION_STRUCTURE_CODES,
    )

    if question_structure_codes != accepted_query["question_structure_codes"]:
        raise EvidenceSufficiencyCaseConstructionError(
            f"{case_id}.question_structure_codes do not match query inventory."
        )

    raw_evidence_passage_ids = _require_sequence(
        case.get("evidence_passage_ids"),
        field_name=f"{case_id}.evidence_passage_ids",
    )

    evidence_passage_ids: list[str] = []
    seen_evidence_passage_ids: set[str] = set()

    for evidence_position, raw_passage_id in enumerate(raw_evidence_passage_ids):
        passage_id = _require_nonempty_string(
            raw_passage_id,
            field_name=(f"{case_id}.evidence_passage_ids[{evidence_position}]"),
        )

        if passage_id in seen_evidence_passage_ids:
            raise EvidenceSufficiencyCaseConstructionError(
                f"duplicate evidence_passage_ids: {passage_id}."
            )

        seen_evidence_passage_ids.add(passage_id)
        evidence_passage_ids.append(passage_id)

    if not evidence_passage_ids:
        raise EvidenceSufficiencyCaseConstructionError(
            f"{case_id}.evidence_passage_ids must be nonempty."
        )

    for passage_id in evidence_passage_ids:
        if passage_id not in passages_by_id:
            raise EvidenceSufficiencyCaseConstructionError(
                f"{case_id} contains unknown evidence_passage_ids value: {passage_id}."
            )

    evidence_structure_codes = _require_unique_strings(
        case.get("evidence_structure_codes"),
        field_name=f"{case_id}.evidence_structure_codes",
        allowed_values=ALLOWED_EVIDENCE_STRUCTURE_CODES,
    )
    evidence_structure_code_set = set(evidence_structure_codes)

    if {
        "one_complete_passage",
        "multiple_complementary_passages",
    }.issubset(evidence_structure_code_set):
        raise EvidenceSufficiencyCaseConstructionError(
            "one_complete_passage and multiple_complementary_passages cannot be combined."
        )

    has_declared_distractors = "topically_related_distractors" in evidence_structure_code_set

    if (
        "one_complete_passage" in evidence_structure_code_set
        and not has_declared_distractors
        and len(evidence_passage_ids) != 1
    ):
        raise EvidenceSufficiencyCaseConstructionError(
            "one_complete_passage requires exactly one evidence "
            "passage when no distractors are declared."
        )

    if (
        "multiple_complementary_passages" in evidence_structure_code_set
        and len(evidence_passage_ids) < 2
    ):
        raise EvidenceSufficiencyCaseConstructionError(
            "multiple_complementary_passages requires at least two evidence passages."
        )

    if has_declared_distractors and len(evidence_passage_ids) < 2:
        raise EvidenceSufficiencyCaseConstructionError(
            "topically_related_distractors requires at least two evidence passages."
        )

    evidence_document_ids = {
        _require_nonempty_string(
            passages_by_id[passage_id].get("document_id"),
            field_name=(f"accepted passage {passage_id}.document_id"),
        )
        for passage_id in evidence_passage_ids
    }

    if "multiple_documents" in evidence_structure_code_set and len(evidence_document_ids) < 2:
        raise EvidenceSufficiencyCaseConstructionError(
            "multiple_documents requires evidence from at least two documents."
        )

    if len(evidence_document_ids) >= 2 and "multiple_documents" not in evidence_structure_code_set:
        raise EvidenceSufficiencyCaseConstructionError(
            "evidence spanning multiple documents requires the multiple_documents code."
        )

    raw_reference_case_id = case.get("complete_reference_case_id")
    complete_reference_case_id: str | None = None

    if raw_reference_case_id is not None:
        complete_reference_case_id = _require_nonempty_string(
            raw_reference_case_id,
            field_name=(f"{case_id}.complete_reference_case_id"),
        )

    has_strict_subset_code = "strict_subset_of_complete_evidence" in evidence_structure_code_set
    has_incomplete_code = "incomplete_evidence_set" in evidence_structure_code_set

    if has_strict_subset_code and complete_reference_case_id is None:
        raise EvidenceSufficiencyCaseConstructionError(
            "strict_subset_of_complete_evidence requires complete_reference_case_id."
        )

    if has_incomplete_code and complete_reference_case_id is None:
        raise EvidenceSufficiencyCaseConstructionError(
            "incomplete_evidence_set requires complete_reference_case_id."
        )

    if complete_reference_case_id is not None and not (
        has_strict_subset_code or has_incomplete_code
    ):
        raise EvidenceSufficiencyCaseConstructionError(
            "complete_reference_case_id requires "
            "strict_subset_of_complete_evidence or "
            "incomplete_evidence_set."
        )

    return {
        "case_id": case_id,
        "query_id": query_id,
        "evidence_passage_ids": evidence_passage_ids,
        "evidence_structure_codes": evidence_structure_codes,
        "complete_reference_case_id": (complete_reference_case_id),
    }


def _validate_case_relationships(
    validated_cases: Sequence[Mapping[str, Any]],
) -> None:
    cases_by_id = {str(case["case_id"]): case for case in validated_cases}

    for case in validated_cases:
        case_id = str(case["case_id"])
        reference_case_id = case["complete_reference_case_id"]

        if reference_case_id is None:
            continue

        if reference_case_id == case_id:
            raise EvidenceSufficiencyCaseConstructionError(
                "complete_reference_case_id cannot reference the case itself."
            )

        if reference_case_id not in cases_by_id:
            raise EvidenceSufficiencyCaseConstructionError(
                f"unknown complete_reference_case_id: {reference_case_id}."
            )

        reference_case = cases_by_id[reference_case_id]

        if reference_case["query_id"] != case["query_id"]:
            raise EvidenceSufficiencyCaseConstructionError(
                "complete_reference_case_id must reference a case with the same query_id."
            )

        reference_structure_codes = set(reference_case["evidence_structure_codes"])
        complete_structure_codes = {
            "one_complete_passage",
            "multiple_complementary_passages",
        }

        if not (reference_structure_codes & complete_structure_codes):
            raise EvidenceSufficiencyCaseConstructionError(
                "reference case must declare a complete evidence structure."
            )

        relational_structure_codes = {
            "strict_subset_of_complete_evidence",
            "incomplete_evidence_set",
        }

        if reference_case["complete_reference_case_id"] is not None or (
            reference_structure_codes & relational_structure_codes
        ):
            raise EvidenceSufficiencyCaseConstructionError(
                "complete reference case must be canonical and must not derive from another case."
            )

        case_evidence = set(case["evidence_passage_ids"])
        reference_evidence = set(reference_case["evidence_passage_ids"])
        case_structure_codes = set(case["evidence_structure_codes"])

        if (
            "strict_subset_of_complete_evidence" in case_structure_codes
            and "incomplete_evidence_set" not in case_structure_codes
        ):
            raise EvidenceSufficiencyCaseConstructionError(
                "strict_subset_of_complete_evidence requires incomplete_evidence_set."
            )

        if (
            "strict_subset_of_complete_evidence" in case_structure_codes
            and not case_evidence < reference_evidence
        ):
            raise EvidenceSufficiencyCaseConstructionError(
                "strict_subset_of_complete_evidence requires the "
                "case evidence to be a strict subset of the "
                "complete reference case."
            )

        if "incomplete_evidence_set" in case_structure_codes and not (
            reference_evidence - case_evidence
        ):
            raise EvidenceSufficiencyCaseConstructionError(
                "incomplete_evidence_set must omit at least one "
                "passage from the complete reference case."
            )


def _validate_unique_query_evidence_pairs(
    validated_cases: Sequence[Mapping[str, Any]],
) -> None:
    seen_pairs: set[tuple[str, tuple[str, ...]]] = set()

    for case in validated_cases:
        pair = (
            str(case["query_id"]),
            tuple(sorted(str(passage_id) for passage_id in case["evidence_passage_ids"])),
        )

        if pair in seen_pairs:
            raise EvidenceSufficiencyCaseConstructionError(
                "duplicate query and evidence passage combination."
            )

        seen_pairs.add(pair)


def validate_evidence_sufficiency_case_construction(
    construction: Mapping[str, Any],
    *,
    query_inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
    query_inventory_sha256: str,
    passage_artifact_sha256: str,
) -> None:
    """Validate one label-blind case-construction artifact."""

    value = _require_mapping(
        construction,
        field_name="case construction",
    )
    corpus_manifest = _require_mapping(
        manifest,
        field_name="manifest",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=CONSTRUCTION_FIELDS,
        object_name="case construction",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != CASE_CONSTRUCTION_SCHEMA_VERSION:
        raise EvidenceSufficiencyCaseConstructionError("schema_version must be 1.0.")

    construction_id = _require_nonempty_string(
        value.get("construction_id"),
        field_name="construction_id",
    )

    if construction_id != CASE_CONSTRUCTION_ID:
        raise EvidenceSufficiencyCaseConstructionError("construction_id is not the accepted ID.")

    _require_version(
        value.get("construction_version"),
        field_name="construction_version",
    )

    accepted_query_inventory_sha256 = _require_sha256(
        query_inventory_sha256,
        field_name="query_inventory_sha256 argument",
    )
    accepted_passage_sha256 = _require_sha256(
        passage_artifact_sha256,
        field_name="passage_artifact_sha256 argument",
    )

    corpus_id = _require_nonempty_string(
        corpus_manifest.get("corpus_id"),
        field_name="manifest.corpus_id",
    )
    corpus_version = _require_nonempty_string(
        corpus_manifest.get("corpus_version"),
        field_name="manifest.corpus_version",
    )
    manifest_document_ids = _manifest_document_ids(corpus_manifest)
    passages_by_id, passage_schema_version = _accepted_passage_contract(
        passages,
        manifest_document_ids=manifest_document_ids,
    )
    queries_by_id, query_inventory_version = _query_contract(
        query_inventory,
        manifest_document_ids=manifest_document_ids,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        passage_schema_version=passage_schema_version,
        passage_artifact_sha256=accepted_passage_sha256,
    )

    bindings = {
        "query_inventory_id": QUERY_INVENTORY_ID,
        "query_inventory_version": query_inventory_version,
        "query_inventory_sha256": (accepted_query_inventory_sha256),
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "passage_schema_version": passage_schema_version,
        "passage_artifact_sha256": accepted_passage_sha256,
    }

    for field_name, expected_value in bindings.items():
        _require_binding(
            value,
            field_name=field_name,
            expected_value=expected_value,
        )

    raw_cases = _require_sequence(
        value.get("cases"),
        field_name="cases",
    )

    if not raw_cases:
        raise EvidenceSufficiencyCaseConstructionError("cases must be nonempty.")

    case_count = _require_nonnegative_integer(
        value.get("case_count"),
        field_name="case_count",
    )

    if case_count != len(raw_cases):
        raise EvidenceSufficiencyCaseConstructionError(
            "case_count does not match the number of cases."
        )

    seen_case_ids: set[str] = set()
    validated_cases: list[dict[str, Any]] = []

    for position, raw_case in enumerate(raw_cases):
        validated_cases.append(
            _validate_case(
                raw_case,
                position=position,
                queries_by_id=queries_by_id,
                passages_by_id=passages_by_id,
                seen_case_ids=seen_case_ids,
            )
        )

    _validate_case_relationships(validated_cases)
    _validate_unique_query_evidence_pairs(validated_cases)


def write_case_construction_json_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically publish a construction artifact without overwriting."""

    try:
        write_annotation_json_artifact(
            artifact,
            output_path,
        )
    except EvidenceSufficiencyAnnotationError as error:
        raise EvidenceSufficiencyCaseConstructionError(str(error)) from error
