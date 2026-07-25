"""Validation for evidence-sufficiency query inventories."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_ID = (
    "policyproof-evidence-sufficiency-query-inventory"
)
EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_SCHEMA_VERSION = "1.0"

VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

INVENTORY_FIELDS = frozenset(
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

FORBIDDEN_QUERY_FIELDS = frozenset(
    {
        "evidence",
        "evidence_passage_ids",
        "expected_behavior",
        "expected_evidence_status",
        "expected_response_action",
        "reason_codes",
        "missing_information",
        "rationale",
        "evaluation_tags",
        "relevance_judgments",
        "policy_prediction",
        "model_score",
        "uncertainty",
        "adjudication",
    }
)

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


class EvidenceSufficiencyQueryInventoryError(ValueError):
    """Raised when a query inventory violates its contract."""


def _require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must be a mapping."
        )

    return value


def _require_sequence(
    value: Any,
    field_name: str,
) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must be an array."
        )

    return value


def _require_nonempty_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must be a non-empty string."
        )

    return value


def _require_version(
    value: Any,
    field_name: str,
) -> str:
    version = _require_nonempty_string(
        value,
        field_name,
    )

    if VERSION_PATTERN.fullmatch(version) is None:
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must use semantic version format."
        )

    return version


def _require_sha256(
    value: Any,
    field_name: str,
) -> str:
    sha256 = _require_nonempty_string(
        value,
        field_name,
    )

    if SHA256_PATTERN.fullmatch(sha256) is None:
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return sha256


def _reject_unknown_fields(
    record: Mapping[str, Any],
    allowed_fields: frozenset[str],
    field_name: str,
) -> None:
    unexpected = sorted(set(record) - allowed_fields)

    if unexpected:
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} contains an unexpected field: "
            f"{unexpected[0]}"
        )


def _require_unique_strings(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    items = _require_sequence(value, field_name)

    if not items and not allow_empty:
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must contain at least one value."
        )

    normalized_items = []

    for index, item in enumerate(items):
        normalized_items.append(
            _require_nonempty_string(
                item,
                f"{field_name}[{index}]",
            )
        )

    if len(set(normalized_items)) != len(normalized_items):
        raise EvidenceSufficiencyQueryInventoryError(
            f"{field_name} must not contain duplicate values."
        )

    return tuple(normalized_items)


def _normalized_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _manifest_document_ids(
    manifest: Mapping[str, Any],
) -> frozenset[str]:
    documents = _require_sequence(
        manifest.get("documents"),
        "manifest.documents",
    )
    document_ids = []

    for index, document in enumerate(documents):
        document_record = _require_mapping(
            document,
            f"manifest.documents[{index}]",
        )
        document_ids.append(
            _require_nonempty_string(
                document_record.get("document_id"),
                f"manifest.documents[{index}].document_id",
            )
        )

    if len(set(document_ids)) != len(document_ids):
        raise EvidenceSufficiencyQueryInventoryError(
            "manifest contains duplicate document IDs."
        )

    return frozenset(document_ids)


def validate_evidence_sufficiency_query_inventory(
    inventory: Mapping[str, Any],
    development_dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
    passage_artifact_sha256: str,
    development_dataset_sha256: str,
) -> None:
    """Validate a label-blind query inventory without mutating inputs."""

    inventory_record = _require_mapping(
        inventory,
        "inventory",
    )
    development_record = _require_mapping(
        development_dataset,
        "development_dataset",
    )
    manifest_record = _require_mapping(
        manifest,
        "manifest",
    )

    _reject_unknown_fields(
        inventory_record,
        INVENTORY_FIELDS,
        "inventory",
    )

    if (
        inventory_record.get("schema_version")
        != EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_SCHEMA_VERSION
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "Unsupported query inventory schema_version."
        )

    if (
        inventory_record.get("inventory_id")
        != EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_ID
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "Unexpected query inventory inventory_id."
        )

    _require_version(
        inventory_record.get("inventory_version"),
        "inventory.inventory_version",
    )

    expected_passage_sha256 = _require_sha256(
        passage_artifact_sha256,
        "passage_artifact_sha256",
    )
    expected_development_sha256 = _require_sha256(
        development_dataset_sha256,
        "development_dataset_sha256",
    )

    if (
        development_record.get("passage_artifact_sha256")
        != expected_passage_sha256
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "development dataset passage_artifact_sha256 "
            "binding mismatch."
        )

    if (
        inventory_record.get("passage_artifact_sha256")
        != expected_passage_sha256
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "inventory passage_artifact_sha256 binding mismatch."
        )

    if (
        inventory_record.get(
            "development_evidence_dataset_sha256"
        )
        != expected_development_sha256
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "inventory development dataset SHA-256 binding mismatch."
        )

    binding_pairs = (
        (
            "corpus_id",
            manifest_record.get("corpus_id"),
        ),
        (
            "corpus_version",
            manifest_record.get("corpus_version"),
        ),
        (
            "corpus_id",
            development_record.get("corpus_id"),
        ),
        (
            "corpus_version",
            development_record.get("corpus_version"),
        ),
        (
            "passage_schema_version",
            development_record.get(
                "passage_schema_version"
            ),
        ),
    )

    for field_name, expected_value in binding_pairs:
        if inventory_record.get(field_name) != expected_value:
            raise EvidenceSufficiencyQueryInventoryError(
                f"inventory {field_name} binding mismatch."
            )

    if (
        inventory_record.get(
            "development_evidence_dataset_id"
        )
        != development_record.get("dataset_id")
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "inventory development dataset ID binding mismatch."
        )

    if (
        inventory_record.get(
            "development_evidence_dataset_version"
        )
        != development_record.get("dataset_version")
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "inventory development dataset version binding mismatch."
        )

    document_ids = _manifest_document_ids(
        manifest_record
    )

    passage_records = _require_sequence(
        passages,
        "passages",
    )
    passage_by_id: dict[str, Mapping[str, Any]] = {}
    source_documents: dict[str, set[str]] = {}
    expected_passage_schema_version = _require_nonempty_string(
        inventory_record.get("passage_schema_version"),
        "inventory.passage_schema_version",
    )

    for index, passage in enumerate(passage_records):
        passage_record = _require_mapping(
            passage,
            f"passages[{index}]",
        )

        if (
            passage_record.get("schema_version")
            != expected_passage_schema_version
        ):
            raise EvidenceSufficiencyQueryInventoryError(
                "passage schema_version binding mismatch at "
                f"passages[{index}]."
            )

        passage_id = _require_nonempty_string(
            passage_record.get("passage_id"),
            f"passages[{index}].passage_id",
        )
        document_id = _require_nonempty_string(
            passage_record.get("document_id"),
            f"passages[{index}].document_id",
        )

        if document_id not in document_ids:
            raise EvidenceSufficiencyQueryInventoryError(
                "passage document_id is not present in the "
                f"manifest: {document_id}"
            )

        logical_source_key = _require_nonempty_string(
            passage_record.get("logical_source_key"),
            f"passages[{index}].logical_source_key",
        )

        if passage_id in passage_by_id:
            raise EvidenceSufficiencyQueryInventoryError(
                f"Duplicate passage_id: {passage_id}"
            )

        passage_by_id[passage_id] = passage_record
        source_documents.setdefault(
            logical_source_key,
            set(),
        ).add(document_id)

    for logical_source_key, documents in source_documents.items():
        if len(documents) > 1:
            raise EvidenceSufficiencyQueryInventoryError(
                "logical source spans multiple documents: "
                f"{logical_source_key}"
            )

    development_cases = _require_sequence(
        development_record.get("cases"),
        "development_dataset.cases",
    )
    development_query_ids: set[str] = set()
    development_questions: set[str] = set()
    development_source_keys: set[str] = set()

    for index, case in enumerate(development_cases):
        case_record = _require_mapping(
            case,
            f"development_dataset.cases[{index}]",
        )
        query_id = _require_nonempty_string(
            case_record.get("query_id"),
            f"development_dataset.cases[{index}].query_id",
        )
        question = _require_nonempty_string(
            case_record.get("question"),
            f"development_dataset.cases[{index}].question",
        )
        evidence_passage_ids = _require_unique_strings(
            case_record.get("evidence_passage_ids"),
            (
                "development_dataset.cases"
                f"[{index}].evidence_passage_ids"
            ),
            allow_empty=True,
        )

        development_query_ids.add(query_id)
        development_questions.add(
            _normalized_question(question)
        )

        for passage_id in evidence_passage_ids:
            passage_record = passage_by_id.get(passage_id)

            if passage_record is None:
                raise EvidenceSufficiencyQueryInventoryError(
                    "development dataset references an unknown "
                    f"passage_id: {passage_id}"
                )

            development_source_keys.add(
                str(passage_record["logical_source_key"])
            )

    queries = _require_sequence(
        inventory_record.get("queries"),
        "inventory.queries",
    )

    if not queries:
        raise EvidenceSufficiencyQueryInventoryError(
            "inventory must contain at least one query."
        )

    query_count = inventory_record.get("query_count")

    if (
        not isinstance(query_count, int)
        or isinstance(query_count, bool)
        or query_count != len(queries)
    ):
        raise EvidenceSufficiencyQueryInventoryError(
            "inventory query_count does not match queries."
        )

    seen_query_ids: set[str] = set()
    seen_questions: set[str] = set()

    for index, query in enumerate(queries):
        query_record = _require_mapping(
            query,
            f"inventory.queries[{index}]",
        )

        forbidden = sorted(
            set(query_record) & FORBIDDEN_QUERY_FIELDS
        )

        if forbidden:
            raise EvidenceSufficiencyQueryInventoryError(
                "inventory query contains forbidden field: "
                f"{forbidden[0]}"
            )

        _reject_unknown_fields(
            query_record,
            QUERY_FIELDS,
            f"inventory.queries[{index}]",
        )

        query_id = _require_nonempty_string(
            query_record.get("query_id"),
            f"inventory.queries[{index}].query_id",
        )
        question = _require_nonempty_string(
            query_record.get("question"),
            f"inventory.queries[{index}].question",
        )

        if query_id in development_query_ids:
            raise EvidenceSufficiencyQueryInventoryError(
                f"query_id {query_id!r} is already used by "
                "development data."
            )

        if query_id in seen_query_ids:
            raise EvidenceSufficiencyQueryInventoryError(
                f"Duplicate query_id: {query_id}"
            )

        seen_query_ids.add(query_id)

        normalized_question = _normalized_question(
            question
        )

        if normalized_question in seen_questions:
            raise EvidenceSufficiencyQueryInventoryError(
                "question is duplicate after normalization."
            )

        if normalized_question in development_questions:
            raise EvidenceSufficiencyQueryInventoryError(
                "question duplicates development data."
            )

        seen_questions.add(normalized_question)

        document_scope = _require_unique_strings(
            query_record.get("document_scope"),
            f"inventory.queries[{index}].document_scope",
            allow_empty=False,
        )

        unknown_documents = sorted(
            set(document_scope) - document_ids
        )

        if unknown_documents:
            raise EvidenceSufficiencyQueryInventoryError(
                "document_scope contains an unknown document_id: "
                f"{unknown_documents[0]}"
            )

        structure_codes = _require_unique_strings(
            query_record.get(
                "question_structure_codes"
            ),
            (
                f"inventory.queries[{index}]"
                ".question_structure_codes"
            ),
            allow_empty=False,
        )
        unknown_structure_codes = sorted(
            set(structure_codes)
            - ALLOWED_QUESTION_STRUCTURE_CODES
        )

        if unknown_structure_codes:
            raise EvidenceSufficiencyQueryInventoryError(
                "Unknown question structure code: "
                f"{unknown_structure_codes[0]}"
            )

        source_keys = _require_unique_strings(
            query_record.get(
                "source_logical_source_keys"
            ),
            (
                f"inventory.queries[{index}]"
                ".source_logical_source_keys"
            ),
            allow_empty=True,
        )

        for source_key in source_keys:
            if source_key in development_source_keys:
                raise EvidenceSufficiencyQueryInventoryError(
                    "Query references a development-touched "
                    f"logical source: {source_key}"
                )

            documents = source_documents.get(source_key)

            if documents is None:
                raise EvidenceSufficiencyQueryInventoryError(
                    f"Query references unknown logical source: "
                    f"{source_key}"
                )

            if not documents.issubset(
                set(document_scope)
            ):
                raise EvidenceSufficiencyQueryInventoryError(
                    "Logical source document is outside "
                    f"document_scope: {source_key}"
                )
