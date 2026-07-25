from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_case_construction import (
    CASE_CONSTRUCTION_ID,
    EvidenceSufficiencyCaseConstructionError,
    validate_evidence_sufficiency_case_construction,
)

QUERY_INVENTORY_SHA256 = "a" * 64
PASSAGE_ARTIFACT_SHA256 = "b" * 64


def manifest() -> dict[str, Any]:
    return {
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "documents": [
            {
                "document_id": "document-a",
            },
            {
                "document_id": "document-b",
            },
        ],
    }


def passages() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "1.1",
            "passage_id": "passage-a",
            "document_id": "document-a",
            "label": "Section A",
            "citation_text": "Accepted citation text for A.",
            "retrieval_text": "Accepted retrieval text for A.",
            "logical_source_key": "source-a",
        },
        {
            "schema_version": "1.1",
            "passage_id": "passage-b",
            "document_id": "document-b",
            "label": "Section B",
            "citation_text": "Accepted citation text for B.",
            "retrieval_text": "Accepted retrieval text for B.",
            "logical_source_key": "source-b",
        },
        {
            "schema_version": "1.1",
            "passage_id": "passage-c",
            "document_id": "document-a",
            "label": "Section C",
            "citation_text": "Accepted citation text for C.",
            "retrieval_text": "Accepted retrieval text for C.",
            "logical_source_key": "source-c",
        },
    ]


def query_inventory() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "inventory_id": ("policyproof-evidence-sufficiency-query-inventory"),
        "inventory_version": "0.2.0",
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "development_evidence_dataset_id": ("policyproof-evidence-sufficiency-evaluation"),
        "development_evidence_dataset_version": "0.1.0",
        "development_evidence_dataset_sha256": "c" * 64,
        "query_count": 1,
        "queries": [
            {
                "query_id": "pilot-001",
                "question": ("What risk does the supplied source identify?"),
                "document_scope": ["document-a"],
                "question_structure_codes": [
                    "direct_factual_lookup",
                ],
                "source_logical_source_keys": ["source-a"],
            }
        ],
    }


def construction() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "construction_id": CASE_CONSTRUCTION_ID,
        "construction_version": "0.1.0",
        "query_inventory_id": ("policyproof-evidence-sufficiency-query-inventory"),
        "query_inventory_version": "0.2.0",
        "query_inventory_sha256": QUERY_INVENTORY_SHA256,
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "case_count": 1,
        "cases": [
            {
                "case_id": "pilot-001-complete",
                "query_id": "pilot-001",
                "question": ("What risk does the supplied source identify?"),
                "evidence_passage_ids": ["passage-a"],
                "question_structure_codes": [
                    "direct_factual_lookup",
                ],
                "evidence_structure_codes": [
                    "one_complete_passage",
                ],
            }
        ],
    }


def validate(
    value: dict[str, Any],
    *,
    query_inventory_value: dict[str, Any] | None = None,
    manifest_value: dict[str, Any] | None = None,
    passages_value: list[dict[str, Any]] | None = None,
    query_inventory_sha256_value: str = QUERY_INVENTORY_SHA256,
    passage_artifact_sha256_value: str = PASSAGE_ARTIFACT_SHA256,
) -> None:
    inventory = query_inventory() if query_inventory_value is None else query_inventory_value
    corpus_manifest = manifest() if manifest_value is None else manifest_value
    accepted_passages = passages() if passages_value is None else passages_value

    validate_evidence_sufficiency_case_construction(
        value,
        query_inventory=inventory,
        manifest=corpus_manifest,
        passages=accepted_passages,
        query_inventory_sha256=(query_inventory_sha256_value),
        passage_artifact_sha256=(passage_artifact_sha256_value),
    )


def test_valid_case_construction_passes_without_mutation() -> None:
    value = construction()
    original = deepcopy(value)

    validate(value)

    assert value == original


@pytest.mark.parametrize(
    "forbidden_field",
    [
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
    ],
)
def test_case_rejects_label_or_evaluation_fields(
    forbidden_field: str,
) -> None:
    value = construction()
    value["cases"][0][forbidden_field] = "must-not-be-present"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=rf"must not expose.*{forbidden_field}",
    ):
        validate(value)


def test_case_question_must_match_query_inventory() -> None:
    value = construction()
    value["cases"][0]["question"] = "Changed question."

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match="question.*query inventory",
    ):
        validate(value)


def test_case_question_structure_must_match_query_inventory() -> None:
    value = construction()
    value["cases"][0]["question_structure_codes"] = [
        "definition",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match="question_structure_codes.*query inventory",
    ):
        validate(value)


def test_case_rejects_unknown_query_id() -> None:
    value = construction()
    value["cases"][0]["query_id"] = "unknown-query"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match="unknown query_id",
    ):
        validate(value)


def test_case_rejects_unknown_evidence_passage_id() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = [
        "unknown-passage",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match="unknown.*evidence_passage_ids",
    ):
        validate(value)


def test_case_rejects_duplicate_evidence_passage_id() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-a",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match="duplicate evidence_passage_ids",
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("query_inventory_version", "9.9.9"),
        ("query_inventory_sha256", "d" * 64),
        ("corpus_id", "other-corpus"),
        ("corpus_version", "9.9.9"),
        ("passage_schema_version", "9.9"),
        ("passage_artifact_sha256", "e" * 64),
    ],
)
def test_case_construction_rejects_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = construction()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=field_name,
    ):
        validate(value)


def test_one_complete_passage_without_distractors_requires_exactly_one_passage() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-b",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"one_complete_passage.*exactly one",
    ):
        validate(value)


def test_multiple_complementary_passages_requires_at_least_two() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "multiple_complementary_passages",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(r"multiple_complementary_passages.*at least two"),
    ):
        validate(value)


def test_multiple_documents_requires_two_evidence_documents() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "multiple_documents",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"multiple_documents.*at least two.*documents",
    ):
        validate(value)


def test_cross_document_evidence_requires_multiple_documents_code() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-b",
    ]
    value["cases"][0]["evidence_structure_codes"] = [
        "multiple_complementary_passages",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"multiple documents.*requires.*multiple_documents",
    ):
        validate(value)


def test_single_and_multiple_complete_codes_are_incompatible() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-b",
    ]
    value["cases"][0]["evidence_structure_codes"] = [
        "one_complete_passage",
        "multiple_complementary_passages",
        "multiple_documents",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"one_complete_passage.*"
            r"multiple_complementary_passages.*cannot be combined"
        ),
    ):
        validate(value)


def test_one_complete_passage_allows_declared_distractors() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-c",
    ]
    value["cases"][0]["evidence_structure_codes"] = [
        "one_complete_passage",
        "topically_related_distractors",
    ]

    validate(value)


def test_topically_related_distractors_requires_additional_passage() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "one_complete_passage",
        "topically_related_distractors",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"topically_related_distractors.*at least two",
    ):
        validate(value)


def add_complete_reference_case(
    value: dict[str, Any],
) -> None:
    value["case_count"] = 2
    value["cases"].append(
        {
            "case_id": "pilot-001-complete-reference",
            "query_id": "pilot-001",
            "question": ("What risk does the supplied source identify?"),
            "evidence_passage_ids": [
                "passage-a",
                "passage-c",
            ],
            "question_structure_codes": [
                "direct_factual_lookup",
            ],
            "evidence_structure_codes": [
                "multiple_complementary_passages",
            ],
        }
    )


def test_strict_subset_requires_complete_reference_case_id() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"strict_subset_of_complete_evidence.*"
            r"complete_reference_case_id"
        ),
    ):
        validate(value)


def test_incomplete_evidence_requires_complete_reference_case_id() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "incomplete_evidence_set",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"incomplete_evidence_set.*"
            r"complete_reference_case_id"
        ),
    ):
        validate(value)


def test_complete_reference_case_id_must_exist() -> None:
    value = construction()
    value["cases"][0]["complete_reference_case_id"] = "missing-complete-reference"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"unknown complete_reference_case_id",
    ):
        validate(value)


def test_complete_reference_case_cannot_reference_itself() -> None:
    value = construction()
    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"complete_reference_case_id.*itself",
    ):
        validate(value)


def test_strict_subset_reference_must_share_query_id() -> None:
    inventory = query_inventory()
    inventory["query_count"] = 2
    inventory["queries"].append(
        {
            "query_id": "pilot-002",
            "question": ("What control does the second source identify?"),
            "document_scope": ["document-b"],
            "question_structure_codes": [
                "direct_factual_lookup",
            ],
            "source_logical_source_keys": ["source-b"],
        }
    )

    value = construction()
    value["cases"][0]["complete_reference_case_id"] = "pilot-002-complete-reference"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]
    value["case_count"] = 2
    value["cases"].append(
        {
            "case_id": "pilot-002-complete-reference",
            "query_id": "pilot-002",
            "question": ("What control does the second source identify?"),
            "evidence_passage_ids": ["passage-b"],
            "question_structure_codes": [
                "direct_factual_lookup",
            ],
            "evidence_structure_codes": [
                "one_complete_passage",
            ],
        }
    )

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(r"complete_reference_case_id.*same query_id"),
    ):
        validate(
            value,
            query_inventory_value=inventory,
        )


def test_strict_subset_must_be_actual_strict_subset() -> None:
    value = construction()
    add_complete_reference_case(value)

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete-reference"
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-c",
    ]
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
        "multiple_complementary_passages",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"strict_subset_of_complete_evidence.*strict subset",
    ):
        validate(value)


def test_incomplete_case_must_omit_reference_evidence() -> None:
    value = construction()
    add_complete_reference_case(value)

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete-reference"
    value["cases"][0]["evidence_passage_ids"] = [
        "passage-a",
        "passage-c",
    ]
    value["cases"][0]["evidence_structure_codes"] = [
        "incomplete_evidence_set",
        "multiple_complementary_passages",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"incomplete_evidence_set.*omit.*reference",
    ):
        validate(value)


def test_reference_case_must_declare_complete_structure() -> None:
    value = construction()
    add_complete_reference_case(value)

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete-reference"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]
    value["cases"][1]["evidence_structure_codes"] = [
        "incomplete_evidence_set",
    ]
    value["cases"][1]["complete_reference_case_id"] = "pilot-001-complete"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"reference case.*complete evidence structure",
    ):
        validate(value)


def test_complete_reference_id_rejected_without_relational_code() -> None:
    value = construction()
    add_complete_reference_case(value)

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete-reference"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"complete_reference_case_id.*requires.*"
            r"strict_subset_of_complete_evidence.*"
            r"incomplete_evidence_set"
        ),
    ):
        validate(value)


def test_valid_relational_construction_passes_without_mutation() -> None:
    value = construction()
    add_complete_reference_case(value)

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete-reference"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]

    original = deepcopy(value)

    validate(value)

    assert value == original


def test_case_construction_rejects_unknown_top_level_field() -> None:
    value = construction()
    value["unexpected_field"] = "not-allowed"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"unknown case construction fields",
    ):
        validate(value)


def test_case_construction_rejects_unknown_case_field() -> None:
    value = construction()
    value["cases"][0]["unexpected_field"] = "not-allowed"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"unknown case construction case fields",
    ):
        validate(value)


def test_case_construction_rejects_wrong_schema_version() -> None:
    value = construction()
    value["schema_version"] = "2.0"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"schema_version",
    ):
        validate(value)


def test_case_construction_rejects_wrong_construction_id() -> None:
    value = construction()
    value["construction_id"] = "other-construction"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"construction_id",
    ):
        validate(value)


def test_case_construction_rejects_invalid_version_format() -> None:
    value = construction()
    value["construction_version"] = "version-one"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"construction_version.*semantic version",
    ):
        validate(value)


def test_case_construction_rejects_empty_cases() -> None:
    value = construction()
    value["case_count"] = 0
    value["cases"] = []

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"cases must be nonempty",
    ):
        validate(value)


def test_case_construction_rejects_case_count_mismatch() -> None:
    value = construction()
    value["case_count"] = 2

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"case_count",
    ):
        validate(value)


def test_case_construction_rejects_duplicate_case_id() -> None:
    value = construction()
    duplicate = deepcopy(value["cases"][0])
    value["case_count"] = 2
    value["cases"].append(duplicate)

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate case_id",
    ):
        validate(value)


def test_duplicate_query_evidence_pair_is_order_insensitive() -> None:
    value = construction()
    add_complete_reference_case(value)

    duplicate = deepcopy(value["cases"][1])
    duplicate["case_id"] = "pilot-001-complete-reference-duplicate"
    duplicate["evidence_passage_ids"].reverse()

    value["case_count"] = 3
    value["cases"].append(duplicate)

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate query and evidence passage",
    ):
        validate(value)


def test_case_rejects_empty_evidence_passage_ids() -> None:
    value = construction()
    value["cases"][0]["evidence_passage_ids"] = []

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"evidence_passage_ids must be nonempty",
    ):
        validate(value)


def test_case_rejects_unknown_question_structure_code() -> None:
    value = construction()
    value["cases"][0]["question_structure_codes"] = [
        "unsupported-question-structure",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(r"question_structure_codes.*unsupported value"),
    ):
        validate(value)


def test_case_rejects_duplicate_question_structure_code() -> None:
    value = construction()
    value["cases"][0]["question_structure_codes"] = [
        "direct_factual_lookup",
        "direct_factual_lookup",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate.*question_structure_codes",
    ):
        validate(value)


def test_case_rejects_empty_question_structure_codes() -> None:
    value = construction()
    value["cases"][0]["question_structure_codes"] = []

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"question_structure_codes must be nonempty",
    ):
        validate(value)


def test_case_rejects_unknown_evidence_structure_code() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "unsupported-evidence-structure",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(r"evidence_structure_codes.*unsupported value"),
    ):
        validate(value)


def test_case_rejects_duplicate_evidence_structure_code() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = [
        "one_complete_passage",
        "one_complete_passage",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate.*evidence_structure_codes",
    ):
        validate(value)


def test_case_rejects_empty_evidence_structure_codes() -> None:
    value = construction()
    value["cases"][0]["evidence_structure_codes"] = []

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"evidence_structure_codes must be nonempty",
    ):
        validate(value)


def test_rejects_invalid_query_inventory_sha256_argument() -> None:
    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"query_inventory_sha256 argument.*"
            r"lowercase SHA-256"
        ),
    ):
        validate(
            construction(),
            query_inventory_sha256_value="not-a-digest",
        )


def test_rejects_invalid_passage_artifact_sha256_argument() -> None:
    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"passage_artifact_sha256 argument.*"
            r"lowercase SHA-256"
        ),
    ):
        validate(
            construction(),
            passage_artifact_sha256_value="not-a-digest",
        )


def test_query_inventory_rejects_wrong_inventory_id() -> None:
    inventory = query_inventory()
    inventory["inventory_id"] = "other-inventory"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"inventory_id.*accepted",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_invalid_version_format() -> None:
    inventory = query_inventory()
    inventory["inventory_version"] = "version-two"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"inventory_version.*semantic version",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_empty_queries() -> None:
    inventory = query_inventory()
    inventory["query_count"] = 0
    inventory["queries"] = []

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"query_inventory\.queries must be nonempty",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_query_count_mismatch() -> None:
    inventory = query_inventory()
    inventory["query_count"] = 2

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"query_inventory\.query_count",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_duplicate_query_id() -> None:
    inventory = query_inventory()
    inventory["query_count"] = 2
    inventory["queries"].append(deepcopy(inventory["queries"][0]))

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate query inventory query_id",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_unknown_document_scope() -> None:
    inventory = query_inventory()
    inventory["queries"][0]["document_scope"] = [
        "unknown-document",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"document_scope.*unknown documents",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_unknown_top_level_field() -> None:
    inventory = query_inventory()
    inventory["unexpected_field"] = "not-allowed"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"unknown query inventory fields",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_query_inventory_rejects_unknown_query_field() -> None:
    inventory = query_inventory()
    inventory["queries"][0]["unexpected_field"] = "not-allowed"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"unknown query inventory query fields",
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_manifest_rejects_duplicate_document_id() -> None:
    corpus_manifest = manifest()
    corpus_manifest["documents"].append(deepcopy(corpus_manifest["documents"][0]))

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate manifest document_id",
    ):
        validate(
            construction(),
            manifest_value=corpus_manifest,
        )


def test_manifest_rejects_empty_documents() -> None:
    corpus_manifest = manifest()
    corpus_manifest["documents"] = []

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"manifest\.documents must be nonempty",
    ):
        validate(
            construction(),
            manifest_value=corpus_manifest,
        )


def test_passage_contract_rejects_empty_passages() -> None:
    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"passages must be nonempty",
    ):
        validate(
            construction(),
            passages_value=[],
        )


def test_passage_contract_rejects_duplicate_passage_id() -> None:
    accepted_passages = passages()
    accepted_passages.append(deepcopy(accepted_passages[0]))

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"duplicate passage_id",
    ):
        validate(
            construction(),
            passages_value=accepted_passages,
        )


def test_passage_contract_rejects_inconsistent_schema_version() -> None:
    accepted_passages = passages()
    accepted_passages[1]["schema_version"] = "9.9"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"inconsistent schema_version",
    ):
        validate(
            construction(),
            passages_value=accepted_passages,
        )


def test_passage_contract_rejects_unknown_document_id() -> None:
    accepted_passages = passages()
    accepted_passages[0]["document_id"] = "unknown-document"

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=r"unknown document_id",
    ):
        validate(
            construction(),
            passages_value=accepted_passages,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("corpus_id", "other-corpus"),
        ("corpus_version", "9.9.9"),
        ("passage_schema_version", "9.9"),
        ("passage_artifact_sha256", "d" * 64),
    ],
)
def test_query_inventory_rejects_accepted_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    inventory = query_inventory()
    inventory[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=field_name,
    ):
        validate(
            construction(),
            query_inventory_value=inventory,
        )


def test_strict_subset_requires_incomplete_evidence_code() -> None:
    value = construction()
    add_complete_reference_case(value)

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-complete-reference"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
    ]

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"strict_subset_of_complete_evidence.*"
            r"requires.*incomplete_evidence_set"
        ),
    ):
        validate(value)


def test_complete_reference_target_must_be_canonical() -> None:
    value = construction()
    accepted_passages = passages()
    accepted_passages.append(
        {
            "schema_version": "1.1",
            "passage_id": "passage-d",
            "document_id": "document-a",
            "label": "Section D",
            "citation_text": ("Accepted citation text for D."),
            "retrieval_text": ("Accepted retrieval text for D."),
            "logical_source_key": "source-d",
        }
    )

    value["cases"][0]["complete_reference_case_id"] = "pilot-001-derived-complete"
    value["cases"][0]["evidence_structure_codes"] = [
        "strict_subset_of_complete_evidence",
        "incomplete_evidence_set",
    ]

    value["cases"].extend(
        [
            {
                "case_id": "pilot-001-derived-complete",
                "query_id": "pilot-001",
                "question": ("What risk does the supplied source identify?"),
                "evidence_passage_ids": [
                    "passage-a",
                    "passage-c",
                ],
                "question_structure_codes": [
                    "direct_factual_lookup",
                ],
                "evidence_structure_codes": [
                    "multiple_complementary_passages",
                    "incomplete_evidence_set",
                ],
                "complete_reference_case_id": ("pilot-001-canonical-complete"),
            },
            {
                "case_id": "pilot-001-canonical-complete",
                "query_id": "pilot-001",
                "question": ("What risk does the supplied source identify?"),
                "evidence_passage_ids": [
                    "passage-a",
                    "passage-c",
                    "passage-d",
                ],
                "question_structure_codes": [
                    "direct_factual_lookup",
                ],
                "evidence_structure_codes": [
                    "multiple_complementary_passages",
                ],
            },
        ]
    )
    value["case_count"] = 3

    with pytest.raises(
        EvidenceSufficiencyCaseConstructionError,
        match=(
            r"complete reference case.*"
            r"must be canonical"
        ),
    ):
        validate(
            value,
            passages_value=accepted_passages,
        )
