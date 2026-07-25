from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from policyproof.evidence_sufficiency_query_inventory import (
    ALLOWED_QUESTION_STRUCTURE_CODES,
    EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_ID,
    EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_SCHEMA_VERSION,
    EvidenceSufficiencyQueryInventoryError,
    validate_evidence_sufficiency_query_inventory,
)

PASSAGE_ARTIFACT_SHA256 = "a" * 64
DEVELOPMENT_DATASET_SHA256 = "b" * 64


def manifest() -> dict[str, Any]:
    return {
        "corpus_id": "policyproof-test-corpus",
        "corpus_version": "0.1.0",
        "documents": [
            {"document_id": "doc-a"},
            {"document_id": "doc-b"},
        ],
    }


def passages() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "1.1",
            "passage_id": "dev-passage",
            "document_id": "doc-a",
            "logical_source_key": "source-dev",
            "label": "Development source",
            "citation_text": "Existing development evidence.",
        },
        {
            "schema_version": "1.1",
            "passage_id": "new-passage-a",
            "document_id": "doc-a",
            "logical_source_key": "source-a",
            "label": "Untouched source A",
            "citation_text": "New source A evidence.",
        },
        {
            "schema_version": "1.1",
            "passage_id": "new-passage-b",
            "document_id": "doc-b",
            "logical_source_key": "source-b",
            "label": "Untouched source B",
            "citation_text": "New source B evidence.",
        },
    ]


def development_dataset() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": "policyproof-evidence-sufficiency-evaluation",
        "dataset_version": "0.1.0",
        "corpus_id": "policyproof-test-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "case_count": 1,
        "cases": [
            {
                "case_id": "existing-001-case",
                "query_id": "existing-001",
                "question": "What does the existing source say?",
                "evidence_passage_ids": ["dev-passage"],
            }
        ],
    }


def inventory() -> dict[str, Any]:
    return {
        "schema_version": (
            EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_SCHEMA_VERSION
        ),
        "inventory_id": EVIDENCE_SUFFICIENCY_QUERY_INVENTORY_ID,
        "inventory_version": "0.1.0",
        "corpus_id": "policyproof-test-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_ARTIFACT_SHA256,
        "development_evidence_dataset_id": (
            "policyproof-evidence-sufficiency-evaluation"
        ),
        "development_evidence_dataset_version": "0.1.0",
        "development_evidence_dataset_sha256": (
            DEVELOPMENT_DATASET_SHA256
        ),
        "query_count": 2,
        "queries": [
            {
                "query_id": "new-001",
                "question": (
                    "How does the first untouched source "
                    "describe its process?"
                ),
                "document_scope": ["doc-a"],
                "question_structure_codes": [
                    "process_or_evaluation_method"
                ],
                "source_logical_source_keys": ["source-a"],
            },
            {
                "query_id": "new-002",
                "question": (
                    "Which policy applies to a future "
                    "organization-specific situation?"
                ),
                "document_scope": ["doc-a", "doc-b"],
                "question_structure_codes": [
                    "policy_interpretation"
                ],
                "source_logical_source_keys": [],
            },
        ],
    }


def validate(value: dict[str, Any]) -> None:
    validate_evidence_sufficiency_query_inventory(
        value,
        development_dataset(),
        manifest(),
        passages(),
        PASSAGE_ARTIFACT_SHA256,
        DEVELOPMENT_DATASET_SHA256,
    )


def test_valid_inventory_passes_without_mutation() -> None:
    value = inventory()
    before = deepcopy(value)

    validate(value)

    assert value == before



def test_validates_without_mutating_any_supplied_input() -> None:
    value = inventory()
    development = development_dataset()
    corpus_manifest = manifest()
    passage_records = passages()

    value_before = deepcopy(value)
    development_before = deepcopy(development)
    manifest_before = deepcopy(corpus_manifest)
    passages_before = deepcopy(passage_records)

    validate_evidence_sufficiency_query_inventory(
        value,
        development,
        corpus_manifest,
        passage_records,
        PASSAGE_ARTIFACT_SHA256,
        DEVELOPMENT_DATASET_SHA256,
    )

    assert value == value_before
    assert development == development_before
    assert corpus_manifest == manifest_before
    assert passage_records == passages_before


def test_rejects_empty_query_collection() -> None:
    value = inventory()
    value["query_count"] = 0
    value["queries"] = []

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="at least one query",
    ):
        validate(value)


def test_rejects_query_id_used_by_development_data() -> None:
    value = inventory()
    value["queries"][0]["query_id"] = "existing-001"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="query_id.*development",
    ):
        validate(value)


def test_rejects_duplicate_normalized_question() -> None:
    value = inventory()
    value["queries"][1]["question"] = (
        "  HOW   DOES THE FIRST UNTOUCHED SOURCE "
        "DESCRIBE ITS PROCESS?  "
    )

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="question.*duplicate",
    ):
        validate(value)


def test_rejects_development_touched_logical_source() -> None:
    value = inventory()
    value["queries"][0]["source_logical_source_keys"] = [
        "source-dev"
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="development.*logical source",
    ):
        validate(value)


def test_rejects_unknown_logical_source() -> None:
    value = inventory()
    value["queries"][0]["source_logical_source_keys"] = [
        "source-missing"
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="unknown logical source",
    ):
        validate(value)


def test_rejects_source_outside_document_scope() -> None:
    value = inventory()
    value["queries"][0]["source_logical_source_keys"] = [
        "source-b"
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="document_scope",
    ):
        validate(value)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "evidence_passage_ids",
        "expected_evidence_status",
        "expected_response_action",
        "reason_codes",
        "relevance_judgments",
        "policy_prediction",
        "model_score",
    ],
)
def test_rejects_label_or_evidence_fields(
    forbidden_field: str,
) -> None:
    value = inventory()
    value["queries"][0][forbidden_field] = []

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="forbidden",
    ):
        validate(value)


def test_rejects_unknown_top_level_field() -> None:
    value = inventory()
    value["unexpected"] = True

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="unexpected field",
    ):
        validate(value)



def test_rejects_development_dataset_passage_binding_mismatch() -> None:
    value = inventory()
    development = development_dataset()
    development["passage_artifact_sha256"] = "c" * 64

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match=(
            "development dataset "
            "passage_artifact_sha256"
        ),
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development,
            manifest(),
            passages(),
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )



def test_rejects_passage_schema_version_mismatch() -> None:
    value = inventory()
    passage_records = passages()
    passage_records[1]["schema_version"] = "1.0"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="passage schema_version",
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development_dataset(),
            manifest(),
            passage_records,
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )


def test_rejects_passage_document_outside_manifest() -> None:
    value = inventory()
    passage_records = passages()
    passage_records[1]["document_id"] = "doc-missing"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="passage document_id",
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development_dataset(),
            manifest(),
            passage_records,
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )


def test_rejects_logical_source_spanning_documents() -> None:
    value = inventory()
    value["queries"][0]["document_scope"] = [
        "doc-a",
        "doc-b",
    ]
    passage_records = passages()
    passage_records.append(
        {
            "schema_version": "1.1",
            "passage_id": "new-passage-a-continuation",
            "document_id": "doc-b",
            "logical_source_key": "source-a",
            "label": "Invalid cross-document continuation",
            "citation_text": "Invalid continuation evidence.",
        }
    )

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="logical source.*multiple documents",
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development_dataset(),
            manifest(),
            passage_records,
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )



ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_QUERY_INVENTORY_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "evidence-sufficiency-query-inventory-v0.1.0.json"
)
REPOSITORY_MANIFEST_PATH = ROOT / "data" / "source_manifest.json"
REPOSITORY_PASSAGES_PATH = (
    ROOT / "data" / "processed" / "retrieval-passages.jsonl"
)
REPOSITORY_DEVELOPMENT_DATASET_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "evidence-sufficiency-evaluation-v0.1.0.json"
)
REPOSITORY_QUERY_INVENTORY_SHA256 = (
    "168a23fdc3c6e6e9664bd112e37efe340"
    "f5e8e1099c30e1b94b0f3de95a937a3"
)

REPOSITORY_QUERY_INVENTORY_V0_2_PATH = (
    REPOSITORY_QUERY_INVENTORY_PATH.with_name(
        "evidence-sufficiency-query-inventory-v0.2.0.json"
    )
)

REPOSITORY_QUERY_INVENTORY_V0_2_SHA256 = (
    "a6e03c5b28f8a99124b3141c7d2fb6e"
    "f7e85f11dfab2477c151f05f4514970f0"
)


def repository_sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def repository_load_jsonl(
    file_path: Path,
) -> list[dict[str, Any]]:
    records = []

    for line_number, line in enumerate(
        file_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise AssertionError(
                f"{file_path}:{line_number} "
                "must contain a JSON object."
            )

        records.append(value)

    return records


def test_repository_query_inventory_validates() -> None:
    repository_inventory = json.loads(
        REPOSITORY_QUERY_INVENTORY_PATH.read_text(
            encoding="utf-8"
        )
    )
    repository_manifest = json.loads(
        REPOSITORY_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    repository_passages = repository_load_jsonl(
        REPOSITORY_PASSAGES_PATH
    )
    repository_development_dataset = json.loads(
        REPOSITORY_DEVELOPMENT_DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )

    validate_evidence_sufficiency_query_inventory(
        repository_inventory,
        repository_development_dataset,
        repository_manifest,
        repository_passages,
        repository_sha256_file(
            REPOSITORY_PASSAGES_PATH
        ),
        repository_sha256_file(
            REPOSITORY_DEVELOPMENT_DATASET_PATH
        ),
    )



def test_published_query_inventory_is_byte_stable() -> None:
    assert (
        repository_sha256_file(
            REPOSITORY_QUERY_INVENTORY_PATH
        )
        == REPOSITORY_QUERY_INVENTORY_SHA256
    )


def test_published_query_inventory_v0_2_is_byte_stable(
) -> None:
    assert (
        repository_sha256_file(
            REPOSITORY_QUERY_INVENTORY_V0_2_PATH
        )
        == REPOSITORY_QUERY_INVENTORY_V0_2_SHA256
    )


def test_repository_query_inventory_v0_2_expands_immutable_v0_1_0(
) -> None:
    assert REPOSITORY_QUERY_INVENTORY_V0_2_PATH.is_file(), (
        "The reviewed v0.2.0 query inventory has not been published."
    )

    original_inventory = json.loads(
        REPOSITORY_QUERY_INVENTORY_PATH.read_text(
            encoding="utf-8"
        )
    )
    expanded_inventory = json.loads(
        REPOSITORY_QUERY_INVENTORY_V0_2_PATH.read_text(
            encoding="utf-8"
        )
    )
    repository_manifest = json.loads(
        REPOSITORY_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    repository_passages = repository_load_jsonl(
        REPOSITORY_PASSAGES_PATH
    )
    repository_development_dataset = json.loads(
        REPOSITORY_DEVELOPMENT_DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )

    validate_evidence_sufficiency_query_inventory(
        expanded_inventory,
        repository_development_dataset,
        repository_manifest,
        repository_passages,
        repository_sha256_file(
            REPOSITORY_PASSAGES_PATH
        ),
        repository_sha256_file(
            REPOSITORY_DEVELOPMENT_DATASET_PATH
        ),
    )

    assert original_inventory["inventory_version"] == "0.1.0"
    assert expanded_inventory["inventory_version"] == "0.2.0"

    binding_fields = (
        "schema_version",
        "inventory_id",
        "corpus_id",
        "corpus_version",
        "passage_schema_version",
        "passage_artifact_sha256",
        "development_evidence_dataset_id",
        "development_evidence_dataset_version",
        "development_evidence_dataset_sha256",
    )

    for field_name in binding_fields:
        assert (
            expanded_inventory[field_name]
            == original_inventory[field_name]
        )

    assert original_inventory["query_count"] == 14
    assert (
        expanded_inventory["query_count"]
        == len(expanded_inventory["queries"])
        == 80
    )
    assert (
        expanded_inventory["queries"][
            : original_inventory["query_count"]
        ]
        == original_inventory["queries"]
    )

    query_ids = {
        query_record["query_id"]
        for query_record in expanded_inventory["queries"]
    }
    expected_query_ids = (
        {
            f"eu-{number:03d}"
            for number in range(5, 27)
        }
        | {
            f"genai-{number:03d}"
            for number in range(5, 25)
        }
        | {
            f"rmf-{number:03d}"
            for number in range(5, 25)
        }
        | {
            f"gpt4o-{number:03d}"
            for number in range(5, 23)
        }
    )

    assert query_ids == expected_query_ids

    logical_source_keys = [
        source_key
        for query_record in expanded_inventory["queries"]
        for source_key in query_record[
            "source_logical_source_keys"
        ]
    ]

    assert len(logical_source_keys) == 80
    assert len(set(logical_source_keys)) == 80

    document_counts = {
        "eu-ai-act-2024-1689": 0,
        "nist-ai-600-1-genai-profile": 0,
        "nist-ai-rmf-1.0": 0,
        "openai-gpt-4o-system-card-2024-08-08": 0,
    }

    for query_record in expanded_inventory["queries"]:
        document_scope = query_record["document_scope"]

        assert len(document_scope) == 1
        document_counts[document_scope[0]] += 1

    assert document_counts == {
        "eu-ai-act-2024-1689": 22,
        "nist-ai-600-1-genai-profile": 20,
        "nist-ai-rmf-1.0": 20,
        "openai-gpt-4o-system-card-2024-08-08": 18,
    }


def test_repository_query_inventory_covers_all_question_structures() -> None:
    repository_inventory = json.loads(
        REPOSITORY_QUERY_INVENTORY_PATH.read_text(
            encoding="utf-8"
        )
    )

    covered_codes = {
        structure_code
        for query_record in repository_inventory["queries"]
        for structure_code in query_record[
            "question_structure_codes"
        ]
    }

    assert covered_codes == ALLOWED_QUESTION_STRUCTURE_CODES


def test_repository_query_inventory_uses_distinct_logical_sources() -> None:
    repository_inventory = json.loads(
        REPOSITORY_QUERY_INVENTORY_PATH.read_text(
            encoding="utf-8"
        )
    )

    logical_source_keys = [
        source_key
        for query_record in repository_inventory["queries"]
        for source_key in query_record[
            "source_logical_source_keys"
        ]
    ]

    assert len(logical_source_keys) == 14
    assert len(set(logical_source_keys)) == 14



def test_rejects_unsupported_inventory_schema() -> None:
    value = inventory()
    value["schema_version"] = "2.0"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="Unsupported query inventory schema_version",
    ):
        validate(value)


def test_rejects_unexpected_inventory_id() -> None:
    value = inventory()
    value["inventory_id"] = "other-inventory"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="Unexpected query inventory inventory_id",
    ):
        validate(value)


def test_rejects_invalid_inventory_version() -> None:
    value = inventory()
    value["inventory_version"] = "version-one"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="semantic version format",
    ):
        validate(value)


def test_rejects_inventory_passage_sha256_binding_mismatch() -> None:
    value = inventory()
    value["passage_artifact_sha256"] = "c" * 64

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="inventory passage_artifact_sha256",
    ):
        validate(value)


def test_rejects_inventory_development_sha256_binding_mismatch() -> None:
    value = inventory()
    value["development_evidence_dataset_sha256"] = "c" * 64

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="development dataset SHA-256",
    ):
        validate(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("corpus_id", "other-corpus"),
        ("corpus_version", "9.9.9"),
        ("passage_schema_version", "9.9"),
    ],
)
def test_rejects_inventory_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = inventory()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match=rf"inventory {field_name} binding mismatch",
    ):
        validate(value)


def test_rejects_development_dataset_id_binding_mismatch() -> None:
    value = inventory()
    value["development_evidence_dataset_id"] = "other-dataset"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="development dataset ID binding mismatch",
    ):
        validate(value)


def test_rejects_development_dataset_version_binding_mismatch() -> None:
    value = inventory()
    value["development_evidence_dataset_version"] = "9.9.9"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="development dataset version binding mismatch",
    ):
        validate(value)


def test_rejects_query_count_mismatch() -> None:
    value = inventory()
    value["query_count"] = 1

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="query_count does not match",
    ):
        validate(value)


def test_rejects_duplicate_query_id() -> None:
    value = inventory()
    value["queries"][1]["query_id"] = (
        value["queries"][0]["query_id"]
    )

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="Duplicate query_id",
    ):
        validate(value)


def test_rejects_question_used_by_development_data() -> None:
    value = inventory()
    value["queries"][0]["question"] = (
        "  WHAT   DOES THE EXISTING SOURCE SAY?  "
    )

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="question duplicates development data",
    ):
        validate(value)


def test_rejects_unknown_question_structure_code() -> None:
    value = inventory()
    value["queries"][0]["question_structure_codes"] = [
        "unsupported_structure"
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="Unknown question structure code",
    ):
        validate(value)


def test_rejects_duplicate_question_structure_code() -> None:
    value = inventory()
    value["queries"][0]["question_structure_codes"] = [
        "process_or_evaluation_method",
        "process_or_evaluation_method",
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="question_structure_codes.*duplicate",
    ):
        validate(value)


def test_rejects_unknown_document_scope() -> None:
    value = inventory()
    value["queries"][0]["document_scope"] = [
        "doc-missing"
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="document_scope contains an unknown document_id",
    ):
        validate(value)


def test_rejects_duplicate_document_scope() -> None:
    value = inventory()
    value["queries"][0]["document_scope"] = [
        "doc-a",
        "doc-a",
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="document_scope.*duplicate",
    ):
        validate(value)


def test_rejects_duplicate_logical_source_key() -> None:
    value = inventory()
    value["queries"][0]["source_logical_source_keys"] = [
        "source-a",
        "source-a",
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="source_logical_source_keys.*duplicate",
    ):
        validate(value)


def test_rejects_duplicate_passage_id() -> None:
    value = inventory()
    passage_records = passages()
    passage_records[2]["passage_id"] = (
        passage_records[1]["passage_id"]
    )

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="Duplicate passage_id",
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development_dataset(),
            manifest(),
            passage_records,
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )


def test_rejects_unknown_development_evidence_passage() -> None:
    value = inventory()
    development = development_dataset()
    development["cases"][0]["evidence_passage_ids"] = [
        "missing-passage"
    ]

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="development dataset references an unknown passage_id",
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development,
            manifest(),
            passages(),
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )


def test_rejects_duplicate_manifest_document_id() -> None:
    value = inventory()
    corpus_manifest = manifest()
    corpus_manifest["documents"][1]["document_id"] = "doc-a"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="manifest contains duplicate document IDs",
    ):
        validate_evidence_sufficiency_query_inventory(
            value,
            development_dataset(),
            corpus_manifest,
            passages(),
            PASSAGE_ARTIFACT_SHA256,
            DEVELOPMENT_DATASET_SHA256,
        )


def test_rejects_unknown_query_field() -> None:
    value = inventory()
    value["queries"][0]["review_note"] = "not allowed"

    with pytest.raises(
        EvidenceSufficiencyQueryInventoryError,
        match="unexpected field",
    ):
        validate(value)
