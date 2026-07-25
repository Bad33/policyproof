from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_case_construction import (
    CASE_CONSTRUCTION_ID,
    validate_evidence_sufficiency_case_construction,
)

ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_CONSTRUCTION_PATH = (
    ROOT / "data/evaluation" / ("evidence-sufficiency-case-construction-pilot-v0.1.0.json")
)
REPOSITORY_QUERY_INVENTORY_PATH = (
    ROOT / "data/evaluation" / "evidence-sufficiency-query-inventory-v0.2.0.json"
)
REPOSITORY_MANIFEST_PATH = ROOT / "data" / "source_manifest.json"
REPOSITORY_PASSAGES_PATH = ROOT / "data/processed" / "retrieval-passages.jsonl"

REPOSITORY_CONSTRUCTION_SHA256 = "a42e680e4b82ef8fd41b8cf3e7c6df4e405018c1bbf851515a9d1550422d0c57"


EXPECTED_QUERY_IDS = {
    "eu-005",
    "eu-006",
    "genai-005",
    "genai-006",
    "rmf-005",
    "rmf-006",
    "gpt4o-005",
    "gpt4o-006",
}

EXPECTED_CASE_IDS = {
    "eu-005-complete-reference",
    "eu-005-system-without-testing",
    "eu-005-testing-without-system",
    "eu-006-complete-reference",
    "eu-006-complete-with-value-chain-distractor",
    "genai-005-complete-reference",
    "genai-005-incident-planning-only",
    "genai-005-monitoring-fallback-contracts-only",
    "genai-006-complete-reference",
    "rmf-005-complete-reference",
    "rmf-005-ai-risk-differences-only",
    ("rmf-005-framework-limitations-without-named-frameworks"),
    "rmf-006-complete-reference",
    "rmf-006-organizational-role-only",
    "rmf-006-continuity-without-full-role",
    "gpt4o-005-complete-reference",
    "gpt4o-005-method-without-limitations",
    "gpt4o-005-limitations-without-method",
    "gpt4o-006-complete-reference",
    "gpt4o-006-definition-without-memory-reliance",
    "gpt4o-006-reliance-without-definition",
}


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))

    assert isinstance(value, dict)
    return value


def load_jsonl(
    file_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        assert isinstance(value, dict), f"{file_path}:{line_number} must contain a JSON object."
        records.append(value)

    return records


def test_repository_pilot_case_construction_validates() -> None:
    assert REPOSITORY_CONSTRUCTION_PATH.is_file(), (
        "The reviewed pilot case-construction artifact has not been published."
    )

    construction = load_json(REPOSITORY_CONSTRUCTION_PATH)
    inventory = load_json(REPOSITORY_QUERY_INVENTORY_PATH)
    manifest = load_json(REPOSITORY_MANIFEST_PATH)
    passages = load_jsonl(REPOSITORY_PASSAGES_PATH)

    inventory_sha256 = sha256_file(REPOSITORY_QUERY_INVENTORY_PATH)
    passage_sha256 = sha256_file(REPOSITORY_PASSAGES_PATH)

    validate_evidence_sufficiency_case_construction(
        construction,
        query_inventory=inventory,
        manifest=manifest,
        passages=passages,
        query_inventory_sha256=inventory_sha256,
        passage_artifact_sha256=passage_sha256,
    )

    assert construction["schema_version"] == "1.0"
    assert construction["construction_id"] == CASE_CONSTRUCTION_ID
    assert construction["construction_version"] == "0.1.0"
    assert construction["case_count"] == 21
    assert len(construction["cases"]) == 21

    query_ids = {case["query_id"] for case in construction["cases"]}
    case_ids = {case["case_id"] for case in construction["cases"]}

    assert query_ids == EXPECTED_QUERY_IDS
    assert case_ids == EXPECTED_CASE_IDS

    assert len(case_ids) == len(construction["cases"])

    complete_reference_count = sum(
        case["case_id"].endswith("-complete-reference") for case in construction["cases"]
    )
    incomplete_subset_count = sum(
        "incomplete_evidence_set" in case["evidence_structure_codes"]
        for case in construction["cases"]
    )
    distractor_case_count = sum(
        "topically_related_distractors" in case["evidence_structure_codes"]
        for case in construction["cases"]
    )

    assert complete_reference_count == 8
    assert incomplete_subset_count == 12
    assert distractor_case_count == 1

    structure_counts = Counter(
        structure_code
        for case in construction["cases"]
        for structure_code in case["evidence_structure_codes"]
    )

    assert structure_counts == {
        "strict_subset_of_complete_evidence": 12,
        "incomplete_evidence_set": 12,
        "multiple_complementary_passages": 6,
        "one_complete_passage": 3,
        "topically_related_distractors": 1,
    }

    query_evidence_keys = [
        (
            case["query_id"],
            tuple(sorted(case["evidence_passage_ids"])),
        )
        for case in construction["cases"]
    ]

    assert len(query_evidence_keys) == len(set(query_evidence_keys))


def test_published_pilot_case_construction_is_byte_stable() -> None:
    assert sha256_file(REPOSITORY_CONSTRUCTION_PATH) == REPOSITORY_CONSTRUCTION_SHA256
