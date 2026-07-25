from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_case_construction import (
    validate_evidence_sufficiency_case_construction,
)

ROOT = Path(__file__).resolve().parents[1]

FULL_CONSTRUCTION_PATH = (
    ROOT / "data/evaluation" / "evidence-sufficiency-case-construction-v0.2.0.json"
)
PILOT_CONSTRUCTION_PATH = (
    ROOT / "data/evaluation" / ("evidence-sufficiency-case-construction-pilot-v0.1.0.json")
)
QUERY_INVENTORY_PATH = ROOT / "data/evaluation" / "evidence-sufficiency-query-inventory-v0.2.0.json"
MANIFEST_PATH = ROOT / "data/source_manifest.json"
PASSAGES_PATH = ROOT / "data/processed" / "retrieval-passages.jsonl"


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
    return [
        json.loads(line)
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_repository_full_case_construction_validates_and_expands_pilot() -> None:
    assert FULL_CONSTRUCTION_PATH.is_file(), (
        "The reviewed full case-construction artifact has not been published."
    )

    full = load_json(FULL_CONSTRUCTION_PATH)
    pilot = load_json(PILOT_CONSTRUCTION_PATH)
    inventory = load_json(QUERY_INVENTORY_PATH)
    manifest = load_json(MANIFEST_PATH)
    passages = load_jsonl(PASSAGES_PATH)

    validate_evidence_sufficiency_case_construction(
        full,
        query_inventory=inventory,
        manifest=manifest,
        passages=passages,
        query_inventory_sha256=sha256_file(QUERY_INVENTORY_PATH),
        passage_artifact_sha256=sha256_file(PASSAGES_PATH),
    )

    assert full["construction_version"] == "0.2.0"
    assert full["case_count"] == len(full["cases"]) == 160
    assert full["cases"][: pilot["case_count"]] == pilot["cases"]

    query_ids = {case["query_id"] for case in full["cases"]}
    assert len(query_ids) == 80

    complete_reference_count = sum(
        case["case_id"].endswith("-complete-reference") for case in full["cases"]
    )
    incomplete_count = sum(
        "incomplete_evidence_set" in case["evidence_structure_codes"] for case in full["cases"]
    )
    distractor_count = sum(
        "topically_related_distractors" in case["evidence_structure_codes"]
        for case in full["cases"]
    )

    assert complete_reference_count == 80
    assert incomplete_count == 79
    assert distractor_count == 1

    structure_counts = Counter(
        structure_code
        for case in full["cases"]
        for structure_code in case["evidence_structure_codes"]
    )

    assert structure_counts == {
        "one_complete_passage": 43,
        "multiple_complementary_passages": 38,
        "strict_subset_of_complete_evidence": 79,
        "incomplete_evidence_set": 79,
        "topically_related_distractors": 1,
    }

    query_evidence_keys = [
        (
            case["query_id"],
            tuple(sorted(case["evidence_passage_ids"])),
        )
        for case in full["cases"]
    ]

    assert len(query_evidence_keys) == len(set(query_evidence_keys))


FULL_CONSTRUCTION_SHA256 = "c78e947a231492ccfece538a234b40bd9a94ca07aacb1acf51d970cecffdf21f"


def test_published_full_case_construction_is_byte_stable() -> None:
    assert sha256_file(FULL_CONSTRUCTION_PATH) == FULL_CONSTRUCTION_SHA256
