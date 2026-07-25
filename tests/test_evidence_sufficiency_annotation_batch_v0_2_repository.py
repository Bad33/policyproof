from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_annotations import (
    validate_annotation_batch,
)

ROOT = Path(__file__).resolve().parents[1]

BATCH_FILE = ROOT / "data/evaluation" / "evidence-sufficiency-annotation-batch-v0.2.0.json"
CONSTRUCTION_FILE = ROOT / "data/evaluation" / "evidence-sufficiency-case-construction-v0.2.0.json"
MANIFEST_FILE = ROOT / "data/source_manifest.json"
PASSAGES_FILE = ROOT / "data/processed" / "retrieval-passages.jsonl"

ANNOTATION_GUIDE_VERSION = "0.1.0"
ANNOTATION_GUIDE_SHA256 = "f188099ffa51b1005a5c607281426a13b5cd087dd51407bbe05808cd4dce893d"
PASSAGE_ARTIFACT_SHA256 = "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"


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


def test_repository_annotation_batch_validates_and_is_blinded() -> None:
    assert BATCH_FILE.is_file(), "The complete blinded annotation batch has not been published."

    batch = load_json(BATCH_FILE)
    construction = load_json(CONSTRUCTION_FILE)
    manifest = load_json(MANIFEST_FILE)
    passages = load_jsonl(PASSAGES_FILE)

    validate_annotation_batch(
        batch,
        manifest=manifest,
        passages=passages,
        passage_artifact_sha256=(PASSAGE_ARTIFACT_SHA256),
        annotation_guide_version=(ANNOTATION_GUIDE_VERSION),
        annotation_guide_sha256=(ANNOTATION_GUIDE_SHA256),
    )

    assert batch["batch_version"] == "0.2.0"
    assert batch["case_count"] == 160
    assert len(batch["cases"]) == 160

    expected_case_ids = [case["case_id"] for case in construction["cases"]]
    actual_case_ids = [case["case_id"] for case in batch["cases"]]

    assert actual_case_ids == expected_case_ids

    forbidden_fields = {
        "question_structure_codes",
        "evidence_structure_codes",
        "complete_reference_case_id",
        "expected_evidence_status",
        "expected_response_action",
        "reason_codes",
        "missing_information",
        "rationale",
        "uncertainty",
        "model_score",
        "retrieval_score",
        "split",
        "split_assignment",
    }

    for batch_case, construction_case in zip(
        batch["cases"],
        construction["cases"],
        strict=True,
    ):
        assert set(batch_case) == {
            "case_id",
            "query_id",
            "question",
            "evidence",
        }
        assert not (set(batch_case) & forbidden_fields)
        assert batch_case["query_id"] == construction_case["query_id"]
        assert batch_case["question"] == construction_case["question"]
        assert [evidence["passage_id"] for evidence in batch_case["evidence"]] == construction_case[
            "evidence_passage_ids"
        ]


BATCH_SHA256 = "1bb6a7bed55a43f59a79ff4861c81c3d36ffa5ed78af1bf12292bceb927bf93c"


def test_published_annotation_batch_is_byte_stable() -> None:
    assert sha256_file(BATCH_FILE) == BATCH_SHA256
