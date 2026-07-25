from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_silver_labels import (
    validate_evidence_sufficiency_silver_labels,
)

ROOT = Path(__file__).resolve().parents[1]

LABEL_SET_FILE = ROOT / "data/evaluation" / "evidence-sufficiency-silver-labels-v0.1.0.json"
CONSTRUCTION_FILE = ROOT / "data/evaluation" / "evidence-sufficiency-case-construction-v0.2.0.json"
BATCH_FILE = ROOT / "data/evaluation" / "evidence-sufficiency-annotation-batch-v0.2.0.json"

CONSTRUCTION_SHA256 = "c78e947a231492ccfece538a234b40bd9a94ca07aacb1acf51d970cecffdf21f"
BATCH_SHA256 = "1bb6a7bed55a43f59a79ff4861c81c3d36ffa5ed78af1bf12292bceb927bf93c"


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


def test_repository_silver_labels_validate_and_are_disclosed() -> None:
    assert LABEL_SET_FILE.is_file(), (
        "The construction-derived silver-label artifact has not been published."
    )

    label_set = load_json(LABEL_SET_FILE)
    construction = load_json(CONSTRUCTION_FILE)
    batch = load_json(BATCH_FILE)

    validate_evidence_sufficiency_silver_labels(
        label_set,
        construction=construction,
        construction_sha256=CONSTRUCTION_SHA256,
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
    )

    assert label_set["label_set_version"] == "0.1.0"
    assert label_set["label_provenance"] == "construction_derived"
    assert label_set["case_count"] == 160
    assert len(label_set["labels"]) == 160

    counts = Counter(label["evidence_status"] for label in label_set["labels"])

    assert counts == {
        "sufficient": 81,
        "insufficient": 79,
    }

    assert sum(label["response_action"] == "answer" for label in label_set["labels"]) == 81

    assert sum(label["response_action"] == "abstain" for label in label_set["labels"]) == 79

    assert (
        sum(label["reason_codes"] == ["incomplete_evidence_set"] for label in label_set["labels"])
        == 79
    )

    assert [label["case_id"] for label in label_set["labels"]] == [
        case["case_id"] for case in construction["cases"]
    ]

    for label in label_set["labels"]:
        assert set(label) == {
            "case_id",
            "evidence_status",
            "response_action",
            "reason_codes",
            "missing_information",
            "derivation_rule",
        }


LABEL_SET_SHA256 = "aa7e12b43d2a9f1fbd93b266c7614cb13e3b486d2456fa450d57cf85d3599531"


def test_published_silver_labels_are_byte_stable() -> None:
    assert sha256_file(LABEL_SET_FILE) == LABEL_SET_SHA256
