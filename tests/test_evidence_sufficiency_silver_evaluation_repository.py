from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from policyproof.evidence_sufficiency_silver_evaluation import (
    validate_baseline,
    validate_grouped_splits,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data/evaluation/evidence-sufficiency-silver-splits-v0.1.0.json"
RESULT_FILE = ROOT / "data/evaluation/evidence-sufficiency-silver-baseline-v0.1.0.json"
BATCH_FILE = ROOT / "data/evaluation/evidence-sufficiency-annotation-batch-v0.2.0.json"
LABEL_FILE = ROOT / "data/evaluation/evidence-sufficiency-silver-labels-v0.1.0.json"

CONSTRUCTION_SHA = "c78e947a231492ccfece538a234b40bd9a94ca07aacb1acf51d970cecffdf21f"
BATCH_SHA = "1bb6a7bed55a43f59a79ff4861c81c3d36ffa5ed78af1bf12292bceb927bf93c"
LABEL_SHA = "aa7e12b43d2a9f1fbd93b266c7614cb13e3b486d2456fa450d57cf85d3599531"


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_repository_silver_evaluation_artifacts_validate() -> None:
    assert SPLIT_FILE.is_file(), "The grouped silver split has not been published."
    assert RESULT_FILE.is_file(), "The silver baseline has not been published."

    split = load_json(SPLIT_FILE)
    result = load_json(RESULT_FILE)
    batch = load_json(BATCH_FILE)
    labels = load_json(LABEL_FILE)

    validate_grouped_splits(
        split,
        batch,
        labels,
        construction_sha256=CONSTRUCTION_SHA,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
    )
    validate_baseline(
        result,
        batch,
        labels,
        split,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
        split_artifact_sha256=sha256_file(SPLIT_FILE),
    )

    assert {item["split_name"]: item["query_count"] for item in split["splits"]} == {
        "train": 48,
        "validation": 16,
        "test": 16,
    }
    assert split["label_provenance"] == "construction_derived"
    assert result["label_provenance"] == "construction_derived"
    assert result["model_type"] == "standardized_logistic_regression"
    assert len(result["test_predictions"]) == next(
        item["case_count"] for item in split["splits"] if item["split_name"] == "test"
    )


SPLIT_SHA256 = "6a0200c31b465bf2784d86e5383c83750ba358c6a83f9cb6b58313df22fa9482"
RESULT_SHA256 = "847c4dedb37ea5c77c94bdf964d408c8685387f43cc57c3acc90a5170146be80"


def test_repository_silver_evaluation_artifacts_are_byte_stable() -> None:
    assert sha256_file(SPLIT_FILE) == SPLIT_SHA256
    assert sha256_file(RESULT_FILE) == RESULT_SHA256
