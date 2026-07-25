from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from policyproof.evidence_sufficiency_silver_evaluation import (
    SilverEvaluationError,
    build_baseline,
    build_grouped_splits,
    extract_features,
    validate_baseline,
    validate_grouped_splits,
    write_json_artifact,
)

CONSTRUCTION_SHA = "a" * 64
BATCH_SHA = "b" * 64
LABEL_SHA = "c" * 64


def batch() -> dict[str, Any]:
    documents = {
        "eu-ai-act-2024-1689": 22,
        "nist-ai-600-1-genai-profile": 20,
        "nist-ai-rmf-1.0": 20,
        "openai-gpt-4o-system-card-2024-08-08": 18,
    }
    cases = []
    ordinal = 0

    for document_id, query_count in documents.items():
        for _ in range(query_count):
            query_id = f"query-{ordinal:03d}"
            ordinal += 1
            cases.append(
                {
                    "case_id": f"{query_id}-complete",
                    "query_id": query_id,
                    "question": "What controls and monitoring procedures apply?",
                    "evidence": [
                        {
                            "passage_id": f"{query_id}-complete-passage",
                            "document_id": document_id,
                            "label": "Controls and monitoring",
                            "citation_text": (
                                "The controls and monitoring procedures apply "
                                "throughout the system lifecycle."
                            ),
                        }
                    ],
                }
            )
            if ordinal % 2 == 0:
                cases.append(
                    {
                        "case_id": f"{query_id}-incomplete",
                        "query_id": query_id,
                        "question": "What controls and monitoring procedures apply?",
                        "evidence": [
                            {
                                "passage_id": f"{query_id}-partial-passage",
                                "document_id": document_id,
                                "label": "Controls",
                                "citation_text": "The controls apply.",
                            }
                        ],
                    }
                )

    return {
        "schema_version": "1.0",
        "batch_id": "policyproof-evidence-sufficiency-annotation-batch",
        "batch_version": "0.2.0",
        "case_count": len(cases),
        "cases": cases,
    }


def labels() -> dict[str, Any]:
    records = [
        {
            "case_id": case["case_id"],
            "evidence_status": (
                "insufficient" if case["case_id"].endswith("-incomplete") else "sufficient"
            ),
        }
        for case in batch()["cases"]
    ]
    return {
        "schema_version": "1.0",
        "label_set_id": "policyproof-evidence-sufficiency-silver-label-set",
        "label_set_version": "0.1.0",
        "case_count": len(records),
        "labels": records,
    }


def test_features_are_deterministic() -> None:
    case = batch()["cases"][0]
    assert extract_features(case) == extract_features(deepcopy(case))
    assert len(extract_features(case)) == 5


def test_grouped_splits_are_deterministic_and_leak_free() -> None:
    first = build_grouped_splits(
        batch(),
        labels(),
        construction_sha256=CONSTRUCTION_SHA,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
    )
    second = build_grouped_splits(
        batch(),
        labels(),
        construction_sha256=CONSTRUCTION_SHA,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
    )
    assert first == second

    validate_grouped_splits(
        first,
        batch(),
        labels(),
        construction_sha256=CONSTRUCTION_SHA,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
    )

    query_sets = [set(item["query_ids"]) for item in first["splits"]]
    assert not (query_sets[0] & query_sets[1])
    assert not (query_sets[0] & query_sets[2])
    assert not (query_sets[1] & query_sets[2])


def test_split_validator_rejects_query_leakage() -> None:
    artifact = build_grouped_splits(
        batch(),
        labels(),
        construction_sha256=CONSTRUCTION_SHA,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
    )
    artifact["splits"][1]["query_ids"][0] = artifact["splits"][0]["query_ids"][0]

    with pytest.raises(SilverEvaluationError, match="leakage"):
        validate_grouped_splits(
            artifact,
            batch(),
            labels(),
            construction_sha256=CONSTRUCTION_SHA,
            annotation_batch_sha256=BATCH_SHA,
            silver_label_set_sha256=LABEL_SHA,
        )


def test_baseline_builds_and_validates() -> None:
    split = build_grouped_splits(
        batch(),
        labels(),
        construction_sha256=CONSTRUCTION_SHA,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
    )
    result = build_baseline(
        batch(),
        labels(),
        split,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
        split_artifact_sha256="d" * 64,
    )

    validate_baseline(
        result,
        batch(),
        labels(),
        split,
        annotation_batch_sha256=BATCH_SHA,
        silver_label_set_sha256=LABEL_SHA,
        split_artifact_sha256="d" * 64,
    )
    assert result["test_predictions"]


def test_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    write_json_artifact({"value": 1}, output)

    with pytest.raises(SilverEvaluationError, match="overwrite"):
        write_json_artifact({"value": 1}, output)
