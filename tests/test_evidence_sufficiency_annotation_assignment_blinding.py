from __future__ import annotations

from copy import deepcopy

import pytest

from policyproof.evidence_sufficiency_annotation_round import (
    build_annotation_assignment_package,
    build_annotation_round_manifest,
)
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
)

BATCH_SHA256 = "1" * 64
ROUND_SHA256 = "2" * 64
GUIDE_SHA256 = "3" * 64
PASSAGE_SHA256 = "4" * 64


def _annotation_batch() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "batch_version": "0.2.0",
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": GUIDE_SHA256,
        "corpus_id": "policyproof-controlled-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_SHA256,
        "case_count": 1,
        "cases": [
            {
                "case_id": "case-001",
                "query_id": "query-001",
                "question": "What does the evidence support?",
                "evidence": [
                    {
                        "passage_id": "passage-001",
                        "document_id": "document-001",
                        "label": "Example section",
                        "citation_text": "Example evidence.",
                    }
                ],
            }
        ],
    }


def _build_assignment(
    batch: dict[str, object],
) -> dict[str, object]:
    manifest = build_annotation_round_manifest(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_version="0.1.0",
        primary_annotator_ids=(
            "annotator-001",
            "annotator-002",
        ),
        adjudicator_id="adjudicator-001",
    )

    return build_annotation_assignment_package(
        annotation_batch=batch,
        annotation_batch_sha256=BATCH_SHA256,
        round_manifest=manifest,
        round_manifest_sha256=ROUND_SHA256,
        annotator_id="annotator-001",
        assignment_version="0.1.0",
    )


def test_assignment_rejects_unknown_batch_field() -> None:
    batch = _annotation_batch()
    batch["construction_labels"] = {
        "case-001": "sufficient",
    }

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            "annotation_batch contains unsupported fields: "
            r"\['construction_labels'\]"
        ),
    ):
        _build_assignment(batch)


def test_assignment_rejects_hidden_case_field() -> None:
    batch = _annotation_batch()
    cases = batch["cases"]
    assert isinstance(cases, list)

    case = deepcopy(cases[0])
    assert isinstance(case, dict)
    case["silver_label"] = "sufficient"
    cases[0] = case

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            r"annotation_batch\.cases\[0\] contains unsupported "
            r"fields: \['silver_label'\]"
        ),
    ):
        _build_assignment(batch)


def test_assignment_rejects_hidden_evidence_field() -> None:
    batch = _annotation_batch()
    cases = batch["cases"]
    assert isinstance(cases, list)

    case = cases[0]
    assert isinstance(case, dict)
    evidence = case["evidence"]
    assert isinstance(evidence, list)

    passage = deepcopy(evidence[0])
    assert isinstance(passage, dict)
    passage["retrieval_score"] = 0.99
    evidence[0] = passage

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=(
            r"annotation_batch\.cases\[0\]\.evidence\[0\] "
            r"contains unsupported fields: "
            r"\['retrieval_score'\]"
        ),
    ):
        _build_assignment(batch)
