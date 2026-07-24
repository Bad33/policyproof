from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    AnnotationComparison,
    EvidenceSufficiencyAnnotationError,
    compare_annotation_record_sets,
)

ANNOTATION_BATCH_SHA256 = "a" * 64
FIRST_RECORD_SET_SHA256 = "b" * 64
SECOND_RECORD_SET_SHA256 = "c" * 64


def annotation_batch() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "batch_version": "0.1.0",
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": "d" * 64,
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": "e" * 64,
        "case_count": 2,
        "cases": [
            {
                "case_id": "pilot-001-reference",
                "query_id": "pilot-001",
                "question": "What risk is identified?",
                "evidence": [
                    {
                        "passage_id": "passage-a",
                        "document_id": "document-a",
                        "label": "Section A",
                        "citation_text": "Risk A is identified.",
                    }
                ],
            },
            {
                "case_id": "pilot-002-incomplete",
                "query_id": "pilot-002",
                "question": "What risk and mitigation are identified?",
                "evidence": [
                    {
                        "passage_id": "passage-b",
                        "document_id": "document-b",
                        "label": "Section B",
                        "citation_text": "Risk B is identified.",
                    }
                ],
            },
        ],
    }


def record_set(
    annotator_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_set_id": (
            "policyproof-evidence-sufficiency-annotation-record-set"
        ),
        "record_set_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.1.0",
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": "d" * 64,
        "passage_artifact_sha256": "e" * 64,
        "annotator_id": annotator_id,
        "annotation_count": 2,
        "annotations": [
            {
                "annotation_id": (
                    f"{annotator_id}:pilot-001-reference"
                ),
                "annotator_id": annotator_id,
                "annotation_guide_version": "0.1.0",
                "case_id": "pilot-001-reference",
                "evidence_status": "sufficient",
                "response_action": "answer",
                "reason_codes": [],
                "missing_information": [],
                "rationale": "The evidence supports the question.",
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-07-22T19:00:00Z",
            },
            {
                "annotation_id": (
                    f"{annotator_id}:pilot-002-incomplete"
                ),
                "annotator_id": annotator_id,
                "annotation_guide_version": "0.1.0",
                "case_id": "pilot-002-incomplete",
                "evidence_status": "insufficient",
                "response_action": "abstain",
                "reason_codes": [
                    "incomplete_evidence_set",
                ],
                "missing_information": [
                    "The supplied evidence does not provide a mitigation."
                ],
                "rationale": (
                    "The risk is supported, but the mitigation is absent."
                ),
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-07-22T19:05:00Z",
            },
        ],
    }


def compare(
    first: dict[str, Any],
    second: dict[str, Any],
) -> AnnotationComparison:
    return compare_annotation_record_sets(
        first,
        second,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
        first_record_set_sha256=FIRST_RECORD_SET_SHA256,
        second_record_set_sha256=SECOND_RECORD_SET_SHA256,
    )


def test_identical_labels_compare_without_mutation() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    original_first = deepcopy(first)
    original_second = deepcopy(second)

    result = compare(first, second)

    assert first == original_first
    assert second == original_second
    assert result.annotator_ids == (
        "annotator-alpha",
        "annotator-beta",
    )
    assert result.record_set_sha256s == (
        FIRST_RECORD_SET_SHA256,
        SECOND_RECORD_SET_SHA256,
    )
    assert tuple(
        comparison.case_id
        for comparison in result.case_comparisons
    ) == (
        "pilot-001-reference",
        "pilot-002-incomplete",
    )
    assert all(
        comparison.disagreement_fields == ()
        for comparison in result.case_comparisons
    )
    assert all(
        not comparison.uncertainty_present
        for comparison in result.case_comparisons
    )
    assert all(
        not comparison.requires_adjudication
        for comparison in result.case_comparisons
    )


def test_comparison_uses_batch_case_order() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    first["annotations"].reverse()
    second["annotations"].reverse()

    result = compare(first, second)

    assert tuple(
        comparison.case_id
        for comparison in result.case_comparisons
    ) == (
        "pilot-001-reference",
        "pilot-002-incomplete",
    )


def test_detects_label_and_reason_code_disagreements() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    annotation = second["annotations"][1]
    annotation["evidence_status"] = "sufficient"
    annotation["response_action"] = "answer"
    annotation["reason_codes"] = []
    annotation["missing_information"] = []

    result = compare(first, second)
    comparison = result.case_comparisons[1]

    assert comparison.disagreement_fields == (
        "evidence_status",
        "response_action",
        "reason_codes",
        "missing_information",
    )
    assert comparison.requires_adjudication


def test_reason_code_comparison_is_order_insensitive() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    for value in (first, second):
        annotation = value["annotations"][1]
        annotation["reason_codes"] = [
            "incomplete_evidence_set",
            "outside_controlled_corpus",
        ]

    second["annotations"][1]["reason_codes"].reverse()

    result = compare(first, second)

    assert (
        result.case_comparisons[1].disagreement_fields
        == ()
    )


def test_missing_information_comparison_is_order_insensitive() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    statements = [
        "The mitigation is absent.",
        "The evidence does not establish implementation details.",
    ]

    for value in (first, second):
        value["annotations"][1]["missing_information"] = list(
            statements
        )

    second["annotations"][1]["missing_information"].reverse()

    result = compare(first, second)

    assert (
        result.case_comparisons[1].disagreement_fields
        == ()
    )


def test_uncertainty_alone_requires_adjudication() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"][0]["uncertainty"] = True
    second["annotations"][0]["adjudication_note"] = (
        "The question wording may be ambiguous."
    )

    result = compare(first, second)
    comparison = result.case_comparisons[0]

    assert comparison.disagreement_fields == ()
    assert comparison.uncertainty_present
    assert comparison.requires_adjudication


def test_rationale_wording_does_not_create_label_disagreement() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"][0]["rationale"] = (
        "Different wording with the same annotation decision."
    )

    result = compare(first, second)

    assert (
        result.case_comparisons[0].disagreement_fields
        == ()
    )
    assert not result.case_comparisons[0].requires_adjudication


def test_requires_distinct_annotators() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-alpha")

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="distinct annotators",
    ):
        compare(first, second)


@pytest.mark.parametrize(
    ("argument_name", "replacement"),
    [
        ("first_record_set_sha256", "not-a-sha"),
        ("second_record_set_sha256", "not-a-sha"),
    ],
)
def test_rejects_invalid_record_set_sha256(
    argument_name: str,
    replacement: str,
) -> None:
    arguments = {
        "annotation_batch": annotation_batch(),
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "first_record_set_sha256": FIRST_RECORD_SET_SHA256,
        "second_record_set_sha256": SECOND_RECORD_SET_SHA256,
    }
    arguments[argument_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=argument_name,
    ):
        compare_annotation_record_sets(
            record_set("annotator-alpha"),
            record_set("annotator-beta"),
            **arguments,
        )


def test_comparison_revalidates_each_record_set() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"].pop()
    second["annotation_count"] = 1

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="cover every batch case exactly once",
    ):
        compare(first, second)
