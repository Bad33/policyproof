from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    ANNOTATION_AGREEMENT_REPORT_ID,
    AnnotationAgreementReport,
    EvidenceSufficiencyAnnotationError,
    build_annotation_agreement_artifact,
    calculate_annotation_agreement,
)

ANNOTATION_BATCH_SHA256 = "a" * 64
FIRST_RECORD_SET_SHA256 = "b" * 64
SECOND_RECORD_SET_SHA256 = "c" * 64
GUIDE_SHA256 = "d" * 64
PASSAGE_SHA256 = "e" * 64


def annotation_batch() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "batch_version": "0.1.0",
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": GUIDE_SHA256,
        "corpus_id": "policyproof-initial-corpus",
        "corpus_version": "0.1.0",
        "passage_schema_version": "1.1",
        "passage_artifact_sha256": PASSAGE_SHA256,
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
                "question": (
                    "What risk and mitigation are identified?"
                ),
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


def record_set(annotator_id: str) -> dict[str, Any]:
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
        "annotation_guide_sha256": GUIDE_SHA256,
        "passage_artifact_sha256": PASSAGE_SHA256,
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
                    (
                        "The supplied evidence does not provide "
                        "the requested mitigation."
                    )
                ],
                "rationale": (
                    "The risk is supported, but the mitigation "
                    "is absent."
                ),
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-07-22T19:05:00Z",
            },
        ],
    }


def calculate(
    first: dict[str, Any],
    second: dict[str, Any],
) -> AnnotationAgreementReport:
    return calculate_annotation_agreement(
        first,
        second,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
        first_record_set_sha256=FIRST_RECORD_SET_SHA256,
        second_record_set_sha256=SECOND_RECORD_SET_SHA256,
    )


def test_identical_annotations_have_perfect_agreement_without_mutation() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    original_first = deepcopy(first)
    original_second = deepcopy(second)

    report = calculate(first, second)

    assert first == original_first
    assert second == original_second
    assert report.annotator_ids == (
        "annotator-alpha",
        "annotator-beta",
    )
    assert report.record_set_sha256s == (
        FIRST_RECORD_SET_SHA256,
        SECOND_RECORD_SET_SHA256,
    )
    assert report.case_count == 2
    assert report.status_agreement_count == 2
    assert report.status_raw_agreement == 1.0
    assert report.cohen_kappa == 1.0
    assert report.reason_code_exact_match_count == 2
    assert report.reason_code_exact_match_agreement == 1.0
    assert report.mean_reason_code_jaccard == 1.0
    assert tuple(
        item.reason_code
        for item in report.per_code_agreement
    ) == ("incomplete_evidence_set",)
    assert report.per_code_agreement[0].true_positive == 1
    assert report.per_code_agreement[0].false_positive == 0
    assert report.per_code_agreement[0].false_negative == 0
    assert report.per_code_agreement[0].precision == 1.0
    assert report.per_code_agreement[0].recall == 1.0
    assert report.per_code_agreement[0].f1 == 1.0
    assert report.macro_reason_code_precision == 1.0
    assert report.macro_reason_code_recall == 1.0
    assert report.macro_reason_code_f1 == 1.0


def test_status_disagreement_calculates_raw_agreement_and_kappa() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    annotation = second["annotations"][1]
    annotation["evidence_status"] = "sufficient"
    annotation["response_action"] = "answer"
    annotation["reason_codes"] = []
    annotation["missing_information"] = []

    report = calculate(first, second)

    assert report.status_agreement_count == 1
    assert report.status_raw_agreement == 0.5
    assert report.cohen_kappa == 0.0


def test_degenerate_status_distribution_reports_undefined_kappa() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    for value in (first, second):
        annotation = value["annotations"][1]
        annotation["evidence_status"] = "sufficient"
        annotation["response_action"] = "answer"
        annotation["reason_codes"] = []
        annotation["missing_information"] = []

    report = calculate(first, second)

    assert report.status_raw_agreement == 1.0
    assert report.cohen_kappa is None


def test_reason_code_metrics_use_first_annotator_as_reference() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"][1]["reason_codes"] = [
        "incomplete_evidence_set",
        "outside_controlled_corpus",
    ]

    report = calculate(first, second)

    assert report.reason_code_exact_match_count == 1
    assert report.reason_code_exact_match_agreement == 0.5
    assert report.mean_reason_code_jaccard == 0.75
    assert tuple(
        item.reason_code
        for item in report.per_code_agreement
    ) == (
        "incomplete_evidence_set",
        "outside_controlled_corpus",
    )

    incomplete, outside = report.per_code_agreement

    assert (
        incomplete.true_positive,
        incomplete.false_positive,
        incomplete.false_negative,
    ) == (1, 0, 0)
    assert (
        incomplete.precision,
        incomplete.recall,
        incomplete.f1,
    ) == (1.0, 1.0, 1.0)

    assert (
        outside.true_positive,
        outside.false_positive,
        outside.false_negative,
    ) == (0, 1, 0)
    assert (
        outside.precision,
        outside.recall,
        outside.f1,
    ) == (0.0, 0.0, 0.0)

    assert report.macro_reason_code_precision == 0.5
    assert report.macro_reason_code_recall == 0.5
    assert report.macro_reason_code_f1 == 0.5


def test_reason_code_sets_are_order_insensitive() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    for value in (first, second):
        value["annotations"][1]["reason_codes"] = [
            "incomplete_evidence_set",
            "outside_controlled_corpus",
        ]

    second["annotations"][1]["reason_codes"].reverse()

    report = calculate(first, second)

    assert report.reason_code_exact_match_agreement == 1.0
    assert report.mean_reason_code_jaccard == 1.0


def test_empty_reason_code_union_has_no_per_code_macro_metrics() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    for value in (first, second):
        annotation = value["annotations"][1]
        annotation["evidence_status"] = "sufficient"
        annotation["response_action"] = "answer"
        annotation["reason_codes"] = []
        annotation["missing_information"] = []

    report = calculate(first, second)

    assert report.per_code_agreement == ()
    assert report.macro_reason_code_precision is None
    assert report.macro_reason_code_recall is None
    assert report.macro_reason_code_f1 is None


def test_missing_information_wording_is_not_scored_as_exact_agreement() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"][1]["missing_information"] = [
        "A mitigation is not supplied."
    ]

    report = calculate(first, second)

    assert report.reason_code_exact_match_agreement == 1.0
    assert report.mean_reason_code_jaccard == 1.0


def test_agreement_revalidates_record_sets() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"].pop()
    second["annotation_count"] = 1

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="cover every batch case exactly once",
    ):
        calculate(first, second)


def test_agreement_requires_distinct_annotators() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-alpha")

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="distinct annotators",
    ):
        calculate(first, second)



def build_artifact(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    report_version: str = "0.1.0",
) -> dict[str, Any]:
    return build_annotation_agreement_artifact(
        first,
        second,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
        first_record_set_sha256=FIRST_RECORD_SET_SHA256,
        second_record_set_sha256=SECOND_RECORD_SET_SHA256,
        report_version=report_version,
    )


def test_build_agreement_artifact_is_deterministic_and_bound() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    original_first = deepcopy(first)
    original_second = deepcopy(second)

    artifact = build_artifact(first, second)

    assert first == original_first
    assert second == original_second
    assert artifact == {
        "schema_version": "1.0",
        "report_id": ANNOTATION_AGREEMENT_REPORT_ID,
        "report_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.1.0",
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "first_record_set_id": (
            "policyproof-evidence-sufficiency-annotation-record-set"
        ),
        "first_record_set_version": "0.1.0",
        "first_record_set_sha256": FIRST_RECORD_SET_SHA256,
        "second_record_set_id": (
            "policyproof-evidence-sufficiency-annotation-record-set"
        ),
        "second_record_set_version": "0.1.0",
        "second_record_set_sha256": SECOND_RECORD_SET_SHA256,
        "annotator_ids": [
            "annotator-alpha",
            "annotator-beta",
        ],
        "case_count": 2,
        "status_agreement": {
            "agreement_count": 2,
            "raw_agreement": 1.0,
            "cohen_kappa": 1.0,
        },
        "reason_code_agreement": {
            "exact_match_count": 2,
            "exact_match_agreement": 1.0,
            "mean_jaccard": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "per_code": [
                {
                    "reason_code": "incomplete_evidence_set",
                    "true_positive": 1,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }
            ],
        },
    }


def test_agreement_artifact_is_independent_of_record_order() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    expected = build_artifact(first, second)

    first["annotations"].reverse()
    second["annotations"].reverse()

    assert build_artifact(first, second) == expected


def test_agreement_artifact_preserves_undefined_metrics() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    for value in (first, second):
        annotation = value["annotations"][1]
        annotation["evidence_status"] = "sufficient"
        annotation["response_action"] = "answer"
        annotation["reason_codes"] = []
        annotation["missing_information"] = []

    artifact = build_artifact(first, second)

    assert artifact["status_agreement"]["cohen_kappa"] is None
    assert artifact["reason_code_agreement"]["per_code"] == []
    assert (
        artifact["reason_code_agreement"]["macro_precision"]
        is None
    )
    assert (
        artifact["reason_code_agreement"]["macro_recall"]
        is None
    )
    assert artifact["reason_code_agreement"]["macro_f1"] is None


@pytest.mark.parametrize(
    "invalid_version",
    [
        "",
        "0.1",
        "v0.1.0",
        "0.1.0-dev",
        1,
    ],
)
def test_agreement_artifact_requires_semantic_report_version(
    invalid_version: object,
) -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="report_version",
    ):
        build_annotation_agreement_artifact(
            record_set("annotator-alpha"),
            record_set("annotator-beta"),
            annotation_batch=annotation_batch(),
            annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
            first_record_set_sha256=FIRST_RECORD_SET_SHA256,
            second_record_set_sha256=SECOND_RECORD_SET_SHA256,
            report_version=invalid_version,  # type: ignore[arg-type]
        )
