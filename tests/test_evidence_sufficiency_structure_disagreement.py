from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    ANNOTATION_STRUCTURE_DISAGREEMENT_REPORT_ID,
    AnnotationStructureDisagreementReport,
    EvidenceSufficiencyAnnotationError,
    build_structure_disagreement_artifact,
    calculate_structure_disagreement_counts,
)

ANNOTATION_BATCH_SHA256 = "a" * 64
FIRST_RECORD_SET_SHA256 = "b" * 64
SECOND_RECORD_SET_SHA256 = "c" * 64
ANALYSIS_METADATA_SHA256 = "d" * 64
GUIDE_SHA256 = "e" * 64
PASSAGE_SHA256 = "f" * 64


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
        "case_count": 3,
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
            {
                "case_id": "pilot-003-comparison",
                "query_id": "pilot-003",
                "question": "Which approach is safer?",
                "evidence": [
                    {
                        "passage_id": "passage-c",
                        "document_id": "document-c",
                        "label": "Section C",
                        "citation_text": (
                            "One approach is described without "
                            "a common comparison basis."
                        ),
                    }
                ],
            },
        ],
    }


def analysis_metadata() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "metadata_id": (
            "policyproof-evidence-sufficiency-"
            "annotation-analysis-metadata"
        ),
        "metadata_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.1.0",
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "case_count": 3,
        "cases": [
            {
                "case_id": "pilot-001-reference",
                "question_structure_codes": [
                    "direct_factual_lookup",
                ],
                "evidence_structure_codes": [
                    "one_complete_passage",
                ],
            },
            {
                "case_id": "pilot-002-incomplete",
                "question_structure_codes": [
                    "risk_and_mitigation",
                    "multi_part_question",
                ],
                "evidence_structure_codes": [
                    "incomplete_evidence_set",
                ],
            },
            {
                "case_id": "pilot-003-comparison",
                "question_structure_codes": [
                    "comparison",
                ],
                "evidence_structure_codes": [
                    "one_complete_passage",
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
        "annotation_count": 3,
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
                    "The requested mitigation is absent."
                ],
                "rationale": (
                    "The risk is supported but the mitigation is absent."
                ),
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-07-22T19:05:00Z",
            },
            {
                "annotation_id": (
                    f"{annotator_id}:pilot-003-comparison"
                ),
                "annotator_id": annotator_id,
                "annotation_guide_version": "0.1.0",
                "case_id": "pilot-003-comparison",
                "evidence_status": "insufficient",
                "response_action": "abstain",
                "reason_codes": [
                    "unsupported_comparison",
                ],
                "missing_information": [
                    "A common comparison basis is absent."
                ],
                "rationale": (
                    "The evidence does not support the requested ranking."
                ),
                "uncertainty": False,
                "adjudication_note": None,
                "annotation_timestamp": "2026-07-22T19:10:00Z",
            },
        ],
    }


def calculate(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> AnnotationStructureDisagreementReport:
    return calculate_structure_disagreement_counts(
        first,
        second,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
        first_record_set_sha256=FIRST_RECORD_SET_SHA256,
        second_record_set_sha256=SECOND_RECORD_SET_SHA256,
        analysis_metadata=metadata or analysis_metadata(),
        analysis_metadata_sha256=ANALYSIS_METADATA_SHA256,
    )


def test_structure_counts_are_deterministic_without_mutation() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    original_first = deepcopy(first)
    original_second = deepcopy(second)

    second_incomplete = second["annotations"][1]
    second_incomplete["evidence_status"] = "sufficient"
    second_incomplete["response_action"] = "answer"
    second_incomplete["reason_codes"] = []
    second_incomplete["missing_information"] = []

    second["annotations"][2]["uncertainty"] = True
    second["annotations"][2]["adjudication_note"] = (
        "The comparison wording may be ambiguous."
    )

    report = calculate(first, second)

    assert first == original_first
    assert second != original_second
    assert report.analysis_metadata_sha256 == (
        ANALYSIS_METADATA_SHA256
    )
    assert report.case_count == 3

    assert tuple(
        item.structure_code
        for item in report.question_structure_counts
    ) == (
        "comparison",
        "direct_factual_lookup",
        "multi_part_question",
        "risk_and_mitigation",
    )

    counts = {
        item.structure_code: item
        for item in report.question_structure_counts
    }

    comparison = counts["comparison"]
    assert comparison.case_count == 1
    assert comparison.disagreement_case_count == 0
    assert comparison.uncertainty_case_count == 1
    assert comparison.requires_adjudication_count == 1

    multi_part = counts["multi_part_question"]
    assert multi_part.case_count == 1
    assert multi_part.disagreement_case_count == 1
    assert multi_part.uncertainty_case_count == 0
    assert multi_part.requires_adjudication_count == 1
    assert multi_part.evidence_status_disagreement_count == 1
    assert multi_part.response_action_disagreement_count == 1
    assert multi_part.reason_codes_disagreement_count == 1
    assert multi_part.missing_information_disagreement_count == 1

    evidence_counts = {
        item.structure_code: item
        for item in report.evidence_structure_counts
    }

    assert evidence_counts["one_complete_passage"].case_count == 2
    assert (
        evidence_counts[
            "one_complete_passage"
        ].disagreement_case_count
        == 0
    )
    assert (
        evidence_counts[
            "one_complete_passage"
        ].uncertainty_case_count
        == 1
    )

    incomplete = evidence_counts["incomplete_evidence_set"]
    assert incomplete.case_count == 1
    assert incomplete.disagreement_case_count == 1


def test_case_with_multiple_codes_counts_once_per_code() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"][1]["reason_codes"] = [
        "incomplete_evidence_set",
        "outside_controlled_corpus",
    ]

    report = calculate(first, second)
    counts = {
        item.structure_code: item
        for item in report.question_structure_counts
    }

    assert counts["multi_part_question"].case_count == 1
    assert (
        counts[
            "multi_part_question"
        ].reason_codes_disagreement_count
        == 1
    )
    assert counts["risk_and_mitigation"].case_count == 1
    assert (
        counts[
            "risk_and_mitigation"
        ].reason_codes_disagreement_count
        == 1
    )


def test_report_revalidates_analysis_metadata() -> None:
    metadata = analysis_metadata()
    metadata["cases"].pop()
    metadata["case_count"] = 2

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="cover every batch case exactly once",
    ):
        calculate(
            record_set("annotator-alpha"),
            record_set("annotator-beta"),
            metadata=metadata,
        )


def test_report_rejects_invalid_metadata_sha256() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="analysis_metadata_sha256",
    ):
        calculate_structure_disagreement_counts(
            record_set("annotator-alpha"),
            record_set("annotator-beta"),
            annotation_batch=annotation_batch(),
            annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
            first_record_set_sha256=FIRST_RECORD_SET_SHA256,
            second_record_set_sha256=SECOND_RECORD_SET_SHA256,
            analysis_metadata=analysis_metadata(),
            analysis_metadata_sha256="not-a-sha",
        )



def build_artifact(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    report_version: str = "0.1.0",
) -> dict[str, Any]:
    return build_structure_disagreement_artifact(
        first,
        second,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
        first_record_set_sha256=FIRST_RECORD_SET_SHA256,
        second_record_set_sha256=SECOND_RECORD_SET_SHA256,
        analysis_metadata=metadata or analysis_metadata(),
        analysis_metadata_sha256=ANALYSIS_METADATA_SHA256,
        report_version=report_version,
    )


def test_build_structure_artifact_is_deterministic_and_bound() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")

    second_incomplete = second["annotations"][1]
    second_incomplete["evidence_status"] = "sufficient"
    second_incomplete["response_action"] = "answer"
    second_incomplete["reason_codes"] = []
    second_incomplete["missing_information"] = []

    second["annotations"][2]["uncertainty"] = True
    second["annotations"][2]["adjudication_note"] = (
        "The comparison wording may be ambiguous."
    )

    original_first = deepcopy(first)
    original_second = deepcopy(second)
    metadata = analysis_metadata()
    original_metadata = deepcopy(metadata)

    artifact = build_artifact(
        first,
        second,
        metadata=metadata,
    )

    assert first == original_first
    assert second == original_second
    assert metadata == original_metadata

    assert artifact["schema_version"] == "1.0"
    assert artifact["report_id"] == (
        ANNOTATION_STRUCTURE_DISAGREEMENT_REPORT_ID
    )
    assert artifact["report_version"] == "0.1.0"

    assert artifact["annotation_batch_id"] == (
        "policyproof-evidence-sufficiency-annotation-batch"
    )
    assert artifact["annotation_batch_version"] == "0.1.0"
    assert artifact["annotation_batch_sha256"] == (
        ANNOTATION_BATCH_SHA256
    )

    assert artifact["first_record_set_id"] == (
        "policyproof-evidence-sufficiency-annotation-record-set"
    )
    assert artifact["first_record_set_version"] == "0.1.0"
    assert artifact["first_record_set_sha256"] == (
        FIRST_RECORD_SET_SHA256
    )

    assert artifact["second_record_set_id"] == (
        "policyproof-evidence-sufficiency-annotation-record-set"
    )
    assert artifact["second_record_set_version"] == "0.1.0"
    assert artifact["second_record_set_sha256"] == (
        SECOND_RECORD_SET_SHA256
    )

    assert artifact["analysis_metadata_id"] == (
        "policyproof-evidence-sufficiency-"
        "annotation-analysis-metadata"
    )
    assert artifact["analysis_metadata_version"] == "0.1.0"
    assert artifact["analysis_metadata_sha256"] == (
        ANALYSIS_METADATA_SHA256
    )
    assert artifact["annotator_ids"] == [
        "annotator-alpha",
        "annotator-beta",
    ]
    assert artifact["case_count"] == 3

    question_counts = {
        item["structure_code"]: item
        for item in artifact["question_structure_counts"]
    }

    assert question_counts["comparison"] == {
        "structure_code": "comparison",
        "case_count": 1,
        "disagreement_case_count": 0,
        "uncertainty_case_count": 1,
        "requires_adjudication_count": 1,
        "evidence_status_disagreement_count": 0,
        "response_action_disagreement_count": 0,
        "reason_codes_disagreement_count": 0,
        "missing_information_disagreement_count": 0,
    }

    assert question_counts["multi_part_question"] == {
        "structure_code": "multi_part_question",
        "case_count": 1,
        "disagreement_case_count": 1,
        "uncertainty_case_count": 0,
        "requires_adjudication_count": 1,
        "evidence_status_disagreement_count": 1,
        "response_action_disagreement_count": 1,
        "reason_codes_disagreement_count": 1,
        "missing_information_disagreement_count": 1,
    }

    evidence_counts = {
        item["structure_code"]: item
        for item in artifact["evidence_structure_counts"]
    }

    assert evidence_counts["one_complete_passage"][
        "case_count"
    ] == 2
    assert evidence_counts["one_complete_passage"][
        "uncertainty_case_count"
    ] == 1
    assert evidence_counts["incomplete_evidence_set"][
        "disagreement_case_count"
    ] == 1


def test_structure_artifact_is_independent_of_input_order() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    metadata = analysis_metadata()

    expected = build_artifact(
        first,
        second,
        metadata=metadata,
    )

    first["annotations"].reverse()
    second["annotations"].reverse()
    metadata["cases"].reverse()

    assert build_artifact(
        first,
        second,
        metadata=metadata,
    ) == expected


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
def test_structure_artifact_requires_semantic_report_version(
    invalid_version: object,
) -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="report_version",
    ):
        build_structure_disagreement_artifact(
            record_set("annotator-alpha"),
            record_set("annotator-beta"),
            annotation_batch=annotation_batch(),
            annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
            first_record_set_sha256=FIRST_RECORD_SET_SHA256,
            second_record_set_sha256=SECOND_RECORD_SET_SHA256,
            analysis_metadata=analysis_metadata(),
            analysis_metadata_sha256=ANALYSIS_METADATA_SHA256,
            report_version=invalid_version,  # type: ignore[arg-type]
        )
