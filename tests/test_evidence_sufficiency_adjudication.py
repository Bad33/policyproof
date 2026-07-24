from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.evidence_sufficiency_annotations import (
    ADJUDICATION_RECORD_SET_ID,
    EvidenceSufficiencyAnnotationError,
    validate_adjudication_record_set,
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


def disagreeing_record_sets() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    annotation = second["annotations"][1]
    annotation["evidence_status"] = "sufficient"
    annotation["response_action"] = "answer"
    annotation["reason_codes"] = []
    annotation["missing_information"] = []
    annotation["rationale"] = (
        "The evidence was interpreted as supporting the request."
    )
    return first, second


def adjudication_record_set() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "adjudication_set_id": ADJUDICATION_RECORD_SET_ID,
        "adjudication_set_version": "0.1.0",
        "annotation_batch_id": (
            "policyproof-evidence-sufficiency-annotation-batch"
        ),
        "annotation_batch_version": "0.1.0",
        "annotation_batch_sha256": ANNOTATION_BATCH_SHA256,
        "annotation_guide_version": "0.1.0",
        "annotation_guide_sha256": GUIDE_SHA256,
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
        "adjudicator_id": "adjudicator-one",
        "record_count": 1,
        "records": [
            {
                "adjudication_id": (
                    "adjudicator-one:pilot-002-incomplete"
                ),
                "case_id": "pilot-002-incomplete",
                "first_annotation_id": (
                    "annotator-alpha:pilot-002-incomplete"
                ),
                "second_annotation_id": (
                    "annotator-beta:pilot-002-incomplete"
                ),
                "disagreement_categories": [
                    "completeness_disagreement",
                    "reason_code_disagreement",
                    "missing_information_disagreement",
                ],
                "final_evidence_status": "insufficient",
                "final_response_action": "abstain",
                "final_reason_codes": [
                    "incomplete_evidence_set",
                ],
                "final_missing_information": [
                    (
                        "The supplied evidence does not provide "
                        "the requested mitigation."
                    )
                ],
                "final_rationale": (
                    "The question requests both a risk and a "
                    "mitigation, but only the risk is supported."
                ),
                "adjudication_rationale": (
                    "The complete-question rule requires support "
                    "for the missing mitigation."
                ),
                "guide_change_required": False,
                "guide_change_summary": None,
                "adjudication_timestamp": "2026-07-22T20:00:00Z",
            }
        ],
    }


def validate(
    value: dict[str, Any],
    *,
    first: dict[str, Any] | None = None,
    second: dict[str, Any] | None = None,
) -> None:
    default_first, default_second = disagreeing_record_sets()

    validate_adjudication_record_set(
        value,
        annotation_batch=annotation_batch(),
        annotation_batch_sha256=ANNOTATION_BATCH_SHA256,
        first_record_set=first or default_first,
        first_record_set_sha256=FIRST_RECORD_SET_SHA256,
        second_record_set=second or default_second,
        second_record_set_sha256=SECOND_RECORD_SET_SHA256,
    )


def test_valid_adjudication_record_set_passes_without_mutation() -> None:
    value = adjudication_record_set()
    original = deepcopy(value)

    validate(value)

    assert value == original


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("annotation_batch_version", "9.9.9"),
        ("annotation_batch_sha256", "f" * 64),
        ("annotation_guide_version", "9.9.9"),
        ("annotation_guide_sha256", "1" * 64),
        ("first_record_set_version", "9.9.9"),
        ("first_record_set_sha256", "2" * 64),
        ("second_record_set_version", "9.9.9"),
        ("second_record_set_sha256", "3" * 64),
    ],
)
def test_adjudication_rejects_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = adjudication_record_set()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match=field_name,
    ):
        validate(value)


def test_adjudication_covers_required_cases_exactly_once() -> None:
    value = adjudication_record_set()
    value["records"] = []
    value["record_count"] = 0

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="records must cover exactly the cases requiring adjudication",
    ):
        validate(value)


def test_adjudication_rejects_record_for_agreed_case() -> None:
    value = adjudication_record_set()
    extra = deepcopy(value["records"][0])
    extra["adjudication_id"] = (
        "adjudicator-one:pilot-001-reference"
    )
    extra["case_id"] = "pilot-001-reference"
    extra["first_annotation_id"] = (
        "annotator-alpha:pilot-001-reference"
    )
    extra["second_annotation_id"] = (
        "annotator-beta:pilot-001-reference"
    )
    value["records"].append(extra)
    value["record_count"] = 2

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="records must cover exactly the cases requiring adjudication",
    ):
        validate(value)


def test_adjudication_binds_original_annotation_ids() -> None:
    value = adjudication_record_set()
    value["records"][0]["first_annotation_id"] = (
        "annotator-alpha:other-case"
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="first_annotation_id",
    ):
        validate(value)


def test_rejects_unknown_disagreement_category() -> None:
    value = adjudication_record_set()
    value["records"][0]["disagreement_categories"] = [
        "unknown_disagreement",
    ]

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="disagreement_categories",
    ):
        validate(value)


def test_label_disagreement_requires_category() -> None:
    value = adjudication_record_set()
    value["records"][0]["disagreement_categories"] = []

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="disagreement_categories.*nonempty",
    ):
        validate(value)


def test_uncertainty_only_adjudication_allows_empty_categories() -> None:
    first = record_set("annotator-alpha")
    second = record_set("annotator-beta")
    second["annotations"][0]["uncertainty"] = True
    second["annotations"][0]["adjudication_note"] = (
        "The question wording may be ambiguous."
    )

    value = adjudication_record_set()
    record = value["records"][0]
    record["adjudication_id"] = (
        "adjudicator-one:pilot-001-reference"
    )
    record["case_id"] = "pilot-001-reference"
    record["first_annotation_id"] = (
        "annotator-alpha:pilot-001-reference"
    )
    record["second_annotation_id"] = (
        "annotator-beta:pilot-001-reference"
    )
    record["disagreement_categories"] = []
    record["final_evidence_status"] = "sufficient"
    record["final_response_action"] = "answer"
    record["final_reason_codes"] = []
    record["final_missing_information"] = []
    record["final_rationale"] = (
        "The evidence directly supports the complete question."
    )
    record["adjudication_rationale"] = (
        "The ambiguity concern does not change the evidence decision."
    )

    validate(value, first=first, second=second)


@pytest.mark.parametrize(
    ("required", "summary"),
    [
        (False, "A guide change was described."),
        (True, None),
    ],
)
def test_guide_change_flag_and_summary_must_agree(
    required: bool,
    summary: str | None,
) -> None:
    value = adjudication_record_set()
    value["records"][0]["guide_change_required"] = required
    value["records"][0]["guide_change_summary"] = summary

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="guide_change",
    ):
        validate(value)


def test_final_sufficient_label_rejects_reason_codes() -> None:
    value = adjudication_record_set()
    record = value["records"][0]
    record["final_evidence_status"] = "sufficient"
    record["final_response_action"] = "answer"
    record["final_reason_codes"] = [
        "incomplete_evidence_set",
    ]
    record["final_missing_information"] = []

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="sufficient.*final_reason_codes",
    ):
        validate(value)


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        "2026-07-22 20:00:00Z",
        "2026-07-22T20:00:00",
        "2026-07-22T20:00:00+00:00",
        "2026-02-30T20:00:00Z",
        "not-a-timestamp",
    ],
)
def test_adjudication_timestamp_requires_canonical_utc_format(
    invalid_timestamp: str,
) -> None:
    value = adjudication_record_set()
    value["records"][0]["adjudication_timestamp"] = (
        invalid_timestamp
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="adjudication_timestamp.*RFC 3339 UTC",
    ):
        validate(value)
