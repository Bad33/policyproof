from __future__ import annotations

from pathlib import Path

import pytest

from policyproof.retrieval_builder import (
    RetrievalBuildResult,
    classify_exclusion_reason,
    direct_coordinates,
    heading_coordinates,
    is_reference_start,
    source_range,
    structural_context_ids,
    write_retrieval_build,
)
from policyproof.retrieval_units import (
    RetrievalUnitError,
    build_document_indexes,
)


def make_page(
    document_id: str,
    page_number: int,
    text: str,
    *,
    page_count: int,
) -> dict:
    return {
        "schema_version": "1.0",
        "page_id": (
            f"{document_id}:page-{page_number:04d}"
        ),
        "document_id": document_id,
        "page_number": page_number,
        "page_count": page_count,
        "text": text,
    }


def coordinate(
    document_id: str,
    page_number: int,
    line_number: int,
) -> dict:
    return {
        "page_id": (
            f"{document_id}:page-{page_number:04d}"
        ),
        "page_number": page_number,
        "line_number": line_number,
    }


def test_source_and_heading_ranges_are_inclusive() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "one\ntwo",
                page_count=2,
            ),
            make_page(
                "document-a",
                2,
                "three\nfour",
                page_count=2,
            ),
        ]
    )
    record = {
        "document_id": "document-a",
        "heading_source": {
            "start": coordinate(
                "document-a",
                1,
                2,
            ),
            "end": coordinate(
                "document-a",
                2,
                1,
            ),
        },
    }

    assert source_range(
        "document-a",
        (1, 2),
        (2, 1),
        indexes,
    ) == (
        (1, 2),
        (2, 1),
    )
    assert heading_coordinates(
        record,
        indexes,
    ) == (
        (1, 2),
        (2, 1),
    )


def test_direct_coordinates_handles_exact_empty_span() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Heading\nBody",
                page_count=1,
            )
        ]
    )
    record = {
        "node_id": "source:a",
        "document_id": "document-a",
        "direct_body": {
            "is_empty": True,
            "included_start": None,
            "included_end": None,
        },
    }

    assert direct_coordinates(
        record,
        indexes,
    ) == ()


def test_direct_coordinates_rejects_inconsistent_empty_span() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Heading",
                page_count=1,
            )
        ]
    )
    record = {
        "node_id": "source:a",
        "document_id": "document-a",
        "direct_body": {
            "is_empty": True,
            "included_start": coordinate(
                "document-a",
                1,
                1,
            ),
            "included_end": None,
        },
    }

    with pytest.raises(
        RetrievalUnitError,
        match="empty direct body",
    ):
        direct_coordinates(
            record,
            indexes,
        )


def test_direct_coordinates_expands_nonempty_span() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Heading\nOne\nTwo",
                page_count=1,
            )
        ]
    )
    record = {
        "node_id": "source:a",
        "document_id": "document-a",
        "direct_body": {
            "is_empty": False,
            "included_start": coordinate(
                "document-a",
                1,
                2,
            ),
            "included_end": coordinate(
                "document-a",
                1,
                3,
            ),
        },
    }

    assert direct_coordinates(
        record,
        indexes,
    ) == (
        (1, 2),
        (1, 3),
    )


def test_eu_reference_disambiguation() -> None:
    assert is_reference_start(
        "(1) OJ L 123, 1.1.2024"
    )
    assert is_reference_start(
        "(15) Regulation (EU) 2024/1689"
    )
    assert not is_reference_start(
        "(1) Artificial intelligence can "
        "provide important benefits."
    )


def test_exclusion_reason_classification() -> None:
    assert classify_exclusion_reason(
        "eu_eli_footer"
    ) == "excluded_page_furniture"
    assert classify_exclusion_reason(
        "eu_numbered_reference_region"
    ) == "excluded_reference_region"
    assert classify_exclusion_reason(
        "gpt_authorship_and_credits"
    ) == "excluded_structural_metadata"


def test_structural_context_is_rmf_specific() -> None:
    pages = [
        make_page(
            "nist-ai-rmf-1.0",
            page_number,
            (
                "one\ntwo\nthree"
                if page_number in {9, 25}
                else "one"
            ),
            page_count=25,
        )
        for page_number in range(1, 26)
    ]
    pages.append(
        make_page(
            "document-a",
            1,
            "one",
            page_count=1,
        )
    )
    indexes = build_document_indexes(
        pages
    )

    assert structural_context_ids(
        "nist-ai-rmf-1.0",
        (9, 3),
        indexes,
    ) == (
        "rmf-part-1-foundational-information",
    )
    assert structural_context_ids(
        "nist-ai-rmf-1.0",
        (25, 3),
        indexes,
    ) == (
        "rmf-part-2-core-and-profiles",
    )
    assert structural_context_ids(
        "document-a",
        (1, 1),
        indexes,
    ) == ()


def test_write_retrieval_build_refuses_preexisting_output(
    tmp_path: Path,
) -> None:
    units_path = tmp_path / "units.jsonl"
    units_path.write_text(
        "existing\n",
        encoding="utf-8",
    )
    result = RetrievalBuildResult(
        units=(),
        ledger=(),
        summary={},
        report="report\n",
    )

    with pytest.raises(
        RetrievalUnitError,
        match="already exists",
    ):
        write_retrieval_build(
            result,
            units_path=units_path,
            ledger_path=(
                tmp_path / "ledger.jsonl"
            ),
            summary_path=(
                tmp_path / "summary.json"
            ),
            report_path=(
                tmp_path / "report.txt"
            ),
        )

    assert units_path.read_text(
        encoding="utf-8"
    ) == "existing\n"


def test_write_retrieval_build_rolls_back_partial_outputs(
    tmp_path: Path,
) -> None:
    units_path = tmp_path / "units.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.txt"
    result = RetrievalBuildResult(
        units=({"unit": 1},),
        ledger=({"ledger": 1},),
        summary={
            "invalid": {1, 2},
        },
        report="report\n",
    )

    with pytest.raises(TypeError):
        write_retrieval_build(
            result,
            units_path=units_path,
            ledger_path=ledger_path,
            summary_path=summary_path,
            report_path=report_path,
        )

    assert not units_path.exists()
    assert not ledger_path.exists()
    assert not summary_path.exists()
    assert not report_path.exists()
