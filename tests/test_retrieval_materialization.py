from __future__ import annotations

import pytest

from policyproof.retrieval_materialization import (
    materialize_retrieval_text,
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
        "page_id": f"{document_id}:page-{page_number:04d}",
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
        "page_id": f"{document_id}:page-{page_number:04d}",
        "page_number": page_number,
        "line_number": line_number,
    }


def heading_body_unit() -> dict:
    document_id = "document-a"

    return {
        "unit_id": (
            "candidate-v2:document-a:"
            "source:document-a:page-0001:line-0001:"
            "part-001"
        ),
        "unit_kind": "heading_body",
        "document_id": document_id,
        "part_number": 1,
        "part_count": 1,
        "source_node_id": (
            "source:document-a:page-0001:line-0001"
        ),
        "content_coordinate_count": 4,
        "content_segments": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    2,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    3,
                ),
            },
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    5,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    5,
                ),
            },
            {
                "included_start": coordinate(
                    document_id,
                    2,
                    2,
                ),
                "included_end": coordinate(
                    document_id,
                    2,
                    2,
                ),
            },
        ],
    }


def test_materialization_preserves_source_and_uses_reviewed_separators() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Heading\n first-\ncontinuation\n \n next body ",
                page_count=2,
            ),
            make_page(
                "document-a",
                2,
                "Page 2\n cross page ",
                page_count=2,
            ),
        ]
    )
    hierarchy = {
        "source:document-a:page-0001:line-0001": {
            "node_id": (
                "source:document-a:page-0001:line-0001"
            ),
            "full_heading": "Reviewed Heading",
        }
    }

    result = materialize_retrieval_text(
        heading_body_unit(),
        hierarchy,
        indexes,
    )

    assert result.raw_source_segments == (
        " first-\ncontinuation",
        " next body ",
        " cross page ",
    )
    assert result.retrieval_body == (
        "first-\ncontinuation\n\nnext body\ncross page"
    )
    assert result.model_text == (
        "Reviewed Heading\n\n"
        "first-\ncontinuation\n\nnext body\ncross page"
    )
    assert result.citation_text == result.retrieval_body
    assert result.label == "Reviewed Heading"
    assert result.logical_source_key == (
        "candidate-v2:document-a:"
        "source:document-a:page-0001:line-0001"
    )


def test_heading_only_uses_reviewed_label_without_duplication() -> None:
    document_id = "document-a"
    node_id = "source:document-a:page-0001:line-0001"
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                "BROKEN\nHEADING",
                page_count=1,
            )
        ]
    )
    unit = {
        "unit_id": (
            "candidate-v2:document-a:"
            f"{node_id}:part-001"
        ),
        "unit_kind": "heading_only",
        "document_id": document_id,
        "part_number": 1,
        "part_count": 1,
        "source_node_id": node_id,
        "content_coordinate_count": 2,
        "content_segments": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    2,
                ),
            }
        ],
    }
    hierarchy = {
        node_id: {
            "node_id": node_id,
            "full_heading": "Corrected Heading",
        }
    }

    result = materialize_retrieval_text(
        unit,
        hierarchy,
        indexes,
    )

    assert result.raw_source_segments == (
        "BROKEN\nHEADING",
    )
    assert result.retrieval_body == "Corrected Heading"
    assert result.model_text == "Corrected Heading"
    assert result.citation_text == "Corrected Heading"


def test_materialization_rejects_invalid_coordinate_count() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Heading\nbody",
                page_count=1,
            )
        ]
    )
    unit = heading_body_unit()
    unit["content_coordinate_count"] = 99
    unit["content_segments"] = [
        {
            "included_start": coordinate(
                "document-a",
                1,
                2,
            ),
            "included_end": coordinate(
                "document-a",
                1,
                2,
            ),
        }
    ]
    hierarchy = {
        "source:document-a:page-0001:line-0001": {
            "node_id": (
                "source:document-a:page-0001:line-0001"
            ),
            "full_heading": "Reviewed Heading",
        }
    }

    with pytest.raises(
        RetrievalUnitError,
        match="content_coordinate_count",
    ):
        materialize_retrieval_text(
            unit,
            hierarchy,
            indexes,
        )
