from __future__ import annotations

import pytest

import policyproof.retrieval_passages as retrieval_passages_module
from policyproof.retrieval_passages import (
    RetrievalPassageBuildResult,
    materialize_passage_text,
    write_retrieval_passage_build,
)
from policyproof.retrieval_tokenizer import count_tokens
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
            f"{document_id}:page-"
            f"{page_number:04d}"
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
            f"{document_id}:page-"
            f"{page_number:04d}"
        ),
        "page_number": page_number,
        "line_number": line_number,
    }


def test_materialize_passage_supports_offsets_and_gaps() -> None:
    document_id = "document-a"
    first_line = (
        "First sentence. Second sentence."
    )
    split = len("First sentence. ")
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                (
                    f"{first_line}\n"
                    "excluded line\n"
                    "Third sentence."
                ),
                page_count=2,
            ),
            make_page(
                document_id,
                2,
                "Fourth sentence.",
                page_count=2,
            ),
        ]
    )
    passage = {
        "document_id": document_id,
        "unit_kind": "heading_body",
        "label": "Reviewed Heading",
        "source_slices": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "start_char_offset": split,
            },
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    3,
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
                    2,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    2,
                    1,
                ),
            },
        ],
    }

    assert materialize_passage_text(
        passage,
        indexes,
    ) == (
        "Reviewed Heading\n\n"
        "Second sentence.\n\n"
        "Third sentence.\n"
        "Fourth sentence."
    )


def test_materialize_passage_applies_end_offset() -> None:
    document_id = "document-a"
    text = "First sentence. Second sentence."
    split = len("First sentence. ")
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                text,
                page_count=1,
            )
        ]
    )
    passage = {
        "document_id": document_id,
        "unit_kind": "heading_body",
        "label": "Heading",
        "source_slices": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "end_char_offset": split,
            }
        ],
    }

    assert materialize_passage_text(
        passage,
        indexes,
    ) == "Heading\n\nFirst sentence."


def test_materialize_passage_citation_text_excludes_retrieval_label() -> None:
    document_id = "document-a"
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                "First sentence.\nexcluded line\nThird sentence.",
                page_count=2,
            ),
            make_page(
                document_id,
                2,
                "Fourth sentence.",
                page_count=2,
            ),
        ]
    )
    passage = {
        "document_id": document_id,
        "unit_kind": "heading_body",
        "label": "Reviewed Heading",
        "source_slices": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    1,
                ),
            },
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    3,
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
                    2,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    2,
                    1,
                ),
            },
        ],
    }

    assert (
        retrieval_passages_module.materialize_passage_citation_text(
            passage,
            indexes,
        )
        == (
            "First sentence.\n\n"
            "Third sentence.\n"
            "Fourth sentence."
        )
    )
    assert materialize_passage_text(
        passage,
        indexes,
    ) == (
        "Reviewed Heading\n\n"
        "First sentence.\n\n"
        "Third sentence.\n"
        "Fourth sentence."
    )


def test_materialize_passage_rejects_empty_offset_range() -> None:
    document_id = "document-a"
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                "Source text",
                page_count=1,
            )
        ]
    )
    passage = {
        "document_id": document_id,
        "unit_kind": "heading_body",
        "label": "Heading",
        "source_slices": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "start_char_offset": 5,
                "end_char_offset": 5,
            }
        ],
    }

    with pytest.raises(
        RetrievalUnitError,
        match="empty source slice",
    ):
        materialize_passage_text(
            passage,
            indexes,
        )


def test_heading_only_uses_label() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "BROKEN HEADING",
                page_count=1,
            )
        ]
    )

    assert materialize_passage_text(
        {
            "document_id": "document-a",
            "unit_kind": "heading_only",
            "label": "Reviewed Heading",
            "source_slices": [],
        },
        indexes,
    ) == "Reviewed Heading"



def test_heading_only_citation_text_uses_reviewed_label() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "BROKEN HEADING",
                page_count=1,
            )
        ]
    )
    passage = {
        "document_id": "document-a",
        "unit_kind": "heading_only",
        "label": "Reviewed Heading",
        "source_slices": [],
    }

    assert (
        retrieval_passages_module.materialize_passage_citation_text(
            passage,
            indexes,
        )
        == "Reviewed Heading"
    )
    assert materialize_passage_text(
        passage,
        indexes,
    ) == "Reviewed Heading"


def test_materialize_passage_record_texts_persists_exact_contract() -> None:
    document_id = "document-a"
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                "First sentence.\nexcluded line\nThird sentence.",
                page_count=2,
            ),
            make_page(
                document_id,
                2,
                "Fourth sentence.",
                page_count=2,
            ),
        ]
    )
    passage = {
        "schema_version": "1.0",
        "passage_id": "source-a:passage-001",
        "document_id": document_id,
        "unit_kind": "heading_body",
        "label": "Reviewed Heading",
        "source_slices": [
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    1,
                    1,
                ),
            },
            {
                "included_start": coordinate(
                    document_id,
                    1,
                    3,
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
                    2,
                    1,
                ),
                "included_end": coordinate(
                    document_id,
                    2,
                    1,
                ),
            },
        ],
    }

    result = (
        retrieval_passages_module.materialize_passage_record_texts(
            passage,
            indexes,
        )
    )

    expected_citation_text = (
        "First sentence.\n\n"
        "Third sentence.\n"
        "Fourth sentence."
    )
    expected_retrieval_text = (
        "Reviewed Heading\n\n"
        f"{expected_citation_text}"
    )

    assert result is not passage
    assert "retrieval_text" not in passage
    assert "citation_text" not in passage
    assert "passage_token_count" not in passage

    assert result["retrieval_text"] == expected_retrieval_text
    assert result["citation_text"] == expected_citation_text
    assert result["passage_token_count"] == count_tokens(
        expected_retrieval_text,
        add_special_tokens=False,
    )


@pytest.mark.parametrize(
    "existing_field",
    (
        "retrieval_text",
        "citation_text",
        "passage_token_count",
    ),
)
def test_materialize_passage_record_texts_rejects_existing_fields(
    existing_field: str,
) -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Source evidence.",
                page_count=1,
            )
        ]
    )
    passage = {
        "schema_version": "1.0",
        "passage_id": "source-a:passage-001",
        "document_id": "document-a",
        "unit_kind": "heading_body",
        "label": "Reviewed Heading",
        "source_slices": [
            {
                "included_start": coordinate(
                    "document-a",
                    1,
                    1,
                ),
                "included_end": coordinate(
                    "document-a",
                    1,
                    1,
                ),
            }
        ],
        existing_field: "stale value",
    }

    with pytest.raises(
        RetrievalUnitError,
        match="already contains",
    ):
        retrieval_passages_module.materialize_passage_record_texts(
            passage,
            indexes,
        )


def test_passage_builder_persists_retrieval_and_citation_text(
    monkeypatch,
) -> None:
    document_id = "document-a"
    node_id = "source:document-a:page-0001:line-0001"
    indexes = build_document_indexes(
        [
            make_page(
                document_id,
                1,
                "Raw heading\nSource evidence.",
                page_count=1,
            )
        ]
    )
    unit = {
        "schema_version": "1.0",
        "unit_id": (
            "candidate-v2:document-a:"
            f"{node_id}:part-001"
        ),
        "unit_kind": "heading_body",
        "document_id": document_id,
        "part_number": 1,
        "part_count": 1,
        "source_node_id": node_id,
        "content_coordinate_count": 1,
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
                    2,
                ),
            }
        ],
    }
    hierarchy = {
        node_id: {
            "node_id": node_id,
            "full_heading": "Reviewed Heading",
        }
    }

    monkeypatch.setattr(
        retrieval_passages_module,
        "validate_reviewed_intraline_anchors",
        lambda indexes: {},
    )
    monkeypatch.setattr(
        retrieval_passages_module,
        "discover_genai_footnotes",
        lambda indexes: ((), frozenset()),
    )

    passages = (
        retrieval_passages_module.build_retrieval_passages(
            (unit,),
            hierarchy,
            indexes,
        )
    )

    assert len(passages) == 1
    passage = passages[0]

    assert passage["schema_version"] == "1.1"
    assert passage["retrieval_text"] == (
        "Reviewed Heading\n\nSource evidence."
    )
    assert passage["citation_text"] == "Source evidence."
    assert passage["passage_token_count"] == count_tokens(
        passage["retrieval_text"],
        add_special_tokens=False,
    )


def sample_build_result() -> RetrievalPassageBuildResult:
    return RetrievalPassageBuildResult(
        passages=(
            {
                "schema_version": "1.0",
                "passage_id": "source-a:passage-001",
            },
        ),
        summary={
            "schema_version": "1.0",
            "passage_count": 1,
        },
        report="PASS\n",
    )


def test_write_passage_build_is_non_overwriting(
    tmp_path,
) -> None:
    passages_path = tmp_path / "passages.jsonl"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "review.txt"
    result = sample_build_result()

    write_retrieval_passage_build(
        result,
        passages_path=passages_path,
        summary_path=summary_path,
        report_path=report_path,
    )

    assert passages_path.exists()
    assert summary_path.exists()
    assert report_path.exists()

    with pytest.raises(
        RetrievalUnitError,
        match="already exists",
    ):
        write_retrieval_passage_build(
            result,
            passages_path=passages_path,
            summary_path=summary_path,
            report_path=report_path,
        )


def test_write_passage_build_rolls_back(
    tmp_path,
    monkeypatch,
) -> None:
    passages_path = tmp_path / "passages.jsonl"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "review.txt"

    def fail_summary_write(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr(
        retrieval_passages_module,
        "write_json_atomically",
        fail_summary_write,
    )

    with pytest.raises(
        OSError,
        match="simulated failure",
    ):
        write_retrieval_passage_build(
            sample_build_result(),
            passages_path=passages_path,
            summary_path=summary_path,
            report_path=report_path,
        )

    assert not passages_path.exists()
    assert not summary_path.exists()
    assert not report_path.exists()


def test_write_passage_build_requires_unique_paths(
    tmp_path,
) -> None:
    shared_path = tmp_path / "shared-output"

    with pytest.raises(
        RetrievalUnitError,
        match="must be unique",
    ):
        write_retrieval_passage_build(
            sample_build_result(),
            passages_path=shared_path,
            summary_path=shared_path,
            report_path=tmp_path / "review.txt",
        )
