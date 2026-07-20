from __future__ import annotations

import pytest

from policyproof.retrieval_policy import (
    EU_ID,
    GENAI_ID,
    GPT_ID,
    RMF_ID,
    accepted_structured_start,
    cleaned_coordinates,
    combined_coordinates,
    discover_genai_footnotes,
    explicit_furniture_reason,
    gap_is_blank_only,
    is_eu_eli,
    logical_source_label,
    node_label,
    reference_entry_groups,
    reference_start_is_valid,
    repack_nonreference,
    safe_boundary_reason,
    semantic_source_key,
)
from policyproof.retrieval_units import (
    RetrievalUnitError,
    build_document_indexes,
    content_segments,
)


def make_page(
    document_id: str,
    page_number: int,
    text: str,
    *,
    page_count: int = 1,
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


def make_unit(
    indexes,
    *,
    unit_id: str,
    document_id: str,
    coordinates: list[tuple[int, int]],
    unit_kind: str = "heading_body",
    part_number: int = 1,
    part_count: int = 1,
    **metadata,
) -> dict:
    return {
        "unit_id": unit_id,
        "unit_kind": unit_kind,
        "document_id": document_id,
        "part_number": part_number,
        "part_count": part_count,
        "content_coordinate_count": len(
            coordinates
        ),
        "content_segments": content_segments(
            document_id,
            coordinates,
            indexes,
        ),
        **metadata,
    }


def test_node_label_uses_first_supported_field() -> None:
    hierarchy = {
        "source:a": {
            "full_heading": "Source heading",
            "synthetic_heading": "Synthetic",
        },
        "synthetic:b": {
            "full_heading": None,
            "synthetic_heading": (
                "Synthetic heading"
            ),
        },
    }

    assert node_label(
        "source:a",
        hierarchy,
    ) == "Source heading"
    assert node_label(
        "synthetic:b",
        hierarchy,
    ) == "Synthetic heading"

    with pytest.raises(
        RetrievalUnitError,
        match="Unknown hierarchy node",
    ):
        node_label(
            "missing",
            hierarchy,
        )


def test_semantic_source_key_and_label() -> None:
    hierarchy = {
        "source:a": {
            "full_heading": "Heading A",
        }
    }

    heading = {
        "unit_id": "heading",
        "unit_kind": "heading_body",
        "source_node_id": "source:a",
    }
    recital = {
        "unit_id": "recital",
        "unit_kind": "eu_recital",
        "recital_number": 7,
    }
    frontmatter = {
        "unit_id": "frontmatter",
        "unit_kind": "frontmatter_body",
        "frontmatter_id": "executive-summary",
    }

    assert semantic_source_key(
        heading
    ) == "source:a"
    assert semantic_source_key(
        recital
    ) == "eu-recital-007"
    assert semantic_source_key(
        frontmatter
    ) == "executive-summary"

    assert logical_source_label(
        heading,
        hierarchy,
    ) == "Heading A"
    assert logical_source_label(
        recital,
        hierarchy,
    ) == "EU recital 7"
    assert logical_source_label(
        frontmatter,
        hierarchy,
    ) == "executive-summary"


def test_combined_coordinates_requires_strict_order() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "one\ntwo\nthree",
            )
        ]
    )
    part_one = make_unit(
        indexes,
        unit_id=(
            "candidate-v2:document-a:a:part-001"
        ),
        document_id="document-a",
        coordinates=[(1, 1), (1, 2)],
        part_number=1,
        part_count=2,
    )
    part_two = make_unit(
        indexes,
        unit_id=(
            "candidate-v2:document-a:a:part-002"
        ),
        document_id="document-a",
        coordinates=[(1, 3)],
        part_number=2,
        part_count=2,
    )

    assert combined_coordinates(
        [part_one, part_two],
        indexes,
    ) == (
        (1, 1),
        (1, 2),
        (1, 3),
    )

    with pytest.raises(
        RetrievalUnitError,
        match="strictly ordered",
    ):
        combined_coordinates(
            [part_two, part_one],
            indexes,
        )


def test_discover_genai_footnotes_links_marker_and_skips_page_number() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                GENAI_ID,
                1,
                (
                    "Header\n"
                    "Body sentence.1\n"
                    "1 Footnote text"
                ),
                page_count=2,
            ),
            make_page(
                GENAI_ID,
                2,
                (
                    "Header\n"
                    "Body sentence.2\n"
                    "2 Another footnote"
                ),
                page_count=2,
            ),
        ]
    )

    records, coordinates = (
        discover_genai_footnotes(indexes)
    )

    assert coordinates == frozenset(
        {
            (1, 3),
            (2, 3),
        }
    )
    assert [
        record["linked_to_previous_marker"]
        for record in records
    ] == [True, True]


def test_discover_genai_footnotes_skips_measured_page_number_pattern() -> None:
    pages = [
        make_page(
            GENAI_ID,
            page_number,
            (
                "Header\n"
                f"{page_number - 4}\n"
                "Body"
            ),
            page_count=5,
        )
        for page_number in range(1, 6)
    ]
    indexes = build_document_indexes(
        pages
    )

    _, coordinates = (
        discover_genai_footnotes(indexes)
    )

    assert (5, 2) not in coordinates


def test_accepted_structured_start_rejects_author_and_footnote() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                GENAI_ID,
                1,
                (
                    "[1] Bracket item\n"
                    "A. Author Name\n"
                    "1 Footnote text\n"
                    "(a) Parent item\n"
                    "- Bullet"
                ),
            )
        ]
    )
    footnotes = frozenset({(1, 3)})

    assert accepted_structured_start(
        GENAI_ID,
        (1, 1),
        indexes,
        footnotes,
    )
    assert not accepted_structured_start(
        GENAI_ID,
        (1, 2),
        indexes,
        footnotes,
    )
    assert not accepted_structured_start(
        GENAI_ID,
        (1, 3),
        indexes,
        footnotes,
    )
    assert accepted_structured_start(
        GENAI_ID,
        (1, 4),
        indexes,
        footnotes,
    )
    assert accepted_structured_start(
        GENAI_ID,
        (1, 5),
        indexes,
        footnotes,
    )


def test_gap_is_blank_only() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "First.\n\nSecond.",
            )
        ]
    )

    assert gap_is_blank_only(
        "document-a",
        (1, 1),
        (1, 3),
        indexes,
    )
    assert not gap_is_blank_only(
        "document-a",
        (1, 1),
        (1, 2),
        indexes,
    )


def test_safe_boundary_strong_terminal_precedes_other_methods() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Sentence ends.\n- Item",
            )
        ]
    )

    assert safe_boundary_reason(
        "document-a",
        [(1, 1), (1, 2)],
        0,
        indexes,
        frozenset(),
    ) == "after_strong_terminal"


def test_safe_boundary_does_not_split_lowercase_url_or_footnote() -> None:
    lowercase_indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Sentence ends.\ncontinuation",
            )
        ]
    )
    url_indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Citation ends.\nhttps://example.test",
            )
        ]
    )
    footnote_indexes = build_document_indexes(
        [
            make_page(
                GENAI_ID,
                1,
                "Sentence ends.\n1 Footnote",
            )
        ]
    )

    assert safe_boundary_reason(
        "document-a",
        [(1, 1), (1, 2)],
        0,
        lowercase_indexes,
        frozenset(),
    ) is None
    assert safe_boundary_reason(
        "document-a",
        [(1, 1), (1, 2)],
        0,
        url_indexes,
        frozenset(),
    ) is None
    assert safe_boundary_reason(
        GENAI_ID,
        [(1, 1), (1, 2)],
        0,
        footnote_indexes,
        frozenset({(1, 2)}),
    ) is None


def test_safe_boundary_structured_start_rejects_introductory_colon() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "Intro:\n(a) First item",
            )
        ]
    )

    assert safe_boundary_reason(
        "document-a",
        [(1, 1), (1, 2)],
        0,
        indexes,
        frozenset(),
    ) is None


def test_safe_boundary_uses_blank_paragraph_gap() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "First clause\n\nSecond Clause",
            )
        ]
    )

    assert safe_boundary_reason(
        "document-a",
        [(1, 1), (1, 3)],
        0,
        indexes,
        frozenset(),
    ) == "at_blank_paragraph_gap"


def test_repack_nonreference_uses_first_safe_boundary_after_target() -> None:
    line_one = " ".join(
        ["alpha"] * 200
    ) + "."
    line_two = " ".join(
        ["beta"] * 190
    ) + "."
    line_three = " ".join(
        ["Gamma"] + ["gamma"] * 129
    )
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "\n".join(
                    [
                        line_one,
                        line_two,
                        line_three,
                    ]
                ),
            )
        ]
    )

    result = repack_nonreference(
        "document-a",
        [(1, 1), (1, 2), (1, 3)],
        context_words=0,
        hard_words=512,
        indexes=indexes,
        genai_footnote_coordinates=(
            frozenset()
        ),
    )

    assert result["passed"]
    assert result["pieces"] == [
        {
            "start": (1, 1),
            "end": (1, 2),
            "content_words": 390,
            "indexed_words": 390,
            "boundary_after": (
                "after_strong_terminal"
            ),
        },
        {
            "start": (1, 3),
            "end": (1, 3),
            "content_words": 130,
            "indexed_words": 130,
            "boundary_after": (
                "end_of_source_unit"
            ),
        },
    ]


def test_repack_nonreference_reports_no_safe_boundary() -> None:
    lines = [
        " ".join(["word"] * 260),
        " ".join(["continuation"] * 260),
    ]
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "\n".join(lines),
            )
        ]
    )

    result = repack_nonreference(
        "document-a",
        [(1, 1), (1, 2)],
        context_words=0,
        hard_words=512,
        indexes=indexes,
        genai_footnote_coordinates=(
            frozenset()
        ),
    )

    assert not result["passed"]
    assert result["reason"] == (
        "no_corrected_safe_boundary"
    )


def test_reference_entry_groups_gpt_preserves_complete_entries() -> None:
    entry_one = (
        "[1] Author. "
        + " ".join(["alpha"] * 190)
    )
    entry_two = (
        "[2] Author. "
        + " ".join(["beta"] * 190)
    )
    entry_three = (
        "[3] Author. "
        + " ".join(["gamma"] * 190)
    )
    indexes = build_document_indexes(
        [
            make_page(
                GPT_ID,
                1,
                "\n".join(
                    [
                        entry_one,
                        "https://one.test",
                        entry_two,
                        "https://two.test",
                        entry_three,
                    ]
                ),
            )
        ]
    )

    result = reference_entry_groups(
        GPT_ID,
        "References",
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
        ],
        indexes,
    )

    assert result["passed"]
    assert len(result["entries"]) == 3
    assert len(result["packs"]) == 2
    assert result["entries"][0][
        "coordinates"
    ] == [
        (1, 1),
        (1, 2),
    ]
    assert result["packs"][0][
        "entries"
    ][-1]["entry_number"] == 2


def test_reference_entry_groups_rejects_gpt_sequence_gap() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                GPT_ID,
                1,
                "[1] First\n[3] Third",
            )
        ]
    )

    result = reference_entry_groups(
        GPT_ID,
        "References",
        [(1, 1), (1, 2)],
        indexes,
    )

    assert not result["passed"]
    assert result["reason"] == (
        "gpt_reference_sequence_not_contiguous"
    )


def test_reference_entry_groups_genai_keeps_url_with_entry() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                GENAI_ID,
                1,
                (
                    "Smith, A. (2024). Title.\n"
                    "https://example.test\n"
                    "Jones, B. (2023). Other."
                ),
            )
        ]
    )

    result = reference_entry_groups(
        GENAI_ID,
        "Appendix B. References",
        [(1, 1), (1, 2), (1, 3)],
        indexes,
    )

    assert result["passed"]
    assert len(result["entries"]) == 2
    assert result["entries"][0][
        "coordinates"
    ] == [
        (1, 1),
        (1, 2),
    ]


def test_reference_entry_groups_rejects_nonblank_prefix() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                GPT_ID,
                1,
                "Preface\n[1] First",
            )
        ]
    )

    result = reference_entry_groups(
        GPT_ID,
        "References",
        [(1, 1), (1, 2)],
        indexes,
    )

    assert not result["passed"]
    assert result["reason"] == (
        "nonblank_reference_prefix"
    )


def test_reference_start_is_valid() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                GPT_ID,
                1,
                "[1] First",
            ),
            make_page(
                GENAI_ID,
                1,
                (
                    "Smith, A. (2024). Title.\n"
                    "https://example.test"
                ),
            ),
        ]
    )

    assert reference_start_is_valid(
        GPT_ID,
        (1, 1),
        indexes,
    )
    assert reference_start_is_valid(
        GENAI_ID,
        (1, 1),
        indexes,
    )
    assert not reference_start_is_valid(
        GENAI_ID,
        (1, 2),
        indexes,
    )


def test_cleaned_coordinates_removes_blank_and_eu_eli() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                EU_ID,
                1,
                (
                    "Content\n\n"
                    "ELI: "
                    "http://data.europa.eu/eli/"
                    "reg/2024/1689/oj 1/144"
                ),
            )
        ]
    )

    assert is_eu_eli(
        EU_ID,
        (1, 3),
        indexes,
    )
    assert cleaned_coordinates(
        EU_ID,
        [(1, 1), (1, 2), (1, 3)],
        indexes,
    ) == ((1, 1),)


def test_explicit_furniture_reasons() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                RMF_ID,
                1,
                (
                    "NIST AI 100-1 AI RMF 1.0\n"
                    "Page 1\n"
                    "Categories Subcategories\n"
                    "Continued on next page"
                ),
            ),
            *[
                make_page(
                    GENAI_ID,
                    page_number,
                    (
                        "Header\n1\nBody"
                        if page_number == 5
                        else "Header\nBody"
                    ),
                    page_count=5,
                )
                for page_number in range(1, 6)
            ],
            make_page(
                EU_ID,
                1,
                (
                    "EN\n"
                    "OJ L, 12.7.2024\n"
                    "Body\n"
                    "Footer"
                ),
            ),
            make_page(
                GPT_ID,
                1,
                "Body\n1",
            ),
        ]
    )

    assert explicit_furniture_reason(
        RMF_ID,
        (1, 1),
        indexes,
    ) == "rmf_running_header"
    assert explicit_furniture_reason(
        RMF_ID,
        (1, 2),
        indexes,
    ) == "rmf_page_label"
    assert explicit_furniture_reason(
        RMF_ID,
        (1, 3),
        indexes,
    ) == "rmf_table_header"
    assert explicit_furniture_reason(
        RMF_ID,
        (1, 4),
        indexes,
    ) == "rmf_table_continuation"
    assert explicit_furniture_reason(
        GENAI_ID,
        (5, 2),
        indexes,
    ) == "genai_page_number"
    assert explicit_furniture_reason(
        EU_ID,
        (1, 1),
        indexes,
    ) == "eu_language_header"
    assert explicit_furniture_reason(
        EU_ID,
        (1, 2),
        indexes,
    ) == "eu_journal_header"
    assert explicit_furniture_reason(
        GPT_ID,
        (1, 2),
        indexes,
    ) == "gpt_page_number"


def test_explicit_furniture_rejects_unknown_document() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "unknown",
                1,
                "Text",
            )
        ]
    )

    with pytest.raises(
        RetrievalUnitError,
        match="Unknown document ID",
    ):
        explicit_furniture_reason(
            "unknown",
            (1, 1),
            indexes,
        )
