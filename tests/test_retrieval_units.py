from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyproof.retrieval_units import (
    RetrievalUnitError,
    build_document_indexes,
    content_segments,
    coordinate_from_record,
    coordinate_record,
    count_words,
    group_units_by_logical_source,
    load_jsonl,
    logical_source_key,
    segment_coordinates,
    unit_coordinates,
    write_json_atomically,
    write_jsonl_atomically,
    write_text_atomically,
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


def make_unit(
    unit_id: str,
    *,
    document_id: str = "document-a",
    unit_kind: str = "heading_body",
    part_number: int = 1,
    part_count: int = 1,
) -> dict:
    return {
        "unit_id": unit_id,
        "unit_kind": unit_kind,
        "document_id": document_id,
        "part_number": part_number,
        "part_count": part_count,
    }


def test_build_document_indexes_preserves_order_and_blank_lines() -> None:
    pages = [
        make_page(
            "document-b",
            1,
            "B one",
            page_count=1,
        ),
        make_page(
            "document-a",
            2,
            "A three",
            page_count=2,
        ),
        make_page(
            "document-a",
            1,
            "A one\n\nA two",
            page_count=2,
        ),
    ]

    indexes = build_document_indexes(pages)

    assert indexes.document_order == (
        "document-b",
        "document-a",
    )
    assert indexes.total_line_count == 5
    assert indexes.by_document[
        "document-a"
    ].coordinates == (
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
    )
    assert indexes.by_document[
        "document-a"
    ].line_text[(1, 2)] == ""


def test_build_document_indexes_rejects_noncontiguous_pages() -> None:
    pages = [
        make_page(
            "document-a",
            1,
            "one",
            page_count=2,
        ),
        make_page(
            "document-a",
            3,
            "three",
            page_count=2,
        ),
    ]

    with pytest.raises(
        RetrievalUnitError,
        match="contiguous from 1",
    ):
        build_document_indexes(pages)


def test_build_document_indexes_rejects_page_count_mismatch() -> None:
    pages = [
        make_page(
            "document-a",
            1,
            "one",
            page_count=3,
        )
    ]

    with pytest.raises(
        RetrievalUnitError,
        match="page_count values",
    ):
        build_document_indexes(pages)


def test_build_document_indexes_rejects_invalid_page_id() -> None:
    page = make_page(
        "document-a",
        1,
        "one",
        page_count=1,
    )
    page["page_id"] = "wrong-page-id"

    with pytest.raises(
        RetrievalUnitError,
        match="does not match",
    ):
        build_document_indexes([page])


def test_coordinate_round_trip_and_validation() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "one\ntwo",
                page_count=1,
            )
        ]
    )
    index = indexes.by_document["document-a"]
    record = coordinate_record(index, (1, 2))

    assert record == {
        "page_id": "document-a:page-0001",
        "page_number": 1,
        "line_number": 2,
    }
    assert coordinate_from_record(
        record,
        index=index,
    ) == (1, 2)

    invalid = dict(record)
    invalid["page_id"] = "document-a:page-9999"

    with pytest.raises(
        RetrievalUnitError,
        match="does not match",
    ):
        coordinate_from_record(
            invalid,
            index=index,
        )


def test_segment_coordinates_can_cross_page_boundary() -> None:
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
    segment = {
        "included_start": {
            "page_id": "document-a:page-0001",
            "page_number": 1,
            "line_number": 2,
        },
        "included_end": {
            "page_id": "document-a:page-0002",
            "page_number": 2,
            "line_number": 1,
        },
    }

    assert segment_coordinates(
        "document-a",
        segment,
        indexes,
    ) == (
        (1, 2),
        (2, 1),
    )


def test_content_segments_preserve_gaps() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "one\ntwo\nthree\nfour",
                page_count=1,
            )
        ]
    )

    segments = content_segments(
        "document-a",
        [
            (1, 1),
            (1, 2),
            (1, 4),
        ],
        indexes,
    )

    assert segments == [
        {
            "included_start": {
                "page_id": "document-a:page-0001",
                "page_number": 1,
                "line_number": 1,
            },
            "included_end": {
                "page_id": "document-a:page-0001",
                "page_number": 1,
                "line_number": 2,
            },
        },
        {
            "included_start": {
                "page_id": "document-a:page-0001",
                "page_number": 1,
                "line_number": 4,
            },
            "included_end": {
                "page_id": "document-a:page-0001",
                "page_number": 1,
                "line_number": 4,
            },
        },
    ]


def test_unit_coordinates_rejects_overlapping_segments() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "one\ntwo\nthree",
                page_count=1,
            )
        ]
    )
    unit = {
        "unit_id": (
            "candidate-v2:document-a:source-a:part-001"
        ),
        "document_id": "document-a",
        "content_coordinate_count": 4,
        "content_segments": [
            {
                "included_start": {
                    "page_id": "document-a:page-0001",
                    "page_number": 1,
                    "line_number": 1,
                },
                "included_end": {
                    "page_id": "document-a:page-0001",
                    "page_number": 1,
                    "line_number": 2,
                },
            },
            {
                "included_start": {
                    "page_id": "document-a:page-0001",
                    "page_number": 1,
                    "line_number": 2,
                },
                "included_end": {
                    "page_id": "document-a:page-0001",
                    "page_number": 1,
                    "line_number": 3,
                },
            },
        ],
    }

    with pytest.raises(
        RetrievalUnitError,
        match="segments overlap",
    ):
        unit_coordinates(unit, indexes)


def test_unit_coordinates_checks_declared_count() -> None:
    indexes = build_document_indexes(
        [
            make_page(
                "document-a",
                1,
                "one\ntwo",
                page_count=1,
            )
        ]
    )
    unit = {
        "unit_id": (
            "candidate-v2:document-a:source-a:part-001"
        ),
        "document_id": "document-a",
        "content_coordinate_count": 3,
        "content_segments": [
            {
                "included_start": {
                    "page_id": "document-a:page-0001",
                    "page_number": 1,
                    "line_number": 1,
                },
                "included_end": {
                    "page_id": "document-a:page-0001",
                    "page_number": 1,
                    "line_number": 2,
                },
            }
        ],
    }

    with pytest.raises(
        RetrievalUnitError,
        match="content_coordinate_count",
    ):
        unit_coordinates(unit, indexes)


def test_logical_source_key_validates_suffix_and_part_number() -> None:
    unit = make_unit(
        "candidate-v2:document-a:source-a:part-002",
        part_number=2,
        part_count=2,
    )

    assert logical_source_key(unit) == (
        "candidate-v2:document-a:source-a"
    )

    invalid = dict(unit)
    invalid["part_number"] = 1

    with pytest.raises(
        RetrievalUnitError,
        match="does not match suffix",
    ):
        logical_source_key(invalid)

    with pytest.raises(
        RetrievalUnitError,
        match="must end",
    ):
        logical_source_key(
            {
                "unit_id": (
                    "candidate-v2:document-a:source-a"
                )
            }
        )


def test_group_units_preserves_source_order_and_sorts_parts() -> None:
    units = [
        make_unit(
            "candidate-v2:document-a:source-b:part-002",
            part_number=2,
            part_count=2,
        ),
        make_unit(
            "candidate-v2:document-a:source-a:part-001",
        ),
        make_unit(
            "candidate-v2:document-a:source-b:part-001",
            part_number=1,
            part_count=2,
        ),
    ]

    grouped = group_units_by_logical_source(units)

    assert grouped.ordered_keys == (
        (
            "document-a",
            "candidate-v2:document-a:source-b",
        ),
        (
            "document-a",
            "candidate-v2:document-a:source-a",
        ),
    )
    assert [
        unit["part_number"]
        for unit in grouped.groups[
            (
                "document-a",
                "candidate-v2:document-a:source-b",
            )
        ]
    ] == [1, 2]


def test_group_units_rejects_missing_part() -> None:
    units = [
        make_unit(
            "candidate-v2:document-a:source-a:part-001",
            part_number=1,
            part_count=3,
        ),
        make_unit(
            "candidate-v2:document-a:source-a:part-003",
            part_number=3,
            part_count=3,
        ),
    ]

    with pytest.raises(
        RetrievalUnitError,
        match="parts must be contiguous",
    ):
        group_units_by_logical_source(units)


def test_group_units_rejects_mixed_kinds() -> None:
    units = [
        make_unit(
            "candidate-v2:document-a:source-a:part-001",
            unit_kind="heading_body",
            part_number=1,
            part_count=2,
        ),
        make_unit(
            "candidate-v2:document-a:source-a:part-002",
            unit_kind="heading_only",
            part_number=2,
            part_count=2,
        ),
    ]

    with pytest.raises(
        RetrievalUnitError,
        match="mixes unit kinds",
    ):
        group_units_by_logical_source(units)


def test_atomic_writers_create_expected_files(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "nested" / "records.jsonl"
    json_path = tmp_path / "summary.json"
    text_path = tmp_path / "report.txt"

    write_jsonl_atomically(
        jsonl_path,
        [{"id": 1}, {"id": 2}],
    )
    write_json_atomically(
        json_path,
        {"count": 2},
    )
    write_text_atomically(
        text_path,
        "report\n",
    )

    assert jsonl_path.read_text(
        encoding="utf-8"
    ) == (
        '{"id": 1}\n'
        '{"id": 2}\n'
    )
    assert json.loads(
        json_path.read_text(encoding="utf-8")
    ) == {"count": 2}
    assert text_path.read_text(
        encoding="utf-8"
    ) == "report\n"


def test_atomic_writer_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"
    output_path.write_text(
        "existing\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RetrievalUnitError,
        match="Output already exists",
    ):
        write_jsonl_atomically(
            output_path,
            [{"replacement": True}],
        )

    assert output_path.read_text(
        encoding="utf-8"
    ) == "existing\n"


def test_atomic_writer_removes_temporary_file_on_failure(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"

    with pytest.raises(TypeError):
        write_jsonl_atomically(
            output_path,
            [{"invalid": {1, 2}}],
        )

    assert not output_path.exists()
    assert not list(
        tmp_path.glob(
            f".{output_path.name}.*.tmp"
        )
    )


def test_load_jsonl_and_count_words(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"id": 1}\n\n{"id": 2}\n',
        encoding="utf-8",
    )

    assert load_jsonl(
        path,
        record_name="fixture",
    ) == [{"id": 1}, {"id": 2}]
    assert count_words(
        "MEASURE 2.6 text"
    ) == 3
