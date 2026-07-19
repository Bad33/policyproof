import json

import pytest

from policyproof.heading_spans import (
    HeadingSpanError,
    build_heading_spans,
    write_jsonl,
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


def make_heading(
    document_id: str,
    page_number: int,
    start_line_number: int,
    *,
    end_line_number: int | None = None,
) -> dict:
    if end_line_number is None:
        end_line_number = start_line_number

    heading_id = f"{document_id}:page-{page_number:04d}:line-{start_line_number:04d}"

    return {
        "schema_version": "1.0",
        "heading_id": heading_id,
        "page_id": f"{document_id}:page-{page_number:04d}",
        "document_id": document_id,
        "page_number": page_number,
        "start_line_number": start_line_number,
        "end_line_number": end_line_number,
        "source_line_numbers": list(range(start_line_number, end_line_number + 1)),
    }


def make_source_node(
    heading: dict,
    *,
    parent_node_id: str | None,
    depth: int,
    document_order: int,
    ancestor_node_ids: list[str],
) -> dict:
    return {
        "schema_version": "1.0",
        "node_id": f"source:{heading['heading_id']}",
        "document_id": heading["document_id"],
        "node_kind": "source",
        "heading_id": heading["heading_id"],
        "parent_node_id": parent_node_id,
        "depth": depth,
        "document_order": document_order,
        "ancestor_node_ids": ancestor_node_ids,
        "page_id": heading["page_id"],
        "page_number": heading["page_number"],
        "start_line_number": heading["start_line_number"],
        "end_line_number": heading["end_line_number"],
        "anchor_heading_id": None,
        "anchor_page_id": None,
        "anchor_page_number": None,
        "anchor_line_number": None,
    }


def make_synthetic_node(
    *,
    node_id: str,
    document_id: str,
    parent_node_id: str,
    depth: int,
    document_order: int,
    ancestor_node_ids: list[str],
    anchor_heading: dict,
) -> dict:
    return {
        "schema_version": "1.0",
        "node_id": node_id,
        "document_id": document_id,
        "node_kind": "synthetic",
        "heading_id": None,
        "parent_node_id": parent_node_id,
        "depth": depth,
        "document_order": document_order,
        "ancestor_node_ids": ancestor_node_ids,
        "page_id": None,
        "page_number": None,
        "start_line_number": None,
        "end_line_number": None,
        "anchor_heading_id": anchor_heading["heading_id"],
        "anchor_page_id": anchor_heading["page_id"],
        "anchor_page_number": anchor_heading["page_number"],
        "anchor_line_number": anchor_heading["start_line_number"],
    }


def records_by_node_id(records: list[dict]) -> dict[str, dict]:
    return {record["node_id"]: record for record in records}


def test_direct_body_stops_before_next_heading_and_subtree_includes_child() -> None:
    document_id = "test-document"
    pages = [
        make_page(
            document_id,
            1,
            "\n".join(
                [
                    "1 Parent",
                    "parent introduction",
                    "1.1 Child",
                    "child body",
                    "2 Sibling",
                    "sibling body",
                ]
            ),
            page_count=1,
        )
    ]
    parent_heading = make_heading(document_id, 1, 1)
    child_heading = make_heading(document_id, 1, 3)
    sibling_heading = make_heading(document_id, 1, 5)

    parent_node = make_source_node(
        parent_heading,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )
    child_node = make_source_node(
        child_heading,
        parent_node_id=parent_node["node_id"],
        depth=2,
        document_order=2,
        ancestor_node_ids=[parent_node["node_id"]],
    )
    sibling_node = make_source_node(
        sibling_heading,
        parent_node_id=None,
        depth=1,
        document_order=3,
        ancestor_node_ids=[],
    )

    records = build_heading_spans(
        pages,
        [
            parent_heading,
            child_heading,
            sibling_heading,
        ],
        [
            parent_node,
            child_node,
            sibling_node,
        ],
    )
    by_node = records_by_node_id(records)
    parent = by_node[parent_node["node_id"]]
    child = by_node[child_node["node_id"]]

    assert parent["direct_body"]["raw_line_count"] == 1
    assert parent["direct_body"]["included_start"]["line_number"] == 2
    assert parent["direct_body"]["included_end"]["line_number"] == 2

    assert parent["subtree"]["raw_line_count"] == 3
    assert parent["subtree"]["included_start"]["line_number"] == 2
    assert parent["subtree"]["included_end"]["line_number"] == 4
    assert parent["subtree"]["includes_descendant_heading_lines"] is True

    assert child["direct_body"]["raw_line_count"] == 1
    assert child["direct_body"]["included_start"]["line_number"] == 4
    assert child["direct_body"]["end_boundary"]["heading_id"] == sibling_heading["heading_id"]


def test_consecutive_headings_produce_exact_empty_direct_body() -> None:
    document_id = "test-document"
    pages = [
        make_page(
            document_id,
            1,
            "1 First\n2 Second\nsecond body",
            page_count=1,
        )
    ]
    first_heading = make_heading(document_id, 1, 1)
    second_heading = make_heading(document_id, 1, 2)
    first_node = make_source_node(
        first_heading,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )
    second_node = make_source_node(
        second_heading,
        parent_node_id=None,
        depth=1,
        document_order=2,
        ancestor_node_ids=[],
    )

    records = build_heading_spans(
        pages,
        [first_heading, second_heading],
        [first_node, second_node],
    )
    first = records[0]["direct_body"]

    assert first["is_empty"] is True
    assert first["is_blank_only"] is False
    assert first["raw_line_count"] == 0
    assert first["included_start"] is None
    assert first["included_end"] is None
    assert first["start_boundary"]["coordinate"]["line_number"] == 1
    assert first["end_boundary"]["coordinate"]["line_number"] == 2


def test_blank_only_direct_body_is_not_exact_empty() -> None:
    document_id = "test-document"
    pages = [
        make_page(
            document_id,
            1,
            "1 First\n\n2 Second",
            page_count=1,
        )
    ]
    first_heading = make_heading(document_id, 1, 1)
    second_heading = make_heading(document_id, 1, 3)
    first_node = make_source_node(
        first_heading,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )
    second_node = make_source_node(
        second_heading,
        parent_node_id=None,
        depth=1,
        document_order=2,
        ancestor_node_ids=[],
    )

    records = build_heading_spans(
        pages,
        [first_heading, second_heading],
        [first_node, second_node],
    )
    first = records[0]["direct_body"]

    assert first["is_empty"] is False
    assert first["is_blank_only"] is True
    assert first["raw_line_count"] == 1
    assert first["nonblank_line_count"] == 0


def test_final_heading_uses_document_end_across_pages() -> None:
    document_id = "test-document"
    pages = [
        make_page(
            document_id,
            1,
            "1 Final\nfirst body line",
            page_count=2,
        ),
        make_page(
            document_id,
            2,
            "second body line",
            page_count=2,
        ),
    ]
    heading = make_heading(document_id, 1, 1)
    node = make_source_node(
        heading,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )

    record = build_heading_spans(
        pages,
        [heading],
        [node],
    )[0]

    assert record["direct_body"]["raw_line_count"] == 2
    assert record["direct_body"]["is_multi_page"] is True
    assert record["direct_body"]["included_start"] == {
        "page_id": f"{document_id}:page-0001",
        "page_number": 1,
        "line_number": 2,
    }
    assert record["direct_body"]["included_end"] == {
        "page_id": f"{document_id}:page-0002",
        "page_number": 2,
        "line_number": 1,
    }
    assert record["direct_body"]["end_boundary"]["kind"] == "after_document_line"
    assert record["subtree"]["end_boundary"]["kind"] == "after_document_line"


def test_synthetic_nodes_receive_only_descendant_envelopes() -> None:
    document_id = "test-document"
    pages = [
        make_page(
            document_id,
            1,
            "\n".join(
                [
                    "3 Actions",
                    "GOVERN 1.1",
                    "govern one body",
                    "GOVERN 1.2",
                    "govern two body",
                    "MAP 1.1",
                    "map body",
                ]
            ),
            page_count=1,
        )
    ]

    root_heading = make_heading(document_id, 1, 1)
    govern_one_heading = make_heading(document_id, 1, 2)
    govern_two_heading = make_heading(document_id, 1, 4)
    map_heading = make_heading(document_id, 1, 6)

    root_node = make_source_node(
        root_heading,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )
    function_node_id = "synthetic:test-document:rmf-function:govern"
    function_node = make_synthetic_node(
        node_id=function_node_id,
        document_id=document_id,
        parent_node_id=root_node["node_id"],
        depth=2,
        document_order=2,
        ancestor_node_ids=[root_node["node_id"]],
        anchor_heading=govern_one_heading,
    )
    category_node_id = "synthetic:test-document:rmf-category:govern:1"
    category_node = make_synthetic_node(
        node_id=category_node_id,
        document_id=document_id,
        parent_node_id=function_node_id,
        depth=3,
        document_order=3,
        ancestor_node_ids=[
            root_node["node_id"],
            function_node_id,
        ],
        anchor_heading=govern_one_heading,
    )
    govern_one_node = make_source_node(
        govern_one_heading,
        parent_node_id=category_node_id,
        depth=4,
        document_order=4,
        ancestor_node_ids=[
            root_node["node_id"],
            function_node_id,
            category_node_id,
        ],
    )
    govern_two_node = make_source_node(
        govern_two_heading,
        parent_node_id=category_node_id,
        depth=4,
        document_order=5,
        ancestor_node_ids=[
            root_node["node_id"],
            function_node_id,
            category_node_id,
        ],
    )
    map_node = make_source_node(
        map_heading,
        parent_node_id=root_node["node_id"],
        depth=2,
        document_order=6,
        ancestor_node_ids=[root_node["node_id"]],
    )

    records = build_heading_spans(
        pages,
        [
            root_heading,
            govern_one_heading,
            govern_two_heading,
            map_heading,
        ],
        [
            root_node,
            function_node,
            category_node,
            govern_one_node,
            govern_two_node,
            map_node,
        ],
    )
    by_node = records_by_node_id(records)
    category = by_node[category_node_id]

    assert category["heading_source"] is None
    assert category["direct_body"] is None
    assert category["subtree"] is None

    envelope = category["source_descendant_envelope"]

    assert envelope["source_descendant_heading_count"] == 2
    assert envelope["included_start"]["line_number"] == 2
    assert envelope["included_end"]["line_number"] == 5
    assert envelope["start_boundary"]["kind"] == "at_source_heading"
    assert envelope["end_boundary"]["kind"] == "before_source_heading"
    assert envelope["end_boundary"]["heading_id"] == map_heading["heading_id"]


def test_output_preserves_exact_hierarchy_input_order() -> None:
    document_a = "document-a"
    document_b = "document-b"

    pages = [
        make_page(
            document_a,
            1,
            "A heading\nA body",
            page_count=1,
        ),
        make_page(
            document_b,
            1,
            "B heading\nB body",
            page_count=1,
        ),
    ]
    heading_a = make_heading(document_a, 1, 1)
    heading_b = make_heading(document_b, 1, 1)
    node_a = make_source_node(
        heading_a,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )
    node_b = make_source_node(
        heading_b,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )

    hierarchy = [node_b, node_a]
    records = build_heading_spans(
        pages,
        [heading_a, heading_b],
        hierarchy,
    )

    assert [record["node_id"] for record in records] == [node["node_id"] for node in hierarchy]
    assert all(record["schema_version"] == "1.0" for record in records)


def test_overlapping_reconstructed_headings_fail_closed() -> None:
    document_id = "test-document"
    pages = [
        make_page(
            document_id,
            1,
            "first heading\ncontinued heading\nsecond heading",
            page_count=1,
        )
    ]
    first_heading = make_heading(
        document_id,
        1,
        1,
        end_line_number=2,
    )
    second_heading = make_heading(document_id, 1, 2)
    first_node = make_source_node(
        first_heading,
        parent_node_id=None,
        depth=1,
        document_order=1,
        ancestor_node_ids=[],
    )
    second_node = make_source_node(
        second_heading,
        parent_node_id=None,
        depth=1,
        document_order=2,
        ancestor_node_ids=[],
    )

    with pytest.raises(
        HeadingSpanError,
        match="overlaps the reconstructed heading",
    ):
        build_heading_spans(
            pages,
            [first_heading, second_heading],
            [first_node, second_node],
        )


def test_writer_rejects_existing_output(tmp_path) -> None:
    output_path = tmp_path / "heading-spans.jsonl"
    output_path.write_text(
        json.dumps({"existing": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HeadingSpanError,
        match="Output already exists",
    ):
        write_jsonl(
            output_path,
            [{"schema_version": "1.0"}],
        )
