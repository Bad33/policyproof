"""Assign deterministic source-coordinate spans to heading hierarchy nodes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class HeadingSpanError(RuntimeError):
    """Raised when safe heading spans cannot be assigned."""


def load_jsonl(
    path: Path,
    *,
    record_name: str,
) -> list[dict[str, Any]]:
    """Load JSON objects from a JSONL file."""
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise HeadingSpanError(f"{record_name} file not found: {path}") from error

    records: list[dict[str, Any]] = []

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HeadingSpanError(
                    f"Invalid JSON in {record_name} file on line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise HeadingSpanError(
                    f"{record_name} record on line {line_number} must be an object."
                )

            records.append(record)

    if not records:
        raise HeadingSpanError(f"{record_name} file contains no records.")

    return records


def require_fields(
    record: dict[str, Any],
    required_fields: set[str],
    *,
    record_name: str,
) -> None:
    """Require fields needed to reconstruct source coordinates."""
    missing_fields = sorted(required_fields - record.keys())

    if missing_fields:
        raise HeadingSpanError(f"{record_name} is missing fields: {', '.join(missing_fields)}")


def build_document_indexes(
    pages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build document-scoped page and line coordinate indexes."""
    pages_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_ids_seen: set[str] = set()

    for record_index, page in enumerate(pages):
        require_fields(
            page,
            {
                "page_id",
                "document_id",
                "page_number",
                "page_count",
                "text",
            },
            record_name=f"Page record {record_index}",
        )

        page_id = page["page_id"]
        document_id = page["document_id"]
        page_number = page["page_number"]
        page_count = page["page_count"]
        text = page["text"]

        if not isinstance(page_id, str) or not page_id:
            raise HeadingSpanError("Page record has no valid page_id.")

        if page_id in page_ids_seen:
            raise HeadingSpanError(f"Duplicate page_id: {page_id}")

        page_ids_seen.add(page_id)

        if not isinstance(document_id, str) or not document_id:
            raise HeadingSpanError(f"{page_id}: invalid document_id.")

        if not isinstance(page_number, int) or page_number < 1:
            raise HeadingSpanError(f"{page_id}: invalid page_number.")

        if not isinstance(page_count, int) or page_count < 1:
            raise HeadingSpanError(f"{page_id}: invalid page_count.")

        if not isinstance(text, str):
            raise HeadingSpanError(f"{page_id}: text must be a string.")

        pages_by_document[document_id].append(page)

    indexes: dict[str, dict[str, Any]] = {}

    for document_id, document_pages in pages_by_document.items():
        ordered_pages = sorted(
            document_pages,
            key=lambda page: page["page_number"],
        )
        page_numbers = [page["page_number"] for page in ordered_pages]
        expected_page_numbers = list(range(1, len(ordered_pages) + 1))

        if page_numbers != expected_page_numbers:
            raise HeadingSpanError(f"{document_id}: pages must be contiguous from page 1.")

        expected_page_count = len(ordered_pages)

        for page in ordered_pages:
            if page["page_count"] != expected_page_count:
                raise HeadingSpanError(
                    f"{page['page_id']}: page_count does not match document pages."
                )

        coordinates: list[tuple[int, int]] = []
        line_text: dict[tuple[int, int], str] = {}
        page_ids: dict[int, str] = {}

        for page in ordered_pages:
            page_number = page["page_number"]
            page_ids[page_number] = page["page_id"]

            for line_number, text in enumerate(
                page["text"].splitlines(),
                start=1,
            ):
                coordinate = (page_number, line_number)
                coordinates.append(coordinate)
                line_text[coordinate] = text

        if not coordinates:
            raise HeadingSpanError(f"{document_id}: document contains no extracted source lines.")

        indexes[document_id] = {
            "coordinates": coordinates,
            "coordinate_indexes": {
                coordinate: index for index, coordinate in enumerate(coordinates)
            },
            "line_text": line_text,
            "page_ids": page_ids,
        }

    return indexes


def index_hierarchy(
    nodes: list[dict[str, Any]],
    document_indexes: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Validate hierarchy identity and index source nodes by heading."""
    nodes_by_id: dict[str, dict[str, Any]] = {}
    source_nodes_by_heading_id: dict[str, dict[str, Any]] = {}

    required_fields = {
        "node_id",
        "document_id",
        "node_kind",
        "heading_id",
        "parent_node_id",
        "depth",
        "document_order",
        "ancestor_node_ids",
        "page_id",
        "page_number",
        "start_line_number",
        "end_line_number",
        "anchor_heading_id",
        "anchor_page_id",
        "anchor_page_number",
        "anchor_line_number",
    }

    for record_index, node in enumerate(nodes):
        require_fields(
            node,
            required_fields,
            record_name=f"Hierarchy node {record_index}",
        )

        node_id = node["node_id"]
        document_id = node["document_id"]
        node_kind = node["node_kind"]

        if not isinstance(node_id, str) or not node_id:
            raise HeadingSpanError("Hierarchy node has no valid node_id.")

        if node_id in nodes_by_id:
            raise HeadingSpanError(f"Duplicate hierarchy node_id: {node_id}")

        if document_id not in document_indexes:
            raise HeadingSpanError(f"{node_id}: hierarchy document has no page records.")

        if node_kind not in {"source", "synthetic"}:
            raise HeadingSpanError(f"{node_id}: unsupported node_kind {node_kind!r}.")

        if not isinstance(node["depth"], int) or node["depth"] < 1:
            raise HeadingSpanError(f"{node_id}: invalid depth.")

        if not isinstance(node["ancestor_node_ids"], list) or not all(
            isinstance(ancestor_id, str) for ancestor_id in node["ancestor_node_ids"]
        ):
            raise HeadingSpanError(f"{node_id}: ancestor_node_ids must be a list of strings.")

        nodes_by_id[node_id] = node

        if node_kind == "source":
            heading_id = node["heading_id"]

            if not isinstance(heading_id, str) or not heading_id:
                raise HeadingSpanError(f"{node_id}: source node has no heading_id.")

            if heading_id in source_nodes_by_heading_id:
                raise HeadingSpanError(f"Duplicate source heading_id in hierarchy: {heading_id}")

            source_nodes_by_heading_id[heading_id] = node
        else:
            direct_provenance_fields = (
                "heading_id",
                "page_id",
                "page_number",
                "start_line_number",
                "end_line_number",
            )

            populated_fields = [
                field_name
                for field_name in direct_provenance_fields
                if node[field_name] is not None
            ]

            if populated_fields:
                raise HeadingSpanError(
                    f"{node_id}: synthetic node has direct provenance: {populated_fields}"
                )

    for node in nodes:
        node_id = node["node_id"]
        parent_node_id = node["parent_node_id"]

        if parent_node_id is None:
            if node["depth"] != 1:
                raise HeadingSpanError(f"{node_id}: root node depth must equal 1.")

            if node["ancestor_node_ids"]:
                raise HeadingSpanError(f"{node_id}: root node cannot have ancestors.")

            continue

        parent = nodes_by_id.get(parent_node_id)

        if parent is None:
            raise HeadingSpanError(f"{node_id}: parent node does not exist.")

        if parent["document_id"] != node["document_id"]:
            raise HeadingSpanError(f"{node_id}: parent belongs to another document.")

        if node["depth"] != parent["depth"] + 1:
            raise HeadingSpanError(f"{node_id}: invalid hierarchy depth.")

        expected_ancestors = [
            *parent["ancestor_node_ids"],
            parent_node_id,
        ]

        if node["ancestor_node_ids"] != expected_ancestors:
            raise HeadingSpanError(f"{node_id}: invalid ancestor_node_ids.")

    return nodes_by_id, source_nodes_by_heading_id


def position_source_headings(
    headings: list[dict[str, Any]],
    source_nodes_by_heading_id: dict[str, dict[str, Any]],
    document_indexes: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """Position source headings and calculate direct and subtree indexes."""
    required_fields = {
        "heading_id",
        "page_id",
        "document_id",
        "page_number",
        "start_line_number",
        "end_line_number",
        "source_line_numbers",
    }

    headings_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    heading_ids_seen: set[str] = set()

    for record_index, heading in enumerate(headings):
        require_fields(
            heading,
            required_fields,
            record_name=f"Reconstructed heading {record_index}",
        )

        heading_id = heading["heading_id"]
        document_id = heading["document_id"]

        if not isinstance(heading_id, str) or not heading_id:
            raise HeadingSpanError("Reconstructed heading has no valid heading_id.")

        if heading_id in heading_ids_seen:
            raise HeadingSpanError(f"Duplicate reconstructed heading_id: {heading_id}")

        heading_ids_seen.add(heading_id)

        if document_id not in document_indexes:
            raise HeadingSpanError(f"{heading_id}: heading document has no page records.")

        headings_by_document[document_id].append(heading)

    if heading_ids_seen != set(source_nodes_by_heading_id):
        missing = sorted(set(source_nodes_by_heading_id) - heading_ids_seen)
        unexpected = sorted(heading_ids_seen - set(source_nodes_by_heading_id))

        raise HeadingSpanError(
            "Reconstructed headings and source hierarchy nodes differ. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    if set(headings_by_document) != set(document_indexes):
        raise HeadingSpanError("Every page document must have reconstructed headings.")

    spans_by_document: dict[str, list[dict[str, Any]]] = {}
    spans_by_node_id: dict[str, dict[str, Any]] = {}

    for document_id, document_headings in headings_by_document.items():
        ordered_headings = sorted(
            document_headings,
            key=lambda heading: (
                heading["page_number"],
                heading["start_line_number"],
            ),
        )

        if [heading["heading_id"] for heading in document_headings] != [
            heading["heading_id"] for heading in ordered_headings
        ]:
            raise HeadingSpanError(
                f"{document_id}: reconstructed headings are not in source order."
            )

        document_index = document_indexes[document_id]
        coordinate_indexes = document_index["coordinate_indexes"]
        page_ids = document_index["page_ids"]
        positioned: list[dict[str, Any]] = []

        for heading in ordered_headings:
            heading_id = heading["heading_id"]
            page_number = heading["page_number"]
            start_line_number = heading["start_line_number"]
            end_line_number = heading["end_line_number"]

            if heading["page_id"] != page_ids.get(page_number):
                raise HeadingSpanError(f"{heading_id}: page_id and page_number do not match.")

            if (
                not isinstance(start_line_number, int)
                or not isinstance(end_line_number, int)
                or start_line_number < 1
                or end_line_number < start_line_number
            ):
                raise HeadingSpanError(f"{heading_id}: invalid heading line range.")

            start_coordinate = (
                page_number,
                start_line_number,
            )
            end_coordinate = (
                page_number,
                end_line_number,
            )

            if start_coordinate not in coordinate_indexes:
                raise HeadingSpanError(
                    f"{heading_id}: heading start is outside source coordinates."
                )

            if end_coordinate not in coordinate_indexes:
                raise HeadingSpanError(f"{heading_id}: heading end is outside source coordinates.")

            source_line_numbers = heading["source_line_numbers"]

            if (
                not isinstance(source_line_numbers, list)
                or not source_line_numbers
                or not all(isinstance(line_number, int) for line_number in source_line_numbers)
                or source_line_numbers != sorted(set(source_line_numbers))
                or source_line_numbers[0] != start_line_number
                or source_line_numbers[-1] != end_line_number
            ):
                raise HeadingSpanError(f"{heading_id}: invalid source_line_numbers.")

            node = source_nodes_by_heading_id[heading_id]

            for field_name in (
                "document_id",
                "page_id",
                "page_number",
                "start_line_number",
                "end_line_number",
            ):
                if node[field_name] != heading[field_name]:
                    raise HeadingSpanError(f"{heading_id}: hierarchy disagrees on {field_name}.")

            positioned.append(
                {
                    "heading": heading,
                    "node": node,
                    "heading_start_coordinate": start_coordinate,
                    "heading_end_coordinate": end_coordinate,
                    "heading_start_index": coordinate_indexes[start_coordinate],
                    "heading_end_index": coordinate_indexes[end_coordinate],
                }
            )

        document_spans: list[dict[str, Any]] = []
        coordinates = document_index["coordinates"]

        for index, current in enumerate(positioned):
            heading = current["heading"]
            node = current["node"]

            direct_start_index = current["heading_end_index"] + 1

            if index + 1 < len(positioned):
                next_heading = positioned[index + 1]
                direct_stop_index = next_heading["heading_start_index"]
                direct_stop_heading_id = next_heading["heading"]["heading_id"]
            else:
                direct_stop_index = len(coordinates)
                direct_stop_heading_id = None

            if direct_stop_index < direct_start_index:
                raise HeadingSpanError(
                    f"{heading['heading_id']}: next heading overlaps the reconstructed heading."
                )

            subtree_start_index = direct_start_index
            subtree_stop_index = len(coordinates)
            subtree_stop_heading_id = None

            for later_heading in positioned[index + 1 :]:
                if later_heading["node"]["depth"] <= node["depth"]:
                    subtree_stop_index = later_heading["heading_start_index"]
                    subtree_stop_heading_id = later_heading["heading"]["heading_id"]
                    break

            if subtree_stop_index < subtree_start_index:
                raise HeadingSpanError(f"{heading['heading_id']}: subtree boundary is inverted.")

            span = {
                "heading": heading,
                "node": node,
                "heading_start_coordinate": current["heading_start_coordinate"],
                "heading_end_coordinate": current["heading_end_coordinate"],
                "heading_start_index": current["heading_start_index"],
                "heading_end_index": current["heading_end_index"],
                "direct_start_index": direct_start_index,
                "direct_stop_index": direct_stop_index,
                "direct_stop_heading_id": direct_stop_heading_id,
                "subtree_start_index": subtree_start_index,
                "subtree_stop_index": subtree_stop_index,
                "subtree_stop_heading_id": subtree_stop_heading_id,
            }

            document_spans.append(span)
            spans_by_node_id[node["node_id"]] = span

        spans_by_document[document_id] = document_spans

    return spans_by_document, spans_by_node_id


def validate_source_containment(
    spans_by_document: dict[str, list[dict[str, Any]]],
    spans_by_node_id: dict[str, dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> None:
    """Validate source headings against all source ancestors."""
    for document_spans in spans_by_document.values():
        for span in document_spans:
            node = span["node"]

            for ancestor_node_id in node["ancestor_node_ids"]:
                ancestor = nodes_by_id[ancestor_node_id]

                if ancestor["node_kind"] != "source":
                    continue

                ancestor_span = spans_by_node_id[ancestor_node_id]

                if not (
                    span["heading_start_index"] >= ancestor_span["subtree_start_index"]
                    and span["heading_end_index"] < ancestor_span["subtree_stop_index"]
                ):
                    raise HeadingSpanError(
                        f"{node['node_id']}: source heading lies outside "
                        f"source ancestor {ancestor_node_id}."
                    )

                if span["subtree_stop_index"] > ancestor_span["subtree_stop_index"]:
                    raise HeadingSpanError(
                        f"{node['node_id']}: subtree extends beyond "
                        f"source ancestor {ancestor_node_id}."
                    )


def coordinate_record(
    document_id: str,
    coordinate: tuple[int, int],
    document_indexes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert an internal coordinate into the production record shape."""
    page_number, line_number = coordinate

    return {
        "page_id": document_indexes[document_id]["page_ids"][page_number],
        "page_number": page_number,
        "line_number": line_number,
    }


def optional_coordinate_record(
    document_id: str,
    coordinate: tuple[int, int] | None,
    document_indexes: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Convert an optional internal source coordinate."""
    if coordinate is None:
        return None

    return coordinate_record(
        document_id,
        coordinate,
        document_indexes,
    )


def build_span_metrics(
    *,
    document_id: str,
    start_index: int,
    stop_index: int,
    document_indexes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate coordinate and line-count measurements for a span."""
    document_index = document_indexes[document_id]
    coordinates = document_index["coordinates"][start_index:stop_index]
    line_text = document_index["line_text"]

    nonblank_line_count = sum(bool(line_text[coordinate].strip()) for coordinate in coordinates)

    return {
        "included_start": optional_coordinate_record(
            document_id,
            coordinates[0] if coordinates else None,
            document_indexes,
        ),
        "included_end": optional_coordinate_record(
            document_id,
            coordinates[-1] if coordinates else None,
            document_indexes,
        ),
        "raw_line_count": len(coordinates),
        "nonblank_line_count": nonblank_line_count,
        "is_empty": not coordinates,
        "is_blank_only": (bool(coordinates) and nonblank_line_count == 0),
        "is_multi_page": (bool(coordinates) and coordinates[0][0] != coordinates[-1][0]),
    }


def build_end_boundary(
    *,
    document_id: str,
    stop_index: int,
    stop_heading_id: str | None,
    document_indexes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an exclusive end boundary for a source span."""
    coordinates = document_indexes[document_id]["coordinates"]

    if stop_index == len(coordinates):
        return {
            "kind": "after_document_line",
            "heading_id": None,
            "coordinate": coordinate_record(
                document_id,
                coordinates[-1],
                document_indexes,
            ),
        }

    if stop_heading_id is None:
        raise HeadingSpanError(f"{document_id}: non-EOF span has no stop heading.")

    return {
        "kind": "before_source_heading",
        "heading_id": stop_heading_id,
        "coordinate": coordinate_record(
            document_id,
            coordinates[stop_index],
            document_indexes,
        ),
    }


def build_heading_spans(
    pages: list[dict[str, Any]],
    headings: list[dict[str, Any]],
    hierarchy_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one coordinate-only span record per hierarchy node."""
    document_indexes = build_document_indexes(pages)
    nodes_by_id, source_nodes_by_heading_id = index_hierarchy(
        hierarchy_nodes,
        document_indexes,
    )
    spans_by_document, spans_by_node_id = position_source_headings(
        headings,
        source_nodes_by_heading_id,
        document_indexes,
    )

    validate_source_containment(
        spans_by_document,
        spans_by_node_id,
        nodes_by_id,
    )

    records: list[dict[str, Any]] = []

    for node in hierarchy_nodes:
        document_id = node["document_id"]

        base_record = {
            "schema_version": "1.0",
            "node_id": node["node_id"],
            "document_id": document_id,
            "node_kind": node["node_kind"],
            "heading_id": node["heading_id"],
            "parent_node_id": node["parent_node_id"],
            "depth": node["depth"],
            "document_order": node["document_order"],
        }

        if node["node_kind"] == "source":
            span = spans_by_node_id[node["node_id"]]
            heading = span["heading"]

            direct_body = {
                "basis": "after_source_heading",
                "start_boundary": {
                    "kind": "after_source_line",
                    "heading_id": heading["heading_id"],
                    "coordinate": coordinate_record(
                        document_id,
                        span["heading_end_coordinate"],
                        document_indexes,
                    ),
                },
                "end_boundary": build_end_boundary(
                    document_id=document_id,
                    stop_index=span["direct_stop_index"],
                    stop_heading_id=span["direct_stop_heading_id"],
                    document_indexes=document_indexes,
                ),
                **build_span_metrics(
                    document_id=document_id,
                    start_index=span["direct_start_index"],
                    stop_index=span["direct_stop_index"],
                    document_indexes=document_indexes,
                ),
                "includes_own_heading_lines": False,
            }

            subtree = {
                "basis": "after_source_heading",
                "start_boundary": {
                    "kind": "after_source_line",
                    "heading_id": heading["heading_id"],
                    "coordinate": coordinate_record(
                        document_id,
                        span["heading_end_coordinate"],
                        document_indexes,
                    ),
                },
                "end_boundary": build_end_boundary(
                    document_id=document_id,
                    stop_index=span["subtree_stop_index"],
                    stop_heading_id=span["subtree_stop_heading_id"],
                    document_indexes=document_indexes,
                ),
                **build_span_metrics(
                    document_id=document_id,
                    start_index=span["subtree_start_index"],
                    stop_index=span["subtree_stop_index"],
                    document_indexes=document_indexes,
                ),
                "includes_own_heading_lines": False,
                "includes_descendant_heading_lines": True,
            }

            records.append(
                {
                    **base_record,
                    "heading_source": {
                        "start": coordinate_record(
                            document_id,
                            span["heading_start_coordinate"],
                            document_indexes,
                        ),
                        "end": coordinate_record(
                            document_id,
                            span["heading_end_coordinate"],
                            document_indexes,
                        ),
                        "source_line_numbers": heading["source_line_numbers"],
                    },
                    "direct_body": direct_body,
                    "subtree": subtree,
                    "source_descendant_envelope": None,
                }
            )
            continue

        document_spans = spans_by_document[document_id]
        descendants = [
            span for span in document_spans if node["node_id"] in span["node"]["ancestor_node_ids"]
        ]

        if not descendants:
            raise HeadingSpanError(f"{node['node_id']}: synthetic node has no source descendants.")

        descendant_positions = [document_spans.index(descendant) for descendant in descendants]
        expected_positions = list(
            range(
                min(descendant_positions),
                max(descendant_positions) + 1,
            )
        )

        if descendant_positions != expected_positions:
            raise HeadingSpanError(f"{node['node_id']}: source descendants are not contiguous.")

        descendants_by_heading_id = {
            descendant["heading"]["heading_id"]: descendant for descendant in descendants
        }
        anchor_heading_id = node["anchor_heading_id"]
        anchor_span = descendants_by_heading_id.get(anchor_heading_id)

        if anchor_span is None:
            raise HeadingSpanError(f"{node['node_id']}: anchor is not a source descendant.")

        expected_anchor = (
            anchor_span["heading"]["page_id"],
            anchor_span["heading"]["page_number"],
            anchor_span["heading"]["start_line_number"],
        )
        actual_anchor = (
            node["anchor_page_id"],
            node["anchor_page_number"],
            node["anchor_line_number"],
        )

        if actual_anchor != expected_anchor:
            raise HeadingSpanError(
                f"{node['node_id']}: anchor coordinates do not match the source heading."
            )

        first_descendant = descendants[0]
        final_descendant = max(
            descendants,
            key=lambda descendant: (
                descendant["subtree_stop_index"],
                descendant["heading_start_index"],
            ),
        )
        envelope_start_index = first_descendant["heading_start_index"]
        envelope_stop_index = final_descendant["subtree_stop_index"]

        parent_node_id = node["parent_node_id"]

        if parent_node_id is not None:
            parent = nodes_by_id[parent_node_id]

            if parent["node_kind"] == "source":
                parent_span = spans_by_node_id[parent_node_id]

                if (
                    envelope_start_index < parent_span["subtree_start_index"]
                    or envelope_stop_index > parent_span["subtree_stop_index"]
                ):
                    raise HeadingSpanError(
                        f"{node['node_id']}: descendant envelope lies "
                        "outside its source parent subtree."
                    )

        records.append(
            {
                **base_record,
                "heading_source": None,
                "direct_body": None,
                "subtree": None,
                "source_descendant_envelope": {
                    "basis": "source_descendant_envelope",
                    "start_boundary": {
                        "kind": "at_source_heading",
                        "heading_id": first_descendant["heading"]["heading_id"],
                        "coordinate": coordinate_record(
                            document_id,
                            first_descendant["heading_start_coordinate"],
                            document_indexes,
                        ),
                    },
                    "end_boundary": build_end_boundary(
                        document_id=document_id,
                        stop_index=envelope_stop_index,
                        stop_heading_id=final_descendant["subtree_stop_heading_id"],
                        document_indexes=document_indexes,
                    ),
                    **build_span_metrics(
                        document_id=document_id,
                        start_index=envelope_start_index,
                        stop_index=envelope_stop_index,
                        document_indexes=document_indexes,
                    ),
                    "includes_descendant_heading_lines": True,
                    "source_descendant_heading_count": len(descendants),
                    "anchor_heading_id": anchor_heading_id,
                },
            }
        )

    if [record["node_id"] for record in records] != [node["node_id"] for node in hierarchy_nodes]:
        raise HeadingSpanError("Output order does not match hierarchy input order.")

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write span records atomically without replacing existing output."""
    if path.exists():
        raise HeadingSpanError(f"Output already exists: {path}")

    if not records:
        raise HeadingSpanError("No heading span records to write.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            for record in records:
                json.dump(
                    record,
                    temporary_file,
                    ensure_ascii=False,
                )
                temporary_file.write("\n")

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    """Command-line entry point for deterministic heading spans."""
    parser = argparse.ArgumentParser(
        description=("Assign direct-body, subtree, and synthetic descendant envelope coordinates.")
    )
    parser.add_argument(
        "pages",
        type=Path,
        help="Path to page-level JSONL.",
    )
    parser.add_argument(
        "headings",
        type=Path,
        help="Path to reconstructed-heading JSONL.",
    )
    parser.add_argument(
        "hierarchy",
        type=Path,
        help="Path to heading-hierarchy JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/heading-spans.jsonl"),
        help="Destination heading-span JSONL.",
    )

    args = parser.parse_args()

    try:
        pages = load_jsonl(
            args.pages,
            record_name="Page",
        )
        headings = load_jsonl(
            args.headings,
            record_name="Reconstructed heading",
        )
        hierarchy_nodes = load_jsonl(
            args.hierarchy,
            record_name="Heading hierarchy",
        )
        records = build_heading_spans(
            pages,
            headings,
            hierarchy_nodes,
        )
        write_jsonl(args.output, records)
    except HeadingSpanError as error:
        print(f"Heading span assignment failed: {error}")
        return 1

    counts = Counter(record["node_kind"] for record in records)
    empty_direct_bodies = sum(
        record["node_kind"] == "source" and record["direct_body"]["is_empty"] for record in records
    )

    print(f"Heading span assignment complete: {len(records)} records")
    print(f"- Source records: {counts['source']}")
    print(f"- Synthetic records: {counts['synthetic']}")
    print(f"- Empty direct bodies: {empty_direct_bodies}")
    print(f"Spans written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
