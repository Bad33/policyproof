"""Deterministic coordinate utilities for retrieval-unit construction."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, TypeAlias

Coordinate: TypeAlias = tuple[int, int]
LogicalSourceKey: TypeAlias = tuple[str, str]

WORD_PATTERN = re.compile(r"\S+")
PART_SUFFIX_PATTERN = re.compile(
    r"^(?P<base>.+):part-(?P<part>[0-9]{3})$"
)

REQUIRED_PAGE_FIELDS = frozenset(
    {
        "page_id",
        "document_id",
        "page_number",
        "text",
    }
)


class RetrievalUnitError(RuntimeError):
    """Raised when retrieval-unit coordinates or outputs are unsafe."""


@dataclass(frozen=True)
class DocumentIndex:
    """Coordinate index for one extracted document."""

    document_id: str
    page_ids: Mapping[int, str]
    coordinates: tuple[Coordinate, ...]
    coordinate_indexes: Mapping[Coordinate, int]
    line_text: Mapping[Coordinate, str]
    page_line_counts: Mapping[int, int]


@dataclass(frozen=True)
class CorpusIndexes:
    """Ordered coordinate indexes for the controlled corpus."""

    document_order: tuple[str, ...]
    by_document: Mapping[str, DocumentIndex]
    total_line_count: int


@dataclass(frozen=True)
class LogicalSourceGroups:
    """Ordered retrieval units grouped by validated logical-source identity."""

    ordered_keys: tuple[LogicalSourceKey, ...]
    groups: Mapping[LogicalSourceKey, tuple[dict[str, Any], ...]]


def require_fields(
    record: Mapping[str, Any],
    required_fields: Iterable[str],
    *,
    record_name: str,
) -> None:
    """Require fields needed by a coordinate-only production record."""

    missing = sorted(
        field
        for field in required_fields
        if field not in record
    )

    if missing:
        raise RetrievalUnitError(
            f"{record_name} is missing required fields: {missing}"
        )


def normalize(text: str) -> str:
    """Collapse whitespace for exact structural comparisons."""

    return " ".join(text.split())


def count_words(text: str) -> int:
    """Count provisional whitespace-delimited words."""

    return len(WORD_PATTERN.findall(text))


def load_jsonl(
    path: Path,
    *,
    record_name: str,
) -> list[dict[str, Any]]:
    """Load JSON objects from a JSONL file and fail closed."""

    records: list[dict[str, Any]] = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RetrievalUnitError(
            f"Could not read {record_name} file {path}: {error}"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RetrievalUnitError(
                f"{path}: invalid JSON at line {line_number}: "
                f"{error.msg}"
            ) from error

        if not isinstance(value, dict):
            raise RetrievalUnitError(
                f"{path}: {record_name} line {line_number} "
                "is not a JSON object."
            )

        records.append(value)

    return records


def build_document_indexes(
    pages: Sequence[Mapping[str, Any]],
) -> CorpusIndexes:
    """Build deterministic document, page, and line-coordinate indexes."""

    if not pages:
        raise RetrievalUnitError(
            "At least one page record is required."
        )

    document_order: list[str] = []
    pages_by_document: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    seen_page_ids: set[str] = set()

    for record_number, page in enumerate(pages, start=1):
        record_name = f"Page record {record_number}"
        require_fields(
            page,
            REQUIRED_PAGE_FIELDS,
            record_name=record_name,
        )

        page_id = page["page_id"]
        document_id = page["document_id"]
        page_number = page["page_number"]
        text = page["text"]

        if not isinstance(page_id, str) or not page_id:
            raise RetrievalUnitError(
                f"{record_name} has an invalid page_id."
            )

        if (
            not isinstance(document_id, str)
            or not document_id
        ):
            raise RetrievalUnitError(
                f"{record_name} has an invalid document_id."
            )

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise RetrievalUnitError(
                f"{record_name} has an invalid page_number."
            )

        if not isinstance(text, str):
            raise RetrievalUnitError(
                f"{record_name} has non-string text."
            )

        expected_page_id = (
            f"{document_id}:page-{page_number:04d}"
        )

        if page_id != expected_page_id:
            raise RetrievalUnitError(
                f"{record_name} page_id {page_id!r} does not "
                f"match {expected_page_id!r}."
            )

        if page_id in seen_page_ids:
            raise RetrievalUnitError(
                f"Duplicate page_id: {page_id}"
            )

        seen_page_ids.add(page_id)

        if document_id not in pages_by_document:
            document_order.append(document_id)

        pages_by_document[document_id].append(page)

    indexes: dict[str, DocumentIndex] = {}
    total_line_count = 0

    for document_id in document_order:
        document_pages = sorted(
            pages_by_document[document_id],
            key=lambda page: page["page_number"],
        )
        actual_page_numbers = [
            page["page_number"]
            for page in document_pages
        ]
        expected_page_numbers = list(
            range(1, len(document_pages) + 1)
        )

        if actual_page_numbers != expected_page_numbers:
            raise RetrievalUnitError(
                f"{document_id}: page numbers must be "
                f"contiguous from 1; found {actual_page_numbers}."
            )

        declared_page_counts = {
            page["page_count"]
            for page in document_pages
            if "page_count" in page
        }

        if declared_page_counts and declared_page_counts != {
            len(document_pages)
        }:
            raise RetrievalUnitError(
                f"{document_id}: page_count values do not "
                f"match {len(document_pages)} pages."
            )

        page_ids: dict[int, str] = {}
        coordinates: list[Coordinate] = []
        coordinate_indexes: dict[Coordinate, int] = {}
        line_text: dict[Coordinate, str] = {}
        page_line_counts: dict[int, int] = {}

        for page in document_pages:
            page_number = page["page_number"]
            page_ids[page_number] = page["page_id"]
            lines = page["text"].splitlines()
            page_line_counts[page_number] = len(lines)

            for line_number, text in enumerate(
                lines,
                start=1,
            ):
                coordinate = (
                    page_number,
                    line_number,
                )
                coordinate_indexes[coordinate] = len(
                    coordinates
                )
                coordinates.append(coordinate)
                line_text[coordinate] = text

        total_line_count += len(coordinates)
        indexes[document_id] = DocumentIndex(
            document_id=document_id,
            page_ids=page_ids,
            coordinates=tuple(coordinates),
            coordinate_indexes=coordinate_indexes,
            line_text=line_text,
            page_line_counts=page_line_counts,
        )

    return CorpusIndexes(
        document_order=tuple(document_order),
        by_document=indexes,
        total_line_count=total_line_count,
    )


def coordinate_from_record(
    record: Mapping[str, Any],
    *,
    record_name: str = "Coordinate",
    index: DocumentIndex | None = None,
) -> Coordinate:
    """Validate and convert a serialized source coordinate."""

    require_fields(
        record,
        {
            "page_id",
            "page_number",
            "line_number",
        },
        record_name=record_name,
    )

    page_id = record["page_id"]
    page_number = record["page_number"]
    line_number = record["line_number"]

    if not isinstance(page_id, str) or not page_id:
        raise RetrievalUnitError(
            f"{record_name} has an invalid page_id."
        )

    for field_name, value in (
        ("page_number", page_number),
        ("line_number", line_number),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise RetrievalUnitError(
                f"{record_name} has an invalid {field_name}."
            )

    coordinate = (page_number, line_number)

    if index is not None:
        expected_page_id = index.page_ids.get(page_number)

        if expected_page_id != page_id:
            raise RetrievalUnitError(
                f"{record_name} page_id {page_id!r} does not "
                f"match document {index.document_id!r}."
            )

        if coordinate not in index.coordinate_indexes:
            raise RetrievalUnitError(
                f"{record_name} coordinate {coordinate} "
                f"does not exist in {index.document_id}."
            )

    return coordinate


def coordinate_record(
    index: DocumentIndex,
    coordinate: Coordinate,
) -> dict[str, Any]:
    """Serialize one validated coordinate."""

    if coordinate not in index.coordinate_indexes:
        raise RetrievalUnitError(
            f"{index.document_id}: unknown coordinate "
            f"{coordinate}."
        )

    page_number, line_number = coordinate

    return {
        "page_id": index.page_ids[page_number],
        "page_number": page_number,
        "line_number": line_number,
    }


def document_index(
    indexes: CorpusIndexes,
    document_id: str,
) -> DocumentIndex:
    """Return one document index or fail closed."""

    try:
        return indexes.by_document[document_id]
    except KeyError as error:
        raise RetrievalUnitError(
            f"Unknown document_id: {document_id}"
        ) from error


def segment_coordinates(
    document_id: str,
    segment: Mapping[str, Any],
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Expand one inclusive serialized segment."""

    require_fields(
        segment,
        {
            "included_start",
            "included_end",
        },
        record_name="Content segment",
    )

    index = document_index(indexes, document_id)
    start = coordinate_from_record(
        segment["included_start"],
        record_name="Content segment start",
        index=index,
    )
    end = coordinate_from_record(
        segment["included_end"],
        record_name="Content segment end",
        index=index,
    )
    start_index = index.coordinate_indexes[start]
    end_index = index.coordinate_indexes[end]

    if end_index < start_index:
        raise RetrievalUnitError(
            f"{document_id}: inverted content segment "
            f"{start} to {end}."
        )

    return index.coordinates[
        start_index:
        end_index + 1
    ]


def unit_coordinates(
    unit: Mapping[str, Any],
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Expand and validate every coordinate owned by one unit."""

    require_fields(
        unit,
        {
            "unit_id",
            "document_id",
            "content_segments",
        },
        record_name="Retrieval unit",
    )

    unit_id = unit["unit_id"]
    document_id = unit["document_id"]
    segments = unit["content_segments"]

    if not isinstance(unit_id, str) or not unit_id:
        raise RetrievalUnitError(
            "Retrieval unit has an invalid unit_id."
        )

    if (
        not isinstance(document_id, str)
        or not document_id
    ):
        raise RetrievalUnitError(
            f"{unit_id}: invalid document_id."
        )

    if not isinstance(segments, list) or not segments:
        raise RetrievalUnitError(
            f"{unit_id}: content_segments must be "
            "a nonempty list."
        )

    index = document_index(indexes, document_id)
    coordinates: list[Coordinate] = []

    for segment in segments:
        if not isinstance(segment, Mapping):
            raise RetrievalUnitError(
                f"{unit_id}: content segment is not "
                "an object."
            )

        coordinates.extend(
            segment_coordinates(
                document_id,
                segment,
                indexes,
            )
        )

    if not coordinates:
        raise RetrievalUnitError(
            f"{unit_id}: no content coordinates."
        )

    if len(set(coordinates)) != len(coordinates):
        raise RetrievalUnitError(
            f"{unit_id}: content segments overlap."
        )

    positions = [
        index.coordinate_indexes[coordinate]
        for coordinate in coordinates
    ]

    if any(
        right <= left
        for left, right in zip(
            positions,
            positions[1:],
        )
    ):
        raise RetrievalUnitError(
            f"{unit_id}: content coordinates are "
            "not in source order."
        )

    declared_count = unit.get(
        "content_coordinate_count"
    )

    if (
        declared_count is not None
        and declared_count != len(coordinates)
    ):
        raise RetrievalUnitError(
            f"{unit_id}: content_coordinate_count "
            f"is {declared_count}, expected "
            f"{len(coordinates)}."
        )

    return tuple(coordinates)


def contiguous_segments(
    document_id: str,
    coordinates: Sequence[Coordinate],
    indexes: CorpusIndexes,
) -> tuple[tuple[Coordinate, ...], ...]:
    """Group ordered coordinates into contiguous source segments."""

    if not coordinates:
        return ()

    index = document_index(indexes, document_id)

    try:
        positions = [
            index.coordinate_indexes[coordinate]
            for coordinate in coordinates
        ]
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown content coordinate "
            f"{error.args[0]}."
        ) from error

    if len(set(coordinates)) != len(coordinates):
        raise RetrievalUnitError(
            f"{document_id}: duplicate content coordinate."
        )

    if any(
        right <= left
        for left, right in zip(
            positions,
            positions[1:],
        )
    ):
        raise RetrievalUnitError(
            f"{document_id}: coordinates are not in "
            "strict source order."
        )

    segments: list[list[Coordinate]] = [
        [coordinates[0]]
    ]

    for coordinate, position in zip(
        coordinates[1:],
        positions[1:],
    ):
        previous_position = index.coordinate_indexes[
            segments[-1][-1]
        ]

        if position == previous_position + 1:
            segments[-1].append(coordinate)
        else:
            segments.append([coordinate])

    return tuple(
        tuple(segment)
        for segment in segments
    )


def content_segments(
    document_id: str,
    coordinates: Sequence[Coordinate],
    indexes: CorpusIndexes,
) -> list[dict[str, Any]]:
    """Serialize ordered coordinates as inclusive source segments."""

    index = document_index(indexes, document_id)

    return [
        {
            "included_start": coordinate_record(
                index,
                segment[0],
            ),
            "included_end": coordinate_record(
                index,
                segment[-1],
            ),
        }
        for segment in contiguous_segments(
            document_id,
            coordinates,
            indexes,
        )
    ]


def logical_source_key(
    unit: Mapping[str, Any],
) -> str:
    """Derive the validated logical-source key from unit_id."""

    unit_id = unit.get("unit_id")

    if not isinstance(unit_id, str) or not unit_id:
        raise RetrievalUnitError(
            "Retrieval unit has no valid unit_id."
        )

    match = PART_SUFFIX_PATTERN.fullmatch(unit_id)

    if match is None:
        raise RetrievalUnitError(
            f"{unit_id}: unit_id must end with "
            "':part-NNN'."
        )

    part_number = int(match.group("part"))

    if part_number < 1:
        raise RetrievalUnitError(
            f"{unit_id}: part number must be positive."
        )

    declared_part = unit.get("part_number")

    if (
        declared_part is not None
        and declared_part != part_number
    ):
        raise RetrievalUnitError(
            f"{unit_id}: part_number {declared_part} "
            f"does not match suffix {part_number}."
        )

    return match.group("base")


def group_units_by_logical_source(
    units: Sequence[dict[str, Any]],
) -> LogicalSourceGroups:
    """Group units while preserving first logical-source occurrence."""

    ordered_keys: list[LogicalSourceKey] = []
    mutable_groups: dict[
        LogicalSourceKey,
        list[dict[str, Any]],
    ] = defaultdict(list)
    seen_unit_ids: set[str] = set()

    for record_number, unit in enumerate(
        units,
        start=1,
    ):
        require_fields(
            unit,
            {
                "unit_id",
                "unit_kind",
                "document_id",
                "part_number",
                "part_count",
            },
            record_name=(
                f"Retrieval unit {record_number}"
            ),
        )

        unit_id = unit["unit_id"]
        document_id = unit["document_id"]
        unit_kind = unit["unit_kind"]
        part_number = unit["part_number"]
        part_count = unit["part_count"]

        if not isinstance(unit_id, str) or not unit_id:
            raise RetrievalUnitError(
                f"Retrieval unit {record_number} has "
                "an invalid unit_id."
            )

        if unit_id in seen_unit_ids:
            raise RetrievalUnitError(
                f"Duplicate unit_id: {unit_id}"
            )

        seen_unit_ids.add(unit_id)

        if (
            not isinstance(document_id, str)
            or not document_id
        ):
            raise RetrievalUnitError(
                f"{unit_id}: invalid document_id."
            )

        if (
            not isinstance(unit_kind, str)
            or not unit_kind
        ):
            raise RetrievalUnitError(
                f"{unit_id}: invalid unit_kind."
            )

        for field_name, value in (
            ("part_number", part_number),
            ("part_count", part_count),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise RetrievalUnitError(
                    f"{unit_id}: invalid {field_name}."
                )

        base_id = logical_source_key(unit)
        key = (document_id, base_id)

        if key not in mutable_groups:
            ordered_keys.append(key)

        mutable_groups[key].append(unit)

    groups: dict[
        LogicalSourceKey,
        tuple[dict[str, Any], ...],
    ] = {}

    for key in ordered_keys:
        members = sorted(
            mutable_groups[key],
            key=lambda unit: unit["part_number"],
        )
        observed_parts = [
            unit["part_number"]
            for unit in members
        ]
        expected_parts = list(
            range(1, len(members) + 1)
        )

        if observed_parts != expected_parts:
            raise RetrievalUnitError(
                f"{key[1]}: parts must be contiguous "
                f"from 1; found {observed_parts}."
            )

        declared_counts = {
            unit["part_count"]
            for unit in members
        }

        if declared_counts != {len(members)}:
            raise RetrievalUnitError(
                f"{key[1]}: part_count values do not "
                f"match {len(members)} parts."
            )

        kinds = {
            unit["unit_kind"]
            for unit in members
        }

        if len(kinds) != 1:
            raise RetrievalUnitError(
                f"{key[1]}: logical source mixes "
                f"unit kinds {sorted(kinds)}."
            )

        documents = {
            unit["document_id"]
            for unit in members
        }

        if documents != {key[0]}:
            raise RetrievalUnitError(
                f"{key[1]}: logical source crosses "
                "documents."
            )

        groups[key] = tuple(members)

    return LogicalSourceGroups(
        ordered_keys=tuple(ordered_keys),
        groups=groups,
    )


def _write_atomically(
    path: Path,
    writer: Callable[[TextIO], None],
) -> None:
    """Write through a same-directory temporary file without overwriting."""

    if path.exists():
        raise RetrievalUnitError(
            f"Output already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())

        try:
            os.link(
                temporary_path,
                path,
            )
        except FileExistsError as error:
            raise RetrievalUnitError(
                f"Output already exists: {path}"
            ) from error
        except OSError as error:
            raise RetrievalUnitError(
                f"Could not publish output {path}: "
                f"{error}"
            ) from error
    finally:
        temporary_path.unlink(
            missing_ok=True
        )


def write_jsonl_atomically(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Write JSONL records atomically without replacing existing output."""

    def write(file: TextIO) -> None:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    _write_atomically(path, write)


def write_json_atomically(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    """Write formatted JSON atomically without replacing existing output."""

    def write(file: TextIO) -> None:
        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    _write_atomically(path, write)


def write_text_atomically(
    path: Path,
    text: str,
) -> None:
    """Write UTF-8 text atomically without replacing existing output."""

    def write(file: TextIO) -> None:
        file.write(text)

    _write_atomically(path, write)
