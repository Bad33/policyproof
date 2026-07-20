"""Deterministic text materialization for validated retrieval units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from policyproof.retrieval_policy import logical_source_label
from policyproof.retrieval_units import (
    Coordinate,
    CorpusIndexes,
    logical_source_key,
    segment_coordinates,
    unit_coordinates,
)


@dataclass(frozen=True)
class RetrievalTextMaterialization:
    """Distinct source, retrieval, model, and citation text representations."""

    unit_id: str
    logical_source_key: str
    label: str
    raw_source_segments: tuple[str, ...]
    retrieval_body: str
    model_text: str
    citation_text: str


@dataclass(frozen=True)
class _ExpandedSegment:
    """One validated source segment with exact extracted lines."""

    coordinates: tuple[Coordinate, ...]
    lines: tuple[str, ...]


def _expanded_segments(
    unit: Mapping[str, Any],
    indexes: CorpusIndexes,
) -> tuple[_ExpandedSegment, ...]:
    """Expand and validate all serialized segments for one unit."""

    unit_coordinates(unit, indexes)

    document_id = unit["document_id"]
    index = indexes.by_document[document_id]
    expanded: list[_ExpandedSegment] = []

    for segment in unit["content_segments"]:
        coordinates = segment_coordinates(
            document_id,
            segment,
            indexes,
        )
        expanded.append(
            _ExpandedSegment(
                coordinates=coordinates,
                lines=tuple(
                    index.line_text[coordinate]
                    for coordinate in coordinates
                ),
            )
        )

    return tuple(expanded)


def _segment_separator(
    left: _ExpandedSegment,
    right: _ExpandedSegment,
) -> str:
    """Preserve same-page separation without inventing cross-page paragraphs."""

    if left.coordinates[-1][0] == right.coordinates[0][0]:
        return "\n\n"

    return "\n"


def _join_segments(
    texts: Sequence[str],
    segments: Sequence[_ExpandedSegment],
) -> str:
    """Join aligned segment texts using reviewed gap semantics."""

    parts: list[str] = []

    for index, text in enumerate(texts):
        if index:
            parts.append(
                _segment_separator(
                    segments[index - 1],
                    segments[index],
                )
            )

        parts.append(text)

    return "".join(parts)


def materialize_retrieval_text(
    unit: Mapping[str, Any],
    hierarchy_by_id: Mapping[str, Mapping[str, Any]],
    indexes: CorpusIndexes,
) -> RetrievalTextMaterialization:
    """Materialize one validated retrieval unit without changing provenance."""

    segments = _expanded_segments(
        unit,
        indexes,
    )
    label = logical_source_label(
        unit,
        hierarchy_by_id,
    )

    raw_source_segments = tuple(
        "\n".join(segment.lines)
        for segment in segments
    )

    normalized_segments = tuple(
        "\n".join(
            line.strip()
            for line in segment.lines
        )
        for segment in segments
    )

    if unit["unit_kind"] == "heading_only":
        retrieval_body = label
        model_text = label
    else:
        retrieval_body = _join_segments(
            normalized_segments,
            segments,
        )
        model_text = f"{label}\n\n{retrieval_body}"

    return RetrievalTextMaterialization(
        unit_id=unit["unit_id"],
        logical_source_key=logical_source_key(unit),
        label=label,
        raw_source_segments=raw_source_segments,
        retrieval_body=retrieval_body,
        model_text=model_text,
        citation_text=retrieval_body,
    )
