"""Build the controlled coordinate-only retrieval corpus."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policyproof.retrieval_policy import (
    EU_ID,
    GENAI_ID,
    GPT_ID,
    LOWERCASE_START_PATTERN,
    RECITAL_29_HARD_WORDS,
    REFERENCE_HEADING_PATTERN,
    RMF_ID,
    STANDARD_HARD_WORDS,
    STRONG_TERMINAL_PATTERN,
    URL_PATTERN,
    accepted_structured_start,
    cleaned_coordinates,
    discover_genai_footnotes,
    explicit_furniture_reason,
    gap_is_blank_only,
    is_eu_eli,
    logical_source_label,
    reference_entry_groups,
    reference_start_is_valid,
    repack_nonreference,
)
from policyproof.retrieval_units import (
    Coordinate,
    CorpusIndexes,
    RetrievalUnitError,
    build_document_indexes,
    content_segments,
    coordinate_from_record,
    coordinate_record,
    count_words,
    document_index,
    group_units_by_logical_source,
    load_jsonl,
    normalize,
    write_json_atomically,
    write_jsonl_atomically,
    write_text_atomically,
)

UNIT_SCHEMA_VERSION = "1.0"
LEDGER_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"

EXPECTED_DOCUMENT_ORDER = (
    RMF_ID,
    GENAI_ID,
    EU_ID,
    GPT_ID,
)
EXPECTED_PAGE_COUNT = 289
EXPECTED_HIERARCHY_COUNT = 380
EXPECTED_SPAN_COUNT = 380
EXPECTED_TOTAL_LINE_COUNT = 12_008
EXPECTED_SOURCE_SPAN_COUNT = 347
EXPECTED_SYNTHETIC_SPAN_COUNT = 33
EXPECTED_LOGICAL_SOURCE_COUNT = 485
EXPECTED_UNIT_COUNT = 579
EXPECTED_INTERNAL_BOUNDARY_COUNT = 94

RECITAL_PATTERN = re.compile(
    r"^\((\d{1,3})\)\s"
)
REFERENCE_PREFIXES = tuple(
    value.casefold()
    for value in (
        "OJ ",
        "Position of the European Parliament",
        "European Council",
        "European Parliament resolution",
        "Regulation (",
        "Directive ",
        "Council Directive",
        "Council Framework Decision",
        "Council Regulation",
        "Council Decision",
        "Commission Recommendation",
        "Commission Decision",
        "Commission Delegated",
        "Commission Implementing",
        "Decision No ",
    )
)

PAGE_FURNITURE_REASONS = frozenset(
    {
        "eu_eli_footer",
        "eu_journal_header",
        "eu_language_header",
        "genai_page_number",
        "gpt_page_number",
        "rmf_page_label",
        "rmf_running_header",
        "rmf_table_continuation",
        "rmf_table_header",
    }
)

EXPECTED_UNIT_KIND_COUNTS = {
    "eu_recital": 181,
    "frontmatter_body": 3,
    "heading_body": 342,
    "heading_only": 53,
}

EXPECTED_DOCUMENT_UNIT_COUNTS = {
    EU_ID: 363,
    GENAI_ID: 76,
    GPT_ID: 34,
    RMF_ID: 106,
}

EXPECTED_BOUNDARY_COUNTS = {
    "after_strong_terminal": 59,
    "before_reference_entry": 8,
    "before_structured_start": 27,
    "end_of_heading_source": 53,
    "end_of_source_unit": 432,
}

EXPECTED_CLASSIFICATION_COUNTS = {
    "excluded_blank_line": 146,
    "excluded_frontmatter_metadata": 209,
    "excluded_page_furniture": 555,
    "excluded_reference_region": 144,
    "excluded_structural_metadata": 311,
    "heading_context": 622,
    "retrieval_content": 10_021,
}

EXPECTED_REASON_COUNTS = {
    "blank_source_line": 146,
    "document_frontmatter_metadata": 209,
    "eu_eli_footer": 72,
    "eu_enacting_formula": 1,
    "eu_journal_header": 143,
    "eu_language_header": 143,
    "eu_numbered_reference_region": 144,
    "genai_page_number": 60,
    "gpt_authorship_and_credits": 308,
    "gpt_page_number": 33,
    "rmf_page_label": 43,
    "rmf_part_1_label": 1,
    "rmf_part_2_label": 1,
    "rmf_running_header": 43,
    "rmf_table_continuation": 7,
    "rmf_table_header": 11,
}

KNOWN_REJECTED_BOUNDARIES = frozenset(
    {
        (
            RMF_ID,
            (15, 6),
            (15, 7),
        ),
        (
            GENAI_ID,
            (7, 37),
            (7, 38),
        ),
        (
            GENAI_ID,
            (13, 35),
            (13, 36),
        ),
        (
            GENAI_ID,
            (53, 1),
            (53, 3),
        ),
        (
            GENAI_ID,
            (61, 3),
            (61, 4),
        ),
        (
            GENAI_ID,
            (62, 12),
            (62, 13),
        ),
        (
            GENAI_ID,
            (63, 31),
            (63, 32),
        ),
        (
            EU_ID,
            (49, 39),
            (49, 40),
        ),
        (
            EU_ID,
            (78, 29),
            (78, 30),
        ),
        (
            GPT_ID,
            (30, 36),
            (30, 37),
        ),
    }
)


@dataclass(frozen=True)
class ExclusionPlan:
    """Reviewed exclusions and EU recital anchors."""

    reasons: Mapping[
        tuple[str, Coordinate],
        str,
    ]
    selected_recitals: tuple[
        tuple[int, Coordinate],
        ...,
    ]
    first_heading_coordinates: Mapping[
        str,
        Coordinate,
    ]


@dataclass(frozen=True)
class LogicalSource:
    """One unsplit semantic source before final packing."""

    document_id: str
    unit_kind: str
    source_key: str
    coordinates: tuple[Coordinate, ...]
    context_word_count: int
    context_node_ids: tuple[str, ...]
    structural_context_ids: tuple[str, ...]
    heading_source: dict[str, Any] | None
    metadata: Mapping[str, Any]
    hard_word_limit: int


@dataclass(frozen=True)
class RetrievalBuildResult:
    """Complete coordinate-only retrieval build."""

    units: tuple[dict[str, Any], ...]
    ledger: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    report: str


def source_records(
    document_id: str,
    spans: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return source span records for one document."""

    return tuple(
        record
        for record in spans
        if (
            record.get("document_id")
            == document_id
            and record.get("node_kind")
            == "source"
        )
    )


def source_range(
    document_id: str,
    start: Coordinate,
    end: Coordinate,
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Expand an inclusive source-coordinate range."""

    index = document_index(
        indexes,
        document_id,
    )

    try:
        start_index = (
            index.coordinate_indexes[start]
        )
        end_index = (
            index.coordinate_indexes[end]
        )
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown source "
            f"coordinate {error.args[0]}."
        ) from error

    if end_index < start_index:
        raise RetrievalUnitError(
            f"{document_id}: inverted source range."
        )

    return index.coordinates[
        start_index:
        end_index + 1
    ]


def heading_coordinates(
    record: Mapping[str, Any],
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Expand one source heading's raw coordinates."""

    document_id = record.get("document_id")
    heading_source = record.get(
        "heading_source"
    )

    if (
        not isinstance(document_id, str)
        or not isinstance(
            heading_source,
            Mapping,
        )
    ):
        raise RetrievalUnitError(
            "Source span has no usable heading_source."
        )

    index = document_index(
        indexes,
        document_id,
    )

    return source_range(
        document_id,
        coordinate_from_record(
            heading_source["start"],
            record_name="Heading source start",
            index=index,
        ),
        coordinate_from_record(
            heading_source["end"],
            record_name="Heading source end",
            index=index,
        ),
        indexes,
    )


def direct_coordinates(
    record: Mapping[str, Any],
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Expand one source heading's direct-body coordinates."""

    document_id = record.get("document_id")
    direct_body = record.get(
        "direct_body"
    )

    if (
        not isinstance(document_id, str)
        or not isinstance(
            direct_body,
            Mapping,
        )
    ):
        raise RetrievalUnitError(
            "Source span has no usable direct_body."
        )

    if direct_body.get("is_empty"):
        if (
            direct_body.get(
                "included_start"
            )
            is not None
            or direct_body.get(
                "included_end"
            )
            is not None
        ):
            raise RetrievalUnitError(
                f"{record.get('node_id')}: empty "
                "direct body has included coordinates."
            )

        return ()

    index = document_index(
        indexes,
        document_id,
    )
    included_start = direct_body.get(
        "included_start"
    )
    included_end = direct_body.get(
        "included_end"
    )

    if (
        not isinstance(
            included_start,
            Mapping,
        )
        or not isinstance(
            included_end,
            Mapping,
        )
    ):
        raise RetrievalUnitError(
            f"{record.get('node_id')}: nonempty "
            "direct body lacks included coordinates."
        )

    return source_range(
        document_id,
        coordinate_from_record(
            included_start,
            record_name="Direct-body start",
            index=index,
        ),
        coordinate_from_record(
            included_end,
            record_name="Direct-body end",
            index=index,
        ),
        indexes,
    )


def structural_context_ids(
    document_id: str,
    start_coordinate: Coordinate,
    indexes: CorpusIndexes,
) -> tuple[str, ...]:
    """Return reviewed RMF part context for one source."""

    if document_id != RMF_ID:
        return ()

    index = document_index(
        indexes,
        document_id,
    )

    try:
        position = (
            index.coordinate_indexes[
                start_coordinate
            ]
        )
        part_one_position = (
            index.coordinate_indexes[(9, 3)]
        )
        part_two_position = (
            index.coordinate_indexes[(25, 3)]
        )
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: missing structural "
            f"anchor {error.args[0]}."
        ) from error

    if position >= part_two_position:
        return (
            "rmf-part-2-core-and-profiles",
        )

    if position >= part_one_position:
        return (
            "rmf-part-1-foundational-information",
        )

    return ()


def is_reference_start(text: str) -> bool:
    """Disambiguate numbered EU references from recitals."""

    stripped = text.strip()
    match = RECITAL_PATTERN.match(
        stripped
    )

    if match is None:
        return False

    remainder = stripped[
        match.end():
    ].lstrip()
    folded = remainder.casefold()

    return any(
        folded.startswith(prefix)
        for prefix in REFERENCE_PREFIXES
    )


def classify_exclusion_reason(
    reason: str,
) -> str:
    """Map one reviewed exclusion reason to its ledger class."""

    if reason in PAGE_FURNITURE_REASONS:
        return "excluded_page_furniture"

    if reason == "eu_numbered_reference_region":
        return "excluded_reference_region"

    return "excluded_structural_metadata"


def _set_exclusion(
    reasons: dict[
        tuple[str, Coordinate],
        str,
    ],
    document_id: str,
    coordinate: Coordinate,
    reason: str,
) -> None:
    """Set one exclusion while preserving the first reviewed reason."""

    reasons.setdefault(
        (
            document_id,
            coordinate,
        ),
        reason,
    )


def _validate_controlled_inputs(
    pages: Sequence[Mapping[str, Any]],
    hierarchy: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    indexes: CorpusIndexes,
) -> None:
    """Validate the pinned four-document production inputs."""

    if len(pages) != EXPECTED_PAGE_COUNT:
        raise RetrievalUnitError(
            f"Expected {EXPECTED_PAGE_COUNT} pages, "
            f"found {len(pages)}."
        )

    if len(hierarchy) != EXPECTED_HIERARCHY_COUNT:
        raise RetrievalUnitError(
            "Expected "
            f"{EXPECTED_HIERARCHY_COUNT} hierarchy "
            f"records, found {len(hierarchy)}."
        )

    if len(spans) != EXPECTED_SPAN_COUNT:
        raise RetrievalUnitError(
            f"Expected {EXPECTED_SPAN_COUNT} span "
            f"records, found {len(spans)}."
        )

    if (
        indexes.document_order
        != EXPECTED_DOCUMENT_ORDER
    ):
        raise RetrievalUnitError(
            "Controlled document order changed: "
            f"{indexes.document_order}."
        )

    if (
        indexes.total_line_count
        != EXPECTED_TOTAL_LINE_COUNT
    ):
        raise RetrievalUnitError(
            "Expected "
            f"{EXPECTED_TOTAL_LINE_COUNT} extracted "
            f"lines, found {indexes.total_line_count}."
        )

    hierarchy_ids = [
        record.get("node_id")
        for record in hierarchy
    ]
    span_ids = [
        record.get("node_id")
        for record in spans
    ]

    if hierarchy_ids != span_ids:
        raise RetrievalUnitError(
            "Hierarchy and span node order differ."
        )

    if len(set(hierarchy_ids)) != len(
        hierarchy_ids
    ):
        raise RetrievalUnitError(
            "Hierarchy contains duplicate node IDs."
        )

    source_count = sum(
        record.get("node_kind") == "source"
        for record in spans
    )
    synthetic_count = sum(
        record.get("node_kind") == "synthetic"
        for record in spans
    )

    if (
        source_count
        != EXPECTED_SOURCE_SPAN_COUNT
        or synthetic_count
        != EXPECTED_SYNTHETIC_SPAN_COUNT
    ):
        raise RetrievalUnitError(
            "Source/synthetic span counts changed: "
            f"{source_count}/{synthetic_count}."
        )


def _build_exclusion_plan(
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    spans: Sequence[Mapping[str, Any]],
    indexes: CorpusIndexes,
) -> ExclusionPlan:
    """Build all reviewed coordinate exclusions and recital anchors."""

    reasons: dict[
        tuple[str, Coordinate],
        str,
    ] = {}

    for document_id in indexes.document_order:
        index = document_index(
            indexes,
            document_id,
        )

        for coordinate in index.coordinates:
            reason = explicit_furniture_reason(
                document_id,
                coordinate,
                indexes,
            )

            if reason is not None:
                _set_exclusion(
                    reasons,
                    document_id,
                    coordinate,
                    reason,
                )

            if is_eu_eli(
                document_id,
                coordinate,
                indexes,
            ):
                _set_exclusion(
                    reasons,
                    document_id,
                    coordinate,
                    "eu_eli_footer",
                )

    rmf_part_one = (9, 2)
    rmf_part_two = (25, 2)
    eu_enacting_formula = (44, 34)
    gpt_authorship_start = (26, 1)

    anchors = (
        (
            RMF_ID,
            rmf_part_one,
            "Part 1: Foundational Information",
        ),
        (
            RMF_ID,
            rmf_part_two,
            "Part 2: Core and Profiles",
        ),
        (
            EU_ID,
            eu_enacting_formula,
            "HAVE ADOPTED THIS REGULATION:",
        ),
        (
            GPT_ID,
            gpt_authorship_start,
            (
                "Authorship, credit attribution, "
                "and acknowledgments"
            ),
        ),
    )

    for (
        document_id,
        coordinate,
        expected_text,
    ) in anchors:
        index = document_index(
            indexes,
            document_id,
        )

        if normalize(
            index.line_text[coordinate]
        ) != expected_text:
            raise RetrievalUnitError(
                f"{document_id}: reviewed anchor "
                f"{coordinate} changed."
            )

    _set_exclusion(
        reasons,
        RMF_ID,
        rmf_part_one,
        "rmf_part_1_label",
    )
    _set_exclusion(
        reasons,
        RMF_ID,
        rmf_part_two,
        "rmf_part_2_label",
    )
    _set_exclusion(
        reasons,
        EU_ID,
        eu_enacting_formula,
        "eu_enacting_formula",
    )

    first_heading_coordinates: dict[
        str,
        Coordinate,
    ] = {}

    for document_id in indexes.document_order:
        records = source_records(
            document_id,
            spans,
        )

        if not records:
            raise RetrievalUnitError(
                f"{document_id}: no source spans."
            )

        first_heading_coordinates[
            document_id
        ] = heading_coordinates(
            records[0],
            indexes,
        )[0]

    gpt_reference_spans = [
        record
        for record in source_records(
            GPT_ID,
            spans,
        )
        if hierarchy_by_id[
            record["node_id"]
        ].get("full_heading")
        == "References"
    ]

    if len(gpt_reference_spans) != 1:
        raise RetrievalUnitError(
            "Expected one GPT References source."
        )

    gpt_reference_start = (
        heading_coordinates(
            gpt_reference_spans[0],
            indexes,
        )[0]
    )
    gpt_index = document_index(
        indexes,
        GPT_ID,
    )
    credit_start_index = (
        gpt_index.coordinate_indexes[
            gpt_authorship_start
        ]
    )
    credit_stop_index = (
        gpt_index.coordinate_indexes[
            gpt_reference_start
        ]
    )

    for coordinate in gpt_index.coordinates[
        credit_start_index:
        credit_stop_index
    ]:
        _set_exclusion(
            reasons,
            GPT_ID,
            coordinate,
            "gpt_authorship_and_credits",
        )

    eu_index = document_index(
        indexes,
        EU_ID,
    )
    eu_first_heading = (
        first_heading_coordinates[EU_ID]
    )
    eu_first_heading_index = (
        eu_index.coordinate_indexes[
            eu_first_heading
        ]
    )
    eu_prefix = eu_index.coordinates[
        :eu_first_heading_index
    ]
    eu_candidates: dict[
        int,
        list[Coordinate],
    ] = defaultdict(list)

    for coordinate in eu_prefix:
        match = RECITAL_PATTERN.match(
            eu_index.line_text[
                coordinate
            ].strip()
        )

        if match is None:
            continue

        number = int(match.group(1))

        if 1 <= number <= 180:
            eu_candidates[number].append(
                coordinate
            )

    selected_recitals: list[
        tuple[int, Coordinate]
    ] = []
    selected_references: list[
        tuple[int, Coordinate]
    ] = []

    for number in range(1, 181):
        candidates = eu_candidates.get(
            number,
            [],
        )

        if len(candidates) == 1:
            selected_recitals.append(
                (
                    number,
                    candidates[0],
                )
            )
            continue

        if len(candidates) != 2:
            raise RetrievalUnitError(
                f"EU number {number}: expected "
                "one or two candidates, found "
                f"{len(candidates)}."
            )

        references = [
            coordinate
            for coordinate in candidates
            if is_reference_start(
                eu_index.line_text[
                    coordinate
                ]
            )
        ]
        recitals = [
            coordinate
            for coordinate in candidates
            if coordinate not in references
        ]

        if (
            len(references) != 1
            or len(recitals) != 1
        ):
            raise RetrievalUnitError(
                f"EU number {number}: ambiguous "
                "recital/reference selection."
            )

        selected_recitals.append(
            (
                number,
                recitals[0],
            )
        )
        selected_references.append(
            (
                number,
                references[0],
            )
        )

    if [
        number
        for number, _ in selected_recitals
    ] != list(range(1, 181)):
        raise RetrievalUnitError(
            "EU recital sequence is incomplete."
        )

    reference_region_start: dict[
        int,
        int,
    ] = {}

    for _, coordinate in selected_references:
        page_number, line_number = coordinate
        existing = reference_region_start.get(
            page_number
        )

        if (
            existing is None
            or line_number < existing
        ):
            reference_region_start[
                page_number
            ] = line_number

    for coordinate in eu_prefix:
        page_number, line_number = coordinate
        start_line = reference_region_start.get(
            page_number
        )

        if (
            start_line is not None
            and line_number >= start_line
        ):
            _set_exclusion(
                reasons,
                EU_ID,
                coordinate,
                "eu_numbered_reference_region",
            )

    return ExclusionPlan(
        reasons=reasons,
        selected_recitals=tuple(
            selected_recitals
        ),
        first_heading_coordinates=(
            first_heading_coordinates
        ),
    )


def _build_logical_sources(
    hierarchy: Sequence[Mapping[str, Any]],
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    spans: Sequence[Mapping[str, Any]],
    indexes: CorpusIndexes,
    exclusion_plan: ExclusionPlan,
) -> tuple[
    tuple[LogicalSource, ...],
    int,
]:
    """Build reviewed unsplit logical retrieval sources."""

    source_descendant_counts: Counter[
        str
    ] = Counter()

    for node in hierarchy:
        if node.get("node_kind") != "source":
            continue

        for ancestor_node_id in node.get(
            "ancestor_node_ids",
            [],
        ):
            source_descendant_counts[
                ancestor_node_id
            ] += 1

    logical_sources: list[
        LogicalSource
    ] = []
    omitted_empty_containers = 0
    reasons = exclusion_plan.reasons

    rmf_index = document_index(
        indexes,
        RMF_ID,
    )
    rmf_summary_anchor = (6, 2)
    rmf_part_one = (9, 2)

    if normalize(
        rmf_index.line_text[
            rmf_summary_anchor
        ]
    ) != "Executive Summary":
        raise RetrievalUnitError(
            "RMF Executive Summary anchor changed."
        )

    summary_start_index = (
        rmf_index.coordinate_indexes[
            rmf_summary_anchor
        ]
        + 1
    )
    summary_stop_index = (
        rmf_index.coordinate_indexes[
            rmf_part_one
        ]
    )
    summary_coordinates = tuple(
        coordinate
        for coordinate in rmf_index.coordinates[
            summary_start_index:
            summary_stop_index
        ]
        if (
            RMF_ID,
            coordinate,
        )
        not in reasons
    )

    logical_sources.append(
        LogicalSource(
            document_id=RMF_ID,
            unit_kind="frontmatter_body",
            source_key=(
                "rmf-executive-summary"
            ),
            coordinates=summary_coordinates,
            context_word_count=2,
            context_node_ids=(),
            structural_context_ids=(
                "rmf-frontmatter-executive-summary",
            ),
            heading_source={
                "start": coordinate_record(
                    rmf_index,
                    rmf_summary_anchor,
                ),
                "end": coordinate_record(
                    rmf_index,
                    rmf_summary_anchor,
                ),
            },
            metadata={
                "frontmatter_id": (
                    "rmf-executive-summary"
                )
            },
            hard_word_limit=(
                STANDARD_HARD_WORDS
            ),
        )
    )

    eu_index = document_index(
        indexes,
        EU_ID,
    )
    selected_recitals = dict(
        exclusion_plan.selected_recitals
    )
    selected_recital_indexes = {
        number: (
            eu_index.coordinate_indexes[
                coordinate
            ]
        )
        for number, coordinate
        in exclusion_plan.selected_recitals
    }
    eu_first_heading_index = (
        eu_index.coordinate_indexes[
            exclusion_plan
            .first_heading_coordinates[
                EU_ID
            ]
        ]
    )

    for number in range(1, 181):
        start_index = (
            selected_recital_indexes[
                number
            ]
        )
        stop_index = (
            selected_recital_indexes[
                number + 1
            ]
            if number < 180
            else eu_first_heading_index
        )
        coordinates = tuple(
            coordinate
            for coordinate in eu_index.coordinates[
                start_index:
                stop_index
            ]
            if (
                EU_ID,
                coordinate,
            )
            not in reasons
        )

        if not coordinates:
            raise RetrievalUnitError(
                f"EU recital {number}: no "
                "retained coordinates."
            )

        if (
            coordinates[0]
            != selected_recitals[number]
        ):
            raise RetrievalUnitError(
                f"EU recital {number}: selected "
                "start was excluded."
            )

        logical_sources.append(
            LogicalSource(
                document_id=EU_ID,
                unit_kind="eu_recital",
                source_key=(
                    f"eu-recital-{number:03d}"
                ),
                coordinates=coordinates,
                context_word_count=2,
                context_node_ids=(),
                structural_context_ids=(
                    "eu-ai-act-recitals",
                ),
                heading_source=None,
                metadata={
                    "recital_number": number,
                },
                hard_word_limit=(
                    RECITAL_29_HARD_WORDS
                    if number == 29
                    else STANDARD_HARD_WORDS
                ),
            )
        )

    for document_id in indexes.document_order:
        for record in source_records(
            document_id,
            spans,
        ):
            node_id = record["node_id"]
            node = hierarchy_by_id[
                node_id
            ]
            direct = direct_coordinates(
                record,
                indexes,
            )
            is_leaf = (
                source_descendant_counts[
                    node_id
                ]
                == 0
            )

            if not direct:
                if is_leaf:
                    heading = (
                        heading_coordinates(
                            record,
                            indexes,
                        )
                    )

                    logical_sources.append(
                        LogicalSource(
                            document_id=(
                                document_id
                            ),
                            unit_kind=(
                                "heading_only"
                            ),
                            source_key=node_id,
                            coordinates=heading,
                            context_word_count=0,
                            context_node_ids=tuple(
                                node[
                                    "ancestor_node_ids"
                                ]
                            ),
                            structural_context_ids=(
                                structural_context_ids(
                                    document_id,
                                    heading[0],
                                    indexes,
                                )
                            ),
                            heading_source=dict(
                                record[
                                    "heading_source"
                                ]
                            ),
                            metadata={
                                "source_node_id": (
                                    node_id
                                ),
                                "heading_id": (
                                    record[
                                        "heading_id"
                                    ]
                                ),
                            },
                            hard_word_limit=(
                                STANDARD_HARD_WORDS
                            ),
                        )
                    )
                else:
                    omitted_empty_containers += 1

                continue

            retained = tuple(
                coordinate
                for coordinate in direct
                if (
                    document_id,
                    coordinate,
                )
                not in reasons
            )

            if not retained:
                continue

            heading = heading_coordinates(
                record,
                indexes,
            )
            full_heading = node.get(
                "full_heading"
            )

            if (
                not isinstance(
                    full_heading,
                    str,
                )
                or not full_heading.strip()
            ):
                raise RetrievalUnitError(
                    f"{node_id}: missing full_heading."
                )

            logical_sources.append(
                LogicalSource(
                    document_id=document_id,
                    unit_kind="heading_body",
                    source_key=node_id,
                    coordinates=retained,
                    context_word_count=(
                        count_words(
                            full_heading
                        )
                    ),
                    context_node_ids=tuple(
                        [
                            *node[
                                "ancestor_node_ids"
                            ],
                            node_id,
                        ]
                    ),
                    structural_context_ids=(
                        structural_context_ids(
                            document_id,
                            heading[0],
                            indexes,
                        )
                    ),
                    heading_source={
                        "start": record[
                            "heading_source"
                        ]["start"],
                        "end": record[
                            "heading_source"
                        ]["end"],
                    },
                    metadata={
                        "source_node_id": node_id,
                        "heading_id": record[
                            "heading_id"
                        ],
                    },
                    hard_word_limit=(
                        STANDARD_HARD_WORDS
                    ),
                )
            )

    if (
        len(logical_sources)
        != EXPECTED_LOGICAL_SOURCE_COUNT
    ):
        raise RetrievalUnitError(
            "Expected "
            f"{EXPECTED_LOGICAL_SOURCE_COUNT} "
            "logical sources, found "
            f"{len(logical_sources)}."
        )

    if omitted_empty_containers != 39:
        raise RetrievalUnitError(
            "Expected 39 omitted empty containers, "
            f"found {omitted_empty_containers}."
        )

    return (
        tuple(logical_sources),
        omitted_empty_containers,
    )


def _build_unit_record(
    source: LogicalSource,
    *,
    part_number: int,
    part_count: int,
    coordinates: Sequence[Coordinate],
    content_word_count: int,
    indexed_word_count: int,
    boundary_after: str,
    indexes: CorpusIndexes,
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    extra_metadata: Mapping[
        str,
        Any,
    ]
    | None = None,
) -> tuple[
    dict[str, Any],
    tuple[Coordinate, ...],
]:
    """Create one validated production retrieval-unit record."""

    if not coordinates:
        raise RetrievalUnitError(
            f"{source.document_id} | "
            f"{source.source_key}: empty unit."
        )

    index = document_index(
        indexes,
        source.document_id,
    )

    if any(
        not index.line_text[
            coordinate
        ].strip()
        for coordinate in coordinates
    ):
        raise RetrievalUnitError(
            f"{source.source_key}: blank "
            "coordinates remain."
        )

    if any(
        is_eu_eli(
            source.document_id,
            coordinate,
            indexes,
        )
        for coordinate in coordinates
    ):
        raise RetrievalUnitError(
            f"{source.source_key}: EU ELI "
            "furniture remains."
        )

    if source.unit_kind == "heading_only":
        source_node_id = source.metadata[
            "source_node_id"
        ]
        node = hierarchy_by_id.get(
            source_node_id
        )

        if node is None:
            raise RetrievalUnitError(
                f"{source_node_id}: missing "
                "heading-only hierarchy node."
            )

        measured_content_words = (
            count_words(
                node["full_heading"]
            )
        )
    else:
        measured_content_words = sum(
            count_words(
                index.line_text[
                    coordinate
                ]
            )
            for coordinate in coordinates
        )

    if (
        measured_content_words
        != content_word_count
    ):
        raise RetrievalUnitError(
            f"{source.source_key}: content-word "
            "count differs from its basis."
        )

    if (
        content_word_count
        + source.context_word_count
        != indexed_word_count
    ):
        raise RetrievalUnitError(
            f"{source.source_key}: indexed-word "
            "count is inconsistent."
        )

    if (
        indexed_word_count
        > source.hard_word_limit
    ):
        raise RetrievalUnitError(
            f"{source.source_key}: unit exceeds "
            "its hard word limit."
        )

    unit_id = (
        f"candidate-v2:{source.document_id}:"
        f"{source.source_key}:"
        f"part-{part_number:03d}"
    )
    record: dict[str, Any] = {
        "schema_version": (
            UNIT_SCHEMA_VERSION
        ),
        "unit_id": unit_id,
        "unit_kind": source.unit_kind,
        "document_id": source.document_id,
        "part_number": part_number,
        "part_count": part_count,
        "context_node_ids": list(
            source.context_node_ids
        ),
        "structural_context_ids": list(
            source.structural_context_ids
        ),
        "heading_source": (
            source.heading_source
        ),
        "content_segments": content_segments(
            source.document_id,
            coordinates,
            indexes,
        ),
        "content_coordinate_count": len(
            coordinates
        ),
        "content_word_count": (
            content_word_count
        ),
        "context_word_count": (
            source.context_word_count
        ),
        "indexed_word_count": (
            indexed_word_count
        ),
        "hard_word_limit": (
            source.hard_word_limit
        ),
        "boundary_after": boundary_after,
        **dict(source.metadata),
    }

    if extra_metadata:
        record.update(
            extra_metadata
        )

    return (
        record,
        tuple(coordinates),
    )


def _materialize_units(
    logical_sources: Sequence[
        LogicalSource
    ],
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    indexes: CorpusIndexes,
) -> tuple[
    tuple[dict[str, Any], ...],
    Mapping[str, tuple[Coordinate, ...]],
    tuple[dict[str, Any], ...],
    frozenset[Coordinate],
]:
    """Pack logical sources into final production units."""

    _, footnote_coordinates = (
        discover_genai_footnotes(indexes)
    )
    units: list[dict[str, Any]] = []
    unit_coordinate_map: dict[
        str,
        tuple[Coordinate, ...],
    ] = {}
    reference_sections: list[
        dict[str, Any]
    ] = []
    def add_unit(
        record: dict[str, Any],
        coordinates: tuple[
            Coordinate,
            ...,
        ],
    ) -> None:
        if record["unit_id"] in (
            unit_coordinate_map
        ):
            raise RetrievalUnitError(
                f"Duplicate unit ID: "
                f"{record['unit_id']}"
            )

        units.append(record)
        unit_coordinate_map[
            record["unit_id"]
        ] = coordinates

    for source in logical_sources:
        coordinates = cleaned_coordinates(
            source.document_id,
            source.coordinates,
            indexes,
        )

        if not coordinates:
            raise RetrievalUnitError(
                f"{source.source_key}: cleaning "
                "removed all coordinates."
            )

        unit_stub = {
            "unit_id": source.source_key,
            "unit_kind": source.unit_kind,
            **dict(source.metadata),
        }
        label = logical_source_label(
            unit_stub,
            hierarchy_by_id,
        )

        if source.unit_kind == "heading_only":
            if source.context_word_count != 0:
                raise RetrievalUnitError(
                    f"{source.source_key}: heading-only "
                    "context must be zero."
                )

            content_words = count_words(
                hierarchy_by_id[
                    source.metadata[
                        "source_node_id"
                    ]
                ]["full_heading"]
            )
            record, owned = (
                _build_unit_record(
                    source,
                    part_number=1,
                    part_count=1,
                    coordinates=coordinates,
                    content_word_count=(
                        content_words
                    ),
                    indexed_word_count=(
                        content_words
                    ),
                    boundary_after=(
                        "end_of_heading_source"
                    ),
                    indexes=indexes,
                    hierarchy_by_id=(
                        hierarchy_by_id
                    ),
                )
            )
            add_unit(
                record,
                owned,
            )
            continue

        if REFERENCE_HEADING_PATTERN.search(
            label
        ):
            result = reference_entry_groups(
                source.document_id,
                label,
                coordinates,
                indexes,
            )

            if not result["passed"]:
                raise RetrievalUnitError(
                    f"{source.source_key}: reference "
                    "packing failed: "
                    f"{result.get('reason')}."
                )

            if (
                count_words(label)
                != source.context_word_count
            ):
                raise RetrievalUnitError(
                    f"{source.source_key}: reference "
                    "context-word basis differs."
                )

            packs = result["packs"]
            reference_sections.append(
                {
                    "document_id": (
                        source.document_id
                    ),
                    "label": label,
                    "entry_count": len(
                        result["entries"]
                    ),
                    "pack_count": len(packs),
                    "maximum_entry_words": max(
                        entry["word_count"]
                        for entry in result[
                            "entries"
                        ]
                    ),
                }
            )

            for part_number, pack in enumerate(
                packs,
                start=1,
            ):
                pack_coordinates = tuple(
                    coordinate
                    for entry in pack[
                        "entries"
                    ]
                    for coordinate in entry[
                        "coordinates"
                    ]
                )
                boundary_after = (
                    "before_reference_entry"
                    if part_number < len(packs)
                    else "end_of_source_unit"
                )
                record, owned = (
                    _build_unit_record(
                        source,
                        part_number=(
                            part_number
                        ),
                        part_count=len(packs),
                        coordinates=(
                            pack_coordinates
                        ),
                        content_word_count=(
                            pack[
                                "content_words"
                            ]
                        ),
                        indexed_word_count=(
                            pack[
                                "indexed_words"
                            ]
                        ),
                        boundary_after=(
                            boundary_after
                        ),
                        indexes=indexes,
                        hierarchy_by_id=(
                            hierarchy_by_id
                        ),
                        extra_metadata={
                            (
                                "reference_entry_"
                                "start_ordinal"
                            ): (
                                pack["entries"][
                                    0
                                ][
                                    "entry_number"
                                ]
                            ),
                            (
                                "reference_entry_"
                                "end_ordinal"
                            ): (
                                pack["entries"][
                                    -1
                                ][
                                    "entry_number"
                                ]
                            ),
                        },
                    )
                )
                add_unit(
                    record,
                    owned,
                )

            continue

        result = repack_nonreference(
            source.document_id,
            coordinates,
            context_words=(
                source.context_word_count
            ),
            hard_words=(
                source.hard_word_limit
            ),
            indexes=indexes,
            genai_footnote_coordinates=(
                footnote_coordinates
            ),
        )

        if not result["passed"]:
            raise RetrievalUnitError(
                f"{source.source_key}: corrected "
                "packing failed: "
                f"{result.get('reason')}."
            )

        positions = {
            coordinate: position
            for position, coordinate
            in enumerate(coordinates)
        }
        rebuilt: list[Coordinate] = []
        pieces = result["pieces"]

        for part_number, piece in enumerate(
            pieces,
            start=1,
        ):
            start_index = positions[
                piece["start"]
            ]
            end_index = positions[
                piece["end"]
            ]

            if end_index < start_index:
                raise RetrievalUnitError(
                    f"{source.source_key}: inverted "
                    "repacked piece."
                )

            piece_coordinates = tuple(
                coordinates[
                    start_index:
                    end_index + 1
                ]
            )
            rebuilt.extend(
                piece_coordinates
            )
            record, owned = (
                _build_unit_record(
                    source,
                    part_number=(
                        part_number
                    ),
                    part_count=len(pieces),
                    coordinates=(
                        piece_coordinates
                    ),
                    content_word_count=(
                        piece[
                            "content_words"
                        ]
                    ),
                    indexed_word_count=(
                        piece[
                            "indexed_words"
                        ]
                    ),
                    boundary_after=(
                        piece[
                            "boundary_after"
                        ]
                    ),
                    indexes=indexes,
                    hierarchy_by_id=(
                        hierarchy_by_id
                    ),
                )
            )
            add_unit(
                record,
                owned,
            )

        if tuple(rebuilt) != coordinates:
            raise RetrievalUnitError(
                f"{source.source_key}: packing "
                "changed source coverage."
            )

    ordered_units: list[dict[str, Any]] = []

    for document_id in indexes.document_order:
        document_units = [
            unit
            for unit in units
            if unit["document_id"] == document_id
        ]

        for document_unit_order, unit in enumerate(
            document_units,
            start=1,
        ):
            unit["document_unit_order"] = (
                document_unit_order
            )
            ordered_units.append(unit)

    if len(ordered_units) != len(units):
        raise RetrievalUnitError(
            "Document-grouped unit ordering lost records."
        )

    units = ordered_units

    if len(units) != EXPECTED_UNIT_COUNT:
        raise RetrievalUnitError(
            f"Expected {EXPECTED_UNIT_COUNT} "
            f"units, found {len(units)}."
        )

    return (
        tuple(units),
        unit_coordinate_map,
        tuple(reference_sections),
        footnote_coordinates,
    )


def _build_ledger(
    units: Sequence[Mapping[str, Any]],
    unit_coordinate_map: Mapping[
        str,
        tuple[Coordinate, ...],
    ],
    logical_sources: Sequence[
        LogicalSource
    ],
    spans: Sequence[Mapping[str, Any]],
    indexes: CorpusIndexes,
    exclusion_plan: ExclusionPlan,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    """Build the complete final coordinate ledger."""

    content_owner: dict[
        tuple[str, Coordinate],
        str,
    ] = {}

    for unit in units:
        unit_id = unit["unit_id"]
        document_id = unit[
            "document_id"
        ]

        for coordinate in unit_coordinate_map[
            unit_id
        ]:
            key = (
                document_id,
                coordinate,
            )

            if key in content_owner:
                raise RetrievalUnitError(
                    "Retrieval-content overlap at "
                    f"{key}: "
                    f"{content_owner[key]} and "
                    f"{unit_id}."
                )

            content_owner[key] = unit_id

    heading_only_source_ids = {
        source.metadata[
            "source_node_id"
        ]
        for source in logical_sources
        if source.unit_kind
        == "heading_only"
    }
    heading_context_owner: dict[
        tuple[str, Coordinate],
        str,
    ] = {}

    for document_id in indexes.document_order:
        for record in source_records(
            document_id,
            spans,
        ):
            node_id = record["node_id"]

            if (
                node_id
                in heading_only_source_ids
            ):
                continue

            for coordinate in heading_coordinates(
                record,
                indexes,
            ):
                key = (
                    document_id,
                    coordinate,
                )

                if key in content_owner:
                    raise RetrievalUnitError(
                        "Heading context overlaps "
                        f"retrieval content at {key}."
                    )

                if key in heading_context_owner:
                    raise RetrievalUnitError(
                        "Duplicate heading context "
                        f"at {key}."
                    )

                heading_context_owner[
                    key
                ] = node_id

    rmf_summary_anchor_key = (
        RMF_ID,
        (6, 2),
    )

    if rmf_summary_anchor_key in (
        content_owner
    ):
        raise RetrievalUnitError(
            "RMF Executive Summary anchor "
            "overlaps retrieval content."
        )

    heading_context_owner[
        rmf_summary_anchor_key
    ] = "rmf-executive-summary"

    first_heading_indexes = {
        document_id: (
            document_index(
                indexes,
                document_id,
            ).coordinate_indexes[
                exclusion_plan
                .first_heading_coordinates[
                    document_id
                ]
            ]
        )
        for document_id
        in indexes.document_order
    }

    ledger: list[dict[str, Any]] = []
    classification_counts: Counter[
        str
    ] = Counter()
    reason_counts: Counter[
        str
    ] = Counter()
    document_classifications: dict[
        str,
        Counter[str],
    ] = {
        document_id: Counter()
        for document_id
        in indexes.document_order
    }

    for document_id in indexes.document_order:
        index = document_index(
            indexes,
            document_id,
        )

        for coordinate in index.coordinates:
            key = (
                document_id,
                coordinate,
            )

            if key in content_owner:
                classification = (
                    "retrieval_content"
                )
                reason = None
                owner_id = (
                    content_owner[key]
                )

            elif key in heading_context_owner:
                classification = (
                    "heading_context"
                )
                reason = None
                owner_id = (
                    heading_context_owner[
                        key
                    ]
                )

            elif key in exclusion_plan.reasons:
                reason = (
                    exclusion_plan.reasons[
                        key
                    ]
                )
                classification = (
                    classify_exclusion_reason(
                        reason
                    )
                )
                owner_id = None

            elif (
                index.coordinate_indexes[
                    coordinate
                ]
                < first_heading_indexes[
                    document_id
                ]
            ):
                classification = (
                    "excluded_frontmatter_metadata"
                )
                reason = (
                    "document_frontmatter_metadata"
                )
                owner_id = None

            elif not index.line_text[
                coordinate
            ].strip():
                classification = (
                    "excluded_blank_line"
                )
                reason = (
                    "blank_source_line"
                )
                owner_id = None

            else:
                raise RetrievalUnitError(
                    "Unclassified post-heading "
                    f"source line: {document_id} "
                    f"{coordinate}."
                )

            classification_counts[
                classification
            ] += 1
            document_classifications[
                document_id
            ][classification] += 1

            if reason is not None:
                reason_counts[
                    reason
                ] += 1

            ledger.append(
                {
                    "schema_version": (
                        LEDGER_SCHEMA_VERSION
                    ),
                    "document_id": (
                        document_id
                    ),
                    "coordinate": (
                        coordinate_record(
                            index,
                            coordinate,
                        )
                    ),
                    "classification": (
                        classification
                    ),
                    "reason": reason,
                    "owner_id": owner_id,
                }
            )

    if (
        len(ledger)
        != EXPECTED_TOTAL_LINE_COUNT
    ):
        raise RetrievalUnitError(
            "Coordinate ledger does not cover "
            "all extracted source lines."
        )

    ledger_by_key = {
        (
            record["document_id"],
            (
                record["coordinate"][
                    "page_number"
                ],
                record["coordinate"][
                    "line_number"
                ],
            ),
        ): record
        for record in ledger
    }

    if len(ledger_by_key) != len(ledger):
        raise RetrievalUnitError(
            "Coordinate ledger contains "
            "duplicate coordinates."
        )

    for document_id in indexes.document_order:
        for record in source_records(
            document_id,
            spans,
        ):
            for coordinate in direct_coordinates(
                record,
                indexes,
            ):
                ledger_record = ledger_by_key[
                    (
                        document_id,
                        coordinate,
                    )
                ]

                if ledger_record[
                    "classification"
                ] not in {
                    "retrieval_content",
                    "excluded_blank_line",
                    "excluded_page_furniture",
                    "excluded_reference_region",
                    "excluded_structural_metadata",
                }:
                    raise RetrievalUnitError(
                        "Direct-body coordinate lacks "
                        "content or exclusion ownership: "
                        f"{document_id} {coordinate}."
                    )

    if (
        dict(classification_counts)
        != EXPECTED_CLASSIFICATION_COUNTS
    ):
        raise RetrievalUnitError(
            "Ledger classification counts differ.\n"
            f"Expected: "
            f"{EXPECTED_CLASSIFICATION_COUNTS}\n"
            f"Actual: "
            f"{dict(classification_counts)}"
        )

    if (
        dict(reason_counts)
        != EXPECTED_REASON_COUNTS
    ):
        raise RetrievalUnitError(
            "Ledger reason counts differ.\n"
            f"Expected: {EXPECTED_REASON_COUNTS}\n"
            f"Actual: {dict(reason_counts)}"
        )

    eli_records = [
        record
        for record in ledger
        if (
            record["document_id"]
            == EU_ID
            and is_eu_eli(
                EU_ID,
                (
                    record["coordinate"][
                        "page_number"
                    ],
                    record["coordinate"][
                        "line_number"
                    ],
                ),
                indexes,
            )
        )
    ]

    if len(eli_records) != 72:
        raise RetrievalUnitError(
            "Expected 72 EU ELI ledger "
            f"records, found {len(eli_records)}."
        )

    if any(
        (
            record["classification"]
            != "excluded_page_furniture"
            or record["reason"]
            != "eu_eli_footer"
        )
        for record in eli_records
    ):
        raise RetrievalUnitError(
            "Not every EU ELI line is explicit "
            "page furniture."
        )

    stats = {
        "classification_counts": dict(
            sorted(
                classification_counts.items()
            )
        ),
        "reason_counts": dict(
            sorted(reason_counts.items())
        ),
        "document_classifications": {
            document_id: dict(
                sorted(
                    counts.items()
                )
            )
            for document_id, counts
            in document_classifications.items()
        },
    }

    return (
        tuple(ledger),
        stats,
    )


def _audit_final_units(
    units: Sequence[Mapping[str, Any]],
    unit_coordinate_map: Mapping[
        str,
        tuple[Coordinate, ...],
    ],
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    indexes: CorpusIndexes,
    footnote_coordinates: frozenset[
        Coordinate
    ],
) -> dict[str, Any]:
    """Audit final unit counts, budgets, and semantic boundaries."""

    grouped = group_units_by_logical_source(
        [
            dict(unit)
            for unit in units
        ]
    )

    if (
        len(grouped.ordered_keys)
        != EXPECTED_LOGICAL_SOURCE_COUNT
    ):
        raise RetrievalUnitError(
            "Final logical-source count changed."
        )

    unit_kind_counts = Counter(
        unit["unit_kind"]
        for unit in units
    )
    document_unit_counts = Counter(
        unit["document_id"]
        for unit in units
    )
    boundary_counts = Counter(
        unit["boundary_after"]
        for unit in units
    )

    if (
        dict(unit_kind_counts)
        != EXPECTED_UNIT_KIND_COUNTS
    ):
        raise RetrievalUnitError(
            "Unit-kind counts differ."
        )

    if (
        dict(document_unit_counts)
        != EXPECTED_DOCUMENT_UNIT_COUNTS
    ):
        raise RetrievalUnitError(
            "Document unit counts differ."
        )

    if (
        dict(boundary_counts)
        != EXPECTED_BOUNDARY_COUNTS
    ):
        raise RetrievalUnitError(
            "Boundary counts differ."
        )

    audited_boundaries: list[
        dict[str, Any]
    ] = []
    semantic_risks: list[
        dict[str, Any]
    ] = []
    boundary_coordinates: set[
        tuple[
            str,
            Coordinate,
            Coordinate,
        ]
    ] = set()

    for key in grouped.ordered_keys:
        source_units = (
            grouped.groups[key]
        )
        first = source_units[0]
        label = logical_source_label(
            first,
            hierarchy_by_id,
        )
        is_reference = bool(
            REFERENCE_HEADING_PATTERN.search(
                label
            )
        )

        for left, right in zip(
            source_units,
            source_units[1:],
        ):
            document_id = left[
                "document_id"
            ]
            left_end = (
                unit_coordinate_map[
                    left["unit_id"]
                ][-1]
            )
            right_start = (
                unit_coordinate_map[
                    right["unit_id"]
                ][0]
            )
            index = document_index(
                indexes,
                document_id,
            )
            left_text = index.line_text[
                left_end
            ].rstrip()
            right_text = index.line_text[
                right_start
            ].lstrip()
            method = left[
                "boundary_after"
            ]
            risks: list[str] = []

            boundary_coordinates.add(
                (
                    document_id,
                    left_end,
                    right_start,
                )
            )

            if (
                method
                == "after_strong_terminal"
            ):
                if not (
                    STRONG_TERMINAL_PATTERN
                    .search(left_text)
                ):
                    risks.append(
                        "missing_strong_terminal"
                    )

                if (
                    LOWERCASE_START_PATTERN
                    .match(right_text)
                ):
                    risks.append(
                        "lowercase_continuation"
                    )

                if URL_PATTERN.match(
                    right_text
                ):
                    risks.append(
                        "url_continuation"
                    )

                if (
                    document_id == GENAI_ID
                    and right_start
                    in footnote_coordinates
                ):
                    risks.append(
                        "genai_footnote_continuation"
                    )

            elif (
                method
                == "before_structured_start"
            ):
                if not accepted_structured_start(
                    document_id,
                    right_start,
                    indexes,
                    footnote_coordinates,
                ):
                    risks.append(
                        "invalid_structured_start"
                    )

                if left_text.endswith(":"):
                    risks.append(
                        "introductory_colon_split"
                    )

            elif (
                method
                == "at_blank_paragraph_gap"
            ):
                if not gap_is_blank_only(
                    document_id,
                    left_end,
                    right_start,
                    indexes,
                ):
                    risks.append(
                        "missing_blank_paragraph_gap"
                    )

                if (
                    LOWERCASE_START_PATTERN
                    .match(right_text)
                ):
                    risks.append(
                        "lowercase_after_blank_gap"
                    )

            elif (
                method
                == "before_reference_entry"
            ):
                if not is_reference:
                    risks.append(
                        "reference_boundary_outside_"
                        "reference_section"
                    )

                if not reference_start_is_valid(
                    document_id,
                    right_start,
                    indexes,
                ):
                    risks.append(
                        "right_unit_does_not_start_"
                        "reference_entry"
                    )

                if right.get(
                    "reference_entry_start_ordinal"
                ) is None:
                    risks.append(
                        "missing_reference_entry_ordinal"
                    )

            else:
                risks.append(
                    "invalid_internal_boundary_"
                    f"method:{method}"
                )

            audit_record = {
                "document_id": document_id,
                "source_key": key[1],
                "label": label,
                "left_part": left[
                    "part_number"
                ],
                "right_part": right[
                    "part_number"
                ],
                "method": method,
                "left_end": left_end,
                "right_start": right_start,
                "risks": risks,
            }
            audited_boundaries.append(
                audit_record
            )

            if risks:
                semantic_risks.append(
                    audit_record
                )

    if (
        len(audited_boundaries)
        != EXPECTED_INTERNAL_BOUNDARY_COUNT
    ):
        raise RetrievalUnitError(
            "Expected "
            f"{EXPECTED_INTERNAL_BOUNDARY_COUNT} "
            "internal boundaries, found "
            f"{len(audited_boundaries)}."
        )

    if semantic_risks:
        raise RetrievalUnitError(
            "Semantic-boundary audit failed: "
            f"{semantic_risks[:5]}"
        )

    remaining_rejected = (
        KNOWN_REJECTED_BOUNDARIES
        & boundary_coordinates
    )

    if remaining_rejected:
        raise RetrievalUnitError(
            "Previously rejected boundaries "
            f"remain: {sorted(remaining_rejected)}"
        )

    semantic_exceptions = [
        unit
        for unit in units
        if unit["indexed_word_count"]
        > STANDARD_HARD_WORDS
    ]

    if len(semantic_exceptions) != 1:
        raise RetrievalUnitError(
            "Expected exactly one unit above "
            "the standard word ceiling."
        )

    semantic_exception = (
        semantic_exceptions[0]
    )

    if (
        semantic_exception[
            "unit_kind"
        ]
        != "eu_recital"
        or semantic_exception[
            "recital_number"
        ]
        != 29
        or semantic_exception[
            "indexed_word_count"
        ]
        != 580
        or semantic_exception[
            "hard_word_limit"
        ]
        != RECITAL_29_HARD_WORDS
    ):
        raise RetrievalUnitError(
            "Semantic exception is not exactly "
            "EU recital 29 at 580/640 words."
        )

    recital_53 = sorted(
        (
            unit
            for unit in units
            if (
                unit["unit_kind"]
                == "eu_recital"
                and unit[
                    "recital_number"
                ]
                == 53
            )
        ),
        key=lambda unit: unit[
            "part_number"
        ],
    )

    if (
        len(recital_53) != 2
        or recital_53[0][
            "boundary_after"
        ]
        != "after_strong_terminal"
    ):
        raise RetrievalUnitError(
            "EU recital 53 accepted split changed."
        )

    reference_units = Counter(
        unit["document_id"]
        for unit in units
        if (
            "reference_entry_start_ordinal"
            in unit
        )
    )

    if dict(reference_units) != {
        GENAI_ID: 5,
        GPT_ID: 5,
    }:
        raise RetrievalUnitError(
            "Reference-unit counts changed."
        )

    maximum_indexed_words = max(
        unit["indexed_word_count"]
        for unit in units
    )
    maximum_segments = max(
        len(unit["content_segments"])
        for unit in units
    )

    if maximum_indexed_words != 580:
        raise RetrievalUnitError(
            "Maximum indexed-word count changed."
        )

    if maximum_segments != 4:
        raise RetrievalUnitError(
            "Maximum content-segment count changed."
        )

    return {
        "logical_source_count": len(
            grouped.ordered_keys
        ),
        "unit_kind_counts": dict(
            sorted(unit_kind_counts.items())
        ),
        "document_unit_counts": dict(
            sorted(
                document_unit_counts.items()
            )
        ),
        "boundary_counts": dict(
            sorted(boundary_counts.items())
        ),
        "internal_boundary_count": len(
            audited_boundaries
        ),
        "semantic_risk_count": 0,
        "known_rejected_boundary_count": len(
            KNOWN_REJECTED_BOUNDARIES
        ),
        "known_rejected_boundaries_remaining": 0,
        "semantic_exception": {
            "unit_id": (
                semantic_exception[
                    "unit_id"
                ]
            ),
            "recital_number": 29,
            "indexed_word_count": 580,
            "hard_word_limit": 640,
        },
        "maximum_indexed_words": (
            maximum_indexed_words
        ),
        "maximum_content_segments": (
            maximum_segments
        ),
        "reference_unit_counts": dict(
            sorted(reference_units.items())
        ),
    }


def build_retrieval_corpus(
    pages: Sequence[Mapping[str, Any]],
    hierarchy: Sequence[
        Mapping[str, Any]
    ],
    spans: Sequence[Mapping[str, Any]],
) -> RetrievalBuildResult:
    """Build and audit the production coordinate-only retrieval corpus."""

    indexes = build_document_indexes(
        pages
    )
    _validate_controlled_inputs(
        pages,
        hierarchy,
        spans,
        indexes,
    )
    hierarchy_by_id = {
        record["node_id"]: record
        for record in hierarchy
    }
    exclusion_plan = (
        _build_exclusion_plan(
            hierarchy_by_id,
            spans,
            indexes,
        )
    )
    (
        logical_sources,
        omitted_empty_containers,
    ) = _build_logical_sources(
        hierarchy,
        hierarchy_by_id,
        spans,
        indexes,
        exclusion_plan,
    )
    (
        units,
        unit_coordinate_map,
        reference_sections,
        footnote_coordinates,
    ) = _materialize_units(
        logical_sources,
        hierarchy_by_id,
        indexes,
    )
    ledger, ledger_stats = _build_ledger(
        units,
        unit_coordinate_map,
        logical_sources,
        spans,
        indexes,
        exclusion_plan,
    )
    unit_stats = _audit_final_units(
        units,
        unit_coordinate_map,
        hierarchy_by_id,
        indexes,
        footnote_coordinates,
    )

    summary = {
        "schema_version": (
            SUMMARY_SCHEMA_VERSION
        ),
        "mode": (
            "production_coordinate_only_"
            "retrieval_corpus"
        ),
        "unit_count": len(units),
        "logical_source_count": len(
            logical_sources
        ),
        "ledger_record_count": len(
            ledger
        ),
        "retrieval_coordinate_count": (
            ledger_stats[
                "classification_counts"
            ]["retrieval_content"]
        ),
        "omitted_empty_container_count": (
            omitted_empty_containers
        ),
        "unit_audit": unit_stats,
        "ledger_audit": ledger_stats,
        "reference_sections": list(
            reference_sections
        ),
        "contains_retrieval_text": False,
        "contains_citation_text": False,
        "contains_embeddings": False,
        "contains_character_offsets": False,
    }

    report_lines = [
        "PolicyProof production coordinate-only retrieval corpus",
        "=" * 78,
        "",
        "Production outputs",
        "-" * 78,
        f"- retrieval units: {len(units)}",
        (
            "- logical sources: "
            f"{len(logical_sources)}"
        ),
        (
            "- coordinate-ledger records: "
            f"{len(ledger)}"
        ),
        (
            "- retrieval-content coordinates: "
            f"{ledger_stats['classification_counts']['retrieval_content']}"
        ),
        (
            "- omitted empty containers: "
            f"{omitted_empty_containers}"
        ),
        "",
        "Semantic integrity",
        "-" * 78,
        (
            "- reviewed internal boundaries: "
            f"{unit_stats['internal_boundary_count']}"
        ),
        "- semantic boundary risks: 0",
        (
            "- rejected boundaries remaining: "
            f"{unit_stats['known_rejected_boundaries_remaining']}"
        ),
        (
            "- maximum indexed words: "
            f"{unit_stats['maximum_indexed_words']}"
        ),
        (
            "- approved exception: EU recital 29 "
            "at 580 words under a 640-word ceiling"
        ),
        "",
        "Deferred materialization",
        "-" * 78,
        "- retrieval text: absent",
        "- citation text: absent",
        "- embeddings: absent",
        "- character offsets: absent",
        "",
        (
            "PASS: production retrieval units and "
            "coordinate ledger satisfy the audited contract."
        ),
    ]

    return RetrievalBuildResult(
        units=units,
        ledger=ledger,
        summary=summary,
        report="\n".join(
            report_lines
        )
        + "\n",
    )


def write_retrieval_build(
    result: RetrievalBuildResult,
    *,
    units_path: Path,
    ledger_path: Path,
    summary_path: Path,
    report_path: Path,
) -> None:
    """Write all production outputs fail-closed with rollback."""

    paths = (
        units_path,
        ledger_path,
        summary_path,
        report_path,
    )

    if len(set(paths)) != len(paths):
        raise RetrievalUnitError(
            "Production output paths must be unique."
        )

    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing:
        raise RetrievalUnitError(
            "Production output already exists: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    created: list[Path] = []

    try:
        write_jsonl_atomically(
            units_path,
            result.units,
        )
        created.append(units_path)

        write_jsonl_atomically(
            ledger_path,
            result.ledger,
        )
        created.append(ledger_path)

        write_json_atomically(
            summary_path,
            result.summary,
        )
        created.append(summary_path)

        write_text_atomically(
            report_path,
            result.report,
        )
        created.append(report_path)

    except Exception:
        for path in reversed(created):
            path.unlink(
                missing_ok=True
            )

        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PolicyProof production "
            "coordinate-only retrieval units."
        )
    )
    parser.add_argument(
        "--pages",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--spans",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--units-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    """Run the production builder from explicit input and output paths."""

    args = _parse_args()
    result = build_retrieval_corpus(
        load_jsonl(
            args.pages,
            record_name="page",
        ),
        load_jsonl(
            args.hierarchy,
            record_name="hierarchy node",
        ),
        load_jsonl(
            args.spans,
            record_name="heading span",
        ),
    )
    write_retrieval_build(
        result,
        units_path=args.units_output,
        ledger_path=args.ledger_output,
        summary_path=args.summary_output,
        report_path=args.report_output,
    )
    print(
        result.report,
        end="",
    )


if __name__ == "__main__":
    main()
