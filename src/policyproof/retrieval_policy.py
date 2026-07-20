"""Document-specific policies for deterministic retrieval-unit boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from policyproof.retrieval_units import (
    Coordinate,
    CorpusIndexes,
    DocumentIndex,
    RetrievalUnitError,
    count_words,
    document_index,
    normalize,
    unit_coordinates,
)

RMF_ID = "nist-ai-rmf-1.0"
GENAI_ID = "nist-ai-600-1-genai-profile"
EU_ID = "eu-ai-act-2024-1689"
GPT_ID = "openai-gpt-4o-system-card-2024-08-08"

TARGET_WORDS = 384
STANDARD_HARD_WORDS = 512
RECITAL_29_HARD_WORDS = 640

RMF_PAGE_PATTERN = re.compile(
    r"^Page\s+(?:\d+|[ivxlcdm]+)$",
    re.IGNORECASE,
)
STRONG_TERMINAL_PATTERN = re.compile(
    r"""[.!?][”’"'')\]]*$"""
)
LOWERCASE_START_PATTERN = re.compile(
    r"^[a-zà-öø-ÿ]"
)
URL_PATTERN = re.compile(
    r"^(?:https?://|www\.)",
    re.IGNORECASE,
)
BRACKET_ITEM_PATTERN = re.compile(
    r"^\[\d+\]\s+"
)
PAREN_ITEM_PATTERN = re.compile(
    r"^\([a-z0-9ivxlcdm]+\)\s+",
    re.IGNORECASE,
)
NUMERIC_ITEM_PATTERN = re.compile(
    r"^\d+(?:\.\d+)*(?:[.)])?\s+"
)
BULLET_PATTERN = re.compile(
    r"^(?:[-–—•▪◦*])\s+"
)
AUTHOR_INITIAL_PATTERN = re.compile(
    r"^[A-Z]\.\s+\S"
)
REFERENCE_HEADING_PATTERN = re.compile(
    r"(?:references|bibliography)",
    re.IGNORECASE,
)
GENAI_REFERENCE_START_PATTERN = re.compile(
    r"^[A-ZÀ-ÖØ-Þ0-9][^\n]{0,240}"
    r"\((?:19|20)\d{2}[a-z]?\)"
)
GENAI_FOOTNOTE_PATTERN = re.compile(
    r"^(\d{1,2})\s+\S"
)
EU_ELI_PATTERN = re.compile(
    r"^ELI:\s+"
    r"http://data\.europa\.eu/eli/reg/2024/1689/oj"
    r"\s+\d+/144$"
)
GPT_REFERENCE_START_PATTERN = re.compile(
    r"^\[(\d+)\]\s+"
)


def node_label(
    node_id: str,
    hierarchy_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the first usable source or synthetic hierarchy label."""

    node = hierarchy_by_id.get(node_id)

    if node is None:
        raise RetrievalUnitError(
            f"Unknown hierarchy node: {node_id}"
        )

    for field in (
        "full_heading",
        "synthetic_heading",
        "title",
        "label",
        "name",
    ):
        value = node.get(field)

        if isinstance(value, str) and value.strip():
            return value

    raise RetrievalUnitError(
        f"{node_id}: no usable label."
    )


def semantic_source_key(
    unit: Mapping[str, Any],
) -> str:
    """Return the source key used inside deterministic candidate IDs."""

    unit_kind = unit.get("unit_kind")

    if unit_kind in {
        "heading_body",
        "heading_only",
    }:
        source_node_id = unit.get(
            "source_node_id"
        )

        if (
            not isinstance(source_node_id, str)
            or not source_node_id
        ):
            raise RetrievalUnitError(
                f"{unit.get('unit_id')}: "
                "source_node_id is missing."
            )

        return source_node_id

    if unit_kind == "eu_recital":
        recital_number = unit.get(
            "recital_number"
        )

        if (
            not isinstance(recital_number, int)
            or isinstance(recital_number, bool)
            or recital_number < 1
        ):
            raise RetrievalUnitError(
                f"{unit.get('unit_id')}: "
                "recital_number is invalid."
            )

        return (
            f"eu-recital-{recital_number:03d}"
        )

    if unit_kind == "frontmatter_body":
        frontmatter_id = unit.get(
            "frontmatter_id"
        )

        if (
            not isinstance(frontmatter_id, str)
            or not frontmatter_id
        ):
            raise RetrievalUnitError(
                f"{unit.get('unit_id')}: "
                "frontmatter_id is missing."
            )

        return frontmatter_id

    raise RetrievalUnitError(
        f"Unexpected unit kind: {unit_kind}"
    )


def logical_source_label(
    unit: Mapping[str, Any],
    hierarchy_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the reviewed label used for packing context."""

    unit_kind = unit.get("unit_kind")

    if unit_kind in {
        "heading_body",
        "heading_only",
    }:
        source_node_id = semantic_source_key(unit)

        return node_label(
            source_node_id,
            hierarchy_by_id,
        )

    if unit_kind == "eu_recital":
        recital_number = unit.get(
            "recital_number"
        )

        if (
            not isinstance(recital_number, int)
            or isinstance(recital_number, bool)
            or recital_number < 1
        ):
            raise RetrievalUnitError(
                f"{unit.get('unit_id')}: "
                "recital_number is invalid."
            )

        return f"Recital {recital_number}"

    if unit_kind == "frontmatter_body":
        return "Executive Summary"

    raise RetrievalUnitError(
        f"Unexpected unit kind: {unit_kind}"
    )


def combined_coordinates(
    source_units: Sequence[Mapping[str, Any]],
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Combine logical-source parts with strict source-order validation."""

    if not source_units:
        raise RetrievalUnitError(
            "At least one source unit is required."
        )

    document_id = source_units[0].get(
        "document_id"
    )

    if (
        not isinstance(document_id, str)
        or not document_id
    ):
        raise RetrievalUnitError(
            "Source unit has an invalid document_id."
        )

    index = document_index(
        indexes,
        document_id,
    )
    result: list[Coordinate] = []

    for unit in source_units:
        if unit.get("document_id") != document_id:
            raise RetrievalUnitError(
                "Logical source crosses documents."
            )

        result.extend(
            unit_coordinates(unit, indexes)
        )

    seen: set[Coordinate] = set()
    previous_index = -1

    for coordinate in result:
        if coordinate in seen:
            raise RetrievalUnitError(
                f"{document_id}: duplicate source "
                f"coordinate {coordinate}."
            )

        current_index = (
            index.coordinate_indexes[
                coordinate
            ]
        )

        if current_index <= previous_index:
            raise RetrievalUnitError(
                f"{document_id}: source coordinates "
                "are not strictly ordered."
            )

        seen.add(coordinate)
        previous_index = current_index

    return tuple(result)


def previous_nonblank_text(
    index: DocumentIndex,
    coordinate: Coordinate,
) -> tuple[Coordinate, str] | None:
    """Return the previous nonblank source line in document order."""

    try:
        position = index.coordinate_indexes[
            coordinate
        ]
    except KeyError as error:
        raise RetrievalUnitError(
            f"{index.document_id}: unknown coordinate "
            f"{coordinate}."
        ) from error

    for candidate in reversed(
        index.coordinates[:position]
    ):
        text = index.line_text[candidate]

        if text.strip():
            return candidate, text

    return None


def discover_genai_footnotes(
    indexes: CorpusIndexes,
) -> tuple[
    tuple[dict[str, Any], ...],
    frozenset[Coordinate],
]:
    """Detect reviewed GenAI footnote starts without treating page numbers as footnotes."""

    index = document_index(
        indexes,
        GENAI_ID,
    )
    records: list[dict[str, Any]] = []
    coordinates: set[Coordinate] = set()

    for coordinate in index.coordinates:
        text = index.line_text[coordinate]
        match = GENAI_FOOTNOTE_PATTERN.match(
            text.strip()
        )

        if match is None:
            continue

        page_number, line_number = coordinate

        if (
            line_number == 2
            and text.strip().isdigit()
            and int(text.strip())
            == page_number - 4
        ):
            continue

        footnote_number = match.group(1)
        previous = previous_nonblank_text(
            index,
            coordinate,
        )
        linked = False
        previous_coordinate = None
        previous_text = None

        if previous is not None:
            (
                previous_coordinate,
                previous_text,
            ) = previous
            linked = bool(
                re.search(
                    rf"{re.escape(footnote_number)}"
                    r"""[”’"'')\]]*$""",
                    previous_text.rstrip(),
                )
            )

        records.append(
            {
                "number": int(
                    footnote_number
                ),
                "coordinate": coordinate,
                "text": text,
                "previous_coordinate": (
                    previous_coordinate
                ),
                "previous_text": previous_text,
                "linked_to_previous_marker": linked,
            }
        )
        coordinates.add(coordinate)

    return (
        tuple(records),
        frozenset(coordinates),
    )


def is_genai_footnote(
    document_id: str,
    coordinate: Coordinate,
    genai_footnote_coordinates: frozenset[
        Coordinate
    ],
) -> bool:
    """Return whether a coordinate is a reviewed GenAI footnote start."""

    return (
        document_id == GENAI_ID
        and coordinate
        in genai_footnote_coordinates
    )


def is_eu_eli(
    document_id: str,
    coordinate: Coordinate,
    indexes: CorpusIndexes,
) -> bool:
    """Return whether a coordinate is the EU ELI footer."""

    if document_id != EU_ID:
        return False

    index = document_index(
        indexes,
        document_id,
    )

    try:
        text = index.line_text[coordinate]
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown coordinate "
            f"{coordinate}."
        ) from error

    return bool(
        EU_ELI_PATTERN.fullmatch(
            normalize(text)
        )
    )


def cleaned_coordinates(
    document_id: str,
    coordinates: Sequence[Coordinate],
    indexes: CorpusIndexes,
) -> tuple[Coordinate, ...]:
    """Remove blank source lines and EU ELI page furniture."""

    index = document_index(
        indexes,
        document_id,
    )

    return tuple(
        coordinate
        for coordinate in coordinates
        if (
            index.line_text[
                coordinate
            ].strip()
            and not is_eu_eli(
                document_id,
                coordinate,
                indexes,
            )
        )
    )


def accepted_structured_start(
    document_id: str,
    coordinate: Coordinate,
    indexes: CorpusIndexes,
    genai_footnote_coordinates: frozenset[
        Coordinate
    ],
) -> bool:
    """Return whether a line begins an accepted structured item."""

    index = document_index(
        indexes,
        document_id,
    )

    try:
        text = index.line_text[
            coordinate
        ].strip()
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown coordinate "
            f"{coordinate}."
        ) from error

    if is_genai_footnote(
        document_id,
        coordinate,
        genai_footnote_coordinates,
    ):
        return False

    if AUTHOR_INITIAL_PATTERN.match(text):
        return False

    return bool(
        BRACKET_ITEM_PATTERN.match(text)
        or PAREN_ITEM_PATTERN.match(text)
        or NUMERIC_ITEM_PATTERN.match(text)
        or BULLET_PATTERN.match(text)
    )


def gap_is_blank_only(
    document_id: str,
    left: Coordinate,
    right: Coordinate,
    indexes: CorpusIndexes,
) -> bool:
    """Return whether only blank lines occur between two coordinates."""

    index = document_index(
        indexes,
        document_id,
    )

    try:
        left_index = (
            index.coordinate_indexes[left]
        )
        right_index = (
            index.coordinate_indexes[right]
        )
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown coordinate "
            f"{error.args[0]}."
        ) from error

    if right_index <= left_index + 1:
        return False

    intervening = index.coordinates[
        left_index + 1:
        right_index
    ]

    return bool(intervening) and all(
        not index.line_text[
            coordinate
        ].strip()
        for coordinate in intervening
    )


def safe_boundary_reason(
    document_id: str,
    coordinates: Sequence[Coordinate],
    end_index: int,
    indexes: CorpusIndexes,
    genai_footnote_coordinates: frozenset[
        Coordinate
    ],
) -> str | None:
    """Return the reviewed boundary method after one candidate line."""

    if (
        end_index < 0
        or end_index >= len(coordinates)
    ):
        raise RetrievalUnitError(
            f"Boundary index {end_index} is outside "
            f"{len(coordinates)} coordinates."
        )

    if end_index + 1 >= len(coordinates):
        return "end_of_source_unit"

    index = document_index(
        indexes,
        document_id,
    )
    current = coordinates[end_index]
    following = coordinates[
        end_index + 1
    ]
    current_text = index.line_text[
        current
    ].rstrip()
    following_text = index.line_text[
        following
    ].lstrip()

    following_lowercase = bool(
        LOWERCASE_START_PATTERN.match(
            following_text
        )
    )
    following_url = bool(
        URL_PATTERN.match(
            following_text
        )
    )
    following_footnote = (
        is_genai_footnote(
            document_id,
            following,
            genai_footnote_coordinates,
        )
    )
    structured_following = (
        accepted_structured_start(
            document_id,
            following,
            indexes,
            genai_footnote_coordinates,
        )
    )

    if (
        STRONG_TERMINAL_PATTERN.search(
            current_text
        )
        and not following_lowercase
        and not following_url
        and not following_footnote
    ):
        return "after_strong_terminal"

    if (
        structured_following
        and not current_text.endswith(":")
        and not following_footnote
    ):
        return "before_structured_start"

    if (
        gap_is_blank_only(
            document_id,
            current,
            following,
            indexes,
        )
        and not following_lowercase
        and not following_url
        and not following_footnote
        and not current_text.endswith(":")
    ):
        return "at_blank_paragraph_gap"

    return None


def repack_nonreference(
    document_id: str,
    coordinates: Sequence[Coordinate],
    *,
    context_words: int,
    hard_words: int,
    indexes: CorpusIndexes,
    genai_footnote_coordinates: frozenset[
        Coordinate
    ],
) -> dict[str, Any]:
    """Pack one non-reference logical source at reviewed semantic boundaries."""

    if not coordinates:
        return {
            "passed": False,
            "reason": "no_coordinates",
            "pieces": [],
        }

    if context_words < 0:
        raise RetrievalUnitError(
            "context_words cannot be negative."
        )

    if hard_words < 1:
        raise RetrievalUnitError(
            "hard_words must be positive."
        )

    index = document_index(
        indexes,
        document_id,
    )
    word_counts = [
        count_words(
            index.line_text[coordinate]
        )
        for coordinate in coordinates
    ]
    hard_content_words = (
        hard_words - context_words
    )

    if hard_content_words < 1:
        return {
            "passed": False,
            "reason": (
                "context_exhausts_hard_budget"
            ),
            "pieces": [],
        }

    target_content_words = max(
        1,
        TARGET_WORDS - context_words,
    )
    pieces: list[dict[str, Any]] = []
    start = 0

    while start < len(coordinates):
        remaining_words = sum(
            word_counts[start:]
        )

        if remaining_words <= hard_content_words:
            pieces.append(
                {
                    "start": coordinates[start],
                    "end": coordinates[-1],
                    "content_words": (
                        remaining_words
                    ),
                    "indexed_words": (
                        remaining_words
                        + context_words
                    ),
                    "boundary_after": (
                        "end_of_source_unit"
                    ),
                }
            )
            break

        running = 0
        hard_end = start - 1

        for index_number in range(
            start,
            len(coordinates),
        ):
            next_words = word_counts[
                index_number
            ]

            if (
                running + next_words
                > hard_content_words
            ):
                break

            running += next_words
            hard_end = index_number

        if hard_end < start:
            return {
                "passed": False,
                "reason": (
                    "single_line_over_budget"
                ),
                "failure_start": (
                    coordinates[start]
                ),
                "pieces": pieces,
            }

        target_running = 0
        target_end = start

        while (
            target_end <= hard_end
            and target_running
            < target_content_words
        ):
            target_running += word_counts[
                target_end
            ]
            target_end += 1

        target_end = min(
            hard_end,
            max(
                start,
                target_end - 1,
            ),
        )

        candidates = [
            (
                index_number,
                safe_boundary_reason(
                    document_id,
                    coordinates,
                    index_number,
                    indexes,
                    genai_footnote_coordinates,
                ),
            )
            for index_number in range(
                start,
                hard_end + 1,
            )
        ]
        candidates = [
            (index_number, reason)
            for index_number, reason
            in candidates
            if reason is not None
        ]
        preferred = [
            item
            for item in candidates
            if item[0] >= target_end
        ]

        if preferred:
            (
                selected_index,
                reason,
            ) = preferred[0]
        elif candidates:
            (
                selected_index,
                reason,
            ) = candidates[-1]
        else:
            return {
                "passed": False,
                "reason": (
                    "no_corrected_safe_boundary"
                ),
                "failure_start": (
                    coordinates[start]
                ),
                "hard_end": (
                    coordinates[hard_end]
                ),
                "pieces": pieces,
            }

        content_words = sum(
            word_counts[
                start:
                selected_index + 1
            ]
        )
        pieces.append(
            {
                "start": coordinates[start],
                "end": coordinates[
                    selected_index
                ],
                "content_words": (
                    content_words
                ),
                "indexed_words": (
                    content_words
                    + context_words
                ),
                "boundary_after": reason,
            }
        )
        start = selected_index + 1

    return {
        "passed": True,
        "pieces": pieces,
    }


def reference_entry_groups(
    document_id: str,
    label: str,
    coordinates: Sequence[Coordinate],
    indexes: CorpusIndexes,
) -> dict[str, Any]:
    """Identify complete bibliography entries and pack them without splitting."""

    if not coordinates:
        return {
            "passed": False,
            "reason": "no_coordinates",
            "prefix_nonblank": [],
            "start_candidates": [],
            "entries": [],
            "oversized_entries": [],
            "packs": [],
        }

    index = document_index(
        indexes,
        document_id,
    )
    starts: list[
        tuple[int, int | None]
    ] = []

    if document_id == GPT_ID:
        for coordinate_index, coordinate in enumerate(
            coordinates
        ):
            match = (
                GPT_REFERENCE_START_PATTERN.match(
                    index.line_text[
                        coordinate
                    ].strip()
                )
            )

            if match is not None:
                starts.append(
                    (
                        coordinate_index,
                        int(match.group(1)),
                    )
                )

        if not starts:
            return {
                "passed": False,
                "reason": (
                    "no_reference_entry_starts"
                ),
                "prefix_nonblank": [],
                "start_candidates": [],
                "entries": [],
                "oversized_entries": [],
                "packs": [],
            }

        numbers = [
            number
            for _, number in starts
            if number is not None
        ]

        if numbers != list(
            range(
                numbers[0],
                numbers[-1] + 1,
            )
        ):
            return {
                "passed": False,
                "reason": (
                    "gpt_reference_sequence_"
                    "not_contiguous"
                ),
                "start_candidates": starts,
                "prefix_nonblank": [],
                "entries": [],
                "oversized_entries": [],
                "packs": [],
            }

    elif document_id == GENAI_ID:
        for coordinate_index, coordinate in enumerate(
            coordinates
        ):
            text = index.line_text[
                coordinate
            ].strip()

            if (
                not URL_PATTERN.match(text)
                and (
                    GENAI_REFERENCE_START_PATTERN.search(
                        text
                    )
                )
            ):
                starts.append(
                    (
                        coordinate_index,
                        None,
                    )
                )

        if not starts:
            return {
                "passed": False,
                "reason": (
                    "no_reference_entry_starts"
                ),
                "prefix_nonblank": [],
                "start_candidates": [],
                "entries": [],
                "oversized_entries": [],
                "packs": [],
            }

    else:
        raise RetrievalUnitError(
            f"Unexpected reference document: "
            f"{document_id}"
        )

    prefix_coordinates = coordinates[
        :starts[0][0]
    ]
    prefix_nonblank = [
        coordinate
        for coordinate in prefix_coordinates
        if index.line_text[
            coordinate
        ].strip()
    ]
    entries: list[dict[str, Any]] = []

    for entry_number, (
        start_index,
        source_number,
    ) in enumerate(
        starts,
        start=1,
    ):
        stop_index = (
            starts[entry_number][0]
            if entry_number < len(starts)
            else len(coordinates)
        )
        entry_coordinates = list(
            coordinates[
                start_index:
                stop_index
            ]
        )
        words = sum(
            count_words(
                index.line_text[
                    coordinate
                ]
            )
            for coordinate
            in entry_coordinates
        )

        entries.append(
            {
                "entry_number": entry_number,
                "source_number": (
                    source_number
                ),
                "start": (
                    entry_coordinates[0]
                ),
                "end": (
                    entry_coordinates[-1]
                ),
                "word_count": words,
                "coordinates": (
                    entry_coordinates
                ),
            }
        )

    context_words = count_words(label)
    hard_content_words = (
        STANDARD_HARD_WORDS
        - context_words
    )
    target_content_words = max(
        1,
        TARGET_WORDS - context_words,
    )
    oversized_entries = [
        entry
        for entry in entries
        if entry["word_count"]
        > hard_content_words
    ]
    packs: list[dict[str, Any]] = []
    current_entries: list[
        dict[str, Any]
    ] = []
    current_words = 0

    for entry in entries:
        if (
            current_entries
            and current_words
            + entry["word_count"]
            > hard_content_words
        ):
            packs.append(
                {
                    "entries": (
                        current_entries
                    ),
                    "content_words": (
                        current_words
                    ),
                }
            )
            current_entries = []
            current_words = 0

        current_entries.append(entry)
        current_words += entry[
            "word_count"
        ]

        if (
            current_words
            >= target_content_words
        ):
            packs.append(
                {
                    "entries": (
                        current_entries
                    ),
                    "content_words": (
                        current_words
                    ),
                }
            )
            current_entries = []
            current_words = 0

    if current_entries:
        packs.append(
            {
                "entries": current_entries,
                "content_words": (
                    current_words
                ),
            }
        )

    for pack in packs:
        pack["indexed_words"] = (
            pack["content_words"]
            + context_words
        )
        pack["start"] = pack[
            "entries"
        ][0]["start"]
        pack["end"] = pack[
            "entries"
        ][-1]["end"]

    passed = (
        not prefix_nonblank
        and not oversized_entries
        and all(
            pack["indexed_words"]
            <= STANDARD_HARD_WORDS
            for pack in packs
        )
    )

    result = {
        "passed": passed,
        "prefix_nonblank": (
            prefix_nonblank
        ),
        "start_candidates": starts,
        "entries": entries,
        "oversized_entries": (
            oversized_entries
        ),
        "packs": packs,
    }

    if not passed:
        if prefix_nonblank:
            result["reason"] = (
                "nonblank_reference_prefix"
            )
        elif oversized_entries:
            result["reason"] = (
                "oversized_reference_entry"
            )
        else:
            result["reason"] = (
                "reference_pack_over_budget"
            )

    return result


def reference_start_is_valid(
    document_id: str,
    coordinate: Coordinate,
    indexes: CorpusIndexes,
) -> bool:
    """Return whether a unit starts at a valid bibliography entry."""

    index = document_index(
        indexes,
        document_id,
    )

    try:
        text = index.line_text[
            coordinate
        ].strip()
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown coordinate "
            f"{coordinate}."
        ) from error

    if document_id == GENAI_ID:
        return bool(
            not URL_PATTERN.match(text)
            and (
                GENAI_REFERENCE_START_PATTERN.search(
                    text
                )
            )
        )

    if document_id == GPT_ID:
        return bool(
            GPT_REFERENCE_START_PATTERN.match(
                text
            )
        )

    return False


def explicit_furniture_reason(
    document_id: str,
    coordinate: Coordinate,
    indexes: CorpusIndexes,
) -> str | None:
    """Return the reviewed page-furniture exclusion reason."""

    index = document_index(
        indexes,
        document_id,
    )

    try:
        text = normalize(
            index.line_text[coordinate]
        )
    except KeyError as error:
        raise RetrievalUnitError(
            f"{document_id}: unknown coordinate "
            f"{coordinate}."
        ) from error

    page_number, line_number = coordinate
    page_line_count = (
        index.page_line_counts[
            page_number
        ]
    )

    if document_id == RMF_ID:
        if text == "NIST AI 100-1 AI RMF 1.0":
            return "rmf_running_header"

        if RMF_PAGE_PATTERN.fullmatch(text):
            return "rmf_page_label"

        if text == "Categories Subcategories":
            return "rmf_table_header"

        if text == "Continued on next page":
            return "rmf_table_continuation"

        return None

    if document_id == GENAI_ID:
        if (
            line_number == 2
            and text.isdigit()
            and int(text) == page_number - 4
        ):
            return "genai_page_number"

        return None

    if document_id == EU_ID:
        if (
            line_number <= 4
            or line_number
            > page_line_count - 4
        ):
            if text == "EN":
                return "eu_language_header"

            if text == "OJ L, 12.7.2024":
                return "eu_journal_header"

        return None

    if document_id == GPT_ID:
        if (
            line_number == page_line_count
            and text == str(page_number)
        ):
            return "gpt_page_number"

        return None

    raise RetrievalUnitError(
        f"Unknown document ID: {document_id}"
    )
