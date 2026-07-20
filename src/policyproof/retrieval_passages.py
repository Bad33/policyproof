"""Build token-safe retrieval passages without changing coordinate ownership."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policyproof.retrieval_policy import (
    EU_ID,
    REFERENCE_HEADING_PATTERN,
    discover_genai_footnotes,
    logical_source_label,
    reference_entry_groups,
    safe_boundary_reason,
)
from policyproof.retrieval_tokenizer import (
    TOKENIZER_CONTRACT,
    count_tokens,
)
from policyproof.retrieval_units import (
    Coordinate,
    CorpusIndexes,
    RetrievalUnitError,
    build_document_indexes,
    coordinate_record,
    document_index,
    group_units_by_logical_source,
    load_jsonl,
    segment_coordinates,
    unit_coordinates,
    write_json_atomically,
    write_jsonl_atomically,
    write_text_atomically,
)

PASSAGE_SCHEMA_VERSION = "1.0"
QUERY_TOKEN_RESERVATION = 64
PAIR_SPECIAL_TOKEN_COUNT = 3
PASSAGE_HARD_TOKEN_LIMIT = 445
PASSAGE_TARGET_TOKEN_COUNT = 384

if (
    QUERY_TOKEN_RESERVATION
    + PAIR_SPECIAL_TOKEN_COUNT
    + PASSAGE_HARD_TOKEN_LIMIT
    != TOKENIZER_CONTRACT.model_max_length
):
    raise RuntimeError(
        "Retrieval passage token budgets do not match "
        "the pinned tokenizer model length."
    )


@dataclass(frozen=True)
class ReviewedIntralineAnchor:
    """One reviewed sentence boundary inside an extracted source line."""

    source_key: str
    coordinate: Coordinate
    split_offset: int
    left_suffix: str
    right_prefix: str


@dataclass(frozen=True)
class PassageAtom:
    """One indivisible source-text range."""

    coordinate: Coordinate
    start_char: int
    end_char: int
    text: str
    endpoint_reason: str | None


@dataclass(frozen=True)
class PassageEndpoint:
    """One permitted passage endpoint."""

    atom_index: int
    reason: str
    reference_entry_ordinal: int | None = None


def _recital_source_key(number: int) -> str:
    return (
        f"candidate-v2:{EU_ID}:"
        f"eu-recital-{number:03d}"
    )


REVIEWED_INTRALINE_ANCHORS = (
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(9),
        coordinate=(3, 12),
        split_offset=74,
        left_suffix=(
            "Council Directive 85/374/EEC (10) remain "
            "unaffected and fully applicable. "
        ),
        right_prefix="Furthermore, in the context of",
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(22),
        coordinate=(6, 37),
        split_offset=90,
        left_suffix=(
            "without that AI system being placed on the "
            "market, put into service or used in the Union. "
        ),
        right_prefix="To prevent the",
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(27),
        coordinate=(8, 13),
        split_offset=82,
        left_suffix=(
            "functioning in a way that can be appropriately "
            "controlled and overseen by humans. "
        ),
        right_prefix="Technical robustness and safety",
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(29),
        coordinate=(8, 51),
        split_offset=41,
        left_suffix=(
            "poverty, ethnic or religious minorities. "
        ),
        right_prefix=(
            "Such AI systems can be placed on the market"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(33),
        coordinate=(10, 12),
        split_offset=14,
        left_suffix="consequences. ",
        right_prefix=(
            "An imminent threat to life or the physical safety"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(53),
        coordinate=(15, 11),
        split_offset=21,
        left_suffix="proper human review. ",
        right_prefix=(
            "Such AI systems include for instance those that"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(58),
        coordinate=(16, 46),
        split_offset=27,
        left_suffix="legal and natural persons. ",
        right_prefix=(
            "In addition, AI systems used to evaluate"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(64),
        coordinate=(18, 42),
        split_offset=53,
        left_suffix=(
            "effective to meet the objectives of this "
            "Regulation. "
        ),
        right_prefix=(
            "Based on the New Legislative Framework"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(65),
        coordinate=(19, 19),
        split_offset=68,
        left_suffix=(
            "significant decisions and actions taken subject "
            "to this Regulation. "
        ),
        right_prefix=(
            "This process should ensure that the provider"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(111),
        coordinate=(29, 26),
        split_offset=57,
        left_suffix=(
            "pre-training, synthetic data generation and "
            "fine-tuning. "
        ),
        right_prefix=(
            "Therefore, an initial threshold of floating point "
            "operations"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(131),
        coordinate=(33, 25),
        split_offset=77,
        left_suffix=(
            "themselves in such database and select the system "
            "that they envisage to use. "
        ),
        right_prefix=(
            "Other deployers should be entitled to do"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(139),
        coordinate=(35, 28),
        split_offset=45,
        left_suffix=(
            "accessibility for SMEs, including start-ups. "
        ),
        right_prefix=(
            "The participation in the AI regulatory sandbox"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(141),
        coordinate=(36, 12),
        split_offset=36,
        left_suffix=(
            "providers or prospective providers. "
        ),
        right_prefix=(
            "Such guarantees should include, inter alia"
        ),
    ),
    ReviewedIntralineAnchor(
        source_key=_recital_source_key(143),
        coordinate=(36, 49),
        split_offset=71,
        left_suffix=(
            "and responding to queries about the implementation "
            "of this Regulation. "
        ),
        right_prefix=(
            "Where appropriate, these channels should"
        ),
    ),
)


def validate_reviewed_intraline_anchors(
    indexes: CorpusIndexes,
) -> Mapping[str, ReviewedIntralineAnchor]:
    """Validate all corpus-specific anchors against exact source text."""

    index = document_index(indexes, EU_ID)
    anchors: dict[str, ReviewedIntralineAnchor] = {}

    for anchor in REVIEWED_INTRALINE_ANCHORS:
        if anchor.source_key in anchors:
            raise RetrievalUnitError(
                "Duplicate reviewed intraline source key: "
                f"{anchor.source_key}."
            )

        if anchor.coordinate not in index.line_text:
            raise RetrievalUnitError(
                f"{anchor.source_key}: reviewed anchor "
                f"coordinate {anchor.coordinate} is missing."
            )

        text = index.line_text[anchor.coordinate]

        if (
            anchor.split_offset < 1
            or anchor.split_offset >= len(text)
        ):
            raise RetrievalUnitError(
                f"{anchor.source_key}: invalid reviewed "
                f"split offset {anchor.split_offset}."
            )

        left = text[: anchor.split_offset]
        right = text[anchor.split_offset :]

        if not left.endswith(anchor.left_suffix):
            raise RetrievalUnitError(
                f"{anchor.source_key}: reviewed anchor left "
                "context no longer matches the source."
            )

        if not right.startswith(anchor.right_prefix):
            raise RetrievalUnitError(
                f"{anchor.source_key}: reviewed anchor right "
                "context no longer matches the source."
            )

        anchors[anchor.source_key] = anchor

    return anchors


def _source_atoms(
    document_id: str,
    source_key: str,
    coordinates: Sequence[Coordinate],
    indexes: CorpusIndexes,
    footnote_coordinates: frozenset[Coordinate],
    anchor: ReviewedIntralineAnchor | None,
) -> tuple[PassageAtom, ...]:
    """Split one source into lines plus its reviewed intraline anchor."""

    index = document_index(indexes, document_id)
    atoms: list[PassageAtom] = []

    for line_position, coordinate in enumerate(coordinates):
        text = index.line_text[coordinate]
        offsets = [0, len(text)]

        if anchor is not None and coordinate == anchor.coordinate:
            offsets.insert(1, anchor.split_offset)

        for atom_position, (start, end) in enumerate(
            zip(offsets, offsets[1:])
        ):
            is_last_fragment = (
                atom_position == len(offsets) - 2
            )
            is_last_coordinate = (
                line_position == len(coordinates) - 1
            )

            if not is_last_fragment:
                endpoint_reason = "after_sentence_in_line"
            elif is_last_coordinate:
                endpoint_reason = "end_of_source_unit"
            else:
                endpoint_reason = safe_boundary_reason(
                    document_id,
                    coordinates,
                    line_position,
                    indexes,
                    footnote_coordinates,
                )

            atoms.append(
                PassageAtom(
                    coordinate=coordinate,
                    start_char=start,
                    end_char=end,
                    text=text[start:end],
                    endpoint_reason=endpoint_reason,
                )
            )

    if not atoms:
        raise RetrievalUnitError(
            f"{source_key}: no passage atoms."
        )

    if (
        anchor is not None
        and not any(
            atom.coordinate == anchor.coordinate
            and atom.end_char == anchor.split_offset
            and atom.endpoint_reason
            == "after_sentence_in_line"
            for atom in atoms
        )
    ):
        raise RetrievalUnitError(
            f"{source_key}: reviewed anchor was not applied."
        )

    return tuple(atoms)


def _serialize_atom_range(
    document_id: str,
    atoms: Sequence[PassageAtom],
    start: int,
    end: int,
    indexes: CorpusIndexes,
) -> list[dict[str, Any]]:
    """Serialize one contiguous atom range as source slices."""

    if start < 0 or end < start or end >= len(atoms):
        raise RetrievalUnitError(
            "Invalid passage atom range."
        )

    index = document_index(indexes, document_id)
    groups: list[list[PassageAtom]] = [
        [atoms[start]]
    ]

    for atom in atoms[start + 1 : end + 1]:
        previous = groups[-1][-1]
        previous_position = index.coordinate_indexes[
            previous.coordinate
        ]
        position = index.coordinate_indexes[
            atom.coordinate
        ]

        if (
            atom.coordinate == previous.coordinate
            or position == previous_position + 1
        ):
            groups[-1].append(atom)
        else:
            groups.append([atom])

    slices: list[dict[str, Any]] = []

    for group in groups:
        first = group[0]
        last = group[-1]
        record: dict[str, Any] = {
            "included_start": coordinate_record(
                index,
                first.coordinate,
            ),
            "included_end": coordinate_record(
                index,
                last.coordinate,
            ),
        }

        if first.start_char:
            record["start_char_offset"] = (
                first.start_char
            )

        last_line_length = len(
            index.line_text[last.coordinate]
        )

        if last.end_char != last_line_length:
            record["end_char_offset"] = last.end_char

        slices.append(record)

    return slices


def _validated_offset(
    value: Any,
    *,
    field_name: str,
    default: int,
) -> int:
    if value is None:
        return default

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise RetrievalUnitError(
            f"Invalid {field_name}: {value!r}."
        )

    return value


def materialize_passage_text(
    passage: Mapping[str, Any],
    indexes: CorpusIndexes,
) -> str:
    """Materialize one token-safe passage from source slices."""

    label = passage.get("label")
    unit_kind = passage.get("unit_kind")
    document_id = passage.get("document_id")
    slices = passage.get("source_slices")

    if not isinstance(label, str) or not label:
        raise RetrievalUnitError(
            "Passage has no valid label."
        )

    if unit_kind == "heading_only":
        return label

    if (
        not isinstance(document_id, str)
        or not document_id
    ):
        raise RetrievalUnitError(
            "Passage has no valid document_id."
        )

    if not isinstance(slices, list) or not slices:
        raise RetrievalUnitError(
            "Passage source_slices must be nonempty."
        )

    index = document_index(indexes, document_id)
    texts: list[str] = []
    expanded: list[tuple[Coordinate, ...]] = []

    for source_slice in slices:
        if not isinstance(source_slice, Mapping):
            raise RetrievalUnitError(
                "Passage source slice is not an object."
            )

        coordinates = segment_coordinates(
            document_id,
            source_slice,
            indexes,
        )
        first_coordinate = coordinates[0]
        last_coordinate = coordinates[-1]
        first_text = index.line_text[first_coordinate]
        last_text = index.line_text[last_coordinate]

        start_offset = _validated_offset(
            source_slice.get("start_char_offset"),
            field_name="start_char_offset",
            default=0,
        )
        end_offset = _validated_offset(
            source_slice.get("end_char_offset"),
            field_name="end_char_offset",
            default=len(last_text),
        )

        if start_offset > len(first_text):
            raise RetrievalUnitError(
                "Passage start offset exceeds source line."
            )

        if end_offset > len(last_text):
            raise RetrievalUnitError(
                "Passage end offset exceeds source line."
            )

        if (
            first_coordinate == last_coordinate
            and end_offset <= start_offset
        ):
            raise RetrievalUnitError(
                "Passage offsets create an empty source slice."
            )

        lines: list[str] = []

        for line_position, coordinate in enumerate(
            coordinates
        ):
            text = index.line_text[coordinate]
            start = (
                start_offset
                if line_position == 0
                else 0
            )
            end = (
                end_offset
                if line_position
                == len(coordinates) - 1
                else len(text)
            )
            lines.append(text[start:end].strip())

        slice_text = "\n".join(lines)

        if not slice_text:
            raise RetrievalUnitError(
                "Passage source slice materialized empty."
            )

        texts.append(slice_text)
        expanded.append(coordinates)

    body_parts: list[str] = []

    for position, text in enumerate(texts):
        if position:
            left = expanded[position - 1]
            right = expanded[position]
            separator = (
                "\n\n"
                if left[-1][0] == right[0][0]
                else "\n"
            )
            body_parts.append(separator)

        body_parts.append(text)

    body = "".join(body_parts)
    return f"{label}\n\n{body}"


def _reference_endpoints(
    document_id: str,
    label: str,
    coordinates: Sequence[Coordinate],
    atoms: Sequence[PassageAtom],
    indexes: CorpusIndexes,
) -> tuple[PassageEndpoint, ...]:
    result = reference_entry_groups(
        document_id,
        label,
        coordinates,
        indexes,
    )

    if not result["passed"]:
        raise RetrievalUnitError(
            "Reference passage parsing failed: "
            f"{result.get('reason')}."
        )

    final_atom_by_coordinate: dict[
        Coordinate,
        int,
    ] = {}

    for atom_index, atom in enumerate(atoms):
        final_atom_by_coordinate[
            atom.coordinate
        ] = atom_index

    endpoints: list[PassageEndpoint] = []

    for entry_position, entry in enumerate(
        result["entries"],
        start=1,
    ):
        coordinate = entry["coordinates"][-1]

        try:
            atom_index = final_atom_by_coordinate[
                coordinate
            ]
        except KeyError as error:
            raise RetrievalUnitError(
                "Reference entry endpoint is absent from "
                "passage atoms."
            ) from error

        reason = (
            "end_of_source_unit"
            if entry_position
            == len(result["entries"])
            else "before_reference_entry"
        )
        endpoints.append(
            PassageEndpoint(
                atom_index=atom_index,
                reason=reason,
                reference_entry_ordinal=entry_position,
            )
        )

    return tuple(endpoints)


def _ordinary_endpoints(
    atoms: Sequence[PassageAtom],
) -> tuple[PassageEndpoint, ...]:
    return tuple(
        PassageEndpoint(
            atom_index=atom_index,
            reason=atom.endpoint_reason,
        )
        for atom_index, atom in enumerate(atoms)
        if atom.endpoint_reason is not None
    )


def _minimum_partition(
    *,
    document_id: str,
    unit_kind: str,
    label: str,
    atoms: Sequence[PassageAtom],
    endpoints: Sequence[PassageEndpoint],
    indexes: CorpusIndexes,
) -> tuple[int, ...]:
    """Choose the fewest token-safe pieces over permitted endpoints."""

    if not endpoints:
        raise RetrievalUnitError(
            "Logical source has no permitted passage endpoint."
        )

    if endpoints[-1].atom_index != len(atoms) - 1:
        raise RetrievalUnitError(
            "Final passage endpoint does not cover the source."
        )

    best_cost: list[
        tuple[int, int, int] | None
    ] = [None] * len(endpoints)
    previous = [-2] * len(endpoints)
    token_cache: dict[tuple[int, int], int] = {}

    def range_tokens(start: int, end: int) -> int:
        key = (start, end)

        if key not in token_cache:
            source_slices = _serialize_atom_range(
                document_id,
                atoms,
                start,
                end,
                indexes,
            )
            text = materialize_passage_text(
                {
                    "document_id": document_id,
                    "unit_kind": unit_kind,
                    "label": label,
                    "source_slices": source_slices,
                },
                indexes,
            )
            token_cache[key] = count_tokens(
                text,
                add_special_tokens=False,
            )

        return token_cache[key]

    for right_position, endpoint in enumerate(
        endpoints
    ):
        for left_position in range(
            -1,
            right_position,
        ):
            if left_position == -1:
                start_atom = 0
                prior_cost = (0, 0, 0)
            else:
                if best_cost[left_position] is None:
                    continue

                start_atom = (
                    endpoints[left_position].atom_index
                    + 1
                )
                prior_cost = best_cost[left_position]

            token_count = range_tokens(
                start_atom,
                endpoint.atom_index,
            )

            if token_count > PASSAGE_HARD_TOKEN_LIMIT:
                continue

            intraline_boundary = int(
                endpoint.reason
                == "after_sentence_in_line"
            )
            candidate_cost = (
                prior_cost[0] + 1,
                prior_cost[1] + intraline_boundary,
                prior_cost[2]
                + abs(
                    PASSAGE_TARGET_TOKEN_COUNT
                    - token_count
                ),
            )

            if (
                best_cost[right_position] is None
                or candidate_cost
                < best_cost[right_position]
            ):
                best_cost[right_position] = (
                    candidate_cost
                )
                previous[right_position] = (
                    left_position
                )

    if best_cost[-1] is None:
        raise RetrievalUnitError(
            "No token-safe partition exists over the "
            "permitted passage boundaries."
        )

    path: list[int] = []
    cursor = len(endpoints) - 1

    while cursor >= 0:
        path.append(cursor)
        cursor = previous[cursor]

    path.reverse()
    return tuple(path)


def build_retrieval_passages(
    units: Sequence[dict[str, Any]],
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    indexes: CorpusIndexes,
) -> tuple[dict[str, Any], ...]:
    """Build token-safe derived passages from accepted retrieval units."""

    grouped = group_units_by_logical_source(units)
    reviewed_anchors = (
        validate_reviewed_intraline_anchors(
            indexes
        )
    )
    _, footnote_coordinates = (
        discover_genai_footnotes(indexes)
    )

    passages: list[dict[str, Any]] = []
    selected_anchor_sources: set[str] = set()

    for group_key in grouped.ordered_keys:
        source_units = grouped.groups[group_key]
        first = source_units[0]
        document_id = first["document_id"]
        source_key = group_key[1]
        unit_kind = first["unit_kind"]
        label = logical_source_label(
            first,
            hierarchy_by_id,
        )
        coordinates = tuple(
            coordinate
            for unit in source_units
            for coordinate in unit_coordinates(
                unit,
                indexes,
            )
        )
        anchor = reviewed_anchors.get(source_key)
        atoms = _source_atoms(
            document_id,
            source_key,
            coordinates,
            indexes,
            footnote_coordinates,
            anchor,
        )

        if unit_kind == "heading_only":
            endpoints = (
                PassageEndpoint(
                    atom_index=len(atoms) - 1,
                    reason="end_of_heading_source",
                ),
            )
        elif REFERENCE_HEADING_PATTERN.search(
            label
        ):
            endpoints = _reference_endpoints(
                document_id,
                label,
                coordinates,
                atoms,
                indexes,
            )
        else:
            endpoints = _ordinary_endpoints(
                atoms
            )

        path = _minimum_partition(
            document_id=document_id,
            unit_kind=unit_kind,
            label=label,
            atoms=atoms,
            endpoints=endpoints,
            indexes=indexes,
        )
        previous_atom = -1
        previous_endpoint_position = -1

        for passage_number, endpoint_position in enumerate(
            path,
            start=1,
        ):
            endpoint = endpoints[
                endpoint_position
            ]
            start_atom = previous_atom + 1
            end_atom = endpoint.atom_index
            source_slices = _serialize_atom_range(
                document_id,
                atoms,
                start_atom,
                end_atom,
                indexes,
            )
            passage_id = (
                f"{source_key}:"
                f"passage-{passage_number:03d}"
            )
            record: dict[str, Any] = {
                "schema_version": (
                    PASSAGE_SCHEMA_VERSION
                ),
                "passage_id": passage_id,
                "logical_source_key": source_key,
                "document_id": document_id,
                "unit_kind": unit_kind,
                "passage_number": passage_number,
                "passage_count": len(path),
                "source_unit_ids": [
                    unit["unit_id"]
                    for unit in source_units
                ],
                "label": label,
                "source_slices": source_slices,
                "source_coordinate_count": len(
                    {
                        atom.coordinate
                        for atom in atoms[
                            start_atom:
                            end_atom + 1
                        ]
                    }
                ),
                "boundary_after": endpoint.reason,
                "boundary_coordinate": (
                    coordinate_record(
                        document_index(
                            indexes,
                            document_id,
                        ),
                        atoms[
                            endpoint.atom_index
                        ].coordinate,
                    )
                ),
            }

            if (
                endpoint.reason
                == "after_sentence_in_line"
            ):
                boundary_atom = atoms[
                    endpoint.atom_index
                ]
                record["boundary_char_offset"] = (
                    boundary_atom.end_char
                )
                selected_anchor_sources.add(
                    source_key
                )

            if (
                endpoint.reference_entry_ordinal
                is not None
            ):
                start_ordinal = (
                    1
                    if previous_endpoint_position < 0
                    else (
                        endpoints[
                            previous_endpoint_position
                        ].reference_entry_ordinal
                        + 1
                    )
                )
                record[
                    "reference_entry_start_ordinal"
                ] = start_ordinal
                record[
                    "reference_entry_end_ordinal"
                ] = (
                    endpoint.reference_entry_ordinal
                )

            text = materialize_passage_text(
                record,
                indexes,
            )
            token_count = count_tokens(
                text,
                add_special_tokens=False,
            )

            if (
                token_count
                > PASSAGE_HARD_TOKEN_LIMIT
            ):
                raise RetrievalUnitError(
                    f"{passage_id}: passage contains "
                    f"{token_count} tokens."
                )

            record["passage_token_count"] = (
                token_count
            )
            passages.append(record)
            previous_atom = end_atom
            previous_endpoint_position = (
                endpoint_position
            )

        if previous_atom != len(atoms) - 1:
            raise RetrievalUnitError(
                f"{source_key}: passage partition "
                "changed source coverage."
            )

    expected_anchor_sources = set(
        reviewed_anchors
    )

    if (
        selected_anchor_sources
        != expected_anchor_sources
    ):
        missing = sorted(
            expected_anchor_sources
            - selected_anchor_sources
        )
        unexpected = sorted(
            selected_anchor_sources
            - expected_anchor_sources
        )
        raise RetrievalUnitError(
            "Reviewed intraline anchor selection "
            f"differs: missing={missing}, "
            f"unexpected={unexpected}."
        )

    passage_ids = [
        passage["passage_id"]
        for passage in passages
    ]

    if len(set(passage_ids)) != len(passage_ids):
        raise RetrievalUnitError(
            "Duplicate token-safe passage ID."
        )

    return tuple(passages)



PASSAGE_SUMMARY_SCHEMA_VERSION = "1.0"
EXPECTED_SOURCE_UNIT_COUNT = 581
EXPECTED_LOGICAL_SOURCE_COUNT = 487
EXPECTED_PASSAGE_COUNT = 707
EXPECTED_MAXIMUM_PASSAGE_TOKENS = 445
EXPECTED_REVIEWED_INTRALINE_BOUNDARY_COUNT = 14
EXPECTED_REFERENCE_PASSAGE_COUNT = 27

EXPECTED_PASSAGE_KIND_COUNTS = {
    "eu_recital": 203,
    "frontmatter_body": 4,
    "heading_body": 447,
    "heading_only": 53,
}

EXPECTED_PASSAGE_DOCUMENT_COUNTS = {
    "eu-ai-act-2024-1689": 423,
    "nist-ai-600-1-genai-profile": 111,
    "nist-ai-rmf-1.0": 120,
    "openai-gpt-4o-system-card-2024-08-08": 53,
}

EXPECTED_PASSAGE_BOUNDARY_COUNTS = {
    "after_sentence_in_line": 14,
    "after_strong_terminal": 123,
    "before_reference_entry": 25,
    "before_structured_start": 58,
    "end_of_heading_source": 53,
    "end_of_source_unit": 434,
}


@dataclass(frozen=True)
class RetrievalPassageBuildResult:
    """Validated token-safe passage outputs."""

    passages: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    report: str


def _counter_dict(
    values: Sequence[str],
) -> dict[str, int]:
    """Return a stable, alphabetically ordered count mapping."""

    return dict(
        sorted(
            Counter(values).items()
        )
    )


def _require_exact_count(
    *,
    name: str,
    actual: int,
    expected: int,
) -> None:
    if actual != expected:
        raise RetrievalUnitError(
            f"Expected {expected} {name}, found {actual}."
        )


def _require_exact_mapping(
    *,
    name: str,
    actual: Mapping[str, int],
    expected: Mapping[str, int],
) -> None:
    if dict(actual) != dict(expected):
        raise RetrievalUnitError(
            f"{name} differs from the controlled corpus: "
            f"actual={dict(actual)}, "
            f"expected={dict(expected)}."
        )


def build_retrieval_passage_artifacts(
    units: Sequence[dict[str, Any]],
    hierarchy_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    indexes: CorpusIndexes,
) -> RetrievalPassageBuildResult:
    """Build and audit token-safe derived passage artifacts."""

    _require_exact_count(
        name="source retrieval units",
        actual=len(units),
        expected=EXPECTED_SOURCE_UNIT_COUNT,
    )

    passages = build_retrieval_passages(
        units,
        hierarchy_by_id,
        indexes,
    )
    logical_source_count = len(
        {
            passage["logical_source_key"]
            for passage in passages
        }
    )
    maximum_tokens = max(
        passage["passage_token_count"]
        for passage in passages
    )
    passage_kind_counts = _counter_dict(
        [
            passage["unit_kind"]
            for passage in passages
        ]
    )
    document_counts = _counter_dict(
        [
            passage["document_id"]
            for passage in passages
        ]
    )
    boundary_counts = _counter_dict(
        [
            passage["boundary_after"]
            for passage in passages
        ]
    )
    reviewed_intraline_count = boundary_counts.get(
        "after_sentence_in_line",
        0,
    )
    reference_passage_count = sum(
        "reference_entry_start_ordinal"
        in passage
        for passage in passages
    )
    passages_over_limit = sum(
        passage["passage_token_count"]
        > PASSAGE_HARD_TOKEN_LIMIT
        for passage in passages
    )

    _require_exact_count(
        name="logical sources",
        actual=logical_source_count,
        expected=EXPECTED_LOGICAL_SOURCE_COUNT,
    )
    _require_exact_count(
        name="token-safe passages",
        actual=len(passages),
        expected=EXPECTED_PASSAGE_COUNT,
    )
    _require_exact_count(
        name="maximum passage tokens",
        actual=maximum_tokens,
        expected=EXPECTED_MAXIMUM_PASSAGE_TOKENS,
    )
    _require_exact_count(
        name="passages over the hard token limit",
        actual=passages_over_limit,
        expected=0,
    )
    _require_exact_count(
        name="reviewed intraline boundaries",
        actual=reviewed_intraline_count,
        expected=(
            EXPECTED_REVIEWED_INTRALINE_BOUNDARY_COUNT
        ),
    )
    _require_exact_count(
        name="reference passages",
        actual=reference_passage_count,
        expected=EXPECTED_REFERENCE_PASSAGE_COUNT,
    )
    _require_exact_mapping(
        name="Passage kind counts",
        actual=passage_kind_counts,
        expected=EXPECTED_PASSAGE_KIND_COUNTS,
    )
    _require_exact_mapping(
        name="Passage document counts",
        actual=document_counts,
        expected=EXPECTED_PASSAGE_DOCUMENT_COUNTS,
    )
    _require_exact_mapping(
        name="Passage boundary counts",
        actual=boundary_counts,
        expected=EXPECTED_PASSAGE_BOUNDARY_COUNTS,
    )

    summary = {
        "schema_version": (
            PASSAGE_SUMMARY_SCHEMA_VERSION
        ),
        "mode": (
            "production_token_safe_derived_passages"
        ),
        "source_unit_count": len(units),
        "logical_source_count": (
            logical_source_count
        ),
        "passage_count": len(passages),
        "passage_kind_counts": (
            passage_kind_counts
        ),
        "document_passage_counts": (
            document_counts
        ),
        "boundary_counts": boundary_counts,
        "query_token_reservation": (
            QUERY_TOKEN_RESERVATION
        ),
        "pair_special_token_count": (
            PAIR_SPECIAL_TOKEN_COUNT
        ),
        "passage_hard_token_limit": (
            PASSAGE_HARD_TOKEN_LIMIT
        ),
        "passage_target_token_count": (
            PASSAGE_TARGET_TOKEN_COUNT
        ),
        "maximum_passage_token_count": (
            maximum_tokens
        ),
        "passages_over_hard_limit": (
            passages_over_limit
        ),
        "reviewed_intraline_boundary_count": (
            reviewed_intraline_count
        ),
        "reference_passage_count": (
            reference_passage_count
        ),
        "contains_retrieval_text": False,
        "contains_citation_text": False,
        "contains_embeddings": False,
        "contains_character_offsets": True,
        "changes_coordinate_ownership": False,
    }

    report_lines = [
        "PolicyProof production token-safe retrieval passages",
        "=" * 78,
        "",
        "Derived passage outputs",
        "-" * 78,
        f"- source retrieval units: {len(units)}",
        (
            "- logical sources: "
            f"{logical_source_count}"
        ),
        f"- token-safe passages: {len(passages)}",
        (
            "- reference passages: "
            f"{reference_passage_count}"
        ),
        "",
        "Tokenizer contract",
        "-" * 78,
        (
            "- query-token reservation: "
            f"{QUERY_TOKEN_RESERVATION}"
        ),
        (
            "- pair special tokens: "
            f"{PAIR_SPECIAL_TOKEN_COUNT}"
        ),
        (
            "- hard passage-token limit: "
            f"{PASSAGE_HARD_TOKEN_LIMIT}"
        ),
        (
            "- target passage tokens: "
            f"{PASSAGE_TARGET_TOKEN_COUNT}"
        ),
        (
            "- maximum observed passage tokens: "
            f"{maximum_tokens}"
        ),
        (
            "- passages over hard limit: "
            f"{passages_over_limit}"
        ),
        "",
        "Provenance",
        "-" * 78,
        (
            "- reviewed intraline sentence anchors: "
            f"{reviewed_intraline_count}"
        ),
        "- coordinate ownership changed: no",
        "- accepted coordinate ledger changed: no",
        "- complete bibliography entries preserved: yes",
        "",
        "Deferred downstream fields",
        "-" * 78,
        "- retrieval text: absent",
        "- citation text: absent",
        "- embeddings: absent",
        "- source character offsets: present where required",
        "",
        (
            "PASS: all controlled logical sources have "
            "token-safe passages under the pinned "
            "512-token pair contract."
        ),
    ]

    return RetrievalPassageBuildResult(
        passages=passages,
        summary=summary,
        report="\n".join(report_lines) + "\n",
    )


def write_retrieval_passage_build(
    result: RetrievalPassageBuildResult,
    *,
    passages_path: Path,
    summary_path: Path,
    report_path: Path,
) -> None:
    """Write passage outputs fail-closed with rollback."""

    paths = (
        passages_path,
        summary_path,
        report_path,
    )

    if len(set(paths)) != len(paths):
        raise RetrievalUnitError(
            "Passage output paths must be unique."
        )

    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing:
        raise RetrievalUnitError(
            "Passage output already exists: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    created: list[Path] = []

    try:
        write_jsonl_atomically(
            passages_path,
            result.passages,
        )
        created.append(passages_path)

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
            "Build PolicyProof production token-safe "
            "retrieval passages."
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
        "--units",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--passages-output",
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
    """Run the passage builder from explicit paths."""

    args = _parse_args()
    page_records = load_jsonl(
        args.pages,
        record_name="page",
    )
    hierarchy = load_jsonl(
        args.hierarchy,
        record_name="hierarchy node",
    )
    units = load_jsonl(
        args.units,
        record_name="retrieval unit",
    )
    indexes = build_document_indexes(
        page_records
    )
    hierarchy_by_id = {
        record["node_id"]: record
        for record in hierarchy
    }
    result = build_retrieval_passage_artifacts(
        units,
        hierarchy_by_id,
        indexes,
    )
    write_retrieval_passage_build(
        result,
        passages_path=args.passages_output,
        summary_path=args.summary_output,
        report_path=args.report_output,
    )
    print(
        result.report,
        end="",
    )


if __name__ == "__main__":
    main()
