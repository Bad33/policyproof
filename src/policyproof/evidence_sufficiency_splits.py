"""Leakage-safe grouping for evidence-sufficiency cases."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

SPLIT_MANIFEST_ID = (
    "policyproof-evidence-sufficiency-split-manifest"
)
SPLIT_MANIFEST_SCHEMA_VERSION = "1.0"
COMPONENT_ALGORITHM_VERSION = "1.0.0"

SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "manifest_version",
        "evidence_dataset_id",
        "evidence_dataset_version",
        "evidence_dataset_sha256",
        "component_algorithm_version",
        "component_count",
        "split_case_counts",
        "assignments",
    }
)


class EvidenceSufficiencySplitError(ValueError):
    """Raised when leakage-component inputs are invalid."""


@dataclass(frozen=True)
class LeakageComponent:
    """One deterministic connected component of evaluation cases."""

    case_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    passage_ids: tuple[str, ...]
    logical_source_keys: tuple[str, ...]


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSufficiencySplitError(
            f"{field_name} must be an object."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSufficiencySplitError(
            f"{field_name} must be a nonempty string."
        )

    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str],
    object_name: str,
) -> None:
    unknown_fields = sorted(set(value) - allowed_fields)

    if unknown_fields:
        raise EvidenceSufficiencySplitError(
            f"unknown {object_name} fields: {unknown_fields}."
        )


def _require_version(
    value: Any,
    *,
    field_name: str,
) -> str:
    version = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if not VERSION_PATTERN.fullmatch(version):
        raise EvidenceSufficiencySplitError(
            f"{field_name} must use semantic version form X.Y.Z."
        )

    return version


def _require_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    sha256 = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if not SHA256_PATTERN.fullmatch(sha256):
        raise EvidenceSufficiencySplitError(
            f"{field_name} must be a lowercase SHA-256 value."
        )

    return sha256


def _require_nonnegative_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise EvidenceSufficiencySplitError(
            f"{field_name} must be a non-negative integer."
        )

    return value


def _require_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise EvidenceSufficiencySplitError(
            f"{field_name} must be an array."
        )

    return value


def _require_unique_strings(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    sequence = _require_sequence(
        value,
        field_name=field_name,
    )

    if not sequence and not allow_empty:
        raise EvidenceSufficiencySplitError(
            f"{field_name} must be nonempty."
        )

    result: list[str] = []
    seen: set[str] = set()

    for position, item in enumerate(sequence):
        text = _require_nonempty_string(
            item,
            field_name=f"{field_name}[{position}]",
        )

        if text in seen:
            raise EvidenceSufficiencySplitError(
                f"{field_name} must not contain duplicates."
            )

        seen.add(text)
        result.append(text)

    return tuple(result)


REQUIRED_SPLIT_NAMES = frozenset(
    {
        "development",
        "validation",
        "test",
    }
)


def validate_split_assignments(
    assignments: Mapping[str, Any],
    *,
    components: Sequence[LeakageComponent],
) -> None:
    """Validate complete assignments without crossing leakage components."""

    split_assignments = _require_mapping(
        assignments,
        field_name="assignments",
    )

    actual_split_names = set(split_assignments)

    if actual_split_names != REQUIRED_SPLIT_NAMES:
        raise EvidenceSufficiencySplitError(
            "split names must be exactly development, validation, "
            "and test."
        )

    component_records = _require_sequence(
        components,
        field_name="components",
    )

    known_case_ids: set[str] = set()

    for position, component in enumerate(component_records):
        if not isinstance(component, LeakageComponent):
            raise EvidenceSufficiencySplitError(
                f"components[{position}] must be a LeakageComponent."
            )

        if not component.case_ids:
            raise EvidenceSufficiencySplitError(
                f"components[{position}].case_ids must be nonempty."
            )

        for case_id in component.case_ids:
            if case_id in known_case_ids:
                raise EvidenceSufficiencySplitError(
                    "components contain duplicate case_id: "
                    f"{case_id}."
                )

            known_case_ids.add(case_id)

    case_to_split: dict[str, str] = {}

    for split_name in (
        "development",
        "validation",
        "test",
    ):
        case_ids = _require_sequence(
            split_assignments[split_name],
            field_name=f"{split_name} split",
        )

        for position, raw_case_id in enumerate(case_ids):
            case_id = _require_nonempty_string(
                raw_case_id,
                field_name=(
                    f"{split_name} split[{position}]"
                ),
            )

            if case_id in case_to_split:
                raise EvidenceSufficiencySplitError(
                    f"case_id assigned more than once: {case_id}."
                )

            if case_id not in known_case_ids:
                raise EvidenceSufficiencySplitError(
                    f"unknown case_id in split assignments: {case_id}."
                )

            case_to_split[case_id] = split_name

    unassigned_case_ids = sorted(
        known_case_ids - set(case_to_split)
    )

    if unassigned_case_ids:
        raise EvidenceSufficiencySplitError(
            "unassigned case_id values: "
            f"{unassigned_case_ids}."
        )

    for component in component_records:
        component_splits = {
            case_to_split[case_id]
            for case_id in component.case_ids
        }

        if len(component_splits) > 1:
            raise EvidenceSufficiencySplitError(
                "leakage component spans multiple splits: "
                f"{component.case_ids}."
            )



def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    evidence_dataset: Mapping[str, Any],
    evidence_dataset_sha256: str,
    components: Sequence[LeakageComponent],
) -> None:
    """Validate one versioned leakage-safe split manifest."""

    value = _require_mapping(
        manifest,
        field_name="split_manifest",
    )
    evidence = _require_mapping(
        evidence_dataset,
        field_name="evidence_dataset",
    )

    _reject_unknown_fields(
        value,
        allowed_fields=SPLIT_MANIFEST_FIELDS,
        object_name="split manifest",
    )

    schema_version = _require_nonempty_string(
        value.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise EvidenceSufficiencySplitError(
            "schema_version must be 1.0."
        )

    manifest_id = _require_nonempty_string(
        value.get("manifest_id"),
        field_name="manifest_id",
    )

    if manifest_id != SPLIT_MANIFEST_ID:
        raise EvidenceSufficiencySplitError(
            "manifest_id is not the evidence-sufficiency "
            "split-manifest ID."
        )

    _require_version(
        value.get("manifest_version"),
        field_name="manifest_version",
    )

    expected_evidence_dataset_id = _require_nonempty_string(
        evidence.get("dataset_id"),
        field_name="evidence_dataset.dataset_id",
    )
    expected_evidence_dataset_version = _require_version(
        evidence.get("dataset_version"),
        field_name="evidence_dataset.dataset_version",
    )

    manifest_evidence_dataset_id = _require_nonempty_string(
        value.get("evidence_dataset_id"),
        field_name="evidence_dataset_id",
    )

    if (
        manifest_evidence_dataset_id
        != expected_evidence_dataset_id
    ):
        raise EvidenceSufficiencySplitError(
            "evidence_dataset_id does not match the bound "
            "evidence dataset."
        )

    manifest_evidence_dataset_version = _require_version(
        value.get("evidence_dataset_version"),
        field_name="evidence_dataset_version",
    )

    if (
        manifest_evidence_dataset_version
        != expected_evidence_dataset_version
    ):
        raise EvidenceSufficiencySplitError(
            "evidence_dataset_version does not match the bound "
            "evidence dataset."
        )

    accepted_evidence_dataset_sha256 = _require_sha256(
        evidence_dataset_sha256,
        field_name="evidence_dataset_sha256 argument",
    )
    manifest_evidence_dataset_sha256 = _require_sha256(
        value.get("evidence_dataset_sha256"),
        field_name="evidence_dataset_sha256",
    )

    if (
        manifest_evidence_dataset_sha256
        != accepted_evidence_dataset_sha256
    ):
        raise EvidenceSufficiencySplitError(
            "evidence_dataset_sha256 does not match the accepted "
            "evidence dataset binding."
        )

    component_algorithm_version = _require_version(
        value.get("component_algorithm_version"),
        field_name="component_algorithm_version",
    )

    if component_algorithm_version != COMPONENT_ALGORITHM_VERSION:
        raise EvidenceSufficiencySplitError(
            "component_algorithm_version is not supported."
        )

    component_records = _require_sequence(
        components,
        field_name="components",
    )
    component_count = _require_nonnegative_integer(
        value.get("component_count"),
        field_name="component_count",
    )

    if component_count != len(component_records):
        raise EvidenceSufficiencySplitError(
            "component_count does not match the supplied "
            "leakage components."
        )

    split_case_counts = _require_mapping(
        value.get("split_case_counts"),
        field_name="split_case_counts",
    )

    if set(split_case_counts) != REQUIRED_SPLIT_NAMES:
        raise EvidenceSufficiencySplitError(
            "split_case_counts names must be exactly development, "
            "validation, and test."
        )

    declared_split_case_counts = {
        split_name: _require_nonnegative_integer(
            split_case_counts[split_name],
            field_name=(
                f"split_case_counts.{split_name}"
            ),
        )
        for split_name in (
            "development",
            "validation",
            "test",
        )
    }

    assignments = _require_mapping(
        value.get("assignments"),
        field_name="assignments",
    )

    validate_split_assignments(
        assignments,
        components=component_records,
    )

    actual_split_case_counts = {
        split_name: len(assignments[split_name])
        for split_name in (
            "development",
            "validation",
            "test",
        )
    }

    if declared_split_case_counts != actual_split_case_counts:
        raise EvidenceSufficiencySplitError(
            "split_case_counts do not match assignments."
        )


def build_leakage_components(
    cases: Sequence[Mapping[str, Any]],
    *,
    passages: Sequence[Mapping[str, Any]],
) -> tuple[LeakageComponent, ...]:
    """Build deterministic components linked by queries or source evidence."""

    case_records = _require_sequence(
        cases,
        field_name="cases",
    )
    passage_records = _require_sequence(
        passages,
        field_name="passages",
    )

    logical_source_by_passage: dict[str, str] = {}

    for position, raw_passage in enumerate(passage_records):
        passage = _require_mapping(
            raw_passage,
            field_name=f"passages[{position}]",
        )
        passage_id = _require_nonempty_string(
            passage.get("passage_id"),
            field_name=f"passages[{position}].passage_id",
        )
        logical_source_key = _require_nonempty_string(
            passage.get("logical_source_key"),
            field_name=(
                f"passages[{position}].logical_source_key"
            ),
        )

        if passage_id in logical_source_by_passage:
            raise EvidenceSufficiencySplitError(
                f"duplicate passage_id: {passage_id}."
            )

        logical_source_by_passage[passage_id] = (
            logical_source_key
        )

    parsed_cases: list[
        tuple[str, str, tuple[str, ...], tuple[str, ...]]
    ] = []
    seen_case_ids: set[str] = set()

    for position, raw_case in enumerate(case_records):
        case = _require_mapping(
            raw_case,
            field_name=f"cases[{position}]",
        )
        case_id = _require_nonempty_string(
            case.get("case_id"),
            field_name=f"cases[{position}].case_id",
        )
        query_id = _require_nonempty_string(
            case.get("query_id"),
            field_name=f"cases[{position}].query_id",
        )
        passage_ids = _require_unique_strings(
            case.get("evidence_passage_ids"),
            field_name=(
                f"cases[{position}].evidence_passage_ids"
            ),
            allow_empty=False,
        )

        if case_id in seen_case_ids:
            raise EvidenceSufficiencySplitError(
                f"duplicate case_id: {case_id}."
            )

        seen_case_ids.add(case_id)

        logical_source_keys: list[str] = []

        for passage_id in passage_ids:
            try:
                logical_source_key = (
                    logical_source_by_passage[passage_id]
                )
            except KeyError as error:
                raise EvidenceSufficiencySplitError(
                    "unknown evidence passage_id: "
                    f"{passage_id}."
                ) from error

            logical_source_keys.append(logical_source_key)

        parsed_cases.append(
            (
                case_id,
                query_id,
                passage_ids,
                tuple(logical_source_keys),
            )
        )

    if not parsed_cases:
        return ()

    parent = list(range(len(parsed_cases)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)

        if left_root == right_root:
            return

        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    first_case_by_query: dict[str, int] = {}
    first_case_by_passage: dict[str, int] = {}
    first_case_by_logical_source: dict[str, int] = {}

    for index, (
        _case_id,
        query_id,
        passage_ids,
        logical_source_keys,
    ) in enumerate(parsed_cases):
        previous_query_case = first_case_by_query.setdefault(
            query_id,
            index,
        )
        union(index, previous_query_case)

        for passage_id in passage_ids:
            previous_passage_case = (
                first_case_by_passage.setdefault(
                    passage_id,
                    index,
                )
            )
            union(index, previous_passage_case)

        for logical_source_key in logical_source_keys:
            previous_source_case = (
                first_case_by_logical_source.setdefault(
                    logical_source_key,
                    index,
                )
            )
            union(index, previous_source_case)

    indices_by_root: dict[int, list[int]] = {}

    for index in range(len(parsed_cases)):
        indices_by_root.setdefault(
            find(index),
            [],
        ).append(index)

    components: list[LeakageComponent] = []

    for indices in indices_by_root.values():
        case_ids = sorted(
            parsed_cases[index][0]
            for index in indices
        )
        query_ids = sorted(
            {
                parsed_cases[index][1]
                for index in indices
            }
        )
        passage_ids = sorted(
            {
                passage_id
                for index in indices
                for passage_id in parsed_cases[index][2]
            }
        )
        logical_source_keys = sorted(
            {
                logical_source_key
                for index in indices
                for logical_source_key
                in parsed_cases[index][3]
            }
        )

        components.append(
            LeakageComponent(
                case_ids=tuple(case_ids),
                query_ids=tuple(query_ids),
                passage_ids=tuple(passage_ids),
                logical_source_keys=tuple(
                    logical_source_keys
                ),
            )
        )

    return tuple(
        sorted(
            components,
            key=lambda component: component.case_ids,
        )
    )
