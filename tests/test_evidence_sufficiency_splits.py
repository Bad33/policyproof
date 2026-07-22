from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from policyproof.evidence_sufficiency_splits import (
    SPLIT_MANIFEST_ID,
    EvidenceSufficiencySplitError,
    build_leakage_components,
    validate_split_assignments,
    validate_split_manifest,
)


def cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "case-a1",
            "query_id": "query-a",
            "evidence_passage_ids": ["passage-a"],
        },
        {
            "case_id": "case-a2",
            "query_id": "query-a",
            "evidence_passage_ids": ["passage-b"],
        },
        {
            "case_id": "case-b1",
            "query_id": "query-b",
            "evidence_passage_ids": ["passage-b"],
        },
        {
            "case_id": "case-c1",
            "query_id": "query-c",
            "evidence_passage_ids": ["passage-c1"],
        },
        {
            "case_id": "case-d1",
            "query_id": "query-d",
            "evidence_passage_ids": ["passage-c2"],
        },
        {
            "case_id": "case-e1",
            "query_id": "query-e",
            "evidence_passage_ids": ["passage-e"],
        },
        {
            "case_id": "case-f1",
            "query_id": "query-f",
            "evidence_passage_ids": ["passage-f"],
        },
    ]


def passages() -> list[dict[str, Any]]:
    return [
        {
            "passage_id": "passage-a",
            "document_id": "document-a",
            "logical_source_key": "source-a",
        },
        {
            "passage_id": "passage-b",
            "document_id": "document-a",
            "logical_source_key": "source-b",
        },
        {
            "passage_id": "passage-c1",
            "document_id": "document-a",
            "logical_source_key": "source-c",
        },
        {
            "passage_id": "passage-c2",
            "document_id": "document-a",
            "logical_source_key": "source-c",
        },
        {
            "passage_id": "passage-e",
            "document_id": "document-a",
            "logical_source_key": "source-e",
        },
        {
            "passage_id": "passage-f",
            "document_id": "document-a",
            "logical_source_key": "source-f",
        },
    ]


def case_groups(
    value: tuple[Any, ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        component.case_ids
        for component in value
    )


def test_builds_transitive_leakage_components() -> None:
    components = build_leakage_components(
        cases(),
        passages=passages(),
    )

    assert case_groups(components) == (
        ("case-a1", "case-a2", "case-b1"),
        ("case-c1", "case-d1"),
        ("case-e1",),
        ("case-f1",),
    )


def test_component_records_include_link_metadata() -> None:
    components = build_leakage_components(
        cases(),
        passages=passages(),
    )

    first = components[0]

    assert first.query_ids == (
        "query-a",
        "query-b",
    )
    assert first.passage_ids == (
        "passage-a",
        "passage-b",
    )
    assert first.logical_source_keys == (
        "source-a",
        "source-b",
    )


def test_same_query_connects_cases() -> None:
    value = [
        {
            "case_id": "case-1",
            "query_id": "shared-query",
            "evidence_passage_ids": ["passage-a"],
        },
        {
            "case_id": "case-2",
            "query_id": "shared-query",
            "evidence_passage_ids": ["passage-e"],
        },
    ]

    components = build_leakage_components(
        value,
        passages=passages(),
    )

    assert case_groups(components) == (
        ("case-1", "case-2"),
    )


def test_shared_passage_connects_different_queries() -> None:
    value = [
        {
            "case_id": "case-1",
            "query_id": "query-1",
            "evidence_passage_ids": ["passage-b"],
        },
        {
            "case_id": "case-2",
            "query_id": "query-2",
            "evidence_passage_ids": ["passage-b"],
        },
    ]

    components = build_leakage_components(
        value,
        passages=passages(),
    )

    assert case_groups(components) == (
        ("case-1", "case-2"),
    )


def test_shared_logical_source_connects_segmented_passages() -> None:
    value = [
        {
            "case_id": "case-1",
            "query_id": "query-1",
            "evidence_passage_ids": ["passage-c1"],
        },
        {
            "case_id": "case-2",
            "query_id": "query-2",
            "evidence_passage_ids": ["passage-c2"],
        },
    ]

    components = build_leakage_components(
        value,
        passages=passages(),
    )

    assert case_groups(components) == (
        ("case-1", "case-2"),
    )


def test_same_document_does_not_connect_distinct_sources() -> None:
    value = [
        {
            "case_id": "case-1",
            "query_id": "query-1",
            "evidence_passage_ids": ["passage-e"],
        },
        {
            "case_id": "case-2",
            "query_id": "query-2",
            "evidence_passage_ids": ["passage-f"],
        },
    ]

    components = build_leakage_components(
        value,
        passages=passages(),
    )

    assert case_groups(components) == (
        ("case-1",),
        ("case-2",),
    )


def test_result_is_deterministic_across_input_order() -> None:
    original_cases = cases()
    original_passages = passages()

    forward = build_leakage_components(
        original_cases,
        passages=original_passages,
    )
    reversed_inputs = build_leakage_components(
        list(reversed(original_cases)),
        passages=list(reversed(original_passages)),
    )

    assert reversed_inputs == forward


def test_does_not_mutate_inputs() -> None:
    case_records = cases()
    passage_records = passages()
    original_cases = deepcopy(case_records)
    original_passages = deepcopy(passage_records)

    build_leakage_components(
        case_records,
        passages=passage_records,
    )

    assert case_records == original_cases
    assert passage_records == original_passages


def test_empty_cases_produce_no_components() -> None:
    assert build_leakage_components(
        [],
        passages=passages(),
    ) == ()


def test_rejects_duplicate_case_ids() -> None:
    value = cases()
    value[1]["case_id"] = value[0]["case_id"]

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="duplicate case_id",
    ):
        build_leakage_components(
            value,
            passages=passages(),
        )


def test_rejects_duplicate_passage_records() -> None:
    passage_records = passages()
    passage_records.append(deepcopy(passage_records[0]))

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="duplicate passage_id",
    ):
        build_leakage_components(
            cases(),
            passages=passage_records,
        )


def test_rejects_unknown_evidence_passage() -> None:
    value = cases()
    value[0]["evidence_passage_ids"] = [
        "unknown-passage",
    ]

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="unknown evidence passage_id",
    ):
        build_leakage_components(
            value,
            passages=passages(),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("case_id", ""),
        ("query_id", ""),
        ("evidence_passage_ids", []),
    ],
)
def test_rejects_invalid_case_contract(
    field_name: str,
    replacement: Any,
) -> None:
    value = cases()
    value[0][field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match=field_name,
    ):
        build_leakage_components(
            value,
            passages=passages(),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("passage_id", ""),
        ("logical_source_key", ""),
    ],
)
def test_rejects_invalid_passage_contract(
    field_name: str,
    replacement: str,
) -> None:
    passage_records = passages()
    passage_records[0][field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match=field_name,
    ):
        build_leakage_components(
            cases(),
            passages=passage_records,
        )


def valid_split_assignments() -> dict[str, list[str]]:
    return {
        "development": [
            "case-a1",
            "case-a2",
            "case-b1",
        ],
        "validation": [
            "case-c1",
            "case-d1",
        ],
        "test": [
            "case-e1",
            "case-f1",
        ],
    }


def test_valid_split_assignments_pass_without_mutation() -> None:
    components = build_leakage_components(
        cases(),
        passages=passages(),
    )
    assignments = valid_split_assignments()
    original_assignments = deepcopy(assignments)
    original_components = deepcopy(components)

    validate_split_assignments(
        assignments,
        components=components,
    )

    assert assignments == original_assignments
    assert components == original_components


def test_allows_empty_validation_and_test_splits() -> None:
    components = build_leakage_components(
        cases(),
        passages=passages(),
    )
    assignments = {
        "development": [
            case_id
            for component in components
            for case_id in component.case_ids
        ],
        "validation": [],
        "test": [],
    }

    validate_split_assignments(
        assignments,
        components=components,
    )


def test_rejects_unknown_split_name() -> None:
    assignments = valid_split_assignments()
    assignments["holdout"] = assignments.pop("test")

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="split names",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


def test_rejects_missing_required_split() -> None:
    assignments = valid_split_assignments()
    del assignments["test"]

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="split names",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


def test_rejects_case_assigned_to_multiple_splits() -> None:
    assignments = valid_split_assignments()
    assignments["test"].append("case-a1")

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="assigned more than once",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


def test_rejects_duplicate_case_within_one_split() -> None:
    assignments = valid_split_assignments()
    assignments["development"].append("case-a1")

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="assigned more than once",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


def test_rejects_unknown_assigned_case() -> None:
    assignments = valid_split_assignments()
    assignments["test"].append("unknown-case")

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="unknown case_id",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


def test_rejects_unassigned_case() -> None:
    assignments = valid_split_assignments()
    assignments["test"].remove("case-f1")

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="unassigned case_id",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


def test_rejects_component_crossing_splits() -> None:
    assignments = valid_split_assignments()
    assignments["development"].remove("case-b1")
    assignments["validation"].append("case-b1")

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="leakage component.*multiple splits",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "case-a1",
        None,
        {"case-a1"},
    ],
)
def test_rejects_non_array_split_values(
    replacement: Any,
) -> None:
    assignments: dict[str, Any] = valid_split_assignments()
    assignments["development"] = replacement

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="development.*array",
    ):
        validate_split_assignments(
            assignments,
            components=build_leakage_components(
                cases(),
                passages=passages(),
            ),
        )


EVIDENCE_DATASET_SHA256 = "c" * 64


def evidence_dataset_binding() -> dict[str, Any]:
    return {
        "dataset_id": (
            "policyproof-evidence-sufficiency-evaluation"
        ),
        "dataset_version": "0.1.0",
    }


def split_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "manifest_id": SPLIT_MANIFEST_ID,
        "manifest_version": "0.1.0",
        "evidence_dataset_id": (
            "policyproof-evidence-sufficiency-evaluation"
        ),
        "evidence_dataset_version": "0.1.0",
        "evidence_dataset_sha256": EVIDENCE_DATASET_SHA256,
        "component_algorithm_version": "1.0.0",
        "component_count": 4,
        "split_case_counts": {
            "development": 3,
            "validation": 2,
            "test": 2,
        },
        "assignments": valid_split_assignments(),
    }


def validate_manifest(value: dict[str, Any]) -> None:
    validate_split_manifest(
        value,
        evidence_dataset=evidence_dataset_binding(),
        evidence_dataset_sha256=EVIDENCE_DATASET_SHA256,
        components=build_leakage_components(
            cases(),
            passages=passages(),
        ),
    )


def test_valid_split_manifest_passes_without_mutation() -> None:
    value = split_manifest()
    original = deepcopy(value)

    validate_manifest(value)

    assert value == original


def test_rejects_unknown_split_manifest_field() -> None:
    value = split_manifest()
    value["unexpected"] = True

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="unknown split manifest fields",
    ):
        validate_manifest(value)


def test_rejects_unsupported_split_manifest_schema() -> None:
    value = split_manifest()
    value["schema_version"] = "2.0"

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="schema_version",
    ):
        validate_manifest(value)


def test_rejects_unexpected_split_manifest_id() -> None:
    value = split_manifest()
    value["manifest_id"] = "other-manifest"

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="manifest_id",
    ):
        validate_manifest(value)


@pytest.mark.parametrize(
    "field_name",
    [
        "manifest_version",
        "component_algorithm_version",
    ],
)
def test_rejects_invalid_split_manifest_version(
    field_name: str,
) -> None:
    value = split_manifest()
    value[field_name] = "version-one"

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match=field_name,
    ):
        validate_manifest(value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        (
            "evidence_dataset_id",
            "other-evidence-dataset",
        ),
        ("evidence_dataset_version", "9.9.9"),
        ("evidence_dataset_sha256", "d" * 64),
    ],
)
def test_rejects_split_manifest_binding_mismatch(
    field_name: str,
    replacement: str,
) -> None:
    value = split_manifest()
    value[field_name] = replacement

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match=field_name,
    ):
        validate_manifest(value)


def test_rejects_invalid_evidence_dataset_sha256() -> None:
    value = split_manifest()
    value["evidence_dataset_sha256"] = "not-a-sha"

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="evidence_dataset_sha256",
    ):
        validate_manifest(value)


def test_rejects_component_count_mismatch() -> None:
    value = split_manifest()
    value["component_count"] = 5

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="component_count",
    ):
        validate_manifest(value)


def test_rejects_split_case_count_mismatch() -> None:
    value = split_manifest()
    value["split_case_counts"]["development"] = 4

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="split_case_counts",
    ):
        validate_manifest(value)


def test_rejects_unknown_split_case_count_name() -> None:
    value = split_manifest()
    value["split_case_counts"]["holdout"] = 0

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="split_case_counts.*names",
    ):
        validate_manifest(value)


def test_split_manifest_rejects_component_crossing() -> None:
    value = split_manifest()
    value["assignments"]["development"].remove("case-b1")
    value["assignments"]["validation"].append("case-b1")
    value["split_case_counts"] = {
        split_name: len(case_ids)
        for split_name, case_ids
        in value["assignments"].items()
    }

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="leakage component.*multiple splits",
    ):
        validate_manifest(value)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_EVIDENCE_DATASET_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "evidence-sufficiency-evaluation-v0.1.0.json"
)
REPOSITORY_SPLIT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "evidence-sufficiency-split-manifest-v0.1.0.json"
)
REPOSITORY_PASSAGES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "retrieval-passages.jsonl"
)
REPOSITORY_EVIDENCE_DATASET_SHA256 = (
    "9ecd30e4ff829561b50d56bf4f1d3d44"
    "c79dcb043ec15661175842597d733a6a"
)
REPOSITORY_SPLIT_MANIFEST_SHA256 = (
    "314d5ca55a1d6557e8f711eea3506ce1"
    "3a85d30f40e706ea27f0afb8226ff4b2"
)


def repository_sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def repository_load_jsonl(
    path: Path,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )


def test_published_split_manifest_is_byte_stable() -> None:
    assert (
        repository_sha256_file(
            REPOSITORY_SPLIT_MANIFEST_PATH
        )
        == REPOSITORY_SPLIT_MANIFEST_SHA256
    )


def test_rejects_unsupported_component_algorithm_version() -> None:
    value = split_manifest()
    value["component_algorithm_version"] = "2.0.0"

    with pytest.raises(
        EvidenceSufficiencySplitError,
        match="component_algorithm_version",
    ):
        validate_manifest(value)


def test_repository_development_split_manifest_validates() -> None:
    evidence_dataset = json.loads(
        REPOSITORY_EVIDENCE_DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )
    split_manifest_value = json.loads(
        REPOSITORY_SPLIT_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    passage_records = repository_load_jsonl(
        REPOSITORY_PASSAGES_PATH
    )

    assert (
        repository_sha256_file(
            REPOSITORY_EVIDENCE_DATASET_PATH
        )
        == REPOSITORY_EVIDENCE_DATASET_SHA256
    )

    components = build_leakage_components(
        evidence_dataset["cases"],
        passages=passage_records,
    )

    validate_split_manifest(
        split_manifest_value,
        evidence_dataset=evidence_dataset,
        evidence_dataset_sha256=(
            REPOSITORY_EVIDENCE_DATASET_SHA256
        ),
        components=components,
    )

    expected_case_ids = sorted(
        case["case_id"]
        for case in evidence_dataset["cases"]
    )

    assert split_manifest_value["manifest_version"] == "0.1.0"
    assert (
        split_manifest_value["component_algorithm_version"]
        == "1.0.0"
    )
    assert split_manifest_value["component_count"] == 19
    assert split_manifest_value["split_case_counts"] == {
        "development": 39,
        "validation": 0,
        "test": 0,
    }
    assert split_manifest_value["assignments"] == {
        "development": expected_case_ids,
        "validation": [],
        "test": [],
    }
