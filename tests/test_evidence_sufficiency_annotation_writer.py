from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import policyproof.evidence_sufficiency_annotations as annotations_module
from policyproof.evidence_sufficiency_annotations import (
    EvidenceSufficiencyAnnotationError,
    write_annotation_json_artifact,
)


def sample_artifact() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_id": (
            "policyproof-evidence-sufficiency-test-artifact"
        ),
        "artifact_version": "0.1.0",
        "description": "Evidence includes naïve and café text.",
        "case_count": 1,
        "cases": [
            {
                "case_id": "pilot-001",
                "uncertainty": False,
            }
        ],
    }


def test_writer_publishes_formatted_utf8_json(
    tmp_path: Path,
) -> None:
    artifact = sample_artifact()
    output_path = tmp_path / "nested" / "artifact.json"

    write_annotation_json_artifact(
        artifact,
        output_path,
    )

    expected = (
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    assert output_path.read_text(
        encoding="utf-8",
    ) == expected
    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_writer_is_byte_stable_for_same_artifact(
    tmp_path: Path,
) -> None:
    artifact = sample_artifact()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_annotation_json_artifact(
        artifact,
        first_path,
    )
    write_annotation_json_artifact(
        artifact,
        second_path,
    )

    assert first_path.read_bytes() == second_path.read_bytes()


def test_writer_refuses_to_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "artifact.json"
    output_path.write_text(
        '{"existing": true}\n',
        encoding="utf-8",
    )
    original_bytes = output_path.read_bytes()

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="Output already exists",
    ):
        write_annotation_json_artifact(
            sample_artifact(),
            output_path,
        )

    assert output_path.read_bytes() == original_bytes


def test_writer_rejects_non_mapping_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="artifact must be a mapping",
    ):
        write_annotation_json_artifact(
            ["not", "a", "mapping"],  # type: ignore[arg-type]
            tmp_path / "artifact.json",
        )


def test_writer_rejects_non_path_output() -> None:
    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="output_path must be a pathlib.Path",
    ):
        write_annotation_json_artifact(
            sample_artifact(),
            "artifact.json",  # type: ignore[arg-type]
        )


def test_writer_removes_temporary_file_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "artifact.json"

    def fail_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        raise OSError(
            f"simulated publication failure: {source} -> {destination}"
        )

    monkeypatch.setattr(
        annotations_module.os,
        "link",
        fail_link,
    )

    with pytest.raises(
        EvidenceSufficiencyAnnotationError,
        match="Unable to publish annotation artifact",
    ):
        write_annotation_json_artifact(
            sample_artifact(),
            output_path,
        )

    assert not output_path.exists()
    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []
