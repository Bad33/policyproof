import copy
import json
from pathlib import Path

import pytest

from policyproof.corpus_manifest import (
    ManifestValidationError,
    load_manifest,
    validate_manifest,
    validate_manifest_file,
)

MANIFEST_PATH = Path("data/source_manifest.json")


def test_actual_source_manifest_is_valid() -> None:
    manifest = validate_manifest_file(MANIFEST_PATH)

    assert manifest["corpus_id"] == "policyproof-initial-corpus"
    assert manifest["corpus_version"] == "0.1.0"
    assert len(manifest["documents"]) == 4


def test_manifest_rejects_unapproved_download_domain() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    invalid_manifest = copy.deepcopy(manifest)

    invalid_manifest["documents"][0]["download_url"] = (
        "https://example.com/copied-framework.pdf"
    )

    with pytest.raises(
        ManifestValidationError,
        match="is not in allowed_domains",
    ):
        validate_manifest(invalid_manifest)


def test_manifest_rejects_duplicate_document_id() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    invalid_manifest = copy.deepcopy(manifest)

    invalid_manifest["documents"][1]["document_id"] = (
        invalid_manifest["documents"][0]["document_id"]
    )

    with pytest.raises(
        ManifestValidationError,
        match="Duplicate document_id",
    ):
        validate_manifest(invalid_manifest)


def test_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"documents": [}', encoding="utf-8")

    with pytest.raises(
        ManifestValidationError,
        match="invalid JSON",
    ):
        load_manifest(invalid_path)


def test_manifest_file_is_formatted_json() -> None:
    raw_text = MANIFEST_PATH.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)

    expected = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"

    assert raw_text == expected
