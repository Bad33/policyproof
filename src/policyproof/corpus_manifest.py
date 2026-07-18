"""Validation for PolicyProof's controlled source manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "corpus_id",
    "corpus_version",
    "verified_date",
    "allowed_domains",
    "documents",
}

REQUIRED_DOCUMENT_FIELDS = {
    "document_id",
    "title",
    "organization",
    "source_url",
    "download_url",
    "doi_url",
    "publication_date",
    "adoption_date",
    "version",
    "jurisdiction",
    "document_category",
    "legal_status",
    "language",
    "source_format",
    "expected_filename",
    "retrieved_date",
}


class ManifestValidationError(ValueError):
    """Raised when the source manifest violates a validation rule."""


def load_manifest(path: Path) -> dict[str, Any]:
    """Read and decode a JSON source manifest."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ManifestValidationError(f"Manifest not found: {path}") from error

    try:
        manifest = json.loads(content)
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"Manifest contains invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(manifest, dict):
        raise ManifestValidationError("Manifest root must be a JSON object.")

    return manifest


def validate_iso_date(value: object, field_name: str, document_id: str) -> None:
    """Validate an ISO-8601 calendar date or a null optional value."""
    if value is None:
        return

    if not isinstance(value, str):
        raise ManifestValidationError(
            f"{document_id}: {field_name} must be an ISO date string or null."
        )

    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ManifestValidationError(
            f"{document_id}: {field_name} is not a valid ISO date: {value}"
        ) from error


def validate_url(
    value: object,
    field_name: str,
    document_id: str,
    allowed_domains: set[str],
    *,
    optional: bool = False,
) -> None:
    """Validate an HTTPS URL whose hostname is explicitly allowed."""
    if optional and value is None:
        return

    if not isinstance(value, str) or not value:
        raise ManifestValidationError(
            f"{document_id}: {field_name} must be a non-empty URL."
        )

    parsed = urlparse(value)

    if parsed.scheme != "https":
        raise ManifestValidationError(
            f"{document_id}: {field_name} must use HTTPS."
        )

    hostname = parsed.hostname
    if hostname not in allowed_domains:
        raise ManifestValidationError(
            f"{document_id}: hostname {hostname!r} from {field_name} "
            "is not in allowed_domains."
        )


def require_non_empty_string(
    document: dict[str, Any],
    field_name: str,
    document_id: str,
) -> str:
    """Return a required string field after checking that it is non-empty."""
    value = document.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"{document_id}: {field_name} must be a non-empty string."
        )

    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate top-level corpus metadata and all document records."""
    missing_top_level = REQUIRED_TOP_LEVEL_FIELDS - manifest.keys()
    if missing_top_level:
        missing = ", ".join(sorted(missing_top_level))
        raise ManifestValidationError(
            f"Manifest is missing top-level fields: {missing}"
        )

    allowed_domains_value = manifest["allowed_domains"]
    if not isinstance(allowed_domains_value, list) or not allowed_domains_value:
        raise ManifestValidationError(
            "allowed_domains must be a non-empty list."
        )

    if not all(
        isinstance(domain, str) and domain.strip()
        for domain in allowed_domains_value
    ):
        raise ManifestValidationError(
            "Every allowed domain must be a non-empty string."
        )

    allowed_domains = set(allowed_domains_value)
    if len(allowed_domains) != len(allowed_domains_value):
        raise ManifestValidationError("allowed_domains contains duplicates.")

    validate_iso_date(
        manifest["verified_date"],
        "verified_date",
        "manifest",
    )

    documents = manifest["documents"]
    if not isinstance(documents, list) or not documents:
        raise ManifestValidationError("documents must be a non-empty list.")

    seen_ids: set[str] = set()
    seen_download_urls: set[str] = set()
    seen_filenames: set[str] = set()

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ManifestValidationError(
                f"Document at index {index} must be a JSON object."
            )

        missing_fields = REQUIRED_DOCUMENT_FIELDS - document.keys()
        provisional_id = document.get("document_id", f"index-{index}")

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ManifestValidationError(
                f"{provisional_id}: missing fields: {missing}"
            )

        document_id = require_non_empty_string(
            document,
            "document_id",
            str(provisional_id),
        )

        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise ManifestValidationError(
                f"{document_id}: document_id must contain only lowercase "
                "letters, numbers, dots, and hyphens."
            )

        if document_id in seen_ids:
            raise ManifestValidationError(
                f"Duplicate document_id: {document_id}"
            )
        seen_ids.add(document_id)

        for field_name in (
            "title",
            "organization",
            "version",
            "jurisdiction",
            "document_category",
            "legal_status",
            "language",
            "source_format",
            "expected_filename",
        ):
            require_non_empty_string(document, field_name, document_id)

        if document["source_format"] != "pdf":
            raise ManifestValidationError(
                f"{document_id}: initial corpus source_format must be pdf."
            )

        filename = document["expected_filename"]
        if not filename.endswith(".pdf"):
            raise ManifestValidationError(
                f"{document_id}: expected_filename must end with .pdf."
            )

        if "/" in filename or "\\" in filename:
            raise ManifestValidationError(
                f"{document_id}: expected_filename must not contain a path."
            )

        if filename in seen_filenames:
            raise ManifestValidationError(
                f"Duplicate expected_filename: {filename}"
            )
        seen_filenames.add(filename)

        validate_url(
            document["source_url"],
            "source_url",
            document_id,
            allowed_domains,
        )
        validate_url(
            document["download_url"],
            "download_url",
            document_id,
            allowed_domains,
        )
        validate_url(
            document["doi_url"],
            "doi_url",
            document_id,
            allowed_domains,
            optional=True,
        )

        download_url = document["download_url"]
        if download_url in seen_download_urls:
            raise ManifestValidationError(
                f"Duplicate download_url: {download_url}"
            )
        seen_download_urls.add(download_url)

        validate_iso_date(
            document["publication_date"],
            "publication_date",
            document_id,
        )
        validate_iso_date(
            document["adoption_date"],
            "adoption_date",
            document_id,
        )
        validate_iso_date(
            document["retrieved_date"],
            "retrieved_date",
            document_id,
        )


def validate_manifest_file(path: Path) -> dict[str, Any]:
    """Load and validate a manifest, returning it when valid."""
    manifest = load_manifest(path)
    validate_manifest(manifest)
    return manifest


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Validate the PolicyProof source manifest."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the source manifest JSON file.",
    )
    args = parser.parse_args()

    try:
        manifest = validate_manifest_file(args.manifest)
    except ManifestValidationError as error:
        print(f"Manifest validation failed: {error}")
        return 1

    print(
        "Manifest valid: "
        f"{len(manifest['documents'])} documents, "
        f"corpus version {manifest['corpus_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
