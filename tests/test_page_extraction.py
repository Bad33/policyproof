import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from policyproof.page_extraction import (
    PageExtractionError,
    extract_corpus_pages,
    extract_page_text,
    normalize_extracted_text,
    select_extraction_method,
    validate_snapshot,
)


class FakePage:
    def __init__(
        self,
        text: str | None,
        expected_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.expected_kwargs = (
            {"extraction_mode": "plain"} if expected_kwargs is None else expected_kwargs
        )

    def extract_text(self, **kwargs: Any) -> str | None:
        assert kwargs == self.expected_kwargs
        return self.text


class FakeReader:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages
        self.is_encrypted = False


def write_fixture_files(
    tmp_path: Path,
    *,
    snapshot_checksum: str | None = None,
) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    pdf_path = raw_dir / "fixture.pdf"
    pdf_content = b"%PDF-controlled-fixture"
    pdf_path.write_bytes(pdf_content)

    checksum = hashlib.sha256(pdf_content).hexdigest()

    manifest = {
        "schema_version": "1.0",
        "corpus_id": "test-corpus",
        "corpus_version": "0.1.0",
        "verified_date": "2026-07-18",
        "allowed_domains": [
            "official.example.gov",
        ],
        "documents": [
            {
                "document_id": "fixture-document",
                "title": "Fixture Document",
                "organization": "Fixture Organization",
                "source_url": ("https://official.example.gov/publication"),
                "download_url": ("https://official.example.gov/fixture.pdf"),
                "doi_url": None,
                "publication_date": "2026-01-01",
                "adoption_date": None,
                "version": "1.0",
                "jurisdiction": "Test",
                "document_category": "fixture",
                "legal_status": "test_only",
                "language": "en",
                "source_format": "pdf",
                "expected_filename": "fixture.pdf",
                "retrieved_date": None,
            }
        ],
    }

    snapshot = {
        "schema_version": "1.0",
        "corpus_id": "test-corpus",
        "corpus_version": "0.1.0",
        "retrieved_date": "2026-07-18",
        "document_count": 1,
        "documents": [
            {
                "document_id": "fixture-document",
                "download_url": ("https://official.example.gov/fixture.pdf"),
                "resolved_url": ("https://official.example.gov/fixture.pdf"),
                "filename": "fixture.pdf",
                "content_type": "application/pdf",
                "size_bytes": len(pdf_content),
                "sha256": snapshot_checksum or checksum,
                "retrieved_date": "2026-07-18",
            }
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    snapshot_path = tmp_path / "snapshot.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest_path, snapshot_path, raw_dir


def test_select_extraction_method_is_document_specific() -> None:
    assert select_extraction_method("eu-ai-act-2024-1689") == "pdfplumber_default"
    assert select_extraction_method("nist-ai-rmf-1.0") == "pypdf_plain"


def test_extract_page_text_uses_pdfplumber_defaults() -> None:
    page = FakePage(
        "Article 33",
        expected_kwargs={},
    )

    assert (
        extract_page_text(
            page,
            "pdfplumber_default",
        )
        == "Article 33"
    )


def test_normalize_extracted_text_normalizes_line_endings() -> None:
    assert normalize_extracted_text("one\r\ntwo\rthree") == ("one\ntwo\nthree")
    assert normalize_extracted_text(None) == ""


def test_validate_snapshot_rejects_corpus_version_mismatch() -> None:
    manifest = {
        "corpus_id": "test-corpus",
        "corpus_version": "1.0",
        "documents": [{"document_id": "document-1"}],
    }
    snapshot = {
        "corpus_id": "test-corpus",
        "corpus_version": "2.0",
        "document_count": 0,
        "documents": [],
    }

    with pytest.raises(
        PageExtractionError,
        match="corpus_version",
    ):
        validate_snapshot(manifest, snapshot)


def test_extract_corpus_writes_page_records(
    tmp_path: Path,
) -> None:
    manifest_path, snapshot_path, raw_dir = write_fixture_files(tmp_path)

    output_path = tmp_path / "processed" / "pages.jsonl"
    summary_path = tmp_path / "processed" / "summary.json"

    fake_reader = FakeReader(
        [
            FakePage("Page one\r\ncontent"),
            FakePage(None),
        ]
    )

    def reader_factory(
        path: str,
        *,
        strict: bool,
    ) -> FakeReader:
        assert path.endswith("fixture.pdf")
        assert strict is False
        return fake_reader

    summary = extract_corpus_pages(
        manifest_path,
        snapshot_path,
        raw_dir,
        output_path,
        summary_path,
        reader_factory=reader_factory,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 2
    assert records[0]["page_id"] == ("fixture-document:page-0001")
    assert records[0]["page_number"] == 1
    assert records[0]["page_count"] == 2
    assert records[0]["text"] == "Page one\ncontent"
    assert records[0]["is_empty"] is False
    assert records[0]["document_sha256"]

    assert records[1]["page_number"] == 2
    assert records[1]["text"] == ""
    assert records[1]["is_empty"] is True

    assert summary["document_count"] == 1
    assert summary["page_count"] == 2
    assert summary["empty_page_count"] == 1
    assert summary_path.exists()


def test_extract_corpus_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, snapshot_path, raw_dir = write_fixture_files(
        tmp_path,
        snapshot_checksum="0" * 64,
    )

    with pytest.raises(
        PageExtractionError,
        match="checksum does not match",
    ):
        extract_corpus_pages(
            manifest_path,
            snapshot_path,
            raw_dir,
            tmp_path / "pages.jsonl",
            tmp_path / "summary.json",
            reader_factory=lambda *args, **kwargs: FakeReader([FakePage("unused")]),
        )


def test_extract_corpus_refuses_to_overwrite_output(
    tmp_path: Path,
) -> None:
    manifest_path, snapshot_path, raw_dir = write_fixture_files(tmp_path)

    output_path = tmp_path / "pages.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(
        PageExtractionError,
        match="already exists",
    ):
        extract_corpus_pages(
            manifest_path,
            snapshot_path,
            raw_dir,
            output_path,
            tmp_path / "summary.json",
            reader_factory=lambda *args, **kwargs: FakeReader([FakePage("unused")]),
        )

    assert output_path.read_text(encoding="utf-8") == ("existing\n")
