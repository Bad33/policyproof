from pathlib import Path
from typing import Any

import pytest

from policyproof.corpus_download import (
    CorpusDownloadError,
    download_document,
    normalize_content_type,
    validate_resolved_url,
)

ALLOWED_DOMAINS = {"official.example.gov"}

DOCUMENT = {
    "document_id": "test-document-1",
    "download_url": "https://official.example.gov/document.pdf",
    "expected_filename": "test-document-1.pdf",
}


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        content_type: str = "application/pdf",
        resolved_url: str = ("https://official.example.gov/document.pdf"),
        content_length: str | None = None,
    ) -> None:
        self.content = content
        self.position = 0
        self.resolved_url = resolved_url
        self.headers: dict[str, str] = {
            "Content-Type": content_type,
        }

        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def geturl(self) -> str:
        return self.resolved_url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            result = self.content[self.position :]
            self.position = len(self.content)
            return result

        end = self.position + size
        result = self.content[self.position : end]
        self.position = min(end, len(self.content))
        return result


class FakeOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def open(
        self,
        request: Any,
        *,
        timeout: float,
    ) -> FakeResponse:
        assert request.full_url == DOCUMENT["download_url"]
        assert timeout > 0
        return self.response


def test_normalize_content_type_removes_parameters() -> None:
    assert normalize_content_type("application/pdf; charset=binary") == "application/pdf"


def test_download_document_writes_pdf_and_checksum(
    tmp_path: Path,
) -> None:
    pdf_content = b"%PDF-1.7\ncontrolled fixture\n%%EOF\n"
    opener = FakeOpener(FakeResponse(pdf_content))

    receipt = download_document(
        DOCUMENT,
        tmp_path,
        ALLOWED_DOMAINS,
        opener=opener,
    )

    destination = tmp_path / DOCUMENT["expected_filename"]

    assert destination.read_bytes() == pdf_content
    assert receipt["document_id"] == DOCUMENT["document_id"]
    assert receipt["size_bytes"] == len(pdf_content)
    assert len(receipt["sha256"]) == 64
    assert receipt["content_type"] == "application/pdf"


def test_download_rejects_non_pdf_content_type(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        b"%PDF-1.7\nfixture",
        content_type="text/html",
    )

    with pytest.raises(
        CorpusDownloadError,
        match="expected application/pdf",
    ):
        download_document(
            DOCUMENT,
            tmp_path,
            ALLOWED_DOMAINS,
            opener=FakeOpener(response),
        )

    assert not (tmp_path / DOCUMENT["expected_filename"]).exists()


def test_download_rejects_missing_pdf_signature(
    tmp_path: Path,
) -> None:
    response = FakeResponse(b"<html>not a PDF</html>")

    with pytest.raises(
        CorpusDownloadError,
        match="does not have a PDF file signature",
    ):
        download_document(
            DOCUMENT,
            tmp_path,
            ALLOWED_DOMAINS,
            opener=FakeOpener(response),
        )

    assert not (tmp_path / DOCUMENT["expected_filename"]).exists()


def test_download_rejects_declared_oversized_file(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        b"%PDF-1.7\nfixture",
        content_length="5000",
    )

    with pytest.raises(
        CorpusDownloadError,
        match="declared file size exceeds",
    ):
        download_document(
            DOCUMENT,
            tmp_path,
            ALLOWED_DOMAINS,
            max_file_size_bytes=100,
            opener=FakeOpener(response),
        )


def test_download_rejects_stream_that_exceeds_limit(
    tmp_path: Path,
) -> None:
    response = FakeResponse(b"%PDF-" + (b"x" * 200))

    with pytest.raises(
        CorpusDownloadError,
        match="downloaded file exceeds",
    ):
        download_document(
            DOCUMENT,
            tmp_path,
            ALLOWED_DOMAINS,
            max_file_size_bytes=100,
            opener=FakeOpener(response),
        )

    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_unapproved_resolved_domain(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        b"%PDF-1.7\nfixture",
        resolved_url="https://unapproved.example.com/document.pdf",
    )

    with pytest.raises(
        CorpusDownloadError,
        match="resolved hostname",
    ):
        download_document(
            DOCUMENT,
            tmp_path,
            ALLOWED_DOMAINS,
            opener=FakeOpener(response),
        )


def test_validate_resolved_url_requires_https() -> None:
    with pytest.raises(
        CorpusDownloadError,
        match="must use HTTPS",
    ):
        validate_resolved_url(
            "http://official.example.gov/document.pdf",
            ALLOWED_DOMAINS,
            document_id=DOCUMENT["document_id"],
        )


def test_download_refuses_to_overwrite_existing_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / DOCUMENT["expected_filename"]
    destination.write_bytes(b"%PDF-existing")

    with pytest.raises(
        CorpusDownloadError,
        match="destination already exists",
    ):
        download_document(
            DOCUMENT,
            tmp_path,
            ALLOWED_DOMAINS,
            opener=FakeOpener(FakeResponse(b"%PDF-new")),
        )

    assert destination.read_bytes() == b"%PDF-existing"
