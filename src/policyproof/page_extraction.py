"""Page-level extraction for the controlled PolicyProof corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pdfplumber
import pypdf
from pypdf import PdfReader

from policyproof.corpus_manifest import (
    ManifestValidationError,
    validate_manifest_file,
)

DEFAULT_EXTRACTION_METHOD = "pypdf_plain"
DOCUMENT_EXTRACTION_METHODS = {
    "eu-ai-act-2024-1689": "pdfplumber_default",
}

ReaderFactory = Callable[..., Any]
PdfPlumberOpen = Callable[..., Any]


class PageExtractionError(RuntimeError):
    """Raised when page-level extraction cannot be completed safely."""


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load a corpus snapshot from JSON."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise PageExtractionError(f"Corpus snapshot not found: {path}") from error

    try:
        snapshot = json.loads(content)
    except json.JSONDecodeError as error:
        raise PageExtractionError(
            f"Corpus snapshot contains invalid JSON at line "
            f"{error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(snapshot, dict):
        raise PageExtractionError("Corpus snapshot root must be a JSON object.")

    return snapshot


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all at once."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(64 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def normalize_extracted_text(text: str | None) -> str:
    """Normalize line endings without otherwise rewriting extracted text."""
    if text is None:
        return ""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def validate_snapshot(
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate snapshot identity and return records by document ID."""
    if snapshot.get("corpus_id") != manifest["corpus_id"]:
        raise PageExtractionError("Snapshot corpus_id does not match the source manifest.")

    if snapshot.get("corpus_version") != manifest["corpus_version"]:
        raise PageExtractionError("Snapshot corpus_version does not match the source manifest.")

    records = snapshot.get("documents")
    if not isinstance(records, list):
        raise PageExtractionError("Snapshot documents must be a list.")

    if snapshot.get("document_count") != len(records):
        raise PageExtractionError("Snapshot document_count does not match its document records.")

    records_by_id: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PageExtractionError(f"Snapshot document at index {index} must be an object.")

        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise PageExtractionError(f"Snapshot document at index {index} has no document_id.")

        if document_id in records_by_id:
            raise PageExtractionError(f"Duplicate snapshot document_id: {document_id}")

        for field_name in (
            "filename",
            "size_bytes",
            "sha256",
            "retrieved_date",
        ):
            if field_name not in record:
                raise PageExtractionError(f"{document_id}: snapshot is missing {field_name}.")

        records_by_id[document_id] = record

    manifest_ids = {document["document_id"] for document in manifest["documents"]}
    snapshot_ids = set(records_by_id)

    if manifest_ids != snapshot_ids:
        missing = sorted(manifest_ids - snapshot_ids)
        unexpected = sorted(snapshot_ids - manifest_ids)

        raise PageExtractionError(
            f"Manifest and snapshot document IDs differ. Missing={missing}, unexpected={unexpected}"
        )

    return records_by_id


def verify_source_file(
    document: dict[str, Any],
    receipt: dict[str, Any],
    raw_dir: Path,
) -> Path:
    """Verify the local source file against its corpus snapshot."""
    document_id = document["document_id"]
    expected_filename = document["expected_filename"]

    if receipt["filename"] != expected_filename:
        raise PageExtractionError(f"{document_id}: snapshot filename does not match manifest.")

    source_path = raw_dir / expected_filename

    if not source_path.is_file():
        raise PageExtractionError(f"{document_id}: source PDF is missing: {source_path}")

    actual_size = source_path.stat().st_size
    if actual_size != receipt["size_bytes"]:
        raise PageExtractionError(f"{document_id}: source PDF size does not match snapshot.")

    actual_checksum = sha256_file(source_path)
    if actual_checksum != receipt["sha256"]:
        raise PageExtractionError(f"{document_id}: source PDF checksum does not match snapshot.")

    return source_path


def select_extraction_method(document_id: str) -> str:
    """Return the measured extraction method for a document."""
    return DOCUMENT_EXTRACTION_METHODS.get(
        document_id,
        DEFAULT_EXTRACTION_METHOD,
    )


def get_extractor_metadata(
    extraction_method: str,
) -> tuple[str, str]:
    """Return the library name and version for an extraction method."""
    if extraction_method == "pypdf_plain":
        return "pypdf", pypdf.__version__

    if extraction_method == "pdfplumber_default":
        return "pdfplumber", pdfplumber.__version__

    raise PageExtractionError(f"Unsupported extraction method: {extraction_method}")


def extract_page_text(
    page: Any,
    extraction_method: str,
) -> str:
    """Extract one page using the selected measured method."""
    if extraction_method == "pypdf_plain":
        text = page.extract_text(
            extraction_mode="plain",
        )
    elif extraction_method == "pdfplumber_default":
        text = page.extract_text()
    else:
        raise PageExtractionError(f"Unsupported extraction method: {extraction_method}")

    return normalize_extracted_text(text)


def build_page_record(
    *,
    manifest: dict[str, Any],
    document: dict[str, Any],
    receipt: dict[str, Any],
    page_number: int,
    page_count: int,
    text: str,
    extraction_method: str,
    extractor: str,
    extractor_version: str,
) -> dict[str, Any]:
    """Build one citation-ready page record."""
    document_id = document["document_id"]

    return {
        "schema_version": "1.0",
        "corpus_id": manifest["corpus_id"],
        "corpus_version": manifest["corpus_version"],
        "page_id": f"{document_id}:page-{page_number:04d}",
        "document_id": document_id,
        "title": document["title"],
        "organization": document["organization"],
        "source_url": document["source_url"],
        "download_url": document["download_url"],
        "publication_date": document["publication_date"],
        "adoption_date": document["adoption_date"],
        "version": document["version"],
        "jurisdiction": document["jurisdiction"],
        "document_category": document["document_category"],
        "legal_status": document["legal_status"],
        "language": document["language"],
        "source_filename": receipt["filename"],
        "document_sha256": receipt["sha256"],
        "retrieved_date": receipt["retrieved_date"],
        "page_number": page_number,
        "page_count": page_count,
        "extraction_method": extraction_method,
        "extractor": extractor,
        "extractor_version": extractor_version,
        "character_count": len(text),
        "is_empty": not bool(text.strip()),
        "text": text,
    }


def open_pdf(
    source_path: Path,
    reader_factory: ReaderFactory,
    document_id: str,
) -> Any:
    """Open a PDF with pypdf and reject unreadable encryption."""
    try:
        reader = reader_factory(str(source_path), strict=False)
    except Exception as error:
        raise PageExtractionError(
            f"{document_id}: pypdf could not open the source PDF: {error}"
        ) from error

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as error:
            raise PageExtractionError(
                f"{document_id}: encrypted PDF could not be decrypted."
            ) from error

        if decrypt_result == 0:
            raise PageExtractionError(f"{document_id}: encrypted PDF requires a password.")

    if len(reader.pages) == 0:
        raise PageExtractionError(f"{document_id}: PDF contains no pages.")

    return reader


def open_pdfplumber_document(
    source_path: Path,
    document_id: str,
    pdfplumber_open: PdfPlumberOpen,
) -> Any:
    """Open a PDF using pdfplumber."""
    try:
        return pdfplumber_open(source_path)
    except Exception as error:
        raise PageExtractionError(
            f"{document_id}: pdfplumber could not open the source PDF: {error}"
        ) from error


def remove_file_if_present(path: Path | None) -> None:
    """Best-effort removal of a temporary or partially created file."""
    if path is None:
        return

    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def write_json_atomically(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write formatted JSON through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                data,
                temporary_file,
                indent=2,
                ensure_ascii=False,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        remove_file_if_present(temporary_path)


def extract_corpus_pages(
    manifest_path: Path,
    snapshot_path: Path,
    raw_dir: Path,
    output_path: Path,
    summary_path: Path,
    *,
    reader_factory: ReaderFactory = PdfReader,
    pdfplumber_open: PdfPlumberOpen = pdfplumber.open,
) -> dict[str, Any]:
    """Extract verified PDFs into page-level JSONL records."""
    if output_path.exists():
        raise PageExtractionError(f"Page output already exists: {output_path}")

    if summary_path.exists():
        raise PageExtractionError(f"Extraction summary already exists: {summary_path}")

    manifest = validate_manifest_file(manifest_path)
    snapshot = load_snapshot(snapshot_path)
    receipts_by_id = validate_snapshot(manifest, snapshot)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_output_path: Path | None = None
    output_created = False

    total_pages = 0
    total_characters = 0
    empty_pages = 0
    document_summaries: list[dict[str, Any]] = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".part",
            dir=output_path.parent,
            delete=False,
        ) as temporary_output:
            temporary_output_path = Path(temporary_output.name)

            for document in manifest["documents"]:
                document_id = document["document_id"]
                receipt = receipts_by_id[document_id]

                source_path = verify_source_file(
                    document,
                    receipt,
                    raw_dir,
                )

                extraction_method = select_extraction_method(document_id)
                extractor, extractor_version = get_extractor_metadata(extraction_method)

                if extraction_method == "pypdf_plain":
                    reader_context = nullcontext(
                        open_pdf(
                            source_path,
                            reader_factory,
                            document_id,
                        )
                    )
                elif extraction_method == "pdfplumber_default":
                    reader_context = open_pdfplumber_document(
                        source_path,
                        document_id,
                        pdfplumber_open,
                    )
                else:
                    raise PageExtractionError(
                        f"{document_id}: unsupported extraction method {extraction_method}"
                    )

                with reader_context as reader:
                    page_count = len(reader.pages)

                    if page_count == 0:
                        raise PageExtractionError(f"{document_id}: PDF contains no pages.")

                    document_characters = 0
                    document_empty_pages = 0

                    for page_index, page in enumerate(
                        reader.pages,
                        start=1,
                    ):
                        try:
                            page_text = extract_page_text(
                                page,
                                extraction_method,
                            )
                        except Exception as error:
                            raise PageExtractionError(
                                f"{document_id}: extraction failed on page {page_index}: {error}"
                            ) from error

                        record = build_page_record(
                            manifest=manifest,
                            document=document,
                            receipt=receipt,
                            page_number=page_index,
                            page_count=page_count,
                            text=page_text,
                            extraction_method=extraction_method,
                            extractor=extractor,
                            extractor_version=extractor_version,
                        )

                        json.dump(
                            record,
                            temporary_output,
                            ensure_ascii=False,
                        )
                        temporary_output.write("\n")

                        character_count = len(page_text)
                        document_characters += character_count
                        total_characters += character_count
                        total_pages += 1

                        if record["is_empty"]:
                            document_empty_pages += 1
                            empty_pages += 1

                document_summaries.append(
                    {
                        "document_id": document_id,
                        "filename": receipt["filename"],
                        "document_sha256": receipt["sha256"],
                        "page_count": page_count,
                        "character_count": document_characters,
                        "empty_page_count": document_empty_pages,
                        "extraction_method": extraction_method,
                        "extractor": extractor,
                        "extractor_version": extractor_version,
                    }
                )

            temporary_output.flush()
            os.fsync(temporary_output.fileno())

        os.replace(temporary_output_path, output_path)
        temporary_output_path = None
        output_created = True

        summary = {
            "schema_version": "1.0",
            "corpus_id": manifest["corpus_id"],
            "corpus_version": manifest["corpus_version"],
            "extracted_date": date.today().isoformat(),
            "extraction_policy": "document_specific_v1",
            "extractors": {
                "pdfplumber": pdfplumber.__version__,
                "pypdf": pypdf.__version__,
            },
            "document_count": len(document_summaries),
            "page_count": total_pages,
            "character_count": total_characters,
            "empty_page_count": empty_pages,
            "output_file": str(output_path),
            "documents": document_summaries,
        }

        write_json_atomically(summary_path, summary)
        return summary

    except Exception:
        remove_file_if_present(temporary_output_path)

        if output_created:
            remove_file_if_present(output_path)

        raise


def main() -> int:
    """Command-line entry point for page-level corpus extraction."""
    parser = argparse.ArgumentParser(
        description="Extract verified corpus PDFs into page-level JSONL."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the validated source manifest.",
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to the verified corpus snapshot.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing downloaded source PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/pages.jsonl"),
        help="Destination page-level JSONL file.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/processed/extraction-summary.json"),
        help="Destination extraction summary.",
    )

    args = parser.parse_args()

    try:
        summary = extract_corpus_pages(
            manifest_path=args.manifest,
            snapshot_path=args.snapshot,
            raw_dir=args.raw_dir,
            output_path=args.output,
            summary_path=args.summary,
        )
    except (
        ManifestValidationError,
        PageExtractionError,
    ) as error:
        print(f"Page extraction failed: {error}")
        return 1

    print(
        "Page extraction complete: "
        f"{summary['document_count']} documents, "
        f"{summary['page_count']} pages, "
        f"{summary['empty_page_count']} empty pages"
    )
    print(f"Extracted characters: {summary['character_count']}")
    print(f"Pages written to: {args.output}")
    print(f"Summary written to: {args.summary}")

    for document in summary["documents"]:
        print(
            f"- {document['document_id']}: "
            f"{document['page_count']} pages, "
            f"{document['character_count']} characters, "
            f"{document['empty_page_count']} empty"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
