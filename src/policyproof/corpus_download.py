"""Safe downloading for PolicyProof's controlled document corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from policyproof.corpus_manifest import (
    ManifestValidationError,
    validate_manifest_file,
)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
READ_CHUNK_SIZE = 64 * 1024
PDF_MAGIC = b"%PDF-"

ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
}

USER_AGENT = (
    "PolicyProof/0.1 (controlled public-document ingestion; https://github.com/Bad33/policyproof)"
)


class CorpusDownloadError(RuntimeError):
    """Raised when a corpus artifact cannot be downloaded safely."""


def normalize_content_type(value: str | None) -> str:
    """Return the media type without charset or other parameters."""
    if value is None:
        return ""

    return value.split(";", maxsplit=1)[0].strip().lower()


def validate_resolved_url(
    url: str,
    allowed_domains: set[str],
    *,
    document_id: str,
) -> None:
    """Require an HTTPS URL on an approved hostname."""
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise CorpusDownloadError(f"{document_id}: resolved URL must use HTTPS.")

    if parsed.hostname not in allowed_domains:
        raise CorpusDownloadError(
            f"{document_id}: resolved hostname {parsed.hostname!r} is not approved."
        )


class AllowlistedRedirectHandler(HTTPRedirectHandler):
    """Reject redirects to non-HTTPS or non-allowlisted locations."""

    def __init__(
        self,
        allowed_domains: set[str],
        document_id: str,
    ) -> None:
        super().__init__()
        self.allowed_domains = allowed_domains
        self.document_id = document_id

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        validate_resolved_url(
            newurl,
            self.allowed_domains,
            document_id=self.document_id,
        )

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


def create_https_opener(
    allowed_domains: set[str],
    document_id: str,
) -> Any:
    """Create an opener that validates every HTTP redirect."""
    redirect_handler = AllowlistedRedirectHandler(
        allowed_domains=allowed_domains,
        document_id=document_id,
    )
    return build_opener(redirect_handler)


def remove_file_if_present(path: Path | None) -> None:
    """Remove a temporary or partially created file when it exists."""
    if path is None:
        return

    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def download_document(
    document: dict[str, Any],
    output_dir: Path,
    allowed_domains: set[str],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    opener: Any | None = None,
) -> dict[str, Any]:
    """Download one approved PDF and return its integrity metadata."""
    document_id = document["document_id"]
    download_url = document["download_url"]
    filename = document["expected_filename"]

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename

    if destination.exists():
        raise CorpusDownloadError(f"{document_id}: destination already exists: {destination}")

    validate_resolved_url(
        download_url,
        allowed_domains,
        document_id=document_id,
    )

    request = Request(
        download_url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )

    selected_opener = opener or create_https_opener(
        allowed_domains,
        document_id,
    )

    temporary_path: Path | None = None

    try:
        with selected_opener.open(
            request,
            timeout=timeout_seconds,
        ) as response:
            resolved_url = response.geturl()

            validate_resolved_url(
                resolved_url,
                allowed_domains,
                document_id=document_id,
            )

            content_type = normalize_content_type(response.headers.get("Content-Type"))

            if content_type not in ALLOWED_PDF_CONTENT_TYPES:
                raise CorpusDownloadError(
                    f"{document_id}: expected application/pdf but received "
                    f"{content_type or 'no content type'}."
                )

            content_length_value = response.headers.get("Content-Length")
            if content_length_value:
                try:
                    content_length = int(content_length_value)
                except ValueError as error:
                    raise CorpusDownloadError(
                        f"{document_id}: invalid Content-Length header."
                    ) from error

                if content_length > max_file_size_bytes:
                    raise CorpusDownloadError(
                        f"{document_id}: declared file size exceeds {max_file_size_bytes} bytes."
                    )

            hasher = hashlib.sha256()
            total_bytes = 0

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{filename}.",
                suffix=".part",
                dir=output_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                first_chunk = response.read(READ_CHUNK_SIZE)

                if not first_chunk:
                    raise CorpusDownloadError(f"{document_id}: downloaded file is empty.")

                if not first_chunk.startswith(PDF_MAGIC):
                    raise CorpusDownloadError(
                        f"{document_id}: downloaded content does not have a PDF file signature."
                    )

                total_bytes += len(first_chunk)
                if total_bytes > max_file_size_bytes:
                    raise CorpusDownloadError(
                        f"{document_id}: downloaded file exceeds {max_file_size_bytes} bytes."
                    )

                hasher.update(first_chunk)
                temporary_file.write(first_chunk)

                while True:
                    chunk = response.read(READ_CHUNK_SIZE)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    if total_bytes > max_file_size_bytes:
                        raise CorpusDownloadError(
                            f"{document_id}: downloaded file exceeds {max_file_size_bytes} bytes."
                        )

                    hasher.update(chunk)
                    temporary_file.write(chunk)

                temporary_file.flush()
                os.fsync(temporary_file.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None

        return {
            "document_id": document_id,
            "download_url": download_url,
            "resolved_url": resolved_url,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": total_bytes,
            "sha256": hasher.hexdigest(),
            "retrieved_date": date.today().isoformat(),
        }

    except CorpusDownloadError:
        remove_file_if_present(temporary_path)
        raise
    except HTTPError as error:
        remove_file_if_present(temporary_path)
        raise CorpusDownloadError(f"{document_id}: server returned HTTP {error.code}.") from error
    except URLError as error:
        remove_file_if_present(temporary_path)
        raise CorpusDownloadError(f"{document_id}: network error: {error.reason}") from error
    except TimeoutError as error:
        remove_file_if_present(temporary_path)
        raise CorpusDownloadError(f"{document_id}: download timed out.") from error
    except OSError as error:
        remove_file_if_present(temporary_path)
        raise CorpusDownloadError(f"{document_id}: filesystem error: {error}") from error


def write_json_atomically(path: Path, data: dict[str, Any]) -> None:
    """Write formatted JSON without leaving a partial destination file."""
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


def download_corpus(
    manifest_path: Path,
    output_dir: Path,
    snapshot_path: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    opener_factory: Any = create_https_opener,
) -> dict[str, Any]:
    """Download every manifest document and write one corpus snapshot."""
    manifest = validate_manifest_file(manifest_path)
    allowed_domains = set(manifest["allowed_domains"])

    if snapshot_path.exists():
        raise CorpusDownloadError(f"Corpus snapshot already exists: {snapshot_path}")

    planned_destinations = [
        output_dir / document["expected_filename"] for document in manifest["documents"]
    ]

    existing_destinations = [path for path in planned_destinations if path.exists()]

    if existing_destinations:
        paths = ", ".join(str(path) for path in existing_destinations)
        raise CorpusDownloadError(f"Refusing to overwrite existing corpus files: {paths}")

    receipts: list[dict[str, Any]] = []
    created_files: list[Path] = []

    try:
        for document in manifest["documents"]:
            opener = opener_factory(
                allowed_domains,
                document["document_id"],
            )

            receipt = download_document(
                document,
                output_dir,
                allowed_domains,
                timeout_seconds=timeout_seconds,
                max_file_size_bytes=max_file_size_bytes,
                opener=opener,
            )
            receipts.append(receipt)
            created_files.append(output_dir / document["expected_filename"])
    except Exception:
        for created_file in created_files:
            remove_file_if_present(created_file)
        raise

    snapshot = {
        "schema_version": "1.0",
        "corpus_id": manifest["corpus_id"],
        "corpus_version": manifest["corpus_version"],
        "retrieved_date": date.today().isoformat(),
        "document_count": len(receipts),
        "documents": receipts,
    }

    try:
        write_json_atomically(snapshot_path, snapshot)
    except OSError:
        for created_file in created_files:
            remove_file_if_present(created_file)
        raise

    return snapshot


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI value."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main() -> int:
    """Command-line entry point for controlled corpus downloading."""
    parser = argparse.ArgumentParser(description="Download the controlled PolicyProof PDF corpus.")
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the validated source manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded PDFs.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/corpus_snapshot.json"),
        help="Path for checksum and retrieval metadata.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-document timeout in seconds.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=positive_int,
        default=25,
        help="Maximum allowed size for each document.",
    )

    args = parser.parse_args()

    try:
        snapshot = download_corpus(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            snapshot_path=args.snapshot,
            timeout_seconds=args.timeout,
            max_file_size_bytes=(args.max_file_size_mb * 1024 * 1024),
        )
    except (
        CorpusDownloadError,
        ManifestValidationError,
    ) as error:
        print(f"Corpus download failed: {error}")
        return 1

    print(f"Corpus download complete: {snapshot['document_count']} documents")

    for document in snapshot["documents"]:
        print(
            f"- {document['filename']}: {document['size_bytes']} bytes, sha256={document['sha256']}"
        )

    print(f"Snapshot written to: {args.snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
