"""Reconstruct complete headings from reviewed heading candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

EU_AI_ACT_ID = "eu-ai-act-2024-1689"
NIST_DOCUMENT_IDS = {
    "nist-ai-rmf-1.0",
    "nist-ai-600-1-genai-profile",
}
OPENAI_GPT4O_ID = "openai-gpt-4o-system-card-2024-08-08"

SUPPORTED_DOCUMENT_IDS = {
    EU_AI_ACT_ID,
    *NIST_DOCUMENT_IDS,
    OPENAI_GPT4O_ID,
}

MAX_NIST_CONTINUATION_LINES = 12
MAX_EU_ANNEX_TITLE_LINES = 3

EU_STRUCTURAL_TYPES = {
    "chapter",
    "section",
    "article",
    "annex",
}

EU_BODY_PATTERNS = (
    re.compile(r"^\d+\.\s"),
    re.compile(
        r"^\([a-z0-9ivxlcdm]+\)\s",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Section|Part)\s+[A-Z]\s+[—-]\s",
        re.IGNORECASE,
    ),
)

EU_BODY_PREFIXES = (
    "for the purpose ",
    "for the purposes ",
    "the following ",
    "this annex ",
)

NIST_STOP_PATTERNS = (
    re.compile(r"^\d+$"),
    re.compile(r"^Page\s+\d+$", re.IGNORECASE),
    re.compile(r"^Action ID\b", re.IGNORECASE),
    re.compile(r"^AI Actor Tasks:", re.IGNORECASE),
    re.compile(r"^Table\s+\d+", re.IGNORECASE),
    re.compile(
        r"^(?:GV|MP|MS|MG)-\d",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Categories|Subcategories)$",
        re.IGNORECASE,
    ),
)


NIST_PRESERVED_WRAP_HYPHENS = frozenset(
    {
        "context-relevant",
        "context-specific",
    }
)

NIST_HEADING_TEXT_CORRECTIONS = {
    "nist-ai-rmf-1.0": (
        ("theMAP", "the MAP"),
    ),
}


class HeadingReconstructionError(RuntimeError):
    """Raised when complete headings cannot be reconstructed safely."""


def normalize_line(line: str) -> str:
    """Collapse internal whitespace without changing words."""
    return " ".join(line.split())


def join_wrapped_lines(lines: list[str]) -> str:
    """Join wrapped lines without deleting source characters."""
    joined = ""

    for raw_line in lines:
        line = normalize_line(raw_line)

        if not line:
            continue

        if not joined:
            joined = line
        elif joined.endswith("-") and line[0].islower():
            joined += line
        else:
            joined += f" {line}"

    return joined


def join_nist_function_lines(
    lines: list[str],
    *,
    document_id: str,
) -> str:
    """Join NIST function headings with reviewed wrap-hyphen handling."""
    joined = ""

    for raw_line in lines:
        line = normalize_line(raw_line)

        if not line:
            continue

        if not joined:
            joined = line
            continue

        if joined.endswith("-") and line[0].islower():
            left_match = re.search(
                r"([A-Za-z]+)-$",
                joined,
            )
            right_match = re.match(
                r"([A-Za-z]+)",
                line,
            )

            if left_match is None or right_match is None:
                raise HeadingReconstructionError(
                    "Could not resolve a NIST line-ending "
                    "hyphen deterministically."
                )

            compound = (
                f"{left_match.group(1)}-"
                f"{right_match.group(1)}"
            ).casefold()

            if compound in NIST_PRESERVED_WRAP_HYPHENS:
                joined += line
            else:
                joined = joined[:-1] + line

            continue

        joined += f" {line}"

    for source, replacement in (
        NIST_HEADING_TEXT_CORRECTIONS.get(
            document_id,
            (),
        )
    ):
        joined = joined.replace(
            source,
            replacement,
        )

    return joined


def load_jsonl(
    path: Path,
    *,
    record_name: str,
) -> list[dict[str, Any]]:
    """Load JSON objects from a JSONL file."""
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise HeadingReconstructionError(f"{record_name} file not found: {path}") from error

    records: list[dict[str, Any]] = []

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HeadingReconstructionError(
                    f"Invalid JSON in {record_name} file on line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise HeadingReconstructionError(
                    f"{record_name} record on line {line_number} must be an object."
                )

            records.append(record)

    if not records:
        raise HeadingReconstructionError(f"{record_name} file contains no records.")

    return records


def build_page_index(
    page_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index page records by stable page ID."""
    pages: dict[str, dict[str, Any]] = {}

    for record in page_records:
        page_id = record.get("page_id")

        if not isinstance(page_id, str) or not page_id:
            raise HeadingReconstructionError("Page record is missing a valid page_id.")

        if page_id in pages:
            raise HeadingReconstructionError(f"Duplicate page_id: {page_id}")

        if not isinstance(record.get("text"), str):
            raise HeadingReconstructionError(f"{page_id}: page text must be a string.")

        pages[page_id] = record

    return pages


def build_candidate_line_index(
    candidates: list[dict[str, Any]],
) -> dict[str, set[int]]:
    """Index accepted candidate line numbers by page."""
    indexes: dict[str, set[int]] = {}

    for candidate in candidates:
        page_id = candidate.get("page_id")
        line_number = candidate.get("line_number")

        if not isinstance(page_id, str) or not page_id:
            raise HeadingReconstructionError("Candidate is missing a valid page_id.")

        if not isinstance(line_number, int) or line_number < 1:
            raise HeadingReconstructionError(f"{page_id}: candidate has invalid line_number.")

        page_indexes = indexes.setdefault(page_id, set())

        if line_number in page_indexes:
            raise HeadingReconstructionError(
                f"{page_id}: duplicate candidate on line {line_number}."
            )

        page_indexes.add(line_number)

    return indexes


def is_eu_body_start(line: str) -> bool:
    """Return whether a line clearly begins EU body content."""
    normalized = normalize_line(line)

    if not normalized:
        return False

    if any(pattern.match(normalized) for pattern in EU_BODY_PATTERNS):
        return True

    casefolded = normalized.casefold()

    return any(casefolded.startswith(prefix) for prefix in EU_BODY_PREFIXES)


def is_nist_continuation_stop(line: str) -> bool:
    """Return whether a line cannot be a heading continuation."""
    normalized = normalize_line(line)

    if not normalized:
        return False

    return any(pattern.match(normalized) for pattern in NIST_STOP_PATTERNS)


def collect_eu_title_lines(
    lines: list[str],
    start_index: int,
    candidate_line_numbers: set[int],
    candidate_type: str,
) -> list[tuple[int, str]]:
    """Collect only the structural title following an EU marker."""
    if candidate_type in {
        "chapter",
        "section",
        "article",
    }:
        maximum_lines = 1
    elif candidate_type == "annex":
        maximum_lines = MAX_EU_ANNEX_TITLE_LINES
    else:
        raise HeadingReconstructionError(f"Unsupported EU candidate type: {candidate_type}")

    collected: list[tuple[int, str]] = []

    for index in range(start_index + 1, len(lines)):
        line_number = index + 1
        normalized = normalize_line(lines[index])

        if not normalized:
            continue

        if line_number in candidate_line_numbers:
            break

        if is_eu_body_start(normalized):
            break

        if (
            candidate_type == "annex"
            and collected
            and (normalized.endswith(":") or normalized.startswith("—"))
        ):
            break

        collected.append((line_number, normalized))

        if len(collected) >= maximum_lines:
            break

    return collected


def collect_nist_function_continuations(
    lines: list[str],
    start_index: int,
    candidate_line_numbers: set[int],
) -> list[tuple[int, str]]:
    """Collect wrapped NIST RMF function-heading lines."""
    starting_line = normalize_line(lines[start_index])

    if starting_line.endswith((".", "?", "!")):
        return []

    collected: list[tuple[int, str]] = []

    for index in range(start_index + 1, len(lines)):
        line_number = index + 1
        normalized = normalize_line(lines[index])

        if not normalized:
            continue

        if line_number in candidate_line_numbers:
            break

        if is_nist_continuation_stop(normalized):
            break

        collected.append((line_number, normalized))

        if normalized.endswith((".", "?", "!")):
            break

        if len(collected) >= MAX_NIST_CONTINUATION_LINES:
            break

    return collected


def collect_nist_appendix_title(
    lines: list[str],
    start_index: int,
    candidate_line_numbers: set[int],
) -> list[tuple[int, str]]:
    """Collect one title line after a split NIST appendix marker."""
    marker_text = normalize_line(lines[start_index])

    if not re.fullmatch(
        r"Appendix\s+[A-Z][.:]?",
        marker_text,
        flags=re.IGNORECASE,
    ):
        return []

    for index in range(start_index + 1, len(lines)):
        line_number = index + 1
        normalized = normalize_line(lines[index])

        if not normalized:
            continue

        if line_number in candidate_line_numbers:
            return []

        if is_nist_continuation_stop(normalized):
            return []

        return [(line_number, normalized)]

    return []


def reconstruct_heading(
    candidate: dict[str, Any],
    page_record: dict[str, Any],
    candidate_line_numbers: set[int],
) -> dict[str, Any]:
    """Reconstruct one complete heading from source-page lines."""
    required_candidate_fields = {
        "corpus_id",
        "corpus_version",
        "page_id",
        "document_id",
        "page_number",
        "line_number",
        "candidate_type",
        "text",
    }
    missing = required_candidate_fields - candidate.keys()

    if missing:
        raise HeadingReconstructionError(
            "Candidate is missing fields: " + ", ".join(sorted(missing))
        )

    document_id = candidate["document_id"]

    if document_id not in SUPPORTED_DOCUMENT_IDS:
        raise HeadingReconstructionError(f"Unsupported document_id: {document_id}")

    if page_record.get("document_id") != document_id:
        raise HeadingReconstructionError(
            f"{candidate['page_id']}: page and candidate document IDs do not match."
        )

    lines = page_record["text"].splitlines()
    line_number = candidate["line_number"]
    start_index = line_number - 1

    if start_index >= len(lines):
        raise HeadingReconstructionError(
            f"{candidate['page_id']}: candidate line {line_number} is outside the page."
        )

    source_marker = normalize_line(lines[start_index])
    candidate_text = normalize_line(candidate["text"])

    if source_marker != candidate_text:
        raise HeadingReconstructionError(
            f"{candidate['page_id']} line {line_number}: "
            "candidate text no longer matches source page."
        )

    candidate_type = candidate["candidate_type"]
    continuations: list[tuple[int, str]] = []
    reconstruction_method = "single_line"

    if document_id == EU_AI_ACT_ID and candidate_type in EU_STRUCTURAL_TYPES:
        continuations = collect_eu_title_lines(
            lines,
            start_index,
            candidate_line_numbers,
            candidate_type,
        )
        reconstruction_method = "eu_structural_title"
    elif document_id in NIST_DOCUMENT_IDS and candidate_type == "rmf_function_heading":
        continuations = collect_nist_function_continuations(
            lines,
            start_index,
            candidate_line_numbers,
        )
        reconstruction_method = "nist_function_continuation"
    elif document_id in NIST_DOCUMENT_IDS and candidate_type == "appendix":
        continuations = collect_nist_appendix_title(
            lines,
            start_index,
            candidate_line_numbers,
        )

        if continuations:
            reconstruction_method = "nist_appendix_title"

    source_line_numbers = [
        line_number,
        *[continuation_line_number for continuation_line_number, _ in continuations],
    ]
    source_lines = [
        source_marker,
        *[continuation_text for _, continuation_text in continuations],
    ]

    if document_id == EU_AI_ACT_ID and candidate_type in EU_STRUCTURAL_TYPES:
        title_text = join_wrapped_lines(
            [continuation_text for _, continuation_text in continuations]
        )
        full_heading = f"{source_marker} — {title_text}" if title_text else source_marker
    elif document_id in NIST_DOCUMENT_IDS and candidate_type == "rmf_function_heading":
        title_text = ""
        full_heading = join_nist_function_lines(
            source_lines,
            document_id=document_id,
        )
    elif document_id in NIST_DOCUMENT_IDS and candidate_type == "appendix" and continuations:
        title_text = join_wrapped_lines(
            [continuation_text for _, continuation_text in continuations]
        )
        full_heading = f"{source_marker} {title_text}"
    else:
        title_text = ""
        full_heading = join_wrapped_lines(source_lines)

    return {
        "schema_version": "1.0",
        "corpus_id": candidate["corpus_id"],
        "corpus_version": candidate["corpus_version"],
        "heading_id": (f"{candidate['page_id']}:line-{line_number:04d}"),
        "page_id": candidate["page_id"],
        "document_id": document_id,
        "page_number": candidate["page_number"],
        "candidate_type": candidate_type,
        "marker_text": source_marker,
        "title_text": title_text,
        "full_heading": full_heading,
        "start_line_number": line_number,
        "end_line_number": source_line_numbers[-1],
        "source_line_numbers": source_line_numbers,
        "source_lines": source_lines,
        "continuation_line_count": len(continuations),
        "reconstruction_method": reconstruction_method,
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records to a new JSONL file."""
    if path.exists():
        raise HeadingReconstructionError(f"Output already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_review_report(
    path: Path,
    headings: list[dict[str, Any]],
) -> None:
    """Write a human-readable reconstruction report."""
    if path.exists():
        raise HeadingReconstructionError(f"Review report already exists: {path}")

    document_counts = Counter(heading["document_id"] for heading in headings)
    reconstructed_counts = Counter(
        heading["document_id"] for heading in headings if heading["continuation_line_count"] > 0
    )

    lines = [
        "PolicyProof reconstructed-heading review",
        "=" * 80,
        "",
        f"Total headings: {len(headings)}",
        (f"Headings with continuation lines: {sum(reconstructed_counts.values())}"),
        "",
        "Headings by document:",
    ]

    for document_id, count in sorted(document_counts.items()):
        lines.append(
            f"- {document_id}: {count} total, {reconstructed_counts[document_id]} reconstructed"
        )

    current_document: str | None = None

    for heading in headings:
        document_id = heading["document_id"]

        if document_id != current_document:
            lines.extend(
                [
                    "",
                    "=" * 80,
                    document_id,
                    "=" * 80,
                ]
            )
            current_document = document_id

        lines.extend(
            [
                "",
                (f"{heading['heading_id']} [{heading['candidate_type']}]"),
                (f"  lines: {heading['start_line_number']}-{heading['end_line_number']}"),
                (f"  method: {heading['reconstruction_method']}"),
                f"  heading: {heading['full_heading']}",
                "  source:",
            ]
        )

        for line_number, source_line in zip(
            heading["source_line_numbers"],
            heading["source_lines"],
            strict=True,
        ):
            lines.append(f"    {line_number}: {source_line}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def reconstruct_headings(
    pages_path: Path,
    candidates_path: Path,
    output_path: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    """Reconstruct all accepted heading candidates."""
    page_records = load_jsonl(
        pages_path,
        record_name="Page",
    )
    candidates = load_jsonl(
        candidates_path,
        record_name="Heading candidate",
    )

    pages = build_page_index(page_records)
    candidate_indexes = build_candidate_line_index(candidates)

    headings: list[dict[str, Any]] = []

    for candidate in candidates:
        page_id = candidate["page_id"]

        if page_id not in pages:
            raise HeadingReconstructionError(f"Candidate references missing page: {page_id}")

        headings.append(
            reconstruct_heading(
                candidate,
                pages[page_id],
                candidate_indexes[page_id],
            )
        )

    write_jsonl(output_path, headings)
    write_review_report(report_path, headings)
    return headings


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=("Reconstruct complete headings from PolicyProof heading candidates.")
    )
    parser.add_argument(
        "pages",
        type=Path,
        help="Path to page-level JSONL records.",
    )
    parser.add_argument(
        "candidates",
        type=Path,
        help="Path to reviewed heading-candidate JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reconstructed-headings.jsonl"),
        help="Destination reconstructed-heading JSONL.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/reconstructed-heading-review.txt"),
        help="Destination reconstruction review report.",
    )

    args = parser.parse_args()

    try:
        headings = reconstruct_headings(
            args.pages,
            args.candidates,
            args.output,
            args.report,
        )
    except HeadingReconstructionError as error:
        print(f"Heading reconstruction failed: {error}")
        return 1

    document_counts = Counter(heading["document_id"] for heading in headings)
    continuation_count = sum(heading["continuation_line_count"] > 0 for heading in headings)

    print(f"Heading reconstruction complete: {len(headings)} headings")
    print(f"Headings with continuation lines: {continuation_count}")

    for document_id, count in sorted(document_counts.items()):
        print(f"- {document_id}: {count}")

    print(f"Headings written to: {args.output}")
    print(f"Review report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
