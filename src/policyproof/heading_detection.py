"""Heading-candidate discovery for extracted PolicyProof pages."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

MAX_HEADING_LENGTH = 180

EU_AI_ACT_ID = "eu-ai-act-2024-1689"
NIST_AI_RMF_ID = "nist-ai-rmf-1.0"
NIST_GENAI_ID = "nist-ai-600-1-genai-profile"
OPENAI_GPT4O_ID = "openai-gpt-4o-system-card-2024-08-08"

SUPPORTED_DOCUMENT_IDS = {
    EU_AI_ACT_ID,
    NIST_AI_RMF_ID,
    NIST_GENAI_ID,
    OPENAI_GPT4O_ID,
}

ROMAN_NUMERAL = r"[IVXLCDM]+"

EU_PATTERNS = (
    (
        "chapter",
        re.compile(
            rf"^CHAPTER\s+{ROMAN_NUMERAL}$",
            re.IGNORECASE,
        ),
    ),
    (
        "section",
        re.compile(
            r"^SECTION\s+\d+[A-Z]?$",
            re.IGNORECASE,
        ),
    ),
    (
        "article",
        re.compile(
            r"^Article\s+\d+[A-Z]?$",
            re.IGNORECASE,
        ),
    ),
    (
        "annex",
        re.compile(
            rf"^ANNEX\s+{ROMAN_NUMERAL}$",
            re.IGNORECASE,
        ),
    ),
)

NIST_PATTERNS = (
    (
        "numbered_heading",
        re.compile(
            r"^\d+\.\s+[A-Z][^\n]{2,}$",
        ),
    ),
    (
        "numbered_subheading",
        re.compile(
            r"^\d+\.\d+(?:\.\d+)*\s+[A-Z][^\n]{2,}$",
        ),
    ),
    (
        "rmf_function_heading",
        re.compile(
            r"^(?:GOVERN|MAP|MEASURE|MANAGE)"
            r"\s+\d+(?:\.\d+)?:\s+\S.*$",
        ),
    ),
    (
        "appendix",
        re.compile(
            r"^Appendix\s+[A-Z](?:[.:]\s*|\s+).+",
            re.IGNORECASE,
        ),
    ),
    (
        "named_heading",
        re.compile(
            r"^(?:References|Glossary|Acknowledgments)$",
            re.IGNORECASE,
        ),
    ),
)

OPENAI_PATTERNS = (
    (
        "numbered_heading",
        re.compile(
            r"^\d+\s+[A-Z][^\n]{2,}$",
        ),
    ),
    (
        "numbered_subheading",
        re.compile(
            r"^\d+\.\d+(?:\.\d+)*\s+[A-Z][^\n]{2,}$",
        ),
    ),
    (
        "appendix",
        re.compile(
            r"^Appendix\s+[A-Z](?:[.:]\s*|\s+).+",
            re.IGNORECASE,
        ),
    ),
    (
        "named_heading",
        re.compile(
            r"^(?:References|Acknowledgments)$",
            re.IGNORECASE,
        ),
    ),
)

NOISE_PATTERNS = (
    re.compile(r"^\d+$"),
    re.compile(r"^Page\s+\d+$", re.IGNORECASE),
    re.compile(r"^EN$"),
    re.compile(r"^OJ\s+L,\s*\d{1,2}\.\d{1,2}\.\d{4}$"),
)


class HeadingDetectionError(RuntimeError):
    """Raised when heading discovery cannot be completed safely."""


def normalize_line(line: str) -> str:
    """Collapse internal whitespace while preserving the words."""
    return " ".join(line.split())


def is_noise_line(line: str) -> bool:
    """Return whether a line is known page furniture or noise."""
    normalized = normalize_line(line)

    if not normalized:
        return True

    return any(pattern.fullmatch(normalized) for pattern in NOISE_PATTERNS)


def is_table_of_contents_page(lines: list[str]) -> bool:
    """Return whether a page is clearly a table of contents."""
    normalized_lines = {normalize_line(line).casefold() for line in lines if normalize_line(line)}

    return bool(
        normalized_lines
        & {
            "contents",
            "table of contents",
        }
    )


def is_numbered_list_item(
    candidate_type: str,
    heading: str,
    next_line: str,
) -> bool:
    """Reject obvious enumerated prose while preserving short headings."""
    if candidate_type not in {
        "numbered_heading",
        "numbered_subheading",
    }:
        return False

    if len(heading) > 120:
        return True

    heading_body = re.sub(
        r"^\d+(?:\.\d+)*\.?\s+",
        "",
        heading,
    )

    if ":" in heading_body and len(heading) > 85:
        return True

    if len(heading) > 75 and next_line and next_line[0].islower():
        return True

    return False


def patterns_for_document(
    document_id: str,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Return heading patterns for one controlled document."""
    if document_id == EU_AI_ACT_ID:
        return EU_PATTERNS

    if document_id in {
        NIST_AI_RMF_ID,
        NIST_GENAI_ID,
    }:
        return NIST_PATTERNS

    if document_id == OPENAI_GPT4O_ID:
        return OPENAI_PATTERNS

    raise HeadingDetectionError(f"Unsupported document_id: {document_id}")


def classify_heading(
    document_id: str,
    line: str,
) -> str | None:
    """Classify one extracted line as a heading candidate."""
    normalized = normalize_line(line)

    if is_noise_line(normalized) or len(normalized) > MAX_HEADING_LENGTH:
        return None

    for candidate_type, pattern in patterns_for_document(document_id):
        if pattern.fullmatch(normalized):
            return candidate_type

    return None


def nearest_non_empty_line(
    lines: list[str],
    start_index: int,
    step: int,
) -> str:
    """Return the nearest non-empty normalized line."""
    index = start_index

    while 0 <= index < len(lines):
        normalized = normalize_line(lines[index])

        if normalized:
            return normalized

        index += step

    return ""


def is_eu_article_reference_inside_annex(
    document_id: str,
    candidate_type: str,
    lines: list[str],
    line_index: int,
) -> bool:
    """Reject standalone article references inside an EU annex heading."""
    if document_id != EU_AI_ACT_ID or candidate_type != "article":
        return False

    prior_lines = [normalize_line(line) for line in lines[:line_index]]

    return any(
        classify_heading(
            EU_AI_ACT_ID,
            prior_line,
        )
        == "annex"
        for prior_line in prior_lines
        if prior_line
    )


def detect_heading_candidates(
    page_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return heading candidates from one page record."""
    required_fields = {
        "corpus_id",
        "corpus_version",
        "page_id",
        "document_id",
        "page_number",
        "text",
    }
    missing_fields = required_fields - page_record.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise HeadingDetectionError(f"Page record is missing fields: {missing}")

    document_id = page_record["document_id"]
    patterns_for_document(document_id)

    text = page_record["text"]

    if not isinstance(text, str):
        raise HeadingDetectionError(f"{page_record['page_id']}: text must be a string.")

    lines = text.splitlines()

    if is_table_of_contents_page(lines):
        return []

    candidates: list[dict[str, Any]] = []

    for line_index, raw_line in enumerate(lines):
        normalized = normalize_line(raw_line)
        candidate_type = classify_heading(
            document_id,
            normalized,
        )

        if candidate_type is None:
            continue

        if is_eu_article_reference_inside_annex(
            document_id,
            candidate_type,
            lines,
            line_index,
        ):
            continue

        previous_line = nearest_non_empty_line(
            lines,
            line_index - 1,
            -1,
        )
        next_line = nearest_non_empty_line(
            lines,
            line_index + 1,
            1,
        )

        if is_numbered_list_item(
            candidate_type,
            normalized,
            next_line,
        ):
            continue

        candidates.append(
            {
                "schema_version": "1.0",
                "corpus_id": page_record["corpus_id"],
                "corpus_version": page_record["corpus_version"],
                "page_id": page_record["page_id"],
                "document_id": document_id,
                "page_number": page_record["page_number"],
                "line_number": line_index + 1,
                "candidate_type": candidate_type,
                "text": normalized,
                "previous_line": previous_line,
                "next_line": next_line,
            }
        )

    return candidates


def load_page_records(path: Path) -> list[dict[str, Any]]:
    """Load page records from JSONL."""
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise HeadingDetectionError(f"Page records not found: {path}") from error

    records: list[dict[str, Any]] = []

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HeadingDetectionError(
                    f"Invalid JSON on line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise HeadingDetectionError(f"Page record on line {line_number} must be an object.")

            records.append(record)

    if not records:
        raise HeadingDetectionError("Page-record file contains no records.")

    return records


def write_candidates(
    path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    """Write heading candidates as JSONL."""
    if path.exists():
        raise HeadingDetectionError(f"Candidate output already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            json.dump(
                candidate,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def write_review_report(
    path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    """Write a human-readable candidate review report."""
    if path.exists():
        raise HeadingDetectionError(f"Review report already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    document_counts = Counter(candidate["document_id"] for candidate in candidates)
    type_counts = Counter(
        (
            candidate["document_id"],
            candidate["candidate_type"],
        )
        for candidate in candidates
    )

    lines = [
        "PolicyProof heading-candidate review",
        "=" * 80,
        "",
        f"Total candidates: {len(candidates)}",
        "",
        "Candidates by document:",
    ]

    for document_id, count in sorted(document_counts.items()):
        lines.append(f"- {document_id}: {count}")

        for (
            counted_document_id,
            candidate_type,
        ), type_count in sorted(type_counts.items()):
            if counted_document_id == document_id:
                lines.append(f"  - {candidate_type}: {type_count}")

    current_document_id: str | None = None

    for candidate in candidates:
        document_id = candidate["document_id"]

        if document_id != current_document_id:
            lines.extend(
                [
                    "",
                    "=" * 80,
                    document_id,
                    "=" * 80,
                ]
            )
            current_document_id = document_id

        lines.extend(
            [
                "",
                (
                    f"{candidate['page_id']} "
                    f"line {candidate['line_number']} "
                    f"[{candidate['candidate_type']}]"
                ),
                f"  heading: {candidate['text']}",
                (f"  previous: {candidate['previous_line'] or '[NONE]'}"),
                (f"  next: {candidate['next_line'] or '[NONE]'}"),
            ]
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def discover_headings(
    pages_path: Path,
    candidates_path: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    """Discover candidates across the controlled corpus."""
    page_records = load_page_records(pages_path)

    candidates = [
        candidate
        for page_record in page_records
        for candidate in detect_heading_candidates(page_record)
    ]

    write_candidates(candidates_path, candidates)
    write_review_report(report_path, candidates)

    return candidates


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=("Discover heading candidates in page-level PolicyProof records.")
    )
    parser.add_argument(
        "pages",
        type=Path,
        help="Path to page-level JSONL records.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/heading-candidates.jsonl"),
        help="Destination heading-candidate JSONL file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/heading-candidate-review.txt"),
        help="Destination human-review report.",
    )

    args = parser.parse_args()

    try:
        candidates = discover_headings(
            args.pages,
            args.candidates,
            args.report,
        )
    except HeadingDetectionError as error:
        print(f"Heading discovery failed: {error}")
        return 1

    document_counts = Counter(candidate["document_id"] for candidate in candidates)

    print(f"Heading discovery complete: {len(candidates)} candidates")

    for document_id, count in sorted(document_counts.items()):
        print(f"- {document_id}: {count}")

    print(f"Candidates written to: {args.candidates}")
    print(f"Review report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
