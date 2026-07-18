"""Compare pypdf plain extraction with pdfplumber extraction."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

SAMPLES = [
    {
        "document_id": "eu-ai-act-2024-1689",
        "path": Path("data/raw/eu-ai-act-2024-1689.pdf"),
        "page_number": 73,
        "expected_phrase": ("Subsidiaries of notified bodies and subcontracting"),
    },
    {
        "document_id": "nist-ai-600-1-genai-profile",
        "path": Path("data/raw/nist-ai-600-1-genai-profile.pdf"),
        "page_number": 33,
        "expected_phrase": ("Implement continuous monitoring of GAI system impacts"),
    },
    {
        "document_id": "nist-ai-rmf-1.0",
        "path": Path("data/raw/nist-ai-rmf-1.0.pdf"),
        "page_number": 25,
        "expected_phrase": ("The AI RMF Core provides outcomes and actions"),
    },
    {
        "document_id": "openai-gpt-4o-system-card-2024-08-08",
        "path": Path("data/raw/openai-gpt-4o-system-card-2024-08-08.pdf"),
        "page_number": 1,
        "expected_phrase": ("GPT-4o is an autoregressive omni model"),
    },
]

OUTPUT_PATH = Path("data/processed/extractor-comparison.txt")


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens."""
    return re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )


def tokens_appear_in_order(
    text: str,
    expected_phrase: str,
) -> bool:
    """Check whether expected tokens occur in order.

    Unrelated tokens may appear between expected tokens. This permits
    citations such as GPT-4o[1] between the model name and sentence text.
    """
    text_tokens = tokenize(text)
    expected_tokens = tokenize(expected_phrase)

    if not expected_tokens:
        return False

    position = 0

    for expected_token in expected_tokens:
        try:
            position = (
                text_tokens.index(
                    expected_token,
                    position,
                )
                + 1
            )
        except ValueError:
            return False

    return True


def calculate_metrics(
    text: str,
    expected_phrase: str,
) -> dict[str, int | float | bool]:
    """Calculate simple extraction-quality indicators."""
    word_tokens = re.findall(
        r"[A-Za-z]+(?:[-'][A-Za-z]+)*",
        text,
    )
    long_alpha_runs = re.findall(
        r"[A-Za-z]{25,}",
        text,
    )

    average_word_length = (
        sum(len(token) for token in word_tokens) / len(word_tokens) if word_tokens else 0.0
    )

    return {
        "characters": len(text),
        "lines": len(text.splitlines()),
        "word_tokens": len(word_tokens),
        "average_word_length": round(
            average_word_length,
            2,
        ),
        "long_alpha_runs_25_plus": len(long_alpha_runs),
        "expected_tokens_in_order": (
            tokens_appear_in_order(
                text,
                expected_phrase,
            )
        ),
    }


def extract_with_pypdf(
    path: Path,
    page_number: int,
) -> str:
    """Extract one page with pypdf plain mode."""
    reader = PdfReader(str(path), strict=False)
    page_index = page_number - 1

    if page_index < 0 or page_index >= len(reader.pages):
        raise ValueError(f"{path}: invalid page {page_number}")

    return reader.pages[page_index].extract_text(extraction_mode="plain") or ""


def extract_with_pdfplumber(
    path: Path,
    page_number: int,
) -> str:
    """Extract one page with pdfplumber defaults."""
    page_index = page_number - 1

    with pdfplumber.open(path) as pdf:
        if page_index < 0 or page_index >= len(pdf.pages):
            raise ValueError(f"{path}: invalid page {page_number}")

        return pdf.pages[page_index].extract_text() or ""


def main() -> None:
    """Write a human-readable comparison report."""
    report: list[str] = []

    for sample in SAMPLES:
        document_id = sample["document_id"]
        path = sample["path"]
        page_number = sample["page_number"]
        expected_phrase = sample["expected_phrase"]

        if not path.is_file():
            raise FileNotFoundError(f"Source PDF not found: {path}")

        extractions = {
            "pypdf_plain": extract_with_pypdf(
                path,
                page_number,
            ),
            "pdfplumber_default": extract_with_pdfplumber(
                path,
                page_number,
            ),
        }

        report.extend(
            [
                "=" * 90,
                f"{document_id} — PDF page {page_number}",
                f"Expected phrase: {expected_phrase}",
                "=" * 90,
            ]
        )

        for extractor, text in extractions.items():
            metrics = calculate_metrics(
                text,
                expected_phrase,
            )

            report.extend(
                [
                    "",
                    f"EXTRACTOR: {extractor}",
                    f"METRICS: {metrics}",
                    "-" * 90,
                    text[:3000],
                    "",
                ]
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(f"Comparison written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
