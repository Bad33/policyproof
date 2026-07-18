"""Compare pypdf plain and layout extraction on representative pages."""

from __future__ import annotations

import re
from pathlib import Path

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

OUTPUT_PATH = Path("data/processed/pypdf-mode-comparison.txt")


def normalize_for_phrase_check(text: str) -> str:
    """Collapse whitespace and lowercase text for phrase checks."""
    return " ".join(text.lower().split())


def calculate_metrics(
    text: str,
    expected_phrase: str,
) -> dict[str, int | float | bool]:
    """Calculate simple indicators of extraction quality."""
    word_tokens = re.findall(
        r"[A-Za-z]+(?:[-'][A-Za-z]+)*",
        text,
    )
    long_alpha_runs = re.findall(r"[A-Za-z]{25,}", text)

    average_word_length = (
        sum(len(token) for token in word_tokens) / len(word_tokens) if word_tokens else 0.0
    )

    normalized_text = normalize_for_phrase_check(text)
    normalized_phrase = normalize_for_phrase_check(expected_phrase)

    return {
        "characters": len(text),
        "lines": len(text.splitlines()),
        "word_tokens": len(word_tokens),
        "average_word_length": round(
            average_word_length,
            2,
        ),
        "long_alpha_runs_25_plus": len(long_alpha_runs),
        "expected_phrase_found": (normalized_phrase in normalized_text),
    }


def extract_modes(
    path: Path,
    page_number: int,
) -> dict[str, str]:
    """Extract one one-based PDF page using both modes."""
    reader = PdfReader(str(path), strict=False)

    page_index = page_number - 1

    if page_index < 0 or page_index >= len(reader.pages):
        raise ValueError(
            f"{path}: page {page_number} is outside the PDF's {len(reader.pages)} pages"
        )

    page = reader.pages[page_index]

    plain_text = page.extract_text(extraction_mode="plain")
    layout_text = page.extract_text(
        extraction_mode="layout",
        layout_mode_space_vertically=False,
    )

    return {
        "plain": plain_text or "",
        "layout": layout_text or "",
    }


def main() -> None:
    """Generate a human-readable extraction comparison."""
    report: list[str] = []

    for sample in SAMPLES:
        document_id = sample["document_id"]
        path = sample["path"]
        page_number = sample["page_number"]
        expected_phrase = sample["expected_phrase"]

        if not path.is_file():
            raise FileNotFoundError(f"Source PDF not found: {path}")

        extracted = extract_modes(path, page_number)

        report.extend(
            [
                "=" * 90,
                f"{document_id} — PDF page {page_number}",
                (f"Expected phrase: {expected_phrase}"),
                "=" * 90,
            ]
        )

        for mode in ("plain", "layout"):
            text = extracted[mode]
            metrics = calculate_metrics(
                text,
                expected_phrase,
            )

            report.extend(
                [
                    "",
                    f"MODE: {mode}",
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
