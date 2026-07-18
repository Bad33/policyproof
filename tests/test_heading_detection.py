import pytest

from policyproof.heading_detection import (
    HeadingDetectionError,
    classify_heading,
    detect_heading_candidates,
)


def make_page_record(
    document_id: str,
    text: str,
) -> dict[str, object]:
    return {
        "corpus_id": "test-corpus",
        "corpus_version": "0.1.0",
        "page_id": f"{document_id}:page-0001",
        "document_id": document_id,
        "page_number": 1,
        "text": text,
    }


def test_detects_eu_article_and_title_context() -> None:
    record = make_page_record(
        "eu-ai-act-2024-1689",
        (
            "Article 33\n"
            "Subsidiaries of notified bodies "
            "and subcontracting\n"
            "1. Where a notified body subcontracts..."
        ),
    )

    candidates = detect_heading_candidates(record)

    assert len(candidates) == 1
    assert candidates[0]["candidate_type"] == "article"
    assert candidates[0]["text"] == "Article 33"
    assert candidates[0]["next_line"] == ("Subsidiaries of notified bodies and subcontracting")


def test_eu_numbered_paragraph_is_not_heading() -> None:
    assert (
        classify_heading(
            "eu-ai-act-2024-1689",
            "1. Where a notified body subcontracts tasks",
        )
        is None
    )


def test_detects_nist_numbered_heading() -> None:
    assert (
        classify_heading(
            "nist-ai-rmf-1.0",
            "5. AI RMF Core",
        )
        == "numbered_heading"
    )


def test_detects_nist_function_heading() -> None:
    assert (
        classify_heading(
            "nist-ai-600-1-genai-profile",
            ("MEASURE 1.3: Internal experts are involved in assessments"),
        )
        == "rmf_function_heading"
    )


def test_detects_openai_numbered_subheading() -> None:
    assert (
        classify_heading(
            ("openai-gpt-4o-system-card-2024-08-08"),
            "3.8 Model autonomy",
        )
        == "numbered_subheading"
    )


def test_page_number_is_not_heading() -> None:
    assert (
        classify_heading(
            ("openai-gpt-4o-system-card-2024-08-08"),
            "17",
        )
        is None
    )


def test_unknown_document_fails_closed() -> None:
    with pytest.raises(
        HeadingDetectionError,
        match="Unsupported document_id",
    ):
        classify_heading(
            "unknown-document",
            "1 Introduction",
        )


def test_table_of_contents_page_is_skipped() -> None:
    record = make_page_record(
        "nist-ai-rmf-1.0",
        (
            "Table of Contents\n"
            "1. Framing Risk 4\n"
            "1.1 Understanding and Addressing Risks 4\n"
            "2. Audience 10"
        ),
    )

    assert detect_heading_candidates(record) == []


def test_long_numbered_risk_definition_is_not_heading() -> None:
    record = make_page_record(
        "nist-ai-600-1-genai-profile",
        (
            "1. CBRN Information or Capabilities: "
            "Eased access to or synthesis of materially "
            "nefarious information or design capabilities\n"
            "related to chemical, biological, radiological, "
            "or nuclear risks."
        ),
    )

    assert detect_heading_candidates(record) == []


def test_numbered_prose_continuation_is_not_heading() -> None:
    record = make_page_record(
        "nist-ai-rmf-1.0",
        (
            "3. Use clear and plain language that is "
            "understandable by a broad audience, including\n"
            "senior executives, government officials, and "
            "non-governmental organization leadership."
        ),
    )

    assert detect_heading_candidates(record) == []


def test_eu_article_reference_inside_annex_is_not_heading() -> None:
    record = make_page_record(
        "eu-ai-act-2024-1689",
        (
            "ANNEX VIII\n"
            "Information to be submitted upon the registration "
            "of high-risk AI systems in accordance with\n"
            "Article 49\n"
            "Section A — Information to be submitted by providers"
        ),
    )

    candidates = detect_heading_candidates(record)

    assert [candidate["candidate_type"] for candidate in candidates] == ["annex"]


def test_eu_real_article_after_body_text_is_preserved() -> None:
    record = make_page_record(
        "eu-ai-act-2024-1689",
        (
            "The CE marking shall indicate compliance.\n"
            "Article 49\n"
            "Registration\n"
            "1. Before placing a high-risk AI system on the market..."
        ),
    )

    candidates = detect_heading_candidates(record)

    assert len(candidates) == 1
    assert candidates[0]["candidate_type"] == "article"
    assert candidates[0]["text"] == "Article 49"
    assert candidates[0]["next_line"] == "Registration"
