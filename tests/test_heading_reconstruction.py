import pytest

from policyproof.heading_reconstruction import (
    HeadingReconstructionError,
    reconstruct_heading,
)


def make_page(
    document_id: str,
    text: str,
) -> dict[str, object]:
    return {
        "page_id": f"{document_id}:page-0001",
        "document_id": document_id,
        "page_number": 1,
        "text": text,
    }


def make_candidate(
    document_id: str,
    candidate_type: str,
    text: str,
    line_number: int = 1,
) -> dict[str, object]:
    return {
        "corpus_id": "test-corpus",
        "corpus_version": "0.1.0",
        "page_id": f"{document_id}:page-0001",
        "document_id": document_id,
        "page_number": 1,
        "line_number": line_number,
        "candidate_type": candidate_type,
        "text": text,
    }


def test_reconstructs_eu_article_title() -> None:
    document_id = "eu-ai-act-2024-1689"
    page = make_page(
        document_id,
        ("Article 49\nRegistration\n1. Before placing the system on the market..."),
    )
    candidate = make_candidate(
        document_id,
        "article",
        "Article 49",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["title_text"] == "Registration"
    assert heading["full_heading"] == ("Article 49 — Registration")
    assert heading["source_line_numbers"] == [1, 2]
    assert heading["end_line_number"] == 2


def test_reconstructs_wrapped_eu_annex_title() -> None:
    document_id = "eu-ai-act-2024-1689"
    page = make_page(
        document_id,
        (
            "ANNEX VIII\n"
            "Information to be submitted upon registration "
            "in accordance with\n"
            "Article 49\n"
            "Section A — Information submitted by providers"
        ),
    )
    candidate = make_candidate(
        document_id,
        "annex",
        "ANNEX VIII",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["title_text"] == (
        "Information to be submitted upon registration in accordance with Article 49"
    )
    assert heading["source_line_numbers"] == [1, 2, 3]
    assert heading["end_line_number"] == 3


def test_eu_reconstruction_stops_at_next_candidate() -> None:
    document_id = "eu-ai-act-2024-1689"
    page = make_page(
        document_id,
        ("CHAPTER I\nGENERAL PROVISIONS\nArticle 1\nSubject matter"),
    )
    candidate = make_candidate(
        document_id,
        "chapter",
        "CHAPTER I",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1, 3},
    )

    assert heading["title_text"] == "GENERAL PROVISIONS"
    assert heading["end_line_number"] == 2


def test_reconstructs_nist_function_heading() -> None:
    document_id = "nist-ai-rmf-1.0"
    page = make_page(
        document_id,
        (
            "GOVERN 1.5: Ongoing monitoring and periodic "
            "review of the\n"
            "risk management process and its outcomes "
            "are planned and organized.\n"
            "GV-1.5-001 Perform an assessment."
        ),
    )
    candidate = make_candidate(
        document_id,
        "rmf_function_heading",
        ("GOVERN 1.5: Ongoing monitoring and periodic review of the"),
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == (
        "GOVERN 1.5: Ongoing monitoring and periodic "
        "review of the risk management process and its "
        "outcomes are planned and organized."
    )
    assert heading["continuation_line_count"] == 1
    assert heading["source_line_numbers"] == [1, 2]


def test_complete_nist_heading_remains_single_line() -> None:
    document_id = "nist-ai-rmf-1.0"
    page = make_page(
        document_id,
        ("MEASURE 1.1: Risks are measured.\nAction ID Suggested Action"),
    )
    candidate = make_candidate(
        document_id,
        "rmf_function_heading",
        "MEASURE 1.1: Risks are measured.",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == ("MEASURE 1.1: Risks are measured.")
    assert heading["continuation_line_count"] == 0


def test_openai_heading_remains_single_line() -> None:
    document_id = "openai-gpt-4o-system-card-2024-08-08"
    page = make_page(
        document_id,
        ("3.8 Model autonomy\nPreparedness Scorecard"),
    )
    candidate = make_candidate(
        document_id,
        "numbered_subheading",
        "3.8 Model autonomy",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == ("3.8 Model autonomy")
    assert heading["reconstruction_method"] == ("single_line")


def test_candidate_text_must_match_source_page() -> None:
    document_id = "nist-ai-rmf-1.0"
    page = make_page(
        document_id,
        "5. AI RMF Core",
    )
    candidate = make_candidate(
        document_id,
        "numbered_heading",
        "5. Different Heading",
    )

    with pytest.raises(
        HeadingReconstructionError,
        match="no longer matches",
    ):
        reconstruct_heading(
            candidate,
            page,
            {1},
        )


def test_unknown_document_fails_closed() -> None:
    document_id = "unknown-document"
    page = make_page(
        document_id,
        "1 Introduction",
    )
    candidate = make_candidate(
        document_id,
        "numbered_heading",
        "1 Introduction",
    )

    with pytest.raises(
        HeadingReconstructionError,
        match="Unsupported document_id",
    ):
        reconstruct_heading(
            candidate,
            page,
            {1},
        )


def test_eu_article_does_not_absorb_body_prose() -> None:
    document_id = "eu-ai-act-2024-1689"
    page = make_page(
        document_id,
        (
            "Article 4\n"
            "AI literacy\n"
            "Providers and deployers shall take measures "
            "to ensure sufficient AI literacy."
        ),
    )
    candidate = make_candidate(
        document_id,
        "article",
        "Article 4",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == ("Article 4 — AI literacy")
    assert heading["source_line_numbers"] == [1, 2]


def test_eu_annex_stops_before_body_introduction() -> None:
    document_id = "eu-ai-act-2024-1689"
    page = make_page(
        document_id,
        (
            "ANNEX II\n"
            "List of criminal offences referred to in "
            "Article 5(1)\n"
            "Criminal offences referred to in Article 5(1):\n"
            "— terrorism"
        ),
    )
    candidate = make_candidate(
        document_id,
        "annex",
        "ANNEX II",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == (
        "ANNEX II — List of criminal offences referred to in Article 5(1)"
    )
    assert heading["source_line_numbers"] == [1, 2]


def test_nist_heading_can_exceed_four_continuation_lines() -> None:
    document_id = "nist-ai-rmf-1.0"
    page = make_page(
        document_id,
        (
            "GOVERN 6: Policies\n"
            "and procedures are\n"
            "in place to address\n"
            "AI risks and benefits\n"
            "arising from\n"
            "third-party software and data."
        ),
    )
    candidate = make_candidate(
        document_id,
        "rmf_function_heading",
        "GOVERN 6: Policies",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == (
        "GOVERN 6: Policies and procedures are in "
        "place to address AI risks and benefits arising "
        "from third-party software and data."
    )
    assert heading["continuation_line_count"] == 5


def test_wrapped_hyphen_does_not_retain_extra_space() -> None:
    document_id = "nist-ai-rmf-1.0"
    page = make_page(
        document_id,
        ("GOVERN 1.2: Trustworthy AI is inte-\ngrated into organizational practices."),
    )
    candidate = make_candidate(
        document_id,
        "rmf_function_heading",
        "GOVERN 1.2: Trustworthy AI is inte-",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == (
        "GOVERN 1.2: Trustworthy AI is inte-grated into organizational practices."
    )
    assert "inte- grated" not in heading["full_heading"]


def test_reconstructs_split_nist_appendix_title() -> None:
    document_id = "nist-ai-rmf-1.0"
    page = make_page(
        document_id,
        ("Appendix D:\nAttributes of the AI RMF\nNIST described several key attributes."),
    )
    candidate = make_candidate(
        document_id,
        "appendix",
        "Appendix D:",
    )

    heading = reconstruct_heading(
        candidate,
        page,
        {1},
    )

    assert heading["full_heading"] == ("Appendix D: Attributes of the AI RMF")
    assert heading["title_text"] == ("Attributes of the AI RMF")
    assert heading["source_line_numbers"] == [1, 2]
    assert heading["reconstruction_method"] == ("nist_appendix_title")
