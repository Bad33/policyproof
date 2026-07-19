import pytest

from policyproof.heading_hierarchy import (
    HeadingHierarchyError,
    build_heading_hierarchy,
)


def make_heading(
    document_id: str,
    marker_text: str,
    candidate_type: str,
    *,
    page_number: int,
    line_number: int,
    full_heading: str | None = None,
) -> dict:
    heading_id = f"{document_id}:page-{page_number:04d}:line-{line_number:04d}"

    return {
        "schema_version": "1.0",
        "corpus_id": "test-corpus",
        "corpus_version": "0.1.0",
        "heading_id": heading_id,
        "page_id": (f"{document_id}:page-{page_number:04d}"),
        "document_id": document_id,
        "page_number": page_number,
        "candidate_type": candidate_type,
        "marker_text": marker_text,
        "title_text": "",
        "full_heading": (full_heading if full_heading is not None else marker_text),
        "start_line_number": line_number,
        "end_line_number": line_number,
        "source_line_numbers": [line_number],
        "source_lines": [marker_text],
        "continuation_line_count": 0,
        "reconstruction_method": "single_line",
    }


def node_by_heading(
    nodes: list[dict],
    full_heading: str,
) -> dict:
    matches = [node for node in nodes if node["full_heading"] == full_heading]

    assert len(matches) == 1
    return matches[0]


def test_eu_chapter_section_article_hierarchy() -> None:
    document_id = "eu-ai-act-2024-1689"
    headings = [
        make_heading(
            document_id,
            "CHAPTER I",
            "chapter",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "SECTION 1",
            "section",
            page_number=1,
            line_number=2,
        ),
        make_heading(
            document_id,
            "Article 1",
            "article",
            page_number=1,
            line_number=3,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    chapter = node_by_heading(nodes, "CHAPTER I")
    section = node_by_heading(nodes, "SECTION 1")
    article = node_by_heading(nodes, "Article 1")

    assert chapter["parent_node_id"] is None
    assert section["parent_node_id"] == chapter["node_id"]
    assert article["parent_node_id"] == section["node_id"]
    assert article["depth"] == 3


def test_eu_article_without_section_uses_chapter() -> None:
    document_id = "eu-ai-act-2024-1689"
    headings = [
        make_heading(
            document_id,
            "CHAPTER I",
            "chapter",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "Article 1",
            "article",
            page_number=1,
            line_number=2,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    chapter = node_by_heading(nodes, "CHAPTER I")
    article = node_by_heading(nodes, "Article 1")

    assert article["parent_node_id"] == chapter["node_id"]


def test_eu_annex_is_top_level() -> None:
    document_id = "eu-ai-act-2024-1689"
    headings = [
        make_heading(
            document_id,
            "CHAPTER I",
            "chapter",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "ANNEX I",
            "annex",
            page_number=2,
            line_number=1,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    annex = node_by_heading(nodes, "ANNEX I")

    assert annex["parent_node_id"] is None
    assert annex["depth"] == 1


def test_openai_numeric_depth_assigns_parent() -> None:
    document_id = "openai-gpt-4o-system-card-2024-08-08"
    headings = [
        make_heading(
            document_id,
            "3 Safety",
            "numbered_heading",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "3.3 Audio",
            "numbered_subheading",
            page_number=1,
            line_number=2,
        ),
        make_heading(
            document_id,
            "3.3.1 Voice",
            "numbered_subheading",
            page_number=1,
            line_number=3,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    level_one = node_by_heading(nodes, "3 Safety")
    level_two = node_by_heading(nodes, "3.3 Audio")
    level_three = node_by_heading(nodes, "3.3.1 Voice")

    assert level_two["parent_node_id"] == level_one["node_id"]
    assert level_three["parent_node_id"] == level_two["node_id"]


def test_numbered_orphan_fails_closed() -> None:
    document_id = "openai-gpt-4o-system-card-2024-08-08"
    headings = [
        make_heading(
            document_id,
            "3.3 Audio",
            "numbered_subheading",
            page_number=1,
            line_number=1,
        ),
    ]

    with pytest.raises(
        HeadingHierarchyError,
        match="has no depth-1 parent",
    ):
        build_heading_hierarchy(headings)


def test_rmf_uses_real_numbered_function_parent() -> None:
    document_id = "nist-ai-rmf-1.0"
    headings = [
        make_heading(
            document_id,
            "5. AI RMF Core",
            "numbered_heading",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "5.1 Govern",
            "numbered_subheading",
            page_number=1,
            line_number=2,
        ),
        make_heading(
            document_id,
            "GOVERN 1.1: Requirement",
            "rmf_function_heading",
            page_number=1,
            line_number=3,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    govern = node_by_heading(nodes, "5.1 Govern")
    category = node_by_heading(nodes, "GOVERN 1")
    child = node_by_heading(
        nodes,
        "GOVERN 1.1: Requirement",
    )

    assert category["node_kind"] == "synthetic"
    assert category["parent_node_id"] == govern["node_id"]
    assert child["parent_node_id"] == category["node_id"]

    synthetic_functions = [node for node in nodes if node["synthetic_type"] == "rmf_function"]
    assert synthetic_functions == []


def test_rmf_uses_explicit_category_node() -> None:
    document_id = "nist-ai-rmf-1.0"
    headings = [
        make_heading(
            document_id,
            "5. AI RMF Core",
            "numbered_heading",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "5.2 Map",
            "numbered_subheading",
            page_number=1,
            line_number=2,
        ),
        make_heading(
            document_id,
            "MAP 1: Context",
            "rmf_function_heading",
            page_number=1,
            line_number=3,
        ),
        make_heading(
            document_id,
            "MAP 1.1: Intended purposes",
            "rmf_function_heading",
            page_number=1,
            line_number=4,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    category = node_by_heading(nodes, "MAP 1: Context")
    child = node_by_heading(
        nodes,
        "MAP 1.1: Intended purposes",
    )

    assert category["node_kind"] == "source"
    assert child["parent_node_id"] == category["node_id"]

    synthetic_categories = [node for node in nodes if node["synthetic_type"] == "rmf_category"]
    assert synthetic_categories == []


def test_genai_creates_function_and_category_nodes() -> None:
    document_id = "nist-ai-600-1-genai-profile"
    headings = [
        make_heading(
            document_id,
            "3. Suggested Actions to Manage GAI Risks",
            "numbered_heading",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "GOVERN 1.1: Requirement",
            "rmf_function_heading",
            page_number=1,
            line_number=2,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    section = node_by_heading(
        nodes,
        "3. Suggested Actions to Manage GAI Risks",
    )
    function = node_by_heading(nodes, "GOVERN")
    category = node_by_heading(nodes, "GOVERN 1")
    child = node_by_heading(
        nodes,
        "GOVERN 1.1: Requirement",
    )

    assert function["node_kind"] == "synthetic"
    assert function["parent_node_id"] == section["node_id"]
    assert category["parent_node_id"] == function["node_id"]
    assert child["parent_node_id"] == category["node_id"]


def test_appendix_clears_numbered_hierarchy() -> None:
    document_id = "nist-ai-rmf-1.0"
    headings = [
        make_heading(
            document_id,
            "6. Profiles",
            "numbered_heading",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "Appendix A: Glossary",
            "appendix",
            page_number=2,
            line_number=1,
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    appendix = node_by_heading(
        nodes,
        "Appendix A: Glossary",
    )

    assert appendix["parent_node_id"] is None


def test_hierarchy_paths_include_all_ancestors() -> None:
    document_id = "openai-gpt-4o-system-card-2024-08-08"
    headings = [
        make_heading(
            document_id,
            "3 Safety",
            "numbered_heading",
            page_number=1,
            line_number=1,
        ),
        make_heading(
            document_id,
            "3.3 Audio",
            "numbered_subheading",
            page_number=1,
            line_number=2,
        ),
    ]

    nodes = build_heading_hierarchy(headings)
    child = node_by_heading(nodes, "3.3 Audio")

    assert child["hierarchy_path"] == [
        "3 Safety",
        "3.3 Audio",
    ]
    assert len(child["ancestor_node_ids"]) == 1


def test_eu_annex_sections_are_siblings() -> None:
    document_id = "eu-ai-act-2024-1689"
    headings = [
        make_heading(
            document_id,
            "ANNEX XI",
            "annex",
            page_number=141,
            line_number=3,
            full_heading=("ANNEX XI — Technical documentation"),
        ),
        make_heading(
            document_id,
            "Section 1",
            "section",
            page_number=141,
            line_number=6,
            full_heading=("Section 1 — Information to be provided by all providers"),
        ),
        make_heading(
            document_id,
            "Section 2",
            "section",
            page_number=141,
            line_number=34,
            full_heading=("Section 2 — Additional information for systemic-risk models"),
        ),
    ]

    nodes = build_heading_hierarchy(headings)

    annex = node_by_heading(
        nodes,
        "ANNEX XI — Technical documentation",
    )
    section_one = node_by_heading(
        nodes,
        ("Section 1 — Information to be provided by all providers"),
    )
    section_two = node_by_heading(
        nodes,
        ("Section 2 — Additional information for systemic-risk models"),
    )

    assert annex["parent_node_id"] is None
    assert section_one["parent_node_id"] == annex["node_id"]
    assert section_two["parent_node_id"] == annex["node_id"]
    assert section_one["depth"] == 2
    assert section_two["depth"] == 2
