from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policyproof.bm25 import BM25Hit
from policyproof.demo import (
    PolicyProofDemo,
    PolicyProofDemoError,
    render_home_page,
)
from policyproof.evidence_sufficiency_silver_evaluation import FEATURE_NAMES


def passages() -> list[dict[str, Any]]:
    return [
        {
            "passage_id": "passage-a",
            "document_id": "document-a",
            "label": "Trustworthy AI",
            "retrieval_text": (
                "Trustworthy AI is valid reliable safe secure resilient "
                "accountable transparent explainable privacy enhanced and fair."
            ),
            "citation_text": (
                "Trustworthy AI is valid and reliable, safe, secure and "
                "resilient, accountable and transparent, explainable and "
                "interpretable, privacy-enhanced, and fair."
            ),
        },
        {
            "passage_id": "passage-b",
            "document_id": "document-b",
            "label": "Risk monitoring",
            "retrieval_text": (
                "Organizations monitor AI risks and document controls throughout the lifecycle."
            ),
            "citation_text": (
                "Organizations monitor AI risks and document controls throughout the lifecycle."
            ),
        },
    ]


def baseline(*, intercept: float) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "result_id": ("policyproof-evidence-sufficiency-silver-baseline"),
        "result_version": "0.1.0",
        "label_provenance": "construction_derived",
        "feature_names": list(FEATURE_NAMES),
        "training": {
            "feature_means": [0.0] * len(FEATURE_NAMES),
            "feature_scales": [1.0] * len(FEATURE_NAMES),
            "coefficients": [0.0] * len(FEATURE_NAMES),
            "intercept": intercept,
            "selected_threshold": 0.5,
        },
    }


def test_query_returns_source_derived_answer_and_citation() -> None:
    app = PolicyProofDemo(
        passages(),
        baseline(intercept=10.0),
    )

    result = app.query("Which characteristics are associated with trustworthy AI?")

    assert result["action"] == "answer"
    assert result["evidence_sufficiency"]["predicted_status"] == "sufficient"
    assert result["retrieval"]["citations"]
    assert "[1]" in result["answer"]
    assert result["retrieval"]["citations"][0]["citation_text"] in result["answer"]


def test_query_abstains_when_no_positive_lexical_match() -> None:
    app = PolicyProofDemo(
        passages(),
        baseline(intercept=10.0),
    )

    result = app.query("zephyr quokka xylophone")

    assert result["action"] == "abstain"
    assert result["reason"] == "no_positive_lexical_match"
    assert result["retrieval"]["citations"] == []


def test_query_abstains_below_sufficiency_threshold() -> None:
    app = PolicyProofDemo(
        passages(),
        baseline(intercept=-10.0),
    )

    result = app.query("How do organizations monitor AI risks?")

    assert result["action"] == "abstain"
    assert result["reason"] == ("evidence_below_silver_sufficiency_threshold")
    assert result["retrieval"]["citations"]


def test_query_is_deterministic() -> None:
    app = PolicyProofDemo(
        passages(),
        baseline(intercept=10.0),
    )
    question = "How do organizations monitor AI risks?"

    assert app.query(question) == app.query(question)


@pytest.mark.parametrize("limit", [0, 11, True])
def test_query_rejects_invalid_limit(limit: Any) -> None:
    app = PolicyProofDemo(
        passages(),
        baseline(intercept=10.0),
    )

    with pytest.raises(
        PolicyProofDemoError,
        match="limit",
    ):
        app.query(
            "How do organizations monitor AI risks?",
            limit=limit,
        )


def test_baseline_input_is_not_mutated() -> None:
    value = baseline(intercept=10.0)
    original = deepcopy(value)

    PolicyProofDemo(
        passages(),
        value,
    )

    assert value == original


def test_home_page_contains_query_interface_and_disclosures() -> None:
    page = render_home_page()

    assert "PolicyProof" in page
    assert "/api/query" in page
    assert "construction-derived silver labels" in page
    assert "not legal advice" in page
    assert (
        "What risks does unauthorized voice generation create, "
        "and how does GPT-4o mitigate them?"
    ) in page


def test_query_excludes_reference_passages() -> None:
    value = passages()
    value.append(
        {
            "passage_id": "reference-passage",
            "document_id": "reference-document",
            "label": "References",
            "retrieval_text": (
                "bibliography zephyr quokka xylophone"
            ),
            "citation_text": (
                "A bibliography entry that must not be used as evidence."
            ),
            "reference_entry_start_ordinal": 1,
            "reference_entry_end_ordinal": 1,
        }
    )

    app = PolicyProofDemo(
        value,
        baseline(intercept=10.0),
    )

    result = app.query(
        "bibliography zephyr quokka xylophone"
    )

    assert result["action"] == "abstain"
    assert result["reason"] == "no_positive_lexical_match"
    assert result["retrieval"]["citations"] == []


def test_second_passage_is_selected_when_logical_source_matches() -> None:
    value = passages()
    value[0]["logical_source_key"] = "shared-source"
    value[1]["logical_source_key"] = "shared-source"

    app = PolicyProofDemo(
        value,
        baseline(intercept=10.0),
    )

    hits = (
        BM25Hit(
            passage_id="passage-a",
            score=100.0,
            accepted_order=0,
        ),
        BM25Hit(
            passage_id="passage-b",
            score=70.0,
            accepted_order=1,
        ),
    )

    assert tuple(
        hit.passage_id
        for hit in app._selected_hits(hits)
    ) == ("passage-a", "passage-b")


def test_second_passage_is_rejected_when_logical_source_differs() -> None:
    value = passages()
    value[0]["logical_source_key"] = "source-a"
    value[1]["logical_source_key"] = "source-b"

    app = PolicyProofDemo(
        value,
        baseline(intercept=10.0),
    )

    hits = (
        BM25Hit(
            passage_id="passage-a",
            score=100.0,
            accepted_order=0,
        ),
        BM25Hit(
            passage_id="passage-b",
            score=99.0,
            accepted_order=1,
        ),
    )

    assert tuple(
        hit.passage_id
        for hit in app._selected_hits(hits)
    ) == ("passage-a",)


def test_second_passage_is_rejected_when_logical_source_is_missing() -> None:
    app = PolicyProofDemo(
        passages(),
        baseline(intercept=10.0),
    )

    hits = (
        BM25Hit(
            passage_id="passage-a",
            score=100.0,
            accepted_order=0,
        ),
        BM25Hit(
            passage_id="passage-b",
            score=99.0,
            accepted_order=1,
        ),
    )

    assert tuple(
        hit.passage_id
        for hit in app._selected_hits(hits)
    ) == ("passage-a",)


def test_second_passage_is_rejected_when_logical_source_is_empty() -> None:
    value = passages()
    value[0]["logical_source_key"] = ""
    value[1]["logical_source_key"] = ""

    app = PolicyProofDemo(
        value,
        baseline(intercept=10.0),
    )

    hits = (
        BM25Hit(
            passage_id="passage-a",
            score=100.0,
            accepted_order=0,
        ),
        BM25Hit(
            passage_id="passage-b",
            score=99.0,
            accepted_order=1,
        ),
    )

    assert tuple(
        hit.passage_id
        for hit in app._selected_hits(hits)
    ) == ("passage-a",)


def test_only_second_ranked_same_source_passage_is_selected() -> None:
    value = passages()
    value.append(
        {
            "passage_id": "passage-c",
            "document_id": "document-c",
            "logical_source_key": "shared-source",
            "label": "Additional evidence",
            "retrieval_text": "Additional trustworthy AI evidence.",
            "citation_text": "Additional trustworthy AI evidence.",
        }
    )
    value[0]["logical_source_key"] = "shared-source"
    value[1]["logical_source_key"] = "shared-source"

    app = PolicyProofDemo(
        value,
        baseline(intercept=10.0),
    )

    hits = (
        BM25Hit(
            passage_id="passage-a",
            score=100.0,
            accepted_order=0,
        ),
        BM25Hit(
            passage_id="passage-b",
            score=90.0,
            accepted_order=1,
        ),
        BM25Hit(
            passage_id="passage-c",
            score=80.0,
            accepted_order=2,
        ),
    )

    assert tuple(
        hit.passage_id
        for hit in app._selected_hits(hits)
    ) == ("passage-a", "passage-b")


def test_demo_excludes_low_information_heading_body_passages() -> None:
    value = passages()
    value.extend(
        [
            {
                "passage_id": "table-continuation",
                "document_id": "nist-ai-rmf-1.0",
                "label": (
                    "MANAGE 3.2: Pre-trained models used for "
                    "development are monitored."
                ),
                "unit_kind": "heading_body",
                "retrieval_text": (
                    "MANAGE 3.2: Pre-trained models used for "
                    "development are monitored.\n\n"
                    "Table 4: Categories and subcategories for "
                    "the MANAGE function. (Continued)"
                ),
                "citation_text": (
                    "Table 4: Categories and subcategories for "
                    "the MANAGE function. (Continued)"
                ),
            },
            {
                "passage_id": "complete-heading-evidence",
                "document_id": "nist-ai-rmf-1.0",
                "label": (
                    "GOVERN 1.1: Legal and regulatory requirements "
                    "involving AI are understood and documented."
                ),
                "unit_kind": "heading_only",
                "retrieval_text": (
                    "GOVERN 1.1: Legal and regulatory requirements "
                    "involving AI are understood and documented."
                ),
                "citation_text": (
                    "GOVERN 1.1: Legal and regulatory requirements "
                    "involving AI are understood and documented."
                ),
            },
        ]
    )

    app = PolicyProofDemo(
        value,
        baseline(intercept=10.0),
    )

    assert "table-continuation" not in app._passages_by_id
    assert "complete-heading-evidence" in app._passages_by_id
