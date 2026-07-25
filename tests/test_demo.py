from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

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
