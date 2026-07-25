from __future__ import annotations

import json
import subprocess
from pathlib import Path

from policyproof.demo import (
    PolicyProofDemo,
    render_home_page,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Which characteristics does the NIST AI RMF associate with trustworthy AI?"


def test_repository_demo_loads_accepted_assets_and_returns_evidence() -> None:
    app = PolicyProofDemo.from_repository(ROOT)
    result = app.query(QUESTION)

    assert result["action"] in {
        "answer",
        "abstain",
    }
    assert result["retrieval"]["method"] == ("bm25_portable_demo")
    assert result["retrieval"]["citations"]
    assert all(citation["bm25_score"] > 0.0 for citation in result["retrieval"]["citations"])
    assert result["evidence_sufficiency"]["label_provenance"] == "construction_derived"
    assert "legal advice" in result["responsible_use_notice"]


def test_repository_demo_cli_outputs_valid_json() -> None:
    completed = subprocess.run(
        [
            str(ROOT / "venv/bin/python"),
            "-m",
            "policyproof.demo",
            "--root",
            str(ROOT),
            "query",
            QUESTION,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)

    assert result["query"] == QUESTION
    assert result["action"] in {
        "answer",
        "abstain",
    }
    assert result["retrieval"]["citations"]


def test_repository_demo_page_is_self_contained() -> None:
    page = render_home_page()

    assert "<!doctype html>" in page
    assert "<script>" in page
    assert "/api/query" in page
    assert "BM25" in page
