from __future__ import annotations

from pathlib import Path

README = Path("README.md")
RESEARCH_PROTOCOL = Path(
    "docs/evidence-sufficiency-research-protocol.md"
)
ENGINEERING_DECISIONS = Path("docs/engineering-decisions.md")


def test_readme_reports_current_repository_test_count() -> None:
    text = README.read_text(encoding="utf-8")

    assert "tests-891%20passing-brightgreen" in text
    assert "- 891 passing tests" in text
    assert "tests-889%20passing-brightgreen" not in text
    assert "- 889 passing tests" not in text


def test_research_protocol_records_readiness_infrastructure() -> None:
    text = RESEARCH_PROTOCOL.read_text(encoding="utf-8")

    required = (
        "annotation batch version `0.2.0`",
        "`160` blinded cases",
        "full-overlap annotation-round manifests",
        "isolated assignment packages",
        "blank record-set templates",
        "independence attestations",
        "round-completion kill-gate",
        "No real human annotations have been collected or accepted",
    )

    for statement in required:
        assert statement in text


def test_research_protocol_removes_stale_case_status() -> None:
    text = RESEARCH_PROTOCOL.read_text(encoding="utf-8")

    stale = (
        "No evidence\n"
        "cases, annotation labels, validation assignments, or "
        "test assignments have yet\n"
        "been accepted."
    )

    assert stale not in text


def test_research_protocol_remaining_work_is_current() -> None:
    text = RESEARCH_PROTOCOL.read_text(encoding="utf-8")

    required = (
        "Recruit two independent primary annotators",
        "Collect one complete raw submission from each primary annotator",
        "Verify all six independence statements",
        "Measure pre-adjudication agreement",
        "Complete written adjudication",
        "Freeze a human-adjudicated dataset",
    )

    for statement in required:
        assert statement in text


def test_engineering_decision_records_annotation_readiness() -> None:
    text = ENGINEERING_DECISIONS.read_text(encoding="utf-8")

    assert (
        "## PP-041: Establish human-annotation operational readiness"
        in text
    )
    assert "the operational readiness suite passes `49` tests" in text
    assert "the annotation subsystem passes `186` tests" in text
    assert "the complete repository passes `881` tests" in text
    assert (
        "No real human annotations have been collected or accepted"
        in text
    )
