from __future__ import annotations

from pathlib import Path

OPERATIONS_GUIDE = Path(
    "docs/evidence-sufficiency-annotation-operations.md"
)

REQUIRED_HEADINGS = (
    "# Evidence-Sufficiency Annotation Operations",
    "## Current status",
    "## Human roles and minimum staffing",
    "## Pseudonymous identity handling",
    "## Frozen inputs",
    "## Round preparation",
    "## Independent annotation",
    "## Submission intake",
    "## Round-completion kill-gate",
    "## Agreement and disagreement analysis",
    "## Adjudication",
    "## Artifact handling",
    "## Acceptance criteria",
    "## Abort and restart conditions",
)

REQUIRED_STATEMENTS = (
    "No human annotation has been collected or accepted",
    "two primary annotators",
    "one adjudicator",
    "100% overlap",
    "counterbalanced case order",
    "outside Git",
    "construction-derived silver labels",
    "must not be shown",
    "agreement analysis must not begin",
    "all six independence statements",
    "raw submissions remain immutable",
    "preserve both original annotations",
    "do not create fake annotation records",
    "blank record-set template",
    "unfilled templates are not valid submissions",
    "unexpected batch, case, or evidence fields",
)

FORBIDDEN_COMPLETION_CLAIMS = (
    "human annotation is complete",
    "human-gold dataset has been created",
    "inter-annotator agreement was measured",
    "adjudication has been completed",
)


def test_annotation_operations_guide_exists() -> None:
    assert OPERATIONS_GUIDE.is_file()


def test_annotation_operations_guide_has_required_structure() -> None:
    text = OPERATIONS_GUIDE.read_text(encoding="utf-8")

    for heading in REQUIRED_HEADINGS:
        assert heading in text


def test_annotation_operations_guide_freezes_core_protocol() -> None:
    text = OPERATIONS_GUIDE.read_text(encoding="utf-8")

    for statement in REQUIRED_STATEMENTS:
        assert statement in text


def test_annotation_operations_guide_makes_no_false_claims() -> None:
    text = OPERATIONS_GUIDE.read_text(
        encoding="utf-8"
    ).lower()

    for claim in FORBIDDEN_COMPLETION_CLAIMS:
        assert claim not in text


def test_annotation_operations_guide_binds_frozen_batch() -> None:
    text = OPERATIONS_GUIDE.read_text(encoding="utf-8")

    assert (
        "evidence-sufficiency-annotation-batch-v0.2.0.json"
        in text
    )
    assert (
        "1bb6a7bed55a43f59a79ff4861c81c3d"
        "36ffa5ed78af1bf12292bceb927bf93c"
        in text
    )
