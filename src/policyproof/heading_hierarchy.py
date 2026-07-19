"""Assign deterministic hierarchy to reconstructed PolicyProof headings."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

EU_AI_ACT_ID = "eu-ai-act-2024-1689"
NIST_AI_RMF_ID = "nist-ai-rmf-1.0"
NIST_GENAI_ID = "nist-ai-600-1-genai-profile"
OPENAI_GPT4O_ID = "openai-gpt-4o-system-card-2024-08-08"

SUPPORTED_DOCUMENT_IDS = {
    EU_AI_ACT_ID,
    NIST_AI_RMF_ID,
    NIST_GENAI_ID,
    OPENAI_GPT4O_ID,
}

NIST_DOCUMENT_IDS = {
    NIST_AI_RMF_ID,
    NIST_GENAI_ID,
}

NUMBERED_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+")

RMF_PATTERN = re.compile(
    r"^(?P<function>GOVERN|MAP|MEASURE|MANAGE)"
    r"\s+(?P<number>\d+(?:\.\d+)?):",
    flags=re.IGNORECASE,
)


class HeadingHierarchyError(RuntimeError):
    """Raised when a safe heading hierarchy cannot be assigned."""


def load_jsonl(
    path: Path,
    *,
    record_name: str,
) -> list[dict[str, Any]]:
    """Load JSON objects from a JSONL file."""
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise HeadingHierarchyError(f"{record_name} file not found: {path}") from error

    records: list[dict[str, Any]] = []

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HeadingHierarchyError(
                    f"Invalid JSON in {record_name} file on line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise HeadingHierarchyError(
                    f"{record_name} record on line {line_number} must be an object."
                )

            records.append(record)

    if not records:
        raise HeadingHierarchyError(f"{record_name} file contains no records.")

    return records


def source_node_id(heading: dict[str, Any]) -> str:
    """Create a stable node ID for a source heading."""
    return f"source:{heading['heading_id']}"


def synthetic_function_node_id(
    document_id: str,
    function_name: str,
) -> str:
    """Create a stable synthetic RMF function-node ID."""
    return f"synthetic:{document_id}:rmf-function:{function_name.lower()}"


def synthetic_category_node_id(
    document_id: str,
    function_name: str,
    category_number: str,
) -> str:
    """Create a stable synthetic RMF category-node ID."""
    return f"synthetic:{document_id}:rmf-category:{function_name.lower()}:{category_number}"


def numbered_marker(
    heading: dict[str, Any],
) -> tuple[str, int] | None:
    """Return a numbered heading key and numeric depth."""
    match = NUMBERED_PATTERN.match(heading["marker_text"])

    if not match:
        return None

    number = match.group("number")
    return number, len(number.split("."))


def rmf_marker(
    heading: dict[str, Any],
) -> tuple[str, str] | None:
    """Return RMF function and category/subcategory number."""
    match = RMF_PATTERN.match(heading["marker_text"])

    if not match:
        return None

    return (
        match.group("function").upper(),
        match.group("number"),
    )


def numbered_title(heading: dict[str, Any]) -> str:
    """Remove the leading numeric marker from a heading."""
    return NUMBERED_PATTERN.sub(
        "",
        heading["full_heading"],
        count=1,
    ).strip()


def heading_matches_function(
    heading: dict[str, Any],
    function_name: str,
) -> bool:
    """Return whether a numbered heading names an RMF function."""
    title = numbered_title(heading)
    first_word = title.split(maxsplit=1)[0] if title else ""

    return first_word.casefold() == function_name.casefold()


def group_by_document(
    headings: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group headings while preserving document order."""
    document_order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}

    for heading in headings:
        document_id = heading.get("document_id")

        if document_id not in SUPPORTED_DOCUMENT_IDS:
            raise HeadingHierarchyError(f"Unsupported document_id: {document_id}")

        if document_id not in groups:
            document_order.append(document_id)
            groups[document_id] = []

        groups[document_id].append(heading)

    return [(document_id, groups[document_id]) for document_id in document_order]


def validate_source_order(
    document_id: str,
    headings: list[dict[str, Any]],
) -> None:
    """Validate page and line ordering within a document."""
    previous_position = (0, 0)

    for heading in headings:
        position = (
            heading["page_number"],
            heading["start_line_number"],
        )

        if position <= previous_position:
            raise HeadingHierarchyError(
                f"{document_id}: headings are not in "
                f"strict source order at "
                f"{heading['heading_id']}."
            )

        previous_position = position


class HierarchyBuilder:
    """Build one ordered hierarchy with validated parent links."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.node_index: dict[str, dict[str, Any]] = {}
        self.document_orders: dict[str, int] = {}

    def add_node(
        self,
        *,
        node_id: str,
        document_id: str,
        node_kind: str,
        full_heading: str,
        parent_node_id: str | None,
        heading: dict[str, Any] | None = None,
        synthetic_type: str | None = None,
        function_name: str | None = None,
        category_number: str | None = None,
        anchor_heading: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add one source or synthetic hierarchy node."""
        if node_id in self.node_index:
            raise HeadingHierarchyError(f"Duplicate hierarchy node ID: {node_id}")

        if parent_node_id is None:
            depth = 1
            ancestor_node_ids: list[str] = []
            hierarchy_path: list[str] = []
        else:
            parent = self.node_index.get(parent_node_id)

            if parent is None:
                raise HeadingHierarchyError(f"Missing parent node: {parent_node_id}")

            if parent["document_id"] != document_id:
                raise HeadingHierarchyError(f"{node_id}: parent belongs to a different document.")

            depth = parent["depth"] + 1
            ancestor_node_ids = [
                *parent["ancestor_node_ids"],
                parent_node_id,
            ]
            hierarchy_path = [
                *parent["hierarchy_path"],
            ]

        document_order = self.document_orders.get(document_id, 0) + 1
        self.document_orders[document_id] = document_order

        if heading is not None:
            record = {
                "schema_version": "1.0",
                "node_id": node_id,
                "document_id": document_id,
                "node_kind": node_kind,
                "synthetic_type": synthetic_type,
                "heading_id": heading["heading_id"],
                "parent_node_id": parent_node_id,
                "depth": depth,
                "document_order": document_order,
                "ancestor_node_ids": ancestor_node_ids,
                "hierarchy_path": [
                    *hierarchy_path,
                    full_heading,
                ],
                "full_heading": full_heading,
                "marker_text": heading["marker_text"],
                "candidate_type": heading["candidate_type"],
                "function_name": function_name,
                "category_number": category_number,
                "page_id": heading["page_id"],
                "page_number": heading["page_number"],
                "start_line_number": heading["start_line_number"],
                "end_line_number": heading["end_line_number"],
                "anchor_heading_id": None,
                "anchor_page_id": None,
                "anchor_page_number": None,
                "anchor_line_number": None,
            }
        else:
            if anchor_heading is None:
                raise HeadingHierarchyError(
                    f"{node_id}: synthetic node requires an anchor heading."
                )

            record = {
                "schema_version": "1.0",
                "node_id": node_id,
                "document_id": document_id,
                "node_kind": node_kind,
                "synthetic_type": synthetic_type,
                "heading_id": None,
                "parent_node_id": parent_node_id,
                "depth": depth,
                "document_order": document_order,
                "ancestor_node_ids": ancestor_node_ids,
                "hierarchy_path": [
                    *hierarchy_path,
                    full_heading,
                ],
                "full_heading": full_heading,
                "marker_text": full_heading,
                "candidate_type": None,
                "function_name": function_name,
                "category_number": category_number,
                "page_id": None,
                "page_number": None,
                "start_line_number": None,
                "end_line_number": None,
                "anchor_heading_id": anchor_heading["heading_id"],
                "anchor_page_id": anchor_heading["page_id"],
                "anchor_page_number": anchor_heading["page_number"],
                "anchor_line_number": anchor_heading["start_line_number"],
            }

        self.nodes.append(record)
        self.node_index[node_id] = record
        return record


def explicit_rmf_categories(
    headings: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return RMF category nodes present in source headings."""
    categories: set[tuple[str, str]] = set()

    for heading in headings:
        marker = rmf_marker(heading)

        if marker is None:
            continue

        function_name, number = marker

        if "." not in number:
            categories.add((function_name, number))

    return categories


def build_eu_hierarchy(
    builder: HierarchyBuilder,
    headings: list[dict[str, Any]],
) -> None:
    """Build chapter/section/article/annex hierarchy."""
    active_chapter: str | None = None
    active_section: str | None = None
    active_annex: str | None = None

    for heading in headings:
        candidate_type = heading["candidate_type"]
        node_id = source_node_id(heading)

        if candidate_type == "chapter":
            parent_node_id = None

        elif candidate_type == "section":
            structural_parent = active_chapter if active_chapter is not None else active_annex

            if structural_parent is None:
                raise HeadingHierarchyError(
                    f"{heading['heading_id']}: EU section has no active chapter or annex."
                )

            parent_node_id = structural_parent

        elif candidate_type == "article":
            if active_chapter is None:
                raise HeadingHierarchyError(
                    f"{heading['heading_id']}: EU article has no active chapter."
                )

            parent_node_id = active_section if active_section is not None else active_chapter

        elif candidate_type == "annex":
            parent_node_id = None

        else:
            raise HeadingHierarchyError(
                f"{heading['heading_id']}: unsupported EU heading type {candidate_type}."
            )

        builder.add_node(
            node_id=node_id,
            document_id=heading["document_id"],
            node_kind="source",
            full_heading=heading["full_heading"],
            parent_node_id=parent_node_id,
            heading=heading,
        )

        if candidate_type == "chapter":
            active_chapter = node_id
            active_section = None
            active_annex = None

        elif candidate_type == "section":
            active_section = node_id

        elif candidate_type == "annex":
            active_chapter = None
            active_section = None
            active_annex = node_id


def build_numbered_source_node(
    builder: HierarchyBuilder,
    heading: dict[str, Any],
    numbered_stack: dict[int, str],
) -> str:
    """Build one numbered heading using exact numeric depth."""
    marker = numbered_marker(heading)

    if marker is None:
        raise HeadingHierarchyError(f"{heading['heading_id']}: expected a numbered heading marker.")

    _, numeric_depth = marker

    if numeric_depth == 1:
        parent_node_id = None
    else:
        parent_node_id = numbered_stack.get(numeric_depth - 1)

        if parent_node_id is None:
            raise HeadingHierarchyError(
                f"{heading['heading_id']}: numbered "
                f"heading has no depth-{numeric_depth - 1} "
                "parent."
            )

    node_id = source_node_id(heading)

    builder.add_node(
        node_id=node_id,
        document_id=heading["document_id"],
        node_kind="source",
        full_heading=heading["full_heading"],
        parent_node_id=parent_node_id,
        heading=heading,
    )

    numbered_stack[numeric_depth] = node_id

    for deeper_depth in list(numbered_stack):
        if deeper_depth > numeric_depth:
            del numbered_stack[deeper_depth]

    return node_id


def deepest_numbered_node(
    numbered_stack: dict[int, str],
) -> str | None:
    """Return the deepest currently active numbered node."""
    if not numbered_stack:
        return None

    return numbered_stack[max(numbered_stack)]


def ensure_rmf_function_container(
    builder: HierarchyBuilder,
    *,
    document_id: str,
    function_name: str,
    heading: dict[str, Any],
    numbered_stack: dict[int, str],
    function_nodes: dict[str, str],
) -> str:
    """Use a real function section or create a synthetic one."""
    existing = function_nodes.get(function_name)

    if existing is not None:
        return existing

    numbered_parent_id = deepest_numbered_node(numbered_stack)

    if numbered_parent_id is not None:
        numbered_parent = builder.node_index[numbered_parent_id]

        if numbered_parent["node_kind"] == "source" and heading_matches_function(
            {
                "marker_text": numbered_parent["marker_text"],
                "full_heading": numbered_parent["full_heading"],
            },
            function_name,
        ):
            function_nodes[function_name] = numbered_parent_id
            return numbered_parent_id

    node_id = synthetic_function_node_id(
        document_id,
        function_name,
    )

    builder.add_node(
        node_id=node_id,
        document_id=document_id,
        node_kind="synthetic",
        full_heading=function_name,
        parent_node_id=numbered_parent_id,
        synthetic_type="rmf_function",
        function_name=function_name,
        anchor_heading=heading,
    )

    function_nodes[function_name] = node_id
    return node_id


def build_nist_hierarchy(
    builder: HierarchyBuilder,
    headings: list[dict[str, Any]],
) -> None:
    """Build numbered, appendix, and RMF hierarchy."""
    document_id = headings[0]["document_id"]
    numbered_stack: dict[int, str] = {}
    function_nodes: dict[str, str] = {}
    category_nodes: dict[tuple[str, str], str] = {}
    explicit_categories = explicit_rmf_categories(headings)

    for heading in headings:
        candidate_type = heading["candidate_type"]
        rmf = rmf_marker(heading)

        if rmf is not None:
            function_name, number = rmf
            category_number = number.split(".")[0]

            function_parent = ensure_rmf_function_container(
                builder,
                document_id=document_id,
                function_name=function_name,
                heading=heading,
                numbered_stack=numbered_stack,
                function_nodes=function_nodes,
            )

            category_key = (
                function_name,
                category_number,
            )

            if "." not in number:
                node_id = source_node_id(heading)

                builder.add_node(
                    node_id=node_id,
                    document_id=document_id,
                    node_kind="source",
                    full_heading=heading["full_heading"],
                    parent_node_id=function_parent,
                    heading=heading,
                    function_name=function_name,
                    category_number=category_number,
                )

                category_nodes[category_key] = node_id
                continue

            category_parent = category_nodes.get(category_key)

            if category_parent is None:
                if category_key in explicit_categories:
                    raise HeadingHierarchyError(
                        f"{heading['heading_id']}: "
                        "explicit RMF category exists but "
                        "has not appeared before its child."
                    )

                category_parent = synthetic_category_node_id(
                    document_id,
                    function_name,
                    category_number,
                )

                builder.add_node(
                    node_id=category_parent,
                    document_id=document_id,
                    node_kind="synthetic",
                    full_heading=(f"{function_name} {category_number}"),
                    parent_node_id=function_parent,
                    synthetic_type="rmf_category",
                    function_name=function_name,
                    category_number=category_number,
                    anchor_heading=heading,
                )

                category_nodes[category_key] = category_parent

            builder.add_node(
                node_id=source_node_id(heading),
                document_id=document_id,
                node_kind="source",
                full_heading=heading["full_heading"],
                parent_node_id=category_parent,
                heading=heading,
                function_name=function_name,
                category_number=category_number,
            )
            continue

        if candidate_type in {
            "numbered_heading",
            "numbered_subheading",
        }:
            build_numbered_source_node(
                builder,
                heading,
                numbered_stack,
            )
            continue

        if candidate_type in {
            "appendix",
            "named_heading",
        }:
            builder.add_node(
                node_id=source_node_id(heading),
                document_id=document_id,
                node_kind="source",
                full_heading=heading["full_heading"],
                parent_node_id=None,
                heading=heading,
            )
            numbered_stack.clear()
            continue

        raise HeadingHierarchyError(
            f"{heading['heading_id']}: unsupported NIST heading type {candidate_type}."
        )


def build_openai_hierarchy(
    builder: HierarchyBuilder,
    headings: list[dict[str, Any]],
) -> None:
    """Build numeric hierarchy for the GPT-4o System Card."""
    numbered_stack: dict[int, str] = {}

    for heading in headings:
        candidate_type = heading["candidate_type"]

        if candidate_type in {
            "numbered_heading",
            "numbered_subheading",
        }:
            build_numbered_source_node(
                builder,
                heading,
                numbered_stack,
            )
        elif candidate_type in {
            "appendix",
            "named_heading",
        }:
            builder.add_node(
                node_id=source_node_id(heading),
                document_id=heading["document_id"],
                node_kind="source",
                full_heading=heading["full_heading"],
                parent_node_id=None,
                heading=heading,
            )
            numbered_stack.clear()
        else:
            raise HeadingHierarchyError(
                f"{heading['heading_id']}: unsupported OpenAI heading type {candidate_type}."
            )


def validate_hierarchy(
    nodes: list[dict[str, Any]],
) -> None:
    """Validate IDs, parent links, depths, and hierarchy paths."""
    node_index = {node["node_id"]: node for node in nodes}

    if len(node_index) != len(nodes):
        raise HeadingHierarchyError("Hierarchy contains duplicate node IDs.")

    source_heading_ids = [node["heading_id"] for node in nodes if node["node_kind"] == "source"]

    if len(source_heading_ids) != len(set(source_heading_ids)):
        raise HeadingHierarchyError("A source heading appears more than once.")

    for node in nodes:
        parent_node_id = node["parent_node_id"]

        if parent_node_id is None:
            if node["depth"] != 1:
                raise HeadingHierarchyError(f"{node['node_id']}: root depth must equal 1.")

            if node["ancestor_node_ids"]:
                raise HeadingHierarchyError(f"{node['node_id']}: root node cannot have ancestors.")
        else:
            parent = node_index.get(parent_node_id)

            if parent is None:
                raise HeadingHierarchyError(f"{node['node_id']}: missing parent.")

            if node["depth"] != parent["depth"] + 1:
                raise HeadingHierarchyError(f"{node['node_id']}: invalid depth.")

            expected_ancestors = [
                *parent["ancestor_node_ids"],
                parent_node_id,
            ]

            if node["ancestor_node_ids"] != expected_ancestors:
                raise HeadingHierarchyError(f"{node['node_id']}: invalid ancestor path.")

        if len(node["hierarchy_path"]) != node["depth"]:
            raise HeadingHierarchyError(
                f"{node['node_id']}: hierarchy path length does not match depth."
            )


def build_heading_hierarchy(
    headings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build hierarchy nodes for every controlled document."""
    builder = HierarchyBuilder()

    for document_id, document_headings in group_by_document(headings):
        validate_source_order(
            document_id,
            document_headings,
        )

        if document_id == EU_AI_ACT_ID:
            build_eu_hierarchy(
                builder,
                document_headings,
            )
        elif document_id in NIST_DOCUMENT_IDS:
            build_nist_hierarchy(
                builder,
                document_headings,
            )
        elif document_id == OPENAI_GPT4O_ID:
            build_openai_hierarchy(
                builder,
                document_headings,
            )
        else:
            raise HeadingHierarchyError(f"Unsupported document_id: {document_id}")

    validate_hierarchy(builder.nodes)
    return builder.nodes


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records to a new JSONL file."""
    if path.exists():
        raise HeadingHierarchyError(f"Output already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def write_review_report(
    path: Path,
    nodes: list[dict[str, Any]],
) -> None:
    """Write an indented hierarchy review report."""
    if path.exists():
        raise HeadingHierarchyError(f"Review report already exists: {path}")

    document_counts = Counter(node["document_id"] for node in nodes)
    synthetic_counts = Counter(
        node["document_id"] for node in nodes if node["node_kind"] == "synthetic"
    )

    lines = [
        "PolicyProof heading-hierarchy review",
        "=" * 100,
        "",
        f"Total hierarchy nodes: {len(nodes)}",
        (f"Source nodes: {sum(node['node_kind'] == 'source' for node in nodes)}"),
        (f"Synthetic nodes: {sum(node['node_kind'] == 'synthetic' for node in nodes)}"),
        "",
        "Nodes by document:",
    ]

    for document_id, count in sorted(document_counts.items()):
        lines.append(f"- {document_id}: {count} total, {synthetic_counts[document_id]} synthetic")

    current_document: str | None = None

    for node in nodes:
        document_id = node["document_id"]

        if document_id != current_document:
            lines.extend(
                [
                    "",
                    "=" * 100,
                    document_id,
                    "=" * 100,
                ]
            )
            current_document = document_id

        indentation = "  " * (node["depth"] - 1)
        kind = "SYNTHETIC" if node["node_kind"] == "synthetic" else "SOURCE"

        lines.append(f"{node['document_order']:03d} {indentation}- [{kind}] {node['full_heading']}")

        if node["node_kind"] == "synthetic":
            lines.append(f"    {indentation}anchor: {node['anchor_heading_id']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=("Assign hierarchy to reconstructed PolicyProof headings.")
    )
    parser.add_argument(
        "headings",
        type=Path,
        help="Path to reconstructed heading JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/heading-hierarchy.jsonl"),
        help="Destination hierarchy JSONL.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/heading-hierarchy-review.txt"),
        help="Destination hierarchy review report.",
    )

    args = parser.parse_args()

    try:
        headings = load_jsonl(
            args.headings,
            record_name="Reconstructed heading",
        )
        nodes = build_heading_hierarchy(headings)
        write_jsonl(args.output, nodes)
        write_review_report(args.report, nodes)
    except HeadingHierarchyError as error:
        print(f"Heading hierarchy failed: {error}")
        return 1

    source_count = sum(node["node_kind"] == "source" for node in nodes)
    synthetic_count = len(nodes) - source_count

    print(f"Heading hierarchy complete: {len(nodes)} nodes")
    print(f"- Source nodes: {source_count}")
    print(f"- Synthetic nodes: {synthetic_count}")

    for document_id, count in sorted(Counter(node["document_id"] for node in nodes).items()):
        document_synthetic = sum(
            node["document_id"] == document_id and node["node_kind"] == "synthetic"
            for node in nodes
        )

        print(f"- {document_id}: {count} total, {document_synthetic} synthetic")

    print(f"Hierarchy written to: {args.output}")
    print(f"Review report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
