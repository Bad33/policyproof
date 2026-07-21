"""Validation for versioned retrieval-evaluation datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

ANSWER_BEHAVIOR = "answer"
ABSTAIN_BEHAVIOR = "abstain"
ALLOWED_EXPECTED_BEHAVIORS = {
    ANSWER_BEHAVIOR,
    ABSTAIN_BEHAVIOR,
}
ALLOWED_RELEVANCE_GRADES = {
    1,
    2,
}

RETRIEVAL_EVALUATION_DATASET_ID = (
    "policyproof-retrieval-evaluation"
)
DATASET_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_version",
        "corpus_id",
        "corpus_version",
        "passage_schema_version",
        "passage_artifact_sha256",
        "query_count",
        "queries",
    }
)
QUERY_FIELDS = frozenset(
    {
        "query_id",
        "question",
        "expected_behavior",
        "document_scope",
        "evaluation_tags",
        "relevance_judgments",
    }
)
RELEVANCE_JUDGMENT_FIELDS = frozenset(
    {
        "passage_id",
        "relevance_grade",
        "rationale",
    }
)


class RetrievalEvaluationError(ValueError):
    """Raised when a retrieval-evaluation dataset is invalid."""


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetrievalEvaluationError(
            f"{field_name} must be an object."
        )

    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str],
    object_name: str,
) -> None:
    unknown_fields = sorted(
        set(value) - allowed_fields
    )

    if unknown_fields:
        raise RetrievalEvaluationError(
            f"unknown {object_name} fields: "
            f"{unknown_fields}."
        )


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalEvaluationError(
            f"{field_name} must be a nonempty string."
        )

    return value


def _require_version(
    value: Any,
    *,
    field_name: str,
) -> str:
    version = _require_nonempty_string(
        value,
        field_name=field_name,
    )

    if not VERSION_PATTERN.fullmatch(version):
        raise RetrievalEvaluationError(
            f"{field_name} must use semantic version form X.Y.Z."
        )

    return version


def _require_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise RetrievalEvaluationError(
            f"{field_name} must be a list."
        )

    return value


def _manifest_document_ids(
    manifest: Mapping[str, Any],
) -> frozenset[str]:
    documents = _require_sequence(
        manifest.get("documents"),
        field_name="manifest.documents",
    )

    document_ids: set[str] = set()

    for position, document_value in enumerate(documents):
        document = _require_mapping(
            document_value,
            field_name=f"manifest.documents[{position}]",
        )
        document_id = _require_nonempty_string(
            document.get("document_id"),
            field_name=(
                f"manifest.documents[{position}].document_id"
            ),
        )

        if document_id in document_ids:
            raise RetrievalEvaluationError(
                f"Duplicate manifest document_id: {document_id}."
            )

        document_ids.add(document_id)

    return frozenset(document_ids)


def _passage_indexes(
    passage_records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, str],
    frozenset[str],
]:
    passage_documents: dict[str, str] = {}
    schema_versions: set[str] = set()

    for position, passage_value in enumerate(passage_records):
        passage = _require_mapping(
            passage_value,
            field_name=f"passage_records[{position}]",
        )
        passage_id = _require_nonempty_string(
            passage.get("passage_id"),
            field_name=(
                f"passage_records[{position}].passage_id"
            ),
        )
        document_id = _require_nonempty_string(
            passage.get("document_id"),
            field_name=(
                f"passage_records[{position}].document_id"
            ),
        )
        schema_version = _require_nonempty_string(
            passage.get("schema_version"),
            field_name=(
                f"passage_records[{position}].schema_version"
            ),
        )

        if passage_id in passage_documents:
            raise RetrievalEvaluationError(
                f"Duplicate passage_id: {passage_id}."
            )

        passage_documents[passage_id] = document_id
        schema_versions.add(schema_version)

    if not passage_documents:
        raise RetrievalEvaluationError(
            "passage_records must be nonempty."
        )

    return (
        passage_documents,
        frozenset(schema_versions),
    )


def _validate_document_scope(
    value: Any,
    *,
    query_id: str,
    known_document_ids: frozenset[str],
) -> tuple[str, ...]:
    scope_values = _require_sequence(
        value,
        field_name=f"{query_id}.document_scope",
    )

    if not scope_values:
        raise RetrievalEvaluationError(
            f"{query_id}.document_scope must be nonempty."
        )

    document_scope: list[str] = []
    seen: set[str] = set()

    for position, document_value in enumerate(scope_values):
        document_id = _require_nonempty_string(
            document_value,
            field_name=(
                f"{query_id}.document_scope[{position}]"
            ),
        )

        if document_id not in known_document_ids:
            raise RetrievalEvaluationError(
                f"{query_id} references unknown document "
                f"{document_id!r}."
            )

        if document_id in seen:
            raise RetrievalEvaluationError(
                f"{query_id} contains duplicate document_scope "
                f"value {document_id!r}."
            )

        seen.add(document_id)
        document_scope.append(document_id)

    return tuple(document_scope)


def _validate_evaluation_tags(
    value: Any,
    *,
    query_id: str,
) -> None:
    tag_values = _require_sequence(
        value,
        field_name=f"{query_id}.evaluation_tags",
    )

    if not tag_values:
        raise RetrievalEvaluationError(
            f"{query_id} requires at least one evaluation tag."
        )

    seen: set[str] = set()

    for position, tag_value in enumerate(tag_values):
        tag = _require_nonempty_string(
            tag_value,
            field_name=(
                f"{query_id}.evaluation_tags[{position}]"
            ),
        )

        if tag in seen:
            raise RetrievalEvaluationError(
                f"{query_id} contains duplicate evaluation tag "
                f"{tag!r}."
            )

        seen.add(tag)


def _validate_relevance_judgments(
    value: Any,
    *,
    query_id: str,
    expected_behavior: str,
    document_scope: tuple[str, ...],
    passage_documents: Mapping[str, str],
) -> None:
    judgments = _require_sequence(
        value,
        field_name=f"{query_id}.relevance_judgments",
    )

    if expected_behavior == ANSWER_BEHAVIOR and not judgments:
        raise RetrievalEvaluationError(
            f"{query_id} requires at least one relevance judgment."
        )

    if expected_behavior == ABSTAIN_BEHAVIOR and judgments:
        raise RetrievalEvaluationError(
            f"{query_id} must not contain relevance judgments."
        )

    seen_passages: set[str] = set()
    has_direct_judgment = False

    for position, judgment_value in enumerate(judgments):
        judgment = _require_mapping(
            judgment_value,
            field_name=(
                f"{query_id}.relevance_judgments[{position}]"
            ),
        )
        _reject_unknown_fields(
            judgment,
            allowed_fields=RELEVANCE_JUDGMENT_FIELDS,
            object_name="relevance judgment",
        )
        passage_id = _require_nonempty_string(
            judgment.get("passage_id"),
            field_name=(
                f"{query_id}.relevance_judgments"
                f"[{position}].passage_id"
            ),
        )

        if passage_id in seen_passages:
            raise RetrievalEvaluationError(
                f"Duplicate passage judgment for {passage_id} "
                f"in {query_id}."
            )

        if passage_id not in passage_documents:
            raise RetrievalEvaluationError(
                f"{query_id} references unknown passage "
                f"{passage_id!r}."
            )

        passage_document_id = passage_documents[passage_id]

        if passage_document_id not in document_scope:
            raise RetrievalEvaluationError(
                f"{query_id} passage {passage_id!r} is outside "
                "document_scope."
            )

        grade = judgment.get("relevance_grade")

        if (
            not isinstance(grade, int)
            or isinstance(grade, bool)
            or grade not in ALLOWED_RELEVANCE_GRADES
        ):
            raise RetrievalEvaluationError(
                f"{query_id} relevance_grade must be integer "
                "1 or 2."
            )

        _require_nonempty_string(
            judgment.get("rationale"),
            field_name=(
                f"{query_id}.relevance_judgments"
                f"[{position}].rationale"
            ),
        )

        seen_passages.add(passage_id)
        has_direct_judgment = (
            has_direct_judgment
            or grade == 2
        )

    if (
        expected_behavior == ANSWER_BEHAVIOR
        and not has_direct_judgment
    ):
        raise RetrievalEvaluationError(
            f"{query_id} requires at least one grade 2 "
            "relevance judgment."
        )


def validate_retrieval_evaluation_dataset(
    dataset: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    passage_records: Sequence[Mapping[str, Any]],
    passage_artifact_sha256: str,
) -> None:
    """Validate one retrieval-evaluation dataset without mutation."""

    dataset = _require_mapping(
        dataset,
        field_name="dataset",
    )
    _reject_unknown_fields(
        dataset,
        allowed_fields=DATASET_FIELDS,
        object_name="dataset",
    )
    manifest = _require_mapping(
        manifest,
        field_name="manifest",
    )

    schema_version = _require_nonempty_string(
        dataset.get("schema_version"),
        field_name="schema_version",
    )

    if schema_version != "1.0":
        raise RetrievalEvaluationError(
            "schema_version must be '1.0'."
        )

    dataset_id = _require_nonempty_string(
        dataset.get("dataset_id"),
        field_name="dataset_id",
    )

    if dataset_id != RETRIEVAL_EVALUATION_DATASET_ID:
        raise RetrievalEvaluationError(
            "dataset_id must be "
            f"{RETRIEVAL_EVALUATION_DATASET_ID!r}."
        )

    _require_version(
        dataset.get("dataset_version"),
        field_name="dataset_version",
    )

    dataset_corpus_id = _require_nonempty_string(
        dataset.get("corpus_id"),
        field_name="corpus_id",
    )
    manifest_corpus_id = _require_nonempty_string(
        manifest.get("corpus_id"),
        field_name="manifest.corpus_id",
    )

    if dataset_corpus_id != manifest_corpus_id:
        raise RetrievalEvaluationError(
            "corpus_id does not match the source manifest."
        )

    dataset_corpus_version = _require_version(
        dataset.get("corpus_version"),
        field_name="corpus_version",
    )
    manifest_corpus_version = _require_version(
        manifest.get("corpus_version"),
        field_name="manifest.corpus_version",
    )

    if dataset_corpus_version != manifest_corpus_version:
        raise RetrievalEvaluationError(
            "corpus_version does not match the source manifest."
        )

    expected_passage_hash = _require_nonempty_string(
        passage_artifact_sha256,
        field_name="passage_artifact_sha256 argument",
    )

    if not SHA256_PATTERN.fullmatch(expected_passage_hash):
        raise RetrievalEvaluationError(
            "passage_artifact_sha256 argument must be a "
            "lowercase SHA-256 value."
        )

    dataset_passage_hash = _require_nonempty_string(
        dataset.get("passage_artifact_sha256"),
        field_name="passage_artifact_sha256",
    )

    if not SHA256_PATTERN.fullmatch(dataset_passage_hash):
        raise RetrievalEvaluationError(
            "passage_artifact_sha256 must be a lowercase "
            "SHA-256 value."
        )

    if dataset_passage_hash != expected_passage_hash:
        raise RetrievalEvaluationError(
            "Dataset passage artifact SHA-256 does not match "
            "the supplied passage artifact."
        )

    passage_documents, passage_schema_versions = (
        _passage_indexes(passage_records)
    )

    dataset_passage_schema = _require_nonempty_string(
        dataset.get("passage_schema_version"),
        field_name="passage_schema_version",
    )

    if passage_schema_versions != {
        dataset_passage_schema
    }:
        raise RetrievalEvaluationError(
            "passage_schema_version does not match all "
            "passage records."
        )

    known_document_ids = _manifest_document_ids(manifest)

    passage_document_ids = set(
        passage_documents.values()
    )
    unknown_passage_documents = sorted(
        passage_document_ids - known_document_ids
    )

    if unknown_passage_documents:
        raise RetrievalEvaluationError(
            "Passage records reference unknown manifest "
            f"documents: {unknown_passage_documents}."
        )

    queries = _require_sequence(
        dataset.get("queries"),
        field_name="queries",
    )

    if not queries:
        raise RetrievalEvaluationError(
            "Dataset must contain at least one query."
        )

    query_count = dataset.get("query_count")

    if (
        not isinstance(query_count, int)
        or isinstance(query_count, bool)
        or query_count < 0
    ):
        raise RetrievalEvaluationError(
            "query_count must be a nonnegative integer."
        )

    if query_count != len(queries):
        raise RetrievalEvaluationError(
            "query_count does not match the number of queries."
        )

    seen_query_ids: set[str] = set()

    for position, query_value in enumerate(queries):
        query = _require_mapping(
            query_value,
            field_name=f"queries[{position}]",
        )
        _reject_unknown_fields(
            query,
            allowed_fields=QUERY_FIELDS,
            object_name="query",
        )
        query_id = _require_nonempty_string(
            query.get("query_id"),
            field_name=f"queries[{position}].query_id",
        )

        if query_id in seen_query_ids:
            raise RetrievalEvaluationError(
                f"Duplicate query_id: {query_id}."
            )

        seen_query_ids.add(query_id)

        _require_nonempty_string(
            query.get("question"),
            field_name=f"{query_id}.question",
        )

        expected_behavior = _require_nonempty_string(
            query.get("expected_behavior"),
            field_name=f"{query_id}.expected_behavior",
        )

        if expected_behavior not in ALLOWED_EXPECTED_BEHAVIORS:
            raise RetrievalEvaluationError(
                f"{query_id}.expected_behavior must be "
                "'answer' or 'abstain'."
            )

        document_scope = _validate_document_scope(
            query.get("document_scope"),
            query_id=query_id,
            known_document_ids=known_document_ids,
        )
        _validate_evaluation_tags(
            query.get("evaluation_tags"),
            query_id=query_id,
        )
        _validate_relevance_judgments(
            query.get("relevance_judgments"),
            query_id=query_id,
            expected_behavior=expected_behavior,
            document_scope=document_scope,
            passage_documents=passage_documents,
        )
