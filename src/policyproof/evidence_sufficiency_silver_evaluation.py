# Query-grouped silver-label evaluation utilities.

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "can",
        "does",
        "for",
        "from",
        "how",
        "if",
        "in",
        "including",
        "is",
        "it",
        "its",
        "may",
        "must",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "these",
        "they",
        "this",
        "to",
        "under",
        "use",
        "used",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
    }
)
FEATURE_NAMES = (
    "evidence_count",
    "log_evidence_word_count",
    "question_content_coverage",
    "citation_label_content_coverage",
    "evidence_content_density",
)
SPLIT_NAMES = ("train", "validation", "test")
DOCUMENT_TARGETS = {
    "eu-ai-act-2024-1689": {"train": 13, "validation": 5, "test": 4},
    "nist-ai-600-1-genai-profile": {"train": 12, "validation": 4, "test": 4},
    "nist-ai-rmf-1.0": {"train": 12, "validation": 4, "test": 4},
    "openai-gpt-4o-system-card-2024-08-08": {
        "train": 11,
        "validation": 3,
        "test": 4,
    },
}


class SilverEvaluationError(ValueError):
    pass


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in WORD_PATTERN.findall(text)]


def _content_tokens(text: str) -> set[str]:
    return {item for item in _tokens(text) if item not in STOPWORDS and len(item) > 1}


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_features(case: dict[str, Any]) -> tuple[float, ...]:
    evidence = case["evidence"]
    if not evidence:
        raise SilverEvaluationError("Evidence must be nonempty.")

    question_tokens = _content_tokens(case["question"])
    citation_text = "\n".join(item["citation_text"] for item in evidence)
    citation_labels = "\n".join(item["label"] for item in evidence)
    evidence_tokens = _content_tokens(citation_text)
    label_tokens = _content_tokens(citation_labels)
    evidence_words = _tokens(citation_text)

    question_coverage = (
        len(question_tokens & evidence_tokens) / len(question_tokens) if question_tokens else 0.0
    )
    label_coverage = (
        len(question_tokens & label_tokens) / len(question_tokens) if question_tokens else 0.0
    )
    density = len(evidence_tokens) / max(1, len(evidence_words))

    return (
        float(len(evidence)),
        math.log1p(len(evidence_words)),
        question_coverage,
        label_coverage,
        density,
    )


def _contracts(
    batch: dict[str, Any],
    labels: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = batch["cases"]
    label_records = labels["labels"]

    if batch["case_count"] != len(cases):
        raise SilverEvaluationError("Batch case_count mismatch.")
    if labels["case_count"] != len(label_records):
        raise SilverEvaluationError("Label case_count mismatch.")

    label_by_id = {item["case_id"]: item for item in label_records}

    if len(label_by_id) != len(label_records):
        raise SilverEvaluationError("Duplicate silver-label case ID.")

    if [item["case_id"] for item in cases] != [item["case_id"] for item in label_records]:
        raise SilverEvaluationError("Batch and label order differ.")

    return cases, label_by_id


def build_grouped_splits(
    batch: dict[str, Any],
    labels: dict[str, Any],
    *,
    construction_sha256: str,
    annotation_batch_sha256: str,
    silver_label_set_sha256: str,
) -> dict[str, Any]:
    cases, label_by_id = _contracts(batch, labels)
    groups: dict[str, dict[str, Any]] = {}

    for case in cases:
        query_id = case["query_id"]
        document_id = case["evidence"][0]["document_id"]
        group = groups.setdefault(
            query_id,
            {
                "query_id": query_id,
                "document_id": document_id,
                "case_ids": [],
                "insufficient_count": 0,
            },
        )
        if group["document_id"] != document_id:
            raise SilverEvaluationError("A query spans multiple documents.")

        group["case_ids"].append(case["case_id"])
        if label_by_id[case["case_id"]]["evidence_status"] == "insufficient":
            group["insufficient_count"] += 1

    by_document: dict[str, list[dict[str, Any]]] = {}
    for group in groups.values():
        by_document.setdefault(group["document_id"], []).append(group)

    if set(by_document) != set(DOCUMENT_TARGETS):
        raise SilverEvaluationError("Unexpected document set.")

    assignment: dict[str, str] = {}

    for document_id in sorted(by_document):
        document_groups = by_document[document_id]
        targets = DOCUMENT_TARGETS[document_id]

        if sum(targets.values()) != len(document_groups):
            raise SilverEvaluationError("Document split targets mismatch.")

        ordered = sorted(
            document_groups,
            key=lambda item: (
                -item["insufficient_count"],
                -len(item["case_ids"]),
                _stable_hash(item["query_id"]),
            ),
        )
        remaining = dict(targets)
        assigned_cases = Counter()
        assigned_incomplete = Counter()

        total_cases = sum(len(item["case_ids"]) for item in document_groups)
        total_incomplete = sum(item["insufficient_count"] for item in document_groups)

        for group in ordered:
            candidates: list[tuple[float, str]] = []

            for split_name in SPLIT_NAMES:
                if remaining[split_name] == 0:
                    continue

                ratio = targets[split_name] / len(document_groups)
                target_cases = total_cases * ratio
                target_incomplete = total_incomplete * ratio
                new_cases = assigned_cases[split_name] + len(group["case_ids"])
                new_incomplete = assigned_incomplete[split_name] + group["insufficient_count"]

                score = (
                    ((new_cases - target_cases) / max(1.0, target_cases)) ** 2
                    + ((new_incomplete - target_incomplete) / max(1.0, target_incomplete)) ** 2
                    + (1.0 / remaining[split_name])
                )
                candidates.append((score, split_name))

            if not candidates:
                raise SilverEvaluationError("No split capacity remains.")

            _, selected = min(
                candidates,
                key=lambda item: (item[0], SPLIT_NAMES.index(item[1])),
            )
            assignment[group["query_id"]] = selected
            remaining[selected] -= 1
            assigned_cases[selected] += len(group["case_ids"])
            assigned_incomplete[selected] += group["insufficient_count"]

        if any(remaining.values()):
            raise SilverEvaluationError("Split target was not filled.")

    split_records = []

    for split_name in SPLIT_NAMES:
        query_ids = sorted(
            query_id for query_id, assigned in assignment.items() if assigned == split_name
        )
        query_set = set(query_ids)
        case_ids = [case["case_id"] for case in cases if case["query_id"] in query_set]
        statuses = [label_by_id[case_id]["evidence_status"] for case_id in case_ids]
        split_records.append(
            {
                "split_name": split_name,
                "query_count": len(query_ids),
                "case_count": len(case_ids),
                "sufficient_count": statuses.count("sufficient"),
                "insufficient_count": statuses.count("insufficient"),
                "query_ids": query_ids,
                "case_ids": case_ids,
            }
        )

    return {
        "schema_version": "1.0",
        "split_id": "policyproof-evidence-sufficiency-silver-splits",
        "split_version": "0.1.0",
        "label_provenance": "construction_derived",
        "construction_sha256": construction_sha256,
        "annotation_batch_sha256": annotation_batch_sha256,
        "silver_label_set_sha256": silver_label_set_sha256,
        "grouping_key": "query_id",
        "assignment_method": "deterministic_document_stratified_greedy",
        "case_count": len(cases),
        "query_count": len(groups),
        "splits": split_records,
    }


def validate_grouped_splits(
    artifact: dict[str, Any],
    batch: dict[str, Any],
    labels: dict[str, Any],
    *,
    construction_sha256: str,
    annotation_batch_sha256: str,
    silver_label_set_sha256: str,
) -> None:
    cases, label_by_id = _contracts(batch, labels)

    expected_top = {
        "schema_version",
        "split_id",
        "split_version",
        "label_provenance",
        "construction_sha256",
        "annotation_batch_sha256",
        "silver_label_set_sha256",
        "grouping_key",
        "assignment_method",
        "case_count",
        "query_count",
        "splits",
    }
    if set(artifact) != expected_top:
        raise SilverEvaluationError("Unexpected split artifact fields.")

    expected_bindings = {
        "schema_version": "1.0",
        "split_id": "policyproof-evidence-sufficiency-silver-splits",
        "split_version": "0.1.0",
        "label_provenance": "construction_derived",
        "construction_sha256": construction_sha256,
        "annotation_batch_sha256": annotation_batch_sha256,
        "silver_label_set_sha256": silver_label_set_sha256,
        "grouping_key": "query_id",
        "assignment_method": "deterministic_document_stratified_greedy",
    }
    for field_name, expected in expected_bindings.items():
        if artifact[field_name] != expected:
            raise SilverEvaluationError(f"{field_name} binding mismatch.")

    if artifact["case_count"] != len(cases):
        raise SilverEvaluationError("Split case coverage mismatch.")

    expected_queries = {case["query_id"] for case in cases}
    if artifact["query_count"] != len(expected_queries):
        raise SilverEvaluationError("Split query coverage mismatch.")

    seen_queries: set[str] = set()
    seen_cases: set[str] = set()

    for split in artifact["splits"]:
        if set(split) != {
            "split_name",
            "query_count",
            "case_count",
            "sufficient_count",
            "insufficient_count",
            "query_ids",
            "case_ids",
        }:
            raise SilverEvaluationError("Unexpected split record fields.")

        query_ids = split["query_ids"]
        case_ids = split["case_ids"]

        if len(query_ids) != len(set(query_ids)):
            raise SilverEvaluationError("Duplicate query in split.")
        if len(case_ids) != len(set(case_ids)):
            raise SilverEvaluationError("Duplicate case in split.")
        if seen_queries & set(query_ids):
            raise SilverEvaluationError("Query leakage across splits.")
        if seen_cases & set(case_ids):
            raise SilverEvaluationError("Case leakage across splits.")

        seen_queries.update(query_ids)
        seen_cases.update(case_ids)

        if split["query_count"] != len(query_ids):
            raise SilverEvaluationError("Split query_count mismatch.")
        if split["case_count"] != len(case_ids):
            raise SilverEvaluationError("Split case_count mismatch.")

        query_set = set(query_ids)
        expected_case_ids = [case["case_id"] for case in cases if case["query_id"] in query_set]
        if case_ids != expected_case_ids:
            raise SilverEvaluationError("Query group or case order mismatch.")

        statuses = [label_by_id[case_id]["evidence_status"] for case_id in case_ids]
        if split["sufficient_count"] != statuses.count("sufficient"):
            raise SilverEvaluationError("Sufficient count mismatch.")
        if split["insufficient_count"] != statuses.count("insufficient"):
            raise SilverEvaluationError("Insufficient count mismatch.")

    if seen_queries != expected_queries:
        raise SilverEvaluationError("Not all query groups are covered.")
    if seen_cases != {case["case_id"] for case in cases}:
        raise SilverEvaluationError("Not all cases are covered.")


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _standardization(rows: list[tuple[float, ...]]) -> tuple[list[float], list[float]]:
    means = []
    scales = []

    for index in range(len(rows[0])):
        values = [row[index] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        scales.append(math.sqrt(variance) or 1.0)

    return means, scales


def _standardize(
    row: tuple[float, ...],
    means: list[float],
    scales: list[float],
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)
    )


def _probability(
    row: tuple[float, ...],
    coefficients: list[float],
    intercept: float,
) -> float:
    return _sigmoid(
        intercept
        + sum(coefficient * value for coefficient, value in zip(coefficients, row, strict=True))
    )


def _fit(
    rows: list[tuple[float, ...]],
    targets: list[int],
) -> tuple[list[float], float]:
    if set(targets) != {0, 1}:
        raise SilverEvaluationError("Training split must contain both classes.")

    coefficients = [0.0] * len(rows[0])
    positive_rate = sum(targets) / len(targets)
    intercept = math.log(positive_rate / (1.0 - positive_rate))
    iterations = 2500
    learning_rate = 0.05
    l2 = 0.01

    for _ in range(iterations):
        coefficient_gradients = [0.0] * len(coefficients)
        intercept_gradient = 0.0

        for row, target in zip(rows, targets, strict=True):
            error = _probability(row, coefficients, intercept) - target
            intercept_gradient += error
            for index, value in enumerate(row):
                coefficient_gradients[index] += error * value

        intercept -= learning_rate * intercept_gradient / len(rows)

        for index in range(len(coefficients)):
            gradient = coefficient_gradients[index] / len(rows) + l2 * coefficients[index]
            coefficients[index] -= learning_rate * gradient

    return coefficients, intercept


def _metrics(
    targets: list[int],
    predictions: list[int],
    split_name: str,
) -> dict[str, Any]:
    tp = sum(a == 1 and b == 1 for a, b in zip(targets, predictions, strict=True))
    tn = sum(a == 0 and b == 0 for a, b in zip(targets, predictions, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(targets, predictions, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(targets, predictions, strict=True))

    accuracy = (tp + tn) / len(targets)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0

    return {
        "split_name": split_name,
        "case_count": len(targets),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": round(accuracy, 12),
        "balanced_accuracy": round((tpr + tnr) / 2.0, 12),
        "precision": round(precision, 12),
        "recall": round(tpr, 12),
        "f1": round(f1, 12),
    }


def build_baseline(
    batch: dict[str, Any],
    labels: dict[str, Any],
    splits: dict[str, Any],
    *,
    annotation_batch_sha256: str,
    silver_label_set_sha256: str,
    split_artifact_sha256: str,
) -> dict[str, Any]:
    cases, label_by_id = _contracts(batch, labels)
    case_by_id = {case["case_id"]: case for case in cases}
    split_case_ids = {item["split_name"]: item["case_ids"] for item in splits["splits"]}

    features = {case["case_id"]: extract_features(case) for case in cases}
    targets = {
        case_id: int(label_by_id[case_id]["evidence_status"] == "sufficient")
        for case_id in case_by_id
    }

    train_ids = split_case_ids["train"]
    train_rows = [features[case_id] for case_id in train_ids]
    train_targets = [targets[case_id] for case_id in train_ids]
    means, scales = _standardization(train_rows)
    standardized = {case_id: _standardize(row, means, scales) for case_id, row in features.items()}
    coefficients, intercept = _fit(
        [standardized[case_id] for case_id in train_ids],
        train_targets,
    )

    validation_ids = split_case_ids["validation"]
    validation_probabilities = [
        _probability(standardized[case_id], coefficients, intercept) for case_id in validation_ids
    ]
    validation_targets = [targets[case_id] for case_id in validation_ids]

    candidates = sorted(
        {
            0.0,
            0.5,
            1.0,
            *validation_probabilities,
            *[
                (left + right) / 2.0
                for left, right in zip(
                    sorted(set(validation_probabilities)),
                    sorted(set(validation_probabilities))[1:],
                )
            ],
        }
    )

    scored_thresholds = []
    for threshold in candidates:
        predictions = [int(probability >= threshold) for probability in validation_probabilities]
        metric = _metrics(validation_targets, predictions, "validation")
        scored_thresholds.append(
            (
                metric["balanced_accuracy"],
                metric["f1"],
                -abs(threshold - 0.5),
                threshold,
            )
        )

    threshold = max(scored_thresholds)[-1]
    metric_records = []

    for split_name in SPLIT_NAMES:
        case_ids = split_case_ids[split_name]
        probabilities = [
            _probability(standardized[case_id], coefficients, intercept) for case_id in case_ids
        ]
        split_targets = [targets[case_id] for case_id in case_ids]
        predictions = [int(probability >= threshold) for probability in probabilities]
        metric_records.append(_metrics(split_targets, predictions, split_name))

    test_predictions = []
    for case_id in split_case_ids["test"]:
        probability = _probability(standardized[case_id], coefficients, intercept)
        test_predictions.append(
            {
                "case_id": case_id,
                "query_id": case_by_id[case_id]["query_id"],
                "gold_evidence_status": label_by_id[case_id]["evidence_status"],
                "predicted_evidence_status": (
                    "sufficient" if probability >= threshold else "insufficient"
                ),
                "probability_sufficient": round(probability, 12),
            }
        )

    return {
        "schema_version": "1.0",
        "result_id": "policyproof-evidence-sufficiency-silver-baseline",
        "result_version": "0.1.0",
        "label_provenance": "construction_derived",
        "annotation_batch_sha256": annotation_batch_sha256,
        "silver_label_set_sha256": silver_label_set_sha256,
        "split_artifact_sha256": split_artifact_sha256,
        "model_type": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "training": {
            "iterations": 2500,
            "learning_rate": 0.05,
            "l2_penalty": 0.01,
            "feature_means": [round(item, 12) for item in means],
            "feature_scales": [round(item, 12) for item in scales],
            "coefficients": [round(item, 12) for item in coefficients],
            "intercept": round(intercept, 12),
            "selected_threshold": round(threshold, 12),
            "threshold_selection_metric": "validation_balanced_accuracy_then_f1",
        },
        "metrics": metric_records,
        "test_predictions": test_predictions,
    }


def validate_baseline(
    artifact: dict[str, Any],
    batch: dict[str, Any],
    labels: dict[str, Any],
    splits: dict[str, Any],
    *,
    annotation_batch_sha256: str,
    silver_label_set_sha256: str,
    split_artifact_sha256: str,
) -> None:
    cases, label_by_id = _contracts(batch, labels)
    case_by_id = {case["case_id"]: case for case in cases}

    expected_fields = {
        "schema_version",
        "result_id",
        "result_version",
        "label_provenance",
        "annotation_batch_sha256",
        "silver_label_set_sha256",
        "split_artifact_sha256",
        "model_type",
        "feature_names",
        "training",
        "metrics",
        "test_predictions",
    }
    if set(artifact) != expected_fields:
        raise SilverEvaluationError("Unexpected baseline fields.")

    expected_bindings = {
        "schema_version": "1.0",
        "result_id": "policyproof-evidence-sufficiency-silver-baseline",
        "result_version": "0.1.0",
        "label_provenance": "construction_derived",
        "annotation_batch_sha256": annotation_batch_sha256,
        "silver_label_set_sha256": silver_label_set_sha256,
        "split_artifact_sha256": split_artifact_sha256,
        "model_type": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
    }
    for field_name, expected in expected_bindings.items():
        if artifact[field_name] != expected:
            raise SilverEvaluationError(f"{field_name} binding mismatch.")

    metric_by_split = {item["split_name"]: item for item in artifact["metrics"]}
    if set(metric_by_split) != set(SPLIT_NAMES):
        raise SilverEvaluationError("Metrics do not cover all splits.")

    test_ids = next(item["case_ids"] for item in splits["splits"] if item["split_name"] == "test")
    predictions = artifact["test_predictions"]

    if [item["case_id"] for item in predictions] != test_ids:
        raise SilverEvaluationError("Test prediction order mismatch.")

    for prediction in predictions:
        case_id = prediction["case_id"]
        if prediction["query_id"] != case_by_id[case_id]["query_id"]:
            raise SilverEvaluationError("Prediction query mismatch.")
        if prediction["gold_evidence_status"] != label_by_id[case_id]["evidence_status"]:
            raise SilverEvaluationError("Prediction gold label mismatch.")
        if not 0.0 <= prediction["probability_sufficient"] <= 1.0:
            raise SilverEvaluationError("Probability is outside [0, 1].")


def write_json_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary_name, output_path)
    except FileExistsError as error:
        raise SilverEvaluationError(
            f"Refusing to overwrite existing artifact: {output_path}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
