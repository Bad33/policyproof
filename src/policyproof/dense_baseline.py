"""Validated orchestration for the deterministic dense baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyproof.dense_evaluation import (
    DenseEvaluationError,
    evaluate_dense,
)
from policyproof.dense_model import (
    DenseModelError,
    create_dense_session,
)
from policyproof.dense_results import (
    DenseResultError,
    build_dense_result_artifact,
    write_dense_result_artifact,
)
from policyproof.retrieval_evaluation import (
    RetrievalEvaluationError,
    validate_retrieval_evaluation_dataset,
)
from policyproof.retrieval_units import (
    RetrievalUnitError,
    load_jsonl,
)


class DenseBaselineError(RuntimeError):
    """Raised when the validated dense baseline cannot be completed."""


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading it fully into memory."""

    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    except OSError as error:
        raise DenseBaselineError(
            f"Could not hash file {path}: {error}"
        ) from error

    return digest.hexdigest()


def load_json_object(
    path: Path,
    *,
    record_name: str,
) -> dict[str, Any]:
    """Load one JSON object and fail closed on malformed input."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DenseBaselineError(
            f"Could not read {record_name} file {path}: {error}"
        ) from error

    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise DenseBaselineError(
            f"{path}: invalid JSON: {error.msg}"
        ) from error

    if not isinstance(value, dict):
        raise DenseBaselineError(
            f"{path}: {record_name} must be a JSON object."
        )

    return value


def run_dense_baseline(
    *,
    manifest_path: Path,
    passages_path: Path,
    benchmark_path: Path,
    model_path: Path,
    output_path: Path,
    result_version: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Validate inputs, evaluate dense retrieval, and publish one result."""

    for field_name, value in (
        ("manifest_path", manifest_path),
        ("passages_path", passages_path),
        ("benchmark_path", benchmark_path),
        ("model_path", model_path),
        ("output_path", output_path),
    ):
        if not isinstance(value, Path):
            raise DenseBaselineError(
                f"{field_name} must be a pathlib.Path."
            )

    if output_path.exists():
        raise DenseBaselineError(
            f"Output already exists: {output_path}"
        )

    manifest = load_json_object(
        manifest_path,
        record_name="corpus manifest",
    )
    benchmark = load_json_object(
        benchmark_path,
        record_name="retrieval benchmark",
    )

    try:
        passages = load_jsonl(
            passages_path,
            record_name="retrieval passage",
        )
    except RetrievalUnitError as error:
        raise DenseBaselineError(str(error)) from error

    benchmark_sha256 = sha256_file(benchmark_path)
    passage_sha256 = sha256_file(passages_path)

    try:
        validate_retrieval_evaluation_dataset(
            benchmark,
            manifest=manifest,
            passage_records=passages,
            passage_artifact_sha256=passage_sha256,
        )

        session = create_dense_session(model_path)

        evaluation = evaluate_dense(
            passages,
            benchmark,
            session=session,
            batch_size=batch_size,
        )

        artifact = build_dense_result_artifact(
            evaluation,
            dataset=benchmark,
            result_version=result_version,
            benchmark_sha256=benchmark_sha256,
            passage_sha256=passage_sha256,
        )

        write_dense_result_artifact(
            artifact,
            output_path,
        )
    except (
        RetrievalEvaluationError,
        DenseModelError,
        DenseEvaluationError,
        DenseResultError,
    ) as error:
        raise DenseBaselineError(str(error)) from error

    if not isinstance(artifact, Mapping):
        raise DenseBaselineError(
            "Dense result artifact must be a mapping."
        )

    return dict(artifact)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from error

    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return parsed


def _parse_args() -> argparse.Namespace:
    """Parse explicit paths for one reproducible dense baseline run."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and publish the deterministic "
            "PolicyProof dense baseline."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--passages",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--result-version",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=32,
    )

    return parser.parse_args()


def main() -> None:
    """Run one validated dense baseline from explicit inputs."""

    args = _parse_args()
    artifact = run_dense_baseline(
        manifest_path=args.manifest,
        passages_path=args.passages,
        benchmark_path=args.benchmark,
        model_path=args.model,
        output_path=args.output,
        result_version=args.result_version,
        batch_size=args.batch_size,
    )

    evaluation = artifact["evaluation"]
    metrics = evaluation["aggregate_metrics"]
    passage_count = artifact["bindings"]["passages"]["count"]

    print(
        "Dense baseline complete: "
        f"{passage_count} passages, "
        f"{evaluation['answer_query_count']} answer queries, "
        f"mean Recall@10={metrics['mean_recall_at_10']:.10f}, "
        f"MRR@10={metrics['mrr_at_10']:.10f}, "
        f"mean nDCG@10={metrics['mean_ndcg_at_10']:.10f}"
    )


if __name__ == "__main__":
    main()
