"""Validated orchestration for hybrid candidate-union coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyproof.dense_model import (
    DenseModelError,
    create_dense_session,
)
from policyproof.hybrid_candidate_evaluation import (
    HybridCandidateEvaluationError,
    evaluate_hybrid_candidates,
)
from policyproof.hybrid_candidate_results import (
    HybridCandidateResultError,
    build_hybrid_candidate_result_artifact,
    write_hybrid_candidate_result_artifact,
)
from policyproof.retrieval_evaluation import (
    RetrievalEvaluationError,
    validate_retrieval_evaluation_dataset,
)
from policyproof.retrieval_units import (
    RetrievalUnitError,
    load_jsonl,
)

DEFAULT_INPUT_DEPTH = 20
DEFAULT_DENSE_BATCH_SIZE = 32


class HybridCandidateBaselineError(RuntimeError):
    """Raised when the hybrid candidate baseline cannot be completed."""


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
        raise HybridCandidateBaselineError(
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
        raise HybridCandidateBaselineError(
            f"Could not read {record_name} file {path}: {error}"
        ) from error

    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise HybridCandidateBaselineError(
            f"{path}: invalid JSON: {error.msg}"
        ) from error

    if not isinstance(value, dict):
        raise HybridCandidateBaselineError(
            f"{path}: {record_name} must be a JSON object."
        )

    return value


def _require_positive_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise HybridCandidateBaselineError(
            f"{field_name} must be a positive integer."
        )

    return value


def run_hybrid_candidate_baseline(
    *,
    manifest_path: Path,
    passages_path: Path,
    benchmark_path: Path,
    model_path: Path,
    output_path: Path,
    result_version: str,
    input_depth: int = DEFAULT_INPUT_DEPTH,
    dense_batch_size: int = DEFAULT_DENSE_BATCH_SIZE,
) -> dict[str, Any]:
    """Validate inputs, evaluate a candidate union, and publish one result."""

    for field_name, value in (
        ("manifest_path", manifest_path),
        ("passages_path", passages_path),
        ("benchmark_path", benchmark_path),
        ("model_path", model_path),
        ("output_path", output_path),
    ):
        if not isinstance(value, Path):
            raise HybridCandidateBaselineError(
                f"{field_name} must be a pathlib.Path."
            )

    validated_input_depth = _require_positive_integer(
        input_depth,
        field_name="input_depth",
    )
    validated_dense_batch_size = _require_positive_integer(
        dense_batch_size,
        field_name="dense_batch_size",
    )

    if not isinstance(result_version, str) or not result_version.strip():
        raise HybridCandidateBaselineError(
            "result_version must be a non-empty string."
        )

    if output_path.exists():
        raise HybridCandidateBaselineError(
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
        raise HybridCandidateBaselineError(str(error)) from error

    if validated_input_depth > len(passages):
        raise HybridCandidateBaselineError(
            "input_depth cannot exceed the corpus passage count."
        )

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

        evaluation = evaluate_hybrid_candidates(
            passages,
            benchmark,
            session=session,
            input_depth=validated_input_depth,
            dense_batch_size=validated_dense_batch_size,
        )

        artifact = build_hybrid_candidate_result_artifact(
            evaluation,
            dataset=benchmark,
            result_version=result_version,
            benchmark_sha256=benchmark_sha256,
            passage_sha256=passage_sha256,
        )

        write_hybrid_candidate_result_artifact(
            artifact,
            output_path,
        )
    except (
        RetrievalEvaluationError,
        DenseModelError,
        HybridCandidateEvaluationError,
        HybridCandidateResultError,
    ) as error:
        raise HybridCandidateBaselineError(str(error)) from error

    if not isinstance(artifact, Mapping):
        raise HybridCandidateBaselineError(
            "Hybrid candidate result artifact must be a mapping."
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
    """Parse explicit paths for one reproducible candidate-union run."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and publish the deterministic PolicyProof "
            "BM25+dense candidate-union baseline."
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
        "--input-depth",
        type=_positive_integer,
        default=DEFAULT_INPUT_DEPTH,
    )
    parser.add_argument(
        "--dense-batch-size",
        type=_positive_integer,
        default=DEFAULT_DENSE_BATCH_SIZE,
    )

    return parser.parse_args()


def main() -> None:
    """Run one validated hybrid candidate-union baseline."""

    args = _parse_args()
    artifact = run_hybrid_candidate_baseline(
        manifest_path=args.manifest,
        passages_path=args.passages,
        benchmark_path=args.benchmark,
        model_path=args.model,
        output_path=args.output,
        result_version=args.result_version,
        input_depth=args.input_depth,
        dense_batch_size=args.dense_batch_size,
    )

    evaluation = artifact["evaluation"]
    metrics = evaluation["aggregate_metrics"]
    passage_count = artifact["bindings"]["passages"]["count"]
    input_depth = artifact["candidate_generation"][
        "input_depth_per_retriever"
    ]

    print(
        "Hybrid candidate baseline complete: "
        f"{passage_count} passages, "
        f"{evaluation['answer_query_count']} answer queries, "
        f"input depth={input_depth}, "
        f"mean candidate recall="
        f"{metrics['mean_candidate_recall']:.10f}, "
        f"direct-evidence hit rate="
        f"{metrics['direct_evidence_hit_rate']:.10f}, "
        f"mean candidate count="
        f"{metrics['mean_candidate_count']:.10f}"
    )


if __name__ == "__main__":
    main()
