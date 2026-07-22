"""Run and publish the deterministic hybrid-candidate reranker baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyproof.reranker_evaluation import (
    RerankerEvaluationError,
    evaluate_reranker,
)
from policyproof.reranker_model import (
    RerankerModelError,
    create_reranker_session,
)
from policyproof.reranker_results import (
    RerankerResultError,
    build_reranker_result_artifact,
    write_reranker_result_artifact,
)
from policyproof.retrieval_evaluation import (
    RetrievalEvaluationError,
    validate_retrieval_evaluation_dataset,
)
from policyproof.retrieval_units import (
    RetrievalUnitError,
    load_jsonl,
)


class RerankerBaselineError(RuntimeError):
    """Raised when the validated reranker baseline cannot be completed."""


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
        raise RerankerBaselineError(
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
        raise RerankerBaselineError(
            f"Could not read {record_name} file {path}: {error}"
        ) from error

    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise RerankerBaselineError(
            f"{path}: invalid JSON: {error.msg}"
        ) from error

    if not isinstance(value, dict):
        raise RerankerBaselineError(
            f"{path}: {record_name} must be a JSON object."
        )

    return value


def run_reranker_baseline(
    *,
    manifest_path: Path,
    passages_path: Path,
    benchmark_path: Path,
    candidate_path: Path,
    model_path: Path,
    output_path: Path,
    result_version: str,
) -> dict[str, Any]:
    """Validate inputs, rerank accepted candidates, and publish one result."""

    for field_name, value in (
        ("manifest_path", manifest_path),
        ("passages_path", passages_path),
        ("benchmark_path", benchmark_path),
        ("candidate_path", candidate_path),
        ("model_path", model_path),
        ("output_path", output_path),
    ):
        if not isinstance(value, Path):
            raise RerankerBaselineError(
                f"{field_name} must be a pathlib.Path."
            )

    if output_path.exists():
        raise RerankerBaselineError(
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
    candidate_artifact = load_json_object(
        candidate_path,
        record_name="hybrid candidate result",
    )

    try:
        passages = load_jsonl(
            passages_path,
            record_name="retrieval passage",
        )
    except RetrievalUnitError as error:
        raise RerankerBaselineError(str(error)) from error

    benchmark_sha256 = sha256_file(benchmark_path)
    passage_sha256 = sha256_file(passages_path)
    candidate_sha256 = sha256_file(candidate_path)

    try:
        validate_retrieval_evaluation_dataset(
            benchmark,
            manifest=manifest,
            passage_records=passages,
            passage_artifact_sha256=passage_sha256,
        )

        session = create_reranker_session(model_path)

        evaluation = evaluate_reranker(
            passages,
            benchmark,
            candidate_artifact,
            session=session,
        )

        artifact = build_reranker_result_artifact(
            evaluation,
            dataset=benchmark,
            candidate_artifact=candidate_artifact,
            result_version=result_version,
            benchmark_sha256=benchmark_sha256,
            passage_sha256=passage_sha256,
            candidate_sha256=candidate_sha256,
        )

        write_reranker_result_artifact(
            artifact,
            output_path,
        )
    except (
        RetrievalEvaluationError,
        RerankerModelError,
        RerankerEvaluationError,
        RerankerResultError,
    ) as error:
        raise RerankerBaselineError(str(error)) from error

    if not isinstance(artifact, Mapping):
        raise RerankerBaselineError(
            "Reranker result artifact must be a mapping."
        )

    return dict(artifact)


def _parse_args() -> argparse.Namespace:
    """Parse explicit paths for one reproducible reranker baseline run."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and publish the deterministic "
            "PolicyProof reranker baseline."
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
        "--candidates",
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

    return parser.parse_args()


def main() -> None:
    """Run one validated reranker baseline from explicit inputs."""

    args = _parse_args()
    artifact = run_reranker_baseline(
        manifest_path=args.manifest,
        passages_path=args.passages,
        benchmark_path=args.benchmark,
        candidate_path=args.candidates,
        model_path=args.model,
        output_path=args.output,
        result_version=args.result_version,
    )

    evaluation = artifact["evaluation"]
    metrics = evaluation["aggregate_metrics"]
    passage_count = artifact["bindings"]["passages"]["count"]

    print(
        "Reranker baseline complete: "
        f"{passage_count} passages, "
        f"{evaluation['answer_query_count']} answer queries, "
        f"mean Recall@10={metrics['mean_recall_at_10']:.10f}, "
        f"MRR@10={metrics['mrr_at_10']:.10f}, "
        f"mean nDCG@10={metrics['mean_ndcg_at_10']:.10f}"
    )


if __name__ == "__main__":
    main()
