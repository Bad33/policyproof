"""Validated orchestration for the deterministic BM25 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyproof.bm25 import BM25Error, BM25Parameters
from policyproof.bm25_evaluation import (
    BM25EvaluationError,
    evaluate_bm25,
)
from policyproof.bm25_results import (
    BM25ResultError,
    build_bm25_result_artifact,
    write_bm25_result_artifact,
)
from policyproof.retrieval_evaluation import (
    RetrievalEvaluationError,
    validate_retrieval_evaluation_dataset,
)
from policyproof.retrieval_units import (
    RetrievalUnitError,
    load_jsonl,
)


class BM25BaselineError(RuntimeError):
    """Raised when the validated BM25 baseline cannot be completed."""


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading the full file into memory."""

    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    except OSError as error:
        raise BM25BaselineError(
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
        raise BM25BaselineError(
            f"Could not read {record_name} file {path}: {error}"
        ) from error

    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise BM25BaselineError(
            f"{path}: invalid JSON: {error.msg}"
        ) from error

    if not isinstance(value, dict):
        raise BM25BaselineError(
            f"{path}: {record_name} must be a JSON object."
        )

    return value


def run_bm25_baseline(
    *,
    manifest_path: Path,
    passages_path: Path,
    benchmark_path: Path,
    output_path: Path,
    result_version: str,
    parameters: BM25Parameters | None = None,
) -> dict[str, Any]:
    """Validate inputs, evaluate corpus-wide BM25, and publish one result."""

    for field_name, value in (
        ("manifest_path", manifest_path),
        ("passages_path", passages_path),
        ("benchmark_path", benchmark_path),
        ("output_path", output_path),
    ):
        if not isinstance(value, Path):
            raise BM25BaselineError(
                f"{field_name} must be a pathlib.Path."
            )

    if output_path.exists():
        raise BM25BaselineError(
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
        raise BM25BaselineError(str(error)) from error

    benchmark_sha256 = sha256_file(benchmark_path)
    passage_sha256 = sha256_file(passages_path)

    try:
        validate_retrieval_evaluation_dataset(
            benchmark,
            manifest=manifest,
            passage_records=passages,
            passage_artifact_sha256=passage_sha256,
        )

        evaluation = evaluate_bm25(
            passages,
            benchmark,
            parameters=parameters,
        )

        artifact = build_bm25_result_artifact(
            evaluation,
            dataset=benchmark,
            result_version=result_version,
            benchmark_sha256=benchmark_sha256,
            passage_sha256=passage_sha256,
        )

        write_bm25_result_artifact(
            artifact,
            output_path,
        )
    except (
        RetrievalEvaluationError,
        BM25Error,
        BM25EvaluationError,
        BM25ResultError,
    ) as error:
        raise BM25BaselineError(str(error)) from error

    if not isinstance(artifact, Mapping):
        raise BM25BaselineError(
            "BM25 result artifact must be a mapping."
        )

    return dict(artifact)



def _parse_args() -> argparse.Namespace:
    """Parse explicit paths for one reproducible BM25 baseline run."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and publish the deterministic "
            "PolicyProof BM25 baseline."
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
    """Run one validated BM25 baseline from explicit inputs."""

    args = _parse_args()
    artifact = run_bm25_baseline(
        manifest_path=args.manifest,
        passages_path=args.passages,
        benchmark_path=args.benchmark,
        output_path=args.output,
        result_version=args.result_version,
    )

    evaluation = artifact["evaluation"]
    metrics = evaluation["aggregate_metrics"]
    passage_count = artifact["bindings"]["passages"]["count"]

    print(
        "BM25 baseline complete: "
        f"{passage_count} passages, "
        f"{evaluation['answer_query_count']} answer queries, "
        f"mean Recall@10={metrics['mean_recall_at_10']:.10f}, "
        f"MRR@10={metrics['mrr_at_10']:.10f}, "
        f"mean nDCG@10={metrics['mean_ndcg_at_10']:.10f}"
    )


if __name__ == "__main__":
    main()
