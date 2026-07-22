from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import policyproof.hybrid_candidate_baseline as baseline_module
from policyproof.dense_model import DenseModelError
from policyproof.hybrid_candidate_baseline import (
    HybridCandidateBaselineError,
    run_hybrid_candidate_baseline,
)
from policyproof.hybrid_candidate_evaluation import (
    HybridCandidateEvaluationError,
    HybridCandidateEvaluationResult,
)
from policyproof.hybrid_candidate_results import (
    HybridCandidateResultError,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
PASSAGES_PATH = ROOT / "data/processed/retrieval-passages.jsonl"
BENCHMARK_PATH = (
    ROOT / "data/evaluation/retrieval-evaluation-v0.1.1.json"
)

BENCHMARK_SHA256 = (
    "42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c"
)
PASSAGE_SHA256 = (
    "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"
)


def sample_evaluation() -> HybridCandidateEvaluationResult:
    return HybridCandidateEvaluationResult(
        corpus_passage_count=707,
        answer_query_count=16,
        abstention_query_count=4,
        input_depth=20,
        dense_batch_size=32,
        query_results=(),
        mean_candidate_recall=1.0,
        direct_evidence_hit_rate=1.0,
        mean_candidate_count=31.3125,
    )


def sample_artifact() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "result_id": "policyproof-hybrid-candidate-baseline",
        "result_version": "0.1.0",
        "bindings": {
            "benchmark": {
                "sha256": BENCHMARK_SHA256,
            },
            "passages": {
                "sha256": PASSAGE_SHA256,
                "count": 707,
            },
        },
        "candidate_generation": {
            "strategy": "deduplicated_union",
            "final_ranking": False,
            "input_depth_per_retriever": 20,
        },
        "evaluation": {
            "answer_query_count": 16,
            "abstention_query_count": 4,
            "aggregate_metrics": {
                "mean_candidate_recall": 1.0,
                "direct_evidence_hit_rate": 1.0,
                "mean_candidate_count": 31.3125,
            },
            "query_results": [],
        },
    }


def install_successful_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    session = object()
    artifact = sample_artifact()

    def fake_create_dense_session(model_path: Path) -> object:
        captured["model_path"] = model_path
        return session

    def fake_evaluate_hybrid_candidates(
        passages,
        dataset,
        *,
        session: object,
        input_depth: int,
        dense_batch_size: int,
    ) -> HybridCandidateEvaluationResult:
        captured["passages"] = passages
        captured["dataset"] = dataset
        captured["session"] = session
        captured["input_depth"] = input_depth
        captured["dense_batch_size"] = dense_batch_size
        return sample_evaluation()

    def fake_build_result(
        evaluation,
        *,
        dataset,
        result_version: str,
        benchmark_sha256: str,
        passage_sha256: str,
    ) -> dict[str, Any]:
        captured["evaluation"] = evaluation
        captured["artifact_dataset"] = dataset
        captured["result_version"] = result_version
        captured["benchmark_sha256"] = benchmark_sha256
        captured["passage_sha256"] = passage_sha256
        return artifact

    def fake_write_result(
        value,
        output_path: Path,
    ) -> None:
        captured["written_artifact"] = value
        captured["output_path"] = output_path
        output_path.write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        fake_create_dense_session,
    )
    monkeypatch.setattr(
        baseline_module,
        "evaluate_hybrid_candidates",
        fake_evaluate_hybrid_candidates,
    )
    monkeypatch.setattr(
        baseline_module,
        "build_hybrid_candidate_result_artifact",
        fake_build_result,
    )
    monkeypatch.setattr(
        baseline_module,
        "write_hybrid_candidate_result_artifact",
        fake_write_result,
    )

    captured["session_object"] = session
    captured["expected_artifact"] = artifact
    return captured


def test_runner_validates_inputs_and_writes_bound_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_successful_pipeline(monkeypatch)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"controlled model fixture")
    output_path = (
        tmp_path / "hybrid-candidate-baseline-v0.1.0.json"
    )

    artifact = run_hybrid_candidate_baseline(
        manifest_path=MANIFEST_PATH,
        passages_path=PASSAGES_PATH,
        benchmark_path=BENCHMARK_PATH,
        model_path=model_path,
        output_path=output_path,
        result_version="0.1.0",
        input_depth=20,
        dense_batch_size=32,
    )

    assert artifact == captured["expected_artifact"]
    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == artifact

    assert captured["model_path"] == model_path
    assert captured["session"] is captured["session_object"]
    assert captured["input_depth"] == 20
    assert captured["dense_batch_size"] == 32
    assert len(captured["passages"]) == 707
    assert captured["dataset"]["dataset_version"] == "0.1.1"
    assert captured["artifact_dataset"] is captured["dataset"]
    assert captured["result_version"] == "0.1.0"
    assert captured["benchmark_sha256"] == BENCHMARK_SHA256
    assert captured["passage_sha256"] == PASSAGE_SHA256
    assert captured["written_artifact"] == artifact
    assert captured["output_path"] == output_path


def test_runner_refuses_to_overwrite_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = (
        tmp_path / "hybrid-candidate-baseline-v0.1.0.json"
    )
    output_path.write_text(
        '{"existing": true}\n',
        encoding="utf-8",
    )

    def unexpected_model_load(path: Path) -> object:
        raise AssertionError(
            f"Model should not load for existing output: {path}"
        )

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        unexpected_model_load,
    )

    with pytest.raises(
        HybridCandidateBaselineError,
        match="Output already exists",
    ):
        run_hybrid_candidate_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=tmp_path / "model.onnx",
            output_path=output_path,
            result_version="0.1.0",
        )

    assert output_path.read_text(encoding="utf-8") == (
        '{"existing": true}\n'
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "manifest_path",
        "passages_path",
        "benchmark_path",
        "model_path",
        "output_path",
    ],
)
def test_runner_requires_path_objects(
    tmp_path: Path,
    field_name: str,
) -> None:
    arguments: dict[str, Any] = {
        "manifest_path": MANIFEST_PATH,
        "passages_path": PASSAGES_PATH,
        "benchmark_path": BENCHMARK_PATH,
        "model_path": tmp_path / "model.onnx",
        "output_path": tmp_path / "result.json",
        "result_version": "0.1.0",
    }
    arguments[field_name] = str(arguments[field_name])

    with pytest.raises(
        HybridCandidateBaselineError,
        match=field_name,
    ):
        run_hybrid_candidate_baseline(**arguments)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("input_depth", 0),
        ("input_depth", True),
        ("dense_batch_size", 0),
        ("dense_batch_size", True),
    ],
)
def test_runner_rejects_invalid_positive_integer_options(
    tmp_path: Path,
    field_name: str,
    value,
) -> None:
    arguments: dict[str, Any] = {
        "manifest_path": MANIFEST_PATH,
        "passages_path": PASSAGES_PATH,
        "benchmark_path": BENCHMARK_PATH,
        "model_path": tmp_path / "model.onnx",
        "output_path": tmp_path / "result.json",
        "result_version": "0.1.0",
        "input_depth": 20,
        "dense_batch_size": 32,
    }
    arguments[field_name] = value

    with pytest.raises(
        HybridCandidateBaselineError,
        match=field_name,
    ):
        run_hybrid_candidate_baseline(**arguments)


def test_runner_wraps_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_model(path: Path) -> object:
        raise DenseModelError("model contract failed")

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        fail_model,
    )

    with pytest.raises(
        HybridCandidateBaselineError,
        match="model contract failed",
    ):
        run_hybrid_candidate_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=tmp_path / "model.onnx",
            output_path=tmp_path / "result.json",
            result_version="0.1.0",
        )


def test_runner_wraps_evaluation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        lambda path: object(),
    )

    def fail_evaluation(*args, **kwargs):
        raise HybridCandidateEvaluationError(
            "hybrid evaluation failed"
        )

    monkeypatch.setattr(
        baseline_module,
        "evaluate_hybrid_candidates",
        fail_evaluation,
    )

    with pytest.raises(
        HybridCandidateBaselineError,
        match="hybrid evaluation failed",
    ):
        run_hybrid_candidate_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=tmp_path / "model.onnx",
            output_path=tmp_path / "result.json",
            result_version="0.1.0",
        )


def test_runner_wraps_result_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_successful_pipeline(monkeypatch)

    def fail_result(*args, **kwargs):
        raise HybridCandidateResultError(
            "hybrid result failed"
        )

    monkeypatch.setattr(
        baseline_module,
        "build_hybrid_candidate_result_artifact",
        fail_result,
    )

    with pytest.raises(
        HybridCandidateBaselineError,
        match="hybrid result failed",
    ):
        run_hybrid_candidate_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=tmp_path / "model.onnx",
            output_path=tmp_path / "result.json",
            result_version="0.1.0",
        )

    assert captured["output_path"] if "output_path" in captured else True
