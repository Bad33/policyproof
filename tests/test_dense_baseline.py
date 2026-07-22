from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import policyproof.dense_baseline as baseline_module
from policyproof.dense import DenseHit
from policyproof.dense_baseline import (
    DenseBaselineError,
    run_dense_baseline,
)
from policyproof.dense_evaluation import (
    DenseEvaluationError,
    DenseEvaluationResult,
    DenseQueryResult,
)
from policyproof.dense_model import DenseModelError
from policyproof.retrieval_units import load_jsonl

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sample_evaluation() -> DenseEvaluationResult:
    query_result = DenseQueryResult(
        query_id="rmf-001",
        ranked_hits=(
            DenseHit(
                passage_id="nist-ai-rmf-1.0:passage-0001",
                score=0.75,
                accepted_order=0,
            ),
        ),
        ranked_passage_ids=(
            "nist-ai-rmf-1.0:passage-0001",
        ),
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        reciprocal_rank_at_10=1.0,
        direct_evidence_hit_at_10=True,
        ndcg_at_10=1.0,
    )

    return DenseEvaluationResult(
        corpus_passage_count=707,
        answer_query_count=1,
        abstention_query_count=4,
        query_results=(query_result,),
        batch_size=16,
        mean_recall_at_1=1.0,
        mean_recall_at_3=1.0,
        mean_recall_at_5=1.0,
        mean_recall_at_10=1.0,
        mrr_at_10=1.0,
        direct_evidence_hit_rate_at_10=1.0,
        mean_ndcg_at_10=1.0,
    )


def test_runner_validates_inputs_and_writes_bound_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"controlled model fixture")
    output_path = tmp_path / "dense-baseline-v0.1.0.json"
    session = object()
    captured: dict[str, Any] = {}

    def fake_create_dense_session(path: Path) -> object:
        captured["model_path"] = path
        return session

    def fake_evaluate_dense(
        passages,
        dataset,
        *,
        session: object,
        batch_size: int,
    ) -> DenseEvaluationResult:
        captured["passages"] = passages
        captured["dataset"] = dataset
        captured["session"] = session
        captured["batch_size"] = batch_size
        return sample_evaluation()

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        fake_create_dense_session,
    )
    monkeypatch.setattr(
        baseline_module,
        "evaluate_dense",
        fake_evaluate_dense,
    )

    artifact = run_dense_baseline(
        manifest_path=MANIFEST_PATH,
        passages_path=PASSAGES_PATH,
        benchmark_path=BENCHMARK_PATH,
        model_path=model_path,
        output_path=output_path,
        result_version="0.1.0",
        batch_size=16,
    )

    persisted = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert persisted == artifact
    assert artifact["result_version"] == "0.1.0"
    assert artifact["bindings"]["benchmark"]["sha256"] == (
        BENCHMARK_SHA256
    )
    assert artifact["bindings"]["passages"]["sha256"] == (
        PASSAGE_SHA256
    )
    assert artifact["bindings"]["passages"]["count"] == 707
    assert artifact["retriever"]["candidate_scope"] == "all_passages"
    assert artifact["retriever"]["batch_size"] == 16

    assert captured["model_path"] == model_path
    assert captured["session"] is session
    assert captured["batch_size"] == 16
    assert len(captured["passages"]) == 707
    assert captured["dataset"]["dataset_version"] == "0.1.1"

    assert sha256_file(BENCHMARK_PATH) == BENCHMARK_SHA256
    assert sha256_file(PASSAGES_PATH) == PASSAGE_SHA256


def test_runner_passes_accepted_passages_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"controlled model fixture")
    expected_passages = load_jsonl(
        PASSAGES_PATH,
        record_name="retrieval passage",
    )
    captured_passages = None

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        lambda path: object(),
    )

    def fake_evaluate_dense(
        passages,
        dataset,
        *,
        session: object,
        batch_size: int,
    ) -> DenseEvaluationResult:
        nonlocal captured_passages
        del dataset, session, batch_size
        captured_passages = passages
        return sample_evaluation()

    monkeypatch.setattr(
        baseline_module,
        "evaluate_dense",
        fake_evaluate_dense,
    )

    run_dense_baseline(
        manifest_path=MANIFEST_PATH,
        passages_path=PASSAGES_PATH,
        benchmark_path=BENCHMARK_PATH,
        model_path=model_path,
        output_path=tmp_path / "result.json",
        result_version="0.1.0",
    )

    assert captured_passages == expected_passages


def test_runner_refuses_to_overwrite_result(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dense-baseline-v0.1.0.json"
    output_path.write_text(
        '{"existing": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DenseBaselineError,
        match="Output already exists",
    ):
        run_dense_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=tmp_path / "model.onnx",
            output_path=output_path,
            result_version="0.1.0",
        )

    assert output_path.read_text(
        encoding="utf-8"
    ) == '{"existing": true}\n'


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
        DenseBaselineError,
        match=field_name,
    ):
        run_dense_baseline(**arguments)


def test_runner_wraps_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"invalid model")

    def fail_model(path: Path) -> object:
        raise DenseModelError("model contract failed")

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        fail_model,
    )

    with pytest.raises(
        DenseBaselineError,
        match="model contract failed",
    ):
        run_dense_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=model_path,
            output_path=tmp_path / "result.json",
            result_version="0.1.0",
        )


def test_runner_wraps_evaluation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"controlled model fixture")

    monkeypatch.setattr(
        baseline_module,
        "create_dense_session",
        lambda path: object(),
    )

    def fail_evaluation(*args, **kwargs):
        raise DenseEvaluationError("dense evaluation failed")

    monkeypatch.setattr(
        baseline_module,
        "evaluate_dense",
        fail_evaluation,
    )

    with pytest.raises(
        DenseBaselineError,
        match="dense evaluation failed",
    ):
        run_dense_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            model_path=model_path,
            output_path=tmp_path / "result.json",
            result_version="0.1.0",
        )
