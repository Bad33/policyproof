from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import pytest

import policyproof.reranker_baseline as baseline_module
from policyproof.reranker import RerankedCandidate
from policyproof.reranker_baseline import (
    RerankerBaselineError,
    main,
    run_reranker_baseline,
)
from policyproof.reranker_evaluation import (
    RerankerEvaluationError,
    RerankerEvaluationResult,
    RerankerQueryResult,
)
from policyproof.reranker_model import RerankerModelError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
PASSAGES_PATH = ROOT / "data/processed/retrieval-passages.jsonl"
BENCHMARK_PATH = ROOT / "data/evaluation/retrieval-evaluation-v0.1.1.json"
CANDIDATE_PATH = (
    ROOT / "data/results/hybrid-candidate-baseline-v0.1.0.json"
)

BENCHMARK_SHA256 = (
    "42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c"
)
PASSAGE_SHA256 = (
    "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"
)
CANDIDATE_SHA256 = (
    "94b98eda3795280ef31aa0dfaa49a44d912c23d77e50a75f33a4f2f26e1fe0d4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_candidate_artifact() -> dict[str, Any]:
    value = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError("Candidate fixture must be a JSON object.")

    return value


def sample_evaluation() -> RerankerEvaluationResult:
    artifact = load_candidate_artifact()
    raw_results = artifact["evaluation"]["query_results"]
    query_results: list[RerankerQueryResult] = []

    for raw_result in raw_results:
        raw_candidates = raw_result["candidates"]
        candidate_count = len(raw_candidates)
        ranked_candidates = tuple(
            RerankedCandidate(
                passage_id=candidate["passage_id"],
                accepted_order=candidate["accepted_order"],
                bm25_rank=candidate["bm25_rank"],
                dense_rank=candidate["dense_rank"],
                reranker_score=float(candidate_count - position),
                reranker_rank=position + 1,
            )
            for position, candidate in enumerate(raw_candidates)
        )

        query_results.append(
            RerankerQueryResult(
                query_id=raw_result["query_id"],
                ranked_candidates=ranked_candidates,
                ranked_passage_ids=tuple(
                    candidate.passage_id
                    for candidate in ranked_candidates
                ),
                candidate_count=candidate_count,
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                reciprocal_rank_at_10=0.0,
                direct_evidence_hit_at_10=False,
                ndcg_at_10=0.0,
            )
        )

    frozen_results = tuple(query_results)

    return RerankerEvaluationResult(
        corpus_passage_count=707,
        answer_query_count=len(frozen_results),
        abstention_query_count=4,
        query_results=frozen_results,
        mean_candidate_count=fmean(
            result.candidate_count
            for result in frozen_results
        ),
        mean_recall_at_1=0.0,
        mean_recall_at_3=0.0,
        mean_recall_at_5=0.0,
        mean_recall_at_10=0.0,
        mrr_at_10=0.0,
        direct_evidence_hit_rate_at_10=0.0,
        mean_ndcg_at_10=0.0,
    )


def test_runner_validates_inputs_and_writes_bound_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"controlled model fixture")
    output_path = tmp_path / "reranker-baseline-v0.1.0.json"
    session = object()
    captured: dict[str, Any] = {}

    def fake_create_reranker_session(path: Path) -> object:
        captured["model_path"] = path
        return session

    def fake_evaluate_reranker(
        passages,
        dataset,
        candidate_artifact,
        *,
        session: object,
    ) -> RerankerEvaluationResult:
        captured["passages"] = passages
        captured["dataset"] = dataset
        captured["candidate_artifact"] = candidate_artifact
        captured["session"] = session
        return sample_evaluation()

    monkeypatch.setattr(
        baseline_module,
        "create_reranker_session",
        fake_create_reranker_session,
    )
    monkeypatch.setattr(
        baseline_module,
        "evaluate_reranker",
        fake_evaluate_reranker,
    )

    artifact = run_reranker_baseline(
        manifest_path=MANIFEST_PATH,
        passages_path=PASSAGES_PATH,
        benchmark_path=BENCHMARK_PATH,
        candidate_path=CANDIDATE_PATH,
        model_path=model_path,
        output_path=output_path,
        result_version="0.1.0",
    )

    persisted = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert persisted == artifact
    assert artifact["result_id"] == "policyproof-reranker-baseline"
    assert artifact["result_version"] == "0.1.0"
    assert artifact["bindings"]["benchmark"]["sha256"] == (
        BENCHMARK_SHA256
    )
    assert artifact["bindings"]["passages"]["sha256"] == (
        PASSAGE_SHA256
    )
    assert artifact["bindings"]["hybrid_candidates"]["sha256"] == (
        CANDIDATE_SHA256
    )
    assert artifact["bindings"]["passages"]["count"] == 707
    assert artifact["reranker"]["candidate_scope"] == (
        "hybrid_candidate_union"
    )
    assert artifact["reranker"]["final_ranking"] is True

    assert captured["model_path"] == model_path
    assert captured["session"] is session
    assert len(captured["passages"]) == 707
    assert captured["dataset"]["dataset_version"] == "0.1.1"
    assert captured["candidate_artifact"]["result_id"] == (
        "policyproof-hybrid-candidate-baseline"
    )

    assert sha256_file(BENCHMARK_PATH) == BENCHMARK_SHA256
    assert sha256_file(PASSAGES_PATH) == PASSAGE_SHA256
    assert sha256_file(CANDIDATE_PATH) == CANDIDATE_SHA256


def test_runner_refuses_to_overwrite_result(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "reranker-baseline-v0.1.0.json"
    output_path.write_text(
        '{"existing": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        RerankerBaselineError,
        match="already exists",
    ):
        run_reranker_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            candidate_path=CANDIDATE_PATH,
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
        "candidate_path",
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
        "candidate_path": CANDIDATE_PATH,
        "model_path": tmp_path / "model.onnx",
        "output_path": tmp_path / "result.json",
        "result_version": "0.1.0",
    }
    arguments[field_name] = str(arguments[field_name])

    with pytest.raises(
        RerankerBaselineError,
        match=field_name,
    ):
        run_reranker_baseline(**arguments)


def test_runner_wraps_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"invalid model")

    def fail_model(path: Path) -> object:
        raise RerankerModelError("model contract failed")

    monkeypatch.setattr(
        baseline_module,
        "create_reranker_session",
        fail_model,
    )

    with pytest.raises(
        RerankerBaselineError,
        match="model contract failed",
    ):
        run_reranker_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            candidate_path=CANDIDATE_PATH,
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
        "create_reranker_session",
        lambda path: object(),
    )

    def fail_evaluation(*args, **kwargs):
        raise RerankerEvaluationError(
            "reranker evaluation failed"
        )

    monkeypatch.setattr(
        baseline_module,
        "evaluate_reranker",
        fail_evaluation,
    )

    with pytest.raises(
        RerankerBaselineError,
        match="reranker evaluation failed",
    ):
        run_reranker_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            candidate_path=CANDIDATE_PATH,
            model_path=model_path,
            output_path=tmp_path / "result.json",
            result_version="0.1.0",
        )


def test_cli_parses_explicit_inputs_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "result.json"
    model_path = tmp_path / "model.onnx"
    captured: dict[str, Any] = {}

    def fake_run_reranker_baseline(**kwargs) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "bindings": {
                "passages": {
                    "count": 707,
                }
            },
            "evaluation": {
                "answer_query_count": 16,
                "aggregate_metrics": {
                    "mean_recall_at_10": 0.875,
                    "mrr_at_10": 0.75,
                    "mean_ndcg_at_10": 0.8,
                },
            },
        }

    monkeypatch.setattr(
        baseline_module,
        "run_reranker_baseline",
        fake_run_reranker_baseline,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "policyproof.reranker_baseline",
            "--manifest",
            str(MANIFEST_PATH),
            "--passages",
            str(PASSAGES_PATH),
            "--benchmark",
            str(BENCHMARK_PATH),
            "--candidates",
            str(CANDIDATE_PATH),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--result-version",
            "0.1.0",
        ],
    )

    main()

    assert captured == {
        "manifest_path": MANIFEST_PATH,
        "passages_path": PASSAGES_PATH,
        "benchmark_path": BENCHMARK_PATH,
        "candidate_path": CANDIDATE_PATH,
        "model_path": model_path,
        "output_path": output_path,
        "result_version": "0.1.0",
    }
    assert capsys.readouterr().out == (
        "Reranker baseline complete: "
        "707 passages, 16 answer queries, "
        "mean Recall@10=0.8750000000, "
        "MRR@10=0.7500000000, "
        "mean nDCG@10=0.8000000000\n"
    )
