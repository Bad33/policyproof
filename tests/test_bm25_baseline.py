from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from policyproof.bm25_baseline import (
    BM25BaselineError,
    run_bm25_baseline,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def test_runner_validates_inputs_and_writes_bound_result(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "bm25-baseline-v0.1.0.json"

    artifact = run_bm25_baseline(
        manifest_path=MANIFEST_PATH,
        passages_path=PASSAGES_PATH,
        benchmark_path=BENCHMARK_PATH,
        output_path=output_path,
        result_version="0.1.0",
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
    assert artifact["retriever"]["parameters"] == {
        "k1": 1.2,
        "b": 0.75,
    }

    metrics = artifact["evaluation"]["aggregate_metrics"]

    assert metrics["mean_recall_at_1"] == pytest.approx(
        0.3125,
        abs=1e-12,
    )
    assert metrics["mean_recall_at_3"] == pytest.approx(
        0.6041666666666666,
        abs=1e-12,
    )
    assert metrics["mean_recall_at_5"] == pytest.approx(
        0.6822916666666666,
        abs=1e-12,
    )
    assert metrics["mean_recall_at_10"] == pytest.approx(
        0.7760416666666666,
        abs=1e-12,
    )
    assert metrics["mrr_at_10"] == pytest.approx(
        0.7433035714285714,
        abs=1e-12,
    )
    assert metrics["direct_evidence_hit_rate_at_10"] == pytest.approx(
        0.9375,
        abs=1e-12,
    )
    assert metrics["mean_ndcg_at_10"] == pytest.approx(
        0.6555156356384406,
        abs=1e-12,
    )

    assert sha256_file(BENCHMARK_PATH) == BENCHMARK_SHA256
    assert sha256_file(PASSAGES_PATH) == PASSAGE_SHA256


def test_runner_refuses_to_overwrite_result(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "bm25-baseline-v0.1.0.json"
    output_path.write_text(
        '{"existing": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        BM25BaselineError,
        match="already exists",
    ):
        run_bm25_baseline(
            manifest_path=MANIFEST_PATH,
            passages_path=PASSAGES_PATH,
            benchmark_path=BENCHMARK_PATH,
            output_path=output_path,
            result_version="0.1.0",
        )

    assert output_path.read_text(encoding="utf-8") == (
        '{"existing": true}\n'
    )



def test_cli_runs_validated_bm25_baseline(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "bm25-baseline-v0.1.0.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "policyproof.bm25_baseline",
            "--manifest",
            str(MANIFEST_PATH),
            "--passages",
            str(PASSAGES_PATH),
            "--benchmark",
            str(BENCHMARK_PATH),
            "--output",
            str(output_path),
            "--result-version",
            "0.1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert completed.stderr == ""
    assert completed.stdout == (
        "BM25 baseline complete: "
        "707 passages, 16 answer queries, "
        "mean Recall@10=0.7760416667, "
        "MRR@10=0.7433035714, "
        "mean nDCG@10=0.6555156356\n"
    )
