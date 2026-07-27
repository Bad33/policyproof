import gzip
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT_GUIDE = ROOT / "docs/deployment.md"
README = ROOT / "README.md"
ENGINEERING_DECISIONS = ROOT / "docs/engineering-decisions.md"
DEPLOYMENT_PASSAGE_ARCHIVE = (
    ROOT / "data/deployment/retrieval-passages-v1.1.jsonl.gz"
)
EXPECTED_PASSAGE_SHA256 = (
    "5ca1db8d2dd56b92d378bdf315bad25e"
    "f83029b4d18017b3755f287bbc26bf96"
)


def test_dockerfile_uses_render_port_contract() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "EXPOSE 10000" in text
    assert "--host 0.0.0.0" in text
    assert (
        "python -m policyproof.demo --root /app serve"
        in text
    )
    assert "${PORT:-10000}" in text
    assert 'CMD ["sh", "-c",' in text
    assert "--port 7860" not in text


def test_deployment_guide_targets_render() -> None:
    text = DEPLOYMENT_GUIDE.read_text(encoding="utf-8")

    required = (
        "Render Web Service",
        "GitHub repository",
        "Docker",
        "Free",
        "PORT",
        "10000",
        "/api/health",
        "onrender.com",
        "15 minutes",
    )

    for phrase in required:
        assert phrase in text

    assert "Hugging Face" not in text


def test_readme_targets_render_demo() -> None:
    text = README.read_text(encoding="utf-8")

    required = (
        "Render",
        "docs/deployment.md",
        "verified Render URL",
        "tests-891%20passing-brightgreen",
        "- 891 passing tests",
    )

    for phrase in required:
        assert phrase in text

    assert "Hugging Face" not in text

def test_engineering_decision_records_render_deployment() -> None:
    text = ENGINEERING_DECISIONS.read_text(encoding="utf-8")

    required = (
        "## PP-042: Deploy the public demo on Render",
        "Render Web Service",
        "PORT",
        "10000",
        "Free",
        "890",
        "Hugging Face",
    )

    for phrase in required:
        assert phrase in text


def test_render_archive_restores_accepted_passage_artifact() -> None:
    assert DEPLOYMENT_PASSAGE_ARCHIVE.is_file()

    restored = gzip.decompress(
        DEPLOYMENT_PASSAGE_ARCHIVE.read_bytes()
    )

    assert hashlib.sha256(restored).hexdigest() == EXPECTED_PASSAGE_SHA256
    assert len(restored.splitlines()) == 707

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "COPY data/deployment/retrieval-passages-v1.1.jsonl.gz "
        "./data/processed/retrieval-passages.jsonl.gz"
    ) in dockerfile
    assert (
        "python -m gzip -d "
        "data/processed/retrieval-passages.jsonl.gz"
    ) in dockerfile
    assert (
        "COPY data/processed/retrieval-passages.jsonl"
        not in dockerfile
    )

    deployment = DEPLOYMENT_GUIDE.read_text(encoding="utf-8")
    decisions = ENGINEERING_DECISIONS.read_text(encoding="utf-8")

    assert (
        "data/deployment/retrieval-passages-v1.1.jsonl.gz"
        in deployment
    )
    assert EXPECTED_PASSAGE_SHA256 in deployment
    assert (
        "cfb26d3393089f8ea29b547961d322b7"
        "3a4a1170d3fd7c9c1999bdcde417d8ee"
        in deployment
    )

    assert (
        "## PP-043: Package the accepted passage corpus for deployment"
        in decisions
    )
    assert EXPECTED_PASSAGE_SHA256 in decisions
    assert "891" in decisions

    assert "python -m policyproof.demo --root /app serve" in deployment
    assert "/usr/local/lib/python3.12" in decisions
    assert "--root /app" in decisions
