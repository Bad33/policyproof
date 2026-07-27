from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT_GUIDE = ROOT / "docs/deployment.md"
README = ROOT / "README.md"
ENGINEERING_DECISIONS = ROOT / "docs/engineering-decisions.md"


def test_dockerfile_uses_render_port_contract() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "EXPOSE 10000" in text
    assert "--host 0.0.0.0" in text
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
        "tests-890%20passing-brightgreen",
        "- 890 passing tests",
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
