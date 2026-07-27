FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/processed/retrieval-passages.jsonl ./data/processed/retrieval-passages.jsonl
COPY data/evaluation/evidence-sufficiency-silver-baseline-v0.1.0.json ./data/evaluation/evidence-sufficiency-silver-baseline-v0.1.0.json

RUN python -m pip install --upgrade pip &&     python -m pip install .

EXPOSE 10000

CMD ["sh", "-c", "python -m policyproof.demo serve --host 0.0.0.0 --port ${PORT:-10000}"]
