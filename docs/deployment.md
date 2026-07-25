# Public Demo Deployment

PolicyProof is prepared for deployment as a Hugging Face Docker Space.

## Create the Space

1. Sign in to Hugging Face.
2. Create a new Space.
3. Select **Docker** as the Space SDK.
4. Use port `7860`.
5. Choose a public Space for portfolio visibility.
6. Clone the new Space repository locally.

## Publish PolicyProof

Copy the PolicyProof repository contents into the Space repository, preserving:

- `Dockerfile`
- `.dockerignore`
- `pyproject.toml`
- `src/`
- `data/processed/retrieval-passages.jsonl`
- `data/evaluation/evidence-sufficiency-silver-baseline-v0.1.0.json`

Commit and push to the Space. Hugging Face builds the container remotely; local
Docker Desktop is not required.

## Verify

After the build succeeds:

- open `/` and submit a policy question
- open `/api/health`
- verify the response reports `bm25_portable_demo`
- verify `label_provenance` is `construction_derived`
- confirm citations contain document, passage, label, excerpt, and BM25 score

Then replace the README's deployment placeholder with the final Space URL and
add a screenshot of the live application.

## Runtime command

The Docker image starts:

```bash
python -m policyproof.demo serve --host 0.0.0.0 --port 7860
```

## Limits

The hosted demo remains extractive and BM25-backed. It does not claim
human-adjudicated gold accuracy, legal correctness, or current-information
coverage.
