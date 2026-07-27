# Public Demo Deployment

PolicyProof is prepared for deployment as a Render Web Service using the
repository's Docker configuration.

## Prerequisites

- a GitHub account containing the public PolicyProof repository
- a Render account connected to GitHub
- the repository's accepted `Dockerfile`
- no API keys or external model credentials

## Create the Render service

1. Sign in to Render.
2. Select **New** and then **Web Service**.
3. Connect the PolicyProof GitHub repository.
4. Select **Docker** as the runtime.
5. Choose the repository's `main` branch.
6. Select the **Free** instance type.
7. Use a descriptive service name such as `policyproof`.
8. Leave the Dockerfile path as `./Dockerfile`.
9. Create the Web Service.

Render builds the container directly from the connected GitHub repository.
Local Docker Desktop is not required.

## Port contract

Render supplies the runtime port through the `PORT` environment variable. The
Docker image starts PolicyProof on `0.0.0.0` and uses port `10000` when `PORT`
is not otherwise provided.

The effective runtime command is:

    python -m policyproof.demo serve --host 0.0.0.0 --port "${PORT:-10000}"

## Verify the deployment

After the deployment reports that it is live:

1. Open the generated `onrender.com` URL.
2. Submit a supported AI-governance policy question.
3. Open `/api/health`.
4. Verify the response reports `bm25_portable_demo`.
5. Verify `label_provenance` is `construction_derived`.
6. Confirm each citation includes:
   - document ID
   - passage ID
   - source label
   - source-derived excerpt
   - BM25 score
7. Submit an unsupported question and confirm that PolicyProof abstains.

Record the final verified Render URL only after these checks pass.

## Free-service behavior

A Free Render Web Service may spin down after approximately 15 minutes without
inbound traffic. The first request after inactivity may therefore take longer
while the service starts again.

The service is stateless and does not require persistent storage, so this
behavior is acceptable for the portfolio demo.

## Automatic deployments

By default, Render can redeploy when new commits reach the connected branch.
Keep the production service connected to `main` only after changes pass the
repository's full verification gates.

## Limits

The hosted demo remains extractive and BM25-backed. Its evidence-sufficiency
model uses construction-derived silver labels.

The deployment does not claim:

- independently human-adjudicated accuracy
- legal correctness
- legal advice
- current-information coverage
- comprehensive coverage beyond the frozen PolicyProof corpus

## Deployment corpus packaging

The accepted passage corpus remains generated locally under
`data/processed/retrieval-passages.jsonl` and remains ignored by Git.

For public Docker deployment, the repository includes the deterministic
transport archive:

`data/deployment/retrieval-passages-v1.1.jsonl.gz`

The Docker build decompresses this archive into the original accepted runtime
location:

`data/processed/retrieval-passages.jsonl`

Integrity bindings:

- restored passage SHA-256:
  `5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96`
- deterministic gzip archive SHA-256:
  `cfb26d3393089f8ea29b547961d322b73a4a1170d3fd7c9c1999bdcde417d8ee`
- restored passage records: `707`
- restored bytes: `2614040`

The application continues to reject the corpus if the restored SHA-256 differs
from the accepted passage binding. The archive is only a deployment transport
format and does not define a new corpus version.

## Container startup root

The Docker container starts the demo with:

`python -m policyproof.demo --root /app serve`

PolicyProof normally derives its repository root from the location of the
installed `policyproof.demo` module. Inside the production container, that
module is installed under `/usr/local/lib/python3.12/site-packages`, while the
deployment corpus and evaluation artifact are copied under `/app`.

Passing `--root /app` makes the existing repository-root contract explicit and
allows the demo to locate:

- `/app/data/processed/retrieval-passages.jsonl`
- `/app/data/evaluation/evidence-sufficiency-silver-baseline-v0.1.0.json`

This does not alter corpus selection or integrity validation.

## Verified public deployment

The accepted public deployment is:

https://policyproof-5uwv.onrender.com/

Live verification completed on 2026-07-26:

- `homepage: PASS (200)`
- `health: PASS (200)`
- `query: PASS (200)`
- query action: `answer`
- query reason: `evidence_passed_silver_sufficiency_threshold`
- returned citations: `2`
- ranking method: `bm25_portable_demo`
- label provenance: `construction_derived`

The health endpoint is:

https://policyproof-5uwv.onrender.com/api/health

The verified query used:

`Which characteristics does the NIST AI RMF associate with trustworthy AI?`

These checks confirm that the deployed service can load the accepted corpus,
validate its integrity bindings, retrieve evidence, evaluate silver-label
evidence sufficiency, and return citations.

## Post-selection-policy live verification

The public service was manually reverified on 2026-07-27 after deployment of
commit:

`e3416c8 fix: tighten demo evidence selection`

The updated homepage example asked:

`What risks does unauthorized voice generation create, and how does GPT-4o mitigate them?`

Observed live behavior:

- homepage loaded successfully
- action: `answer`
- reason: `evidence_passed_silver_sufficiency_threshold`
- silver sufficiency probability: `0.966787274217`
- threshold: `0.772084750192`
- returned citations: `2`
- both citations came from document:
  `openai-gpt-4o-system-card-2024-08-08`
- both citations came from section:
  `3.3.1 Unauthorized voice generation`

The first passage describes unauthorized voice-generation risks. The second
describes preset-voice restrictions, output classification, and blocking
behavior. No unrelated passage was displayed.

This confirms that the live service is using the conservative
same-logical-source evidence-selection policy accepted in PP-046. It does not
change the limitations of the construction-derived silver sufficiency model.
