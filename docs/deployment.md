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
