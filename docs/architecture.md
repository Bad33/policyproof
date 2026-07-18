# Architecture

Status: Phase 0 proposal.

PolicyProof will use a framework-independent Python pipeline for ingestion,
retrieval, reranking, evidence-sufficiency assessment, grounded generation,
claim extraction, citation verification, and structured tracing.

FastAPI will later expose the stable pipeline but will not contain core retrieval
or verification logic.
