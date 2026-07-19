# Architecture

Status: Phase 1.4c implemented ingestion foundation.

PolicyProof will use a framework-independent Python pipeline for ingestion,
retrieval, reranking, evidence-sufficiency assessment, grounded generation,
claim extraction, citation verification, and structured tracing.

FastAPI will later expose the stable pipeline but will not contain core retrieval
or verification logic.

## Implemented pipeline through Phase 1.4c

The current deterministic ingestion foundation consists of these independently
callable stages:

1. Validate the version-controlled source manifest.
2. Download approved documents and record checksum snapshots.
3. Extract page-level text while retaining stable document and page identity.
4. Detect document-specific heading candidates.
5. Reconstruct complete headings with exact source-line provenance.
6. Assign document-specific parent-child hierarchy.
7. Assign coordinate-only direct-body, subtree, and synthetic descendant
   envelope spans.

Each stage fails closed on invalid identity, ordering, provenance, or overwrite
conditions.

## Heading-span boundary model

Every source hierarchy node has two distinct governed ranges:

- `direct_body` starts after the final source line used by the node's own
  reconstructed heading and stops immediately before the next source heading.
- `subtree` starts at the same position and stops immediately before the next
  source heading at the same or a shallower hierarchy depth, or at document end.

These ranges retain page and line coordinates. They do not store assembled body
text or citation chunks.

Synthetic hierarchy nodes do not receive source-heading provenance,
`direct_body`, or `subtree` ownership. They may receive a
`source_descendant_envelope`, which is explicitly identified as an aggregate of
their contiguous source descendants rather than as text printed under a
synthetic heading.

## Generated artifacts

Generated corpus artifacts under `data/processed/` remain local and are ignored
by Git. The repository stores the deterministic builders and tests needed to
reproduce them.

The Phase 1.4c local artifact is:

- `data/processed/heading-spans.jsonl`

It is derived from:

- `data/processed/pages.jsonl`
- `data/processed/reconstructed-headings.jsonl`
- `data/processed/heading-hierarchy.jsonl`

## Deferred pipeline stages

Phase 1.4c does not create retrieval or citation chunks. Chunk construction,
indexing, retrieval, reranking, evidence-sufficiency checks, generation, and
citation verification remain downstream stages.
