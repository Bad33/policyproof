# PolicyProof

PolicyProof is a research-driven RAG and citation-verification system for public
AI-governance and regulatory documents.

The system is designed to retrieve supporting evidence, produce claim-level
citations, identify insufficient evidence, abstain from unsupported questions,
and evaluate each pipeline component.

## Status

Phase 1 ingestion and retrieval-data preparation is complete.

Completed:

- Approved four-document AI-governance corpus
- Secure PDF downloading with SHA-256 verification
- Page extraction with document-specific extractor selection
- Reviewed heading detection, reconstruction, hierarchy, and source spans
- 581 coordinate-only retrieval units from 487 logical sources
- Complete 12,008-record coordinate ledger with explicit ownership and exclusions
- Pinned offline BERT WordPiece tokenizer contract
- 707 token-safe passages under a 445-token hard limit
- Persisted `retrieval_text` and source-derived `citation_text`
- Character-level provenance for the 14 reviewed intra-line passage boundaries
- Deterministic, non-overwriting artifact generation with regression tests
- 144 passing tests

Current passage artifacts use schema version `1.1`. They contain retrieval and
citation text with complete source provenance, but no embeddings or vector-index
records.

Next: define a versioned retrieval-evaluation contract before implementing
BM25, dense-vector, hybrid retrieval, and reranking.

## Responsible-use notice

PolicyProof is a research and compliance-support application. It does not provide
legal advice, determine legal compliance, or replace review by qualified
professionals.
