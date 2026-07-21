# PolicyProof

PolicyProof is a research-driven RAG and citation-verification system for public
AI-governance and regulatory documents.

The system is designed to retrieve supporting evidence, produce claim-level
citations, identify insufficient evidence, abstain from unsupported questions,
and evaluate each pipeline component.

## Status

Phase 1 ingestion and retrieval-data preparation is complete. The initial
retrieval-evaluation contract and reviewed benchmark are also complete.

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
- Strict retrieval-evaluation schema version `1.0`
- Reviewed benchmark version `0.1.1` with 20 queries and 36 graded judgments
- Published benchmark version `0.1.0` retained byte-for-byte for reproducibility
- 16 answerable questions balanced across all four documents
- 4 explicit abstention cases for unsupported or out-of-scope requests
- 174 passing tests

Current passage artifacts use schema version `1.1`. They contain retrieval and
citation text with complete source provenance, but no embeddings or vector-index
records.

The retrieval benchmark uses schema version `1.0`. Dataset version `0.1.1` is
the current benchmark and contains manually reviewed answerable and abstention
cases bound to the accepted passage artifact. Published version `0.1.0` remains
available unchanged so earlier measurements remain reproducible.

Next: implement and measure a deterministic BM25 baseline against the fixed
retrieval benchmark before adding dense-vector, hybrid retrieval, or reranking.

## Responsible-use notice

PolicyProof is a research and compliance-support application. It does not provide
legal advice, determine legal compliance, or replace review by qualified
professionals.
