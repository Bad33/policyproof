# PolicyProof

PolicyProof is a research-driven RAG and citation-verification system for public
AI-governance and regulatory documents.

The system is designed to retrieve supporting evidence, produce claim-level
citations, identify insufficient evidence, abstain from unsupported questions,
and evaluate each pipeline component.

## Status

Phase 1: Controlled corpus ingestion.

Completed:

- Approved four-document source manifest
- Secure PDF downloading and SHA-256 snapshots
- Page-level extraction with document-specific extractor selection
- Automated manifest, download, and extraction tests
- Document-specific heading detection and reviewed reconstruction
- Reviewed NIST RMF heading normalization with immutable raw provenance
- Document-specific hierarchy and coordinate-only heading spans
- Tested production builder for 579 coordinate-only retrieval units
- Complete 12,008-record coordinate ledger with explicit ownership and exclusions
- Exact semantic parity with the independently audited retrieval design

Next: define and test retrieval-text materialization, tokenizer, and citation-unit requirements before creating embeddings or indexes.

## Responsible-use notice

PolicyProof is a research and compliance-support application. It does not provide
legal advice, determine legal compliance, or replace review by qualified
professionals.
