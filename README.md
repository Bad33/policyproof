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
- Independently audited 579-unit coordinate-only retrieval design

Next: promote the audited coordinate-only retrieval-unit design into a tested production builder before materializing retrieval text, selecting a tokenizer, creating citation units, embeddings, or indexes.

## Responsible-use notice

PolicyProof is a research and compliance-support application. It does not provide
legal advice, determine legal compliance, or replace review by qualified
professionals.
