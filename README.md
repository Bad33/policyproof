# PolicyProof

PolicyProof is a research-driven RAG and citation-verification system for public
AI-governance and regulatory documents.

The system is designed to retrieve supporting evidence, produce claim-level
citations, identify insufficient evidence, abstain from unsupported questions,
and evaluate each pipeline component.

## Status

Phase 1 ingestion and retrieval-data preparation is complete. The retrieval-
evaluation foundation, deterministic BM25 and dense rankings, hybrid candidate
generation, and pinned cross-encoder baseline are also complete.

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
- Plain-Python corpus-wide BM25 implementation with deterministic tie-breaking
- Versioned BM25 result artifact with byte-identical regeneration tests
- Pinned BGE-small dense retriever through direct CPU ONNX inference
- Hash-verified external model asset with no automatic runtime download
- Versioned dense result artifact with byte-identical regeneration tests
- Explicit rejection of equal-weight reciprocal-rank fusion as a final ranking
- Deterministic top-20 BM25+dense candidate union for bounded reranking
- Versioned hybrid candidate-coverage artifact with immutable regression tests
- Pinned MiniLM cross-encoder through deterministic direct CPU ONNX inference
- Versioned reranker result with byte-identical regeneration and manual review
- Dense retrieval retained as the selected ranking after reranker comparison
- Offline regression coverage for corpus, benchmark, metrics, and result bindings
- 429 passing tests

Current passage artifacts use schema version `1.1`. They contain retrieval and
citation text with complete source provenance. Dense embeddings are calculated
during evaluation but are not yet persisted as embedding or vector-index artifacts.

The retrieval benchmark uses schema version `1.0`. Dataset version `0.1.1` is
the current benchmark and contains manually reviewed answerable and abstention
cases bound to the accepted passage artifact. Published version `0.1.0` remains
available unchanged so earlier measurements remain reproducible.

The accepted BM25 result is `data/results/bm25-baseline-v0.1.0.json`. It searches
all 707 passages without using benchmark document scope and records mean
Recall@10 of `0.7760`, MRR@10 of `0.7433`, direct-evidence hit rate@10 of
`0.9375`, and mean nDCG@10 of `0.6555`.

The accepted dense result is `data/results/dense-baseline-v0.1.0.json`. It uses
the pinned `BAAI/bge-small-en-v1.5` ONNX model over the same full corpus and
records mean Recall@10 of `0.9688`, MRR@10 of `0.9062`, direct-evidence hit
rate@10 of `1.0000`, and mean nDCG@10 of `0.8866`. The model asset is supplied
locally and verified by exact size and SHA-256; it is not committed or downloaded
automatically at runtime.

The accepted hybrid candidate result is
`data/results/hybrid-candidate-baseline-v0.1.0.json`. It takes the top 20
full-corpus results from BM25 and dense retrieval, forms a deduplicated union,
and preserves both source ranks without assigning a fused score or final rank.
The union contains all reviewed evidence for all 16 answerable benchmark queries,
with mean candidate count `31.3125`. Candidate depth 20 was selected after
diagnostics on the fixed benchmark, so this coverage result is benchmark-informed
and is not an out-of-sample generalization claim.

The accepted reranker result is
`data/results/reranker-baseline-v0.1.0.json`. It uses the pinned
`cross-encoder/ms-marco-MiniLM-L6-v2` ONNX model to score only the accepted
hybrid candidate union. It records mean Recall@10 of `0.9271`, MRR@10 of
`0.8250`, direct-evidence hit rate@10 of `1.0000`, and mean nDCG@10 of
`0.7893`. The result is byte reproducible and improves over BM25, but it trails
the accepted dense baseline. Dense retrieval therefore remains the selected
ranking; the cross-encoder is retained as a reproducible comparison baseline.

Next: close the retrieval phase with final audits and staged-change review.
Generation, API, and UI remain later phases.

## Responsible-use notice

PolicyProof is a research and compliance-support application. It does not provide
legal advice, determine legal compliance, or replace review by qualified
professionals.
