# Architecture

Status: ingestion, passage materialization, retrieval evaluation, hybrid candidate generation, cross-encoder comparison, and the evidence-sufficiency evaluation contract are implemented, independently audited, and regression tested.

PolicyProof will use a framework-independent Python pipeline for ingestion,
retrieval, reranking, evidence-sufficiency assessment, grounded generation,
claim extraction, citation verification, and structured tracing.

FastAPI will later expose the stable pipeline but will not contain core retrieval
or verification logic.

## Implemented ingestion pipeline

The current deterministic ingestion foundation consists of these independently
callable stages:

1. Validate the version-controlled source manifest.
2. Download approved documents and record checksum snapshots.
3. Extract page-level text while retaining stable document and page identity.
4. Detect document-specific heading candidates.
5. Reconstruct complete headings with exact source-line provenance and reviewed,
   document-specific display-text normalization.
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

## Reviewed heading normalization

Raw page text and every heading's `source_lines` remain immutable provenance.
Display text in `full_heading` may apply deterministic, corpus-reviewed
normalization when PDF extraction splits an ordinary word at a visual line
wrap.

For NIST AI RMF function headings, the implemented policy:

- removes 51 reviewed PDF wrap hyphens
- preserves the legitimate compounds `context-specific` and
  `context-relevant`
- corrects the reviewed extraction glue `theMAP` to `the MAP`
- preserves the original page, line, marker, and `source_lines` fields
- does not use dictionary lookup, language-model inference, or generic
  corpus-wide dehyphenation

The normalization changes 40 of 349 reconstructed headings. It does not change
heading identifiers, source ordering, hierarchy parents, hierarchy depths,
direct-body coordinates, subtree coordinates, or synthetic envelopes.

Accepted local artifact checksums are:

- `heading-candidates.jsonl`:
  `0842bb2e6cfaa05103918cd24579ad252203e384bab6015a3ee864218b8796d0`
- `reconstructed-headings.jsonl`:
  `fb3c6d3d9b3615e78bcca5f9e075c88bda50eed5c5ef1a3643335fcd45c5d621`
- `heading-hierarchy.jsonl`:
  `a49c97741378e0ee60c531b0eee2d3f6d66c37057b56556e8c06523e8f19b928`
- `heading-spans.jsonl`:
  `67ad7444bd77a384df85e0fdef8f3f18aba18c76646b08a661c405c216817871`

## Generated artifacts

Generated corpus artifacts under `data/processed/` remain local and are ignored
by Git. The repository stores the deterministic builders and tests needed to
reproduce them.

The accepted local ingestion artifacts are:

- `data/processed/pages.jsonl`
- `data/processed/heading-candidates.jsonl`
- `data/processed/reconstructed-headings.jsonl`
- `data/processed/heading-hierarchy.jsonl`
- `data/processed/heading-spans.jsonl`

`heading-spans.jsonl` is derived from the pages, reconstructed headings, and
hierarchy artifacts. Reviewed display-text normalization does not alter source
coordinates. The compact GPT-4o appendix correction deliberately adds two
source spans and shortens the existing `References` span to page 31, line 29.

## Production Phase 1.4d retrieval-unit builder

Phase 1.4d promotes the independently audited coordinate-only design into
production Python modules:

- `src/policyproof/retrieval_units.py`
- `src/policyproof/retrieval_policy.py`
- `src/policyproof/retrieval_builder.py`

The builder accepts explicit paths for `pages.jsonl`,
`heading-hierarchy.jsonl`, and `heading-spans.jsonl`. It has no `runpy`
coupling, environment-variable namespace coupling, embedded temporary paths,
or prototype-script dependency.

The production build contains:

- 581 retrieval units from 487 logical sources
- 10,019 retrieval-content coordinates, each owned exactly once
- 624 heading-context coordinates
- a complete 12,008-record coordinate ledger
- 53 heading-only evidence units
- 180 complete EU recitals represented by 181 units
- 94 reviewed internal boundaries with zero known semantic defects
- 146 explicit blank-line exclusions
- 72 explicit EU ELI footer exclusions
- 39 omitted empty hierarchy containers
- one reviewed semantic exception: EU recital 29 at 580 indexed words under a
  640-word ceiling

The final unit distribution is:

- 344 heading-body units
- 53 heading-only units
- 181 EU-recital units
- 3 frontmatter-body units

Outputs remain coordinate-only. They contain no retrieval text, citation text,
embeddings, character offsets, or vector-index records.

The production builder preserves deterministic document ordering, terminal
`:part-NNN` unit IDs, complete bibliography entries, URL continuation,
footnote continuation, list structure, and the reviewed EU recital 53 split.
All outputs refuse overwrite and are written atomically with rollback across
the multi-file build.

Generated artifacts remain local and ignored. Their accepted production
SHA-256 checksums are:

- `retrieval-units.jsonl`:
  `4726f293b6cea614e86c6d61bd240f4da87c5fb139169ae2b5e81faba7d658c0`
- `retrieval-coordinate-ledger.jsonl`:
  `dc599de4b2766e588adabb584912b1eb20080cca18b03403ab5bddd8b5f569e8`
- `retrieval-units-summary.json`:
  `324f146626c620917fe7178e0fd9a721104d224531894f4e2b6545676a60f2a4`
- `retrieval-units-review.txt`:
  `46167f087bd99a6a2f5a2076fd59248b226dc89d60e08c25a6bdc03fdc47ff17`

The accepted temporary prototype remains useful only as an independent parity
oracle. It is not a runtime dependency or a committed generated artifact.

## Production Phase 1.4e token-safe passage layer

Phase 1.4e derives model-safe passages from the accepted Phase 1.4d
coordinate-only retrieval units without changing their coordinate ownership or
the accepted 12,008-record ledger.

The implementation consists of:

- `src/policyproof/retrieval_materialization.py`
- `src/policyproof/retrieval_passages.py`
- `src/policyproof/retrieval_tokenizer.py`

The passage builder uses the pinned BERT WordPiece tokenizer and the following
pair-input contract:

- model maximum length: 512 tokens
- reserved query budget: 64 tokens
- pair special tokens: 3
- passage hard limit: 445 tokens
- passage packing target: 384 tokens

The derived corpus contains:

- 707 token-safe passages
- all 487 logical sources
- all 581 accepted coordinate-only retrieval units as provenance inputs
- 203 EU-recital passages
- 4 frontmatter-body passages
- 447 heading-body passages
- 53 heading-only passages
- 27 complete-reference-entry passages
- 14 reviewed sentence boundaries inside source lines
- 0 passages above the 445-token hard limit
- a maximum observed passage length of exactly 445 tokens

The 14 intra-line boundaries are exact corpus-specific anchors. Each anchor
includes the logical source key, source coordinate, character offset, and
expected text on both sides. They are used only where complete source lines
cannot satisfy the pinned token budget.

Character offsets exist only in the derived passage `source_slices`. The
accepted retrieval units remain coordinate-only, and the coordinate ledger
continues to assign each extracted source line to exactly one retrieval-content
owner.

A complete provenance audit confirmed:

- every accepted source character is represented exactly once within its
  logical source
- each split line forms two adjacent character ranges
- there are no character gaps or overlaps
- passage numbering and passage counts are contiguous
- every passage links to the complete ordered set of source retrieval units
- reference-entry ordinal ranges are continuous
- no unreviewed intra-line split was introduced

The production passage writer refuses overwrite, writes atomically, and rolls
back earlier outputs if any later output fails.

Generated passage artifacts remain local and ignored. Their accepted SHA-256
checksums are:

- `retrieval-passages.jsonl`:
  `918e6d30f2e1900386f5f3e9f5311042f47560c6aaca90168bfc4008f807f874`
- `retrieval-passages-summary.json`:
  `a4d0e9afc004f8da8fa93d5ff895d3ff3cf5540c65cfb0c1b182c026f985a626`
- `retrieval-passages-review.txt`:
  `9f03dbe06bb74f2d59cc05b49c0ca36758ad47398b5cfe8e4e150ead6658fa45`

The passage artifacts contain source-slice provenance and token counts. They do
not yet persist retrieval text, citation text, embeddings, or vector-index
records.

## Production Phase 1.4f persisted passage text

Phase 1.4f finalizes the text representations stored with each accepted
Phase 1.4e passage. It does not change passage identity, source slices,
coordinate ownership, semantic boundaries, passage ordering, or token budgets.

The passage schema is now version `1.1` and persists two distinct strings:

- `retrieval_text` is the exact label-prefixed text used for token accounting,
  indexing, embedding, and reranking.
- `citation_text` is the source-derived evidence text without the
  retrieval-only label prefix.
- For the 53 heading-only passages, the reviewed reconstructed heading is both
  the retrieval text and the citation evidence because the heading itself is
  the accepted evidence unit.

`passage_token_count` is calculated from the exact persisted
`retrieval_text`. Records that already contain `retrieval_text`,
`citation_text`, or `passage_token_count` are rejected before materialization,
preventing stale or externally supplied text from silently replacing the
deterministic source-slice reconstruction.

The accepted corpus remains:

- 707 passages
- 487 logical sources
- 581 source retrieval units represented as provenance
- 53 heading-only passages
- 27 reference passages
- 14 reviewed intra-line boundaries
- maximum passage length of 445 tokens
- zero passages over the hard limit
- zero changed pre-existing passage fields other than the schema version

The persisted corpus contains:

- 903,356 retrieval-text characters
- 865,405 citation-text characters
- no embeddings
- no vector-index records
- no change to the accepted coordinate ledger

The previous schema `1.0` passage artifacts were verified against their
accepted hashes and archived locally under:

`data/processed/archive/retrieval-passages-schema-1.0-pre-persisted-text-3bde189/`

Generated artifacts remain local and ignored. The accepted schema `1.1`
SHA-256 checksums are:

- `retrieval-passages.jsonl`:
  `5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96`
- `retrieval-passages-summary.json`:
  `aff873450e04744f71580fdf2792b56edceceb61259b3f20ea54bc735f3c1bb9`
- `retrieval-passages-review.txt`:
  `8cd2574c360765a8eb8fdac1cae7232db97ce1cf372a7a24b89a69c9c6f29bec`

## Implemented retrieval and ranking pipeline

The accepted retrieval implementation consists of independently callable,
framework-free Python stages:

1. Load and validate the 707 accepted schema-`1.1` passage records.
2. Build a full-corpus dense index with the pinned
   `BAAI/bge-small-en-v1.5` ONNX model.
3. Rank a user question against all passages with `rank_dense`.
4. Resolve equal similarity scores by accepted passage order and passage ID.
5. Return deterministic `DenseHit` records containing passage identity,
   accepted order, similarity score, and relevance rank.

BM25 remains an accepted lexical baseline. The top-20 BM25 and top-20 dense
rankings can also be combined into a deduplicated candidate union with complete
reviewed-evidence coverage on the current benchmark.

The candidate union is bounded input for reranker experiments. It is not the
selected production ranking and must not be interpreted as relevance-ordered
output.

## Cross-encoder comparison

PolicyProof includes a deterministic comparison implementation for
`cross-encoder/ms-marco-MiniLM-L6-v2` using direct CPU ONNX Runtime inference.

The reranker:

- scores only the accepted hybrid candidate union
- encodes pairs as `[CLS] query [SEP] passage [SEP]`
- uses raw classification logits without normalization
- rejects pairs above the 512-token limit
- preserves BM25 and dense source-rank provenance
- resolves equal logits by accepted passage order and passage ID
- requires an explicitly supplied, size- and SHA-256-verified model file
- never downloads or commits the model asset

All 501 accepted benchmark query-candidate pairs fit the model input limit; the
maximum observed pair length is 467 tokens.

The reranker improves over BM25 but underperforms the accepted dense ranking.
It is retained as a reproducible experimental baseline rather than selected as
the production ranking.

## Selected retrieval contract

Dense retrieval remains the selected ranking for the next pipeline phase.

The selected runtime path is:

- construct one `DenseIndex` from all accepted passages
- use no benchmark `document_scope` filtering
- apply the fixed query instruction only to the question
- rank by normalized dot-product similarity
- expose the top retrieved passages with stable IDs and accepted-order
  provenance
- use each passage's `citation_text`, not its label-prefixed `retrieval_text`,
  when presenting evidence or constructing citations

The current accepted retrieval benchmark measures ranking quality only for the
16 answerable questions. The four abstention questions are now represented in
the evidence-sufficiency evaluation dataset, but they have not been used to set
or validate a runtime decision policy.

Retrieval scores therefore must not be interpreted as calibrated confidence,
proof that an answer is supported, or permission to generate an answer.

## Evidence-sufficiency evaluation boundary

PolicyProof implements a separate evaluation-only contract for deciding whether
a reviewed passage set supports the complete source question.

Implemented components:

- `src/policyproof/evidence_sufficiency_evaluation.py`
- `data/evaluation/evidence-sufficiency-evaluation-v0.1.0.json`
- `tests/test_evidence_sufficiency_evaluation.py`

The validator is fail-closed. It verifies:

- exact corpus, passage, and retrieval-benchmark bindings
- schema and dataset identity
- source-query identity and exact question text
- accepted passage IDs
- source expected behavior
- source relevance grades
- sufficient/answer and insufficient/abstain cross-field rules
- allowed reason codes
- explicit missing-information statements
- unique case IDs and evidence IDs
- rejection of unknown fields

The accepted artifact contains 39 manually reviewed cases over all 20 source
queries:

- 16 sufficient reference cases
- 19 incomplete-evidence cases
- 4 required-abstention cases using actual dense top-five evidence

The evaluation layer is downstream of retrieval and upstream of any future
runtime decision policy:

    accepted passages
            |
    selected dense ranking
            |
    question + candidate evidence set
            |
    evidence-sufficiency evaluation contract
            |
    future runtime sufficiency policy
            |
    future grounded answer or informative abstention

The committed dataset does not execute the sufficiency decision at runtime. It
defines the expected result for reviewed question-and-evidence combinations.

Retrieval similarity, BM25 score, dense score, cross-encoder logit, rank, and
candidate count are not treated as calibrated answer confidence.

Insufficient cases preserve two outputs needed by future policy evaluation:

- machine-readable reason codes
- concrete descriptions of missing information

These outputs distinguish incomplete retrieved evidence from other boundaries,
including:

- information outside the controlled corpus
- current information
- organization-specific conclusions
- legal-advice boundaries
- high-stakes recommendations
- unsupported comparisons

The four source abstention cases use actual dense top-five passage sets. No
production `document_scope` filter, query-specific threshold, benchmark label,
or hidden evaluation metadata is used to construct retrieval candidates.

No runtime classifier, threshold, prompt-based judge, language-model evaluator,
grounded generator, or abstention response generator has been selected.

## Leakage-safe evaluation splits

PolicyProof now implements deterministic leakage-component construction and
split-manifest validation for evidence-sufficiency evaluation.

Implemented components:

- `src/policyproof/evidence_sufficiency_splits.py`
- `tests/test_evidence_sufficiency_splits.py`
- `data/evaluation/evidence-sufficiency-split-manifest-v0.1.0.json`

Cases are connected into one leakage component when they share:

- a source query ID
- an exact evidence passage ID
- an accepted passage `logical_source_key`

The component relation is transitive. If one case shares a query with a second
case and the second shares evidence with a third case, all three remain in one
component.

The implementation deliberately does not connect all passages from the same
document. Document-level grouping would be unnecessarily coarse and would
prevent meaningful document representation across future splits.

Component construction is:

- deterministic across case and passage input order
- non-mutating
- fail-closed for duplicate case or passage IDs
- fail-closed for unknown evidence passage IDs
- fail-closed for missing or malformed query, case, passage, or logical-source
  identifiers

Split assignments must contain exactly:

- `development`
- `validation`
- `test`

Every accepted case must be assigned exactly once. A complete leakage component
cannot span more than one split.

The first published split manifest binds to evidence dataset version `0.1.0`:

- manifest schema version: `1.0`
- manifest version: `0.1.0`
- size: `2103` bytes
- component algorithm version: `1.0.0`
- component count: `19`
- development cases: `39`
- validation cases: `0`
- test cases: `0`
- SHA-256:
  `314d5ca55a1d6557e8f711eea3506ce13a85d30f40e706ea27f0afb8226ff4b2`

The 39 development cases originate from 20 source queries but form only 19
independent leakage components.

One component crosses source-query IDs:

- `abstain-004`
- `gpt4o-003`

Those queries share the accepted passage:

`candidate-v2:openai-gpt-4o-system-card-2024-08-08:source:openai-gpt-4o-system-card-2024-08-08:page-0012:line-0035:passage-001`

They also share its `logical_source_key`. They therefore must remain in the same
split.

All existing cases are intentionally assigned to development because dataset
version `0.1.0` was inspected while the evidence schema, reason codes,
construction rules, and validators were designed.

The split manifest does not manufacture validation or test data. New
independently annotated query groups are required before those splits can be
populated.

No runtime evidence-sufficiency policy has been selected or evaluated by this
split infrastructure.

## Deferred production stages

Runtime evidence-sufficiency policy, abstention execution, grounded answer
generation, claim extraction, citation verification, structured tracing, API
exposure, and UI remain downstream stages.

Persisted embeddings and a local vector-index artifact also remain optional
future optimizations. Their absence does not block the next correctness phase
because the accepted dense index can be constructed deterministically from the
passage corpus and explicitly supplied model asset.
