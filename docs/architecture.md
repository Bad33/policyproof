# Architecture

Status: Phase 1.4d coordinate-only retrieval-unit builder implemented, independently audited, and regression tested.

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

The normalization changes 40 of 347 reconstructed headings. It does not change
heading identifiers, source ordering, hierarchy parents, hierarchy depths,
direct-body coordinates, subtree coordinates, or synthetic envelopes.

Accepted local artifact checksums are:

- `reconstructed-headings.jsonl`:
  `911f8379ba1633ffa189102143514aeff7e7e98e163fe89c7af806fffb169356`
- `heading-hierarchy.jsonl`:
  `690f640e5cc44a4dc76785b5b2ec6e4878811f956a646bdd757c8432ae790303`
- `heading-spans.jsonl`:
  `e863d78800faceeeedbd08ea2b5a406bb4e8e81cecf9dfae9bf08d6600604c5d`

## Generated artifacts

Generated corpus artifacts under `data/processed/` remain local and are ignored
by Git. The repository stores the deterministic builders and tests needed to
reproduce them.

The accepted local ingestion artifacts are:

- `data/processed/pages.jsonl`
- `data/processed/reconstructed-headings.jsonl`
- `data/processed/heading-hierarchy.jsonl`
- `data/processed/heading-spans.jsonl`

`heading-spans.jsonl` is derived from the first three artifacts. The
normalization correction changes reconstructed display text and derived
hierarchy paths, but the coordinate-only span artifact remains byte-identical.

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

- 579 retrieval units from 485 logical sources
- 10,021 retrieval-content coordinates, each owned exactly once
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

- 342 heading-body units
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
  `e9675edc15a8cc7651a17ad8c9134f4b9166a5fc039d679602c7db542cf2aa07`
- `retrieval-coordinate-ledger.jsonl`:
  `0b59132e7cdd6b68b667e07ad54efe762ba7b6a7572584f4c2fd94fcc8bf3a78`
- `retrieval-units-summary.json`:
  `7b4160740e682bf759f3f506c24d0e4fcd7e56bfa106e4ed09e30d671c0fdd15`
- `retrieval-units-review.txt`:
  `f683f4dd2d5a3487704ad397ec4a93d005054068990e16f0df19f87cd31dddaa`

The accepted temporary prototype remains useful only as an independent parity
oracle. It is not a runtime dependency or a committed generated artifact.

## Deferred production stages

Citation-unit identity, retrieval-text materialization, tokenizer selection,
final token budgets, indexing, retrieval, reranking, evidence-sufficiency
checks, generation, and citation verification remain downstream stages.
