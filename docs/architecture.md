# Architecture

Status: Phase 1.4c ingestion foundation implemented; Phase 1.4d retrieval-unit design independently audited.

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

## Audited Phase 1.4d retrieval-unit design

Phase 1.4d has produced an independently audited, temporary coordinate-only
retrieval-unit candidate. It is a design artifact, not a production chunk
corpus and not a committed generated artifact.

The audited candidate contains:

- 579 retrieval units from 485 logical sources
- 10,021 retrieval-content coordinates, each owned exactly once
- 53 heading-only evidence units covering 188 NIST AI RMF source coordinates
- 180 complete EU recitals represented by 181 units
- 94 reviewed internal boundaries with zero known semantic defects
- 146 explicit blank-line exclusions
- 72 EU ELI footer exclusions
- 5 GenAI reference units and 5 GPT-4o reference units
- one approved semantic-integrity exception: EU recital 29 at 580 words
  under a 640-word exception ceiling

The candidate stores coordinates and metadata only. It contains no assembled
retrieval text, citation text, embeddings, or character offsets.

Its accepted temporary checksums are:

- corrected candidate JSONL:
  `2fc8308ad1a221ebee7bde31c9ff9d76fda8de42391b0632ff5d673cb47c723f`
- corrected coordinate ledger:
  `d5b258efa67e0067146777893d2817955b29025a992ab68e2a8b8955c8bd1fc5`
- final independent audit:
  `57c93cbba08e6026577665a77c76b8f164491a72ee6144af922116e81b42a2be`

## Deferred production stages

A tested production retrieval-unit builder has not yet been added to the
repository. Citation-unit identity, retrieval-text materialization, tokenizer
selection, final token budgets, indexing, retrieval, reranking,
evidence-sufficiency checks, generation, and citation verification remain
downstream stages.
