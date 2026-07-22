# Engineering Decisions

This log records decisions that materially affect PolicyProof's scope,
architecture, evaluation, reproducibility, security, or maintainability.

---

## PP-001: Evidence-focused product boundary

**Decision:**

PolicyProof will report what selected public sources state. It will not provide
legal advice, determine legal compliance, or make organization-specific legal
conclusions.

**Context:**

The corpus includes regulatory and AI-governance material that users could
mistake for personalized legal guidance.

**Options considered:**

- Legal-advice assistant
- Automated compliance checker
- Evidence-focused research assistant

**Selected option:**

Evidence-focused research assistant.

**Why:**

This scope is safer, more testable, and appropriate for the available public
document corpus.

**Trade-offs:**

Some questions must be rejected even when a language model could generate a
plausible response.

**How we will verify it:**

Include legal-advice and organization-specific compliance questions in the
answerability evaluation dataset.

**Date:**

2026-07-18

---

## PP-002: Evaluation before interface development

**Decision:**

Build the controlled corpus and manually reviewed evaluation dataset before a
polished API or user interface.

**Context:**

Without evaluation data, retrieval and verification improvements cannot be
measured objectively.

**Options considered:**

- UI-first development
- API-first development
- Evaluation-first pipeline development

**Selected option:**

Evaluation-first pipeline development.

**Why:**

It allows BM25, dense retrieval, hybrid retrieval, and reranking to be compared
using the same corpus and questions.

**Trade-offs:**

The project will not have a polished visual demonstration during its early
phases.

**How we will verify it:**

No frontend work begins before retrieval and answerability baselines have been
measured.

**Date:**

2026-07-18

---

## PP-003: Framework-independent core pipeline

**Decision:**

Implement ingestion, retrieval, evaluation, and citation-verification logic in
plain Python instead of using LangChain or LangGraph.

**Context:**

The important engineering and evaluation logic must remain visible,
understandable, and independently testable.

**Options considered:**

- LangChain
- LangGraph
- LlamaIndex
- Plain Python

**Selected option:**

Plain Python.

**Why:**

The initial pipeline does not require framework-based orchestration.

**Trade-offs:**

We will write and maintain more integration code ourselves.

**How we will verify it:**

Every pipeline stage must be callable independently from tests and command-line
evaluation scripts.

**Date:**

2026-07-18

---

## PP-004: Python 3.12 development environment

**Decision:**

Develop and test PolicyProof using Python 3.12 and the standard `venv` module.

**Context:**

Some AI and machine-learning libraries may not support newer Python releases
reliably.

**Options considered:**

- Python 3.11
- Python 3.12
- Python 3.14

**Selected option:**

Python 3.12.

**Why:**

It provides modern Python functionality while maintaining broad library
compatibility.

**Trade-offs:**

The project intentionally restricts local and CI environments to Python 3.12
until compatibility requirements change.

**How we will verify it:**

`pyproject.toml` requires Python `>=3.12,<3.13`, and GitHub Actions will later
run against Python 3.12.

**Date:**

2026-07-18

---

## PP-005: File-based storage before PostgreSQL

**Decision:**

Use JSON, JSONL, Markdown, and generated local index files before introducing a
database.

**Context:**

The first corpus contains only a few controlled documents and is maintained by
one developer.

**Options considered:**

- Local files
- SQLite
- PostgreSQL

**Selected option:**

Local files.

**Why:**

They are inspectable, easy to version, and sufficient for initial offline
experiments.

**Trade-offs:**

Local files are not ideal for concurrent writes, complex querying, or large
trace datasets.

**How we will verify it:**

PostgreSQL will be introduced only after identifying a persistence, concurrency,
or analytical-query requirement that local files cannot handle cleanly.

**Date:**

2026-07-18

---

## PP-006: Official-source allowlist

**Decision:**

The initial corpus will contain documents downloaded only from manually approved
official source domains.

**Context:**

Third-party summaries may omit, reinterpret, or reproduce outdated policy text.

**Options considered:**

- General web crawling
- Search-engine result ingestion
- Official-source allowlist

**Selected option:**

Official-source allowlist.

**Why:**

It improves provenance, version control, and confidence that the original
document was retrieved.

**Trade-offs:**

Corpus expansion will be slower and may require source-specific extraction
logic.

**How we will verify it:**

The ingestion pipeline will reject source URLs whose domains are absent from the
approved source manifest.

**Date:**

2026-07-18

---

## PP-007: No paid model calls in normal CI

**Decision:**

Pull-request CI will use deterministic fixtures, local computations, and mocked
generation responses.

**Context:**

External model calls introduce cost, nondeterminism, latency, rate limits, and
secret-management risks.

**Options considered:**

- Live model calls in every pull request
- Fully offline pull-request CI
- Offline pull-request CI with optional scheduled live evaluation

**Selected option:**

Offline pull-request CI with the possibility of scheduled live evaluation later.

**Why:**

Normal CI must be repeatable and safe to run from a public repository.

**Trade-offs:**

Offline CI cannot detect every behavior change introduced by an external model
provider.

**How we will verify it:**

The default test suite and pull-request workflow must pass without API keys or
network access.

**Date:**

2026-07-18

---

## PP-008: Research entries require implementation

**Decision:**

A paper will be added to `docs/research-notes.md` only when PolicyProof
implements or evaluates one of its findings.

**Context:**

A long bibliography does not make a project research-driven unless the research
changes an implementation or experiment.

**Options considered:**

- General bibliography
- Broad literature survey
- Implementation-linked research notes

**Selected option:**

Implementation-linked research notes.

**Why:**

This creates defensible engineering decisions and interview explanations.

**Trade-offs:**

The research-notes file will grow more slowly.

**How we will verify it:**

Every completed paper entry must identify a PolicyProof baseline, experiment,
result, or documented implementation difference.

**Date:**

2026-07-18

---

## PP-009: Manifest-controlled corpus entry

**Decision:**

Every source document must be declared in a version-controlled source manifest
before it can be downloaded or processed.

**Context:**

Retrieval experiments are meaningful only when the exact corpus identity,
version, provenance, and source locations are known.

**Options considered:**

- Hard-coded download URLs inside scripts
- Manual document downloads without metadata
- Version-controlled source manifest

**Selected option:**

Version-controlled source manifest.

**Why:**

The manifest separates source approval from downloading and extraction. It also
creates stable document identifiers and an explicit source-domain allowlist.

**Trade-offs:**

Adding or replacing a document requires a reviewed manifest change before
ingestion can proceed.

**How we will verify it:**

Manifest validation must pass before any download command runs. The downloader
will accept only records from the validated manifest.

**Date:**

2026-07-18

---

## PP-010: Controlled downloads with checksum snapshots

**Decision:**

Download approved corpus artifacts through a small standard-library downloader
that validates HTTPS destinations, file type, PDF signatures, size limits, and
SHA-256 checksums.

**Context:**

Manual downloads do not provide consistent safety controls or reproducible
artifact-integrity metadata.

**Options considered:**

- Manual browser or curl downloads
- Third-party HTTP client dependency
- Python standard-library downloader

**Selected option:**

Python standard-library downloader.

**Why:**

The initial corpus contains only four static documents. Python's standard
library is sufficient for streamed HTTPS downloads, redirect checks, timeouts,
atomic writes, and hashing.

**Trade-offs:**

The urllib API is less convenient than a dedicated HTTP client and does not
provide advanced retry or connection-pooling features.

**How we will verify it:**

Offline tests must cover successful downloads, unapproved destinations,
incorrect content types, invalid PDF signatures, oversized files, and overwrite
protection. Real downloads must produce a checksum snapshot for every approved
document.

**Date:**

2026-07-18

---

## PP-011: pypdf for initial page extraction

**Status:**

Superseded by PP-012 after measured extraction tests showed that pypdf layout
mode produced severe spacing corruption in the controlled corpus.

**Decision:**

Use pypdf layout-mode extraction as the initial page-level PDF extraction
method.

**Context:**

The controlled corpus consists of four digitally generated PDFs. PolicyProof
needs page text and page-number provenance before it needs advanced table,
coordinate, or OCR capabilities.

**Options considered:**

- pypdf
- pdfplumber
- PyMuPDF
- OCR-first extraction

**Selected option:**

pypdf using layout extraction with excess vertical spacing disabled.

**Why:**

pypdf is a small pure-Python dependency, provides page-level extraction, and is
sufficient for establishing an inspectable baseline. Extraction quality will be
measured manually before chunking begins.

**Trade-offs:**

Reading order, tables, footnotes, and multi-column layouts may not be recovered
correctly. pypdf cannot extract text from image-only pages.

**How we will verify it:**

Extract every page, report empty pages and character counts, and manually compare
representative extracted pages against the rendered PDFs. A more advanced parser
will be considered only for confirmed extraction failures.

**Date:**

2026-07-18

---

## PP-012: Document-specific PDF extraction

**Decision:**

Use pdfplumber default extraction for the EU AI Act and pypdf plain extraction
for the other initial corpus documents.

**Context:**

The initial pypdf layout extraction completed structurally but produced severe
spacing corruption. A controlled comparison tested pypdf plain and pdfplumber
default extraction on the same representative pages.

**Options considered:**

- pypdf layout for all documents
- pypdf plain for all documents
- pdfplumber default for all documents
- Document-specific extraction

**Selected option:**

Document-specific extraction.

**Why:**

On the evaluated pages, pypdf plain preserved expected phrases for the NIST AI
RMF, NIST Generative AI Profile, and GPT-4o System Card. pdfplumber preserved
the EU AI Act heading and text that both pypdf modes split internally.

No single tested extractor produced the best result for all four documents.

**Trade-offs:**

The extraction pipeline now has two PDF dependencies and records different
extraction methods across documents. Future corpus additions require a small
extraction-quality review before selecting a method.

**How we will verify it:**

Every page record stores its extraction method, library, and version. The final
corpus extraction will be manually checked on representative pages and tested
for contiguous page numbers, stable checksums, unique page IDs, and readable
expected phrases.

**Date:**

2026-07-18

---

## PP-013: Document-specific heading candidate detection

**Decision:**

Use deterministic, document-specific heading patterns to generate reviewable
heading candidates before constructing sections.

**Context:**

The four controlled documents use different heading systems. The EU AI Act uses
chapters, sections, articles, and annexes. NIST publications use numbered
headings and AI RMF function identifiers. The GPT-4o System Card uses numbered
sections and subsections.

A single generic numbered-heading rule produced table-of-contents entries,
numbered risk definitions, and numbered prose items as false positives.

**Options considered:**

- One generic heading pattern for all documents
- LLM-based heading classification
- Document-specific deterministic patterns
- Building chunks directly from page boundaries

**Selected option:**

Document-specific deterministic candidate detection followed by measured manual
review.

**Why:**

The controlled corpus is small and structurally distinct. Explicit rules are
auditable, reproducible, offline, and preserve exact page and line provenance.
They also fail closed for unknown documents.

**Measured refinements:**

The initial detector produced 397 candidates. Conservative filters removed
table-of-contents pages, long numbered definitions, numbered prose
continuations, and an annex cross-reference incorrectly detected as Article 49.

The accepted candidate baseline contains 347 candidates:

- EU AI Act: 157
- NIST Generative AI Profile: 54
- NIST AI RMF: 108
- GPT-4o System Card: 28

All EU AI Act Articles 1 through 113 were detected exactly once.

**Trade-offs:**

These records are heading candidates, not completed sections. Some headings span
multiple extracted lines, and function-category tables may contain semantic
headings that require continuation-line reconstruction. New document formats
will require an explicit extraction and heading review.

**How we will verify it:**

Automated tests cover known heading formats and false-positive cases. Stable
anchor headings are checked across all four documents, and EU article numbering
is validated for completeness and uniqueness.

**Date:**

2026-07-18

---

## PP-014: Provenance-preserving heading reconstruction

**Status:**

Superseded in part by PP-017 for reviewed NIST AI RMF display-text normalization. The source-line provenance and document-specific boundary rules remain governing decisions.

**Decision:**

Reconstruct complete heading text from reviewed heading candidates while
preserving the exact page lines that contributed to every reconstructed
heading.

**Context:**

The controlled documents contain several structurally different heading forms.
EU legal headings separate structural markers such as Article 49 or Annex VIII
from their title text. NIST AI RMF function statements frequently wrap across
multiple extracted lines. Some NIST appendix markers are also separated from
their titles.

Using the candidate line alone produced incomplete headings. An initial generic
continuation policy also introduced several errors:

- EU article body paragraphs were appended to article titles.
- A four-line continuation limit truncated valid NIST function statements.
- Joining wrapped lines introduced spaces after extraction hyphens.
- Numbered Appendix D attributes were misclassified as headings while the split
  Appendix D marker was missed.

**Selected approach:**

Use document-specific reconstruction policies:

- EU chapters, sections, and articles consume one following title line.
- EU annexes may consume multiple title lines with conservative body-start
  stopping rules.
- NIST RMF function statements continue until terminal punctuation, the next
  heading candidate, an explicit table/action boundary, or a twelve-line safety
  limit.
- Split NIST appendix markers consume one following title line.
- Generic wrapped source lines are joined without inserting an additional
  space after a line-ending hyphen. NIST AI RMF function headings then apply
  the reviewed normalization policy defined in PP-017.
- Every output stores the exact source lines and source line numbers used.

**Measured result:**

The accepted output contains 347 reconstructed headings from 347 reviewed
candidates:

- EU AI Act: 157
- NIST Generative AI Profile: 54
- NIST AI RMF: 108
- GPT-4o System Card: 28

Validation confirmed:

- 347 unique heading identifiers
- EU Articles 1 through 113 detected exactly once
- Appendix D reconstructed as `Appendix D: Attributes of the AI RMF`
- No late NIST numbered-list candidates
- No spaces introduced after line-ending hyphens
- No NIST action identifiers appended
- No headings reached the twelve-line continuation limit
- No RMF function headings ended with suspicious continuation words

**Trade-offs:**

The initial accepted baseline preserved extracted hyphen characters and could
therefore retain forms such as `inte-grated`. A later corpus-wide audit showed
that this reduced heading and heading-only evidence quality. PP-017 supersedes
that part of the policy with a closed, manually reviewed normalization set while
keeping exact raw source lines unchanged.

The output represents complete heading labels, but hierarchy and source-span
ownership remain separate downstream stages.

**Date:**

2026-07-18

---

## PP-015: Document-specific heading hierarchy with explicit synthetic nodes

**Decision:**

Assign hierarchy using document-specific structural rules. Introduce synthetic
RMF function or category nodes only where a document uses subcategory labels
without providing the corresponding source heading.

**Context:**

The reconstructed corpus contains 347 source headings, but the documents do not
share one hierarchy convention.

The EU AI Act uses chapters, sections, articles, and annexes. Most sections
belong to chapters, while Annex XI contains two internal sections.

The GPT-4o System Card and ordinary NIST numbered headings use dotted numeric
depth.

NIST RMF function statements use identifiers such as `GOVERN 1.1` and
`MEASURE 2.3`. Some category headings such as `MAP 1` are present in the source,
while others such as `GOVERN 1` are omitted. The NIST Generative AI Profile
contains neither explicit function containers nor explicit category headings.

**Selected approach:**

- EU chapters and annexes are top-level nodes.
- EU sections attach to the active chapter or annex.
- EU articles attach to the active section, or directly to the active chapter
  when no section is active.
- Numbered headings use the number of dotted components as their structural
  depth.
- NIST AI RMF uses the real `5.1 Govern`, `5.2 Map`, `5.3 Measure`, and
  `5.4 Manage` headings as function containers.
- Source RMF category headings such as `MAP 1` remain source nodes.
- Missing RMF categories become synthetic nodes anchored to the first source
  subcategory that requires them.
- The NIST Generative AI Profile receives four synthetic function containers
  under `3. Suggested Actions to Manage GAI Risks`.
- Synthetic nodes never claim page-level source provenance. They store an
  explicit anchor heading instead.

**Measured result:**

The accepted hierarchy contains 380 nodes:

- 347 source nodes
- 33 synthetic nodes
- 4 synthetic RMF function nodes
- 29 synthetic RMF category nodes

Document totals:

- EU AI Act: 157
- NIST Generative AI Profile: 77
- NIST AI RMF: 118
- GPT-4o System Card: 28

Validation confirmed:

- Every reconstructed heading appears exactly once.
- All parent links resolve within the same document.
- Depth, ancestor lists, and hierarchy paths are consistent.
- No parent cycles exist.
- Source and synthetic provenance remain distinct.
- Annex XI contains two sibling sections.
- NIST AI RMF uses four real source function containers.
- The Generative AI Profile uses four explicit synthetic function containers.

**Trade-offs:**

Synthetic nodes improve traversal and retrieval consistency but are not source
claims. Downstream citation and user-facing output must distinguish synthetic
organizational metadata from headings printed in the original documents.

This phase assigns parent-child relationships only. It does not yet determine
the body-text span governed by each heading.

**Date:**

2026-07-19

---

## PP-016: Coordinate-only heading spans with separate synthetic envelopes

**Decision:**

Assign every source hierarchy node both a direct-body span and a subtree span
using deterministic page-and-line coordinates. Synthetic hierarchy nodes will
not claim direct source spans; they may receive a separately labeled source
descendant envelope.

**Context:**

Phase 1.4b assigned parent-child hierarchy to 347 reconstructed source headings
and 33 synthetic RMF organizational nodes, but it did not determine which source
lines each node governed.

A single section span was insufficient for downstream retrieval design because
two different ranges are needed:

- body text belonging directly to the current heading
- the complete source range governed by the heading and all descendants

Synthetic nodes also require special treatment. They improve hierarchy
navigation but do not represent headings printed in the source documents.

**Options considered:**

- Assign only one body span to every hierarchy node
- Copy descendant text into synthetic nodes
- Assemble and store body text during boundary assignment
- Store separate coordinate-only source spans and synthetic descendant envelopes

**Selected approach:**

- A source node's `direct_body` starts after the final source line used to
  reconstruct its own heading.
- The direct body stops immediately before the next source heading, regardless
  of hierarchy depth.
- A source node's `subtree` starts after its own heading and stops before the
  next source heading at the same or a shallower depth, or at document end.
- Consecutive headings produce explicit empty direct-body spans.
- Final headings terminate using an explicit document-end boundary.
- Multi-page spans retain exact page and line coordinates.
- Source subtree spans may contain descendant heading lines, but never the
  current node's own heading lines.
- Synthetic nodes have null source-heading provenance, null direct bodies, and
  null source subtrees.
- A synthetic node may receive a `source_descendant_envelope` covering its
  contiguous source descendants, beginning at the first descendant heading.
- Span records contain coordinates and derived measurements only. They do not
  contain assembled text or citation chunks.
- Output order exactly preserves `heading-hierarchy.jsonl` order.
- Generated `data/processed/` artifacts remain local and ignored by Git.

**Measured result:**

The accepted production output contains 380 span records:

- 347 source records
- 33 synthetic records
- 92 exact-empty source direct bodies
- 0 blank-only source direct bodies
- 166 multi-page source direct bodies
- 201 multi-page source subtrees
- 33 synthetic source descendant envelopes

Boundary combinations are:

- 686 source spans from `after_source_line` to `before_source_heading`
- 8 source spans from `after_source_line` to `after_document_line`
- 33 synthetic envelopes from `at_source_heading` to
  `before_source_heading`

The production artifact contains no assembled text, source-line copies,
retrieval chunks, or citation chunks.

Its accepted SHA-256 checksum is:

`e863d78800faceeeedbd08ea2b5a406bb4e8e81cecf9dfae9bf08d6600604c5d`

**Why:**

Coordinate-only spans preserve auditable source provenance while postponing text
materialization until chunking requirements are defined. Separating direct-body
and subtree ranges supports both leaf-level retrieval and hierarchy-aware
aggregation without duplicating source ownership.

Keeping synthetic descendant envelopes distinct prevents organizational nodes
created by PolicyProof from being mistaken for headings or passages present in
the original documents.

**Trade-offs:**

The JSONL records repeat boundary metadata and are larger than a minimal pair of
integer offsets. The accepted local artifact is approximately 792 KiB for 380
records.

Consumers must resolve coordinates against `pages.jsonl` before reading span
text. This adds a materialization step but preserves one authoritative copy of
the extracted source text.

Synthetic envelopes contain descendant heading lines by design and therefore
must not be presented as direct body text belonging to the synthetic label.

**How we verified it:**

- Unit tests cover direct-body boundaries, subtree containment, exact-empty
  spans, blank-only spans, multi-page EOF spans, synthetic envelopes, hierarchy
  order, overlapping-heading rejection, and overwrite protection.
- The full test suite passes with 67 tests.
- Ruff passes across `src` and `tests`.
- The production builder reproduces the accepted discovery prototype exactly
  after normalizing schema version and hierarchy order.
- Manual review covers EU Chapter III, EU Annex XI, NIST AI RMF source and
  synthetic nodes, NIST Generative AI Profile source and synthetic nodes,
  GPT-4o depth-three headings, exact-empty spans, multi-page ranges, and all
  four final-heading EOF boundaries.
- The generated artifact is byte-identical to the accepted dry run.

**Date:**

2026-07-19
---

## PP-017: Reviewed heading normalization with immutable raw provenance

**Decision:**

Normalize reviewed PDF line-wrap artifacts in NIST AI RMF function-heading
display text while retaining the exact extracted source lines and coordinates
as immutable provenance.

**Context:**

The PP-014 reconstruction policy correctly prevented `or- ganizational`-style
spacing defects, but it intentionally preserved every extracted hyphen. A
corpus-wide manual audit of 347 source headings found 53 NIST AI RMF
line-ending-hyphen boundaries:

- 51 were visual PDF wrap artifacts inside ordinary words.
- `context-specific` and `context-relevant` were legitimate compounds.
- One separate extraction defect joined `theMAP` without a space.

The artifacts affected heading display text and heading-only retrieval evidence.
They did not indicate a source-coordinate or heading-boundary failure.

**Options considered:**

- Preserve every extracted character.
- Apply dictionary-based dehyphenation.
- Apply generic language-model correction.
- Apply a deterministic, document-scoped reviewed correction policy.
- Modify raw extracted page text.

**Selected option:**

Apply a deterministic, document-scoped reviewed correction policy only to NIST
function headings.

- When a source line ends with a hyphen and the next source line begins with a
  lowercase letter, join the two fragments without the wrap hyphen.
- Preserve only the reviewed compounds `context-specific` and
  `context-relevant`.
- Apply the exact document correction `theMAP` to `the MAP`.
- Keep `source_lines`, `source_line_numbers`, page identity, line identity,
  marker text, and candidate identity unchanged.
- Keep generic reconstruction behavior unchanged for other heading types and
  documents.
- Do not use a dictionary, probabilistic model, or broad text-rewriting rule.

**Measured result:**

Temporary full-corpus regeneration confirmed:

- 347 reconstructed headings retained.
- 40 headings received corrected display text.
- 51 visual wrap hyphens were removed.
- 2 legitimate compounds were preserved.
- 1 `theMAP` extraction glue was corrected.
- 0 raw source-line changes occurred.
- 380 hierarchy nodes were retained.
- 380 span records were retained.
- Hierarchy parentage, depth, source order, and coordinate ownership were
  unchanged.
- `heading-spans.jsonl` remained byte-identical.
- All 53 heading-only evidence units match normalized reconstructed-heading word
  counts.
- MEASURE 2.6 now contains 69 normalized heading words.

Accepted local checksums:

- `reconstructed-headings.jsonl`:
  `911f8379ba1633ffa189102143514aeff7e7e98e163fe89c7af806fffb169356`
- `heading-hierarchy.jsonl`:
  `690f640e5cc44a4dc76785b5b2ec6e4878811f956a646bdd757c8432ae790303`
- `heading-spans.jsonl`:
  `e863d78800faceeeedbd08ea2b5a406bb4e8e81cecf9dfae9bf08d6600604c5d`

**Why:**

A closed reviewed correction set improves retrieval text without weakening
provenance. The normalized display text is reproducible and testable, while the
raw extracted lines remain available for audit, citation verification, and
future extractor comparisons.

Restricting the policy to the audited NIST function-heading format prevents a
local PDF repair rule from silently rewriting unrelated documents.

**Trade-offs:**

New NIST function-heading extraction patterns may require another explicit
review. The allowlist and exact correction map are corpus-specific by design.

The normalized heading is suitable for indexing and display, but citations must
continue to resolve to raw page-and-line provenance rather than treating the
normalized string as the original PDF byte sequence.

**How we verified it:**

- Focused heading-reconstruction tests cover ordinary wrap dehyphenation,
  preservation of both legitimate compounds, multiple corrections in one
  heading, the exact `theMAP` repair, and immutable raw source lines.
- Full temporary reconstruction, hierarchy, and span generation was compared
  against the previously accepted artifacts.
- The changed-heading set exactly matched the coordinate-derived reviewed set.
- An independent corrected-corpus audit confirmed 10,021 retrieval coordinates
  with zero missing, unexpected, or overlapping ownership.
- The full project suite passes with 71 tests.
- Ruff and `git diff --check` pass.

**Date:**

2026-07-19
---

## PP-018: Production coordinate-only retrieval builder

**Decision:**

Promote the independently audited Phase 1.4d coordinate-only retrieval design
into production modules with explicit inputs, deterministic outputs, complete
coordinate ownership, atomic multi-file publication, and exact semantic parity
with the accepted prototype.

**Context:**

The accepted Phase 1.4d design established the correct logical sources,
document-specific exclusions, semantic boundaries, bibliography packing,
heading-only evidence behavior, and coordinate ledger. Its implementation,
however, depended on temporary scripts, `runpy`, environment variables, global
mutable state, absolute temporary paths, and destructive output cleanup.

Those properties were suitable for controlled discovery but not for a
production ingestion pipeline.

**Options considered:**

- Keep the temporary scripts as the production implementation.
- Copy the temporary scripts into the package with minimal changes.
- Reimplement only the final unit serialization layer.
- Port the accepted policies into explicit, independently tested production
  modules.
- Materialize retrieval text and citations during the same change.

**Selected option:**

Create three production modules:

- `retrieval_units.py` for immutable corpus indexes, coordinate expansion,
  logical-source grouping, validation, and atomic writers.
- `retrieval_policy.py` for explicit document policies, semantic-boundary
  selection, GenAI footnote handling, EU ELI removal, and entry-aware
  bibliography packing.
- `retrieval_builder.py` for controlled source construction, exclusion
  planning, unit materialization, complete ledger generation, semantic audits,
  explicit command-line inputs and outputs, and multi-output rollback.

The builder remains coordinate-only. It does not create retrieval text,
citation text, embeddings, character offsets, or indexes.

The accepted `candidate-v2` terminal `:part-NNN` identity contract is retained
for parity. It is not yet declared a public external API.

**Measured result:**

The production builder creates:

- 579 retrieval units
- 485 logical sources
- 12,008 coordinate-ledger records
- 10,021 retrieval-content coordinates
- 622 heading-context coordinates
- 94 internal boundaries
- 0 semantic-boundary risks
- 0 remaining previously rejected boundaries
- 39 omitted empty hierarchy containers
- 53 heading-only units
- 10 reference units across two reference sections
- one standard-ceiling exception: EU recital 29 at 580/640 indexed words

Unit kinds are:

- 342 heading-body
- 53 heading-only
- 181 EU-recital
- 3 frontmatter-body

The builder reproduces the accepted prototype units and ledger exactly after
normalizing the production schema version. A second explicit CLI execution
produces identical output hashes.

Accepted production SHA-256 checksums:

- `retrieval-units.jsonl`:
  `e9675edc15a8cc7651a17ad8c9134f4b9166a5fc039d679602c7db542cf2aa07`
- `retrieval-coordinate-ledger.jsonl`:
  `0b59132e7cdd6b68b667e07ad54efe762ba7b6a7572584f4c2fd94fcc8bf3a78`
- `retrieval-units-summary.json`:
  `7b4160740e682bf759f3f506c24d0e4fcd7e56bfa106e4ed09e30d671c0fdd15`
- `retrieval-units-review.txt`:
  `f683f4dd2d5a3487704ad397ec4a93d005054068990e16f0df19f87cd31dddaa`

**Why:**

Separating indexing, policy, and orchestration makes document-specific behavior
explicit and testable without preserving prototype coupling.

Coordinate-only outputs retain one authoritative source-text representation in
`pages.jsonl`, prevent premature citation or text-schema commitments, and allow
coverage validation to remain independent from word-count validation.

Fail-closed atomic publication prevents a partially written corpus from being
mistaken for a successful production build.

**Trade-offs:**

The production code intentionally contains corpus-specific anchors, expected
counts, and reviewed exception checks. A corpus update must fail closed and
undergo a new review rather than silently adapting.

The builder is larger than the temporary prototype's final serialization
layer because it owns input validation, rollback, semantic-boundary auditing,
ledger completeness, and deterministic output ordering.

Generated artifacts remain local and ignored, so users must run the builder to
inspect them.

**How we verified it:**

- 38 focused foundation and policy tests pass.
- 9 focused builder tests pass.
- The complete project suite passes with 118 tests.
- Ruff and `git diff --check` pass.
- Production units have exact semantic parity with all 579 accepted units.
- The production ledger has exact semantic parity with all 12,008 accepted
  ledger records.
- An explicit module CLI run reproduces all four accepted production hashes.
- The semantic audit checks all 94 internal boundaries and confirms zero risks.
- The builder confirms zero overlap, zero lost retained coordinates, complete
  bibliography entries, no blank or ELI coordinates in units, and exact
  document-grouped ordering.

**Date:**

2026-07-20
---

## PP-019: Separate semantic source identity from retrieval context labels

**Decision:**

Keep machine-stable semantic source keys separate from the human-readable
labels that will prefix retrieval text.

For the existing non-heading unit kinds:

- EU recital source keys remain `eu-recital-NNN`, while their retrieval labels
  are `Recital N`.
- The RMF executive-summary source key remains `rmf-executive-summary`, while
  its retrieval label is `Executive Summary`.

Heading-body and heading-only labels continue to use the reviewed reconstructed
heading text.

**Context:**

Phase 1.4e requires retrieval text to be materialized from the accepted
coordinate-only units. During the tokenizer audit, the retrieval-label helper
was compared with each unit's accepted `context_word_count`.

The audit found that all 181 EU-recital units and all three RMF
executive-summary units had inconsistent label/count contracts:

- `EU recital N` contains three whitespace-delimited words, while the accepted
  packing basis reserves two.
- `rmf-executive-summary` contains one whitespace-delimited token, while the
  accepted packing basis reserves two.

The accepted two-word budgets correspond to `Recital N` and
`Executive Summary`.

**Options considered:**

- Change the accepted context-word counts to match the existing helper labels.
- Use semantic source keys directly as retrieval labels.
- Treat the mismatch as an undocumented prototype convention.
- Preserve semantic source keys and correct only the human-readable labels.

**Selected option:**

Preserve all semantic source keys, unit IDs, coordinates, boundaries, word
budgets, and artifact schemas. Correct only `logical_source_label()` so its
human-readable output matches the accepted packing basis.

This keeps identifiers stable while making retrieval-text materialization
explicit and internally consistent.

**Measured result:**

- 579 of 579 units have consistent label/count bases.
- Semantic source keys and unit IDs are unchanged.
- All four regenerated coordinate-only artifacts retain their accepted
  SHA-256 checksums.
- Unit, logical-source, coordinate-ledger, and semantic-boundary counts are
  unchanged.

**Why:**

Identifiers optimize for stability and uniqueness. Retrieval labels optimize
for readable semantic context. Requiring one string to serve both purposes
would either expose implementation-oriented IDs to retrieval or destabilize
accepted identities when display wording changes.

Matching the label to its packing basis also prevents later tokenizer and
splitting calculations from using text that differs from the context assumed
when the unit was accepted.

**Trade-offs:**

Retrieval-text materialization must explicitly choose the retrieval label
rather than assuming that a semantic source key is display text.

Changing a retrieval label in the future may alter token counts even when
coordinates and identifiers remain unchanged, so label changes require
token-budget regression tests.

**How we verified it:**

- A failing regression test first demonstrated the old labels.
- The focused retrieval-policy suite passes with 21 tests.
- The complete project suite passes with 118 tests.
- Ruff and `git diff --check` pass.
- Temporary production regeneration reproduced all accepted hashes:
  - units:
    `e9675edc15a8cc7651a17ad8c9134f4b9166a5fc039d679602c7db542cf2aa07`
  - ledger:
    `0b59132e7cdd6b68b667e07ad54efe762ba7b6a7572584f4c2fd94fcc8bf3a78`
  - summary:
    `7b4160740e682bf759f3f506c24d0e4fcd7e56bfa106e4ed09e30d671c0fdd15`
  - report:
    `f683f4dd2d5a3487704ad397ec4a93d005054068990e16f0df19f87cd31dddaa`

**Date:**

2026-07-20
---

## PP-020: Pin an offline BERT WordPiece tokenizer contract

**Decision:**

Use `tokenizers==0.22.2` with an explicitly constructed BERT WordPiece pipeline
for retrieval tokenization and token-budget accounting.

The tokenizer contract is pinned by:

- library name and exact version,
- implementation strategy,
- normalization behavior,
- special-token identities,
- model maximum length,
- vocabulary size,
- vocabulary SHA-256,
- canonical vocabulary source,
- and canonical source revision.

The canonical vocabulary is packaged inside PolicyProof so tokenization never
requires a network download at runtime.

**Context:**

Phase 1.4e requires token budgets to be based on the tokenizer used by the
retrieval architecture rather than the provisional whitespace word count.

The reviewed candidate dense retriever and reranker are:

- `BAAI/bge-small-en-v1.5` at revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- `cross-encoder/ms-marco-MiniLM-L6-v2` at revision
  `c5ee24cb16019beea0893ab7796b1df96625c6b8`

Both declare `BertTokenizer`, a 512-position BERT contract, the same special
tokens, the same normalization behavior, and the same 30,522-entry vocabulary.

Their shared vocabulary is byte-identical to
`google-bert/bert-base-uncased` at revision
`86b5e0934494bd15c9632b12f734a8a67f723594`.

The vocabulary SHA-256 is:

`07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`

**Options considered:**

- Continue using provisional whitespace-delimited word counts.
- Use `tiktoken` with `cl100k_base`.
- Add the full `transformers` runtime dependency.
- Use `tokenizers` while downloading model assets on first use.
- Use `tokenizers` with a packaged, hash-verified canonical vocabulary.

**Selected option:**

Add `tokenizers==0.22.2` as a pinned runtime dependency and construct the
tokenizer from low-level components:

- `Tokenizer`
- `WordPiece.from_file`
- `BertNormalizer`
- `BertPreTokenizer`
- `BertProcessing`
- the WordPiece decoder

The production contract uses:

- lowercase normalization,
- BERT clean-text behavior,
- Chinese-character handling,
- model-default accent stripping,
- `[PAD]` at ID 0,
- `[UNK]` at ID 100,
- `[CLS]` at ID 101,
- `[SEP]` at ID 102,
- `[MASK]` at ID 103,
- and a model maximum length of 512 tokens.

The vocabulary, Apache-2.0 license, and source notice are packaged under
`src/policyproof/assets/`.

The selected tokenizer contract does not finalize the dense retriever or
reranker model choice. It establishes the shared tokenizer behavior needed for
Phase 1.4e token accounting.

**Measured result:**

The production tokenizer matched `transformers.BertTokenizer` exactly across
12,979 reviewed cases:

- 579 materialized candidate retrieval texts,
- 12,008 extracted source lines,
- 380 hierarchy headings,
- and 12 targeted Unicode and control-character cases.

For every case, the following were identical:

- WordPiece token strings,
- token IDs without special tokens,
- and token IDs with BERT special tokens.

The built PolicyProof wheel contains:

- `bert-base-uncased-vocab.txt`
- `bert-base-uncased-LICENSE.txt`
- `bert-base-uncased-NOTICE.txt`

The wheel metadata pins `tokenizers==0.22.2`.

**Why:**

Token-budget calculations must remain deterministic across machines, CI runs,
and future model downloads.

Packaging the exact vocabulary removes network availability, cache state, and
upstream mutable references from the tokenization path.

Using the lower-level `tokenizers` API preserves the reviewed BERT behavior
without adding the full model-loading and framework surface of
`transformers`.

Hash, vocabulary-size, UTF-8, special-token-position, package-version, and
packaged-resource checks make the tokenizer fail closed when its contract is
not satisfied.

**Trade-offs:**

The repository and wheel now include approximately 231 KB of vocabulary data
plus its license and attribution notice.

`tokenizers` introduces additional transitive packaging dependencies even
though PolicyProof does not use its network-facing utilities.

Any future retrieval model with a different tokenizer, vocabulary,
normalization policy, or sequence-length contract will require an explicit new
decision and corpus-wide token audit.

The 512-token model limit does not itself define the final passage budget.
Query reservation, pair special tokens, splitting policy, and long-unit
handling remain separate Phase 1.4e decisions.

**How we verified it:**

- The tokenizer foundation was created through a failing-first test.
- Six focused tokenizer tests pass.
- The complete project suite passes with 124 tests.
- Deprecation warnings are treated as test failures.
- Ruff and `git diff --check` pass.
- The production tokenizer has exact parity across all 12,979 reviewed cases.
- A built wheel contains all three packaged assets.
- The packaged vocabulary and license retain their accepted SHA-256 values.
- The packaged notice records the canonical source, pinned revision,
  vocabulary hash, entry count, and Apache-2.0 license.
- Wheel metadata declares `tokenizers==0.22.2`.

**Date:**

2026-07-20
---

## PP-021: Anchor reviewed compact GPT-4o appendix headings

**Decision:**

Recognize the two compact appendix headings in the GPT-4o System Card through a
document-scoped coordinate-and-text allowlist, while leaving the generic heading
classifier unchanged.

**Context:**

During retrieval token-budget auditing, the final GPT bibliography entry was
found to contain safety-evaluation material and the Figure 3 caption.

The underlying `References` source span began on page 29 and incorrectly
continued through page 33 because these source lines had not been detected as
headings:

- page 31, line 30:
  `A Violative & Disallowed Content - Full Evaluations`
- page 32, line 13:
  `B Sample tasks from METR Evaluations`

The source uses compact `A ...` and `B ...` forms rather than explicit
`Appendix A ...` and `Appendix B ...` markers.

The document also contains ordinary prose beginning with `A`, so a generic
single-letter heading rule would introduce false positives.

**Options considered:**

- Leave the accepted heading set unchanged and repair bibliography parsing.
- Treat every line beginning with one capital letter as an appendix.
- Add word-count, capitalization, or punctuation heuristics.
- Add exact document-and-coordinate anchors.
- Add exact document, coordinate, and normalized-text anchors.

**Selected option:**

Add two exact GPT-4o System Card anchors containing:

- document ID,
- page number,
- line number,
- normalized source text,
- and the `appendix` candidate type.

Apply the anchors only when the generic classifier returns no heading type.

Keep the generic classifier unchanged so ordinary prose such as
`A second concern may be whether...` remains non-heading.

Require the expected text to match the pinned coordinate. A changed source line
therefore fails closed rather than being accepted solely because it occupies a
previously reviewed location.

**Measured result:**

The regenerated heading pipeline produces:

- 349 candidates instead of 347,
- 349 reconstructed headings instead of 347,
- 382 hierarchy nodes instead of 380,
- 349 source nodes and 33 synthetic nodes,
- 382 span records instead of 380,
- and 2 new root-level GPT appendix nodes.

The only changed pre-existing span is GPT `References`:

- previous included end: page 33, line 2,
- corrected included end: page 31, line 29.

The production retrieval corpus now produces:

- 581 units,
- 487 logical sources,
- 12,008 ledger records,
- 10,019 retrieval-content coordinates,
- 624 heading-context coordinates,
- 344 heading-body units,
- 53 heading-only units,
- 181 EU-recital units,
- 3 frontmatter-body units,
- 36 GPT units,
- 94 internal boundaries,
- 0 semantic risks,
- and 0 remaining previously rejected boundaries.

Reference packing remains unchanged at five units for each of the two reference
sections. The maximum indexed-word count remains 580, with EU recital 29 as the
only approved standard-ceiling exception.

Accepted heading artifact SHA-256 checksums:

- `heading-candidates.jsonl`:
  `0842bb2e6cfaa05103918cd24579ad252203e384bab6015a3ee864218b8796d0`
- `reconstructed-headings.jsonl`:
  `fb3c6d3d9b3615e78bcca5f9e075c88bda50eed5c5ef1a3643335fcd45c5d621`
- `heading-hierarchy.jsonl`:
  `a49c97741378e0ee60c531b0eee2d3f6d66c37057b56556e8c06523e8f19b928`
- `heading-spans.jsonl`:
  `67ad7444bd77a384df85e0fdef8f3f18aba18c76646b08a661c405c216817871`

Accepted retrieval artifact SHA-256 checksums:

- `retrieval-units.jsonl`:
  `4726f293b6cea614e86c6d61bd240f4da87c5fb139169ae2b5e81faba7d658c0`
- `retrieval-coordinate-ledger.jsonl`:
  `dc599de4b2766e588adabb584912b1eb20080cca18b03403ab5bddd8b5f569e8`
- `retrieval-units-summary.json`:
  `324f146626c620917fe7178e0fd9a721104d224531894f4e2b6545676a60f2a4`
- `retrieval-units-review.txt`:
  `46167f087bd99a6a2f5a2076fd59248b226dc89d60e08c25a6bdc03fdc47ff17`

**Why:**

The root cause is a missing structural boundary, not a bibliography-packing
defect. Repairing the section boundary preserves the architecture in which
heading detection defines spans and downstream retrieval consumes those spans.

A broad single-letter rule would weaken precision across the corpus. Exact
reviewed anchors are appropriate because the production corpus is pinned,
corpus-specific invariants already fail closed, and both coordinates have been
manually verified.

**Trade-offs:**

The allowlist is intentionally corpus-specific. A revised GPT-4o System Card
layout or changed source text will require explicit review.

Exact anchors do not generalize to unknown documents. Future ingestion of
unreviewed documents will need a separate compact-heading discovery policy
rather than silently inheriting these two exceptions.

The corrected structure changes accepted downstream counts and hashes, so
retrieval artifacts, snapshots, documentation, and token-budget audits must be
regenerated together.

**How we verified it:**

- Failing-first tests demonstrated that the two headings were initially missed.
- Positive tests cover both reviewed compact appendices.
- A negative regression test preserves ordinary single-letter prose.
- The exact temporary structural comparison showed no removed records and no
  changed existing candidate, reconstructed-heading, or hierarchy records.
- Only the existing `References` span changed.
- The original seven heading artifacts were archived with verified SHA-256
  equality before regeneration.
- Snapshot values and hashes were recomputed from the regenerated artifacts.
- A path-level snapshot audit confirmed only reviewed fields changed.
- The complete production retrieval builder passed all controlled validations.
- All 12,008 source coordinates remain classified exactly once.
- All 94 internal boundaries pass with zero semantic risks.
- Temporary and accepted retrieval builds are byte-identical.
- All 130 project tests pass.
- Ruff and `git diff --check` pass.

**Date:**

2026-07-20

---

## PP-022: Derive token-safe passages without changing coordinate ownership

**Decision:**

Create a separate token-safe passage layer derived from the accepted
coordinate-only retrieval units.

Do not add character-offset ownership to the Phase 1.4d retrieval units or
coordinate ledger.

Use exact reviewed character offsets only for the 14 EU recitals that cannot
fit the 445-token passage limit using existing line-level semantic boundaries.

**Context:**

The pinned tokenizer has a 512-token model limit. Reserving 64 tokens for the
query and three pair special tokens leaves a hard passage budget of 445 tokens.

A corpus-wide audit found:

- 122 of the original 581 materialized units exceeded 512 single-sequence
  tokens
- 174 units exceeded the 445-token passage budget
- all complete bibliography entries individually fit the budget
- 14 EU recitals could not fit using only existing line-level boundaries
- the limiting existing-boundary source required 714 tokens

The blocking recitals contained valid sentence endings inside extracted source
lines. A reviewed dynamic-programming audit found one suitable intra-line
sentence boundary for each recital.

The accepted coordinate ledger requires every source coordinate to have exactly
one retrieval-content owner. Splitting a coordinate between two Phase 1.4d units
would therefore violate the existing ownership contract.

**Options considered:**

- Increase the passage limit above the model-compatible budget.
- Reduce the reserved query budget.
- Truncate long passages.
- Apply a generic sentence tokenizer to all source text.
- Add character offsets directly to the accepted retrieval units and ledger.
- Derive a separate passage layer with optional reviewed character offsets.

**Selected option:**

Keep the 581 accepted retrieval units and their coordinate ledger unchanged.

Build a derived passage schema containing:

- a deterministic passage ID
- logical-source identity
- ordered source-unit IDs
- source coordinate slices
- optional start and end character offsets
- the reviewed boundary reason
- the pinned tokenizer count

Use a minimum-piece dynamic-programming partition over approved semantic
endpoints. Minimize, in order:

1. passage count
2. reviewed intra-line boundaries
3. deviation from the 384-token packing target

Require every passage to remain at or below 445 tokens.

Reference sections may split only between complete parsed bibliography entries.

**Measured result:**

The production passage corpus contains:

- 707 passages
- 487 logical sources
- 203 EU-recital passages
- 4 frontmatter-body passages
- 447 heading-body passages
- 53 heading-only passages
- 27 reference passages
- 14 reviewed intra-line boundaries
- 0 passages over 445 tokens
- maximum passage length of 445 tokens

Document passage counts are:

- EU AI Act: 423
- NIST Generative AI Profile: 111
- NIST AI RMF: 120
- GPT-4o System Card: 53

Boundary counts are:

- 14 `after_sentence_in_line`
- 123 `after_strong_terminal`
- 25 `before_reference_entry`
- 58 `before_structured_start`
- 53 `end_of_heading_source`
- 434 `end_of_source_unit`

Accepted artifact SHA-256 checksums:

- `retrieval-passages.jsonl`:
  `918e6d30f2e1900386f5f3e9f5311042f47560c6aaca90168bfc4008f807f874`
- `retrieval-passages-summary.json`:
  `a4d0e9afc004f8da8fa93d5ff895d3ff3cf5540c65cfb0c1b182c026f985a626`
- `retrieval-passages-review.txt`:
  `9f03dbe06bb74f2d59cc05b49c0ca36758ad47398b5cfe8e4e150ead6658fa45`

**Why:**

The coordinate-only units and ledger answer which extracted lines belong to
each reviewed logical source. Token-safe passages answer how those accepted
sources should be packed for a specific model-input contract.

Separating those responsibilities preserves the already audited provenance
layer while allowing a narrow character-offset representation where the model
budget requires it.

Exact source-key, coordinate, offset, and text-context checks make all 14
exceptions fail closed if the pinned corpus changes.

**Trade-offs:**

The passage corpus contains more records than the coordinate-only corpus.

Fourteen source lines are represented by two adjacent passage slices, although
each line still has only one owner in the accepted coordinate ledger.

The 445-token budget is tied to the current 64-token query reservation and
BERT-compatible 512-token pair contract. A different query reservation,
tokenizer, or model limit requires a new corpus-wide audit and an explicit
decision.

The reviewed sentence anchors are intentionally corpus-specific and should not
be generalized to unknown documents.

**How we verified it:**

- All 14 blocking recitals received exactly one reviewed intra-line boundary.
- Every projected passage fit the 445-token limit before implementation.
- Focused passage and materialization tests pass.
- The complete project suite passes with 134 tests.
- Ruff and `git diff --check` pass before documentation finalization.
- Temporary and permanent passage builds are byte-identical.
- All 707 passage IDs are unique.
- All 487 logical sources are represented.
- Maximum passage length is 445 tokens.
- No passage exceeds the hard limit.
- All bibliography entry ordinal ranges are continuous.
- Every accepted source character is covered exactly once within its logical
  source.
- All 14 split coordinates form adjacent ranges with no gaps or overlaps.
- Passage-to-unit links and passage numbering are complete.
- The production writer was tested for non-overwrite behavior, unique output
  paths, and rollback after a simulated intermediate failure.
- The generated passage artifacts remain ignored under `data/processed/`.

**Date:**

2026-07-20
---

## PP-023: Persist retrieval and citation text on accepted passages

**Decision:**

Persist deterministic `retrieval_text` and `citation_text` fields directly on
the accepted token-safe passage records.

Use schema version `1.1` and calculate `passage_token_count` from the exact
persisted `retrieval_text`.

**Context:**

Phase 1.4e established 707 token-safe passages with complete source-slice
provenance but intentionally deferred persisted text.

The passage builder already reconstructed label-prefixed text to calculate token
counts and then discarded that string. Citation/body semantics existed only in
the earlier retrieval-unit materializer.

Leaving text unpersisted would require every downstream embedding, reranking,
evaluation, and citation consumer to rematerialize it independently from
`pages.jsonl`. That would create multiple opportunities for separator, label,
offset, or normalization behavior to diverge from the text whose token count was
accepted.

**Options considered:**

- Continue rematerializing all text at downstream runtime.
- Create a separate passage-text artifact synchronized by passage ID.
- Persist only retrieval text.
- Persist retrieval and citation text directly on each passage.
- Add per-record text hashes in addition to the whole-artifact checksum.

**Selected option:**

Extend each passage record with:

- `retrieval_text`
- `citation_text`
- `passage_token_count` derived from `retrieval_text`

For non-heading-only passages:

- `citation_text` is the materialized source-slice body.
- `retrieval_text` is `label`, two newlines, and `citation_text`.

For heading-only passages, both fields contain the reviewed reconstructed
heading because the heading is the accepted evidence itself.

Reject any input passage that already contains `retrieval_text`,
`citation_text`, or `passage_token_count` before deterministic materialization.

Do not introduce a parallel passage-text artifact or per-record hashes. The
canonical passage record and its whole-artifact SHA-256 are sufficient and
avoid maintaining synchronized duplicate records.

**Measured result:**

The schema `1.1` production corpus contains:

- 707 persisted passage records
- 903,356 retrieval-text characters
- 865,405 citation-text characters
- 53 heading-only passages with identical retrieval and citation text
- 27 complete-reference-entry passages
- 14 reviewed intra-line boundaries
- maximum passage size of 445 tokens
- zero passages above the hard token limit
- zero changed passage IDs
- zero changed ordering
- zero changed source slices
- zero changed boundary fields
- zero changed source-unit provenance links
- zero changed token counts
- zero changes to coordinate ownership or the accepted coordinate ledger

Accepted schema `1.1` artifact SHA-256 checksums:

- `retrieval-passages.jsonl`:
  `5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96`
- `retrieval-passages-summary.json`:
  `aff873450e04744f71580fdf2792b56edceceb61259b3f20ea54bc735f3c1bb9`
- `retrieval-passages-review.txt`:
  `8cd2574c360765a8eb8fdac1cae7232db97ce1cf372a7a24b89a69c9c6f29bec`

**Why:**

Persisting the exact accepted text makes the passage artifact self-contained for
the next retrieval-data stages while retaining source-slice provenance for
audit and citation verification.

Separating retrieval context from citation evidence prevents human-readable
labels from being presented as though they were extracted body text while still
allowing normalized headings to improve retrieval.

Calculating the stored token count from the persisted retrieval string ensures
that token safety and downstream model input refer to the same bytes.

**Trade-offs:**

The passage artifact is larger because it now stores two related strings.

Citation text still reflects deterministic line trimming and reviewed
same-page/cross-page separators rather than preserving original PDF layout.

The reviewed heading text may differ from raw extracted heading lines where
document-scoped normalization corrected known PDF wrapping defects. Raw source
coordinates remain authoritative provenance.

Any future change to labels, separators, normalization, tokenizer behavior, or
citation-text policy requires schema review and a full corpus regeneration.

**How we verified it:**

- Failing-first tests established distinct retrieval and citation behavior.
- Focused tests cover same-page gaps, page transitions, start and end character
  offsets, heading-only evidence, immutable input records, and rejection of
  preexisting materialized fields.
- A builder-level regression proves production records contain both text fields
  and derive token counts from persisted retrieval text.
- A temporary 707-record production build was compared field by field with the
  accepted schema `1.0` corpus after excluding only the two new fields and
  normalizing the schema version.
- No previously accepted identity, ordering, provenance, boundary, slice, or
  token-count field changed.
- The old three artifacts were hash-verified and archived before regeneration.
- The permanent schema `1.1` artifacts exactly match the audited temporary
  hashes.
- The complete project suite passes with 144 tests.
- Ruff and `git diff --check` pass.

**Date:**

2026-07-20
---

## PP-024: Bind retrieval evaluation to reviewed passage evidence

**Decision:**

Define a versioned, manually reviewed retrieval-evaluation dataset before
implementing BM25, dense retrieval, hybrid retrieval, or reranking.

Bind each dataset version to the exact corpus version, passage schema version,
and accepted passage-artifact SHA-256.

**Context:**

PolicyProof requires objective comparison of retrieval approaches using the same
questions and accepted evidence. Building a retriever first would create a risk
that benchmark questions or relevance judgments were selected after observing
which passages a particular implementation retrieved.

The product boundary also requires explicit abstention for questions that seek
legal advice, organization-specific compliance conclusions, current facts not
contained in the controlled corpus, unsupported comparisons, or high-stakes
recommendations.

**Options considered:**

- Implement BM25 first and create evaluation data afterward.
- Store only answerable questions with one exact gold passage each.
- Use logical source identifiers rather than accepted passage identifiers.
- Define a versioned benchmark with graded passage judgments and abstentions.
- Allow loosely structured evaluation records for easier manual editing.

**Selected option:**

Use retrieval-evaluation schema version `1.0` with:

- fixed dataset identity `policyproof-retrieval-evaluation`
- semantic dataset versioning
- corpus ID and corpus version binding
- passage schema version binding
- whole passage-artifact SHA-256 binding
- stable query IDs
- exact document scope
- explicit `answer` or `abstain` behavior
- non-empty evaluation tags
- graded passage relevance:
  - grade `2` for direct evidence
  - grade `1` for useful supporting context
- a required rationale for every relevance judgment

Answerable queries require at least one grade `2` judgment. Abstention queries
must contain no relevance judgments.

The validator rejects:

- unsupported schema versions
- unexpected dataset identities
- unknown fields at dataset, query, or judgment level
- empty query collections
- duplicate query IDs
- empty or duplicate evaluation tags
- empty or duplicate document-scope entries
- unknown document IDs
- unknown or duplicate passage IDs
- evidence outside the declared document scope
- invalid relevance grades
- empty judgment rationales
- answerable queries without direct evidence
- abstention queries containing gold evidence
- corpus, passage-schema, or passage-artifact mismatches

**Measured result:**

The initial dataset, version `0.1.0`, contains:

- 20 manually reviewed queries
- 16 answerable queries
- 4 required-abstention queries
- 4 answerable queries for each corpus document
- 35 passage-level relevance judgments
- 31 direct-evidence judgments at grade `2`
- 4 supporting-context judgments at grade `1`
- 11 multi-passage answerable queries
- 2 heading-only evidence cases

Accepted evaluation dataset SHA-256:

- `retrieval-evaluation-v0.1.0.json`:
  `135a29586cf86c3489038514a821198381d15eafb89345de31b4999ce1091d86`

The dataset is bound to accepted passage artifact SHA-256:

- `retrieval-passages.jsonl`:
  `5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96`

**Trade-offs:**

The initial benchmark is intentionally small and cannot represent every possible
question or relevance interpretation. Manual passage review takes more effort
than automatically generated labels, and future passage changes will require a
new dataset version or a deliberate evidence migration.

Graded relevance introduces more judgment complexity than binary relevance, but
it preserves the distinction between passages that directly answer a question
and passages that only provide useful context.

**How we will verify it:**

- Validate the committed dataset against the source manifest and all accepted
  passage records.
- Recalculate and compare the passage-artifact SHA-256 during tests.
- Require focused tests for every fail-closed schema rule.
- Keep the default test suite offline and independent of API keys.
- Use this unchanged dataset to compare BM25, dense, hybrid, and reranked
  retrieval results.

At acceptance:

- 29 focused retrieval-evaluation tests pass.
- 173 total tests pass.
- Ruff reports no violations.
- The committed dataset validates against all 707 accepted passages.

**Date:**

2026-07-20
---

## PP-025: Correct benchmark evidence through an immutable patch version

**Decision:**

Retain the published retrieval-evaluation dataset version `0.1.0` byte-for-byte
and issue corrected dataset version `0.1.1` for subsequent retrieval
measurements.

Validate every retained benchmark version in the repository while using the
latest version for current metric assertions.

**Context:**

The first corpus-wide BM25 diagnostic retrieved an unjudged passage at rank one
for query `genai-001`.

Manual review confirmed that passage
`candidate-v2:nist-ai-600-1-genai-profile:source:nist-ai-600-1-genai-profile:page-0006:line-0017:passage-008`
directly:

- defines confabulation as confidently presented erroneous or false content
- describes output that diverges from prompts or prior statements
- explains that confabulation follows from generative models predicting from
  training-data distributions

The passage therefore satisfies the benchmark's grade `2` definition for direct
evidence. It was omitted from version `0.1.0` even though two neighboring
passages from the same reviewed logical source were already judged relevant.

Other weak-query results reviewed during the same audit were either contextual,
references, neighboring legal provisions, or genuine BM25 ranking failures.
They remain unjudged.

**Options considered:**

- Edit the published `0.1.0` file in place.
- Ignore the omitted passage to avoid changing measured results.
- Add the passage only to BM25-specific evaluation logic.
- Create patch version `0.1.1` and retain `0.1.0` unchanged.
- Replace all passage judgments with logical-source-level relevance.

**Selected option:**

Create `retrieval-evaluation-v0.1.1.json` from version `0.1.0` with exactly two
semantic changes:

- update `dataset_version` from `0.1.0` to `0.1.1`
- add the reviewed `genai-001` passage as relevance grade `2`

Keep version `0.1.0` at its accepted SHA-256 and add a regression test that
fails if its bytes change.

Repository tests validate both retained dataset versions against:

- the corpus manifest
- all 707 accepted passage records
- passage schema version `1.1`
- the accepted passage-artifact SHA-256

Current benchmark balance and metric assertions use version `0.1.1`.

**Measured result:**

Version `0.1.1` contains:

- 20 queries
- 16 answerable queries
- 4 required-abstention queries
- 36 passage-level relevance judgments
- 32 direct-evidence judgments at grade `2`
- 4 supporting-context judgments at grade `1`
- 3 direct-evidence passages for `genai-001`

Dataset SHA-256 values:

- immutable `retrieval-evaluation-v0.1.0.json`:
  `135a29586cf86c3489038514a821198381d15eafb89345de31b4999ce1091d86`
- current `retrieval-evaluation-v0.1.1.json`:
  `42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c`

The correction improved every tested corpus-wide tokenizer because the newly
judged passage ranked first for all three variants. On version `0.1.1`, the
simple punctuation-splitting lexical tokenizer produced:

- mean Recall@10: `0.7760`
- MRR@10: `0.7433`
- direct-evidence hit rate@10: `0.9375`
- mean nDCG@10: `0.6555`

The same tokenizer remained stronger than the punctuation-preserving lexical
variant and the BERT WordPiece variant on Recall@10, direct-evidence hit rate,
and nDCG@10.

**Trade-offs:**

Discovering an omitted judgment while examining retrieval output creates a risk
of post-hoc benchmark tuning. That risk is controlled by:

- retaining the original dataset unchanged
- recording the exact evidence-based correction
- applying it independently of tokenizer or retriever
- comparing all variants against both dataset versions
- refusing to add other retrieved passages unless manual review establishes
  direct or supporting relevance

Keeping multiple benchmark versions adds maintenance and test cost, but it
preserves reproducibility and makes benchmark evolution auditable.

**How we will verify it:**

- Assert the exact SHA-256 of published version `0.1.0`.
- Validate all retained retrieval-evaluation files.
- Assert that current version `0.1.1` has 36 judgments with grade distribution
  `{1: 4, 2: 32}`.
- Assert that `genai-001` has three reviewed judgments.
- Use only version `0.1.1` for the production BM25 baseline.
- Continue separating retrieval ranking metrics from later evidence-sufficiency
  and abstention evaluation.

At acceptance:

- 30 focused retrieval-evaluation tests pass.
- 174 total tests pass.
- Ruff reports no violations.
- Published version `0.1.0` retains its accepted SHA-256.
- Both retained datasets validate against all 707 accepted passages.

**Date:**

2026-07-20

---

## PP-026: Establish a deterministic corpus-wide BM25 baseline

**Decision:**

Implement the first production retrieval baseline as inspectable plain-Python
BM25 over all 707 accepted passages, evaluate it only against answerable queries
in benchmark `v0.1.1`, and publish a versioned non-overwriting result artifact.

**Context:**

PolicyProof needs a reproducible lexical baseline before dense retrieval,
hybrid retrieval, reranking, generation, API, or UI work can be evaluated.
Benchmark `document_scope` cannot be used as a production candidate filter
because it is gold metadata unavailable for ordinary user queries.

The earlier tokenizer diagnostic selected simple punctuation-splitting lexical
terms, but its query-term deduplication and ranking behavior had not yet been
promoted into an explicit tested production contract.

**Options considered:**

- Add `rank_bm25` or another retrieval dependency.
- Reuse the BERT WordPiece token-budget tokenizer for BM25.
- Restrict candidates with benchmark document scope.
- Tune BM25 parameters on the 20-query benchmark.
- Implement plain-Python corpus-wide BM25 with a fixed lexical contract.

**Selected option:**

Use plain Python with:

- Unicode NFKC normalization
- lowercase text
- terms matching `[a-z0-9]+`
- punctuation, apostrophes, and hyphens as term boundaries
- query terms deduplicated in first-seen order
- ordinary passage term frequency
- `k1 = 1.2`
- `b = 0.75`
- IDF `log(1 + (N - df + 0.5) / (df + 0.5))`
- descending score followed by accepted passage order and passage ID
- all 707 passages as production candidates

Query-term frequency is intentionally ignored. Repeating a term in a natural-
language question should not silently increase its importance, and preserving
the exploratory behavior avoids an unreviewed contract change.

Evaluate Recall@1, Recall@3, Recall@5, Recall@10, MRR@10,
direct-evidence hit rate@10, and graded nDCG@10 across only the 16 answerable
queries. Keep the four abstention cases for a later evidence-sufficiency phase.

**Measured result:**

The accepted corpus-wide result contains:

- 707 candidate passages
- 16 answerable queries
- 4 excluded abstention queries
- mean Recall@1: `0.3125000000`
- mean Recall@3: `0.6041666667`
- mean Recall@5: `0.6822916667`
- mean Recall@10: `0.7760416667`
- MRR@10: `0.7433035714`
- direct-evidence hit rate@10: `0.9375000000`
- mean nDCG@10: `0.6555156356`

The production implementation reproduces all four previously recorded rounded
exploratory metrics.

Accepted result artifact:

- `data/results/bm25-baseline-v0.1.0.json`
- SHA-256:
  `5609b146b0901fc84851789d3b6c2799ec6aad0545e33b9c80afaa29c9d80003`

The result binds benchmark `v0.1.1`, its SHA-256, passage schema `1.1`, the
accepted passage SHA-256, the corpus identity and version, tokenizer behavior,
actual BM25 parameters, candidate scope, and deterministic tie-breaking.

A weak-query audit found six answerable queries below perfect Recall@10.
`eu-002` is the only complete top-10 miss and remains weak even under a
diagnostic EU-only ranking, identifying a within-document lexical limitation.
No new benchmark correction was established.

**Trade-offs:**

The simple lexical tokenizer is transparent and dependency-free but cannot
reliably connect paraphrases, legal references, and semantically related terms.
Multi-passage questions can have strong first-hit performance while retaining
incomplete Recall@10.

Ignoring query-term frequency simplifies the contract but prevents deliberate
repeated-term weighting. Fixed parameters provide a fair baseline rather than
benchmark-optimized performance.

Persisting top-10 scores and accepted-order positions enlarges the result
artifact, but makes ranking and tie-breaking auditable.

**How we will verify it:**

- Unit-test normalization, punctuation splitting, BM25 scoring, query-term
  deduplication, tie-breaking, and fail-closed inputs.
- Unit-test Recall, reciprocal rank, direct-evidence hit rate, and graded nDCG.
- Validate the benchmark before evaluation.
- Assert exact repository input hashes and aggregate metrics.
- Refuse result overwrite and publish through an atomic writer.
- Assert the exact accepted result SHA-256.
- Regenerate the result byte-for-byte in repository tests.
- Keep document-scoped retrieval diagnostic-only.
- Preserve all published benchmark versions unchanged.

At acceptance:

- 30 focused BM25 tests cover scoring, metrics, evaluation, serialization,
  orchestration, CLI behavior, and repository result reproducibility.
- The complete offline project suite passes with 204 tests.
- The accepted result regenerates byte-for-byte with SHA-256
  `5609b146b0901fc84851789d3b6c2799ec6aad0545e33b9c80afaa29c9d80003`.

Ruff reports no violations, final diff checks pass, and staged manual review
found no blocking issues.

**Date:**

2026-07-21
---

## PP-027: Establish a pinned corpus-wide dense retrieval baseline

**Decision:**

Implement the first production dense baseline with the pinned
`BAAI/bge-small-en-v1.5` ONNX model over all 707 accepted passages.

Use direct CPU inference through `onnxruntime==1.27.0` and
`numpy==2.5.1`, rather than adding PyTorch, Transformers, or
Sentence-Transformers.

Require an explicit local ONNX model path. Validate the model file by exact
size and SHA-256 before session creation, apply the model's retrieval
instruction only to queries, use CLS pooling, L2-normalize all embeddings, and
rank with normalized dot product.

Evaluate only answerable queries from benchmark `v0.1.1`, while recording all
four abstention cases outside ranking metrics. Publish a versioned,
non-overwriting result artifact.

**Context:**

The accepted BM25 baseline established a reproducible lexical reference, but it
cannot reliably connect paraphrases, related legal concepts, and semantically
equivalent wording.

The reviewed dense-model candidate from PP-020 is:

- model: `BAAI/bge-small-en-v1.5`
- revision:
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- ONNX file: `onnx/model.onnx`
- size: `133093490` bytes
- SHA-256:
  `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35`
- embedding dimension: `384`
- maximum sequence length: `512`
- license: MIT

A binary-only dependency audit found:

- direct ONNX Runtime: 5 packages
- PyTorch plus Transformers: 34 packages
- Sentence-Transformers: 40 packages

The direct ONNX route can consume the already pinned PolicyProof tokenizer
without introducing model-loading frameworks, runtime network access, or
additional tokenization behavior.

**Options considered:**

- Use Sentence-Transformers and its full dependency stack.
- Use PyTorch, Transformers, and Safetensors directly.
- Use ONNX Runtime with automatic model downloading.
- Commit the approximately 133 MB model binary to the repository.
- Use direct ONNX Runtime with an explicit local, hash-verified model asset.
- Restrict candidates using benchmark `document_scope`.
- Tune the embedding model or query instruction on the 20-query benchmark.

**Selected option:**

Use direct ONNX Runtime with:

- `onnxruntime==1.27.0`
- `numpy==2.5.1`
- `CPUExecutionProvider` only
- the exact three-input ONNX interface:
  - `input_ids`
  - `attention_mask`
  - `token_type_ids`
- output `last_hidden_state`
- CLS pooling from token position zero
- 384-dimensional float32 embeddings
- L2 normalization
- normalized dot-product similarity
- query instruction:
  `Represent this sentence for searching relevant passages: `
- no passage instruction
- no truncation
- bounded passage batches, with accepted baseline batch size `32`
- full-corpus candidate scope
- deterministic tie-breaking by:
  1. score descending
  2. accepted passage order
  3. passage ID

The model binary is not committed. Production execution requires an explicit
local path to the exact pinned asset. Default tests remain offline and use
controlled fake sessions or the committed result artifact.

**Measured result:**

The accepted dense baseline searched all 707 passages and evaluated:

- 16 answerable queries
- 4 abstention queries excluded from ranking metrics

Aggregate metrics:

- mean Recall@1: `0.4583333333333333`
- mean Recall@3: `0.8802083333333334`
- mean Recall@5: `0.8802083333333334`
- mean Recall@10: `0.96875`
- MRR@10: `0.90625`
- direct-evidence hit rate@10: `1.0`
- mean nDCG@10: `0.8866302400292934`

Compared with the accepted BM25 baseline, dense retrieval improved:

- mean Recall@10 by `0.1927083333333334`
- MRR@10 by `0.1629464285714286`
- direct-evidence hit rate@10 by `0.0625`
- mean nDCG@10 by `0.2311146043908528`

Fourteen of sixteen answerable queries achieved complete Recall@10. The two
partial-recall queries were:

- `rmf-002`: Recall@10 `0.75`
  - all three grade-2 passages were retrieved
  - one grade-1 GOVERN 1.3 supporting passage was outside the top ten
- `eu-003`: Recall@10 `0.75`
  - three of four grade-2 Article 26 passage segments were retrieved
  - the specialized post-remote-biometric-identification segment was outside
    the top ten

All sixteen answerable queries retrieved direct evidence within the top ten.
No benchmark correction was established by the miss audit.

Accepted result artifact:

- `data/results/dense-baseline-v0.1.0.json`
- size: `68170` bytes
- SHA-256:
  `fd6477ba09c8d9a4a3d36eeeaa2455882a90fd26fb0a9f82f3363c16189f6c5d`

A second complete execution using the same model, corpus, benchmark, and batch
size regenerated the artifact byte-for-byte with the same SHA-256.

**Why:**

Direct ONNX inference keeps the production dependency and implementation
surface small while preserving the exact reviewed model behavior.

Hash and size validation make the external model asset fail closed. Requiring
an explicit path prevents ordinary library imports, tests, or CLI discovery
from silently downloading a model.

Using the accepted persisted `retrieval_text` avoids downstream
rematerialization differences. Applying the instruction only to queries follows
the model's retrieval contract, while CLS pooling and L2 normalization reproduce
the reviewed model configuration.

Full-corpus ranking preserves realistic production behavior and prevents gold
benchmark metadata from influencing candidates.

**Trade-offs:**

The 133 MB model remains an external local prerequisite and must be obtained
separately before a real dense-baseline execution.

CPU inference is slower than BM25. The accepted run took approximately 57
seconds on the reviewed Apple Silicon development machine.

Exact runtime pins improve reproducibility but require deliberate dependency
updates for new Python, NumPy, or ONNX Runtime releases.

The current implementation rebuilds passage embeddings for each complete
baseline run. Persisted embedding or vector-index artifacts remain a later
decision and must bind to the exact passage, tokenizer, and model contracts.

The initial benchmark is small. Strong results establish a baseline but do not
justify model tuning or broad claims about unseen queries.

**How we verified it:**

- A binary-only dependency audit confirmed Python 3.12 Apple Silicon wheels.
- The exact ONNX asset size and SHA-256 were measured.
- The ONNX graph exposes the expected three inputs and one
  384-dimensional hidden-state output.
- Repeated CPU inference was byte-identical.
- Embeddings remained stable across batching, padding, and input order.
- All 707 passages fit the accepted 445-token budget.
- All instructed benchmark queries fit the 64-token reservation.
- Failing-first tests cover:
  - model and runtime contracts
  - asset integrity
  - session interfaces
  - input preparation
  - CLS pooling and normalization
  - bounded passage batching
  - similarity ranking and tie-breaking
  - benchmark evaluation
  - result serialization
  - orchestration
  - immutable repository-result bindings
- The result regenerated byte-for-byte.
- The two partial-recall cases were manually audited.
- The complete offline project suite passes with 291 tests.
- Ruff, dependency checks, and `git diff --check` pass.

**Date:**

2026-07-22

---

## PP-028: Use hybrid retrieval for candidate generation, not final ranking

**Decision:**

Use BM25 and dense retrieval to generate a deterministic candidate union for
later cross-encoder reranking.

Retrieve the top 20 passages from each full-corpus retriever, deduplicate by
passage ID, preserve BM25 and dense source ranks, and serialize candidates in
accepted passage order followed by passage ID.

Do not assign a fused score or final relevance rank at this stage.

**Context:**

The accepted dense baseline substantially outperformed BM25 but retained two
partial Recall@10 cases:

- `rmf-002`: one grade-`1` supporting passage outside dense top ten
- `eu-003`: one grade-`2` specialized Article 26 passage outside dense top ten

BM25 contains complementary evidence for both cases. However, equal-weight
reciprocal-rank fusion with constant `60` degraded the overall final ranking.

Full-corpus RRF measured:

- mean Recall@10: `0.880208333333`
- MRR@10: `0.8125`
- direct-evidence hit rate@10: `1.0`
- mean nDCG@10: `0.744151796205`

The accepted dense baseline remains stronger as a final ranking:

- mean Recall@10: `0.96875`
- MRR@10: `0.90625`
- direct-evidence hit rate@10: `1.0`
- mean nDCG@10: `0.8866302400292934`

**Options considered:**

- Use equal-weight RRF over the published top-ten lists.
- Use equal-weight RRF over complete corpus rankings.
- Tune RRF constants or retriever weights on the 20-query benchmark.
- Normalize and combine incompatible BM25 and cosine-similarity scores.
- Keep dense retrieval as the only candidate generator.
- Form a deduplicated BM25+dense union and defer ordering to a cross-encoder.

**Selected option:**

Use a candidate-only union with:

- BM25 input depth `20`
- dense input depth `20`
- full-corpus source rankings
- deduplication by passage ID
- source-rank provenance on every candidate
- no fused score
- no final rank
- deterministic serialization by accepted passage order and passage ID
- dense passage batch size `32`
- answerable-query coverage metrics only
- abstention queries recorded but excluded from candidate metrics

Candidate-depth diagnostics measured:

- depth `5`: mean recall `0.9166666667`, mean union size `7.5625`
- depth `10`: mean recall `0.984375`, mean union size `15.4375`
- depth `20`: mean recall `1.0`, mean union size `31.3125`
- depth `30`: mean recall `1.0`, mean union size `46.3125`
- depth `50`: mean recall `1.0`, mean union size `77.625`
- depth `100`: mean recall `1.0`, mean union size `155.0`

Depth 20 is the smallest tested depth with complete reviewed-passage coverage.
This depth was selected after inspecting the fixed benchmark. The accepted
coverage measurement is therefore benchmark-informed and must not be described
as untuned or out-of-sample performance.

**Measured result:**

The accepted hybrid candidate baseline evaluates:

- 707 accepted passages
- 16 answerable queries
- 4 abstention queries excluded from candidate metrics
- top 20 BM25 candidates per query
- top 20 dense candidates per query

Aggregate candidate coverage:

- mean candidate recall: `1.0`
- direct-evidence hit rate: `1.0`
- mean candidate count: `31.3125`

Every reviewed grade-`1` and grade-`2` passage is present in the candidate union
for every answerable query.

Accepted result artifact:

- `data/results/hybrid-candidate-baseline-v0.1.0.json`
- size: `195077` bytes
- SHA-256:
  `94b98eda3795280ef31aa0dfaa49a44d912c23d77e50a75f33a4f2f26e1fe0d4`

The artifact contains no benchmark `document_scope`, evaluation tags, relevance
judgments, fused scores, or final ranks.

**Why:**

Candidate union preserves complementary lexical and semantic evidence without
forcing incomparable retriever signals into a premature ranking rule.

The approach also creates a bounded reranker workload. At accepted depth 20,
the union averages approximately 31 candidates rather than requiring the
cross-encoder to score all 707 passages.

Preserving source ranks makes candidate origin auditable while keeping the
cross-encoder evaluation independent of a hidden fused-score transformation.

**Trade-offs:**

Candidate depth 20 is benchmark-informed. Additional evaluation queries are
needed before treating that depth as generally sufficient.

The union is larger than the dense top ten, increasing cross-encoder inference
cost. It is intentionally not suitable for direct answer generation because
accepted passage order is not relevance order.

The current runner rebuilds dense embeddings for each complete execution.
Persisted embeddings remain a separate future decision.

**How we verified it:**

- Failing-first tests cover union construction, validation, deduplication,
  source-rank provenance, candidate coverage, serialization, orchestration,
  atomic publication, and immutable repository bindings.
- Equal-weight RRF was evaluated at both top-ten and full-corpus source depths.
- Candidate union was measured at depths `5`, `10`, `20`, `30`, `50`, and
  `100`.
- The accepted real-model run uses the exact pinned dense model and runtime
  contracts.
- The committed artifact is locked by exact size and SHA-256.
- The complete offline project suite passes with 351 tests.
- Ruff, dependency checks, and `git diff --check` pass.

**Date:**

2026-07-22
