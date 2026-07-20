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
