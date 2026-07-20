# Failure Analysis

## PDF text extraction baseline

### Scope

Four controlled public AI-governance documents were extracted into page-level
records before section detection or chunking.

### Rejected baseline

The first baseline used pypdf 6.14.2 in layout mode for every document.

It extracted all 289 pages and reported no empty pages, but manual inspection
found severe text-quality problems:

- Words were frequently concatenated in the NIST AI RMF.
- Large sections of the GPT-4o System Card lost spaces between words.
- Words in the EU AI Act were split internally.
- NIST Generative AI Profile table rows contained irregular spacing and broken
  action identifiers.
- Layout-position padding substantially inflated character counts.

The rejected baseline produced 2,038,929 characters. This total was not treated
as evidence of better extraction because much of the additional content was
layout whitespace or corrupted spacing.

### Controlled comparison

Representative pages were compared using:

- pypdf layout mode
- pypdf plain mode
- pdfplumber default extraction

The same pages and expected phrases were used for each comparison.

Results showed:

- pypdf plain was preferred for the NIST AI RMF.
- pypdf plain was preferred for the NIST Generative AI Profile.
- pypdf plain was preferred for the GPT-4o System Card.
- pdfplumber default was preferred for the EU AI Act.

No single tested extractor produced the best result for all four documents.

### Accepted baseline

The accepted `document_specific_v1` policy uses:

- pdfplumber 0.11.10 for the EU AI Act
- pypdf 6.14.2 plain mode for the other three documents

Measured accepted output:

- Documents: 4
- Pages: 289
- Extracted characters: 941,574
- Empty pages: 0
- Targeted representative phrase checks passed: 4 of 4
- Automated project tests passed: 21

### Remaining limitations

- Some pypdf pages still contain occasional concatenated words.
- PDF tables are flattened into text and may not preserve logical row structure.
- Reading order may still fail on complex multi-column pages.
- Headers, footers, footnotes, and captions have not yet been removed.
- No OCR is used because the initial documents contain extractable digital text.
- Representative checks do not prove that every page is error-free.

These limitations must be considered during section detection, chunking,
retrieval evaluation, and citation verification.

## Heading detection false positives

### Initial result

The first document-specific heading detector produced 397 candidates.

Manual inspection found three recurring false-positive classes:

- Numbered entries on table-of-contents pages
- Numbered GAI risk definitions mistaken for section headings
- Numbered prose or design principles mistaken for headings

The EU AI Act also produced a duplicate Article 49 candidate. The second
occurrence was not an article heading; it was a wrapped reference inside the
title of Annex VIII.

### Correction

The detector was refined to:

- Skip pages explicitly identified as contents pages
- Reject unusually long numbered prose candidates
- Reject numbered candidates whose text continues as ordinary prose
- Reject standalone Article references appearing after an Annex heading on the
  same EU AI Act page

### Accepted result

The refined detector produced 347 candidates while retaining the expected anchor
headings in all four documents.

EU legal-structure validation confirmed:

- 113 article headings
- Complete range from Article 1 through Article 113
- No missing article numbers
- No duplicate article numbers

### Remaining limitations

- Heading titles that wrap across lines are not yet merged.
- Table-contained AI RMF subcategories require hierarchy-aware reconstruction.
- Candidate type does not yet represent a completed section hierarchy.
- Page furniture removal remains conservative.

## Heading reconstruction boundary failures

### Rejected initial reconstruction

The first reconstruction policy generated 347 headings, but manual audits found
three recurring issues:

- EU article and annex headings could absorb body paragraphs.
- A four-line NIST continuation limit truncated valid RMF function statements.
- Joining extracted lines with ordinary spaces produced artifacts such as
  `or- ganizational`.

A later audit also found numbered prose items in NIST AI RMF Appendix D being
treated as headings while the split `Appendix D:` structural marker was missed.

### Correction

The accepted reconstruction policy:

- Limits EU chapter, section, and article titles to one following line.
- Applies conservative multi-line title handling only to EU annexes.
- Extends NIST function reconstruction until punctuation or a structural stop.
- Uses a twelve-line safety limit rather than the rejected four-line limit.
- Preserves exact raw source lines while applying reviewed NIST RMF display-text
  normalization for visual wrap hyphens.
- Detects split NIST appendix markers and joins them to one following title line.
- Rejects numbered attribute-list items on the Appendix D page.

### Accepted result

The final candidate and reconstruction sets each contain 347 records.

Quality validation confirmed:

- Appendix D appears once as a structural heading.
- No numbered Appendix D attributes remain as candidates.
- No late-document NIST numbered-list candidates remain.
- No hyphen-space artifacts remain.
- No action identifiers were appended.
- No NIST heading reached the continuation safety limit.
- No RMF function statement has a suspicious grammatical ending.

### Remaining limitation

This phase reconstructs complete labels but does not yet infer hierarchy,
section spans, or cross-page section boundaries.

## EU section parent assumption failed for Annex XI

### Failure

The first EU hierarchy policy required every section to have an active chapter.

Generation failed at:

`eu-ai-act-2024-1689:page-0141:line-0006`

The source sequence showed that Annex XI contains two internal sections and no
active chapter.

### Root cause

The initial implementation modeled only the dominant legal structure:

`chapter → section → article`

It did not account for annex-local subdivisions.

### Correction

The hierarchy builder now tracks both an active chapter and an active annex.

- A section attaches to the active chapter when one exists.
- Otherwise, it attaches to the active annex.
- Starting a chapter clears annex context.
- Starting an annex clears chapter and section context.
- Consecutive sections under an annex remain siblings.

### Validation

Annex XI now contains exactly two source section children:

- Section 1 — Information to be provided by all providers of general-purpose
  AI models
- Section 2 — Additional information to be provided by providers of
  general-purpose AI models with systemic risk

Both sections have depth 2 and the same Annex XI parent.
## NIST heading display-text normalization failure

### Failure

The first accepted heading-reconstruction baseline retained extracted
line-ending hyphens in reconstructed display text.

This avoided inserting spaces into forms such as `or- ganizational`, but it
produced headings containing visual PDF wrap artifacts such as:

- `inte-grated`
- `or-ganizational`
- `decom-missioning`
- `docu-mented`
- `transfer-ring`

A separate NIST AI RMF extraction defect produced `theMAP`.

### Root cause

The generic wrapped-line joiner treated character preservation as more important
than reconstructed semantic text. That was a safe provenance policy, but the
same reconstructed heading was also used as the indexed content for 53
heading-only evidence units.

Raw provenance and retrieval/display text had been treated as though they
required the same normalization policy.

### Correction

The reconstruction pipeline now separates those responsibilities:

- Raw page text and stored `source_lines` remain unchanged.
- NIST function-heading display text removes reviewed visual wrap hyphens.
- `context-specific` and `context-relevant` remain hyphenated.
- The exact extraction glue `theMAP` becomes `the MAP`.
- Other document types retain their existing reconstruction rules.
- No dictionary or probabilistic text correction is used.

### Validation

A full temporary downstream regeneration confirmed:

- 347 reconstructed headings
- 40 headings with corrected display text
- 51 removed wrap hyphens
- 2 preserved legitimate compounds
- 1 corrected `theMAP` occurrence
- 0 changes to raw source-line provenance
- 0 changes to hierarchy structure
- 0 changes to direct-body or subtree coordinates
- a byte-identical `heading-spans.jsonl`
- 53 heading-only units with normalized word-count agreement
- 71 passing project tests
- passing Ruff checks

### Downstream impact

The corrected coordinate-only retrieval candidate still contains 579 units.
All 10,021 retrieval coordinates remain owned exactly once.

The MEASURE 2.6 heading-only unit increases to 69 normalized content words
because `theMAP` becomes two words. This does not violate the provisional
512-word standard ceiling.

Citation verification must continue to use raw source coordinates and extracted
source lines. Normalized heading text is an indexing and display representation,
not a replacement for source provenance.

## Production retrieval-unit output ordering mismatch

### Failure

The first production builder generated the correct 579 units and correct
coordinate ownership, but semantic parity failed at record 4.

The implementation returned logical-source construction order:

1. RMF executive-summary frontmatter
2. EU recitals
3. source-heading units

The accepted artifact used deterministic document-grouped order.

### Root cause

`document_unit_order` was assigned while units were being constructed.
Construction order is an implementation detail and does not match the output
contract established by the accepted prototype.

The builder had preserved unit content, identity, boundaries, and ownership but
had not reproduced final corpus serialization order.

### Correction

The builder now completes all unit construction first, then:

1. iterates through the validated corpus document order,
2. collects units for each document without reordering them internally,
3. assigns `document_unit_order` within that document, and
4. publishes the resulting document-grouped sequence.

### Validation

After the correction:

- all 579 production units have exact semantic parity with the accepted units,
- all 12,008 ledger records have exact semantic parity with the accepted
  ledger,
- explicit CLI and in-memory builds produce identical hashes,
- all 118 project tests pass,
- Ruff and `git diff --check` pass.

### Lesson

Deterministic content is insufficient when artifact ordering is part of the
reviewed contract. Production parity checks must compare ordered records, not
only sets of unit IDs or coordinate ownership.
