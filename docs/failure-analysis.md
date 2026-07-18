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
- Preserves line-ending hyphens without inserting additional whitespace.
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
