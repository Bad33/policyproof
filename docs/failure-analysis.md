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
