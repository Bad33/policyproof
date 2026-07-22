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
## Retrieval context-label word-count mismatch

### Failure

While defining retrieval-text materialization, an audit compared each unit's
human-readable label with the `context_word_count` used by the accepted packing
policy.

The audit found 184 mismatches:

- 181 EU-recital units used labels such as `EU recital 7`, which contain three
  whitespace-delimited words, but stored a two-word context budget.
- Three RMF executive-summary units used the machine identifier
  `rmf-executive-summary`, which counts as one whitespace-delimited token, but
  stored a two-word context budget.

Heading-body and heading-only units were consistent.

### Root cause

`logical_source_label()` served several prototype purposes:

- identifying reference sections,
- providing review labels,
- and acting as a candidate retrieval-text prefix.

For heading units, the semantic source key resolves to reviewed heading text.
For recitals and frontmatter, however, semantic identity and display text were
not separated consistently.

The production builder explicitly assigned the intended two-word packing
budgets but did not validate label length for non-reference units. The existing
validation applied only to bibliography packing, so the mismatch passed the
production audits.

### Correction

The semantic identifiers remain unchanged:

- `eu-recital-NNN`
- `rmf-executive-summary`

Only the human-readable retrieval labels changed:

- `EU recital N` became `Recital N`.
- `rmf-executive-summary` became `Executive Summary`.

A regression test now distinguishes semantic source keys from retrieval labels.

### Validation

After the correction:

- all 579 units have label/count agreement,
- all 118 project tests pass,
- Ruff and `git diff --check` pass,
- all 579 units and 12,008 ledger records remain unchanged,
- all four regenerated production artifacts match their accepted hashes,
- the maximum indexed-word count remains 580,
- semantic-boundary risk count remains zero.

### Lesson

Packing metadata must be validated against the exact text that will eventually
be indexed. Stable identity strings, display labels, retrieval prefixes, and
citation text are related but distinct representations and should not be
silently substituted for one another.
## Deprecated tokenizer wrapper file-loading path

### Failure

The first production tokenizer implementation used
`BertWordPieceTokenizer` with the packaged vocabulary path.

Its focused tests passed, but `tokenizers==0.22.2` emitted a
`DeprecationWarning` stating that the `WordPiece` constructor would no longer
create a model directly from files and that `WordPiece.from_file` should be
used instead.

### Root cause

`BertWordPieceTokenizer` is a convenience wrapper whose constructor still
creates its internal model through the deprecated
`WordPiece(vocab_path, ...)` path.

The wrapper reproduced the intended BERT behavior, but accepting its warning
would leave PolicyProof dependent on an API path that is already scheduled for
removal.

Suppressing the warning would hide a known compatibility defect rather than
correct it.

### Correction

Replace the convenience wrapper with an explicit low-level pipeline:

- load the packaged vocabulary through `WordPiece.from_file`,
- create a `Tokenizer`,
- register the five BERT special tokens,
- apply `BertNormalizer`,
- apply `BertPreTokenizer`,
- apply `BertProcessing` for `[CLS]` and `[SEP]`,
- and configure the WordPiece decoder.

The tokenizer contract now names this implementation explicitly as
`Tokenizer+WordPiece.from_file`.

### Validation

After the correction:

- focused tests pass with `DeprecationWarning` promoted to an error,
- all 124 project tests pass,
- Ruff and `git diff --check` pass,
- all 12,979 reviewed corpus and Unicode cases exactly match
  `transformers.BertTokenizer`,
- token IDs with and without BERT special tokens are identical,
- and the built wheel contains the vocabulary, license, and source notice.

### Lesson

A behaviorally correct wrapper is not sufficient when it relies on a
deprecated construction path.

Warnings from pinned dependencies should be evaluated as compatibility signals,
not automatically suppressed. When the lower-level public API can reproduce
the reviewed behavior exactly, it provides a clearer and more durable
production contract.

## GPT-4o compact appendix headings absorbed into References

### Failure

The accepted GPT-4o System Card heading set omitted two compact appendix
headings:

- page 31, line 30:
  `A Violative & Disallowed Content - Full Evaluations`
- page 32, line 13:
  `B Sample tasks from METR Evaluations`

Because neither line was recognized as a heading, the `References` span that
began on page 29 continued through the end of the document.

The final parsed reference entry therefore absorbed Appendix A, Appendix B,
evaluation tables, and the Figure 3 caption.

### Root cause

The generic detector recognized explicit forms such as `Appendix A`, but the
GPT-4o System Card used compact single-letter appendix headings.

A generic pattern for lines beginning with a single capital letter was unsafe.
The same document also contains ordinary prose such as:

`A second concern may be whether...`

Word-count or capitalization heuristics could not reliably distinguish that
sentence from the two appendix headings.

The bibliography parser was not the source of the defect. It correctly followed
the incorrect upstream `References` span.

### Correction

The heading detector now contains a document-scoped, coordinate-and-text
allowlist for the two reviewed compact appendix headings.

The rule requires all of the following to match:

- the GPT-4o System Card document ID,
- the exact page number,
- the exact line number,
- and the normalized expected text.

Both candidates are classified as `appendix` headings. The general heading
classifier remains unchanged, and ordinary single-letter prose remains
unclassified.

Requiring exact text as well as coordinates makes future source changes fail
closed instead of silently accepting a stale anchor.

### Validation

The exact structural delta is limited to:

- 2 added heading candidates,
- 2 added reconstructed headings,
- 2 added root-level hierarchy source nodes,
- 2 added span records,
- and 1 changed existing span: `References`.

The corrected `References` direct-body and subtree spans now end at page 31,
line 29 instead of page 33, line 2.

The regenerated heading artifacts contain:

- 349 candidates,
- 349 reconstructed headings,
- 382 hierarchy nodes,
- 382 span records,
- 349 source nodes,
- and 33 synthetic nodes.

The rebuilt production retrieval corpus contains:

- 581 retrieval units,
- 487 logical sources,
- 12,008 coordinate-ledger records,
- 10,019 retrieval-content coordinates,
- 624 heading-context coordinates,
- 94 reviewed internal boundaries,
- and 0 semantic-boundary risks.

The corrected GPT sources contain:

- 5 `References` units,
- 1 Appendix A unit,
- and 1 Appendix B unit.

Coordinate ownership confirms:

- both appendix heading lines are `heading_context`,
- Appendix A body content belongs to Appendix A,
- the Figure 3 caption belongs to Appendix B,
- and GPT page-number lines remain excluded page furniture.

All 130 project tests pass. Ruff and `git diff --check` pass.

### Lesson

A bibliography parser cannot compensate for a missing upstream section
boundary.

Compact headings that are visually clear to a human may be indistinguishable
from prose under a generic text pattern. When the corpus is pinned and the
ambiguous headings are known, exact document, coordinate, and text anchors are
safer than broad heuristics.

Downstream token-budget and retrieval audits must be rerun whenever section
boundaries change, even when the raw extracted text is unchanged.


## Line-only retrieval boundaries exceeded the pinned token budget

### Failure

The accepted coordinate-only retrieval units were semantically valid but were
not all compatible with the pinned 512-token model-input contract.

With 64 tokens reserved for a query and three tokens reserved for BERT pair
formatting, passages must contain no more than 445 tokens.

The initial token audit found:

- 174 accepted units above 445 passage tokens
- 14 EU recitals that could not be split below 445 tokens using only existing
  line-level boundaries
- a minimum required cap of 714 tokens if only the existing boundaries were
  allowed

Increasing the cap to 461 or 477 did not solve the structural problem while
remaining compatible with the intended query reservation.

### Root cause

The extracted corpus stores source coordinates at line granularity.

Several EU recital lines contain multiple complete sentences. The retrieval
packing policy recognized safe boundaries only after an entire extracted line,
so valid sentence endings inside those lines were unavailable to the packer.

The coordinate-only ledger also requires each source line to have exactly one
retrieval-content owner. Reusing the same coordinate in two accepted retrieval
units would create an ownership overlap.

This was therefore not a tokenizer defect, a section-boundary defect, or a
bibliography-entry defect. It was a mismatch between line-granular provenance
ownership and model-compatible passage packing.

### Correction

Keep the accepted coordinate-only units and ledger unchanged.

Add a separate derived passage layer with optional character offsets inside
source slices.

A controlled audit selected exactly one sentence boundary for each of the 14
blocking recitals. Each boundary is anchored by:

- logical source key
- page and line coordinate
- character offset
- expected left-side source text
- expected right-side source text

The passage builder validates every anchor against the pinned source before
using it.

All other boundaries continue to use the previously reviewed line-level
semantic policy. Bibliography passages split only between complete entries.

### Validation

The corrected passage build produces:

- 707 passages from all 487 logical sources
- 14 reviewed intra-line boundaries
- 27 complete-reference-entry passages
- maximum passage length of 445 tokens
- 0 passages over the hard limit

A character-level provenance audit confirmed:

- all accepted source coordinates remain represented
- every accepted source character is covered exactly once within its logical
  source
- each reviewed split line forms two adjacent ranges
- no character gap or overlap exists
- no unexpected character split exists
- all passage-to-unit links are complete
- all passage numbers and passage counts are consistent
- all reference-entry ordinal ranges are continuous

The generated artifacts are reproducible: temporary and permanent builds have
identical SHA-256 checksums.

The complete project suite passes with 134 tests, and the focused passage and
materialization suite passes with 10 tests.

### Lesson

A semantically correct coordinate unit is not automatically a valid model
passage.

Provenance ownership and model packing should be separate layers when their
granularity requirements differ.

Character offsets should be introduced narrowly, only after proving that
line-level boundaries are insufficient, and should be protected by exact
source-context validation rather than a broad unreviewed sentence splitter.

## Token-safe passage artifacts omitted their accepted text

### Failure

The Phase 1.4e passage artifact stored source slices, labels, boundaries, and
token counts but did not store the retrieval string whose token count had been
accepted.

It also did not store the citation evidence string that downstream grounded
generation and citation verification will need.

### Root cause

Phase 1.4e deliberately separated model-compatible passage packing from the
final persisted-text contract.

The builder materialized label-prefixed passage text internally to calculate
tokens, then discarded it. Citation/body behavior remained represented only by
the earlier retrieval-unit materializer.

This left downstream consumers responsible for independently recreating
separator, offset, label, and heading-only behavior.

### Correction

Persist two explicitly governed passage representations:

- `retrieval_text` for indexing and model-facing retrieval
- `citation_text` for source evidence and citation display

Calculate `passage_token_count` from the exact persisted `retrieval_text`.

For ordinary passages, exclude the retrieval-only label from citation text. For
heading-only passages, retain the reviewed heading as both representations
because it is the accepted evidence.

Reject records that already contain any governed text/count field before
materialization.

### Validation

The regenerated schema `1.1` corpus contains all 707 accepted passages and
preserves every pre-existing identity, ordering, provenance, boundary,
source-slice, and token-count field.

The audit confirmed:

- 903,356 retrieval-text characters
- 865,405 citation-text characters
- 53 heading-only passages
- 27 reference passages
- 14 reviewed intra-line boundaries
- maximum passage size of 445 tokens
- zero passages above the limit
- zero coordinate-ledger changes
- zero changed pre-existing passage fields beyond the schema version

The previous schema `1.0` artifacts were hash-verified and archived before the
schema `1.1` artifacts were published.

### Lesson

A token count is not a complete retrieval-data contract unless the exact text
being counted is also governed.

Retrieval context and citation evidence should be persisted separately when one
contains normalized labels that are useful for indexing but are not extracted
body evidence.

## BM25 misses Article 6 classification evidence

### Failure

For query `eu-002` — “When is an AI system classified as high-risk under
Article 6 of the EU AI Act?” — the corpus-wide BM25 baseline retrieves no
grade-`2` evidence in the first 10 results.

The two reviewed passages rank:

- passage 1: corpus rank 62, diagnostic EU-only rank 47
- passage 2: corpus rank 25, diagnostic EU-only rank 24

### Root cause

The query contains frequent legal and corpus terms such as `AI`, `system`,
`high-risk`, and `Article`. Other EU provisions contain denser combinations of
those terms, including Articles 25, 71, and 80.

The exact Article 6 evidence is therefore outranked even when candidates are
restricted diagnostically to the EU AI Act. The problem is not primarily
cross-document competition or an omitted benchmark judgment. It is a
within-document lexical-ranking limitation.

BM25 does not infer that “classified as high-risk under Article 6” specifically
targets the Article 6 classification rule rather than other provisions that
repeatedly discuss high-risk systems.

### Correction

Do not alter the benchmark, filter production candidates with gold
`document_scope`, tune BM25 parameters on this query, or add query-specific
rules.

Retain the failure as an accepted lexical-baseline limitation. Later dense,
hybrid, and reranked systems must be compared against the same fixed benchmark
and corpus-wide candidate set.

### Validation

The weak-query audit found:

- corpus Recall@10: `0.0`
- corpus reciprocal rank@10: `0.0`
- corpus direct-evidence hit@10: false
- corpus nDCG@10: `0.0`
- best gold corpus rank: 25
- best gold diagnostic EU-only rank: 24

Five additional weak queries retain at least one direct-evidence hit in the
first 10 and represent partial-recall or graded-ordering weaknesses rather than
complete retrieval failure.

### Lesson

Document filtering cannot repair every lexical failure. A lexical retriever can
rank neighboring or downstream legal provisions above the governing provision
even when all candidates come from the correct document.

Baseline failures should remain visible rather than being hidden through gold
metadata, benchmark-specific tuning, or post-hoc relevance expansion.

## Dense retrieval retains two partial-recall cases

### Failure

The accepted dense baseline achieves mean Recall@10 of `0.96875`, but two of
the 16 answerable queries retrieve only three of four reviewed passages in the
first ten.

For `rmf-002` — “How does organizational risk tolerance affect AI
risk-management priorities?” — the dense top ten contains:

- all three grade-`2` passages
- no grade-`1` GOVERN 1.3 supporting passage

For `eu-003` — “What obligations does Article 26 impose on deployers of
high-risk AI systems?” — the dense top ten contains:

- the instructions, oversight, and input-data segment
- the monitoring, incident-reporting, and log-retention segment
- the police-file documentation and reporting segment
- no specialized post-remote-biometric-identification segment

Both queries therefore have Recall@10 of `0.75`.

### Root cause

The `rmf-002` miss is a supporting-context ranking issue rather than a missing
direct answer. The three grade-`2` passages that define risk tolerance and
connect assessed risk to prioritization all rank in the first ten. The missed
grade-`1` GOVERN passage expresses a narrower management requirement and is
outscored by other semantically related RMF and GenAI passages.

The `eu-003` question asks broadly about Article 26 obligations, while the
article spans four accepted passage segments. The missed segment concerns a
specialized conditional use case: post-remote biometric identification. Dense
similarity ranks the three generally applicable deployer-obligation segments
above that narrower segment and also ranks nearby recitals and provisions in
the top ten.

These are not truncation, passage-boundary, model-interface, or cross-document
candidate-filter failures. All input passages satisfy the accepted token
contract, and both queries retrieve direct evidence at rank one.

### Correction

Do not:

- alter benchmark judgments
- use benchmark `document_scope` as a production filter
- add query-specific boosting
- increase the top-ten evaluation cutoff
- merge accepted passage segments
- tune the model or query instruction on these two examples

Retain both cases as visible dense-baseline limitations. Evaluate whether
deterministic hybrid retrieval or later reranking improves complete
multi-passage coverage against the same fixed benchmark.

### Validation

The accepted dense baseline records:

- `rmf-002` Recall@10: `0.75`
- `rmf-002` reciprocal rank@10: `1.0`
- `rmf-002` direct-evidence hit@10: true
- `eu-003` Recall@10: `0.75`
- `eu-003` reciprocal rank@10: `1.0`
- `eu-003` direct-evidence hit@10: true
- 14 other answerable queries with Recall@10 of `1.0`
- direct-evidence hit rate@10 of `1.0`

Manual review found no omitted benchmark evidence. A second full dense
execution regenerated the accepted result byte-for-byte.

### Lesson

Strong first-hit accuracy does not guarantee complete evidence coverage for
multi-passage questions.

Semantic retrieval can prioritize the central or generally applicable parts of
a policy provision while ranking narrower supporting or conditional passages
below the cutoff. Hybrid retrieval and reranking should therefore be evaluated
for both first-hit quality and complete reviewed-passage recall.

## Equal-weight RRF degrades the dense ranking

### Failure

Equal-weight reciprocal-rank fusion was evaluated as a possible deterministic
hybrid final ranking. With constant `60` and complete 707-passage rankings from
both retrievers, mean Recall@10 fell from the accepted dense value of `0.96875`
to `0.8802083333`. MRR@10 fell to `0.8125`, and mean nDCG@10 fell to
`0.7441517962`.

The fused ranking repaired `eu-003`, where BM25 ranks the missing Article 26
segment fifth, but degraded several queries whose relevant passages had strong
dense ranks and weak lexical ranks.

Examples include:

- `rmf-002`: relevant dense ranks `2` and `6` were moved to RRF ranks `16` and
  `35`
- `rmf-004`: a relevant dense rank `7` passage moved to RRF rank `11`
- `genai-001`: a relevant dense rank `2` passage moved to RRF rank `21`
- `eu-002`: a relevant dense rank `1` passage moved to RRF rank `15`

### Root cause

Equal-weight RRF assumes both retrievers provide similarly useful rank evidence
for each candidate. That assumption does not hold uniformly in this corpus.

When dense retrieval correctly resolves paraphrases or legal references but
BM25 assigns a weak rank, the lexical rank contribution can pull the relevant
passage below the final cutoff. Conversely, passages with moderate ranks from
both retrievers can outrank passages that are highly relevant under only one
retriever.

Increasing source-ranking depth does not fix this behavior. Full-corpus fusion
performed worse than fusion restricted to published top-ten lists, confirming
that the issue is the final-rank rule rather than missing candidate availability.

### Correction

Do not productionize equal-weight RRF as PolicyProof's final hybrid ranking.

Use BM25 and dense retrieval as complementary candidate generators:

- retrieve the top 20 from each full-corpus ranking
- deduplicate by passage ID
- preserve both source ranks
- assign no fused score
- defer final relevance ordering to the pinned cross-encoder phase

The top-20 union contains all reviewed passages for every answerable benchmark
query and averages `31.3125` candidates.

### Validation

The accepted candidate artifact records:

- mean candidate recall: `1.0`
- direct-evidence hit rate: `1.0`
- mean candidate count: `31.3125`
- zero missed reviewed passages
- no fused scores or final ranks
- SHA-256
  `94b98eda3795280ef31aa0dfaa49a44d912c23d77e50a75f33a4f2f26e1fe0d4`

Depth 20 was chosen after fixed-benchmark diagnostics. The coverage result is
therefore benchmark-informed and must be validated on additional queries before
making general retrieval claims.

### Lesson

Hybrid retrieval does not require score or rank fusion.

When retrievers have complementary failure modes, a candidate union can retain
their combined evidence coverage while avoiding a premature ranking rule that
damages the stronger retriever. Final ordering should be evaluated separately
with a model designed for query-passage relevance scoring.
