# Evaluation Methodology

## Retrieval-ranking scope

Retrieval baselines rank the complete accepted passage corpus. The current
corpus contains 707 passages from four documents.

Benchmark `document_scope` is manually reviewed evaluation metadata. It is not
available to an ordinary production query and is therefore never used to filter
production BM25 candidates. Document-scoped rankings may be calculated only as
explicitly labeled diagnostics.

Passage-ranking metrics use the 16 queries whose `expected_behavior` is
`answer`. The four required-abstention queries are excluded because passage
ranking alone does not decide whether retrieved evidence is sufficient. They
will be evaluated in a later evidence-sufficiency phase.

## BM25 lexical contract

The deterministic BM25 baseline:

- normalizes passage and query text with Unicode NFKC
- lowercases normalized text
- extracts terms matching `[a-z0-9]+`
- treats punctuation, apostrophes, and hyphens as term boundaries
- ignores query-term frequency by deduplicating terms in first-seen order
- retains normal passage term frequency
- uses `k1 = 1.2` and `b = 0.75`
- uses `log(1 + (N - df + 0.5) / (df + 0.5))` for IDF
- ranks by descending score
- resolves equal scores by accepted passage order and then passage ID

Parameters are an explicit baseline contract. They were not tuned on the
20-query benchmark.

## Ranking metrics

Let the gold set contain every passage with relevance grade `1` or `2`.

**Recall@k** is the number of gold passages appearing in the first `k` results
divided by the total number of gold passages for that query. Mean Recall@k is
the arithmetic mean across the 16 answerable queries.

**Reciprocal rank@10** is the reciprocal of the rank of the first grade-`1` or
grade-`2` passage in the first 10 results. It is zero when no relevant passage
appears. MRR@10 is the arithmetic mean across answerable queries.

**Direct-evidence hit rate@10** is the fraction of answerable queries with at
least one grade-`2` passage in the first 10 results.

**nDCG@10** uses gain `2^grade - 1` and logarithmic discount
`1 / log2(rank + 1)`. The ideal ranking sorts all reviewed grades in descending
order and truncates at 10. Mean nDCG@10 is the arithmetic mean across answerable
queries.

The committed result reports mean Recall@1, Recall@3, Recall@5, Recall@10,
MRR@10, direct-evidence hit rate@10, and mean nDCG@10.

## Integrity and reproducibility

The current evaluation uses only:

`data/evaluation/retrieval-evaluation-v0.1.1.json`

It is bound to passage schema `1.1` and the accepted 707-passage artifact. The
runner validates the manifest, benchmark, passage records, and passage-artifact
checksum before evaluation.

Result artifacts bind:

- benchmark identity, schema version, dataset version, and SHA-256
- corpus identity and version
- passage schema version, count, and SHA-256
- BM25 parameters
- lexical-tokenizer behavior
- candidate scope and deterministic tie-breaking
- aggregate and per-query metrics
- top-10 passage IDs, scores, ranks, and accepted-order positions

Result publication is atomic and non-overwriting. Repository regression tests
require byte-identical regeneration.

Retrieved but unjudged passages are not automatically benchmark errors. Any
benchmark correction requires independent manual evidence review, a new
immutable dataset version, an exact rationale, comparison against earlier
versions, and regression tests preserving prior dataset hashes.
