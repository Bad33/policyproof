# Experiment Results

## Deterministic BM25 baseline

The first production retrieval experiment evaluates plain-Python BM25 against
retrieval benchmark `v0.1.1` over all 707 accepted passages.

Contract:

- candidate scope: all passages
- lexical tokenizer: NFKC, lowercase, `[a-z0-9]+`
- query-term frequency: ignored
- BM25 parameters: `k1 = 1.2`, `b = 0.75`
- answerable queries evaluated: 16
- abstention queries excluded from ranking metrics: 4

Aggregate results:

| Metric | Result |
| --- | ---: |
| Mean Recall@1 | `0.3125000000` |
| Mean Recall@3 | `0.6041666667` |
| Mean Recall@5 | `0.6822916667` |
| Mean Recall@10 | `0.7760416667` |
| MRR@10 | `0.7433035714` |
| Direct-evidence hit rate@10 | `0.9375000000` |
| Mean nDCG@10 | `0.6555156356` |

Accepted result artifact:

`data/results/bm25-baseline-v0.1.0.json`

SHA-256:

`5609b146b0901fc84851789d3b6c2799ec6aad0545e33b9c80afaa29c9d80003`

The result regenerates byte-for-byte from the accepted manifest, passage
artifact, benchmark, and fixed BM25 contract.

## Weak-query audit

Six of the 16 answerable queries have Recall@10 below `1.0`.

Query `eu-002`, asking when Article 6 classifies an AI system as high-risk, is
the only query without any direct evidence in the first 10 results. Its two
grade-`2` passages rank 25 and 62 corpus-wide and 24 and 47 in an explicitly
diagnostic EU-only ranking. This is a within-document lexical-ranking failure,
not a consequence of cross-document competition.

The other five weak queries retrieve at least one direct-evidence passage but
miss additional reviewed passages or order them below competing lexical
matches. The audit did not establish a new benchmark omission, so benchmark
`v0.1.1` remains unchanged.
