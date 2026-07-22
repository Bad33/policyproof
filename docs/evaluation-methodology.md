# Evaluation Methodology

## Retrieval-ranking scope

Retrieval baselines rank the complete accepted passage corpus. The current
corpus contains 707 passages from four documents.

Benchmark `document_scope` is manually reviewed evaluation metadata. It is not
available to an ordinary production query and is therefore never used to filter
production retrieval candidates. Document-scoped rankings may be calculated
only as explicitly labeled diagnostics.

Passage-ranking metrics use the 16 queries whose `expected_behavior` is
`answer`. The four required-abstention queries are excluded because passage
ranking alone does not decide whether retrieved evidence is sufficient. They
are now represented by a separate, manually reviewed evidence-sufficiency evaluation dataset; runtime decision policy remains deferred.

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

## Dense embedding contract

The deterministic dense baseline uses:

- model `BAAI/bge-small-en-v1.5`
- revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- model file `onnx/model.onnx`
- model SHA-256
  `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35`
- `onnxruntime==1.27.0`
- `numpy==2.5.1`
- `CPUExecutionProvider` only
- the packaged PolicyProof BERT WordPiece tokenizer
- query instruction
  `Represent this sentence for searching relevant passages: `
- no passage instruction
- no truncation
- CLS pooling
- L2-normalized 384-dimensional embeddings
- normalized dot-product similarity
- accepted passage batch size `32`
- descending score order
- equal-score resolution by accepted passage order and then passage ID

The model asset is supplied through an explicit local path and validated by
exact byte size and SHA-256 before session creation. PolicyProof does not
download the model automatically or commit the model binary.

The model, instruction, pooling, normalization, runtime versions, provider,
batch size, and ranking behavior are fixed baseline contracts. They were not
tuned on the 20-query benchmark.

## Hybrid candidate-generation contract

The hybrid stage does not combine BM25 and dense scores into a final ranking.
It independently retrieves the top 20 passages from each full-corpus retriever,
deduplicates the two sets by passage ID, and records:

- accepted passage order
- BM25 source rank when present
- dense source rank when present

Candidate records are serialized in accepted passage order followed by passage
ID. That order is deterministic storage order, not relevance order. The stage
does not emit a fused score, reciprocal-rank-fusion score, or final rank.

Input depth 20 was selected after fixed-benchmark diagnostics compared depths
`5`, `10`, `20`, `30`, `50`, and `100`. Depth 20 was the smallest tested depth
whose union contained every reviewed passage for all answerable queries.
Therefore, hybrid candidate-coverage measurements are benchmark-informed and
must not be presented as out-of-sample performance.

Candidate coverage uses the 16 answerable queries. Abstention queries remain
outside candidate metrics.

**Candidate recall** is the number of reviewed grade-`1` or grade-`2` passages
present anywhere in the candidate union divided by the total reviewed passages
for that query.

**Direct-evidence hit rate** is the fraction of answerable queries whose union
contains at least one grade-`2` passage.

**Mean candidate count** is the arithmetic mean of deduplicated union size
across answerable queries.

## Cross-encoder reranking contract

The comparison reranker uses:

- model `cross-encoder/ms-marco-MiniLM-L6-v2`
- revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- model file `onnx/model.onnx`
- model SHA-256
  `5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a`
- `onnxruntime==1.27.0`
- `numpy==2.5.1`
- `CPUExecutionProvider` only
- one intra-op thread and one inter-op thread
- sequential execution
- deterministic compute enabled
- the packaged PolicyProof BERT WordPiece tokenizer
- pair template `[CLS] query [SEP] passage [SEP]`
- query token-type ID `0`
- passage token-type ID `1`
- no query or passage instruction
- maximum sequence length `512`
- rejection rather than truncation for overlength pairs
- raw scalar logit scoring
- descending-logit ranking
- equal-score resolution by accepted passage order and passage ID

The model asset is supplied through an explicit local path and verified by
exact byte size and SHA-256 before session creation. PolicyProof does not
download the model automatically or commit the model binary.

The reranker scores only the accepted hybrid candidate union. It does not
search the corpus independently and does not remove or add candidates. Every
ranked result retains its BM25 and dense source ranks.

The current model and contract were evaluated without benchmark-specific query
instructions, score fusion, query-specific boosting, or fine-tuning.

All 501 accepted query-candidate pairs fit the model limit. The maximum observed
pair length is 467 tokens.

## Ranking-model selection rule

A later pipeline stage is not accepted merely because it is more specialized.

Ranking selection compares each candidate implementation against the strongest
accepted baseline using:

- mean Recall@1, Recall@3, Recall@5, and Recall@10
- MRR@10
- direct-evidence hit rate@10
- mean nDCG@10
- per-query residual failures
- byte reproducibility
- semantic review of high-ranked and missed passages
- absence of benchmark-only production filtering or query-specific tuning

The pinned cross-encoder improves over BM25 but underperforms dense retrieval on
every graded aggregate ranking metric except tying direct-evidence hit rate@10.
Dense retrieval therefore remains the selected ranking.

The cross-encoder result remains a valid, immutable experimental baseline for
future model comparison.

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
- retriever-specific tokenizer or model identity
- retriever-specific parameters, runtime, pooling, normalization, and similarity
- candidate scope and deterministic tie-breaking
- aggregate and per-query metrics

Full-corpus BM25 and dense ranking artifacts record top-10 passage IDs, scores,
ranks, and accepted-order positions. The hybrid candidate artifact instead
records the deduplicated candidate IDs, accepted order, source-retriever ranks,
and coverage measurements. It deliberately contains no fused score or final
relevance rank.

The reranker artifact records the complete bounded candidate ranking for each
answerable query, raw finite logits, accepted order, BM25 rank, dense rank, and
ranking metrics. It binds the exact hybrid candidate artifact, model, tokenizer,
runtime, corpus, passages, and benchmark. It contains no benchmark
`document_scope`, evaluation tags, relevance judgments, or model binary.

Result publication is atomic and non-overwriting. Repository regression tests
lock exact accepted hashes. Baseline acceptance also requires a separate
byte-identical regeneration audit. Dense and hybrid repository-result tests
remain offline and do not require the external model binary.

Retrieved but unjudged passages are not automatically benchmark errors. Any
benchmark correction requires independent manual evidence review, a new
immutable dataset version, an exact rationale, comparison against earlier
versions, and regression tests preserving prior dataset hashes.

## Evidence-sufficiency evaluation contract

Evidence sufficiency is evaluated separately from retrieval ranking.

Accepted artifact:

`data/evaluation/evidence-sufficiency-evaluation-v0.1.0.json`

Accepted artifact properties:

- schema version: `1.0`
- dataset version: `0.1.0`
- size: `46673` bytes
- SHA-256:
  `9ecd30e4ff829561b50d56bf4f1d3d44c79dcb043ec15661175842597d733a6a`

The artifact binds exactly to:

- passage schema version `1.1`
- passage artifact SHA-256:
  `5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96`
- source retrieval dataset version `0.1.1`
- source retrieval dataset SHA-256:
  `42e7e0e1a824b1c48973bb2163aca7664d53161632fcd699068931cd9fe80a7c`

### Evaluation unit

The evaluation unit is a source question plus an ordered set of accepted passage
IDs.

Each case is labeled as either:

- `sufficient`, with expected response action `answer`; or
- `insufficient`, with expected response action `abstain`

A sufficient label means the reviewed evidence set supports the complete
question at the accepted passage granularity.

It does not evaluate:

- generated answer wording
- citation placement
- language-model reasoning
- factual consistency outside the supplied evidence
- production answer quality

An insufficient label means the selected evidence set does not support the
complete requested conclusion.

It does not claim that the answer is unknowable outside the controlled corpus.

### Sufficient-case requirements

A sufficient case must:

- derive from a source retrieval query labeled `answer`
- contain at least one accepted passage
- use only passages reviewed for that source query
- include at least one grade-`2` passage
- require the `answer` action
- contain no insufficiency reason codes
- contain no missing-information statements

Each of the 16 answerable source queries has exactly one sufficient reference
case.

### Insufficient-case requirements

An insufficient case must:

- require the `abstain` action
- include at least one allowed reason code
- state at least one concrete item of missing information
- explain why the supplied evidence does not support the complete question

Allowed reason codes are:

- `outside_controlled_corpus`
- `current_information_required`
- `organization_specific_conclusion`
- `legal_advice_boundary`
- `high_stakes_recommendation`
- `unsupported_comparison`
- `incomplete_evidence_set`
- `conflicting_evidence`

The accepted dataset currently uses every code except
`conflicting_evidence`.

A source retrieval query labeled `abstain` cannot be relabeled sufficient.

### Manual construction procedure

All 20 source queries were manually reviewed against the actual accepted
`citation_text`.

For answerable queries:

1. Review all grade-`2` passages and any grade-`1` supporting passages.
2. Select one reference set sufficient for the complete question.
3. Review strict subsets of multi-passage reference sets.
4. Add an insufficient case only when the subset omits a material component of
   the question.
5. Do not manufacture an incomplete case when one accepted passage already
   contains the complete answer.

For source abstention queries:

1. Use the actual top five passages returned by the selected dense retriever.
2. Review whether those passages support the requested conclusion.
3. Record the product or evidence boundary using reason codes.
4. State the exact information that remains unavailable.

No empty-evidence control cases were added.

### Dataset composition

The accepted dataset contains 39 cases over all 20 source queries:

- 16 sufficient reference cases
- 19 incomplete-evidence cases from answerable queries
- 4 required-abstention cases using actual dense top-five evidence

The dataset has:

- complete source-query coverage
- exactly one sufficient case for each answerable query
- no sufficient cases for source abstention queries
- no duplicate case IDs
- no duplicate query/evidence-set contracts
- no conflicting labels

### Similarity diagnostic

No retrieval-similarity threshold was selected.

Under the exact pinned dense model:

- minimum answerable top-one score: `0.728191`
- maximum abstention top-one score: `0.731699`

The ranges overlap. A cutoff high enough to reject the strongest abstention
case would also reject at least one answerable case.

Observed top-three means, top-five means, and score margins also overlap between
the answerable and abstention groups.

These measurements reject the assumption that one observed similarity cutoff is
a sufficient production answer policy.

### Current evaluation boundary

The published artifact is a gold evaluation contract, not a runtime-policy
result.

It contains no:

- selected sufficiency classifier
- selected score threshold
- prompt-based judge
- language-model evaluator
- confusion matrix
- calibration result
- grounded answer
- generated abstention response

Any future policy, classifier, prompt, or model evaluated on this fixed dataset
must be described as benchmark-informed unless it is selected independently or
evaluated on held-out cases.

The dataset must not be changed merely to improve a future policy's measured
performance.

## Leakage-safe split methodology

Evidence-sufficiency cases must be split by leakage component rather than by
individual case.

Accepted split artifact:

`data/evaluation/evidence-sufficiency-split-manifest-v0.1.0.json`

Artifact properties:

- schema version: `1.0`
- manifest version: `0.1.0`
- size: `2103` bytes
- component algorithm version: `1.0.0`
- component count: `19`
- development cases: `39`
- validation cases: `0`
- test cases: `0`
- SHA-256:
  `314d5ca55a1d6557e8f711eea3506ce13a85d30f40e706ea27f0afb8226ff4b2`

The manifest binds exactly to evidence dataset version `0.1.0` and its accepted
SHA-256:

`9ecd30e4ff829561b50d56bf4f1d3d44c79dcb043ec15661175842597d733a6a`

### Leakage-component construction

A case is connected to another case when they share at least one of:

- source query ID
- exact evidence passage ID
- accepted passage `logical_source_key`

Connectivity is transitive.

For example, if:

- case A shares a query with case B
- case B shares a logical source with case C

then A, B, and C form one indivisible component.

This protects against leakage from:

- repeated question wording
- complete and strict-subset evidence pairs
- an identical passage reused by different questions
- separately segmented passages from one logical source
- transitive combinations of those relationships

Document identity alone does not connect cases. Grouping every case from one
document would be unnecessarily coarse and would prevent future validation and
test splits from representing all source documents.

### Determinism

Component construction is deterministic from:

- stable case IDs
- stable query IDs
- stable passage IDs
- stable logical-source keys
- component algorithm version `1.0.0`

Reversing the order of case records or passage records produces the same
component records and ordering.

Each component records sorted:

- case IDs
- query IDs
- passage IDs
- logical-source keys

### Split-assignment validation

A split manifest must contain exactly:

- `development`
- `validation`
- `test`

The validator requires:

- every known case is assigned
- every case is assigned exactly once
- no unknown case is assigned
- no duplicate case assignment exists
- no leakage component crosses splits
- declared case counts match the assignments
- declared component count matches reconstruction
- evidence dataset ID, version, and SHA-256 match
- component algorithm version is supported
- unknown manifest fields are rejected

Validation and test arrays may be empty when no legitimately held-out cases
exist.

### Existing-dataset classification

The accepted 39-case dataset forms:

- 20 source-query groups
- 19 leakage components

The reduction from 20 query groups to 19 components occurs because
`abstain-004` and `gpt4o-003` share both:

- the same accepted GPT-4o passage
- the same logical source

These two queries cannot be assigned to different splits.

All 39 existing cases are assigned to `development`.

This is intentional because dataset version `0.1.0` was inspected during:

- schema design
- reason-code design
- case-construction rule development
- validator development
- similarity diagnostics
- research-protocol development

The current manifest therefore does not support held-out performance claims.

### Requirements for future validation and test data

Validation and test splits must be populated only with new independently
annotated query groups.

Future cases must first undergo:

1. annotation under the versioned guide
2. independent second annotation
3. disagreement measurement
4. written adjudication
5. leakage-component reconstruction
6. deterministic component-level split assignment
7. manifest validation
8. immutable artifact publication

Cases must not be moved individually to improve class balance or policy
performance.

When perfect stratification conflicts with leakage isolation, leakage isolation
takes priority.

### Current boundary

The split infrastructure does not:

- choose a sufficiency policy
- choose a model
- choose a prompt
- set a threshold
- generate predictions
- calculate policy performance
- manufacture held-out cases
- convert development cases into test cases

It establishes the data-governance boundary required before those activities.
