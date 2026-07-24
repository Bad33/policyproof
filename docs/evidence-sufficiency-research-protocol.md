# Evidence-Sufficiency Research Protocol

Status: accepted protocol for the current PolicyProof
evidence-sufficiency research phase.

This document defines how PolicyProof will expand, split, annotate, and evaluate
evidence-sufficiency data without contaminating held-out results.

## Research objective

The central research question is:

> Can a structured evidence-sufficiency policy reduce unsupported answers in
> policy RAG while preserving useful answer coverage?

The policy must decide whether a supplied evidence set supports the complete
question. When evidence is insufficient, it should identify why and state what
information is missing.

The research phase evaluates evidence decisions only. Grounded answer
generation and claim-level citation verification remain later stages.

## Current dataset status

The accepted dataset is:

`data/evaluation/evidence-sufficiency-evaluation-v0.1.0.json`

It contains:

- 20 source queries
- 39 evidence cases
- 16 sufficient cases
- 23 insufficient cases

This dataset was manually constructed during retrieval and sufficiency
development. It was inspected while the schema, reason codes, evidence-set
construction rules, and validator were designed.

Therefore, version `0.1.0` is classified as:

- development and diagnostic data
- suitable for validator testing
- suitable for baseline prototyping
- unsuitable for unbiased held-out performance claims
- unsuitable for final policy selection by itself

Any result measured on version `0.1.0` must be described as
benchmark-informed.

## Research hypotheses

The initial hypotheses are:

1. Dense retrieval similarity alone cannot reliably separate sufficient from
   insufficient evidence.
2. Question-and-evidence reasoning improves sufficiency classification over
   score-only baselines.
3. Structured reason codes improve the usefulness and auditability of
   abstentions.
4. Explicit missing-information prediction reduces unsupported answers and
   generic refusals.
5. Removing one necessary passage from a complete evidence set should cause a
   reliable change from answer to abstain.
6. Adding topically related but irrelevant passages should not turn an
   insufficient set into a sufficient one.

These hypotheses must be evaluated on frozen data not used to select the final
policy.

## Evaluation unit

The atomic evaluation unit is:

- one source question
- one ordered evidence-passage set
- one expected evidence status
- one expected response action
- zero or more reason codes
- zero or more missing-information statements

Cases derived from the same source question are not independent observations.

## Leakage-safe grouping

Random case-level splitting is prohibited.

All cases connected through any of the following relationships must remain in
the same split:

1. The cases share the same source query ID.
2. The cases share an exact passage ID.
3. Their passages originate from the same accepted logical source span when one
   passage is a continuation or segmentation of another.
4. One case is a strict subset or superset of another case's evidence.
5. The questions are paraphrases created from the same annotated source fact.
6. The cases were created from the same synthetic transformation or template.

These relationships form a leakage component. A complete component must be
assigned to exactly one split.

At minimum, the split builder must construct connected components over:

- source query IDs
- exact evidence passage IDs

Before publication, manual review must check for additional section-level,
logical-span, paraphrase, and template leakage.

## Dataset expansion target

The next benchmark version should target at least:

- 80 new source-query groups
- 160 to 240 new evidence cases
- all four existing source documents
- both single-passage and multi-passage questions
- both answerable and corpus-relative abstention queries
- complete and incomplete evidence sets
- evidence sets containing topically related distractors

The target is a coverage goal, not permission to manufacture weak examples.
Annotation quality takes priority over an arbitrary case count.

The existing 20 query groups remain development data and do not count toward
the new held-out test target.

## Coverage dimensions

New source queries should cover:

### Question structure

- direct factual lookup
- definition
- factual list
- risk and mitigation
- process or evaluation method
- policy interpretation
- legal classification
- legal obligations
- comparison
- multi-part question

### Evidence structure

- one complete passage
- multiple complementary passages
- one necessary passage removed
- one incomplete half of a multi-part answer
- topically related but non-answering evidence
- distractor passages added to complete evidence
- distractor passages added to incomplete evidence
- evidence from multiple documents where justified

### Abstention boundary

The benchmark must cover the exact machine-readable reason-code contract:

- `outside_controlled_corpus`
- `current_information_required`
- `organization_specific_conclusion`
- `legal_advice_boundary`
- `high_stakes_recommendation`
- `unsupported_comparison`
- `incomplete_evidence_set`
- `conflicting_evidence`

The `conflicting_evidence` reason code requires deliberate coverage because it
does not appear in dataset version `0.1.0`.

## Proposed split policy

The expanded benchmark should use three immutable splits:

### Development split

Purpose:

- implement baseline policies
- debug serialization and evaluation
- inspect errors
- define metric calculations

The current dataset version `0.1.0` belongs entirely to development use.

### Validation split

Purpose:

- select model families
- select prompts or deterministic rules
- select score features
- choose thresholds
- choose abstention operating points

Validation labels may be inspected during policy development.

### Test split

Purpose:

- report final locked performance
- compare accepted baselines
- run final ablations
- support paper tables

The test split must be frozen before final policy selection. Test labels must not
be used to change:

- prompts
- thresholds
- features
- reason-code rules
- model choice
- evidence construction
- metric definitions

If test results cause a policy change, the test set becomes development data and
a new held-out test set must be created.

## Split assignment principles

Split assignment must operate on leakage components, not individual cases.

The assignment should aim for:

- no component crossing splits
- representation from every document in validation and test
- representation of sufficient and insufficient cases
- representation of major question structures
- representation of major abstention reasons
- similar evidence-size distributions where feasible

Perfect stratification is not required when it would break leakage isolation.

Split construction must be deterministic from:

- a declared component graph
- a fixed algorithm version
- an explicit seed
- stable case and passage IDs

The resulting split manifest must be committed and hash-locked.

## Annotation process

### Annotation guide

Before new labeling begins, create a versioned annotation guide defining:

- sufficient evidence
- insufficient evidence
- complete support for multi-part questions
- treatment of implicit versus explicit support
- treatment of legal interpretation
- treatment of current or outside-corpus facts
- treatment of contradictory passages
- reason-code definitions
- missing-information requirements
- examples and counterexamples

### Primary annotation

Each new case must be labeled independently by two annotators for:

- evidence status
- response action
- reason codes
- whether each missing-information statement is materially correct

Annotators must not see the other annotator's labels.

### Adjudication

Disagreements must be resolved through written adjudication.

The adjudication record should preserve:

- both original labels
- disagreement category
- final label
- adjudication rationale
- annotation-guide change, if any

The final dataset should not erase the fact that disagreement occurred.

## Agreement measurement

Report at least:

- raw agreement for sufficient versus insufficient
- Cohen's kappa for sufficient versus insufficient
- exact-match agreement for reason-code sets
- macro-averaged per-code precision, recall, and F1 between annotators
- Jaccard similarity for multi-label reason-code sets
- disagreement counts by question and evidence structure

Missing-information text should not be reduced to exact string agreement.
Instead, adjudicators should assess whether both annotations identify the same
material information gap.

Agreement must be reported before adjudication and separately from final
adjudicated labels.

## Baseline policy families

The research evaluation should include at least:

### Constant baselines

- always answer
- always abstain
- majority class

### Retrieval-score baselines

Potential inputs:

- top-one similarity
- top-three mean
- top-five mean
- top-one minus top-two margin
- top-one minus top-five margin
- evidence-set size

These are baselines only. Retrieval scores are not assumed to be calibrated
answer confidence.

### Deterministic feature baseline

A small interpretable model using frozen numeric and structural features.

No feature may include benchmark-only fields such as:

- expected behavior
- relevance grades
- evaluation tags
- document scope
- reason codes
- missing-information labels

### Pairwise or set-level semantic baseline

A model that evaluates the question jointly with the supplied evidence set.

The model contract must pin:

- model identity
- exact revision
- tokenizer
- runtime
- input construction
- truncation behavior
- scoring transformation
- deterministic tie-breaking

### Structured judge baseline

A judge that emits:

- sufficient or insufficient
- answer or abstain
- reason codes
- missing information
- evidence passage IDs used in the decision

All structured outputs must be schema validated.

## Policy-selection rule

The accepted policy must not be selected by overall accuracy alone.

The primary safety objective is to minimize unsafe answers:

> predicted answer when expected action is abstain

The initial policy-selection order is:

1. satisfy a declared maximum unsafe-answer rate on validation
2. among policies satisfying that constraint, maximize answer coverage
3. use macro-F1 and reason-code quality as secondary criteria
4. prefer the simpler and more reproducible policy when results are materially
   similar

The maximum acceptable unsafe-answer rate must be declared before test
evaluation.

## Core metrics

Report at least:

- accuracy
- sufficient precision, recall, and F1
- insufficient precision, recall, and F1
- unsafe-answer rate
- unnecessary-abstention rate
- answer coverage
- selective accuracy
- selective risk
- confusion matrix
- per-document metrics
- per-question-structure metrics
- per-evidence-structure metrics
- per-reason-code precision, recall, and F1
- exact reason-code set match
- missing-information semantic correctness
- latency per case
- model or API cost per case where applicable

Macro-averaged metrics must accompany micro-averaged metrics.

## Robustness evaluation

For suitable source questions, construct paired perturbations:

### Necessary-evidence removal

Start from a sufficient evidence set and remove one necessary passage.

Expected behavior:

- sufficient becomes insufficient
- answer becomes abstain
- the missing-information output identifies the removed support

### Irrelevant-evidence addition

Add one or more topically related distractors.

Expected behavior:

- the original sufficiency label remains unchanged
- reason codes remain stable unless the new evidence genuinely resolves a gap

### Evidence-order permutation

Reorder the same evidence IDs.

Expected behavior:

- the decision should remain unchanged unless the policy contract explicitly
  defines order as meaningful

### Cross-document distractors

Add relevant-looking passages from another source document.

Expected behavior:

- the policy should not infer unsupported conclusions from thematic overlap

## Evaluation granularity

Because multiple cases may derive from one source question, report both:

- case-level metrics
- query-group-level metrics

Query-group bootstrap intervals or grouped resampling should be preferred over
treating all cases as independent.

Confidence intervals must never resample strict-subset cases independently from
their parent query group.

## Reproducibility requirements

Every published evaluation must bind:

- corpus ID and version
- passage artifact schema and SHA-256
- source retrieval benchmark and SHA-256
- evidence dataset and SHA-256
- split manifest and SHA-256
- policy implementation version
- model assets and revisions
- runtime package versions
- deterministic seed where applicable
- exact metric implementation version

Publication must be atomic and non-overwriting.

## Test-set governance

Before running final test evaluation:

1. Freeze the test dataset.
2. Freeze the split manifest.
3. Freeze metric definitions.
4. Freeze the selected policy.
5. Record all hashes.
6. Confirm the working tree is clean.
7. Run the test evaluation once for the primary result.
8. Preserve the complete output artifact.
9. Do not silently rerun after inspecting failures.

Additional test runs are allowed for reproducibility verification but must use
the identical frozen contract and produce byte-identical outputs where the
policy is deterministic.

## Publication claims

The initial four-document corpus does not justify broad claims about all policy
or legal RAG systems.

Permitted claims must be scoped to:

- the accepted PolicyProof corpus
- the published benchmark version
- the evaluated policy contracts
- the measured question and evidence categories

Any broader claim requires additional corpora, annotators, and held-out
evaluation.

## Implementation status and next steps

Completed infrastructure:

1. The research protocol has been reviewed and accepted.
2. Annotation guide version `0.1.0` has been created and is immutable once used
   for a published annotation round.
3. Deterministic leakage-component construction and split-manifest validation
   have been implemented.
4. The existing 39-case dataset has been correctly classified as
   development-only.
5. Blinded annotation-batch validation has been implemented.
6. Independent raw annotation-record validation has been implemented.
7. Pre-adjudication comparison and agreement reporting have been implemented.
8. Written adjudication validation has been implemented.
9. Separate question-structure and evidence-structure analysis metadata has
   been implemented.
10. Deterministic, atomic, non-overwriting JSON artifact publication has been
    implemented.

Remaining controlled workflow:

1. Define and review the new query inventory before any evidence labels are
   created.
2. Construct new cases from accepted passages without inspecting model
   predictions.
3. Publish blinded annotation batches bound to the accepted guide and passage
   artifact.
4. Obtain two independent annotations for every new case.
5. Publish pre-adjudication agreement and structure-disagreement reports.
6. Complete and publish written adjudication for every disagreement or
   uncertainty case.
7. Construct a new adjudicated evidence-sufficiency dataset version while
   preserving both original annotations and the adjudication history.
8. Reconstruct leakage components across existing and new cases.
9. Assign components deterministically to development, validation, and test.
10. Freeze the dataset, split manifest, hashes, and metric contracts before any
    runtime policy selection.

No new validation or test cases have yet been accepted.
