# Evidence-Sufficiency Annotation Guide

Version: `0.1.0`

Status: proposed annotation contract for PolicyProof benchmark expansion.

This guide defines how human annotators label whether a supplied evidence set
supports a complete policy question.

It must be used together with:

`docs/evidence-sufficiency-research-protocol.md`

## Annotation objective

For each question-and-evidence-set case, determine:

1. whether the supplied evidence supports the complete question
2. whether the expected system action is answer or abstain
3. why the evidence is insufficient, when applicable
4. what material information is missing
5. which evidence passages were considered

Annotators evaluate only the supplied evidence set.

They must not answer from:

- memory
- general knowledge
- internet search
- other passages in the corpus
- assumptions about the organization or user
- benchmark relevance grades
- another annotator's decision

## Annotation unit

Each case contains:

- one stable case ID
- one source query ID
- one question
- an ordered list of evidence passage IDs
- the corresponding citation text
- optional document and section metadata

The complete ordered evidence set is the annotation unit.

Do not label individual passages independently unless the assigned task
explicitly asks for passage-level support judgments.

## Primary labels

### Sufficient

Label the evidence `sufficient` only when it supports every material component
of the question.

The associated response action is:

`answer`

A sufficient case must not contain:

- reason codes
- missing-information statements

### Insufficient

Label the evidence `insufficient` when any material component of the question
cannot be supported from the supplied evidence.

The associated response action is:

`abstain`

An insufficient case must contain:

- at least one reason code
- at least one concrete missing-information statement
- a concise rationale

## Core decision rule

Use this test:

> Could a careful answerer respond to the complete question using only this
> evidence, without adding an unsupported factual claim or conclusion?

If yes, label `sufficient`.

If no, label `insufficient`.

Do not require the evidence to contain polished answer wording. The evidence may
support a concise synthesis across passages.

Do not label evidence sufficient merely because it is topically related.

## Annotation decision sequence

Apply these steps in order.

### Step 1: Parse the question

Identify every material request in the question.

Examples:

- a list of named items
- both a risk and its mitigation
- both provider and deployer obligations
- a rule and its exception
- a process and how its results were used
- a current fact and a recommendation based on that fact

Write a private checklist of these requested components before reading the
evidence.

### Step 2: Review the complete evidence set

Read every supplied passage.

Consider:

- explicit statements
- definitions
- lists
- headings when headings carry accepted evidence
- relationships that can be directly synthesized across passages

Do not use omitted passages or nearby source text.

### Step 3: Map evidence to question components

For every material question component, identify the passage or passages that
support it.

A case cannot be sufficient if one material component has no support.

### Step 4: Check for prohibited inference

The evidence must not require the answerer to invent:

- a current fact
- an organization-specific conclusion
- a legal conclusion about a specific party
- a high-stakes recommendation
- a comparison not made by the sources
- a resolution of conflicting evidence without a stated basis

### Step 5: Assign the status and action

Use only these combinations:

- `sufficient` with `answer`
- `insufficient` with `abstain`

Any other combination is invalid.

### Step 6: Assign reason codes

Insufficient cases require one or more exact reason codes.

Use only the codes defined below.

### Step 7: State missing information

Describe the smallest material information gap that prevents a supported answer.

The statement must explain what is absent, not merely repeat that the evidence
is insufficient.

### Step 8: Write the rationale

Explain briefly why the selected status follows from the supplied evidence and
the question.

## Complete support for multi-part questions

A multi-part question is sufficient only when every requested part is supported.

Examples:

- risk without mitigation is insufficient for a risk-and-mitigation question
- mitigation without the risk description is insufficient
- general obligations without specialized obligations are insufficient when
  both are requested
- a rule without its material exception or override is insufficient
- process organization without how results were used is insufficient when both
  are requested

Do not average support across question parts.

One strongly supported part does not compensate for an unsupported part.

## Lists and enumerations

For a question requesting a list, determine whether it asks for:

- all listed items
- principal items
- examples
- one or more items

If the wording requests the full list, omission of a material listed item makes
the evidence insufficient.

If the wording asks for examples, a supported non-exhaustive response may be
sufficient.

Do not assume a list is exhaustive unless the question or evidence establishes
that requirement.

## Definitions and consequences

A definition-only passage is insufficient when the question also asks for:

- consequences
- risks
- examples
- mitigations
- operational implications

A consequences-only passage is insufficient when the requested concept itself
is not defined or identifiable from the evidence.

## Rules, exceptions, and overrides

A legal or policy rule may be incomplete without:

- an exception
- an override
- a threshold
- a condition
- a scope limitation
- a special case

If the omitted element can materially change the conclusion, label the evidence
insufficient.

Do not independently interpret how a rule applies to a real organization unless
the evidence and question remain at the general source-text level.

## Implicit versus explicit support

Explicit support is preferred but not always required.

A limited synthesis is permitted when:

- every factual premise is present in the evidence
- the relationship between premises is direct
- no specialized outside knowledge is needed
- no new legal, causal, comparative, or organizational conclusion is introduced

Label insufficient when the answer would require a material inference not stated
or directly supported by the evidence.

When uncertain, record the uncertainty for adjudication rather than stretching
the evidence.

## Heading-only evidence

A heading may be evidence when the accepted passage itself is a heading and the
question asks for the heading's exact concept or label.

Do not infer the contents of an unseen section from its heading.

## Current information

Use:

`current_information_required`

when answering the question requires information that may have changed after
the controlled documents were published.

Examples include:

- current office holders
- current laws or amendments
- current product capabilities or prices
- current organizational practices
- current comparative rankings

A historical source can support what was true at publication time, but not an
unstated present-day fact.

## Outside-corpus information

Use:

`outside_controlled_corpus`

when the requested fact is not contained in the supplied controlled evidence.

This code may apply even when the evidence is topically related.

Do not use it merely because a strict subset omits another accepted passage from
the same reviewed answer set. Use `incomplete_evidence_set` for that situation.

## Organization-specific conclusions

Use:

`organization_specific_conclusion`

when the question asks what a particular organization:

- must do
- should do
- currently does
- complies with
- violates
- qualifies as

and the evidence contains only general policy or legal text without the
organization-specific facts needed for the conclusion.

## Legal-advice boundary

Use:

`legal_advice_boundary`

when the requested response would apply legal rules to a specific party,
situation, obligation, liability, or compliance determination beyond what the
supplied general evidence supports.

This code does not prohibit direct lookup or neutral summary of legal text.

## High-stakes recommendations

Use:

`high_stakes_recommendation`

when the question requests a consequential recommendation involving areas such
as:

- healthcare
- legal action
- finance
- employment
- safety
- major purchases

and the evidence lacks the current, individualized, or independently reviewed
facts needed to support the recommendation.

## Unsupported comparisons

Use:

`unsupported_comparison`

when the question asks which option is:

- better
- safer
- cheaper
- more compliant
- more effective
- more current

but the evidence does not provide a common comparison basis.

Separate descriptions of two items do not automatically support a comparative
ranking.

## Incomplete evidence sets

Use:

`incomplete_evidence_set`

when the question is answerable from the controlled corpus, but the supplied
evidence set omits one or more necessary accepted passages or answer components.

Typical cases include:

- one half of a complementary pair
- some but not all required obligations
- a rule without its material override
- a definition without requested consequences
- actions 1 through 4 without actions 5 through 9

## Conflicting evidence

Use:

`conflicting_evidence`

when supplied passages make materially incompatible claims and the evidence does
not provide a supported way to resolve the conflict.

Do not use this code for:

- complementary passages
- different levels of detail
- a general rule plus a stated exception
- passages covering different time periods when the time distinction resolves
  the apparent conflict

The missing-information statement should identify what authority, date, scope,
or adjudicating fact is needed to resolve the conflict.

## Multiple reason codes

Assign every materially applicable reason code, but avoid redundant coding.

Examples:

A request for current legal advice about one company may require:

- `current_information_required`
- `organization_specific_conclusion`
- `legal_advice_boundary`

A current product recommendation may require:

- `outside_controlled_corpus`
- `current_information_required`
- `high_stakes_recommendation`

Reason codes describe distinct boundaries. They are not severity scores.

## Missing-information statements

Every insufficient case requires at least one missing-information statement.

A good statement is:

- specific
- factual
- minimal
- connected to the question
- understandable without seeing the reason-code definition

Good example:

> The evidence does not provide the organization's current deployment facts
> needed to determine whether the stated legal condition applies.

Weak examples:

- More context is needed.
- The answer is not in the passage.
- Insufficient evidence.
- Current information is missing.

When several independent gaps exist, use separate statements.

Do not prescribe where the missing information must be obtained unless that is
part of the evaluation task.

## Rationale requirements

The rationale should explain:

- which question components are supported
- which components are unsupported, if any
- why the assigned status follows

It should not:

- introduce outside facts
- quote long passages
- speculate about model behavior
- describe retrieval quality instead of evidence completeness
- rely on benchmark labels as justification

## Evidence order

Annotate the supplied evidence in its presented order, but base the status on
the complete set.

Evidence-order permutation should not change the label unless order itself
carries meaning in the source contract.

If order appears to change the interpretation, flag the case for adjudication.

## Distractor evidence

Topically related distractors do not make an incomplete set sufficient.

Irrelevant additions also do not make a sufficient set insufficient unless they
introduce a genuine unresolved contradiction.

Annotators should identify support, not reward topical density or passage count.

## Reference examples from dataset version 0.1.0

These examples illustrate the guide. They are development examples and must not
be reused as held-out test cases.

### Complete multi-passage evidence

`gpt4o-001-complete-reference`

The question asks for unauthorized voice-generation risks and mitigations.

One passage describes impersonation, fraud, and misinformation risks. A second
describes preset voices, output classification, blocking, and residual-risk
evaluation.

Together they are sufficient.

### Risk without mitigation

`gpt4o-001-risk-without-mitigation`

The evidence supports the risk portion but not the mitigation portion.

Label:

- status: `insufficient`
- action: `abstain`
- reason: `incomplete_evidence_set`

### Rule without override

`eu-002-core-rules-without-profiling-override`

The evidence contains the core classification rules but omits the profiling
override needed for the complete requested legal rule.

Label:

- status: `insufficient`
- action: `abstain`
- reason: `incomplete_evidence_set`

### Split obligations

`eu-004-provider-obligations-only`

Provider obligations alone are insufficient when the question also requests
deployer obligations.

### Complete single passage

`gpt4o-002-complete-reference`

One passage contains the privacy risk, surveillance risk, refusal behavior,
allowed exceptions, and evaluation result.

Do not manufacture an incomplete case merely because it is a single passage.

### Current unsupported comparison

`abstain-004-dense-top5`

The retrieved evidence does not provide current comparable evidence for the
requested conclusion.

Applicable reasons include:

- `outside_controlled_corpus`
- `current_information_required`
- `unsupported_comparison`

## Counterexamples

### Topical relevance is not sufficiency

A passage discussing general AI risk is not sufficient for a question asking
for a named document's complete list of risk categories.

### Retrieval rank is not sufficiency

A top-ranked passage may be insufficient. A lower-ranked passage may contain
the missing answer component.

Do not inspect or use retrieval rank when assigning the gold sufficiency label.

### Relevance grade is not sufficiency

A grade-1 passage may independently contain the requested facts for a narrowly
phrased question.

A grade-2 passage may still be insufficient for a broader multi-part question.

### Passage count is not sufficiency

One passage can be complete.

Five passages can remain insufficient.

## Independent annotation procedure

Each new case must be labeled by two annotators independently.

Annotators receive:

- question
- evidence text
- evidence IDs
- neutral source metadata
- this guide

Annotators must not receive:

- another annotator's label
- adjudicated labels
- expected behavior
- retrieval relevance grades
- evaluation tags
- policy predictions
- model scores

## Required annotation record

Each annotation record must contain:

- annotation ID
- annotator pseudonymous ID
- guide version
- case ID
- evidence status
- response action
- reason codes
- missing-information statements
- rationale
- uncertainty flag
- optional adjudication note
- annotation timestamp

Annotator identities should be stored separately from public pseudonymous IDs.

## Uncertainty flag

Set the uncertainty flag when:

- the question is materially ambiguous
- passage boundaries obscure required context
- support depends on a debatable inference
- legal scope is unclear
- passages appear contradictory
- the reason code is uncertain
- source formatting may have altered meaning

An uncertainty flag does not replace the required label.

Uncertain cases must be prioritized for adjudication.

## Disagreement taxonomy

Adjudication should classify disagreements as one or more of:

- question decomposition disagreement
- evidence interpretation disagreement
- implicit-inference disagreement
- completeness disagreement
- reason-code disagreement
- missing-information disagreement
- source-boundary disagreement
- legal-scope disagreement
- current-information disagreement
- annotation error
- guide ambiguity
- source extraction defect

## Adjudication procedure

For each disagreement:

1. Preserve both original annotations.
2. Identify the disagreement category.
3. Re-read the question and evidence.
4. Apply the current guide without viewing policy predictions.
5. Record the final adjudicated label.
6. Write a concise adjudication rationale.
7. Record whether the guide requires revision.
8. Re-review earlier affected cases if the guide changes materially.

Do not silently overwrite original labels.

## Quality-control checklist

Before accepting an annotation, verify:

- every question component was identified
- every component was mapped to evidence or a gap
- no outside information was used
- status and action are consistent
- insufficient cases have reason codes
- insufficient cases have concrete missing information
- sufficient cases have neither reason codes nor missing information
- rationale explains the decision
- exact machine-readable reason codes are used
- uncertainty is flagged when appropriate

## Guide versioning

This guide is immutable once used for a published annotation round.

Material changes require:

- a new guide version
- a written change summary
- identification of affected cases
- re-annotation or formal migration review
- updated dataset and annotation metadata

Typographical corrections that cannot affect labels may be documented without
re-annotation, but still require a recorded revision.

## Current limitations

This guide was developed from four AI-governance documents and the existing
PolicyProof development benchmark.

It may not fully cover:

- other legal systems
- case law
- contracts
- highly technical standards
- multilingual evidence
- numerical reasoning
- temporal conflicts across many document versions
- evidence requiring specialist domain interpretation

New domains require pilot annotation and guide review before they are added to a
held-out benchmark.
