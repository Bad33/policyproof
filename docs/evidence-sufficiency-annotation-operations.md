# Evidence-Sufficiency Annotation Operations

## Current status

PolicyProof is operationally preparing for genuine independent human
annotation of the frozen evidence-sufficiency batch.

No human annotation has been collected or accepted. No annotation record,
agreement report, adjudication record, or human-gold dataset is created by
this readiness phase.

The existing construction-derived silver labels are engineering artifacts.
They are not human labels and must remain strictly separated from the human
annotation workflow.

## Human roles and minimum staffing

One annotation round requires at least three distinct people:

1. two primary annotators;
2. one adjudicator.

The two primary annotators independently label every assigned case. The
adjudicator reviews only cases requiring adjudication after both primary
submissions have been accepted and pre-adjudication agreement has been
measured.

The adjudicator must not serve as either primary annotator for the same round.

An intake operator may validate and preserve submitted artifacts. The intake
operator performs administrative validation and does not change annotation
decisions.

## Pseudonymous identity handling

Repository artifacts use pseudonymous identifiers for annotators,
adjudicators, and intake operators.

The mapping between a pseudonymous identifier and a person's real identity
must be stored separately, outside Git, outside public artifacts, and outside
the distributable assignment packages.

Real names, email addresses, payment details, contact information, and other
private identity fields must not be added to annotation manifests,
assignments, submissions, receipts, agreement reports, or adjudication
artifacts.

A pseudonymous identifier alone does not prove independence. Each annotator
must provide a record-set-bound independence attestation.

## Frozen inputs

The annotation round is bound to:

- blinded batch:
  `data/evaluation/evidence-sufficiency-annotation-batch-v0.2.0.json`
- blinded-batch SHA-256:
  `1bb6a7bed55a43f59a79ff4861c81c3d36ffa5ed78af1bf12292bceb927bf93c`
- annotation guide version: `0.1.0`
- accepted passage artifact referenced by the blinded batch.

The blinded batch is immutable. Operational preparation must not modify its
case content, evidence content, evidence order within a case, identifiers,
bindings, or metadata.

The following information must not be shown to primary annotators:

- construction-derived silver labels;
- construction labels or expected outcomes;
- split assignments;
- another annotator's work;
- adjudicated labels;
- retrieval scores or ranks;
- model scores or probabilities;
- policy predictions;
- hidden construction metadata.

## Round preparation

The default PolicyProof round uses two primary annotators with 100% overlap:
both primary annotators receive all 160 frozen cases.

Assignment packages may use a counterbalanced case order to reduce shared
sequence and fatigue effects. Counterbalancing changes only the order in which
complete cases are presented. It must not change a case snapshot or reorder
evidence within a case.

The annotation-round manifest binds:

- the frozen batch and its SHA-256;
- the annotation guide and passage bindings;
- the pseudonymous primary-annotator IDs;
- the pseudonymous adjudicator ID;
- the full-overlap assignment policy;
- the exact case order assigned to each primary annotator.

Each primary annotator receives an isolated assignment package. That package
contains only the assigned annotator's pseudonymous ID and blinded cases. It
does not expose the other primary annotator or the adjudicator.

Before distribution, the assignment builder rejects unexpected batch, case, or evidence fields. This prevents hidden labels, retrieval scores, model outputs, or undeclared metadata from being copied into an annotator package.

Operational IDs used in automated tests are synthetic test fixtures. They are
not evidence that real people have been recruited or assigned.

## Independent annotation

Primary annotators must work separately and must not discuss case decisions
until both raw submissions have been accepted.

Each annotator uses only:

- the isolated assignment package;
- the frozen annotation guide;
- administrative instructions that do not disclose labels or predictions.

Each annotation must include the fields required by the frozen annotation
guide and annotation-record schema.

The annotator must truthfully answer all six independence statements:

1. `completed_without_collaboration`;
2. `did_not_view_other_annotations`;
3. `did_not_view_construction_or_silver_labels`;
4. `did_not_view_split_assignments`;
5. `did_not_view_retrieval_or_model_scores`;
6. `used_only_assigned_materials`.

A false statement is preserved as truthful operational metadata. It is not
silently changed to true.

## Submission intake

For each validated assignment, the tooling may generate a blank record-set template containing the bound annotator ID, assigned case IDs, and empty decision fields in assignment order. It does not copy questions, evidence text, labels, scores, or predictions.

The template is only a fillable operational artifact: unfilled templates are not valid submissions and must fail submission validation until every required human decision field is completed.

A raw annotation submission is accepted only when it:

- validates against the frozen batch;
- is bound to the assigned pseudonymous annotator;
- contains every assigned case exactly once;
- preserves the assigned case order;
- follows the annotation guide and record relationships;
- uses canonical UTC timestamps;
- passes unknown-field and binding checks.

The intake process records a metadata-only receipt bound to the exact
assignment package and raw record-set SHA-256. The receipt must not copy case
content, labels, reason codes, rationales, or missing-information text.

The operational rule is that raw submissions remain immutable after intake. Corrections require a new
versioned artifact and an explicit audit trail; they must not overwrite the
original submission.

## Round-completion kill-gate

The governing rule is that agreement analysis must not begin until the round-completion validator confirms
all of the following:

- every primary annotator assigned by the round manifest submitted exactly one
  bundle;
- there are no duplicate submission annotators;
- every assignment package is valid and correctly bound;
- every raw record set is valid and correctly bound;
- every submission receipt is valid and correctly bound;
- every independence attestation is valid and correctly bound;
- all six independence statements are affirmative for every primary
  annotator;
- record-set SHA-256 values are distinct.

A missing submission, duplicate annotator, failed independence statement,
binding mismatch, altered assignment, invalid receipt, or repeated record-set
hash blocks the round.

Passing this software gate establishes artifact consistency. It does not by
itself prove that a person acted independently; operational supervision and
truthful attestations remain required.

## Agreement and disagreement analysis

After the round-completion kill-gate passes, calculate pre-adjudication
agreement using the two original raw record sets.

Report agreement before adjudication and separately from final adjudicated
labels.

At minimum, report:

- raw sufficient-versus-insufficient agreement;
- Cohen's kappa when defined;
- exact reason-code-set agreement;
- reason-code Jaccard similarity;
- per-code and macro precision, recall, and F1;
- disagreement counts by question and evidence structure.

Missing-information text must not be judged by exact string matching alone.

No agreement value may be reported until genuine validated human submissions
exist.

## Adjudication

Adjudication covers every case with a label disagreement, relevant structural
disagreement, or uncertainty flag.

The adjudicator must:

1. preserve both original annotations;
2. identify the disagreement category;
3. reread the blinded question and evidence;
4. apply the frozen annotation guide;
5. avoid construction labels, silver labels, model outputs, and policy
   predictions;
6. record the final decision;
7. record a written adjudication rationale;
8. record whether a guide change is required.

The workflow must preserve both original annotations and keep adjudication
records separate. Original labels must never be silently overwritten.

A material guide change requires a new guide version and review of affected
cases before a human-gold artifact can be frozen.

## Artifact handling

Published operational artifacts must be deterministic where their content is
deterministic, UTF-8 encoded, atomically written, and non-overwriting.

Private identity mappings remain outside Git. Raw human submissions may also
require access-controlled storage rather than publication in the public
repository.

Every accepted artifact must retain its relevant version and SHA-256 bindings.

The artifact-handling rule is: do not create fake annotation records, fake annotators, fake timestamps,
fabricated agreement values, fabricated adjudication, or placeholder
human-gold results.

Test fixtures must remain clearly synthetic and must never be presented as
collected human data.

## Acceptance criteria

Phase 1 human-annotation readiness is accepted only after:

- the frozen batch hash is unchanged;
- the real batch passes blinded assignment-package validation;
- the round-manifest contract is tested;
- isolated assignment packages are tested;
- raw-submission intake and receipt contracts are tested;
- independence attestations are tested;
- the round-completion kill-gate is tested;
- this operations guide is reviewed;
- focused and complete repository test suites pass;
- Ruff, compilation, and text-quality checks pass;
- all changed files receive manual review.

Phase 1 readiness means PolicyProof can begin recruiting and assigning real
independent annotators. It does not claim that annotation collection has
occurred.

## Abort and restart conditions

Abort the current annotation round when any of the following occurs:

- an annotator sees construction-derived silver labels or other hidden labels;
- an annotator sees another annotator's decisions before submission;
- split assignments, retrieval scores, or model scores are disclosed;
- cases or evidence snapshots differ from the frozen batch;
- the annotation guide changes materially during the round;
- real identities are exposed in repository artifacts;
- independence cannot be truthfully attested;
- a raw submission is overwritten or its provenance is lost;
- the same person is used as a primary annotator and adjudicator;
- a required submission or immutable source artifact cannot be verified.

After an abort, preserve the affected artifacts for audit, document the
failure, create new versioned operational artifacts where appropriate, and
restart only with clean blinded assignments. Never silently repair a
contaminated annotation round.
