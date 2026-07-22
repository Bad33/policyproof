# Research Notes

A paper is added only when PolicyProof implements or evaluates one of its
findings.

Inclusion here does not mean that PolicyProof reproduces the paper's model,
training procedure, dataset, or reported performance.

## Evidence sufficiency and abstention

### Unanswerability Evaluation for Retrieval Augmented Generation

Reference:

- ACL 2025
- Anthology ID: `2025.acl-long.415`

Finding used by PolicyProof:

RAG evaluation should include questions that are unanswerable relative to the
available knowledge base, rather than evaluating only answerable questions.

PolicyProof application:

- the source benchmark contains 16 answerable queries and 4 queries that require
  abstention relative to the controlled corpus
- the evidence-sufficiency dataset preserves both behaviors
- the four abstention cases use actual retrieved passages rather than empty
  context

Not implemented:

- automatic unanswerable-query generation
- the paper's complete taxonomy or experimental pipeline

### Sufficient Context: A New Lens on Retrieval Augmented Generation Systems

Reference:

- arXiv: `2411.06037`

Finding used by PolicyProof:

Whether retrieved context is sufficient to answer a question is separate from
whether a language model correctly uses sufficient context.

PolicyProof application:

- question-and-evidence-set sufficiency is labeled independently of generation
- sufficient evidence cases are not treated as approved generated answers
- retrieval similarity is not treated as calibrated answer confidence

Not implemented:

- a learned context-sufficiency classifier
- selective answer generation
- the paper's model experiments

### S2G-RAG: Structured Sufficiency and Gap Judging for Iterative
Retrieval-Augmented QA

Reference:

- ACL 2026
- Anthology ID: `2026.acl-long.1185`

Finding used by PolicyProof:

Insufficient evidence can be represented through structured descriptions of the
information that remains missing.

PolicyProof application:

- every insufficient case contains one or more reason codes
- every insufficient case states concrete missing information
- partial multi-passage cases identify the unsupported component of the
  question

Not implemented:

- iterative retrieval
- an automated sufficiency controller
- conversion of missing-information gaps into follow-up searches

### Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via
Verifiable RL

Reference:

- Findings of ACL 2026
- Anthology ID: `2026.findings-acl.985`

Finding used by PolicyProof:

A useful abstention should identify why the available evidence is insufficient,
rather than returning an unsupported answer or only a generic refusal.

PolicyProof application:

- insufficient cases require explicit missing-information statements
- reason codes distinguish incomplete evidence from current-information,
  outside-corpus, organization-specific, legal, high-stakes, and comparison
  gaps
- the dataset can later evaluate informative abstention separately from binary
  refusal

Not implemented:

- reinforcement learning
- the Abstain-R1 model
- generated clarification responses
