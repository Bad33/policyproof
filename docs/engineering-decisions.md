# Engineering Decisions

This log records decisions that materially affect PolicyProof's scope,
architecture, evaluation, reproducibility, security, or maintainability.

---

## PP-001: Evidence-focused product boundary

**Decision:**

PolicyProof will report what selected public sources state. It will not provide
legal advice, determine legal compliance, or make organization-specific legal
conclusions.

**Context:**

The corpus includes regulatory and AI-governance material that users could
mistake for personalized legal guidance.

**Options considered:**

- Legal-advice assistant
- Automated compliance checker
- Evidence-focused research assistant

**Selected option:**

Evidence-focused research assistant.

**Why:**

This scope is safer, more testable, and appropriate for the available public
document corpus.

**Trade-offs:**

Some questions must be rejected even when a language model could generate a
plausible response.

**How we will verify it:**

Include legal-advice and organization-specific compliance questions in the
answerability evaluation dataset.

**Date:**

2026-07-18

---

## PP-002: Evaluation before interface development

**Decision:**

Build the controlled corpus and manually reviewed evaluation dataset before a
polished API or user interface.

**Context:**

Without evaluation data, retrieval and verification improvements cannot be
measured objectively.

**Options considered:**

- UI-first development
- API-first development
- Evaluation-first pipeline development

**Selected option:**

Evaluation-first pipeline development.

**Why:**

It allows BM25, dense retrieval, hybrid retrieval, and reranking to be compared
using the same corpus and questions.

**Trade-offs:**

The project will not have a polished visual demonstration during its early
phases.

**How we will verify it:**

No frontend work begins before retrieval and answerability baselines have been
measured.

**Date:**

2026-07-18

---

## PP-003: Framework-independent core pipeline

**Decision:**

Implement ingestion, retrieval, evaluation, and citation-verification logic in
plain Python instead of using LangChain or LangGraph.

**Context:**

The important engineering and evaluation logic must remain visible,
understandable, and independently testable.

**Options considered:**

- LangChain
- LangGraph
- LlamaIndex
- Plain Python

**Selected option:**

Plain Python.

**Why:**

The initial pipeline does not require framework-based orchestration.

**Trade-offs:**

We will write and maintain more integration code ourselves.

**How we will verify it:**

Every pipeline stage must be callable independently from tests and command-line
evaluation scripts.

**Date:**

2026-07-18

---

## PP-004: Python 3.12 development environment

**Decision:**

Develop and test PolicyProof using Python 3.12 and the standard `venv` module.

**Context:**

Some AI and machine-learning libraries may not support newer Python releases
reliably.

**Options considered:**

- Python 3.11
- Python 3.12
- Python 3.14

**Selected option:**

Python 3.12.

**Why:**

It provides modern Python functionality while maintaining broad library
compatibility.

**Trade-offs:**

The project intentionally restricts local and CI environments to Python 3.12
until compatibility requirements change.

**How we will verify it:**

`pyproject.toml` requires Python `>=3.12,<3.13`, and GitHub Actions will later
run against Python 3.12.

**Date:**

2026-07-18

---

## PP-005: File-based storage before PostgreSQL

**Decision:**

Use JSON, JSONL, Markdown, and generated local index files before introducing a
database.

**Context:**

The first corpus contains only a few controlled documents and is maintained by
one developer.

**Options considered:**

- Local files
- SQLite
- PostgreSQL

**Selected option:**

Local files.

**Why:**

They are inspectable, easy to version, and sufficient for initial offline
experiments.

**Trade-offs:**

Local files are not ideal for concurrent writes, complex querying, or large
trace datasets.

**How we will verify it:**

PostgreSQL will be introduced only after identifying a persistence, concurrency,
or analytical-query requirement that local files cannot handle cleanly.

**Date:**

2026-07-18

---

## PP-006: Official-source allowlist

**Decision:**

The initial corpus will contain documents downloaded only from manually approved
official source domains.

**Context:**

Third-party summaries may omit, reinterpret, or reproduce outdated policy text.

**Options considered:**

- General web crawling
- Search-engine result ingestion
- Official-source allowlist

**Selected option:**

Official-source allowlist.

**Why:**

It improves provenance, version control, and confidence that the original
document was retrieved.

**Trade-offs:**

Corpus expansion will be slower and may require source-specific extraction
logic.

**How we will verify it:**

The ingestion pipeline will reject source URLs whose domains are absent from the
approved source manifest.

**Date:**

2026-07-18

---

## PP-007: No paid model calls in normal CI

**Decision:**

Pull-request CI will use deterministic fixtures, local computations, and mocked
generation responses.

**Context:**

External model calls introduce cost, nondeterminism, latency, rate limits, and
secret-management risks.

**Options considered:**

- Live model calls in every pull request
- Fully offline pull-request CI
- Offline pull-request CI with optional scheduled live evaluation

**Selected option:**

Offline pull-request CI with the possibility of scheduled live evaluation later.

**Why:**

Normal CI must be repeatable and safe to run from a public repository.

**Trade-offs:**

Offline CI cannot detect every behavior change introduced by an external model
provider.

**How we will verify it:**

The default test suite and pull-request workflow must pass without API keys or
network access.

**Date:**

2026-07-18

---

## PP-008: Research entries require implementation

**Decision:**

A paper will be added to `docs/research-notes.md` only when PolicyProof
implements or evaluates one of its findings.

**Context:**

A long bibliography does not make a project research-driven unless the research
changes an implementation or experiment.

**Options considered:**

- General bibliography
- Broad literature survey
- Implementation-linked research notes

**Selected option:**

Implementation-linked research notes.

**Why:**

This creates defensible engineering decisions and interview explanations.

**Trade-offs:**

The research-notes file will grow more slowly.

**How we will verify it:**

Every completed paper entry must identify a PolicyProof baseline, experiment,
result, or documented implementation difference.

**Date:**

2026-07-18

---

## PP-009: Manifest-controlled corpus entry

**Decision:**

Every source document must be declared in a version-controlled source manifest
before it can be downloaded or processed.

**Context:**

Retrieval experiments are meaningful only when the exact corpus identity,
version, provenance, and source locations are known.

**Options considered:**

- Hard-coded download URLs inside scripts
- Manual document downloads without metadata
- Version-controlled source manifest

**Selected option:**

Version-controlled source manifest.

**Why:**

The manifest separates source approval from downloading and extraction. It also
creates stable document identifiers and an explicit source-domain allowlist.

**Trade-offs:**

Adding or replacing a document requires a reviewed manifest change before
ingestion can proceed.

**How we will verify it:**

Manifest validation must pass before any download command runs. The downloader
will accept only records from the validated manifest.

**Date:**

2026-07-18
