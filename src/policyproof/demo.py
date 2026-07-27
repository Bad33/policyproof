"""Portable BM25-backed PolicyProof CLI and local web demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from policyproof.bm25 import BM25Hit, build_bm25_index, rank_bm25
from policyproof.evidence_sufficiency_silver_evaluation import (
    FEATURE_NAMES,
    extract_features,
)

EXPECTED_PASSAGE_SHA256 = "5ca1db8d2dd56b92d378bdf315bad25ef83029b4d18017b3755f287bbc26bf96"
EXPECTED_BASELINE_SHA256 = "847c4dedb37ea5c77c94bdf964d408c8685387f43cc57c3acc90a5170146be80"
PASSAGE_RELATIVE_PATH = Path("data/processed/retrieval-passages.jsonl")
BASELINE_RELATIVE_PATH = Path("data/evaluation/evidence-sufficiency-silver-baseline-v0.1.0.json")
MAX_QUERY_CHARACTERS = 2_000
MAX_REQUEST_BYTES = 16_384
DEFAULT_LIMIT = 5
MAX_LIMIT = 10

RESPONSIBLE_USE_NOTICE = (
    "PolicyProof is a research and compliance-support demo. It does not "
    "provide legal advice, determine compliance, or replace qualified review."
)
RANKING_DISCLOSURE = (
    "The benchmark-selected dense ranker requires a local, uncommitted model "
    "asset. This portable demo uses deterministic BM25 so it runs from the "
    "repository without downloading models."
)
METRIC_DISCLOSURE = (
    "The sufficiency threshold was evaluated on construction-derived silver "
    "labels, not independently adjudicated human gold labels."
)


class PolicyProofDemoError(ValueError):
    """Raised when portable demo inputs or accepted assets are invalid."""


@dataclass(frozen=True)
class DemoPaths:
    """Accepted repository paths used by the portable demo."""

    root: Path
    passages: Path
    baseline: Path


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _load_json(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise PolicyProofDemoError(f"{file_path} must contain a JSON object.")

    return value


def _load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise PolicyProofDemoError(f"{file_path}:{line_number} must contain a JSON object.")

        records.append(value)

    if not records:
        raise PolicyProofDemoError(f"{file_path} must contain passages.")

    return records


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_paths(root: Path) -> DemoPaths:
    resolved_root = root.expanduser().resolve()

    return DemoPaths(
        root=resolved_root,
        passages=resolved_root / PASSAGE_RELATIVE_PATH,
        baseline=resolved_root / BASELINE_RELATIVE_PATH,
    )


def _validate_baseline(baseline: dict[str, Any]) -> None:
    if baseline.get("schema_version") != "1.0":
        raise PolicyProofDemoError("Unsupported sufficiency baseline schema.")

    if baseline.get("result_id") != ("policyproof-evidence-sufficiency-silver-baseline"):
        raise PolicyProofDemoError("Unsupported sufficiency baseline ID.")

    if baseline.get("result_version") != "0.1.0":
        raise PolicyProofDemoError("Unsupported sufficiency baseline version.")

    if baseline.get("label_provenance") != "construction_derived":
        raise PolicyProofDemoError(
            "The portable demo requires disclosed construction-derived labels."
        )

    if tuple(baseline.get("feature_names", ())) != FEATURE_NAMES:
        raise PolicyProofDemoError("Sufficiency feature contract mismatch.")

    training = baseline.get("training")

    if not isinstance(training, dict):
        raise PolicyProofDemoError("Sufficiency baseline requires training metadata.")

    for field_name in (
        "feature_means",
        "feature_scales",
        "coefficients",
    ):
        values = training.get(field_name)

        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise PolicyProofDemoError(f"training.{field_name} must match the feature contract.")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            raise PolicyProofDemoError(f"training.{field_name} must contain finite numbers.")

    for field_name in ("intercept", "selected_threshold"):
        value = training.get(field_name)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise PolicyProofDemoError(f"training.{field_name} must be finite.")

    threshold = float(training["selected_threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise PolicyProofDemoError("training.selected_threshold must be between zero and one.")

    if any(float(scale) <= 0.0 for scale in training["feature_scales"]):
        raise PolicyProofDemoError("Feature scales must be greater than zero.")


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)

    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class PolicyProofDemo:
    """In-memory portable retrieval, sufficiency, and citation pipeline."""

    def __init__(
        self,
        passages: list[dict[str, Any]],
        baseline: dict[str, Any],
    ) -> None:
        _validate_baseline(baseline)

        passages_by_id = {
            str(passage["passage_id"]): passage
            for passage in passages
        }

        if len(passages_by_id) != len(passages):
            raise PolicyProofDemoError("Passages contain duplicate passage IDs.")

        eligible_passages = [
            passage
            for passage in passages
            if not (
                (
                    "reference_entry_start_ordinal" in passage
                    and "reference_entry_end_ordinal" in passage
                )
                or (
                    passage.get("unit_kind") == "heading_body"
                    and len(passage["citation_text"]) <= 120
                )
            )
        ]

        self._passages = tuple(eligible_passages)
        self._passages_by_id = {
            str(passage["passage_id"]): passage
            for passage in eligible_passages
        }
        self._baseline = baseline
        self._index = build_bm25_index(eligible_passages)

    @classmethod
    def from_repository(
        cls,
        root: Path | None = None,
    ) -> "PolicyProofDemo":
        paths = _resolve_paths(root or _repository_root())

        if not paths.passages.is_file():
            raise PolicyProofDemoError(f"Accepted passage artifact not found: {paths.passages}")

        if not paths.baseline.is_file():
            raise PolicyProofDemoError(f"Accepted sufficiency baseline not found: {paths.baseline}")

        if _sha256_file(paths.passages) != EXPECTED_PASSAGE_SHA256:
            raise PolicyProofDemoError("Accepted passage SHA-256 mismatch.")

        if _sha256_file(paths.baseline) != EXPECTED_BASELINE_SHA256:
            raise PolicyProofDemoError("Accepted baseline SHA-256 mismatch.")

        return cls(
            _load_jsonl(paths.passages),
            _load_json(paths.baseline),
        )

    def _probability_sufficient(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[float, tuple[float, ...]]:
        case = {
            "question": question,
            "evidence": evidence,
        }
        features = extract_features(case)
        training = self._baseline["training"]
        means = [float(value) for value in training["feature_means"]]
        scales = [float(value) for value in training["feature_scales"]]
        coefficients = [float(value) for value in training["coefficients"]]
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(
                features,
                means,
                scales,
                strict=True,
            )
        ]
        logit = float(training["intercept"]) + sum(
            coefficient * value
            for coefficient, value in zip(
                coefficients,
                standardized,
                strict=True,
            )
        )

        return _sigmoid(logit), features

    def _selected_hits(
        self,
        hits: tuple[BM25Hit, ...],
    ) -> tuple[BM25Hit, ...]:
        positive_hits = tuple(hit for hit in hits if hit.score > 0.0)

        if not positive_hits:
            return ()

        selected = [positive_hits[0]]

        if len(positive_hits) > 1:
            first_passage = self._passages_by_id[
                positive_hits[0].passage_id
            ]
            second_passage = self._passages_by_id[
                positive_hits[1].passage_id
            ]
            first_source = first_passage.get(
                "logical_source_key"
            )
            second_source = second_passage.get(
                "logical_source_key"
            )

            if (
                first_source
                and second_source == first_source
            ):
                selected.append(positive_hits[1])

        return tuple(selected)

    def query(
        self,
        question: str,
        *,
        limit: int = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise PolicyProofDemoError("question must be a nonempty string.")

        normalized_question = question.strip()

        if len(normalized_question) > MAX_QUERY_CHARACTERS:
            raise PolicyProofDemoError(
                f"question must not exceed {MAX_QUERY_CHARACTERS} characters."
            )

        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
            raise PolicyProofDemoError(f"limit must be an integer between 1 and {MAX_LIMIT}.")

        ranked_hits = rank_bm25(
            self._index,
            normalized_question,
            limit=limit,
        )
        selected_hits = self._selected_hits(ranked_hits)

        if not selected_hits:
            return {
                "query": normalized_question,
                "action": "abstain",
                "reason": "no_positive_lexical_match",
                "answer": (
                    "I could not find a positive lexical match in the "
                    "controlled PolicyProof corpus."
                ),
                "missing_information": [
                    "Evidence matching the question is absent from the controlled corpus."
                ],
                "evidence_sufficiency": {
                    "probability_sufficient": 0.0,
                    "selected_threshold": float(self._baseline["training"]["selected_threshold"]),
                    "predicted_status": "insufficient",
                    "label_provenance": "construction_derived",
                },
                "retrieval": {
                    "method": "bm25_portable_demo",
                    "candidate_limit": limit,
                    "positive_candidate_count": 0,
                    "selected_passage_count": 0,
                    "citations": [],
                },
                "ranking_disclosure": RANKING_DISCLOSURE,
                "metric_disclosure": METRIC_DISCLOSURE,
                "responsible_use_notice": RESPONSIBLE_USE_NOTICE,
            }

        evidence: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []

        for citation_number, hit in enumerate(
            selected_hits,
            start=1,
        ):
            passage = self._passages_by_id[hit.passage_id]
            evidence_record = {
                "passage_id": passage["passage_id"],
                "document_id": passage["document_id"],
                "label": passage["label"],
                "citation_text": passage["citation_text"],
            }
            evidence.append(evidence_record)
            citations.append(
                {
                    "citation_number": citation_number,
                    **evidence_record,
                    "bm25_score": round(hit.score, 12),
                }
            )

        probability, features = self._probability_sufficient(
            normalized_question,
            evidence,
        )
        threshold = float(self._baseline["training"]["selected_threshold"])
        sufficient = probability >= threshold
        predicted_status = "sufficient" if sufficient else "insufficient"

        if sufficient:
            action = "answer"
            reason = "evidence_passed_silver_sufficiency_threshold"
            answer = "\n\n".join(
                f"[{item['citation_number']}] {item['citation_text']}" for item in citations
            )
            missing_information: list[str] = []
        else:
            action = "abstain"
            reason = "evidence_below_silver_sufficiency_threshold"
            answer = (
                "I found related evidence, but it did not pass the "
                "evidence-sufficiency threshold. Review the excerpts below "
                "or refine the question."
            )
            missing_information = [
                "Additional directly responsive evidence is required before the system can answer."
            ]

        return {
            "query": normalized_question,
            "action": action,
            "reason": reason,
            "answer": answer,
            "missing_information": missing_information,
            "evidence_sufficiency": {
                "probability_sufficient": round(probability, 12),
                "selected_threshold": round(threshold, 12),
                "predicted_status": predicted_status,
                "label_provenance": "construction_derived",
                "feature_names": list(FEATURE_NAMES),
                "feature_values": [round(value, 12) for value in features],
            },
            "retrieval": {
                "method": "bm25_portable_demo",
                "candidate_limit": limit,
                "positive_candidate_count": sum(hit.score > 0.0 for hit in ranked_hits),
                "selected_passage_count": len(citations),
                "citations": citations,
            },
            "ranking_disclosure": RANKING_DISCLOSURE,
            "metric_disclosure": METRIC_DISCLOSURE,
            "responsible_use_notice": RESPONSIBLE_USE_NOTICE,
        }


def render_home_page() -> str:
    """Return the self-contained local demo interface."""

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PolicyProof</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: #f4f6f8;
      color: #17202a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main {
      width: min(980px, calc(100% - 32px));
      margin: 36px auto 64px;
    }
    header, section {
      background: #ffffff;
      border: 1px solid #dde3e8;
      border-radius: 16px;
      box-shadow: 0 8px 28px rgba(20, 33, 45, 0.06);
    }
    header { padding: 28px; }
    section { margin-top: 18px; padding: 24px; }
    h1 { margin: 0 0 8px; font-size: 2rem; }
    h2 { margin-top: 0; }
    p { line-height: 1.55; }
    .badge {
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      background: #eaf3ff;
      color: #164b7a;
      font-size: 0.8rem;
      font-weight: 700;
    }
    textarea {
      width: 100%;
      min-height: 110px;
      padding: 14px;
      border: 1px solid #b9c4ce;
      border-radius: 10px;
      font: inherit;
      resize: vertical;
    }
    button {
      margin-top: 12px;
      padding: 11px 18px;
      border: 0;
      border-radius: 10px;
      background: #17202a;
      color: #ffffff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .status { font-weight: 800; text-transform: uppercase; }
    .answer {
      white-space: pre-wrap;
      padding: 16px;
      background: #f7f9fb;
      border-radius: 10px;
      line-height: 1.55;
    }
    article {
      border-top: 1px solid #e4e9ed;
      padding: 16px 0;
    }
    article:first-child { border-top: 0; }
    code { word-break: break-all; }
    .muted { color: #5f6b76; }
    .error { color: #a61b1b; }
  </style>
</head>
<body>
<main>
  <header>
    <span class="badge">Evidence-first AI governance RAG</span>
    <h1>PolicyProof</h1>
    <p>
      Ask a question about the controlled NIST, EU AI Act, and OpenAI system-card
      corpus. The portable demo retrieves passages, estimates evidence
      sufficiency, returns source-derived excerpts with citations, or abstains.
    </p>
  </header>

  <section>
    <h2>Ask the corpus</h2>
    <textarea id="question">What risks does unauthorized voice generation create, and how does GPT-4o mitigate them?</textarea>
    <br>
    <button id="submit">Run PolicyProof</button>
    <p id="error" class="error"></p>
  </section>

  <section id="result" hidden>
    <h2>Decision</h2>
    <p class="status" id="action"></p>
    <p id="reason" class="muted"></p>
    <div id="answer" class="answer"></div>
    <p id="probability" class="muted"></p>
  </section>

  <section id="evidence" hidden>
    <h2>Retrieved evidence</h2>
    <div id="citations"></div>
  </section>

  <section>
    <h2>Important disclosures</h2>
    <p><strong>Portable ranking:</strong> deterministic BM25 is used because the
      benchmark-selected dense model asset is intentionally not committed.</p>
    <p><strong>Evaluation:</strong> sufficiency results use
      construction-derived silver labels, not human-adjudicated gold labels.</p>
    <p><strong>Responsible use:</strong> this is research and compliance support,
      not legal advice or a compliance determination.</p>
  </section>
</main>
<script>
const button = document.getElementById("submit");
const question = document.getElementById("question");
const error = document.getElementById("error");
const result = document.getElementById("result");
const evidence = document.getElementById("evidence");

function text(id, value) {
  document.getElementById(id).textContent = value;
}

button.addEventListener("click", async () => {
  button.disabled = true;
  error.textContent = "";
  result.hidden = true;
  evidence.hidden = true;

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: question.value})
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Request failed.");
    }

    text("action", payload.action);
    text("reason", payload.reason);
    text("answer", payload.answer);
    text(
      "probability",
      `Sufficiency probability ${payload.evidence_sufficiency.probability_sufficient} · threshold ${payload.evidence_sufficiency.selected_threshold}`
    );

    const citations = document.getElementById("citations");
    citations.replaceChildren();

    for (const item of payload.retrieval.citations) {
      const article = document.createElement("article");
      const heading = document.createElement("strong");
      heading.textContent = `[${item.citation_number}] ${item.label}`;
      const meta = document.createElement("p");
      meta.className = "muted";
      meta.textContent = `${item.document_id} · BM25 ${item.bm25_score}`;
      const excerpt = document.createElement("p");
      excerpt.textContent = item.citation_text;
      article.append(heading, meta, excerpt);
      citations.append(article);
    }

    result.hidden = false;
    evidence.hidden = payload.retrieval.citations.length === 0;
  } catch (requestError) {
    error.textContent = requestError.message;
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


def _handler_class(app: PolicyProofDemo) -> type[BaseHTTPRequestHandler]:
    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "PolicyProofDemo/0.1"

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = render_home_page().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "service": "policyproof-portable-demo",
                        "ranking_method": "bm25_portable_demo",
                        "label_provenance": "construction_derived",
                    }
                )
                return

            self._send_json(
                {"error": "Not found."},
                status=HTTPStatus.NOT_FOUND,
            )

        def do_POST(self) -> None:
            if self.path != "/api/query":
                self._send_json(
                    {"error": "Not found."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            content_length_text = self.headers.get("Content-Length", "0")

            try:
                content_length = int(content_length_text)
            except ValueError:
                self._send_json(
                    {"error": "Invalid Content-Length."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            if not 0 < content_length <= MAX_REQUEST_BYTES:
                self._send_json(
                    {"error": "Request body size is invalid."},
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return

            try:
                raw_body = self.rfile.read(content_length)
                request = json.loads(raw_body.decode("utf-8"))

                if not isinstance(request, dict):
                    raise PolicyProofDemoError("Request body must be a JSON object.")

                question_value = request.get("question")
                limit_value = request.get("limit", DEFAULT_LIMIT)
                payload = app.query(
                    question_value,
                    limit=limit_value,
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                PolicyProofDemoError,
            ) as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            self._send_json(payload)

    return DemoRequestHandler


def serve(
    app: PolicyProofDemo,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise PolicyProofDemoError("port must be between 1 and 65535.")

    server = ThreadingHTTPServer(
        (host, port),
        _handler_class(app),
    )
    url = f"http://{host}:{port}/"
    print(f"PolicyProof demo: {url}")
    print("Press Ctrl-C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PolicyProof demo.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run the portable PolicyProof retrieval, sufficiency, and citation demo.")
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_repository_root(),
        help="PolicyProof repository root.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Run one question and print JSON.",
    )
    query_parser.add_argument("question")
    query_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the local browser demo.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    serve_parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
    )

    args = parser.parse_args()
    app = PolicyProofDemo.from_repository(args.root)

    if args.command == "query":
        print(
            json.dumps(
                app.query(
                    args.question,
                    limit=args.limit,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    serve(
        app,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()
