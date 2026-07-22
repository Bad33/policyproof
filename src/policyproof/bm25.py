"""Deterministic plain-Python BM25 lexical retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

LEXICAL_TERM_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class BM25Error(ValueError):
    """Raised when BM25 input or configuration is invalid."""


@dataclass(frozen=True)
class BM25Parameters:
    """Explicit BM25 scoring parameters."""

    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise BM25Error("k1 must be a finite number greater than zero.")

        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise BM25Error("b must be a finite number between zero and one.")


@dataclass(frozen=True)
class BM25Document:
    """One indexed passage in accepted corpus order."""

    passage_id: str
    accepted_order: int
    term_frequencies: Mapping[str, int]
    length: int


@dataclass(frozen=True)
class BM25Index:
    """Immutable data required for deterministic BM25 ranking."""

    documents: tuple[BM25Document, ...]
    document_frequencies: Mapping[str, int]
    average_document_length: float
    parameters: BM25Parameters


@dataclass(frozen=True)
class BM25Hit:
    """One ranked BM25 result."""

    passage_id: str
    score: float
    accepted_order: int


def lexical_terms(text: str) -> tuple[str, ...]:
    """Return NFKC-normalized lowercase ASCII letter-and-digit terms."""

    if not isinstance(text, str):
        raise BM25Error("Lexical input must be a string.")

    normalized = unicodedata.normalize("NFKC", text).lower()
    return tuple(LEXICAL_TERM_PATTERN.findall(normalized))


def _unique_terms_in_order(terms: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate terms while preserving first-seen order."""

    result: list[str] = []
    seen: set[str] = set()

    for term in terms:
        if term not in seen:
            seen.add(term)
            result.append(term)

    return tuple(result)


def build_bm25_index(
    passages: Sequence[Mapping[str, Any]],
    *,
    parameters: BM25Parameters | None = None,
) -> BM25Index:
    """Build a deterministic BM25 index from passages in accepted order."""

    if not passages:
        raise BM25Error("At least one passage is required.")

    effective_parameters = parameters or BM25Parameters()
    documents: list[BM25Document] = []
    document_frequencies: Counter[str] = Counter()
    total_document_length = 0
    seen_passage_ids: set[str] = set()

    for accepted_order, passage in enumerate(passages):
        if not isinstance(passage, Mapping):
            raise BM25Error(
                f"Passage at accepted order {accepted_order} must be a mapping."
            )

        passage_id = passage.get("passage_id")
        retrieval_text = passage.get("retrieval_text")

        if not isinstance(passage_id, str) or not passage_id:
            raise BM25Error(
                f"Passage at accepted order {accepted_order} requires "
                "a non-empty passage_id."
            )

        if passage_id in seen_passage_ids:
            raise BM25Error(f"Duplicate passage_id: {passage_id}")

        if not isinstance(retrieval_text, str) or not retrieval_text.strip():
            raise BM25Error(
                f"{passage_id}: retrieval_text must be a non-empty string."
            )

        seen_passage_ids.add(passage_id)
        terms = lexical_terms(retrieval_text)

        if not terms:
            raise BM25Error(
                f"{passage_id}: retrieval_text must contain "
                "at least one lexical term."
            )

        term_frequencies = Counter(terms)

        for term in term_frequencies:
            document_frequencies[term] += 1

        document = BM25Document(
            passage_id=passage_id,
            accepted_order=accepted_order,
            term_frequencies=dict(term_frequencies),
            length=len(terms),
        )
        documents.append(document)
        total_document_length += document.length

    return BM25Index(
        documents=tuple(documents),
        document_frequencies=dict(document_frequencies),
        average_document_length=total_document_length / len(documents),
        parameters=effective_parameters,
    )


def _inverse_document_frequency(
    *,
    document_count: int,
    document_frequency: int,
) -> float:
    """Calculate the contracted BM25 inverse-document-frequency value."""

    return math.log(
        1
        + (
            document_count
            - document_frequency
            + 0.5
        )
        / (
            document_frequency
            + 0.5
        )
    )


def rank_bm25(
    index: BM25Index,
    query: str,
    *,
    limit: int,
) -> tuple[BM25Hit, ...]:
    """Rank all indexed passages using unique first-seen query terms."""

    if not isinstance(index, BM25Index):
        raise BM25Error("index must be a BM25Index.")

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise BM25Error("limit must be a positive integer.")

    query_terms = _unique_terms_in_order(lexical_terms(query))
    document_count = len(index.documents)
    parameters = index.parameters
    hits: list[BM25Hit] = []

    for document in index.documents:
        score = 0.0

        if document.length > 0 and index.average_document_length > 0:
            length_normalization = (
                1
                - parameters.b
                + parameters.b
                * document.length
                / index.average_document_length
            )

            for term in query_terms:
                term_frequency = document.term_frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = index.document_frequencies[term]
                inverse_document_frequency = _inverse_document_frequency(
                    document_count=document_count,
                    document_frequency=document_frequency,
                )
                score += inverse_document_frequency * (
                    term_frequency
                    * (parameters.k1 + 1)
                    / (
                        term_frequency
                        + parameters.k1
                        * length_normalization
                    )
                )

        hits.append(
            BM25Hit(
                passage_id=document.passage_id,
                score=score,
                accepted_order=document.accepted_order,
            )
        )

    hits.sort(
        key=lambda hit: (
            -hit.score,
            hit.accepted_order,
            hit.passage_id,
        )
    )

    return tuple(hits[:limit])
