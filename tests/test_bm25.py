from __future__ import annotations

import math

import pytest

from policyproof.bm25 import (
    BM25Parameters,
    build_bm25_index,
    lexical_terms,
    rank_bm25,
)


def passage(passage_id: str, retrieval_text: str) -> dict[str, str]:
    return {
        "passage_id": passage_id,
        "retrieval_text": retrieval_text,
    }


def test_lexical_terms_apply_nfkc_lowercase_and_punctuation_splitting() -> None:
    assert lexical_terms(
        "ＡＩ—Risk's third-party controls; ISO/IEC 42001."
    ) == (
        "ai",
        "risk",
        "s",
        "third",
        "party",
        "controls",
        "iso",
        "iec",
        "42001",
    )


def test_bm25_uses_explicit_parameters_and_reviewed_idf_formula() -> None:
    passages = [
        passage("passage-b", "alpha alpha beta"),
        passage("passage-a", "alpha gamma"),
        passage("passage-c", "gamma delta"),
    ]
    parameters = BM25Parameters(k1=1.2, b=0.75)
    index = build_bm25_index(
        passages,
        parameters=parameters,
    )

    hits = rank_bm25(index, "alpha beta", limit=3)

    document_count = 3
    average_length = 7 / 3

    alpha_idf = math.log(
        1 + (document_count - 2 + 0.5) / (2 + 0.5)
    )
    beta_idf = math.log(
        1 + (document_count - 1 + 0.5) / (1 + 0.5)
    )

    length_normalization = (
        1 - parameters.b
        + parameters.b * (3 / average_length)
    )
    alpha_score = alpha_idf * (
        2 * (parameters.k1 + 1)
        / (2 + parameters.k1 * length_normalization)
    )
    beta_score = beta_idf * (
        1 * (parameters.k1 + 1)
        / (1 + parameters.k1 * length_normalization)
    )

    assert [hit.passage_id for hit in hits] == [
        "passage-b",
        "passage-a",
        "passage-c",
    ]
    assert hits[0].score == pytest.approx(
        alpha_score + beta_score
    )


def test_repeated_query_terms_do_not_change_scores_or_ranking() -> None:
    index = build_bm25_index(
        [
            passage("passage-1", "risk risk management"),
            passage("passage-2", "risk governance"),
            passage("passage-3", "unrelated material"),
        ]
    )

    single = rank_bm25(index, "risk management", limit=3)
    repeated = rank_bm25(
        index,
        "risk risk management risk",
        limit=3,
    )

    assert repeated == single


def test_equal_scores_preserve_accepted_passage_order() -> None:
    index = build_bm25_index(
        [
            passage("passage-z", "alpha"),
            passage("passage-a", "beta"),
            passage("passage-m", "gamma"),
        ]
    )

    hits = rank_bm25(index, "unknown-term", limit=3)

    assert [hit.passage_id for hit in hits] == [
        "passage-z",
        "passage-a",
        "passage-m",
    ]
    assert [hit.score for hit in hits] == [0.0, 0.0, 0.0]



def test_index_rejects_passage_without_lexical_terms() -> None:
    with pytest.raises(
        ValueError,
        match="at least one lexical term",
    ):
        build_bm25_index(
            [
                passage("passage-1", "--- ... !!!"),
            ]
        )
