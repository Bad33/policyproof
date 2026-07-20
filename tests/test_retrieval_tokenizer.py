from __future__ import annotations

import hashlib
from importlib.resources import files

import pytest

from policyproof.retrieval_tokenizer import (
    TOKENIZER_CONTRACT,
    RetrievalTokenizerError,
    count_tokens,
    token_ids,
    tokenize_text,
    validate_tokenizer_contract,
)


def vendored_vocab_bytes() -> bytes:
    return (
        files("policyproof")
        .joinpath(TOKENIZER_CONTRACT.vocab_resource)
        .read_bytes()
    )


def test_tokenizer_contract_is_fully_pinned() -> None:
    assert TOKENIZER_CONTRACT.library_name == "tokenizers"
    assert TOKENIZER_CONTRACT.library_version == "0.22.2"
    assert (
        TOKENIZER_CONTRACT.implementation
        == "Tokenizer+WordPiece.from_file"
    )
    assert (
        TOKENIZER_CONTRACT.vocab_source_model
        == "google-bert/bert-base-uncased"
    )
    assert (
        TOKENIZER_CONTRACT.vocab_source_revision
        == "86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    assert (
        TOKENIZER_CONTRACT.vocab_resource
        == "assets/bert-base-uncased-vocab.txt"
    )
    assert (
        TOKENIZER_CONTRACT.vocab_license_resource
        == "assets/bert-base-uncased-LICENSE.txt"
    )
    assert (
        TOKENIZER_CONTRACT.vocab_notice_resource
        == "assets/bert-base-uncased-NOTICE.txt"
    )
    assert (
        TOKENIZER_CONTRACT.vocab_sha256
        == "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
    )
    assert TOKENIZER_CONTRACT.vocab_size == 30_522
    assert TOKENIZER_CONTRACT.model_max_length == 512
    assert TOKENIZER_CONTRACT.lowercase is True
    assert TOKENIZER_CONTRACT.strip_accents is None
    assert TOKENIZER_CONTRACT.tokenize_chinese_chars is True
    assert TOKENIZER_CONTRACT.unknown_token == "[UNK]"
    assert TOKENIZER_CONTRACT.separator_token == "[SEP]"
    assert TOKENIZER_CONTRACT.padding_token == "[PAD]"
    assert TOKENIZER_CONTRACT.classification_token == "[CLS]"
    assert TOKENIZER_CONTRACT.mask_token == "[MASK]"


def test_vendored_vocab_matches_the_pinned_contract() -> None:
    content = vendored_vocab_bytes()
    entries = content.decode("utf-8").splitlines()

    assert hashlib.sha256(content).hexdigest() == (
        TOKENIZER_CONTRACT.vocab_sha256
    )
    assert len(entries) == TOKENIZER_CONTRACT.vocab_size
    assert entries[:3] == [
        "[PAD]",
        "[unused0]",
        "[unused1]",
    ]
    assert entries[-3:] == [
        "##：",
        "##？",
        "##～",
    ]


def test_contract_validation_accepts_only_exact_assets() -> None:
    content = vendored_vocab_bytes()

    validate_tokenizer_contract(
        installed_version="0.22.2",
        vocab_bytes=content,
    )

    with pytest.raises(
        RetrievalTokenizerError,
        match="tokenizers version",
    ):
        validate_tokenizer_contract(
            installed_version="0.22.3",
            vocab_bytes=content,
        )

    with pytest.raises(
        RetrievalTokenizerError,
        match="vocabulary SHA-256",
    ):
        validate_tokenizer_contract(
            installed_version="0.22.2",
            vocab_bytes=content + b"\ncorrupt",
        )


def test_tokenization_matches_the_reviewed_bert_contract() -> None:
    assert tokenize_text(
        "organization’s responsibility"
    ) == (
        "organization",
        "’",
        "s",
        "responsibility",
    )
    assert tokenize_text(
        "café naïve résumé"
    ) == (
        "cafe",
        "naive",
        "resume",
    )
    assert token_ids(
        "AI safety 🤖"
    ) == (
        9932,
        3808,
        100,
    )


def test_special_tokens_are_explicitly_accounted_for() -> None:
    assert token_ids(
        "AI safety",
        add_special_tokens=False,
    ) == (
        9932,
        3808,
    )
    assert token_ids(
        "AI safety",
        add_special_tokens=True,
    ) == (
        101,
        9932,
        3808,
        102,
    )
    assert count_tokens(
        "AI safety",
        add_special_tokens=False,
    ) == 2
    assert count_tokens(
        "AI safety",
        add_special_tokens=True,
    ) == 4

def test_packaged_attribution_files_are_present() -> None:
    package_files = files("policyproof")

    license_text = package_files.joinpath(
        TOKENIZER_CONTRACT.vocab_license_resource
    ).read_text(encoding="utf-8")
    notice_text = package_files.joinpath(
        TOKENIZER_CONTRACT.vocab_notice_resource
    ).read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert TOKENIZER_CONTRACT.vocab_source_model in notice_text
    assert TOKENIZER_CONTRACT.vocab_source_revision in notice_text
    assert TOKENIZER_CONTRACT.vocab_sha256 in notice_text
    assert "Apache-2.0" in notice_text
