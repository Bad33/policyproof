"""Pinned offline tokenizer contract for PolicyProof retrieval text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import BertProcessing


class RetrievalTokenizerError(RuntimeError):
    """Raised when the retrieval tokenizer contract cannot be verified."""


@dataclass(frozen=True)
class RetrievalTokenizerContract:
    """Immutable tokenizer configuration shared by retrieval models."""

    library_name: str
    library_version: str
    implementation: str
    vocab_source_model: str
    vocab_source_revision: str
    vocab_resource: str
    vocab_license_resource: str
    vocab_notice_resource: str
    vocab_sha256: str
    vocab_size: int
    model_max_length: int
    lowercase: bool
    strip_accents: bool | None
    tokenize_chinese_chars: bool
    unknown_token: str
    separator_token: str
    padding_token: str
    classification_token: str
    mask_token: str


TOKENIZER_CONTRACT = RetrievalTokenizerContract(
    library_name="tokenizers",
    library_version="0.22.2",
    implementation="Tokenizer+WordPiece.from_file",
    vocab_source_model="google-bert/bert-base-uncased",
    vocab_source_revision=(
        "86b5e0934494bd15c9632b12f734a8a67f723594"
    ),
    vocab_resource="assets/bert-base-uncased-vocab.txt",
    vocab_license_resource=(
        "assets/bert-base-uncased-LICENSE.txt"
    ),
    vocab_notice_resource=(
        "assets/bert-base-uncased-NOTICE.txt"
    ),
    vocab_sha256=(
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
    ),
    vocab_size=30_522,
    model_max_length=512,
    lowercase=True,
    strip_accents=None,
    tokenize_chinese_chars=True,
    unknown_token="[UNK]",
    separator_token="[SEP]",
    padding_token="[PAD]",
    classification_token="[CLS]",
    mask_token="[MASK]",
)


def validate_tokenizer_contract(
    *,
    installed_version: str,
    vocab_bytes: bytes,
) -> None:
    """Validate the exact library and vocabulary used for token counting."""

    if installed_version != TOKENIZER_CONTRACT.library_version:
        raise RetrievalTokenizerError(
            "Installed tokenizers version "
            f"{installed_version!r} does not match pinned version "
            f"{TOKENIZER_CONTRACT.library_version!r}."
        )

    actual_hash = hashlib.sha256(vocab_bytes).hexdigest()

    if actual_hash != TOKENIZER_CONTRACT.vocab_sha256:
        raise RetrievalTokenizerError(
            "Tokenizer vocabulary SHA-256 "
            f"{actual_hash} does not match pinned value "
            f"{TOKENIZER_CONTRACT.vocab_sha256}."
        )

    try:
        entries = vocab_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RetrievalTokenizerError(
            "Tokenizer vocabulary is not valid UTF-8."
        ) from error

    if len(entries) != TOKENIZER_CONTRACT.vocab_size:
        raise RetrievalTokenizerError(
            "Tokenizer vocabulary contains "
            f"{len(entries)} entries; expected "
            f"{TOKENIZER_CONTRACT.vocab_size}."
        )

    expected_special_tokens = {
        TOKENIZER_CONTRACT.padding_token: 0,
        TOKENIZER_CONTRACT.unknown_token: 100,
        TOKENIZER_CONTRACT.classification_token: 101,
        TOKENIZER_CONTRACT.separator_token: 102,
        TOKENIZER_CONTRACT.mask_token: 103,
    }

    for token, expected_id in expected_special_tokens.items():
        if entries[expected_id] != token:
            raise RetrievalTokenizerError(
                f"Tokenizer vocabulary ID {expected_id} contains "
                f"{entries[expected_id]!r}; expected {token!r}."
            )


def _installed_tokenizers_version() -> str:
    try:
        return version(TOKENIZER_CONTRACT.library_name)
    except PackageNotFoundError as error:
        raise RetrievalTokenizerError(
            "Pinned tokenizers package is not installed."
        ) from error


def _vocab_resource():
    resource = files("policyproof").joinpath(
        TOKENIZER_CONTRACT.vocab_resource
    )

    if not resource.is_file():
        raise RetrievalTokenizerError(
            "Packaged tokenizer vocabulary is missing: "
            f"{TOKENIZER_CONTRACT.vocab_resource}"
        )

    return resource


@lru_cache(maxsize=1)
def retrieval_tokenizer() -> Tokenizer:
    """Load and validate the pinned tokenizer once per process."""

    resource = _vocab_resource()
    vocab_bytes = resource.read_bytes()

    validate_tokenizer_contract(
        installed_version=_installed_tokenizers_version(),
        vocab_bytes=vocab_bytes,
    )

    with as_file(resource) as vocab_path:
        model = WordPiece.from_file(
            str(vocab_path),
            unk_token=TOKENIZER_CONTRACT.unknown_token,
        )

    tokenizer = Tokenizer(model)

    special_tokens = (
        TOKENIZER_CONTRACT.unknown_token,
        TOKENIZER_CONTRACT.separator_token,
        TOKENIZER_CONTRACT.classification_token,
        TOKENIZER_CONTRACT.padding_token,
        TOKENIZER_CONTRACT.mask_token,
    )
    tokenizer.add_special_tokens(list(special_tokens))

    tokenizer.normalizer = BertNormalizer(
        clean_text=True,
        handle_chinese_chars=(
            TOKENIZER_CONTRACT.tokenize_chinese_chars
        ),
        strip_accents=TOKENIZER_CONTRACT.strip_accents,
        lowercase=TOKENIZER_CONTRACT.lowercase,
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()

    separator_id = tokenizer.token_to_id(
        TOKENIZER_CONTRACT.separator_token
    )
    classification_id = tokenizer.token_to_id(
        TOKENIZER_CONTRACT.classification_token
    )

    if separator_id is None or classification_id is None:
        raise RetrievalTokenizerError(
            "Required BERT special tokens are missing from "
            "the pinned vocabulary."
        )

    tokenizer.post_processor = BertProcessing(
        (
            TOKENIZER_CONTRACT.separator_token,
            separator_id,
        ),
        (
            TOKENIZER_CONTRACT.classification_token,
            classification_id,
        ),
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    if tokenizer.get_vocab_size() != TOKENIZER_CONTRACT.vocab_size:
        raise RetrievalTokenizerError(
            "Loaded tokenizer vocabulary size does not match "
            "the pinned contract."
        )

    return tokenizer


def _require_text(text: str) -> None:
    if not isinstance(text, str):
        raise RetrievalTokenizerError(
            "Tokenizer input must be a string."
        )


def tokenize_text(text: str) -> tuple[str, ...]:
    """Return WordPiece tokens without model special tokens."""

    _require_text(text)

    encoding = retrieval_tokenizer().encode(
        text,
        add_special_tokens=False,
    )
    return tuple(encoding.tokens)


def token_ids(
    text: str,
    *,
    add_special_tokens: bool = False,
) -> tuple[int, ...]:
    """Return deterministic token IDs for one text sequence."""

    _require_text(text)

    encoding = retrieval_tokenizer().encode(
        text,
        add_special_tokens=add_special_tokens,
    )
    return tuple(encoding.ids)


def count_tokens(
    text: str,
    *,
    add_special_tokens: bool = False,
) -> int:
    """Count deterministic tokenizer IDs for one text sequence."""

    return len(
        token_ids(
            text,
            add_special_tokens=add_special_tokens,
        )
    )
