"""Pinned dense-embedding model contract and deterministic input handling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policyproof.retrieval_tokenizer import retrieval_tokenizer


class DenseModelError(RuntimeError):
    """Raised when the dense-model contract cannot be satisfied."""


@dataclass(frozen=True)
class DenseModelContract:
    """Immutable runtime, model-asset, and embedding behavior contract."""

    runtime_library: str
    runtime_version: str
    array_library: str
    array_version: str
    model_id: str
    model_revision: str
    model_filename: str
    model_size_bytes: int
    model_sha256: str
    license_id: str
    execution_provider: str
    max_sequence_length: int
    embedding_dimension: int
    pooling: str
    normalization: str
    query_instruction: str
    input_names: tuple[str, ...]
    output_name: str


DENSE_MODEL_CONTRACT = DenseModelContract(
    runtime_library="onnxruntime",
    runtime_version="1.27.0",
    array_library="numpy",
    array_version="2.5.1",
    model_id="BAAI/bge-small-en-v1.5",
    model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    model_filename="onnx/model.onnx",
    model_size_bytes=133_093_490,
    model_sha256="828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    license_id="mit",
    execution_provider="CPUExecutionProvider",
    max_sequence_length=512,
    embedding_dimension=384,
    pooling="cls",
    normalization="l2",
    query_instruction=(
        "Represent this sentence for searching relevant passages: "
    ),
    input_names=(
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ),
    output_name="last_hidden_state",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DenseModelError(
            f"Unable to read dense model asset: {path}"
        ) from error

    return digest.hexdigest()


def validate_model_asset(
    path: Path,
    *,
    contract: DenseModelContract = DENSE_MODEL_CONTRACT,
) -> None:
    """Require the exact pinned model size and SHA-256."""

    model_path = Path(path)

    if not model_path.is_file():
        raise DenseModelError(
            f"Dense model asset is missing: {model_path}"
        )

    try:
        actual_size = model_path.stat().st_size
    except OSError as error:
        raise DenseModelError(
            f"Unable to inspect dense model asset: {model_path}"
        ) from error

    if actual_size != contract.model_size_bytes:
        raise DenseModelError(
            "Dense model asset size "
            f"{actual_size} does not match pinned size "
            f"{contract.model_size_bytes}."
        )

    actual_sha256 = _sha256_file(model_path)

    if actual_sha256 != contract.model_sha256:
        raise DenseModelError(
            "Dense model asset SHA-256 "
            f"{actual_sha256} does not match pinned value "
            f"{contract.model_sha256}."
        )


def _require_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(
        texts,
        Sequence,
    ):
        raise DenseModelError(
            "Dense model input must be a sequence of strings."
        )

    validated = tuple(texts)

    if not validated:
        raise DenseModelError(
            "Dense model input must contain at least one string."
        )

    for index, text in enumerate(validated):
        if not isinstance(text, str):
            raise DenseModelError(
                f"Dense model input at index {index} must be a string."
            )

        if not text:
            raise DenseModelError(
                f"Dense model input at index {index} must not be empty."
            )

    return validated


def prepare_dense_inputs(
    texts: Sequence[str],
    *,
    is_query: bool,
    contract: DenseModelContract = DENSE_MODEL_CONTRACT,
) -> dict[str, np.ndarray]:
    """Tokenize and pad dense-model inputs without truncation."""

    if not isinstance(is_query, bool):
        raise DenseModelError("is_query must be a boolean.")

    validated_texts = _require_texts(texts)
    tokenizer = retrieval_tokenizer()

    model_texts = (
        tuple(
            contract.query_instruction + text
            for text in validated_texts
        )
        if is_query
        else validated_texts
    )

    encodings = [
        tokenizer.encode(
            text,
            add_special_tokens=True,
        )
        for text in model_texts
    ]
    lengths = [len(encoding.ids) for encoding in encodings]

    for index, length in enumerate(lengths):
        if length > contract.max_sequence_length:
            raise DenseModelError(
                f"Dense model input at index {index} contains "
                f"{length} tokens and exceeds the "
                f"{contract.max_sequence_length}-token limit."
            )

    maximum_length = max(lengths)
    batch_size = len(encodings)

    input_ids = np.zeros(
        (batch_size, maximum_length),
        dtype=np.int64,
    )
    attention_mask = np.zeros_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)

    for row, encoding in enumerate(encodings):
        length = lengths[row]
        input_ids[row, :length] = np.asarray(
            encoding.ids,
            dtype=np.int64,
        )
        attention_mask[row, :length] = 1

    return {
        contract.input_names[0]: input_ids,
        contract.input_names[1]: attention_mask,
        contract.input_names[2]: token_type_ids,
    }


def normalize_embeddings(
    embeddings: np.ndarray,
    *,
    expected_dimension: int,
) -> np.ndarray:
    """Return finite float32 row vectors normalized to unit L2 length."""

    if (
        not isinstance(expected_dimension, int)
        or isinstance(expected_dimension, bool)
        or expected_dimension <= 0
    ):
        raise DenseModelError(
            "expected_dimension must be a positive integer."
        )

    values = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if values.ndim != 2:
        raise DenseModelError(
            "Embeddings must be a rank-2 array."
        )

    if values.shape[1] != expected_dimension:
        raise DenseModelError(
            "Embedding dimension "
            f"{values.shape[1]} does not match expected dimension "
            f"{expected_dimension}."
        )

    if not np.all(np.isfinite(values)):
        raise DenseModelError(
            "Embeddings must contain only finite values."
        )

    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise DenseModelError(
            "Cannot normalize a zero-length embedding."
        )

    return np.asarray(
        values / norms,
        dtype=np.float32,
    )


def validate_runtime_contract(
    *,
    runtime_version: str,
    array_version: str,
    contract: DenseModelContract = DENSE_MODEL_CONTRACT,
) -> None:
    """Require the exact numerical and ONNX runtime versions."""

    if runtime_version != contract.runtime_version:
        raise DenseModelError(
            "Installed onnxruntime version "
            f"{runtime_version!r} does not match pinned version "
            f"{contract.runtime_version!r}."
        )

    if array_version != contract.array_version:
        raise DenseModelError(
            "Installed numpy version "
            f"{array_version!r} does not match pinned version "
            f"{contract.array_version!r}."
        )


def validate_session_contract(
    session: object,
    *,
    contract: DenseModelContract = DENSE_MODEL_CONTRACT,
) -> None:
    """Validate the provider and public ONNX input/output interface."""

    try:
        providers = tuple(session.get_providers())  # type: ignore[attr-defined]
        inputs = tuple(session.get_inputs())  # type: ignore[attr-defined]
        outputs = tuple(session.get_outputs())  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise DenseModelError(
            "Dense model session does not expose the required ONNX interface."
        ) from error

    if providers != (contract.execution_provider,):
        raise DenseModelError(
            "Dense model session must use only "
            f"{contract.execution_provider}; received {providers!r}."
        )

    expected_input_contract = tuple(
        (
            name,
            "tensor(int64)",
            2,
        )
        for name in contract.input_names
    )
    actual_input_contract = tuple(
        (
            getattr(node, "name", None),
            getattr(node, "type", None),
            len(getattr(node, "shape", ())),
        )
        for node in inputs
    )

    if actual_input_contract != expected_input_contract:
        raise DenseModelError(
            "Dense model input contract does not match the pinned "
            f"interface: {actual_input_contract!r}."
        )

    if len(outputs) != 1:
        raise DenseModelError(
            "Dense model output contract must contain exactly one output."
        )

    output = outputs[0]
    output_name = getattr(output, "name", None)
    output_type = getattr(output, "type", None)
    output_shape = getattr(output, "shape", ())

    if (
        output_name != contract.output_name
        or output_type != "tensor(float)"
        or len(output_shape) != 3
        or output_shape[-1] != contract.embedding_dimension
    ):
        raise DenseModelError(
            "Dense model output contract does not match the pinned "
            "last-hidden-state interface."
        )


def embed_dense_texts(
    session: object,
    texts: Sequence[str],
    *,
    is_query: bool,
    contract: DenseModelContract = DENSE_MODEL_CONTRACT,
) -> np.ndarray:
    """Run validated CLS-pooled, L2-normalized dense embedding inference."""

    validate_session_contract(
        session,
        contract=contract,
    )
    inputs = prepare_dense_inputs(
        texts,
        is_query=is_query,
        contract=contract,
    )

    try:
        outputs = session.run(  # type: ignore[attr-defined]
            [contract.output_name],
            inputs,
        )
    except Exception as error:
        raise DenseModelError(
            "Dense model ONNX inference failed."
        ) from error

    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise DenseModelError(
            "Dense model inference must return exactly one output."
        )

    hidden_state = np.asarray(outputs[0])

    if hidden_state.ndim != 3:
        raise DenseModelError(
            "Dense model output must be a rank-3 last hidden state."
        )

    expected_batch_size = inputs[contract.input_names[0]].shape[0]

    if hidden_state.shape[0] != expected_batch_size:
        raise DenseModelError(
            "Dense model output batch size does not match its input batch."
        )

    if hidden_state.shape[1] < 1:
        raise DenseModelError(
            "Dense model output contains no CLS token position."
        )

    if hidden_state.shape[2] != contract.embedding_dimension:
        raise DenseModelError(
            "Dense model output embedding dimension "
            f"{hidden_state.shape[2]} does not match pinned dimension "
            f"{contract.embedding_dimension}."
        )

    cls_embeddings = hidden_state[:, 0, :]

    return normalize_embeddings(
        cls_embeddings,
        expected_dimension=contract.embedding_dimension,
    )


def create_dense_session(
    model_path: Path,
    *,
    contract: DenseModelContract = DENSE_MODEL_CONTRACT,
    runtime: object | None = None,
    runtime_version: str | None = None,
    array_version: str | None = None,
) -> object:
    """Create one hash-verified, CPU-only ONNX inference session."""

    validated_path = Path(model_path)

    validate_model_asset(
        validated_path,
        contract=contract,
    )

    selected_runtime = runtime

    if selected_runtime is None:
        try:
            import onnxruntime as selected_runtime
        except ImportError as error:
            raise DenseModelError(
                "Pinned onnxruntime package is not installed."
            ) from error

    selected_runtime_version = runtime_version

    if selected_runtime_version is None:
        selected_runtime_version = getattr(
            selected_runtime,
            "__version__",
            None,
        )

    if not isinstance(selected_runtime_version, str):
        raise DenseModelError(
            "Unable to determine installed onnxruntime version."
        )

    selected_array_version = (
        np.__version__
        if array_version is None
        else array_version
    )

    validate_runtime_contract(
        runtime_version=selected_runtime_version,
        array_version=selected_array_version,
        contract=contract,
    )

    try:
        session = selected_runtime.InferenceSession(  # type: ignore[attr-defined]
            str(validated_path),
            providers=[contract.execution_provider],
        )
    except Exception as error:
        raise DenseModelError(
            "Dense model ONNX session creation failed."
        ) from error

    validate_session_contract(
        session,
        contract=contract,
    )

    return session
