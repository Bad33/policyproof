"""Pinned cross-encoder reranker contract and deterministic pair inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policyproof.retrieval_tokenizer import retrieval_tokenizer


class RerankerModelError(RuntimeError):
    """Raised when the reranker model contract cannot be satisfied."""


@dataclass(frozen=True)
class RerankerModelContract:
    """Immutable runtime, model-asset, and ranking behavior contract."""

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
    input_names: tuple[str, ...]
    output_name: str
    output_dimension: int
    output_interpretation: str
    ranking_order: str
    query_instruction: str
    passage_instruction: str
    truncation: str


RERANKER_MODEL_CONTRACT = RerankerModelContract(
    runtime_library="onnxruntime",
    runtime_version="1.27.0",
    array_library="numpy",
    array_version="2.5.1",
    model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    model_revision="c5ee24cb16019beea0893ab7796b1df96625c6b8",
    model_filename="onnx/model.onnx",
    model_size_bytes=91_011_230,
    model_sha256=(
        "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"
    ),
    license_id="apache-2.0",
    execution_provider="CPUExecutionProvider",
    max_sequence_length=512,
    input_names=(
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ),
    output_name="logits",
    output_dimension=1,
    output_interpretation="raw_logit",
    ranking_order="descending",
    query_instruction="",
    passage_instruction="",
    truncation="reject",
)


def _require_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Validate and freeze non-empty question-passage pairs."""

    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise RerankerModelError(
            "Reranker input must be a sequence of question-passage pairs."
        )

    validated = tuple(pairs)

    if not validated:
        raise RerankerModelError(
            "Reranker input must contain at least one pair."
        )

    normalized: list[tuple[str, str]] = []

    for index, pair in enumerate(validated):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise RerankerModelError(
                f"Reranker input at index {index} must contain exactly "
                "one question and one passage."
            )

        question, passage = pair

        if not isinstance(question, str):
            raise RerankerModelError(
                f"Reranker question at index {index} must be a string."
            )

        if not isinstance(passage, str):
            raise RerankerModelError(
                f"Reranker passage at index {index} must be a string."
            )

        if not question:
            raise RerankerModelError(
                f"Reranker question at index {index} must not be empty."
            )

        if not passage:
            raise RerankerModelError(
                f"Reranker passage at index {index} must not be empty."
            )

        normalized.append((question, passage))

    return tuple(normalized)


def prepare_reranker_inputs(
    pairs: Sequence[tuple[str, str]],
    *,
    contract: RerankerModelContract = RERANKER_MODEL_CONTRACT,
) -> dict[str, np.ndarray]:
    """Tokenize and pad BERT question-passage pairs without truncation."""

    validated_pairs = _require_pairs(pairs)
    tokenizer = retrieval_tokenizer()

    encodings = [
        tokenizer.encode(
            contract.query_instruction + question,
            pair=contract.passage_instruction + passage,
            add_special_tokens=True,
        )
        for question, passage in validated_pairs
    ]

    lengths = [len(encoding.ids) for encoding in encodings]

    for index, length in enumerate(lengths):
        if length > contract.max_sequence_length:
            raise RerankerModelError(
                f"Reranker pair at index {index} contains {length} tokens "
                f"and exceeds the {contract.max_sequence_length}-token limit."
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
        token_type_ids[row, :length] = np.asarray(
            encoding.type_ids,
            dtype=np.int64,
        )

    return {
        contract.input_names[0]: input_ids,
        contract.input_names[1]: attention_mask,
        contract.input_names[2]: token_type_ids,
    }



def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one readable model asset."""

    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RerankerModelError(
            f"Unable to read reranker model asset: {path}"
        ) from error

    return digest.hexdigest()


def validate_reranker_model_asset(
    path: Path,
    *,
    contract: RerankerModelContract = RERANKER_MODEL_CONTRACT,
) -> None:
    """Require the exact pinned reranker model size and SHA-256."""

    validated_path = Path(path)

    try:
        actual_size = validated_path.stat().st_size
    except OSError as error:
        raise RerankerModelError(
            f"Unable to inspect reranker model asset: {validated_path}"
        ) from error

    if actual_size != contract.model_size_bytes:
        raise RerankerModelError(
            "Reranker model asset size "
            f"{actual_size} does not match pinned size "
            f"{contract.model_size_bytes}."
        )

    actual_sha256 = _sha256_file(validated_path)

    if actual_sha256 != contract.model_sha256:
        raise RerankerModelError(
            "Reranker model asset SHA-256 "
            f"{actual_sha256} does not match pinned SHA-256 "
            f"{contract.model_sha256}."
        )


def validate_reranker_runtime_contract(
    *,
    runtime_version: str,
    array_version: str,
    contract: RerankerModelContract = RERANKER_MODEL_CONTRACT,
) -> None:
    """Require the exact pinned ONNX Runtime and NumPy versions."""

    if runtime_version != contract.runtime_version:
        raise RerankerModelError(
            "Installed onnxruntime version "
            f"{runtime_version!r} does not match pinned version "
            f"{contract.runtime_version!r}."
        )

    if array_version != contract.array_version:
        raise RerankerModelError(
            "Installed numpy version "
            f"{array_version!r} does not match pinned version "
            f"{contract.array_version!r}."
        )


def validate_reranker_session_contract(
    session: object,
    *,
    contract: RerankerModelContract = RERANKER_MODEL_CONTRACT,
) -> None:
    """Validate the provider and public reranker ONNX interface."""

    try:
        providers = tuple(session.get_providers())  # type: ignore[attr-defined]
        inputs = tuple(session.get_inputs())  # type: ignore[attr-defined]
        outputs = tuple(session.get_outputs())  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise RerankerModelError(
            "Reranker session does not expose the required ONNX interface."
        ) from error

    if providers != (contract.execution_provider,):
        raise RerankerModelError(
            "Reranker session must use only "
            f"{contract.execution_provider}; received {providers!r}."
        )

    expected_inputs = tuple(
        (
            name,
            "tensor(int64)",
            2,
        )
        for name in contract.input_names
    )
    actual_inputs = tuple(
        (
            getattr(node, "name", None),
            getattr(node, "type", None),
            len(getattr(node, "shape", ())),
        )
        for node in inputs
    )

    if actual_inputs != expected_inputs:
        raise RerankerModelError(
            "Reranker model input contract does not match the pinned "
            f"interface: {actual_inputs!r}."
        )

    if len(outputs) != 1:
        raise RerankerModelError(
            "Reranker model output contract must contain exactly one output."
        )

    output = outputs[0]
    output_name = getattr(output, "name", None)
    output_type = getattr(output, "type", None)
    output_shape = getattr(output, "shape", ())

    if (
        output_name != contract.output_name
        or output_type != "tensor(float)"
        or len(output_shape) != 2
        or output_shape[-1] != contract.output_dimension
    ):
        raise RerankerModelError(
            "Reranker model output contract does not match the pinned "
            "single-logit interface."
        )


def score_reranker_pairs(
    session: object,
    pairs: Sequence[tuple[str, str]],
    *,
    contract: RerankerModelContract = RERANKER_MODEL_CONTRACT,
) -> np.ndarray:
    """Return one finite raw float32 relevance logit per input pair."""

    validate_reranker_session_contract(
        session,
        contract=contract,
    )
    inputs = prepare_reranker_inputs(
        pairs,
        contract=contract,
    )
    batch_size = inputs[contract.input_names[0]].shape[0]

    try:
        outputs = session.run(  # type: ignore[attr-defined]
            [contract.output_name],
            inputs,
        )
    except Exception as error:
        raise RerankerModelError(
            "Reranker ONNX inference failed."
        ) from error

    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise RerankerModelError(
            "Reranker inference must return exactly one output tensor."
        )

    logits = outputs[0]

    if not isinstance(logits, np.ndarray):
        raise RerankerModelError(
            "Reranker output must be a NumPy array."
        )

    expected_shape = (batch_size, contract.output_dimension)

    if logits.shape != expected_shape:
        raise RerankerModelError(
            f"Reranker output shape {logits.shape!r} does not match "
            f"expected shape {expected_shape!r}."
        )

    if logits.dtype != np.float32:
        raise RerankerModelError(
            f"Reranker output dtype {logits.dtype} must be float32."
        )

    if not np.isfinite(logits).all():
        raise RerankerModelError(
            "Reranker output contains non-finite logits."
        )

    return np.asarray(logits[:, 0], dtype=np.float32)



def create_reranker_session(
    model_path: Path,
    *,
    contract: RerankerModelContract = RERANKER_MODEL_CONTRACT,
    runtime: object | None = None,
    runtime_version: str | None = None,
    array_version: str | None = None,
) -> object:
    """Create one hash-verified deterministic CPU ONNX session."""

    validated_path = Path(model_path)

    validate_reranker_model_asset(
        validated_path,
        contract=contract,
    )

    selected_runtime = runtime

    if selected_runtime is None:
        try:
            import onnxruntime as selected_runtime
        except ImportError as error:
            raise RerankerModelError(
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
        raise RerankerModelError(
            "Unable to determine installed onnxruntime version."
        )

    selected_array_version = (
        np.__version__
        if array_version is None
        else array_version
    )

    validate_reranker_runtime_contract(
        runtime_version=selected_runtime_version,
        array_version=selected_array_version,
        contract=contract,
    )

    try:
        options = selected_runtime.SessionOptions()  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise RerankerModelError(
            "ONNX Runtime does not expose session options required "
            "by the reranker contract."
        ) from error

    if not hasattr(options, "use_deterministic_compute"):
        raise RerankerModelError(
            "ONNX Runtime session options do not support deterministic compute."
        )

    try:
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = (  # type: ignore[attr-defined]
            selected_runtime.ExecutionMode.ORT_SEQUENTIAL
        )
        options.graph_optimization_level = (  # type: ignore[attr-defined]
            selected_runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.use_deterministic_compute = True
    except (AttributeError, TypeError, ValueError) as error:
        raise RerankerModelError(
            "Unable to configure deterministic reranker session options."
        ) from error

    try:
        session = selected_runtime.InferenceSession(  # type: ignore[attr-defined]
            str(validated_path),
            sess_options=options,
            providers=[contract.execution_provider],
        )
    except Exception as error:
        raise RerankerModelError(
            "Reranker model ONNX session creation failed."
        ) from error

    validate_reranker_session_contract(
        session,
        contract=contract,
    )

    return session
