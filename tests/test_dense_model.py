from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from policyproof.dense_model import (
    DENSE_MODEL_CONTRACT,
    DenseModelError,
    normalize_embeddings,
    prepare_dense_inputs,
    validate_model_asset,
)
from policyproof.retrieval_tokenizer import token_ids


def test_dense_model_contract_is_fully_pinned() -> None:
    assert DENSE_MODEL_CONTRACT.runtime_library == "onnxruntime"
    assert DENSE_MODEL_CONTRACT.runtime_version == "1.27.0"
    assert DENSE_MODEL_CONTRACT.array_library == "numpy"
    assert DENSE_MODEL_CONTRACT.array_version == "2.5.1"

    assert DENSE_MODEL_CONTRACT.model_id == "BAAI/bge-small-en-v1.5"
    assert (
        DENSE_MODEL_CONTRACT.model_revision
        == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )
    assert DENSE_MODEL_CONTRACT.model_filename == "onnx/model.onnx"
    assert DENSE_MODEL_CONTRACT.model_size_bytes == 133_093_490
    assert (
        DENSE_MODEL_CONTRACT.model_sha256
        == "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
    )
    assert DENSE_MODEL_CONTRACT.license_id == "mit"

    assert DENSE_MODEL_CONTRACT.execution_provider == (
        "CPUExecutionProvider"
    )
    assert DENSE_MODEL_CONTRACT.max_sequence_length == 512
    assert DENSE_MODEL_CONTRACT.embedding_dimension == 384
    assert DENSE_MODEL_CONTRACT.pooling == "cls"
    assert DENSE_MODEL_CONTRACT.normalization == "l2"
    assert DENSE_MODEL_CONTRACT.query_instruction == (
        "Represent this sentence for searching relevant passages: "
    )
    assert DENSE_MODEL_CONTRACT.input_names == (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )
    assert DENSE_MODEL_CONTRACT.output_name == "last_hidden_state"


def test_model_asset_validation_accepts_exact_custom_contract(
    tmp_path,
) -> None:
    content = b"controlled ONNX fixture"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        DENSE_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )

    validate_model_asset(
        model_path,
        contract=contract,
    )


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        (b"wrong length", "size"),
        (b"wrong-content", "SHA-256"),
    ],
)
def test_model_asset_validation_fails_closed(
    tmp_path,
    content: bytes,
    expected_message: str,
) -> None:
    accepted = b"accepted-data"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        DENSE_MODEL_CONTRACT,
        model_size_bytes=len(accepted),
        model_sha256=hashlib.sha256(accepted).hexdigest(),
    )

    with pytest.raises(
        DenseModelError,
        match=expected_message,
    ):
        validate_model_asset(
            model_path,
            contract=contract,
        )


def test_prepare_dense_inputs_adds_instruction_only_to_queries() -> None:
    text = "risk governance"

    passage_inputs = prepare_dense_inputs(
        [text],
        is_query=False,
    )
    query_inputs = prepare_dense_inputs(
        [text],
        is_query=True,
    )

    passage_length = int(
        passage_inputs["attention_mask"][0].sum()
    )
    query_length = int(
        query_inputs["attention_mask"][0].sum()
    )

    assert tuple(
        passage_inputs["input_ids"][0, :passage_length]
    ) == token_ids(
        text,
        add_special_tokens=True,
    )
    assert tuple(
        query_inputs["input_ids"][0, :query_length]
    ) == token_ids(
        DENSE_MODEL_CONTRACT.query_instruction + text,
        add_special_tokens=True,
    )

    assert query_length > passage_length
    assert np.all(
        passage_inputs["token_type_ids"] == 0
    )
    assert np.all(
        query_inputs["token_type_ids"] == 0
    )


def test_prepare_dense_inputs_pads_without_truncation() -> None:
    inputs = prepare_dense_inputs(
        [
            "short text",
            "a longer controlled retrieval passage",
        ],
        is_query=False,
    )

    assert tuple(inputs) == DENSE_MODEL_CONTRACT.input_names
    assert inputs["input_ids"].dtype == np.int64
    assert inputs["attention_mask"].dtype == np.int64
    assert inputs["token_type_ids"].dtype == np.int64
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    assert inputs["input_ids"].shape == inputs["token_type_ids"].shape

    first_length = int(inputs["attention_mask"][0].sum())

    assert np.all(
        inputs["input_ids"][0, first_length:] == 0
    )
    assert np.all(
        inputs["attention_mask"][0, first_length:] == 0
    )


def test_prepare_dense_inputs_rejects_overlong_text() -> None:
    with pytest.raises(
        DenseModelError,
        match="512",
    ):
        prepare_dense_inputs(
            ["risk " * 600],
            is_query=False,
        )


def test_normalize_embeddings_produces_unit_vectors() -> None:
    embeddings = np.asarray(
        [
            [3.0, 4.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_embeddings(
        embeddings,
        expected_dimension=2,
    )

    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized, axis=1) == pytest.approx(
        np.ones(2),
        abs=1e-6,
    )


def test_normalize_embeddings_rejects_zero_vectors() -> None:
    with pytest.raises(
        DenseModelError,
        match="zero-length",
    ):
        normalize_embeddings(
            np.zeros((1, 384), dtype=np.float32),
            expected_dimension=384,
        )


class FakeNode:
    def __init__(
        self,
        name: str,
        node_type: str,
        shape: list[object],
    ) -> None:
        self.name = name
        self.type = node_type
        self.shape = shape


class FakeSession:
    def __init__(
        self,
        output: np.ndarray,
        *,
        providers: list[str] | None = None,
    ) -> None:
        self.output = output
        self.providers = providers or ["CPUExecutionProvider"]
        self.received_output_names: list[str] | None = None
        self.received_inputs: dict[str, np.ndarray] | None = None

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[FakeNode]:
        return [
            FakeNode(
                "input_ids",
                "tensor(int64)",
                ["batch_size", "sequence_length"],
            ),
            FakeNode(
                "attention_mask",
                "tensor(int64)",
                ["batch_size", "sequence_length"],
            ),
            FakeNode(
                "token_type_ids",
                "tensor(int64)",
                ["batch_size", "sequence_length"],
            ),
        ]

    def get_outputs(self) -> list[FakeNode]:
        return [
            FakeNode(
                "last_hidden_state",
                "tensor(float)",
                ["batch_size", "sequence_length", 384],
            )
        ]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.received_output_names = output_names
        self.received_inputs = inputs
        return [self.output]


def test_runtime_contract_accepts_only_exact_versions() -> None:
    from policyproof.dense_model import validate_runtime_contract

    validate_runtime_contract(
        runtime_version="1.27.0",
        array_version="2.5.1",
    )

    with pytest.raises(
        DenseModelError,
        match="onnxruntime",
    ):
        validate_runtime_contract(
            runtime_version="1.27.1",
            array_version="2.5.1",
        )

    with pytest.raises(
        DenseModelError,
        match="numpy",
    ):
        validate_runtime_contract(
            runtime_version="1.27.0",
            array_version="2.5.0",
        )


def test_session_contract_accepts_exact_cpu_onnx_interface() -> None:
    from policyproof.dense_model import validate_session_contract

    output = np.zeros((2, 5, 384), dtype=np.float32)
    session = FakeSession(output)

    validate_session_contract(session)


def test_session_contract_rejects_wrong_provider() -> None:
    from policyproof.dense_model import validate_session_contract

    output = np.zeros((1, 3, 384), dtype=np.float32)
    session = FakeSession(
        output,
        providers=["CoreMLExecutionProvider"],
    )

    with pytest.raises(
        DenseModelError,
        match="CPUExecutionProvider",
    ):
        validate_session_contract(session)


def test_session_contract_rejects_wrong_input_interface() -> None:
    from policyproof.dense_model import validate_session_contract

    class WrongInputSession(FakeSession):
        def get_inputs(self) -> list[FakeNode]:
            return [
                FakeNode(
                    "input_ids",
                    "tensor(int32)",
                    ["batch_size", "sequence_length"],
                )
            ]

    session = WrongInputSession(
        np.zeros((1, 3, 384), dtype=np.float32)
    )

    with pytest.raises(
        DenseModelError,
        match="input contract",
    ):
        validate_session_contract(session)


def test_embed_dense_texts_uses_cls_pooling_and_l2_normalization() -> None:
    from policyproof.dense_model import embed_dense_texts

    output = np.zeros((2, 8, 384), dtype=np.float32)
    output[0, 0, 0] = 3.0
    output[0, 0, 1] = 4.0
    output[1, 0, 0] = 5.0
    output[1, 0, 1] = 12.0

    session = FakeSession(output)

    embeddings = embed_dense_texts(
        session,
        [
            "govern AI risk",
            "manage generative AI risk",
        ],
        is_query=False,
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    assert embeddings[0, :2] == pytest.approx(
        np.asarray([0.6, 0.8], dtype=np.float32),
        abs=1e-6,
    )
    assert embeddings[1, :2] == pytest.approx(
        np.asarray([5 / 13, 12 / 13], dtype=np.float32),
        abs=1e-6,
    )
    assert np.linalg.norm(
        embeddings,
        axis=1,
    ) == pytest.approx(
        np.ones(2),
        abs=1e-6,
    )

    assert session.received_output_names == [
        DENSE_MODEL_CONTRACT.output_name
    ]
    assert session.received_inputs is not None
    assert tuple(
        session.received_inputs
    ) == DENSE_MODEL_CONTRACT.input_names


def test_embed_dense_texts_applies_query_instruction() -> None:
    from policyproof.dense_model import embed_dense_texts

    output = np.zeros((1, 32, 384), dtype=np.float32)
    output[0, 0, 0] = 1.0
    session = FakeSession(output)

    embed_dense_texts(
        session,
        ["What is AI risk management?"],
        is_query=True,
    )

    assert session.received_inputs is not None

    actual_length = int(
        session.received_inputs["attention_mask"][0].sum()
    )
    expected_length = len(
        token_ids(
            DENSE_MODEL_CONTRACT.query_instruction
            + "What is AI risk management?",
            add_special_tokens=True,
        )
    )

    assert actual_length == expected_length


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 384), dtype=np.float32),
        np.zeros((2, 3, 384), dtype=np.float32),
        np.zeros((1, 3, 383), dtype=np.float32),
        np.full((1, 3, 384), np.nan, dtype=np.float32),
    ],
)
def test_embed_dense_texts_rejects_invalid_model_output(
    output: np.ndarray,
) -> None:
    from policyproof.dense_model import embed_dense_texts

    session = FakeSession(output)

    with pytest.raises(DenseModelError):
        embed_dense_texts(
            session,
            ["controlled query"],
            is_query=False,
        )


class FakeRuntime:
    __version__ = "1.27.0"

    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def InferenceSession(
        self,
        path: str,
        *,
        providers: list[str],
    ) -> FakeSession:
        self.calls.append(
            (
                path,
                tuple(providers),
            )
        )
        return self.session


def test_create_dense_session_validates_asset_runtime_and_interface(
    tmp_path,
) -> None:
    from policyproof.dense_model import create_dense_session

    content = b"controlled-model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        DENSE_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )
    session = FakeSession(
        np.zeros((1, 3, 384), dtype=np.float32)
    )
    runtime = FakeRuntime(session)

    created = create_dense_session(
        model_path,
        contract=contract,
        runtime=runtime,
        runtime_version="1.27.0",
        array_version="2.5.1",
    )

    assert created is session
    assert runtime.calls == [
        (
            str(model_path),
            ("CPUExecutionProvider",),
        )
    ]


def test_create_dense_session_rejects_asset_before_runtime_call(
    tmp_path,
) -> None:
    from policyproof.dense_model import create_dense_session

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"corrupt")
    session = FakeSession(
        np.zeros((1, 3, 384), dtype=np.float32)
    )
    runtime = FakeRuntime(session)

    with pytest.raises(
        DenseModelError,
        match="size",
    ):
        create_dense_session(
            model_path,
            runtime=runtime,
            runtime_version="1.27.0",
            array_version="2.5.1",
        )

    assert runtime.calls == []


def test_create_dense_session_rejects_runtime_before_session_creation(
    tmp_path,
) -> None:
    from policyproof.dense_model import create_dense_session

    content = b"controlled-model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        DENSE_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )
    runtime = FakeRuntime(
        FakeSession(
            np.zeros((1, 3, 384), dtype=np.float32)
        )
    )

    with pytest.raises(
        DenseModelError,
        match="onnxruntime",
    ):
        create_dense_session(
            model_path,
            contract=contract,
            runtime=runtime,
            runtime_version="1.27.1",
            array_version="2.5.1",
        )

    assert runtime.calls == []


def test_create_dense_session_wraps_runtime_failure(
    tmp_path,
) -> None:
    from policyproof.dense_model import create_dense_session

    content = b"controlled-model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        DENSE_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )

    class FailingRuntime:
        def InferenceSession(
            self,
            path: str,
            *,
            providers: list[str],
        ) -> FakeSession:
            raise RuntimeError("simulated ONNX failure")

    with pytest.raises(
        DenseModelError,
        match="session creation failed",
    ):
        create_dense_session(
            model_path,
            contract=contract,
            runtime=FailingRuntime(),
            runtime_version="1.27.0",
            array_version="2.5.1",
        )
