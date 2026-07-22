from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from policyproof.reranker_model import (
    RERANKER_MODEL_CONTRACT,
    RerankerModelError,
    create_reranker_session,
    prepare_reranker_inputs,
    score_reranker_pairs,
    validate_reranker_model_asset,
    validate_reranker_runtime_contract,
    validate_reranker_session_contract,
)
from policyproof.retrieval_tokenizer import retrieval_tokenizer


def test_reranker_model_contract_is_fully_pinned() -> None:
    assert RERANKER_MODEL_CONTRACT.runtime_library == "onnxruntime"
    assert RERANKER_MODEL_CONTRACT.runtime_version == "1.27.0"
    assert RERANKER_MODEL_CONTRACT.array_library == "numpy"
    assert RERANKER_MODEL_CONTRACT.array_version == "2.5.1"

    assert (
        RERANKER_MODEL_CONTRACT.model_id
        == "cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    assert (
        RERANKER_MODEL_CONTRACT.model_revision
        == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    )
    assert RERANKER_MODEL_CONTRACT.model_filename == "onnx/model.onnx"
    assert RERANKER_MODEL_CONTRACT.model_size_bytes == 91_011_230
    assert (
        RERANKER_MODEL_CONTRACT.model_sha256
        == "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"
    )
    assert RERANKER_MODEL_CONTRACT.license_id == "apache-2.0"

    assert (
        RERANKER_MODEL_CONTRACT.execution_provider
        == "CPUExecutionProvider"
    )
    assert RERANKER_MODEL_CONTRACT.max_sequence_length == 512
    assert RERANKER_MODEL_CONTRACT.input_names == (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )
    assert RERANKER_MODEL_CONTRACT.output_name == "logits"
    assert RERANKER_MODEL_CONTRACT.output_dimension == 1
    assert RERANKER_MODEL_CONTRACT.output_interpretation == "raw_logit"
    assert RERANKER_MODEL_CONTRACT.ranking_order == "descending"
    assert RERANKER_MODEL_CONTRACT.query_instruction == ""
    assert RERANKER_MODEL_CONTRACT.passage_instruction == ""
    assert RERANKER_MODEL_CONTRACT.truncation == "reject"


def test_prepare_reranker_inputs_builds_bert_pairs_and_pads_batch() -> None:
    pairs = [
        ("What is AI risk governance?", "Governance manages AI risks."),
        (
            "What must providers document?",
            "Providers must maintain technical documentation and records.",
        ),
    ]

    inputs = prepare_reranker_inputs(pairs)

    assert tuple(inputs) == RERANKER_MODEL_CONTRACT.input_names
    assert inputs["input_ids"].dtype == np.int64
    assert inputs["attention_mask"].dtype == np.int64
    assert inputs["token_type_ids"].dtype == np.int64

    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    assert inputs["input_ids"].shape == inputs["token_type_ids"].shape
    assert inputs["input_ids"].shape[0] == 2

    tokenizer = retrieval_tokenizer()

    for row, (question, passage) in enumerate(pairs):
        expected = tokenizer.encode(
            question,
            pair=passage,
            add_special_tokens=True,
        )
        length = len(expected.ids)

        assert tuple(inputs["input_ids"][row, :length]) == tuple(expected.ids)
        assert tuple(inputs["token_type_ids"][row, :length]) == tuple(
            expected.type_ids
        )
        assert np.all(inputs["attention_mask"][row, :length] == 1)

        assert np.all(inputs["input_ids"][row, length:] == 0)
        assert np.all(inputs["attention_mask"][row, length:] == 0)
        assert np.all(inputs["token_type_ids"][row, length:] == 0)

        first_separator = expected.ids.index(102)

        assert expected.ids[0] == 101
        assert expected.ids[-1] == 102
        assert expected.ids.count(102) == 2
        assert all(
            type_id == 0
            for type_id in expected.type_ids[: first_separator + 1]
        )
        assert all(
            type_id == 1
            for type_id in expected.type_ids[first_separator + 1 :]
        )


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [("", "valid passage")],
        [("valid question", "")],
        [("valid question", 123)],
        ["not a pair"],
        [("question only",)],
        [("question", "passage", "extra")],
    ],
)
def test_prepare_reranker_inputs_rejects_invalid_pairs(pairs) -> None:
    with pytest.raises(RerankerModelError):
        prepare_reranker_inputs(pairs)


def test_prepare_reranker_inputs_rejects_overlong_pair_without_truncation() -> None:
    with pytest.raises(
        RerankerModelError,
        match="512",
    ):
        prepare_reranker_inputs(
            [
                (
                    "controlled question",
                    "risk " * 600,
                )
            ]
        )


def test_prepare_reranker_inputs_uses_contract_sequence_limit() -> None:
    restrictive_contract = replace(
        RERANKER_MODEL_CONTRACT,
        max_sequence_length=8,
    )

    with pytest.raises(
        RerankerModelError,
        match="8",
    ):
        prepare_reranker_inputs(
            [
                (
                    "what is governed",
                    "artificial intelligence risk is governed",
                )
            ],
            contract=restrictive_contract,
        )


def test_reranker_model_asset_validation_accepts_exact_custom_contract(
    tmp_path,
) -> None:
    content = b"controlled reranker ONNX fixture"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        RERANKER_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )

    validate_reranker_model_asset(
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
def test_reranker_model_asset_validation_fails_closed(
    tmp_path,
    content: bytes,
    expected_message: str,
) -> None:
    accepted = b"accepted-data"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        RERANKER_MODEL_CONTRACT,
        model_size_bytes=len(accepted),
        model_sha256=hashlib.sha256(accepted).hexdigest(),
    )

    with pytest.raises(
        RerankerModelError,
        match=expected_message,
    ):
        validate_reranker_model_asset(
            model_path,
            contract=contract,
        )


def test_reranker_runtime_contract_accepts_only_exact_versions() -> None:
    validate_reranker_runtime_contract(
        runtime_version="1.27.0",
        array_version="2.5.1",
    )

    with pytest.raises(
        RerankerModelError,
        match="onnxruntime",
    ):
        validate_reranker_runtime_contract(
            runtime_version="1.27.1",
            array_version="2.5.1",
        )

    with pytest.raises(
        RerankerModelError,
        match="numpy",
    ):
        validate_reranker_runtime_contract(
            runtime_version="1.27.0",
            array_version="2.5.0",
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


class FakeRerankerSession:
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
                "logits",
                "tensor(float)",
                ["batch_size", 1],
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


def test_reranker_session_contract_accepts_exact_cpu_onnx_interface() -> None:
    session = FakeRerankerSession(
        np.zeros((2, 1), dtype=np.float32)
    )

    validate_reranker_session_contract(session)


def test_reranker_session_contract_rejects_wrong_provider() -> None:
    session = FakeRerankerSession(
        np.zeros((1, 1), dtype=np.float32),
        providers=["CoreMLExecutionProvider"],
    )

    with pytest.raises(
        RerankerModelError,
        match="CPUExecutionProvider",
    ):
        validate_reranker_session_contract(session)


def test_reranker_session_contract_rejects_wrong_output_interface() -> None:
    class WrongOutputSession(FakeRerankerSession):
        def get_outputs(self) -> list[FakeNode]:
            return [
                FakeNode(
                    "probabilities",
                    "tensor(float)",
                    ["batch_size", 2],
                )
            ]

    session = WrongOutputSession(
        np.zeros((1, 2), dtype=np.float32)
    )

    with pytest.raises(
        RerankerModelError,
        match="output contract",
    ):
        validate_reranker_session_contract(session)


def test_score_reranker_pairs_returns_raw_float32_logits() -> None:
    expected = np.asarray(
        [
            [1.25],
            [-0.75],
        ],
        dtype=np.float32,
    )
    session = FakeRerankerSession(expected)

    pairs = [
        ("What is governed?", "AI risks are governed."),
        ("What is documented?", "Technical records are maintained."),
    ]

    scores = score_reranker_pairs(
        session,
        pairs,
    )

    assert scores.shape == (2,)
    assert scores.dtype == np.float32
    assert scores == pytest.approx([1.25, -0.75])

    assert session.received_output_names == [
        RERANKER_MODEL_CONTRACT.output_name
    ]
    assert session.received_inputs is not None
    assert tuple(session.received_inputs) == (
        RERANKER_MODEL_CONTRACT.input_names
    )

    expected_inputs = prepare_reranker_inputs(pairs)

    for name in RERANKER_MODEL_CONTRACT.input_names:
        assert np.array_equal(
            session.received_inputs[name],
            expected_inputs[name],
        )


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((2,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float64),
        np.asarray([[0.0], [np.nan]], dtype=np.float32),
        np.asarray([[0.0], [np.inf]], dtype=np.float32),
    ],
)
def test_score_reranker_pairs_rejects_invalid_model_outputs(
    output: np.ndarray,
) -> None:
    session = FakeRerankerSession(output)

    with pytest.raises(RerankerModelError):
        score_reranker_pairs(
            session,
            [
                ("question one", "passage one"),
                ("question two", "passage two"),
            ],
        )



class FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.execution_mode = None
        self.graph_optimization_level = None
        self.use_deterministic_compute = False


class FakeExecutionMode:
    ORT_SEQUENTIAL = "ORT_SEQUENTIAL"


class FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "ORT_ENABLE_ALL"


class FakeRerankerRuntime:
    __version__ = "1.27.0"
    SessionOptions = FakeSessionOptions
    ExecutionMode = FakeExecutionMode
    GraphOptimizationLevel = FakeGraphOptimizationLevel

    def __init__(self, session: FakeRerankerSession) -> None:
        self.session = session
        self.calls: list[dict[str, object]] = []

    def InferenceSession(
        self,
        path: str,
        *,
        sess_options: FakeSessionOptions,
        providers: list[str],
    ) -> FakeRerankerSession:
        self.calls.append(
            {
                "path": path,
                "sess_options": sess_options,
                "providers": tuple(providers),
            }
        )
        return self.session


def test_create_reranker_session_uses_deterministic_cpu_options(
    tmp_path,
) -> None:
    content = b"controlled reranker model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        RERANKER_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )
    session = FakeRerankerSession(
        np.zeros((1, 1), dtype=np.float32)
    )
    runtime = FakeRerankerRuntime(session)

    created = create_reranker_session(
        model_path,
        contract=contract,
        runtime=runtime,
        runtime_version="1.27.0",
        array_version="2.5.1",
    )

    assert created is session
    assert len(runtime.calls) == 1

    call = runtime.calls[0]
    assert call["path"] == str(model_path)
    assert call["providers"] == ("CPUExecutionProvider",)

    options = call["sess_options"]
    assert isinstance(options, FakeSessionOptions)
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == FakeExecutionMode.ORT_SEQUENTIAL
    assert (
        options.graph_optimization_level
        == FakeGraphOptimizationLevel.ORT_ENABLE_ALL
    )
    assert options.use_deterministic_compute is True


def test_create_reranker_session_rejects_asset_before_runtime_use(
    tmp_path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"corrupt")

    runtime = FakeRerankerRuntime(
        FakeRerankerSession(
            np.zeros((1, 1), dtype=np.float32)
        )
    )

    with pytest.raises(
        RerankerModelError,
        match="size",
    ):
        create_reranker_session(
            model_path,
            runtime=runtime,
            runtime_version="1.27.0",
            array_version="2.5.1",
        )

    assert runtime.calls == []


def test_create_reranker_session_rejects_runtime_before_session_creation(
    tmp_path,
) -> None:
    content = b"controlled reranker model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        RERANKER_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )
    runtime = FakeRerankerRuntime(
        FakeRerankerSession(
            np.zeros((1, 1), dtype=np.float32)
        )
    )

    with pytest.raises(
        RerankerModelError,
        match="onnxruntime",
    ):
        create_reranker_session(
            model_path,
            contract=contract,
            runtime=runtime,
            runtime_version="1.27.1",
            array_version="2.5.1",
        )

    assert runtime.calls == []


def test_create_reranker_session_rejects_runtime_without_determinism_option(
    tmp_path,
) -> None:
    content = b"controlled reranker model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        RERANKER_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )

    class IncompleteSessionOptions:
        def __init__(self) -> None:
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.execution_mode = None
            self.graph_optimization_level = None

    class IncompleteRuntime(FakeRerankerRuntime):
        SessionOptions = IncompleteSessionOptions

    runtime = IncompleteRuntime(
        FakeRerankerSession(
            np.zeros((1, 1), dtype=np.float32)
        )
    )

    with pytest.raises(
        RerankerModelError,
        match="deterministic",
    ):
        create_reranker_session(
            model_path,
            contract=contract,
            runtime=runtime,
            runtime_version="1.27.0",
            array_version="2.5.1",
        )

    assert runtime.calls == []


def test_create_reranker_session_wraps_runtime_failure(
    tmp_path,
) -> None:
    content = b"controlled reranker model"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(content)

    contract = replace(
        RERANKER_MODEL_CONTRACT,
        model_size_bytes=len(content),
        model_sha256=hashlib.sha256(content).hexdigest(),
    )

    class FailingRuntime(FakeRerankerRuntime):
        def InferenceSession(
            self,
            path: str,
            *,
            sess_options: FakeSessionOptions,
            providers: list[str],
        ) -> FakeRerankerSession:
            raise RuntimeError("simulated ONNX failure")

    runtime = FailingRuntime(
        FakeRerankerSession(
            np.zeros((1, 1), dtype=np.float32)
        )
    )

    with pytest.raises(
        RerankerModelError,
        match="session creation failed",
    ):
        create_reranker_session(
            model_path,
            contract=contract,
            runtime=runtime,
            runtime_version="1.27.0",
            array_version="2.5.1",
        )
