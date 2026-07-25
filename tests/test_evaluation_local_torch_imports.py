"""Regression contracts for unused local PyTorch evaluation imports."""

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP10 = ROOT / "scripts" / "evaluation" / "experiments" / "exp10_advanced_routing.py"
EXP11 = ROOT / "scripts" / "evaluation" / "experiments" / "exp11_ablation_optimization.py"
EXP8 = ROOT / "scripts" / "evaluation" / "experiments" / "exp8_inference_efficiency.py"
SPECS = [
    pytest.param(
        EXP10,
        "_jaccard_topk",
        "torch",
        "27033aed8400f985acc1a9a33f199a50928baa8385e8aa54aca8691cf151301d",
        id="exp10-jaccard",
    ),
    pytest.param(
        EXP10,
        "_train_router",
        "F",
        "20f63d8ccf41e2e6d85b8a1eaefac8f4129b8c107aa77a81a8c64fc0cefc21a3",
        id="exp10-train-router",
    ),
    pytest.param(
        EXP10,
        "_run_output_ensemble",
        "torch",
        "3f0b7fab466f4275194e4e8a34079b7a2bfc29cebbc688eb33eebf0ae214da3a",
        id="exp10-output-ensemble",
    ),
    pytest.param(
        EXP10,
        "_decode_from_logits",
        "torch",
        "8fdde30468843ead482a9ee129f1a1ff993fbeb9a62be7f2e13eb6d0fa463efc",
        id="exp10-decode",
    ),
    pytest.param(
        EXP11,
        "run_phase1",
        "torch",
        "a43aedd2557a4ca36bed87bd589b9daab59adc2f442e69b42108165dcaa1b026",
        id="exp11-phase1",
    ),
    pytest.param(
        EXP8,
        "_benchmark_gpu_method",
        "torch",
        "f25b21f369594cc2996b1b65e7cea17ff9997414345e5b2e3ea45978c833f8fa",
        id="exp8-gpu-benchmark",
    ),
]


class _StripBinding(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name

    def visit_Import(self, node):
        node.names = [
            alias
            for alias in node.names
            if (alias.asname or alias.name.split(".")[0]) != self.name
        ]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        node.names = [
            alias
            for alias in node.names
            if (alias.asname or alias.name) != self.name
        ]
        return node if node.names else None


def _load_function_node(path, function_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _imported_bindings(function):
    bindings = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Import):
            bindings.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            bindings.update(alias.asname or alias.name for alias in node.names)
    return bindings


def _direct_calls(function):
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _normalized_hash(function, binding):
    normalized = _StripBinding(binding).visit(copy.deepcopy(function))
    ast.fix_missing_locations(normalized)
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _load_function(path, function_name):
    function = _load_function_node(path, function_name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


@pytest.mark.parametrize(
    ("path", "function_name", "removed_binding", "expected_hash"),
    SPECS,
)
def test_local_import_removal_preserves_each_function_contract(
    path,
    function_name,
    removed_binding,
    expected_hash,
):
    function = _load_function_node(path, function_name)
    loaded_targets = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == removed_binding
    }

    assert removed_binding not in _imported_bindings(function)
    assert loaded_targets == set()
    assert _normalized_hash(function, removed_binding) == expected_hash


def test_heavy_loops_keep_their_runtime_dependency_paths():
    train_router = _load_function_node(EXP10, "_train_router")
    output_ensemble = _load_function_node(EXP10, "_run_output_ensemble")
    phase1 = _load_function_node(EXP11, "run_phase1")
    benchmark = _load_function_node(EXP8, "_benchmark_gpu_method")

    assert {"torch", "nn", "DataLoader", "TensorDataset"} <= (
        _imported_bindings(train_router)
    )
    assert {"PeftModel", "LanguageModel"} <= _imported_bindings(
        output_ensemble
    )
    assert "_logit_ensemble_generate_batched" in _direct_calls(
        output_ensemble
    )
    assert {"PeftModel", "LanguageModel"} <= _imported_bindings(phase1)
    assert {"ZeroShotGenerator", "TextExpert"} <= _imported_bindings(
        benchmark
    )
    assert {
        "_clear_gpu",
        "_gpu_current_mb",
        "_gpu_peak_mb",
        "_gpu_sync",
    } <= _direct_calls(benchmark)


def test_jaccard_topk_preserves_tensor_protocol_behavior():
    class _Row:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class _Indices:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, index):
            return _Row(self.rows[index])

    class _Probabilities:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows),)

        def topk(self, k, dim):
            assert k == 2  # noqa: PLR2004 - top-2 routing contract
            assert dim == -1
            return SimpleNamespace(indices=_Indices(self.rows))

    jaccard_topk = _load_function(EXP10, "_jaccard_topk")
    result = jaccard_topk(
        _Probabilities([[1, 2], [3, 4]]),
        _Probabilities([[2, 3], [3, 4]]),
        k=2,
    )

    assert result == pytest.approx([1 / 3, 1.0])


def test_decode_from_logits_preserves_stop_and_decode_behavior():
    class _ArgmaxResult:
        def __init__(self, token):
            self.token = token

        def item(self):
            return self.token

    class _Logits:
        def __init__(self, token):
            self.token = token

        def argmax(self, dim):
            assert dim == -1
            return _ArgmaxResult(self.token)

    class _Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __init__(self):
            self.calls = []

        def decode(self, tokens, skip_special_tokens):
            self.calls.append((tokens, skip_special_tokens))
            return "decoded"

    decode = _load_function(EXP10, "_decode_from_logits")
    tokenizer = _Tokenizer()

    assert decode(tokenizer, [_Logits(4), _Logits(7), _Logits(99), _Logits(8)]) == "decoded"
    assert tokenizer.calls == [([4, 7], True)]
