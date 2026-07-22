"""Regression contracts for Experiment 8 CPU benchmark callables."""

import ast
import copy
import hashlib
import sys
import types
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP8 = ROOT / "scripts" / "evaluation" / "experiments" / "exp8_inference_efficiency.py"
EXPECTED_NORMALIZED_HASH = (
    "cf1a40bf1a28274ff64b047a0627960ac5c909923a9323b118c544afe1eec694"
)
CALLABLE_NAMES = {"predict_one", "predict_batch"}


def _load_function_node():
    tree = ast.parse(EXP8.read_text(encoding="utf-8"), filename=str(EXP8))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_benchmark_cpu_method"
    )


def _normalized_hash(function):
    function = copy.deepcopy(function)
    for owner in ast.walk(function):
        for field_name in ("body", "orelse", "finalbody"):
            body = getattr(owner, field_name, None)
            if not isinstance(body, list):
                continue
            normalized = []
            for statement in body:
                if (
                    isinstance(statement, ast.FunctionDef)
                    and statement.name in CALLABLE_NAMES
                    and len(statement.body) == 1
                    and isinstance(statement.body[0], ast.Return)
                ):
                    normalized.append(
                        ast.Assign(
                            targets=[
                                ast.Name(id=statement.name, ctx=ast.Store())
                            ],
                            value=ast.Lambda(
                                args=statement.args,
                                body=statement.body[0].value,
                            ),
                        )
                    )
                else:
                    normalized.append(statement)
            setattr(owner, field_name, normalized)
    ast.fix_missing_locations(function)
    dumped = ast.dump(function, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _load_benchmark_function(clock):
    function = _load_function_node()
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"time": clock}
    exec(compile(module, str(EXP8), "exec"), namespace)
    return namespace["_benchmark_cpu_method"]


class _Clock:
    def __init__(self):
        self._values = iter(
            [
                10.0,
                10.25,
                20.0,
                20.01,
                30.0,
                30.02,
                40.0,
                40.5,
            ]
        )

    def perf_counter(self):
        return next(self._values)


def _retriever_class(kind, events):
    class _Retriever:
        def __init__(self, **kwargs):
            events.append(("init", kind, kwargs))

        def build_index(self, train_data):
            events.append(("build", kind, tuple(train_data)))

        def batch_retrieve(self, inputs):
            events.append(("batch", kind, tuple(inputs)))
            return [f"{kind}:{value}" for value in inputs]

    return _Retriever


def _template_class(events):
    class _TemplateFiller:
        def __init__(self):
            events.append(("init", "template", {}))

        def batch_fill(self, inputs):
            events.append(("batch", "template", tuple(inputs)))
            return [f"template:{value}" for value in inputs]

    return _TemplateFiller


def _install_fake_baselines(monkeypatch, events):
    ir_methods = types.ModuleType("src.baselines.ir_methods")
    ir_methods.BM25Retriever = _retriever_class("bm25", events)
    ir_methods.LSARetriever = _retriever_class("lsa", events)
    template_filling = types.ModuleType("src.baselines.template_filling")
    template_filling.TemplateFiller = _template_class(events)
    monkeypatch.setitem(sys.modules, ir_methods.__name__, ir_methods)
    monkeypatch.setitem(
        sys.modules,
        template_filling.__name__,
        template_filling,
    )


def test_cpu_benchmark_ast_preserves_all_three_callable_pairs():
    function = _load_function_node()
    lambda_bindings = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Lambda)
        and any(
            isinstance(target, ast.Name)
            and target.id in CALLABLE_NAMES
            for target in node.targets
        )
    ]
    nested_callables = Counter(
        node.name
        for node in ast.walk(function)
        if isinstance(node, ast.FunctionDef)
        and node.name in CALLABLE_NAMES
    )

    assert lambda_bindings == []
    assert nested_callables == Counter(
        {"predict_one": 3, "predict_batch": 3}
    )
    assert _normalized_hash(function) == EXPECTED_NORMALIZED_HASH


@pytest.mark.parametrize(
    ("method", "expected_init", "expects_build"),
    [
        pytest.param("bm25", ("init", "bm25", {}), True, id="bm25"),
        pytest.param(
            "lsa",
            ("init", "lsa", {"n_components": 100}),
            True,
            id="lsa",
        ),
        pytest.param(
            "template",
            ("init", "template", {}),
            False,
            id="template",
        ),
    ],
)
def test_cpu_benchmark_preserves_each_method_call_trace(
    monkeypatch,
    method,
    expected_init,
    expects_build,
):
    events = []
    _install_fake_baselines(monkeypatch, events)
    benchmark = _load_benchmark_function(_Clock())
    train_data = ["train-a", "train-b"]
    test_inputs = ["warmup", "latency-a", "latency-b", "unused"]

    load_time, latencies, throughput, memory = benchmark(
        method,
        train_data,
        test_inputs,
        n_warmup=1,
        n_latency=2,
        n_throughput=3,
    )

    assert events[0] == expected_init
    event_offset = 1
    if expects_build:
        assert events[1] == ("build", method, tuple(train_data))
        event_offset = 2
    assert events[event_offset:] == [
        ("batch", method, ("warmup",)),
        ("batch", method, ("latency-a",)),
        ("batch", method, ("latency-b",)),
        ("batch", method, tuple(test_inputs[:3])),
    ]
    assert load_time == pytest.approx(0.25)
    assert latencies == pytest.approx([10.0, 20.0])
    assert throughput == {
        "n_samples": 3,
        "wall_time_s": 0.5,
        "batch_size": 3,
        "samples_per_sec": 6.0,
    }
    assert memory == []
