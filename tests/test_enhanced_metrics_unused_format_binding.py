"""Regression contracts for the discarded format-metrics return value."""

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
ENHANCED_METRICS = ROOT / "src" / "utils" / "enhanced_metrics.py"
EXPECTED_NORMALIZED_HASH = (
    "fed9b5e3010c4ae1d375bbf7fcfad6904b3e22b1902a4b793de2747b22dd55d2"
)


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)


def _parse():
    return ast.parse(
        ENHANCED_METRICS.read_text(encoding="utf-8"),
        filename=str(ENHANCED_METRICS),
    )


def _class_node(tree, class_name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _method_node(class_node, method_name):
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _assigns_format_results(statement):
    return isinstance(statement, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "format_results"
        for target in statement.targets
    )


def _normalized_hash(method):
    method = copy.deepcopy(method)
    for owner in ast.walk(method):
        body = getattr(owner, "body", None)
        if not isinstance(body, list):
            continue
        owner.body = [
            ast.Expr(value=statement.value)
            if _assigns_format_results(statement)
            else statement
            for statement in body
        ]
    ast.fix_missing_locations(method)
    dumped = ast.dump(method, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _load_metrics_class():
    tree = _parse()
    module = ast.Module(
        body=[
            _class_node(tree, "EvaluationThresholds"),
            _class_node(tree, "EnhancedMetrics"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    logger = _Logger()
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "logger": logger,
    }
    exec(compile(module, str(ENHANCED_METRICS), "exec"), namespace)
    return namespace["EnhancedMetrics"], logger


def test_binary_metrics_ast_preserves_the_format_metrics_call():
    tree = _parse()
    metrics_class = _class_node(tree, "EnhancedMetrics")
    method = _method_node(
        metrics_class,
        "calculate_binary_classification_metrics",
    )

    format_result_bindings = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Name)
        and node.id == "format_results"
        and isinstance(node.ctx, (ast.Store, ast.Load))
    ]
    format_call_statements = [
        statement
        for statement in method.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "calculate_format_metrics"
    ]

    assert format_result_bindings == []
    assert len(format_call_statements) == 1
    assert _normalized_hash(method) == EXPECTED_NORMALIZED_HASH


def test_binary_metrics_preserves_call_order_and_result_contract():
    metrics_class, logger = _load_metrics_class()
    metrics = metrics_class(use_bertscore=False)
    events = []
    discarded_summary = object()

    def calculate_format_metrics(values):
        events.append(("format_metrics", tuple(values)))
        return discarded_summary

    def lazy_load_metrics():
        events.append(("lazy_load",))

    def check_single_format(value):
        events.append(("single_format", value))
        return {
            "has_definition": True,
            "has_emphasis": True,
            "has_avoid": True,
            "format_score": 1.0,
        }

    class _RougeMetric:
        def compute(self, **kwargs):
            events.append(
                (
                    "rouge",
                    tuple(kwargs["predictions"]),
                    tuple(kwargs["references"]),
                    kwargs["use_aggregator"],
                )
            )
            return {"rougeL": [0.5]}

    metrics.calculate_format_metrics = calculate_format_metrics
    metrics._lazy_load_metrics = lazy_load_metrics
    metrics._check_single_format = check_single_format
    metrics.rouge_metric = _RougeMetric()

    predictions = ["Definition: inspect the item"]
    references = ["Reference instruction"]
    results = metrics.calculate_binary_classification_metrics(
        predictions,
        references,
        format_threshold=1.0,
        rouge_threshold=0.4,
        bertscore_threshold=0.82,
        use_and_logic=True,
        precomputed_bertscore_f1=[0.9],
    )

    assert events == [
        ("format_metrics", tuple(predictions)),
        ("lazy_load",),
        ("rouge", tuple(predictions), tuple(references), False),
        ("single_format", predictions[0]),
    ]
    assert results == {
        "TP": 1,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "accuracy": 1.0,
        "total_samples": 1,
        "valid_samples": [0],
        "invalid_samples": [],
        "format_threshold": 1.0,
        "rouge_threshold": 0.4,
        "bertscore_threshold": 0.82,
        "use_and_logic": True,
        "use_bertscore": False,
    }
    assert logger.error_messages == []
