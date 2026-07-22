"""Regression contracts for side-effect-free evaluation import cleanup."""

import ast
import copy
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    pytest.param(
        "scripts/evaluation/experiments/exp8_inference_efficiency.py",
        {"group_split_by_input"},
        "f3b4e2510c1e83538a40c1b9f87a50f343b37d725f1a07f2730dfddfc8b2b560",
        id="exp8-group-split-binding",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp11_ablation_optimization.py",
        {
            "HiddenStateExtractor",
            "TextDatasetLoader",
            "ImageDatasetLoader",
            "UMLDatasetLoader",
        },
        "b27155d49d1da931158021fb98f7eb1f983ef914362b85911092de6a132204fa",
        id="exp11-router-and-loader-bindings",
    ),
]


class _StripBindings(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Import(self, node):
        node.names = [
            alias
            for alias in node.names
            if (alias.asname or alias.name.split(".")[0]) not in self.names
        ]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        node.names = [
            alias
            for alias in node.names
            if (alias.asname or alias.name) not in self.names
        ]
        return node if node.names else None


def _parse(relative_path):
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_bindings(tree):
    bindings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bindings.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            bindings.update(alias.asname or alias.name for alias in node.names)
    return bindings


def _normalized_hash(tree, stripped_names):
    normalized = _StripBindings(stripped_names).visit(copy.deepcopy(tree))
    ast.fix_missing_locations(normalized)
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    ("relative_path", "removed_bindings", "expected_hash"),
    SPECS,
)
def test_unused_bindings_are_absent_and_ast_contract_is_preserved(
    relative_path,
    removed_bindings,
    expected_hash,
):
    tree = _parse(relative_path)
    loaded_targets = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in removed_bindings
    }

    assert removed_bindings.isdisjoint(_imported_bindings(tree))
    assert loaded_targets == set()
    assert _normalized_hash(tree, removed_bindings) == expected_hash


def test_exp8_keeps_used_dataset_imports_and_group_split_is_definition_only():
    exp8_tree = _parse(
        "scripts/evaluation/experiments/exp8_inference_efficiency.py"
    )
    assert {"TextDatasetLoader", "split_dataset_for_expert"} <= (
        _imported_bindings(exp8_tree)
    )

    group_split_tree = _parse("src/utils/group_split.py")
    executable_nodes = [
        node
        for node in group_split_tree.body
        if not (
            isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        )
    ]
    assert executable_nodes == []


def test_exp11_keeps_used_router_and_dataset_bindings():
    tree = _parse(
        "scripts/evaluation/experiments/exp11_ablation_optimization.py"
    )
    assert {
        "RouterMLP",
        "EXPERT_TO_IDX",
        "IDX_TO_EXPERT",
        "GeneralDatasetLoader",
        "split_dataset_for_expert",
    } <= _imported_bindings(tree)
