"""Focused regression coverage for BaseTrainer early-stopping patience."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER = ROOT / "src" / "training" / "base_trainer.py"


def _load_patience_method():
    tree = ast.parse(
        BASE_TRAINER.read_text(encoding="utf-8"),
        filename=str(BASE_TRAINER),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseTrainer"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_early_stopping_patience"
    )
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {}
    exec(compile(isolated_module, str(BASE_TRAINER), "exec"), namespace)
    return namespace["_get_early_stopping_patience"]


PATIENCE = _load_patience_method()


class _Dataset:
    def __init__(self, size):
        self.data = [None] * size

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize(
    ("expert_type", "expected"),
    [
        ("text", 3),
        ("general", 3),
        ("image", 4),
        ("uml", 4),
        ("unknown", 3),
    ],
)
@pytest.mark.parametrize("dataset_size", [None, 0, 1, 1000])
def test_early_stopping_patience_is_independent_of_dataset_size(
    expert_type,
    expected,
    dataset_size,
):
    dataset = None if dataset_size is None else _Dataset(dataset_size)
    trainer = SimpleNamespace(
        expert_type=expert_type,
        train_dataset=dataset,
    )

    assert PATIENCE(trainer) == expected
