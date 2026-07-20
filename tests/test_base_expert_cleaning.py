"""Focused regression coverage for BaseExpert instruction-line cleanup."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_EXPERT = ROOT / "src" / "experts" / "base_expert.py"


def _load_cleaner():
    tree = ast.parse(
        BASE_EXPERT.read_text(encoding="utf-8"),
        filename=str(BASE_EXPERT),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseExpert"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_clean_instruction_line"
    )
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {}
    exec(compile(isolated_module, str(BASE_EXPERT), "exec"), namespace)
    return namespace["_clean_instruction_line"]


CLEAN_LINE = _load_cleaner()


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Unprefixed content", "Unprefixed content"),
        (
            "Definition: Definition: Definition: A payment service",
            "Definition: A payment service.",
        ),
        (
            "Definition: A payment service.is a software used by teams",
            "Definition: A payment service.",
        ),
        (
            "Emphasis & Caution: Validate inputs. 中文说明",
            "Emphasis & Caution: Validate inputs.",
        ),
        (
            "Things to Avoid: Avoid invalid input. unfinished detail",
            "Things to Avoid: Avoid invalid input.",
        ),
        ("Definition: -", "Definition: -"),
    ],
)
def test_clean_instruction_line_preserves_current_output_contract(line, expected):
    assert CLEAN_LINE(None, line) == expected
