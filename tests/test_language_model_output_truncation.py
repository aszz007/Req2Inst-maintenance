"""Focused regression coverage for the three-section output boundary."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_MODEL = ROOT / "models" / "language_model.py"


def _load_truncator():
    tree = ast.parse(
        LANGUAGE_MODEL.read_text(encoding="utf-8"),
        filename=str(LANGUAGE_MODEL),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LanguageModel"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_truncate_after_three_parts"
    )
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {}
    exec(compile(isolated_module, str(LANGUAGE_MODEL), "exec"), namespace)
    return namespace["_truncate_after_three_parts"]


TRUNCATE = _load_truncator()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Definition: A requirement.\n"
            "Emphasis & Caution: Validate it.\n"
            "Things to Avoid: Do not guess. Do not omit checks.\n\n"
            "Trailing commentary.",
            "Definition: A requirement.\n"
            "Emphasis & Caution: Validate it.\n"
            "Things to Avoid: Do not guess. Do not omit checks.",
        ),
        (
            "Definition: A requirement.\n"
            "Emphasis and Caution: Validate it.\n"
            "Things to Avoid: Do not one. Do not two. Do not three. "
            "Do not four. Do not five. Do not six. Do not seven.",
            "Definition: A requirement.\n"
            "Emphasis and Caution: Validate it.\n"
            "Things to Avoid: Do not one. Do not two. Do not three. "
            "Do not four. Do not five. Do not six.",
        ),
        (
            "Definition: A requirement. Emphasis & Caution: Validate it. "
            "Things to Avoid: Do not guess.",
            "Definition: A requirement.\n"
            "Emphasis & Caution: Validate it.\n"
            "Things to Avoid: Do not guess.",
        ),
        (
            "Definition: A requirement.\n"
            "Emphasis & Caution: Validate it.\n"
            "Things to Avoid:\nDo not guess.\nDo not omit checks.\n"
            "Definition: unrelated repeated section",
            "Definition: A requirement.\n"
            "Emphasis & Caution: Validate it.\n"
            "Things to Avoid: Do not guess. Do not omit checks.",
        ),
        (
            "A response without the required section headers.",
            "A response without the required section headers.",
        ),
    ],
)
def test_truncate_after_three_parts_preserves_current_contract(text, expected):
    assert TRUNCATE(None, text) == expected
