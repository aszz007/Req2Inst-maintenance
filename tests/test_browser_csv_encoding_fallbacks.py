"""Regression coverage for browser-script CSV encoding fallbacks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "scripts/preprocessing/build_final_dataset/image/generate_instructions.py",
    ROOT / "scripts/preprocessing/build_final_dataset/text/generate_instructions.py",
    ROOT / "scripts/preprocessing/build_final_dataset/uml/generate_instructions_uml.py",
    ROOT / "scripts/preprocessing/build_final_dataset/image/regenerate_failed.py",
    ROOT / "scripts/preprocessing/build_final_dataset/text/regenerate_failed.py",
    ROOT / "scripts/preprocessing/build_final_dataset/uml/regenerate_failed_uml.py",
)


def _is_read_csv_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pd"
        and node.func.attr == "read_csv"
    )


def _load_fallback(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(_is_read_csv_call(call) for statement in node.body for call in ast.walk(statement))
        and any(
            isinstance(statement, ast.For)
            for handler in node.handlers
            for statement in handler.body
        )
    )
    function = ast.FunctionDef(
        name="run_fallback",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="pd"), ast.arg(arg="csv_path"), ast.arg(arg="encoding")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[fallback, ast.Return(value=ast.Name(id="df", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_fallback"]


def _script_id(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


@pytest.mark.parametrize("path", SCRIPTS, ids=_script_id)
def test_fallback_order_and_success_are_preserved(path):
    fallback = _load_fallback(path)
    calls = []
    result = object()

    def read_csv(_path, encoding):
        calls.append(encoding)
        if encoding == "gb18030":
            return result
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "fixture")

    actual = fallback(
        SimpleNamespace(read_csv=read_csv),
        "dataset.csv",
        "detected-encoding",
    )

    assert actual is result
    assert calls == ["detected-encoding", "utf-8", "gbk", "gb18030"]


@pytest.mark.parametrize("path", SCRIPTS, ids=_script_id)
def test_initial_read_does_not_swallow_keyboard_interrupt(path):
    fallback = _load_fallback(path)
    calls = 0

    def interrupt(_path, encoding):
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fallback(SimpleNamespace(read_csv=interrupt), "dataset.csv", "detected")

    assert calls == 1


@pytest.mark.parametrize("path", SCRIPTS, ids=_script_id)
def test_fallback_read_does_not_swallow_keyboard_interrupt(path):
    fallback = _load_fallback(path)
    calls = 0

    def interrupt_fallback(_path, encoding):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UnicodeDecodeError("utf-8", b"x", 0, 1, "fixture")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fallback(
            SimpleNamespace(read_csv=interrupt_fallback),
            "dataset.csv",
            "detected",
        )

    assert calls == 2
