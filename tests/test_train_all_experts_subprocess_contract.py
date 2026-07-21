"""Regression coverage for the training orchestrator subprocess contract."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "training" / "train_all_experts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "req2inst_train_all_experts",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _is_subprocess_run(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    )


def _prepare_task(tmp_path, monkeypatch, run):
    relative_path = Path("scripts") / "training" / "fake_task.py"
    full_path = tmp_path / relative_path
    full_path.parent.mkdir(parents=True)
    full_path.write_text("raise SystemExit(0)\n", encoding="utf-8")

    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE.subprocess, "run", run)
    clock = iter((100.0, 105.0))
    monkeypatch.setattr(MODULE.time, "time", lambda: next(clock))
    return relative_path, full_path


def test_subprocess_call_is_an_unassigned_expression():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_training_task"
    )
    assigned_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _is_subprocess_run(node.value)
    ]
    expression_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Expr) and _is_subprocess_run(node.value)
    ]

    assert assigned_calls == []
    assert len(expression_calls) == 1


def test_success_preserves_command_cwd_environment_and_options(tmp_path, monkeypatch):
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    relative_path, full_path = _prepare_task(tmp_path, monkeypatch, run)

    success = MODULE.run_training_task("lora_moe", "text", str(relative_path))

    assert success is True
    command = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert command == [sys.executable, str(full_path)]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"]["PYTHONPATH"] == str(tmp_path)
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is False


def test_called_process_error_still_returns_false(tmp_path, monkeypatch):
    run = MagicMock(
        side_effect=subprocess.CalledProcessError(
            returncode=7,
            cmd=[sys.executable, "fake_task.py"],
        )
    )
    relative_path, _ = _prepare_task(tmp_path, monkeypatch, run)

    success = MODULE.run_training_task("lora_moe", "text", str(relative_path))

    assert success is False


def test_keyboard_interrupt_still_propagates(tmp_path, monkeypatch):
    run = MagicMock(side_effect=KeyboardInterrupt)
    relative_path, _ = _prepare_task(tmp_path, monkeypatch, run)

    with pytest.raises(KeyboardInterrupt):
        MODULE.run_training_task("lora_moe", "text", str(relative_path))


def test_missing_script_returns_false_without_starting_a_process(tmp_path, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    success = MODULE.run_training_task(
        "lora_moe",
        "text",
        "scripts/training/missing.py",
    )

    assert success is False
    run.assert_not_called()
