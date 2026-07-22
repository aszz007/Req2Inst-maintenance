"""Regression contracts for GPU detection in training launchers."""

import ast
import builtins
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_LAUNCHERS = [
    "scripts/training/lora_moe/train_general_expert.py",
    "scripts/training/lora_moe/train_image_expert.py",
    "scripts/training/lora_moe/train_text_expert.py",
    "scripts/training/lora_moe/train_uml_expert.py",
    "scripts/training/lora_single/train_unified_expert.py",
    "scripts/training/p_tuning/train_general_expert.py",
    "scripts/training/p_tuning/train_image_expert.py",
    "scripts/training/p_tuning/train_text_expert.py",
    "scripts/training/p_tuning/train_uml_expert.py",
    "scripts/training/prompt_tuning/train_general_expert.py",
    "scripts/training/prompt_tuning/train_image_expert.py",
    "scripts/training/prompt_tuning/train_text_expert.py",
    "scripts/training/prompt_tuning/train_uml_expert.py",
]
TRAINING_LAUNCHERS = [ROOT / path for path in RELATIVE_LAUNCHERS]


def _function_node(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "detect_rtx4090"
    ]
    assert len(matches) == 1
    return matches[0]


def _load_detector(path: Path):
    module = ast.Module(body=[_function_node(path)], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["detect_rtx4090"]


def _install_fake_torch(monkeypatch, *, available, gpu_name="", error=None):
    def is_available():
        if error is not None:
            raise error
        return available

    def get_device_name(index):
        assert index == 0
        return gpu_name

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=is_available,
        get_device_name=get_device_name,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_all_launchers_share_the_same_detector_ast():
    reference = ast.dump(
        _function_node(TRAINING_LAUNCHERS[0]),
        include_attributes=False,
    )
    assert all(
        ast.dump(_function_node(path), include_attributes=False) == reference
        for path in TRAINING_LAUNCHERS[1:]
    )


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
def test_detector_catches_only_exception_subclasses(path):
    handlers = [
        handler
        for node in ast.walk(_function_node(path))
        if isinstance(node, ast.Try)
        for handler in node.handlers
    ]
    assert len(handlers) == 1
    assert isinstance(handlers[0].type, ast.Name)
    assert handlers[0].type.id == "Exception"


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
def test_failed_torch_import_falls_back_to_false(monkeypatch, path):
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert _load_detector(path)() is False


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
def test_cuda_unavailable_returns_false(monkeypatch, path):
    _install_fake_torch(monkeypatch, available=False)
    assert _load_detector(path)() is False


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
@pytest.mark.parametrize(
    "gpu_name",
    ["NVIDIA GeForce RTX 4090", "NVIDIA RTX 4090D"],
)
def test_supported_gpu_names_return_true(monkeypatch, path, gpu_name):
    _install_fake_torch(monkeypatch, available=True, gpu_name=gpu_name)
    assert _load_detector(path)() is True


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
def test_other_gpu_returns_false(monkeypatch, path):
    _install_fake_torch(
        monkeypatch,
        available=True,
        gpu_name="NVIDIA GeForce RTX 4080",
    )
    assert _load_detector(path)() is False


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
def test_ordinary_cuda_error_falls_back_to_false(monkeypatch, path):
    _install_fake_torch(
        monkeypatch,
        available=True,
        error=RuntimeError("CUDA query failed"),
    )
    assert _load_detector(path)() is False


@pytest.mark.parametrize("path", TRAINING_LAUNCHERS, ids=RELATIVE_LAUNCHERS)
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(monkeypatch, path, error_type):
    _install_fake_torch(
        monkeypatch,
        available=True,
        error=error_type(),
    )
    with pytest.raises(error_type):
        _load_detector(path)()
