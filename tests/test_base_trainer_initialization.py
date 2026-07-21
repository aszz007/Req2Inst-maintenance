"""Focused regression coverage for BaseTrainer path initialization."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER = ROOT / "src" / "training" / "base_trainer.py"


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class _HistoryCallback:
    pass


def _load_initializer():
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
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    project_root = Path("project-root")
    path_cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        get_text_model_path=lambda: project_root / "base_models" / "Qwen3-8B",
    )
    train_cfg = SimpleNamespace(num_epochs=3)
    device_cfg = object()
    model_cfg = SimpleNamespace(version="configured_version")
    namespace = {
        "Optional": Optional,
        "os": os,
        "Path": Path,
        "get_path_config": lambda: path_cfg,
        "get_training_config": lambda: train_cfg,
        "get_device_config": lambda: device_cfg,
        "get_model_config": lambda: model_cfg,
        "TrainingHistoryCallback": _HistoryCallback,
        "logger": _Logger(),
    }
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec(compile(isolated_module, str(BASE_TRAINER), "exec"), namespace)
    return namespace["__init__"], path_cfg, train_cfg, device_cfg, model_cfg


@pytest.mark.parametrize(
    ("base_model_path", "output_dir", "expected_version", "expected_output"),
    [
        (
            None,
            None,
            "qwen3_8b",
            Path("project-root") / "checkpoints" / "lora_moe" / "image_expert",
        ),
        (
            "custom/model",
            "custom-output",
            "configured_version",
            Path("custom-output"),
        ),
    ],
)
def test_base_trainer_initialization_preserves_checkpoint_paths(
    monkeypatch,
    base_model_path,
    output_dir,
    expected_version,
    expected_output,
):
    monkeypatch.delenv("TRAIN_EPOCHS", raising=False)
    initializer, path_cfg, train_cfg, device_cfg, model_cfg = _load_initializer()
    trainer = SimpleNamespace()

    initializer(
        trainer,
        expert_type="image",
        method_name="lora_moe",
        base_model_path=base_model_path,
        output_dir=output_dir,
    )

    assert trainer.path_cfg is path_cfg
    assert trainer.train_cfg is train_cfg
    assert trainer.device_cfg is device_cfg
    assert trainer.model_cfg is model_cfg
    assert trainer.model_version == expected_version
    assert trainer.output_dir == expected_output
    assert trainer.checkpoint_dir == expected_output / "training_checkpoints"
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.train_dataset is None
    assert trainer.val_dataset is None
    assert trainer.test_dataset is None
    assert isinstance(trainer.history_callback, _HistoryCallback)
