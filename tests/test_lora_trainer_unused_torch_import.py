"""Regression contracts for the LoRA trainer's PyTorch dependency boundary."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
LORA_TRAINER = ROOT / "src" / "training" / "lora_trainer.py"
BASE_TRAINER = ROOT / "src" / "training" / "base_trainer.py"
NON_IMPORT_AST_HASH = (
    "73e86c4553161126185908985636575f8ccce817fb38f5cc4dac2a0ef51e3b7f"
)
EXPECTED_TOP_LEVEL_IMPORTS = [
    ("from", "typing", ("Optional",)),
    (
        "from",
        "peft",
        ("LoraConfig", "get_peft_model", "TaskType"),
    ),
    ("from", "src.training.base_trainer", ("BaseTrainer",)),
    ("from", "src.utils.logger", ("get_logger",)),
]


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_import_signature(path):
    signature = []
    for node in _parse(path).body:
        if isinstance(node, ast.Import):
            signature.append(
                ("import", None, tuple(alias.name for alias in node.names))
            )
        elif isinstance(node, ast.ImportFrom):
            signature.append(
                (
                    "from",
                    node.module,
                    tuple(alias.name for alias in node.names),
                )
            )
    return signature


def _loads_name(path, name):
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == name
        for node in ast.walk(_parse(path))
    )


def _load_lora_trainer(namespace):
    tree = _parse(LORA_TRAINER)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LoRATrainer"
    )
    module = ast.Module(body=[trainer_class], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LORA_TRAINER), "exec"), namespace)
    return namespace["LoRATrainer"]


def test_non_import_body_and_top_level_imports_match_the_cleanup_contract():
    tree = _parse(LORA_TRAINER)
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ],
        type_ignores=[],
    )
    digest = hashlib.sha256(
        ast.dump(module, include_attributes=False).encode("utf-8")
    ).hexdigest()

    assert digest == NON_IMPORT_AST_HASH
    assert _top_level_import_signature(LORA_TRAINER) == (
        EXPECTED_TOP_LEVEL_IMPORTS
    )
    assert not _loads_name(LORA_TRAINER, "torch")


def test_base_trainer_keeps_the_runtime_torch_dependency():
    assert ("import", None, ("torch",)) in _top_level_import_signature(
        BASE_TRAINER
    )
    assert _loads_name(BASE_TRAINER, "torch")


def test_setup_model_preserves_the_lora_configuration_and_wrapping_loop():
    captured = {}

    class _Logger:
        def __init__(self):
            self.info_messages = []
            self.error_messages = []

        def info(self, message):
            self.info_messages.append(message)

        def error(self, message):
            self.error_messages.append(message)

    class _BaseTrainer:
        pass

    class _LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Parameter:
        def __init__(self, count, requires_grad):
            self.count = count
            self.requires_grad = requires_grad

        def numel(self):
            return self.count

    class _AdapterModel:
        def parameters(self):
            return [
                _Parameter(10, True),
                _Parameter(90, False),
            ]

    adapter_model = _AdapterModel()

    def get_peft_model(model, config):
        captured["base_model"] = model
        captured["config"] = config
        return adapter_model

    logger = _Logger()
    trainer_class = _load_lora_trainer(
        {
            "BaseTrainer": _BaseTrainer,
            "Optional": Optional,
            "LoraConfig": _LoraConfig,
            "get_peft_model": get_peft_model,
            "TaskType": SimpleNamespace(CAUSAL_LM="causal-lm"),
            "logger": logger,
        }
    )
    trainer = object.__new__(trainer_class)
    base_model = object()
    trainer.model = base_model
    trainer.use_4bit = False
    trainer.lora_rank = 16
    trainer.lora_alpha = 32
    trainer.lora_dropout = 0.1
    trainer.target_modules = ["q_proj", "v_proj"]

    def load_base_model(use_4bit):
        captured["use_4bit"] = use_4bit
        return True

    trainer._load_base_model = load_base_model

    assert trainer.setup_model() is True
    assert captured["use_4bit"] is False
    assert captured["base_model"] is base_model
    assert captured["config"].kwargs == {
        "task_type": "causal-lm",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    }
    assert trainer.model is adapter_model
    assert logger.error_messages == []
