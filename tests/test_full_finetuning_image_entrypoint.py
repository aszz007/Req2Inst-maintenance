"""Focused regression coverage for the image full-finetuning entry point."""

import argparse
import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (
    ROOT
    / "scripts"
    / "training"
    / "full_finetuning"
    / "train_image_expert.py"
)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _load_main(namespace):
    tree = ast.parse(
        ENTRYPOINT.read_text(encoding="utf-8"),
        filename=str(ENTRYPOINT),
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    isolated_module = ast.Module(body=[main], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec(compile(isolated_module, str(ENTRYPOINT), "exec"), namespace)
    return namespace["main"]


def test_image_full_finetuning_main_preserves_config_and_training_order(
    monkeypatch,
    capsys,
):
    events = []

    def get_path_config():
        events.append("get_path_config")
        return object()

    class FakeTrainer:
        output_dir = Path("checkpoints/full_finetuning/image_expert")

        def __init__(self, **kwargs):
            events.append(("trainer_init", kwargs))

        def setup_model(self):
            events.append("setup_model")
            return True

        def prepare_data(self):
            events.append("prepare_data")
            return True

        def train(self):
            events.append("train")
            return True

    def print_header():
        events.append("print_header")

    namespace = {
        "argparse": argparse,
        "get_path_config": get_path_config,
        "FullFineTuningTrainer": FakeTrainer,
        "logger": _Logger(),
        "print_header": print_header,
    }
    main = _load_main(namespace)
    monkeypatch.setattr(sys, "argv", [str(ENTRYPOINT), "--no_4bit"])

    assert main() == 0
    assert events == [
        "print_header",
        "get_path_config",
        (
            "trainer_init",
            {
                "expert_type": "image",
                "use_4bit": False,
                "use_rtx4090_optimization": True,
            },
        ),
        "setup_model",
        "prepare_data",
        "train",
    ]
    output = capsys.readouterr().out
    assert "4-bit quantization: False" in output
    assert "Sample coverage: 100% of Image" in output
    assert "Training completed successfully!" in output
    assert "checkpoints\\full_finetuning\\image_expert" in output
