"""Regression checks for behavior-preserving console output cleanup."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_UNIFIED = ROOT / "scripts/training/lora_single/train_unified_expert.py"
MULTI_EXPERT_TRAINING = {
    "Text Expert Training": ROOT / "scripts/training/lora_moe/train_text_expert.py",
    "Image Expert Training": ROOT / "scripts/training/lora_moe/train_image_expert.py",
    "FlowChart Expert Training": ROOT / "scripts/training/lora_moe/train_uml_expert.py",
    "General Expert Training": ROOT / "scripts/training/lora_moe/train_general_expert.py",
}


def _print_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_unified_training_console_output_has_no_decorative_separators():
    tree = ast.parse(
        TRAIN_UNIFIED.read_text(encoding="utf-8"),
        filename=str(TRAIN_UNIFIED),
    )

    for call in _print_calls(tree):
        if not call.args or not isinstance(call.args[0], ast.Constant):
            continue
        value = call.args[0].value
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        assert not stripped or set(stripped) not in ({"="}, {"-"})


def test_unified_training_console_output_keeps_operational_information():
    source = TRAIN_UNIFIED.read_text(encoding="utf-8")

    required_messages = (
        "LoRA (Unified) Expert Training",
        "Checking runtime environment...",
        "Transformers version:",
        "PEFT version:",
        "PyTorch version:",
        "Comparison configuration:",
        "Prompt templates:",
        "Training samples:",
        "Validation samples:",
        "Training completed successfully!",
        "LoRA weights saved to:",
        "An error occurred during training; check the logs",
    )
    for message in required_messages:
        assert message in source


def test_unified_training_console_output_reports_the_selected_prompt_mode():
    tree = ast.parse(
        TRAIN_UNIFIED.read_text(encoding="utf-8"),
        filename=str(TRAIN_UNIFIED),
    )
    prompt_mode = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "prompt_mode"
            for target in node.targets
        )
    )

    assert isinstance(prompt_mode, ast.IfExp)
    assert ast.unparse(prompt_mode.test) == "args.use_domain_templates"
    assert ast.literal_eval(prompt_mode.body) == "domain-specific"
    assert ast.literal_eval(prompt_mode.orelse) == "unified General"


def _constant_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.BinOp):
        return None

    left = _constant_text(node.left)
    right = _constant_text(node.right)
    if isinstance(node.op, ast.Add) and left is not None and right is not None:
        return left + right
    if isinstance(node.op, ast.Mult):
        if left is not None and isinstance(node.right, ast.Constant):
            if isinstance(node.right.value, int):
                return left * node.right.value
        if right is not None and isinstance(node.left, ast.Constant):
            if isinstance(node.left.value, int):
                return right * node.left.value
    return None


def test_multi_expert_training_console_output_has_no_display_noise():
    for path in MULTI_EXPERT_TRAINING.values():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in _print_calls(tree):
            assert call.args, f"Empty print call remains in {path}"
            value = _constant_text(call.args[0])
            if value is None:
                continue
            stripped = value.strip()
            assert not stripped or set(stripped) not in ({"="}, {"-"})


def test_multi_expert_training_console_output_keeps_operational_information():
    common_messages = (
        "Training configuration:",
        "Checking runtime environment...",
        "Transformers version:",
        "PEFT version:",
        "PyTorch version:",
        "Dataset statistics:",
        "Training samples:",
        "Validation samples:",
        "Training completed successfully!",
        "LoRA weights saved to:",
        "Checkpoint directory:",
        "An error occurred during training; check the logs",
    )
    for header, path in MULTI_EXPERT_TRAINING.items():
        source = path.read_text(encoding="utf-8")
        assert header in source
        for message in common_messages:
            assert message in source
        assert "Training started - this may take a while" not in source
