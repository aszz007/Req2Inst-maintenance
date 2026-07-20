"""Static regression checks for the image failure-regeneration boundary."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPAIR_SCRIPT = (
    ROOT
    / "scripts"
    / "preprocessing"
    / "build_final_dataset"
    / "image"
    / "regenerate_failed.py"
)
GENERATION_SCRIPT = (
    ROOT
    / "scripts"
    / "preprocessing"
    / "build_final_dataset"
    / "image"
    / "generate_instructions.py"
)


def _class_methods(path: Path, class_name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
    }


def _self_calls(method: ast.FunctionDef) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def test_image_failure_regeneration_keeps_a_single_repair_entrypoint():
    repair_methods = _class_methods(REPAIR_SCRIPT, "ImageBatchRepairer")
    generation_methods = _class_methods(GENERATION_SCRIPT, "GPTAutomator")

    assert "repair_file" in repair_methods
    assert "process_file" not in repair_methods
    assert "process_file" in generation_methods

    run_calls = _self_calls(repair_methods["run"])
    assert "repair_file" in run_calls
    assert "process_file" not in run_calls
