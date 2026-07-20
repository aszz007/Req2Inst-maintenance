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
    assert "clean_json_data" not in repair_methods
    assert "parse_instructions" not in repair_methods
    assert "process_file" in generation_methods

    run_calls = _self_calls(repair_methods["run"])
    assert "repair_file" in run_calls
    assert "process_file" not in run_calls

def test_input_box_fallback_catches_normal_exceptions_only():
    methods = _class_methods(REPAIR_SCRIPT, "ImageBatchRepairer")
    handlers = sorted(
        (
            node
            for node in ast.walk(methods["find_input_box"])
            if isinstance(node, ast.ExceptHandler)
        ),
        key=lambda node: node.lineno,
    )

    assert len(handlers) == 2
    assert all(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in handlers
    )

    cached_selector_reset = handlers[0].body[0]
    assert isinstance(cached_selector_reset, ast.Assign)
    assert isinstance(cached_selector_reset.targets[0], ast.Attribute)
    assert cached_selector_reset.targets[0].attr == "cached_input_selector"
    assert isinstance(cached_selector_reset.value, ast.Constant)
    assert cached_selector_reset.value.value is None

    assert any(isinstance(node, ast.Continue) for node in handlers[1].body)


def test_submit_button_fallback_catches_normal_exceptions_only():
    methods = _class_methods(REPAIR_SCRIPT, "ImageBatchRepairer")
    method = methods["find_submit_button"]
    handlers = sorted(
        (
            node
            for node in ast.walk(method)
            if isinstance(node, ast.ExceptHandler)
        ),
        key=lambda node: node.lineno,
    )

    assert len(handlers) == 2
    assert all(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in handlers
    )

    cached_selector_reset = handlers[0].body[0]
    assert isinstance(cached_selector_reset, ast.Assign)
    assert isinstance(cached_selector_reset.targets[0], ast.Attribute)
    assert cached_selector_reset.targets[0].attr == "cached_button_selector"
    assert isinstance(cached_selector_reset.value, ast.Constant)
    assert cached_selector_reset.value.value is None

    assert any(isinstance(node, ast.Continue) for node in handlers[1].body)

    selectors_assignment = next(
        node
        for node in method.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "selectors"
    )
    assert ast.literal_eval(selectors_assignment.value) == [
        "button[data-testid='send-button']",
        "button[type='submit']",
        "button:has(svg)",
        "button[aria-label*='Send']",
        "button[aria-label*='\u53d1\u9001']",
    ]

    fallback = method.body[-1]
    assert isinstance(fallback, ast.Return)
