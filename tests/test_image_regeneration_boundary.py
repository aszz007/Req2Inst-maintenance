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
    assert isinstance(fallback.value, ast.Constant)
    assert fallback.value.value is None


def test_response_text_fallbacks_catch_normal_exceptions_only():
    methods = _class_methods(REPAIR_SCRIPT, "ImageBatchRepairer")

    extract_method = methods["extract_response"]
    extract_handlers = sorted(
        (
            node
            for node in ast.walk(extract_method)
            if isinstance(node, ast.ExceptHandler)
        ),
        key=lambda node: node.lineno,
    )
    assert len(extract_handlers) == 2
    assert all(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in extract_handlers
    )

    extract_fallback = extract_handlers[0].body[0]
    assert isinstance(extract_fallback, ast.Assign)
    assert isinstance(extract_fallback.targets[0], ast.Name)
    assert extract_fallback.targets[0].id == "response_text"
    assert isinstance(extract_fallback.value, ast.Attribute)
    assert isinstance(extract_fallback.value.value, ast.Name)
    assert extract_fallback.value.value.id == "last_response"
    assert extract_fallback.value.attr == "text"

    validate_method = methods["_validate_new_response"]
    validate_handlers = sorted(
        (
            node
            for node in ast.walk(validate_method)
            if isinstance(node, ast.ExceptHandler)
        ),
        key=lambda node: node.lineno,
    )
    assert len(validate_handlers) == 2
    assert all(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in validate_handlers
    )

    validate_fallback = validate_handlers[0].body[0]
    assert isinstance(validate_fallback, ast.Assign)
    assert isinstance(validate_fallback.targets[0], ast.Name)
    assert validate_fallback.targets[0].id == "text"
    assert isinstance(validate_fallback.value, ast.Call)
    assert isinstance(validate_fallback.value.func, ast.Attribute)
    assert validate_fallback.value.func.attr == "strip"
    assert isinstance(validate_fallback.value.func.value, ast.Attribute)
    assert validate_fallback.value.func.value.attr == "text"
    assert isinstance(validate_fallback.value.func.value.value, ast.Name)
    assert validate_fallback.value.func.value.value.id == "last_response"
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in validate_handlers[1].body
    )

def test_csv_encoding_fallback_catches_normal_exceptions_only():
    methods = _class_methods(REPAIR_SCRIPT, "ImageBatchRepairer")
    method = methods["repair_file"]
    handlers = sorted(
        (
            node
            for node in ast.walk(method)
            if isinstance(node, ast.ExceptHandler)
        ),
        key=lambda node: node.lineno,
    )

    assert len(handlers) == 3
    assert all(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in handlers
    )

    encoding_loop = handlers[0].body[0]
    assert isinstance(encoding_loop, ast.For)
    assert isinstance(encoding_loop.target, ast.Name)
    assert encoding_loop.target.id == "enc"
    assert ast.literal_eval(encoding_loop.iter) == [
        "utf-8",
        "gbk",
        "gb18030",
        "latin1",
    ]
    assert any(isinstance(node, ast.Break) for node in ast.walk(encoding_loop))
    assert any(isinstance(node, ast.Continue) for node in handlers[1].body)

    failure_raise = encoding_loop.orelse[0]
    assert isinstance(failure_raise, ast.Raise)
    assert isinstance(failure_raise.exc, ast.Call)
    assert isinstance(failure_raise.exc.func, ast.Name)
    assert failure_raise.exc.func.id == "Exception"
    assert ast.literal_eval(failure_raise.exc.args[0]) == "Failed to read the file"

    outer_fallback = next(
        node
        for node in handlers[2].body
        if isinstance(node, ast.Return)
    )
    assert isinstance(outer_fallback.value, ast.Constant)
    assert outer_fallback.value.value == 0

def test_response_count_fallback_catches_normal_exceptions_only():
    methods = _class_methods(REPAIR_SCRIPT, "ImageBatchRepairer")
    method = methods["get_current_response_count"]
    handlers = sorted(
        (
            node
            for node in ast.walk(method)
            if isinstance(node, ast.ExceptHandler)
        ),
        key=lambda node: node.lineno,
    )

    assert len(handlers) == 3
    assert all(
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        for handler in handlers
    )
    assert any(isinstance(node, ast.Continue) for node in handlers[0].body)
    assert any(isinstance(node, ast.Continue) for node in handlers[1].body)

    outer_fallback = handlers[2].body[0]
    assert isinstance(outer_fallback, ast.Return)
    assert isinstance(outer_fallback.value, ast.Constant)
    assert outer_fallback.value.value == 0

    selectors_assignment = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "response_selectors"
    )
    assert ast.literal_eval(selectors_assignment.value) == [
        "div[data-message-author-role='assistant']",
        "article[data-turn='assistant']",
        "article[data-testid*='conversation-turn'] div.markdown.prose",
    ]

    outer_try = next(node for node in method.body if isinstance(node, ast.Try))
    normal_fallback = next(
        node
        for node in outer_try.body
        if isinstance(node, ast.Return)
    )
    assert isinstance(normal_fallback.value, ast.Constant)
    assert normal_fallback.value.value == 0
