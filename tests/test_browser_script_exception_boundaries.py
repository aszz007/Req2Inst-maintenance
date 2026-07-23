"""Offline regression coverage for browser-script exception boundaries."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
IMAGE_GENERATE = ROOT / "scripts/preprocessing/build_final_dataset/image/generate_instructions.py"
TEXT_GENERATE = ROOT / "scripts/preprocessing/build_final_dataset/text/generate_instructions.py"
UML_GENERATE = ROOT / "scripts/preprocessing/build_final_dataset/uml/generate_instructions_uml.py"
IMAGE_REGENERATE = ROOT / "scripts/preprocessing/build_final_dataset/image/regenerate_failed.py"
TEXT_REGENERATE = ROOT / "scripts/preprocessing/build_final_dataset/text/regenerate_failed.py"
UML_REGENERATE = ROOT / "scripts/preprocessing/build_final_dataset/uml/regenerate_failed_uml.py"

SCRIPT_CASES = (
    (IMAGE_GENERATE, "GPTAutomator"),
    (TEXT_GENERATE, "GPTAutomator"),
    (UML_GENERATE, "GPTAutomator"),
    (IMAGE_REGENERATE, "ImageBatchRepairer"),
    (TEXT_REGENERATE, "TextBatchRepairer"),
    (UML_REGENERATE, "UMLBatchRepairer"),
)
PROGRESS_CASES = tuple(case for case in SCRIPT_CASES if case[0] != TEXT_REGENERATE)
EXTRACTION_CASES = tuple(case for case in SCRIPT_CASES if case[0] != TEXT_REGENERATE)
VALIDATION_CASES = (
    (IMAGE_GENERATE, "GPTAutomator"),
    (UML_GENERATE, "GPTAutomator"),
    (IMAGE_REGENERATE, "ImageBatchRepairer"),
    (UML_REGENERATE, "UMLBatchRepairer"),
)
WAIT_CASES = (
    (TEXT_GENERATE, "GPTAutomator", "wait_for_response_complete"),
    (TEXT_REGENERATE, "TextBatchRepairer", "wait_for_response"),
)


class _MissingElement(Exception):
    pass


class _FailingDriver:
    def __init__(self, error_type):
        self.error_type = error_type

    def raise_failure(self):
        raise self.error_type("offline browser fixture failure")

    def find_element(self, *_args, **_kwargs):
        self.raise_failure()

    def find_elements(self, *_args, **_kwargs):
        self.raise_failure()


class _FixtureWait:
    def __init__(self, driver, _timeout):
        self.driver = driver

    def until(self, _condition):
        self.driver.raise_failure()


class _Response:
    def __init__(self, text, nested_error=RuntimeError):
        self.text = text
        self.nested_error = nested_error

    def find_element(self, *_args, **_kwargs):
        raise self.nested_error("nested response fixture failure")


class _ResponseDriver:
    def __init__(self, response):
        self.response = response

    def find_elements(self, *_args, **_kwargs):
        return [self.response]

    def find_element(self, *_args, **_kwargs):
        return self.response


def _case_id(value):
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)).replace("\\", "/")
    return value


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _load_method(path, class_name, method_name):
    class_node = next(
        node
        for node in _parse(path).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "By": SimpleNamespace(
            CSS_SELECTOR="css selector",
            TAG_NAME="tag name",
            XPATH="xpath",
        ),
        "CONTENT_STABLE_CHECKS": 3,
        "EC": SimpleNamespace(presence_of_element_located=lambda locator: locator),
        "NoSuchElementException": _MissingElement,
        "WebDriverWait": _FixtureWait,
        "time": time,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_browser_scripts_have_no_bare_exception_handlers(path, class_name):
    del class_name
    bare_handlers = [
        node
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]

    assert bare_handlers == []


@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_input_box_lookup_preserves_regular_failure_fallback(path, class_name):
    method = _load_method(path, class_name, "find_input_box")
    target = SimpleNamespace(
        cached_input_selector="cached selector",
        driver=_FailingDriver(RuntimeError),
    )

    try:
        result = method(target)
    except _MissingElement:
        result = None

    assert result is None
    assert target.cached_input_selector is None


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_input_box_lookup_does_not_swallow_base_exceptions(path, class_name, error_type):
    method = _load_method(path, class_name, "find_input_box")
    target = SimpleNamespace(
        cached_input_selector="cached selector",
        driver=_FailingDriver(error_type),
    )

    with pytest.raises(error_type):
        method(target)


@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_submit_button_lookup_preserves_regular_failure_fallback(path, class_name):
    method = _load_method(path, class_name, "find_submit_button")
    target = SimpleNamespace(
        cached_button_selector="cached selector",
        driver=_FailingDriver(RuntimeError),
    )

    assert method(target) is None
    assert target.cached_button_selector is None


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_submit_button_lookup_does_not_swallow_base_exceptions(path, class_name, error_type):
    method = _load_method(path, class_name, "find_submit_button")
    target = SimpleNamespace(
        cached_button_selector="cached selector",
        driver=_FailingDriver(error_type),
    )

    with pytest.raises(error_type):
        method(target)


@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_response_count_preserves_zero_fallback(path, class_name):
    method = _load_method(path, class_name, "get_current_response_count")
    target = SimpleNamespace(driver=_FailingDriver(RuntimeError))

    assert method(target) == 0


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("path,class_name", SCRIPT_CASES, ids=_case_id)
def test_response_count_does_not_swallow_base_exceptions(path, class_name, error_type):
    method = _load_method(path, class_name, "get_current_response_count")
    target = SimpleNamespace(driver=_FailingDriver(error_type))

    with pytest.raises(error_type):
        method(target)


@pytest.mark.parametrize("path,class_name", PROGRESS_CASES, ids=_case_id)
def test_progress_check_preserves_false_fallback(path, class_name):
    method = _load_method(path, class_name, "check_response_still_updating")
    target = SimpleNamespace(
        driver=_FailingDriver(RuntimeError),
        response_count_before_send=0,
    )

    assert method(target) is False


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("path,class_name", PROGRESS_CASES, ids=_case_id)
def test_progress_check_does_not_swallow_base_exceptions(path, class_name, error_type):
    method = _load_method(path, class_name, "check_response_still_updating")
    target = SimpleNamespace(
        driver=_FailingDriver(error_type),
        response_count_before_send=0,
    )

    with pytest.raises(error_type):
        method(target)


@pytest.mark.parametrize("path,class_name", EXTRACTION_CASES, ids=_case_id)
def test_response_extraction_preserves_regular_fallback(path, class_name):
    method = _load_method(path, class_name, "extract_response")
    response_text = "Definition: offline fixture response"
    target = SimpleNamespace(
        driver=_ResponseDriver(_Response(response_text)),
        response_count_before_send=0,
    )

    assert method(target) == response_text


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("path,class_name", EXTRACTION_CASES, ids=_case_id)
def test_response_extraction_does_not_swallow_base_exceptions(path, class_name, error_type):
    method = _load_method(path, class_name, "extract_response")
    target = SimpleNamespace(
        driver=_FailingDriver(error_type),
        response_count_before_send=0,
    )

    with pytest.raises(error_type):
        method(target)


@pytest.mark.parametrize("path,class_name", VALIDATION_CASES, ids=_case_id)
def test_response_validation_preserves_parent_text_fallback(path, class_name):
    method = _load_method(path, class_name, "_validate_new_response")
    target = SimpleNamespace(
        driver=_ResponseDriver(_Response("validated offline response")),
        response_count_before_send=0,
    )

    assert method(target) is True


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("path,class_name", VALIDATION_CASES, ids=_case_id)
def test_response_validation_does_not_swallow_base_exceptions(path, class_name, error_type):
    method = _load_method(path, class_name, "_validate_new_response")
    target = SimpleNamespace(
        driver=_ResponseDriver(
            _Response("validated offline response", nested_error=error_type)
        ),
        response_count_before_send=0,
    )

    with pytest.raises(error_type):
        method(target)


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize(
    "path,class_name,method_name",
    WAIT_CASES,
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_wait_loops_do_not_swallow_base_exceptions(
    path,
    class_name,
    method_name,
    error_type,
):
    method = _load_method(path, class_name, method_name)

    def interrupt():
        raise error_type("offline counter fixture failure")

    target = SimpleNamespace(
        get_current_response_count=interrupt,
        response_count_before_send=0,
    )

    with pytest.raises(error_type):
        method(target)
