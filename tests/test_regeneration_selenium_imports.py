"""Regression contracts for regeneration-script Selenium imports."""

import ast
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEXT_REGENERATION = (
    ROOT
    / "scripts"
    / "preprocessing"
    / "build_final_dataset"
    / "text"
    / "regenerate_failed.py"
)
UML_REGENERATION = (
    ROOT
    / "scripts"
    / "preprocessing"
    / "build_final_dataset"
    / "uml"
    / "regenerate_failed_uml.py"
)
BODY_HASHES = {
    TEXT_REGENERATION: "eb75aaae205ea5a8628ed7eff0225ac0ab06e3e50e0f55994dba857cf91d292a",
    UML_REGENERATION: "7c4a91129ce61c3fa736d7b94f65aabaeb56d5045243ec4d1803fa251b8c4785",
}


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _body_hash(path):
    tree = _parse(path)
    body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    module = ast.Module(body=body, type_ignores=[])
    return hashlib.sha256(
        ast.dump(module, include_attributes=False).encode()
    ).hexdigest()


def _top_level_bindings(path):
    bindings = set()
    for node in _parse(path).body:
        if isinstance(node, ast.Import):
            bindings.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bindings.update(alias.asname or alias.name for alias in node.names)
    return bindings


@pytest.mark.parametrize("path", BODY_HASHES)
def test_non_import_body_matches_the_approved_contract(path):
    assert _body_hash(path) == BODY_HASHES[path]


def test_text_regeneration_keeps_the_used_exception_binding_only():
    tree = _parse(TEXT_REGENERATION)
    exception_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "selenium.common.exceptions"
    )

    assert [alias.name for alias in exception_import.names] == [
        "NoSuchElementException"
    ]
    assert "TimeoutException" not in _top_level_bindings(TEXT_REGENERATION)


def test_uml_regeneration_keeps_only_used_top_level_selenium_bindings():
    bindings = _top_level_bindings(UML_REGENERATION)

    assert {"webdriver", "By", "Keys"} <= bindings
    assert {
        "WebDriverWait",
        "EC",
        "ActionChains",
        "TimeoutException",
        "NoSuchElementException",
    }.isdisjoint(bindings)


def test_uml_regeneration_preserves_the_local_action_chains_import():
    tree = _parse(UML_REGENERATION)
    start_new_chat = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "start_new_chat"
    )
    local_imports = [
        node
        for node in ast.walk(start_new_chat)
        if isinstance(node, ast.ImportFrom)
        and node.module == "selenium.webdriver.common.action_chains"
    ]
    loaded_names = {
        node.id
        for node in ast.walk(start_new_chat)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    assert len(local_imports) == 1
    assert [alias.name for alias in local_imports[0].names] == ["ActionChains"]
    assert "ActionChains" in loaded_names
