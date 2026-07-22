"""Regression contracts for BaseExpert import and initialization order."""

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_EXPERT = ROOT / "src" / "experts" / "base_expert.py"
EXPECTED_NON_IMPORT_HASH = (
    "23bf4cd29e9d86a9a36e0bd545972f407904776b98ad4e39cd700500d2a8d687"
)
EXPECTED_IMPORT_SIGNATURE = [
    ("import", None, ("os",)),
    ("from", "abc", ("ABC", "abstractmethod")),
    ("from", "pathlib", ("Path",)),
    ("from", "typing", ("Optional", "Dict", "Any")),
    ("import", None, ("torch",)),
    ("from", "models.language_model", ("LanguageModel",)),
    ("from", "src.utils.logger", ("get_logger",)),
]


def _parse():
    return ast.parse(
        BASE_EXPERT.read_text(encoding="utf-8"),
        filename=str(BASE_EXPERT),
    )


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _import_signature(tree):
    signature = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            signature.append(
                (
                    "import",
                    None,
                    tuple(alias.name for alias in node.names),
                )
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


def _non_import_hash(tree):
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    dumped = ast.dump(module, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _call_index(tree, dotted_name):
    return next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == dotted_name
    )


def _import_index(tree, module_name, binding_name):
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Import) and module_name is None:
            if any(alias.name == binding_name for alias in node.names):
                return index
        elif isinstance(node, ast.ImportFrom) and node.module == module_name:
            if any(alias.name == binding_name for alias in node.names):
                return index
    raise AssertionError(f"Missing import binding: {binding_name}")


def _assignment_index(tree, binding_name):
    return next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == binding_name
            for target in node.targets
        )
    )


def test_base_expert_imports_preserve_initialization_boundaries():
    tree = _parse()
    env_index = _call_index(tree, "os.environ.setdefault")
    torch_index = _import_index(tree, None, "torch")
    cpu_count_index = _assignment_index(tree, "_cpu_count")
    set_threads_index = _call_index(tree, "torch.set_num_threads")
    set_interop_index = _call_index(
        tree,
        "torch.set_num_interop_threads",
    )
    language_model_index = _import_index(
        tree,
        "models.language_model",
        "LanguageModel",
    )
    logger_index = _import_index(
        tree,
        "src.utils.logger",
        "get_logger",
    )

    assert _import_signature(tree) == EXPECTED_IMPORT_SIGNATURE
    assert _non_import_hash(tree) == EXPECTED_NON_IMPORT_HASH
    assert _import_signature(tree).count(
        ("import", None, ("torch",))
    ) == 1
    assert max(
        _import_index(tree, "abc", "ABC"),
        _import_index(tree, "pathlib", "Path"),
        _import_index(tree, "typing", "Optional"),
    ) < env_index
    assert (
        env_index
        < torch_index
        < cpu_count_index
        < set_threads_index
        < set_interop_index
        < language_model_index
        < logger_index
    )
