"""Regression contracts for redundant PyTorch routing imports."""

import ast
import copy
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEARNED_ROUTER = ROOT / "src" / "routing" / "learned_router.py"
SOFT_ROUTER = ROOT / "src" / "routing" / "soft_router.py"
EXTRACT_HASH = (
    "85c962b7d65ddd94a7e708f899ad6bb6d5ee04b5430dc837d0205da1874fcae6"
)
SOFT_NON_IMPORT_HASH = (
    "90306c9408d2c92f2d1da1155616591839d955fdb62cb4c07e95f888d22b78c4"
)
EXPECTED_SOFT_IMPORTS = [
    ("import", None, ("json",)),
    ("import", None, ("shutil",)),
    ("import", None, ("tempfile",)),
    ("from", "pathlib", ("Path",)),
    ("from", "typing", ("Dict", "List")),
    ("from", "dataclasses", ("dataclass",)),
    ("from", "peft", ("PeftModel",)),
    ("from", "src.utils.logger", ("get_logger",)),
]


class _StripTorch(ast.NodeTransformer):
    def visit_Import(self, node):
        node.names = [
            alias
            for alias in node.names
            if (alias.asname or alias.name.split(".")[0]) != "torch"
        ]
        return node if node.names else None


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


def _class_node(path, class_name):
    return next(
        node
        for node in _parse(path).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _method_node(class_node, method_name):
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _imported_bindings(node):
    bindings = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            bindings.update(
                alias.asname or alias.name.split(".")[0]
                for alias in child.names
            )
        elif isinstance(child, ast.ImportFrom):
            bindings.update(alias.asname or alias.name for alias in child.names)
    return bindings


def _loads_name(node, name):
    return any(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id == name
        for child in ast.walk(node)
    )


def _load_class(path, class_name, namespace):
    class_node = _class_node(path, class_name)
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


def test_routing_ast_and_dependency_boundaries_are_preserved():
    extractor_class = _class_node(LEARNED_ROUTER, "HiddenStateExtractor")
    extract = _method_node(extractor_class, "extract")
    extract_batch = _method_node(extractor_class, "_extract_batch")
    normalized_extract = _StripTorch().visit(copy.deepcopy(extract))
    ast.fix_missing_locations(normalized_extract)
    extract_digest = hashlib.sha256(
        ast.dump(normalized_extract, include_attributes=False).encode("utf-8")
    ).hexdigest()

    soft_tree = _parse(SOFT_ROUTER)
    soft_non_import = ast.Module(
        body=[
            node
            for node in soft_tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ],
        type_ignores=[],
    )
    soft_digest = hashlib.sha256(
        ast.dump(soft_non_import, include_attributes=False).encode("utf-8")
    ).hexdigest()

    assert extract_digest == EXTRACT_HASH
    assert "torch" not in _imported_bindings(extract)
    assert not _loads_name(extract, "torch")
    assert "torch" in _imported_bindings(extract_batch)
    assert _loads_name(extract_batch, "torch")
    assert soft_digest == SOFT_NON_IMPORT_HASH
    assert _top_level_import_signature(SOFT_ROUTER) == EXPECTED_SOFT_IMPORTS
    assert not _loads_name(soft_tree, "torch")


def test_hidden_state_extract_preserves_batching_and_normalization():
    class _Logger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    class _Model:
        def __init__(self):
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1

    logger = _Logger()
    model = _Model()
    extractor_class = _load_class(
        LEARNED_ROUTER,
        "HiddenStateExtractor",
        {
            "List": List,
            "Optional": Optional,
            "np": np,
            "logger": logger,
        },
    )
    extractor = extractor_class(model, tokenizer=object(), max_length=128)
    batches = []
    vectors = {
        "a": [3.0, 4.0],
        "b": [0.0, 0.0],
        "c": [1.0, 0.0],
    }

    def extract_batch(batch):
        batches.append(list(batch))
        return np.asarray([vectors[item] for item in batch], dtype=float)

    extractor._extract_batch = extract_batch
    features = extractor.extract(["a", "b", "c"], batch_size=2)

    assert model.eval_calls == 1
    assert batches == [["a", "b"], ["c"]]
    np.testing.assert_allclose(
        features,
        np.asarray([[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]]),
    )


def test_soft_router_preserves_weighted_adapter_merge_behavior():
    class _Logger:
        def __init__(self):
            self.info_messages = []
            self.error_messages = []

        def info(self, message):
            self.info_messages.append(message)

        def error(self, message):
            self.error_messages.append(message)

        def warning(self, _message):
            pass

    class _PeftModel:
        def __init__(self):
            self.add_calls = []
            self.set_calls = []

        def add_weighted_adapter(self, **kwargs):
            self.add_calls.append(kwargs)

        def set_adapter(self, adapter_name):
            self.set_calls.append(adapter_name)

    logger = _Logger()
    router_class = _load_class(
        SOFT_ROUTER,
        "SoftRouter",
        {
            "Dict": Dict,
            "Path": Path,
            "PeftModel": object,
            "_clean_adapter_config": lambda path: path,
            "logger": logger,
            "shutil": object(),
        },
    )
    router = router_class(
        base_model=object(),
        tokenizer=object(),
        adapter_paths={"text_expert": "text", "general_expert": "general"},
    )
    peft_model = _PeftModel()
    router.peft_model = peft_model
    router._adapters_loaded = True

    assert router.merge_adapters(
        {"text_expert": 0.7, "general_expert": 0.3},
        merged_name="weighted",
    ) is True
    assert peft_model.add_calls == [
        {
            "adapters": ["text_expert", "general_expert"],
            "weights": [0.7, 0.3],
            "adapter_name": "weighted",
            "combination_type": "linear",
        }
    ]
    assert peft_model.set_calls == ["weighted"]
    assert router._current_merged_name == "weighted"
    assert logger.error_messages == []
