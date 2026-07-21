"""Regression contracts for full-finetuning entry-point imports."""

import ast
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FULL_FINETUNING_DIR = ROOT / "scripts" / "training" / "full_finetuning"
ENTRYPOINT_BODY_HASHES = {
    FULL_FINETUNING_DIR / "train_text_expert.py": (
        "6b9efc3411333f6f997f42a6dae70bf5971466b559d54a0738ce8ad76adeb976"
    ),
    FULL_FINETUNING_DIR / "train_image_expert.py": (
        "b4605590a4e4d51105bd125381188dcf522717881a92865e734238af95f613e9"
    ),
    FULL_FINETUNING_DIR / "train_uml_expert.py": (
        "386ade5bbab71f00debb8f0f455b9f9cdb95577f7f7799e3bd309af51f871201"
    ),
    FULL_FINETUNING_DIR / "train_general_expert.py": (
        "6feda915359d7453cdd38eb11cfa0f9dfeb0e2db3ac22c79f5320850f26879a3"
    ),
}
EXPECTED_ENTRYPOINT_IMPORTS = [
    ("import", None, ("sys",)),
    ("import", None, ("argparse",)),
    ("from", "pathlib", ("Path",)),
    ("from", "config.settings", ("get_path_config",)),
    (
        "from",
        "src.training.full_finetuning_trainer",
        ("FullFineTuningTrainer",),
    ),
    ("from", "src.utils.logger", ("get_logger",)),
]
DOWNSTREAM_TORCH_MODULES = [
    ROOT / "config" / "settings.py",
    ROOT / "src" / "training" / "full_finetuning_trainer.py",
    ROOT / "src" / "training" / "base_trainer.py",
]


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _non_import_body_hash(path):
    tree = _parse(path)
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ],
        type_ignores=[],
    )
    return hashlib.sha256(
        ast.dump(module, include_attributes=False).encode("utf-8")
    ).hexdigest()


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


@pytest.mark.parametrize("path,expected_hash", ENTRYPOINT_BODY_HASHES.items())
def test_entrypoint_non_import_body_matches_pre_cleanup_contract(
    path,
    expected_hash,
):
    assert _non_import_body_hash(path) == expected_hash


@pytest.mark.parametrize("path", ENTRYPOINT_BODY_HASHES)
def test_entrypoint_keeps_all_required_imports_except_torch(path):
    assert _top_level_import_signature(path) == EXPECTED_ENTRYPOINT_IMPORTS
    assert not _loads_name(path, "torch")


@pytest.mark.parametrize("path", DOWNSTREAM_TORCH_MODULES)
def test_downstream_runtime_modules_keep_their_torch_dependency(path):
    assert ("import", None, ("torch",)) in _top_level_import_signature(path)
    assert _loads_name(path, "torch")
