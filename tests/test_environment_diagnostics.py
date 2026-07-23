"""Regression coverage for the read-only environment diagnostic."""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = ROOT / "scripts" / "diagnostics" / "check_environment.py"


def _load_diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "req2inst_environment_diagnostic",
        DIAGNOSTIC_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = _load_diagnostic_module()


def test_runtime_requirements_are_classified_by_a_profile():
    requirements = DIAGNOSTIC._load_requirements(ROOT / "requirements.txt")
    classified = set(DIAGNOSTIC.OPTIONAL_PACKAGES)
    for packages in DIAGNOSTIC.PROFILE_PACKAGES.values():
        classified.update(packages)

    assert set(requirements) == classified


def test_exact_pins_accept_local_cuda_build_versions():
    if DIAGNOSTIC.Requirement is None:
        pytest.skip("packaging is unavailable")

    requirement = DIAGNOSTIC.Requirement("torch==2.7.1")

    assert DIAGNOSTIC._version_satisfies(requirement, "2.7.1+cu128") is True
    assert DIAGNOSTIC._version_satisfies(requirement, "2.13.0+cpu") is False


def test_missing_requirements_short_circuits_dependency_checks(
    tmp_path,
    monkeypatch,
):
    def unexpected_pip_check(*_args, **_kwargs):
        pytest.fail("pip check must not run without requirements.txt")

    monkeypatch.setattr(DIAGNOSTIC.subprocess, "run", unexpected_pip_check)

    assert DIAGNOSTIC.check_dependencies("training", tmp_path) == [
        DIAGNOSTIC.CheckResult(
            "Dependencies",
            "requirements.txt",
            "FAIL",
            f"Missing: {tmp_path / 'requirements.txt'}",
        )
    ]


def test_dependency_check_preserves_package_and_pip_outcomes(
    tmp_path,
    monkeypatch,
):
    if DIAGNOSTIC.Requirement is None:
        pytest.skip("packaging is unavailable")

    (tmp_path / "requirements.txt").write_text("fixture\n", encoding="utf-8")
    requirements = {
        "demo-mismatch": DIAGNOSTIC.Requirement("demo-mismatch>=2"),
        "demo-missing": DIAGNOSTIC.Requirement("demo-missing"),
        "demo-present": DIAGNOSTIC.Requirement("demo-present>=1"),
    }
    monkeypatch.setattr(
        DIAGNOSTIC,
        "_selected_packages",
        lambda _profile: {
            "demo-mismatch",
            "demo-missing",
            "demo-present",
            "undeclared",
        },
    )
    monkeypatch.setattr(
        DIAGNOSTIC,
        "_load_requirements",
        lambda _path: requirements,
    )

    def fake_version(name):
        if name == "demo-missing":
            raise DIAGNOSTIC.metadata.PackageNotFoundError
        return {
            "demo-mismatch": "1.0",
            "demo-present": "1.5",
        }[name]

    monkeypatch.setattr(DIAGNOSTIC.metadata, "version", fake_version)
    monkeypatch.setattr(
        DIAGNOSTIC.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="No broken requirements found.\n",
            stderr="",
        ),
    )

    assert DIAGNOSTIC.check_dependencies("training", tmp_path) == [
        DIAGNOSTIC.CheckResult(
            "Dependencies",
            "demo-mismatch",
            "FAIL",
            "Installed 1.0; expected demo-mismatch>=2",
        ),
        DIAGNOSTIC.CheckResult(
            "Dependencies",
            "demo-missing",
            "FAIL",
            "Missing; declared: demo-missing",
        ),
        DIAGNOSTIC.CheckResult(
            "Dependencies",
            "demo-present",
            "PASS",
            "Installed 1.5",
        ),
        DIAGNOSTIC.CheckResult(
            "Dependencies",
            "undeclared",
            "FAIL",
            "Referenced by the profile but absent from requirements.txt",
        ),
        DIAGNOSTIC.CheckResult(
            "Dependency consistency",
            "pip check",
            "PASS",
            "No broken requirements found.",
        ),
    ]


def test_dependency_check_preserves_pip_failure_fallback(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "requirements.txt").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(DIAGNOSTIC, "_selected_packages", lambda _profile: set())
    monkeypatch.setattr(DIAGNOSTIC, "_load_requirements", lambda _path: {})

    def fail_pip_check(*_args, **_kwargs):
        raise OSError("pip unavailable")

    monkeypatch.setattr(DIAGNOSTIC.subprocess, "run", fail_pip_check)

    assert DIAGNOSTIC.check_dependencies("training", tmp_path) == [
        DIAGNOSTIC.CheckResult(
            "Dependency consistency",
            "pip check",
            "WARN",
            "pip unavailable",
        )
    ]


def test_all_profile_preserves_optional_dependency_results(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "requirements.txt").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(DIAGNOSTIC, "_selected_packages", lambda _profile: set())
    monkeypatch.setattr(DIAGNOSTIC, "_load_requirements", lambda _path: {})
    monkeypatch.setattr(
        DIAGNOSTIC,
        "OPTIONAL_PACKAGES",
        {
            "optional-present": "present fixture",
            "optional-missing": "missing fixture",
        },
    )
    monkeypatch.setattr(
        DIAGNOSTIC.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    def fake_version(name):
        if name == "optional-missing":
            raise DIAGNOSTIC.metadata.PackageNotFoundError
        return "4.0"

    monkeypatch.setattr(DIAGNOSTIC.metadata, "version", fake_version)

    assert DIAGNOSTIC.check_dependencies("all", tmp_path) == [
        DIAGNOSTIC.CheckResult(
            "Dependency consistency",
            "pip check",
            "PASS",
            "Exited with code 0",
        ),
        DIAGNOSTIC.CheckResult(
            "Optional dependencies",
            "optional-present",
            "PASS",
            "Installed 4.0; used for present fixture",
        ),
        DIAGNOSTIC.CheckResult(
            "Optional dependencies",
            "optional-missing",
            "WARN",
            "Missing; only needed for missing fixture",
        ),
    ]


def _directory_rule(name, relative_path, required_groups):
    return DIAGNOSTIC.AssetRule(
        "Test assets",
        name,
        relative_path,
        ("inference",),
        required_groups=required_groups,
    )


def test_model_directory_requires_config_tokenizer_and_weights(tmp_path):
    rule = _directory_rule(
        "test model",
        "model",
        (("config.json",), ("tokenizer_config.json",), DIAGNOSTIC.MODEL_WEIGHTS),
    )
    model_path = tmp_path / "model"

    assert DIAGNOSTIC._check_asset(rule, tmp_path).status == "FAIL"

    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_path / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    assert DIAGNOSTIC._check_asset(rule, tmp_path).status == "PASS"


def test_vision_model_directory_requires_processor_metadata(tmp_path):
    rule = _directory_rule(
        "vision model",
        "vision",
        (
            ("config.json",),
            ("tokenizer_config.json",),
            ("preprocessor_config.json", "processor_config.json"),
            DIAGNOSTIC.MODEL_WEIGHTS,
        ),
    )
    model_path = tmp_path / "vision"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text(
        "{}",
        encoding="utf-8",
    )

    assert DIAGNOSTIC._check_asset(rule, tmp_path).status == "FAIL"

    (model_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    assert DIAGNOSTIC._check_asset(rule, tmp_path).status == "PASS"


def test_adapter_directory_requires_config_and_weights(tmp_path):
    rule = _directory_rule(
        "test adapter",
        "adapter",
        (("adapter_config.json",), DIAGNOSTIC.ADAPTER_WEIGHTS),
    )
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert DIAGNOSTIC._check_asset(rule, tmp_path).status == "FAIL"

    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert DIAGNOSTIC._check_asset(rule, tmp_path).status == "PASS"


def test_diagnostic_source_contains_no_write_or_model_load_operations():
    tree = ast.parse(
        DIAGNOSTIC_PATH.read_text(encoding="utf-8"),
        filename=str(DIAGNOSTIC_PATH),
    )
    banned_attributes = {
        "download",
        "from_pretrained",
        "load_state_dict",
        "mkdir",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "unlink",
        "write_bytes",
        "write_text",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in banned_attributes
    }

    assert calls == set()


def test_fragmented_config_validator_is_removed():
    settings = (ROOT / "config" / "settings.py").read_text(encoding="utf-8")
    package_init = (ROOT / "config" / "__init__.py").read_text(encoding="utf-8")

    assert "def validate_config(" not in settings
    assert "validate_config" not in package_init


def test_unused_reserved_memory_query_is_removed_from_base_trainer():
    source = (ROOT / "src" / "training" / "base_trainer.py").read_text(
        encoding="utf-8"
    )

    assert "torch.cuda.memory_reserved()" not in source


def test_summary_fails_only_when_a_check_fails():
    healthy = [
        DIAGNOSTIC.CheckResult("test", "pass", "PASS", "ok"),
        DIAGNOSTIC.CheckResult("test", "warn", "WARN", "review"),
        DIAGNOSTIC.CheckResult("test", "info", "INFO", "context"),
    ]
    failed = [
        *healthy,
        DIAGNOSTIC.CheckResult("test", "fail", "FAIL", "missing"),
    ]

    assert DIAGNOSTIC.summarize(healthy)["ready"] is True
    assert DIAGNOSTIC.summarize(failed)["ready"] is False


def test_cli_defaults_to_the_full_workflow_profile():
    args = DIAGNOSTIC.parse_args([])

    assert args.profile == "all"
    assert args.json is False


def test_json_cli_reports_readiness_without_loading_models():
    result = subprocess.run(
        [
            sys.executable,
            str(DIAGNOSTIC_PATH),
            "--profile",
            "inference",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode in {0, 1}
    payload = json.loads(result.stdout)
    assert payload["profile"] == "inference"
    assert payload["ready"] is (result.returncode == 0)
    assert payload["checks"]
    assert {check["status"] for check in payload["checks"]} <= {
        "PASS",
        "WARN",
        "FAIL",
        "INFO",
    }
    assert "Traceback" not in result.stderr
