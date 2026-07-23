"""Tests for the cloc-based code statistics wrapper."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnostics/code_stats.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("req2inst_code_stats", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_cloc_uses_json_quiet_and_parses_sum(monkeypatch):
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"SUM": {"nFiles": 2, "blank": 3, "comment": 4, "code": 5}}
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    summary = module._run_cloc("cloc.exe", ("config",))

    assert summary == {"nFiles": 2, "blank": 3, "comment": 4, "code": 5}
    assert calls[0][0] == ["cloc.exe", "--json", "--quiet", "config"]
    assert calls[0][1]["cwd"] == module.PROJECT_ROOT


def test_collect_stats_keeps_main_and_test_scopes_separate(monkeypatch):
    module = _load_module()
    summaries = {
        tuple(module.MAIN_ROOTS): {"nFiles": 3, "blank": 4, "comment": 5, "code": 6},
        tuple(module.TEST_ROOTS): {"nFiles": 1, "blank": 2, "comment": 3, "code": 4},
    }

    monkeypatch.setattr(module, "_resolve_cloc", lambda _command: "cloc")
    monkeypatch.setattr(
        module,
        "_run_cloc",
        lambda _cloc, roots: summaries[tuple(roots)],
    )

    report = module.collect_stats()

    assert report["scopes"]["main"]["summary"] == summaries[tuple(module.MAIN_ROOTS)]
    assert report["scopes"]["tests"]["summary"] == summaries[tuple(module.TEST_ROOTS)]
    assert report["total"] == {"nFiles": 4, "blank": 6, "comment": 8, "code": 10}


def test_missing_cloc_has_an_actionable_error(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda _command: None)

    with pytest.raises(FileNotFoundError, match="Install cloc separately"):
        module._resolve_cloc()
