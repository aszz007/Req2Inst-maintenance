"""Repository-level checks that do not import the runtime model stack."""

import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"(?<!!)(?:\[[^\]]+\])\(([^)\s]+)")


def _tracked_paths(pattern: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", pattern],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def test_all_tracked_python_files_parse():
    failures = []
    for path in _tracked_paths("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")

    assert not failures, "\n".join(failures)


def test_relative_markdown_links_exist():
    missing = []
    for path in _tracked_paths("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK_RE.findall(text):
            target = target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue

            relative_target = parsed.path
            if not relative_target:
                continue
            resolved = (path.parent / relative_target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

    assert not missing, "Missing relative Markdown links:\n" + "\n".join(missing)


def test_plantuml_runtime_is_local_only():
    assert not _tracked_paths("scripts/plantuml.jar")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    reproducibility = (ROOT / "docs/reproducibility.md").read_text(
        encoding="utf-8"
    )
    artifact_policy = (ROOT / "docs/data-and-artifacts.md").read_text(
        encoding="utf-8"
    )

    assert "/scripts/plantuml.jar" in gitignore
    assert "PlantUML 1.2024.3" in reproducibility
    assert "519A4A7284C6A0357C369E4BB0CAF72C4BFBBDE851B8C6D6BBDB7AF3C01FC82F" in (
        reproducibility
    )
    assert "scripts/plantuml.jar" in artifact_policy
