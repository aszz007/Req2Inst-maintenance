"""Report project code size through the external cloc tool."""

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOTS = ("config", "models", "src", "scripts")
TEST_ROOTS = ("tests",)
SUMMARY_FIELDS = ("nFiles", "blank", "comment", "code")


def _resolve_cloc(command: str | None = None) -> str:
    """Resolve an existing cloc executable without downloading or installing it."""
    candidates = []
    if command:
        requested = Path(command)
        candidates.extend(
            (
                requested,
                PROJECT_ROOT / requested,
            )
        )
        resolved = shutil.which(command)
        if resolved:
            return resolved
    else:
        candidates.extend(
            (
                PROJECT_ROOT / "cloc.exe",
                PROJECT_ROOT / "cloc",
            )
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    resolved = shutil.which("cloc")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "cloc was not found. Install cloc separately or pass "
        "--cloc-command PATH; the repository does not vendor the executable."
    )


def _run_cloc(cloc_command: str, roots: Sequence[str]) -> dict[str, int]:
    """Run cloc for one scope and return its aggregate SUM row."""
    missing = [root for root in roots if not (PROJECT_ROOT / root).exists()]
    if missing:
        raise FileNotFoundError(
            f"Configured code scope is missing from the repository: {', '.join(missing)}"
        )

    completed = subprocess.run(
        [cloc_command, "--json", "--quiet", *roots],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cloc failed with exit code {completed.returncode}: {detail}")

    payload = json.loads(completed.stdout)
    summary = payload.get("SUM", {})
    return {
        field: int(summary.get(field, 0))
        for field in SUMMARY_FIELDS
    }


def _add(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Add two cloc summary rows."""
    return {field: left[field] + right[field] for field in SUMMARY_FIELDS}


def _empty_summary() -> dict[str, int]:
    """Return a zero row for an optional code scope that is not present."""
    return {field: 0 for field in SUMMARY_FIELDS}


def collect_stats(cloc_command: str | None = None) -> dict[str, object]:
    """Collect main-code, optional local-test, and combined totals."""
    cloc = _resolve_cloc(cloc_command)
    main = _run_cloc(cloc, MAIN_ROOTS)
    tests = (
        _run_cloc(cloc, TEST_ROOTS)
        if all((PROJECT_ROOT / root).exists() for root in TEST_ROOTS)
        else _empty_summary()
    )
    return {
        "tool": "cloc",
        "project_root": str(PROJECT_ROOT),
        "scopes": {
            "main": {"roots": list(MAIN_ROOTS), "summary": main},
            "tests": {"roots": list(TEST_ROOTS), "summary": tests},
        },
        "total": _add(main, tests),
    }


def _render_text(report: dict[str, object]) -> None:
    """Render a compact human-readable report."""
    print("Req2Inst code statistics (cloc)")
    print(f"Project: {report['project_root']}")
    print("Scope        Files    Blank  Comment     Code")
    print("-----------  -------  -------  -------  -------")
    for label in ("main", "tests"):
        summary = report["scopes"][label]["summary"]
        print(
            f"{label:<11}  {summary['nFiles']:>7}  {summary['blank']:>7}  "
            f"{summary['comment']:>7}  {summary['code']:>7}"
        )
    total = report["total"]
    print("-----------  -------  -------  -------  -------")
    print(
        f"{'total':<11}  {total['nFiles']:>7}  {total['blank']:>7}  "
        f"{total['comment']:>7}  {total['code']:>7}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Count Req2Inst main and optional local test code through cloc "
            "without modifying project files"
        )
    )
    parser.add_argument(
        "--cloc-command",
        default=None,
        help="Optional cloc executable name or path (auto-detected by default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the code-statistics command."""
    args = parse_args(argv)
    try:
        report = collect_stats(args.cloc_command)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _render_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
