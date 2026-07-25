"""Report whether the local environment can run Req2Inst workflows.

The diagnostic is read-only: it does not create directories, download
packages, load model weights, or start training/inference.
"""

import argparse
import ast
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Sequence

try:
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - minimal Python installations only
    Requirement = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_CHOICES = ("all", "inference", "training", "evaluation")
WORKFLOW_PROFILES = PROFILE_CHOICES[1:]

CORE_PACKAGES = {
    "accelerate",
    "bitsandbytes",
    "einops",
    "peft",
    "psutil",
    "safetensors",
    "sentencepiece",
    "tiktoken",
    "torch",
    "torchaudio",
    "torchvision",
    "tqdm",
    "transformers",
    "transformers-stream-generator",
}
PROFILE_PACKAGES = {
    "inference": CORE_PACKAGES | {"numpy", "pillow", "qwen-vl-utils"},
    "training": CORE_PACKAGES
    | {
        "chardet",
        "datasets",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    },
    "evaluation": CORE_PACKAGES
    | {
        "bert-score",
        "datasets",
        "evaluate",
        "matplotlib",
        "nltk",
        "numpy",
        "pandas",
        "rank-bm25",
        "rouge-score",
        "sacrebleu",
        "scikit-learn",
        "scipy",
        "seaborn",
    },
}
OPTIONAL_PACKAGES = {
    "selenium": "browser-assisted historical dataset construction",
}

ENTRYPOINTS = {
    "inference": (
        "config/settings.py",
        "models/language_model.py",
        "models/vision_model.py",
        "scripts/inference/generate_instructions.py",
        "scripts/inference/recognize_inputs.py",
        "src/instruction_generation/generator.py",
    ),
    "training": (
        "config/settings.py",
        "scripts/training/train_all_experts.py",
        "src/training/base_trainer.py",
        "src/training/data_loader.py",
    ),
    "evaluation": (
        "config/settings.py",
        "scripts/evaluation/experiments/run_all_experiments.py",
        "src/utils/enhanced_metrics.py",
    ),
}

MODEL_WEIGHTS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
)
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
STANDARD_EXPERTS = ("text", "image", "uml", "general")
DATASET_PATHS = (
    "data/dataset/image/image_dataset.csv",
    "data/dataset/uml/uml_dataset.csv",
    "data/dataset/text/CCHIT_dataset.csv",
    "data/dataset/text/CM1_dataset.csv",
    "data/dataset/text/GANNT_dataset.csv",
    "data/dataset/text/InfusionPump_dataset.csv",
    "data/dataset/text/Modis_dataset.csv",
    "data/dataset/text/WARC_dataset.csv",
)


@dataclass(frozen=True)
class CheckResult:
    """One diagnostic result."""

    section: str
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class AssetRule:
    """A required local file or directory contract."""

    section: str
    name: str
    relative_path: str
    profiles: tuple[str, ...]
    kind: str = "directory"
    required_groups: tuple[tuple[str, ...], ...] = ()


def _adapter_rules(
    method: str,
    experts: Sequence[str],
    profiles: tuple[str, ...],
) -> tuple[AssetRule, ...]:
    return tuple(
        AssetRule(
            "Checkpoints",
            f"checkpoints/{method}/{expert}_expert",
            f"checkpoints/{method}/{expert}_expert",
            profiles,
            required_groups=(("adapter_config.json",), ADAPTER_WEIGHTS),
        )
        for expert in experts
    )


ASSET_RULES = (
    AssetRule(
        "Models",
        "Qwen3-8B text model",
        "base_models/qwen3-8B/Qwen/Qwen3-8B",
        WORKFLOW_PROFILES,
        required_groups=(("config.json",), ("tokenizer_config.json",), MODEL_WEIGHTS),
    ),
    AssetRule(
        "Models",
        "Qwen3-VL-8B-Instruct vision model",
        "base_models/qwen3-VL-8B/qwen/Qwen3-VL-8B-Instruct",
        ("inference",),
        required_groups=(
            ("config.json",),
            ("tokenizer_config.json",),
            ("preprocessor_config.json", "processor_config.json"),
            MODEL_WEIGHTS,
        ),
    ),
    *_adapter_rules("lora_moe", STANDARD_EXPERTS, ("inference", "evaluation")),
    AssetRule(
        "Checkpoints",
        "checkpoints/lora_single/unified_expert",
        "checkpoints/lora_single/unified_expert",
        ("evaluation",),
        required_groups=(("adapter_config.json",), ADAPTER_WEIGHTS),
    ),
    *_adapter_rules("p_tuning", STANDARD_EXPERTS, ("evaluation",)),
    *_adapter_rules("prompt_tuning", STANDARD_EXPERTS, ("evaluation",)),
    *_adapter_rules("full_finetuning", STANDARD_EXPERTS, ("evaluation",)),
    *tuple(
        AssetRule(
            "Datasets",
            relative_path,
            relative_path,
            ("training", "evaluation"),
            kind="file",
        )
        for relative_path in DATASET_PATHS
    ),
)


RUNTIME_TARGETS = {
    "inference": ("logs", "outputs"),
    "training": ("checkpoints", "logs", "outputs"),
    "evaluation": ("checkpoints", "logs", "outputs"),
}


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _profiles(profile: str) -> tuple[str, ...]:
    return WORKFLOW_PROFILES if profile == "all" else (profile,)


def _selected_packages(profile: str) -> set[str]:
    selected = set()
    for workflow in _profiles(profile):
        selected.update(PROFILE_PACKAGES[workflow])
    return selected


def _load_requirements(path: Path) -> dict[str, object]:
    requirements = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        if Requirement is not None:
            parsed = Requirement(line)
            requirements[_canonical_name(parsed.name)] = parsed
        else:
            match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", line)
            if match:
                requirements[_canonical_name(match.group(1))] = match.groups()
    return requirements


def _requirement_name(requirement: object) -> str:
    return requirement.name if Requirement and isinstance(requirement, Requirement) else requirement[0]


def _requirement_label(requirement: object) -> str:
    if Requirement and isinstance(requirement, Requirement):
        return str(requirement)
    return "".join(requirement)


def _version_satisfies(requirement: object, installed: str) -> bool | None:
    if Requirement is None or not isinstance(requirement, Requirement):
        return None
    return not requirement.specifier or requirement.specifier.contains(
        installed,
        prereleases=True,
    )


def check_python() -> list[CheckResult]:
    version = sys.version_info[:2]
    version_status = "PASS" if version == (3, 10) else "FAIL" if version < (3, 10) else "WARN"
    isolated = (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or bool(os.environ.get("CONDA_PREFIX"))
        or bool(os.environ.get("VIRTUAL_ENV"))
    )
    return [
        CheckResult(
            "Python",
            "Interpreter version",
            version_status,
            f"{platform.python_version()} at {sys.executable}; documented baseline: 3.10",
        ),
        CheckResult(
            "Python",
            "Environment isolation",
            "PASS" if isolated else "WARN",
            f"sys.prefix={sys.prefix}",
        ),
        CheckResult(
            "Python",
            "Operating system",
            "INFO",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        ),
    ]


def _check_required_package(
    package: str,
    declared: dict[str, object],
) -> CheckResult:
    requirement = declared.get(_canonical_name(package))
    if requirement is None:
        return CheckResult(
            "Dependencies",
            package,
            "FAIL",
            "Referenced by the profile but absent from requirements.txt",
        )

    name = _requirement_name(requirement)
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        return CheckResult(
            "Dependencies",
            name,
            "FAIL",
            f"Missing; declared: {_requirement_label(requirement)}",
        )

    matches = _version_satisfies(requirement, installed)
    if matches is False:
        status = "FAIL"
        detail = f"Installed {installed}; expected {_requirement_label(requirement)}"
    elif matches is None:
        status = "WARN"
        detail = f"Installed {installed}; packaging unavailable, constraint not checked"
    else:
        status = "PASS"
        detail = f"Installed {installed}"
    return CheckResult("Dependencies", name, status, detail)


def _check_pip_consistency(root: Path) -> CheckResult:
    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("Dependency consistency", "pip check", "WARN", str(exc))

    output = (pip_check.stdout or pip_check.stderr).strip()
    return CheckResult(
        "Dependency consistency",
        "pip check",
        "PASS" if pip_check.returncode == 0 else "FAIL",
        output or f"Exited with code {pip_check.returncode}",
    )


def _check_optional_package(package: str, purpose: str) -> CheckResult:
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        status, detail = "WARN", f"Missing; only needed for {purpose}"
    else:
        status, detail = "PASS", f"Installed {installed}; used for {purpose}"
    return CheckResult("Optional dependencies", package, status, detail)


def check_dependencies(profile: str, root: Path) -> list[CheckResult]:
    requirements_path = root / "requirements.txt"
    if not requirements_path.is_file():
        return [
            CheckResult(
                "Dependencies",
                "requirements.txt",
                "FAIL",
                f"Missing: {requirements_path}",
            )
        ]

    declared = _load_requirements(requirements_path)
    results = [
        _check_required_package(package, declared)
        for package in sorted(_selected_packages(profile))
    ]
    results.append(_check_pip_consistency(root))

    if profile == "all":
        results.extend(
            _check_optional_package(package, purpose)
            for package, purpose in OPTIONAL_PACKAGES.items()
        )
    return results


def check_accelerator(profile: str) -> list[CheckResult]:
    required = profile in {"all", "inference", "training"}
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        return [
            CheckResult(
                "Accelerator",
                "PyTorch CUDA runtime",
                "FAIL" if required else "WARN",
                f"Unable to import torch: {exc}",
            )
        ]

    if not torch.cuda.is_available():
        return [
            CheckResult(
                "Accelerator",
                "CUDA availability",
                "FAIL" if required else "WARN",
                (
                    f"torch={torch.__version__}, compiled CUDA={torch.version.cuda}; "
                    "the documented full workflow requires an NVIDIA GPU"
                ),
            )
        ]

    name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    memory_pass_threshold_gb = 20.0
    return [
        CheckResult(
            "Accelerator",
            "CUDA availability",
            "PASS",
            f"{name}; torch={torch.__version__}; CUDA={torch.version.cuda}",
        ),
        CheckResult(
            "Accelerator",
            "GPU memory tier",
            "PASS" if memory_gb >= memory_pass_threshold_gb else "WARN",
            f"{memory_gb:.2f} GB detected; documented baseline: 24 GB-class",
        ),
    ]


def check_entrypoints(profile: str, root: Path) -> list[CheckResult]:
    relative_paths = sorted(
        {path for workflow in _profiles(profile) for path in ENTRYPOINTS[workflow]}
    )
    results = []
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            status, detail = "FAIL", "File is missing"
        else:
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                status, detail = "FAIL", str(exc)
            else:
                status, detail = "PASS", "File exists and parses"
        results.append(CheckResult("Repository", relative_path, status, detail))
    return results


def _matches_group(path: Path, patterns: Sequence[str]) -> bool:
    return any(next(path.glob(pattern), None) is not None for pattern in patterns)


def _check_asset(rule: AssetRule, root: Path) -> CheckResult:
    path = root / rule.relative_path
    if rule.kind == "file":
        if not path.is_file():
            return CheckResult(rule.section, rule.name, "FAIL", f"Missing: {path}")
        if path.stat().st_size == 0:
            return CheckResult(rule.section, rule.name, "FAIL", f"Empty: {path}")
        return CheckResult(
            rule.section,
            rule.name,
            "PASS",
            f"{path} ({path.stat().st_size} bytes)",
        )

    if not path.is_dir():
        return CheckResult(rule.section, rule.name, "FAIL", f"Missing: {path}")
    missing = [
        " or ".join(group)
        for group in rule.required_groups
        if not _matches_group(path, group)
    ]
    if missing:
        return CheckResult(
            rule.section,
            rule.name,
            "FAIL",
            f"Incomplete at {path}; missing: {', '.join(missing)}",
        )
    return CheckResult(rule.section, rule.name, "PASS", str(path))


def check_assets(profile: str, root: Path) -> list[CheckResult]:
    selected = set(_profiles(profile))
    return [
        _check_asset(rule, root)
        for rule in ASSET_RULES
        if selected.intersection(rule.profiles)
    ]


def _nearest_existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_runtime_targets(profile: str, root: Path) -> list[CheckResult]:
    targets = sorted(
        {
            target
            for workflow in _profiles(profile)
            for target in RUNTIME_TARGETS[workflow]
        }
    )
    results = []
    for relative_path in targets:
        path = root / relative_path
        probe = path if path.exists() else _nearest_existing_parent(path)
        writable = os.access(probe, os.W_OK)
        if path.exists():
            status = "PASS" if writable else "FAIL"
            detail = f"Existing {'writable' if writable else 'read-only'} path: {path}"
        else:
            status = "WARN" if writable else "FAIL"
            detail = f"Missing; nearest parent is {'writable' if writable else 'read-only'}: {probe}"
        results.append(CheckResult("Runtime targets", relative_path, status, detail))

    free_gb = shutil.disk_usage(root).free / 1024**3
    results.append(
        CheckResult(
            "Runtime targets",
            "Free disk space",
            "INFO",
            f"{free_gb:.2f} GB free on {root.anchor}",
        )
    )
    return results


def run_diagnostics(profile: str, root: Path = PROJECT_ROOT) -> list[CheckResult]:
    checks = []
    checks.extend(check_python())
    checks.extend(check_dependencies(profile, root))
    checks.extend(check_accelerator(profile))
    checks.extend(check_entrypoints(profile, root))
    checks.extend(check_assets(profile, root))
    checks.extend(check_runtime_targets(profile, root))
    return checks


def summarize(checks: Sequence[CheckResult]) -> dict[str, object]:
    counts = {
        status: sum(check.status == status for check in checks)
        for status in ("PASS", "WARN", "FAIL", "INFO")
    }
    return {"ready": counts["FAIL"] == 0, "counts": counts}


def render_text(profile: str, checks: Sequence[CheckResult]) -> None:
    print("Req2Inst environment diagnostics")
    print(f"Profile: {profile}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")

    section = None
    for check in checks:
        if check.section != section:
            section = check.section
            print(f"\n[{section}]")
        print(f"{check.status:>4}  {check.name}: {check.detail}")

    summary = summarize(checks)
    counts = summary["counts"]
    outcome = "READY" if summary["ready"] else "NOT READY"
    print(
        f"\nSummary: {outcome} ({counts['PASS']} pass, {counts['WARN']} warn, "
        f"{counts['FAIL']} fail, {counts['INFO']} info)"
    )
    print(
        "Static/read-only result only; it does not prove that a model-backed "
        "training or inference run succeeded."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Req2Inst readiness without loading model weights or "
            "modifying local artifacts"
        )
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="all",
        help="Workflow to check (default: all)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checks = run_diagnostics(args.profile)
    summary = summarize(checks)
    if args.json:
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "project_root": str(PROJECT_ROOT),
                    "python_executable": sys.executable,
                    **summary,
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
            )
        )
    else:
        render_text(args.profile, checks)
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
