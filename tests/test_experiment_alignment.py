"""Regression checks for manuscript-aligned experiment defaults."""

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unified_lora_uses_general_template_by_default_with_opt_in_override():
    path = ROOT / "scripts/training/lora_single/train_unified_expert.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    template_flag = None
    trainer_call = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--use_domain_templates"
            ):
                template_flag = node
        if isinstance(node.func, ast.Name) and node.func.id == "LoRATrainer":
            trainer_call = node

    assert template_flag is not None
    flag_keywords = {keyword.arg: keyword.value for keyword in template_flag.keywords}
    assert ast.literal_eval(flag_keywords["action"]) == "store_true"
    assert "default" not in flag_keywords

    assert trainer_call is not None
    trainer_keywords = {keyword.arg: keyword.value for keyword in trainer_call.keywords}
    template_value = trainer_keywords["use_domain_templates"]
    assert isinstance(template_value, ast.Attribute)
    assert isinstance(template_value.value, ast.Name)
    assert template_value.value.id == "args"
    assert template_value.attr == "use_domain_templates"


def test_exp10_keeps_the_last_complete_optional_implementation():
    path = ROOT / "scripts/evaluation/experiments/exp10_advanced_routing.py"
    source = path.read_text(encoding="utf-8")

    assert "v15: PoE + confidence-adaptive weighting" in source
    assert "adaptive_w1 = w1_t * conf1" in source
    assert "--debug-ensemble" in source


def test_exp10_docs_separate_manuscript_routing_from_later_variants():
    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "docs/experiments.md",
            ROOT / "docs/architecture.md",
        )
    )

    assert "Learned Router" in documentation
    assert "Output Ensemble" in documentation
    assert "top-2 logit fusion" in documentation
    assert "v13-v15" in documentation
    assert "repository-only" in documentation


def test_historical_local_experiment_artifacts_remain_untracked():
    historical_artifacts = {
        "evaluation_report.json",
        "expert_rouge_heatmap.pdf",
        "heatmap.pdf",
        "scripts/utils/chart.py",
        "scripts/utils/combined_metrics_plot.py",
        "scripts/utils/comparison_plot.py",
        "scripts/utils/expert_selection_plot.py",
        "scripts/utils/heatmap.py",
        "src/utils/eval_data.json",
        "src/utils/eval_data_chatgpt.json",
        "src/utils/eval_data_gemini.json",
        "src/utils/quick_eval.py",
    }
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tracked = {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }

    assert historical_artifacts.isdisjoint(tracked)
