"""Regression checks for manuscript-facing terminology and legacy keys."""

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/README.md",
    ROOT / "docs/architecture.md",
    ROOT / "docs/experiments.md",
    ROOT / "docs/reproducibility.md",
    ROOT / "inputs/README.md",
)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name!r} not found in {path}")


class ManuscriptTerminologyTests(unittest.TestCase):
    def test_public_docs_use_manuscript_facing_names(self):
        combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
        self.assertIsNone(re.search(r"\bUML\b", combined))
        self.assertNotIn("LoRA-MoE", combined)
        self.assertNotIn("LoRA-Single", combined)
        self.assertIn("FlowChart", combined)
        self.assertIn("Multi-Expert LoRA", combined)
        self.assertIn("LoRA (Unified)", combined)

    def test_legacy_internal_keys_remain_unchanged(self):
        exp2 = ROOT / "scripts/evaluation/experiments/exp2_compare_finetuning_methods.py"
        self.assertEqual(
            _assignment(exp2, "METHODS"),
            ["lora_moe", "lora_single", "p_tuning", "prompt_tuning", "full_finetuning"],
        )
        self.assertEqual(
            _assignment(exp2, "EXPERT_TYPES"),
            ["text", "image", "uml", "general"],
        )

        recognition_source = (
            ROOT / "scripts/inference/recognize_inputs.py"
        ).read_text(encoding="utf-8")
        self.assertIn("choices=['image', 'uml']", recognition_source)

    def test_paper_facing_display_maps_do_not_replace_result_keys(self):
        exp2 = ROOT / "scripts/evaluation/experiments/exp2_compare_finetuning_methods.py"
        self.assertEqual(_assignment(exp2, "METHOD_DISPLAY_NAMES")["lora_single"], "LoRA (Unified)")
        self.assertEqual(_assignment(exp2, "EXPERT_DISPLAY_NAMES")["uml"], "FlowChart")

        exp3 = ROOT / "scripts/evaluation/experiments/exp3_moe_architecture_validation.py"
        self.assertEqual(
            _assignment(exp3, "ARCH_DISPLAY_NAMES"),
            {
                "MoE-4": "Multi-Expert-4",
                "MoE-3": "Multi-Expert-3",
                "Single": "LoRA (Unified)",
            },
        )
        exp3_source = exp3.read_text(encoding="utf-8")
        self.assertIn("'MoE-4': {'rougeL'", exp3_source)
        self.assertIn("'MoE-3': {", exp3_source)

    def test_behavior_sensitive_prompts_keep_legacy_uml_wording(self):
        protected_paths = (
            ROOT / "models/prompt_templates/general_template.py",
            ROOT / "models/prompt_templates/uml_template.py",
            ROOT / "scripts/preprocessing/build_final_dataset/uml/generate_instructions_uml.py",
            ROOT / "scripts/preprocessing/build_final_dataset/uml/regenerate_failed_uml.py",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in protected_paths)
        self.assertIn("UML Use Case Diagram", combined)
        self.assertIn("[UML Diagram:", combined)
        self.assertIn("Definition:", combined)
        self.assertIn("Emphasis & Caution:", combined)
        self.assertIn("Things to Avoid:", combined)


if __name__ == "__main__":
    unittest.main()