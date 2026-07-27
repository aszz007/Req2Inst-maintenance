"""Lightweight compatibility checks for the current Qwen3-only baseline."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/README.md",
    ROOT / "docs/architecture.md",
    ROOT / "CONTRIBUTING.md",
)


class ModelCompatibilityTests(unittest.TestCase):
    def test_current_docs_describe_the_qwen3_only_baseline(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in BASELINE_DOCS
        )
        self.assertNotIn("Qwen-7B", combined)
        self.assertIn("Qwen3-8B", combined)
        self.assertIn("Qwen3-VL-8B-Instruct", combined)

    def test_config_exposes_only_supported_vision_version(self):
        sys.path.insert(0, str(ROOT))
        from config.settings import get_path_config, get_vision_model_config

        self.assertEqual(get_vision_model_config().SUPPORTED_VERSIONS, ["qwen3"])
        path_cfg = get_path_config()
        self.assertEqual(list(path_cfg.VISION_MODEL_PATHS), ["qwen3"])
        with self.assertRaisesRegex(ValueError, "Unsupported vision model version"):
            path_cfg.get_vision_model_path("qwen2.5")

    def test_cli_version_choices_are_qwen3_only(self):
        cli_paths = (
            ROOT / "scripts/inference/recognize_inputs.py",
            ROOT / "scripts/inference/generate_instructions.py",
        )

        for path in cli_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            version_choices = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                if node.args[0].value not in {"--version", "--vision-version"}:
                    continue
                choices = next(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "choices"
                )
                version_choices.append(ast.literal_eval(choices))
            self.assertEqual(version_choices, [["qwen3"]], str(path))

    def test_obsolete_multi_environment_wrapper_is_removed(self):
        self.assertFalse((ROOT / "scripts/run_with_env.py").exists())

    def test_vision_wrapper_uses_requested_version_for_path_resolution(self):
        path = ROOT / "models/vision_model.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn(
            "self.version = version or get_vision_model_config().version",
            source,
        )
        self.assertIn(
            "configured_model_path = path_cfg.get_vision_model_path(self.version)",
            source,
        )

    def test_expert_registry_uses_current_text_model_version(self):
        module_path = ROOT / "src/routing/expert_router.py"
        spec = importlib.util.spec_from_file_location("req2inst_expert_router", module_path)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        versions = [
            expert.model_version
            for expert in module.ExpertRouter().list_experts()
        ]
        self.assertEqual(versions, ["qwen3_8b"] * 4)

    def test_lora_moe_config_helpers_use_current_base_model_path(self):
        script_paths = (
            ROOT / "scripts/training/lora_moe/train_general_expert.py",
            ROOT / "scripts/training/lora_moe/train_image_expert.py",
            ROOT / "scripts/training/lora_moe/train_text_expert.py",
            ROOT / "scripts/training/lora_moe/train_uml_expert.py",
        )
        for path in script_paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("QWEN_7B_CHAT_PATH", source, str(path))
            self.assertIn("QWEN3_8B_PATH", source, str(path))
if __name__ == "__main__":
    unittest.main()
