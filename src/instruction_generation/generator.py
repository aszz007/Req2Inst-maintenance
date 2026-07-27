"""Expose the unified instruction-generation interface."""

import json
from datetime import datetime
from pathlib import Path

from config.settings import get_path_config
from src.routing.moe_model import MoEModel
from src.utils.logger import get_logger

logger = get_logger('instruction_generation.generator')


class InstructionGenerator:
    """Generate instructions with the configured model interface."""

    def __init__(
            self,
            lora_weights_dir: str | None = None,
            base_models_dir: str | None = None
    ):
        """Initialize the instance."""
        path_cfg = get_path_config()

        if lora_weights_dir is None:
            # Use checkpoints/lora_moe/ as the standard weights directory per framework
            lora_weights_dir = str(path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe')
        if base_models_dir is None:
            base_models_dir = str(path_cfg.BASE_MODELS_DIR)

        self.moe_model = MoEModel(
            lora_weights_dir=lora_weights_dir,
            base_models_dir=base_models_dir
        )

        logger.info("Instruction generator initialized")
        logger.info(f"LoRA weights directory: {lora_weights_dir}")
        logger.info(f"Base models directory: {base_models_dir}")

    def generate(
            self,
            input_data: str | dict,
            output_format: str = 'text',
            expert_variant: str | None = None,
            **generation_kwargs
    ) -> str | dict:
        """Generate output."""
        logger.info("Starting instruction generation")

        result = self.moe_model.generate_instruction(
            input_data=input_data,
            expert_variant=expert_variant,
            **generation_kwargs
        )

        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if output_format == 'json':
            logger.info("Returning JSON output")
            return result
        elif output_format == 'markdown':
            logger.info("Returning Markdown output")
            return self._format_markdown(result)
        else:  # text
            logger.info("Returning text output")
            return result['instruction']

    def batch_generate(
            self,
            input_list: list[str | dict],
            output_format: str = 'text',
            expert_variant: str | None = None,
            save_path: str | None = None,
            **generation_kwargs
    ) -> list[str | dict]:
        """Generate outputs in batches."""
        logger.info(f"Batch instruction generation - {len(input_list)} samples")

        results = []

        for i, input_data in enumerate(input_list, 1):
            logger.info(f"\nProcessing sample {i}/{len(input_list)}")

            try:
                result = self.generate(
                    input_data=input_data,
                    output_format=output_format,
                    expert_variant=expert_variant,
                    **generation_kwargs
                )
                results.append(result)
                logger.info(f"Sample {i} generated successfully")

            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                if output_format == 'json':
                    results.append({
                        'instruction': '',
                        'expert_used': 'none',
                        'error': str(e),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    results.append('')

        if save_path:
            self._save_results(results, save_path, output_format)

        logger.info(f"Batch generation complete - successful: {len([r for r in results if r])}/{len(input_list)}")

        return results

    def generate_from_file(
            self,
            input_file: str,
            output_file: str | None = None,
            output_format: str = 'json',
            **generation_kwargs
    ) -> list[dict]:
        """Generate from file."""
        logger.info(f"Generating instructions from file: {input_file}")

        input_list = self._load_input_file(input_file)
        logger.info(f"Loaded {len(input_list)} inputs")

        results = self.batch_generate(
            input_list=input_list,
            output_format=output_format,
            save_path=output_file,
            **generation_kwargs
        )

        return results

    def _load_input_file(self, file_path: str) -> list[str | dict]:
        """Load input file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {file_path}")

        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines

        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _save_results(
            self,
            results: list[str | dict],
            save_path: str,
            output_format: str
    ):
        """Save results."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        elif output_format == 'markdown':
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("# Generated Instructions\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"## Instruction {i}\n\n")
                    f.write(result + "\n\n")
                    f.write("---\n\n")

        else:  # text
            with open(save_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"=== Instruction {i} ===\n")
                    f.write(result + "\n\n")

        logger.info(f"Results saved to: {save_path}")

    def _format_markdown(self, result: dict) -> str:
        """Format markdown."""
        md = "# Generated Instruction\n\n"

        md += "## Instruction\n\n"
        md += result['instruction'] + "\n\n"

        md += "## Metadata\n\n"
        md += f"- **Expert Used**: {result['expert_used']}\n"
        md += f"- **Expert Type**: {result['expert_type']}\n"
        md += f"- **Timestamp**: {result.get('timestamp', 'N/A')}\n"
        md += f"- **Reasoning**: {result['reasoning']}\n"

        return md

    def get_statistics(self) -> dict:
        """Return statistics."""
        return self.moe_model.get_router_statistics()

    def reset_statistics(self):
        """Reset statistics."""
        self.moe_model.reset_router_statistics()
        logger.info("Statistics reset")

    def list_available_experts(self, expert_type: str | None = None) -> list[str]:
        """List available experts."""
        return self.moe_model.list_available_experts(expert_type)
