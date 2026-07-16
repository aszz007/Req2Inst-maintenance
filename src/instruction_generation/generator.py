"""Expose the unified instruction-generation interface."""

import json
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from src.routing.moe_model import MoEModel
from src.utils.logger import get_logger
from config.settings import get_path_config

logger = get_logger('instruction_generation.generator')


class InstructionGenerator:
    """Generate instructions with the configured model interface."""

    def __init__(
            self,
            lora_weights_dir: Optional[str] = None,
            base_models_dir: Optional[str] = None
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

        logger.info("指令生成器初始化完成")
        logger.info(f"LoRA权重目录: {lora_weights_dir}")
        logger.info(f"基础模型目录: {base_models_dir}")

    def generate(
            self,
            input_data: Union[str, dict],
            output_format: str = 'text',
            expert_variant: Optional[str] = None,
            **generation_kwargs
    ) -> Union[str, dict]:
        """Generate output."""
        logger.info("=" * 80)
        logger.info("开始生成指令")
        logger.info("=" * 80)

        result = self.moe_model.generate_instruction(
            input_data=input_data,
            expert_variant=expert_variant,
            **generation_kwargs
        )

        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if output_format == 'json':
            logger.info("返回JSON格式")
            return result
        elif output_format == 'markdown':
            logger.info("返回Markdown格式")
            return self._format_markdown(result)
        else:  # text
            logger.info("返回文本格式")
            return result['instruction']

    def batch_generate(
            self,
            input_list: List[Union[str, dict]],
            output_format: str = 'text',
            expert_variant: Optional[str] = None,
            save_path: Optional[str] = None,
            **generation_kwargs
    ) -> List[Union[str, dict]]:
        """Generate outputs in batches."""
        logger.info("=" * 80)
        logger.info(f"批量生成指令 - 共{len(input_list)}个样本")
        logger.info("=" * 80)

        results = []

        for i, input_data in enumerate(input_list, 1):
            logger.info(f"\n处理样本 {i}/{len(input_list)}")

            try:
                result = self.generate(
                    input_data=input_data,
                    output_format=output_format,
                    expert_variant=expert_variant,
                    **generation_kwargs
                )
                results.append(result)
                logger.info(f"样本 {i} 生成成功")

            except Exception as e:
                logger.error(f"样本 {i} 生成失败: {e}")
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

        logger.info("=" * 80)
        logger.info(f"批量生成完成 - 成功: {len([r for r in results if r])}/{len(input_list)}")
        logger.info("=" * 80)

        return results

    def generate_from_file(
            self,
            input_file: str,
            output_file: Optional[str] = None,
            output_format: str = 'json',
            **generation_kwargs
    ) -> List[dict]:
        """Generate from file."""
        logger.info(f"从文件生成指令: {input_file}")

        input_list = self._load_input_file(input_file)
        logger.info(f"加载了 {len(input_list)} 个输入")

        results = self.batch_generate(
            input_list=input_list,
            output_format=output_format,
            save_path=output_file,
            **generation_kwargs
        )

        return results

    def _load_input_file(self, file_path: str) -> List[Union[str, dict]]:
        """Load input file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {file_path}")

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
                raise ValueError(f"不支持的JSON格式: {type(data)}")

        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _save_results(
            self,
            results: List[Union[str, dict]],
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

        logger.info(f"结果已保存至: {save_path}")

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

    def get_statistics(self) -> Dict:
        """Return statistics."""
        return self.moe_model.get_router_statistics()

    def reset_statistics(self):
        """Reset statistics."""
        self.moe_model.reset_router_statistics()
        logger.info("统计信息已重置")

    def list_available_experts(self, expert_type: Optional[str] = None) -> List[str]:
        """List available experts."""
        return self.moe_model.list_available_experts(expert_type)
