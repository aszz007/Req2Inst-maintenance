"""Implement the text-domain expert."""

import json
from pathlib import Path
from typing import Optional

from src.experts.base_expert import BaseExpert
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.text')



def _build_prompt_for_domain(input_data):
    """Build prompt for domain."""
    if isinstance(input_data, dict):
        data = input_data
        text_fallback = str(input_data)
    elif isinstance(input_data, str):
        try:
            data = json.loads(input_data)
            text_fallback = input_data
        except json.JSONDecodeError:
            return TextInstructionTemplate.build_prompt(input_data), 'text'
    else:
        return TextInstructionTemplate.build_prompt(str(input_data)), 'text'

    if isinstance(data, dict):
        if 'actors' in data and 'use_cases' in data:
            return UMLInstructionTemplate.build_prompt(input_data), 'uml'
        details = data.get('details', {})
        if 'description' in data and ('objects' in details or 'scene' in details):
            return ImageInstructionTemplate.build_prompt(input_data), 'image'

    return TextInstructionTemplate.build_prompt(text_fallback), 'text'


class TextExpert(BaseExpert):
    """Generate instructions for text-domain inputs."""

    def __init__(self, lora_path: Optional[str] = None, use_4bit: bool = True):
        """Initialize the instance."""
        path_cfg = get_path_config()

        if lora_path is None:
            lora_weight_path = path_cfg.EXPERT_LORA_PATHS.get('text_expert')
            if lora_weight_path is None:
                logger.warning("配置中未找到text_expert的LoRA权重路径,将使用基础模型")
                lora_path = None
            else:
                lora_path_obj = Path(lora_weight_path)
                if not lora_path_obj.exists():
                    logger.warning(f"LoRA权重路径不存在: {lora_path_obj},将使用基础模型")
                    lora_path = None
                elif not lora_path_obj.is_dir():
                    logger.warning(f"LoRA权重路径不是目录: {lora_path_obj},将使用基础模型")
                    lora_path = None
                else:
                    lora_path = str(lora_path_obj)
                    logger.info(f"找到LoRA权重路径: {lora_path}")

        super().__init__(
            expert_name='text_expert',
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("文本专家初始化完成")

    def generate_instruction(self, input_data: str, sample_index: int = None) -> str:
        """Generate instruction."""
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            prompt, detected_domain = _build_prompt_for_domain(input_data)
            if detected_domain != 'text' and (sample_index is None or sample_index < 3):
                logger.warning(
                    f"输入数据检测为{detected_domain}类型，使用对应模板（跨域评估场景）"
                )

            if sample_index is None or sample_index < 3:
                logger.debug(f"生成指令 - 输入需求: {input_data[:100]}...")

            infer_cfg = get_inference_config()
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty,
                sample_index=sample_index,
                verbose=(sample_index is None or sample_index < 3)
            )

            instruction = self._normalize_instruction(instruction)

            if sample_index is None or sample_index < 3:
                logger.info("=" * 80)
                logger.info("模型原始输出:")
                logger.info("-" * 80)
                logger.info(instruction)
                logger.info("=" * 80)

            if self.validate_output(instruction):
                logger.info("指令生成成功,格式验证通过")
            else:
                if sample_index is None or sample_index < 3:
                    logger.warning("指令格式验证失败，直接返回模型输出")
                    logger.warning(f"验证未通过的指令内容：\n{instruction}")
            return instruction

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 16) -> list:
        """Generate instructions in batches."""
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return [""] * len(input_data_list)

        try:
            logger.info(f"批量生成指令 - 共{len(input_data_list)}个样本，batch_size={batch_size}")

            prompts = []
            for _idx, _data in enumerate(input_data_list):
                _prompt, _domain = _build_prompt_for_domain(_data)
                if _domain != 'text' and _idx < 3:
                    logger.warning(
                        f"样本{_idx}输入检测为{_domain}类型，使用对应模板（跨域评估场景）"
                    )
                prompts.append(_prompt)

            for i in range(min(3, len(input_data_list))):
                logger.info("=" * 80)
                logger.info(f"[样本 {i+1}/{len(input_data_list)}] 输入需求:")
                logger.info("-" * 80)
                logger.info(input_data_list[i][:200] + ("..." if len(input_data_list[i]) > 200 else ""))
                logger.info("=" * 80)

            infer_cfg = get_inference_config()
            instructions = self._generate_batch_with_model(
                prompts=prompts,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty,
                batch_size=batch_size,
                start_index=0,
                verbose=True
            )

            for i in range(min(3, len(instructions))):
                logger.info("=" * 80)
                logger.info(f"[样本 {i+1}/{len(input_data_list)}] 生成的指令:")
                logger.info("-" * 80)
                logger.info(instructions[i])
                logger.info("=" * 80)

            validated_instructions = []
            for i, instruction in enumerate(instructions):
                instruction = self._normalize_instruction(instruction)
                if not self.validate_output(instruction):
                    if i < 3:
                        logger.warning(f"样本{i+1}格式验证失败，直接使用模型输出")
                validated_instructions.append(instruction)

            return validated_instructions

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [""] * len(input_data_list)

    def validate_output(self, instruction: str) -> bool:
        """Validate output."""
        if not instruction or len(instruction.strip()) < 50:
            logger.debug("指令内容过短")
            return False

        result = TextInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"格式验证失败: {result['errors']}")
            return False

        return True

    def _fallback_generation(self, input_data: str) -> str:
        """Generate fallback output."""
        logger.info("使用回退方案生成指令")

        fallback_instruction = """Definition: In this task, implement and test the specified requirement.
Emphasis & Caution: Ensure thorough testing and validation of all functionality.
Things to Avoid: Do not skip error handling or edge case testing."""

        return fallback_instruction
