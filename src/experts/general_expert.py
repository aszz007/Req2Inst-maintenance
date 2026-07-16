"""Implement the general-domain expert."""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.general_template import GeneralInstructionTemplate
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.general')

def _build_prompt_for_domain(input_data):
    """Build prompt for domain."""
    if isinstance(input_data, dict):
        data = input_data
    elif isinstance(input_data, str):
        try:
            data = json.loads(input_data)
        except (json.JSONDecodeError, ValueError):
            return TextInstructionTemplate.build_prompt(input_data), 'text'
    else:
        return TextInstructionTemplate.build_prompt(str(input_data)), 'text'

    if isinstance(data, dict):
        if 'actors' in data and 'use_cases' in data:
            return UMLInstructionTemplate.build_prompt(input_data), 'uml'
        details = data.get('details', {})
        if isinstance(details, dict) and ('objects' in details or 'scene' in details):
            return ImageInstructionTemplate.build_prompt(input_data), 'image'

    return TextInstructionTemplate.build_prompt(input_data), 'text'

class GeneralExpert(BaseExpert):
    """Generate instructions for mixed-domain inputs."""

    def __init__(self,
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """Initialize the instance."""
        path_cfg = get_path_config()

        expert_name = 'general_expert'

        if lora_path is None:
            lora_weight_path = path_cfg.EXPERT_LORA_PATHS.get(expert_name)
            if lora_weight_path is None:
                logger.warning(f"配置中未找到{expert_name}的LoRA权重路径,将使用基础模型")
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
            expert_name=expert_name,
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("通用专家初始化完成")

    def generate_instruction(self, input_data: Union[str, dict], sample_index: int = None) -> str:
        """Generate instruction."""
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            show_debug = sample_index is None or sample_index < 3

            if show_debug:
                logger.info("=" * 80)
                logger.info("[调试] 原始输入数据:")
                logger.info("-" * 80)
                if isinstance(input_data, dict):
                    logger.info(f"输入类型: dict")
                    logger.info(f"输入内容（前500字符）: {str(input_data)[:500]}")
                else:
                    logger.info(f"输入类型: {type(input_data).__name__}")
                    logger.info(f"输入内容（前500字符）: {str(input_data)[:500]}")
                logger.info("=" * 80)

            if show_debug:
                input_type = self._detect_input_type(input_data)
                logger.info(f"[调试] 识别输入类型: {input_type}")

            prompt, _ = _build_prompt_for_domain(input_data)

            infer_cfg = get_inference_config()
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty,
                sample_index=sample_index,
                verbose=show_debug
            )

            instruction = self._normalize_instruction(instruction)

            if show_debug:
                logger.info("=" * 80)
                logger.info("模型原始输出:")
                logger.info("-" * 80)
                logger.info(instruction)
                logger.info("=" * 80)

            if self.validate_output(instruction):
                logger.info("指令生成成功,格式验证通过")
                return instruction
            else:
                if show_debug:
                    logger.warning(f"指令格式验证未通过，直接使用normalize后的输出")
                if not instruction or not instruction.strip():
                    logger.warning("输出为空，使用fallback兜底")
                    return self._fallback_generation(input_data)
                return instruction

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 8) -> list:
        """Generate instructions in batches."""
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return [""] * len(input_data_list)

        try:
            logger.info(f"批量生成指令 - 共{len(input_data_list)}个样本，batch_size={batch_size}")

            prompts = [_build_prompt_for_domain(data)[0] for data in input_data_list]

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
                        logger.warning(
                            f"样本{i+1}格式验证未通过，直接使用normalize后的输出"
                        )
                    if not instruction or not instruction.strip():
                        logger.warning(f"样本{i+1}输出为空，使用fallback兜底")
                        instruction = self._fallback_generation(input_data_list[i])
                validated_instructions.append(instruction)

            return validated_instructions

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return [""] * len(input_data_list)

    def _detect_input_type(self, input_data: Union[str, dict]) -> str:
        """Detect input type."""
        if isinstance(input_data, dict):
            if 'actors' in input_data and 'use_cases' in input_data:
                return 'uml'
            elif 'description' in input_data or 'details' in input_data:
                return 'image'
            else:
                return 'unknown'

        elif isinstance(input_data, str):
            try:
                parsed = json.loads(input_data)
                return self._detect_input_type(parsed)
            except json.JSONDecodeError:
                return 'text'

        return 'unknown'

    def validate_output(self, instruction: str) -> bool:
        """Validate output."""
        if not instruction or len(instruction.strip()) < 50:
            logger.debug("指令内容过短")
            return False

        result = GeneralInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"格式验证失败: {result['errors']}")
            return False

        return True

    def _fallback_generation(self, input_data: Union[str, dict]) -> str:
        """Generate fallback output."""
        logger.info("使用回退方案生成指令")

        fallback_instruction = """Definition: In this task, implement or test the specified requirement.
Emphasis & Caution: Ensure comprehensive testing and validation of all functionality.
Things to Avoid: Do not skip error handling or edge case validation."""

        return fallback_instruction
