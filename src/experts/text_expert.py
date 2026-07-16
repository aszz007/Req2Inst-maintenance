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
                logger.warning("No LoRA weight path is configured for text_expert; using the base model")
                lora_path = None
            else:
                lora_path_obj = Path(lora_weight_path)
                if not lora_path_obj.exists():
                    logger.warning(f"LoRA weight path does not exist: {lora_path_obj}; using the base model")
                    lora_path = None
                elif not lora_path_obj.is_dir():
                    logger.warning(f"LoRA weight path is not a directory: {lora_path_obj}; using the base model")
                    lora_path = None
                else:
                    lora_path = str(lora_path_obj)
                    logger.info(f"Found LoRA weight path: {lora_path}")

        super().__init__(
            expert_name='text_expert',
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("Text expert initialized")

    def generate_instruction(self, input_data: str, sample_index: int = None) -> str:
        """Generate instruction."""
        if not self.is_model_loaded:
            logger.warning("Model is not loaded; attempting to load it...")
            if not self.load_model():
                logger.error("Failed to load model")
                return ""

        try:
            prompt, detected_domain = _build_prompt_for_domain(input_data)
            if detected_domain != 'text' and (sample_index is None or sample_index < 3):
                logger.warning(
                    f"Input detected as {detected_domain}; using the matching template for cross-domain evaluation"
                )

            if sample_index is None or sample_index < 3:
                logger.debug(f"Generating instruction - input requirement: {input_data[:100]}...")

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
                logger.info("Raw model output:")
                logger.info(instruction)

            if self.validate_output(instruction):
                logger.info("Instruction generated successfully and passed format validation")
            else:
                if sample_index is None or sample_index < 3:
                    logger.warning("Instruction failed format validation; returning the model output directly")
                    logger.warning(f"Instruction that failed validation:\n{instruction}")
            return instruction

        except Exception as e:
            logger.error(f"Instruction generation failed: {e}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 16) -> list:
        """Generate instructions in batches."""
        if not self.is_model_loaded:
            logger.warning("Model is not loaded; attempting to load it...")
            if not self.load_model():
                logger.error("Failed to load model")
                return [""] * len(input_data_list)

        try:
            logger.info(f"Batch instruction generation - {len(input_data_list)} samples, batch_size={batch_size}")

            prompts = []
            for _idx, _data in enumerate(input_data_list):
                _prompt, _domain = _build_prompt_for_domain(_data)
                if _domain != 'text' and _idx < 3:
                    logger.warning(
                        f"Sample {_idx} detected as {_domain}; using the matching template for cross-domain evaluation"
                    )
                prompts.append(_prompt)

            for i in range(min(3, len(input_data_list))):
                logger.info(f"[Sample {i+1}/{len(input_data_list)}] Input requirement:")
                logger.info(input_data_list[i][:200] + ("..." if len(input_data_list[i]) > 200 else ""))

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
                logger.info(f"[Sample {i+1}/{len(input_data_list)}] Generated instruction:")
                logger.info(instructions[i])

            validated_instructions = []
            for i, instruction in enumerate(instructions):
                instruction = self._normalize_instruction(instruction)
                if not self.validate_output(instruction):
                    if i < 3:
                        logger.warning(f"Sample {i+1} failed format validation; using the model output directly")
                validated_instructions.append(instruction)

            return validated_instructions

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [""] * len(input_data_list)

    def validate_output(self, instruction: str) -> bool:
        """Validate output."""
        if not instruction or len(instruction.strip()) < 50:
            logger.debug("Instruction is too short")
            return False

        result = TextInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"Format validation failed: {result['errors']}")
            return False

        return True

    def _fallback_generation(self, input_data: str) -> str:
        """Generate fallback output."""
        logger.info("Using fallback instruction generation")

        fallback_instruction = """Definition: In this task, implement and test the specified requirement.
Emphasis & Caution: Ensure thorough testing and validation of all functionality.
Things to Avoid: Do not skip error handling or edge case testing."""

        return fallback_instruction
