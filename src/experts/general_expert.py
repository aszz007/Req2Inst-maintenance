"""Implement the general-domain expert."""

import json
from pathlib import Path

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
                 lora_path: str | None = None,
                 use_4bit: bool = True):
        """Initialize the instance."""
        path_cfg = get_path_config()

        expert_name = 'general_expert'

        if lora_path is None:
            lora_weight_path = path_cfg.EXPERT_LORA_PATHS.get(expert_name)
            if lora_weight_path is None:
                logger.error(f"No LoRA weight path is configured for {expert_name}; expert loading will fail")
                lora_path = None
            else:
                lora_path_obj = Path(lora_weight_path)
                if not lora_path_obj.exists():
                    logger.error(f"LoRA weight path does not exist: {lora_path_obj}; expert loading will fail")
                    lora_path = None
                elif not lora_path_obj.is_dir():
                    logger.error(f"LoRA weight path is not a directory: {lora_path_obj}; expert loading will fail")
                    lora_path = None
                else:
                    lora_path = str(lora_path_obj)
                    logger.info(f"Found LoRA weight path: {lora_path}")

        super().__init__(
            expert_name=expert_name,
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("General expert initialized")

    def generate_instruction(self, input_data: str | dict, sample_index: int = None) -> str:
        """Generate instruction."""
        if not self.is_model_loaded:
            logger.warning("Model is not loaded; attempting to load it...")
            if not self.load_model():
                logger.error("Failed to load model")
                return ""

        try:
            show_debug = sample_index is None or sample_index < 3

            if show_debug:
                logger.info("[Debug] Raw input data:")
                if isinstance(input_data, dict):
                    logger.info("Input type: dict")
                    logger.info(f"First 500 characters of input: {str(input_data)[:500]}")
                else:
                    logger.info(f"Input type: {type(input_data).__name__}")
                    logger.info(f"First 500 characters of input: {str(input_data)[:500]}")

            if show_debug:
                input_type = self._detect_input_type(input_data)
                logger.info(f"[Debug] Detected input type: {input_type}")

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
                logger.info("Raw model output:")
                logger.info(instruction)

            if self.validate_output(instruction):
                logger.info("Instruction generated successfully and passed format validation")
                return instruction
            else:
                if show_debug:
                    logger.warning("Instruction failed format validation; using the normalized output directly")
                if not instruction or not instruction.strip():
                    logger.warning("Output is empty; using fallback generation")
                    return self._fallback_generation(input_data)
                return instruction

        except Exception as e:
            logger.error(f"Instruction generation failed: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 8) -> list:
        """Generate instructions in batches."""
        if not self.is_model_loaded:
            logger.warning("Model is not loaded; attempting to load it...")
            if not self.load_model():
                logger.error("Failed to load model")
                return [""] * len(input_data_list)

        try:
            logger.info(f"Batch instruction generation - {len(input_data_list)} samples, batch_size={batch_size}")

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
                logger.info(f"[Sample {i+1}/{len(input_data_list)}] Generated instruction:")
                logger.info(instructions[i])

            validated_instructions = []
            for i, instruction in enumerate(instructions):
                instruction = self._normalize_instruction(instruction)
                if not self.validate_output(instruction):
                    if i < 3:
                        logger.warning(
                            f"Sample {i+1} failed format validation; using the normalized output directly"
                        )
                    if not instruction or not instruction.strip():
                        logger.warning(f"Sample {i+1} output is empty; using fallback generation")
                        instruction = self._fallback_generation(input_data_list[i])
                validated_instructions.append(instruction)

            return validated_instructions

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            return [""] * len(input_data_list)

    def _detect_input_type(self, input_data: str | dict) -> str:
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
            logger.debug("Instruction is too short")
            return False

        result = GeneralInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"Format validation failed: {result['errors']}")
            return False

        return True

    def _fallback_generation(self, input_data: str | dict) -> str:
        """Generate fallback output."""
        logger.info("Using fallback instruction generation")

        fallback_instruction = """Definition: In this task, implement or test the specified requirement.
Emphasis & Caution: Ensure comprehensive testing and validation of all functionality.
Things to Avoid: Do not skip error handling or edge case validation."""

        return fallback_instruction
