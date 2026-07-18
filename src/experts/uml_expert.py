"""Implement the FlowChart-domain expert."""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.uml_template import UMLInstructionTemplate
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.uml')



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


class UMLExpert(BaseExpert):
    """Generate instructions for FlowChart-domain inputs."""

    def __init__(self,
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """Initialize the instance."""
        path_cfg = get_path_config()

        expert_name = 'uml_expert'

        if lora_path is None:
            lora_weight_path = path_cfg.EXPERT_LORA_PATHS.get(expert_name)
            if lora_weight_path is None:
                logger.warning(f"No LoRA weight path is configured for {expert_name}; using the base model")
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
            expert_name=expert_name,
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("FlowChart expert initialized")

    def generate_instruction(self, input_data: Union[str, dict], sample_index: int = None) -> str:
        """Generate instruction."""
        if not self.is_model_loaded:
            logger.warning("Model is not loaded; attempting to load it...")
            if not self.load_model():
                logger.error("Failed to load model")
                return ""

        try:
            show_debug = sample_index is None or sample_index < 3

            if show_debug:
                logger.info("[FlowChart expert debug] Raw input data:")
                logger.info(f"Data type: {type(input_data).__name__}")

                if isinstance(input_data, dict):
                    logger.info("Data content (dict):")
                    logger.info(json.dumps(input_data, indent=2, ensure_ascii=False))
                elif isinstance(input_data, str):
                    logger.info("Data content (str, first 500 characters):")
                    logger.info(input_data[:500])
                    try:
                        parsed = json.loads(input_data)
                        logger.info("\nParsed as JSON:")
                        logger.info(json.dumps(parsed, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError:
                        logger.info("\nCould not parse as JSON")
                else:
                    logger.info(f"Unknown data type: {input_data}")

            prompt, detected_domain = _build_prompt_for_domain(input_data)
            if detected_domain == 'uml':
                uml_data = json.loads(input_data) if isinstance(input_data, str) else input_data
                if show_debug:
                    elements = UMLInstructionTemplate.extract_key_elements(uml_data)
                    logger.debug(
                        f"Generating instruction - actors: {elements['actors']}, use cases: {len(elements['use_cases'])}"
                    )
            else:
                logger.warning(
                    f"Input detected as {detected_domain}; using the matching template for cross-domain evaluation"
                )
                uml_data = {}

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
            else:
                if show_debug:
                    logger.warning("Instruction failed format validation; returning the model output directly")
                    logger.warning(f"Instruction that failed validation:\n{instruction}")
            return instruction

        except Exception as e:
            logger.error(f"Instruction generation failed: {e}")
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

            parsed_data_list = [{}] * len(input_data_list)
            valid_indices = []
            valid_prompts = []

            for idx, data in enumerate(input_data_list):
                _prompt, _domain = _build_prompt_for_domain(data)
                if _domain == 'uml':
                    parsed_data_list[idx] = (
                        json.loads(data) if isinstance(data, str) else data
                    )
                else:
                    if idx < 3:
                        logger.warning(
                            f"Sample {idx} detected as {_domain}; using the matching template for cross-domain evaluation"
                        )
                    parsed_data_list[idx] = {}
                valid_indices.append(idx)
                valid_prompts.append(_prompt)

            raw_instructions = [""] * len(input_data_list)
            if valid_prompts:
                infer_cfg = get_inference_config()
                generated = self._generate_batch_with_model(
                    prompts=valid_prompts,
                    max_new_tokens=infer_cfg.max_new_tokens,
                    temperature=infer_cfg.temperature,
                    top_p=infer_cfg.top_p,
                    top_k=infer_cfg.top_k,
                    repetition_penalty=infer_cfg.repetition_penalty,
                    batch_size=batch_size,
                    start_index=0,
                    verbose=True
                )
                for orig_idx, instruction in zip(valid_indices, generated):
                    raw_instructions[orig_idx] = instruction

            instructions = raw_instructions

            shown = 0
            for i in range(len(instructions)):
                if shown >= 3:
                    break
                if i in valid_indices:
                    logger.info(f"[Sample {i+1}/{len(input_data_list)}] Generated instruction:")
                    logger.info(instructions[i])
                    shown += 1

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

        result = UMLInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"Format validation failed: {result['errors']}")
            return False

        if not result['has_business_logic']:
            logger.debug("Business-logic implementation requirement is missing")
            return False

        return True

    def _fallback_generation(self, uml_data: dict) -> str:
        """Generate fallback output."""
        logger.info("Using fallback instruction generation")

        fallback_instruction = """Definition: In this task, implement the system workflow with specified actors interacting with defined use cases.
Emphasis & Caution: Ensure all mandatory steps and conditional extensions are properly implemented.
Things to Avoid: Do not focus on UI positioning or visual layout. Avoid implementing frontend styling."""

        return fallback_instruction
