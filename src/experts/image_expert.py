"""Implement the image-domain expert."""

import json
from pathlib import Path

from src.experts.base_expert import BaseExpert
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.image')



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


class ImageExpert(BaseExpert):
    """Generate instructions for image-domain inputs."""

    def __init__(self,
                 lora_path: str | None = None,
                 use_4bit: bool = True):
        """Initialize the instance."""
        path_cfg = get_path_config()

        expert_name = 'image_expert'

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

        logger.info("Image expert initialized")

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
                logger.info("[Image expert debug] Raw input data:")
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
                        logger.info("\nCould not parse as JSON; treating input as a plain-text description")
                else:
                    logger.info(f"Unknown data type: {input_data}")

            prompt, detected_domain = _build_prompt_for_domain(input_data)
            if detected_domain != 'image' and show_debug:
                logger.warning(
                    f"Input detected as {detected_domain}; using the matching template for cross-domain evaluation"
                )

            if show_debug:
                logger.debug(f"Generating instruction - input data type: {type(input_data).__name__}")

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
            elif show_debug:
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
                if _domain != 'image' and _idx < 3:
                    logger.warning(
                        f"Sample {_idx} detected as {_domain}; using the matching template for cross-domain evaluation"
                    )
                prompts.append(_prompt)

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

        result = ImageInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"Format validation failed: {result['errors']}")
            return False

        if not result['has_bounding_boxes']:
            logger.debug("Bounding-box requirement is missing")
            return False

        return True

    def _fallback_generation(self, input_data: str | dict) -> str:
        """Generate fallback output."""
        logger.info("Using fallback instruction generation")

        fallback_instruction = """Definition: In this task, draw bounding boxes around all visible objects in the image.
Emphasis & Caution: Focus on accurately identifying and labeling all foreground objects.
Things to Avoid: Do not annotate background elements or partial objects."""

        return fallback_instruction
