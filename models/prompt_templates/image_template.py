"""Define the image-domain prompt template for three-part instruction generation."""

import json
from typing import Union


class ImageInstructionTemplate:
    """Build image-domain instruction prompts."""

    IMAGE_RECOGNITION_PROMPT = """Please describe this image in detail and output in JSON format with the following fields:
1. description: Overall description of the image (summarize in one sentence)
2. details: Contains the following sub-fields
   - objects: List of main objects in the image
   - scene: Scene type (e.g., "urban street", "indoor scene", etc.)
   - spatial_info: Spatial position information of objects

Please output strictly in JSON format with no other content. Use ONLY English in all fields."""

    @staticmethod
    def get_recognition_prompt() -> str:
        """Return recognition prompt."""
        return ImageInstructionTemplate.IMAGE_RECOGNITION_PROMPT

    SYSTEM_PROMPT = """You are a computer vision data expert and crowdsourcing task designer. Based on the input image analysis structured data, write an English image annotation instruction for crowdsourcing workers.

Core Principles:
1. Annotation Focus: The instruction must explicitly require workers to draw bounding boxes.
2. Foreground Extraction: Extract main foreground objects (e.g., people, vehicles) from the objects list as annotation targets. Ignore background elements.
3. Direct Reference: Use English terms directly from the JSON data. Do not replace with synonyms.
4. Extreme Conciseness: Keep Emphasis and Avoid sections brief. Use "-" if no significant visual features or distractors exist."""

    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the annotation targets. Must start with "In this task," and explicitly mention "draw bounding boxes around".
Emphasis & Caution: Only list highly distinctive visual features (e.g., specific colors, positions). Use "-" if nothing specific to emphasize.
Things to Avoid: Only list confusing background distractors. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must include "draw bounding boxes around" and list specific objects from JSON data
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def build_prompt(image_description: Union[str, dict]) -> str:
        """Build prompt."""
        if isinstance(image_description, dict):
            filtered_data = {
                k: v for k, v in image_description.items()
                if k not in ['confidence', 'recognition_status', 'processing_time']
            }
            json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        elif isinstance(image_description, str):
            try:
                parsed = json.loads(image_description)
                filtered_data = {
                    k: v for k, v in parsed.items()
                    if k not in ['confidence', 'recognition_status', 'processing_time']
                }
                json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                json_str = json.dumps({
                    "description": image_description,
                    "details": {
                        "objects": [],
                        "scene": "unknown",
                        "spatial_info": ""
                    }
                }, ensure_ascii=False, separators=(',', ':'))
        else:
            raise TypeError("image_description must be a str or dict")

        user_message = f"""Image analysis structured data (JSON format):
```json
{json_str}
```

{ImageInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        prompt = f"""<|im_start|>system
{ImageInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        return prompt

    @staticmethod
    def build_batch_prompt(image_descriptions: list) -> list:
        """Build batch prompt."""
        return [
            ImageInstructionTemplate.build_prompt(desc)
            for desc in image_descriptions
        ]

    @staticmethod
    def extract_description_from_json(json_data: Union[str, dict]) -> str:
        """Extract description from JSON."""
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                return json_data
        else:
            data = json_data

        return data.get('description', str(data))

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """Validate instruction."""
        result = {
            'is_valid': True,
            'has_definition': False,
            'has_bounding_boxes': False,
            'has_emphasis': False,
            'has_avoid': False,
            'errors': []
        }

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            result['errors'].append(f'指令行数不足，期望至少3行，实际{len(lines)}行')
            result['is_valid'] = False
            return result

        for line in lines:
            line_lower = line.lower()

            if line.startswith('Definition:'):
                content = line[len('Definition:'):].strip()
                if content:
                    result['has_definition'] = True
                    if 'bounding box' in line_lower or 'draw box' in line_lower:
                        result['has_bounding_boxes'] = True
                else:
                    result['errors'].append('Definition部分内容为空')

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True

        if not result['has_definition']:
            result['errors'].append('缺少"Definition:"部分或格式错误')

        if not result['has_bounding_boxes']:
            result['errors'].append('Definition未明确要求画边框（draw bounding boxes）')

        if not result['has_emphasis']:
            result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')

        if not result['has_avoid']:
            result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

        result['is_valid'] = all([
            result['has_definition'],
            result['has_bounding_boxes'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result
