"""Define the general prompt template for three-part instruction generation."""

import json
from typing import Union


class GeneralInstructionTemplate:
    """Build general-domain instruction prompts."""

    SYSTEM_PROMPT = """You are an expert crowdsourcing task designer. Based on the input (which may be a text requirement, image description, or UML diagram description), write an English task instruction for crowdsourcing workers.

Core Principles:
1. Adapt to Input Type: Recognize whether the input is a text requirement, image annotation task, or UML diagram task, and generate appropriate instructions.
2. Extreme Conciseness: Keep all sections brief and focused.
3. Structured Format: Strictly follow the three-part format.
4. English Output: Always output in English."""

    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the main task. Must start with "In this task,".
Emphasis & Caution: Highlight key requirements or common errors. Use "-" if nothing specific to emphasize.
Things to Avoid: List prohibited operations or confusing elements. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- For image tasks, explicitly mention "draw bounding boxes" in Definition
- For UML tasks, specify the diagram type (class/sequence/use case) in Definition
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def detect_input_type(input_data: Union[str, dict]) -> str:
        """Detect input type."""
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data)
        else:
            input_str = str(input_data)

        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, dict):
                if 'description' in parsed and 'details' in parsed:
                    details = parsed.get('details', {})
                    if 'objects' in details and 'scene' in details:
                        return 'image'
                    if 'diagram_type' in details:
                        return 'uml'

                if 'diagram_type' in parsed:
                    return 'uml'

                if 'description' in parsed:
                    return 'text'

        except (json.JSONDecodeError, TypeError):
            pass

        return 'text'

    @staticmethod
    def build_prompt(input_data: Union[str, dict], force_type: str = None) -> str:
        """Build prompt."""
        if force_type:
            input_type = force_type
        else:
            input_type = GeneralInstructionTemplate.detect_input_type(input_data)

        if isinstance(input_data, dict):
            if input_type == 'uml' and 'actors' in input_data:
                import copy
                input_data_copy = copy.deepcopy(input_data)
                if isinstance(input_data_copy['actors'], list):
                    filtered_actors = []
                    for actor in input_data_copy['actors']:
                        if isinstance(actor, dict):
                            filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                            filtered_actors.append(filtered_actor)
                        else:
                            filtered_actors.append(actor)
                    input_data_copy['actors'] = filtered_actors
                input_data = input_data_copy

            input_str = json.dumps(input_data, ensure_ascii=False, separators=(',', ':'))
        elif isinstance(input_data, str):
            try:
                parsed = json.loads(input_data)
                if input_type == 'uml' and 'actors' in parsed:
                    import copy
                    parsed_copy = copy.deepcopy(parsed)
                    if isinstance(parsed_copy['actors'], list):
                        filtered_actors = []
                        for actor in parsed_copy['actors']:
                            if isinstance(actor, dict):
                                filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                                filtered_actors.append(filtered_actor)
                            else:
                                filtered_actors.append(actor)
                        parsed_copy['actors'] = filtered_actors
                    parsed = parsed_copy

                input_str = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                input_str = input_data
        else:
            input_str = str(input_data)

        if input_type == 'image':
            user_message = f"""Image description (JSON format):
```json
{input_str}
```

Task: Generate an image annotation instruction for crowdsourcing workers to draw bounding boxes around objects.

{GeneralInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        elif input_type == 'uml':
            user_message = f"""UML diagram description (JSON format):
```json
{input_str}
```

Task: Generate a UML diagram analysis instruction for crowdsourcing workers.

{GeneralInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        else:  # text
            user_message = f"""Requirement text:
{input_str}

{GeneralInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        prompt = f"""<|im_start|>system
{GeneralInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        return prompt

    @staticmethod
    def build_batch_prompt(input_data_list: list) -> list:
        """Build batch prompt."""
        return [
            GeneralInstructionTemplate.build_prompt(data)
            for data in input_data_list
        ]

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """Validate instruction."""
        result = {
            'is_valid': True,
            'has_definition': False,
            'has_emphasis': False,
            'has_avoid': False,
            'errors': []
        }

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            result['errors'].append(f'指令行数不足,期望至少3行,实际{len(lines)}行')
            result['is_valid'] = False
            return result

        for line in lines:
            if line.startswith('Definition:'):
                content = line[len('Definition:'):].strip()
                if content:
                    result['has_definition'] = True
                else:
                    result['errors'].append('Definition部分内容为空')

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True

        if not result['has_definition']:
            result['errors'].append('缺少"Definition:"部分或格式错误')

        if not result['has_emphasis']:
            result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')

        if not result['has_avoid']:
            result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

        result['is_valid'] = all([
            result['has_definition'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result
