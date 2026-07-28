"""Define the text-domain prompt template for three-part instruction generation."""


class TextInstructionTemplate:
    """Build text-domain instruction prompts."""

    SYSTEM_PROMPT = """You are a crowdsourcing task design expert. Based on the input requirement text, write an English task instruction for crowdsourcing workers.

Core Principles:
1. Extreme Conciseness: Crowdsourcing workers value time. Use the most concise language possible.
2. Structured Format: Strictly follow the three-part format defined below.
3. English Output: Output must be in English regardless of input language."""

    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the main objective. Must start with "In this task,".
Emphasis & Caution: Only highlight conditions most prone to error or that must be met. Use "-" if nothing specific to emphasize.
Things to Avoid: Only list prohibited operations. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def build_prompt(low_requirement: str) -> str:
        """Build prompt."""
        user_message = f"""Requirement text:
{low_requirement}

{TextInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        prompt = f"""<|im_start|>system
{TextInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        return prompt

    @staticmethod
    def build_batch_prompt(low_requirements: list) -> list:
        """Build batch prompt."""
        return [
            TextInstructionTemplate.build_prompt(req)
            for req in low_requirements
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
            result['errors'].append(
                f'Instruction has too few lines; expected at least 3, got {len(lines)}'
            )
            result['is_valid'] = False
            return result

        for i, line in enumerate(lines):
            if line.startswith('Definition:'):
                content = line[len('Definition:'):].strip()
                if content:
                    result['has_definition'] = True
                else:
                    result['errors'].append('Definition section is empty')

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True

        if not result['has_definition']:
            result['errors'].append('Missing "Definition:" section or invalid format')

        if not result['has_emphasis']:
            result['errors'].append(
                'Missing "Emphasis & Caution:" section or invalid format'
            )

        if not result['has_avoid']:
            result['errors'].append(
                'Missing "Things to Avoid:" section or invalid format'
            )

        result['is_valid'] = all([
            result['has_definition'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result
