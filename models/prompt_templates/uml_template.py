"""Define the FlowChart-domain prompt template for three-part instruction generation."""

import json


class UMLInstructionTemplate:
    """Build FlowChart-domain instruction prompts."""

    UML_RECOGNITION_PROMPT = """Please carefully analyze this Use Case Diagram and output the recognition results in JSON format.

A Use Case Diagram is a type of UML diagram used to describe system functions and user interactions. Please identify:

1. **actors**: List of actors (typically stick figures or text labels)
   - Each actor includes: name, position (e.g., "left", "right")

2. **use_cases**: List of use cases (typically ovals)
   - Each use case includes: name, description (brief description)

3. **system_boundary**: System boundary
   - Includes: name (system name), is_present (whether boundary box exists)

4. **relationships**: List of relationships
   - Each relationship includes:
     - type ("association", "include", "extend", "generalization")
     - from (starting element)
     - to (ending element)
     - description (relationship description)

5. **overall_description**: Overall description (summarize the system functionality in one paragraph)

Please output strictly in JSON format. Example:
{
  "actors": [{"name": "User", "position": "left"}],
  "use_cases": [{"name": "Login System", "description": "User login functionality"}],
  "system_boundary": {"name": "System Name", "is_present": true},
  "relationships": [{"type": "association", "from": "User", "to": "Login System", "description": "User can login"}],
  "overall_description": "This is a use case diagram..."
}

If the image is not a use case diagram or cannot be recognized, please explain in overall_description.
Important: Ensure complete JSON output with all brackets properly closed. Use English for all content."""

    @staticmethod
    def get_recognition_prompt() -> str:
        """Return recognition prompt."""
        return UMLInstructionTemplate.UML_RECOGNITION_PROMPT

    SYSTEM_PROMPT = """You are a software architecture and crowdsourcing task design expert. Based on the input UML Use Case Diagram structured data (JSON format), write an English task instruction for crowdsourcing workers.

Core Principles:
1. Data-Driven: Actor names and Use Case names in the instruction must strictly reference the original names from JSON source data. Do not omit, abbreviate, or rewrite.
2. Logic Priority, Visuals Secondary: Completely ignore visual layout information like position (e.g., top_left) in input data. Focus on parsing business logic in relationships.
3. Relationship Semantics Translation:
   - include -> Translate to "Mandatory step" or "Required prerequisite"
   - extend -> Translate to "Conditional flow" or "Optional"
   - association -> Translate to "Interaction" or "Access"
4. Structured Format: Strictly follow the three-part format defined below."""

    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the core system objective. Must start with "In this task,".
Emphasis & Caution: Highlight mandatory flows (include) and conditional extension flows (extend). Use "-" if none.
Things to Avoid: List prohibited operations (e.g., focusing on node positions, implementing UI styles). Use "-" if nothing specific.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must start with "In this task," and explicitly list actors and use cases from JSON data
- Translate relationship types (include/extend/association) to business logic terms
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def build_prompt(uml_json: str | dict) -> str:
        """Build prompt."""
        if isinstance(uml_json, dict):
            filtered_data = {
                k: v for k, v in uml_json.items()
                if k not in ['confidence', 'recognition_status', 'processing_time']
            }
            if 'actors' in filtered_data and isinstance(filtered_data['actors'], list):
                filtered_actors = []
                for actor in filtered_data['actors']:
                    if isinstance(actor, dict):
                        filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                        filtered_actors.append(filtered_actor)
                    else:
                        filtered_actors.append(actor)
                filtered_data['actors'] = filtered_actors

            json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        elif isinstance(uml_json, str):
            try:
                parsed = json.loads(uml_json)
                filtered_data = {
                    k: v for k, v in parsed.items()
                    if k not in ['confidence', 'recognition_status', 'processing_time']
                }
                if 'actors' in filtered_data and isinstance(filtered_data['actors'], list):
                    filtered_actors = []
                    for actor in filtered_data['actors']:
                        if isinstance(actor, dict):
                            filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                            filtered_actors.append(filtered_actor)
                        else:
                            filtered_actors.append(actor)
                    filtered_data['actors'] = filtered_actors

                json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                json_str = uml_json
        else:
            raise TypeError("uml_json must be a str or dict")

        user_message = f"""UML Use Case Diagram structured data (JSON format):
```json
{json_str}
```

{UMLInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        prompt = f"""<|im_start|>system
{UMLInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        return prompt

    @staticmethod
    def build_batch_prompt(uml_jsons: list) -> list:
        """Build batch prompt."""
        return [
            UMLInstructionTemplate.build_prompt(uml)
            for uml in uml_jsons
        ]

    @staticmethod
    def extract_key_elements(uml_data: str | dict) -> dict:
        """Extract key elements."""
        if isinstance(uml_data, str):
            try:
                data = json.loads(uml_data)
            except json.JSONDecodeError:
                return {
                    'actors': [],
                    'use_cases': [],
                    'include_relations': [],
                    'extend_relations': [],
                    'associations': []
                }
        else:
            data = uml_data

        actors = [
            actor.get('name', actor) if isinstance(actor, dict) else actor
            for actor in data.get('actors', [])
        ]

        use_cases = []
        for uc in data.get('use_cases', []):
            if isinstance(uc, dict):
                use_cases.append({
                    'name': uc.get('name', ''),
                    'description': uc.get('description', '')
                })
            else:
                use_cases.append({'name': str(uc), 'description': ''})

        relationships = data.get('relationships', [])
        include_relations = []
        extend_relations = []
        associations = []

        for rel in relationships:
            if isinstance(rel, dict):
                rel_type = rel.get('type', '').lower()
                relation_info = {
                    'from': rel.get('from', ''),
                    'to': rel.get('to', ''),
                    'description': rel.get('description', '')
                }

                if 'include' in rel_type:
                    include_relations.append(relation_info)
                elif 'extend' in rel_type:
                    extend_relations.append(relation_info)
                elif 'association' in rel_type:
                    associations.append(relation_info)

        return {
            'actors': actors,
            'use_cases': use_cases,
            'include_relations': include_relations,
            'extend_relations': extend_relations,
            'associations': associations
        }

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """Validate instruction."""
        result = {
            'is_valid': True,
            'has_definition': False,
            'has_business_logic': False,
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
                    business_keywords = [
                        'implement', 'functionality', 'workflow', 'process',
                        'interaction', 'interact', 'trigger', 'system',
                        'analyze', 'manage', 'execute', 'perform'
                    ]
                    if any(keyword in line_lower for keyword in business_keywords):
                        result['has_business_logic'] = True
                else:
                    result['errors'].append('Definition部分内容为空')

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True

        if not result['has_definition']:
            result['errors'].append('缺少"Definition:"部分或格式错误')

        if not result['has_business_logic']:
            result['errors'].append('Definition未体现业务逻辑实现要求')

        if not result['has_emphasis']:
            result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')

        if not result['has_avoid']:
            result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

        result['is_valid'] = all([
            result['has_definition'],
            result['has_business_logic'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result
