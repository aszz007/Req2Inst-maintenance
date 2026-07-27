"""Coordinate expert loading, routing, and instruction generation."""

from pathlib import Path

from .expert_router import ExpertRouter, RoutingResult


class MoEModel:
    """Coordinate expert routing, loading, and generation."""

    def __init__(
            self,
            lora_weights_dir: str = "lora_weights/experts",
            base_models_dir: str = "base_models"
    ):
        """Initialize the instance."""
        self.lora_weights_dir = Path(lora_weights_dir)
        self.base_models_dir = Path(base_models_dir)

        self.router = ExpertRouter(lora_weights_dir)

        self._loaded_experts = {}

    def generate_instruction(
            self,
            input_data: str | dict,
            expert_variant: str | None = None,
            **generation_kwargs
    ) -> dict:
        """Generate instruction."""
        formatted_input = self._format_input(input_data)

        routing_result = self.router.route(
            formatted_input,
            expert_variant=expert_variant
        )

        expert = self._get_or_load_expert(routing_result)

        instruction = expert.generate_instruction(
            formatted_input['content'],
            **generation_kwargs
        )

        return {
            'instruction': instruction,
            'expert_used': routing_result.expert_name,
            'expert_type': routing_result.expert_type,
            'reasoning': routing_result.reasoning
        }

    def _format_input(self, input_data: str | dict) -> dict:
        """Format input."""
        if isinstance(input_data, str):
            return {
                'type': 'text',
                'content': input_data
            }
        elif isinstance(input_data, dict):
            if 'type' not in input_data or 'content' not in input_data:
                raise ValueError("Input dict must contain 'type' and 'content' fields")
            return input_data
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _get_or_load_expert(self, routing_result: RoutingResult):
        """Return or load expert."""
        expert_name = routing_result.expert_name

        if expert_name not in self._loaded_experts:
            self._loaded_experts[expert_name] = self._load_expert(routing_result)

        return self._loaded_experts[expert_name]

    def _load_expert(self, routing_result: RoutingResult):
        """Load expert."""
        from src.experts import GeneralExpert, ImageExpert, TextExpert, UMLExpert

        expert_type = routing_result.expert_type
        expert_path = routing_result.expert_path

        if expert_type == 'text':
            return TextExpert(lora_path=expert_path)

        elif expert_type == 'image':
            return ImageExpert(lora_path=expert_path)

        elif expert_type == 'uml':
            return UMLExpert(lora_path=expert_path)

        elif expert_type == 'general':
            return GeneralExpert(lora_path=expert_path)

        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def batch_generate(
            self,
            input_list: list[str | dict],
            expert_variant: str | None = None,
            **generation_kwargs
    ) -> list[dict]:
        """Generate outputs in batches."""
        results = []
        for input_data in input_list:
            result = self.generate_instruction(
                input_data,
                expert_variant=expert_variant,
                **generation_kwargs
            )
            results.append(result)
        return results

    def get_router_statistics(self) -> dict:
        """Return router statistics."""
        return self.router.get_routing_statistics()

    def reset_router_statistics(self):
        """Reset router statistics."""
        self.router.reset_statistics()

    def list_available_experts(
            self,
            expert_type: str | None = None
    ) -> list[str]:
        """List available experts."""
        if expert_type:
            return self.router.get_available_variants(expert_type)
        else:
            all_experts = []
            for etype in ['text', 'image', 'uml', 'general']:
                all_experts.extend(self.router.get_available_variants(etype))
            return all_experts
