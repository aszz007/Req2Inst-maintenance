"""Select a domain expert with rule-based input-type routing."""

import json
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RoutingResult:
    """Store expert-routing output and metadata."""
    expert_name: str
    expert_path: str
    expert_type: str
    reasoning: str


@dataclass
class ExpertConfig:
    """Store expert model and adapter configuration."""
    name: str
    expert_type: str
    model_version: str
    dataset_variant: Optional[str]
    path: str
    is_default: bool


class ExpertRouter:
    """Select experts with rule-based input-type routing."""

    DEFAULT_EXPERTS = {
        'text': 'text_expert',
        'image': 'image_expert',
        'uml': 'uml_expert',
        'general': 'general_expert'
    }

    def __init__(self, lora_weights_dir: str = "lora_weights/experts"):
        """Initialize the instance."""
        self.lora_weights_dir = Path(lora_weights_dir)
        self.expert_registry = self._build_expert_registry()
        self.routing_stats = defaultdict(int)

    def _build_expert_registry(self) -> Dict[str, List[ExpertConfig]]:
        """Build expert registry."""
        registry = {
            'text': [],
            'image': [],
            'uml': [],
            'general': []
        }

        registry['text'].append(ExpertConfig(
            name='text_expert',
            expert_type='text',
            model_version='qwen3_8b',
            dataset_variant=None,
            path=str(self.lora_weights_dir / 'text_expert'),
            is_default=True
        ))

        registry['image'].append(ExpertConfig(
            name='image_expert',
            expert_type='image',
            model_version='qwen3_8b',
            dataset_variant=None,
            path=str(self.lora_weights_dir / 'image_expert'),
            is_default=True
        ))

        registry['uml'].append(ExpertConfig(
            name='uml_expert',
            expert_type='uml',
            model_version='qwen3_8b',
            dataset_variant=None,
            path=str(self.lora_weights_dir / 'uml_expert'),
            is_default=True
        ))

        registry['general'].append(ExpertConfig(
            name='general_expert',
            expert_type='general',
            model_version='qwen3_8b',
            dataset_variant=None,
            path=str(self.lora_weights_dir / 'general_expert'),
            is_default=True
        ))

        return registry

    def route(
            self,
            input_data: dict,
            expert_variant: Optional[str] = None
    ) -> RoutingResult:
        """Route an input to an expert."""
        input_type = input_data.get('type', '').lower()

        if input_type not in ['text', 'image', 'uml', 'general']:
            input_type = 'general'

        if expert_variant:
            expert = self._get_expert_by_name(expert_variant)
            if expert:
                self.routing_stats[expert.name] += 1
                return RoutingResult(
                    expert_name=expert.name,
                    expert_path=expert.path,
                    expert_type=expert.expert_type,
                    reasoning=f"Explicitly specified expert variant: {expert_variant}"
                )
            else:
                raise ValueError(f"Expert variant '{expert_variant}' not found in registry")

        default_expert_name = self.DEFAULT_EXPERTS[input_type]
        expert = self._get_expert_by_name(default_expert_name)

        self.routing_stats[expert.name] += 1

        return RoutingResult(
            expert_name=expert.name,
            expert_path=expert.path,
            expert_type=expert.expert_type,
            reasoning=f"Default expert for {input_type} input"
        )

    def _get_expert_by_name(self, expert_name: str) -> Optional[ExpertConfig]:
        """Return expert by name."""
        for expert_list in self.expert_registry.values():
            for expert in expert_list:
                if expert.name == expert_name:
                    return expert
        return None

    def list_experts(
            self,
            expert_type: Optional[str] = None,
            only_defaults: bool = False
    ) -> List[ExpertConfig]:
        """List configured experts."""
        if expert_type:
            experts = self.expert_registry.get(expert_type, [])
        else:
            experts = []
            for expert_list in self.expert_registry.values():
                experts.extend(expert_list)

        if only_defaults:
            experts = [e for e in experts if e.is_default]

        return experts

    def get_available_variants(self, expert_type: str) -> List[str]:
        """Return available variants."""
        experts = self.expert_registry.get(expert_type, [])
        return [e.name for e in experts]

    def get_routing_statistics(self) -> Dict:
        """Return routing statistics."""
        total_routings = sum(self.routing_stats.values())

        stats = {
            'total_routings': total_routings,
            'expert_usage_count': dict(self.routing_stats),
            'expert_usage_percentage': {
                expert: (count / total_routings * 100) if total_routings > 0 else 0
                for expert, count in self.routing_stats.items()
            },
            'total_experts': sum(len(experts) for experts in self.expert_registry.values()),
            'experts_by_type': {
                expert_type: len(expert_list)
                for expert_type, expert_list in self.expert_registry.items()
            }
        }

        return stats

    def reset_statistics(self):
        """Reset statistics."""
        self.routing_stats.clear()
