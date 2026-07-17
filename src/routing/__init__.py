"""Initialize the routing package."""

from .expert_router import ExpertRouter, RoutingResult, ExpertConfig
from .moe_model import MoEModel
from .soft_router import SoftRouter, SoftRoutingConfig, build_type_aware_weights
from .learned_router import (
    RouterMLP,
    HiddenStateExtractor,
    LearnedRouterInference,
    load_router_from_checkpoint,
    EXPERT_TO_IDX,
    IDX_TO_EXPERT,
    ALL_EXPERTS,
)

__all__ = [
    # Hard Routing
    'ExpertRouter',
    'RoutingResult',
    'ExpertConfig',
    'MoEModel',
    # Soft Routing (exp9)
    'SoftRouter',
    'SoftRoutingConfig',
    'build_type_aware_weights',
    # Learned Routing (exp10)
    'RouterMLP',
    'HiddenStateExtractor',
    'LearnedRouterInference',
    'load_router_from_checkpoint',
    'EXPERT_TO_IDX',
    'IDX_TO_EXPERT',
    'ALL_EXPERTS',
]

__version__ = '1.2.0'