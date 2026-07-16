"""Implement weighted LoRA-adapter routing for routing experiments."""

import torch
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from peft import PeftModel

from src.utils.logger import get_logger

logger = get_logger('routing.soft_router')


@dataclass
class SoftRoutingConfig:
    """Store soft-routing configuration."""
    alpha: float = 0.5
    combination_type: str = "linear"


def check_peft_version():
    """Check peft version."""
    try:
        import peft
        version = peft.__version__
        major, minor = [int(x) for x in version.split('.')[:2]]
        supported = (major > 0) or (major == 0 and minor >= 6)
        if not supported:
            logger.error(f"PEFT版本过低: {version}，需要 >= 0.6.0")
        else:
            logger.info(f"PEFT版本检查通过: {version}")
        return supported
    except Exception as e:
        logger.error(f"PEFT版本检查失败: {e}")
        return False


def _clean_adapter_config(adapter_path: Path) -> Path:
    """Clean adapter config."""
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        return adapter_path

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    incompatible_params = [
        'alora_invocation_tokens', 'alora_prefix',
        'alora_suffix', 'arrow_config',
    ]

    needs_cleaning = any(p in config for p in incompatible_params)
    if not needs_cleaning:
        return adapter_path

    temp_dir = Path(tempfile.mkdtemp(prefix="lora_soft_router_"))
    for item in adapter_path.iterdir():
        if item.is_file():
            shutil.copy2(item, temp_dir / item.name)

    for param in incompatible_params:
        if param in config:
            del config[param]

    with open(temp_dir / "adapter_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"已清理adapter配置: {adapter_path.name}")
    return temp_dir


class SoftRouter:
    """Blend LoRA adapters for soft-routing experiments."""

    EXPERT_NAMES = ['text_expert', 'image_expert', 'uml_expert', 'general_expert']

    def __init__(
        self,
        base_model,
        tokenizer,
        adapter_paths: Dict[str, str],
    ):
        """Initialize the instance."""
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.adapter_paths = adapter_paths
        self.peft_model = None
        self._temp_dirs = []
        self._adapters_loaded = False
        self._current_merged_name = None

    def load_all_adapters(self) -> bool:
        """Load all adapters."""
        try:
            adapter_names = list(self.adapter_paths.keys())
            if not adapter_names:
                logger.error("没有可用的adapter路径")
                return False

            first_name = adapter_names[0]
            first_path = Path(self.adapter_paths[first_name])
            cleaned_path = _clean_adapter_config(first_path)
            if cleaned_path != first_path:
                self._temp_dirs.append(cleaned_path)

            logger.info(f"加载第一个adapter: {first_name} <- {first_path}")
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                str(cleaned_path),
                adapter_name=first_name,
                is_trainable=False,
            )

            for name in adapter_names[1:]:
                adapter_path = Path(self.adapter_paths[name])
                cleaned_path = _clean_adapter_config(adapter_path)
                if cleaned_path != adapter_path:
                    self._temp_dirs.append(cleaned_path)

                logger.info(f"加载adapter: {name} <- {adapter_path}")
                self.peft_model.load_adapter(
                    str(cleaned_path),
                    adapter_name=name,
                )

            self._adapters_loaded = True
            logger.info(f"全部 {len(adapter_names)} 个adapter加载完成")
            return True

        except Exception as e:
            logger.error(f"加载adapter失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def merge_adapters(self, weights: Dict[str, float], merged_name: str = "merged") -> bool:
        """Merge weighted LoRA adapters."""
        if not self._adapters_loaded:
            logger.error("adapter未加载，请先调用 load_all_adapters()")
            return False

        try:
            if self._current_merged_name is not None:
                try:
                    self.peft_model.delete_adapter(self._current_merged_name)
                except Exception:
                    pass

            adapter_names = list(weights.keys())
            adapter_weights = [weights[name] for name in adapter_names]

            logger.info(f"融合adapter: {dict(zip(adapter_names, adapter_weights))}")

            self.peft_model.add_weighted_adapter(
                adapters=adapter_names,
                weights=adapter_weights,
                adapter_name=merged_name,
                combination_type="linear",
            )

            self.peft_model.set_adapter(merged_name)
            self._current_merged_name = merged_name
            logger.info(f"adapter融合完成，已切换到: {merged_name}")
            return True

        except Exception as e:
            logger.error(f"adapter融合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def set_single_adapter(self, adapter_name: str) -> bool:
        """Activate one LoRA adapter."""
        if not self._adapters_loaded:
            logger.error("adapter未加载")
            return False

        try:
            self.peft_model.set_adapter(adapter_name)
            logger.info(f"已切换到单adapter: {adapter_name}")
            return True
        except Exception as e:
            logger.error(f"切换adapter失败: {e}")
            return False

    def cleanup(self):
        """Release temporary resources."""
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
        self._temp_dirs.clear()

        self._current_merged_name = None
        self._adapters_loaded = False

    def __del__(self):
        """Release owned resources."""
        try:
            self.cleanup()
        except Exception:
            pass


def build_type_aware_weights(
    data_type: str,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """Build type aware weights."""
    type_to_expert = {
        'text': 'text_expert',
        'image': 'image_expert',
        'uml': 'uml_expert',
    }

    specialized_expert = type_to_expert.get(data_type)
    if specialized_expert is None:
        logger.warning(f"未知数据类型: {data_type}，使用纯general权重")
        return {'general_expert': 1.0}

    return {
        specialized_expert: alpha,
        'general_expert': 1.0 - alpha,
    }


def group_general_samples_by_type(
    test_data: List[Dict],
) -> Dict[str, List[int]]:
    """Group general samples by input type."""
    groups = {}
    for idx, sample in enumerate(test_data):
        dt = sample.get('data_type', 'unknown')
        if dt not in groups:
            groups[dt] = []
        groups[dt].append(idx)

    for dt, indices in groups.items():
        logger.info(f"  General样本分组: {dt} -> {len(indices)} 条")

    return groups
