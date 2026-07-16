"""Convert UML inputs into structured JSON descriptions."""

from pathlib import Path
from typing import Dict, Optional
import time

from models.vision_model import VisionModel
from src.utils.logger import get_logger

logger = get_logger('preprocessing.uml_to_json')

_vision_model = None


def get_vision_model() -> VisionModel:
    """Return vision model."""
    global _vision_model
    if _vision_model is None:
        logger.info("初始化视觉模型（Qwen3-VL-8B for UML recognition）...")
        _vision_model = VisionModel(version='qwen3')
    return _vision_model


def convert_uml_to_json(
    uml_path: str,
    save_path: Optional[str] = None,
    max_retries: int = 2
) -> Dict:
    """Convert UML to JSON."""
    logger.info(f"处理UML图: {Path(uml_path).name}")

    try:
        start_time = time.time()

        model = get_vision_model()
        result = model.recognize_uml(uml_path, max_retries=max_retries)

        processing_time = round(time.time() - start_time, 2)

        if result.get('success', False):
            output_data = {
                "description": result['description'],
                "processing_time": processing_time,
                "recognition_status": "success"
            }
        else:
            output_data = {
                "description": "",
                "processing_time": processing_time,
                "recognition_status": "failed",
                "error": result.get('error', '未知错误')
            }

        if save_path:
            from src.utils.file_utils import save_json
            save_json({"description": output_data["description"]}, save_path)
            logger.info(f"结果已保存至: {save_path}")

        return output_data

    except Exception as e:
        logger.error(f"UML处理失败: {e}")
        return {
            "description": "",
            "recognition_status": "failed",
            "error": str(e)
        }


def batch_convert_umls(
    uml_paths: list,
    output_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Dict:
    """Convert UML inputs in batches."""
    results = []
    success = 0
    failed = 0

    for idx, uml_path in enumerate(uml_paths, 1):
        result = convert_uml_to_json(
            uml_path,
            save_path=f"{output_dir}/{Path(uml_path).stem}.json" if output_dir else None
        )

        if result['recognition_status'] == 'success':
            success += 1
        else:
            failed += 1

        results.append(result)

        if progress_callback:
            progress_callback(idx, len(uml_paths), result)

    return {
        'success': success,
        'failed': failed,
        'total': len(uml_paths),
        'results': results
    }
