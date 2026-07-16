"""Convert image inputs into structured JSON descriptions."""

from pathlib import Path
from typing import Dict, Optional
import time

from models.vision_model import VisionModel
from src.utils.logger import get_logger

logger = get_logger('preprocessing.image_to_json')

_vision_model = None


def get_vision_model() -> VisionModel:
    """Return vision model."""
    global _vision_model
    if _vision_model is None:
        logger.info("初始化视觉模型（Qwen3-VL-8B for image recognition）...")
        _vision_model = VisionModel(version='qwen3')
    return _vision_model


def convert_image_to_json(
    image_path: str,
    save_path: Optional[str] = None,
    return_processing_time: bool = True
) -> Dict:
    """Convert image to JSON."""
    logger.info(f"处理图像: {Path(image_path).name}")

    try:
        start_time = time.time()

        model = get_vision_model()
        result = model.recognize_image(image_path)

        if return_processing_time:
            result['processing_time'] = round(time.time() - start_time, 2)

        if save_path:
            from src.utils.file_utils import save_json
            save_json(result, save_path)
            logger.info(f"结果已保存至: {save_path}")

        return result

    except Exception as e:
        logger.error(f"图像处理失败: {e}")
        return {
            "description": "",
            "details": {"objects": [], "scene": "unknown", "spatial_info": ""},
            "confidence": 0.0,
            "recognition_status": "failed",
            "error": str(e)
        }


def batch_convert_images(
    image_paths: list,
    output_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Dict:
    """Convert images in batches."""
    results = []
    success = 0
    failed = 0

    for idx, img_path in enumerate(image_paths, 1):
        result = convert_image_to_json(
            img_path,
            save_path=f"{output_dir}/{Path(img_path).stem}.json" if output_dir else None
        )

        if result['recognition_status'] == 'success':
            success += 1
        else:
            failed += 1

        results.append(result)

        if progress_callback:
            progress_callback(idx, len(image_paths), result)

    return {
        'success': success,
        'failed': failed,
        'total': len(image_paths),
        'results': results
    }
