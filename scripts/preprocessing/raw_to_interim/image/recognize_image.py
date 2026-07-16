"""Recognize raw images and write interim records."""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vision_model import VisionModel
from config.settings import get_path_config


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='批量识别一般图像')
    parser.add_argument(
        '--version',
        type=str,
        default='qwen3',
        choices=['qwen2.5', 'qwen3'],
        help='选择视觉模型版本（默认: qwen3）'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='输入图片文件夹路径（默认使用配置中的测试目录）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出JSON文件路径（默认输出到outputs/recognition_results/image/）'
    )
    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='单张图片路径（用于快速测试）'
    )
    return parser.parse_args()


def recognize_single_image(image_path: str, version: str = 'qwen3') -> Dict:
    """Recognize single image."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"\n{'='*80}")
    print(f"单图识别")
    print(f"{'='*80}")
    print(f"模型版本: {version.upper()}")
    print(f"图片路径: {image_path}")
    print(f"{'='*80}\n")

    print(f"[模型加载] 正在加载 {version.upper()} 视觉模型...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[模型信息] {model_info['model_name']}")
    print(f"[设备] {model_info['device']}\n")

    print(f"[识别中] 正在处理图片...")
    result = model.recognize_image(str(image_path))

    result['image_path'] = str(image_path)
    result['image_name'] = image_path.name
    result['model_version'] = version

    print(f"\n{'='*80}")
    print(f"识别结果")
    print(f"{'='*80}")
    if result.get('recognition_status') == 'success':
        print(f"✓ 识别成功")
        print(f"置信度: {result.get('confidence', 0):.3f}")
        print(f"\n描述: {result.get('description', '')}")

        details = result.get('details', {})
        if details:
            print(f"\n详细信息:")
            print(f"  场景: {details.get('scene', 'unknown')}")
            print(f"  对象: {', '.join(details.get('objects', []))}")
            if details.get('spatial_info'):
                print(f"  空间信息: {details.get('spatial_info')}")
    else:
        print(f"✗ 识别失败: {result.get('error', '未知错误')}")

    print(f"{'='*80}\n")

    return result


def batch_recognize_images(
    image_folder: str,
    version: str = 'qwen3',
    output_file: str = None
) -> List[Dict]:
    """Recognize images."""
    image_folder = Path(image_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = set()
    for ext in image_extensions:
        image_files.update(image_folder.glob(f"*{ext}"))
        image_files.update(image_folder.glob(f"*{ext.upper()}"))

    image_files = sorted(list(image_files))
    total_images = len(image_files)

    print(f"\n{'='*80}")
    print(f"批量识别图像")
    print(f"{'='*80}")
    print(f"模型版本: {version.upper()}")
    print(f"图片文件夹: {image_folder}")
    print(f"找到图片数量: {total_images}")
    print(f"{'='*80}\n")

    if total_images == 0:
        print("[警告] 未找到任何图片文件")
        return []

    print(f"[模型加载] 正在加载 {version.upper()} 视觉模型...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[模型信息] {model_info['model_name']}")
    print(f"[设备] {model_info['device']}\n")

    results = []
    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{total_images}] 处理: {image_path.name}")
        print("-" * 70)

        try:
            result = model.recognize_image(str(image_path))

            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['model_version'] = version

            results.append(result)

            if result.get('recognition_status') == 'success':
                success_count += 1
                print(f"✓ 识别成功")
                print(f"  置信度: {result.get('confidence', 0):.3f}")
                print(f"  描述: {result.get('description', '')[:80]}...")
            else:
                fail_count += 1
                print(f"✗ 识别失败: {result.get('error', '未知错误')}")

        except Exception as e:
            fail_count += 1
            print(f"✗ 处理失败: {str(e)}")
            results.append({
                'image_path': str(image_path),
                'image_name': image_path.name,
                'model_version': version,
                'recognition_status': 'failed',
                'error': str(e)
            })

    if output_file is None:
        path_cfg = get_path_config()
        output_dir = path_cfg.IMAGE_RECOGNITION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"image_recognition_{version}_{timestamp}.json"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"批量识别完成")
    print(f"{'='*80}")
    print(f"总图片数: {total_images}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"成功率: {success_count/total_images*100:.1f}%")
    print(f"结果已保存至: {output_file}")
    print(f"{'='*80}\n")

    return results


def main():
    """Run the command-line entry point."""
    args = parse_args()

    print("=" * 80)
    print(" " * 25 + f"图像识别系统")
    print("=" * 80)
    print(f"模型版本: {args.version.upper()}")
    print(f"功能: 识别图像内容并生成结构化描述")
    print(f"输出: 英文JSON格式结果")
    print("=" * 80 + "\n")

    try:
        if args.single:
            result = recognize_single_image(args.single, args.version)

            path_cfg = get_path_config()
            output_dir = path_cfg.IMAGE_RECOGNITION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"single_image_{timestamp}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"结果已保存至: {output_file}")

        else:
            if args.input:
                image_folder = args.input
            else:
                path_cfg = get_path_config()
                image_folder = path_cfg.COCO_500_DIR
                print(f"[提示] 使用默认输入目录: {image_folder}")
                print(f"[提示] 可使用 --input 参数指定其他目录\n")

            results = batch_recognize_images(
                image_folder=image_folder,
                version=args.version,
                output_file=args.output
            )

            if results and results[0].get('recognition_status') == 'success':
                print("\n" + "="*80)
                print("结果示例（第一张图片）")
                print("="*80)
                first_result = results[0]
                sample = {
                    'image_name': first_result.get('image_name'),
                    'model_version': first_result.get('model_version'),
                    'recognition_status': first_result.get('recognition_status'),
                    'confidence': first_result.get('confidence'),
                    'description': first_result.get('description', '')[:150] + "..."
                }
                print(json.dumps(sample, ensure_ascii=False, indent=2))
                print("="*80)

        print("\n✓ 所有识别任务完成！")

    except Exception as e:
        print(f"\n✗ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
