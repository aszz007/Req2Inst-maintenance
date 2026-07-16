"""Recognize raw UML diagrams and write interim records."""

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


def extract_metadata_from_path(image_path: Path) -> dict:
    """Extract metadata from path."""
    metadata = {
        'domain': 'unknown',
        'complexity': 'unknown'
    }

    try:
        parent_dir = image_path.parent.name
        metadata['domain'] = parent_dir

        filename = image_path.stem
        parts = filename.split('_')

        if len(parts) >= 2:
            complexity = parts[1]
            if complexity in ['simple', 'medium', 'complex']:
                metadata['complexity'] = complexity
    except Exception:
        pass

    return metadata


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='批量识别UML用例图')
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
        help='输出JSON文件路径（默认输出到outputs/recognition_results/uml/）'
    )
    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='单张图片路径（用于快速测试）'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='启用流式输出模式（实时显示生成内容）'
    )
    return parser.parse_args()


def recognize_single_uml(image_path: str, version: str = 'qwen3', streaming: bool = False) -> Dict:
    """Recognize single UML."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"\n{'='*80}")
    print(f"单图识别 - UML用例图")
    print(f"{'='*80}")
    print(f"模型版本: {version.upper()}")
    print(f"图片路径: {image_path}")
    print(f"{'='*80}\n")

    print(f"[模型加载] 正在加载 {version.upper()} 视觉模型...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[模型信息] {model_info['model_name']}")
    print(f"[设备] {model_info['device']}\n")

    print(f"[识别中] 正在处理UML图...")
    result = model.recognize_uml(str(image_path), streaming=streaming)

    result['image_path'] = str(image_path)
    result['image_name'] = image_path.name
    result['model_version'] = version

    metadata = extract_metadata_from_path(image_path)
    result['domain'] = metadata['domain']
    result['complexity'] = metadata['complexity']

    print(f"\n{'='*80}")
    print(f"识别结果")
    print(f"{'='*80}")
    if result.get('success', False):
        print(f"识别成功")

        try:
            desc = json.loads(result['description'])
            print(f"\n参与者数量: {len(desc.get('actors', []))}")
            print(f"用例数量: {len(desc.get('use_cases', []))}")
            print(f"关系数量: {len(desc.get('relationships', []))}")

            if desc.get('actors'):
                print(f"\n参与者: {', '.join(desc.get('actors', []))}")
            if desc.get('use_cases'):
                use_cases = [uc.get('name', '') for uc in desc.get('use_cases', [])]
                print(f"用例: {', '.join(use_cases[:3])}{'...' if len(use_cases) > 3 else ''}")
        except:
            print(f"\n描述: {result.get('description', '')[:200]}...")
    else:
        print(f"识别失败: {result.get('error', '未知错误')}")

    print(f"{'='*80}\n")

    return result


def batch_recognize_uml(
    image_folder: str,
    version: str = 'qwen3',
    output_file: str = None,
    streaming: bool = False
) -> List[Dict]:
    """Recognize UML."""
    image_folder = Path(image_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = set()
    for ext in image_extensions:
        image_files.update(image_folder.rglob(f"*{ext}"))
        image_files.update(image_folder.rglob(f"*{ext.upper()}"))

    image_files = sorted(list(image_files))
    total_images = len(image_files)

    print(f"\n{'='*80}")
    print(f"批量识别UML用例图")
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
            result = model.recognize_uml(str(image_path), streaming=streaming)

            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['model_version'] = version

            metadata = extract_metadata_from_path(image_path)
            result['domain'] = metadata['domain']
            result['complexity'] = metadata['complexity']

            results.append(result)

            if result.get('success', False):
                success_count += 1
                print(f"✓ 识别成功")

                try:
                    desc = json.loads(result['description'])
                    print(f"  参与者数量: {len(desc.get('actors', []))}")
                    print(f"  用例数量: {len(desc.get('use_cases', []))}")
                    print(f"  关系数量: {len(desc.get('relationships', []))}")
                except:
                    pass
            else:
                fail_count += 1
                print(f"✗ 识别失败: {result.get('error', '未知错误')}")

        except Exception as e:
            fail_count += 1
            print(f"✗ 处理失败: {str(e)}")

            metadata = extract_metadata_from_path(image_path)

            results.append({
                'image_path': str(image_path),
                'image_name': image_path.name,
                'model_version': version,
                'domain': metadata['domain'],
                'complexity': metadata['complexity'],
                'success': False,
                'error': str(e)
            })

    if output_file is None:
        path_cfg = get_path_config()
        output_dir = path_cfg.UML_RECOGNITION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"uml_recognition_{version}_{timestamp}.json"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"统计报告")
    print(f"{'='*80}")

    domain_stats = {}
    overall_description_complete = 0
    overall_description_incomplete = 0

    for result in results:
        domain = result.get('domain', 'unknown')
        if domain not in domain_stats:
            domain_stats[domain] = {'total': 0, 'success': 0, 'failed': 0}

        domain_stats[domain]['total'] += 1
        if result.get('success', False):
            domain_stats[domain]['success'] += 1

            try:
                desc = json.loads(result['description'])
                if 'overall_description' in desc and desc['overall_description']:
                    overall_description_complete += 1
                else:
                    overall_description_incomplete += 1
                    print(f"[警告] {result['image_name']} - overall_description缺失或为空")
            except:
                overall_description_incomplete += 1
        else:
            domain_stats[domain]['failed'] += 1

    print(f"\n按领域统计:")
    print(f"{'领域':<25} {'总数':>8} {'成功':>8} {'失败':>8} {'成功率':>10}")
    print("-" * 70)
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{domain:<25} {stats['total']:>8} {stats['success']:>8} {stats['failed']:>8} {success_rate:>9.1f}%")

    print(f"\noverall_description完整性检查:")
    print(f"  完整: {overall_description_complete}")
    print(f"  不完整/缺失: {overall_description_incomplete}")
    if overall_description_incomplete > 0:
        print(f"  [提示] 如果不完整率较高，可能需要增加max_new_tokens参数")

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
    print(" " * 25 + f"UML用例图识别系统")
    print("=" * 80)
    print(f"模型版本: {args.version.upper()}")
    print(f"功能: 识别用例图中的元素和逻辑关系")
    print(f"输出: 英文JSON格式结果")
    print("=" * 80 + "\n")

    try:
        if args.single:
            result = recognize_single_uml(args.single, args.version, args.streaming)

            path_cfg = get_path_config()
            output_dir = path_cfg.UML_RECOGNITION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"single_uml_{timestamp}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"结果已保存至: {output_file}")

        else:
            if args.input:
                image_folder = args.input
            else:
                path_cfg = get_path_config()
                image_folder = path_cfg.PLANT_UML_DIR
                print(f"[提示] 使用默认输入目录: {image_folder}")
                print(f"[提示] 可使用 --input 参数指定其他目录\n")

            results = batch_recognize_uml(
                image_folder=image_folder,
                version=args.version,
                output_file=args.output,
                streaming=args.streaming
            )

            if results and results[0].get('success', False):
                print("\n" + "="*80)
                print("结果示例（第一张图片）")
                print("="*80)
                first_result = results[0]
                sample = {
                    'image_name': first_result.get('image_name'),
                    'model_version': first_result.get('model_version'),
                    'success': first_result.get('success'),
                    'description_preview': first_result.get('description', '')[:200] + "..."
                }
                print(json.dumps(sample, ensure_ascii=False, indent=2))
                print("="*80)

        print("\n所有识别任务完成")

    except Exception as e:
        print(f"\n程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
