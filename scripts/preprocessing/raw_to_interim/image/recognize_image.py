"""Recognize raw images and write interim records."""

import argparse
import json
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
    parser = argparse.ArgumentParser(description='Batch-recognize general images')
    parser.add_argument(
        '--version',
        type=str,
        default='qwen3',
        choices=['qwen3'],
        help='Select the vision model version (default: qwen3)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input image directory (uses the configured test directory by default)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON path (defaults to outputs/recognition_results/image/)'
    )
    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='Path to a single image (for quick testing)'
    )
    return parser.parse_args()


def recognize_single_image(image_path: str, version: str = 'qwen3') -> Dict:
    """Recognize single image."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"\n{'='*80}")
    print("Single-image recognition")
    print(f"{'='*80}")
    print(f"Model version: {version.upper()}")
    print(f"Image path: {image_path}")
    print(f"{'='*80}\n")

    print(f"[Model loading] Loading the {version.upper()} vision model...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[Model info] {model_info['model_name']}")
    print(f"[Device] {model_info['device']}\n")

    print("[Recognizing] Processing image...")
    result = model.recognize_image(str(image_path))

    result['image_path'] = str(image_path)
    result['image_name'] = image_path.name
    result['model_version'] = version

    print(f"\n{'='*80}")
    print("Recognition result")
    print(f"{'='*80}")
    if result.get('recognition_status') == 'success':
        print(" Recognition succeeded")
        print(f"Confidence: {result.get('confidence', 0):.3f}")
        print(f"\nDescription: {result.get('description', '')}")

        details = result.get('details', {})
        if details:
            print("\nDetails:")
            print(f"  Scene: {details.get('scene', 'unknown')}")
            print(f"  Objects: {', '.join(details.get('objects', []))}")
            if details.get('spatial_info'):
                print(f"  Spatial information: {details.get('spatial_info')}")
    else:
        print(f" Recognition failed: {result.get('error', 'Unknown error')}")

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
    print("Batch image recognition")
    print(f"{'='*80}")
    print(f"Model version: {version.upper()}")
    print(f"Image folder: {image_folder}")
    print(f"Images found: {total_images}")
    print(f"{'='*80}\n")

    if total_images == 0:
        print("[Warning] No image files found")
        return []

    print(f"[Model loading] Loading the {version.upper()} vision model...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[Model info] {model_info['model_name']}")
    print(f"[Device] {model_info['device']}\n")

    results = []
    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{total_images}] Processing: {image_path.name}")
        print("-" * 70)

        try:
            result = model.recognize_image(str(image_path))

            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['model_version'] = version

            results.append(result)

            if result.get('recognition_status') == 'success':
                success_count += 1
                print(" Recognition succeeded")
                print(f"  Confidence: {result.get('confidence', 0):.3f}")
                print(f"  Description: {result.get('description', '')[:80]}...")
            else:
                fail_count += 1
                print(f" Recognition failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            fail_count += 1
            print(f" Processing failed: {str(e)}")
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
    print("Batch recognition completed")
    print(f"{'='*80}")
    print(f"Total images: {total_images}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {success_count/total_images*100:.1f}%")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    return results


def main():
    """Run the command-line entry point."""
    args = parse_args()

    print("=" * 80)
    print(" " * 25 + "Image Recognition System")
    print("=" * 80)
    print(f"Model version: {args.version.upper()}")
    print("Function: Recognize image content and generate a structured description")
    print("Output: English JSON result")
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

            print(f"Results saved to: {output_file}")

        else:
            if args.input:
                image_folder = args.input
            else:
                path_cfg = get_path_config()
                image_folder = path_cfg.COCO_500_DIR
                print(f"[Info] Using the default input directory: {image_folder}")
                print("[Info] Use --input to specify another directory\n")

            results = batch_recognize_images(
                image_folder=image_folder,
                version=args.version,
                output_file=args.output
            )

            if results and results[0].get('recognition_status') == 'success':
                print("\n" + "="*80)
                print("Example result (first image)")
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

        print("\n All recognition tasks completed!")

    except Exception as e:
        print(f"\n Program execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
