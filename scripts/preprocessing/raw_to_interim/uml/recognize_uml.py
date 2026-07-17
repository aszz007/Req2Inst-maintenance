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
    parser = argparse.ArgumentParser(description='Batch-recognize UML use case diagrams')
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
        help='Output JSON path (defaults to outputs/recognition_results/uml/)'
    )
    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='Path to a single image (for quick testing)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Enable streaming output (show generated content in real time)'
    )
    return parser.parse_args()


def recognize_single_uml(image_path: str, version: str = 'qwen3', streaming: bool = False) -> Dict:
    """Recognize single UML."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"\n{'='*80}")
    print(f"Single-image recognition - UML use case diagram")
    print(f"{'='*80}")
    print(f"Model version: {version.upper()}")
    print(f"Image path: {image_path}")
    print(f"{'='*80}\n")

    print(f"[Model loading] Loading the {version.upper()} vision model...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[Model info] {model_info['model_name']}")
    print(f"[Device] {model_info['device']}\n")

    print(f"[Recognizing] Processing UML diagram...")
    result = model.recognize_uml(str(image_path), streaming=streaming)

    result['image_path'] = str(image_path)
    result['image_name'] = image_path.name
    result['model_version'] = version

    metadata = extract_metadata_from_path(image_path)
    result['domain'] = metadata['domain']
    result['complexity'] = metadata['complexity']

    print(f"\n{'='*80}")
    print(f"Recognition result")
    print(f"{'='*80}")
    if result.get('success', False):
        print(f"Recognition succeeded")

        try:
            desc = json.loads(result['description'])
            print(f"\nActor count: {len(desc.get('actors', []))}")
            print(f"Use case count: {len(desc.get('use_cases', []))}")
            print(f"Relationship count: {len(desc.get('relationships', []))}")

            if desc.get('actors'):
                print(f"\nActors: {', '.join(desc.get('actors', []))}")
            if desc.get('use_cases'):
                use_cases = [uc.get('name', '') for uc in desc.get('use_cases', [])]
                print(f"Use cases: {', '.join(use_cases[:3])}{'...' if len(use_cases) > 3 else ''}")
        except:
            print(f"\nDescription: {result.get('description', '')[:200]}...")
    else:
        print(f"Recognition failed: {result.get('error', 'Unknown error')}")

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
    print(f"Batch UML use case diagram recognition")
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
                print(f" Recognition succeeded")

                try:
                    desc = json.loads(result['description'])
                    print(f"  Actor count: {len(desc.get('actors', []))}")
                    print(f"  Use case count: {len(desc.get('use_cases', []))}")
                    print(f"  Relationship count: {len(desc.get('relationships', []))}")
                except:
                    pass
            else:
                fail_count += 1
                print(f" Recognition failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            fail_count += 1
            print(f" Processing failed: {str(e)}")

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
    print(f"Statistics report")
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
                    print(f"[Warning] {result['image_name']} - overall_description is missing or empty")
            except:
                overall_description_incomplete += 1
        else:
            domain_stats[domain]['failed'] += 1

    print(f"\nStatistics by domain:")
    print(f"{'Domain':<25} {'Total':>8} {'Succeeded':>8} {'Failed':>8} {'Success rate':>10}")
    print("-" * 70)
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{domain:<25} {stats['total']:>8} {stats['success']:>8} {stats['failed']:>8} {success_rate:>9.1f}%")

    print(f"\noverall_description completeness check:")
    print(f"  Complete: {overall_description_complete}")
    print(f"  Incomplete/missing: {overall_description_incomplete}")
    if overall_description_incomplete > 0:
        print(f"  [Info] A high incomplete rate may require increasing max_new_tokens")

    print(f"\n{'='*80}")
    print(f"Batch recognition completed")
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
    print(" " * 25 + f"UML Use Case Diagram Recognition System")
    print("=" * 80)
    print(f"Model version: {args.version.upper()}")
    print(f"Function: Recognize elements and logical relationships in use case diagrams")
    print(f"Output: English JSON result")
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

            print(f"Results saved to: {output_file}")

        else:
            if args.input:
                image_folder = args.input
            else:
                path_cfg = get_path_config()
                image_folder = path_cfg.PLANT_UML_DIR
                print(f"[Info] Using the default input directory: {image_folder}")
                print(f"[Info] Use --input to specify another directory\n")

            results = batch_recognize_uml(
                image_folder=image_folder,
                version=args.version,
                output_file=args.output,
                streaming=args.streaming
            )

            if results and results[0].get('success', False):
                print("\n" + "="*80)
                print("Example result (first image)")
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

        print("\nAll recognition tasks completed")

    except Exception as e:
        print(f"\nProgram execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
