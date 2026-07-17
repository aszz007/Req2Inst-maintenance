"""Match image records and build the interim image dataset."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime


class ImageDatasetBuilder:
    """Build the interim image dataset from matched records."""

    def __init__(self):
        """Initialize the instance."""
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        self.errors = []
        self.stats = {
            'total_images': 0,
            'total_json_entries': 0,
            'matched': 0,
            'images_without_json': 0,
            'json_without_images': 0,
            'failed_recognitions': 0
        }

    def get_all_images(self, image_folder: Path) -> Set[str]:
        """Return all images."""
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        image_names = set()

        for ext in self.image_extensions:
            for file_path in image_folder.glob(f"*{ext}"):
                image_names.add(file_path.stem)
            for file_path in image_folder.glob(f"*{ext.upper()}"):
                image_names.add(file_path.stem)

        self.stats['total_images'] = len(image_names)
        return image_names

    def load_json_results(self, json_file: Path) -> Dict[str, Dict]:
        """Load JSON results."""
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        json_mapping = {}

        for entry in results:
            image_name = entry.get('image_name', '')
            if not image_name:
                self.errors.append({
                    'type': 'INVALID_JSON_ENTRY',
                    'message': 'JSON条目缺少image_name',
                    'entry': entry
                })
                continue

            name_without_ext = Path(image_name).stem

            recognition_status = entry.get('recognition_status', '')
            success_flag = entry.get('success', False)

            is_failed = (recognition_status != 'success' and not success_flag)

            if is_failed:
                self.stats['failed_recognitions'] += 1
                error_msg = entry.get('error', f'Recognition status: {recognition_status}' if recognition_status else 'Unknown error')
                self.errors.append({
                    'type': 'FAILED_RECOGNITION',
                    'image_name': name_without_ext,
                    'error': error_msg
                })

            json_mapping[name_without_ext] = entry

        self.stats['total_json_entries'] = len(json_mapping)
        return json_mapping

    def validate_mapping(self, image_names: Set[str], json_mapping: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
        """Validate mapping."""
        images_without_json = list(image_names - json_mapping.keys())

        json_without_images = list(json_mapping.keys() - image_names)

        self.stats['images_without_json'] = len(images_without_json)
        self.stats['json_without_images'] = len(json_without_images)
        self.stats['matched'] = len(image_names & json_mapping.keys())

        for img_name in images_without_json:
            self.errors.append({
                'type': 'IMAGE_WITHOUT_JSON',
                'image_name': img_name,
                'message': '图片存在但没有JSON识别结果'
            })

        for json_name in json_without_images:
            self.errors.append({
                'type': 'JSON_WITHOUT_IMAGE',
                'image_name': json_name,
                'message': 'JSON结果存在但没有对应图片'
            })

        return images_without_json, json_without_images

    def prepare_json_string(self, recognition_info: Dict) -> str:
        """Prepare JSON string."""
        recognition_status = recognition_info.get('recognition_status', '')
        success_flag = recognition_info.get('success', False)

        is_success = (recognition_status == 'success' or success_flag)

        if not is_success:
            error_msg = recognition_info.get('error', f'Recognition status: {recognition_status}' if recognition_status else 'Unknown error')
            return json.dumps({
                'recognition_status': 'failed',
                'error': error_msg
            }, ensure_ascii=False)

        try:
            metadata_fields = {
                'confidence',
                'recognition_status',
                'image_path',
                'image_name',
                'model_version',
                'success',
                'timestamp',
                'error'
            }

            clean_info = {}
            for key, value in recognition_info.items():
                if key not in metadata_fields:
                    clean_info[key] = value

            if not clean_info:
                clean_info = {'description': recognition_info.get('description', '')}

            return json.dumps(clean_info, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                'recognition_status': 'error',
                'error': f'Unexpected error: {str(e)}',
                'raw_description': str(recognition_info.get('description', ''))[:200]
            }, ensure_ascii=False)

    def build_csv_dataset(
        self,
        image_folder: str,
        json_file: str,
        output_csv: str = None,
        error_log: str = None,
        include_failed: bool = True
    ) -> str:
        """Build CSV dataset."""
        print("="*80)
        print(" "*25 + "Image Dataset Builder")
        print("="*80)

        image_folder = Path(image_folder)
        json_file = Path(json_file)

        print("\n[Step 1/5] Scanning the image directory...")
        print(f"Image folder: {image_folder}")
        image_names = self.get_all_images(image_folder)
        print(f"Found {len(image_names)} unique images")

        print("\n[Step 2/5] Loading JSON recognition results...")
        print(f"JSON file: {json_file}")
        json_mapping = self.load_json_results(json_file)
        print(f"Loaded {len(json_mapping)} JSON records")
        if self.stats['failed_recognitions'] > 0:
            print(f"     {self.stats['failed_recognitions']} recognition(s) failed")

        print("\n[Step 3/5] Validating image-to-JSON mappings...")
        images_without_json, json_without_images = self.validate_mapping(image_names, json_mapping)

        print(f"Matched pairs: {self.stats['matched']}")
        if self.stats['images_without_json'] > 0:
            print(f"  Images missing JSON: {self.stats['images_without_json']}")
        if self.stats['json_without_images'] > 0:
            print(f"  JSON records missing images: {self.stats['json_without_images']}")

        print("\n[Step 4/5] Building the CSV dataset...")

        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"image_interim_{timestamp}.csv"

        output_csv = Path(output_csv)

        rows = []
        for image_name in sorted(image_names):
            if image_name in json_mapping:
                recognition_info = json_mapping[image_name]

                recognition_status = recognition_info.get('recognition_status', '')
                success_flag = recognition_info.get('success', False)
                is_success = (recognition_status == 'success' or success_flag)

                if not include_failed and not is_success:
                    continue

                header = image_name
                description = self.prepare_json_string(recognition_info)
                instruction = ""

                rows.append({
                    'Header': header,
                    'Description': description,
                    'Instruction': instruction
                })
            else:
                rows.append({
                    'Header': image_name,
                    'Description': json.dumps({
                        'recognition_status': 'missing',
                        'error': 'No JSON recognition result found'
                    }, ensure_ascii=False),
                    'Instruction': ""
                })

        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            fieldnames = ['Header', 'Description', 'Instruction']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV dataset created: {output_csv}")
        print(f"Total rows: {len(rows)}")

        print("\n[Step 5/5] Saving the error log...")

        if error_log is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log = f"image_dataset_errors_{timestamp}.json"

        error_log = Path(error_log)

        with open(error_log, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'errors': self.errors
            }, f, ensure_ascii=False, indent=2)

        print(f"Error log saved to: {error_log}")
        print(f"Total errors: {len(self.errors)}")

        print("\n" + "="*80)
        print(" "*30 + "Summary")
        print("="*80)
        print(f"Total images: {self.stats['total_images']}")
        print(f"Total JSON entries: {self.stats['total_json_entries']}")
        print(f"Matched pairs: {self.stats['matched']}")
        print(f"CSV rows created: {len(rows)}")
        print("-"*80)
        print(f"Images missing JSON: {self.stats['images_without_json']}")
        print(f"JSON records missing images: {self.stats['json_without_images']}")
        print(f"Recognition failures: {self.stats['failed_recognitions']}")
        print(f"Logged errors: {len(self.errors)}")
        print("="*80 + "\n")

        return str(output_csv)


def main():
    """Run the command-line entry point."""


    IMAGE_FOLDER = r"data/raw/image/coco_1k"

    JSON_FILE = r"outputs/recognition_results/image/image_recognition_qwen3_20260215_210034.json"

    OUTPUT_CSV = r"data/interim/image/image_interim_coco_1k.csv"

    ERROR_LOG = r"data/interim/image/image_dataset_errors.json"

    INCLUDE_FAILED = True


    print("="*80)
    print(" "*20 + "Image Dataset Builder")
    print("="*80)
    print(f"Purpose: Convert images and JSON results into a CSV dataset")
    print(f"Output format: Header | Description | Instruction")
    print(f"Description contents: All image recognition information (description, details, etc.)")
    print(f"Filtered fields: confidence, recognition_status, image_path, image_name, model_version")
    print("="*80 + "\n")

    try:
        builder = ImageDatasetBuilder()

        output_path = builder.build_csv_dataset(
            image_folder=IMAGE_FOLDER,
            json_file=JSON_FILE,
            output_csv=OUTPUT_CSV,
            error_log=ERROR_LOG,
            include_failed=INCLUDE_FAILED
        )

        print(f" Dataset build completed!")
        print(f" Output file: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Review the error log for any issues")
        print(f"2. Populate the 'Instruction' column as needed")
        print(f"3. Use the dataset for model training or other applications")

    except Exception as e:
        print(f"\n Dataset build failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
