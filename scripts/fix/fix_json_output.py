"""Repair malformed generated JSON output files."""

import json
import os

input_path = "outputs/recognition_results/uml/uml_recognition_qwen3_20260210_052354.json"
output_path = "outputs/recognition_results/uml/uml_recognition_qwen3_20260210_052354_fixed.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    for item in data:
        if 'description' in item and isinstance(item['description'], str):
            try:
                item['description'] = json.loads(item['description'])
            except json.JSONDecodeError as e:
                print(f"Failed to parse the description field (preserving the original string): {item.get('image_name', 'unknown')} - Error: {e}")
elif isinstance(data, dict):
    if 'description' in data and isinstance(data['description'], str):
        try:
            data['description'] = json.loads(data['description'])
        except json.JSONDecodeError as e:
            print(f"Failed to parse the top-level description: {e}")
else:
    raise ValueError("Unknown JSON structure: expected a list or dictionary")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Repair completed; saved to: {output_path}")
