"""Build the evaluation manifest (barcode_to_images_map.json) from the extracted dataset.

Run from the repository root. Stores image paths relative to
mpr_dataset.path_to_images from src/data/paths.yaml.

The tracked manifest was converted from the original evaluation run by
stripping the dataset-root prefix, preserving the original path ordering.
Regenerating with this script yields the same path set per barcode, but
glob enumeration order may differ across filesystems (sorted here for
determinism). Ordering has no effect on the metrics.
"""

import json
import os
from glob import glob
from tqdm import tqdm
import yaml

with open("src/data/paths.yaml", "r") as f:
    datasets_paths = yaml.safe_load(f)

mpr_dataset = datasets_paths.get("mpr_dataset", {})

if mpr_dataset is None:
    raise ValueError("No mpr_dataset found in paths.yaml")

path_to_images = mpr_dataset.get("path_to_images", None)

single_frame_front_view_path = os.path.join(path_to_images, "single_frame_front_view")
if not os.path.exists(single_frame_front_view_path):
    raise ValueError(f"Path {single_frame_front_view_path} does not exist")
single_frame_front_drop_path = os.path.join(path_to_images, "single_frame_front_drop")
if not os.path.exists(single_frame_front_drop_path):
    raise ValueError(f"Path {single_frame_front_drop_path} does not exist")

synthetic_dataset = datasets_paths.get("synthetic", {})
if synthetic_dataset is None:
    raise ValueError("No synthetic dataset found in paths.yaml")
path_to_synthetic_labels = synthetic_dataset.get("path_to_synthetic_labels", None)
if path_to_synthetic_labels is None or not os.path.exists(path_to_synthetic_labels):
    raise ValueError(f"Path {path_to_synthetic_labels} does not exist")

with open(path_to_synthetic_labels, "r") as f:
    synthetic_labels = json.load(f)

barcode_to_image_filepaths = {}

for label in synthetic_labels:
    barcode_to_image_filepaths[label["product_id"]] = []

single_frame_front_view_images = os.listdir(single_frame_front_view_path)
single_frame_front_drop_images = os.listdir(single_frame_front_drop_path)


no_matching_files_in_front_view = []

barcodes = list(barcode_to_image_filepaths.keys())

for barcode in tqdm(barcodes, desc="Mapping barcodes to images in front view"):
    matching_files = sorted(glob(os.path.join(single_frame_front_view_path, f"*{barcode}*")))
    if matching_files:
        barcode_to_image_filepaths[barcode].extend(
            os.path.relpath(p, path_to_images) for p in matching_files)
    else:
        no_matching_files_in_front_view.append(barcode)

for barcode in tqdm(no_matching_files_in_front_view, desc="Mapping remaining barcodes to front drop images"):
    matching_files = sorted(glob(os.path.join(single_frame_front_drop_path, f"*{barcode}*")))
    if matching_files:
        barcode_to_image_filepaths[barcode].extend(
            os.path.relpath(p, path_to_images) for p in matching_files)
    else:
        raise FileNotFoundError(f"No matching files found for barcode {barcode} in front drop images")

print(f"Total barcodes with at least one image: {len(barcode_to_image_filepaths):,}")

extracted_image_filepaths = list(barcode_to_image_filepaths.values())
total_images = sum(len(filepaths) for filepaths in extracted_image_filepaths)
print(f"Total images found: {total_images:,}")

average_images_per_barcode = total_images / len(barcode_to_image_filepaths) if barcode_to_image_filepaths else 0
print(f"Average images per barcode: {average_images_per_barcode:.2f}")


# Export statistics
# Show which barcode (label) has the most images by sorting
# the barcode_to_image_filepaths
data_statistics = {}
# Key: barcode, Values: 'label' (str), 'num_images' (int)

# synthetic_label is a list that follows {'product_id': str, 'label': str} per item
# Change this to a dictionary for easier access
synthetic_label_dict = {item['product_id']: item['label'] for item in synthetic_labels}

for barcode, filepaths in barcode_to_image_filepaths.items():
    data_statistics[barcode] = {
        "barcode": barcode,
        "label": synthetic_label_dict.get(barcode, ""),
        "num_images": len(filepaths)
    }

sorted_data_statistics = dict(sorted(data_statistics.items(), key=lambda item: item[1]['num_images'], reverse=True))

with open("src/data/barcode_to_images_statistics.json", "w") as f:
    json.dump(sorted_data_statistics, f, indent=4)

print("Data statistics saved to 'src/data/barcode_to_images_statistics.json'")
print("Showing data statistics:")
for barcode, stats in sorted_data_statistics.items():
    print(f"Barcode: {barcode}, Label: {stats['label']}, Number of Images: {stats['num_images']}")

eval_dataset = {}

for barcode, filepaths in barcode_to_image_filepaths.items():
    eval_dataset[barcode] = {
        "image_paths": filepaths,
        "label": synthetic_label_dict.get(barcode, "")
    }

with open("src/data/barcode_to_images_map.json", "w") as f:
    json.dump(eval_dataset, f, indent=4)
