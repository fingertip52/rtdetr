import os
import json
import argparse
from PIL import Image

# 注意：这里改成 0~9
CATEGORIES = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "people"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "van"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "tricycle"},
    {"id": 7, "name": "awning-tricycle"},
    {"id": 8, "name": "bus"},
    {"id": 9, "name": "motor"},
]

def convert(image_dir, label_dir, output_json):
    images = []
    annotations = []
    ann_id = 1

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    for img_id, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 8:
                    continue

                x, y, w, h = map(float, parts[:4])
                score = int(parts[4])       # 1: valid, 0: ignored
                cls_id = int(parts[5])      # 原始 VisDrone: 1~10
                trunc = int(parts[6])
                occ = int(parts[7])

                # 跳过 ignored region
                if score == 0:
                    continue

                # 跳过非法类别
                if cls_id < 1 or cls_id > 10:
                    continue

                # 改成 0~9
                cls_id = cls_id - 1

                x = max(0.0, x)
                y = max(0.0, y)
                w = max(0.0, w)
                h = max(0.0, h)

                if x + w > width:
                    w = width - x
                if y + h > height:
                    h = height - y

                if w <= 0 or h <= 0:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,   # 0~9
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "truncation": trunc,
                    "occlusion": occ,
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"Saved to {output_json}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(annotations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert(args.image_dir, args.label_dir, args.output)