import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor

from src.core import YAMLConfig


# 你当前这版：category_id 已经是 0~9，且 remap_mscoco_category=False
# 0 行人
# 1 人群
# 2 自行车
# 3 汽车
# 4 面包车
# 5 卡车
# 6 三轮车
# 7 篷车
# 8 公交车
# 9 摩托车
CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
CLASS_COLORS = {
    0: (255, 0, 0),        # red
    1: (255, 165, 0),      # orange
    2: (255, 255, 0),      # yellow
    3: (0, 255, 0),        # lime
    4: (0, 255, 255),      # cyan
    5: (0, 0, 255),        # blue
    6: (128, 0, 128),      # purple
    7: (255, 0, 255),      # magenta
    8: (165, 42, 42),      # brown
    9: (255, 192, 203),    # pink
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_default_font():
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def collect_images(input_path):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"输入文件不是支持的图片格式: {input_path}")
        return [p]

    files = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            files.append(f)

    return files


def load_model(config_path, ckpt_path, device):
    cfg = YAMLConfig(config_path, resume=ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 官方 export_onnx.py 的加载逻辑：优先 ema
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("Loaded EMA weights")
    elif "model" in checkpoint:
        state = checkpoint["model"]
        print("Loaded model weights")
    else:
        raise KeyError("checkpoint 中既没有 'ema' 也没有 'model'")

    cfg.model.load_state_dict(state)

    model = cfg.model.deploy().to(device)
    model.eval()

    postprocessor = cfg.postprocessor.deploy()
    if hasattr(postprocessor, "to"):
        postprocessor = postprocessor.to(device)
    if hasattr(postprocessor, "eval"):
        postprocessor.eval()

    return model, postprocessor


@torch.no_grad()
def infer_one(
    model,
    postprocessor,
    image_path,
    device,
    conf_thres=0.35,
    input_size=640,
):
    original_im = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_im.size

    resized_im = original_im.resize((input_size, input_size))
    im_tensor = ToTensor()(resized_im).unsqueeze(0).to(device)

    target_sizes = torch.tensor(
        [[input_size, input_size]], dtype=torch.float32, device=device
    )

    outputs = model(im_tensor)
    labels, boxes, scores = postprocessor(outputs, target_sizes)

    labels = labels[0].detach().cpu()
    boxes = boxes[0].detach().cpu()
    scores = scores[0].detach().cpu()

    keep = scores > conf_thres
    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    # 从 input_size 映射回原图
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= float(orig_w) / float(input_size)
        boxes[:, [1, 3]] *= float(orig_h) / float(input_size)

        # 夹到图像范围内，避免出现负数或越界
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h - 1)

    results = []
    for lab, box, score in zip(labels, boxes, scores):
        lab = int(lab.item())
        score = float(score.item())
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]

        # 保证坐标顺序正确
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 跳过退化框
        if x2 <= x1 or y2 <= y1:
            continue

        results.append({
            "label": lab,
            "class_name": CLASS_NAMES.get(lab, f"class_{lab}"),
            "score": score,
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    return original_im, results


def draw_results(image, results, save_path, line_width=2):
    draw = ImageDraw.Draw(image)
    font = get_default_font()
    img_w, img_h = image.size

    for obj in results:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        cls_id = obj["label"]
        class_name = obj["class_name"]
        score = obj["score"]

        color = CLASS_COLORS.get(cls_id, "red")

        # 再做一次保险裁剪
        x1 = max(0, min(float(x1), img_w - 1))
        y1 = max(0, min(float(y1), img_h - 1))
        x2 = max(0, min(float(x2), img_w - 1))
        y2 = max(0, min(float(y2), img_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        text = f"{class_name} {score:.2f}"

        # 画目标框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 计算文字尺寸
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top
        except Exception:
            text_w, text_h = draw.textsize(text, font=font)

        pad = 2

        # 优先放在框上方
        text_x0 = x1
        text_y1 = y1 - 1
        text_y0 = text_y1 - text_h - 2 * pad

        if text_y0 < 0:
            text_y0 = y1 + 1
            text_y1 = text_y0 + text_h + 2 * pad

        text_x1 = min(text_x0 + text_w + 2 * pad, img_w - 1)
        text_x0 = max(0, text_x1 - (text_w + 2 * pad))

        if text_y1 >= img_h:
            text_y1 = img_h - 1
            text_y0 = max(0, text_y1 - (text_h + 2 * pad))

        if text_x1 > text_x0 and text_y1 > text_y0:
            draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill=color)
            draw.text(
                (text_x0 + pad, text_y0 + pad),
                text,
                fill="black" if color in ["yellow", "white", "lime", "cyan"] else "white",
                font=font
            )

    image.save(save_path)


def save_json(results, image_path, json_path):
    data = {
        "image_path": str(image_path),
        "detections": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="配置文件路径")
    parser.add_argument("-r", "--resume", required=True, type=str, help="checkpoint 路径")
    parser.add_argument("-i", "--input", required=True, type=str, help="单张图片或图片文件夹")
    parser.add_argument("-o", "--output-dir", default="vis_outputs", type=str, help="输出目录")
    parser.add_argument("--device", default="cuda:0", type=str, help="例如 cuda:0 或 cpu")
    parser.add_argument("--conf-thres", default=0.35, type=float, help="置信度阈值")
    parser.add_argument("--input-size", default=640, type=int, help="推理输入尺寸")
    parser.add_argument("--save-json", action="store_true", help="是否同时保存每张图的检测结果 JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model from: {args.resume}")

    model, postprocessor = load_model(args.config, args.resume, device)
    image_list = collect_images(args.input)

    if len(image_list) == 0:
        raise FileNotFoundError(f"在 {args.input} 中没有找到图片")

    print(f"Found {len(image_list)} image(s)")

    for img_path in image_list:
        vis_im, results = infer_one(
            model=model,
            postprocessor=postprocessor,
            image_path=str(img_path),
            device=device,
            conf_thres=args.conf_thres,
            input_size=args.input_size,
        )

        save_img_path = os.path.join(args.output_dir, img_path.name)
        draw_results(vis_im, results, save_img_path)

        if args.save_json:
            json_name = img_path.stem + ".json"
            save_json_path = os.path.join(args.output_dir, json_name)
            save_json(results, img_path, save_json_path)

        print(f"[OK] {img_path.name} -> {save_img_path}, dets={len(results)}")

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()