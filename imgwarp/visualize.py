import argparse
import cv2
import os
from pathlib import Path
import yaml
from glob import glob

# YOLO normalized [cls x y w h] -> pixel [x1 y1 x2 y2]
def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    cx, cy = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    x1 = max(0, int(round(cx - bw / 2)))
    y1 = max(0, int(round(cy - bh / 2)))
    x2 = min(img_w - 1, int(round(cx + bw / 2)))
    y2 = min(img_h - 1, int(round(cy + bh / 2)))
    return x1, y1, x2, y2

def load_class_names(dataset_yaml):
    with open(dataset_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    # names can be dict or list depending on yaml
    if isinstance(names, dict):
        # convert to list ordered by key index
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names

def color_for_class(idx):
    # deterministic distinct-ish colors
    palette = [
        (255, 99, 71),    # tomato
        (30, 144, 255),   # dodger blue
        (60, 179, 113),   # medium sea green
        (255, 215, 0),    # gold
        (186, 85, 211),   # medium orchid
        (244, 164, 96),   # sandy brown
        (0, 206, 209),    # dark turquoise
        (255, 105, 180),  # hot pink
        (154, 205, 50),   # yellow green
        (70, 130, 180),   # steel blue
    ]
    return palette[idx % len(palette)]

def draw_label(img, x1, y1, x2, y2, text, color, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # text background
    font_scale = max(0.4, min(1.0, img.shape[1] / 1600.0))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    th = int(th * 1.4)
    y_top = max(0, y1 - th - 2)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y_top + th + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, y_top + th - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def find_label_path(image_path, root_images, root_labels):
    # Replace root_images with root_labels and image extension with .txt
    rel = Path(image_path).relative_to(root_images)
    label_rel = rel.with_suffix(".txt")
    return Path(root_labels) / label_rel

def collect_images(root_images):
    image_paths = []
    
    # Convert to Path object for easier handling
    path = Path(root_images)
    
    if not path.exists():
        print(f"Directory {root_images} does not exist")
        return []

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'} 
    # Iterate through all files in the directory
    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
    
    return sorted(image_paths)

def visualize(dataset_yaml, out_dir, subset=None, thickness=2, draw_missing=False):
    names = load_class_names(dataset_yaml)
    with open(dataset_yaml, "r") as f:
        data = yaml.safe_load(f)
    root = Path(data["path"])
    root_images = root / data["train"] if subset is None else root / data[subset]
    # Determine images and labels roots robustly
    # If dataset follows standard YOLOv5 layout:
    #   images/train, labels/train; images/val, labels/val
    # We'll derive labels root by swapping "images" -> "labels" in the path.
    if "images" in str(root_images):
        root_labels = Path(str(root_images).replace(os.sep + "images", os.sep + "labels"))
    else:
        # fallback: root/labels
        root_labels = root / "labels"

    imgs = collect_images(root_images)
    if not imgs:
        print(f"No images found under {root_images} (subset={subset}).")
        return

    out_dir = Path(out_dir)
    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue
        H, W = img.shape[:2]
        label_path = find_label_path(img_path, root_images, root_labels)
        vis = img.copy()

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                conf = None
                if len(parts) >= 6:
                    try:
                        conf = float(parts[5])
                    except:
                        conf = None
                x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, W, H)
                cname = names[cls] if names and 0 <= cls < len(names) else f"class_{cls}"
                label_text = f"{cname}" + (f" {conf:.2f}" if conf is not None else "")
                color = color_for_class(cls)
                draw_label(vis, x1, y1, x2, y2, label_text, color, thickness=thickness)
        else:
            if draw_missing:
                cv2.putText(vis, "No label file", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # write output preserving relative folder structure under out_dir
        rel = Path(img_path).relative_to(root_images)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

    print(f"Done. Visualizations saved under: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLOv5 labels on images.")
    parser.add_argument("--dataset_yaml", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save visualized images")
    parser.add_argument("--subset", type=str, default=None, choices=[None, "train", "val", "test"],
                        help="Optional subset to visualize (train/val/test). Default: auto-detect all.")
    parser.add_argument("--thickness", type=int, default=2, help="Rectangle thickness")
    parser.add_argument("--draw_missing", action="store_true", help="Overlay 'No label file' when a label is missing")
    args = parser.parse_args()
    visualize(args.dataset_yaml, args.out_dir, subset=args.subset, thickness=args.thickness, draw_missing=args.draw_missing)
