import cv2
import numpy as np
import os
import glob

# --- helpers ---

def parse_polygon_line(line, img_w, img_h):
    parts = list(map(float, line.strip().split()))
    label = int(parts[0])
    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= img_w
    coords[:, 1] *= img_h
    return label, coords

def force_quad(polygon):
    # Konveks hull al
    hull = cv2.convexHull(polygon)

    if len(hull) <= 4:
        return hull[:4].astype(np.float32)

    s = hull[:, 0, 0] + hull[:, 0, 1]  # x + y
    diff = hull[:, 0, 0] - hull[:, 0, 1]  # x - y

    top_left = hull[np.argmin(s)][0]
    bottom_right = hull[np.argmax(s)][0]
    top_right = hull[np.argmin(diff)][0]
    bottom_left = hull[np.argmax(diff)][0]

    quad = np.array([top_left, bottom_left, bottom_right,  top_right],
                    dtype=np.float32)
    return quad

def yolo_to_corners_px(line, W, H):
    # line: "cls cx cy w h" normalized
    cls, cx, cy, w, h = map(float, line.split()[:5])
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    # corners in order TL, TR, BR, BL (axis-aligned in source)
    return int(cls), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

def warp_points(pts_xy, M):
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)

def aabb_from_points(pts):
    x1 = float(np.min(pts[:,0])); y1 = float(np.min(pts[:,1]))
    x2 = float(np.max(pts[:,0])); y2 = float(np.max(pts[:,1]))
    return x1, y1, x2, y2

def xyxy_to_yolo_norm(x1, y1, x2, y2, W, H):
    # clip
    x1 = max(0.0, min(x1, W-1)); y1 = max(0.0, min(y1, H-1))
    x2 = max(0.0, min(x2, W-1)); y2 = max(0.0, min(y2, H-1))
    w = x2 - x1; h = y2 - y1
    if w <= 1e-3 or h <= 1e-3:
        return None
    cx = x1 + w/2.0; cy = y1 + h/2.0
    return cx/W, cy/H, w/W, h/H

# --- main warper ---

def warp_image_and_yolo_boxes(img, label_lines, src_quad_px, dst_size, classes_to_keep=None):
    """
    img: BGR image
    label_lines: list of 'cls cx cy w h' normalized (YOLO)
    src_quad_px: 4x2 float32 source polygon in pixel coords (order: TL, TR, BR, BL)
    dst_size: (dst_w, dst_h)  e.g., (800, 800)
    classes_to_keep: iterable of class ids to keep; None -> keep all
    """
    dst_w, dst_h = dst_size
    Hs, Ws = img.shape[:2]

    dst_quad = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(src_quad_px, np.float32), dst_quad)
    warped_img = cv2.warpPerspective(img, M, (dst_w, dst_h))

    new_lines = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if classes_to_keep is not None and cls not in classes_to_keep:
            continue

        _, corners_px = yolo_to_corners_px(line, Ws, Hs)
        warped_pts = warp_points(corners_px, M)          # (4,2) in warped image
        x1, y1, x2, y2 = aabb_from_points(warped_pts)    # axis-aligned bbox after warp
        yolo_vals = xyxy_to_yolo_norm(x1, y1, x2, y2, dst_w, dst_h)
        if yolo_vals is None:
            continue
        cx, cy, w, h = yolo_vals
        new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return warped_img, new_lines

if __name__ == "__main__":
    #input_dir, already_warped = "../../tatamovski/", False
    input_dir, already_warped = "../../warpedhata_detection/", True
    output_root_dir = "warpeddata/"
    dest_size = (800, 800)

    for split in ["train/", "valid/"]:
        image_dir = input_dir + split + "images"
        label_dir = input_dir + split + "labels"
        output_dir = output_root_dir + split
        for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
            base = os.path.splitext(os.path.basename(label_path))[0]
            image_path = os.path.join(image_dir, base + ".jpg")

            if not os.path.exists(image_path):
                print(f'Error! Image file not found that correspond to the label file: {label_path}')
                continue

            img = cv2.imread(image_path)
            with open(label_path, "r") as f:
                labels = [l.strip() for l in f if l.strip()]

            img_h, img_w = img.shape[:2]
            plank_quad = None
            bbox_lines = []
            for line in labels:
                if (not already_warped) and line[0] == "2":
                    _, plank_poly = parse_polygon_line(line, img_w, img_h)
                    plank_quad = force_quad(plank_poly)
                else:
                    bbox_lines.append("0" + line[1:])

            if already_warped:
                src_h, src_w = img.shape[:2]
                plank_quad = np.array([[0, 0], [src_w-1, 0], [src_w-1, src_h-1], [0, src_h-1]], dtype=np.float32)
            warped_img, new_labels = warp_image_and_yolo_boxes(img, bbox_lines, plank_quad, dest_size, classes_to_keep=None)

            # Sonuçları kaydet
            output_img_dir = os.path.join(output_dir, 'images')
            output_lbl_dir = os.path.join(output_dir, 'labels')
            for pth in (output_dir, output_img_dir, output_lbl_dir):
                os.makedirs(pth, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_img_dir, base + "_warped.jpg"), warped_img)
            with open(os.path.join(output_lbl_dir, base + "_warped.txt"), "w") as f:
                f.write("\n".join(new_labels))
