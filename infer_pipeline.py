import os
from pathlib import Path
import torch
import numpy as np
import cv2
import requests

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots
from utils.general import (
    LOGGER, check_file, check_img_size, increment_path,
    non_max_suppression, scale_boxes
)
from utils.segment.general import process_mask, process_mask_native
from utils.torch_utils import select_device
import utils.anom_visualizer as avis

# when hosting on windows and serving for wsl linux, use gateway ip
SERVER_IP = "127.0.0.1"

# --------------------------
# Helpers
# --------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)


def extract_segmentation_polygon(mask: np.ndarray) -> np.ndarray:
    """
    mask: numpy array HxW float or uint8 (0..1 or 0..255)
    returns: 4x2 float32 quad in image pixel coords in order TL,TR,BR,BL
             or empty list if not found
    """
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(largest)
        quad = cv2.boxPoints(rect).astype(np.float32)

    return order_points(quad)


# --------------------------
# Model loader
# --------------------------
def load_model(weights, device="", data=None, half=False, dnn=False):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    return model


# --------------------------
# Segmentation model wrapper
# --------------------------
class SegmentationModel:
    """
    Sarıcı sınıf: DetectMultiBackend ile çalışır.
    segment(image) -> dict: {
        "mask": (H,W) numpy 0..1 float,
        "box": [x1,y1,x2,y2] (scaled to original im0),
        "conf": float,
        "cls": int,
        "proto": proto_tensor (model-specific, may be None)
    }
    """
    def __init__(self, weights, device="", half=False, dnn=False, retina_masks=True):
        self.model = load_model(weights, device=device, half=half, dnn=dnn)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.retina_masks = retina_masks

    def segment(self, im0: np.ndarray, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45):
        """
        im0: original BGR image (np.ndarray)
        returns dict or None if no detections
        """
        # prepare tensor
        im = cv2.resize(im0, imgsz)
        im_rgb = im[:, :, ::-1].transpose(2, 0, 1)
        im_rgb = np.ascontiguousarray(im_rgb, dtype=np.float32) / 255.0
        im_tensor = torch.from_numpy(im_rgb).unsqueeze(0).to(self.model.device)
        if self.model.fp16:
            im_tensor = im_tensor.half()
        else:
            im_tensor = im_tensor.float()

        pred, proto = self.model(im_tensor)[:2]

        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000, nm=32)
        if len(pred[0]) == 0:
            return None

        dets = pred[0]  # (n, 6 + nm)
        # pick highest confidence
        best_idx = torch.argmax(dets[:, 4]).item()
        best_det = dets[best_idx].unsqueeze(0)  # (1, ...)
        # scale box to original image size
        best_det[:, :4] = scale_boxes(im_tensor.shape[2:], best_det[:, :4], im0.shape).round()

        # get mask (consider retina_masks)
        if self.retina_masks:
            # process_mask_native expects proto[0], mask logits and boxes in im0 coords
            mask = process_mask_native(proto[0], best_det[:, 6:], best_det[:, :4], im0.shape[:2])[0]
        else:
            mask = process_mask(proto[0], best_det[:, 6:], best_det[:, :4], im_tensor.shape[2:], upsample=True)[0]
            # ensure boxes scaled already
            best_det[:, :4] = scale_boxes(im_tensor.shape[2:], best_det[:, :4], im0.shape).round()
        mask = mask.detach().cpu().numpy()
        box = best_det[0, :4].detach().cpu().numpy().tolist()
        conf = float(best_det[0, 4].detach().cpu().item())
        cls = int(best_det[0, 5].detach().cpu().item())

        return {
            "mask": mask,             # HxW 0..1 float or uint8
            "box": box,               # x1,y1,x2,y2 in im0 coords
            "conf": conf,
            "cls": cls,
            "proto": proto
        }


# --------------------------
# Warper class
# --------------------------
class Warper:
    """
    Warper:
      - quad çıkarır (extract_segmentation_polygon)
      - warp eder (warp_image_with_matrix)
      - inverse ile warp edilmiş görüntü üzerindeki kutuları orijinale map eder
    """
    def __init__(self, dst_size=(640, 640)):
        self.dst_size = (int(dst_size[0]), int(dst_size[1]))

    def warp_from_mask(self, im0: np.ndarray, mask: np.ndarray):
        """
        im0: original BGR image
        mask: 2D array same size as im0 HxW (0..1 or 0..255)
        returns: warped_img (BGR), M (3x3), M_inv (3x3), quad (4x2)
        """
        quad = extract_segmentation_polygon(mask)
        if quad is None or len(quad) != 4:
            raise ValueError("Could not extract quad from mask")

        dst_quad = np.array(
            [[0, 0], [self.dst_size[0], 0], [self.dst_size[0], self.dst_size[1]], [0, self.dst_size[1]]],
            dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_quad)
        M_inv = cv2.getPerspectiveTransform(dst_quad, quad.astype(np.float32))
        warped = cv2.warpPerspective(im0, M, self.dst_size)
        return warped, M, M_inv, quad

    def unwarp_boxes(self, boxes, M_inv):
        """
        boxes: list of ((x1,y1,x2,y2), conf, cls) in warped image coords
        M_inv: 3x3 matrix mapping warped->orig
        returns: list of ((x_min,y_min,x_max,y_max), conf, cls) in original image coords
        """
        mapped = []
        for (x1, y1, x2, y2), conf, cls in boxes:
            # ensure float32
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
            pts_back = cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)
            x_min, y_min = float(pts_back[:, 0].min()), float(pts_back[:, 1].min())
            x_max, y_max = float(pts_back[:, 0].max()), float(pts_back[:, 1].max())
            mapped.append(((x_min, y_min, x_max, y_max), conf, cls))
        return mapped


# --------------------------
# Detector wrapper
# --------------------------
class Detector:
    """
    detect(model, image) -> list of ((x1,y1,x2,y2), conf, cls) in image coords
    """
    def __init__(self, weights, device="", half=False, dnn=False):
        self.model = load_model(weights, device=device, half=half, dnn=dnn)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt

    def detect(self, image_bgr: np.ndarray, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=300, classes=None, agnostic_nms=False):
        im_resized = cv2.resize(image_bgr, imgsz)
        im_rgb = im_resized[:, :, ::-1].transpose(2, 0, 1)
        im_rgb = np.ascontiguousarray(im_rgb, dtype=np.float32) / 255.0
        im_tensor = torch.from_numpy(im_rgb).unsqueeze(0).to(self.model.device)
        if self.model.fp16:
            im_tensor = im_tensor.half()
        else:
            im_tensor = im_tensor.float()

        pred = self.model(im_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, max_det=max_det)

        boxes_out = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im_resized.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    boxes_out.append(((x1, y1, x2, y2), float(conf), int(cls)))
        return boxes_out


# --------------------------
# Pipeline function
# --------------------------
def run_seg_warp_detect_pipeline(
    seg_weights,
    det_weights,
    source="images",
    project="runs/predict-seg2det",
    name="exp",
    imgsz_seg=(320, 320),
    imgsz_det=(640, 640),
    seg_conf_thres=0.30,
    seg_iou_thres=0.50,
    det_conf_thres=0.25,
    det_iou_thres=0.45,
    retina_masks=True,
    device_seg="",
    device_det="",
    save_vis=True,
    exist_ok=False,
    get_imgs_from_cam=False
):
    """
    High-level pipeline orchestration. For each image in `source`:
      - run segmentation (SegmentationModel)
      - warp image using Warper
      - run detection on the warped image (Detector)
      - unwarp detected boxes back to original coords
      - draw boxes on original, save and return results
    Returns: list of results per image, each a dict:
      {
        "path": str,
        "orig_image_with_boxes": np.ndarray,
        "warped_image": np.ndarray,
        "mapped_boxes": [((x1,y1,x2,y2), conf, cls), ...],
        "det_boxes_on_warp": [..],
        "seg_info": {...}
      }
    """
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    seg_model = SegmentationModel(seg_weights, device=device_seg, half=False, dnn=False, retina_masks=retina_masks)
    detect_model = Detector(det_weights, device=device_det, half=False, dnn=False)
    warper = Warper(dst_size=imgsz_det)

    imgsz_seg_checked = check_img_size(imgsz_seg, s=seg_model.stride)
    imgsz_det_checked = check_img_size(imgsz_det, s=detect_model.stride)

    # dataloader
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    if not get_imgs_from_cam:
        if screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz_seg_checked, stride=seg_model.stride, auto=seg_model.pt)
        else:
            dataset = LoadImages(source, img_size=imgsz_seg_checked, stride=seg_model.stride, auto=seg_model.pt)

    seg_model.model.warmup(imgsz=(1, 3, *imgsz_seg_checked))
    detect_model.model.warmup(imgsz=(1, 3, *imgsz_det_checked))

    results, imgcnt = [], 0
    global SERVER_IP
    while True:
        if get_imgs_from_cam:
            SERVER_URL = f"http://{SERVER_IP}:8554/frame"  # adjust host if connecting from WSL/container
            resp = requests.get(SERVER_URL, stream=True)
            if resp.status_code == 200:
                jpg_bytes = resp.content
                nparr = np.frombuffer(jpg_bytes, np.uint8)
                im0s = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                print('Could not get the image!')
        else:
            if imgcnt >= len(dataset):
                break
            path, im, im0s, _, _ = dataset[imgcnt]
            p = Path(path)
        imgcnt += 1
        im0 = im0s.copy()

        seg_result = seg_model.segment(im0, imgsz=imgsz_seg_checked, conf_thres=seg_conf_thres, iou_thres=seg_iou_thres)
        if (not get_imgs_from_cam) and seg_result is None:
            LOGGER.info(f"{p.name}: no segmentation detections")
            continue

        if seg_result is None:
            print('plank not found')
            cv2.imshow("Frame from server", im0)
            pressed = cv2.waitKey(10)
            if pressed == 27:
                print('Quit!')
                break
            continue

        mask = seg_result["mask"]
        # warp
        try:
            warped, M, M_inv, quad = warper.warp_from_mask(im0, mask)
        except Exception as e:
            print('Warp error')
            continue

        # detect on warped
        det_boxes_on_warp = detect_model.detect(warped, imgsz=imgsz_det_checked, conf_thres=det_conf_thres, iou_thres=det_iou_thres)

        # map boxes back
        mapped_boxes = warper.unwarp_boxes(det_boxes_on_warp, M_inv)

        im0_draw = avis.create_visualization(im0, mask, warped,
                                  det_boxes_on_warp, mapped_boxes, str(detect_model.names))

        if get_imgs_from_cam:
            cv2.imshow("Frame from server", im0_draw)
            pressed = cv2.waitKey(10)
            if pressed == 27:
                print('Quit!')
                break
        else:
            save_path = str(save_dir / f"{p.stem}_det_on_orig.jpg")
            if save_vis:
                cv2.imwrite(save_path, im0_draw)

            res = {
                "path": str(p),
                "orig_image_with_boxes": im0_draw,
                "warped_image": warped,
                "mapped_boxes": mapped_boxes,
                "det_boxes_on_warp": det_boxes_on_warp,
                "seg_info": seg_result,
                "save_path": save_path
            }
            results.append(res)
            LOGGER.info(f"{p.name}: best-seg conf={seg_result['conf']:.3f}, mapped {len(mapped_boxes)} boxes, saved -> {save_path}")

    return results


# --------------------------
# Example usage in __main__
# --------------------------
if __name__ == "__main__":
    # paths to your weights
    inp_dir = "/home/a249s197/shared/datasets/tatamovski/val/images"
    seg_weights = "trained/yolov5l-seg-plank-500epoch.pt"
    det_weights = "trained/best.pt"

    results = run_seg_warp_detect_pipeline(
        seg_weights=seg_weights,
        det_weights=det_weights,
        source=inp_dir,
        project="runs/predict-seg2det",
        name="exp",
        imgsz_seg=(320, 320),
        imgsz_det=(640, 640),
        seg_conf_thres=0.30,
        seg_iou_thres=0.50,
        det_conf_thres=0.1,
        det_iou_thres=0.45,
        retina_masks=True,
        device_seg="cpu",
        device_det="cpu",
        save_vis=True,
        exist_ok=True,
        get_imgs_from_cam=True
    )

    print(f"Processed {len(results)} images. Done.")
