import json
import os

import cv2
import numpy as np


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
_CLAHE_CACHE = {}


def centered_roi_bounds(image_width, image_height, width_ratio=0.5, height_ratio=0.5):
    """Return an integer (x0, y0, x1, y1) ROI centered in one camera image."""
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image_width and image_height must be positive")
    width_ratio = float(width_ratio)
    height_ratio = float(height_ratio)
    if not (0.0 < width_ratio <= 1.0 and 0.0 < height_ratio <= 1.0):
        raise ValueError("ROI width/height ratios must be in (0, 1]")

    roi_width = max(1, min(width, int(round(width * width_ratio))))
    roi_height = max(1, min(height, int(round(height * height_ratio))))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    return x0, y0, x0 + roi_width, y0 + roi_height


def draw_sbs_center_rois(
    frame,
    width_ratio=0.5,
    height_ratio=0.5,
    color=(0, 255, 255),
    thickness=2,
):
    """Draw the same centered RT ROI on both halves of an SBS frame."""
    height, full_width = frame.shape[:2]
    half_width = full_width // 2
    if half_width <= 0:
        return frame

    x0, y0, x1, y1 = centered_roi_bounds(
        half_width, height, width_ratio, height_ratio)
    for label, offset_x in (("L RT ROI", 0), ("R RT ROI", half_width)):
        p0 = (offset_x + x0, y0)
        p1 = (offset_x + x1 - 1, y1 - 1)
        cv2.rectangle(frame, p0, p1, color, int(thickness), cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (p0[0] + 8, max(24, p0[1] + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def get_clahe(clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_GRID_SIZE):
    key = (clip_limit, tile_size)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return _CLAHE_CACHE[key]


def preprocess_gray(gray_img, enable_clahe=True, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_GRID_SIZE):
    if enable_clahe:
        return get_clahe(clip_limit, tile_size).apply(gray_img)
    return gray_img


def load_json_camera_params(json_path):
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到相機參數檔案 {json_path}")
        return None, None, None, None, None, None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mtx_L = np.array(data["intrinsic_L"]["matrix"], dtype=np.float32)
    dist_L = np.array(data["intrinsic_L"]["distortion"], dtype=np.float32)
    mtx_R = np.array(data["intrinsic_R"]["matrix"], dtype=np.float32)
    dist_R = np.array(data["intrinsic_R"]["distortion"], dtype=np.float32)
    extrinsic = data.get("extrinsic", {})
    R_rel = np.array(extrinsic.get("R", np.eye(3)))
    t_rel = np.array(extrinsic.get("T", np.zeros(3))).reshape(3, 1)
    F_orig = np.array(extrinsic.get("F")) if "F" in extrinsic else None

    # Zebra forvideo mode keeps both sides on the same intrinsic model.
    mtx_R = mtx_L
    dist_R = dist_L

    return mtx_L, dist_L, mtx_R, dist_R, extrinsic, F_orig


def build_undistort_processor(mtx, dist, image_size, alpha=1.0):
    newK, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, alpha, image_size)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newK, image_size, cv2.CV_16SC2)

    def process_view(img, K=None, dist=None, nK=None):
        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        return undist, 1.0

    return newK, map1, map2, process_view
