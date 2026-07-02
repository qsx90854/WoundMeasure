import json
import os

import cv2
import numpy as np


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
_CLAHE_CACHE = {}


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
