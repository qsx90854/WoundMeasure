"""
depth_measure_multi_aruco_sbs_camera.py
================
互動式深度量測工具 (SBS 併排影片 + JSON 標定參數版本)。

特點：
- 支援影片輸入，可指定左圖幀與多個右圖候選幀
- 整合 JSON 標定參數，支援左右相機不對稱的內參與畸變修正
- 雙內參精確幾何：三角測距、單應性映射與基本矩陣均使用獨立的 KL/KR
- 多幀平均量測：點擊左圖後同時計算所有候選幀深度並平均
- 採用 Grad-SIFT 匹配演算法
- 動態 UI：提供右圖候選幀切換選單與匹配狀態切換
"""

import os, sys, glob, json, threading, queue, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import ConnectionPatch, Rectangle, Polygon
from matplotlib.widgets import RadioButtons, Button, CheckButtons, TextBox
import onnxruntime as ort
from Algorithm import aruco_pose as aruco_algo
from Algorithm import camera_preprocess as camera_algo
from Algorithm import video_pose_analysis as video_pose_algo
from Algorithm.perf_timer import StageTimer
from Algorithm.specular_detection import (
    compute_specular_mask_bgr_wound_adaptive,
    compute_rt_aligned_temporal_specular_mask_bgr,
    overlay_specular_mask_rgb,
)
from Algorithm import stereo_matching as stereo_algo
from Algorithm.stereo_matching import (
    get_patch, score_patch_match, score_zncc_patch_match,
    score_warped_patch_match, score_warped_zncc_patch_match,
    get_local_homography_warped_patch, project_point_to_line,
    search_match_on_epipolar_band, point_in_roi,
    plane_homography_from_cand, predict_right_seed_from_geometry,
    enforce_point_on_epipolar, pyramid_ecc_refinement, find_precise_match,
    compute_rgb_sift_descriptors, compute_opponent_sift_descriptors,
    check_color_histogram_similarity, run_improved_matching_flow,
    run_grad_sift_matching_flow,
)

BASE_DIR = Path(__file__).resolve().parent
WOUND_DETECTION_DIR = BASE_DIR / "wound_detection_model"
WOUND_MODEL_PATH = WOUND_DETECTION_DIR / "model" / "assets" / "v9-t-seg_320.onnx"
WOUND_OVERLAY_ALPHA = 0.45
_WOUND_DETECTOR = None
_WOUND_DETECTOR_ERROR_LOGGED = False

# ==================== 全局設定區 ====================
VIDEO_PATH            = r"test_video_Zebra//video_20260601_172436.mp4"        # 影片檔案路徑
RECORD_SAVE_DIR       = "test_video_Zebra"                                    # 錄影儲存資料夾路徑
START_FRAME_COUNT     = 30                         # 前段評估幀數 (N)

# ----------------- 全局日誌收集區 -----------------
ANALYSIS_LOG = []
COMBINATION_LOG = []

def log_and_print(msg):
    print(msg)
    ANALYSIS_LOG.append(str(msg))
END_FRAME_COUNT       = 30                         # 後段評估幀數 (M)
FRAME_RANGE_MODE      = "half_half"                # "fixed" (使用 START/END_FRAME_COUNT) 或 "half_half" (影片前半段與後半段)
POSE_SELECT_MODE      = "reproj_min"               # "reproj_min" (最小重投影誤差), "average" (平均姿態去噪) 或 "best_pair"
MEASURE_MODE          = "dual_direct"              # "dual_direct", "multi_dedrift", "multi_pure"
FLOW_FB_THRESHOLD     = 0.8                        # 雙向光流一致性誤差閾值 (pixels)
EPIPOLAR_DIST_THRESHOLD = 0.8                      # 極線幾何約束距離閾值 (pixels)
LOOP_CLOSURE_DRIFT_THRESHOLD = 1.5                 # 閉環誤差校正門檻值 (pixels)
LOOP_CLOSURE_FAIL_THRESHOLD = 5.0                  # 閉環失敗退回雙目門檻值 (pixels)

MAX_EXTRA_PAIRS       = 5                          # 除了最優對之外，最多再存 N 組次優配對
MAX_REPROJ_ERR_THRES  = 0.5                        # 次優配對的重投影誤差上限門檻 M (px)
FUSE_DEPTH_TOL_REL    = 0.03                       # 融合一致性閘門: 候選與最優對深度相對差容許
FUSE_DEPTH_TOL_ABS_MM = 5.0                        # 融合一致性閘門: 絕對差容許 (取兩者較大)

CAMERA_WIDTH          = 1920                       # 相機解析度寬
CAMERA_HEIGHT         = 1080                       # 相機解析度高

PARAMS_JSON_PATH      = "calibration_result_Zebra_1_no_dis.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM =  8.25#16.5#8.25                       # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
MIN_BASELINE_MM       = 20.0                        # 最小基準線限制 (mm)
MAX_BASELINE_MM       = 220.0                      # 最大基準線限制 (mm)
AUTO_CALC_INTERVAL_SEC = 0.2                       # 連續計算模式下的計算時間間隔 (秒)
ENFORCE_COPLANAR      = False                      # 強制共面對齊優化
SAVE_ARUCO_DEBUG_IMG  = False                      # 是否存出 ArUco 偵測結果圖片
ENABLE_POSE_SMOOTHING  = True                      # 是否啟用時序平滑濾波 (EMA)
POSE_SMOOTHING_ALPHA   = 0.3                         # 平滑係數
ENABLE_CLAHE_DEFAULT  = False                      # 預設是否啟用 CLAHE
CLAHE_CLIP_LIMIT      = 2.0                        # CLAHE 對比度限制閾值 (數值愈大對比愈強，雜訊也愈大)
CLAHE_TILE_GRID_SIZE  = (8, 8)                     # CLAHE 分塊大小 (8, 8) 代表 8x8 的網格
ENABLE_IMPROVED_MATCHING_DEFAULT = False          # 預設是否啟用改良版特徵匹配流程 (高光遮罩 + Harris Corner + 收緊幾何門檻 + 金字塔 ECC)
SHOW_SCORE_DEFAULT = False                         # 預設是否顯示匹配品質與信心分數
DISABLE_EXTRA_CANDS_ECC_PRECISE = True            # 預設是否在多影格融合的次要影格中停用 ECC 與 Precise 精修 (設為 True 可大幅提升點選反應速度)
ENABLE_EPIPOLAR_BAND_SEARCH_DEFAULT = False        # 用候選點只估初始範圍，再沿點選點自己的極線重新搜尋最佳匹配
EPIPOLAR_SEARCH_HALF_LEN = 55                      # 極線方向搜尋半長度 (pixels)
EPIPOLAR_SEARCH_BAND_RADIUS = 2                    # 極線法線方向 band 半徑 (pixels)
EPIPOLAR_SEARCH_MIN_SCORE = 0.35                   # masked ZNCC / gradient NCC 最低接受分數
ENABLE_SIFT_PNP_ASSIST = False                      # 當僅有 1 個 ArUco 標籤時，是否啟用 SIFT 特徵點輔助 RT 與 baseline 解算
EPIPOLAR_SEARCH_DESC_OK = 0.30                      # Epi-band search: normal descriptor threshold
EPIPOLAR_SEARCH_ZNCC_OK = 0.15                      # Epi-band search: normal ZNCC threshold
EPIPOLAR_SEARCH_DESC_STRONG = 0.50                  # Epi-band search: descriptor can rescue a weak ZNCC
EPIPOLAR_SEARCH_ZNCC_STRONG = 0.35                  # Epi-band search: ZNCC can rescue a weak descriptor
EPIPOLAR_SEARCH_WEAK_FLOOR = 0.05                   # Epi-band search: weak score floor when the other score is strong

# ----------------- 交互特徵點匹配搜索設定 -----------------
LEFT_PATCH_SEARCH_RADIUS      = 30#18                         # 左圖點選候選點周圍的搜索半徑 (pixels)
RIGHT_PATCH_SEARCH_RADIUS     = 90#30                         # 右圖預測投影點周圍的搜索半徑 (pixels)
GRAD_SIFT_MAX_RT_ADJUST_PX    = 40.0                       # v1 Grad-SIFT 允許相對 RT/平面預測 seed 的最大微調量 (pixels)
LEFT_GRADIENT_POINTS_COUNT    = 150                        # 左圖周圍取梯度最高的特徵點數量
RIGHT_GRADIENT_POINTS_COUNT   = 300                       # 右圖周圍取梯度最高的特徵點數量
LEFT_MID_GRADIENT_POINTS_COUNT = 150                       # 左圖周圍取梯度中等的特徵點數量
RIGHT_MID_GRADIENT_POINTS_COUNT = 300                      # 右圖周圍取梯度中等的特徵點數量

CIRCLE_LABEL_MATCH_ENABLED = True              # Experimental: snap/match circular labels pasted on the target.
CIRCLE_LABEL_SNAP_RADIUS_PX = 18.0             # Max click distance for snapping the left point to a circle center.
CIRCLE_LABEL_MIN_RADIUS_PX = 15                 # Expected circle label radius range after resizing/undistortion.
CIRCLE_LABEL_MAX_RADIUS_PX = 100
CIRCLE_LABEL_MIN_AREA_PX = 500
CIRCLE_LABEL_MAX_AREA_PX = 25000
CIRCLE_LABEL_MERGE_DISTANCE_PX = 6.0
CIRCLE_LABEL_LIGHT_RING_MIN_FRACTION = 0.30    # Outside of a true label should still look like the bright target surface.
CIRCLE_LABEL_LOW_SAT_RING_MIN_FRACTION = 0.20
CIRCLE_LABEL_INK_MIN_FRACTION = 0.26
CIRCLE_LABEL_MIN_ELLIPSE_AXIS_RATIO = 0.45
CIRCLE_LABEL_MIN_ELLIPSE_SUPPORT = 0.55
CIRCLE_LABEL_ELLIPSE_VOTE_TOL = 0.28
CIRCLE_LABEL_LOCAL_BG_KERNEL = 51
CIRCLE_LABEL_LOCAL_DARK_DELTA = 16.0
CIRCLE_LABEL_COLOR_SAT_MIN = 65
CIRCLE_LABEL_COLOR_SAT_DELTA = 12.0
CIRCLE_LABEL_VERY_DARK_GRAY_MAX = 78
CIRCLE_LABEL_HOUGH_ENABLED = True
CIRCLE_LABEL_HOUGH_DP = 1.2
CIRCLE_LABEL_HOUGH_PARAM1 = 90
CIRCLE_LABEL_HOUGH_PARAM2 = 13
CIRCLE_LABEL_HOUGH_MAX_DIM = 960
CIRCLE_LABEL_HOUGH_MAX_CANDIDATES = 128
CIRCLE_LABEL_DISK_MIN_INK_FRACTION = 0.16
CIRCLE_LABEL_DISK_MIN_CONTRAST = 14.0
CIRCLE_LABEL_DISK_MIN_SAT_DELTA = 16.0
CIRCLE_LABEL_DISK_MIN_COLOR_SAT = 58.0
CIRCLE_LABEL_DEBUG_REASON_ORDER = (
    'accepted',
    'area',
    'points',
    'perimeter',
    'bbox aspect',
    'radius',
    'border',
    'ellipse axis',
    'ellipse vote',
    'empty ring',
    'ink fraction',
    'light ring',
    'low-sat ring',
    'fill ratio',
    'disk evidence',
    'unknown',
)
CIRCLE_LABEL_DEBUG_REASON_COLORS_RGB = {
    'accepted': (0, 255, 80),
    'area': (255, 64, 64),
    'points': (255, 150, 40),
    'perimeter': (255, 225, 40),
    'bbox aspect': (175, 85, 255),
    'radius': (40, 175, 255),
    'border': (80, 90, 255),
    'ellipse axis': (255, 50, 210),
    'ellipse vote': (0, 220, 170),
    'empty ring': (255, 255, 255),
    'ink fraction': (130, 255, 40),
    'light ring': (255, 130, 180),
    'low-sat ring': (80, 255, 255),
    'fill ratio': (190, 125, 35),
    'disk evidence': (255, 185, 90),
    'unknown': (180, 180, 180),
}

GRAD_SIFT_RATIO_TEST          = 0.78                       # v1 Grad-SIFT KNN ratio test threshold
GRAD_SIFT_EPIPOLAR_TOL_PX     = 3.0                        # max point-to-epipolar-line distance for local SIFT matches
GRAD_SIFT_OFFSET_MEDIAN_TOL_PX = 8.0                       # reject local matches whose disparity differs too much from median
GRAD_SIFT_RANSAC_REPROJ_PX    = 2.5                        # local affine RANSAC reprojection threshold
GRAD_SIFT_MIN_GROUP_INLIERS   = 3                          # minimum inliers for accepting one high/mid gradient group
GRAD_SIFT_GUIDED_RADIUS_PX    = 10.0                       # guided fallback: search right refs near RT/plane-predicted location
GRAD_SIFT_GUIDED_RATIO_TEST   = 0.95                       # guided fallback uses geometry, so descriptor ambiguity can be looser
UI_LOOP_SLEEP_SEC             = 0.03                       # idle UI loop delay; lower is smoother but uses more CPU
IDEAL_BASELINE_MM             = 45.0                       # preferred baseline for pair selection
PAIR_SCORE_REPROJ_W           = 1.00                       # pair selection weight: reprojection error
PAIR_SCORE_BASELINE_W         = 0.18                       # pair selection weight: baseline away from ideal
PAIR_SCORE_BLUR_W             = 0.18                       # pair selection weight: blur penalty
PAIR_SCORE_COVER_W            = 0.12                       # pair selection weight: weak ArUco coverage
PAIR_SCORE_MARKER_W           = 0.08                       # pair selection weight: too few shared markers
# ===================================================

# 將主檔頂部的可調常數注入 stereo_matching 模組 (調參仍集中在本檔)
for _const_name in stereo_algo.TUNABLE_CONSTANTS:
    setattr(stereo_algo, _const_name, globals()[_const_name])

def _empty_points():
    return np.empty((0, 2), dtype=np.float32)

def merge_circle_candidates(candidates, merge_dist=CIRCLE_LABEL_MERGE_DISTANCE_PX):
    if not candidates:
        return _empty_points(), np.empty((0,), dtype=np.float32)

    merged = []
    for center, radius, score in sorted(candidates, key=lambda item: item[2], reverse=True):
        center = np.asarray(center, dtype=np.float32)
        duplicate = False
        for item in merged:
            if np.linalg.norm(center - item['center']) <= merge_dist:
                duplicate = True
                if score > item['score']:
                    item['center'] = center
                    item['radius'] = float(radius)
                    item['score'] = float(score)
                break
        if not duplicate:
            merged.append({'center': center, 'radius': float(radius), 'score': float(score)})

    centers = np.array([item['center'] for item in merged], dtype=np.float32)
    radii = np.array([item['radius'] for item in merged], dtype=np.float32)
    return centers, radii

def ellipse_contour_vote_score(contour, ellipse, vote_tol=CIRCLE_LABEL_ELLIPSE_VOTE_TOL):
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    major = max(float(axis_a), float(axis_b))
    minor = min(float(axis_a), float(axis_b))
    if major <= 1e-6 or minor <= 1e-6:
        return 0.0, 0.0

    pts = contour.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    a = major / 2.0
    b = minor / 2.0

    supports = []
    for theta in (np.deg2rad(angle_deg), np.deg2rad(angle_deg + 90.0)):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        supports.append(float(np.mean(np.abs(rho - 1.0) <= vote_tol)))
    support = max(supports)
    axis_ratio = minor / major
    return support, axis_ratio

def _circle_label_bg_kernel(gray):
    h, w = gray.shape[:2]
    max_kernel = max(5, min(h, w))
    if max_kernel % 2 == 0:
        max_kernel -= 1
    kernel = min(int(CIRCLE_LABEL_LOCAL_BG_KERNEL), max_kernel)
    if kernel % 2 == 0:
        kernel -= 1
    return max(5, kernel)

def _circle_label_local_features(gray, hsv):
    kernel = _circle_label_bg_kernel(gray)
    gray_f = gray.astype(np.float32)
    sat_f = hsv[:, :, 1].astype(np.float32)
    local_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0).astype(np.float32)
    local_sat = cv2.GaussianBlur(hsv[:, :, 1], (kernel, kernel), 0).astype(np.float32)
    dark_delta = local_gray - gray_f
    sat_delta = sat_f - local_sat
    return local_gray, local_sat, dark_delta, sat_delta

def _circle_label_contour_from_disk(x, y, radius, num_points=64):
    angles = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False)
    pts = np.stack([
        float(x) + np.cos(angles) * float(radius),
        float(y) + np.sin(angles) * float(radius),
    ], axis=1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)

def _circle_label_radial_roi(center, radius, image_shape, inner_scale, ring_inner_scale, ring_outer_scale):
    h, w = image_shape[:2]
    x, y = float(center[0]), float(center[1])
    outer_radius = float(radius) * float(ring_outer_scale)
    x0 = max(0, int(np.floor(x - outer_radius)))
    x1 = min(w, int(np.ceil(x + outer_radius)) + 1)
    y0 = max(0, int(np.floor(y - outer_radius)))
    y1 = min(h, int(np.ceil(y + outer_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    local_y, local_x = np.ogrid[y0:y1, x0:x1]
    dist2 = (local_x - x) ** 2 + (local_y - y) ** 2
    inner = dist2 <= (float(radius) * float(inner_scale)) ** 2
    ring = (
        (dist2 >= (float(radius) * float(ring_inner_scale)) ** 2) &
        (dist2 <= outer_radius ** 2)
    )
    return (slice(y0, y1), slice(x0, x1)), inner, ring

def build_circle_label_ink_mask(bgr):
    if bgr is None or bgr.size == 0:
        return None, None, None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if bgr.ndim == 3 else cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    _local_gray, _local_sat, dark_delta, sat_delta = _circle_label_local_features(gray, hsv)

    # Absolute gray thresholds make shaded gray paper turn into one connected blob.
    # Use local contrast for black disks and saturation contrast for colored disks.
    very_dark = gray <= CIRCLE_LABEL_VERY_DARK_GRAY_MAX
    locally_dark = (dark_delta >= CIRCLE_LABEL_LOCAL_DARK_DELTA) & (gray < 185)
    colored = (
        (sat >= CIRCLE_LABEL_COLOR_SAT_MIN) &
        (
            (sat_delta >= CIRCLE_LABEL_COLOR_SAT_DELTA) |
            (dark_delta >= 5.0) |
            (val <= 220)
        )
    )
    dark_or_colored_label = very_dark | locally_dark | colored
    ink_mask = dark_or_colored_label.astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return gray, hsv, ink_mask

def evaluate_circle_label_disk_candidate(center, radius, gray, hsv, ink_mask, source='hough'):
    h, w = gray.shape[:2]
    x, y = float(center[0]), float(center[1])
    radius = float(radius)
    cnt = _circle_label_contour_from_disk(x, y, radius)
    info = {
        'contour': cnt,
        'ellipse': ((x, y), (radius * 2.0, radius * 2.0), 0.0),
        'accepted': False,
        'reason': '',
        'source': source,
    }

    if radius < CIRCLE_LABEL_MIN_RADIUS_PX or radius > CIRCLE_LABEL_MAX_RADIUS_PX:
        info['reason'] = 'radius'
        return None, info
    if x < radius or y < radius or x > w - radius or y > h - radius:
        info['reason'] = 'border'
        return None, info

    radial_roi = _circle_label_radial_roi((x, y), radius, gray.shape, 0.85, 1.18, 2.30)
    if radial_roi is None:
        info['reason'] = 'empty ring'
        return None, info
    roi, inner, ring = radial_roi
    inner_count = int(np.count_nonzero(inner))
    ring_count = int(np.count_nonzero(ring))
    if inner_count == 0 or ring_count == 0:
        info['reason'] = 'empty ring'
        return None, info

    gray_roi = gray[roi]
    sat_roi = hsv[roi][:, :, 1]
    ink_roi = ink_mask[roi]
    inner_gray = gray_roi[inner].astype(np.float32)
    ring_gray = gray_roi[ring].astype(np.float32)
    inner_sat = sat_roi[inner].astype(np.float32)
    ring_sat = sat_roi[ring].astype(np.float32)

    ink_fraction = float(np.count_nonzero(ink_roi[inner]) / inner_count)
    ring_light_fraction = float(np.count_nonzero(ring_gray > 65) / ring_count)
    ring_low_sat_fraction = float(np.count_nonzero(ring_sat < 115) / ring_count)
    gray_contrast = float(np.percentile(ring_gray, 60) - np.percentile(inner_gray, 40))
    sat_delta = float(np.percentile(inner_sat, 70) - np.percentile(ring_sat, 60))
    inner_sat_ref = float(np.percentile(inner_sat, 70))
    inner_gray_std = float(np.std(inner_gray))

    has_dark_disk = gray_contrast >= CIRCLE_LABEL_DISK_MIN_CONTRAST
    has_colored_disk = (
        inner_sat_ref >= CIRCLE_LABEL_DISK_MIN_COLOR_SAT and
        sat_delta >= CIRCLE_LABEL_DISK_MIN_SAT_DELTA
    )
    has_mask_support = ink_fraction >= CIRCLE_LABEL_DISK_MIN_INK_FRACTION
    is_aruco_like_texture = inner_gray_std > 55.0 and inner_sat_ref < CIRCLE_LABEL_DISK_MIN_COLOR_SAT
    if is_aruco_like_texture or not (has_dark_disk or has_colored_disk or has_mask_support):
        info['reason'] = 'disk evidence'
        return None, info
    if ring_light_fraction < CIRCLE_LABEL_LIGHT_RING_MIN_FRACTION:
        info['reason'] = 'light ring'
        return None, info
    if ring_low_sat_fraction < CIRCLE_LABEL_LOW_SAT_RING_MIN_FRACTION:
        info['reason'] = 'low-sat ring'
        return None, info

    score = (
        1.0 +
        min(1.0, ink_fraction * 2.0) +
        min(1.0, max(0.0, gray_contrast) / 35.0) +
        min(1.0, max(0.0, sat_delta) / 45.0) +
        ring_light_fraction +
        ring_low_sat_fraction
    )
    info.update({
        'accepted': True,
        'reason': 'accepted',
        'center': (x, y),
        'radius': radius,
        'support': float(has_dark_disk or has_colored_disk or has_mask_support),
        'axis_ratio': 1.0,
        'ink_fraction': ink_fraction,
        'ring_light_fraction': ring_light_fraction,
        'ring_low_sat_fraction': ring_low_sat_fraction,
        'gray_contrast': gray_contrast,
        'sat_delta': sat_delta,
        'fill_ratio': 1.0,
    })
    return ((x, y), radius, score), info

def collect_circle_label_hough_candidates(gray, hsv, ink_mask):
    if not CIRCLE_LABEL_HOUGH_ENABLED:
        return [], []

    h, w = gray.shape[:2]
    scale = min(1.0, float(CIRCLE_LABEL_HOUGH_MAX_DIM) / float(max(h, w)))
    if scale < 1.0:
        small_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        gray_search = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
        hsv_search = cv2.resize(hsv, small_size, interpolation=cv2.INTER_AREA)
    else:
        gray_search = gray
        hsv_search = hsv

    sat = hsv_search[:, :, 1].astype(np.float32)
    _local_gray, _local_sat, dark_delta, sat_delta = _circle_label_local_features(gray_search, hsv_search)
    dark_evidence = np.clip(dark_delta * 5.0, 0.0, 255.0)
    color_evidence = np.where(
        sat >= CIRCLE_LABEL_COLOR_SAT_MIN,
        np.clip(np.maximum(sat, sat_delta * 6.0), 0.0, 255.0),
        0.0
    )
    evidence = np.maximum(dark_evidence, color_evidence).astype(np.uint8)
    evidence = cv2.GaussianBlur(evidence, (5, 5), 0)

    min_radius = max(3, int(round(CIRCLE_LABEL_MIN_RADIUS_PX * 0.75 * scale)))
    max_radius = max(min_radius + 1, int(round(CIRCLE_LABEL_MAX_RADIUS_PX * 1.05 * scale)))
    min_dist = max(5, int(round(CIRCLE_LABEL_MIN_RADIUS_PX * 1.35 * scale)))
    circles = cv2.HoughCircles(
        evidence,
        cv2.HOUGH_GRADIENT,
        dp=CIRCLE_LABEL_HOUGH_DP,
        minDist=min_dist,
        param1=CIRCLE_LABEL_HOUGH_PARAM1,
        param2=CIRCLE_LABEL_HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return [], []

    candidates = []
    infos = []
    hough_rows = circles[0][:CIRCLE_LABEL_HOUGH_MAX_CANDIDATES].astype(np.float32)
    hough_rows /= float(scale)
    for x, y, radius in hough_rows:
        candidate, info = evaluate_circle_label_disk_candidate((x, y), radius, gray, hsv, ink_mask, source='hough')
        infos.append(info)
        if candidate is not None:
            candidates.append(candidate)
    return candidates, infos

def evaluate_circle_label_contour(cnt, gray, hsv, ink_mask):
    h, w = gray.shape[:2]
    sat = hsv[:, :, 1]
    info = {'contour': cnt, 'ellipse': None, 'accepted': False, 'reason': ''}

    area = float(cv2.contourArea(cnt))
    if area < CIRCLE_LABEL_MIN_AREA_PX or area > CIRCLE_LABEL_MAX_AREA_PX:
        info['reason'] = 'area'
        return None, info
    if len(cnt) < 5:
        info['reason'] = 'points'
        return None, info
    perim = float(cv2.arcLength(cnt, True))
    if perim <= 1e-6:
        info['reason'] = 'perimeter'
        return None, info

    x_box, y_box, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / max(1.0, float(bh))
    if aspect < 0.40 or aspect > 2.50:
        info['reason'] = 'bbox aspect'
        return None, info

    ellipse = cv2.fitEllipse(cnt)
    info['ellipse'] = ellipse
    (x, y), (axis_a, axis_b), angle = ellipse
    major = max(float(axis_a), float(axis_b))
    minor = min(float(axis_a), float(axis_b))
    radius = 0.25 * (major + minor)
    if radius < CIRCLE_LABEL_MIN_RADIUS_PX or radius > CIRCLE_LABEL_MAX_RADIUS_PX:
        info['reason'] = 'radius'
        return None, info
    if x < radius or y < radius or x > w - radius or y > h - radius:
        info['reason'] = 'border'
        return None, info
    axis_ratio = minor / max(major, 1e-6)
    if axis_ratio < CIRCLE_LABEL_MIN_ELLIPSE_AXIS_RATIO:
        info['reason'] = 'ellipse axis'
        return None, info
    support, vote_axis_ratio = ellipse_contour_vote_score(cnt, ellipse)
    if support < CIRCLE_LABEL_MIN_ELLIPSE_SUPPORT:
        info['reason'] = 'ellipse vote'
        return None, info

    radial_roi = _circle_label_radial_roi((x, y), radius, gray.shape, 0.95, 1.20, 2.40)
    if radial_roi is None:
        info['reason'] = 'empty ring'
        return None, info
    roi, inner, ring = radial_roi
    inner_count = int(np.count_nonzero(inner))
    ring_count = int(np.count_nonzero(ring))
    if inner_count == 0 or ring_count == 0:
        info['reason'] = 'empty ring'
        return None, info

    gray_roi = gray[roi]
    sat_roi = sat[roi]
    ink_roi = ink_mask[roi]
    ink_fraction = float(np.count_nonzero(ink_roi[inner]) / inner_count)
    ring_light_fraction = float(np.count_nonzero(gray_roi[ring] > 65) / ring_count)
    ring_low_sat_fraction = float(np.count_nonzero(sat_roi[ring] < 115) / ring_count)
    if ink_fraction < CIRCLE_LABEL_INK_MIN_FRACTION:
        info['reason'] = 'ink fraction'
        return None, info
    if ring_light_fraction < CIRCLE_LABEL_LIGHT_RING_MIN_FRACTION:
        info['reason'] = 'light ring'
        return None, info
    if ring_low_sat_fraction < CIRCLE_LABEL_LOW_SAT_RING_MIN_FRACTION:
        info['reason'] = 'low-sat ring'
        return None, info

    ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
    fill_ratio = area / (ellipse_area + 1e-6)
    if fill_ratio < 0.24 or fill_ratio > 1.35:
        info['reason'] = 'fill ratio'
        return None, info

    score = support + vote_axis_ratio + ink_fraction + ring_light_fraction + ring_low_sat_fraction
    info.update({
        'accepted': True,
        'reason': 'accepted',
        'center': (float(x), float(y)),
        'radius': float(radius),
        'support': float(support),
        'axis_ratio': float(vote_axis_ratio),
        'ink_fraction': ink_fraction,
        'ring_light_fraction': ring_light_fraction,
        'ring_low_sat_fraction': ring_low_sat_fraction,
        'fill_ratio': float(fill_ratio),
    })
    return ((x, y), radius, score), info

def collect_circle_label_contours(bgr):
    gray, hsv, ink_mask = build_circle_label_ink_mask(bgr)
    if gray is None:
        return {
            'gray': None,
            'mask': None,
            'contours': [],
            'hough': [],
            'centers': _empty_points(),
            'radii': np.empty((0,), dtype=np.float32),
        }

    # The bright area outside the target can enclose every label in the contour
    # hierarchy. RETR_EXTERNAL would then return only that one outer component.
    contours, _ = cv2.findContours(ink_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_candidates = []
    contour_infos = []
    for cnt in contours:
        candidate, info = evaluate_circle_label_contour(cnt, gray, hsv, ink_mask)
        contour_infos.append(info)
        if candidate is not None:
            contour_candidates.append(candidate)

    # Hough is only a last-resort fallback. Mixing all of its trial circles into
    # the contour result creates many large false candidates and a misleading
    # contour debug overlay.
    hough_candidates = []
    hough_infos = []
    if not contour_candidates:
        hough_candidates, hough_infos = collect_circle_label_hough_candidates(gray, hsv, ink_mask)

    candidates = contour_candidates + hough_candidates
    centers, radii = merge_circle_candidates(candidates)
    return {
        'gray': gray,
        'mask': ink_mask,
        'contours': contour_infos,
        'hough': hough_infos,
        'centers': centers,
        'radii': radii,
    }

def detect_circle_label_centers(bgr):
    debug = collect_circle_label_contours(bgr)
    return debug['centers'], debug['radii']

def circle_label_debug_reason(info):
    if info.get('accepted'):
        return 'accepted'
    return info.get('reason') or 'unknown'

def circle_label_debug_color(reason):
    return CIRCLE_LABEL_DEBUG_REASON_COLORS_RGB.get(
        reason,
        CIRCLE_LABEL_DEBUG_REASON_COLORS_RGB['unknown']
    )

def draw_circle_label_debug_legend(overlay, reason_counts):
    if not reason_counts:
        return overlay

    ordered_reasons = [
        reason for reason in CIRCLE_LABEL_DEBUG_REASON_ORDER
        if reason in reason_counts
    ]
    extra_reasons = sorted(
        reason for reason in reason_counts
        if reason not in CIRCLE_LABEL_DEBUG_REASON_ORDER
    )
    items = ordered_reasons + extra_reasons

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    row_h = 15
    pad = 7
    title = "contour filter"
    labels = [f"{reason}: {reason_counts[reason]}" for reason in items]
    max_text_w = cv2.getTextSize(title, font, font_scale, thickness)[0][0]
    for label in labels:
        max_text_w = max(max_text_w, cv2.getTextSize(label, font, font_scale, thickness)[0][0])

    panel_w = min(overlay.shape[1] - 8, max_text_w + 34)
    panel_h = min(overlay.shape[0] - 8, pad * 2 + row_h * (len(items) + 1))
    if panel_w <= 0 or panel_h <= 0:
        return overlay

    panel = overlay.copy()
    cv2.rectangle(panel, (4, 4), (4 + panel_w, 4 + panel_h), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.58, overlay, 0.42, 0)

    y = 4 + pad + 8
    cv2.putText(overlay, title, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    y += row_h
    max_rows = max(0, (panel_h - pad * 2 - row_h) // row_h)
    for reason in items[:max_rows]:
        color = circle_label_debug_color(reason)
        cv2.rectangle(overlay, (10, y - 9), (20, y + 1), color, -1)
        cv2.putText(
            overlay,
            f"{reason}: {reason_counts[reason]}",
            (26, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        y += row_h
    return overlay

def draw_circle_label_contour_debug_overlay(bgr, debug):
    if bgr is None:
        return None
    overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    overlay = overlay.copy()

    reason_counts = {}
    for info in debug.get('contours', []):
        cnt = info.get('contour')
        if cnt is None:
            continue
        reason = circle_label_debug_reason(info)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        color = circle_label_debug_color(reason)
        cv2.drawContours(overlay, [cnt], -1, color, 1, cv2.LINE_AA)
        ellipse = info.get('ellipse')
        if ellipse is not None:
            cv2.ellipse(overlay, ellipse, color, 1, cv2.LINE_AA)
        if info.get('accepted') and info.get('center') is not None:
            x, y = info['center']
            cv2.drawMarker(
                overlay, (int(round(x)), int(round(y))), (255, 255, 0),
                markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1,
                line_type=cv2.LINE_AA
            )

    overlay = draw_circle_label_debug_legend(overlay, reason_counts)
    return overlay

def find_nearest_circle_center(point, centers, max_dist):
    centers = np.asarray(centers if centers is not None else _empty_points(), dtype=np.float32)
    if centers.size == 0:
        return None, None
    point = np.asarray(point, dtype=np.float32)
    dists = np.linalg.norm(centers - point, axis=1)
    idx = int(np.argmin(dists))
    if float(dists[idx]) <= max_dist:
        return centers[idx].copy(), float(dists[idx])
    return None, None

def circle_label_search_rect(left_point, cand, K_L, image_shape):
    seed_pt, seed_method = predict_right_seed_from_geometry(left_point, cand, K_L)
    if seed_pt is None or not np.all(np.isfinite(seed_pt)):
        return None, None, seed_method

    h, w = image_shape[:2]
    rad = float(RIGHT_PATCH_SEARCH_RADIUS)
    x0 = max(0.0, float(seed_pt[0]) - rad)
    y0 = max(0.0, float(seed_pt[1]) - rad)
    x1 = min(float(w - 1), float(seed_pt[0]) + rad)
    y1 = min(float(h - 1), float(seed_pt[1]) + rad)
    if x1 <= x0 or y1 <= y0:
        return np.asarray(seed_pt, dtype=np.float32), None, seed_method
    return np.asarray(seed_pt, dtype=np.float32), (x0, y0, x1 - x0, y1 - y0), seed_method

def find_circle_center_in_rect(centers, rect, preferred_point=None):
    centers = np.asarray(centers if centers is not None else _empty_points(), dtype=np.float32)
    if centers.size == 0 or rect is None:
        return None
    x, y, rw, rh = rect
    mask = (
        (centers[:, 0] >= x) & (centers[:, 0] <= x + rw) &
        (centers[:, 1] >= y) & (centers[:, 1] <= y + rh)
    )
    if not np.any(mask):
        return None
    roi_centers = centers[mask]
    if preferred_point is None:
        preferred_point = np.array([x + rw / 2.0, y + rh / 2.0], dtype=np.float32)
    dists = np.linalg.norm(roi_centers - np.asarray(preferred_point, dtype=np.float32), axis=1)
    return roi_centers[int(np.argmin(dists))].copy()

def get_wound_detector():
    """Lazy-load the v9-t-seg_320 wound segmentation model."""
    global _WOUND_DETECTOR, _WOUND_DETECTOR_ERROR_LOGGED
    if _WOUND_DETECTOR is not None:
        return _WOUND_DETECTOR
    if not WOUND_MODEL_PATH.exists():
        if not _WOUND_DETECTOR_ERROR_LOGGED:
            print(f"[Wound] Cannot find model: {WOUND_MODEL_PATH}")
            _WOUND_DETECTOR_ERROR_LOGGED = True
        return None

    old_cwd = Path.cwd()
    wound_dir_str = str(WOUND_DETECTION_DIR)
    try:
        if wound_dir_str not in sys.path:
            sys.path.insert(0, wound_dir_str)
        os.chdir(WOUND_DETECTION_DIR)
        from wound_detector import WoundDetector

        _WOUND_DETECTOR = WoundDetector(WOUND_MODEL_PATH)
        model = _WOUND_DETECTOR.model
        print(f"[Wound] Loaded {model.model_path} input_shape={model.input_shape}")
        return _WOUND_DETECTOR
    except Exception as exc:
        if not _WOUND_DETECTOR_ERROR_LOGGED:
            print(f"[Wound] Failed to load wound detector: {exc}")
            _WOUND_DETECTOR_ERROR_LOGGED = True
        return None
    finally:
        os.chdir(old_cwd)


def predict_wound_regions_bgr(bgr):
    detector = get_wound_detector()
    if detector is None:
        return None
    try:
        return detector.predict(bgr.copy(), draw_result=False)
    except Exception as exc:
        print(f"[Wound] Inference failed: {exc}")
        return None


def count_wound_detections(prediction):
    if not prediction:
        return 0
    first = prediction[0]
    if first is None or len(first) < 4:
        return 0
    return int(len(first[0]))


def extract_wound_rect(prediction, image_shape):
    if not prediction:
        return None
    first = prediction[0]
    if first is None or len(first) < 4:
        return None

    h, w = image_shape[:2]
    _classes, bboxes, scores, masks = first
    best_contour = None
    best_bbox = None
    best_area = 0.0
    for bbox, score, mask in zip(bboxes, scores, masks):
        conf_val = float(score[0] if isinstance(score, np.ndarray) else score)
        if conf_val <= 0.01:
            continue
        mask_f = mask.astype(np.float32)
        if mask_f.shape != (h, w):
            mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_u8 = ((mask_f > 0.5).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area > best_area:
                best_area = area
                best_contour = contour
                best_bbox = bbox

    if best_contour is None or best_area <= 0:
        return None

    min_area_box = cv2.boxPoints(cv2.minAreaRect(best_contour)).astype(np.float32)
    h, w = image_shape[:2]
    if best_bbox is not None:
        x_min, y_min, x_max, y_max = map(float, best_bbox)
        x_min, x_max = sorted((max(0.0, min(w - 1.0, x_min)), max(0.0, min(w - 1.0, x_max))))
        y_min, y_max = sorted((max(0.0, min(h - 1.0, y_min)), max(0.0, min(h - 1.0, y_max))))
        bbox_box = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
    else:
        bbox_box = min_area_box.copy()
    return {'box': min_area_box, 'min_area_box': min_area_box, 'bbox_box': bbox_box, 'area_px': best_area}


def prediction_to_wound_mask(prediction, image_shape):
    if not prediction:
        return None
    first = prediction[0]
    if first is None or len(first) < 4:
        return None
    h, w = image_shape[:2]
    _classes, _bboxes, scores, masks = first
    combined = np.zeros((h, w), dtype=np.uint8)
    for score, mask in zip(scores, masks):
        conf_val = float(score[0] if isinstance(score, np.ndarray) else score)
        if conf_val <= 0.01:
            continue
        mask_f = mask.astype(np.float32)
        if mask_f.shape != (h, w):
            mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
        combined[mask_f > 0.5] = 255
    return combined if np.any(combined) else None


def draw_wound_size_label_rgb(rgb, size_info, title="Wound", fallback_text=None):
    if not size_info and not fallback_text:
        return rgb
    out = rgb.copy()
    h, _w = out.shape[:2]
    if size_info:
        long_v = size_info['long']
        short_v = size_info['short']
        unit = size_info['unit']
        text = f"{title} L:{long_v:.1f}{unit} W:{short_v:.1f}{unit}"
    else:
        text = f"{title} {fallback_text}"
    font_scale = max(0.5, h / 1100.0)
    thickness = max(1, int(h / 420))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = 12, 28
    cv2.rectangle(out, (x - 6, y - th - 8), (x + tw + 6, y + base + 6), (22, 22, 22), -1)
    cv2.rectangle(out, (x - 6, y - th - 8), (x + tw + 6, y + base + 6), (255, 230, 40), thickness)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 230, 40), thickness, cv2.LINE_AA)
    return out


def overlay_wound_prediction_rgb(rgb, prediction, alpha=WOUND_OVERLAY_ALPHA, draw_bbox=True):
    if not prediction:
        return rgb
    first = prediction[0]
    if first is None or len(first) < 4:
        return rgb

    out = rgb.copy()
    h, w = out.shape[:2]
    _classes, bboxes, scores, masks = first
    mask_color = np.array([255, 64, 64], dtype=np.float32)
    edge_color = (255, 230, 40)
    font_scale = max(0.45, h / 1200.0)
    thickness = max(1, int(h / 420))

    for bbox, score, mask in zip(bboxes, scores, masks):
        conf_val = float(score[0] if isinstance(score, np.ndarray) else score)
        if conf_val <= 0.01:
            continue

        mask_f = mask.astype(np.float32)
        if mask_f.shape != (h, w):
            mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
        bool_mask = mask_f > 0.5
        out[bool_mask] = out[bool_mask].astype(np.float32) * (1.0 - alpha) + mask_color * alpha

        mask_u8 = bool_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, edge_color, thickness, cv2.LINE_AA)

        if draw_bbox:
            x_min, y_min, x_max, y_max = map(int, bbox)
            x_min, x_max = sorted((max(0, min(w - 1, x_min)), max(0, min(w - 1, x_max))))
            y_min, y_max = sorted((max(0, min(h - 1, y_min)), max(0, min(h - 1, y_max))))
            cv2.rectangle(out, (x_min, y_min), (x_max, y_max), edge_color, thickness, cv2.LINE_AA)
            label = f"wound {conf_val:.0%}"
            cv2.putText(out, label, (x_min, max(14, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, edge_color, thickness, cv2.LINE_AA)

    return out.astype(np.uint8)


def draw_wound_corner_points_rgb(rgb, points, title_prefix, line_closed=False, color=(255, 230, 40)):
    if rgb is None or points is None:
        return rgb
    out = rgb.copy()
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return out

    h, w = out.shape[:2]
    valid = np.isfinite(pts).all(axis=1)
    radius = max(4, int(round(min(h, w) / 180.0)))
    thickness = max(2, int(round(min(h, w) / 420.0)))
    font_scale = max(0.45, h / 1250.0)
    edge = (20, 20, 20)

    drawable = []
    for i, pt in enumerate(pts):
        if not valid[i]:
            continue
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        drawable.append((x, y))
        cv2.circle(out, (x, y), radius + 2, edge, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius + 2, color, thickness, cv2.LINE_AA)
        label = f"{title_prefix}{i + 1}"
        cv2.putText(out, label, (x + radius + 4, y - radius - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, edge, thickness + 2, cv2.LINE_AA)
        cv2.putText(out, label, (x + radius + 4, y - radius - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    if line_closed and len(drawable) >= 2:
        poly = np.asarray(drawable, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [poly], isClosed=len(drawable) >= 3, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def compute_fundamental_matrix(K_L, K_R, R_rel, t_rel):
    t = t_rel.flatten()
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    E = tx @ R_rel
    K_R_inv, K_L_inv = np.linalg.inv(K_R.astype(np.float64)), np.linalg.inv(K_L.astype(np.float64))
    return K_R_inv.T @ E @ K_L_inv

def triangulate_point_3d(pt_A, pt_B, K_L, K_R, R_rel, t_rel, F=None):
    # 提供 F 時先做 Hartley-Sturm 最佳修正 (cv2.correctMatches)，
    # 同時最小幅度微調左右點使其嚴格滿足極線幾何，優於只單邊投影右點
    if F is not None:
        try:
            ptsA_in = np.array([[[float(pt_A[0]), float(pt_A[1])]]], dtype=np.float64)
            ptsB_in = np.array([[[float(pt_B[0]), float(pt_B[1])]]], dtype=np.float64)
            ptsA_c, ptsB_c = cv2.correctMatches(F.astype(np.float64), ptsA_in, ptsB_in)
            if np.all(np.isfinite(ptsA_c)) and np.all(np.isfinite(ptsB_c)):
                pt_A = ptsA_c[0, 0]
                pt_B = ptsB_c[0, 0]
        except cv2.error:
            pass
    P0 = (K_L.astype(np.float64) @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
    P1 = (K_R.astype(np.float64) @ np.hstack([R_rel, t_rel])).astype(np.float32)
    # 強制使用 float32，避免 OpenCV 在處理整數點陣列時發生隱性記憶體錯亂 (計算出極端錯誤的負深度)
    ptsA_f32 = np.array([[pt_A[0]], [pt_A[1]]], dtype=np.float32)
    ptsB_f32 = np.array([[pt_B[0]], [pt_B[1]]], dtype=np.float32)
    pts4d = cv2.triangulatePoints(P0, P1, ptsA_f32, ptsB_f32)
    pt3d = (pts4d[:3] / pts4d[3]).flatten()
    return pt3d

def epipolar_line(F, pt, img_w):
    l = F @ np.array([pt[0], pt[1], 1.0])
    a, b, c = l
    if abs(b) > 1e-8: return (0, int(-c/b)), (img_w-1, int(-(a*(img_w-1)+c)/b))
    return (int(-c/a), 0), (int(-c/a), img_w-1)

def average_rotations_svd(R_list):
    """
    對多個 3x3 旋轉矩陣進行 SVD 平均，獲得在 SO(3) 群上的正交最小二乘平均矩陣
    """
    if len(R_list) == 0:
        return np.eye(3, dtype=np.float32)
    M = np.zeros((3, 3), dtype=np.float64)
    for R in R_list:
        M += R.astype(np.float64)
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg.astype(np.float32)

def multi_view_triangulation(P_matrices, points_2d):
    """
    N-View 三角化 (DLT 算法)
    P_matrices: list of 3x4 projection matrices [R | t] (單位：米)
    points_2d: list of (x, y) normalized coordinates
    """
    if len(P_matrices) < 2:
        return None
    A = []
    for P, (x, y) in zip(P_matrices, points_2d):
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-6:
        return None
    return (X[:3] / X[3])

def fuse_candidate_results(res_list, best_idx=None):
    """
    多候選影格量測結果融合：
    1. 候選一致性閘門：與最優對 (best_idx) 深度差超過容許值
       max(FUSE_DEPTH_TOL_ABS_MM, FUSE_DEPTH_TOL_REL * z_best) 的候選不參與融合
       (兩個候選時 MAD 無法運作，靠此閘門擋掉與最優對不一致的壞樣本)
    2. 深度中位數 + MAD 剔除離群 (|z - med| > 3 * 1.4826 * MAD, n>=3)
    3. 依 (baseline / z^2)^2 加權平均 (三角測距深度不確定度 ∝ z^2 / (f * baseline))
    res_list 每個元素需含 'p3d'，可含 'baseline'、'p3d_w'、'cand_idx'。
    """
    if not res_list:
        return None
    dropped = []
    pool = list(res_list)
    if best_idx is not None:
        ref = next((r for r in pool if r.get('cand_idx') == best_idx), None)
        if ref is not None:
            z_ref = float(ref['p3d'][2])
            tol = max(FUSE_DEPTH_TOL_ABS_MM, FUSE_DEPTH_TOL_REL * abs(z_ref))
            gated = []
            for r in pool:
                dz = abs(float(r['p3d'][2]) - z_ref)
                if dz <= tol:
                    gated.append(r)
                else:
                    r['drop_reason'] = f"深度偏離最優對 {dz:.1f}mm (>容許 {tol:.1f}mm)"
                    dropped.append(r)
            pool = gated
    zs = np.array([float(r['p3d'][2]) for r in pool], dtype=np.float64)
    keep = np.ones(len(pool), dtype=bool)
    if len(pool) >= 3:
        med = float(np.median(zs))
        mad = float(np.median(np.abs(zs - med)))
        if mad > 1e-6:
            keep = np.abs(zs - med) <= 3.0 * 1.4826 * mad
            if not np.any(keep):
                keep[:] = True
    kept = [r for r, k in zip(pool, keep) if k]
    for r, k in zip(pool, keep):
        if not k:
            r['drop_reason'] = 'MAD 離群剔除'
            dropped.append(r)

    def _weights(items):
        ws = []
        for r in items:
            b = float(r.get('baseline') or 0.0)
            z = max(float(r['p3d'][2]), 1e-6)
            ws.append((b / (z * z)) ** 2 if b > 0 else 0.0)
        ws = np.array(ws, dtype=np.float64)
        if not np.all(np.isfinite(ws)) or ws.sum() <= 0:
            ws = np.ones(len(items), dtype=np.float64)
        return ws / ws.sum()

    ws = _weights(kept)
    p3d = np.sum(np.array([r['p3d'] for r in kept], dtype=np.float64) * ws[:, None], axis=0)
    kept_w = [r for r in kept if r.get('p3d_w') is not None]
    p3d_w = None
    if kept_w:
        ws_w = _weights(kept_w)
        p3d_w = np.sum(np.array([r['p3d_w'] for r in kept_w], dtype=np.float64) * ws_w[:, None], axis=0)
    return {'p3d': p3d, 'd': float(np.linalg.norm(p3d)), 'p3d_w': p3d_w,
            'kept': kept, 'dropped': dropped, 'weights': ws}


def fit_plane_to_points(pts, ransac_thresh_mm=1.5, ransac_iters=200):
    """
    SVD 平面擬合；點數 >= 6 時先以 RANSAC 剔除離群點（誤匹配的 3D 點）。
    回傳 (n, c, inlier_mask, residuals)，residuals 為所有輸入點到平面的有號距離 (mm)。
    """
    pts = np.asarray(pts, dtype=np.float64)
    inlier_mask = np.ones(len(pts), dtype=bool)
    if len(pts) >= 6:
        best_inliers = None
        rng = np.random.default_rng(0)
        for _ in range(ransac_iters):
            idx = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx]
            n_h = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n_h)
            if norm < 1e-9:
                continue
            n_h = n_h / norm
            d = np.abs((pts - p0) @ n_h)
            inl = d <= ransac_thresh_mm
            if best_inliers is None or inl.sum() > best_inliers.sum():
                best_inliers = inl
        if best_inliers is not None and best_inliers.sum() >= 3:
            inlier_mask = best_inliers
    sub = pts[inlier_mask]
    c = sub.mean(axis=0)
    _, _, Vt = np.linalg.svd(sub - c)
    n = Vt[-1]
    if np.dot(n, c) > 0:
        n = -n  # 法向量朝向相機
    residuals = (pts - c) @ n
    return n, c, inlier_mask, residuals

def apply_dedrift_correction(trajectory, p_end_match):
    """
    對光流軌跡進行閉環去漂移修正
    trajectory: list of (f_idx, [u, v])
    p_end_match: [u, v] 終點影格的最優匹配點真值
    """
    if len(trajectory) < 2:
        return trajectory
    p_end_flow = np.array(trajectory[-1][1])
    total_drift = np.array(p_end_match) - p_end_flow
    corrected_trajectory = []
    n = len(trajectory) - 1
    for i, (f_idx, pt) in enumerate(trajectory):
        factor = i / n
        corr_pt = np.array(pt) + factor * total_drift
        corrected_trajectory.append((f_idx, corr_pt.tolist()))
    return corrected_trajectory

def track_feature_and_verify(all_frames, start_f_idx, end_f_idx, p_start, valid_poses, K_L, dist_L):
    """
    使用雙向光流 (KLT) 與極線幾何硬約束對特徵點進行時序追蹤，並進行極線正交投影校正。
    """
    step = -1 if start_f_idx > end_f_idx else 1
    curr_f_idx = start_f_idx
    curr_pt = np.array(p_start, dtype=np.float32).reshape(-1, 2)
    
    trajectory = [(curr_f_idx, curr_pt[0].tolist())]
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                     
    # 局部快取，避免在多幀光流迴圈中重複將相同的彩色幀轉換成灰階並執行對比度增強
    gray_cache = {}
    def get_processed_gray(idx):
        if idx not in gray_cache:
            g = cv2.cvtColor(all_frames[idx], cv2.COLOR_BGR2GRAY)
            g = preprocess_gray(g, True)
            gray_cache[idx] = g
        return gray_cache[idx]

    while curr_f_idx != end_f_idx:
        next_f_idx = curr_f_idx + step
        
        img_prev = get_processed_gray(curr_f_idx)
        img_next = get_processed_gray(next_f_idx)
        
        # 1. Forward 追蹤
        p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_next, curr_pt, None, **lk_params)
        if st is None or st[0][0] == 0:
            print(f"⚠️ 光流追蹤在影格 {curr_f_idx} -> {next_f_idx} 斷線")
            break
            
        # 2. Backward 追蹤
        p0_re, st_re, _ = cv2.calcOpticalFlowPyrLK(img_next, img_prev, p1, None, **lk_params)
        if st_re is None or st_re[0][0] == 0:
            print(f"⚠️ 雙向光流在影格 {next_f_idx} -> {curr_f_idx} 斷線")
            break
            
        # FB-Consistency 檢查
        fb_err = np.linalg.norm(curr_pt[0] - p0_re[0])
        if fb_err > FLOW_FB_THRESHOLD:
            print(f"⚠️ 雙向光流偏差過大 ({fb_err:.2f} px > {FLOW_FB_THRESHOLD} px)，拒絕影格 {next_f_idx}")
            curr_pt = p1
            curr_f_idx = next_f_idx
            continue
            
        # 3. 極線幾何約束檢查與正交投影校正
        pt_verified = p1[0].copy()
        if curr_f_idx in valid_poses and next_f_idx in valid_poses:
            R_prev, t_prev = valid_poses[curr_f_idx]
            R_next, t_next = valid_poses[next_f_idx]
            
            R_f = R_next @ R_prev.T
            t_f = t_next - R_f @ t_prev
            
            F = compute_fundamental_matrix(K_L, K_L, R_f, t_f)
            l = F @ np.array([curr_pt[0][0], curr_pt[0][1], 1.0])
            a, b, c = l
            denom = a**2 + b**2
            if denom > 1e-9:
                dist_epi = abs(a * pt_verified[0] + b * pt_verified[1] + c) / np.sqrt(denom)
                if dist_epi > EPIPOLAR_DIST_THRESHOLD:
                    print(f"⚠️ 幾何極線檢查失敗 ({dist_epi:.2f} px > {EPIPOLAR_DIST_THRESHOLD} px)，拒絕影格 {next_f_idx}")
                    curr_pt = p1
                    curr_f_idx = next_f_idx
                    continue
                else:
                    # 正交投影校正
                    pt_verified[0] = pt_verified[0] - a * (a * pt_verified[0] + b * pt_verified[1] + c) / denom
                    pt_verified[1] = pt_verified[1] - b * (a * pt_verified[0] + b * pt_verified[1] + c) / denom
                    
        curr_pt = pt_verified.reshape(-1, 2)
        trajectory.append((next_f_idx, curr_pt[0].tolist()))
        curr_f_idx = next_f_idx
        
    return trajectory

def analyze_video_frames(video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm, select_mode="average", range_mode="fixed", progress_callback=None):
    video_pose_algo.log_and_print = log_and_print
    video_pose_algo.RECORD_SAVE_DIR = RECORD_SAVE_DIR
    video_pose_algo.MIN_BASELINE_MM = MIN_BASELINE_MM
    video_pose_algo.MAX_BASELINE_MM = MAX_BASELINE_MM
    video_pose_algo.IDEAL_BASELINE_MM = IDEAL_BASELINE_MM
    video_pose_algo.PAIR_SCORE_REPROJ_W = PAIR_SCORE_REPROJ_W
    video_pose_algo.PAIR_SCORE_BASELINE_W = PAIR_SCORE_BASELINE_W
    video_pose_algo.PAIR_SCORE_BLUR_W = PAIR_SCORE_BLUR_W
    video_pose_algo.PAIR_SCORE_COVER_W = PAIR_SCORE_COVER_W
    video_pose_algo.PAIR_SCORE_MARKER_W = PAIR_SCORE_MARKER_W
    video_pose_algo.preprocess_gray = preprocess_gray
    video_pose_algo.average_rotations_svd = average_rotations_svd
    return video_pose_algo.analyze_video_frames(
        video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm,
        select_mode, range_mode, progress_callback=progress_callback
    )

_clahe_cache = {}

def get_clahe(clip_limit, tile_size):
    return camera_algo.get_clahe(clip_limit, tile_size)

def preprocess_gray(gray_img, enable_clahe=True):
    return camera_algo.preprocess_gray(gray_img, enable_clahe, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    return aruco_algo.compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=log_and_print)

def get_joint_relative_pose(imgA_gray, imgB_gray, K_L, K_R, marker_size_mm, global_plane_n=None, global_plane_c=None, prev_marker_poses=None, prev_rel_pose=None, marker_map=None, map_calibrated=False):
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cA, idsA, _ = detector.detectMarkers(imgA_gray)
        cB, idsB, _ = detector.detectMarkers(imgB_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
        cB, idsB, _ = cv2.aruco.detectMarkers(imgB_gray, dict_4x4, parameters=params)
    
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    if cA is not None:
        for c in cA: cv2.cornerSubPix(imgA_gray, c, (3, 3), (-1, -1), term)
    if cB is not None:
        for c in cB: cv2.cornerSubPix(imgB_gray, c, (3, 3), (-1, -1), term)

    if idsA is None or idsB is None: return None, map_calibrated
    idsA_l, idsB_l = [i[0] for i in idsA], [i[0] for i in idsB]
    shared = list(set(idsA_l).intersection(set(idsB_l)))
    if not shared: return None, map_calibrated
    
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    cA_dict, cB_dict = {}, {}
    for mid in shared:
        idxA, idxB = idsA_l.index(mid), idsB_l.index(mid)
        cA_dict[mid], cB_dict[mid] = cA[idxA][0], cB[idxB][0]

    # 計算共享標籤角點在左右圖之間的平均像素位移 (視差)
    all_dists = []
    for mid in shared:
        if mid in cA_dict and mid in cB_dict:
            dists = np.linalg.norm(cA_dict[mid] - cB_dict[mid], axis=1)
            all_dists.extend(dists)
    mean_disparity = np.mean(all_dists) if all_dists else 0.0
    
    if mean_disparity < 2.0:
        print(f"⚠️ [外參解算] 左右圖平均像素位移過小 ({mean_disparity:.2f} px < 2.0 px)，判定為無視差退化狀態，跳過此幀。")
        return None, map_calibrated

    # ---------------- 1. 全域標籤地圖在線自標定 (Map Auto-Calibration) ----------------
    if marker_map is not None and not map_calibrated and len(shared) >= 2:
        print("🛠️ 偵測到多個標籤，開始進行全域標籤地圖在線自標定...")
        marker_poses_L = {}
        for mid in shared:
            idxA = idsA_l.index(mid)
            ok, rv, tv = cv2.solvePnP(canon, cA[idxA][0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                marker_poses_L[mid] = (rv, tv)
        
        if len(marker_poses_L) >= 2:
            min_id = min(marker_poses_L.keys())
            rv_ref, tv_ref = marker_poses_L[min_id]
            R_ref, _ = cv2.Rodrigues(rv_ref)
            
            marker_map[min_id] = (np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
            
            for mid, (rv_m, tv_m) in marker_poses_L.items():
                if mid == min_id: continue
                R_m, _ = cv2.Rodrigues(rv_m)
                R_m2ref = R_ref.T @ R_m
                T_m2ref = R_ref.T @ (tv_m - tv_ref)
                marker_map[mid] = (R_m2ref.astype(np.float32), T_m2ref.astype(np.float32))
                print(f"  - 標定標籤 {mid} 到世界原點標籤 {min_id} 的相對平移: {T_m2ref.flatten()} mm")
            
            map_calibrated = True
            print("✅ 全域標籤地圖在線自標定完成！")

    # ---------------- 2. 多標籤聯合 PnP 求解 (Compound PnP) ----------------
    if marker_map is not None and map_calibrated:
        active_mids = [mid for mid in shared if mid in marker_map]
        if len(active_mids) > 0:
            joint_objW = []
            joint_imgA = []
            joint_imgB = []
            
            for mid in active_mids:
                idxA = idsA_l.index(mid)
                idxB = idsB_l.index(mid)
                R_m2o, T_m2o = marker_map[mid]
                pts_W = (R_m2o @ canon.T).T + T_m2o.T
                joint_objW.append(pts_W)
                joint_imgA.append(cA[idxA][0])
                joint_imgB.append(cB[idxB][0])
            
            joint_objW = np.vstack(joint_objW).astype(np.float32)
            joint_imgA = np.vstack(joint_imgA).astype(np.float32)
            joint_imgB = np.vstack(joint_imgB).astype(np.float32)
            
            min_id = min(active_mids)
            idxA_min = idsA_l.index(min_id)
            idxB_min = idsB_l.index(min_id)
            
            ok_L_init, rv_L_init, tv_L_init = cv2.solvePnP(canon, cA[idxA_min][0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            ok_R_init, rv_R_init, tv_R_init = cv2.solvePnP(canon, cB[idxB_min][0], K_R, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            if ok_L_init and ok_R_init:
                ok_L, rv_L, tv_L = cv2.solvePnP(joint_objW, joint_imgA, K_L, np.zeros(5), rvec=rv_L_init.copy().astype(np.float32), tvec=tv_L_init.copy().astype(np.float32), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                ok_R, rv_R, tv_R = cv2.solvePnP(joint_objW, joint_imgB, K_R, np.zeros(5), rvec=rv_R_init.copy().astype(np.float32), tvec=tv_R_init.copy().astype(np.float32), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok_L, rv_L, tv_L = cv2.solvePnP(joint_objW, joint_imgA, K_L, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
                ok_R, rv_R, tv_R = cv2.solvePnP(joint_objW, joint_imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
            
            if ok_L and ok_R:
                R_L, _ = cv2.Rodrigues(rv_L)
                R_R, _ = cv2.Rodrigues(rv_R)
                R_rel = R_R @ R_L.T
                t_rel = tv_R - R_rel @ tv_L
                baseline_val = float(np.linalg.norm(t_rel))
                
                curr_marker_poses = {}
                for mid in active_mids:
                    R_m2o, T_m2o = marker_map[mid]
                    R_m_L = R_L @ R_m2o
                    T_m_L = R_L @ T_m2o + tv_L
                    rv_m_L, _ = cv2.Rodrigues(R_m_L)
                    curr_marker_poses[mid] = (rv_m_L, T_m_L)
                
                pts_C_L = (R_L @ joint_objW.T).T + tv_L.T
                rv_rel, _ = cv2.Rodrigues(R_rel)
                
                cA_dict_sub = {mid: cA_dict[mid] for mid in active_mids}
                cB_dict_sub = {mid: cB_dict[mid] for mid in active_mids}
                
                return (R_rel, t_rel, baseline_val, pts_C_L, active_mids, cA_dict_sub, cB_dict_sub, curr_marker_poses, rv_rel), map_calibrated

    # ---------------- 3. 降級方案 (獨立解算 / 原 IPPE 算法) ----------------
    objA, imgB = [], []
    curr_marker_poses = {}
    
    for mid in shared:
        idxA, idxB = idsA_l.index(mid), idsB_l.index(mid)
        ok, rv, tv = cv2.solvePnP(canon, cA[idxA][0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            curr_marker_poses[mid] = (rv, tv)
            R, _ = cv2.Rodrigues(rv)
            objA.append((R @ canon.T).T + tv.T)
            imgB.append(cB[idxB][0])

    if not objA: return None, map_calibrated
    objA = np.vstack(objA).astype(np.float32)
    imgB = np.vstack(imgB).astype(np.float32)

    # 全域共面對齊優化 (Global Coplanar Refinement)
    if ENFORCE_COPLANAR and global_plane_n is not None and global_plane_c is not None:
        normal = global_plane_n
        d_val = np.dot(normal, global_plane_c)
        K_L_inv = np.linalg.inv(K_L.astype(np.float64))
        refined_objA = []
        for mid in shared:
            pts_2d = cA_dict[mid]
            rays = np.hstack([pts_2d, np.ones((4, 1))]) @ K_L_inv.T
            t_vals = d_val / (rays @ normal)
            refined_objA.append(rays * t_vals[:, np.newaxis])
        objA = np.vstack(refined_objA).astype(np.float32)

    use_guess_rel = False
    rv_rel_init, tv_rel_init = None, None
    if prev_rel_pose is not None:
        rv_rel_init = prev_rel_pose[0].copy().astype(np.float32)
        tv_rel_init = prev_rel_pose[1].copy().astype(np.float32)
        use_guess_rel = True

    ok_rel = False
    rv_rel = None
    tv_rel = None

    if len(shared) == 1:
        if ENABLE_SIFT_PNP_ASSIST:
            # 1. 首先解出一個粗略的相對姿態做為初值與極線、單應性參考
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_rel:
                R_rel_init, _ = cv2.Rodrigues(rv_rel)
                
                # 計算該單個標籤的世界平面法向量 n 與中心 c
                v1 = objA[1] - objA[0]
                v2 = objA[3] - objA[0]
                n_plane = np.cross(v1, v2)
                n_norm = np.linalg.norm(n_plane)
                if n_norm > 1e-6:
                    n_plane = n_plane / n_norm
                    if n_plane[2] > 0:
                        n_plane = -n_plane
                    c_plane = np.mean(objA, axis=0)
                    d_plane = np.dot(n_plane, c_plane)
                    
                    H, W = imgA_gray.shape
                    mid_label = shared[0]
                    corners_A = cA_dict[mid_label]
                    center_2d = np.mean(corners_A, axis=0)
                    
                    # 定義左圖 ROI (1/3 影片大小) 與右圖較大的檢測區 (1/2 影片大小)
                    roi_w, roi_h = int(W / 3), int(H / 3)
                    x_min = max(0, int(center_2d[0] - roi_w / 2))
                    x_max = min(W, int(center_2d[0] + roi_w / 2))
                    y_min = max(0, int(center_2d[1] - roi_h / 2))
                    y_max = min(H, int(center_2d[1] + roi_h / 2))
                    
                    roi_w_R, roi_h_R = int(W / 2), int(H / 2)
                    x_min_R = max(0, int(center_2d[0] - roi_w_R / 2))
                    x_max_R = min(W, int(center_2d[0] + roi_w_R / 2))
                    y_min_R = max(0, int(center_2d[1] - roi_h_R / 2))
                    y_max_R = min(H, int(center_2d[1] + roi_h_R / 2))
                    
                    # 提取 SIFT
                    sift_pnp = cv2.SIFT_create(contrastThreshold=0.005)
                    roi_imgA = imgA_gray[y_min:y_max, x_min:x_max]
                    kps_A_sub, des_A = sift_pnp.detectAndCompute(roi_imgA, None)
                    
                    roi_imgB = imgB_gray[y_min_R:y_max_R, x_min_R:x_max_R]
                    kps_B_sub, des_B = sift_pnp.detectAndCompute(roi_imgB, None)
                    
                    if (kps_A_sub is not None and len(kps_A_sub) > 0 and 
                        kps_B_sub is not None and len(kps_B_sub) > 0 and 
                        des_A is not None and des_B is not None):
                        
                        kps_A = [cv2.KeyPoint(kp.pt[0] + x_min, kp.pt[1] + y_min, kp.size) for kp in kps_A_sub]
                        kps_B = [cv2.KeyPoint(kp.pt[0] + x_min_R, kp.pt[1] + y_min_R, kp.size) for kp in kps_B_sub]
                        
                        K_L_inv = np.linalg.inv(K_L.astype(np.float64))
                        H_AB_init = K_R @ (R_rel_init + (tv_rel @ n_plane.reshape(1, 3)) / d_plane) @ K_L_inv
                        
                        t = tv_rel.flatten()
                        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
                        E = tx @ R_rel_init
                        K_R_inv = np.linalg.inv(K_R.astype(np.float64))
                        F_init = K_R_inv.T @ E @ K_L_inv
                        
                        bf = cv2.BFMatcher()
                        matches = bf.knnMatch(des_A, des_B, k=2)
                        
                        extra_objA = []
                        extra_imgB = []
                        
                        for m, n_match in matches:
                            # 第一重：Lowe's Ratio Test (<0.6)
                            if m.distance < 0.6 * n_match.distance:
                                pt_L = np.array(kps_A[m.queryIdx].pt)
                                pt_R = np.array(kps_B[m.trainIdx].pt)
                                
                                # 第二重：平面單應性距離約束 (< 40px)
                                pt_L_h = np.array([pt_L[0], pt_L[1], 1.0])
                                pt_R_proj_h = H_AB_init @ pt_L_h
                                if abs(pt_R_proj_h[2]) > 1e-6:
                                    pt_R_proj = np.array([pt_R_proj_h[0]/pt_R_proj_h[2], pt_R_proj_h[1]/pt_R_proj_h[2]])
                                    if np.linalg.norm(pt_R - pt_R_proj) < 40.0:
                                        
                                        # 第三重：極線距離約束 (< 2.0px)
                                        l_R = F_init @ pt_L_h
                                        denom_epi = l_R[0]**2 + l_R[1]**2
                                        if denom_epi > 1e-9:
                                            dist_epi = abs(l_R[0]*pt_R[0] + l_R[1]*pt_R[1] + l_R[2]) / np.sqrt(denom_epi)
                                            if dist_epi < 2.0:
                                                
                                                # 第四重：共面反投影得到 3D 點
                                                ray = K_L_inv @ pt_L_h
                                                denom_ray = np.dot(n_plane, ray)
                                                if abs(denom_ray) > 1e-6:
                                                    lambda_val = d_plane / denom_ray
                                                    if lambda_val > 0:
                                                        pt_3D = ray * lambda_val
                                                        extra_objA.append(pt_3D)
                                                        extra_imgB.append(pt_R)
                        
                        # 第五重：RANSAC 與二次精修
                        if len(extra_objA) >= 8:
                            total_objA = np.vstack([objA, np.array(extra_objA, dtype=np.float32)])
                            total_imgB = np.vstack([imgB, np.array(extra_imgB, dtype=np.float32)])
                            
                            ok_ransac, rv_ransac, tv_ransac, inliers = cv2.solvePnPRansac(
                                total_objA, total_imgB, K_R, np.zeros(5),
                                reprojectionError=2.0, iterationsCount=150, flags=cv2.SOLVEPNP_ITERATIVE
                            )
                            
                            if ok_ransac and inliers is not None and len(inliers) >= 6:
                                inliers = inliers.flatten()
                                objA_inliers = total_objA[inliers]
                                imgB_inliers = total_imgB[inliers]
                                
                                ok_refine, rv_refine, tv_refine = cv2.solvePnP(
                                    objA_inliers, imgB_inliers, K_R, np.zeros(5),
                                    rvec=rv_ransac, tvec=tv_ransac, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                                )
                                if ok_refine:
                                    rv_rel, tv_rel = rv_refine, tv_refine
                                    ok_rel = True
                                    print(f"🚀 [SIFT-PnP 輔助成功] 使用 {len(inliers)} 個內點二次精修相對外參，Baseline: {np.linalg.norm(tv_rel):.2f} mm")
            
            # 安全降級：如果 SIFT 輔助解算未成功，則保留最初解出的粗估位姿
            if not ok_rel:
                print("⚠️ [SIFT-PnP 輔助未成功或點數不足] 降級使用純 ArUco 角點的初始 PnP 解")
        else:
            # 初始 solvePnP 也失敗，直接降級
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        # 共享標籤大於 1 個時
        if use_guess_rel:
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), rvec=rv_rel_init, tvec=tv_rel_init, useExtrinsicGuess=True)
        else:
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok_rel: return None, map_calibrated
    R_rel, _ = cv2.Rodrigues(rv_rel)
    return (R_rel, tv_rel, float(np.linalg.norm(tv_rel)), objA, shared, cA_dict, cB_dict, curr_marker_poses, rv_rel), map_calibrated


def snap_to_aruco_corner(x, y, corners_dict):
    pt = np.array([x, y])
    for corners in corners_dict.values():
        dists = np.linalg.norm(corners - pt, axis=1)
        if np.min(dists) < 15:
            return float(corners[np.argmin(dists)][0]), float(corners[np.argmin(dists)][1])
    return x, y

def record_video_from_camera():
    import datetime
    # 建立影片儲存資料夾（如果不存在的話）
    save_path = RECORD_SAVE_DIR
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 開啟相機
    cap = cv2.VideoCapture(0)#, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟相機")
        return None

    # 設定相機解析度為 1920x1080 且設定編碼格式
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 目前接收到的串流解析度: {width} x {height}")
    print("操作說明：")
    print("  按下 's' 鍵 - 開始/停止錄影")
    print("  按下 'q' 鍵 - 當錄影完成後，結束預覽並載入影片")

    is_recording = False
    video_writer = None
    video_name = None
    has_recorded = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法接收畫面，錄影中斷...")
            break

        # 錄影寫入
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # 顯示錄影狀態指示
        if is_recording:
            cv2.circle(display_frame, (30, h - 30), 15, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (55, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 'S' to STOP recording", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            if has_recorded:
                cv2.putText(display_frame, "Recorded! Press 'Q' to start depth measure or 'S' to re-record", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Press 'S' to START recording", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # 4. 偵測按鍵事件
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            if has_recorded and not is_recording:
                log_and_print(f"🎬 錄影就緒，準備載入: {video_name}")
                break
            elif is_recording:
                print("⚠️ 正在錄影中，請先按 's' 停止錄影後再按 'q' 離開。")
            else:
                print("⚠️ 尚未錄製任何影片，請按 's' 錄製一段影片。")
        
        elif key == ord('s'):
            if not is_recording:
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_name = os.path.join(save_path, f"video_{now_str}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 100: fps = 25.0  # 保底 FPS
                video_writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))
                is_recording = True
                has_recorded = True
                print(f"🎬 開始錄影：{video_name}")
            else:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("🛑 錄影結束")
        
        # 縮放預覽，避免影像太大
        display_small = cv2.resize(display_frame, (int(w//2), int(h//2)))
        cv2.imshow('Camera Recording Window', display_small)

    cap.release()
    cv2.destroyAllWindows()
    return video_name

def save_measurement_to_txt(video_path, res, cand, wound_z_offset, custom_plane_n, custom_plane_c, custom_plane_fitted, measure_mode):
    import datetime
    if video_path is None:
        return
    txt_path = os.path.splitext(video_path)[0] + ".txt"
    
    # 取得現在時間
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    u, v = res['u'], res['v']
    m_pt = res['pt']
    m_pt_raw = res.get('pt_raw')
    p3d = res['p3d']
    p3d_best = res.get('p3d_best') if res.get('p3d_best') is not None else p3d
    p3d_w = res.get('p3d_w')
    d_val = res['d']
    method = res['method']
    fail_reason = res.get('fail_reason', '')
    
    m_pt_str = f"({m_pt[0]:.2f}, {m_pt[1]:.2f})" if m_pt is not None else "None"
    m_pt_raw_str = f"({m_pt_raw[0]:.2f}, {m_pt_raw[1]:.2f})" if m_pt_raw is not None else "None"
    p3d_str = f"[{p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f}]" if p3d is not None else "None"
    p3d_w_str = f"[{p3d_w[0]:.2f}, {p3d_w[1]:.2f}, {p3d_w[2]:.2f}]" if p3d_w is not None else "None"
    p3d_w_offset_str = f"[{p3d_w[0]:.2f}, {p3d_w[1]:.2f}, {p3d_w[2] + wound_z_offset:.2f}]" if p3d_w is not None else "None"
    d_str = f"{d_val:.2f} mm" if d_val is not None else "None"
    
    # 距標記平面深度 (signed：Above 表示在平面靠相機側)
    p_dist_str = "None"
    if p3d_best is not None and cand.get('plane_n') is not None and cand.get('plane_c') is not None:
        p_dist = np.dot(cand['plane_n'], p3d_best - cand['plane_c'])
        p_dist_str = f"{'Above' if p_dist > 0 else 'Below'} {abs(p_dist):.2f} mm"
        
    # 距自訂平面深度
    cp_dist_str = "None"
    proj_dist_signed = 0.0
    if custom_plane_fitted and p3d is not None and custom_plane_n is not None and custom_plane_c is not None:
        proj_dist_signed = np.dot(custom_plane_n, p3d - custom_plane_c)
        status = "Above" if proj_dist_signed > 0 else "Below"
        cp_dist_str = f"{status} {abs(proj_dist_signed):.2f} mm"
        
    # 幾何外參
    R_rel = cand.get('R_rel')
    t_rel = cand.get('t_rel')
    baseline = cand.get('baseline')
    idx_B = cand.get('idx')
    
    R_rel_str = np.array2string(R_rel, precision=6, separator=', ', suppress_small=True) if R_rel is not None else "None"
    t_rel_str = np.array2string(t_rel.flatten(), precision=6, separator=', ', suppress_small=True) if t_rel is not None else "None"
    
    # 內參
    K_R = cand.get('K_R')
    K_R_str = np.array2string(K_R, precision=6, separator=', ', suppress_small=True) if K_R is not None else "None"
    
    # 格式化人讀文字
    text_lines = []
    text_lines.append("==================================================")
    text_lines.append(f"時間戳記: {now_str}")
    text_lines.append(f"量測狀態: {'成功' if d_val is not None else '失敗'}")
    if d_val is None:
        text_lines.append(f"失敗原因: {fail_reason}")
    text_lines.append(f"左圖點擊座標 (u, v): ({u:.2f}, {v:.2f})")
    text_lines.append(f"右圖匹配座標 (u_R, v_R): {m_pt_str}")
    text_lines.append(f"右圖原始匹配座標 (u_R_raw, v_R_raw): {m_pt_raw_str}")
    text_lines.append(f"匹配演算法: {method}")
    text_lines.append(f"量測模式: {measure_mode}")
    text_lines.append(f"歐式距離 (d): {d_str}")
    text_lines.append(f"相機 3D 座標 (X_c, Y_c, Z_c): {p3d_str}")
    text_lines.append(f"世界 3D 座標 (X_w, Y_w, Z_w): {p3d_w_str}")
    text_lines.append(f"世界 3D 座標 (含傷口高度補償 {wound_z_offset:.2f} mm): {p3d_w_offset_str}")
    text_lines.append(f"距標記平面深度: {p_dist_str}")
    text_lines.append(f"自訂平面擬合狀態: {'已擬合' if custom_plane_fitted else '未擬合'}")
    text_lines.append(f"距自訂平面深度: {cp_dist_str}")
    text_lines.append(f"左圖影格索引: {idx_B}")
    text_lines.append(f"基準線 (Baseline): {baseline:.2f} mm" if baseline is not None else "基準線 (Baseline): None")
    text_lines.append(f"相對平移向量 (T_rel): {t_rel_str}")
    text_lines.append(f"相對旋轉矩陣 (R_rel):\n{R_rel_str}")
    text_lines.append(f"相機內參 (KL):\n{K_R_str}")
    if "multi_res" in res:
        text_lines.append("--------------------------------------------------")
        text_lines.append(f"多對融合結果 (共 {len(res['multi_res'])} 組成功):")
        for sub in res['multi_res']:
            is_best = " (最優)" if sub['cand_idx'] == cand.get('idx') else ""
            text_lines.append(f"  - 右圖 F{sub['cand_idx']}{is_best}: 深度 = {sub['d']:.2f} mm, 3D = [{sub['p3d'][0]:.2f}, {sub['p3d'][1]:.2f}, {sub['p3d'][2]:.2f}]")
    text_lines.append("==================================================")
    
    # 格式化機讀 JSON
    import json
    json_data = {
        "timestamp": now_str,
        "status": "success" if d_val is not None else "failed",
        "fail_reason": fail_reason,
        "u": float(u),
        "v": float(v),
        "u_R": float(m_pt[0]) if m_pt is not None else None,
        "v_R": float(m_pt[1]) if m_pt is not None else None,
        "u_R_raw": float(m_pt_raw[0]) if m_pt_raw is not None else None,
        "v_R_raw": float(m_pt_raw[1]) if m_pt_raw is not None else None,
        "method": method,
        "measure_mode": measure_mode,
        "d_mm": float(d_val) if d_val is not None else None,
        "p3d_camera": p3d.tolist() if p3d is not None else None,
        "p3d_world": p3d_w.tolist() if p3d_w is not None else None,
        "p3d_world_compensated": [float(p3d_w[0]), float(p3d_w[1]), float(p3d_w[2] + wound_z_offset)] if p3d_w is not None else None,
        "wound_z_offset_mm": float(wound_z_offset),
        "dist_to_marker_plane_mm": float(p_dist) if (p3d is not None and cand.get('plane_n') is not None and cand.get('plane_c') is not None) else None,
        "custom_plane_fitted": bool(custom_plane_fitted),
        "dist_to_custom_plane_mm": float(abs(proj_dist_signed)) if (custom_plane_fitted and p3d is not None and custom_plane_n is not None and custom_plane_c is not None) else None,
        "idx_left_frame": int(idx_B) if idx_B is not None else None,
        "baseline_mm": float(baseline) if baseline is not None else None,
        "t_rel": t_rel.flatten().tolist() if t_rel is not None else None,
        "R_rel": R_rel.tolist() if R_rel is not None else None,
        "KL": K_R.tolist() if K_R is not None else None
    }
    if "multi_res" in res:
        json_data["multi_fusion"] = {
            "num_successful_pairs": len(res['multi_res']),
            "details": [
                {
                    "cand_idx": int(sub['cand_idx']),
                    "is_best": bool(sub['cand_idx'] == cand.get('idx')),
                    "d_mm": float(sub['d']),
                    "p3d_camera": sub['p3d'].tolist()
                }
                for sub in res['multi_res']
            ]
        }
    
    # 寫入檔案
    try:
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(text_lines) + "\n")
            f.write("JSON: " + json.dumps(json_data, ensure_ascii=False) + "\n\n")
        print(f"💾 量測數據已儲存至: {txt_path}")
    except Exception as e:
        print(f"❌ 儲存量測數據失敗: {e}")

def ui_progress_status_english(status_text):
    text = str(status_text or "")
    replacements = (
        ("階段處理中", "Processing stage"),
        ("階段", "Stage"),
        ("：", ": "),
        ("載入影片", "Loading video"),
        ("載入完成", "Loading complete"),
        ("分析影像", "Analyzing frames"),
        ("影像校正", "Calibrating images"),
        ("基準計算", "Computing baseline"),
        ("資料準備", "Preparing data"),
        ("完成", "Complete"),
        ("準備開始", "Preparing"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text

def ui_failure_reason_english(reason):
    text = str(reason or "No matching point")
    replacements = (
        ("未偵測到 ArUco", "ArUco not detected"),
        ("視差不合規範", "Baseline out of range"),
        ("追蹤影格數不足", "Not enough tracked frames"),
        ("三角化失敗", "Triangulation failed"),
        ("無匹配點", "No matching point"),
        ("深度為負(在相機後方)", "Negative depth (behind camera)"),
        ("超過最大深度", "Exceeds max depth"),
        ("匹配點偏離RT/平面預測", "Match point deviates from RT/plane prediction"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = text.replace("Baseline out of range(", "Baseline out of range (")
    return text

def select_video_source():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.title("Interactive Measurement - Select Video Source")
    
    # 設置視窗大小與置中
    window_width = 450
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_c = (screen_width - window_width) // 2
    y_c = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_c}+{y_c}")
    root.configure(bg="#2D2D2D")
    root.resizable(False, False)

    # 設置字型與樣式
    title_font = ("Microsoft JhengHei", 12, "bold")
    btn_font = ("Microsoft JhengHei", 10, "bold")

    # 回傳變數
    selected_path = {"path": None, "action": None}

    # 標題
    title_label = tk.Label(root, text="Choose a video source for depth measurement:", font=title_font, fg="#FFFFFF", bg="#2D2D2D", pady=25)
    title_label.pack()

    def on_camera():
        selected_path["action"] = "camera"
        root.destroy()

    def on_file():
        selected_path["action"] = "file"
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
        )
        if file_path:
            selected_path["path"] = file_path
            root.destroy()
        else:
            selected_path["action"] = None

    # 按鈕容器
    btn_frame = tk.Frame(root, bg="#2D2D2D")
    btn_frame.pack(pady=5)

    btn_cam = tk.Button(
        btn_frame, 
        text="📷 Record from Camera", 
        font=btn_font, 
        command=on_camera, 
        bg="#007ACC", 
        fg="#FFFFFF", 
        activebackground="#005A9E", 
        activeforeground="#FFFFFF",
        width=18,
        height=2,
        relief="flat"
    )
    btn_cam.pack(side="left", padx=15)

    btn_file = tk.Button(
        btn_frame, 
        text="📁 Load Video File", 
        font=btn_font, 
        command=on_file, 
        bg="#28A745", 
        fg="#FFFFFF", 
        activebackground="#1E7E34", 
        activeforeground="#FFFFFF",
        width=18,
        height=2,
        relief="flat"
    )
    btn_file.pack(side="right", padx=15)

    root.mainloop()

    return selected_path["action"], selected_path["path"]

def analyze_video_with_progress_bar(video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm, select_mode, range_mode):
    import tkinter as tk
    from tkinter import ttk
    import threading
    import queue

    root = tk.Tk()
    root.title("Analysis Progress")
    
    # 視窗置中
    window_width = 450
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_c = (screen_width - window_width) // 2
    y_c = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_c}+{y_c}")
    root.configure(bg="#2D2D2D")
    root.resizable(False, False)

    # 狀態文字與進度變數
    status_var = tk.StringVar(value="Preparing...")
    progress_var = tk.DoubleVar(value=0.0)

    # UI 元件
    title_label = tk.Label(root, text="🎥 Processing...", font=("Microsoft JhengHei", 12, "bold"), fg="#FFFFFF", bg="#2D2D2D", pady=10)
    title_label.pack()

    status_label = tk.Label(root, textvariable=status_var, font=("Microsoft JhengHei", 10), fg="#E0E0E0", bg="#2D2D2D", wraplength=400)
    status_label.pack(pady=5)

    # 美化進度條樣式
    style = ttk.Style()
    style.theme_use('default')
    style.configure("TProgressbar", thickness=15, troughcolor="#404040", background="#28A745")
    
    progress_bar = ttk.Progressbar(root, length=380, mode="determinate", variable=progress_var, style="TProgressbar")
    progress_bar.pack(pady=10)

    # thread 安全的更新機制
    update_queue = queue.Queue()

    def progress_callback(percent, status_text):
        update_queue.put((percent, ui_progress_status_english(status_text)))

    # 用於儲存執行結果的字典
    result_container = {"data": None, "error": None}

    def worker():
        try:
            res = analyze_video_frames(
                video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm, 
                select_mode, range_mode, progress_callback=progress_callback
            )
            result_container["data"] = res
        except Exception as e:
            result_container["error"] = e
        finally:
            update_queue.put("DONE")

    # 啟動背景計算線程
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # 定期檢查 Queue 並更新 UI
    def check_queue():
        try:
            while True:
                msg = update_queue.get_nowait()
                if msg == "DONE":
                    root.destroy()
                    return
                else:
                    percent, text = msg
                    progress_var.set(percent)
                    status_var.set(text)
                    root.update_idletasks()
        except queue.Empty:
            pass
        root.after(100, check_queue)

    root.after(100, check_queue)
    root.mainloop()

    if result_container["error"]:
        raise result_container["error"]
    return result_container["data"]


def precompute_masks_with_progress_window(extra_cands, compute_masks_fn):
    """次佳影格高光遮罩預計算：顯示獨立讀取條視窗（階段處理中），算完自動關閉。"""
    if not extra_cands:
        return
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Video Analysis Progress")

    window_width = 450
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_c = (screen_width - window_width) // 2
    y_c = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_c}+{y_c}")
    root.configure(bg="#2D2D2D")
    root.resizable(False, False)

    status_var = tk.StringVar(value="Processing stage...")
    progress_var = tk.DoubleVar(value=0.0)

    title_label = tk.Label(root, text="🎥 Processing...", font=("Microsoft JhengHei", 12, "bold"), fg="#FFFFFF", bg="#2D2D2D", pady=10)
    title_label.pack()
    status_label = tk.Label(root, textvariable=status_var, font=("Microsoft JhengHei", 10), fg="#E0E0E0", bg="#2D2D2D", wraplength=400)
    status_label.pack(pady=5)

    style = ttk.Style()
    style.theme_use('default')
    style.configure("TProgressbar", thickness=15, troughcolor="#404040", background="#28A745")
    progress_bar = ttk.Progressbar(root, length=380, mode="determinate", variable=progress_var, style="TProgressbar")
    progress_bar.pack(pady=10)

    update_queue = queue.Queue()

    def worker():
        try:
            total = len(extra_cands)
            for i, cand_bg in enumerate(extra_cands):
                update_queue.put(((i / total) * 100.0, f"Processing stage ({i + 1}/{total})..."))
                try:
                    if cand_bg.get('spec_mask') is None and cand_bg.get('rgb') is not None:
                        _m, _sm, _tm = compute_masks_fn(
                            cv2.cvtColor(cand_bg['rgb'], cv2.COLOR_RGB2BGR), cand_bg.get('idx'))
                        cand_bg['spec_mask'] = _m
                        cand_bg['spec_spatial_mask'] = _sm
                        cand_bg['spec_temporal_mask'] = _tm
                except Exception as exc:
                    print(f"⚠️ [遮罩預計算] F{cand_bg.get('idx')} 失敗: {exc}")
        finally:
            update_queue.put("DONE")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def check_queue():
        try:
            while True:
                msg = update_queue.get_nowait()
                if msg == "DONE":
                    root.destroy()
                    return
                percent, text = msg
                progress_var.set(percent)
                status_var.set(text)
                root.update_idletasks()
        except queue.Empty:
            pass
        root.after(100, check_queue)

    root.after(100, check_queue)
    root.mainloop()
    print("✅ [遮罩預計算] 次佳影格高光遮罩預計算完成")


def main():
    import collections, time
    
    # 0. 選擇影片來源 (UI 視窗)
    action, selected_path = select_video_source()
    if action is None:
        print("❌ 未選擇任何影片來源，程式結束。")
        sys.exit(0)
        
    global VIDEO_PATH
    if action == "camera":
        recorded_path = record_video_from_camera()
        if recorded_path is None or not os.path.exists(recorded_path):
            print("❌ 錄影失敗或未錄製影片，程式結束。")
            sys.exit(1)
        VIDEO_PATH = recorded_path
    else:
        if not selected_path or not os.path.exists(selected_path):
            print("❌ 載入檔案無效或取消選取，程式結束。")
            sys.exit(0)
        VIDEO_PATH = selected_path
        log_and_print(f"📂 已載入指定影片：{VIDEO_PATH}")
    
    startup_timer = StageTimer("啟動流程 (選定影片 → UI 就緒)")

    # 1. 讀取相機內參
    mtxL_o, distL, mtxR_o, distR, extrinsic, F_orig = camera_algo.load_json_camera_params(PARAMS_JSON_PATH)
    
    # 由於去畸變時需要影像尺寸，我們先用 VideoCapture 打開影片讀取第一影格取得原影像寬高
    cap_temp = cv2.VideoCapture(VIDEO_PATH)
    if not cap_temp.isOpened():
        print(f"❌ 無法開啟影片檔: {VIDEO_PATH}")
        sys.exit(1)
    ret, first_frame = cap_temp.read()
    cap_temp.release()
    if not ret:
        print("❌ 無法讀取影片首影格")
        sys.exit(1)
        
    h_raw, w_raw = first_frame.shape[:2]
    w_alg = w_raw
    active_u = w_alg // 2
    active_v = h_raw // 2
    
    newKL_o, _map1, _map2, process_view = camera_algo.build_undistort_processor(
        mtxL_o, distL, (w_alg, h_raw), alpha=1.0
    )
    KL = newKL_o.copy().astype(np.float64)
    startup_timer.stage("相機參數+去畸變映射表")
    
    # 預先建立去畸變查找表
    log_and_print("🔄 正在分析影片中開頭與結尾影格的 ArUco 標籤與最優姿態對...")
    video_data = analyze_video_with_progress_bar(VIDEO_PATH, START_FRAME_COUNT, END_FRAME_COUNT, KL, distL, mtxL_o, ACTUAL_MARKER_SIZE_MM, POSE_SELECT_MODE, FRAME_RANGE_MODE)
    if video_data is None:
        print("❌ 影片 Pose 分析失敗，無法啟動測量工具")
        sys.exit(1)
    startup_timer.stage("影片分析(ArUco配對+RT解算)")

    use_wound_adaptive_spatial_specular = True

    def compute_wound_adaptive_spatial_mask(bgr, wound_prediction=None):
        if use_wound_adaptive_spatial_specular and wound_prediction is not None:
            wound_mask = prediction_to_wound_mask(wound_prediction, bgr.shape)
            return compute_specular_mask_bgr_wound_adaptive(bgr, wound_mask)
        return None

    def compute_locked_spec_mask(bgr, frame_idx=None, wound_prediction=None):
        combined_mask, spatial_mask, temporal_mask = compute_locked_spec_masks(bgr, frame_idx, wound_prediction)
        return combined_mask

    def compute_locked_spec_masks(bgr, frame_idx=None, wound_prediction=None):
        combined_mask, spatial_mask, temporal_mask = compute_rt_aligned_temporal_specular_mask_bgr(
            bgr,
            frame_idx,
            video_data,
            KL,
            ACTUAL_MARKER_SIZE_MM,
            process_view,
            return_parts=True,
            preprocess_gray_fn=preprocess_gray,
        )
        adaptive_spatial_mask = compute_wound_adaptive_spatial_mask(bgr, wound_prediction)
        if adaptive_spatial_mask is not None:
            spatial_mask = adaptive_spatial_mask
            if temporal_mask is None:
                combined_mask = spatial_mask
            else:
                combined_mask = cv2.bitwise_or(spatial_mask, temporal_mask)
        return combined_mask, spatial_mask, temporal_mask
        
    # 去畸變處理挑選出的最優左右圖
    imgA_bgr, _ = process_view(video_data['frame_B']) # 結尾最優影格作為左圖 (B)
    imgB_bgr, _ = process_view(video_data['frame_A']) # 開頭最優影格作為右圖 (A)
    
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, ENABLE_CLAHE_DEFAULT)
    h, w = imgA_gray.shape
    startup_timer.stage("最優影格去畸變+灰階")
    
    # 使用分析得到的相對 R, t 和 baseline
    R_r = video_data['R_rel']
    t_r = video_data['t_rel']
    _rt_left_value = video_data.get('rt_sift_points_left')
    _rt_right_value = video_data.get('rt_sift_points_right')
    rt_sift_points_left = np.asarray(
        [] if _rt_left_value is None else _rt_left_value, dtype=np.float64).reshape(-1, 2)
    rt_sift_points_right = np.asarray(
        [] if _rt_right_value is None else _rt_right_value, dtype=np.float64).reshape(-1, 2)
    rt_sift_inlier_count = min(len(rt_sift_points_left), len(rt_sift_points_right))
    rt_sift_points_left = rt_sift_points_left[:rt_sift_inlier_count]
    rt_sift_points_right = rt_sift_points_right[:rt_sift_inlier_count]
    rt_sift_match_count = int(video_data.get('rt_sift_match_count', rt_sift_inlier_count))
    rt_sift_applied = bool(video_data.get('rt_sift_applied', False))
    rt_sift_role = video_data.get(
        'rt_sift_role', 'final_rt' if rt_sift_applied else 'validation_only')
    rt_sift_diagnostics_path = video_data.get('rt_sift_diagnostics_path')
    
    sift = cv2.SIFT_create(contrastThreshold=0.005)
    orb = cv2.ORB_create(nfeatures=1000)
    
    # 優先使用進度條執行期間在背景預先計算的平面
    global_plane_n = video_data.get('global_plane_n')
    global_plane_c = video_data.get('global_plane_c')
    if global_plane_n is None or global_plane_c is None:
        global_plane_n, global_plane_c = compute_global_plane(imgA_gray, KL, ACTUAL_MARKER_SIZE_MM)
    
    # 預設直接鎖定
    locked_L = imgA_bgr.copy()
    locked_R = imgB_bgr.copy()
    locked_L_clean = locked_L.copy()
    locked_R_clean = locked_R.copy()
    locked_L_idx = video_data['idx_B']
    locked_R_idx = video_data['idx_A']
    locked_L_spec_mask, locked_L_spec_spatial_mask, locked_L_spec_temporal_mask = compute_locked_spec_masks(locked_L_clean, locked_L_idx)
    locked_R_spec_mask, locked_R_spec_spatial_mask, locked_R_spec_temporal_mask = compute_locked_spec_masks(locked_R_clean, locked_R_idx)
    if CIRCLE_LABEL_MATCH_ENABLED:
        circle_centers_L, circle_radii_L = detect_circle_label_centers(locked_L_clean)
        circle_centers_R, circle_radii_R = detect_circle_label_centers(locked_R_clean)
        print(f"[CircleLabel] detected left={len(circle_centers_L)} right={len(circle_centers_R)} centers")
    else:
        circle_centers_L, circle_radii_L = _empty_points(), np.empty((0,), dtype=np.float32)
        circle_centers_R, circle_radii_R = _empty_points(), np.empty((0,), dtype=np.float32)
    startup_timer.stage("全域平面+左右高光遮罩")
    live_L = False
    live_R = False
    has_set_L = True
    has_set_R = True
    
    current_cand = {
        'idx': video_data['idx_B'], # 左圖（量測起點圖）索引
        'rgb': cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB), 
        'gray': cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY),
        'K_R': KL, 
        'R_rel': R_r, 
        't_rel': t_r, 
        'F': compute_fundamental_matrix(KL, KL, R_r, t_r),
        'cornersA': video_data['cornersB'], # 對應左圖 (B)
        'cornersB': video_data['cornersA'], # 對應右圖 (A)
        'kpB': video_data.get('best_kpB', []), 
        'desB': video_data.get('best_desB'),
        'plane_n': global_plane_n, 
        'plane_c': global_plane_c,
        'pose_valid': True,
        'baseline': video_data['baseline'],
        'pose_info': f"ArUco multi-frame average (Bsl: {video_data['baseline']:.1f}mm)" if POSE_SELECT_MODE=="average" else f"ArUco best pair (Bsl: {video_data['baseline']:.1f}mm)",
        'marker_map': video_data['marker_map'],
        'map_calibrated': True
    }
    current_cand['spec_mask'] = locked_R_spec_mask
    current_cand['spec_spatial_mask'] = locked_R_spec_spatial_mask
    current_cand['spec_temporal_mask'] = locked_R_spec_temporal_mask
    current_cand['circle_centers'] = circle_centers_R
    current_cand['circle_radii'] = circle_radii_R
    
    # 若背景計算因任何理由未獲得特徵，則降級在主線程中計算
    if not current_cand['kpB'] or current_cand['desB'] is None:
        kb, db = sift.detectAndCompute(current_cand['gray'], None)
        current_cand.update({'kpB': kb, 'desB': db})
    
    extra_candidates_list = []
    for extra in video_data.get('extra_candidates', []):
        imgB_extra_bgr, _ = process_view(extra['frame_A'])
        # 高光遮罩延遲計算：開啟 Reject SpecPts 後第一次用到該影格時才在 compute_measure 內計算並回填
        extra_cand = {
            'idx': extra['idx_A'],
            'rgb': cv2.cvtColor(imgB_extra_bgr, cv2.COLOR_BGR2RGB),
            'gray': cv2.cvtColor(imgB_extra_bgr, cv2.COLOR_BGR2GRAY),
            'K_R': KL,
            'R_rel': extra['R_rel'],
            't_rel': extra['t_rel'],
            'F': compute_fundamental_matrix(KL, KL, extra['R_rel'], extra['t_rel']),
            'cornersA': video_data['cornersB'],  # 對應左圖 (B)
            'cornersB': extra['cornersA'],       # 對應右圖 (A)
            'kpB': extra.get('kpB', []),
            'desB': extra.get('desB'),
            'plane_n': global_plane_n,
            'plane_c': global_plane_c,
            'pose_valid': True,
            'baseline': extra['baseline'],
            'pose_info': f"ArUco alternate pair (Bsl: {extra['baseline']:.1f}mm)",
            'spec_mask': None,
            'spec_spatial_mask': None,
            'spec_temporal_mask': None,
        }
        if CIRCLE_LABEL_MATCH_ENABLED:
            extra_cand['circle_centers'], extra_cand['circle_radii'] = detect_circle_label_centers(imgB_extra_bgr)
            print(f"[CircleLabel] detected right F{extra_cand['idx']}={len(extra_cand['circle_centers'])} centers")
        else:
            extra_cand['circle_centers'] = _empty_points()
            extra_cand['circle_radii'] = np.empty((0,), dtype=np.float32)
        # 若背景中未成功提取，才在主線程中提取特徵
        if not extra_cand['kpB'] or extra_cand['desB'] is None:
            kb_e, db_e = sift.detectAndCompute(extra_cand['gray'], None)
            extra_cand.update({'kpB': kb_e, 'desB': db_e})
        extra_candidates_list.append(extra_cand)
        
    candidates = [current_cand]
    startup_timer.stage(f"次佳影格處理({len(extra_candidates_list)} 個)")

    # 進 UI 前預先算好次佳影格高光遮罩 (Reject SpecPts 預設開啟)，
    # 期間以獨立讀取條視窗顯示「階段處理中」，避免第一次點擊卡住
    precompute_masks_with_progress_window(extra_candidates_list, compute_locked_spec_masks)
    startup_timer.stage("高光遮罩預計算")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1E1E1E')
    fig.canvas.manager.set_window_title("MeasureTool")
    try:
        fig.canvas.toolbar.pack_forget() # 隱藏底部的功能條
    except:
        pass
    fig.subplots_adjust(top=0.58, right=0.98, left=0.05, bottom=0.08)
    ax_A, ax_B = axes
    im_A = ax_A.imshow(cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB))
    im_B = ax_B.imshow(current_cand['rgb'])
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor('#1E1E1E')
    ax_B.set_visible(False)
        
    # 加入專業感的影像外框
    from matplotlib.patches import Rectangle
    border_A = Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='#00FF00', lw=2, alpha=0.8) # 綠色代表 Live
    border_B = Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='#FFCC00', lw=2, alpha=0.8) # 黃色代表鎖定/參考
    ax_A.add_patch(border_A)
    ax_B.add_patch(border_B)
    
    # 設定標題為白色
    ax_A.set_title('Camera (Live)', color='white', fontsize=10, fontweight='bold', pad=10)
    ax_B.set_title('右圖 (Locked)', color='white', fontsize=10, fontweight='bold', pad=10)

    def draw_aruco(ax, corners):
        if not hasattr(ax, 'art'): ax.art = []
        for a in ax.art: a.remove()
        ax.art = []
        if not corners: return
        min_id = min(corners.keys())
        for mid, pts in corners.items():
            p = np.vstack((pts, pts[0])); l, = ax.plot(p[:,0], p[:,1], 'cyan', lw=1.5)
            t_str = f"ID:{mid}"
            if mid == min_id:
                t_str += " (World Origin)"
                center = np.mean(pts, axis=0)
                c_pt, = ax.plot(center[0], center[1], 'r+', markersize=10, markeredgewidth=2, zorder=4)
                ax.art.append(c_pt)
            t = ax.text(pts[0,0], pts[0,1]-5, t_str, color='cyan', fontsize=8, fontweight='bold' if mid == min_id else 'normal', zorder=4)
            ax.art.extend([l, t])
            
            # 用四種不同顏色標示四個角點：紅(0)、綠(1)、藍(2)、黃(3)，以協助確認平面方向是否正確
            c_colors = ['ro', 'go', 'bo', 'yo']
            for i in range(4):
                c_pt, = ax.plot(pts[i,0], pts[i,1], c_colors[i], markersize=6, zorder=4)
                ax.art.append(c_pt)

    draw_aruco(ax_A, current_cand['cornersA'])
    draw_aruco(ax_B, current_cand['cornersB'])

    def draw_rt_consistency_overlay(cornersA_d, cornersB_d, R_rel, t_rel):
        """
        RT 一致性圖層：以量測用的最終 RT 三角化左右圖共享標籤角點，再重投影回兩視角繪製。
        取代舊的「單標籤 PnP → RT 轉換」畫法——單標籤 PnP 有 IPPE 分支歧義，
        畫出的偏移混入 PnP 自身誤差，無法判讀 RT 好壞。
        另印出三角化邊長 vs 已知邊長的尺度檢查 (極線殘差看不到沿極線的尺度滑動)。
        """
        for ax_t in (ax_A, ax_B):
            if not hasattr(ax_t, 'reproj_art'):
                ax_t.reproj_art = []
            for a in ax_t.reproj_art:
                try: a.remove()
                except: pass
            ax_t.reproj_art = []
        shared = set(cornersA_d.keys()) & set(cornersB_d.keys())
        if not shared:
            return
        K64 = KL.astype(np.float64)
        P0 = (K64 @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
        P1 = (K64 @ np.hstack([np.asarray(R_rel, np.float64),
                               np.asarray(t_rel, np.float64).reshape(3, 1)])).astype(np.float32)
        rvec_rel, _ = cv2.Rodrigues(np.asarray(R_rel, np.float64))
        for mid in sorted(shared):
            ptsA = np.asarray(cornersA_d[mid], dtype=np.float32)
            ptsB = np.asarray(cornersB_d[mid], dtype=np.float32)
            pts4d = cv2.triangulatePoints(P0, P1, ptsA.T, ptsB.T)
            w = pts4d[3]
            if np.any(np.abs(w) < 1e-12):
                continue
            X = (pts4d[:3] / w).T
            projA, _ = cv2.projectPoints(X.astype(np.float32), np.zeros(3), np.zeros(3), KL, np.zeros(5))
            projB, _ = cv2.projectPoints(X.astype(np.float32), rvec_rel,
                                         np.asarray(t_rel, np.float64).reshape(3, 1), KL, np.zeros(5))
            projA = projA.reshape(4, 2)
            projB = projB.reshape(4, 2)
            errA = float(np.mean(np.linalg.norm(projA - ptsA, axis=1)))
            errB = float(np.mean(np.linalg.norm(projB - ptsB, axis=1)))
            edges = [float(np.linalg.norm(X[(i + 1) % 4] - X[i])) for i in range(4)]
            edge_mean = float(np.mean(edges))
            scale_err = (edge_mean / ACTUAL_MARKER_SIZE_MM - 1.0) * 100.0
            log_and_print(
                f"📊 [RT一致性] 標籤 {mid} | 三角化重投影 左 {errA:.2f}px / 右 {errB:.2f}px (量極線幾何) | "
                f"三角化邊長 {edge_mean:.2f}mm vs 已知 {ACTUAL_MARKER_SIZE_MM}mm (尺度偏差 {scale_err:+.1f}%)"
            )
            for ax_t, proj in ((ax_A, projA), (ax_B, projB)):
                p = np.vstack((proj, proj[0]))
                l, = ax_t.plot(p[:, 0], p[:, 1], color='#FF00FF', linestyle='--', lw=1.5, alpha=0.8, zorder=3)
                ax_t.reproj_art.append(l)
                for i in range(4):
                    pt, = ax_t.plot(proj[i, 0], proj[i, 1], color='#FF00FF', marker='+', markersize=6, zorder=3)
                    ax_t.reproj_art.append(pt)

    draw_rt_consistency_overlay(current_cand['cornersA'], current_cand['cornersB'], R_r, t_r)

    # 初始化 TXT 檔案，寫入影片分析與 Baseline 組合日誌
    init_txt_path = os.path.splitext(VIDEO_PATH)[0] + ".txt"
    try:
        with open(init_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== 影片分析與挑選最優對日誌 ===\n")
            for line in ANALYSIS_LOG:
                f.write(line + "\n")
            f.write("\n")
            
            for line in COMBINATION_LOG:
                f.write(line + "\n")
            f.write("\n=== RT SIFT recoverPose inlier pixel pairs ===\n")
            f.write(
                f"role={rt_sift_role}, applied={rt_sift_applied}, "
                f"inliers={rt_sift_inlier_count}/{rt_sift_match_count}\n")
            for i, (pt_left, pt_right) in enumerate(
                    zip(rt_sift_points_left, rt_sift_points_right), start=1):
                f.write(
                    f"#{i:03d}: left=({pt_left[0]:.3f}, {pt_left[1]:.3f}), "
                    f"right=({pt_right[0]:.3f}, {pt_right[1]:.3f})\n")
            f.write("\n==================================================\n\n")
        print(f"💾 已初始化分析日誌至: {init_txt_path}")
    except Exception as e:
        print(f"❌ 初始化日誌失敗: {e}")

    scatter_A = ax_A.scatter([], [], s=80, c='red', marker='x', zorder=5)
    scatter_A_reproj = ax_A.scatter([], [], s=120, facecolors='none', edgecolors='#FF00FF', marker='o', linestyle='--', lw=1.5, zorder=6)
    scatter_B = ax_B.scatter([], [], s=80, c='lime', marker='x', zorder=5)
    scatter_B_reproj = ax_B.scatter([], [], s=120, facecolors='none', edgecolors='#FF00FF', marker='o', linestyle='--', lw=1.5, zorder=6)
    scatter_grad_ref_A = ax_A.scatter([], [], s=5, c='#8FD3FF', alpha=0.65, zorder=3)
    scatter_grad_ref_B = ax_B.scatter([], [], s=5, c='#8FD3FF', alpha=0.65, zorder=3)
    scatter_mid_grad_ref_A = ax_A.scatter([], [], s=5, c='#FFD29A', alpha=0.65, zorder=3)
    scatter_mid_grad_ref_B = ax_B.scatter([], [], s=5, c='#FFD29A', alpha=0.65, zorder=3)
    scatter_grad_inject = ax_A.scatter([], [], s=18, c='#0047AB', alpha=0.9, zorder=4)
    scatter_grad_match = ax_B.scatter([], [], s=18, c='#0047AB', alpha=0.9, zorder=4)
    scatter_mid_grad_inject = ax_A.scatter([], [], s=18, c='#FF8C00', alpha=0.9, zorder=4)
    scatter_mid_grad_match = ax_B.scatter([], [], s=18, c='#FF8C00', alpha=0.9, zorder=4)
    rt_sift_colors = np.linspace(0.0, 1.0, rt_sift_inlier_count) if rt_sift_inlier_count else []
    scatter_rt_sift_A = ax_A.scatter(
        rt_sift_points_left[:, 0], rt_sift_points_left[:, 1],
        s=30, c=rt_sift_colors, cmap='turbo', vmin=0.0, vmax=1.0,
        edgecolors='black', linewidths=0.35, alpha=0.95, zorder=7, visible=False)
    scatter_rt_sift_B = ax_B.scatter(
        rt_sift_points_right[:, 0], rt_sift_points_right[:, 1],
        s=30, c=rt_sift_colors, cmap='turbo', vmin=0.0, vmax=1.0,
        edgecolors='black', linewidths=0.35, alpha=0.95, zorder=7, visible=False)
    scatter_circle_A = ax_A.scatter(
        circle_centers_L[:, 0] if len(circle_centers_L) else [],
        circle_centers_L[:, 1] if len(circle_centers_L) else [],
        s=75, facecolors='none', edgecolors='#00E5FF', marker='o', linewidths=1.4,
        alpha=0.95, zorder=4.5
    )
    scatter_circle_B = ax_B.scatter(
        current_cand['circle_centers'][:, 0] if len(current_cand.get('circle_centers', _empty_points())) else [],
        current_cand['circle_centers'][:, 1] if len(current_cand.get('circle_centers', _empty_points())) else [],
        s=75, facecolors='none', edgecolors='#FFE066', marker='o', linewidths=1.4,
        alpha=0.95, zorder=4.5
    )
    epi_line, = ax_B.plot([], [], 'yellow', lw=1, alpha=0.6, zorder=4)
    sift_rect = Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='magenta', facecolor='none', linestyle='--', alpha=0.8, zorder=4)
    ax_B.add_patch(sift_rect)
    sift_rect.set_visible(False)
    sift_rect_center, = ax_B.plot([], [], '+', color='magenta', markersize=12, markeredgewidth=1.5, zorder=5)
    sift_rect_center.set_visible(False)
    # HUD 風格的文字面板
    depth_text = fig.text(0.53, 0.35, "", transform=fig.transFigure,
                          color='white', fontweight='bold', fontsize=13,
                          bbox=dict(facecolor='#121212', alpha=0.7, edgecolor='#00FFFF', lw=1))
    fps_text = ax_A.text(0.01, 1.03, "FPS: --", transform=ax_A.transAxes,
                         color='#00FF00', fontsize=10, fontweight='bold', va='bottom',
                         bbox=dict(facecolor='#121212', alpha=0.6, edgecolor='none'), zorder=10,
                         clip_on=False)
                         
    pose_err = video_data.get('min_reproj_err')
    if pose_err is None:
        pose_status_str = "姿態估計狀態: 未知"
        pose_status_color = "#FFFFFF"
    elif pose_err < 0.3:
        pose_status_str = "姿態估計效果理想"
        pose_status_color = "#00FF00"
    elif pose_err < 0.5:
        pose_status_str = "姿態估計效果正常"
        pose_status_color = "#FFFF00"
    elif pose_err < 1.0:
        pose_status_str = "姿態預測效果不佳"
        pose_status_color = "#FF9900"
    else:
        pose_status_str = "姿態預測效果異常"
        pose_status_color = "#FF0000"
        
    if pose_err is not None:
        pose_status_str += f" ({pose_err:.2f} px)"
        
    pose_status_text = fig.text(0.975, 0.025, pose_status_str, transform=fig.transFigure,
                                 color=pose_status_color, fontsize=10, fontweight='bold',
                                 ha='right', va='bottom',
                                 bbox=dict(facecolor='#121212', alpha=0.7, edgecolor=pose_status_color, lw=1), zorder=10)
                                 
    # Blit 最佳化：標記每幀會改變的 artists 為 animated，防止它們被無謂嫚入靜態背景圖
    im_A.set_animated(True)
    im_B.set_animated(True)
    fps_text.set_animated(True)
    pose_status_text.set_animated(True)
    # blit_state: 管理背景圖狀態
    blit_state = {'bg': None, 'needs_refresh': True}

    def request_blit_refresh():
        """UI 元件有治變時呼叫，主迴圈下一幀會重新全圖儲存新背景。"""
        blit_state['needs_refresh'] = True

    # 顯示影像轉換快取：locked_L/locked_R 與疊圖狀態沒變時，重繪直接重用上次轉換結果
    display_cache = {'key': None, 'disp_A': None, 'disp_B': None, 'version': 0}

    def mark_display_dirty():
        """locked_L/locked_R 內容或疊圖來源 (遮罩/傷口預測) 改變時呼叫，使顯示快取失效。"""
        display_cache['version'] += 1

    wound_state = {
        'show': False,
        'left_pred': None,
        'right_pred': None,
        'left_count': 0,
        'right_count': 0,
        'left_size': None,
        'right_size': None,
        'v1_size': None,
        'size_error': None,
        'dirty': False,
        'corner_source': 'min_area',
    }

    def update_wound_size_from_current_v1(reason="state change"):
        wound_state['size_error'] = None
        wound_state['v1_size'] = compute_wound_size_with_current_v1()
        wound_state['left_size'] = wound_state['v1_size']
        wound_state['right_size'] = wound_state['v1_size']
        wound_state['dirty'] = False
        v1_size = wound_state['v1_size']
        size_msg = "N/A" if v1_size is None else (
            f"{v1_size['long']:.1f}x{v1_size['short']:.1f}{v1_size['unit']} "
            f"(valid {v1_size['valid_points']}/4)"
        )
        if v1_size and v1_size.get('corner_candidate_frames'):
            frame_parts = []
            for i, frames_used in enumerate(v1_size['corner_candidate_frames']):
                if frames_used:
                    frame_parts.append(f"L{i + 1}:F{','.join(str(int(f)) for f in frames_used)}")
                else:
                    frame_parts.append(f"L{i + 1}:N/A")
            size_msg += " | " + " ".join(frame_parts)
        print(f"[Wound] V1 size refresh {reason}: {size_msg}")
        mark_display_dirty()

    def refresh_wound_predictions(reason="selected"):
        wound_state['left_pred'] = predict_wound_regions_bgr(locked_L_clean)
        wound_state['right_pred'] = predict_wound_regions_bgr(locked_R_clean)
        wound_state['left_count'] = count_wound_detections(wound_state['left_pred'])
        wound_state['right_count'] = count_wound_detections(wound_state['right_pred'])
        wound_state['v1_size'] = None
        wound_state['left_size'] = None
        wound_state['right_size'] = None
        wound_state['size_error'] = None
        wound_state['dirty'] = True
        print(
            f"[Wound] Pre-inference {reason}: "
            f"left={wound_state['left_count']} right={wound_state['right_count']}"
        )
        mark_display_dirty()
        if use_wound_adaptive_spatial_specular:
            recompute_locked_spec_masks_from_wound(f"wound prediction {reason}")

    def recompute_locked_spec_masks_from_wound(reason="adaptive spatial"):
        nonlocal locked_L_spec_mask, locked_L_spec_spatial_mask, locked_L_spec_temporal_mask
        nonlocal locked_R_spec_mask, locked_R_spec_spatial_mask, locked_R_spec_temporal_mask
        locked_L_spec_mask, locked_L_spec_spatial_mask, locked_L_spec_temporal_mask = compute_locked_spec_masks(
            locked_L_clean,
            locked_L_idx,
            wound_state.get('left_pred'),
        )
        locked_R_spec_mask, locked_R_spec_spatial_mask, locked_R_spec_temporal_mask = compute_locked_spec_masks(
            locked_R_clean,
            locked_R_idx,
            wound_state.get('right_pred'),
        )
        current_cand['spec_mask'] = locked_R_spec_mask
        current_cand['spec_spatial_mask'] = locked_R_spec_spatial_mask
        current_cand['spec_temporal_mask'] = locked_R_spec_temporal_mask
        print(
            f"[Specular] {'Adaptive wound spatial' if use_wound_adaptive_spatial_specular else 'Fixed spatial'} "
            f"masks refreshed ({reason})"
        )
        mark_display_dirty()

    def mark_wound_size_dirty(reason="state change"):
        wound_state['dirty'] = True
        if wound_state.get('show', False):
            print(f"[Wound] Matching state changed ({reason}); recomputing displayed V1 size...")
            update_wound_size_from_current_v1(reason)

    def apply_wound_overlay_if_enabled(disp_A, disp_B):
        if not wound_state.get('show', False):
            return disp_A, disp_B
        use_min_area_rect = wound_state.get('corner_source', 'min_area') == 'min_area'
        disp_A = overlay_wound_prediction_rgb(
            disp_A,
            wound_state.get('left_pred'),
            draw_bbox=not use_min_area_rect,
        )
        disp_B = overlay_wound_prediction_rgb(disp_B, wound_state.get('right_pred'), draw_bbox=False)
        v1_size = wound_state.get('v1_size')
        if v1_size:
            left_corner_color = (40, 150, 255) if use_min_area_rect else (255, 230, 40)
            disp_A = draw_wound_corner_points_rgb(
                disp_A,
                v1_size.get('left_box'),
                "L",
                line_closed=True,
                color=left_corner_color,
            )
            disp_B = draw_wound_corner_points_rgb(disp_B, v1_size.get('right_points'), "R", line_closed=False)
        fallback = None
        if v1_size is None:
            fallback = "N/A" if not wound_state.get('size_error') else f"N/A: {wound_state['size_error']}"
        disp_A = draw_wound_size_label_rgb(disp_A, v1_size, "V1 3D", fallback)
        disp_B = draw_wound_size_label_rgb(disp_B, v1_size, "V1 3D", fallback)
        return disp_A, disp_B


    # 勾選框面板 (改成兩行排列，每顆獨立以利排版)
    # 由於 Matplotlib 的 CheckButtons 在不同版本間極難著色，這裡改用標準 Button 來模擬勾選框！
    ax_c1 = fig.add_axes([0.05, 0.92, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c2 = fig.add_axes([0.17, 0.92, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c3 = fig.add_axes([0.29, 0.92, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c4 = fig.add_axes([0.05, 0.86, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c5 = fig.add_axes([0.17, 0.86, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c6 = fig.add_axes([0.29, 0.86, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c7 = fig.add_axes([0.05, 0.80, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c8 = fig.add_axes([0.17, 0.80, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c9 = fig.add_axes([0.29, 0.80, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c10 = fig.add_axes([0.05, 0.74, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c11 = fig.add_axes([0.17, 0.74, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c12 = fig.add_axes([0.29, 0.74, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c13 = fig.add_axes([0.05, 0.68, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c14 = fig.add_axes([0.17, 0.68, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c15 = fig.add_axes([0.29, 0.68, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c16 = fig.add_axes([0.05, 0.62, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c17 = fig.add_axes([0.17, 0.62, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c19 = fig.add_axes([0.29, 0.62, 0.11, 0.04], facecolor='#1E1E1E')
    
    # 建立標準按鈕，文字開頭加上 [X] 或 [ ] 代表勾選狀態
    btn_opt_style = dict(color='#1A1A1A', hovercolor='#333333')
    c1 = Button(ax_c1, "[X] 嚴格精細匹配", **btn_opt_style)
    c2 = Button(ax_c2, "[X] 梯度 SIFT 匹配", **btn_opt_style)
    c3 = Button(ax_c3, "[X] 強制極線對齊", **btn_opt_style)
    c4 = Button(ax_c4, "[X] 啟用 ECC 精修", **btn_opt_style)
    c5 = Button(ax_c5, "[ ] 手動匹配模式", **btn_opt_style)
    c6 = Button(ax_c6, "[X] 啟用 CLAHE 增強" if ENABLE_CLAHE_DEFAULT else "[ ] 啟用 CLAHE 增強", **btn_opt_style)
    c7 = Button(ax_c7, "[X] 改良匹配流程" if ENABLE_IMPROVED_MATCHING_DEFAULT else "[ ] 改良匹配流程", **btn_opt_style)
    c8 = Button(ax_c8, "[X] 顯示匹配分數" if SHOW_SCORE_DEFAULT else "[ ] 顯示匹配分數", **btn_opt_style)
    c9 = Button(ax_c9, "[ ] 色彩直方圖約束", **btn_opt_style)
    c10 = Button(ax_c10, "[ ] 啟用 RGB-SIFT", **btn_opt_style)
    c11 = Button(ax_c11, "[ ] Opponent-SIFT", **btn_opt_style)
    c12 = Button(ax_c12, "[ ] 過濾高光反光", **btn_opt_style)
    c13 = Button(ax_c13, "[ ] 進階高光過濾", **btn_opt_style)
    c14 = Button(ax_c14, "[X] Epi-band Search" if ENABLE_EPIPOLAR_BAND_SEARCH_DEFAULT else "[ ] Epi-band Search", **btn_opt_style)
    c15 = Button(ax_c15, "[ ] Show Spatial", **btn_opt_style)
    c16 = Button(ax_c16, "[ ] Show Temporal", **btn_opt_style)
    c17 = Button(ax_c17, "[X] Reject SpecPts", **btn_opt_style)
    c19 = Button(ax_c19, "[X] Adaptive Spatial", **btn_opt_style)

    view_state = {'precise': True, 'grad_sift': True, 'enforce_epi': True, 'ecc': True, 'manual': False,
                  'use_hamming': True, 'enable_clahe': ENABLE_CLAHE_DEFAULT,
                  'use_improved_matching': ENABLE_IMPROVED_MATCHING_DEFAULT,
                  'show_score': SHOW_SCORE_DEFAULT,
                  'use_color_hist': False,
                  'use_rgb_sift': False,
                  'use_opponent_sift': False,
                  'filter_specular': False,
                  'filter_specular_hsv_mser': False,
                  'epipolar_band_search': ENABLE_EPIPOLAR_BAND_SEARCH_DEFAULT,
                  'show_spatial_specular_mask': False,
                  'show_temporal_specular_mask': False,
                  'reject_specular_candidates': True,
                  'adaptive_spatial_specular': True,
                  'show_high_grad_points': False,
                  'show_mid_grad_points': False,
                  'show_aruco_overlay': False,
                  'show_rt_sift_points': False,
                  'manual_pt_A': None, 'lines': [], 'grad_lines': [], 'show_grad_lines': False,
                  'highlighted_grad_line': None, 'highlighted_grad_line_artist': None,
                  'grad_data': None, 'restart': False}  # grad_data = {'ptsA': ndarray, 'ptsB': ndarray}

    # HighPts / MidPts 預設關閉：初始同步散點顯示狀態
    for _artist in (scatter_grad_ref_A, scatter_grad_ref_B, scatter_grad_inject, scatter_grad_match):
        _artist.set_visible(view_state['show_high_grad_points'])
    for _artist in (scatter_mid_grad_ref_A, scatter_mid_grad_ref_B, scatter_mid_grad_inject, scatter_mid_grad_match):
        _artist.set_visible(view_state['show_mid_grad_points'])

    # 建立測量模式單選框，置於中間空白處
    ax_mode = fig.add_axes([0.42, 0.86, 0.13, 0.10], facecolor='#1E1E1E')
    ax_mode.patch.set_edgecolor('white')
    ax_mode.patch.set_linewidth(1.0)
    radio_mode = RadioButtons(ax_mode, ('雙幀直接', '多幀去漂移', '多幀純光流'),
                              active=0 if MEASURE_MODE=="dual_direct" else (1 if MEASURE_MODE=="multi_dedrift" else 2),
                              activecolor='#00FFFF')
    
    # 調整單選框字型與色彩
    for label in radio_mode.labels:
        label.set_color('white')
        label.set_fontsize(8)
        
    def on_mode_change(label_text):
        global MEASURE_MODE
        if label_text == '雙幀直接':
            MEASURE_MODE = 'dual_direct'
        elif label_text == '多幀去漂移':
            MEASURE_MODE = 'multi_dedrift'
        elif label_text == '多幀純光流':
            MEASURE_MODE = 'multi_pure'
        print(f"🔄 量測模式已切換為: {MEASURE_MODE}")
        mark_wound_size_dirty('measure_mode')
        if last_click:
            do_measure(last_click[0], last_click[1])
            
    radio_mode.on_clicked(on_mode_change)

    # 統一設定文字顏色為白色，並將按鈕外框設為白色
    for c in [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c19]:
        c.label.set_color('white')
        c.label.set_fontsize(8)
        c.ax.patch.set_edgecolor('white')
        c.ax.patch.set_linewidth(1.0)
            
    # 使用閉包來處理點擊事件與文字切換
    # 使用閉包來處理點擊事件與文字切換
    def make_on_opt(btn, key, label_text):
        def _on_opt(event):
            view_state[key] = not view_state[key]
            # 根據狀態切換 [X] 或 [ ]
            prefix = "[X] " if view_state[key] else "[ ] "
            btn.label.set_text(prefix + label_text)
            
            if key == 'manual' and not view_state['manual']:
                view_state['manual_pt_A'] = None
            request_blit_refresh()
            
            if key in ('precise', 'grad_sift', 'enforce_epi', 'ecc',
                       'enable_clahe', 'use_improved_matching', 'use_color_hist',
                       'filter_specular', 'filter_specular_hsv_mser', 'epipolar_band_search',
                       'reject_specular_candidates'):
                mark_wound_size_dirty(key)
                if last_click:
                    do_measure(last_click[0], last_click[1])
        return _on_opt
        
    c1.on_clicked(make_on_opt(c1, 'precise', "嚴格精細匹配"))
    c2.on_clicked(make_on_opt(c2, 'grad_sift', "梯度 SIFT 匹配"))
    c3.on_clicked(make_on_opt(c3, 'enforce_epi', "強制極線對齊"))
    c4.on_clicked(make_on_opt(c4, 'ecc', "啟用 ECC 精修"))
    c5.on_clicked(make_on_opt(c5, 'manual', "手動匹配模式"))
    c6.on_clicked(make_on_opt(c6, 'enable_clahe', "啟用 CLAHE 增強"))
    c7.on_clicked(make_on_opt(c7, 'use_improved_matching', "改良匹配流程"))
    c8.on_clicked(make_on_opt(c8, 'show_score', "顯示匹配分數"))
    c9.on_clicked(make_on_opt(c9, 'use_color_hist', "色彩直方圖約束"))
    c14.on_clicked(make_on_opt(c14, 'epipolar_band_search', "Epi-band Search"))
    c15.on_clicked(make_on_opt(c15, 'show_spatial_specular_mask', "Show Spatial"))
    c16.on_clicked(make_on_opt(c16, 'show_temporal_specular_mask', "Show Temporal"))
    c17.on_clicked(make_on_opt(c17, 'reject_specular_candidates', "Reject SpecPts"))

    def on_adaptive_spatial_specular(event):
        nonlocal use_wound_adaptive_spatial_specular
        view_state['adaptive_spatial_specular'] = not view_state['adaptive_spatial_specular']
        use_wound_adaptive_spatial_specular = view_state['adaptive_spatial_specular']
        c19.label.set_text("[X] Adaptive Spatial" if use_wound_adaptive_spatial_specular else "[ ] Adaptive Spatial")
        if wound_state.get('left_pred') is None and wound_state.get('right_pred') is None:
            refresh_wound_predictions("adaptive spatial toggle")
        else:
            recompute_locked_spec_masks_from_wound("adaptive spatial toggle")
        request_blit_refresh()

    c19.on_clicked(on_adaptive_spatial_specular)
    def on_c10_clicked(event):
        view_state['use_rgb_sift'] = not view_state['use_rgb_sift']
        c10.label.set_text("[X] 啟用 RGB-SIFT" if view_state['use_rgb_sift'] else "[ ] 啟用 RGB-SIFT")
        if view_state['use_rgb_sift'] and view_state.get('use_opponent_sift', False):
            view_state['use_opponent_sift'] = False
            c11.label.set_text("[ ] Opponent-SIFT")
        request_blit_refresh()
        mark_wound_size_dirty('use_rgb_sift')
        if last_click:
            do_measure(last_click[0], last_click[1])

    def on_c11_clicked(event):
        view_state['use_opponent_sift'] = not view_state['use_opponent_sift']
        c11.label.set_text("[X] Opponent-SIFT" if view_state['use_opponent_sift'] else "[ ] Opponent-SIFT")
        if view_state['use_opponent_sift'] and view_state.get('use_rgb_sift', False):
            view_state['use_rgb_sift'] = False
            c10.label.set_text("[ ] 啟用 RGB-SIFT")
        request_blit_refresh()
        mark_wound_size_dirty('use_opponent_sift')
        if last_click:
            do_measure(last_click[0], last_click[1])

    c10.on_clicked(on_c10_clicked)
    c11.on_clicked(on_c11_clicked)
    c12.on_clicked(make_on_opt(c12, 'filter_specular', "過濾高光反光"))
    c13.on_clicked(make_on_opt(c13, 'filter_specular_hsv_mser', "進階高光過濾"))

    def redraw_grad_lines(highlight_idx=None):
        """清除所有梯度SIFT連線 (包含高亮), 依 view_state['grad_data'] 重新繪製."""
        # 清除所有舊的連線 Artist
        for item in view_state['grad_lines']:
            try: item.remove()
            except: pass
        view_state['grad_lines'] = []
        old_h = view_state.get('highlighted_grad_line_artist')
        if old_h is not None:
            try: old_h.remove()
            except: pass
        view_state['highlighted_grad_line_artist'] = None
        
        gd = view_state.get('grad_data')
        if gd is None or not view_state['show_grad_lines']:
            return
        
        ptsA, ptsB = gd['ptsA'], gd['ptsB']
        for i in range(len(ptsA)):
            if i == highlight_idx:
                continue  # 跟高亮連線分開畫
            con = ConnectionPatch(xyA=ptsA[i], xyB=ptsB[i], coordsA="data", coordsB="data",
                                  axesA=ax_A, axesB=ax_B, color="#0047AB", lw=0.8, alpha=0.45, zorder=4)
            ax_B.add_artist(con)
            view_state['grad_lines'].append(con)
        
        if highlight_idx is not None and 0 <= highlight_idx < len(ptsA):
            hl = ConnectionPatch(xyA=ptsA[highlight_idx], xyB=ptsB[highlight_idx], coordsA="data", coordsB="data",
                                 axesA=ax_A, axesB=ax_B, color="red", lw=2.5, alpha=1.0, zorder=10)
            ax_B.add_artist(hl)
            view_state['highlighted_grad_line_artist'] = hl
        view_state['highlighted_grad_line'] = highlight_idx


    measure_results = {}
    last_click = None
    spec_mask_lock = threading.Lock()  # 高光遮罩計算互斥：背景預計算 vs 點擊時延遲計算

    # ---- 純計算（可在背景執行緒安全呼叫，不觸碰 Matplotlib）----
    def compute_measure(u, v, snap_cand, snap_imgA_gray, snap_view_state, manual_match_pt=None, left_cache=None):
        """純計算版 do_measure，回傳結果 dict，不更新任何 UI 元件。"""
        nonlocal locked_L, locked_R, locked_L_spec_mask, locked_R_spec_mask, current_cand
        cand = snap_cand
        t_cm_start = time.perf_counter()
        t_prof = {}
        left_spec_mask = locked_L_spec_mask
        if cand.get('idx') == current_cand.get('idx'):
            right_spec_mask = locked_R_spec_mask
        else:
            right_spec_mask = cand.get('spec_mask')
            # 延遲計算：只有實際會用到遮罩 (Reject SpecPts 開啟) 時才計算；
            # 與背景預計算執行緒以 spec_mask_lock 互斥，先到先算、後到直接取用
            if (right_spec_mask is None and cand.get('rgb') is not None
                    and snap_view_state.get('reject_specular_candidates', False)):
                with spec_mask_lock:
                    right_spec_mask = cand.get('spec_mask')
                    if right_spec_mask is None:
                        right_spec_mask, right_spec_spatial_mask, right_spec_temporal_mask = compute_locked_spec_masks(cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR), cand.get('idx'))
                        cand['spec_mask'] = right_spec_mask
                        cand['spec_spatial_mask'] = right_spec_spatial_mask
                        cand['spec_temporal_mask'] = right_spec_temporal_mask
        m_pt, method, neighbors = None, "", []
        direct_match_methods = ("ArUco", "CircleLabel")
        rt_bound_reject_reason = None
        g_ptsA, g_ptsB, g_groups, g_refA, g_refB, g_refA_groups, g_refB_groups, g_kptsB, g_rect = None, None, None, None, None, None, None, None, None
        trajectory_res = None

        if not cand.get('pose_valid', True):
            print(f"❌ [測量失敗] 當前候選影格位姿無效 (pose_valid == False)，原因: {cand.get('pose_info', '未知')}")
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_groups': None,
                    'g_refA': None, 'g_refB': None, 'g_refA_groups': None, 'g_refB_groups': None,
                    'g_kptsB': None, 'g_rect': None,
                    'fail_reason': '未偵測到 ArUco', 'u': u, 'v': v, 'trajectory': None}
        bsl = cand.get('baseline', 0.0)
        if bsl < MIN_BASELINE_MM or bsl > MAX_BASELINE_MM:
            print(f"❌ [測量失敗] 基準線不合規範 ({bsl:.2f} mm，限制: {MIN_BASELINE_MM}~{MAX_BASELINE_MM} mm)")
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_groups': None,
                    'g_refA': None, 'g_refB': None, 'g_refA_groups': None, 'g_refB_groups': None,
                    'g_kptsB': None, 'g_rect': None,
                    'fail_reason': f'視差不合規範({MIN_BASELINE_MM}~{MAX_BASELINE_MM}mm)', 'u': u, 'v': v, 'trajectory': None}

        # 1. 只有在需要匹配點的模式下進行匹配
        if MEASURE_MODE in ("dual_direct", "multi_dedrift"):
            if manual_match_pt is not None:
                m_pt, method = manual_match_pt, "手動點選"

            if m_pt is None:
                if snap_view_state.get('circle_label_match_active', False):
                    seed_pt, circle_rect, seed_method = circle_label_search_rect((u, v), cand, KL, cand['gray'].shape)
                    if circle_rect is not None:
                        g_rect = circle_rect
                        circle_match = find_circle_center_in_rect(cand.get('circle_centers'), circle_rect, seed_pt)
                        if circle_match is not None:
                            m_pt = circle_match.astype(np.float32)
                            method = "CircleLabel"
                            g_ptsA = np.array([[u, v]], dtype=np.float32)
                            g_ptsB = np.array([m_pt], dtype=np.float32)
                            g_groups = np.array(["circle"])
                            print(
                                f"[CircleLabel] right circle matched in ROI via {seed_method}: "
                                f"({m_pt[0]:.1f}, {m_pt[1]:.1f})"
                            )
                        else:
                            print(f"[CircleLabel] no right circle center inside predicted ROI ({seed_method})")
                if m_pt is None:
                    for mid, cA in cand['cornersA'].items():
                        d = np.linalg.norm(cA - np.array([u, v]), axis=1)
                        if np.min(d) < 10:
                            best_idx = np.argmin(d)
                            u, v = cA[best_idx] # 🌟 同步校正左圖座標為精確角點
                            m_pt, method = cand['cornersB'][mid][best_idx], "ArUco"
                            break
                if m_pt is None and snap_view_state['grad_sift']:
                    _t_blk = time.perf_counter()
                    if snap_view_state.get('use_improved_matching', False):
                        m_pt, method, g_ptsA, g_ptsB, g_rect = run_improved_matching_flow(
                            snap_imgA_gray, cand['gray'], u, v, cand, KL,
                            snap_view_state.get('use_hamming', False), orb, sift,
                            snap_view_state.get('use_color_hist', False),
                            snap_view_state.get('use_rgb_sift', False),
                            snap_view_state.get('use_opponent_sift', False),
                            left_spec_mask,
                            right_spec_mask,
                            snap_view_state.get('reject_specular_candidates', False),
                            left_bgr=locked_L
                        )
                    else:
                        gs = run_grad_sift_matching_flow(
                            snap_imgA_gray, cand['gray'], u, v, cand, KL,
                            snap_view_state, orb, sift,
                            locked_L, locked_R,
                            left_spec_mask, right_spec_mask,
                            is_best_cand=(cand['idx'] == current_cand['idx']),
                            left_cache=left_cache
                        )
                        m_pt = gs['m_pt']
                        if gs['method']:
                            method = gs['method']
                        g_ptsA, g_ptsB, g_groups = gs['g_ptsA'], gs['g_ptsB'], gs['g_groups']
                        g_refA, g_refB = gs['g_refA'], gs['g_refB']
                        g_refA_groups, g_refB_groups = gs['g_refA_groups'], gs['g_refB_groups']
                        g_kptsB, g_rect = gs['g_kptsB'], gs['g_rect']
                        if gs['reject_reason']:
                            rt_bound_reject_reason = gs['reject_reason']
                    t_prof['Grad/Improved匹配'] = time.perf_counter() - _t_blk
                if (m_pt is None and snap_view_state['precise']):
                    _t_blk = time.perf_counter()
                    res_p = find_precise_match(snap_imgA_gray, cand['gray'], (u, v), cand['F'],
                                               KL, cand['K_R'], cand['R_rel'], cand['t_rel'],
                                               cand['plane_n'], cand['plane_c'])
                    if res_p: m_pt, method = np.array(res_p), "Precise"
                    t_prof['Precise匹配'] = time.perf_counter() - _t_blk
            
            m_pt_raw = m_pt.copy() if m_pt is not None else None

            if (m_pt is not None and method not in direct_match_methods and manual_match_pt is None
                    and snap_view_state.get('epipolar_band_search', False)):
                _t_blk = time.perf_counter()
                rt_seed_for_bound, _rt_seed_method = predict_right_seed_from_geometry((u, v), cand, KL)
                epi_pt, epi_score = search_match_on_epipolar_band(
                    snap_imgA_gray, cand['gray'], (u, v), m_pt, cand['F'],
                    patch_size=31,
                    half_len=EPIPOLAR_SEARCH_HALF_LEN,
                    band_radius=EPIPOLAR_SEARCH_BAND_RADIUS,
                    min_score=EPIPOLAR_SEARCH_MIN_SCORE,
                    search_roi=g_rect
                )
                if epi_pt is not None:
                    print(f"   [Epi-band] 沿極線重新搜尋成功: seed=({m_pt[0]:.1f},{m_pt[1]:.1f}) -> ({epi_pt[0]:.1f},{epi_pt[1]:.1f}), score={epi_score:.3f}")
                    m_pt = epi_pt
                    m_pt_raw = m_pt.copy()
                    method += f"+EpiBand({epi_score:.2f})"
                    rt_dev = float(np.linalg.norm(m_pt - np.array(rt_seed_for_bound, dtype=np.float32)))
                    if rt_dev > GRAD_SIFT_MAX_RT_ADJUST_PX:
                        print(f"❌ [RT邊界] Epi-band 結果偏離 RT/平面預測 {rt_dev:.1f}px (> {GRAD_SIFT_MAX_RT_ADJUST_PX:.0f}px)，判定匹配失敗。")
                        rt_bound_reject_reason = f"匹配點偏離RT/平面預測 {rt_dev:.0f}px"
                        m_pt = None
                        m_pt_raw = None
                else:
                    print(f"   [Epi-band] 沿極線重新搜尋未通過門檻，保留原始候選點 (best score={epi_score:.3f})")
                t_prof['Epi-band搜尋'] = time.perf_counter() - _t_blk

            if (m_pt is not None and snap_view_state['enforce_epi'] and method not in direct_match_methods):
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                      m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])
                    method += "+極線對齊"

            if (m_pt is not None and method not in direct_match_methods and manual_match_pt is None):
                rt_seed_final, _rt_seed_method = predict_right_seed_from_geometry((u, v), cand, KL)
                rt_dev = float(np.linalg.norm(np.array(m_pt, dtype=np.float32) - np.array(rt_seed_final, dtype=np.float32)))
                if rt_dev > GRAD_SIFT_MAX_RT_ADJUST_PX:
                    print(f"❌ [RT邊界] 匹配點偏離 RT/平面預測 {rt_dev:.1f}px (> {GRAD_SIFT_MAX_RT_ADJUST_PX:.0f}px)，判定匹配失敗。")
                    rt_bound_reject_reason = f"匹配點偏離RT/平面預測 {rt_dev:.0f}px"
                    m_pt = None
            if (m_pt is not None and snap_view_state['ecc'] and method not in direct_match_methods):
                _t_blk = time.perf_counter()
                if snap_view_state.get('use_improved_matching', False):
                    m_pt, ecc_method = pyramid_ecc_refinement(snap_imgA_gray, cand['gray'], (u, v), m_pt, 45, 91)
                    method += ecc_method
                else:
                    tmpl = get_patch(snap_imgA_gray, (u, v), 45)
                    roi = get_patch(cand['gray'], m_pt, 91)
                    if tmpl is not None and roi is not None:
                        warp = np.eye(2, 3, dtype=np.float32)
                        warp[0, 2] = (91 - 45) / 2.0; warp[1, 2] = (91 - 45) / 2.0
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
                        try:
                            _, warp = cv2.findTransformECC(tmpl, roi, warp, cv2.MOTION_TRANSLATION, criteria)
                            m_pt = np.array([m_pt[0] - 45.5 + warp[0, 2] + 22.5,
                                             m_pt[1] - 45.5 + warp[1, 2] + 22.5])
                            method += "+ECC精修"
                        except: method += "+ECC失敗"
                t_prof['ECC精修'] = time.perf_counter() - _t_blk
            if (m_pt is not None and snap_view_state['enforce_epi'] and method not in direct_match_methods):
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                      m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])

        # 2. 分流計算三維點
        d_val, p3d_val, p3d_w_val, fail_reason = None, None, None, ""
        p3d = None
        
        if (m_pt is not None and method not in direct_match_methods and manual_match_pt is None):
            rt_seed_final, _rt_seed_method = predict_right_seed_from_geometry((u, v), cand, KL)
            rt_dev = float(np.linalg.norm(np.array(m_pt, dtype=np.float32) - np.array(rt_seed_final, dtype=np.float32)))
            if rt_dev > GRAD_SIFT_MAX_RT_ADJUST_PX:
                print(f"❌ [RT邊界] 精修後匹配點偏離 RT/平面預測 {rt_dev:.1f}px (> {GRAD_SIFT_MAX_RT_ADJUST_PX:.0f}px)，判定匹配失敗。")
                rt_bound_reject_reason = f"匹配點偏離RT/平面預測 {rt_dev:.0f}px"
                m_pt = None

        if MEASURE_MODE == "multi_dedrift":
            if m_pt is None:
                print("⚠️ [閉環光流] 雙幀直接匹配失敗，無法取得閉環真值點，退回雙幀直接模式。")
            else:
                _t_blk = time.perf_counter()
                trajectory = track_feature_and_verify(
                    video_data['all_frames'], video_data['idx_B'], video_data['idx_A'],
                    (u, v), video_data['valid_poses'], KL, distL
                )
                t_prof['光流追蹤'] = time.perf_counter() - _t_blk
                if len(trajectory) >= 3:
                    p_end_flow = np.array(trajectory[-1][1])
                    drift = np.linalg.norm(p_end_flow - np.array(m_pt))
                    print(f"📊 [閉環光流] 光流追蹤終點: {p_end_flow} | 閉環真值: {m_pt} | 累積漂移: {drift:.2f} px")
                    
                    if drift > LOOP_CLOSURE_FAIL_THRESHOLD:
                        print(f"⚠️ [閉環光流] 累積漂移過大 ({drift:.2f} px)，安全退回雙幀匹配模式！")
                    else:
                        if drift >= LOOP_CLOSURE_DRIFT_THRESHOLD:
                            print(f"🔧 [閉環光流] 漂移 ({drift:.2f} px) 超過門檻值 ({LOOP_CLOSURE_DRIFT_THRESHOLD} px)，執行去漂移修正...")
                            trajectory = apply_dedrift_correction(trajectory, m_pt)
                        else:
                            print("✅ [閉環光流] 累積漂移在容許範圍內，無需進行去漂移補償。")
                            
                        P_matrices = []
                        points_2d = []
                        for f_idx, pt in trajectory:
                            if f_idx in video_data['valid_poses']:
                                R, t = video_data['valid_poses'][f_idx]
                                P = np.hstack([R, t / 1000.0]) # 轉米
                                P_matrices.append(P)
                                pt_hom = np.linalg.inv(KL) @ np.array([pt[0], pt[1], 1.0])
                                points_2d.append((pt_hom[0]/pt_hom[2], pt_hom[1]/pt_hom[2]))
                                
                        p3d_m = multi_view_triangulation(P_matrices, points_2d)
                        if p3d_m is not None:
                            p3d = p3d_m * 1000.0
                            method = f"閉環光流多影格 ({len(trajectory)}幀)"
                            trajectory_res = trajectory
                        else:
                            print("❌ [閉環光流] 多幀三角化失敗，退回雙幀匹配結果")
                else:
                    print("⚠️ [閉環光流] 有效追蹤影格數不足 3，退回雙影格匹配。")
                    
        elif MEASURE_MODE == "multi_pure":
            _t_blk = time.perf_counter()
            trajectory = track_feature_and_verify(
                video_data['all_frames'], video_data['idx_B'], video_data['idx_A'],
                (u, v), video_data['valid_poses'], KL, distL
            )
            t_prof['光流追蹤'] = time.perf_counter() - _t_blk
            if len(trajectory) < 3:
                fail_reason = "追蹤影格數不足"
                print("⚠️ [純光流] 有效追蹤影格數不足 3，無法進行多視角三角化。")
            else:
                P_matrices = []
                points_2d = []
                for f_idx, pt in trajectory:
                    if f_idx in video_data['valid_poses']:
                        R, t = video_data['valid_poses'][f_idx]
                        P = np.hstack([R, t / 1000.0]) # 轉米
                        P_matrices.append(P)
                        pt_hom = np.linalg.inv(KL) @ np.array([pt[0], pt[1], 1.0])
                        points_2d.append((pt_hom[0]/pt_hom[2], pt_hom[1]/pt_hom[2]))
                        
                p3d_m = multi_view_triangulation(P_matrices, points_2d)
                if p3d_m is not None:
                    p3d = p3d_m * 1000.0
                    method = f"純光流多影格 ({len(trajectory)}幀)"
                    m_pt = np.array(trajectory[-1][1])
                    trajectory_res = trajectory
                else:
                    fail_reason = "三角化失敗"
                    print("❌ [純光流] 多幀三角化失敗。")

        # 雙幀退回方案或直接雙幀模式
        if p3d is None and m_pt is not None:
            print(f"👉 [深度計算] 左圖座標: ({u:.1f}, {v:.1f}) | 右圖匹配座標: ({m_pt[0]:.1f}, {m_pt[1]:.1f}) | 匹配方式: {method}")
            R_str = np.array2string(cand['R_rel'].flatten(), precision=4, suppress_small=True)
            t_str = np.array2string(cand['t_rel'].flatten(), precision=2, suppress_small=True)
            print(f"   [當前外參] R_rel: {R_str} | t_rel: {t_str}")
            _t_blk = time.perf_counter()
            p3d = triangulate_point_3d((u, v), m_pt, KL, cand['K_R'], cand['R_rel'], cand['t_rel'], F=cand.get('F'))
            t_prof['三角化'] = time.perf_counter() - _t_blk
        elif p3d is None:
            fail_reason = rt_bound_reject_reason or "無匹配點"
            print(f"❌ [計算失敗] 在右圖中找不到與左圖點 ({u:.1f}, {v:.1f}) 的匹配點。請試著點選特徵較明顯的邊緣。")

        # 3. 計算最後的三維座標和距離
        if p3d is not None:
            if p3d[2] <= 0:
                fail_reason = "深度為負(在相機後方)"
                print(f"   [計算失敗] 原因: {fail_reason} | 原始算出Z: {p3d[2]:.2f} mm")
            elif p3d[2] > MAX_DEPTH_MM:
                fail_reason = "超過最大深度"
                print(f"   [計算失敗] 原因: {fail_reason} | 原始算出Z: {p3d[2]:.2f} mm")
            else:
                d_val = np.linalg.norm(p3d); p3d_val = p3d
                p_dist_str = "N/A"
                if cand['plane_n'] is not None:
                    p_dist = np.dot(cand['plane_n'], p3d - cand['plane_c'])
                    p_dist_str = f"{'Above' if p_dist > 0 else 'Below'} {abs(p_dist):.2f} mm"
                print(f"   [計算結果] 歐式距離: {d_val:.2f} mm | 距平面深度: {p_dist_str}")
                
                # 計算世界座標 (以 ID 最小的 ArUco 標籤中心為原點)
                if cand.get('cornersA'):
                    min_id = min(cand['cornersA'].keys())
                    
                    ok_origin = False
                    if 'curr_marker_poses' in cand and min_id in cand['curr_marker_poses']:
                        rv_o, tv_o = cand['curr_marker_poses'][min_id]
                        ok_origin = True
                    else:
                        half = ACTUAL_MARKER_SIZE_MM / 2.0
                        canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
                        ok_origin, rv_o, tv_o = cv2.solvePnP(
                            canon, cand['cornersA'][min_id], KL, np.zeros(5),
                            flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                            
                    if ok_origin:
                        R_o, _ = cv2.Rodrigues(rv_o)
                        p3d_w = R_o.T @ (p3d_val.reshape(3, 1) - tv_o.reshape(3, 1))
                        p3d_w_val = p3d_w.flatten()
        else:
            if not fail_reason:
                fail_reason = rt_bound_reject_reason or "無匹配點"
            print(f"👉 [深度計算] 左圖座標: ({u:.1f}, {v:.1f}) | 右圖匹配座標: N/A | 匹配方式: N/A")
            print(f"   [計算失敗] 原因: {fail_reason}")

        depth_z = p3d_val[2] if p3d_val is not None else None
        reproj_err = None
        if p3d_val is not None and m_pt is not None:
            rvec_rel, _ = cv2.Rodrigues(cand['R_rel'])
            pt_reproj_B, _ = cv2.projectPoints(p3d_val.reshape(1, 1, 3), rvec_rel, cand['t_rel'], KL, np.zeros(5))
            pt_reproj_B = pt_reproj_B.reshape(2)
            reproj_err = float(np.linalg.norm(m_pt - pt_reproj_B))

        # 品質與信心分數評估指標計算
        _t_blk = time.perf_counter()
        d_epi = 999.0
        zncc_score = 0.0
        masked_score = -1.0
        confidence_score = 0.0
        if p3d_val is not None and m_pt is not None:
            # 1. 極線偏離距離
            if m_pt_raw is not None:
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    d_epi = float(abs(l_B[0]*m_pt_raw[0] + l_B[1]*m_pt_raw[1] + l_B[2]) / np.sqrt(denom))
            # 2. ZNCC 外觀相似度 (使用 size=31 窗口)
            tmpl = get_patch(snap_imgA_gray, (u, v), 31)
            roi = get_patch(cand['gray'], m_pt, 31)
            if tmpl is not None and roi is not None:
                res_zncc = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                zncc_score = float(res_zncc[0, 0])
            masked_score = score_patch_match(snap_imgA_gray, cand['gray'], (u, v), m_pt, patch_size=31)
            # 3. 綜合信心度分數 (幾何與外觀聯立)
            sigma = 1.5
            geom_factor = np.exp(-(d_epi**2) / (2.0 * sigma**2)) if d_epi != 999.0 else 0.0
            confidence_score = float(max(0.0, zncc_score) * max(0.0, masked_score) * geom_factor)
            print(f"📊 [品質評估] 極線偏差: {d_epi:.2f} px | ZNCC相似度: {zncc_score:.3f} | MaskedScore: {masked_score:.3f} | 信心度: {confidence_score:.3f}")

        t_prof['品質評估'] = time.perf_counter() - _t_blk
        _detail = " | ".join(f"{k} {v * 1000.0:.0f}ms" for k, v in t_prof.items())
        print(f"⏱️ [F{cand.get('idx')}] 單影格量測 {(time.perf_counter() - t_cm_start) * 1000.0:.0f} ms（{_detail}）")
        return {'pt': m_pt, 'pt_raw': m_pt_raw, 'p3d': p3d_val, 'p3d_w': p3d_w_val, 'd': d_val, 'depth': depth_z, 'error': reproj_err, 'method': method, 'neighbors': neighbors,
                'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_groups': g_groups,
                'g_refA': g_refA, 'g_refB': g_refB,
                'g_refA_groups': g_refA_groups, 'g_refB_groups': g_refB_groups,
                'g_kptsB': g_kptsB, 'g_rect': g_rect,
                'fail_reason': fail_reason, 'u': u, 'v': v, 'trajectory': trajectory_res,
                'd_epi': d_epi, 'zncc_score': zncc_score, 'masked_score': masked_score, 'confidence_score': confidence_score}

    def compute_wound_size_with_current_v1():
        left_rect = extract_wound_rect(wound_state.get('left_pred'), locked_L_clean.shape)
        if left_rect is None:
            wound_state['size_error'] = "no left mask"
            print("[Wound V1] no left wound mask; cannot measure size")
            return None

        snap_vs = dict(view_state)
        left_img_gray = cv2.cvtColor(locked_L_clean, cv2.COLOR_BGR2GRAY)
        left_img_gray = preprocess_gray(left_img_gray, snap_vs['enable_clahe'])
        corner_source = wound_state.get('corner_source', 'min_area')
        box_key = 'bbox_box' if corner_source == 'bbox' else 'min_area_box'
        box = np.asarray(left_rect.get(box_key, left_rect['box']), dtype=np.float32)
        points_3d = []
        results = []
        right_points = []

        def compute_size_point_v1(u, v):
            all_cands = [current_cand] + extra_candidates_list
            valid_results = []
            current_res = None
            corner_left_cache = {}  # 同一角點在各候選影格間重用左圖特徵
            for cand in all_cands:
                cand_vs = dict(snap_vs)
                if DISABLE_EXTRA_CANDS_ECC_PRECISE and cand['idx'] != current_cand['idx']:
                    cand_vs['ecc'] = False
                    cand_vs['precise'] = False
                res_c = compute_measure(u, v, cand, left_img_gray, cand_vs, manual_match_pt=None, left_cache=corner_left_cache)
                res_c['cand_idx'] = cand['idx']
                res_c['baseline'] = cand.get('baseline')
                if cand['idx'] == current_cand['idx']:
                    current_res = res_c
                if res_c.get('d') is not None and res_c.get('p3d') is not None:
                    valid_results.append(res_c)

            if valid_results:
                fused = fuse_candidate_results(valid_results, best_idx=current_cand['idx'])
                best_res = dict(current_res) if current_res is not None else dict(valid_results[0])
                best_res['p3d'] = fused['p3d']
                best_res['multi_res'] = valid_results
                best_res['valid_candidate_count'] = len(fused['kept'])
                best_res['candidate_frames'] = [int(r['cand_idx']) for r in fused['kept']]
                return best_res
            return current_res

        print("[Wound V1] Measuring left-mask rectangle corners with current matching options...")
        for i, pt in enumerate(box):
            u, v = float(pt[0]), float(pt[1])
            print(f"[Wound V1] corner {i + 1}/4 left=({u:.1f}, {v:.1f})")
            res = compute_size_point_v1(u, v)
            results.append(res)
            right_pt = res.get('pt') if res is not None else None
            if right_pt is not None and np.all(np.isfinite(right_pt)):
                right_points.append(np.asarray(right_pt, dtype=np.float32))
            else:
                right_points.append(np.array([np.nan, np.nan], dtype=np.float32))
            p3d = res.get('p3d') if res is not None else None
            if p3d is None or not np.all(np.isfinite(p3d)):
                fail = res.get('fail_reason', 'no 3D') if res else 'no result'
                wound_state['size_error'] = f"corner {i + 1} failed"
                print(f"[Wound V1] corner {i + 1} failed: {fail}")
                return None
            if p3d[2] <= 0 or p3d[2] > MAX_DEPTH_MM:
                wound_state['size_error'] = f"corner {i + 1} bad depth"
                print(f"[Wound V1] corner {i + 1} invalid depth: {p3d[2]:.1f} mm")
                return None
            points_3d.append(np.asarray(p3d, dtype=np.float64))
            frames_used = res.get('candidate_frames') if res is not None else None
            if frames_used:
                print(f"[Wound V1] corner {i + 1} valid frames: {', '.join('F' + str(int(f)) for f in frames_used)}")

        points_3d = np.asarray(points_3d, dtype=np.float64)
        edges_3d = [float(np.linalg.norm(points_3d[(i + 1) % 4] - points_3d[i])) for i in range(4)]
        edges_px = [float(np.linalg.norm(box[(i + 1) % 4] - box[i])) for i in range(4)]
        long_mm = max(edges_3d)
        short_mm = min(edges_3d)
        if long_mm <= 0 or short_mm <= 0 or long_mm > MAX_DEPTH_MM:
            wound_state['size_error'] = "invalid edge length"
            return None

        wound_state['size_error'] = None
        print(f"[Wound V1] size={long_mm:.1f} x {short_mm:.1f} mm")
        return {
            'long': long_mm,
            'short': short_mm,
            'unit': 'mm',
            'pixel_long': max(edges_px),
            'pixel_short': min(edges_px),
            'left_box': box,
            'right_points': np.asarray(right_points, dtype=np.float32),
            'points_3d': points_3d,
            'corner_results': results,
            'valid_points': 4,
            'candidate_counts': [len(r.get('multi_res', [])) for r in results],
            'corner_candidate_frames': [r.get('candidate_frames', []) if r else [] for r in results],
            'area_px': left_rect['area_px'],
            'corner_source': corner_source,
            'method': 'current_v1',
        }

    flow_line_artists = []

    def clear_flow_lines():
        nonlocal flow_line_artists
        for art in flow_line_artists:
            try: art.remove()
            except: pass
        flow_line_artists.clear()

    def draw_trajectory_on_ui(trajectory):
        nonlocal flow_line_artists
        if len(trajectory) < 2: return
        pts = np.array([pt for f_idx, pt in trajectory])
        l_A, = ax_A.plot(pts[:, 0], pts[:, 1], color='#00FFFF', linestyle='-', linewidth=1.5, marker='o', markersize=2, alpha=0.8, zorder=4)
        l_B, = ax_B.plot(pts[:, 0], pts[:, 1], color='#00FFFF', linestyle='-', linewidth=1.5, marker='o', markersize=2, alpha=0.8, zorder=4)
        flow_line_artists.extend([l_A, l_B])

    def do_measure(u, v, manual_match_pt=None):
        """同步計算並立即更新 UI"""
        nonlocal last_click, locked_L, locked_R
        circle_snap_active = False
        circle_snap_dist = None
        if CIRCLE_LABEL_MATCH_ENABLED and manual_match_pt is None:
            snapped_center, snap_dist = find_nearest_circle_center(
                (u, v), circle_centers_L, CIRCLE_LABEL_SNAP_RADIUS_PX
            )
            if snapped_center is not None:
                u, v = float(snapped_center[0]), float(snapped_center[1])
                circle_snap_active = True
                circle_snap_dist = snap_dist
                print(f"[CircleLabel] snapped left click to circle center ({u:.1f}, {v:.1f}), dist={snap_dist:.1f}px")
        last_click = (u, v)
        print("\n" + "=" * 80)
        print(f"🖱️ [新點選量測] 左圖點選座標: ({float(u):.1f}, {float(v):.1f})")
        print("=" * 80)
        click_timer = StageTimer("點擊量測流程")
        locked_L = locked_L_clean.copy()
        locked_R = locked_R_clean.copy()
        mark_display_dirty()  # locked 影像重置 (量測中可能被高光過濾塗黑)
        if len(plane_dist_history) > 0:
            plane_dist_history.clear()
        for l in view_state['lines']: l.remove()
        view_state['lines'] = []
        view_state['grad_data'] = None
        redraw_grad_lines(None)  # 清除舊連線
        sift_rect.set_visible(False)
        sift_rect_center.set_visible(False)
        clear_flow_lines()
        
        snap_vs = dict(view_state)
        snap_vs['circle_label_match_active'] = circle_snap_active
        snap_vs['circle_label_snap_dist'] = circle_snap_dist
        left_img_gray = cv2.cvtColor(locked_L, cv2.COLOR_BGR2GRAY)
        left_img_gray = preprocess_gray(left_img_gray, snap_vs['enable_clahe'])
        click_timer.stage("左圖灰階+CLAHE前處理")
        
        all_cands = [current_cand] + extra_candidates_list
        res_list = []
        current_res = None
        click_left_cache = {}  # 左圖特徵/描述子在各候選影格間重用 (同一點選點必然相同)
        for cand in all_cands:
            cand_role = "BEST" if cand['idx'] == current_cand['idx'] else "EXTRA"
            print(f"-------- Right F{cand['idx']} [{cand_role}] --------")
            # 性能優化：依據變數控制是否在次要影格中停用耗時的 ECC 亞像素精修與精細匹配
            cand_vs = dict(snap_vs)
            if DISABLE_EXTRA_CANDS_ECC_PRECISE and cand['idx'] != current_cand['idx']:
                cand_vs['ecc'] = False
                cand_vs['precise'] = False
            res_c = compute_measure(u, v, cand, left_img_gray, cand_vs, manual_match_pt, left_cache=click_left_cache)
            click_timer.stage(f"右圖F{cand['idx']} 匹配+三角化")
            res_c['cand_idx'] = cand['idx']
            res_c['baseline'] = cand.get('baseline')
            if cand['idx'] == current_cand['idx']:
                current_res = res_c
            if res_c.get('d') is not None and res_c.get('p3d') is not None:
                res_list.append(res_c)
                
        if current_res is None:
            print(f"-------- Right F{current_cand['idx']} [BEST-RETRY] --------")
            current_res = compute_measure(u, v, current_cand, left_img_gray, snap_vs, manual_match_pt, left_cache=click_left_cache)
            click_timer.stage(f"F{current_cand['idx']} 重試")
            current_res['cand_idx'] = current_cand['idx']

        res = dict(current_res)
        res['p3d_best'] = res.get('p3d')  # 最優對自身的 3D 點 (與參考平面同一幾何鏈)
        if res_list:
            fused = fuse_candidate_results(res_list, best_idx=current_cand['idx'])
            avg_p3d = fused['p3d']
            avg_d = fused['d']
            avg_depth = avg_p3d[2]
            kept = fused['kept']
            avg_error = np.mean([r['error'] for r in kept if r.get('error') is not None]) if any(r.get('error') is not None for r in kept) else 0.0
            avg_p3d_w = fused['p3d_w']

            res['multi_res'] = res_list
            res['multi_avg_p3d'] = avg_p3d
            res['multi_avg_d'] = avg_d
            res['multi_avg_depth'] = avg_depth
            res['multi_avg_error'] = avg_error
            res['multi_avg_p3d_w'] = avg_p3d_w
            # 顯示與存檔統一使用融合結果（與傷口 V1 尺寸量測行為一致）
            res['p3d'] = avg_p3d
            res['d'] = avg_d
            res['depth'] = avg_depth
            if avg_p3d_w is not None:
                res['p3d_w'] = avg_p3d_w
            res['fused_count'] = len(kept)
            res['fused_dropped'] = [int(r['cand_idx']) for r in fused['dropped']]

            print(f"📊 [多對融合深度] 左圖 F{current_cand['idx']} 與最多 {len(all_cands)} 個右圖進行計算：")
            for r, w in zip(kept, fused['weights']):
                is_best = " (最優)" if r['cand_idx'] == current_cand['idx'] else ""
                print(f"  - 右圖 F{r['cand_idx']}{is_best}: 深度 = {r['d']:.2f} mm, 權重 = {w:.2f}, 3D = [{r['p3d'][0]:.2f}, {r['p3d'][1]:.2f}, {r['p3d'][2]:.2f}]")
            for r in fused['dropped']:
                print(f"  - 右圖 F{r['cand_idx']}: 深度 = {r['d']:.2f} mm ({r.get('drop_reason', '離群剔除')})")
            print(f"  ➡️ 加權融合結果 (採用 {len(kept)}/{len(res_list)} 組): 深度 = {avg_d:.2f} mm, 3D = [{avg_p3d[0]:.2f}, {avg_p3d[1]:.2f}, {avg_p3d[2]:.2f}]")
        click_timer.stage("多影格融合")
        
        if custom_plane_mode:
            if res.get('p3d') is not None:
                custom_plane_pts_3d.append(res['p3d'])
                custom_plane_pts_2d.append((res['u'], res['v']))
                c_pt, = ax_A.plot(res['u'], res['v'], 'mo', markersize=6, zorder=5)
                t_lbl = ax_A.text(res['u'] + 5, res['v'] - 5, f"P{len(custom_plane_pts_3d)}", 
                                  color='magenta', fontsize=9, fontweight='bold', zorder=5)
                custom_plane_artists.extend([c_pt, t_lbl])
                redraw_custom_plane_poly()
                btn_custom_plane.label.set_text(f"Finish Fit ({len(custom_plane_pts_3d)})")
                print(f"🎯 自訂平面已新增點 P{len(custom_plane_pts_3d)}: (u, v)=({res['u']:.1f}, {res['v']:.1f}), 3D={res['p3d']}")
            else:
                print("❌ 點選點之深度計算無效，無法加入自訂平面點！")
            res['custom_plane_pick_mode'] = True
            res['custom_plane_pick_valid'] = res.get('p3d') is not None
            res['custom_plane_pick_count'] = len(custom_plane_pts_3d)
                
        if res.get('trajectory') is not None:
            draw_trajectory_on_ui(res['trajectory'])
                
        measure_results[current_cand['idx']] = res
        all_d = [r['d'] for r in measure_results.values() if r['d'] is not None]
        summary = [f"F{current_cand['idx']}: {res['d']:.1f}" if res['d'] is not None else f"F{current_cand['idx']}: N/A"]
        
        # 存出數據至 txt 檔案
        save_measurement_to_txt(
            VIDEO_PATH, res, current_cand, wound_z_offset, 
            custom_plane_n, custom_plane_c, custom_plane_fitted, MEASURE_MODE
        )
        click_timer.stage("結果整理+數據存檔")
        
        apply_measure_result(res, np.mean(all_d) if all_d else None, summary)
        click_timer.stage("UI 更新繪製")
        click_timer.report()


    def apply_measure_result(res, avg, summary):
        """在主執行緒中，用 compute_measure 的純資料結果更新所有 Matplotlib UI 元件。"""
        nonlocal last_click
        u, v = res['u'], res['v']
        last_click = (u, v)
        for l in view_state['lines']: l.remove()
        view_state['lines'] = []
        view_state['grad_data'] = None
        redraw_grad_lines(None)
        sift_rect.set_visible(False)
        sift_rect_center.set_visible(False)
        update_display(avg, summary)
    
    plane_dist_history = collections.deque(maxlen=15)
    
    def update_display(avg, summary):
        res = measure_results.get(current_cand['idx'], {'pt': None, 'neighbors': [], 'p3d': None, 'g_ptsA': None, 'g_groups': None})
        u, v = last_click if last_click else (0, 0)
        scatter_A.set_offsets([[u, v]])
        for l in view_state['lines']: l.remove()
        view_state['lines'] = []
        # 清除舊連線 Artists
        for item in view_state['grad_lines']:
            try:
                con = item[0] if isinstance(item, tuple) else item
                con.remove()
            except: pass
        view_state['grad_lines'] = []
        old_h = view_state.get('highlighted_grad_line_artist')
        if old_h is not None:
            try: old_h.remove()
            except: pass
        view_state['highlighted_grad_line_artist'] = None
        view_state['highlighted_grad_line'] = None
        view_state['grad_data'] = None
        
        sift_rect.set_visible(False)
        sift_rect_center.set_visible(False)
        scatter_grad_ref_A.set_offsets(np.empty((0,2)))
        scatter_grad_ref_B.set_offsets(np.empty((0,2)))
        scatter_mid_grad_ref_A.set_offsets(np.empty((0,2)))
        scatter_mid_grad_ref_B.set_offsets(np.empty((0,2)))
        scatter_grad_inject.set_offsets(np.empty((0,2)))
        scatter_grad_match.set_offsets(np.empty((0,2)))
        scatter_mid_grad_inject.set_offsets(np.empty((0,2)))
        scatter_mid_grad_match.set_offsets(np.empty((0,2)))
        scatter_circle_A.set_offsets(circle_centers_L if len(circle_centers_L) else _empty_points())
        scatter_circle_B.set_offsets(current_cand.get('circle_centers', _empty_points()))
        
        if res.get('g_refA') is not None and res.get('g_refB') is not None:
            refA_groups = res.get('g_refA_groups')
            refB_groups = res.get('g_refB_groups')
            if refA_groups is not None and refB_groups is not None:
                refA_groups = np.asarray(refA_groups)
                refB_groups = np.asarray(refB_groups)
                refA_high_mask = refA_groups != "mid"
                refA_mid_mask = refA_groups == "mid"
                refB_high_mask = refB_groups != "mid"
                refB_mid_mask = refB_groups == "mid"
                scatter_grad_ref_A.set_offsets(res['g_refA'][refA_high_mask] if np.any(refA_high_mask) else np.empty((0,2)))
                scatter_grad_ref_B.set_offsets(res['g_refB'][refB_high_mask] if np.any(refB_high_mask) else np.empty((0,2)))
                scatter_mid_grad_ref_A.set_offsets(res['g_refA'][refA_mid_mask] if np.any(refA_mid_mask) else np.empty((0,2)))
                scatter_mid_grad_ref_B.set_offsets(res['g_refB'][refB_mid_mask] if np.any(refB_mid_mask) else np.empty((0,2)))
            else:
                scatter_grad_ref_A.set_offsets(res['g_refA'])
                scatter_grad_ref_B.set_offsets(res['g_refB'])
        if res.get('g_rect') is not None:
            sift_rect.set_bounds(*res['g_rect'])
            sift_rect.set_visible(True)
            # 更新 Rect 中心標記
            rx, ry, rw, rh = res['g_rect']
            sift_rect_center.set_data([rx + rw/2], [ry + rh/2])
            sift_rect_center.set_visible(True)
            
            if res.get('g_ptsA') is not None:
                groups = res.get('g_groups')
                if groups is not None:
                    groups = np.asarray(groups)
                    high_mask = groups != "mid"
                    mid_mask = groups == "mid"
                    scatter_grad_inject.set_offsets(res['g_ptsA'][high_mask] if np.any(high_mask) else np.empty((0,2)))
                    scatter_grad_match.set_offsets(res['g_ptsB'][high_mask] if np.any(high_mask) else np.empty((0,2)))
                    scatter_mid_grad_inject.set_offsets(res['g_ptsA'][mid_mask] if np.any(mid_mask) else np.empty((0,2)))
                    scatter_mid_grad_match.set_offsets(res['g_ptsB'][mid_mask] if np.any(mid_mask) else np.empty((0,2)))
                else:
                    scatter_grad_inject.set_offsets(res['g_ptsA'])
                    scatter_grad_match.set_offsets(res['g_ptsB'])
                    scatter_mid_grad_inject.set_offsets(np.empty((0,2)))
                    scatter_mid_grad_match.set_offsets(np.empty((0,2)))
                view_state['grad_data'] = {'ptsA': res['g_ptsA'], 'ptsB': res['g_ptsB'], 'groups': groups}
                redraw_grad_lines(None)  # 初始無高亮
            else:
                scatter_grad_inject.set_offsets(np.empty((0,2)))
                scatter_grad_match.set_offsets(np.empty((0,2)))
                scatter_mid_grad_inject.set_offsets(np.empty((0,2)))
                scatter_mid_grad_match.set_offsets(np.empty((0,2)))
        elif current_cand.get('pose_valid', False):
            seed_pt, seed_method = predict_right_seed_from_geometry((u, v), current_cand, KL)
            u_exp, v_exp = float(seed_pt[0]), float(seed_pt[1])
            if 0 <= u_exp < w and 0 <= v_exp < h:
                rad = RIGHT_PATCH_SEARCH_RADIUS
                sift_rect.set_bounds(u_exp - rad, v_exp - rad, rad*2, rad*2)
                sift_rect.set_visible(True)
                sift_rect_center.set_data([u_exp], [v_exp])
                sift_rect_center.set_visible(True)
            else:
                print(f"⚠️ [預估搜尋框繪製失敗] {seed_method} 預測點 ({u_exp:.1f}, {v_exp:.1f}) 超出影像邊界。")
                

        pose_info_str = current_cand.get('pose_info', '')
        if res['pt'] is not None:
            scatter_B.set_offsets([[res['pt'][0], res['pt'][1]]])
            p0, p1 = epipolar_line(current_cand['F'], (u, v), w); epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            
            # 計算三角化 3D 點重投影
            pt_reproj_B_tri, pt_reproj_A_tri = None, None
            if res['p3d'] is not None:
                rvec_rel, _ = cv2.Rodrigues(current_cand['R_rel'])
                pt_reproj_B_tri, _ = cv2.projectPoints(res['p3d'].reshape(1, 1, 3), rvec_rel, current_cand['t_rel'], KL, np.zeros(5))
                pt_reproj_B_tri = pt_reproj_B_tri.reshape(2)
                
                pt_reproj_A_tri, _ = cv2.projectPoints(res['p3d'].reshape(1, 1, 3), np.zeros(3), np.zeros(3), KL, np.zeros(5))
                pt_reproj_A_tri = pt_reproj_A_tri.reshape(2)

            # 畫面紫色圓圈 (預設使用平面單應性，若無效退回三角化)
            has_plane_reproj = False
            if current_cand.get('plane_n') is not None:
                d_plane = np.dot(current_cand['plane_n'], current_cand['plane_c'])
                if abs(d_plane) > 1e-6:
                    try:
                        H_AB = current_cand['K_R'] @ (current_cand['R_rel'] + (current_cand['t_rel'] @ current_cand['plane_n'].reshape(1, 3)) / d_plane) @ np.linalg.inv(KL)
                        H_BA = np.linalg.inv(H_AB)
                        
                        pt_p_B = H_AB @ np.array([u, v, 1.0])
                        pt_reproj_B_plane = np.array([pt_p_B[0]/pt_p_B[2], pt_p_B[1]/pt_p_B[2]])
                        
                        pt_p_A = H_BA @ np.array([res['pt'][0], res['pt'][1], 1.0])
                        pt_reproj_A_plane = np.array([pt_p_A[0]/pt_p_A[2], pt_p_A[1]/pt_p_A[2]])
                        
                        scatter_B_reproj.set_offsets([[pt_reproj_B_plane[0], pt_reproj_B_plane[1]]])
                        scatter_A_reproj.set_offsets([[pt_reproj_A_plane[0], pt_reproj_A_plane[1]]])
                        
                        err_L_plane = np.linalg.norm(np.array([u, v]) - pt_reproj_A_plane)
                        err_R_plane = np.linalg.norm(np.array(res['pt']) - pt_reproj_B_plane)
                        print(f"📊 [平面單應性重投影誤差] 左圖 (點選點 vs 右圖點平面反投影): {err_L_plane:.2f} px | 右圖 (匹配點 vs 左圖點平面正投影): {err_R_plane:.2f} px")
                        has_plane_reproj = True
                    except Exception as e:
                        print(f"⚠️ [單應性計算出錯] {e}，退回傳統三角化重投影")
            
            if not has_plane_reproj:
                if pt_reproj_B_tri is not None and pt_reproj_A_tri is not None:
                    scatter_B_reproj.set_offsets([[pt_reproj_B_tri[0], pt_reproj_B_tri[1]]])
                    scatter_A_reproj.set_offsets([[pt_reproj_A_tri[0], pt_reproj_A_tri[1]]])
                else:
                    scatter_B_reproj.set_offsets(np.empty((0, 2)))
                    scatter_A_reproj.set_offsets(np.empty((0, 2)))
            
            # 列印三角化重投影誤差資訊供 Debug
            if pt_reproj_B_tri is not None and pt_reproj_A_tri is not None:
                err_L_tri = np.linalg.norm(np.array([u, v]) - pt_reproj_A_tri)
                err_R_tri_aligned = np.linalg.norm(np.array(res['pt']) - pt_reproj_B_tri)
                print(f"📊 [三角化 3D 重投影誤差]")
                print(f"   - 左圖 (點選點 vs 3D點投影): {err_L_tri:.2f} px")
                print(f"   - 右圖 (極線對齊點 vs 3D點投影): {err_R_tri_aligned:.2f} px")
                if res.get('pt_raw') is not None:
                    err_R_tri_raw = np.linalg.norm(np.array(res['pt_raw']) - pt_reproj_B_tri)
                    print(f"   - 右圖 (原始未對齊匹配點 vs 3D點投影): {err_R_tri_raw:.2f} px (💡 反映特徵點偏離極線程度)")
                
            p_dist_str = ""
            if (custom_plane_fitted and res['p3d'] is not None
                    and custom_plane_n is not None and custom_plane_c is not None):
                p_dist = np.dot(custom_plane_n, res['p3d'] - custom_plane_c)
                if auto_calc_active:
                    plane_dist_history.append(p_dist)
                    p_dist_str = f"\nWound Height (Custom Plane): {np.mean(plane_dist_history):.1f}mm"
                else:
                    plane_dist_history.clear()
                    p_dist_str = f"\nWound Height (Custom Plane): {p_dist:.1f}mm"
            elif res['p3d'] is not None and current_cand['plane_n'] is not None:
                # 與平面同鏈: 高度用最優對的 p3d (平面即由最優對 RT 三角化)，誤差相消才成立
                _p3d_plane = res['p3d_best'] if res.get('p3d_best') is not None else res['p3d']
                p_dist = (np.dot(current_cand['plane_n'], _p3d_plane - current_cand['plane_c']))
                if auto_calc_active:
                    plane_dist_history.append(p_dist)
                    p_dist_str = f"\nWound Height: {np.mean(plane_dist_history):.1f}mm"
                else:
                    plane_dist_history.clear()
                    p_dist_str = f"\nWound Height: {p_dist:.1f}mm"
            
            if res['depth'] is not None:
                # 這裡的 res['depth'] 就是左相機坐標系下的 z 座標
                #main_text = f"深度: {res['depth']:.1f}mm{p_dist_str}{h_diff_str}\n誤差: {res['error']:.3f}px\n配對: {res['method']}\n外參來源: {pose_info_str}"
                score_str = ""
                if view_state.get('show_score', False) and res.get('confidence_score') is not None:
                    score_str = f"\nConfidence: {res['confidence_score']:.3f} (Epipolar:{res['d_epi']:.1f}px, ZNCC:{res['zncc_score']:.2f})"
                main_text = f"Camera-to-Selected Position Distance: {res['depth']:.1f}mm{p_dist_str}{score_str}\n"
            
            
            else:
                main_text = f"Depth: calculation failed\nPose source: {pose_info_str}"
            
        else:
            scatter_B.set_offsets(np.empty((0,2)))
            scatter_B_reproj.set_offsets(np.empty((0,2)))
            scatter_A_reproj.set_offsets(np.empty((0,2)))
            epi_line.set_data([], [])
            fail_reason = ui_failure_reason_english(res.get('fail_reason', 'No matching point'))
            main_text = f"Invalid point ({fail_reason})\nPose source: {pose_info_str}"
        if res.get('custom_plane_pick_mode'):
            if res.get('custom_plane_pick_valid'):
                main_text = (
                    f"Custom plane mode\n"
                    "Continue selecting points or press Finish Fit."
                )
            else:
                main_text = (
                    "Custom plane mode\n"
                    "Invalid point. Try another location."
                )
        depth_text.set_text(main_text)
        request_blit_refresh()


    pan_state = {'pressing': False, 'x': None, 'y': None, 'ax': None, 'dragged': False, 'dragging_hud': False}
    def on_press(event):
        if event.button != 1: return
        
        # 檢查是否點擊在深度數值 HUD 區域內
        try:
            bbox = depth_text.get_window_extent(fig.canvas.get_renderer())
            # 擴大偵測框以提升點擊靈敏度
            bbox_padded = bbox.expanded(1.2, 1.2)
            if bbox_padded.contains(event.x, event.y):
                pan_state['dragging_hud'] = True
                # 計算滑鼠相對 Figure 座標系與文字原點的位移量，避免拖曳起步時瞬移
                inv = fig.transFigure.inverted()
                mx, my = inv.transform((event.x, event.y))
                tx, ty = depth_text.get_position()
                pan_state['hud_offset'] = (tx - mx, ty - my)
                return
        except Exception:
            pass
            
        if event.inaxes not in (ax_A, ax_B): return
        pan_state.update({'pressing': True, 'dragged': False, 'x': event.x, 'y': event.y, 'ax': event.inaxes})
 
    def on_release(event):
        if pan_state.get('dragging_hud', False):
            pan_state['dragging_hud'] = False
            return
        if not pan_state['pressing']: return
        pan_state['pressing'] = False
        if not pan_state['dragged'] and event.xdata is not None:
            ux, vx = float(event.xdata), float(event.ydata)
            
            # 自動吸附 ArUco 角點
            if pan_state['ax'] == ax_A:
                ux, vx = snap_to_aruco_corner(ux, vx, current_cand['cornersA'])
            elif pan_state['ax'] == ax_B:
                ux, vx = snap_to_aruco_corner(ux, vx, current_cand['cornersB'])
                
            if view_state['manual']:
                if pan_state['ax'] == ax_A:
                    view_state['manual_pt_A'] = (ux, vx); nonlocal last_click; last_click = (ux, vx)
                    scatter_A.set_offsets([[ux, vx]])
                    p0, p1 = epipolar_line(current_cand['F'], (ux, vx), w)
                    epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                    depth_text.set_text("Manual mode: select the matching point on the right epipolar line")
                    request_blit_refresh()
                elif pan_state['ax'] == ax_B:
                    # 有 grad_data 時，點擊右圖做高亮（不管連線目前是否顯示）
                    if view_state.get('grad_data') and not view_state['manual_pt_A']:
                        ptsB_arr = view_state['grad_data']['ptsB']
                        dists = np.linalg.norm(ptsB_arr - np.array([ux, vx]), axis=1)
                        nearest_idx = int(np.argmin(dists))
                        view_state['show_grad_lines'] = True
                        btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線')
                        redraw_grad_lines(nearest_idx)
                        request_blit_refresh()
                    elif view_state['manual_pt_A']:
                        do_measure(view_state['manual_pt_A'][0], view_state['manual_pt_A'][1], manual_match_pt=np.array([ux, vx]))
            else:
                if pan_state['ax'] == ax_A:
                    nonlocal active_u, active_v
                    active_u = int(round(ux))
                    active_v = int(round(vx))
                    do_measure(active_u, active_v)
                elif pan_state['ax'] == ax_B and view_state.get('grad_data'):
                    ptsB_arr = view_state['grad_data']['ptsB']
                    dists = np.linalg.norm(ptsB_arr - np.array([ux, vx]), axis=1)
                    nearest_idx = int(np.argmin(dists))
                    view_state['show_grad_lines'] = True
                    btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線')
                    redraw_grad_lines(nearest_idx)
                    request_blit_refresh()
    def on_motion(event):
        if pan_state.get('dragging_hud', False):
            # 直接使用 Figure 座標系之逆變換計算新位置，避免綁定 ax_B 導致跨 axes 拖曳卡死
            inv = fig.transFigure.inverted()
            mx, my = inv.transform((event.x, event.y))
            ox, oy = pan_state.get('hud_offset', (0, 0))
            new_x = max(0.01, min(0.95, mx + ox))
            new_y = max(0.01, min(0.95, my + oy))
            depth_text.set_position((new_x, new_y))
            request_blit_refresh()
            return
            
        if not pan_state['pressing'] or event.inaxes != pan_state['ax']: return
        dx, dy = event.x - pan_state['x'], event.y - pan_state['y']
        if not pan_state['dragged'] and abs(dx) < 3 and abs(dy) < 3: return
        pan_state['dragged'] = True
        ax = pan_state['ax']
        inv = ax.transData.inverted()
        p0, p1 = inv.transform((pan_state['x'], pan_state['y'])), inv.transform((event.x, event.y))
        dx_d, dy_d = p1 - p0
        ax.set_xlim(ax.get_xlim() - dx_d); ax.set_ylim(ax.get_ylim() - dy_d)
        pan_state.update({'x': event.x, 'y': event.y})
        request_blit_refresh()

    def on_scroll(event):
        if event.inaxes not in (ax_A, ax_B): return
        ax, f = event.inaxes, 1.2 if event.button == 'down' else 1/1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim([x - (x-xl[0])*f, x + (xl[1]-x)*f]); ax.set_ylim([y - (y-yl[0])*f, y + (yl[1]-y)*f])
        request_blit_refresh()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('button_press_event', on_press)

    # 手動鎖定模式的狀態變數
    locked_L = imgA_bgr.copy()
    locked_R = imgB_bgr.copy()
    live_L = False
    live_R = False
    has_set_L = True
    has_set_R = True
    reset_pose_history = True
    
    # 自訂平面擬合狀態變數
    custom_plane_mode = False
    custom_plane_pts_3d = []
    custom_plane_pts_2d = []
    custom_plane_n = None
    custom_plane_c = None
    custom_plane_fitted = False
    custom_plane_artists = []
    custom_plane_poly_artist = None

    def redraw_custom_plane_poly():
        nonlocal custom_plane_poly_artist
        if custom_plane_poly_artist is not None:
            try: custom_plane_poly_artist.remove()
            except: pass
            custom_plane_poly_artist = None
            
        if len(custom_plane_pts_2d) >= 3:
            pts2d = np.array(custom_plane_pts_2d)
            poly = Polygon(pts2d, closed=True, facecolor='magenta', edgecolor='magenta', alpha=0.15, zorder=3)
            ax_A.add_patch(poly)
            custom_plane_poly_artist = poly
    
    # 離線模式下已在 main 函數中初始化 current_cand，此處無需重設
    
    # 建立按鈕 (已統一尺寸、排列，並升級為精緻的「微發光邊框」與「功能分色」設計)
    # 使用更深邃的背景色 (#1A1A1A)，與主背景形成對比
    btn_style = dict(color='#1A1A1A', hovercolor='#333333')
    
    ax_btn_lock_L = fig.add_axes([0.58, 0.92, 0.08, 0.04])
    btn_lock_L = Button(ax_btn_lock_L, "鎖定左圖", **btn_style)
    
    ax_btn_lock_R = fig.add_axes([0.68, 0.92, 0.08, 0.04])
    btn_lock_R = Button(ax_btn_lock_R, "鎖定右圖", **btn_style)
    
    ax_btn_hide_R = fig.add_axes([0.78, 0.92, 0.08, 0.04])
    btn_hide_R = Button(ax_btn_hide_R, "顯示右圖", **btn_style)
    
    ax_btn_norm = fig.add_axes([0.88, 0.92, 0.08, 0.04])
    btn_norm_toggle = Button(ax_btn_norm, '使用 HAMMING', **btn_style)
    
    ax_btn_calc = fig.add_axes([0.58, 0.86, 0.08, 0.04])
    btn_calc = Button(ax_btn_calc, "單次計算深度", **btn_style)
    
    ax_btn_auto_calc = fig.add_axes([0.68, 0.86, 0.08, 0.04])
    btn_auto_calc = Button(ax_btn_auto_calc, "連續計算: 關", **btn_style)
    
    ax_btn_grad = fig.add_axes([0.78, 0.86, 0.08, 0.04])
    btn_grad_toggle = Button(ax_btn_grad, '顯示梯度 SIFT 連線', **btn_style)
    
    ax_btn_custom_plane = fig.add_axes([0.88, 0.86, 0.08, 0.04])
    btn_custom_plane = Button(ax_btn_custom_plane, "Custom Plane", **btn_style)
    
    ax_btn_high_grad_pts = fig.add_axes([0.58, 0.80, 0.08, 0.04])
    btn_high_grad_pts = Button(ax_btn_high_grad_pts, "HighPts: Off", **btn_style)
    
    ax_btn_mid_grad_pts = fig.add_axes([0.68, 0.80, 0.08, 0.04])
    btn_mid_grad_pts = Button(ax_btn_mid_grad_pts, "MidPts: Off", **btn_style)
    
    ax_btn_rt_diff = fig.add_axes([0.78, 0.80, 0.08, 0.04])
    btn_rt_diff = Button(ax_btn_rt_diff, "RT Diff", **btn_style)
    
    ax_btn_return_menu = fig.add_axes([0.88, 0.80, 0.08, 0.04])
    btn_return_menu = Button(ax_btn_return_menu, "Back to Menu", **btn_style)
    
    # 建立 TextBox 用於傷口高度補償
    ax_btn_wound = fig.add_axes([0.58, 0.74, 0.08, 0.04])
    btn_wound_toggle = Button(ax_btn_wound, "Wound: Off", **btn_style)

    ax_btn_wound_pts = fig.add_axes([0.68, 0.74, 0.08, 0.04])
    btn_wound_pts_toggle = Button(ax_btn_wound_pts, "Pts: Rect", **btn_style)

    ax_btn_aruco_overlay = fig.add_axes([0.78, 0.74, 0.08, 0.04])
    btn_aruco_overlay = Button(ax_btn_aruco_overlay, "ArUco標記: Off", **btn_style)

    ax_btn_contour_debug = fig.add_axes([0.88, 0.74, 0.08, 0.04])
    btn_contour_debug = Button(ax_btn_contour_debug, "Show Contours", **btn_style)

    ax_btn_rt_sift = fig.add_axes([0.88, 0.68, 0.08, 0.04])
    btn_rt_sift = Button(ax_btn_rt_sift, "RT SIFT: Off", **btn_style)

    wound_z_offset = 0.0
    ax_box = fig.add_axes([0.02, 0.02, 0.04, 0.04])
    text_box = TextBox(ax_box, "", initial="0.0", color='#1A1A1A', hovercolor='#333333')#傷口高度補償(mm): 
    text_box.label.set_color('#E0E0E0')
    text_box.label.set_fontsize(8)
    text_box.text_disp.set_color('#E0E0E0')
    text_box.text_disp.set_fontsize(8)
    ax_box.patch.set_linewidth(1.2)
    ax_box.patch.set_edgecolor('#007ACC')
    
    def submit_z_offset(text):
        nonlocal wound_z_offset
        try:
            wound_z_offset = float(text)
            print(f"✏️ 已設定傷口 Z 軸高度補償量: {wound_z_offset} mm")
            if last_click:
                do_measure(last_click[0], last_click[1])
        except ValueError:
            print("⚠️ 請輸入有效的數字")
            
    text_box.on_submit(submit_z_offset)
    
    # 統一設定字型、文字顏色與邊框寬度
    for b in [btn_lock_L, btn_lock_R, btn_hide_R, btn_norm_toggle, btn_calc, btn_auto_calc, btn_grad_toggle, btn_custom_plane, btn_high_grad_pts, btn_mid_grad_pts, btn_rt_diff, btn_return_menu, btn_wound_toggle, btn_wound_pts_toggle, btn_aruco_overlay, btn_contour_debug, btn_rt_sift]:
        b.label.set_color('#E0E0E0') # 質感白
        b.label.set_fontsize(8)
        b.ax.patch.set_linewidth(1.2) # 細緻邊框
        
    # 依功能進行邊框分色（專業軟體常見的語意化色彩）
    # 1. 影像鎖定/控制類：使用專業藍 (#007ACC)
    for b in [btn_lock_L, btn_lock_R, btn_hide_R]:
        b.ax.patch.set_edgecolor('#007ACC')
        
    # 2. 深度計算類：使用警告橘/強調橘 (#D83B01)
    for b in [btn_calc, btn_auto_calc]:
        b.ax.patch.set_edgecolor('#D83B01')
        
    # 3. 功能切換類：使用中性的深灰 (#555555)
    for b in [btn_norm_toggle, btn_grad_toggle, btn_custom_plane, btn_high_grad_pts, btn_mid_grad_pts, btn_rt_diff, btn_wound_toggle, btn_wound_pts_toggle, btn_aruco_overlay, btn_contour_debug]:
        b.ax.patch.set_edgecolor('#555555')
        
    # 4. 導覽/返回選單類：使用翡翠綠 (#28A745)
    btn_return_menu.ax.patch.set_edgecolor('#28A745')
        
    btn_grad_toggle.label.set_fontsize(7) # 特長文字微調
    btn_contour_debug.label.set_fontsize(7)
    
    def on_grad_toggle(event):
        view_state['show_grad_lines'] = not view_state['show_grad_lines']
        btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線' if view_state['show_grad_lines'] else '顯示梯度 SIFT 連線')
        redraw_grad_lines(view_state.get('highlighted_grad_line'))
        request_blit_refresh()
        
    def on_high_grad_pts_toggle(event):
        view_state['show_high_grad_points'] = not view_state['show_high_grad_points']
        visible = view_state['show_high_grad_points']
        btn_high_grad_pts.label.set_text("HighPts: On" if visible else "HighPts: Off")
        for artist in (scatter_grad_ref_A, scatter_grad_ref_B, scatter_grad_inject, scatter_grad_match):
            artist.set_visible(visible)
        request_blit_refresh()

    def on_mid_grad_pts_toggle(event):
        view_state['show_mid_grad_points'] = not view_state['show_mid_grad_points']
        visible = view_state['show_mid_grad_points']
        btn_mid_grad_pts.label.set_text("MidPts: On" if visible else "MidPts: Off")
        for artist in (scatter_mid_grad_ref_A, scatter_mid_grad_ref_B, scatter_mid_grad_inject, scatter_mid_grad_match):
            artist.set_visible(visible)
        request_blit_refresh()

    def on_wound_toggle(event):
        wound_state['show'] = not wound_state['show']
        if wound_state['left_pred'] is None and wound_state['right_pred'] is None:
            refresh_wound_predictions("toggle")
        if wound_state['show'] and (wound_state.get('dirty', False) or wound_state.get('v1_size') is None):
            update_wound_size_from_current_v1("toggle")
        btn_wound_toggle.label.set_text("Wound: On" if wound_state['show'] else "Wound: Off")
        btn_wound_toggle.ax.patch.set_edgecolor('#28A745' if wound_state['show'] else '#555555')
        print(
            f"[Wound] Overlay {'shown' if wound_state['show'] else 'hidden'} "
            f"(left={wound_state['left_count']} right={wound_state['right_count']})"
        )
        request_blit_refresh()

    def on_wound_pts_toggle(event):
        wound_state['corner_source'] = 'bbox' if wound_state.get('corner_source') == 'min_area' else 'min_area'
        use_bbox = wound_state['corner_source'] == 'bbox'
        btn_wound_pts_toggle.label.set_text("Pts: BBox" if use_bbox else "Pts: Rect")
        btn_wound_pts_toggle.ax.patch.set_edgecolor('#28A745' if use_bbox else '#555555')
        print(f"[Wound] Corner source: {'AI bbox' if use_bbox else 'minAreaRect'}")
        mark_wound_size_dirty('corner_source')
        request_blit_refresh()

    def on_norm_toggle(event):
        view_state['use_hamming'] = not view_state['use_hamming']
        btn_norm_toggle.label.set_text('使用 HAMMING' if view_state['use_hamming'] else '使用 L2')
        mark_wound_size_dirty('use_hamming')
        request_blit_refresh()

    def on_rt_diff(event):
        if current_cand.get('plane_n') is None or current_cand.get('plane_c') is None:
            print("⚠️ [RT Diff] 缺少 plane_n / plane_c，無法用 RT + 平面單應性 warp 左圖。")
            return
        d_plane = float(np.dot(current_cand['plane_n'], current_cand['plane_c']))
        if abs(d_plane) < 1e-6:
            print("⚠️ [RT Diff] 平面距離 d_plane 趨近 0，無法計算 homography。")
            return

        left_gray = cv2.cvtColor(locked_L_clean, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(locked_R_clean, cv2.COLOR_BGR2GRAY)
        hR, wR = right_gray.shape[:2]
        H_AB = current_cand['K_R'] @ (
            current_cand['R_rel'] + (current_cand['t_rel'] @ current_cand['plane_n'].reshape(1, 3)) / d_plane
        ) @ np.linalg.inv(KL)

        warped_left = cv2.warpPerspective(
            left_gray, H_AB, (wR, hR),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        valid = cv2.warpPerspective(
            np.ones(left_gray.shape[:2], dtype=np.uint8) * 255, H_AB, (wR, hR),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        diff = cv2.absdiff(warped_left, right_gray)
        diff[valid == 0] = 0

        diff_fig, diff_ax = plt.subplots(1, 1, figsize=(9, 6), facecolor='#1E1E1E')
        diff_fig.canvas.manager.set_window_title("RT Warp Gray Difference")
        diff_ax.imshow(diff, cmap='gray', vmin=0, vmax=255)
        diff_ax.set_title("abs(gray(warp(left by RT+plane)) - gray(right))", color='white')
        diff_ax.axis("off")
        diff_ax.set_facecolor('#1E1E1E')
        diff_fig.tight_layout()
        diff_fig.show()
        valid_mean = float(np.mean(diff[valid > 0])) if np.any(valid > 0) else 0.0
        print(f"📊 [RT Diff] 已產生相減圖 | d_plane={d_plane:.3f} | valid diff mean={valid_mean:.2f}")

    def on_show_contours(event):
        left_debug = collect_circle_label_contours(locked_L_clean)
        right_bgr = locked_R_clean
        right_debug = collect_circle_label_contours(right_bgr)

        left_overlay = draw_circle_label_contour_debug_overlay(locked_L_clean, left_debug)
        right_overlay = draw_circle_label_contour_debug_overlay(right_bgr, right_debug)

        dbg_fig, dbg_axes = plt.subplots(2, 2, figsize=(13, 8), facecolor='#1E1E1E')
        try:
            dbg_fig.canvas.manager.set_window_title("Circle Label Contour Debug")
        except Exception:
            pass

        panels = [
            (dbg_axes[0, 0], left_debug['mask'], "Left binary mask", 'gray'),
            (dbg_axes[0, 1], left_overlay, f"Left contours: {len(left_debug['centers'])} accepted", None),
            (dbg_axes[1, 0], right_debug['mask'], "Right binary mask", 'gray'),
            (dbg_axes[1, 1], right_overlay, f"Right contours: {len(right_debug['centers'])} accepted", None),
        ]
        for ax_dbg, img_dbg, title, cmap in panels:
            ax_dbg.set_facecolor('#1E1E1E')
            if img_dbg is not None:
                ax_dbg.imshow(img_dbg, cmap=cmap)
            ax_dbg.set_title(title, color='white')
            ax_dbg.axis('off')

        dbg_fig.tight_layout()
        dbg_fig.show()
        print(
            f"[CircleLabel Debug] left accepted={len(left_debug['centers'])}/{len(left_debug['contours'])}, "
            f"right accepted={len(right_debug['centers'])}/{len(right_debug['contours'])}"
        )
        
    def on_return_menu(event):
        view_state['restart'] = True
        plt.close(fig)
        print("🔄 正在關閉目前量測介面並返回影片來源選單...")
    
    def on_hide_R(event):
        visible = ax_B.get_visible()
        ax_B.set_visible(not visible)
        btn_hide_R.label.set_text("顯示右圖" if visible else "隱藏右圖")
        if visible:
            depth_text.set_position((0.53, 0.35))
        else:
            depth_text.set_position((0.53, 0.02))
        request_blit_refresh()
    
    auto_calc_active = False
    
    def on_auto_calc(event):
        nonlocal auto_calc_active, reset_pose_history
        if custom_plane_mode:
            print("⚠️ 自訂平面選點中，無法開啟連續計算！")
            return
        auto_calc_active = not auto_calc_active
        if auto_calc_active:
            btn_auto_calc.label.set_text("連續計算: 開")
            print("▶️ 開啟連續計算模式")
        else:
            btn_auto_calc.label.set_text("連續計算: 關")
            print("⏸️ 關閉連續計算模式")
        reset_pose_history = True
        request_blit_refresh()
    
    def on_lock_L(event):
        # 離線影片模式沒有 Live 串流可供重新鎖定，左圖固定為分析挑出的最優影格
        print("⚠️ 離線影片模式不支援重新鎖定左圖，左圖固定為分析挑出的最優影格。")

    def on_lock_R(event):
        # 離線影片模式沒有 Live 串流可供重新鎖定，右圖固定為分析挑出的最優影格
        print("⚠️ 離線影片模式不支援重新鎖定右圖，右圖固定為分析挑出的最優影格。")
        
    def on_custom_plane(event):
        nonlocal custom_plane_mode, custom_plane_n, custom_plane_c, custom_plane_fitted, auto_calc_active
        
        # 1. 檢查先決條件：必須鎖定右圖，且外參有效
        if live_R:
            print("⚠️ 請先鎖定右圖再開始自訂平面選點！")
            depth_text.set_text("Hint: lock the right image before selecting a custom plane")
            request_blit_refresh()
            return
        if not current_cand.get('pose_valid', False):
            print("⚠️ 外參/Baseline無效，無法計算3D座標，請先確保 ArUco 定位成功！")
            depth_text.set_text("Hint: pose is invalid. Make sure ArUco is detected")
            request_blit_refresh()
            return
            
        if not custom_plane_mode:
            # 2. 進入選點模式
            custom_plane_mode = True
            custom_plane_pts_3d.clear()
            custom_plane_pts_2d.clear()
            custom_plane_n = None
            custom_plane_c = None
            custom_plane_fitted = False
            
            # 清除舊的 matplotlib 標記與多邊形
            for art in custom_plane_artists:
                try: art.remove()
                except: pass
            custom_plane_artists.clear()
            redraw_custom_plane_poly()
            
            # 強制關閉連續計算
            if auto_calc_active:
                auto_calc_active = False
                btn_auto_calc.label.set_text("連續計算: 關")
                print("⏸️ 連續計算模式已自動關閉")
                
            btn_custom_plane.label.set_text("Finish Fit (0)")
            btn_custom_plane.ax.patch.set_facecolor('#8A2BE2') # 變為紫羅蘭色
            btn_custom_plane.ax.patch.set_edgecolor('#8A2BE2')
            print("🎯 已進入「自訂平面選點模式」，請在左圖點選至少 3 個點...")
            depth_text.set_text("Custom plane mode: select at least 3 points on the left image")
            request_blit_refresh()
        else:
            # 3. 按下按鈕完成或取消擬合
            if len(custom_plane_pts_3d) == 0:
                # 0 個點，取消此模式並清除平面
                custom_plane_mode = False
                custom_plane_fitted = False
                btn_custom_plane.label.set_text("Custom Wound Plane")
                btn_custom_plane.ax.patch.set_facecolor('#1A1A1A')
                btn_custom_plane.ax.patch.set_edgecolor('#555555')
                print("❌ 已取消自訂平面擬合，並清除自訂平面。")
                if last_click:
                    do_measure(last_click[0], last_click[1])
                else:
                    depth_text.set_text("Custom plane cleared")
                    request_blit_refresh()
                return
                
            if len(custom_plane_pts_3d) < 3:
                print(f"⚠️ 點數不足 (當前僅 {len(custom_plane_pts_3d)} 個點)，擬合平面至少需要 3 個點！")
                depth_text.set_text(f"Error: not enough points ({len(custom_plane_pts_3d)}/3). Continue selecting points")
                request_blit_refresh()
                return
                
            # 4. RANSAC (>=6 點時) + SVD 擬合平面，並回報各點殘差供品質判斷
            pts = np.array(custom_plane_pts_3d, dtype=np.float64)
            n, c, inlier_mask, resid = fit_plane_to_points(pts)

            custom_plane_n = n
            custom_plane_c = c
            custom_plane_fitted = True
            custom_plane_mode = False

            btn_custom_plane.label.set_text("Custom Wound Plane")
            btn_custom_plane.ax.patch.set_facecolor('#1A1A1A')
            btn_custom_plane.ax.patch.set_edgecolor('#555555')
            n_inl = int(np.count_nonzero(inlier_mask))
            rms = float(np.sqrt(np.mean(resid[inlier_mask] ** 2)))
            print(f"✅ 自訂平面擬合成功！")
            print(f"  - 擬合點數: {n_inl}/{len(pts)}" + (" (RANSAC 已剔除離群點)" if n_inl < len(pts) else ""))
            print(f"  - 平面中心: {c}")
            print(f"  - 平面法向: {n}")
            for i, r_val in enumerate(resid):
                tag = "" if inlier_mask[i] else " ⚠️ (離群，未參與擬合)"
                print(f"  - P{i + 1} 殘差: {r_val:+.2f} mm{tag}")
            print(f"  - 內點 RMS 殘差: {rms:.2f} mm" + (" ⚠️ 殘差偏大，建議重新選點" if rms > 2.0 else ""))
            
            # 重新計算當前選取點，以獲得與新平面的距離
            if last_click:
                do_measure(last_click[0], last_click[1])
            else:
                depth_text.set_text(f"Custom plane fitted. Points: {len(pts)}")
                request_blit_refresh()
        
    def on_calc(event):
        if last_click:
            do_measure(last_click[0], last_click[1])
        else:
            do_measure(active_u, active_v)
    def apply_aruco_overlay_visibility():
        visible = view_state['show_aruco_overlay']
        for _ax in (ax_A, ax_B):
            for _a in getattr(_ax, 'art', []):
                _a.set_visible(visible)
            # 重投影框 (洋紅: 左投右 / 橘: 右投左) 一併控制
            for _a in getattr(_ax, 'reproj_art', []):
                _a.set_visible(visible)

    def on_aruco_overlay_toggle(event):
        view_state['show_aruco_overlay'] = not view_state['show_aruco_overlay']
        btn_aruco_overlay.label.set_text("ArUco標記: On" if view_state['show_aruco_overlay'] else "ArUco標記: Off")
        apply_aruco_overlay_visibility()
        request_blit_refresh()

    btn_aruco_overlay.on_clicked(on_aruco_overlay_toggle)
    apply_aruco_overlay_visibility()  # 預設隱藏 ArUco 偵測框與 ID 標籤

    def on_rt_sift_toggle(event):
        if rt_sift_inlier_count == 0:
            print("⚠️ 此最佳影像對沒有可顯示的 RT SIFT recoverPose 內點。")
            return
        view_state['show_rt_sift_points'] = not view_state['show_rt_sift_points']
        visible = view_state['show_rt_sift_points']
        scatter_rt_sift_A.set_visible(visible)
        scatter_rt_sift_B.set_visible(visible)
        btn_rt_sift.label.set_text("RT SIFT: On" if visible else "RT SIFT: Off")
        if visible:
            role_text = "已套用於最終 RT" if rt_sift_applied else "僅參與 RT 驗證，最終保留 ArUco RT"
            diagnostics_text = rt_sift_diagnostics_path or init_txt_path
            print(
                f"📍 RT SIFT recoverPose 內點: {rt_sift_inlier_count}/{rt_sift_match_count}，"
                f"{role_text}；完整分層診斷已記錄於 {diagnostics_text}")
        request_blit_refresh()

    btn_rt_sift.on_clicked(on_rt_sift_toggle)

    btn_lock_L.on_clicked(on_lock_L)
    btn_lock_R.on_clicked(on_lock_R)
    btn_calc.on_clicked(on_calc)
    btn_auto_calc.on_clicked(on_auto_calc)
    btn_hide_R.on_clicked(on_hide_R)
    btn_grad_toggle.on_clicked(on_grad_toggle)
    btn_high_grad_pts.on_clicked(on_high_grad_pts_toggle)
    btn_mid_grad_pts.on_clicked(on_mid_grad_pts_toggle)
    btn_wound_toggle.on_clicked(on_wound_toggle)
    btn_wound_pts_toggle.on_clicked(on_wound_pts_toggle)
    btn_norm_toggle.on_clicked(on_norm_toggle)
    btn_custom_plane.on_clicked(on_custom_plane)
    btn_rt_diff.on_clicked(on_rt_diff)
    btn_contour_debug.on_clicked(on_show_contours)
    btn_return_menu.on_clicked(on_return_menu)
    # ---- 三區按鈕顯示/隱藏控制：左上角三個圓點，預設全部隱藏（返回主選單與自訂傷口平面不受影響）----
    panel_defs = [
        ('#00BFFF', [ax_c1, ax_c2, ax_c3, ax_c4, ax_c5, ax_c6, ax_c7, ax_c8, ax_c9,
                     ax_c10, ax_c11, ax_c12, ax_c13, ax_c14, ax_c15, ax_c16, ax_c17, ax_c19]),
        ('#00FF88', [ax_mode]),
        ('#FFAA00', [ax_btn_lock_L, ax_btn_lock_R, ax_btn_hide_R, ax_btn_norm,
                     ax_btn_calc, ax_btn_auto_calc, ax_btn_grad,
                     ax_btn_high_grad_pts, ax_btn_mid_grad_pts, ax_btn_rt_diff,
                     ax_btn_wound, ax_btn_wound_pts, ax_btn_aruco_overlay, ax_btn_rt_sift]),
        ('#FF6688', [pose_status_text]),  # 右下角姿態估計狀態 label (set_visible 對 Text artist 同樣有效)
    ]
    panel_visible = [False, False, False, False]
    panel_dot_buttons = []

    def make_panel_toggle(idx):
        def _toggle(event):
            panel_visible[idx] = not panel_visible[idx]
            for ax_p in panel_defs[idx][1]:
                ax_p.set_visible(panel_visible[idx])
            panel_dot_buttons[idx].label.set_color(
                panel_defs[idx][0] if panel_visible[idx] else '#555555')
            request_blit_refresh()
        return _toggle

    for _i, (_color, _axes_list) in enumerate(panel_defs):
        for _ax_p in _axes_list:
            _ax_p.set_visible(panel_visible[_i])
        _ax_dot = fig.add_axes([0.005 + _i * 0.025, 0.965, 0.02, 0.03])
        _dot = Button(_ax_dot, '●', color='#1E1E1E', hovercolor='#333333')
        _dot.label.set_color(_color if panel_visible[_i] else '#555555')
        _dot.label.set_fontsize(11)
        _ax_dot.patch.set_edgecolor('none')
        for _spine in _ax_dot.spines.values():
            _spine.set_visible(False)
        panel_dot_buttons.append(_dot)
        _dot.on_clicked(make_panel_toggle(_i))

    startup_timer.stage("UI 元件建立")
    refresh_wound_predictions("initial selection")
    startup_timer.stage("傷口模型預載+推論")

    # ---- Blit 初始化 ----
    # 切斷 im_A/im_B 的 stale propagation callback：
    # im.set_data() 會把 artist 標為 stale，stale 向上傳遞到 figure 後
    # 觸發 canvas.draw_idle()，最終讓 flush_events() 執行完整重繪。
    # 由於 im_A/im_B 由我們的 blit 路徑手動管理，不需要這個機制。
    im_A._stale_callback = None
    im_B._stale_callback = None

    # Monkey-patch draw_idle：按鈕/Widget 觸發的 draw_idle 只需設 flag
    fig.canvas.draw_idle = request_blit_refresh

    # 顯示視窗並做初始全繪，存成靜態背景
    plt.show(block=False)
    fig.canvas.draw()
    blit_state['bg'] = fig.canvas.copy_from_bbox(fig.bbox)
    blit_state['needs_refresh'] = False
    startup_timer.stage("首次繪製")
    startup_timer.report()

    import time as _time
    _fps_t0 = _time.perf_counter()
    _fps_counter = 0
    _fps_val = 0.0
    # 各階段耗時累計 (單位: ms)
    _t_cap = _t_buf = _t_calc_q = _t_result_q = _t_proc = _t_setdata = _t_pause = 0.0
    _perf_frames = 0
    _perf_t0 = _time.perf_counter()
    _last_auto_calc_time = 0.0

    # 更改 FPS 文字為靜態影片標籤
    fps_text.set_text("Mode: Video (Offline)")
    
    while plt.fignum_exists(fig.number):
        if not blit_state['needs_refresh'] and blit_state['bg'] is not None:
            fig.canvas.flush_events()
            _time.sleep(UI_LOOP_SLEEP_SEC)
            continue
        # 依前處理開關，動態切換顯示畫面（使肉眼可見差異）。
        # 影像內容 (version) 與疊圖相關開關沒變時，直接重用上次的轉換結果。
        disp_key = (
            display_cache['version'],
            view_state['enable_clahe'],
            view_state.get('show_spatial_specular_mask', False),
            view_state.get('show_temporal_specular_mask', False),
            wound_state.get('show', False),
            wound_state.get('corner_source'),
        )
        if disp_key != display_cache['key']:
            if view_state['enable_clahe']:
                gray_A = cv2.cvtColor(locked_L, cv2.COLOR_BGR2GRAY)
                gray_A_enh = preprocess_gray(gray_A, True)
                disp_A = cv2.cvtColor(gray_A_enh, cv2.COLOR_GRAY2RGB)

                gray_B = cv2.cvtColor(locked_R, cv2.COLOR_BGR2GRAY)
                gray_B_enh = preprocess_gray(gray_B, True)
                disp_B = cv2.cvtColor(gray_B_enh, cv2.COLOR_GRAY2RGB)
            else:
                disp_A = cv2.cvtColor(locked_L, cv2.COLOR_BGR2RGB)
                disp_B = cv2.cvtColor(locked_R, cv2.COLOR_BGR2RGB)

            if view_state.get('show_spatial_specular_mask', False) or view_state.get('show_temporal_specular_mask', False):
                empty_L = np.zeros_like(locked_L_spec_mask) if locked_L_spec_mask is not None else None
                empty_R = np.zeros_like(locked_R_spec_mask) if locked_R_spec_mask is not None else None
                spatial_A = locked_L_spec_spatial_mask if view_state.get('show_spatial_specular_mask', False) else empty_L
                temporal_A = locked_L_spec_temporal_mask if view_state.get('show_temporal_specular_mask', False) else empty_L
                spatial_B = locked_R_spec_spatial_mask if view_state.get('show_spatial_specular_mask', False) else empty_R
                temporal_B = locked_R_spec_temporal_mask if view_state.get('show_temporal_specular_mask', False) else empty_R
                disp_A = overlay_specular_mask_rgb(disp_A, spatial_A, temporal_A)
                disp_B = overlay_specular_mask_rgb(disp_B, spatial_B, temporal_B)

            disp_A, disp_B = apply_wound_overlay_if_enabled(disp_A, disp_B)
            display_cache['key'] = disp_key
            display_cache['disp_A'] = disp_A
            display_cache['disp_B'] = disp_B

        im_A.set_data(display_cache['disp_A'])
        im_B.set_data(display_cache['disp_B'])

        # ---- Blit 渲染 ----
        if blit_state['needs_refresh'] or blit_state['bg'] is None:
            fig.canvas.draw()
            blit_state['bg'] = fig.canvas.copy_from_bbox(fig.bbox)
            blit_state['needs_refresh'] = False
        else:
            fig.canvas.restore_region(blit_state['bg'])

        ax_A.draw_artist(im_A)
        if custom_plane_poly_artist is not None:
            ax_A.draw_artist(custom_plane_poly_artist)
        ax_A.draw_artist(scatter_A)
        ax_A.draw_artist(scatter_A_reproj)
        ax_A.draw_artist(scatter_grad_ref_A)
        ax_A.draw_artist(scatter_mid_grad_ref_A)
        ax_A.draw_artist(scatter_grad_inject)
        ax_A.draw_artist(scatter_mid_grad_inject)
        ax_A.draw_artist(scatter_rt_sift_A)
        ax_A.draw_artist(scatter_circle_A)
        for a in custom_plane_artists:
            ax_A.draw_artist(a)
        
        if ax_B.get_visible():
            ax_B.draw_artist(im_B)
            ax_B.draw_artist(scatter_B)
            ax_B.draw_artist(scatter_B_reproj)
            ax_B.draw_artist(scatter_grad_ref_B)
            ax_B.draw_artist(scatter_mid_grad_ref_B)
            ax_B.draw_artist(scatter_grad_match)
            ax_B.draw_artist(scatter_mid_grad_match)
            ax_B.draw_artist(scatter_rt_sift_B)
            ax_B.draw_artist(scatter_circle_B)
            ax_B.draw_artist(epi_line)
            ax_B.draw_artist(sift_rect)
            ax_B.draw_artist(sift_rect_center)
            
            for line in view_state.get('grad_lines', []):
                ax_B.draw_artist(line)
            if view_state.get('highlighted_grad_line_artist'):
                ax_B.draw_artist(view_state['highlighted_grad_line_artist'])
                
        fig.draw_artist(depth_text)

        for ax in [ax_A, ax_B]:
            if hasattr(ax, 'art'):
                if ax == ax_B and not ax_B.get_visible():
                    continue
                for a in ax.art:
                    ax.draw_artist(a)
            if hasattr(ax, 'reproj_art'):
                if ax == ax_B and not ax_B.get_visible():
                    continue
                for a in ax.reproj_art:
                    ax.draw_artist(a)

        ax_A.draw_artist(fps_text)
        fig.draw_artist(pose_status_text)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        _time.sleep(UI_LOOP_SLEEP_SEC)

    return view_state.get('restart', False)

if __name__ == "__main__":
    while True:
        should_restart = main()
        if not should_restart:
            break
        import matplotlib.pyplot as plt
        plt.close('all')
