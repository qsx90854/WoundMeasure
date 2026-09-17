"""
depth_measure_multi_aruco_sbs_camera.py
================
互動式深度量測工具 (SBS 併排影片 + JSON 標定參數版本)。

特點：
- 支援影片輸入，可指定左圖幀與多個右圖候選幀
- 整合 JSON 標定參數，支援左右相機不對稱的內參與畸
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
from Algorithm.aruco_id_filter import filter_marker_detections, filter_marker_mapping
from Algorithm import camera_preprocess as camera_algo
# Isolated temporal RT implementation; the original video_pose_analysis.py is
# intentionally retained unchanged for direct A/B fallback.
#from Algorithm import video_pose_analysis_temporal as video_pose_algo #old version
from Algorithm import video_pose_analysis_temporal_unified_pattern_guided_local_window as video_pose_algo
from Algorithm.perf_timer import StageTimer
from Algorithm.lossless_recording import open_lossless_writer
from Algorithm.block_height_accuracy import (
    BlockAccuracyConfig, BlockAccuracyReport, build_block_plan, measurement_record,
    block_mae_color, summarize_block_records,
)
from Algorithm.specular_detection import (
    BlockSpatialSpecularConfig,
    compute_specular_mask_bgr_block_adaptive,
    compute_specular_mask_bgr_wound_adaptive,
    compute_rt_aligned_temporal_specular_mask_bgr,
    overlay_specular_mask_rgb,
)
from Algorithm import stereo_matching as stereo_algo
from Algorithm.Region_SIFT_Matching import (
    RegionSIFTConfig,
    run_region_sift_matching,
)
from Algorithm.region_sift_settings import show_region_sift_settings
from Algorithm.region_sift_diagnostics import RegionSIFTDiagnostics
from Algorithm.drag_height_profile import DragHeightProfile, DragProfileConfig
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
# Drag-path samples use displayed image coordinates, not physical millimetres.
DRAG_PROFILE_CONFIG = DragProfileConfig(spacing_px=5.0, max_points=100, pick_radius_px=12.0)
# RT and reference-plane marker allowlist. None=all, []=none. Restart after editing.
ARUCO_ALLOWED_PATTERN_IDS = None #[9,12]#[2, 5, 9, 12]
# Stepped calibration model: 6 columns x 4 rows, row-major heights in mm.
# 25/50/75% in each cell = nine anchor centers, inset 5 mm for a 20 mm cell.
BLOCK_ACCURACY_CONFIG = BlockAccuracyConfig()
BLOCK_ACCURACY_OUTPUT_DIR = BASE_DIR / 'measurement_accuracy'
WOUND_DETECTION_DIR = BASE_DIR / "wound_detection_model"
WOUND_MODEL_PATH = WOUND_DETECTION_DIR / "model" / "assets" / "v9-t-seg_320.onnx"
WOUND_OVERLAY_ALPHA = 0.45
ENABLE_WOUND_AI = False                           # False: 不載入傷口 AI 模型，也不執行推論
# Adaptive Spatial 開啟但無有效傷口遮罩時，全圖按影像小區塊調整門檻。
# 與手動圈選的 6x4 高度模型無關；其餘可調欄位見 BlockSpatialSpecularConfig。
SPATIAL_BLOCK_CONFIG = BlockSpatialSpecularConfig(
    block_width_px=64,
    block_height_px=64,
    v_percentile= 10,#90.0,
    rgb_percentile=5.0, #92
    local_hot_percentile=10,#85.0,
    interpolate_thresholds=True,
    dilate_px=0,
    enable_prominence_gate=True,           # False 回到原本三條分支 OR 的結果
    prominence_min=3.0,                    # 局部亮度突出門檻下限；降低較寬鬆
    prominence_mad_multiplier=2.0,         # 區塊亮度差 MAD 倍數；降低較寬鬆
    enable_strong_highlight_exception=True,
    strong_v_min=250.0,                    # 強反光例外：近飽和 + 同時符合兩項顏色限制
    strong_s_max=30.0,
    strong_whiteness_max=20.0,
)
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
# Frame-pair proposal mode: "original" keeps the historical five probes;
# "angle_guided" scans ArUco poses and targets right/A=15 deg, left/B=35 deg.
FRAME_PAIR_SELECTION_MODE = "original"#"angle_guided"
ANGLE_GUIDED_DIRECTION_MODE = "auto"  # auto / 15_to_35 / 35_to_15
ANGLE_GUIDED_RIGHT_TARGET_DEG = 15.0
ANGLE_GUIDED_LEFT_TARGET_DEG = 35.0
ANGLE_GUIDED_TARGET_TOLERANCE_DEG = 6.0
ANGLE_GUIDED_COARSE_SAMPLES_PER_SEGMENT = 12
ANGLE_GUIDED_MAX_SCAN_FRAMES_PER_SEGMENT = 18
ANGLE_GUIDED_CANDIDATES_PER_SIDE = 3
ANGLE_GUIDED_PAIR_SCORE_WEIGHT = 0.35
# RT SIFT-only ROI. ArUco detection and pair-geometry scoring remain full-frame.
# Ratio format: (x, y, width, height), normalized to the original camera frame.
ENABLE_RT_SIFT_ROI = True
RT_SIFT_ROI_RATIO = (0.10, 0.10, 0.80, 0.80)
RT_SIFT_IMAGE_SCALE = 0.5
# True: Marker 世界座標重投影 + SIFT Sampson residual 聯合精修最終 RT。
# False: 最終 RT 只做 marker-only world optimization；SIFT 仍保留作為
#        影格配對與 validation-only 品質檢查，不會改動 R/t。
ENABLE_WORLD_MARKER_SIFT_RT_REFINE = True
# True: 保留 PRE / MARKER / FINAL 的完整 RT 與 delta Log；整段會用
#       醒目的分隔線包住，避免與選幀、SIFT 及品質 Log 混在一起。
WORLD_RT_AUDIT_VERBOSE = True
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

PARAMS_JSON_PATH      = "calibration_result_Zebra_1_monocular.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM = 8.25                       # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
DEFAULT_WOUND_HEIGHT_OFFSET_MM = 0.0              # 未使用自定義平面時，Wound Height 顯示扣除值 (mm)
MIN_BASELINE_MM       = 35.0#8.0                        # 最小基準線限制 (mm)
MAX_BASELINE_MM       = 220.0                      # 最大基準線限制 (mm)
AUTO_CALC_INTERVAL_SEC = 0.2                       # 連續計算模式下的計算時間間隔 (秒)
ENFORCE_COPLANAR      = False                      # 強制共面對齊優化
SAVE_ARUCO_DEBUG_IMG  = False                      # 是否存出 ArUco 偵測結果圖片
ENABLE_POSE_SMOOTHING  = True                      # 是否啟用時序平滑濾波 (EMA)
POSE_SMOOTHING_ALPHA   = 0.3                         # 平滑係數
ENABLE_ECC_REFINEMENT_DEFAULT = False              # Grad-SIFT/ORB debug：先關閉點擊匹配後的 ECC 精修
ENABLE_CLAHE_DEFAULT  = False                      # 預設是否啟用 CLAHE
CLAHE_CLIP_LIMIT      = 2.0                        # CLAHE 對比度限制閾值 (數值愈大對比愈強，雜訊也愈大)
CLAHE_TILE_GRID_SIZE  = (8, 8)                     # CLAHE 分塊大小 (8, 8) 代表 8x8 的網格
ENABLE_IMPROVED_MATCHING_DEFAULT = False          # 預設是否啟用改良版特徵匹配流程 (高光遮罩 + Harris Corner + 收緊幾何門檻 + 金字塔 ECC)
SHOW_SCORE_DEFAULT = False                         # 預設是否顯示匹配品質與信心分數
SHOW_LEFT_REPROJECTION_CIRCLE = False              # 是否顯示右圖匹配點反投影回左圖的粉紅虛線圓圈
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
RIGHT_PATCH_SEARCH_RADIUS     = 75#75#40#30                         # 右圖預測投影點周圍的搜索半徑 (pixels)
GRAD_SIFT_MAX_RT_ADJUST_PX    = 75#75#40.0                       # v1 Grad-SIFT 允許相對 RT/平面預測 seed 的最大微調量 (pixels)
LEFT_GRADIENT_POINTS_COUNT    = 150                         # 左圖周圍取梯度最高的特徵點數量
RIGHT_GRADIENT_POINTS_COUNT   = 500                       # 右圖周圍取梯度最高的特徵點數量
LEFT_MID_GRADIENT_POINTS_COUNT = 150                        # 左圖周圍取梯度中等的特徵點數量
RIGHT_MID_GRADIENT_POINTS_COUNT = 500                      # 右圖周圍取梯度中等的特徵點數量

GRAD_SIFT_RATIO_TEST          = 0.78                       # v1 Grad-SIFT KNN ratio test threshold
GRAD_SIFT_EPIPOLAR_TOL_PX     = 3.0                        # max point-to-epipolar-line distance for local SIFT matches
GRAD_SIFT_OFFSET_MEDIAN_TOL_PX = 8.0                       # reject local matches whose disparity differs too much from median
GRAD_SIFT_RANSAC_REPROJ_PX    = 2.5                        # local affine RANSAC reprojection threshold
GRAD_SIFT_MIN_GROUP_INLIERS   = 3                          # minimum inliers for accepting one high/mid gradient group
GRAD_SIFT_GUIDED_RADIUS_PX    = 10.0                       # guided fallback: search right refs near RT/plane-predicted location
GRAD_SIFT_GUIDED_RATIO_TEST   = 0.95                       # guided fallback uses geometry, so descriptor ambiguity can be looser
# Region-SIFT tuning.  Defaults produce 27 automatic samples + click P,
# a 28x128 descriptor matrix and a rotated 5x75 epipolar band. GroupScore
# blends best-21 with cell-balanced ALL-row L2; only reliable frames add a penalty.
REGION_SIFT_CONFIG = RegionSIFTConfig(
    grid_rows=3,
    grid_cols=3,
    cell_width_px=10,
    cell_height_px=10,
    points_per_cell=3,
    sobel_ksize=3,
    min_point_distance_px=1.0,
    exclude_click_from_cell_points=True,
    auto_scale_orientation=True,
    scale_keypoint_sizes_px=(3.2, 4.0, 5.0, 6.4, 8.0, 10.0, 12.0, 16.0),
    sift_n_octave_layers=3,
    sift_sigma=1.6,
    descriptor_max_support_radius_px=40.0,  # nominal SIFT radius; None disables cap
    flat_keypoint_size_px=3.2,
    scale_min_confidence=0.05,
    orientation_min_confidence=0.10,
    scale_boundary_is_reliable=False,
    scale_dog_ratio=2.0 ** (1.0 / 3.0),
    scale_response_pool_sigma_factor=0.75,
    scale_response_floor=1e-4,
    orientation_bins=36,
    orientation_sigma_factor=1.5,
    orientation_radius_factor=3.0,
    orientation_hist_smooth_passes=2,
    frame_coordinate_quantization_px=0.0,  # 0=exact; raise only for speed tuning
    # Fallback frame only; used in locally flat areas or when auto is off.
    keypoint_size_px=10.0,
    keypoint_angle_deg=0.0,
    require_all_descriptors=True,
    normalize_descriptors=False,
    search_length_px=95.0,
    search_width_px=5.0,
    search_along_step_px=1.0,
    search_across_step_px=1.0,
    keep_best_ratio=0.75,
    keep_best_count=None,
    max_group_score=500.0,           # BestG > 500 時匹配失敗；None 關閉
    max_objective_score_ratio=0.95,  # ObjRatio > 0.95 時匹配失敗；None 關閉
    epipolar_penalty_weight=0.0,
    second_best_exclusion_radius_px=3.0,
    group_balance_weight=0.35,  # G = 0.65 * Trim21 + 0.35 * CellAll
    frame_consistency_weight=0.05,
    frame_scale_tolerance_log2=0.5,
    frame_angle_tolerance_deg=30.0,
    frame_min_reliable_pairs=3,
    reject_uninformative_group=True,
    reject_flat_score_surface=True,
    flat_score_relative_tolerance=1e-6,
    warp_interpolation=cv2.INTER_LINEAR,
    min_valid_warp_ratio=1.0,
    descriptor_border_margin_px=8,
    descriptor_batch_size=4096,
    specular_check_support=False,  # Reject SpecPts: 完整 support 避反光；False 只避中心點
)
# Debug-only alternative to the original guided fallback.  Ratio-rejected
# points may select only their original Global Top-1/Top-2 using exact H(pL).
TOP2_GEOMETRY_RESCUE_DEFAULT  = False
TOP2_GEOMETRY_MAX_DIST_PX     = 35.0
# Debug-only physical block audit.  This never rejects a match or changes a
# Grad-SIFT score; it only visualizes/records local sparse-depth consistency.
DEBUG_METRIC_BLOCK_SIZE_MM    = 5.0
DEBUG_METRIC_BLOCKS_DEFAULT   = False
DEBUG_METRIC_BLOCK_GRID_RADIUS = 1                         # selected block + one neighboring block on each side
DEBUG_METRIC_BLOCK_MIN_POINTS = 3
DEBUG_METRIC_BLOCK_BASE_TOL_MM = 1.0
DEBUG_METRIC_BLOCK_SIGMA_D_PX = 1.0
DEBUG_METRIC_BLOCK_SIGMA_MULT = 3.0
DEBUG_METRIC_BLOCK_MAX_REPROJ_PX = 3.5
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

def get_wound_detector():
    """Lazy-load the v9-t-seg_320 wound segmentation model."""
    global _WOUND_DETECTOR, _WOUND_DETECTOR_ERROR_LOGGED
    if not ENABLE_WOUND_AI:
        return None
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
    if not ENABLE_WOUND_AI:
        return None
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


def compute_shared_marker_corner_plane(corners_left, corners_right,
                                       K_L, K_R, R_rel, t_rel, F=None,
                                       reference_normal=None,
                                       min_shared_markers=2):
    """Triangulate every valid corner of shared markers and fit one 3D plane.

    The returned plane is expressed in the left-camera coordinate system.  A
    marker only participates when all four corresponding corners triangulate
    to finite points in front of both cameras.  Unlike ``fit_plane_to_points``,
    this diagnostic reference intentionally uses every accepted marker corner
    (no point-level RANSAC), so the fitted plane means exactly "all shared
    patterns" and the residual report exposes any non-coplanarity.
    """
    diag = {
        'available': False,
        'reason': '',
        'shared_marker_ids': [],
        'used_marker_ids': [],
        'skipped_markers': {},
        'point_count': 0,
        'rms_mm': None,
        'p90_abs_mm': None,
        'max_abs_mm': None,
        'per_marker': {},
    }
    if not isinstance(corners_left, dict) or not isinstance(corners_right, dict):
        diag['reason'] = 'marker corner dictionaries are unavailable'
        return None, None, diag

    corners_left = filter_marker_mapping(corners_left, ARUCO_ALLOWED_PATTERN_IDS)
    corners_right = filter_marker_mapping(corners_right, ARUCO_ALLOWED_PATTERN_IDS)
    shared_ids = sorted(set(corners_left.keys()) & set(corners_right.keys()))
    diag['shared_marker_ids'] = [int(mid) for mid in shared_ids]
    if len(shared_ids) < int(min_shared_markers):
        diag['reason'] = (
            f'need at least {int(min_shared_markers)} shared markers; '
            f'found {len(shared_ids)}')
        return None, None, diag

    R = np.asarray(R_rel, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_rel, dtype=np.float64).reshape(3)
    all_points = []
    point_marker_ids = []
    for mid in shared_ids:
        try:
            pts_l = np.asarray(corners_left[mid], dtype=np.float64).reshape(-1, 2)
            pts_r = np.asarray(corners_right[mid], dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError):
            diag['skipped_markers'][int(mid)] = 'invalid corner array'
            continue
        if len(pts_l) != 4 or len(pts_r) != 4:
            diag['skipped_markers'][int(mid)] = (
                f'expected 4/4 corners, got {len(pts_l)}/{len(pts_r)}')
            continue
        if not np.all(np.isfinite(pts_l)) or not np.all(np.isfinite(pts_r)):
            diag['skipped_markers'][int(mid)] = 'non-finite 2D corner'
            continue

        marker_points = []
        marker_error = None
        for corner_idx, (pt_l, pt_r) in enumerate(zip(pts_l, pts_r)):
            try:
                p3d = np.asarray(triangulate_point_3d(
                    pt_l, pt_r, K_L, K_R, R, t.reshape(3, 1), F=F),
                    dtype=np.float64).reshape(3)
            except (cv2.error, FloatingPointError, TypeError, ValueError) as exc:
                marker_error = f'corner {corner_idx}: triangulation failed ({exc})'
                break
            if not np.all(np.isfinite(p3d)):
                marker_error = f'corner {corner_idx}: non-finite 3D point'
                break
            right_z = float((R @ p3d + t)[2])
            if p3d[2] <= 0.0 or right_z <= 0.0:
                marker_error = f'corner {corner_idx}: point is behind a camera'
                break
            if p3d[2] > MAX_DEPTH_MM or right_z > MAX_DEPTH_MM:
                marker_error = f'corner {corner_idx}: depth exceeds {MAX_DEPTH_MM} mm'
                break
            marker_points.append(p3d)

        if marker_error is not None or len(marker_points) != 4:
            diag['skipped_markers'][int(mid)] = marker_error or 'incomplete 3D corners'
            continue
        all_points.extend(marker_points)
        point_marker_ids.extend([int(mid)] * 4)
        diag['used_marker_ids'].append(int(mid))

    if len(diag['used_marker_ids']) < int(min_shared_markers):
        diag['reason'] = (
            f'only {len(diag["used_marker_ids"])} shared markers have four valid '
            f'3D corners; need {int(min_shared_markers)}')
        return None, None, diag

    pts = np.asarray(all_points, dtype=np.float64)
    c = pts.mean(axis=0)
    try:
        _, singular_values, Vt = np.linalg.svd(pts - c, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        diag['reason'] = f'plane SVD failed ({exc})'
        return None, None, diag
    if len(singular_values) < 3 or singular_values[1] <= 1e-9:
        diag['reason'] = 'triangulated corners are geometrically degenerate'
        return None, None, diag

    n = np.asarray(Vt[-1], dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if not np.isfinite(n_norm) or n_norm <= 1e-12:
        diag['reason'] = 'fitted plane normal is invalid'
        return None, None, diag
    n /= n_norm
    if reference_normal is not None:
        ref_n = np.asarray(reference_normal, dtype=np.float64).reshape(3)
        ref_norm = float(np.linalg.norm(ref_n))
        if ref_norm > 1e-12 and float(np.dot(n, ref_n / ref_norm)) < 0.0:
            n = -n
    elif float(np.dot(n, c)) > 0.0:
        n = -n

    residuals = (pts - c) @ n
    abs_residuals = np.abs(residuals)
    diag['point_count'] = int(len(pts))
    diag['rms_mm'] = float(np.sqrt(np.mean(residuals ** 2)))
    diag['p90_abs_mm'] = float(np.percentile(abs_residuals, 90))
    diag['max_abs_mm'] = float(np.max(abs_residuals))
    for mid in diag['used_marker_ids']:
        mask = np.asarray(point_marker_ids, dtype=np.int64) == int(mid)
        marker_residuals = residuals[mask]
        diag['per_marker'][int(mid)] = {
            'rms_mm': float(np.sqrt(np.mean(marker_residuals ** 2))),
            'max_abs_mm': float(np.max(np.abs(marker_residuals))),
            'signed_residuals_mm': [float(x) for x in marker_residuals],
        }
    diag['available'] = True
    diag['reason'] = 'ok'
    return n, c, diag

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

def configure_world_marker_sift_rt_refine():
    """Configure the imported pose module without modifying the shared Algorithm file.

    The Debug wrapper also logs seed -> marker-only -> final RT differences.
    With refinement disabled, SIFT statistics are restored after marker-only
    optimization for validation, but cannot feed back into R/t.
    """
    enabled = bool(ENABLE_WORLD_MARKER_SIFT_RT_REFINE)
    optimizer_name = '_unified_optimize_endpoint_world_poses'
    original_name = '_grad_sift_debug_original_world_optimizer'
    original_optimizer = getattr(video_pose_algo, original_name, None)
    if original_optimizer is None:
        original_optimizer = getattr(video_pose_algo, optimizer_name)
        setattr(video_pose_algo, original_name, original_optimizer)

    # Keep the legacy/fallback feature-refine path consistent with the World
    # endpoint optimizer controlled below.
    if hasattr(video_pose_algo, 'ENABLE_FEATURE_RT_REFINE'):
        video_pose_algo.ENABLE_FEATURE_RT_REFINE = enabled

    def audited_world_optimizer(*args, **kwargs):
        # Optional arguments begin after marker_size_mm in the shared optimizer.
        def input_value(position, name):
            return args[position] if len(args) > position else kwargs.get(name)

        feature_points_B = input_value(11, 'feature_points_B')
        feature_points_A = input_value(12, 'feature_points_A')
        feature_mask = input_value(13, 'feature_mask')
        feature_K = input_value(9, 'feature_K')
        R_A0 = np.asarray(input_value(0, 'R_A0'), dtype=np.float64).reshape(3, 3)
        t_A0 = np.asarray(input_value(1, 't_A0'), dtype=np.float64).reshape(3, 1)
        R_B0 = np.asarray(input_value(2, 'R_B0'), dtype=np.float64).reshape(3, 3)
        t_B0 = np.asarray(input_value(3, 't_B0'), dtype=np.float64).reshape(3, 1)
        corners_A = input_value(4, 'corners_A')
        corners_B = input_value(5, 'corners_B')
        marker_map = input_value(6, 'marker_map')
        camera_matrix = input_value(7, 'camera_matrix')
        distortion = input_value(8, 'distortion')
        marker_size_mm = input_value(10, 'marker_size_mm')

        def run_marker_only():
            marker_args = list(args)
            marker_kwargs = dict(kwargs)
            for position, name in (
                    (11, 'feature_points_B'),
                    (12, 'feature_points_A'),
                    (13, 'feature_mask')):
                if len(marker_args) > position:
                    marker_args[position] = None
                else:
                    marker_kwargs[name] = None
            return original_optimizer(*marker_args, **marker_kwargs)

        def feature_stats_for_rt(R_rel_value, t_rel_value):
            """Evaluate the same optimizer-input SIFT pairs against one RT."""
            try:
                pts_B = np.asarray(
                    feature_points_B, dtype=np.float64).reshape(-1, 2)
                pts_A = np.asarray(
                    feature_points_A, dtype=np.float64).reshape(-1, 2)
                if len(pts_B) != len(pts_A) or len(pts_B) == 0:
                    return None
                if feature_mask is not None:
                    mask = np.asarray(feature_mask, dtype=bool).reshape(-1)
                    if len(mask) == len(pts_B):
                        pts_B = pts_B[mask]
                        pts_A = pts_A[mask]
                if len(pts_B) == 0:
                    return None
                residuals = np.abs(video_pose_algo._unified_signed_sampson(
                    pts_B, pts_A, R_rel_value, t_rel_value, feature_K))
                residuals = residuals[np.isfinite(residuals)]
                if len(residuals) == 0:
                    return None
                inlier_threshold = float(
                    video_pose_algo.FEATURE_FINAL_INLIER_PX)
                inlier_values = residuals[residuals <= inlier_threshold]
                return {
                    'pair_count': int(len(residuals)),
                    'inlier_count': int(len(inlier_values)),
                    'inlier_threshold_px': inlier_threshold,
                    'inlier_median_px': float(np.median(inlier_values))
                        if len(inlier_values) else float('inf'),
                    'inlier_p90_px': float(np.percentile(inlier_values, 90))
                        if len(inlier_values) else float('inf'),
                    'all_median_px': float(np.median(residuals)),
                    'all_p90_px': float(np.percentile(residuals, 90)),
                }
            except (TypeError, ValueError, KeyError, np.linalg.LinAlgError):
                return None

        if enabled:
            result = original_optimizer(*args, **kwargs)
            if result is None:
                log_and_print("⚠️ [World RT Audit] 聯合最佳化未產生有效結果")
                return None
            # The shared optimizer does not expose its internal marker-only pose
            # when the joint candidate wins. Re-run only that small final audit
            # solve so the Debug log can compare both actual candidates.
            marker_result = (
                result if result.get('role') == 'marker_only_world'
                else run_marker_only())
        else:
            marker_result = run_marker_only()
            result = marker_result
        if result is None:
            log_and_print("⚠️ [World RT Audit] Marker-only 最佳化未產生有效結果")
            return None
        result = dict(result)

        if not enabled:
            result['applied_feature'] = False
            result['uses_feature'] = False

            # Compute Sampson statistics strictly after marker-only optimization.
            feature_stats = feature_stats_for_rt(
                result['R_rel'], result['t_rel'])

            result['feature'] = feature_stats
            result['feature_ok'] = bool(
                feature_stats is not None
                and feature_stats['inlier_count']
                >= int(video_pose_algo.FEATURE_MIN_MATCHES)
                and feature_stats['inlier_p90_px']
                <= float(video_pose_algo.FEATURE_FINAL_P90_MAX_PX))
            result['role'] = 'marker_only_world'

        def relative_pose(R_A, t_A, R_B, t_B):
            R_rel = np.asarray(R_A, dtype=np.float64).reshape(3, 3) @ np.asarray(
                R_B, dtype=np.float64).reshape(3, 3).T
            t_rel = (np.asarray(t_A, dtype=np.float64).reshape(3, 1)
                     - R_rel @ np.asarray(t_B, dtype=np.float64).reshape(3, 1))
            return R_rel, t_rel

        def rotation_delta_deg(R_from, R_to):
            delta = np.asarray(R_to) @ np.asarray(R_from).T
            cos_angle = np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))

        def marker_stats(R_A, t_A, R_B, t_B):
            try:
                return video_pose_algo._unified_pair_reprojection_stats(
                    R_A, t_A, corners_A, R_B, t_B, corners_B, marker_map,
                    camera_matrix, distortion, marker_size_mm)
            except (TypeError, ValueError, KeyError, np.linalg.LinAlgError, cv2.error):
                return None

        def format_rt(label, R_rel, t_rel, stats):
            R_text = np.array2string(
                np.asarray(R_rel).reshape(-1), precision=6,
                separator=',', suppress_small=True)
            t_vec = np.asarray(t_rel, dtype=np.float64).reshape(3)
            t_text = np.array2string(
                t_vec, precision=4, separator=',', suppress_small=True)
            reproj_text = "N/A" if stats is None else (
                f"RMS={stats['rms_px']:.4f}px, max={stats['max_px']:.4f}px")
            log_and_print(
                f"   {label}: R={R_text} | t={t_text} mm | "
                f"baseline={np.linalg.norm(t_vec):.4f} mm | marker {reproj_text}")

        def format_delta(label, R_from, t_from, R_to, t_to):
            delta_t = (np.asarray(t_to, dtype=np.float64).reshape(3)
                       - np.asarray(t_from, dtype=np.float64).reshape(3))
            log_and_print(
                f"   {label}: ΔR={rotation_delta_deg(R_from, R_to):.6f} deg | "
                f"Δt={np.linalg.norm(delta_t):.6f} mm | "
                f"Δt_xyz={np.array2string(delta_t, precision=6, separator=',')} | "
                f"Δbaseline={np.linalg.norm(t_to) - np.linalg.norm(t_from):+.6f} mm")

        def format_feature_transition(stats_from, stats_to):
            if stats_from is None or stats_to is None:
                log_and_print(
                    "   SIFT epipolar MARKER-ONLY→FINAL: N/A "
                    "(no common optimizer-input feature pairs)")
                return
            pair_count = min(
                int(stats_from['pair_count']), int(stats_to['pair_count']))
            threshold = float(stats_to['inlier_threshold_px'])
            log_and_print(
                "   SIFT epipolar MARKER-ONLY→FINAL "
                f"(same {pair_count} pairs, inlier≤{threshold:.3f}px):")
            log_and_print(
                f"      inliers {stats_from['inlier_count']}/{pair_count}→"
                f"{stats_to['inlier_count']}/{pair_count} | "
                f"inlier median {stats_from['inlier_median_px']:.4f}→"
                f"{stats_to['inlier_median_px']:.4f}px | "
                f"P90 {stats_from['inlier_p90_px']:.4f}→"
                f"{stats_to['inlier_p90_px']:.4f}px")
            log_and_print(
                f"      all-pair median {stats_from['all_median_px']:.4f}→"
                f"{stats_to['all_median_px']:.4f}px | "
                f"P90 {stats_from['all_p90_px']:.4f}→"
                f"{stats_to['all_p90_px']:.4f}px")

        def format_compact_transition(
                label, R_from, t_from, stats_from, R_to, t_to, stats_to):
            delta_t = (np.asarray(t_to, dtype=np.float64).reshape(3)
                       - np.asarray(t_from, dtype=np.float64).reshape(3))
            log_and_print(
                f"   {label}: ΔR={rotation_delta_deg(R_from, R_to):.6f}° | "
                f"Δt={np.linalg.norm(delta_t):.6f}mm "
                f"xyz={np.array2string(delta_t, precision=5, separator=',')} | "
                f"baseline {np.linalg.norm(t_from):.4f}→{np.linalg.norm(t_to):.4f}mm")
            if stats_from is None or stats_to is None:
                log_and_print("      marker reprojection: N/A")
            else:
                log_and_print(
                    f"      marker RMS {stats_from['rms_px']:.4f}→"
                    f"{stats_to['rms_px']:.4f}px | max "
                    f"{stats_from['max_px']:.4f}→{stats_to['max_px']:.4f}px")

        R_seed, t_seed = relative_pose(R_A0, t_A0, R_B0, t_B0)
        seed_stats = marker_stats(R_A0, t_A0, R_B0, t_B0)
        marker_result = dict(marker_result) if marker_result is not None else result
        R_marker = np.asarray(marker_result['R_rel'], dtype=np.float64).reshape(3, 3)
        t_marker = np.asarray(marker_result['t_rel'], dtype=np.float64).reshape(3, 1)
        marker_result_stats = marker_stats(
            marker_result['R_A'], marker_result['t_A'],
            marker_result['R_B'], marker_result['t_B'])
        R_final = np.asarray(result['R_rel'], dtype=np.float64).reshape(3, 3)
        t_final = np.asarray(result['t_rel'], dtype=np.float64).reshape(3, 1)
        final_stats = marker_stats(
            result['R_A'], result['t_A'], result['R_B'], result['t_B'])
        marker_feature_stats = feature_stats_for_rt(R_marker, t_marker)
        final_audit_feature_stats = feature_stats_for_rt(R_final, t_final)
        mapped_A = sorted(set(int(x) for x in (corners_A or {})) & set(marker_map or {}))
        mapped_B = sorted(set(int(x) for x in (corners_B or {})) & set(marker_map or {}))
        audit_separator = "=" * 96
        log_and_print(audit_separator)
        log_and_print("📐 [WORLD RT AUDIT：最佳化前／Marker-only／最終 RT 比較]")
        log_and_print(
            f"📐 [RT Audit] ids A/B={mapped_A}/{mapped_B} | "
            f"corners={len(mapped_A) * 4}/{len(mapped_B) * 4} | "
            f"SIFT={'ON' if enabled else 'OFF'} | role={result.get('role')}")
        if WORLD_RT_AUDIT_VERBOSE:
            format_rt("PRE seed (ArUco/temporal; not guaranteed raw solvePnP)",
                      R_seed, t_seed, seed_stats)
            format_rt("MARKER-ONLY", R_marker, t_marker, marker_result_stats)
            format_delta("PRE→MARKER", R_seed, t_seed, R_marker, t_marker)
            format_rt("FINAL", R_final, t_final, final_stats)
            format_delta("MARKER→FINAL", R_marker, t_marker, R_final, t_final)
            format_feature_transition(
                marker_feature_stats, final_audit_feature_stats)
        else:
            final_is_marker_only = bool(
                result.get('role') == 'marker_only_world'
                or (rotation_delta_deg(R_marker, R_final) < 1e-10
                    and np.linalg.norm(t_final - t_marker) < 1e-10))
            if final_is_marker_only:
                format_compact_transition(
                    "PRE→FINAL (Marker-only)",
                    R_seed, t_seed, seed_stats,
                    R_final, t_final, final_stats)
            else:
                format_compact_transition(
                    "PRE→MARKER", R_seed, t_seed, seed_stats,
                    R_marker, t_marker, marker_result_stats)
                format_compact_transition(
                    "MARKER→FINAL (SIFT)",
                    R_marker, t_marker, marker_result_stats,
                    R_final, t_final, final_stats)
            format_feature_transition(
                marker_feature_stats, final_audit_feature_stats)
            final_rvec = cv2.Rodrigues(R_final)[0].reshape(3)
            log_and_print(
                "   FINAL RT: rvec(rad)="
                + np.array2string(final_rvec, precision=6, separator=',')
                + " | t(mm)="
                + np.array2string(t_final.reshape(3), precision=5, separator=','))
        log_and_print(audit_separator)
        return result

    setattr(video_pose_algo, optimizer_name, audited_world_optimizer)
    log_and_print(
        "🔧 [World Marker+SIFT RT] enabled=" + str(enabled) + " | "
        + ("SIFT residuals may refine final R/t after passing gates"
           if enabled else
           "final R/t uses marker-only world optimization; SIFT is validation_only"))


def analyze_video_frames(video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm, select_mode="average", range_mode="fixed", progress_callback=None):
    video_pose_algo.log_and_print = log_and_print
    configure_world_marker_sift_rt_refine()
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
    video_pose_algo.FEATURE_IMAGE_SCALE = float(RT_SIFT_IMAGE_SCALE)
    feature_roi_ratio = (
        RT_SIFT_ROI_RATIO if bool(ENABLE_RT_SIFT_ROI) else None)
    selection_mode = str(FRAME_PAIR_SELECTION_MODE).strip().lower()
    if selection_mode not in ("original", "angle_guided"):
        raise ValueError(
            "FRAME_PAIR_SELECTION_MODE must be 'original' or 'angle_guided'")
    angle_guided_config = {
        'enabled': selection_mode == "angle_guided",
        'direction_mode': str(ANGLE_GUIDED_DIRECTION_MODE),
        'normalize_output_roles': True,
        'target_frame_A_deg': float(ANGLE_GUIDED_RIGHT_TARGET_DEG),
        'target_frame_B_deg': float(ANGLE_GUIDED_LEFT_TARGET_DEG),
        'target_tolerance_deg': float(ANGLE_GUIDED_TARGET_TOLERANCE_DEG),
        'coarse_samples_per_segment': int(
            ANGLE_GUIDED_COARSE_SAMPLES_PER_SEGMENT),
        'max_scan_frames_per_segment': int(
            ANGLE_GUIDED_MAX_SCAN_FRAMES_PER_SEGMENT),
        'candidates_per_side': int(ANGLE_GUIDED_CANDIDATES_PER_SIDE),
        'pair_score_weight': float(ANGLE_GUIDED_PAIR_SCORE_WEIGHT),
    }
    log_and_print(
        f"🧭 [Frame pair mode] {selection_mode} | "
        f"direction={ANGLE_GUIDED_DIRECTION_MODE} | "
        f"right/A={ANGLE_GUIDED_RIGHT_TARGET_DEG:.1f}deg, "
        f"left/B={ANGLE_GUIDED_LEFT_TARGET_DEG:.1f}deg")
    log_and_print(
        f"🎯 [RT SIFT ROI config] enabled={bool(ENABLE_RT_SIFT_ROI)} | "
        f"ratio={feature_roi_ratio} | scale={float(RT_SIFT_IMAGE_SCALE):.3f} | "
        "ArUco=full-frame | pair-geometry=full-frame")
    return video_pose_algo.analyze_video_frames(
        video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm,
        select_mode, range_mode, progress_callback=progress_callback,
        detection_roi_ratio=None,
        feature_roi_ratio=feature_roi_ratio,
        pair_geometry_roi_ratio=None,
        angle_guided_config=angle_guided_config,
        allowed_marker_ids=ARUCO_ALLOWED_PATTERN_IDS
    )

_clahe_cache = {}

def get_clahe(clip_limit, tile_size):
    return camera_algo.get_clahe(clip_limit, tile_size)

def preprocess_gray(gray_img, enable_clahe=True):
    return camera_algo.preprocess_gray(gray_img, enable_clahe, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    return aruco_algo.compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=log_and_print,
                                          allowed_marker_ids=ARUCO_ALLOWED_PATTERN_IDS)


def compute_marker_pose_plane(valid_poses, frame_idx, reference_normal=None):
    """Return the anchor-marker z=0 plane in the selected camera frame.

    ``valid_poses[frame_idx]`` follows the temporal analyzer convention
    T_camera<-anchor_marker.  The marker center is therefore ``t`` and its
    plane normal is the third column of ``R``.  Align the sign with the legacy
    plane when available so switching the display mode cannot invert the wound
    height sign.
    """
    if not isinstance(valid_poses, dict):
        return None, None
    pose = valid_poses.get(int(frame_idx))
    if pose is None or len(pose) != 2:
        return None, None
    try:
        R = np.asarray(pose[0], dtype=np.float64).reshape(3, 3)
        c = np.asarray(pose[1], dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None, None
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(c)):
        return None, None
    n = R[:, 2].copy()
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return None, None
    n /= n_norm
    if reference_normal is not None:
        ref_n = np.asarray(reference_normal, dtype=np.float64).reshape(3)
        ref_norm = float(np.linalg.norm(ref_n))
        if ref_norm > 1e-12 and float(np.dot(n, ref_n / ref_norm)) < 0.0:
            n = -n
    return n, c

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
    
    cA, idsA = filter_marker_detections(cA, idsA, ARUCO_ALLOWED_PATTERN_IDS)
    cB, idsB = filter_marker_detections(cB, idsB, ARUCO_ALLOWED_PATTERN_IDS)
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


def draw_high_contrast_preview_text(image, text, origin, font_scale=1.0):
    """Draw blue preview text with a thick white outline for readability."""
    cv2.putText(image, str(text), tuple(origin), cv2.FONT_HERSHEY_SIMPLEX,
                float(font_scale), (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(image, str(text), tuple(origin), cv2.FONT_HERSHEY_SIMPLEX,
                float(font_scale), (255, 70, 0), 2, cv2.LINE_AA)


def draw_rt_sift_roi_preview(image):
    """Draw the configured RT-SIFT-only ROI on the unrecorded live preview."""
    if not bool(ENABLE_RT_SIFT_ROI):
        return
    height, width = image.shape[:2]
    configured = RT_SIFT_ROI_RATIO
    if isinstance(configured, dict):
        entries = [
            ("A", configured.get("frame_A"), (255, 0, 255)),
            ("B", configured.get("frame_B"), (0, 200, 255)),
        ]
    else:
        entries = [("", configured, (255, 0, 255))]
    for role, ratio, color in entries:
        if ratio is None:
            raise ValueError("RT_SIFT_ROI_RATIO dict requires frame_A and frame_B")
        if len(ratio) == 2:
            x0, y0, x1, y1 = camera_algo.centered_roi_bounds(
                width, height, ratio[0], ratio[1])
        elif len(ratio) == 4:
            x0, y0, x1, y1 = camera_algo.normalized_roi_bounds(
                width, height, ratio[0], ratio[1], ratio[2], ratio[3])
        else:
            raise ValueError(
                "RT_SIFT_ROI_RATIO must contain width/height or x/y/width/height")
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 4, cv2.LINE_AA)
        scaled_w = max(1, int(round((x1 - x0) * float(RT_SIFT_IMAGE_SCALE))))
        scaled_h = max(1, int(round((y1 - y0) * float(RT_SIFT_IMAGE_SCALE))))
        role_text = f" {role}" if role else ""
        draw_high_contrast_preview_text(
            image,
            f"RT SIFT ROI{role_text} | scale {float(RT_SIFT_IMAGE_SCALE):.2f} "
            f"| crop {scaled_w}x{scaled_h}",
            (x0 + 10, min(height - 12, max(y0 + 30, y1 - 16))),
            font_scale=0.68)


def estimate_aruco_pattern_distances(
        frame_bgr, camera_matrix, distortion, marker_size_mm,
        detector=None, calibration_image_size=None):
    """Estimate camera-centre to ArUco-centre distances in a raw camera frame.

    The returned distance is ``||tvec||`` (not only optical-axis Z). Raw image
    corners are paired with the original calibrated K/distortion model. When
    the live stream resolution differs from the calibration resolution, K is
    scaled to the live frame before PnP.
    """
    if frame_bgr is None or frame_bgr.size == 0 or camera_matrix is None:
        return {}
    marker_size_mm = float(marker_size_mm)
    if not np.isfinite(marker_size_mm) or marker_size_mm <= 0.0:
        return {}

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    dist = (np.zeros((5, 1), dtype=np.float64) if distortion is None else
            np.asarray(distortion, dtype=np.float64).reshape(-1, 1))
    frame_h, frame_w = frame_bgr.shape[:2]
    if calibration_image_size is not None:
        calib_w, calib_h = map(float, calibration_image_size)
        if calib_w > 0.0 and calib_h > 0.0:
            sx = float(frame_w) / calib_w
            sy = float(frame_h) / calib_h
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if detector is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(
                dictionary, cv2.aruco.DetectorParameters())
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params)
    corners, ids = filter_marker_detections(corners, ids, ARUCO_ALLOWED_PATTERN_IDS)
    if ids is None or len(ids) == 0:
        return {}

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    for corner in corners:
        try:
            cv2.cornerSubPix(gray, corner, (4, 4), (-1, -1), term)
        except cv2.error:
            # A marker very close to the image boundary can lack a complete
            # refinement window; its detector coordinates remain usable.
            pass

    half = marker_size_mm * 0.5
    object_points = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)
    estimates = {}
    for marker_id_raw, corner in zip(ids.reshape(-1), corners):
        image_points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
        pose_candidates = []
        try:
            solved = cv2.solvePnPGeneric(
                object_points, image_points, K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if solved and bool(solved[0]):
                for rvec, tvec in zip(solved[1], solved[2]):
                    pose_candidates.append((
                        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                        np.asarray(tvec, dtype=np.float64).reshape(3, 1)))
        except cv2.error:
            pose_candidates = []
        if not pose_candidates:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, K, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    pose_candidates.append((rvec.reshape(3, 1), tvec.reshape(3, 1)))
            except cv2.error:
                continue

        scored = []
        for rvec, tvec in pose_candidates:
            if not np.all(np.isfinite(tvec)) or float(tvec[2, 0]) <= 0.0:
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
            projected = projected.reshape(4, 2)
            rms = float(np.sqrt(np.mean(np.sum(
                (projected - image_points) ** 2, axis=1))))
            scored.append((rms, rvec, tvec))
        if not scored:
            continue
        rms, rvec, tvec = min(scored, key=lambda item: item[0])
        R_marker_to_camera, _ = cv2.Rodrigues(rvec)
        marker_normal_camera = R_marker_to_camera[:, 2]
        marker_to_camera = -tvec.reshape(3)
        marker_to_camera /= max(float(np.linalg.norm(marker_to_camera)), 1e-12)
        # abs() makes the result independent of which side of the mathematical
        # marker normal is selected: 0 degrees means a fronto-parallel view.
        cos_view_angle = float(np.clip(abs(np.dot(
            marker_normal_camera, marker_to_camera)), 0.0, 1.0))
        view_angle_deg = float(np.degrees(np.arccos(cos_view_angle)))
        estimates[int(marker_id_raw)] = {
            "distance_mm": float(np.linalg.norm(tvec)),
            "z_mm": float(tvec[2, 0]),
            "view_angle_deg": view_angle_deg,
            "reprojection_rms_px": rms,
            "rvec": rvec,
            "tvec": tvec,
            "corners": image_points,
        }
    return estimates


def record_video_from_camera(camera_matrix=None, distortion=None,
                             marker_size_mm=ACTUAL_MARKER_SIZE_MM):
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
    camera_algo.log_camera_stream_settings(cap)
    print('🎞️ 錄影格式: FFV1 / AVI 無損（保留 cap.read() 的 BGR 像素；檔案較大）')
    print("操作說明：")
    print("  按下 's' 鍵 - 開始/停止錄影")
    print("  按下 'q' 鍵 - 當錄影完成後，結束預覽並載入影片")

    is_recording = False
    video_writer = None
    video_name = None
    has_recorded = False
    recorded_frames = 0
    recording_size = None

    # The preview estimates distance only; overlays are not written to video.
    preview_detector = None
    if camera_matrix is not None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        if hasattr(cv2.aruco, "ArucoDetector"):
            preview_detector = cv2.aruco.ArucoDetector(
                dictionary, cv2.aruco.DetectorParameters())
    last_distance_update = 0.0
    distance_estimates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法接收畫面，錄影中斷...")
            break

        # 錄影寫入
        if is_recording and video_writer is not None:
            if (frame.shape[1], frame.shape[0]) != recording_size:
                print('❌ 相機畫面尺寸改變，停止錄影以避免遺失或裁切影格')
                break
            video_writer.write(frame)
            recorded_frames += 1

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        draw_rt_sift_roi_preview(display_frame)

        # ArUco detection is throttled so the recording preview remains fluid.
        now_monotonic = time.monotonic()
        if now_monotonic - last_distance_update >= 0.12:
            distance_estimates = estimate_aruco_pattern_distances(
                frame, camera_matrix, distortion, marker_size_mm,
                detector=preview_detector,
                calibration_image_size=(CAMERA_WIDTH, CAMERA_HEIGHT))
            last_distance_update = now_monotonic

        if distance_estimates:
            distances = [v["distance_mm"] for v in distance_estimates.values()]
            view_angles = [v["view_angle_deg"] for v in distance_estimates.values()]
            median_distance = float(np.median(distances))
            median_view_angle = float(np.median(view_angles))
            distance_text = (
                f"Pattern distance: {median_distance:.1f} mm "
                f"({median_distance / 10.0:.1f} cm) | "
                f"Angle: {median_view_angle:.1f} deg")
            draw_high_contrast_preview_text(
                display_frame, distance_text, (30, 103), font_scale=1.02)
            for row, (marker_id, estimate) in enumerate(
                    sorted(distance_estimates.items())):
                pts = np.rint(estimate["corners"]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2,
                              cv2.LINE_AA)
                origin = tuple(pts[0, 0].tolist())
                draw_high_contrast_preview_text(
                    display_frame,
                    f"ID {marker_id}: {estimate['distance_mm']:.1f} mm  "
                    f"{estimate['view_angle_deg']:.1f} deg",
                    (origin[0], max(24, origin[1] - 10)),
                    font_scale=0.68)
                if row < 3:
                    draw_high_contrast_preview_text(
                        display_frame,
                        f"ID {marker_id}: {estimate['distance_mm']:.1f} mm  "
                        f"Angle {estimate['view_angle_deg']:.1f} deg  "
                        f"RMS {estimate['reprojection_rms_px']:.2f}px",
                        (30, 143 + row * 38), font_scale=0.68)
        else:
            status = ("Pattern distance: calibration unavailable" if
                      camera_matrix is None else
                      "Pattern distance: ArUco not detected")
            draw_high_contrast_preview_text(
                display_frame, status, (30, 103), font_scale=0.86)

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
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                next_video_name = os.path.join(save_path, f"video_{now_str}_lossless.avi")
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0 or fps > 100: fps = 25.0  # 保底 FPS
                try:
                    video_writer = open_lossless_writer(next_video_name, fps, (w, h))
                except (ValueError, RuntimeError, cv2.error) as exc:
                    print(f'❌ 無損錄影未開始: {exc}')
                    continue
                video_name = next_video_name
                recording_size = (w, h)
                recorded_frames = 0
                is_recording = True
                has_recorded = False
                print(f"🎬 開始無損錄影：{video_name} | FFV1 | {w}x{h} | {fps:g} fps")
            else:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                has_recorded = recorded_frames > 0
                print(f"🛑 無損錄影結束，共 {recorded_frames} frames")
        
        # 縮放預覽，避免影像太大
        display_small = cv2.resize(display_frame, (int(w//2), int(h//2)))
        cv2.imshow('Camera Recording Window', display_small)

    if video_writer is not None:
        video_writer.release()  # Finalize the AVI even if the camera disconnects.
        has_recorded = recorded_frames > 0
    cap.release()
    cv2.destroyAllWindows()
    return video_name if has_recorded else None

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
    height_source = res.get('height_reference_source')
    height_signed = res.get('height_reference_signed_mm')
    height_display = res.get('height_display_mm')
    text_lines.append(f"目前高度參考平面: {height_source or 'None'}")
    text_lines.append(
        f"目前高度參考平面有號距離: {height_signed:.3f} mm"
        if height_signed is not None else "目前高度參考平面有號距離: None")
    text_lines.append(
        f"目前畫面高度值: {height_display:.3f} mm"
        if height_display is not None else "目前畫面高度值: None")
    shared_plane_diag = res.get('shared_pattern_plane_diag')
    if isinstance(shared_plane_diag, dict):
        text_lines.append(
            "共同 Pattern 角點平面: "
            f"available={shared_plane_diag.get('available', False)}, "
            f"markers={shared_plane_diag.get('used_marker_ids', [])}, "
            f"corners={shared_plane_diag.get('point_count', 0)}, "
            f"RMS={shared_plane_diag.get('rms_mm')} mm, "
            f"P90={shared_plane_diag.get('p90_abs_mm')} mm, "
            f"max={shared_plane_diag.get('max_abs_mm')} mm, "
            f"reason={shared_plane_diag.get('reason', '')}")
    text_lines.append(f"左圖影格索引: {idx_B}")
    text_lines.append(f"基準線 (Baseline): {baseline:.2f} mm" if baseline is not None else "基準線 (Baseline): None")
    text_lines.append(f"相對平移向量 (T_rel): {t_rel_str}")
    text_lines.append(f"相對旋轉矩陣 (R_rel):\n{R_rel_str}")
    text_lines.append(f"相機內參 (KL):\n{K_R_str}")
    block_audit = res.get('debug_metric_block_audit')
    block_log = (
        block_audit.get('log')
        if isinstance(block_audit, dict) else None)
    if isinstance(block_log, dict):
        text_lines.append("--------------------------------------------------")
        text_lines.append("5x5 mm Block Debug（唯讀，不影響匹配）:")
        text_lines.append(
            f"  grid={block_log.get('grid_source', 'N/A')}, "
            f"block={block_log.get('selected_block')}, "
            f"projected_px={block_log.get('projected_size_left_px')}")
        text_lines.append(
            f"  A-centered support={block_log.get('support_count', 0)}, "
            f"valid3D={block_log.get('valid_3d_count', 0)}, "
            f"H/M={block_log.get('high_count', 0)}/{block_log.get('mid_count', 0)}")
        text_lines.append(
            f"  median={block_log.get('median_height_mm')} mm, "
            f"MAD={block_log.get('mad_mm')} mm, "
            f"P90-P10={block_log.get('robust_span_mm')} mm, "
            f"tol={block_log.get('tolerance_mm')} mm")
        text_lines.append(
            f"  A_delta={block_log.get('a_delta_mm')} mm, "
            f"status={block_log.get('status', 'N/A')}")
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
        "height_reference_source": height_source,
        "height_reference_signed_mm": float(height_signed) if height_signed is not None else None,
        "height_display_mm": float(height_display) if height_display is not None else None,
        "idx_left_frame": int(idx_B) if idx_B is not None else None,
        "baseline_mm": float(baseline) if baseline is not None else None,
        "t_rel": t_rel.flatten().tolist() if t_rel is not None else None,
        "R_rel": R_rel.tolist() if R_rel is not None else None,
        "KL": K_R.tolist() if K_R is not None else None
    }
    if isinstance(shared_plane_diag, dict):
        json_data["shared_pattern_plane_diag"] = shared_plane_diag
    if isinstance(block_log, dict):
        json_data["debug_metric_block_audit"] = block_log
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
    text = str(reason or "No Valid Depth")
    replacements = (
        ("未偵測到 ArUco", "ArUco not detected"),
        ("視差不合規範", "Baseline out of range"),
        ("追蹤影格數不足", "Not enough tracked frames"),
        ("三角化失敗", "Triangulation failed"),
        ("無匹配點", "No Valid Depth"),
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

    # Load calibration before opening the live preview so the same calibrated
    # K/distortion model can be used for the on-screen Pattern distance.
    mtxL_o, distL, mtxR_o, distR, extrinsic, F_orig = \
        camera_algo.load_json_camera_params(PARAMS_JSON_PATH)
    if mtxL_o is None or distL is None:
        print(f"❌ 無法載入相機標定參數: {PARAMS_JSON_PATH}")
        sys.exit(1)
        
    global VIDEO_PATH
    if action == "camera":
        recorded_path = record_video_from_camera(
            mtxL_o, distL, ACTUAL_MARKER_SIZE_MM)
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

    # 1. The intrinsics loaded above are shared by live preview and analysis.
    
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
    _angle_diag = video_data.get('angle_guided_diagnostics', {}) or {}
    _angle_final = _angle_diag.get('final_selected_pair', {}) or {}
    _angle_pair = _angle_final.get('pair_measurement') or {}
    log_and_print(
        f"⏱️ [Frame pair result] mode={FRAME_PAIR_SELECTION_MODE} | "
        f"status={_angle_diag.get('status', 'N/A')} | "
        f"A/right=F{video_data.get('idx_A')} "
        f"{_angle_pair.get('incidence_A_deg', 'N/A')}deg | "
        f"B/left=F{video_data.get('idx_B')} "
        f"{_angle_pair.get('incidence_B_deg', 'N/A')}deg | "
        f"angle_scan={float(_angle_diag.get('elapsed_s', 0.0)):.3f}s | "
        f"RT_total={float(video_data.get('analysis_total_elapsed_s', 0.0)):.3f}s")
    startup_timer.stage("影片分析(ArUco配對+RT解算)")

    use_wound_adaptive_spatial_specular = True
    log_and_print(
        f'[Specular] Adaptive Spatial: valid wound mask when available; otherwise '
        f'full-image {SPATIAL_BLOCK_CONFIG.block_width_px}x{SPATIAL_BLOCK_CONFIG.block_height_px}px tiles '
        f'(AI enabled={ENABLE_WOUND_AI})')

    def compute_wound_adaptive_spatial_mask(bgr, wound_prediction=None):
        if not use_wound_adaptive_spatial_specular or bgr is None or bgr.size == 0:
            return None
        if wound_prediction is not None:
            wound_mask = prediction_to_wound_mask(wound_prediction, bgr.shape)
            if wound_mask is not None and np.count_nonzero(wound_mask) >= 20:
                return compute_specular_mask_bgr_wound_adaptive(bgr, wound_mask)
        return compute_specular_mask_bgr_block_adaptive(bgr, SPATIAL_BLOCK_CONFIG)

    def compute_locked_spec_mask(bgr, frame_idx=None, wound_prediction=None):
        combined_mask, spatial_mask, temporal_mask = compute_locked_spec_masks(bgr, frame_idx, wound_prediction)
        return combined_mask

    def compute_locked_spec_masks(bgr, frame_idx=None, wound_prediction=None):
        adaptive_spatial_mask = compute_wound_adaptive_spatial_mask(bgr, wound_prediction)
        combined_mask, spatial_mask, temporal_mask = compute_rt_aligned_temporal_specular_mask_bgr(
            bgr,
            frame_idx,
            video_data,
            KL,
            ACTUAL_MARKER_SIZE_MM,
            process_view,
            return_parts=True,
            preprocess_gray_fn=preprocess_gray,
            base_mask=adaptive_spatial_mask,
        )
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
    # Keep the historical triangulated/SVD plane untouched for matching and for
    # A/B comparison.  The alternate display-only plane comes directly from the
    # temporally selected anchor-marker pose in the left image (frame B).
    legacy_height_plane_n = None if global_plane_n is None else np.asarray(
        global_plane_n, dtype=np.float64).reshape(3).copy()
    legacy_height_plane_c = None if global_plane_c is None else np.asarray(
        global_plane_c, dtype=np.float64).reshape(3).copy()
    pose_height_plane_n, pose_height_plane_c = compute_marker_pose_plane(
        video_data.get('valid_poses', {}), video_data['idx_B'],
        reference_normal=legacy_height_plane_n)
    if pose_height_plane_n is not None:
        if legacy_height_plane_n is not None:
            _legacy_n_unit = legacy_height_plane_n / max(
                float(np.linalg.norm(legacy_height_plane_n)), 1e-12)
            _plane_angle_deg = float(np.degrees(np.arccos(np.clip(
                abs(float(np.dot(_legacy_n_unit, pose_height_plane_n))),
                -1.0, 1.0))))
            log_and_print(
                f"[Height Plane] Marker-pose plane ready; legacy normal delta "
                f"{_plane_angle_deg:.3f} deg")
        else:
            log_and_print("[Height Plane] Marker-pose plane ready; no legacy plane for comparison")
    else:
        log_and_print(
            "[Height Plane] Marker-pose plane unavailable; height display will remain in legacy mode")
    
    # 預設直接鎖定
    locked_L = imgA_bgr.copy()
    locked_R = imgB_bgr.copy()
    locked_L_clean = locked_L.copy()
    locked_R_clean = locked_R.copy()
    locked_L_idx = video_data['idx_B']
    locked_R_idx = video_data['idx_A']
    locked_L_spec_mask, locked_L_spec_spatial_mask, locked_L_spec_temporal_mask = compute_locked_spec_masks(locked_L_clean, locked_L_idx)
    locked_R_spec_mask, locked_R_spec_spatial_mask, locked_R_spec_temporal_mask = compute_locked_spec_masks(locked_R_clean, locked_R_idx)
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

    # Debug-only height reference: triangulate all four corners of every marker
    # present in both selected images, then fit one plane in left-camera 3D.
    # This is deliberately separate from current_cand['plane_n'/'plane_c'] so
    # enabling it cannot change RT, matching, homography or triangulation.
    (shared_height_plane_n,
     shared_height_plane_c,
     shared_height_plane_diag) = compute_shared_marker_corner_plane(
        current_cand.get('cornersA'), current_cand.get('cornersB'),
        KL, current_cand['K_R'], current_cand['R_rel'], current_cand['t_rel'],
        F=current_cand.get('F'), reference_normal=legacy_height_plane_n,
        min_shared_markers=2)
    if shared_height_plane_diag.get('available'):
        _shared_ids = shared_height_plane_diag['used_marker_ids']
        log_and_print(
            f"[Shared Pattern Plane] ready; "
            f"shared={shared_height_plane_diag['shared_marker_ids']}, "
            f"used={_shared_ids}, "
            f"corners={shared_height_plane_diag['point_count']}, "
            f"RMS={shared_height_plane_diag['rms_mm']:.3f} mm, "
            f"P90={shared_height_plane_diag['p90_abs_mm']:.3f} mm, "
            f"max={shared_height_plane_diag['max_abs_mm']:.3f} mm")
        for _mid in _shared_ids:
            _marker_diag = shared_height_plane_diag['per_marker'][_mid]
            log_and_print(
                f"[Shared Pattern Plane] marker {_mid}: "
                f"RMS={_marker_diag['rms_mm']:.3f} mm, "
                f"max={_marker_diag['max_abs_mm']:.3f} mm")
        for _mid, _reason in shared_height_plane_diag['skipped_markers'].items():
            log_and_print(
                f"[Shared Pattern Plane] skipped marker {_mid}: {_reason}")
        if shared_height_plane_diag['rms_mm'] > 2.0:
            log_and_print(
                "[Shared Pattern Plane] warning: plane residual RMS exceeds "
                "2.0 mm; inspect marker detection and stereo RT")
    else:
        log_and_print(
            "[Shared Pattern Plane] unavailable: "
            + shared_height_plane_diag.get('reason', 'unknown reason'))
    
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
        extra_candidates_list.append(extra_cand)
        
    candidates = [current_cand]
    startup_timer.stage(f"次佳影格處理({len(extra_candidates_list)} 個)")

    # 進 UI 前預先算好次佳影格高光遮罩 (Reject SpecPts 預設開啟)，
    # 期間以獨立讀取條視窗顯示「階段處理中」，避免第一次點擊卡住
    precompute_masks_with_progress_window(extra_candidates_list, compute_locked_spec_masks)
    startup_timer.stage("高光遮罩預計算")

    # Debug 版改為上下兩列：上方保留原本左右量測圖，
    # 下方顯示點擊區域的放大匹配診斷圖。
    fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1E1E1E')
    fig.canvas.manager.set_window_title("MeasureTool")
    try:
        fig.canvas.toolbar.pack_forget() # 隱藏底部的功能條
    except:
        pass
    fig.subplots_adjust(
        top=0.735, right=0.98, left=0.05, bottom=0.130,
        hspace=0.10, wspace=0.04)
    (ax_A, ax_B), (ax_debug_A, ax_debug_B) = axes_grid
    # Debug 說明移到影像外的獨立左右資訊列，避免長文字覆蓋 ROI，
    # 並保留中間間距防止兩欄互相重疊。
    ax_debug_info_A = fig.add_axes(
        [0.05, 0.012, 0.445, 0.105], facecolor='#101010')
    ax_debug_info_B = fig.add_axes(
        [0.535, 0.012, 0.445, 0.105], facecolor='#101010')
    for ax_info, border_color in (
            (ax_debug_info_A, '#8FD3FF'),
            (ax_debug_info_B, '#FFD29A')):
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(0.8)
    im_A = ax_A.imshow(cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB))
    im_B = ax_B.imshow(current_cand['rgb'])
    # Debug panels show the actual grayscale inputs used by the default
    # Grad-SIFT/ORB path, rather than the display-only RGB/overlay images.
    im_debug_A = ax_debug_A.imshow(cv2.cvtColor(imgA_gray, cv2.COLOR_GRAY2RGB))
    im_debug_B = ax_debug_B.imshow(
        cv2.cvtColor(current_cand['gray'], cv2.COLOR_GRAY2RGB))
    for ax in axes_grid.flat:
        ax.axis("off")
        ax.set_facecolor('#1E1E1E')
    ax_B.set_visible(False)
    ax_debug_B.set_visible(False)
    ax_debug_info_B.set_visible(False)
        
    # 加入專業感的影像外框
    from matplotlib.patches import Rectangle
    border_A = Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='#00FF00', lw=2, alpha=0.8) # 綠色代表 Live
    border_B = Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='#FFCC00', lw=2, alpha=0.8) # 黃色代表鎖定/參考
    ax_A.add_patch(border_A)
    ax_B.add_patch(border_B)
    
    # 設定標題為白色
    ax_A.set_title('Camera (Live)', color='white', fontsize=10, fontweight='bold', pad=5)
    ax_B.set_title('右圖 (Locked)', color='white', fontsize=10, fontweight='bold', pad=5)
    ax_debug_A.set_title(
        'Grad-SIFT/ORB Debug - Left ROI (wheel zoom; double-click reset)', color='#8FD3FF',
        fontsize=10, fontweight='bold', pad=5)
    ax_debug_B.set_title(
        'Grad-SIFT/ORB Debug - Right ROI (wheel zoom; double-click reset)', color='#FFD29A',
        fontsize=10, fontweight='bold', pad=5)

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
    scatter_A_reproj = ax_A.scatter(
        [], [], s=120, facecolors='none', edgecolors='#FF00FF', marker='o',
        linestyle='--', lw=1.5, zorder=6, visible=SHOW_LEFT_REPROJECTION_CIRCLE)
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
    epi_line, = ax_B.plot([], [], 'yellow', lw=1, alpha=0.6, zorder=4)
    sift_rect = Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='magenta', facecolor='none', linestyle='--', alpha=0.8, zorder=4)
    ax_B.add_patch(sift_rect)
    sift_rect.set_visible(False)
    sift_rect_center, = ax_B.plot([], [], '+', color='magenta', markersize=12, markeredgewidth=1.5, zorder=5)
    sift_rect_center.set_visible(False)

    # ---------------- Grad-SIFT/ORB 放大 Debug 圖層 ----------------
    # 淺色小點：全部候選參考點；實心大點：最後通過的匹配點。
    dbg_ref_high_A = ax_debug_A.scatter(
        [], [], s=9, c='#8FD3FF', alpha=0.45, zorder=3)
    dbg_ref_high_B = ax_debug_B.scatter(
        [], [], s=9, c='#8FD3FF', alpha=0.45, zorder=3)
    dbg_ref_mid_A = ax_debug_A.scatter(
        [], [], s=9, c='#FFD29A', alpha=0.45, zorder=3)
    dbg_ref_mid_B = ax_debug_B.scatter(
        [], [], s=9, c='#FFD29A', alpha=0.45, zorder=3)
    dbg_inlier_high_A = ax_debug_A.scatter(
        [], [], s=38, facecolors='none', edgecolors='#00BFFF',
        linewidths=1.3, zorder=6)
    dbg_inlier_high_B = ax_debug_B.scatter(
        [], [], s=38, facecolors='none', edgecolors='#00BFFF',
        linewidths=1.3, zorder=6)
    dbg_inlier_mid_A = ax_debug_A.scatter(
        [], [], s=38, facecolors='none', edgecolors='#FF8C00',
        linewidths=1.3, zorder=6)
    dbg_inlier_mid_B = ax_debug_B.scatter(
        [], [], s=38, facecolors='none', edgecolors='#FF8C00',
        linewidths=1.3, zorder=6)
    dbg_click_A = ax_debug_A.scatter(
        [], [], s=120, c='white', marker='x', linewidths=2.0, zorder=9)
    dbg_seed_B = ax_debug_B.scatter(
        [], [], s=130, c='#FF00FF', marker='+', linewidths=2.0, zorder=8)
    dbg_raw_B = ax_debug_B.scatter(
        [], [], s=90, facecolors='none', edgecolors='white', marker='D',
        linewidths=1.5, zorder=9)
    dbg_final_B = ax_debug_B.scatter(
        [], [], s=120, c='#00FF66', marker='x', linewidths=2.2, zorder=10)
    dbg_pred_B = ax_debug_B.scatter(
        [], [], s=28, facecolors='none', edgecolors='#FF00FF', marker='o',
        linewidths=0.9, alpha=0.8, zorder=5)
    # Descriptor audit selection.  These artists only explain the KNN result;
    # they never feed back into Grad-SIFT/ORB matching.
    dbg_audit_selected_A = ax_debug_A.scatter(
        [], [], s=150, facecolors='none', edgecolors='#FFFF00', marker='*',
        linewidths=1.8, zorder=11)
    dbg_audit_top1_B = ax_debug_B.scatter(
        [], [], s=120, facecolors='none', edgecolors='#39FF14', marker='o',
        linewidths=2.0, zorder=11)
    dbg_audit_top2_B = ax_debug_B.scatter(
        [], [], s=115, facecolors='none', edgecolors='#FF3333', marker='s',
        linewidths=1.8, zorder=11)
    dbg_audit_local_seed_B = ax_debug_B.scatter(
        [], [], s=90, c='#FF00FF', marker='x', linewidths=1.5, zorder=10)
    dbg_epi_line, = ax_debug_B.plot(
        [], [], color='#FFFF00', lw=1.0, alpha=0.75, zorder=4)
    dbg_hull_line, = ax_debug_A.plot(
        [], [], color='#00FF66', lw=1.4, linestyle='--', alpha=0.9, zorder=5)
    dbg_hull_line_B, = ax_debug_B.plot(
        [], [], color='#00FF66', lw=1.4, linestyle='--', alpha=0.9, zorder=5)
    dbg_search_rect = Rectangle(
        (0, 0), 0, 0, linewidth=1.2, edgecolor='#FF00FF',
        facecolor='none', linestyle='--', alpha=0.8, zorder=4)
    ax_debug_B.add_patch(dbg_search_rect)
    dbg_search_rect.set_visible(False)
    dbg_info_A = ax_debug_info_A.text(
        0.012, 0.95, "Click the upper-left image to inspect matching",
        transform=ax_debug_info_A.transAxes, color='white', fontsize=7,
        va='top', ha='left', zorder=12, clip_on=True)
    dbg_info_B = ax_debug_info_B.text(
        0.012, 0.95,
        "purple +=RT/plane seed | white diamond=raw | green x=final",
        transform=ax_debug_info_B.transAxes, color='white', fontsize=7,
        va='top', ha='left', zorder=12, clip_on=True)
    # 每次點擊都會重建的跨圖連線、Homography residual 線與編號標籤。
    debug_pair_artists = []
    # Subset of debug_pair_artists controlled by the H-residual visibility
    # button.  The magenta prediction circles are the persistent dbg_pred_B.
    debug_homography_residual_artists = []
    # 5x5 mm physical-grid polygons/text are kept separate so the overlay can
    # be toggled without disturbing descriptor-audit or correspondence artists.
    debug_metric_block_artists = []
    debug_metric_block_state = {
        'last_audit': None,
        'last_message': '5x5 mm block audit has not run',
    }
    # Descriptor-audit selection lines are managed separately so clicking a
    # raw reference can replace only the Top-1/Top-2 explanation.
    debug_audit_artists = []
    debug_audit_state = {
        'records': [], 'selected_index': None,
        'base_left_text': '', 'base_right_text': '',
        'descriptor_name': 'N/A', 'audit': None,
    }
    # Region-SIFT uses the same lower axes but keeps its dynamic 3x3 grid,
    # rotated search band and numbered point annotations separate.
    debug_region_artists = []
    region_diagnostics = RegionSIFTDiagnostics()
    region_diagnostic_latest = {'result': None}
    height_profile = None
    profile_disabled_widgets = []
    debug_region_state = {
        'records': [], 'selected_index': None,
        'base_left_text': '', 'base_right_text': '',
        'support_artists': [],
    }
    # Home limits are refreshed for every measurement.  Debug zoom/reset only
    # changes the axes view and never feeds coordinates back into matching.
    debug_zoom_home = {'A': None, 'B': None}

    # HUD 風格的文字面板
    depth_text = fig.text(0.53, 0.50, "", transform=fig.transFigure,
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
        
    # 右側控制區少一列，將姿態狀態放在該空列，避免覆蓋放大的影像。
    pose_status_text = fig.text(0.975, 0.785, pose_status_str, transform=fig.transFigure,
                                 color=pose_status_color, fontsize=8, fontweight='bold',
                                 ha='right', va='center',
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
        if view_state.get('show_rt_warp_view', False):
            print(
                f"⚠️ [RT Warp View] 跳過 Wound V1 深度/尺寸重算 ({reason})；"
                "純顯示模式不執行任何深度計算。")
            return
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
        # Extra right frames were precomputed under the previous spatial mode.
        # Invalidate them so matching lazily rebuilds consistent masks.
        for extra in extra_candidates_list:
            extra['spec_mask'] = None
            extra['spec_spatial_mask'] = None
            extra['spec_temporal_mask'] = None
        print(
            f"[Specular] {'Adaptive spatial (wound mask or full-image tiles)' if use_wound_adaptive_spatial_specular else 'Fixed spatial'} "
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
    # 頂端保留 0.965 以上給面板顯示切換小圓點；其餘控制項緊密
    # 排在 0.772~0.958，讓下方 2x2 影像區能向上延伸。
    control_row_y = (0.932, 0.905, 0.878, 0.851, 0.824, 0.797, 0.770)
    control_h = 0.026
    ax_c1 = fig.add_axes([0.05, control_row_y[0], 0.11, control_h], facecolor='#1E1E1E')
    ax_c2 = fig.add_axes([0.17, control_row_y[0], 0.11, control_h], facecolor='#1E1E1E')
    ax_c18 = fig.add_axes([0.29, control_row_y[0], 0.11, control_h], facecolor='#1E1E1E')
    ax_c3 = fig.add_axes([0.05, control_row_y[1], 0.11, control_h], facecolor='#1E1E1E')
    ax_c4 = fig.add_axes([0.17, control_row_y[1], 0.11, control_h], facecolor='#1E1E1E')
    ax_c5 = fig.add_axes([0.29, control_row_y[1], 0.11, control_h], facecolor='#1E1E1E')
    ax_c6 = fig.add_axes([0.05, control_row_y[2], 0.11, control_h], facecolor='#1E1E1E')
    ax_c7 = fig.add_axes([0.17, control_row_y[2], 0.11, control_h], facecolor='#1E1E1E')
    ax_c8 = fig.add_axes([0.29, control_row_y[2], 0.11, control_h], facecolor='#1E1E1E')
    ax_c9 = fig.add_axes([0.05, control_row_y[3], 0.11, control_h], facecolor='#1E1E1E')
    ax_c10 = fig.add_axes([0.17, control_row_y[3], 0.11, control_h], facecolor='#1E1E1E')
    ax_c11 = fig.add_axes([0.29, control_row_y[3], 0.11, control_h], facecolor='#1E1E1E')
    ax_c12 = fig.add_axes([0.05, control_row_y[4], 0.11, control_h], facecolor='#1E1E1E')
    ax_c13 = fig.add_axes([0.17, control_row_y[4], 0.11, control_h], facecolor='#1E1E1E')
    ax_c14 = fig.add_axes([0.29, control_row_y[4], 0.11, control_h], facecolor='#1E1E1E')
    ax_c15 = fig.add_axes([0.05, control_row_y[5], 0.11, control_h], facecolor='#1E1E1E')
    ax_c16 = fig.add_axes([0.17, control_row_y[5], 0.11, control_h], facecolor='#1E1E1E')
    ax_c17 = fig.add_axes([0.29, control_row_y[5], 0.11, control_h], facecolor='#1E1E1E')
    ax_c19 = fig.add_axes([0.05, control_row_y[6], 0.11, control_h], facecolor='#1E1E1E')
    
    # 建立標準按鈕，文字開頭加上 [X] 或 [ ] 代表勾選狀態
    btn_opt_style = dict(color='#1A1A1A', hovercolor='#333333')
    c1 = Button(ax_c1, "[ ] 嚴格精細匹配", **btn_opt_style)
    c2 = Button(ax_c2, "[ ] 梯度 SIFT 匹配", **btn_opt_style)
    c18 = Button(ax_c18, "[X] Region-SIFT 匹配", **btn_opt_style)
    c3 = Button(ax_c3, "[ ] 強制極線對齊", **btn_opt_style)
    c4 = Button(
        ax_c4,
        "[X] 啟用 ECC 精修" if ENABLE_ECC_REFINEMENT_DEFAULT else "[ ] 啟用 ECC 精修",
        **btn_opt_style)
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

    view_state = {'precise': False, 'grad_sift': False, 'region_sift': True,
                  'enforce_epi': False,
                  'ecc': ENABLE_ECC_REFINEMENT_DEFAULT, 'manual': False,
                  'use_hamming': False, 'enable_clahe': ENABLE_CLAHE_DEFAULT,
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
                  'show_metric_blocks': DEBUG_METRIC_BLOCKS_DEFAULT,
                  'show_homography_residual': True,
                  'top2_geometry_rescue': TOP2_GEOMETRY_RESCUE_DEFAULT,
                  'show_rt_warp_view': False,
                  'manual_pt_A': None, 'lines': [], 'grad_lines': [], 'show_grad_lines': False,
                  'highlighted_grad_line': None, 'highlighted_grad_line_artist': None,
                  'grad_data': None, 'restart': False}  # grad_data = {'ptsA': ndarray, 'ptsB': ndarray}
    # 純顯示用：保存「右圖經 RT + 基準平面 Homography 投回左圖座標」的 RGB 畫面。
    # 此狀態不會覆寫 locked_R/current_cand，也不會進入 matcher 或三角化流程。
    rt_warp_view_state = {
        'frame_rgb': None,
        'valid_ratio': None,
        'forced_ax_b_visible': False,
    }
    # Display-only selector.  RT, baseline, matching and triangulated p3d never
    # change when this state is toggled.
    height_plane_state = {
        'use_pose_plane': False,
        'use_shared_plane': True,
    }
    block_accuracy_state = {'mode': 'idle', 'corners': [], 'plan': None,
                            'report': None, 'timer': None, 'artists': [],
                            'disabled_widgets': [], 'started': None,
                            'summaries': None, 'show_mae_colors': False}

    def get_selected_height_plane():
        """Return (normal, center, UI label) for the display-only height plane."""
        if (height_plane_state['use_shared_plane']
                and shared_height_plane_n is not None
                and shared_height_plane_c is not None):
            return shared_height_plane_n, shared_height_plane_c, 'Shared Pattern Plane'
        if (height_plane_state['use_pose_plane']
                and pose_height_plane_n is not None
                and pose_height_plane_c is not None):
            return pose_height_plane_n, pose_height_plane_c, 'Marker Pose Plane'
        return legacy_height_plane_n, legacy_height_plane_c, 'Legacy Plane'

    # HighPts / MidPts 預設關閉：初始同步散點顯示狀態
    for _artist in (scatter_grad_ref_A, scatter_grad_ref_B, scatter_grad_inject, scatter_grad_match):
        _artist.set_visible(view_state['show_high_grad_points'])
    for _artist in (scatter_mid_grad_ref_A, scatter_mid_grad_ref_B, scatter_mid_grad_inject, scatter_mid_grad_match):
        _artist.set_visible(view_state['show_mid_grad_points'])

    # 建立測量模式單選框，置於中間空白處
    ax_mode = fig.add_axes([0.42, 0.836, 0.13, 0.12], facecolor='#1E1E1E')
    ax_mode.patch.set_edgecolor('white')
    ax_mode.patch.set_linewidth(1.0)
    radio_mode = RadioButtons(ax_mode, ('雙幀直接', '多幀去漂移', '多幀純光流'),
                              active=0 if MEASURE_MODE=="dual_direct" else (1 if MEASURE_MODE=="multi_dedrift" else 2),
                              activecolor='#00FFFF')
    
    # 調整單選框字型與色彩
    for label in radio_mode.labels:
        label.set_color('white')
        label.set_fontsize(7)
        
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
    for c in [c1, c2, c18, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c19]:
        c.label.set_color('white')
        c.label.set_fontsize(7)
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

    def on_grad_sift_mode_clicked(event):
        enabling = not view_state['grad_sift']
        view_state['grad_sift'] = enabling
        c2.label.set_text("[X] 梯度 SIFT 匹配" if enabling else "[ ] 梯度 SIFT 匹配")
        if enabling and view_state.get('region_sift', False):
            view_state['region_sift'] = False
            c18.label.set_text("[ ] Region-SIFT 匹配")
        request_blit_refresh()
        mark_wound_size_dirty('grad_sift')
        if last_click:
            do_measure(last_click[0], last_click[1])

    def on_region_sift_mode_clicked(event):
        enabling = not view_state.get('region_sift', False)
        view_state['region_sift'] = enabling
        c18.label.set_text("[X] Region-SIFT 匹配" if enabling else "[ ] Region-SIFT 匹配")
        if enabling and view_state.get('grad_sift', False):
            view_state['grad_sift'] = False
            c2.label.set_text("[ ] 梯度 SIFT 匹配")
        if enabling and not ax_debug_B.get_visible():
            # Region mode needs both lower panels to explain the shared group
            # shift; keep the upper/lower right visibility controls coherent.
            ax_B.set_visible(True)
            ax_debug_B.set_visible(True)
            ax_debug_info_B.set_visible(True)
            btn_hide_R.label.set_text("隱藏右圖")
        request_blit_refresh()
        mark_wound_size_dirty('region_sift')
        if last_click:
            do_measure(last_click[0], last_click[1])

    c2.on_clicked(on_grad_sift_mode_clicked)
    c18.on_clicked(on_region_sift_mode_clicked)
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
        if (ENABLE_WOUND_AI and use_wound_adaptive_spatial_specular
                and wound_state.get('left_pred') is None and wound_state.get('right_pred') is None):
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
    grad_matcher_config_lock = threading.Lock()  # Debug-only temporary matcher config

    def build_grad_descriptor_audit(ref_a, ref_b, ref_a_groups, ref_b_groups,
                                    left_gray, right_gray, left_bgr, cand,
                                    click_pt, snap_view_state,
                                    final_pts_a=None, final_pts_b=None,
                                    final_groups=None):
        """Recompute the read-only global KNN distances used by Grad-SIFT/ORB."""
        ratio_limit = float(getattr(stereo_algo, 'GRAD_SIFT_RATIO_TEST', 0.78))
        epi_limit = float(getattr(stereo_algo, 'GRAD_SIFT_EPIPOLAR_TOL_PX', 3.0))
        seed_limit = float(getattr(stereo_algo, 'GRAD_SIFT_MAX_RT_ADJUST_PX', 80.0))
        guided_radius = float(getattr(
            stereo_algo, 'GRAD_SIFT_GUIDED_RADIUS_PX', 10.0))
        guided_ratio_limit = float(getattr(
            stereo_algo, 'GRAD_SIFT_GUIDED_RATIO_TEST', 0.95))
        min_group_inliers = int(getattr(
            stereo_algo, 'GRAD_SIFT_MIN_GROUP_INLIERS', 3))
        use_hamming = bool(snap_view_state.get('use_hamming', False))
        use_rgb_sift = bool(snap_view_state.get('use_rgb_sift', False))
        use_opponent_sift = bool(snap_view_state.get('use_opponent_sift', False))
        top2_geometry_mode = bool(
            snap_view_state.get('top2_geometry_rescue', False))

        if use_hamming:
            descriptor_name = 'ORB/Hamming'
            match_threshold = 100.0
            norm_type = cv2.NORM_HAMMING
        elif use_rgb_sift:
            descriptor_name = 'RGB-SIFT/L2'
            match_threshold = 780.0
            norm_type = cv2.NORM_L2
        elif use_opponent_sift:
            descriptor_name = 'Opponent-SIFT/L2'
            match_threshold = 780.0
            norm_type = cv2.NORM_L2
        else:
            descriptor_name = 'Gray-SIFT/L2'
            match_threshold = 450.0
            norm_type = cv2.NORM_L2

        audit = {
            'kind': 'global_knn',
            'descriptor_name': descriptor_name,
            'ratio_limit': ratio_limit,
            'guided_radius': guided_radius,
            'guided_ratio_limit': guided_ratio_limit,
            'min_group_inliers': min_group_inliers,
            'match_threshold': match_threshold,
            'top2_geometry_mode': top2_geometry_mode,
            'top2_geometry_max_dist_px': float(TOP2_GEOMETRY_MAX_DIST_PX),
            'top2_geometry_result': None,
            'records': [],
            'groups': {},
            'default_index': None,
        }
        if ref_a is None or ref_b is None or left_gray is None or right_gray is None:
            return audit

        ref_a = np.asarray(ref_a, dtype=np.float32).reshape(-1, 2)
        ref_b = np.asarray(ref_b, dtype=np.float32).reshape(-1, 2)
        labels_a = np.asarray(
            ref_a_groups if ref_a_groups is not None
            else ['high'] * len(ref_a), dtype=object).reshape(-1)
        labels_b = np.asarray(
            ref_b_groups if ref_b_groups is not None
            else ['high'] * len(ref_b), dtype=object).reshape(-1)
        if len(labels_a) != len(ref_a):
            labels_a = np.asarray(['high'] * len(ref_a), dtype=object)
        if len(labels_b) != len(ref_b):
            labels_b = np.asarray(['high'] * len(ref_b), dtype=object)

        right_bgr = None
        if cand.get('rgb') is not None:
            right_bgr = cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR)

        def compute_descriptors(gray, bgr, keypoints):
            if not keypoints:
                return None, None
            if use_hamming:
                return orb.compute(gray, keypoints)
            if use_rgb_sift:
                return compute_rgb_sift_descriptors(bgr, keypoints, sift)
            if use_opponent_sift:
                return compute_opponent_sift_descriptors(bgr, keypoints, sift)
            return sift.compute(gray, keypoints)

        click_arr = np.asarray(click_pt, dtype=np.float32).reshape(2)
        try:
            rt_seed, _ = predict_right_seed_from_geometry(click_arr, cand, KL)
            rt_seed = np.asarray(rt_seed, dtype=np.float32).reshape(2)
        except Exception:
            rt_seed = click_arr.copy()

        H_ab = None
        try:
            plane_n = np.asarray(cand.get('plane_n'), dtype=np.float64).reshape(3)
            plane_c = np.asarray(cand.get('plane_c'), dtype=np.float64).reshape(3)
            d_plane = float(np.dot(plane_n, plane_c))
            if abs(d_plane) > 1e-8:
                H_ab = (
                    np.asarray(cand['K_R'], dtype=np.float64)
                    @ (
                        np.asarray(cand['R_rel'], dtype=np.float64).reshape(3, 3)
                        + np.asarray(cand['t_rel'], dtype=np.float64).reshape(3, 1)
                        @ plane_n.reshape(1, 3) / d_plane
                    )
                    @ np.linalg.inv(np.asarray(KL, dtype=np.float64)))
        except (TypeError, ValueError, np.linalg.LinAlgError):
            H_ab = None

        def project_h_point(point):
            if H_ab is None:
                return None
            point_h = H_ab @ np.array(
                [float(point[0]), float(point[1]), 1.0], dtype=np.float64)
            if not np.all(np.isfinite(point_h)) or abs(float(point_h[2])) <= 1e-9:
                return None
            return np.asarray(point_h[:2] / point_h[2], dtype=np.float32)

        def point_epi_distance(point_left, point_right):
            if cand.get('F') is None:
                return None
            line = cand['F'] @ np.array(
                [float(point_left[0]), float(point_left[1]), 1.0],
                dtype=np.float64)
            denom = float(np.hypot(line[0], line[1]))
            if denom <= 1e-8:
                return None
            return abs(float(
                line[0] * float(point_right[0])
                + line[1] * float(point_right[1]) + line[2])) / denom

        top2_group_results = {}

        def build_top2_geometry_group(group_records, group_label, group_summary):
            """Run Top-1/Top-2 geometry rescue and report every sequential gate."""
            gate = {
                'ratio_rejected': 0,
                'within_click_radius': 0,
                'has_h_prediction': 0,
                'abs_any': 0,
                'mutual_any': 0,
                'epi_any': 0,
                'h_distance_any': 0,
                'global_proposals': 0,
                'rescue_proposals': 0,
                'pre_unique': 0,
                'after_unique': 0,
                'after_offset': 0,
                'ransac_inliers': None,
            }
            first_reject = {
                'CLICK_RADIUS': 0,
                'NO_H': 0,
                'ABS': 0,
                'MUTUAL': 0,
                'EPIPOLAR': 0,
                'H_DISTANCE': 0,
            }
            h_min_samples = []
            for record in group_records:
                record['top2_geo_selected'] = False
                record['top2_geo_selected_pR'] = None
                record['top2_geo_source'] = None
                record['top2_geo_geom_dist'] = None
                record['top2_geo_rank'] = None
                record['top2_geo_reject_stage'] = None

            def print_gate_summary(status):
                group_summary['top2_geo_gates'] = dict(gate)
                group_summary['top2_geo_first_reject'] = dict(first_reject)
                flow = (
                    f"start={gate['ratio_rejected']} -> "
                    f"click<50={gate['within_click_radius']} -> "
                    f"H={gate['has_h_prediction']} -> "
                    f"abs={gate['abs_any']} -> "
                    f"mutual={gate['mutual_any']} -> "
                    f"epi={gate['epi_any']} -> "
                    f"Hdist<={TOP2_GEOMETRY_MAX_DIST_PX:g}="
                    f"{gate['h_distance_any']}")
                print(f"   [Top2-Geo gates {group_label}] ratio-fail points: {flow}")
                print(
                    f"   [Top2-Geo rejects {group_label}] first-fail "
                    + ", ".join(
                        f"{name}={count}" for name, count in first_reject.items()))
                if h_min_samples:
                    h_values = np.asarray(h_min_samples, dtype=np.float64)
                    print(
                        f"   [Top2-Geo Hdist {group_label}] after epi, nearest "
                        f"Top1/Top2 min/median/p90="
                        f"{np.min(h_values):.2f}/"
                        f"{np.median(h_values):.2f}/"
                        f"{np.percentile(h_values, 90):.2f}px "
                        f"(limit<={TOP2_GEOMETRY_MAX_DIST_PX:g}px)")
                ransac_text = (
                    'N/A' if gate['ransac_inliers'] is None
                    else str(gate['ransac_inliers']))
                print(
                    f"   [Top2-Geo group {group_label}] "
                    f"global={gate['global_proposals']} + "
                    f"rescued={gate['rescue_proposals']} -> "
                    f"pre_unique={gate['pre_unique']} -> "
                    f"unique={gate['after_unique']} -> "
                    f"offset={gate['after_offset']} -> "
                    f"RANSAC={ransac_text}; status={status}")
                if gate['ratio_rejected'] > 0:
                    dominant_name, dominant_count = max(
                        first_reject.items(), key=lambda item: item[1])
                    if dominant_count > 0:
                        dominant_pct = (
                            100.0 * dominant_count / gate['ratio_rejected'])
                        print(
                            f"   [Top2-Geo bottleneck {group_label}] "
                            f"{dominant_name} rejected the most: "
                            f"{dominant_count}/{gate['ratio_rejected']} "
                            f"({dominant_pct:.1f}%)")
                    else:
                        print(
                            f"   [Top2-Geo bottleneck {group_label}] "
                            "none in point gates; inspect unique/offset/RANSAC "
                            "in the group flow above")

            proposals = []
            for record in group_records:
                p_left = np.asarray(record['pL'], dtype=np.float32)
                exact_seed = record.get('exact_h_seed')

                if record['decision'] == 'KNN_PASS':
                    # Normal Global Top-1 path.  It still uses exact H(pL) for
                    # the downstream seed bound while Top2-Geo mode is active.
                    if exact_seed is None:
                        exact_seed = record.get('translated_seed')
                    if exact_seed is None:
                        continue
                    epi = record.get('epi_distance')
                    if epi is not None and float(epi) > epi_limit:
                        continue
                    geom_dist = float(np.linalg.norm(
                        np.asarray(record['top1'], dtype=np.float32)
                        - np.asarray(exact_seed, dtype=np.float32)))
                    if geom_dist > seed_limit:
                        continue
                    selected = {
                        'pR': np.asarray(record['top1'], dtype=np.float32),
                        'train_index': int(record['top1_train_index']),
                        'descriptor_distance': float(record['d1']),
                        'geom_dist': geom_dist,
                        'source': 'GLOBAL_KNN',
                        'rank': 'Top1',
                    }
                    gate['global_proposals'] += 1
                elif record['pass_distance'] and not record['pass_ratio']:
                    gate['ratio_rejected'] += 1
                    if float(np.linalg.norm(p_left - click_arr)) >= 50.0:
                        first_reject['CLICK_RADIUS'] += 1
                        record['top2_geo_reject_stage'] = 'CLICK_RADIUS'
                        continue
                    gate['within_click_radius'] += 1
                    if exact_seed is None:
                        first_reject['NO_H'] += 1
                        record['top2_geo_reject_stage'] = 'NO_H'
                        continue
                    gate['has_h_prediction'] += 1
                    exact_seed = np.asarray(exact_seed, dtype=np.float32)
                    raw_candidates = [
                        {
                            'rank': 'Top1',
                            'pR': record.get('top1'),
                            'distance': record.get('d1'),
                            'train_index': record.get('top1_train_index'),
                            'mutual': record.get('pass_mutual'),
                            'epi': record.get('epi_distance'),
                        },
                        {
                            'rank': 'Top2',
                            'pR': record.get('top2'),
                            'distance': record.get('d2'),
                            'train_index': record.get('top2_train_index'),
                            'mutual': record.get('top2_pass_mutual'),
                            'epi': record.get('top2_epi_distance'),
                        },
                    ]
                    candidates = [
                        item for item in raw_candidates
                        if (item['pR'] is not None
                            and item['distance'] is not None
                            and item['train_index'] is not None
                            and float(item['distance']) < match_threshold)
                    ]
                    if not candidates:
                        first_reject['ABS'] += 1
                        record['top2_geo_reject_stage'] = 'ABS'
                        continue
                    gate['abs_any'] += 1
                    candidates = [item for item in candidates if item['mutual']]
                    if not candidates:
                        first_reject['MUTUAL'] += 1
                        record['top2_geo_reject_stage'] = 'MUTUAL'
                        continue
                    gate['mutual_any'] += 1
                    candidates = [
                        item for item in candidates
                        if item['epi'] is None or float(item['epi']) <= epi_limit]
                    if not candidates:
                        first_reject['EPIPOLAR'] += 1
                        record['top2_geo_reject_stage'] = 'EPIPOLAR'
                        continue
                    gate['epi_any'] += 1
                    for item in candidates:
                        item['geom_dist'] = float(np.linalg.norm(
                            np.asarray(item['pR'], dtype=np.float32) - exact_seed))
                    h_min_samples.append(min(
                        item['geom_dist'] for item in candidates))
                    candidates = [
                        item for item in candidates
                        if item['geom_dist'] <= TOP2_GEOMETRY_MAX_DIST_PX]
                    if not candidates:
                        first_reject['H_DISTANCE'] += 1
                        record['top2_geo_reject_stage'] = 'H_DISTANCE'
                        continue
                    gate['h_distance_any'] += 1
                    selected_candidate = min(
                        candidates,
                        key=lambda item: (
                            item['geom_dist'], float(item['distance'])))
                    selected = {
                        'pR': np.asarray(
                            selected_candidate['pR'], dtype=np.float32),
                        'train_index': int(selected_candidate['train_index']),
                        'descriptor_distance': float(
                            selected_candidate['distance']),
                        'geom_dist': float(selected_candidate['geom_dist']),
                        'source': 'TOP2_GEO_RESCUE',
                        'rank': selected_candidate['rank'],
                    }
                    gate['rescue_proposals'] += 1
                else:
                    continue

                if snap_view_state.get('use_color_hist', False):
                    if not check_color_histogram_similarity(
                            left_bgr, p_left, selected['pR'], cand,
                            patch_size=16, threshold=0.45):
                        continue
                proposals.append({
                    'record': record,
                    'pL': p_left,
                    'pR': selected['pR'],
                    'off': selected['pR'] - p_left,
                    'dist': selected['descriptor_distance'],
                    'geom_dist': selected['geom_dist'],
                    'train_index': selected['train_index'],
                    'source': selected['source'],
                    'rank': selected['rank'],
                })

            gate['pre_unique'] = len(proposals)
            best_by_right = {}
            for item in proposals:
                previous = best_by_right.get(item['train_index'])
                item_key = (
                    0 if item['source'] == 'GLOBAL_KNN' else 1,
                    item['geom_dist'], item['dist'])
                if previous is None:
                    best_by_right[item['train_index']] = item
                    continue
                previous_key = (
                    0 if previous['source'] == 'GLOBAL_KNN' else 1,
                    previous['geom_dist'], previous['dist'])
                if item_key < previous_key:
                    best_by_right[item['train_index']] = item
            proposals = list(best_by_right.values())
            gate['after_unique'] = len(proposals)

            if len(proposals) >= 3:
                offsets = np.asarray(
                    [item['off'] for item in proposals], dtype=np.float32)
                median_offset = np.median(offsets, axis=0)
                proposals = [
                    item for item in proposals
                    if float(np.linalg.norm(item['off'] - median_offset))
                    < GRAD_SIFT_OFFSET_MEDIAN_TOL_PX]
            gate['after_offset'] = len(proposals)
            if len(proposals) < GRAD_SIFT_MIN_GROUP_INLIERS:
                print_gate_summary('REJECT_SUPPORT')
                return None

            pts_a = np.asarray(
                [item['pL'] for item in proposals], dtype=np.float32)
            pts_b = np.asarray(
                [item['pR'] for item in proposals], dtype=np.float32)
            M_local, inliers = cv2.estimateAffinePartial2D(
                pts_a, pts_b, method=cv2.RANSAC,
                ransacReprojThreshold=GRAD_SIFT_RANSAC_REPROJ_PX)
            if M_local is None:
                gate['ransac_inliers'] = 0
                print_gate_summary('REJECT_AFFINE')
                return None
            if inliers is not None:
                inlier_mask = inliers.ravel().astype(bool)
                gate['ransac_inliers'] = int(np.count_nonzero(inlier_mask))
                if gate['ransac_inliers'] < GRAD_SIFT_MIN_GROUP_INLIERS:
                    print_gate_summary('REJECT_RANSAC')
                    return None
                pts_a = pts_a[inlier_mask]
                pts_b = pts_b[inlier_mask]
                proposals = [
                    item for item, keep in zip(proposals, inlier_mask) if keep]
            else:
                gate['ransac_inliers'] = len(proposals)

            # Recompute after RANSAC: rejected points cannot leak into fallback.
            weights = 1.0 / (
                np.sum((pts_a - click_arr) ** 2, axis=1) + 1e-5)
            weighted_group = (
                click_arr
                + np.sum((pts_b - pts_a) * weights[:, None], axis=0)
                / np.sum(weights))
            mapped_group = (
                M_local @ np.array(
                    [float(click_arr[0]), float(click_arr[1]), 1.0])
            )[:2]
            center_b = np.mean(pts_b, axis=0)
            radius_b = max(float(np.percentile(
                np.linalg.norm(pts_b - center_b, axis=1), 90)), 3.0)
            if float(np.linalg.norm(mapped_group - center_b)) > radius_b * 1.25:
                mapped_group = weighted_group
            if float(np.linalg.norm(mapped_group - rt_seed)) > seed_limit:
                print_gate_summary('REJECT_FINAL_RT')
                return None

            for item in proposals:
                record = item['record']
                record['top2_geo_selected'] = True
                record['top2_geo_selected_pR'] = item['pR'].copy()
                record['top2_geo_source'] = item['source']
                record['top2_geo_geom_dist'] = float(item['geom_dist'])
                record['top2_geo_rank'] = item['rank']
                record['top2_geo_reject_stage'] = None
            offsets = pts_b - pts_a
            offset_spread = float(np.median(np.linalg.norm(
                offsets - np.median(offsets, axis=0), axis=1)))
            mean_dist = float(np.mean([item['dist'] for item in proposals]))
            score = len(proposals) / (
                1.0 + offset_spread + mean_dist / max(match_threshold, 1.0))
            rescued_final = sum(
                item['source'] == 'TOP2_GEO_RESCUE' for item in proposals)
            rescued_top2 = sum(
                item['source'] == 'TOP2_GEO_RESCUE'
                and item['rank'] == 'Top2' for item in proposals)
            print_gate_summary('ACCEPT')
            print(
                f"   [Top2-Geo {group_label}] accepted: "
                f"inliers={len(proposals)}, rescued={rescued_final} "
                f"(Top2={rescued_top2}), spread={offset_spread:.2f}, "
                f"score={score:.2f}")
            return {
                'mapped': np.asarray(mapped_group, dtype=np.float32),
                'ptsA': pts_a,
                'ptsB': pts_b,
                'score': float(score),
            }

        for group_key, group_label in (('high', 'HIGH'), ('mid', 'MID')):
            points_left = ref_a[labels_a == group_key]
            points_right = ref_b[labels_b == group_key]
            group_summary = {
                'raw_left': int(len(points_left)),
                'raw_right': int(len(points_right)),
                'desc_left': 0,
                'desc_right': 0,
                'record_count': 0,
            }
            audit['groups'][group_key] = group_summary
            if len(points_left) == 0 or len(points_right) == 0:
                print(
                    f"   [Descriptor Audit {descriptor_name} {group_label}] "
                    f"no raw references: left={len(points_left)}, right={len(points_right)}")
                continue

            kpts_left_raw = [
                cv2.KeyPoint(float(point[0]), float(point[1]), 31.0)
                for point in points_left
            ]
            kpts_right_raw = [
                cv2.KeyPoint(float(point[0]), float(point[1]), 31.0)
                for point in points_right
            ]
            try:
                kpts_left, des_left = compute_descriptors(
                    left_gray, left_bgr, kpts_left_raw)
                kpts_right, des_right = compute_descriptors(
                    right_gray, right_bgr, kpts_right_raw)
            except cv2.error as exc:
                group_summary['error'] = str(exc)
                print(
                    f"   [Descriptor Audit {descriptor_name} {group_label}] "
                    f"descriptor error: {exc}")
                continue
            if (kpts_left is None or des_left is None or
                    kpts_right is None or des_right is None or
                    len(des_left) == 0 or len(des_right) == 0):
                print(
                    f"   [Descriptor Audit {descriptor_name} {group_label}] "
                    "no descriptors")
                continue

            n_left = min(len(kpts_left), len(des_left))
            n_right = min(len(kpts_right), len(des_right))
            kpts_left = list(kpts_left[:n_left])
            kpts_right = list(kpts_right[:n_right])
            des_left = des_left[:n_left]
            des_right = des_right[:n_right]
            group_summary['desc_left'] = int(n_left)
            group_summary['desc_right'] = int(n_right)
            if n_left == 0 or n_right == 0:
                continue

            matcher = cv2.BFMatcher(norm_type)
            try:
                knn_lr = matcher.knnMatch(des_left, des_right, k=2)
                knn_rl = matcher.knnMatch(des_right, des_left, k=1)
            except cv2.error as exc:
                group_summary['error'] = str(exc)
                print(
                    f"   [Descriptor Audit {descriptor_name} {group_label}] "
                    f"KNN error: {exc}")
                continue
            reverse_best = {
                match.queryIdx: match.trainIdx
                for pair in knn_rl for match in pair[:1]
            }

            group_records = []
            for pair in knn_lr:
                if not pair:
                    continue
                best = pair[0]
                second = pair[1] if len(pair) > 1 else None
                p_left = np.asarray(
                    kpts_left[best.queryIdx].pt, dtype=np.float32)
                p_top1 = np.asarray(
                    kpts_right[best.trainIdx].pt, dtype=np.float32)
                p_top2 = (
                    np.asarray(kpts_right[second.trainIdx].pt, dtype=np.float32)
                    if second is not None else None
                )
                d1 = float(best.distance)
                d2 = float(second.distance) if second is not None else None
                if d2 is None:
                    ratio = None
                elif d2 == 0.0:
                    ratio = (
                        float('nan') if d1 == 0.0 else float('inf'))
                else:
                    ratio = d1 / d2
                pass_distance = d1 < match_threshold
                # This intentionally mirrors the matcher: with one neighbour
                # there is no ratio rejection; with d2==0 even 0/0 is rejected.
                pass_ratio = (
                    second is None or d1 < ratio_limit * float(second.distance))
                pass_mutual = reverse_best.get(best.trainIdx) == best.queryIdx
                if not pass_distance:
                    decision = 'DIST'
                elif not pass_ratio:
                    decision = 'RATIO'
                elif not pass_mutual:
                    decision = 'MUTUAL'
                else:
                    decision = 'KNN_PASS'

                epi_distance = point_epi_distance(p_left, p_top1)
                top2_epi_distance = (
                    point_epi_distance(p_left, p_top2)
                    if p_top2 is not None else None)
                translated_seed = rt_seed + (p_left - click_arr)
                exact_h_seed = project_h_point(p_left)
                local_seed = (
                    exact_h_seed if top2_geometry_mode and exact_h_seed is not None
                    else translated_seed)
                seed_distance = float(np.linalg.norm(p_top1 - local_seed))
                top2_seed_distance = (
                    float(np.linalg.norm(p_top2 - local_seed))
                    if p_top2 is not None else None)
                top12_spatial_distance = (
                    float(np.linalg.norm(p_top1 - p_top2))
                    if p_top2 is not None else None
                )
                record = {
                    'audit_index': len(audit['records']),
                    'group': group_key,
                    'group_label': group_label,
                    'query_index': int(best.queryIdx),
                    'top1_train_index': int(best.trainIdx),
                    'top2_train_index': (
                        int(second.trainIdx) if second is not None else None),
                    'pL': p_left,
                    'top1': p_top1,
                    'top2': p_top2,
                    'local_seed': local_seed,
                    'd1': d1,
                    'd2': d2,
                    'ratio': ratio,
                    'gap': (d2 - d1) if d2 is not None else None,
                    'top12_spatial_distance': top12_spatial_distance,
                    'pass_distance': bool(pass_distance),
                    'pass_ratio': bool(pass_ratio),
                    'pass_mutual': bool(pass_mutual),
                    'top2_pass_distance': bool(
                        second is not None
                        and float(second.distance) < match_threshold),
                    'top2_pass_mutual': bool(
                        second is not None
                        and reverse_best.get(second.trainIdx) == best.queryIdx),
                    'decision': decision,
                    'epi_distance': epi_distance,
                    'pass_epi': (
                        None if epi_distance is None
                        else bool(epi_distance <= epi_limit)
                    ),
                    'seed_distance': seed_distance,
                    'pass_seed': bool(seed_distance <= seed_limit),
                    'top2_epi_distance': top2_epi_distance,
                    'top2_pass_epi': (
                        None if top2_epi_distance is None
                        else bool(top2_epi_distance <= epi_limit)),
                    'top2_seed_distance': top2_seed_distance,
                    'translated_seed': translated_seed,
                    'exact_h_seed': exact_h_seed,
                    'match_threshold': match_threshold,
                    'ratio_limit': ratio_limit,
                }
                audit['records'].append(record)
                group_records.append(record)

            # Replay the matcher-guided candidate restriction without feeding
            # anything back into matching.  This uses the same returned
            # keypoints/descriptors and the same strict gates as
            # map_from_gradient_group().
            global_good_count = sum(
                record['decision'] == 'KNN_PASS' for record in group_records)
            guided_triggered = (
                not top2_geometry_mode
                and global_good_count < min_group_inliers)
            group_summary['guided_triggered'] = bool(guided_triggered)
            group_summary['guided_attempted'] = 0
            group_summary['guided_candidate_pass'] = 0
            pts_right_arr = np.asarray(
                [kp.pt for kp in kpts_right], dtype=np.float32)
            for record in group_records:
                record.update({
                    'guided_triggered': bool(guided_triggered),
                    'guided_radius': guided_radius,
                    'guided_ratio_limit': guided_ratio_limit,
                    'guided_attempted': False,
                    'guided_radius_candidate_count': 0,
                    'guided_candidate_count': 0,
                    'guided_top1': None,
                    'guided_top2': None,
                    'guided_d1': None,
                    'guided_d2': None,
                    'guided_ratio': None,
                    'guided_pass_distance': None,
                    'guided_pass_ratio': None,
                    'guided_candidate_pass': False,
                    'guided_decision': (
                        'GLOBAL_ALREADY_GOOD'
                        if record['decision'] == 'KNN_PASS'
                        else 'NOT_TRIGGERED'),
                })
                if not guided_triggered or record['decision'] == 'KNN_PASS':
                    continue

                p_left = record['pL']
                if float(np.linalg.norm(p_left - click_arr)) >= 50.0:
                    record['guided_decision'] = 'OUTSIDE_CLICK_RADIUS'
                    continue
                record['guided_attempted'] = True
                group_summary['guided_attempted'] += 1
                local_seed = record['local_seed']
                spatial_d = np.linalg.norm(pts_right_arr - local_seed, axis=1)
                cand_idx = np.flatnonzero(spatial_d <= guided_radius)
                record['guided_radius_candidate_count'] = int(len(cand_idx))
                if len(cand_idx) == 0:
                    record['guided_decision'] = 'NO_RADIUS_CANDIDATES'
                    continue

                if cand.get('F') is not None:
                    epi_line = cand['F'] @ np.array(
                        [p_left[0], p_left[1], 1.0], dtype=np.float64)
                    denom = float(np.hypot(epi_line[0], epi_line[1]))
                    if denom > 1e-8:
                        epi_d = np.abs(
                            epi_line[0] * pts_right_arr[cand_idx, 0]
                            + epi_line[1] * pts_right_arr[cand_idx, 1]
                            + epi_line[2]
                        ) / denom
                        cand_idx = cand_idx[epi_d <= epi_limit]
                record['guided_candidate_count'] = int(len(cand_idx))
                if len(cand_idx) == 0:
                    record['guided_decision'] = 'NO_EPI_CANDIDATES'
                    continue

                qi = int(record['query_index'])
                if use_hamming:
                    guided_dists = np.asarray([
                        cv2.norm(
                            des_left[qi], des_right[int(ri)],
                            cv2.NORM_HAMMING)
                        for ri in cand_idx
                    ], dtype=np.float32)
                else:
                    guided_diff = (
                        des_right[cand_idx].astype(np.float32)
                        - des_left[qi].astype(np.float32))
                    guided_dists = np.linalg.norm(guided_diff, axis=1)
                order = np.argsort(guided_dists)
                best_pos = int(order[0])
                best_ri = int(cand_idx[best_pos])
                guided_d1 = float(guided_dists[best_pos])
                if len(order) > 1:
                    second_pos = int(order[1])
                    second_ri = int(cand_idx[second_pos])
                    guided_d2 = float(guided_dists[second_pos])
                    guided_top2 = pts_right_arr[second_ri].copy()
                    if guided_d2 == 0.0:
                        guided_ratio = (
                            float('nan') if guided_d1 == 0.0
                            else float('inf'))
                    else:
                        guided_ratio = guided_d1 / guided_d2
                else:
                    guided_d2 = None
                    guided_top2 = None
                    guided_ratio = None

                guided_pass_distance = guided_d1 < match_threshold
                guided_pass_ratio = (
                    guided_d2 is None
                    or guided_d1 < guided_ratio_limit * guided_d2)
                if not guided_pass_distance:
                    guided_decision = 'DIST'
                elif not guided_pass_ratio:
                    guided_decision = 'RATIO'
                else:
                    guided_decision = 'CANDIDATE_PASS'
                    group_summary['guided_candidate_pass'] += 1
                record.update({
                    'guided_top1': pts_right_arr[best_ri].copy(),
                    'guided_top2': guided_top2,
                    'guided_d1': guided_d1,
                    'guided_d2': guided_d2,
                    'guided_ratio': guided_ratio,
                    'guided_pass_distance': bool(guided_pass_distance),
                    'guided_pass_ratio': bool(guided_pass_ratio),
                    'guided_candidate_pass': bool(
                        guided_pass_distance and guided_pass_ratio),
                    'guided_decision': guided_decision,
                })

            group_summary['record_count'] = len(group_records)
            group_summary['distance_pass'] = int(sum(
                record['pass_distance'] for record in group_records))
            group_summary['ratio_pass'] = int(sum(
                record['pass_ratio'] for record in group_records))
            group_summary['mutual_pass'] = int(sum(
                record['pass_mutual'] for record in group_records))
            group_summary['ratio_stage_reached'] = int(sum(
                record['pass_distance'] for record in group_records))
            group_summary['ratio_stage_pass'] = int(sum(
                record['pass_distance'] and record['pass_ratio']
                for record in group_records))
            group_summary['mutual_stage_reached'] = int(sum(
                record['pass_distance'] and record['pass_ratio']
                for record in group_records))
            group_summary['knn_pass'] = int(sum(
                record['decision'] == 'KNN_PASS' for record in group_records))
            group_summary['reject_dist'] = int(sum(
                record['decision'] == 'DIST' for record in group_records))
            group_summary['reject_ratio'] = int(sum(
                record['decision'] == 'RATIO' for record in group_records))
            group_summary['reject_mutual'] = int(sum(
                record['decision'] == 'MUTUAL' for record in group_records))
            group_summary['d2_zero'] = int(sum(
                record.get('d2') == 0.0 for record in group_records))
            finite_ratios = np.asarray([
                record['ratio'] for record in group_records
                if record['ratio'] is not None and np.isfinite(record['ratio'])
            ], dtype=np.float64)
            if finite_ratios.size:
                group_summary['ratio_min'] = float(np.min(finite_ratios))
                group_summary['ratio_median'] = float(np.median(finite_ratios))
                group_summary['ratio_p90'] = float(
                    np.percentile(finite_ratios, 90))
                ratio_stats = (
                    f"min/median/p90={group_summary['ratio_min']:.3f}/"
                    f"{group_summary['ratio_median']:.3f}/"
                    f"{group_summary['ratio_p90']:.3f}")
            else:
                ratio_stats = 'min/median/p90=N/A'
            sensitivity = ', '.join(
                f"<{limit:.2f}:{int(np.count_nonzero(finite_ratios < limit))}"
                for limit in (ratio_limit, 0.85, 0.90, 0.95)
            )
            print(
                f"   [Descriptor Audit {descriptor_name} {group_label}] "
                f"n={len(group_records)}, abs<{match_threshold:g}:"
                f"{group_summary['distance_pass']}, "
                f"Lowe<{ratio_limit:.2f}:{group_summary['ratio_pass']}, "
                f"mutual(any):{group_summary['mutual_pass']}, "
                f"KNN_PASS:{group_summary['knn_pass']}, "
                f"stage reject D/R/M="
                f"{group_summary['reject_dist']}/"
                f"{group_summary['reject_ratio']}/"
                f"{group_summary['reject_mutual']}, "
                f"d2=0:{group_summary['d2_zero']} | "
                f"ratio {ratio_stats} | {sensitivity}")
            global_rejects = {
                'DIST': group_summary['reject_dist'],
                'RATIO': group_summary['reject_ratio'],
                'MUTUAL': group_summary['reject_mutual'],
            }
            dominant_gate, dominant_count = max(
                global_rejects.items(), key=lambda item: item[1])
            dominant_pct = (
                100.0 * dominant_count / len(group_records)
                if group_records else 0.0)
            largest_text = (
                f"{dominant_gate} {dominant_count}/{len(group_records)} "
                f"({dominant_pct:.1f}%)"
                if dominant_count > 0 else 'NONE (all KNN-pass)')
            print(
                f"   [Descriptor Audit bottleneck {group_label}] "
                f"Global KNN first-fail: DIST={global_rejects['DIST']}, "
                f"RATIO={global_rejects['RATIO']}, "
                f"MUTUAL={global_rejects['MUTUAL']}; largest={largest_text}")
            if top2_geometry_mode:
                print(
                    f"   [Descriptor Audit guided {group_label}] "
                    "paused by Top2-Geo mode; no arbitrary local candidates")
                top2_group_results[group_key] = build_top2_geometry_group(
                    group_records, group_label, group_summary)
            else:
                print(
                    f"   [Descriptor Audit guided {group_label}] "
                    f"triggered={guided_triggered} "
                    f"(global_good={global_good_count} < "
                    f"min={min_group_inliers}), "
                    f"attempted={group_summary['guided_attempted']}, "
                    f"candidate_pass={group_summary['guided_candidate_pass']}, "
                    f"radius<={guided_radius:g}px, epi<={epi_limit:g}px, "
                    f"ratio<{guided_ratio_limit:.2f}")
                if guided_triggered:
                    guided_pool = sum(
                        record['decision'] != 'KNN_PASS'
                        for record in group_records)
                    radius_nonempty = sum(
                        record.get('guided_radius_candidate_count', 0) > 0
                        for record in group_records)
                    epi_nonempty = sum(
                        record.get('guided_candidate_count', 0) > 0
                        for record in group_records)
                    guided_abs_pass = sum(
                        record.get('guided_pass_distance') is True
                        for record in group_records)
                    guided_ratio_pass = sum(
                        record.get('guided_candidate_pass') is True
                        for record in group_records)
                    guided_rejects = {}
                    for record in group_records:
                        decision = record.get('guided_decision')
                        if (decision is not None
                                and decision not in (
                                    'GLOBAL_ALREADY_GOOD', 'NOT_TRIGGERED',
                                    'CANDIDATE_PASS')):
                            guided_rejects[decision] = (
                                guided_rejects.get(decision, 0) + 1)
                    reject_text = (
                        ', '.join(
                            f"{key}={value}"
                            for key, value in sorted(guided_rejects.items()))
                        or 'none')
                    print(
                        f"   [Guided gates {group_label}] "
                        f"non-global={guided_pool} -> click<50="
                        f"{group_summary['guided_attempted']} -> "
                        f"radius-hit={radius_nonempty} -> "
                        f"epi-hit={epi_nonempty} -> "
                        f"abs-pass={guided_abs_pass} -> "
                        f"ratio-pass={guided_ratio_pass}; "
                        f"first-fail {reject_text}")

        if top2_geometry_mode:
            valid_top2_groups = [
                (group_key, result)
                for group_key, result in top2_group_results.items()
                if result is not None
            ]
            if valid_top2_groups:
                total_score = sum(
                    max(float(result['score']), 1e-6)
                    for _, result in valid_top2_groups)
                mapped = sum(
                    np.asarray(result['mapped'], dtype=np.float32)
                    * max(float(result['score']), 1e-6)
                    for _, result in valid_top2_groups
                ) / total_score
                top2_pts_a = np.vstack([
                    result['ptsA'] for _, result in valid_top2_groups
                ]).astype(np.float32)
                top2_pts_b = np.vstack([
                    result['ptsB'] for _, result in valid_top2_groups
                ]).astype(np.float32)
                top2_groups = np.concatenate([
                    np.asarray(
                        [group_key] * len(result['ptsA']), dtype=object)
                    for group_key, result in valid_top2_groups
                ])
                valid_keys = {key for key, _ in valid_top2_groups}
                if valid_keys == {'high', 'mid'}:
                    top2_method = 'Grad-SIFT+Top2Geo+MidGradInterp'
                elif 'mid' in valid_keys:
                    top2_method = 'Grad-SIFT+Top2Geo+MidGrad'
                else:
                    top2_method = 'Grad-SIFT+Top2Geo+HighGrad'
                audit['top2_geometry_result'] = {
                    'm_pt': np.asarray(mapped, dtype=np.float32),
                    'method': top2_method,
                    'ptsA': top2_pts_a,
                    'ptsB': top2_pts_b,
                    'groups': top2_groups,
                    'reject_reason': None,
                }
            else:
                audit['top2_geometry_result'] = {
                    'm_pt': None,
                    'method': '',
                    'ptsA': None,
                    'ptsB': None,
                    'groups': None,
                    'reject_reason': (
                        'Top2-Geo: HIGH/MID 均未通過支援點與 RANSAC 門檻'),
                }

        if top2_geometry_mode and audit['top2_geometry_result'] is not None:
            final_pts_a = audit['top2_geometry_result']['ptsA']
            final_pts_b = audit['top2_geometry_result']['ptsB']
            final_groups = audit['top2_geometry_result']['groups']

        # Compare the global-KNN audit with the matcher output that actually
        # survived geometry/offset/RANSAC and participated in interpolation.
        # If a final support point failed global KNN, its only possible source
        # in the current matcher is the guided fallback.
        if final_pts_a is None or final_pts_b is None:
            accepted_left = np.empty((0, 2), dtype=np.float32)
            accepted_right = np.empty((0, 2), dtype=np.float32)
            accepted_groups = np.empty((0,), dtype=object)
        else:
            accepted_left = np.asarray(
                final_pts_a, dtype=np.float32).reshape(-1, 2)
            accepted_right = np.asarray(
                final_pts_b, dtype=np.float32).reshape(-1, 2)
            accepted_count = min(len(accepted_left), len(accepted_right))
            accepted_left = accepted_left[:accepted_count]
            accepted_right = accepted_right[:accepted_count]
            if final_groups is None:
                accepted_groups = np.asarray(
                    ['high'] * accepted_count, dtype=object)
            else:
                accepted_groups = np.asarray(
                    final_groups, dtype=object).reshape(-1)
                if len(accepted_groups) != accepted_count:
                    accepted_groups = np.asarray(
                        ['high'] * accepted_count, dtype=object)

        for record in audit['records']:
            record['used_in_interpolation'] = False
            record['guided_rescue'] = False
            record['support_source'] = 'NOT_USED'
            record['accepted_pR'] = None
            record['guided_matches_final'] = None
            if len(accepted_left) == 0:
                continue
            same_group = np.flatnonzero(accepted_groups == record['group'])
            if len(same_group) == 0:
                continue
            distances = np.linalg.norm(
                accepted_left[same_group] - record['pL'], axis=1)
            nearest_pos = int(np.argmin(distances))
            if float(distances[nearest_pos]) > 0.25:
                continue
            support_index = int(same_group[nearest_pos])
            record['used_in_interpolation'] = True
            record['accepted_pR'] = accepted_right[support_index].copy()
            if (top2_geometry_mode
                    and record.get('top2_geo_selected')
                    and record.get('top2_geo_source') == 'TOP2_GEO_RESCUE'):
                record['support_source'] = 'TOP2_GEO_RESCUE'
            elif record['decision'] == 'KNN_PASS':
                record['support_source'] = 'GLOBAL_KNN'
            else:
                record['guided_rescue'] = True
                record['support_source'] = 'GUIDED_RESCUE'
                guided_top1 = record.get('guided_top1')
                if guided_top1 is not None:
                    record['guided_matches_final'] = bool(
                        np.linalg.norm(
                            np.asarray(guided_top1, dtype=np.float32)
                            - record['accepted_pR']) <= 0.25)

        for group_key, group_label in (('high', 'HIGH'), ('mid', 'MID')):
            group_records = [
                record for record in audit['records']
                if record['group'] == group_key
            ]
            final_support_count = sum(
                record['used_in_interpolation'] for record in group_records)
            guided_rescue_count = sum(
                record['guided_rescue'] for record in group_records)
            global_support_count = sum(
                record['support_source'] == 'GLOBAL_KNN'
                for record in group_records)
            top2_geo_rescue_count = sum(
                record['support_source'] == 'TOP2_GEO_RESCUE'
                for record in group_records)
            audit['groups'].setdefault(group_key, {})[
                'final_support_count'] = int(final_support_count)
            audit['groups'][group_key][
                'guided_rescue_count'] = int(guided_rescue_count)
            audit['groups'][group_key][
                'top2_geo_rescue_count'] = int(top2_geo_rescue_count)
            print(
                f"   [Descriptor Audit support {group_label}] "
                f"final={final_support_count}, "
                f"global={global_support_count}, "
                f"guided_rescue={guided_rescue_count}, "
                f"top2_geo_rescue={top2_geo_rescue_count}")

        if audit['records']:
            nearest = min(
                audit['records'],
                key=lambda record: float(np.linalg.norm(record['pL'] - click_arr)))
            audit['default_index'] = int(nearest['audit_index'])
            d2_text = 'N/A' if nearest['d2'] is None else f"{nearest['d2']:.3f}"
            if nearest['ratio'] is None:
                ratio_text = 'N/A'
            elif np.isnan(nearest['ratio']):
                ratio_text = 'undefined(0/0)'
            elif np.isinf(nearest['ratio']):
                ratio_text = 'inf'
            else:
                ratio_text = f"{nearest['ratio']:.4f}"
            print(
                f"   [Descriptor Audit nearest] #{nearest['audit_index']:03d} "
                f"{nearest['group_label']} "
                f"L=({nearest['pL'][0]:.1f},{nearest['pL'][1]:.1f}) "
                f"Top1=({nearest['top1'][0]:.1f},{nearest['top1'][1]:.1f}) "
                f"d1={nearest['d1']:.3f}, d2={d2_text}, "
                f"d1/d2={ratio_text} -> {nearest['decision']}, "
                f"support={nearest['support_source']}")
        return audit

    # ---- 純計算（可在背景執行緒安全呼叫，不觸碰 Matplotlib）----
    def compute_measure(u, v, snap_cand, snap_imgA_gray, snap_view_state, manual_match_pt=None, left_cache=None):
        """純計算版 do_measure，回傳結果 dict，不更新任何 UI 元件。"""
        nonlocal locked_L, locked_R, locked_L_spec_mask, locked_R_spec_mask, current_cand
        cand = snap_cand
        region_config = REGION_SIFT_CONFIG  # Immutable snapshot for this computation.
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
        rt_bound_reject_reason = None
        g_ptsA, g_ptsB, g_groups, g_refA, g_refB, g_refA_groups, g_refB_groups, g_kptsB, g_rect = None, None, None, None, None, None, None, None, None
        trajectory_res = None
        grad_descriptor_audit = None
        region_debug = None

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
                region_mode = bool(snap_view_state.get('region_sift', False))
                if region_mode:
                    _t_blk = time.perf_counter()
                    region_right_gray = preprocess_gray(
                        cand['gray'], snap_view_state.get('enable_clahe', False))
                    t_prof['右圖灰階/CLAHE前處理'] = time.perf_counter() - _t_blk
                    _t_blk = time.perf_counter()
                    region_result = run_region_sift_matching(
                        snap_imgA_gray, region_right_gray, (u, v), cand, KL,
                        sift=None, config=region_config,
                        left_cache=left_cache,
                        reject_specular=snap_view_state.get('reject_specular_candidates', False),
                        left_spec_mask=left_spec_mask, right_spec_mask=right_spec_mask)
                    t_prof['Region-SIFT匹配'] = time.perf_counter() - _t_blk
                    _t_blk = time.perf_counter()
                    _region_ms = region_result.get('elapsed_ms', 0.0)
                    _region_counts = region_result.get('timing_counts', {})
                    print(f"[Region-SIFT Specular] enabled={_region_counts.get('reject_specular', False)} "
                          f"| support={_region_counts.get('specular_check_support', False)} "
                          f"| center rejects={_region_counts.get('specular_center_rejected_candidates', 0)} "
                          f"| incomplete support rejects={_region_counts.get('incomplete_support_rejected_candidates', 0)}")
                    print(f"⏱️ [F{cand.get('idx')} Region-SIFT細分] {_region_ms:.1f} ms "
                          f"| 左圖cache={'hit' if _region_counts.get('left_cache_hit') else 'miss'} "
                          f"| 右圖descriptor={_region_counts.get('right_descriptor_rows', 0)} rows/"
                          f"{_region_counts.get('right_descriptor_batches', 0)} batches")
                    for _stage_name, _stage_ms in region_result.get('timing_ms', {}).items():
                        _stage_pct = 100.0 * _stage_ms / _region_ms if _region_ms > 0 else 0.0
                        print(f"   - {_stage_name}: {_stage_ms:.1f} ms ({_stage_pct:.1f}%)")
                    region_debug = region_result.get('region_debug')
                    if region_debug is not None:
                        region_debug['diagnostic_match_status'] = (
                            region_result.get('reject_reason') or 'accepted')
                        region_debug['diagnostic_frame_index'] = cand.get('idx')
                        region_debug['diagnostic_geometry'] = {
                            'K_L': np.asarray(KL).copy(), 'K_R': np.asarray(cand['K_R']).copy(),
                            'R': np.asarray(cand['R_rel']).copy(), 't': np.asarray(cand['t_rel']).copy()}
                        second_candidate = region_debug.get('second_candidate')
                        candidate_p3ds = []
                        for candidate_pt in (region_debug['best_center_right'],
                                             None if second_candidate is None else second_candidate['center_right']):
                            xyz = None
                            if candidate_pt is not None:
                                try:
                                    value = triangulate_point_3d(
                                        (u, v), candidate_pt, KL, cand['K_R'],
                                        cand['R_rel'], cand['t_rel'], F=cand.get('F'))
                                    right_value = cand['R_rel'] @ value + np.asarray(cand['t_rel']).reshape(3)
                                    if (np.all(np.isfinite(value)) and 0 < value[2] <= MAX_DEPTH_MM
                                            and right_value[2] > 0):
                                        xyz = value
                                except (ValueError, cv2.error, FloatingPointError):
                                    pass
                            candidate_p3ds.append(xyz)
                        region_debug['diagnostic_candidate_p3ds'] = candidate_p3ds
                    m_pt = region_result.get('m_pt')
                    method = region_result.get('method', '')
                    if region_result.get('reject_reason'):
                        rt_bound_reject_reason = region_result['reject_reason']
                        print(
                            f"   [Region-SIFT] rejected: "
                            f"{region_result['reject_reason']}")
                    if region_debug is not None:
                        delta = np.asarray(
                            region_debug['best_displacement_warp']).reshape(2)
                        print(
                            f"   [Region-SIFT] {'rejected candidate' if region_result.get('reject_reason') else 'accepted'}: "
                            f"points={region_debug['point_count']}, "
                            f"keep={region_debug['keep_count']}, "
                            f"candidates={region_debug['valid_candidate_count']}/"
                            f"{region_debug['candidate_count']}, "
                            f"deltaW=({delta[0]:.1f},{delta[1]:.1f}), "
                            f"GroupScore={region_debug['group_score']:.3f}")
                        print(
                            f"   [Region-SIFT score] "
                            f"Trim{region_debug['keep_count']}={region_debug['trimmed_score']:.3f}, "
                            f"CellAll={region_debug['balanced_score']:.3f}, "
                            f"balanceWeight={region_config.group_balance_weight:.2f}, "
                            f"FramePenalty={region_debug['frame_penalty']:.3f}, "
                            f"Total={region_debug['objective_score']:.3f}; "
                            + (f"all {region_debug['point_count']} rows contribute to CellAll"
                               if region_config.group_balance_weight > 0 else
                               "CellAll diagnostic only (weight=0)"))
                        print(
                            f"   [Region-SIFT frame/support] "
                            f"reliablePairs scale/angle={region_debug['frame_scale_pair_count']}/"
                            f"{region_debug['frame_orientation_pair_count']}, "
                            f"groupScaleRatio={2.0 ** region_debug['frame_scale_center_log2']:.3f}, "
                            f"groupAngleDelta={region_debug['frame_angle_center_deg']:.2f}deg, "
                            f"leftSizes={np.unique(region_debug['left_frame_sizes_px']).round(2).tolist()}, "
                            f"supportMaxL/R={np.max(region_debug['left_frames']['support_radius_px']):.1f}/"
                            f"{np.max(region_debug['right_frames']['support_radius_px']):.1f}px")
                        kept_stats = region_debug.get('kept_l2_stats', {})
                        print(
                            f"   [Region-SIFT ambiguity] "
                            f"bestG={region_debug['group_score']:.3f}, "
                            f"secondG={region_debug.get('second_group_score', float('nan')):.3f} "
                            f"(outside {region_debug.get('second_best_exclusion_radius_px', 0.0):.1f}px), "
                            f"marginG={region_debug.get('group_score_margin', float('nan')):.3f}, "
                            f"relativeG={region_debug.get('relative_group_score', float('nan')):.4f}, "
                            f"bestObj={region_debug.get('objective_score', float('nan')):.3f}, "
                            f"secondObj={region_debug.get('second_objective_score', float('nan')):.3f}, "
                            f"ratioObj={region_debug.get('objective_score_ratio', float('nan')):.4f}")
                        print(
                            f"   [Region-SIFT distribution] "
                            f"KEEP-L2 mean/median/std/min/max="
                            f"{kept_stats.get('mean', float('nan')):.3f}/"
                            f"{kept_stats.get('median', float('nan')):.3f}/"
                            f"{kept_stats.get('std', float('nan')):.3f}/"
                            f"{kept_stats.get('min', float('nan')):.3f}/"
                            f"{kept_stats.get('max', float('nan')):.3f}, "
                            f"cells={region_debug.get('kept_cell_coverage', 0)}/"
                            f"{region_debug.get('total_cell_count', 0)}")
                        print(
                            f"   [Region-SIFT geometry/frame] "
                            f"bestOffset(along,across)=("
                            f"{region_debug.get('best_along_offset_px', float('nan')):.1f},"
                            f"{region_debug.get('best_across_offset_px', float('nan')):.1f})px, "
                            f"secondOffset=("
                            f"{region_debug.get('second_along_offset_px', float('nan')):.1f},"
                            f"{region_debug.get('second_across_offset_px', float('nan')):.1f})px, "
                            f"scaleRatio median/MAD="
                            f"{region_debug.get('kept_scale_ratio_median', float('nan')):.3f}/"
                            f"{region_debug.get('kept_scale_ratio_mad', float('nan')):.3f}, "
                            f"absAngleDelta median/MAD="
                            f"{region_debug.get('kept_abs_angle_delta_median', float('nan')):.1f}/"
                            f"{region_debug.get('kept_abs_angle_delta_mad', float('nan')):.1f}deg")
                    t_prof['Region-SIFT結果處理與log輸出'] = time.perf_counter() - _t_blk

                if not region_mode:
                    for mid, cA in cand['cornersA'].items():
                        d = np.linalg.norm(cA - np.array([u, v]), axis=1)
                        if np.min(d) < 10:
                            best_idx = np.argmin(d)
                            u, v = cA[best_idx] # 🌟 同步校正左圖座標為精確角點
                            m_pt, method = cand['cornersB'][mid][best_idx], "ArUco"
                            break
                if (m_pt is None and not region_mode
                        and snap_view_state['grad_sift']):
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
                        top2_geometry_mode = bool(
                            snap_view_state.get('top2_geometry_rescue', False))
                        with grad_matcher_config_lock:
                            saved_guided_radius = getattr(
                                stereo_algo, 'GRAD_SIFT_GUIDED_RADIUS_PX',
                                GRAD_SIFT_GUIDED_RADIUS_PX)
                            if top2_geometry_mode:
                                # Keep the shared matcher untouched on disk.  A
                                # negative radius makes its original local guided
                                # candidate set empty for this Debug-only call.
                                stereo_algo.GRAD_SIFT_GUIDED_RADIUS_PX = -1.0
                                print(
                                    "   [Top2-Geo] original Guided Fallback disabled; "
                                    "ratio rejects use Global Top-1/Top-2 + exact H(pL)")
                            try:
                                gs = run_grad_sift_matching_flow(
                                    snap_imgA_gray, cand['gray'], u, v, cand, KL,
                                    snap_view_state, orb, sift,
                                    locked_L, locked_R,
                                    left_spec_mask, right_spec_mask,
                                    is_best_cand=(cand['idx'] == current_cand['idx']),
                                    left_cache=left_cache
                                )
                            finally:
                                stereo_algo.GRAD_SIFT_GUIDED_RADIUS_PX = (
                                    saved_guided_radius)
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
                    # Debug-only: replay the same global KNN descriptor stage
                    # for the displayed/best right frame.  This is deliberately
                    # outside the matcher and cannot alter good/inliers/m_pt.
                    if (not snap_view_state.get('use_improved_matching', False)
                            and (cand.get('idx') == current_cand.get('idx')
                                 or snap_view_state.get(
                                     'top2_geometry_rescue', False))
                            and g_refA is not None and g_refB is not None):
                        _t_audit = time.perf_counter()
                        try:
                            grad_descriptor_audit = build_grad_descriptor_audit(
                                g_refA, g_refB,
                                g_refA_groups, g_refB_groups,
                                snap_imgA_gray, cand['gray'],
                                locked_L, cand, (u, v), snap_view_state,
                                final_pts_a=g_ptsA,
                                final_pts_b=g_ptsB,
                                final_groups=g_groups)
                        except Exception as exc:
                            # A diagnostic overlay must never make a valid
                            # measurement fail.
                            print(f"   [Descriptor Audit] unavailable: {exc}")
                            grad_descriptor_audit = {
                                'kind': 'global_knn',
                                'descriptor_name': 'ERROR',
                                'records': [],
                                'groups': {},
                                'error': str(exc),
                            }
                        t_prof['Descriptor audit'] = (
                            time.perf_counter() - _t_audit)
                        if (snap_view_state.get('top2_geometry_rescue', False)
                                and grad_descriptor_audit is not None):
                            top2_result = grad_descriptor_audit.get(
                                'top2_geometry_result')
                            if top2_result is not None:
                                m_pt = top2_result.get('m_pt')
                                method = top2_result.get('method', '')
                                g_ptsA = top2_result.get('ptsA')
                                g_ptsB = top2_result.get('ptsB')
                                g_groups = top2_result.get('groups')
                                rt_bound_reject_reason = (
                                    None if m_pt is not None
                                    else top2_result.get('reject_reason'))
                if (m_pt is None and not region_mode
                        and snap_view_state['precise']):
                    _t_blk = time.perf_counter()
                    res_p = find_precise_match(snap_imgA_gray, cand['gray'], (u, v), cand['F'],
                                               KL, cand['K_R'], cand['R_rel'], cand['t_rel'],
                                               cand['plane_n'], cand['plane_c'])
                    if res_p: m_pt, method = np.array(res_p), "Precise"
                    t_prof['Precise匹配'] = time.perf_counter() - _t_blk
            
            m_pt_raw = m_pt.copy() if m_pt is not None else None

            if (m_pt is not None and method != "ArUco" and manual_match_pt is None
                    and not snap_view_state.get('region_sift', False)
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

            if (m_pt is not None and snap_view_state['enforce_epi'] and method != "ArUco"
                    and not snap_view_state.get('region_sift', False)):
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                      m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])
                    method += "+極線對齊"

            if (m_pt is not None and method != "ArUco" and manual_match_pt is None
                    and not snap_view_state.get('region_sift', False)):
                rt_seed_final, _rt_seed_method = predict_right_seed_from_geometry((u, v), cand, KL)
                rt_dev = float(np.linalg.norm(np.array(m_pt, dtype=np.float32) - np.array(rt_seed_final, dtype=np.float32)))
                if rt_dev > GRAD_SIFT_MAX_RT_ADJUST_PX:
                    print(f"❌ [RT邊界] 匹配點偏離 RT/平面預測 {rt_dev:.1f}px (> {GRAD_SIFT_MAX_RT_ADJUST_PX:.0f}px)，判定匹配失敗。")
                    rt_bound_reject_reason = f"匹配點偏離RT/平面預測 {rt_dev:.0f}px"
                    m_pt = None
            if (m_pt is not None and snap_view_state['ecc']
                    and not snap_view_state.get('region_sift', False)):
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
            if (m_pt is not None and snap_view_state['enforce_epi'] and method != "ArUco"
                    and not snap_view_state.get('region_sift', False)):
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                      m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])

        # 2. 分流計算三維點
        d_val, p3d_val, p3d_w_val, fail_reason = None, None, None, ""
        p3d = None
        
        if (m_pt is not None and method != "ArUco" and manual_match_pt is None
                and not snap_view_state.get('region_sift', False)):
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
        _measure_seconds = time.perf_counter() - t_cm_start
        t_prof['其他準備、幾何檢查與結果處理'] = max(
            0.0, _measure_seconds - sum(t_prof.values()))
        print(f"⏱️ [F{cand.get('idx')}] 匹配+三角化細分 {_measure_seconds * 1000.0:.1f} ms")
        for _stage_name, _stage_seconds in t_prof.items():
            _stage_pct = 100.0 * _stage_seconds / _measure_seconds if _measure_seconds > 0 else 0.0
            print(f"   - {_stage_name}: {_stage_seconds * 1000.0:.1f} ms ({_stage_pct:.1f}%)")
        return {'pt': m_pt, 'pt_raw': m_pt_raw, 'p3d': p3d_val, 'p3d_w': p3d_w_val, 'd': d_val, 'depth': depth_z, 'error': reproj_err, 'method': method, 'neighbors': neighbors,
                'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_groups': g_groups,
                'g_refA': g_refA, 'g_refB': g_refB,
                'g_refA_groups': g_refA_groups, 'g_refB_groups': g_refB_groups,
                'g_kptsB': g_kptsB, 'g_rect': g_rect,
                'grad_descriptor_audit': grad_descriptor_audit,
                'region_debug': region_debug,
                'region_diagnostic_config': region_config,
                'fail_reason': fail_reason, 'u': u, 'v': v, 'trajectory': trajectory_res,
                'd_epi': d_epi, 'zncc_score': zncc_score, 'masked_score': masked_score, 'confidence_score': confidence_score,
                # Read-only references for the lower debug panels.  These are
                # the exact grayscale arrays passed into Grad-SIFT/ORB.
                'debug_left_gray': snap_imgA_gray,
                'debug_right_gray': cand['gray']}

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

    def do_measure(u, v, manual_match_pt=None, *, accuracy_batch=False):
        """同步計算並立即更新 UI"""
        if height_profile is not None and height_profile.enabled and not accuracy_batch:
            return None  # Profile owns measurement until the user exits its mode.
        if block_accuracy_state['mode'] != 'idle' and not accuracy_batch:
            print('[Block Accuracy] Finish/cancel the block run before manual measurement')
            return None
        if view_state.get('show_rt_warp_view', False):
            print(
                "⚠️ [RT Warp View] 目前為純顯示模式，深度計算已停用；"
                "請先按 RT Warp: On 關閉此模式。")
            depth_text.set_text(
                "RT+Plane Warp View (display only)\n"
                "Depth calculation is disabled")
            request_blit_refresh()
            return None
        nonlocal last_click, locked_L, locked_R; last_click = (u, v)
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

        # Debug-only sparse block audit.  It consumes the already accepted
        # support pairs and never feeds a decision back into matching/fusion.
        if snap_vs.get('show_metric_blocks', False):
            try:
                block_pts_a = res.get('g_ptsA')
                block_pts_b = res.get('g_ptsB')
                block_pts_a = (
                    np.empty((0, 2), dtype=np.float32) if block_pts_a is None
                    else np.asarray(block_pts_a, dtype=np.float32).reshape(-1, 2))
                block_pts_b = (
                    np.empty((0, 2), dtype=np.float32) if block_pts_b is None
                    else np.asarray(block_pts_b, dtype=np.float32).reshape(-1, 2))
                block_count = min(len(block_pts_a), len(block_pts_b))
                block_pts_a = block_pts_a[:block_count]
                block_pts_b = block_pts_b[:block_count]
                block_groups = res.get('g_groups')
                if (block_groups is None
                        or len(np.asarray(block_groups).reshape(-1)) != block_count):
                    block_groups = np.asarray(['high'] * block_count, dtype=object)
                else:
                    block_groups = np.asarray(
                        block_groups, dtype=object).reshape(-1)[:block_count]
                res['debug_metric_block_audit'] = build_metric_block_audit(
                    res, u, v, block_pts_a, block_pts_b, block_groups)
                print_metric_block_audit(res['debug_metric_block_audit'])
            except Exception as exc:
                print(f"   [5mm Block Debug] unavailable: {exc}")
                res['debug_metric_block_audit'] = {
                    'available': False, 'reason': str(exc),
                    'log': {
                        'grid_source': 'error', 'selected_block': None,
                        'status': 'UNAVAILABLE', 'read_only': True,
                        'block_filter_applied': False,
                    },
                }

        # Persist the exact plane source used by the height readout.  This is
        # metadata only and does not feed back into any matching calculation.
        res['shared_pattern_plane_diag'] = shared_height_plane_diag
        res['height_reference_source'] = None
        res['height_reference_signed_mm'] = None
        res['height_display_mm'] = None
        if (custom_plane_fitted and res.get('p3d') is not None
                and custom_plane_n is not None and custom_plane_c is not None):
            _height_signed = float(np.dot(
                custom_plane_n, res['p3d'] - custom_plane_c))
            res['height_reference_source'] = 'Custom Plane'
            res['height_reference_signed_mm'] = _height_signed
            res['height_display_mm'] = _height_signed
        elif res.get('p3d') is not None:
            _height_n, _height_c, _height_source = get_selected_height_plane()
            if _height_n is not None and _height_c is not None:
                _height_p3d = (
                    res['p3d_best'] if res.get('p3d_best') is not None
                    else res['p3d'])
                _height_signed = float(np.dot(
                    _height_n, _height_p3d - _height_c))
                res['height_reference_source'] = _height_source
                res['height_reference_signed_mm'] = _height_signed
                res['height_display_mm'] = (
                    _height_signed - DEFAULT_WOUND_HEIGHT_OFFSET_MM)
                
        measure_results[current_cand['idx']] = res
        all_d = [r['d'] for r in measure_results.values() if r['d'] is not None]
        summary = [f"F{current_cand['idx']}: {res['d']:.1f}" if res['d'] is not None else f"F{current_cand['idx']}: N/A"]
        
        # 存出數據至 txt 檔案
        if not accuracy_batch:
            save_measurement_to_txt(
                VIDEO_PATH, res, current_cand, wound_z_offset,
                custom_plane_n, custom_plane_c, custom_plane_fitted, MEASURE_MODE
            )
        click_timer.stage("結果整理+數據存檔")
        
        if not accuracy_batch:
            apply_measure_result(res, np.mean(all_d) if all_d else None, summary)
        click_timer.stage("UI 更新繪製")
        click_timer.report()
        return res


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

    # ------------------------------------------------------------------
    # Debug-only 5x5 mm physical block audit
    # ------------------------------------------------------------------
    def clear_metric_block_debug_artists():
        for artist in debug_metric_block_artists:
            try:
                artist.remove()
            except Exception:
                pass
        debug_metric_block_artists.clear()

    def get_metric_grid_frame():
        """Return origin/x/y/normal for a physical grid in left-camera mm."""
        pose = None
        valid_poses = video_data.get('valid_poses', {})
        try:
            pose = valid_poses.get(int(locked_L_idx))
        except (TypeError, ValueError, AttributeError):
            pose = None
        if pose is not None and len(pose) == 2:
            try:
                R_marker = np.asarray(pose[0], dtype=np.float64).reshape(3, 3)
                origin = np.asarray(pose[1], dtype=np.float64).reshape(3)
                x_axis = R_marker[:, 0].copy()
                y_axis = R_marker[:, 1].copy()
                if (np.all(np.isfinite(R_marker))
                        and np.all(np.isfinite(origin))):
                    x_norm = float(np.linalg.norm(x_axis))
                    if x_norm > 1e-9:
                        x_axis /= x_norm
                        y_axis -= x_axis * float(np.dot(x_axis, y_axis))
                        y_norm = float(np.linalg.norm(y_axis))
                        if y_norm > 1e-9:
                            y_axis /= y_norm
                            normal = np.cross(x_axis, y_axis)
                            n_norm = float(np.linalg.norm(normal))
                            if n_norm > 1e-9:
                                normal /= n_norm
                                if float(np.dot(normal, R_marker[:, 2])) < 0.0:
                                    y_axis = -y_axis
                                    normal = -normal
                                return {
                                    'origin': origin, 'x_axis': x_axis,
                                    'y_axis': y_axis, 'normal': normal,
                                    'source': 'marker-pose',
                                }
            except (TypeError, ValueError):
                pass

        # Pose axes are preferable because they keep block IDs stable.  If the
        # temporal anchor pose is unavailable, retain metric scale with an
        # arbitrary but deterministic basis on the existing reference plane.
        plane_n = current_cand.get('plane_n')
        plane_c = current_cand.get('plane_c')
        if plane_n is None or plane_c is None:
            return None
        try:
            normal = np.asarray(plane_n, dtype=np.float64).reshape(3)
            origin = np.asarray(plane_c, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return None
        n_norm = float(np.linalg.norm(normal))
        if (n_norm <= 1e-9 or not np.all(np.isfinite(normal))
                or not np.all(np.isfinite(origin))):
            return None
        normal /= n_norm
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= normal * float(np.dot(normal, x_axis))
        if float(np.linalg.norm(x_axis)) <= 1e-6:
            x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis -= normal * float(np.dot(normal, x_axis))
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
        y_axis = np.cross(normal, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
        return {
            'origin': origin, 'x_axis': x_axis,
            'y_axis': y_axis, 'normal': normal,
            'source': 'plane-basis-fallback',
        }

    def metric_pixel_to_grid_xy(point_uv, grid_frame):
        """Intersect an undistorted left-image ray with the metric grid plane."""
        try:
            point_uv = np.asarray(point_uv, dtype=np.float64).reshape(2)
            ray = np.linalg.inv(np.asarray(KL, dtype=np.float64)) @ np.array(
                [point_uv[0], point_uv[1], 1.0], dtype=np.float64)
            normal = grid_frame['normal']
            denom = float(np.dot(normal, ray))
            if abs(denom) <= 1e-9:
                return None, None
            scale = float(np.dot(normal, grid_frame['origin']) / denom)
            if not np.isfinite(scale) or scale <= 0.0:
                return None, None
            point_3d = ray * scale
            delta = point_3d - grid_frame['origin']
            xy = np.array([
                np.dot(delta, grid_frame['x_axis']),
                np.dot(delta, grid_frame['y_axis']),
            ], dtype=np.float64)
            return xy, point_3d
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None, None

    def metric_grid_xy_to_left_3d(xy_points, grid_frame):
        xy_points = np.asarray(xy_points, dtype=np.float64).reshape(-1, 2)
        return (
            grid_frame['origin'][None, :]
            + xy_points[:, 0:1] * grid_frame['x_axis'][None, :]
            + xy_points[:, 1:2] * grid_frame['y_axis'][None, :]
        )

    def project_metric_points(points_left_3d, cand, right=False):
        points = np.asarray(points_left_3d, dtype=np.float64).reshape(-1, 3)
        K_proj = np.asarray(KL, dtype=np.float64)
        if right:
            R_rel = np.asarray(cand['R_rel'], dtype=np.float64).reshape(3, 3)
            t_rel = np.asarray(cand['t_rel'], dtype=np.float64).reshape(3)
            points = (R_rel @ points.T).T + t_rel[None, :]
            K_proj = np.asarray(cand['K_R'], dtype=np.float64).reshape(3, 3)
        if len(points) == 0 or np.any(points[:, 2] <= 1e-6):
            return None
        projected_h = (K_proj @ points.T).T
        if np.any(np.abs(projected_h[:, 2]) <= 1e-9):
            return None
        projected = projected_h[:, :2] / projected_h[:, 2:3]
        return projected if np.all(np.isfinite(projected)) else None

    def summarize_metric_height_samples(samples, tolerance_mm):
        if not samples:
            return {
                'count': 0, 'high_count': 0, 'mid_count': 0,
                'median': None, 'mad': None, 'p10': None, 'p90': None,
                'span': None, 'status': 'INSUFFICIENT',
            }
        heights = np.asarray([sample['height'] for sample in samples], dtype=np.float64)
        median = float(np.median(heights))
        mad = float(np.median(np.abs(heights - median)))
        p10, p90 = np.percentile(heights, [10.0, 90.0])
        span = float(p90 - p10)
        if len(samples) < DEBUG_METRIC_BLOCK_MIN_POINTS:
            status = 'INSUFFICIENT'
        elif span <= tolerance_mm:
            status = 'STABLE'
        elif span <= 2.0 * tolerance_mm:
            status = 'CAUTION'
        else:
            status = 'INCONSISTENT'
        return {
            'count': int(len(samples)),
            'high_count': int(sum(sample['group'] != 'mid' for sample in samples)),
            'mid_count': int(sum(sample['group'] == 'mid' for sample in samples)),
            'median': median, 'mad': mad,
            'p10': float(p10), 'p90': float(p90),
            'span': span, 'status': status,
        }

    def build_metric_block_audit(res, u, v, pts_a, pts_b, pair_groups):
        """Build a read-only sparse-depth audit from final matcher supports."""
        grid_frame = get_metric_grid_frame()
        if grid_frame is None:
            return {
                'available': False,
                'reason': 'marker pose / reference plane unavailable',
                'log': {
                    'grid_source': 'unavailable', 'selected_block': None,
                    'status': 'UNAVAILABLE',
                },
            }
        click_xy, click_plane_point = metric_pixel_to_grid_xy((u, v), grid_frame)
        if click_xy is None:
            return {
                'available': False, 'reason': 'click ray is parallel to grid plane',
                'log': {
                    'grid_source': grid_frame['source'], 'selected_block': None,
                    'status': 'UNAVAILABLE',
                },
            }

        block_size = float(DEBUG_METRIC_BLOCK_SIZE_MM)
        selected_block = (
            int(np.floor(click_xy[0] / block_size)),
            int(np.floor(click_xy[1] / block_size)),
        )
        pts_a = np.asarray(pts_a, dtype=np.float32).reshape(-1, 2)
        pts_b = np.asarray(pts_b, dtype=np.float32).reshape(-1, 2)
        pair_count = min(len(pts_a), len(pts_b))
        pts_a, pts_b = pts_a[:pair_count], pts_b[:pair_count]
        pair_groups = np.asarray(pair_groups, dtype=object).reshape(-1)
        if len(pair_groups) != pair_count:
            pair_groups = np.asarray(['high'] * pair_count, dtype=object)

        p3d_reference = res.get('p3d_best')
        if p3d_reference is None:
            p3d_reference = res.get('p3d')
        if p3d_reference is not None:
            try:
                ref_z = float(np.asarray(p3d_reference).reshape(3)[2])
            except (TypeError, ValueError):
                ref_z = float(click_plane_point[2])
        else:
            ref_z = float(click_plane_point[2])
        baseline = float(np.linalg.norm(
            np.asarray(current_cand['t_rel'], dtype=np.float64).reshape(3)))
        focal = 0.5 * (float(KL[0, 0]) + float(KL[1, 1]))
        sigma_z = (
            (ref_z * ref_z / (focal * baseline))
            * float(DEBUG_METRIC_BLOCK_SIGMA_D_PX)
            if ref_z > 0.0 and focal > 0.0 and baseline > 1e-6 else 0.0)
        tolerance_mm = max(
            float(DEBUG_METRIC_BLOCK_BASE_TOL_MM),
            float(DEBUG_METRIC_BLOCK_SIGMA_MULT) * sigma_z)

        samples = []
        support_count_centered = 0
        half_block = block_size * 0.5
        R_rel = np.asarray(current_cand['R_rel'], dtype=np.float64).reshape(3, 3)
        t_rel = np.asarray(current_cand['t_rel'], dtype=np.float64).reshape(3)
        for index in range(pair_count):
            support_xy, _ = metric_pixel_to_grid_xy(pts_a[index], grid_frame)
            if support_xy is None:
                continue
            in_centered_window = bool(
                abs(float(support_xy[0] - click_xy[0])) <= half_block
                and abs(float(support_xy[1] - click_xy[1])) <= half_block)
            if in_centered_window:
                support_count_centered += 1
            try:
                support_3d = triangulate_point_3d(
                    pts_a[index], pts_b[index], KL, current_cand['K_R'],
                    current_cand['R_rel'], current_cand['t_rel'],
                    F=current_cand.get('F'))
            except Exception:
                continue
            support_3d = np.asarray(support_3d, dtype=np.float64).reshape(3)
            support_right_3d = R_rel @ support_3d + t_rel
            if (not np.all(np.isfinite(support_3d))
                    or support_3d[2] <= 0.0
                    or support_3d[2] > MAX_DEPTH_MM
                    or support_right_3d[2] <= 0.0):
                continue
            proj_left = project_metric_points([support_3d], current_cand, right=False)
            proj_right = project_metric_points([support_3d], current_cand, right=True)
            if proj_left is None or proj_right is None:
                continue
            reproj_rms = float(np.sqrt(0.5 * (
                np.sum((proj_left[0] - pts_a[index]) ** 2)
                + np.sum((proj_right[0] - pts_b[index]) ** 2))))
            if reproj_rms > DEBUG_METRIC_BLOCK_MAX_REPROJ_PX:
                continue
            cell = (
                int(np.floor(support_xy[0] / block_size)),
                int(np.floor(support_xy[1] / block_size)),
            )
            samples.append({
                'index': int(index), 'cell': cell,
                'xy': support_xy, 'p3d': support_3d,
                'height': float(np.dot(
                    grid_frame['normal'], support_3d - grid_frame['origin'])),
                'group': str(pair_groups[index]),
                'reproj_rms': reproj_rms,
                'in_centered_window': in_centered_window,
            })

        samples_by_cell = {}
        for sample in samples:
            samples_by_cell.setdefault(sample['cell'], []).append(sample)
        block_summaries = {
            cell: summarize_metric_height_samples(cell_samples, tolerance_mm)
            for cell, cell_samples in samples_by_cell.items()
        }
        centered_samples = [
            sample for sample in samples if sample['in_centered_window']]
        centered_summary = summarize_metric_height_samples(
            centered_samples, tolerance_mm)
        selected_summary = block_summaries.get(
            selected_block,
            summarize_metric_height_samples([], tolerance_mm))

        a_height = None
        a_delta = None
        if p3d_reference is not None:
            try:
                p3d_reference = np.asarray(p3d_reference, dtype=np.float64).reshape(3)
                if np.all(np.isfinite(p3d_reference)):
                    a_height = float(np.dot(
                        grid_frame['normal'],
                        p3d_reference - grid_frame['origin']))
            except (TypeError, ValueError):
                a_height = None
        if a_height is not None and centered_summary['median'] is not None:
            a_delta = abs(a_height - centered_summary['median'])
        status = centered_summary['status']
        if (status != 'INSUFFICIENT' and a_delta is not None
                and a_delta > 2.0 * tolerance_mm):
            status = 'A_OUTLIER'

        cells = []
        radius = int(DEBUG_METRIC_BLOCK_GRID_RADIUS)
        for cell_y in range(selected_block[1] - radius, selected_block[1] + radius + 1):
            for cell_x in range(selected_block[0] - radius, selected_block[0] + radius + 1):
                x0, y0 = cell_x * block_size, cell_y * block_size
                corners_xy = np.array([
                    [x0, y0], [x0 + block_size, y0],
                    [x0 + block_size, y0 + block_size], [x0, y0 + block_size],
                ], dtype=np.float64)
                corners_3d = metric_grid_xy_to_left_3d(corners_xy, grid_frame)
                left_poly = project_metric_points(corners_3d, current_cand, right=False)
                right_poly = project_metric_points(corners_3d, current_cand, right=True)
                if left_poly is None or right_poly is None:
                    continue
                cell_key = (cell_x, cell_y)
                cells.append({
                    'cell': cell_key, 'left': left_poly, 'right': right_poly,
                    'summary': block_summaries.get(
                        cell_key, summarize_metric_height_samples([], tolerance_mm)),
                    'selected': cell_key == selected_block,
                })

        centered_xy = np.array([
            [click_xy[0] - half_block, click_xy[1] - half_block],
            [click_xy[0] + half_block, click_xy[1] - half_block],
            [click_xy[0] + half_block, click_xy[1] + half_block],
            [click_xy[0] - half_block, click_xy[1] + half_block],
        ], dtype=np.float64)
        centered_3d = metric_grid_xy_to_left_3d(centered_xy, grid_frame)
        centered_left = project_metric_points(centered_3d, current_cand, right=False)
        centered_right = project_metric_points(centered_3d, current_cand, right=True)

        selected_cell_entry = next(
            (cell for cell in cells if cell['selected']), None)
        projected_size_left = None
        projected_size_right = None
        if selected_cell_entry is not None:
            def polygon_metric_size(poly):
                widths = [
                    np.linalg.norm(poly[1] - poly[0]),
                    np.linalg.norm(poly[2] - poly[3]),
                ]
                heights = [
                    np.linalg.norm(poly[3] - poly[0]),
                    np.linalg.norm(poly[2] - poly[1]),
                ]
                return float(np.mean(widths)), float(np.mean(heights))
            projected_size_left = polygon_metric_size(selected_cell_entry['left'])
            projected_size_right = polygon_metric_size(selected_cell_entry['right'])

        def rounded(value, digits=3):
            return None if value is None else round(float(value), digits)

        log_data = {
            'read_only': True,
            'block_filter_applied': False,
            'block_size_mm': block_size,
            'grid_source': grid_frame['source'],
            'selected_block': [int(selected_block[0]), int(selected_block[1])],
            'click_grid_xy_mm': [rounded(click_xy[0]), rounded(click_xy[1])],
            'projected_size_left_px': (
                None if projected_size_left is None else
                [rounded(projected_size_left[0], 2), rounded(projected_size_left[1], 2)]),
            'projected_size_right_px': (
                None if projected_size_right is None else
                [rounded(projected_size_right[0], 2), rounded(projected_size_right[1], 2)]),
            'support_count': int(support_count_centered),
            'valid_3d_count': int(centered_summary['count']),
            'high_count': int(centered_summary['high_count']),
            'mid_count': int(centered_summary['mid_count']),
            'median_height_mm': rounded(centered_summary['median']),
            'mad_mm': rounded(centered_summary['mad']),
            'robust_span_mm': rounded(centered_summary['span']),
            'tolerance_mm': rounded(tolerance_mm),
            'sigma_z_mm': rounded(sigma_z),
            'a_height_mm': rounded(a_height),
            'a_delta_mm': rounded(a_delta),
            'status': status,
        }
        return {
            'available': True, 'grid_frame': grid_frame,
            'selected_block': selected_block,
            'click_xy': click_xy, 'cells': cells,
            'centered_left': centered_left, 'centered_right': centered_right,
            'centered_summary': centered_summary,
            'selected_summary': selected_summary,
            'tolerance_mm': tolerance_mm, 'a_delta': a_delta,
            'status': status, 'samples': samples, 'log': log_data,
        }

    def print_metric_block_audit(audit):
        log_data = audit.get('log', {}) if isinstance(audit, dict) else {}
        if not audit or not audit.get('available'):
            print(
                f"   [5mm Block Debug] unavailable: "
                f"{audit.get('reason', 'unknown') if isinstance(audit, dict) else 'unknown'}")
            return
        print(
            f"   [5mm Block Debug] source={log_data.get('grid_source')} | "
            f"block={log_data.get('selected_block')} | "
            f"L_size={log_data.get('projected_size_left_px')}px | "
            f"R_size={log_data.get('projected_size_right_px')}px")
        print(
            f"   [5mm Block Debug A-window] support={log_data.get('support_count')} | "
            f"valid3D={log_data.get('valid_3d_count')} | "
            f"H/M={log_data.get('high_count')}/{log_data.get('mid_count')} | "
            f"median={log_data.get('median_height_mm')}mm | "
            f"MAD={log_data.get('mad_mm')}mm | "
            f"P90-P10={log_data.get('robust_span_mm')}mm | "
            f"tol={log_data.get('tolerance_mm')}mm | "
            f"A_delta={log_data.get('a_delta_mm')}mm | "
            f"status={log_data.get('status')} | read_only=True")

    def render_metric_block_audit(audit):
        clear_metric_block_debug_artists()
        if (not view_state.get('show_metric_blocks', False)
                or not isinstance(audit, dict) or not audit.get('available')):
            return
        status_colors = {
            'STABLE': '#00DD66', 'CAUTION': '#FFCC00',
            'INCONSISTENT': '#FF3333', 'A_OUTLIER': '#FF3333',
            'INSUFFICIENT': '#888888',
        }
        for cell in audit.get('cells', []):
            cell_status = cell['summary'].get('status', 'INSUFFICIENT')
            color = status_colors.get(cell_status, '#888888')
            for axis, key in ((ax_debug_A, 'left'), (ax_debug_B, 'right')):
                patch = Polygon(
                    cell[key], closed=True,
                    facecolor=color, edgecolor=color,
                    alpha=0.13, linewidth=0.9, zorder=4)
                axis.add_patch(patch)
                debug_metric_block_artists.append(patch)
                if cell.get('selected'):
                    outline = Polygon(
                        cell[key], closed=True, fill=False,
                        edgecolor='white', linewidth=1.8, zorder=7)
                    axis.add_patch(outline)
                    debug_metric_block_artists.append(outline)
        for axis, poly in (
                (ax_debug_A, audit.get('centered_left')),
                (ax_debug_B, audit.get('centered_right'))):
            if poly is not None:
                window_outline = Polygon(
                    poly, closed=True, fill=False, edgecolor='#00FFFF',
                    linestyle=':', linewidth=1.5, zorder=7)
                axis.add_patch(window_outline)
                debug_metric_block_artists.append(window_outline)
        selected = next(
            (cell for cell in audit.get('cells', []) if cell.get('selected')),
            None)
        if selected is not None:
            center_left = np.mean(selected['left'], axis=0)
            label = ax_debug_A.text(
                center_left[0], center_left[1],
                f"5mm {audit.get('status', 'N/A')}",
                color='white', fontsize=6.5, fontweight='bold',
                ha='center', va='center', zorder=8,
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=1.0))
            debug_metric_block_artists.append(label)

    def ensure_metric_block_audit(res, u, v, pts_a, pts_b, pair_groups,
                                  announce=False):
        audit = res.get('debug_metric_block_audit')
        if not isinstance(audit, dict):
            audit = build_metric_block_audit(
                res, u, v, pts_a, pts_b, pair_groups)
            res['debug_metric_block_audit'] = audit
            announce = True
        debug_metric_block_state['last_audit'] = audit
        if announce:
            print_metric_block_audit(audit)
        render_metric_block_audit(audit)
        return audit

    def select_grad_descriptor_audit(index=None, screen_xy=None,
                                     announce=False, refresh=False):
        """Select one audited left reference and expose its Top-1/Top-2."""
        empty = np.empty((0, 2), dtype=np.float32)
        records = debug_audit_state.get('records', [])

        if not records:
            for artist in debug_audit_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            debug_audit_artists.clear()
            debug_audit_state['selected_index'] = None
            dbg_audit_selected_A.set_offsets(empty)
            dbg_audit_top1_B.set_offsets(empty)
            dbg_audit_top2_B.set_offsets(empty)
            dbg_audit_local_seed_B.set_offsets(empty)
            dbg_info_A.set_text(
                debug_audit_state.get('base_left_text', '')
                + "\nGlobal KNN audit unavailable for this result")
            dbg_info_B.set_text(debug_audit_state.get('base_right_text', ''))
            if refresh:
                request_blit_refresh()
            return False

        if screen_xy is not None:
            left_points = np.asarray(
                [record['pL'] for record in records], dtype=np.float32)
            display_points = ax_debug_A.transData.transform(left_points)
            mouse_display = np.asarray(screen_xy, dtype=np.float64).reshape(2)
            display_distances = np.linalg.norm(
                display_points - mouse_display, axis=1)
            index = int(np.argmin(display_distances))
            # Prevent an accidental click on empty space from silently
            # selecting a distant, densely packed raw reference.
            if float(display_distances[index]) > 12.0:
                if announce:
                    print(
                        "   [Descriptor Audit] 請點下方左圖的 High/Mid 候選點 "
                        "(距游標 12 screen px 內)")
                return False
        elif index is None:
            index = 0

        index = int(index)
        if index < 0 or index >= len(records):
            return False
        for artist in debug_audit_artists:
            try:
                artist.remove()
            except Exception:
                pass
        debug_audit_artists.clear()
        record = records[index]
        debug_audit_state['selected_index'] = index

        p_left = np.asarray(record['pL'], dtype=np.float32).reshape(2)
        p_top1 = np.asarray(record['top1'], dtype=np.float32).reshape(2)
        p_top2 = (
            None if record.get('top2') is None
            else np.asarray(record['top2'], dtype=np.float32).reshape(2)
        )
        local_seed = (
            None if record.get('local_seed') is None
            else np.asarray(record['local_seed'], dtype=np.float32).reshape(2)
        )
        dbg_audit_selected_A.set_offsets([p_left])
        dbg_audit_top1_B.set_offsets([p_top1])
        dbg_audit_top2_B.set_offsets(
            [p_top2] if p_top2 is not None else empty)
        dbg_audit_local_seed_B.set_offsets(
            [local_seed] if local_seed is not None else empty)

        # The lower-right yellow line follows the reference selected in the
        # lower-left Debug panel.  It is this pL's own epipolar line, not the
        # line of the original measurement click.
        if current_cand.get('F') is not None:
            try:
                p0_selected, p1_selected = epipolar_line(
                    current_cand['F'],
                    (float(p_left[0]), float(p_left[1])), w)
                dbg_epi_line.set_data(
                    [p0_selected[0], p1_selected[0]],
                    [p0_selected[1], p1_selected[1]])
            except Exception:
                dbg_epi_line.set_data([], [])
        else:
            dbg_epi_line.set_data([], [])

        top1_line = ConnectionPatch(
            xyA=p_left, xyB=p_top1,
            coordsA='data', coordsB='data',
            axesA=ax_debug_A, axesB=ax_debug_B,
            color='#39FF14', lw=1.7, alpha=0.9, zorder=10)
        ax_debug_B.add_artist(top1_line)
        debug_audit_artists.append(top1_line)
        if p_top2 is not None:
            top2_line = ConnectionPatch(
                xyA=p_left, xyB=p_top2,
                coordsA='data', coordsB='data',
                axesA=ax_debug_A, axesB=ax_debug_B,
                color='#FF3333', lw=1.2, linestyle='--',
                alpha=0.8, zorder=9)
            ax_debug_B.add_artist(top2_line)
            debug_audit_artists.append(top2_line)
        accepted_pR_for_line = record.get('accepted_pR')
        if (record.get('support_source') == 'TOP2_GEO_RESCUE'
                and accepted_pR_for_line is not None):
            top2_selected_line = ConnectionPatch(
                xyA=p_left,
                xyB=np.asarray(accepted_pR_for_line, dtype=np.float32),
                coordsA='data', coordsB='data',
                axesA=ax_debug_A, axesB=ax_debug_B,
                color='#FFD700', lw=2.4, alpha=1.0, zorder=11)
            ax_debug_B.add_artist(top2_selected_line)
            debug_audit_artists.append(top2_selected_line)

        d2_text = (
            'N/A' if record.get('d2') is None
            else f"{record['d2']:.3f}")
        ratio_value = record.get('ratio')
        if ratio_value is None:
            ratio_text = 'N/A'
        elif np.isnan(ratio_value):
            ratio_text = 'undefined(0/0)'
        elif np.isinf(ratio_value):
            ratio_text = 'inf'
        else:
            ratio_text = f"{ratio_value:.4f}"
        gap_text = (
            'N/A' if record.get('gap') is None
            else f"{record['gap']:.3f}")
        spatial_text = (
            'N/A' if record.get('top12_spatial_distance') is None
            else f"{record['top12_spatial_distance']:.1f}px")
        epi_text = (
            'N/A' if record.get('epi_distance') is None
            else f"{record['epi_distance']:.2f}px/"
                 f"{'PASS' if record.get('pass_epi') else 'FAIL'}")
        seed_text = (
            'N/A' if record.get('seed_distance') is None
            else f"{record['seed_distance']:.1f}px/"
                 f"{'PASS' if record.get('pass_seed') else 'FAIL'}")
        abs_text = 'PASS' if record.get('pass_distance') else 'FAIL'
        if not record.get('pass_distance'):
            ratio_gate_text = 'NOT_REACHED'
            mutual_text = 'NOT_REACHED'
        elif not record.get('pass_ratio'):
            ratio_gate_text = 'FAIL'
            mutual_text = 'NOT_REACHED'
        else:
            ratio_gate_text = 'PASS'
            mutual_text = 'PASS' if record.get('pass_mutual') else 'FAIL'
        support_source = record.get('support_source', 'NOT_USED')
        accepted_pR = record.get('accepted_pR')
        if accepted_pR is not None:
            accepted_pR = np.asarray(
                accepted_pR, dtype=np.float32).reshape(2)
            support_text = (
                f"{support_source}, used R="
                f"({accepted_pR[0]:.1f},{accepted_pR[1]:.1f})")
        else:
            support_text = support_source
        if support_source == 'TOP2_GEO_RESCUE':
            geometry_role = (
                'ratio rejected; exact H(pL) selected Global Top1/Top2')
        elif record.get('guided_rescue'):
            geometry_role = (
                'global Top1 diagnostic; guided used the final R below')
        elif record.get('decision') == 'KNN_PASS':
            geometry_role = 'next gates'
        else:
            geometry_role = 'diagnostic only; pipeline stopped earlier'
        ratio_limit = float(record.get('ratio_limit', 0.78))
        match_threshold = float(record.get('match_threshold', 0.0))
        guided_detail = ''
        if support_source == 'TOP2_GEO_RESCUE':
            selected_rank = record.get('top2_geo_rank', 'N/A')
            geom_dist = record.get('top2_geo_geom_dist')
            geom_dist_text = (
                'N/A' if geom_dist is None else f'{float(geom_dist):.2f}px')
            selected_pt = record.get('top2_geo_selected_pR')
            selected_pt_text = (
                'N/A' if selected_pt is None
                else f'({selected_pt[0]:.1f},{selected_pt[1]:.1f})')
            guided_detail += (
                f"\nTOP2-GEO: selected {selected_rank}={selected_pt_text}; "
                f"Hdist={geom_dist_text}/"
                f"{TOP2_GEOMETRY_MAX_DIST_PX:g}px")
        elif record.get('top2_geo_reject_stage'):
            guided_detail += (
                f"\nTOP2-GEO rejected at "
                f"{record.get('top2_geo_reject_stage')}")
        if record.get('guided_triggered'):
            guided_decision = record.get('guided_decision', 'N/A')
            if record.get('guided_attempted'):
                guided_d1 = record.get('guided_d1')
                guided_d2 = record.get('guided_d2')
                guided_ratio = record.get('guided_ratio')
                guided_d1_text = (
                    'N/A' if guided_d1 is None else f'{guided_d1:.3f}')
                guided_d2_text = (
                    'N/A(single)' if guided_d2 is None
                    else f'{guided_d2:.3f}')
                if guided_ratio is None:
                    guided_ratio_text = 'N/A(single)'
                elif np.isnan(guided_ratio):
                    guided_ratio_text = 'undefined(0/0)'
                elif np.isinf(guided_ratio):
                    guided_ratio_text = 'inf'
                else:
                    guided_ratio_text = f'{guided_ratio:.4f}'
                guided_detail = (
                    f"\nGUIDED subset radius/epi candidates="
                    f"{record.get('guided_radius_candidate_count', 0)}/"
                    f"{record.get('guided_candidate_count', 0)}; "
                    f"d1={guided_d1_text}, d2={guided_d2_text}, "
                    f"ratio={guided_ratio_text} => {guided_decision}")
            else:
                guided_detail = f"\nGUIDED: {guided_decision}"

        dbg_info_A.set_text(
            debug_audit_state.get('base_left_text', '')
            + f"\nSelected global-KNN {record['group_label']}"
              f" #{record['audit_index']:03d} at "
              f"({p_left[0]:.1f},{p_left[1]:.1f}); "
              "click another raw point")
        dbg_info_B.set_text(
            debug_audit_state.get('base_right_text', '')
            + f"\nGLOBAL {record['group_label']} #{record['audit_index']:03d}: "
              f"d1={record['d1']:.3f}, d2={d2_text}, "
              f"d1/d2={ratio_text} (need <{ratio_limit:.2f}) "
              f"=> {record['decision']}"
            + f"\nabs d1<{match_threshold:g}: {abs_text} | "
              f"Lowe gate: {ratio_gate_text} | mutual gate: {mutual_text}"
            + f"\nd2-d1={gap_text} | Top1↔Top2={spatial_text}"
            + f"\n{geometry_role}: epi={epi_text} | seedΔ={seed_text}"
            + guided_detail
            + f"\nfinal interpolation support: {support_text}"
            + "\nlime circle=Top1, red square=Top2, magenta x=local seed; "
              "gold line=Top2-Geo selection; yellow line=selected pL epiline")

        if announce:
            print(
                f"   [Descriptor Audit select] "
                f"#{record['audit_index']:03d} {record['group_label']} "
                f"L=({p_left[0]:.1f},{p_left[1]:.1f}) "
                f"Top1=({p_top1[0]:.1f},{p_top1[1]:.1f}) "
                f"d1={record['d1']:.3f}, d2={d2_text}, "
                f"d1/d2={ratio_text} (limit < {ratio_limit:.2f}), "
                f"abs={abs_text}, Lowe={ratio_gate_text}, "
                f"mutual={mutual_text}, decision={record['decision']}, "
                f"support={support_source}")
            if support_source == 'TOP2_GEO_RESCUE' and accepted_pR is not None:
                print(
                    f"   [Descriptor Audit TOP2_GEO_RESCUE] "
                    f"#{record['audit_index']:03d} {record['group_label']} "
                    f"selected={record.get('top2_geo_rank')} "
                    f"R=({accepted_pR[0]:.1f},{accepted_pR[1]:.1f}), "
                    f"Hdist={float(record.get('top2_geo_geom_dist', float('nan'))):.2f}px; "
                    "this point participated in Grad-SIFT interpolation")
            if record.get('guided_rescue') and accepted_pR is not None:
                guided_top1 = record.get('guided_top1')
                guided_top2 = record.get('guided_top2')
                guided_d1 = record.get('guided_d1')
                guided_d2 = record.get('guided_d2')
                guided_ratio = record.get('guided_ratio')
                guided_top1_text = (
                    'N/A' if guided_top1 is None
                    else f"({guided_top1[0]:.1f},{guided_top1[1]:.1f})")
                guided_top2_text = (
                    'N/A(single)' if guided_top2 is None
                    else f"({guided_top2[0]:.1f},{guided_top2[1]:.1f})")
                guided_d1_text = (
                    'N/A' if guided_d1 is None else f'{guided_d1:.3f}')
                guided_d2_text = (
                    'N/A(single)' if guided_d2 is None
                    else f'{guided_d2:.3f}')
                if guided_ratio is None:
                    guided_ratio_text = 'N/A(single)'
                elif np.isnan(guided_ratio):
                    guided_ratio_text = 'undefined(0/0)'
                elif np.isinf(guided_ratio):
                    guided_ratio_text = 'inf'
                else:
                    guided_ratio_text = f'{guided_ratio:.4f}'
                print(
                    f"   [Descriptor Audit GUIDED_RESCUE] "
                    f"#{record['audit_index']:03d} {record['group_label']} "
                    f"L=({p_left[0]:.1f},{p_left[1]:.1f}) "
                    f"global={record['decision']} but final_support=YES, "
                    f"guided_R=({accepted_pR[0]:.1f},"
                    f"{accepted_pR[1]:.1f}); "
                    "this point participated in Grad-SIFT interpolation")
                print(
                    f"   [Descriptor Audit GUIDED subset] "
                    f"radius_candidates="
                    f"{record.get('guided_radius_candidate_count', 0)}, "
                    f"after_epi={record.get('guided_candidate_count', 0)}, "
                    f"Top1={guided_top1_text}, Top2={guided_top2_text}, "
                    f"d1={guided_d1_text}, d2={guided_d2_text}, "
                    f"d1/d2={guided_ratio_text} "
                    f"(limit < {float(record.get('guided_ratio_limit', 0.95)):.2f}), "
                    f"decision={record.get('guided_decision', 'N/A')}, "
                    f"Top1==final_R={record.get('guided_matches_final')}")
        if refresh:
            request_blit_refresh()
        return True

    def clear_region_sift_debug_artists():
        for artist in debug_region_artists:
            try:
                artist.remove()
            except Exception:
                pass
        debug_region_artists.clear()
        debug_region_state['records'] = []
        debug_region_state['selected_index'] = None
        debug_region_state['support_artists'] = []

    def select_region_sift_debug_point(index=None, screen_xy=None,
                                       announce=False, refresh=False):
        records = debug_region_state.get('records', [])
        if not records:
            return False
        selected = index
        if selected is None and screen_xy is not None:
            mouse_xy = np.asarray(screen_xy, dtype=np.float64).reshape(2)
            display_points = np.asarray([
                ax_debug_A.transData.transform(record['left_point'])
                for record in records
            ], dtype=np.float64)
            distances = np.linalg.norm(display_points - mouse_xy, axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) > 12.0:
                return False
            selected = nearest
        if selected is None or not 0 <= int(selected) < len(records):
            return False
        selected = int(selected)
        record = records[selected]
        debug_region_state['selected_index'] = selected
        point_left = np.asarray(record['left_point'], dtype=np.float32)
        dbg_audit_selected_A.set_offsets([point_left])

        label = 'P' if record.get('is_click') else f"#{selected + 1:02d}"
        cell_text = (
            f"({int(record['cell_row'])},{int(record['cell_col'])})")
        keep_text = 'KEEP in TrimK' if record.get('kept') else 'TRIM from TrimK'
        keep_text += ('; CellAll active' if record.get('balanced_score_active')
                      else '; CellAll disabled (weight=0)')
        point_warp = np.asarray(record['right_point_warp'], dtype=np.float32)
        point_right = np.asarray(record['right_point_original'], dtype=np.float32)
        for support_artist in debug_region_state.get('support_artists', []):
            support_artist.remove()
            if support_artist in debug_region_artists:
                debug_region_artists.remove(support_artist)
        support_artists = []
        for support_ax, support_point, radius_key in (
                (ax_debug_A, point_left, 'left_support_radius_px'),
                (ax_debug_B, point_warp, 'right_support_radius_px')):
            radius = float(record.get(radius_key, 0.0))
            support_box = Rectangle(
                (support_point[0] - radius, support_point[1] - radius),
                2.0 * radius, 2.0 * radius, fill=False,
                edgecolor='#FFFF66', linestyle=':', linewidth=1.0, zorder=6)
            support_ax.add_patch(support_box)
            support_artists.append(support_box)
        debug_region_artists.extend(support_artists)
        debug_region_state['support_artists'] = support_artists
        dbg_info_A.set_text(
            debug_region_state.get('base_left_text', '')
            + f"\nSelected {label}: cell={cell_text}, "
              f"level={record.get('gradient_label')}, "
              f"gradient={float(record.get('gradient_magnitude', 0.0)):.2f}"
            + f"\nL frame: size={float(record.get('left_scale_px', 0.0)):.1f}px, "
              f"angle={float(record.get('left_angle_deg', 0.0)):.1f}deg, "
              f"support={float(record.get('left_support_radius_px', 0.0)):.1f}px"
            + f"\nreliable scale/angle pair={record.get('scale_pair_reliable')}/"
              f"{record.get('angle_pair_reliable')}; dotted=full support bounds")
        dbg_info_B.set_text(
            debug_region_state.get('base_right_text', '')
            + f"\n{label}: L2={float(record['l2_distance']):.3f}, "
              f"rank={int(record['distance_rank'])}/{len(records)} => {keep_text}"
            + f"\nL=({point_left[0]:.1f},{point_left[1]:.1f}), "
              f"RW=({point_warp[0]:.1f},{point_warp[1]:.1f}), "
              f"R=({point_right[0]:.1f},{point_right[1]:.1f})"
            + f"\nR frame: size={float(record.get('right_scale_px', 0.0)):.1f}px, "
              f"angle={float(record.get('right_angle_deg', 0.0)):.1f}deg | "
              f"scale-ratio={float(record.get('scale_ratio', 0.0)):.2f}, "
              f"dAngle={float(record.get('angle_delta_deg', 0.0)):+.1f}deg"
            + f" | support={float(record.get('right_support_radius_px', 0.0)):.1f}px")
        if announce:
            print(
                f"   [Region-SIFT point] {label} cell={cell_text} "
                f"level={record.get('gradient_label')} "
                f"gradient={float(record.get('gradient_magnitude', 0.0)):.2f} "
                f"L-frame={float(record.get('left_scale_px', 0.0)):.1f}px/"
                f"{float(record.get('left_angle_deg', 0.0)):.1f}deg "
                f"R-frame={float(record.get('right_scale_px', 0.0)):.1f}px/"
                f"{float(record.get('right_angle_deg', 0.0)):.1f}deg "
                f"packedOctave L/R={record.get('left_octave')}/{record.get('right_octave')} "
                f"support L/R={record.get('left_support_radius_px')}/{record.get('right_support_radius_px')}px "
                f"reliableScale/Angle={record.get('scale_pair_reliable')}/{record.get('angle_pair_reliable')} "
                f"L2={float(record['l2_distance']):.3f} "
                f"rank={int(record['distance_rank'])}/{len(records)} "
                f"status={keep_text}")
        if refresh:
            request_blit_refresh()
        return True

    def update_region_sift_debug_views(res, u, v):
        """Draw the exact Region-SIFT inputs and the winning shared shift."""
        empty = np.empty((0, 2), dtype=np.float32)
        clear_region_sift_debug_artists()
        clear_metric_block_debug_artists()
        for artist in debug_pair_artists:
            try:
                artist.remove()
            except Exception:
                pass
        debug_pair_artists.clear()
        debug_homography_residual_artists.clear()
        for artist in debug_audit_artists:
            try:
                artist.remove()
            except Exception:
                pass
        debug_audit_artists.clear()
        debug_audit_state['records'] = []
        debug_audit_state['selected_index'] = None
        for scatter in (dbg_ref_high_A, dbg_ref_high_B, dbg_ref_mid_A,
                        dbg_ref_mid_B, dbg_inlier_high_A, dbg_inlier_high_B,
                        dbg_inlier_mid_A, dbg_inlier_mid_B, dbg_pred_B,
                        dbg_audit_top1_B, dbg_audit_top2_B,
                        dbg_audit_local_seed_B):
            scatter.set_offsets(empty)
        dbg_hull_line.set_data([], [])
        dbg_hull_line_B.set_data([], [])
        dbg_search_rect.set_visible(False)
        dbg_raw_B.set_offsets(empty)
        dbg_audit_selected_A.set_offsets(empty)

        ax_debug_A.set_title(
            'Region-SIFT Debug - Left 3x3 Sampling Grid', color='#8FD3FF',
            fontsize=10, fontweight='bold', pad=5)
        ax_debug_B.set_title(
            'Region-SIFT Debug - Warped Right / Epipolar Band', color='#FFD29A',
            fontsize=10, fontweight='bold', pad=5)
        debug_left_gray = res.get('debug_left_gray')
        if debug_left_gray is not None:
            im_debug_A.set_data(cv2.cvtColor(
                np.asarray(debug_left_gray), cv2.COLOR_GRAY2RGB))

        debug = res.get('region_debug')
        region_diagnostic_latest['result'] = res
        if not isinstance(debug, dict):
            if (block_accuracy_state['mode'] == 'idle' and not auto_calc_active
                    and not (height_profile is not None and height_profile.enabled)):
                region_diagnostics.show(res)
            debug_right_gray = res.get('debug_right_gray')
            if debug_right_gray is not None:
                im_debug_B.set_data(cv2.cvtColor(
                    np.asarray(debug_right_gray), cv2.COLOR_GRAY2RGB))
            dbg_click_A.set_offsets([[float(u), float(v)]])
            dbg_seed_B.set_offsets(empty)
            dbg_final_B.set_offsets(empty)
            dbg_epi_line.set_data([], [])
            fail_reason = res.get('fail_reason') or 'Region-SIFT unavailable'
            dbg_info_A.set_text('Region-SIFT did not produce debug point data')
            dbg_info_B.set_text(f"Region-SIFT failed: {fail_reason}")
            return

        warped_right = np.asarray(debug['warped_right_gray'])
        im_debug_B.set_data(cv2.cvtColor(warped_right, cv2.COLOR_GRAY2RGB))
        left_points = np.asarray(debug['left_points'], dtype=np.float32).reshape(-1, 2)
        right_points = np.asarray(debug['best_points_warp'], dtype=np.float32).reshape(-1, 2)
        records = list(debug.get('point_metadata', []))
        keep_mask = np.asarray(debug['keep_mask'], dtype=bool).reshape(-1)
        auto_count = int(debug.get('auto_point_count', max(0, len(left_points) - 1)))
        auto_left = left_points[:auto_count]
        auto_right = right_points[:auto_count]

        color_map = {
            'high': '#FF4D4D',
            'mid': '#FFD84D',
            'low': '#36C5F0',
        }
        auto_colors = [
            color_map.get(str(records[index].get('gradient_label')), '#B084FF')
            for index in range(auto_count)
        ]
        left_scatter = ax_debug_A.scatter(
            auto_left[:, 0], auto_left[:, 1], s=40, c=auto_colors,
            edgecolors='black', linewidths=0.45, zorder=7)
        debug_region_artists.append(left_scatter)
        kept_auto = keep_mask[:auto_count]
        if np.any(kept_auto):
            kept_scatter = ax_debug_B.scatter(
                auto_right[kept_auto, 0], auto_right[kept_auto, 1], s=44,
                facecolors='none', edgecolors='#00FF66', linewidths=1.4,
                zorder=8)
            debug_region_artists.append(kept_scatter)
        if np.any(~kept_auto):
            trim_scatter = ax_debug_B.scatter(
                auto_right[~kept_auto, 0], auto_right[~kept_auto, 1], s=34,
                c='#888888', marker='x', linewidths=1.1, zorder=7)
            debug_region_artists.append(trim_scatter)

        dbg_click_A.set_offsets([left_points[-1]])
        dbg_seed_B.set_offsets([np.asarray(debug['seed_on_line']).reshape(2)])
        dbg_final_B.set_offsets([np.asarray(debug['best_center_warp']).reshape(2)])

        polygon = np.asarray(debug['search_band_polygon_warp'], dtype=np.float32)
        band_patch = Polygon(
            polygon, closed=True, fill=False, edgecolor='#FF00FF',
            linewidth=1.3, linestyle='--', alpha=0.9, zorder=5)
        ax_debug_B.add_patch(band_patch)
        debug_region_artists.append(band_patch)

        line = np.asarray(debug['line_warp'], dtype=np.float64).reshape(3)
        image_h, image_w = warped_right.shape[:2]
        if abs(line[1]) >= abs(line[0]) and abs(line[1]) > 1e-9:
            line_x = np.array([0.0, float(image_w - 1)])
            line_y = -(line[0] * line_x + line[2]) / line[1]
        elif abs(line[0]) > 1e-9:
            line_y = np.array([0.0, float(image_h - 1)])
            line_x = -(line[1] * line_y + line[2]) / line[0]
        else:
            line_x = line_y = np.array([], dtype=np.float64)
        dbg_epi_line.set_data(line_x, line_y)

        x0, y0, roi_w, roi_h = [float(value) for value in debug['left_roi']]
        cfg = debug['config']
        for col in range(int(cfg.grid_cols) + 1):
            x = x0 + col * float(cfg.cell_width_px)
            grid_line, = ax_debug_A.plot(
                [x, x], [y0, y0 + roi_h], color='white', lw=0.55,
                alpha=0.55, zorder=4)
            debug_region_artists.append(grid_line)
        for row in range(int(cfg.grid_rows) + 1):
            y = y0 + row * float(cfg.cell_height_px)
            grid_line, = ax_debug_A.plot(
                [x0, x0 + roi_w], [y, y], color='white', lw=0.55,
                alpha=0.55, zorder=4)
            debug_region_artists.append(grid_line)

        for index in range(auto_count):
            color = '#00FF66' if keep_mask[index] else '#AAAAAA'
            label_a = ax_debug_A.text(
                left_points[index, 0] + 0.7, left_points[index, 1] - 0.7,
                str(index + 1), color=auto_colors[index], fontsize=6,
                fontweight='bold', zorder=9)
            label_b = ax_debug_B.text(
                right_points[index, 0] + 0.7, right_points[index, 1] - 0.7,
                str(index + 1), color=color, fontsize=6,
                fontweight='bold', zorder=9)
            connection = ConnectionPatch(
                xyA=left_points[index], xyB=right_points[index],
                coordsA='data', coordsB='data', axesA=ax_debug_A,
                axesB=ax_debug_B, color=color, lw=0.5, alpha=0.25,
                zorder=4)
            ax_debug_B.add_artist(connection)
            debug_region_artists.extend([label_a, label_b, connection])

        # Each short ray is the independently estimated SIFT frame: its
        # length follows keypoint size and its direction follows orientation.
        # The point locations themselves still differ only by one shared shift.
        left_sizes = np.asarray(
            debug['left_frame_sizes_px'], dtype=np.float32).reshape(-1)
        left_angles = np.asarray(
            debug['left_frame_angles_deg'], dtype=np.float32).reshape(-1)
        right_sizes = np.asarray(
            debug['right_frame_sizes_px'], dtype=np.float32).reshape(-1)
        right_angles = np.asarray(
            debug['right_frame_angles_deg'], dtype=np.float32).reshape(-1)
        for index in range(len(left_points)):
            angle_l = np.deg2rad(float(left_angles[index]))
            angle_r = np.deg2rad(float(right_angles[index]))
            length_l = max(2.0, 0.5 * float(left_sizes[index]))
            length_r = max(2.0, 0.5 * float(right_sizes[index]))
            end_l = left_points[index] + length_l * np.array(
                [np.cos(angle_l), np.sin(angle_l)], dtype=np.float32)
            end_r = right_points[index] + length_r * np.array(
                [np.cos(angle_r), np.sin(angle_r)], dtype=np.float32)
            color_l = 'white' if index == len(left_points) - 1 else auto_colors[index]
            color_r = '#00FF66' if keep_mask[index] else '#888888'
            frame_l, = ax_debug_A.plot(
                [left_points[index, 0], end_l[0]],
                [left_points[index, 1], end_l[1]],
                color=color_l, lw=1.0, alpha=0.9, zorder=8)
            frame_r, = ax_debug_B.plot(
                [right_points[index, 0], end_r[0]],
                [right_points[index, 1], end_r[1]],
                color=color_r, lw=1.0, alpha=0.9, zorder=8)
            debug_region_artists.extend([frame_l, frame_r])
        p_label_a = ax_debug_A.text(
            left_points[-1, 0] + 1.0, left_points[-1, 1] - 1.0, 'P',
            color='white', fontsize=8, fontweight='bold', zorder=10)
        p_label_b = ax_debug_B.text(
            right_points[-1, 0] + 1.0, right_points[-1, 1] - 1.0, "P'",
            color='#00FF66', fontsize=8, fontweight='bold', zorder=10)
        debug_region_artists.extend([p_label_a, p_label_b])

        pad = max(5.0, float(np.max(debug['left_frames']['support_radius_px'])))
        ax_debug_A.set_xlim(max(-0.5, x0 - pad), min(float(image_w) - 0.5, x0 + roi_w + pad))
        ax_debug_A.set_ylim(min(float(image_h) - 0.5, y0 + roi_h + pad), max(-0.5, y0 - pad))
        debug_zoom_home['A'] = (ax_debug_A.get_xlim(), ax_debug_A.get_ylim())
        right_radii = np.asarray(debug['right_frames']['support_radius_px']).reshape(-1, 1)
        right_cloud = np.vstack([right_points - right_radii,
                                 right_points + right_radii, polygon])
        rx0 = max(-0.5, float(np.min(right_cloud[:, 0])) - 7.0)
        rx1 = min(float(image_w) - 0.5, float(np.max(right_cloud[:, 0])) + 7.0)
        ry0 = max(-0.5, float(np.min(right_cloud[:, 1])) - 7.0)
        ry1 = min(float(image_h) - 0.5, float(np.max(right_cloud[:, 1])) + 7.0)
        ax_debug_B.set_xlim(rx0, rx1)
        ax_debug_B.set_ylim(ry1, ry0)
        debug_zoom_home['B'] = (ax_debug_B.get_xlim(), ax_debug_B.get_ylim())

        delta = np.asarray(debug['best_displacement_warp']).reshape(2)
        best_right = np.asarray(debug['best_center_right']).reshape(2)
        descriptor_shape = tuple(np.asarray(debug['left_descriptors']).shape)
        kept_stats = debug.get('kept_l2_stats', {})
        base_left_text = (
            f"Region-SIFT | grid={cfg.grid_rows}x{cfg.grid_cols}, "
            f"cell={cfg.cell_width_px}x{cfg.cell_height_px}px | "
            f"auto={auto_count}+P={debug['point_count']} | desc={descriptor_shape}\n"
            "each ray=size+angle; red=High, yellow=Mid, cyan=Low; click for details")
        base_right_text = (
            f"band={cfg.search_width_px:g}x{cfg.search_length_px:g}px, "
            f"step={cfg.search_across_step_px:g}x{cfg.search_along_step_px:g} | "
            f"candidates={debug['valid_candidate_count']}/{debug['candidate_count']}\n"
            f"bestG={debug['group_score']:.3f}, "
            f"relativeG={debug.get('relative_group_score', float('nan')):.3f}, "
            f"secondG={debug.get('second_group_score', float('nan')):.3f}, "
            f"margin={debug.get('group_score_margin', float('nan')):.3f}, "
            f"ObjRatio={debug.get('objective_score_ratio', float('nan')):.4f}\n"
            f"Trim{debug['keep_count']}={debug['trimmed_score']:.2f}, "
            f"CellAll={debug['balanced_score']:.2f} (w={cfg.group_balance_weight:.2f}), "
            f"Frame+={debug['frame_penalty']:.2f}, Total={debug['objective_score']:.2f}\n"
            f"KEEP={debug['keep_count']}/{debug['point_count']}, "
            f"L2 med/std/max={kept_stats.get('median', float('nan')):.1f}/"
            f"{kept_stats.get('std', float('nan')):.1f}/"
            f"{kept_stats.get('max', float('nan')):.1f}, "
            f"cells={debug.get('kept_cell_coverage', 0)}/"
            f"{debug.get('total_cell_count', 0)}\n"
            f"deltaW=({delta[0]:.1f},{delta[1]:.1f}) | "
            f"P' R=({best_right[0]:.1f},{best_right[1]:.1f}) | "
            f"{debug['elapsed_ms']:.0f}ms\n"
            + ("green=TrimK+CellAll, gray=CellAll"
               if cfg.group_balance_weight > 0 else
               "green=TrimK, gray=trimmed; CellAll diagnostic only (weight=0)")
            + "; flat frames skip consistency penalty")
        if res.get('fail_reason'):
            base_right_text += f"\nResult: {res['fail_reason']}"
        debug_region_state['records'] = records
        debug_region_state['base_left_text'] = base_left_text
        debug_region_state['base_right_text'] = base_right_text
        dbg_info_A.set_text(base_left_text)
        dbg_info_B.set_text(base_right_text)
        # Default to P so its descriptor contribution is visible immediately.
        select_region_sift_debug_point(index=len(records) - 1)

        if (block_accuracy_state['mode'] == 'idle' and not auto_calc_active
                and not (height_profile is not None and height_profile.enabled)):
            show_region_diagnostics()

    def show_region_diagnostics(event=None, *, result=None, display=True):
        res = result if result is not None else region_diagnostic_latest['result']
        if res is None:
            print('[Region-SIFT Debug] No candidate scores; first measure a Region-SIFT point.')
            return
        if not isinstance(res.get('region_debug'), dict):
            if display:
                region_diagnostics.show(res)
            return
        debug = res['region_debug']
        n, c, label = get_selected_height_plane()
        offset = DEFAULT_WOUND_HEIGHT_OFFSET_MM
        if custom_plane_fitted:
            n, c, label = custom_plane_n, custom_plane_c, 'Custom Plane'
            offset = 0.0
        debug['diagnostic_heights_mm'] = [
            float(np.dot(n, xyz-c)) - offset
            if xyz is not None and n is not None and c is not None else float('nan')
            for xyz in debug.get('diagnostic_candidate_p3ds', [None, None])]
        debug['diagnostic_plane_label'] = f'{label}; F{debug.get("diagnostic_frame_index")}; no temporal fusion'
        debug['diagnostic_height_plane'] = {
            'n': None if n is None else np.asarray(n).reshape(3).copy(),
            'c': None if c is None else np.asarray(c).reshape(3).copy(), 'offset': offset}
        if display:
            region_diagnostics.show(res)

    def update_grad_match_debug_views(res, u, v):
        """Update the lower zoomed views without changing any matching decision."""
        if view_state.get('region_sift', False):
            update_region_sift_debug_views(res, u, v)
            return
        clear_region_sift_debug_artists()
        ax_debug_A.set_title(
            'Grad-SIFT/ORB Debug - Left ROI (wheel zoom; double-click reset)',
            color='#8FD3FF', fontsize=10, fontweight='bold', pad=5)
        ax_debug_B.set_title(
            'Grad-SIFT/ORB Debug - Right ROI (wheel zoom; double-click reset)',
            color='#FFD29A', fontsize=10, fontweight='bold', pad=5)
        empty = np.empty((0, 2), dtype=np.float32)

        def as_points(value):
            if value is None:
                return empty
            arr = np.asarray(value, dtype=np.float32)
            if arr.size == 0:
                return empty
            return arr.reshape(-1, 2)

        def split_groups(points, labels):
            if len(points) == 0:
                return empty, empty
            if labels is None:
                return points, empty
            labels_arr = np.asarray(labels, dtype=object).reshape(-1)
            if len(labels_arr) != len(points):
                return points, empty
            mid_mask = labels_arr == "mid"
            return points[~mid_mask], points[mid_mask]

        debug_left_gray = res.get('debug_left_gray')
        debug_right_gray = res.get('debug_right_gray')
        if debug_left_gray is not None:
            debug_left_gray = np.asarray(debug_left_gray)
            if debug_left_gray.ndim == 2 and debug_left_gray.size:
                im_debug_A.set_data(
                    cv2.cvtColor(debug_left_gray, cv2.COLOR_GRAY2RGB))
        if debug_right_gray is not None:
            debug_right_gray = np.asarray(debug_right_gray)
            if debug_right_gray.ndim == 2 and debug_right_gray.size:
                im_debug_B.set_data(
                    cv2.cvtColor(debug_right_gray, cv2.COLOR_GRAY2RGB))

        # Remove click-specific ConnectionPatch/text/residual artists from the
        # previous measurement.  Persistent scatters are updated in place below.
        for artist in debug_pair_artists:
            try:
                artist.remove()
            except Exception:
                pass
        debug_pair_artists.clear()
        debug_homography_residual_artists.clear()

        ref_a = as_points(res.get('g_refA'))
        ref_b = as_points(res.get('g_refB'))
        ref_a_high, ref_a_mid = split_groups(ref_a, res.get('g_refA_groups'))
        ref_b_high, ref_b_mid = split_groups(ref_b, res.get('g_refB_groups'))
        dbg_ref_high_A.set_offsets(ref_a_high)
        dbg_ref_mid_A.set_offsets(ref_a_mid)
        dbg_ref_high_B.set_offsets(ref_b_high)
        dbg_ref_mid_B.set_offsets(ref_b_mid)

        pts_a = as_points(res.get('g_ptsA'))
        pts_b = as_points(res.get('g_ptsB'))
        pair_count = min(len(pts_a), len(pts_b))
        pts_a = pts_a[:pair_count]
        pts_b = pts_b[:pair_count]
        pair_groups = res.get('g_groups')
        if pair_groups is None or len(np.asarray(pair_groups).reshape(-1)) != pair_count:
            pair_groups = np.array(["high"] * pair_count, dtype=object)
        else:
            pair_groups = np.asarray(pair_groups, dtype=object).reshape(-1)[:pair_count]
        in_a_high, in_a_mid = split_groups(pts_a, pair_groups)
        in_b_high, in_b_mid = split_groups(pts_b, pair_groups)
        dbg_inlier_high_A.set_offsets(in_a_high)
        dbg_inlier_mid_A.set_offsets(in_a_mid)
        dbg_inlier_high_B.set_offsets(in_b_high)
        dbg_inlier_mid_B.set_offsets(in_b_mid)

        metric_block_audit = None
        if view_state.get('show_metric_blocks', False):
            try:
                metric_block_audit = ensure_metric_block_audit(
                    res, u, v, pts_a, pts_b, pair_groups,
                    announce=False)
            except Exception as exc:
                clear_metric_block_debug_artists()
                metric_block_audit = {
                    'available': False, 'reason': str(exc),
                    'log': {'status': 'UNAVAILABLE'},
                }
                print(f"   [5mm Block Debug] overlay unavailable: {exc}")
        else:
            clear_metric_block_debug_artists()

        click_pt = np.array([float(u), float(v)], dtype=np.float32)
        dbg_click_A.set_offsets([click_pt])

        seed_pt = None
        seed_method = "N/A"
        try:
            seed_value, seed_method = predict_right_seed_from_geometry(
                (float(u), float(v)), current_cand, KL)
            seed_pt = np.asarray(seed_value, dtype=np.float32).reshape(2)
        except Exception:
            seed_pt = None
        dbg_seed_B.set_offsets([seed_pt] if seed_pt is not None else empty)

        raw_pt = res.get('pt_raw')
        raw_pt = (None if raw_pt is None else
                  np.asarray(raw_pt, dtype=np.float32).reshape(2))
        final_pt = res.get('pt')
        final_pt = (None if final_pt is None else
                    np.asarray(final_pt, dtype=np.float32).reshape(2))
        dbg_raw_B.set_offsets([raw_pt] if raw_pt is not None else empty)
        dbg_final_B.set_offsets([final_pt] if final_pt is not None else empty)

        if current_cand.get('F') is not None:
            try:
                p0_dbg, p1_dbg = epipolar_line(
                    current_cand['F'], (float(u), float(v)), w)
                dbg_epi_line.set_data(
                    [p0_dbg[0], p1_dbg[0]], [p0_dbg[1], p1_dbg[1]])
            except Exception:
                dbg_epi_line.set_data([], [])
        else:
            dbg_epi_line.set_data([], [])

        rect = res.get('g_rect')
        if rect is not None:
            rx, ry, rw, rh = [float(value) for value in rect]
            dbg_search_rect.set_bounds(rx, ry, rw, rh)
            dbg_search_rect.set_visible(True)
        else:
            rx = ry = rw = rh = None
            dbg_search_rect.set_visible(False)

        # Show where the plane homography predicts every accepted left support
        # point.  The short magenta residual segment ends at the actual match.
        predicted_b = empty
        if pair_count > 0:
            try:
                plane_n = current_cand.get('plane_n')
                plane_c = current_cand.get('plane_c')
                if plane_n is not None and plane_c is not None:
                    plane_n = np.asarray(plane_n, dtype=np.float64).reshape(3)
                    plane_c = np.asarray(plane_c, dtype=np.float64).reshape(3)
                    d_plane = float(np.dot(plane_n, plane_c))
                    if abs(d_plane) > 1e-8:
                        H_ab = (
                            np.asarray(current_cand['K_R'], dtype=np.float64)
                            @ (
                                np.asarray(current_cand['R_rel'], dtype=np.float64).reshape(3, 3)
                                + np.asarray(current_cand['t_rel'], dtype=np.float64).reshape(3, 1)
                                @ plane_n.reshape(1, 3) / d_plane
                            )
                            @ np.linalg.inv(np.asarray(KL, dtype=np.float64))
                        )
                        pts_h = np.column_stack(
                            [pts_a.astype(np.float64), np.ones(pair_count)])
                        pred_h = (H_ab @ pts_h.T).T
                        valid_h = np.abs(pred_h[:, 2]) > 1e-9
                        predicted_b = np.full((pair_count, 2), np.nan, dtype=np.float32)
                        predicted_b[valid_h] = (
                            pred_h[valid_h, :2] / pred_h[valid_h, 2:3]
                        ).astype(np.float32)
            except Exception:
                predicted_b = empty
        valid_pred = (
            np.all(np.isfinite(predicted_b), axis=1)
            if len(predicted_b) == pair_count and pair_count > 0
            else np.zeros(pair_count, dtype=bool)
        )
        dbg_pred_B.set_offsets(
            predicted_b[valid_pred] if np.any(valid_pred) else empty)
        dbg_pred_B.set_visible(
            view_state.get('show_homography_residual', True))

        # Accepted correspondences are numbered consistently in both zoomed
        # panels.  Only the nearest 30 labels are drawn to prevent text clutter.
        label_ids = set()
        if pair_count:
            nearest_order = np.argsort(np.linalg.norm(pts_a - click_pt, axis=1))
            label_ids = set(int(index) for index in nearest_order[:30])
        for index, (point_a, point_b) in enumerate(zip(pts_a, pts_b)):
            is_mid = pair_groups[index] == "mid"
            color = '#FF8C00' if is_mid else '#00BFFF'
            connection = ConnectionPatch(
                xyA=point_a, xyB=point_b,
                coordsA="data", coordsB="data",
                axesA=ax_debug_A, axesB=ax_debug_B,
                color=color, lw=0.85, alpha=0.55, zorder=5)
            ax_debug_B.add_artist(connection)
            debug_pair_artists.append(connection)
            if index < len(valid_pred) and valid_pred[index]:
                residual_line, = ax_debug_B.plot(
                    [predicted_b[index, 0], point_b[0]],
                    [predicted_b[index, 1], point_b[1]],
                    color='#FF00FF', lw=0.75, alpha=0.65, zorder=5,
                    visible=view_state.get(
                        'show_homography_residual', True))
                debug_pair_artists.append(residual_line)
                debug_homography_residual_artists.append(residual_line)
            if index in label_ids:
                label_a = ax_debug_A.text(
                    point_a[0] + 1.2, point_a[1] - 1.2, str(index),
                    color=color, fontsize=7, fontweight='bold', zorder=8)
                label_b = ax_debug_B.text(
                    point_b[0] + 1.2, point_b[1] - 1.2, str(index),
                    color=color, fontsize=7, fontweight='bold', zorder=8)
                debug_pair_artists.extend([label_a, label_b])

        hull_state = "N/A"
        hull_area = 0.0
        if pair_count >= 3:
            try:
                # Keep the left-hull vertex indices so the lower-right panel
                # connects the exact corresponding matches in the same order.
                # An independent right convex hull could hide a crossed or
                # distorted correspondence polygon, which is useful evidence
                # when diagnosing a bad match.
                hull_indices = cv2.convexHull(
                    pts_a.astype(np.float32), returnPoints=False).reshape(-1)
                hull = pts_a[hull_indices].astype(np.float32)
                hull_area = float(cv2.contourArea(hull.astype(np.float32)))
                if len(hull) >= 3 and hull_area > 1e-6:
                    hull_closed = np.vstack([hull, hull[0]])
                    dbg_hull_line.set_data(hull_closed[:, 0], hull_closed[:, 1])
                    right_hull = pts_b[hull_indices].astype(np.float32)
                    right_hull_closed = np.vstack([right_hull, right_hull[0]])
                    dbg_hull_line_B.set_data(
                        right_hull_closed[:, 0], right_hull_closed[:, 1])
                    inside = cv2.pointPolygonTest(
                        hull.astype(np.float32), (float(u), float(v)), False) >= 0
                    hull_state = "inside" if inside else "OUTSIDE"
                else:
                    dbg_hull_line.set_data([], [])
                    dbg_hull_line_B.set_data([], [])
                    hull_state = "degenerate"
            except Exception:
                dbg_hull_line.set_data([], [])
                dbg_hull_line_B.set_data([], [])
                hull_state = "error"
        else:
            dbg_hull_line.set_data([], [])
            dbg_hull_line_B.set_data([], [])

        left_radius = max(float(LEFT_PATCH_SEARCH_RADIUS) + 7.0, 30.0)
        left_x0 = max(-0.5, float(u) - left_radius)
        left_x1 = min(float(w) - 0.5, float(u) + left_radius)
        left_y0 = max(-0.5, float(v) - left_radius)
        left_y1 = min(float(h) - 0.5, float(v) + left_radius)
        ax_debug_A.set_xlim(left_x0, left_x1)
        ax_debug_A.set_ylim(left_y1, left_y0)
        debug_zoom_home['A'] = (
            (left_x0, left_x1), (left_y1, left_y0))

        right_points = [point for point in (seed_pt, raw_pt, final_pt) if point is not None]
        if rect is not None:
            right_x0, right_x1 = rx, rx + rw
            right_y0, right_y1 = ry, ry + rh
        elif seed_pt is not None:
            right_x0 = float(seed_pt[0]) - RIGHT_PATCH_SEARCH_RADIUS
            right_x1 = float(seed_pt[0]) + RIGHT_PATCH_SEARCH_RADIUS
            right_y0 = float(seed_pt[1]) - RIGHT_PATCH_SEARCH_RADIUS
            right_y1 = float(seed_pt[1]) + RIGHT_PATCH_SEARCH_RADIUS
        elif final_pt is not None:
            right_x0 = float(final_pt[0]) - RIGHT_PATCH_SEARCH_RADIUS
            right_x1 = float(final_pt[0]) + RIGHT_PATCH_SEARCH_RADIUS
            right_y0 = float(final_pt[1]) - RIGHT_PATCH_SEARCH_RADIUS
            right_y1 = float(final_pt[1]) + RIGHT_PATCH_SEARCH_RADIUS
        else:
            right_x0, right_x1 = 0.0, float(w)
            right_y0, right_y1 = 0.0, float(h)
        if right_points:
            right_arr = np.asarray(right_points, dtype=np.float32).reshape(-1, 2)
            right_x0 = min(right_x0, float(np.min(right_arr[:, 0])) - 8.0)
            right_x1 = max(right_x1, float(np.max(right_arr[:, 0])) + 8.0)
            right_y0 = min(right_y0, float(np.min(right_arr[:, 1])) - 8.0)
            right_y1 = max(right_y1, float(np.max(right_arr[:, 1])) + 8.0)
        right_x0 = max(-0.5, right_x0)
        right_x1 = min(float(w) - 0.5, right_x1)
        right_y0 = max(-0.5, right_y0)
        right_y1 = min(float(h) - 0.5, right_y1)
        if right_x1 <= right_x0:
            right_x0, right_x1 = 0.0, float(w)
        if right_y1 <= right_y0:
            right_y0, right_y1 = 0.0, float(h)
        ax_debug_B.set_xlim(right_x0, right_x1)
        ax_debug_B.set_ylim(right_y1, right_y0)
        debug_zoom_home['B'] = (
            (right_x0, right_x1), (right_y1, right_y0))

        audit = res.get('grad_descriptor_audit')
        descriptor_name = (
            audit.get('descriptor_name')
            if isinstance(audit, dict) and audit.get('descriptor_name')
            else ("ORB/Hamming" if view_state.get('use_hamming', False)
                  else "Gray-SIFT/L2")
        )
        if (not isinstance(audit, dict)
                and not view_state.get('use_hamming', False)):
            if view_state.get('use_rgb_sift', False):
                descriptor_name = "RGB-SIFT/L2"
            elif view_state.get('use_opponent_sift', False):
                descriptor_name = "Opponent-SIFT/L2"
        raw_final_shift = (
            float(np.linalg.norm(final_pt - raw_pt))
            if raw_pt is not None and final_pt is not None else float('nan'))
        high_count = int(np.count_nonzero(pair_groups != "mid"))
        mid_count = int(np.count_nonzero(pair_groups == "mid"))
        base_left_text = (
            f"{descriptor_name} | refs H/M={len(ref_a_high)}/{len(ref_a_mid)} | "
            f"accepted H/M={high_count}/{mid_count}\n"
            f"support hull={hull_state}, area={hull_area:.1f}px² | "
            "cyan=High, orange=Mid")
        if isinstance(metric_block_audit, dict):
            block_log = metric_block_audit.get('log', {})
            if metric_block_audit.get('available'):
                base_left_text += (
                    f"\n5mm grid={block_log.get('grid_source')} | "
                    f"block={block_log.get('selected_block')} | "
                    f"Lpx={block_log.get('projected_size_left_px')}")
            else:
                base_left_text += "\n5mm grid unavailable"
        if isinstance(audit, dict):
            ratio_limit = float(audit.get('ratio_limit', 0.78))
            group_bits = []
            for group_key, short_name in (('high', 'H'), ('mid', 'M')):
                group_summary = audit.get('groups', {}).get(group_key, {})
                count = int(group_summary.get('record_count', 0))
                ratio_reached = int(group_summary.get(
                    'ratio_stage_reached', count))
                ratio_pass = int(group_summary.get(
                    'ratio_stage_pass',
                    group_summary.get('ratio_pass', 0)))
                median = group_summary.get('ratio_median')
                median_text = 'N/A' if median is None else f"{median:.3f}"
                d2_zero = int(group_summary.get('d2_zero', 0))
                support_count = int(group_summary.get(
                    'final_support_count', 0))
                guided_count = int(group_summary.get(
                    'guided_rescue_count', 0))
                top2_geo_count = int(group_summary.get(
                    'top2_geo_rescue_count', 0))
                group_bits.append(
                    f"{short_name} Lowe-stage<{ratio_limit:.2f}:"
                    f"{ratio_pass}/{ratio_reached}, med={median_text}, "
                    f"d2=0:{d2_zero}, support={support_count}"
                    f"(guided={guided_count}, top2geo={top2_geo_count})")
            if group_bits:
                base_left_text += "\n" + "\n".join(group_bits)
        raw_text = "None" if raw_pt is None else f"({raw_pt[0]:.1f},{raw_pt[1]:.1f})"
        final_text = "None" if final_pt is None else f"({final_pt[0]:.1f},{final_pt[1]:.1f})"
        shift_text = "N/A" if not np.isfinite(raw_final_shift) else f"{raw_final_shift:.2f}px"
        base_right_text = (
            f"seed={seed_method} | raw={raw_text} | final={final_text} | "
            f"raw→final={shift_text}\n"
            "purple +=seed/circles=H(pL), white diamond=raw, green x=final, "
            "green dashed=matched left hull")
        if isinstance(metric_block_audit, dict) and metric_block_audit.get('available'):
            block_log = metric_block_audit.get('log', {})
            base_right_text += (
                f"\n5mm A-window n={block_log.get('valid_3d_count')} "
                f"H/M={block_log.get('high_count')}/{block_log.get('mid_count')} | "
                f"med/MAD/span={block_log.get('median_height_mm')}/"
                f"{block_log.get('mad_mm')}/{block_log.get('robust_span_mm')}mm | "
                f"AΔ={block_log.get('a_delta_mm')} => {block_log.get('status')}")
        debug_audit_state['audit'] = audit
        debug_audit_state['records'] = (
            list(audit.get('records', []))
            if isinstance(audit, dict) else [])
        debug_audit_state['selected_index'] = None
        debug_audit_state['descriptor_name'] = descriptor_name
        debug_audit_state['base_left_text'] = base_left_text
        debug_audit_state['base_right_text'] = base_right_text
        default_index = (
            audit.get('default_index')
            if isinstance(audit, dict) else None)
        select_grad_descriptor_audit(index=default_index)

    
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
            elif res['p3d'] is not None:
                # 與平面同鏈: 高度用最優對的 p3d (平面即由最優對 RT 三角化)，誤差相消才成立
                _height_plane_n, _height_plane_c, _plane_label = get_selected_height_plane()
                if _height_plane_n is not None and _height_plane_c is not None:
                    _p3d_plane = res['p3d_best'] if res.get('p3d_best') is not None else res['p3d']
                    p_dist = float(np.dot(_height_plane_n, _p3d_plane - _height_plane_c))
                    if auto_calc_active:
                        plane_dist_history.append(p_dist)
                        display_wound_height = np.mean(plane_dist_history) - DEFAULT_WOUND_HEIGHT_OFFSET_MM
                    else:
                        plane_dist_history.clear()
                        display_wound_height = p_dist - DEFAULT_WOUND_HEIGHT_OFFSET_MM
                    p_dist_str = f"\nWound Height ({_plane_label}): {display_wound_height:.1f}mm"
            
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
            fail_reason = ui_failure_reason_english(res.get('fail_reason', 'No Valid Depth'))
            if fail_reason == "No Valid Depth":
                main_text = fail_reason
            else:
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
        update_grad_match_debug_views(res, u, v)
        depth_text.set_text(main_text)
        request_blit_refresh()


    pan_state = {'pressing': False, 'x': None, 'y': None, 'ax': None, 'dragged': False, 'dragging_hud': False}

    def reset_debug_zoom(ax):
        """Restore the click-specific home ROI for one lower debug view."""
        key = 'A' if ax is ax_debug_A else 'B'
        home = debug_zoom_home.get(key)
        if home is None:
            return False
        ax.set_xlim(*home[0])
        ax.set_ylim(*home[1])
        request_blit_refresh()
        return True

    def on_press(event):
        if height_profile is not None and height_profile.on_press(event):
            return
        if block_accuracy_state['mode'] in ('running', 'preview') and event.inaxes in (ax_A, ax_B):
            return
        # Lower debug views: double-left-click or right-click resets the view.
        # Handle this before the left-button-only interaction below.
        if event.inaxes in (ax_debug_A, ax_debug_B):
            if event.button == 3 or (
                    event.button == 1 and getattr(event, 'dblclick', False)):
                reset_debug_zoom(event.inaxes)
                return
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

        # 下方左圖只用來挑選 descriptor audit 參考點；不會重新量測，
        # 也不會改變正式 matcher 的結果。以螢幕像素找最近點，避免縮放
        # 程度影響 12 px 的點選容許範圍。
        if event.inaxes == ax_debug_A:
            if event.x is not None and event.y is not None:
                if view_state.get('region_sift', False):
                    select_region_sift_debug_point(
                        screen_xy=(event.x, event.y),
                        announce=True, refresh=True)
                else:
                    select_grad_descriptor_audit(
                        screen_xy=(event.x, event.y),
                        announce=True, refresh=True)
            return

        if event.inaxes not in (ax_A, ax_B): return
        pan_state.update({'pressing': True, 'dragged': False, 'x': event.x, 'y': event.y, 'ax': event.inaxes})
 
    def on_release(event):
        if height_profile is not None and height_profile.on_release(event):
            return
        if pan_state.get('dragging_hud', False):
            pan_state['dragging_hud'] = False
            return
        if not pan_state['pressing']: return
        pan_state['pressing'] = False
        if not pan_state['dragged'] and event.xdata is not None:
            if (view_state.get('show_rt_warp_view', False)
                    and pan_state['ax'] in (ax_A, ax_B)):
                print(
                    "⚠️ [RT Warp View] 主畫面點擊量測已停用；"
                    "可繼續縮放/平移，關閉 RT Warp 後才能計算深度。")
                depth_text.set_text(
                    "RT+Plane Warp View (display only)\n"
                    "Depth calculation is disabled")
                request_blit_refresh()
                return
            ux, vx = float(event.xdata), float(event.ydata)
            if block_accuracy_state['mode'] == 'picking':
                if pan_state['ax'] == ax_A and event.inaxes == ax_A:
                    pick_block_accuracy_corner(ux, vx)
                return
            
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
        if height_profile is not None and height_profile.on_motion(event):
            return
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
        if height_profile is not None and height_profile.mode == 'drawing':
            return
        if event.inaxes not in (ax_A, ax_B, ax_debug_A, ax_debug_B):
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        f = 1.2 if event.button == 'down' else 1/1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata

        if ax in (ax_debug_A, ax_debug_B):
            # Keep a useful minimum window and do not zoom outside the image.
            # Preserve reversed image y-limits while clamping the visible span.
            def scaled_limits(limits, center, bound_low, bound_high):
                first, second = map(float, limits)
                reversed_axis = second < first
                low, high = min(first, second), max(first, second)
                old_span = max(high - low, 1e-9)
                max_span = float(bound_high - bound_low)
                new_span = float(np.clip(old_span * f, 6.0, max_span))
                anchor = float(np.clip((center - low) / old_span, 0.0, 1.0))
                new_low = float(center) - anchor * new_span
                new_high = new_low + new_span
                if new_low < bound_low:
                    new_high += bound_low - new_low
                    new_low = float(bound_low)
                if new_high > bound_high:
                    new_low -= new_high - bound_high
                    new_high = float(bound_high)
                result = (new_low, new_high)
                return result[::-1] if reversed_axis else result

            new_xl = scaled_limits(xl, x, -0.5, float(w) - 0.5)
            new_yl = scaled_limits(yl, y, -0.5, float(h) - 0.5)
            ax.set_xlim(*new_xl)
            ax.set_ylim(*new_yl)
        else:
            ax.set_xlim([x - (x-xl[0])*f, x + (xl[1]-x)*f])
            ax.set_ylim([y - (y-yl[0])*f, y + (yl[1]-y)*f])
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
    
    ax_btn_lock_L = fig.add_axes([0.58, control_row_y[0], 0.08, control_h])
    btn_lock_L = Button(ax_btn_lock_L, "鎖定左圖", **btn_style)
    
    ax_btn_lock_R = fig.add_axes([0.68, control_row_y[0], 0.08, control_h])
    btn_lock_R = Button(ax_btn_lock_R, "鎖定右圖", **btn_style)
    
    ax_btn_hide_R = fig.add_axes([0.78, control_row_y[0], 0.08, control_h])
    btn_hide_R = Button(ax_btn_hide_R, "顯示右圖", **btn_style)
    if view_state['region_sift']:
        ax_B.set_visible(True)
        ax_debug_B.set_visible(True)
        ax_debug_info_B.set_visible(True)
        btn_hide_R.label.set_text('隱藏右圖')
    
    ax_btn_norm = fig.add_axes([0.88, control_row_y[0], 0.08, control_h])
    btn_norm_toggle = Button(ax_btn_norm, '使用 L2', **btn_style)
    
    ax_btn_calc = fig.add_axes([0.58, control_row_y[1], 0.08, control_h])
    btn_calc = Button(ax_btn_calc, "單次計算深度", **btn_style)
    
    ax_btn_auto_calc = fig.add_axes([0.68, control_row_y[1], 0.08, control_h])
    btn_auto_calc = Button(ax_btn_auto_calc, "連續計算: 關", **btn_style)
    
    ax_btn_grad = fig.add_axes([0.78, control_row_y[1], 0.08, control_h])
    btn_grad_toggle = Button(ax_btn_grad, '顯示梯度 SIFT 連線', **btn_style)
    
    ax_btn_custom_plane = fig.add_axes([0.88, control_row_y[1], 0.08, control_h])
    btn_custom_plane = Button(ax_btn_custom_plane, "Custom Plane", **btn_style)
    
    ax_btn_high_grad_pts = fig.add_axes([0.58, control_row_y[2], 0.08, control_h])
    btn_high_grad_pts = Button(ax_btn_high_grad_pts, "HighPts: Off", **btn_style)
    
    ax_btn_mid_grad_pts = fig.add_axes([0.68, control_row_y[2], 0.08, control_h])
    btn_mid_grad_pts = Button(ax_btn_mid_grad_pts, "MidPts: Off", **btn_style)
    
    ax_btn_rt_diff = fig.add_axes([0.78, control_row_y[2], 0.08, control_h])
    btn_rt_diff = Button(ax_btn_rt_diff, "RT Diff", **btn_style)
    
    ax_btn_return_menu = fig.add_axes([0.88, control_row_y[2], 0.08, control_h])
    btn_return_menu = Button(ax_btn_return_menu, "Back to Menu", **btn_style)
    
    # 建立 TextBox 用於傷口高度補償
    ax_btn_wound = fig.add_axes([0.58, control_row_y[3], 0.08, control_h])
    btn_wound_toggle = Button(ax_btn_wound, "Wound: Off", **btn_style)

    ax_btn_wound_pts = fig.add_axes([0.68, control_row_y[3], 0.08, control_h])
    btn_wound_pts_toggle = Button(ax_btn_wound_pts, "Pts: Rect", **btn_style)

    ax_btn_aruco_overlay = fig.add_axes([0.78, control_row_y[3], 0.08, control_h])
    btn_aruco_overlay = Button(ax_btn_aruco_overlay, "ArUco標記: Off", **btn_style)

    ax_btn_rt_sift = fig.add_axes([0.88, control_row_y[3], 0.08, control_h])
    btn_rt_sift = Button(ax_btn_rt_sift, "RT SIFT: Off", **btn_style)

    ax_btn_height_plane = fig.add_axes([0.58, control_row_y[4], 0.12, control_h])
    btn_height_plane = Button(ax_btn_height_plane, "Height: Legacy", **btn_style)

    ax_btn_metric_blocks = fig.add_axes([0.88, control_row_y[4], 0.08, control_h])
    btn_metric_blocks = Button(
        ax_btn_metric_blocks,
        "5mm Grid: On" if view_state['show_metric_blocks'] else "5mm Grid: Off",
        **btn_style)

    ax_btn_shared_plane = fig.add_axes([0.58, control_row_y[5], 0.09, control_h])
    btn_shared_plane = Button(ax_btn_shared_plane, "Shared: On", **btn_style)
    btn_shared_plane.ax.patch.set_facecolor('#145A32')

    ax_btn_top2_geo = fig.add_axes([0.68, control_row_y[5], 0.09, control_h])
    btn_top2_geo = Button(
        ax_btn_top2_geo,
        "Top2Geo: On" if view_state['top2_geometry_rescue']
        else "Top2Geo: Off",
        **btn_style)

    ax_btn_h_residual = fig.add_axes([0.78, control_row_y[5], 0.09, control_h])
    btn_h_residual = Button(
        ax_btn_h_residual,
        "H-Resid: On" if view_state['show_homography_residual']
        else "H-Resid: Off",
        **btn_style)

    ax_btn_rt_warp_view = fig.add_axes([0.88, control_row_y[5], 0.09, control_h])
    btn_rt_warp_view = Button(ax_btn_rt_warp_view, "RT Warp: Off", **btn_style)

    ax_btn_block_accuracy = fig.add_axes([0.58, control_row_y[6], 0.19, control_h])
    btn_block_accuracy = Button(ax_btn_block_accuracy, "Block 誤差統計", **btn_style)
    ax_btn_block_cancel = fig.add_axes([0.78, control_row_y[6], 0.09, control_h])
    btn_block_cancel = Button(ax_btn_block_cancel, "取消統計", **btn_style)
    ax_btn_block_mae = fig.add_axes([0.88, control_row_y[6], 0.09, control_h])
    # The app owns full-figure blitting; this widget must not independently
    # blit a stale Off label over the updated figure background.
    btn_block_mae = Button(ax_btn_block_mae, "MAE 著色: Off", useblit=False, **btn_style)
    for _button in (btn_block_accuracy, btn_block_cancel, btn_block_mae):
        _button.label.set_color('#E0E0E0')
        _button.label.set_fontsize(7)
        _button.ax.patch.set_edgecolor('#D83B01')

    # Region-SIFT coordinate replay.  Values copied from points.csv are
    # rounded exactly like the block-accuracy batch before entering do_measure.
    ax_region_u = fig.add_axes([0.185, control_row_y[6], 0.045, control_h])
    ax_region_v = fig.add_axes([0.255, control_row_y[6], 0.045, control_h])
    ax_btn_region_replay = fig.add_axes([0.315, control_row_y[6], 0.085, control_h])
    text_region_u = TextBox(
        ax_region_u, 'U ', initial='', color='#1A1A1A', hovercolor='#333333')
    text_region_v = TextBox(
        ax_region_v, 'V ', initial='', color='#1A1A1A', hovercolor='#333333')
    btn_region_replay = Button(ax_btn_region_replay, '座標重現', **btn_opt_style)
    for _coordinate_box in (text_region_u, text_region_v):
        _coordinate_box.label.set_color('#E0E0E0')
        _coordinate_box.label.set_fontsize(7)
        _coordinate_box.text_disp.set_color('#E0E0E0')
        _coordinate_box.text_disp.set_fontsize(7)
        _coordinate_box.ax.patch.set_linewidth(1.0)
        _coordinate_box.ax.patch.set_edgecolor('#00BFFF')
    btn_region_replay.label.set_color('#E0E0E0')
    btn_region_replay.label.set_fontsize(7)
    btn_region_replay.ax.patch.set_edgecolor('#00BFFF')
    btn_region_replay.ax.patch.set_linewidth(1.0)

    ax_btn_region_settings = fig.add_axes([0.410, control_row_y[6], 0.075, control_h])
    btn_region_settings = Button(ax_btn_region_settings, 'SIFT 參數', **btn_opt_style)
    btn_region_settings.label.set_color('#E0E0E0')
    btn_region_settings.label.set_fontsize(7)
    btn_region_settings.ax.patch.set_edgecolor('#00BFFF')
    ax_btn_region_diagnostics = fig.add_axes([0.495, control_row_y[6], 0.075, control_h])
    btn_region_diagnostics = Button(ax_btn_region_diagnostics, 'SIFT Debug', **btn_opt_style)
    btn_region_diagnostics.label.set_color('#E0E0E0')
    btn_region_diagnostics.label.set_fontsize(7)
    btn_region_diagnostics.on_clicked(show_region_diagnostics)
    ax_btn_height_profile = fig.add_axes([0.415, control_row_y[5], 0.15, control_h])
    btn_height_profile = Button(ax_btn_height_profile, '拖曳高度剖面', useblit=False, **btn_opt_style)
    btn_height_profile.label.set_color('#E0E0E0')
    btn_height_profile.label.set_fontsize(7)

    def apply_region_settings(updated):
        global REGION_SIFT_CONFIG
        if height_profile is not None and height_profile.enabled:
            raise ValueError('請先關閉拖曳高度剖面，再修改參數')
        if block_accuracy_state['mode'] != 'idle':
            raise ValueError('請先完成或取消 Block 誤差統計，再修改參數')
        previous = REGION_SIFT_CONFIG
        REGION_SIFT_CONFIG = updated
        # Left descriptor cache keys include the whole immutable config;
        # the next measurement also creates a new click-local cache.
        for name, value in vars(updated).items():
            if getattr(previous, name) != value:
                print(f'[Region-SIFT Settings] {name}: {getattr(previous, name)} -> {value}')
        depth_text.set_text('Region-SIFT 參數已套用；請重新點選或按座標重現')
        request_blit_refresh()

    def on_region_settings(event):
        if block_accuracy_state['mode'] != 'idle':
            depth_text.set_text('請先完成或取消 Block 誤差統計，再修改參數')
            request_blit_refresh()
            return
        show_region_sift_settings(fig.canvas.manager.window, REGION_SIFT_CONFIG,
                                  apply_region_settings)

    def on_region_coordinate_replay(event):
        if not view_state.get('region_sift', False):
            message = '座標重現需要先啟用 Region-SIFT 匹配'
            print(f'[Region-SIFT Replay] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        if block_accuracy_state['mode'] != 'idle':
            message = '請先完成或取消 Block 誤差統計'
            print(f'[Region-SIFT Replay] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        try:
            input_u = float(text_region_u.text.strip())
            input_v = float(text_region_v.text.strip())
        except (TypeError, ValueError):
            message = 'U、V 必須是有效數字'
            print(f'[Region-SIFT Replay] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        if not np.isfinite(input_u) or not np.isfinite(input_v):
            message = 'U、V 必須是有限數字'
            print(f'[Region-SIFT Replay] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        measured_u, measured_v = int(round(input_u)), int(round(input_v))
        image_h, image_w = locked_L_clean.shape[:2]
        if not (0 <= measured_u < image_w and 0 <= measured_v < image_h):
            message = (f'座標超出左圖範圍：U=0..{image_w - 1}, '
                       f'V=0..{image_h - 1}')
            print(f'[Region-SIFT Replay] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        print(f'[Region-SIFT Replay] CSV input=({input_u:.6f}, {input_v:.6f}) '
              f'-> measured=({measured_u}, {measured_v})')
        do_measure(measured_u, measured_v)

    wound_z_offset = 0.0
    # 原本位於左下角，會壓到新的 Debug 資訊列；移入右側控制區空位。
    ax_box = fig.add_axes([0.78, control_row_y[4], 0.08, control_h])
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
    for b in [btn_lock_L, btn_lock_R, btn_hide_R, btn_norm_toggle, btn_calc, btn_auto_calc, btn_grad_toggle, btn_custom_plane, btn_high_grad_pts, btn_mid_grad_pts, btn_rt_diff, btn_return_menu, btn_wound_toggle, btn_wound_pts_toggle, btn_aruco_overlay, btn_rt_sift, btn_height_plane, btn_metric_blocks, btn_shared_plane, btn_top2_geo, btn_h_residual, btn_rt_warp_view]:
        b.label.set_color('#E0E0E0') # 質感白
        b.label.set_fontsize(7)
        b.ax.patch.set_linewidth(1.2) # 細緻邊框
        
    # 依功能進行邊框分色（專業軟體常見的語意化色彩）
    # 1. 影像鎖定/控制類：使用專業藍 (#007ACC)
    for b in [btn_lock_L, btn_lock_R, btn_hide_R]:
        b.ax.patch.set_edgecolor('#007ACC')
        
    # 2. 深度計算類：使用警告橘/強調橘 (#D83B01)
    for b in [btn_calc, btn_auto_calc]:
        b.ax.patch.set_edgecolor('#D83B01')
        
    # 3. 功能切換類：使用中性的深灰 (#555555)
    for b in [btn_norm_toggle, btn_grad_toggle, btn_custom_plane, btn_high_grad_pts, btn_mid_grad_pts, btn_rt_diff, btn_wound_toggle, btn_wound_pts_toggle, btn_aruco_overlay, btn_height_plane, btn_metric_blocks, btn_shared_plane, btn_top2_geo, btn_h_residual, btn_rt_warp_view]:
        b.ax.patch.set_edgecolor('#555555')
        
    # 4. 導覽/返回選單類：使用翡翠綠 (#28A745)
    btn_return_menu.ax.patch.set_edgecolor('#28A745')
        
    btn_grad_toggle.label.set_fontsize(7) # 特長文字微調
    
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

    def on_metric_blocks_toggle(event):
        view_state['show_metric_blocks'] = not view_state['show_metric_blocks']
        visible = view_state['show_metric_blocks']
        btn_metric_blocks.label.set_text(
            "5mm Grid: On" if visible else "5mm Grid: Off")
        res = measure_results.get(current_cand['idx'])
        if last_click is not None and res is not None:
            update_grad_match_debug_views(
                res, float(last_click[0]), float(last_click[1]))
        else:
            clear_metric_block_debug_artists()
        print(
            f"[5mm Block Debug] {'enabled' if visible else 'disabled'}; "
            "read_only=True, block_filter_applied=False")
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
        # RT 由所有 mapped markers 求得；RT Diff 也優先使用所有左右共視、
        # 四角皆有效的 markers 擬合共同平面，不再固定使用 reference ID 平面。
        corners_left_raw = current_cand.get('cornersA') or {}
        corners_right_raw = current_cand.get('cornersB') or {}
        # 正規化 ID 型別，避免來源字典使用 np.int / 字串 ID 時，
        # 交集判定成功卻無法用 Python int 取回角點。
        corners_left = {int(mid): pts for mid, pts in corners_left_raw.items()}
        corners_right = {int(mid): pts for mid, pts in corners_right_raw.items()}
        mapped_ids = set(int(mid) for mid in (current_cand.get('marker_map') or {}))
        participant_ids_left = sorted(
            set(int(mid) for mid in corners_left) & mapped_ids)
        participant_ids_right = sorted(
            set(int(mid) for mid in corners_right) & mapped_ids)
        comparable_ids = sorted(
            set(participant_ids_left) & set(participant_ids_right))
        rt_plane_left = {mid: corners_left[mid] for mid in comparable_ids}
        rt_plane_right = {mid: corners_right[mid] for mid in comparable_ids}
        rt_diff_plane_n, rt_diff_plane_c, rt_diff_plane_diag = (
            compute_shared_marker_corner_plane(
                rt_plane_left, rt_plane_right,
                KL, current_cand['K_R'],
                current_cand['R_rel'], current_cand['t_rel'],
                F=current_cand.get('F'),
                reference_normal=current_cand.get('plane_n'),
                min_shared_markers=2)
            if len(comparable_ids) >= 2
            else (None, None, {
                'available': False,
                'reason': f'only {len(comparable_ids)} common mapped RT patterns',
                'used_marker_ids': comparable_ids,
            }))
        use_all_pattern_plane = bool(
            rt_diff_plane_n is not None
            and rt_diff_plane_c is not None
            and rt_diff_plane_diag.get('available', False))
        if use_all_pattern_plane:
            plane_n = np.asarray(rt_diff_plane_n, dtype=np.float64).reshape(3)
            plane_c = np.asarray(rt_diff_plane_c, dtype=np.float64).reshape(3)
            plane_marker_ids = [
                int(mid) for mid in rt_diff_plane_diag.get('used_marker_ids', [])]
            plane_source = "all shared RT patterns"
        else:
            plane_n = current_cand.get('plane_n')
            plane_c = current_cand.get('plane_c')
            plane_marker_ids = []
            plane_source = "reference marker fallback"
        if plane_n is None or plane_c is None:
            print("⚠️ [RT Diff] 缺少共同平面與 reference plane，無法做 RT warp。")
            return
        plane_n = np.asarray(plane_n, dtype=np.float64).reshape(3)
        plane_c = np.asarray(plane_c, dtype=np.float64).reshape(3)
        d_plane = float(np.dot(plane_n, plane_c))
        if abs(d_plane) < 1e-6:
            print("⚠️ [RT Diff] 平面距離 d_plane 趨近 0，無法計算 homography。")
            return

        left_gray = cv2.cvtColor(locked_L_clean, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(locked_R_clean, cv2.COLOR_BGR2GRAY)
        hR, wR = right_gray.shape[:2]
        H_AB = current_cand['K_R'] @ (
            np.asarray(current_cand['R_rel'], dtype=np.float64).reshape(3, 3)
            + (np.asarray(current_cand['t_rel'], dtype=np.float64).reshape(3, 1)
               @ plane_n.reshape(1, 3)) / d_plane
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

        # Red/green false-colour overlay makes alignment easier to inspect than
        # a scalar absolute-difference image.  Warped left contributes green,
        # right contributes red; aligned content becomes yellow, while a
        # displacement leaves separate green/red edges.  Keep ``diff`` above
        # for the existing numeric gray-difference diagnostics only.
        left_level = warped_left.astype(np.float32) / 255.0
        right_level = right_gray.astype(np.float32) / 255.0
        overlay_rgb = np.empty((hR, wR, 3), dtype=np.float32)
        overlay_rgb[..., 0] = np.clip(
            0.90 * right_level + 0.30 * left_level, 0.0, 1.0)
        overlay_rgb[..., 1] = np.clip(
            0.30 * right_level + 0.90 * left_level, 0.0, 1.0)
        overlay_rgb[..., 2] = np.clip(
            0.22 * (right_level + left_level), 0.0, 1.0)
        overlay_rgb[valid == 0] = 0.0

        diff_fig, diff_ax = plt.subplots(1, 1, figsize=(9, 6), facecolor='#1E1E1E')
        diff_fig.canvas.manager.set_window_title("RT Warp Red/Green Overlay")
        diff_ax.imshow(overlay_rgb)
        diff_ax.set_title(
            "RT Overlay: warped left=GREEN, right=RED (aligned=YELLOW) | "
            "all-pattern shared plane"
            if use_all_pattern_plane
            else "RT Overlay: warped left=GREEN, right=RED (aligned=YELLOW) | "
                 "reference-plane fallback",
            color='white')

        # 直接量測「這張共同 Homography」對每個 Pattern 的實際效果。
        # 這與 RT/PnP 重投影誤差不同：若 Pattern 不共面，即使 RT 正確，
        # 各 ID 的 H-warp corner error 仍可能明顯不同。
        per_marker_lines = []
        for mid in comparable_ids:
            try:
                pts_left = np.asarray(corners_left[mid], dtype=np.float64).reshape(4, 2)
                pts_right = np.asarray(corners_right[mid], dtype=np.float64).reshape(4, 2)
                projected = cv2.perspectiveTransform(
                    pts_left.reshape(-1, 1, 2).astype(np.float64), H_AB
                ).reshape(4, 2)
            except (TypeError, ValueError, cv2.error):
                continue
            corner_error = np.linalg.norm(projected - pts_right, axis=1)
            corner_rms = float(np.sqrt(np.mean(corner_error ** 2)))
            corner_max = float(np.max(corner_error))
            marker_mask = np.zeros_like(right_gray, dtype=np.uint8)
            cv2.fillConvexPoly(
                marker_mask, np.rint(pts_right).astype(np.int32), 255)
            marker_valid = (marker_mask > 0) & (valid > 0)
            photo_mean = (
                float(np.mean(diff[marker_valid]))
                if np.any(marker_valid) else float('nan'))
            per_marker_lines.append(
                f"ID {mid}: H-corner RMS={corner_rms:.3f}px, "
                f"max={corner_max:.3f}px, gray-diff={photo_mean:.2f}")

            polygon = np.vstack([pts_right, pts_right[0]])
            diff_ax.plot(
                polygon[:, 0], polygon[:, 1], color='#00FFFF',
                linewidth=1.0, alpha=0.9)
            center = np.mean(pts_right, axis=0)
            diff_ax.text(
                float(center[0]), float(center[1]),
                f"ID {mid}\nH {corner_rms:.2f}px",
                color='#00FFFF', fontsize=7, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none'))
        diff_ax.axis("off")
        diff_ax.set_facecolor('#1E1E1E')
        diff_fig.tight_layout()
        diff_fig.show()
        valid_mean = float(np.mean(diff[valid > 0])) if np.any(valid > 0) else 0.0
        print("=" * 96)
        if use_all_pattern_plane:
            print(
                "📊 [RT Diff] 使用全部共視 RT Pattern 擬合的共同平面 | "
                f"IDs={plane_marker_ids} | corners={rt_diff_plane_diag.get('point_count')} | "
                f"plane RMS={rt_diff_plane_diag.get('rms_mm'):.3f}mm, "
                f"P90={rt_diff_plane_diag.get('p90_abs_mm'):.3f}mm, "
                f"max={rt_diff_plane_diag.get('max_abs_mm'):.3f}mm")
        else:
            print(
                "⚠️ [RT Diff] 全 Pattern 共同平面不可用，已退回 reference marker plane | "
                f"reason={rt_diff_plane_diag.get('reason', 'unknown')}")
        print(
            f"   source={plane_source} | d_plane={d_plane:.3f} | "
            f"valid gray-diff mean={valid_mean:.2f}")
        print(
            "   display=red/green overlay: warped left=GREEN, "
            "right=RED, aligned=YELLOW; gray-diff remains numeric-only")
        if participant_ids_left != comparable_ids or participant_ids_right != comparable_ids:
            print(
                f"   RT participants right(A)/left(B)="
                f"{participant_ids_right}/{participant_ids_left}; "
                f"RT Diff can compare only common IDs={comparable_ids}")
        for line in per_marker_lines:
            print("   " + line)
        print("=" * 96)
        
    def on_return_menu(event):
        view_state['restart'] = True
        plt.close(fig)
        print("🔄 正在關閉目前量測介面並返回影片來源選單...")
    
    def on_hide_R(event):
        visible = ax_B.get_visible()
        ax_B.set_visible(not visible)
        ax_debug_B.set_visible(not visible)
        ax_debug_info_B.set_visible(not visible)
        btn_hide_R.label.set_text("顯示右圖" if visible else "隱藏右圖")
        # Debug 版下方已有 ROI 圖，HUD 固定留在上方主圖區，
        # 避免右圖隱藏時掉到最下方蓋住 Debug 圖。
        depth_text.set_position((0.53, 0.50))
        request_blit_refresh()
    
    auto_calc_active = False
    
    def on_auto_calc(event):
        nonlocal auto_calc_active, reset_pose_history
        if view_state.get('show_rt_warp_view', False):
            print(
                "⚠️ [RT Warp View] 純顯示模式中，無法開啟連續計算；"
                "請先關閉 RT Warp。")
            depth_text.set_text(
                "RT+Plane Warp View (display only)\n"
                "Depth calculation is disabled")
            request_blit_refresh()
            return
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

    def build_rt_warp_right_display():
        """用既有 RT + 基準平面 Homography，把原始右圖投回左圖座標（僅顯示）。"""
        plane_n = current_cand.get('plane_n')
        plane_c = current_cand.get('plane_c')
        if plane_n is None or plane_c is None:
            raise ValueError("缺少 plane_n / plane_c")

        plane_n = np.asarray(plane_n, dtype=np.float64).reshape(3)
        plane_c = np.asarray(plane_c, dtype=np.float64).reshape(3)
        d_plane = float(np.dot(plane_n, plane_c))
        if not np.isfinite(d_plane) or abs(d_plane) < 1e-6:
            raise ValueError(f"無效的平面距離 d_plane={d_plane}")

        K_left = np.asarray(KL, dtype=np.float64).reshape(3, 3)
        K_right = np.asarray(current_cand['K_R'], dtype=np.float64).reshape(3, 3)
        R_rel = np.asarray(current_cand['R_rel'], dtype=np.float64).reshape(3, 3)
        t_rel = np.asarray(current_cand['t_rel'], dtype=np.float64).reshape(3, 1)
        H_left_to_right = K_right @ (
            R_rel + (t_rel @ plane_n.reshape(1, 3)) / d_plane
        ) @ np.linalg.inv(K_left)
        if (not np.all(np.isfinite(H_left_to_right))
                or abs(float(np.linalg.det(H_left_to_right))) < 1e-12):
            raise ValueError("left-to-right Homography 無效或不可逆")

        H_right_to_left = np.linalg.inv(H_left_to_right)
        h_left, w_left = locked_L_clean.shape[:2]
        h_right, w_right = locked_R_clean.shape[:2]
        warped_right_bgr = cv2.warpPerspective(
            locked_R_clean,
            H_right_to_left,
            (w_left, h_left),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        valid = cv2.warpPerspective(
            np.full((h_right, w_right), 255, dtype=np.uint8),
            H_right_to_left,
            (w_left, h_left),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_ratio = float(np.count_nonzero(valid)) / max(float(valid.size), 1.0)
        return cv2.cvtColor(warped_right_bgr, cv2.COLOR_BGR2RGB), valid_ratio, d_plane

    def on_rt_warp_view_toggle(event):
        """切換右上圖的 RT+Plane warp 預覽；不修改任何計算影像或結果。"""
        nonlocal auto_calc_active
        enabling = not view_state.get('show_rt_warp_view', False)

        if enabling:
            if custom_plane_mode:
                print(
                    "⚠️ [RT Warp View] 自訂平面選點尚未結束；"
                    "請先完成或取消後再開啟 warp 顯示。")
                return
            try:
                frame_rgb, valid_ratio, d_plane = build_rt_warp_right_display()
            except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
                print(f"❌ [RT Warp View] 無法建立顯示影像: {exc}")
                depth_text.set_text(f"RT Warp unavailable: {exc}")
                request_blit_refresh()
                return

            rt_warp_view_state['frame_rgb'] = frame_rgb
            rt_warp_view_state['valid_ratio'] = valid_ratio
            rt_warp_view_state['hud_text'] = depth_text.get_text()
            rt_warp_view_state['forced_ax_b_visible'] = not ax_B.get_visible()
            if rt_warp_view_state['forced_ax_b_visible']:
                ax_B.set_visible(True)
                btn_hide_R.label.set_text("隱藏右圖")
            view_state['show_rt_warp_view'] = True
            if auto_calc_active:
                auto_calc_active = False
                btn_auto_calc.label.set_text("連續計算: 關")
                print("⏸️ [RT Warp View] 連續計算已自動關閉")
            view_state['manual_pt_A'] = None
            btn_rt_warp_view.label.set_text("RT Warp: On")
            btn_rt_warp_view.ax.patch.set_facecolor('#145A32')
            ax_B.set_title(
                '右圖 (RT+Plane Warp → Left; DISPLAY ONLY)',
                color='#7CFFB2', fontsize=10, fontweight='bold', pad=5)
            depth_text.set_text(
                "RT+Plane Warp View (display only)\n"
                "Depth calculation is disabled")
            print(
                "👁️ [RT Warp View] ON: 右上圖已切換為原始右圖經 "
                "RT + 基準平面 Homography 投回左圖座標；"
                f"valid={valid_ratio * 100.0:.1f}%, d_plane={d_plane:.3f}. ")
            print(
                "   僅顯示影像；locked_R、Grad-SIFT、匹配、內插、"
                "三角化與下方 Debug 右圖完全不變。所有深度計算入口已停用。")
        else:
            view_state['show_rt_warp_view'] = False
            rt_warp_view_state['frame_rgb'] = None
            rt_warp_view_state['valid_ratio'] = None
            if rt_warp_view_state.get('forced_ax_b_visible', False):
                ax_B.set_visible(False)
                btn_hide_R.label.set_text("顯示右圖")
            rt_warp_view_state['forced_ax_b_visible'] = False
            btn_rt_warp_view.label.set_text("RT Warp: Off")
            btn_rt_warp_view.ax.patch.set_facecolor('#1A1A1A')
            ax_B.set_title(
                '右圖 (Locked)', color='white', fontsize=10,
                fontweight='bold', pad=5)
            depth_text.set_text(rt_warp_view_state.pop('hud_text', ''))
            print(
                "👁️ [RT Warp View] OFF: 右上圖已恢復原始右圖；"
                "深度計算功能已恢復。")

        mark_display_dirty()
        request_blit_refresh()
    
    def on_lock_L(event):
        # 離線影片模式沒有 Live 串流可供重新鎖定，左圖固定為分析挑出的最優影格
        print("⚠️ 離線影片模式不支援重新鎖定左圖，左圖固定為分析挑出的最優影格。")

    def on_lock_R(event):
        # 離線影片模式沒有 Live 串流可供重新鎖定，右圖固定為分析挑出的最優影格
        print("⚠️ 離線影片模式不支援重新鎖定右圖，右圖固定為分析挑出的最優影格。")
        
    def on_custom_plane(event):
        nonlocal custom_plane_mode, custom_plane_n, custom_plane_c, custom_plane_fitted, auto_calc_active

        if view_state.get('show_rt_warp_view', False):
            print(
                "⚠️ [RT Warp View] 純顯示模式中，無法進入自訂平面量測；"
                "請先關閉 RT Warp。")
            depth_text.set_text(
                "RT+Plane Warp View (display only)\n"
                "Depth calculation is disabled")
            request_blit_refresh()
            return
        
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

    def on_height_plane_toggle(event):
        # The original selector and the new shared-pattern selector are
        # mutually exclusive.  Pressing this button always returns control to
        # the Legacy/Marker-Pose pair before toggling that pair.
        if height_plane_state['use_shared_plane']:
            height_plane_state['use_shared_plane'] = False
            btn_shared_plane.label.set_text("Shared: Off")
            btn_shared_plane.ax.patch.set_facecolor('#1A1A1A')
        if pose_height_plane_n is None or pose_height_plane_c is None:
            height_plane_state['use_pose_plane'] = False
            btn_height_plane.label.set_text("Height: Legacy")
            btn_height_plane.ax.patch.set_facecolor('#1A1A1A')
            plane_dist_history.clear()
            print("[Height Plane] Marker-pose plane is unavailable; using Legacy mode")
            if last_click and current_cand['idx'] in measure_results:
                update_display(None, [])
            else:
                request_blit_refresh()
            return
        height_plane_state['use_pose_plane'] = not height_plane_state['use_pose_plane']
        use_pose = height_plane_state['use_pose_plane']
        btn_height_plane.label.set_text(
            "Height: Marker" if use_pose else "Height: Legacy")
        btn_height_plane.ax.patch.set_facecolor('#145A32' if use_pose else '#1A1A1A')
        plane_dist_history.clear()
        print(
            "[Height Plane] Display mode -> "
            + ("Marker Pose Plane" if use_pose else "Legacy Triangulated/SVD Plane")
            + "; RT, baseline, matching and p3d unchanged")
        if custom_plane_fitted:
            print("[Height Plane] Custom Plane is active and still has display priority")
        if last_click and current_cand['idx'] in measure_results:
            update_display(None, [])
        else:
            request_blit_refresh()

    def on_shared_plane_toggle(event):
        if (not height_plane_state['use_shared_plane']
                and (shared_height_plane_n is None or shared_height_plane_c is None)):
            print(
                "[Shared Pattern Plane] unavailable; keeping current height plane: "
                + shared_height_plane_diag.get('reason', 'unknown reason'))
            return

        use_shared = not height_plane_state['use_shared_plane']
        height_plane_state['use_shared_plane'] = use_shared
        if use_shared:
            height_plane_state['use_pose_plane'] = False
            btn_height_plane.label.set_text("Height: Legacy")
            btn_height_plane.ax.patch.set_facecolor('#1A1A1A')
        btn_shared_plane.label.set_text(
            "Shared: On" if use_shared else "Shared: Off")
        btn_shared_plane.ax.patch.set_facecolor('#145A32' if use_shared else '#1A1A1A')
        plane_dist_history.clear()
        if use_shared:
            print(
                "[Shared Pattern Plane] Display mode -> ON; "
                f"markers={shared_height_plane_diag['used_marker_ids']}, "
                f"corners={shared_height_plane_diag['point_count']}, "
                f"RMS={shared_height_plane_diag['rms_mm']:.3f} mm, "
                f"P90={shared_height_plane_diag['p90_abs_mm']:.3f} mm, "
                f"max={shared_height_plane_diag['max_abs_mm']:.3f} mm; "
                "RT, baseline, matching and p3d unchanged")
        else:
            print(
                "[Shared Pattern Plane] Display mode -> OFF; returning to "
                "Legacy Triangulated/SVD Plane; RT, baseline, matching and p3d unchanged")
        if custom_plane_fitted:
            print("[Shared Pattern Plane] Custom Plane is active and still has display priority")
        if last_click and current_cand['idx'] in measure_results:
            update_display(None, [])
        else:
            request_blit_refresh()

    def on_top2_geometry_toggle(event):
        enabled = not view_state['top2_geometry_rescue']
        view_state['top2_geometry_rescue'] = enabled
        btn_top2_geo.label.set_text(
            "Top2Geo: On" if enabled else "Top2Geo: Off")
        btn_top2_geo.ax.patch.set_facecolor(
            '#145A32' if enabled else '#1A1A1A')
        if enabled:
            print(
                "[Top2-Geo] ON: original Guided Fallback is paused; "
                "ratio-rejected references test only Global Top-1/Top-2, "
                f"then choose the candidate nearest exact H(pL) within "
                f"{TOP2_GEOMETRY_MAX_DIST_PX:g}px. Gate statistics are logged.")
        else:
            print(
                "[Top2-Geo] OFF: restored the original Guided Fallback path.")
        mark_wound_size_dirty('top2_geometry_rescue')
        if last_click:
            do_measure(last_click[0], last_click[1])
        else:
            request_blit_refresh()

    def on_h_residual_toggle(event):
        visible = not view_state['show_homography_residual']
        view_state['show_homography_residual'] = visible
        btn_h_residual.label.set_text(
            "H-Resid: On" if visible else "H-Resid: Off")
        btn_h_residual.ax.patch.set_facecolor(
            '#145A32' if visible else '#1A1A1A')
        dbg_pred_B.set_visible(visible)
        for artist in debug_homography_residual_artists:
            artist.set_visible(visible)
        print(
            f"[Homography Residual] {'shown' if visible else 'hidden'}: "
            "magenta H(pL) circles and residual segments only; "
            "matching and interpolation are unchanged")
        request_blit_refresh()

    # Accuracy runs use one ordinary click calculation per timer tick on the
    # main thread. No Matplotlib or shared matcher state is accessed by a worker.
    # Controls are locked until completion/cancel to keep one run comparable.
    def clear_block_accuracy_overlay():
        for artist in block_accuracy_state['artists']:
            artist.remove()
        block_accuracy_state['artists'] = []

    def block_accuracy_control_widgets():
        widgets = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
                   c13, c14, c15, c16, c17, c18, c19, radio_mode, text_box,
                   btn_lock_L, btn_lock_R, btn_hide_R, btn_norm_toggle,
                   btn_calc, btn_auto_calc, btn_grad_toggle, btn_custom_plane,
                   btn_high_grad_pts, btn_mid_grad_pts, btn_rt_diff, btn_return_menu,
                   btn_wound_toggle, btn_wound_pts_toggle, btn_aruco_overlay,
                   btn_rt_sift, btn_height_plane, btn_metric_blocks, btn_shared_plane,
                   btn_top2_geo, btn_h_residual, btn_rt_warp_view]
        widgets.extend([text_region_u, text_region_v, btn_region_replay, btn_region_settings])
        return widgets

    def lock_block_accuracy_controls():
        widgets = block_accuracy_control_widgets() + [btn_height_profile]
        block_accuracy_state['disabled_widgets'] = [(widget, widget.active) for widget in widgets]
        for widget in widgets:
            # RadioButtons.set_active(index) SELECTS an option; it does not
            # disable events. The inherited active property is the common
            # event-enable API for Buttons, RadioButtons and TextBox alike.
            widget.active = False

    def draw_block_accuracy_plan(block_summaries=None):
        clear_block_accuracy_overlay()
        plan = block_accuracy_state['plan']
        summaries = {row['block_id']: row for row in (block_summaries or [])}
        for block in plan['blocks']:
            row = summaries.get(block['block_id'])
            color = (block_mae_color(row.get('mae_mm'))
                     if block_accuracy_state['show_mae_colors'] and row and row['valid_count'] else None)
            poly = Polygon(block['corners_px'], closed=True, fill=color is not None,
                           facecolor=color or 'none', alpha=0.45 if color else 1.0,
                           edgecolor='#00FFFF', linewidth=0.8, zorder=8)
            ax_A.add_patch(poly)
            block_accuracy_state['artists'].append(poly)
            if block_accuracy_state['show_mae_colors']:
                continue  # Keep the colored block unobstructed by labels.
            label = f"B{block['block_id']:02d}\nT={block['true_height_mm']:g}mm"
            if row:
                if row['valid_count']:
                    label += (f"\nH={row['mean_height_mm']:.2f}\nE={row['mean_height_error_mm']:+.2f}"
                              f" ({row['valid_count']}/{row['expected_count']})")
                else:
                    label += '\nH=N/A'
            x, y = block['center_px']
            text = ax_A.text(x, y, label, ha='center', va='center', fontsize=6,
                             color='yellow', zorder=10,
                             bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=1))
            block_accuracy_state['artists'].append(text)
        if not block_accuracy_state['show_mae_colors']:
            points = np.array([[point['u'], point['v']] for point in plan['samples']])
            scatter = ax_A.scatter(points[:, 0], points[:, 1], s=5, c='#FF55FF', zorder=9)
            block_accuracy_state['artists'].append(scatter)
        if block_accuracy_state['show_mae_colors']:
            legend = ax_A.text(
                0.01, 0.99, 'MAE (mm): green <0.5 | orange 0.5-1.0 | red >1.0\n'
                'Unfilled: no valid data. Toggle Off for labels / sample points.',
                transform=ax_A.transAxes, ha='left', va='top', fontsize=7,
                color='white', zorder=12,
                bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=3))
            block_accuracy_state['artists'].append(legend)

    def on_block_mae_toggle(event):
        if block_accuracy_state['mode'] != 'idle':
            message = f"Block MAE: finish/cancel current operation ({block_accuracy_state['mode']}) first"
            print(f'[Block MAE] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        # Reconstruct from the current in-memory point records, even if CSV
        # export failed after measurement or finalization lost its return value.
        report = block_accuracy_state.get('report')
        if report is not None and report.records:
            blocks, _ = summarize_block_records(report.plan, report.records)
            block_accuracy_state['plan'] = report.plan
            block_accuracy_state['summaries'] = blocks
        if block_accuracy_state['summaries'] is None or block_accuracy_state['plan'] is None:
            message = 'Block MAE: no statistics in this window. Run block accuracy first.'
            print(f'[Block MAE] {message}')
            depth_text.set_text(message)
            request_blit_refresh()
            return
        enabled = not block_accuracy_state['show_mae_colors']
        block_accuracy_state['show_mae_colors'] = enabled
        btn_block_mae.label.set_text('MAE 著色: On' if enabled else 'MAE 著色: Off')
        draw_block_accuracy_plan(block_accuracy_state['summaries'])
        count = sum(row['valid_count'] > 0 for row in block_accuracy_state['summaries'])
        print(f"[Block MAE] {'On' if enabled else 'Off'}; valid blocks={count}/"
              f"{len(block_accuracy_state['summaries'])}; green <0.5, orange 0.5-1.0, red >1.0 mm")
        request_blit_refresh()

    def pick_block_accuracy_corner(u, v):
        image_h, image_w = locked_L_clean.shape[:2]
        if not (0 <= u <= image_w - 1 and 0 <= v <= image_h - 1):
            return
        corners = block_accuracy_state['corners']
        corners.append((float(u), float(v)))
        dot, = ax_A.plot(u, v, 'yo', markersize=5, zorder=10)
        label = ax_A.text(u + 3, v - 3, ('TL', 'TR', 'BR', 'BL')[len(corners) - 1],
                          color='yellow', fontsize=9, zorder=10)
        block_accuracy_state['artists'].extend([dot, label])
        if len(corners) < 4:
            depth_text.set_text('Block accuracy: select ' + ('TL', 'TR', 'BR', 'BL')[len(corners)])
        else:
            try:
                plan = build_block_plan(corners, locked_L_clean.shape, BLOCK_ACCURACY_CONFIG)
            except ValueError as exc:
                print(f'[Block Accuracy] Invalid corners: {exc}; select all four again')
                block_accuracy_state['corners'] = []
                clear_block_accuracy_overlay()
                depth_text.set_text('Invalid quadrilateral. Select TL, TR, BR, BL again')
                request_blit_refresh()
                return
            block_accuracy_state['plan'] = plan
            block_accuracy_state['mode'] = 'preview'
            draw_block_accuracy_plan()
            btn_block_accuracy.label.set_text(f"開始 {len(plan['samples'])} 點統計")
            depth_text.set_text('Check grid / sample points on EACH block top.\n'
                                'Heights: row-major, in mm. Press Start or Cancel.\n'
                                'Stepped surfaces: four-corner mapping is approximate.')
            print('[Block Accuracy] Preview ready. Verify every grid cell and all sample centers; '
                  'an inset anchor does not guarantee its Region-SIFT support stays inside a block.')
        request_blit_refresh()

    def finish_block_accuracy(status='cancelled', reason='', redraw=True):
        timer = block_accuracy_state['timer']
        if timer is not None:
            timer.stop()
        report = block_accuracy_state['report']
        try:
            if report is not None and not report.closed:
                # Display data must not depend on successful disk export.
                blocks, _ = summarize_block_records(report.plan, report.records)
                block_accuracy_state['summaries'] = blocks
                blocks, summary = report.finish(status, reason)
                block_accuracy_state['summaries'] = blocks
                if redraw:
                    draw_block_accuracy_plan(blocks)
                print(f"\n[Block Accuracy] {status}: {report.directory}")
                print(f"  Valid={summary['valid_count']}/{summary['expected_count']}, "
                      f"failed={summary['failed_count']}, unmeasured={summary['unmeasured_count']}")
                for row in blocks:
                    values = (f"mean={row['mean_height_mm']:.3f} mm, "
                              f"error={row['mean_height_error_mm']:+.3f} mm"
                              if row['valid_count'] else 'mean=N/A, error=N/A')
                    print(f"  B{row['block_id']:02d} true={row['true_height_mm']:g} mm: "
                          f"{values}, valid={row['valid_count']}/{row['expected_count']}")
                metric = summary['block_mean_mae_mm']
                metric_text = f'{metric:.3f} mm' if metric is not None else 'N/A'
                depth_text.set_text(f"Block accuracy: {status}\n"
                                    f"Valid {summary['valid_count']}/{summary['expected_count']} | "
                                    f"Block-mean MAE: {metric_text}\n"
                                    f"Saved: {report.directory.name}")
            elif redraw:
                clear_block_accuracy_overlay()
                depth_text.set_text('Block accuracy cancelled')
        except Exception as exc:
            print(f'[Block Accuracy] Report finalization failed: {exc}; '
                  f'already flushed point data: {report.directory if report else "N/A"}')
            if redraw:
                depth_text.set_text(f'Block report save failed: {exc}')
        finally:
            for widget, was_active in block_accuracy_state['disabled_widgets']:
                widget.active = was_active
            block_accuracy_state['disabled_widgets'] = []
            block_accuracy_state['mode'] = 'idle'
            block_accuracy_state['timer'] = None
            btn_block_mae.set_active(True)
            btn_block_mae.eventson = True
            btn_block_accuracy.label.set_text('Block 誤差統計')
            if redraw:
                request_blit_refresh()

    def run_next_block_accuracy_point():
        if block_accuracy_state['mode'] != 'running':
            return
        report = block_accuracy_state['report']
        sample = report.plan['samples'][len(report.records)]
        started = time.perf_counter()
        try:
            # Match the ordinary left-click pixel rounding. CSV keeps BOTH
            # the projected grid center and the actual measured pixel.
            result = do_measure(int(round(sample['u'])), int(round(sample['v'])),
                                accuracy_batch=True)
        except Exception as exc:
            result = {'fail_reason': f'{type(exc).__name__}: {exc}',
                      'u': int(round(sample['u'])), 'v': int(round(sample['v']))}
        try:
            record = measurement_record(sample, result, time.perf_counter() - started)
            report.append(record)
        except Exception as exc:
            finish_block_accuracy('error', str(exc))
            return
        done, total = len(report.records), len(report.plan['samples'])
        scatter_A.set_offsets([[sample['u'], sample['v']]])
        elapsed = time.perf_counter() - block_accuracy_state['started']
        remaining = elapsed / done * (total - done)
        print(f"[Block Accuracy] {done}/{total}, B{sample['block_id']:02d}, "
              f"height={record['wound_height_mm']}, error={record['error_mm']}, {record['status']}")
        btn_block_accuracy.label.set_text(f'統計中 {done}/{total}')
        depth_text.set_text(f"Block accuracy {done}/{total} | B{sample['block_id']:02d}\n"
                            f"Height={record['wound_height_mm']} mm, error={record['error_mm']} mm\n"
                            f"ETA ~{remaining / 60:.1f} min | Cancel between points")
        request_blit_refresh()
        if done == total:
            finish_block_accuracy('completed')

    def on_block_accuracy(event):
        nonlocal auto_calc_active
        mode = block_accuracy_state['mode']
        if mode in ('running', 'picking'):
            return
        if mode == 'preview':
            n, c, source = get_selected_height_plane()
            if custom_plane_fitted:
                n, c, source = custom_plane_n, custom_plane_c, 'Custom Plane'
            if n is None or c is None:
                print('[Block Accuracy] No valid height reference plane; cancel and select a reference')
                return
            metadata = dict(video_path=str(VIDEO_PATH), measure_mode=MEASURE_MODE,
                            best_right_frame=current_cand['idx'],
                            right_frames=[cand['idx'] for cand in [current_cand] + extra_candidates_list],
                            height_reference_source=source, plane_normal=n, plane_center=c,
                            height_offset_mm=0.0 if custom_plane_fitted else DEFAULT_WOUND_HEIGHT_OFFSET_MM,
                            shared_pattern_plane_diag=shared_height_plane_diag,
                            options={key: value for key, value in view_state.items()
                                     if isinstance(value, (bool, int, float, str))},
                            region_sift_config=vars(REGION_SIFT_CONFIG),
                            camera_matrix_left=KL,
                            right_geometry=[{key: cand.get(key) for key in
                                             ('idx', 'K_R', 'R_rel', 't_rel', 'F', 'plane_n', 'plane_c')}
                                            for cand in [current_cand] + extra_candidates_list],
                            sample_coordinate_policy='projective grid centers rounded to nearest image pixel; no ArUco snapping',
                            error_definition='measured Wound Height minus true height (mm); no outlier removal',
                            std_definition='population standard deviation, valid points only')
            try:
                report = BlockAccuracyReport(BLOCK_ACCURACY_OUTPUT_DIR,
                                             block_accuracy_state['plan'], metadata)
            except Exception as exc:
                print(f'[Block Accuracy] Cannot create report: {exc}')
                return
            block_accuracy_state.update(mode='running', report=report, started=time.perf_counter())
            timer = fig.canvas.new_timer(interval=100)
            timer.add_callback(run_next_block_accuracy_point)
            block_accuracy_state['timer'] = timer
            timer.start()
            print(f'[Block Accuracy] Starting {len(report.plan["samples"])} points; output: {report.directory}')
            return
        if custom_plane_mode or view_state['manual'] or view_state.get('show_rt_warp_view'):
            print('[Block Accuracy] Exit Custom Plane picking, manual matching and RT Warp display before starting')
            return
        auto_calc_active = False
        btn_auto_calc.label.set_text('連續計算: 關')
        clear_block_accuracy_overlay()
        block_accuracy_state.update(mode='picking', corners=[], plan=None, report=None)
        block_accuracy_state.update(summaries=None, show_mae_colors=False)
        btn_block_mae.label.set_text('MAE 著色: Off')
        lock_block_accuracy_controls()
        btn_block_accuracy.label.set_text('左圖選四角…')
        depth_text.set_text('Block accuracy: select TL, TR, BR, BL on LEFT image.\n'
                            '6 columns x 4 rows; 12.5 to 1.0 mm, row-major.\n'
                            'Then inspect preview and press Start.')
        request_blit_refresh()

    def on_block_accuracy_close(event):
        if block_accuracy_state['mode'] != 'idle':
            finish_block_accuracy('cancelled', 'window closed', redraw=False)

    btn_block_accuracy.on_clicked(on_block_accuracy)
    btn_block_mae.on_clicked(on_block_mae_toggle)
    btn_region_replay.on_clicked(on_region_coordinate_replay)
    btn_region_settings.on_clicked(on_region_settings)
    btn_block_cancel.on_clicked(lambda event: finish_block_accuracy('cancelled', 'user cancelled')
                                if block_accuracy_state['mode'] != 'idle' else None)
    fig.canvas.mpl_connect('close_event', on_block_accuracy_close)

    btn_height_plane.on_clicked(on_height_plane_toggle)
    btn_shared_plane.on_clicked(on_shared_plane_toggle)
    btn_top2_geo.on_clicked(on_top2_geometry_toggle)
    btn_h_residual.on_clicked(on_h_residual_toggle)
    btn_rt_warp_view.on_clicked(on_rt_warp_view_toggle)

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
    btn_metric_blocks.on_clicked(on_metric_blocks_toggle)
    btn_return_menu.on_clicked(on_return_menu)

    def profile_notify(message):
        print(f'[Height Profile] {message}')
        depth_text.set_text(message)
        request_blit_refresh()

    def activate_height_profile():
        nonlocal auto_calc_active
        if (block_accuracy_state['mode'] != 'idle' or custom_plane_mode
                or view_state['manual'] or view_state.get('show_rt_warp_view')):
            profile_notify('Finish other selection modes and turn RT Warp off first.')
            return False
        if not view_state.get('region_sift', False):
            profile_notify('Enable Region-SIFT before starting a height profile.')
            return False
        n, c, _ = get_selected_height_plane()
        if not custom_plane_fitted and (n is None or c is None):
            profile_notify('A valid reference plane is required for Wound Height.')
            return False
        auto_calc_active = False
        btn_auto_calc.label.set_text('連續計算: 關')
        pan_state.update(pressing=False, dragging_hud=False)
        ax_B.set_visible(True)
        ax_debug_B.set_visible(True)
        ax_debug_info_B.set_visible(True)
        btn_hide_R.label.set_text('隱藏右圖')
        widgets = block_accuracy_control_widgets() + [
            btn_block_accuracy, btn_block_cancel, btn_block_mae, btn_region_diagnostics]
        profile_disabled_widgets[:] = [(widget, widget.active) for widget in widgets]
        for widget in widgets:
            widget.active = False
        return True

    def deactivate_height_profile():
        for widget, active in profile_disabled_widgets:
            widget.active = active
        profile_disabled_widgets.clear()

    def measure_profile_point(u, v):
        result = do_measure(u, v, accuracy_batch=True)
        if result is not None:
            # Freeze the plane used by candidate-height diagnostics with this point.
            show_region_diagnostics(result=result, display=False)
        return result

    def replay_profile_point(result):
        update_region_sift_debug_views(result, result['u'], result['v'])
        scatter_A.set_offsets([[result['u'], result['v']]])
        point = result.get('pt')
        scatter_B.set_offsets(np.asarray(point).reshape(1, 2)
                              if point is not None else np.empty((0, 2)))
        # Replay the stored candidate scores, geometry and plane; never remeasure.
        region_diagnostics.show(result)
        request_blit_refresh()

    height_profile = DragHeightProfile(
        fig, ax_A, ax_B, btn_height_profile, DRAG_PROFILE_CONFIG,
        activate=activate_height_profile, deactivate=deactivate_height_profile,
        measure=measure_profile_point, replay=replay_profile_point,
        notify=profile_notify, refresh=request_blit_refresh,
        image_shape=lambda: locked_L_clean.shape)
    # ---- 三區按鈕顯示/隱藏控制：左上角三個圓點，預設全部隱藏（返回主選單與自訂傷口平面不受影響）----
    panel_defs = [
        ('#00BFFF', [ax_c1, ax_c2, ax_c18, ax_c3, ax_c4, ax_c5, ax_c6, ax_c7, ax_c8, ax_c9,
                     ax_c10, ax_c11, ax_c12, ax_c13, ax_c14, ax_c15, ax_c16, ax_c17, ax_c19,
                     ax_region_u, ax_region_v, ax_btn_region_replay, ax_btn_region_settings,
                     ax_btn_region_diagnostics, ax_btn_height_profile]),
        ('#00FF88', [ax_mode]),
        ('#FFAA00', [ax_btn_lock_L, ax_btn_lock_R, ax_btn_hide_R, ax_btn_norm,
                     ax_btn_calc, ax_btn_auto_calc, ax_btn_grad,
                     ax_btn_high_grad_pts, ax_btn_mid_grad_pts, ax_btn_rt_diff,
                     ax_btn_wound, ax_btn_wound_pts, ax_btn_aruco_overlay, ax_btn_rt_sift,
                     ax_btn_height_plane, ax_btn_metric_blocks, ax_btn_shared_plane,
                     ax_btn_top2_geo, ax_btn_h_residual, ax_btn_rt_warp_view,
                     ax_btn_block_accuracy, ax_btn_block_cancel, ax_btn_block_mae]),
        ('#FF6688', [pose_status_text]),  # 右下角姿態估計狀態 label (set_visible 對 Text artist 同樣有效)
    ]
    panel_visible = [True, True, True, True]
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
    if ENABLE_WOUND_AI:
        refresh_wound_predictions("initial selection")
        startup_timer.stage("傷口模型預載+推論")
    else:
        log_and_print("ℹ️ [Wound] AI disabled: model loading and inference skipped")
        startup_timer.stage("傷口 AI 停用")

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
            view_state.get('show_rt_warp_view', False),
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
            # Warp 模式只替換右上「顯示影像」。刻意放在所有正常疊圖之後，
            # 避免把仍位於原始右圖座標的量測點/傷口框誤畫到左圖座標系。
            if (view_state.get('show_rt_warp_view', False)
                    and rt_warp_view_state.get('frame_rgb') is not None):
                disp_B = rt_warp_view_state['frame_rgb'].copy()
            display_cache['key'] = disp_key
            display_cache['disp_A'] = disp_A
            display_cache['disp_B'] = disp_B

        im_A.set_data(display_cache['disp_A'])
        im_B.set_data(display_cache['disp_B'])

        # ---- Blit 渲染 ----
        if blit_state['needs_refresh'] or blit_state['bg'] is None:
            # 右圖的點、線、ArUco 及傷口量測 artist 都使用原始右圖座標。
            # 建立 warp 模式背景時暫時隱藏，畫完即恢復其內部狀態，關閉
            # warp 後所有顯示選項仍能原樣回來。
            hidden_right_artists = []
            if view_state.get('show_rt_warp_view', False):
                right_overlay_artists = (
                    list(ax_B.collections)
                    + list(ax_B.lines)
                    + list(ax_B.texts)
                    + [patch for patch in ax_B.patches if patch is not border_B]
                )
                seen_artist_ids = set()
                for artist in right_overlay_artists:
                    artist_id = id(artist)
                    if artist_id in seen_artist_ids:
                        continue
                    seen_artist_ids.add(artist_id)
                    hidden_right_artists.append((artist, artist.get_visible()))
                    artist.set_visible(False)
            try:
                fig.canvas.draw()
                blit_state['bg'] = fig.canvas.copy_from_bbox(fig.bbox)
            finally:
                for artist, was_visible in hidden_right_artists:
                    artist.set_visible(was_visible)
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
        for artist in block_accuracy_state['artists']:
            ax_A.draw_artist(artist)
        for a in custom_plane_artists:
            ax_A.draw_artist(a)
        
        if ax_B.get_visible():
            ax_B.draw_artist(im_B)
            if not view_state.get('show_rt_warp_view', False):
                ax_B.draw_artist(scatter_B)
                ax_B.draw_artist(scatter_B_reproj)
                ax_B.draw_artist(scatter_grad_ref_B)
                ax_B.draw_artist(scatter_mid_grad_ref_B)
                ax_B.draw_artist(scatter_grad_match)
                ax_B.draw_artist(scatter_mid_grad_match)
                ax_B.draw_artist(scatter_rt_sift_B)
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
                if (ax == ax_B and (
                        not ax_B.get_visible()
                        or view_state.get('show_rt_warp_view', False))):
                    continue
                for a in ax.art:
                    ax.draw_artist(a)
            if hasattr(ax, 'reproj_art'):
                if (ax == ax_B and (
                        not ax_B.get_visible()
                        or view_state.get('show_rt_warp_view', False))):
                    continue
                for a in ax.reproj_art:
                    ax.draw_artist(a)

        ax_A.draw_artist(fps_text)
        fig.draw_artist(pose_status_text)
        height_profile.draw_overlays()
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
