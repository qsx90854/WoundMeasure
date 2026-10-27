# -*- coding: utf-8 -*-
"""Region-SIFT 精簡版量測核心（無 UI）。

對外 demo 用：只保留 Region-SIFT 量測流程，結果與主程式逐位元相同。
主程式（下稱 MAIN）= depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py。
演算法仍以 MAIN 為主體開發；本檔不 import MAIN（避免載入 onnxruntime 等），
而是把 Region-SIFT 生產路徑需要的頂層函式／常數「複製」過來，每段註明來源 MAIN 行號
（2026-10-06 版本）。MAIN 的這些區段若有改動，必須同步改本檔，見 README_regionsift_lite.md。

Session 與點擊量測流程改寫自 D:\\Lightglue\\RegionSIFT_Cpp\\golden\\export_golden.py
（已用 GUI 量測紀錄驗證可逐位元重現 MAIN）。Algorithm/ 下的檔案只 import、不修改。

用法（無 GUI）：
    import regionsift_lite_core as core
    session = core.build_session(video_path)
    res = core.measure_click(session, u, v, core.REGION_SIFT_CONFIG)
"""
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Algorithm import aruco_pose as aruco_algo
from Algorithm.aruco_id_filter import filter_marker_mapping
from Algorithm import camera_preprocess as camera_algo
from Algorithm import video_pose_analysis_temporal_unified_pattern_guided_local_window as video_pose_algo
from Algorithm.specular_detection import (
    BlockSpatialSpecularConfig,
    compute_specular_mask_bgr_block_adaptive,
)
from Algorithm import stereo_matching as stereo_algo
from Algorithm.Region_SIFT_Matching import (
    RegionSIFTConfig,
    run_region_sift_matching,
)
from Algorithm.stereo_matching import get_patch, score_patch_match

# ===========================================================================
# 常數（複製自 MAIN 全局設定區，MAIN:77-303；只留生產路徑會用到的）
# ===========================================================================
# MAIN:81 RT 與參考平面的 marker 白名單（None=全部）
ARUCO_ALLOWED_PATTERN_IDS = None
# MAIN:92-106 自製遮罩模式用的 block-adaptive 空間反光遮罩設定
SPATIAL_BLOCK_CONFIG = BlockSpatialSpecularConfig(
    block_width_px=64,
    block_height_px=64,
    v_percentile=10,
    rgb_percentile=5.0,
    local_hot_percentile=10,
    interpolate_thresholds=True,
    dilate_px=0,
    enable_prominence_gate=True,
    prominence_min=3.0,
    prominence_mad_multiplier=2.0,
    enable_strong_highlight_exception=True,
    strong_v_min=250.0,
    strong_s_max=30.0,
    strong_whiteness_max=20.0,
)
# MAIN:116-128 影片分析
RECORD_SAVE_DIR = "test_video_Zebra"
START_FRAME_COUNT = 30
END_FRAME_COUNT = 30
FRAME_RANGE_MODE = "half_half"
# MAIN:131-139 選幀模式（angle_guided 關閉，仍要傳設定給 POSE）
FRAME_PAIR_SELECTION_MODE = "original"
ANGLE_GUIDED_DIRECTION_MODE = "auto"
ANGLE_GUIDED_RIGHT_TARGET_DEG = 15.0
ANGLE_GUIDED_LEFT_TARGET_DEG = 35.0
ANGLE_GUIDED_TARGET_TOLERANCE_DEG = 6.0
ANGLE_GUIDED_COARSE_SAMPLES_PER_SEGMENT = 12
ANGLE_GUIDED_MAX_SCAN_FRAMES_PER_SEGMENT = 18
ANGLE_GUIDED_CANDIDATES_PER_SIDE = 3
ANGLE_GUIDED_PAIR_SCORE_WEIGHT = 0.35
# MAIN:142-148 RT SIFT ROI 與世界 marker+SIFT 聯合精修
ENABLE_RT_SIFT_ROI = True
RT_SIFT_ROI_RATIO = (0.10, 0.10, 0.80, 0.80)
RT_SIFT_IMAGE_SCALE = 0.5
ENABLE_WORLD_MARKER_SIFT_RT_REFINE = True
# MAIN:152 位姿選擇
POSE_SELECT_MODE = "reproj_min"
# MAIN:161-162 融合一致性閘門
FUSE_DEPTH_TOL_REL = 0.03
FUSE_DEPTH_TOL_ABS_MM = 5.0
# MAIN:167-173 標定、標籤、深度與基線限制
PARAMS_JSON_PATH = "calibration_result_Zebra_1_monocular.json"
ACTUAL_MARKER_SIZE_MM = 8.25
MAX_DEPTH_MM = 2000
DEFAULT_WOUND_HEIGHT_OFFSET_MM = 0.0
MIN_BASELINE_MM = 35.0
MAX_BASELINE_MM = 220.0
# MAIN:180-182 CLAHE（Demo 固定關閉，常數只用於 preprocess_gray(…, False)）
ENABLE_CLAHE_DEFAULT = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# MAIN:199-213 stereo_matching 可調常數（Region-SIFT 路徑不使用，但 MAIN 會注入，照做以保持一致）
LEFT_PATCH_SEARCH_RADIUS = 30
RIGHT_PATCH_SEARCH_RADIUS = 75
GRAD_SIFT_MAX_RT_ADJUST_PX = 75
LEFT_GRADIENT_POINTS_COUNT = 150
RIGHT_GRADIENT_POINTS_COUNT = 500
LEFT_MID_GRADIENT_POINTS_COUNT = 150
RIGHT_MID_GRADIENT_POINTS_COUNT = 500
GRAD_SIFT_RATIO_TEST = 0.78
GRAD_SIFT_EPIPOLAR_TOL_PX = 3.0
GRAD_SIFT_OFFSET_MEDIAN_TOL_PX = 8.0
GRAD_SIFT_RANSAC_REPROJ_PX = 2.5
GRAD_SIFT_MIN_GROUP_INLIERS = 3
GRAD_SIFT_GUIDED_RADIUS_PX = 10.0
GRAD_SIFT_GUIDED_RATIO_TEST = 0.95
EPIPOLAR_SEARCH_HALF_LEN = 55
EPIPOLAR_SEARCH_BAND_RADIUS = 2
EPIPOLAR_SEARCH_MIN_SCORE = 0.35
EPIPOLAR_SEARCH_DESC_OK = 0.30
EPIPOLAR_SEARCH_ZNCC_OK = 0.15
EPIPOLAR_SEARCH_DESC_STRONG = 0.50
EPIPOLAR_SEARCH_ZNCC_STRONG = 0.35
EPIPOLAR_SEARCH_WEAK_FLOOR = 0.05
# MAIN:297-302 POSE 選配對權重
IDEAL_BASELINE_MM = 45.0
PAIR_SCORE_REPROJ_W = 1.00
PAIR_SCORE_BASELINE_W = 0.18
PAIR_SCORE_BLUR_W = 0.18
PAIR_SCORE_COVER_W = 0.12
PAIR_SCORE_MARKER_W = 0.08

# MAIN:222-277 Region-SIFT 生產參數（Demo 以此為預設，UI 可於執行期修改）
REGION_SIFT_CONFIG = RegionSIFTConfig(
    two_stage_search=True,
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
    descriptor_max_support_radius_px=40.0,
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
    frame_coordinate_quantization_px=0.0,
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
    max_group_score=500.0,
    max_objective_score_ratio=0.95,
    epipolar_penalty_weight=0.0,
    second_best_exclusion_radius_px=3.0,
    group_balance_weight=0.35,
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
    specular_check_support=False,
)

# MAIN:306-307 把可調常數注入 stereo_matching 模組（與 MAIN 相同）
for _const_name in stereo_algo.TUNABLE_CONSTANTS:
    setattr(stereo_algo, _const_name, globals()[_const_name])

# 日誌：MAIN 以 log_and_print 收集；Demo 預設只 print，可由 UI 換成自訂函式
_LOG_FN = print


def set_log_fn(fn):
    global _LOG_FN
    _LOG_FN = fn if fn is not None else (lambda *_a, **_k: None)


def log_and_print(msg):
    _LOG_FN(str(msg))


# ===========================================================================
# 頂層函式（複製自 MAIN）
# ===========================================================================
def compute_fundamental_matrix(K_L, K_R, R_rel, t_rel):  # MAIN:534-539
    t = t_rel.flatten()
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    E = tx @ R_rel
    K_R_inv, K_L_inv = np.linalg.inv(K_R.astype(np.float64)), np.linalg.inv(K_L.astype(np.float64))
    return K_R_inv.T @ E @ K_L_inv


def triangulate_point_3d(pt_A, pt_B, K_L, K_R, R_rel, t_rel, F=None):  # MAIN:541-561
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
    # 強制使用 float32，避免 OpenCV 在處理整數點陣列時發生隱性記憶體錯亂
    ptsA_f32 = np.array([[pt_A[0]], [pt_A[1]]], dtype=np.float32)
    ptsB_f32 = np.array([[pt_B[0]], [pt_B[1]]], dtype=np.float32)
    pts4d = cv2.triangulatePoints(P0, P1, ptsA_f32, ptsB_f32)
    pt3d = (pts4d[:3] / pts4d[3]).flatten()
    return pt3d


def epipolar_line(F, pt, img_w):  # MAIN:563-567（UI 畫極線用）
    l = F @ np.array([pt[0], pt[1], 1.0])
    a, b, c = l
    if abs(b) > 1e-8:
        return (0, int(-c / b)), (img_w - 1, int(-(a * (img_w - 1) + c) / b))
    return (int(-c / a), 0), (int(-c / a), img_w - 1)


def average_rotations_svd(R_list):  # MAIN:569-583（POSE 注入用）
    """對多個 3x3 旋轉矩陣進行 SVD 平均，獲得在 SO(3) 群上的正交最小二乘平均矩陣"""
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


def fuse_candidate_results(res_list, best_idx=None):  # MAIN:604-666
    """
    多候選影格量測結果融合：
    1. 候選一致性閘門：與最優對 (best_idx) 深度差超過容許值
       max(FUSE_DEPTH_TOL_ABS_MM, FUSE_DEPTH_TOL_REL * z_best) 的候選不參與融合
    2. 深度中位數 + MAD 剔除離群 (|z - med| > 3 * 1.4826 * MAD, n>=3)
    3. 依 (baseline / z^2)^2 加權平均
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


def compute_shared_marker_corner_plane(corners_left, corners_right,  # MAIN:703-837
                                       K_L, K_R, R_rel, t_rel, F=None,
                                       reference_normal=None,
                                       min_shared_markers=2):
    """Triangulate every valid corner of shared markers and fit one 3D plane.

    The returned plane is expressed in the left-camera coordinate system.  A
    marker only participates when all four corresponding corners triangulate
    to finite points in front of both cameras (no point-level RANSAC).
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


def preprocess_gray(gray_img, enable_clahe=True):  # MAIN:1274-1275
    return camera_algo.preprocess_gray(gray_img, enable_clahe, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)


def compute_global_plane(imgA_gray, K_L, marker_size_mm):  # MAIN:1277-1279
    return aruco_algo.compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=log_and_print,
                                           allowed_marker_ids=ARUCO_ALLOWED_PATTERN_IDS)


def configure_world_marker_sift_rt_refine():  # MAIN:935-1212 中影響數值的部分
    """MAIN 版本在 POSE 的世界最佳化器外包一層「稽核 wrapper」：
    enabled 時直接回傳原最佳化器結果的淺拷貝（只多印 log、多做一次不回饋的 marker-only 稽核解算），
    因此對 R/t 沒有數值影響；真正影響數值的只有下面這一行旗標。
    本檔省略稽核 wrapper（最佳化器為純函式，已用逐位元驗證確認等價）。
    """
    enabled = bool(ENABLE_WORLD_MARKER_SIFT_RT_REFINE)
    if hasattr(video_pose_algo, 'ENABLE_FEATURE_RT_REFINE'):
        video_pose_algo.ENABLE_FEATURE_RT_REFINE = enabled          # MAIN:952-953


def analyze_video_frames(video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm,
                         select_mode="average", range_mode="fixed", progress_callback=None):  # MAIN:1215-1267
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
    feature_roi_ratio = (RT_SIFT_ROI_RATIO if bool(ENABLE_RT_SIFT_ROI) else None)
    selection_mode = str(FRAME_PAIR_SELECTION_MODE).strip().lower()
    if selection_mode not in ("original", "angle_guided"):
        raise ValueError("FRAME_PAIR_SELECTION_MODE must be 'original' or 'angle_guided'")
    angle_guided_config = {
        'enabled': selection_mode == "angle_guided",
        'direction_mode': str(ANGLE_GUIDED_DIRECTION_MODE),
        'normalize_output_roles': True,
        'target_frame_A_deg': float(ANGLE_GUIDED_RIGHT_TARGET_DEG),
        'target_frame_B_deg': float(ANGLE_GUIDED_LEFT_TARGET_DEG),
        'target_tolerance_deg': float(ANGLE_GUIDED_TARGET_TOLERANCE_DEG),
        'coarse_samples_per_segment': int(ANGLE_GUIDED_COARSE_SAMPLES_PER_SEGMENT),
        'max_scan_frames_per_segment': int(ANGLE_GUIDED_MAX_SCAN_FRAMES_PER_SEGMENT),
        'candidates_per_side': int(ANGLE_GUIDED_CANDIDATES_PER_SIDE),
        'pair_score_weight': float(ANGLE_GUIDED_PAIR_SCORE_WEIGHT),
    }
    return video_pose_algo.analyze_video_frames(
        video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm,
        select_mode, range_mode, progress_callback=progress_callback,
        detection_roi_ratio=None,
        feature_roi_ratio=feature_roi_ratio,
        pair_geometry_roi_ratio=None,
        angle_guided_config=angle_guided_config,
        allowed_marker_ids=ARUCO_ALLOWED_PATTERN_IDS
    )


def snap_to_aruco_corner(x, y, corners_dict):  # MAIN:1624-1630（UI 點擊吸附）
    pt = np.array([x, y])
    for corners in corners_dict.values():
        dists = np.linalg.norm(corners - pt, axis=1)
        if np.min(dists) < 15:
            return float(corners[np.argmin(dists)][0]), float(corners[np.argmin(dists)][1])
    return x, y


def ui_failure_reason_english(reason):  # MAIN:2173-2188
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


def ui_progress_status_english(status_text):  # MAIN:2154-2171
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


# ===========================================================================
# Session 層：改寫自 export_golden.build_session（MAIN main() 2476-2729）
# ===========================================================================
def build_session(video_path, progress_callback=None, analysis_copy_dir=None):
    """影片 -> POSE 分析 -> 去畸變 -> F -> 高度平面。回傳 session dict。

    analysis_copy_dir: 若指定，先把影片複製到該資料夾再分析，POSE 的診斷檔
        (<影片名>_rt_sift_diagnostics.txt) 就寫在那裡而不是影片旁（驗證腳本用；
        不影響數值）。None＝與 MAIN 相同，診斷檔寫在影片旁。
    """
    video_path = str(video_path)
    params_path = str(BASE_DIR / PARAMS_JSON_PATH)
    mtxL_o, distL, _mtxR, _distR, _ext, _F = camera_algo.load_json_camera_params(params_path)
    if mtxL_o is None or distL is None:
        raise RuntimeError(f'無法載入相機標定參數: {params_path}')
    cap = cv2.VideoCapture(video_path)
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'無法讀取影片: {video_path}')
    h_raw, w_raw = first_frame.shape[:2]
    newKL_o, _m1, _m2, process_view = camera_algo.build_undistort_processor(
        mtxL_o, distL, (w_raw, h_raw), alpha=1.0)                            # MAIN:2491
    KL = newKL_o.copy().astype(np.float64)                                    # MAIN:2494

    analysis_path = video_path
    if analysis_copy_dir is not None:
        Path(analysis_copy_dir).mkdir(parents=True, exist_ok=True)
        analysis_path = str(Path(analysis_copy_dir) / Path(video_path).name)
        shutil.copy2(video_path, analysis_path)
    log_and_print("[Session] 分析影片中開頭與結尾影格的 ArUco 標籤與最優姿態對...")
    video_data = analyze_video_frames(                                        # MAIN:2499
        analysis_path, START_FRAME_COUNT, END_FRAME_COUNT, KL, distL, mtxL_o,
        ACTUAL_MARKER_SIZE_MM, POSE_SELECT_MODE, FRAME_RANGE_MODE,
        progress_callback=progress_callback)
    if video_data is None:
        raise RuntimeError('影片 Pose 分析失敗（偵測不到足夠的 ArUco 標籤或配對）')

    imgA_bgr, _ = process_view(video_data['frame_B'])                         # MAIN:2566 左圖
    imgB_bgr, _ = process_view(video_data['frame_A'])                         # MAIN:2567 右圖
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, ENABLE_CLAHE_DEFAULT)              # MAIN:2570

    R_r, t_r = video_data['R_rel'], video_data['t_rel']
    global_plane_n = video_data.get('global_plane_n')                         # MAIN:2596-2599
    global_plane_c = video_data.get('global_plane_c')
    if global_plane_n is None or global_plane_c is None:
        global_plane_n, global_plane_c = compute_global_plane(imgA_gray, KL, ACTUAL_MARKER_SIZE_MM)
    legacy_n = None if global_plane_n is None else np.asarray(global_plane_n, np.float64).reshape(3).copy()
    legacy_c = None if global_plane_c is None else np.asarray(global_plane_c, np.float64).reshape(3).copy()

    current_cand = {                                                          # MAIN:2641-2660
        'idx': video_data['idx_B'],
        'rgb': cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB),
        'gray': cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY),
        'bgr': imgB_bgr,
        'K_R': KL, 'R_rel': R_r, 't_rel': t_r,
        'F': compute_fundamental_matrix(KL, KL, R_r, t_r),
        'cornersA': video_data['cornersB'], 'cornersB': video_data['cornersA'],
        'plane_n': global_plane_n, 'plane_c': global_plane_c,
        'pose_valid': True, 'baseline': video_data['baseline'],
        'right_frame_idx': video_data['idx_A'],
    }
    shared_n, shared_c, shared_diag = compute_shared_marker_corner_plane(     # MAIN:2669-2675
        current_cand.get('cornersA'), current_cand.get('cornersB'),
        KL, current_cand['K_R'], current_cand['R_rel'], current_cand['t_rel'],
        F=current_cand.get('F'), reference_normal=legacy_n, min_shared_markers=2)
    if shared_diag.get('available'):
        log_and_print(f"[Shared Pattern Plane] ready; used={shared_diag['used_marker_ids']}, "
                      f"RMS={shared_diag['rms_mm']:.3f} mm")
    else:
        log_and_print("[Shared Pattern Plane] unavailable: " + shared_diag.get('reason', 'unknown reason')
                      + "（高度改用 Legacy Plane）")

    cands = [current_cand]
    for extra in video_data.get('extra_candidates', []):                      # MAIN:2705-2729
        imgB_extra_bgr, _ = process_view(extra['frame_A'])
        cands.append({
            'idx': extra['idx_A'],
            'rgb': cv2.cvtColor(imgB_extra_bgr, cv2.COLOR_BGR2RGB),
            'gray': cv2.cvtColor(imgB_extra_bgr, cv2.COLOR_BGR2GRAY),
            'bgr': imgB_extra_bgr,
            'K_R': KL, 'R_rel': extra['R_rel'], 't_rel': extra['t_rel'],
            'F': compute_fundamental_matrix(KL, KL, extra['R_rel'], extra['t_rel']),
            'cornersA': video_data['cornersB'], 'cornersB': extra['cornersA'],
            'plane_n': global_plane_n, 'plane_c': global_plane_c,
            'pose_valid': True, 'baseline': extra['baseline'],
            'right_frame_idx': extra['idx_A'],
        })

    # 高度平面選擇（MAIN get_selected_height_plane:3359-3378，預設 use_shared_plane=True）
    if shared_n is not None and shared_c is not None:
        height_plane = (shared_n, shared_c, 'Shared Pattern Plane')
    else:
        height_plane = (legacy_n, legacy_c, 'Legacy Plane')

    return {
        'video_path': video_path, 'KL': KL, 'K_raw': mtxL_o, 'dist': distL,
        'image_size': (w_raw, h_raw), 'left_bgr': imgA_bgr, 'left_gray': imgA_gray,
        'left_frame_idx': video_data['idx_B'], 'cands': cands,
        'legacy_plane': (legacy_n, legacy_c),
        'shared_plane': (shared_n, shared_c), 'shared_plane_diag': shared_diag,
        'height_plane': height_plane,
        'left_mask': None,   # 自製遮罩模式第一次用到時才計算（惰性）
    }


def ensure_spatial_masks(session, cand):
    """自製遮罩式 descriptor 用的空間遮罩（惰性計算並快取）。

    MAIN:2525-2563、4744-4758：預設 ENABLE_WOUND_AI=False、SPECULAR_MODEL_V2 關閉，
    遮罩即 compute_specular_mask_bgr_block_adaptive(bgr, SPATIAL_BLOCK_CONFIG)。
    """
    if session.get('left_mask') is None:
        session['left_mask'] = compute_specular_mask_bgr_block_adaptive(
            session['left_bgr'], SPATIAL_BLOCK_CONFIG)
    if cand.get('spec_spatial_mask') is None:
        cand['spec_spatial_mask'] = compute_specular_mask_bgr_block_adaptive(
            cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR), SPATIAL_BLOCK_CONFIG)
    return session['left_mask'], cand['spec_spatial_mask']


# ===========================================================================
# Click 層：MAIN compute_measure（4687-5326）Region-SIFT 分支 + do_measure（5445-5629）
# ===========================================================================
def compute_measure(session, cand, u, v, config, left_cache):
    KL = session['KL']
    out = {'pt': None, 'pt_raw': None, 'p3d': None, 'p3d_w': None, 'd': None, 'depth': None,
           'error': None, 'method': '', 'fail_reason': '', 'u': u, 'v': v,
           'd_epi': 999.0, 'zncc_score': 0.0, 'masked_score': -1.0, 'confidence_score': 0.0,
           'reject_reason': None,
           'cand_idx': cand['idx'], 'right_frame_idx': cand['right_frame_idx'],
           'baseline': cand.get('baseline')}
    bsl = cand.get('baseline', 0.0)                                           # MAIN:4726-4733
    if bsl < MIN_BASELINE_MM or bsl > MAX_BASELINE_MM:
        out['fail_reason'] = f'視差不合規範({MIN_BASELINE_MM}~{MAX_BASELINE_MM}mm)'
        return out

    if config.use_masked_sift:                                                # MAIN:4744-4758
        left_gray = cv2.cvtColor(session['left_bgr'], cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2GRAY)
        left_mask, right_mask = ensure_spatial_masks(session, cand)
    else:                                                                     # MAIN:4760-4763
        left_gray = session['left_gray']
        right_gray = preprocess_gray(cand['gray'], False)
        left_mask = None   # reject_specular=False 時 Region_SIFT_Matching 內部也會丟棄
        right_mask = None
    region_result = run_region_sift_matching(                                 # MAIN:4766-4771
        left_gray, right_gray, (u, v), cand, KL, sift=None, config=config,
        left_cache=left_cache, reject_specular=False,
        left_spec_mask=left_mask, right_spec_mask=right_mask)
    m_pt = region_result.get('m_pt')                                          # MAIN:4863
    method = region_result.get('method', '')
    reject_reason = region_result.get('reject_reason')
    m_pt_raw = m_pt.copy() if m_pt is not None else None                      # MAIN:5050
    out.update(pt=m_pt, pt_raw=m_pt_raw, method=method, reject_reason=reject_reason)

    p3d = None
    fail_reason = ''
    d_val = p3d_val = p3d_w_val = None
    if m_pt is not None:                                                      # MAIN:5216-5223
        p3d = triangulate_point_3d((u, v), m_pt, KL, cand['K_R'], cand['R_rel'],
                                   cand['t_rel'], F=cand.get('F'))
    else:
        fail_reason = reject_reason or "無匹配點"                              # MAIN:5224-5226
    if p3d is not None:                                                       # MAIN:5229-5263
        if p3d[2] <= 0:
            fail_reason = "深度為負(在相機後方)"
        elif p3d[2] > MAX_DEPTH_MM:
            fail_reason = "超過最大深度"
        else:
            d_val = np.linalg.norm(p3d)
            p3d_val = p3d
            if cand.get('cornersA'):                                          # 世界座標：ID 最小的標籤為原點
                min_id = min(cand['cornersA'].keys())
                half = ACTUAL_MARKER_SIZE_MM / 2.0
                canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0],
                                  [-half, -half, 0]], dtype=np.float32)
                ok_origin, rv_o, tv_o = cv2.solvePnP(
                    canon, cand['cornersA'][min_id], KL, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok_origin:
                    R_o, _ = cv2.Rodrigues(rv_o)
                    p3d_w = R_o.T @ (p3d_val.reshape(3, 1) - tv_o.reshape(3, 1))
                    p3d_w_val = p3d_w.flatten()
                out['world_origin_marker_id'] = int(min_id)

    reproj_err = None                                                         # MAIN:5270-5276
    if p3d_val is not None and m_pt is not None:
        rvec_rel, _ = cv2.Rodrigues(cand['R_rel'])
        pt_reproj, _ = cv2.projectPoints(p3d_val.reshape(1, 1, 3), rvec_rel, cand['t_rel'], KL, np.zeros(5))
        reproj_err = float(np.linalg.norm(m_pt - pt_reproj.reshape(2)))

    d_epi, zncc_score, masked_score, confidence = 999.0, 0.0, -1.0, 0.0      # MAIN:5280-5302
    if p3d_val is not None and m_pt is not None:
        if m_pt_raw is not None:
            l_B = cand['F'] @ np.array([u, v, 1.0])
            denom = l_B[0] ** 2 + l_B[1] ** 2
            if denom > 1e-9:
                d_epi = float(abs(l_B[0] * m_pt_raw[0] + l_B[1] * m_pt_raw[1] + l_B[2]) / np.sqrt(denom))
        tmpl = get_patch(left_gray, (u, v), 31)
        roi = get_patch(cand['gray'], m_pt, 31)
        if tmpl is not None and roi is not None:
            zncc_score = float(cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        masked_score = score_patch_match(left_gray, cand['gray'], (u, v), m_pt, patch_size=31)
        sigma = 1.5
        geom = np.exp(-(d_epi ** 2) / (2.0 * sigma ** 2)) if d_epi != 999.0 else 0.0
        confidence = float(max(0.0, zncc_score) * max(0.0, masked_score) * geom)

    out.update(p3d=p3d_val, p3d_w=p3d_w_val, d=d_val, depth=None if p3d_val is None else p3d_val[2],
               error=reproj_err, fail_reason=fail_reason,
               d_epi=d_epi, zncc_score=zncc_score, masked_score=masked_score,
               confidence_score=confidence)
    return out


def measure_click(session, u, v, config):
    """點擊左圖 (u, v)（呼叫端須先 int(round()) 成整數像素）→ 多配對量測、融合、高度。

    回傳 res（鍵與 MAIN do_measure 的 res 相容），另含 'per_pair'（每個右圖配對的結果）。
    """
    click_left_cache = {}                                                     # MAIN:5490
    cands = session['cands']
    per_pair, res_list = [], []
    for cand in cands:                                                        # MAIN:5491-5506
        res_c = compute_measure(session, cand, u, v, config, click_left_cache)
        per_pair.append(res_c)
        if res_c.get('d') is not None and res_c.get('p3d') is not None:
            res_list.append(res_c)
    best = per_pair[0]
    res = dict(best)                                                          # MAIN:5514-5515
    res['p3d_best'] = res.get('p3d')
    res['fused_count'] = 0
    res['fused_dropped'] = []
    if res_list:                                                              # MAIN:5516-5538
        fused = fuse_candidate_results(res_list, best_idx=cands[0]['idx'])
        avg_p3d = fused['p3d']
        kept = fused['kept']
        res['multi_res'] = res_list
        res['multi_avg_p3d'] = avg_p3d
        res['multi_avg_d'] = fused['d']
        res['multi_avg_depth'] = avg_p3d[2]
        res['multi_avg_p3d_w'] = fused['p3d_w']
        res['p3d'] = avg_p3d
        res['d'] = fused['d']
        res['depth'] = avg_p3d[2]
        if fused['p3d_w'] is not None:
            res['p3d_w'] = fused['p3d_w']
        res['fused_count'] = len(kept)
        res['fused_dropped'] = [int(r['cand_idx']) for r in fused['dropped']]
        res['fused_weights'] = np.asarray(fused['weights'], np.float64)
        res['fused_kept_idx'] = [int(r['cand_idx']) for r in kept]
    res['shared_pattern_plane_diag'] = session.get('shared_plane_diag')       # MAIN:5607-5629
    res['height_reference_source'] = None
    res['height_reference_signed_mm'] = None
    res['height_display_mm'] = None
    if res.get('p3d') is not None:
        n, c, source = session['height_plane']
        if n is not None and c is not None:
            hp = res['p3d_best'] if res.get('p3d_best') is not None else res['p3d']
            res['height_reference_source'] = source
            res['height_reference_signed_mm'] = float(np.dot(n, hp - c))
            res['height_display_mm'] = res['height_reference_signed_mm'] - DEFAULT_WOUND_HEIGHT_OFFSET_MM
    res['per_pair'] = per_pair
    return res


# ===========================================================================
# 量測紀錄：MAIN save_measurement_to_txt（1967-2152），移除自訂平面／block audit 以外的欄位照舊
# ===========================================================================
def record_txt_path(video_path):
    return os.path.splitext(str(video_path))[0] + ".txt"


def save_measurement_to_txt(video_path, res, cand, measure_mode="dual_direct", txt_path=None):
    """在影片旁同名 .txt 追加一筆紀錄（格式同 MAIN）。txt_path 可覆寫輸出位置（自測用）。
    wound_z_offset 固定 0.0、自訂平面固定「未擬合」（Demo 無自訂平面）。回傳實際寫入路徑或 None。
    """
    import datetime
    if video_path is None:
        return None
    txt_path = txt_path or record_txt_path(video_path)
    wound_z_offset = 0.0
    custom_plane_fitted = False

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

    p_dist_str = "None"
    if p3d_best is not None and cand.get('plane_n') is not None and cand.get('plane_c') is not None:
        p_dist = np.dot(cand['plane_n'], p3d_best - cand['plane_c'])
        p_dist_str = f"{'Above' if p_dist > 0 else 'Below'} {abs(p_dist):.2f} mm"
    cp_dist_str = "None"

    R_rel = cand.get('R_rel')
    t_rel = cand.get('t_rel')
    baseline = cand.get('baseline')
    idx_B = cand.get('idx')
    R_rel_str = np.array2string(R_rel, precision=6, separator=', ', suppress_small=True) if R_rel is not None else "None"
    t_rel_str = np.array2string(t_rel.flatten(), precision=6, separator=', ', suppress_small=True) if t_rel is not None else "None"
    K_R = cand.get('K_R')
    K_R_str = np.array2string(K_R, precision=6, separator=', ', suppress_small=True) if K_R is not None else "None"

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
    if "multi_res" in res:
        text_lines.append("--------------------------------------------------")
        text_lines.append(f"多對融合結果 (共 {len(res['multi_res'])} 組成功):")
        for sub in res['multi_res']:
            is_best = " (最優)" if sub['cand_idx'] == cand.get('idx') else ""
            text_lines.append(f"  - 右圖 F{sub['cand_idx']}{is_best}: 深度 = {sub['d']:.2f} mm, 3D = [{sub['p3d'][0]:.2f}, {sub['p3d'][1]:.2f}, {sub['p3d'][2]:.2f}]")
    text_lines.append("==================================================")

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
        "dist_to_custom_plane_mm": None,
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
    try:
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(text_lines) + "\n")
            f.write("JSON: " + json.dumps(json_data, ensure_ascii=False) + "\n\n")
        log_and_print(f"[Record] 量測數據已儲存至: {txt_path}")
        return txt_path
    except Exception as e:
        log_and_print(f"[Record] 儲存量測數據失敗: {e}")
        return None
