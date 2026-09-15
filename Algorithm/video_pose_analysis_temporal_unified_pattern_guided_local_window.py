"""Temporal, drop-in variant of :mod:`Algorithm.video_pose_analysis`.

The original module is intentionally kept unchanged.  This copy preserves its
public ``analyze_video_frames`` interface and final ArUco/SIFT refinement, but
adds a lightweight pattern-pose front end:

* robust multi-frame marker-to-reference map estimation;
* per-frame IPPE pose hypotheses in one common marker coordinate system;
* an exact second-order DP that selects a temporally consistent hypothesis path;
* optional neighbouring ArUco probes without adding any full-frame SIFT work;
* bounded pattern-guided ArUco-only endpoint proposals using existing metric poses;
* bounded selected-endpoint local windows with ArUco/KLT validation and endpoint replacement.

Temporal translation costs use camera centres in the common marker frame.  A
PnP translation vector cannot be compared directly across moving camera frames.
"""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.optimize import least_squares

from .aruco_pose import average_rotations_svd, compute_global_plane as _compute_global_plane
from .camera_preprocess import (
    centered_roi_bounds,
    normalized_roi_bounds,
    preprocess_gray,
)
from .perf_timer import StageTimer

RECORD_SAVE_DIR = "test_video_Zebra"
SAVE_DEBUG_PAIR_IMAGES = False
SAVE_RT_SIFT_DIAGNOSTICS = True
MIN_BASELINE_MM = 20.0
# Candidate pairs intentionally keep a small margin above the externally injected
# MIN_BASELINE_MM.  The effective value MUST be resolved at analysis time because
# Zebra injects MIN_BASELINE_MM through the module alias after import.
PAIR_CANDIDATE_MIN_BASELINE_MARGIN_MM = 2.0
# Deprecated compatibility snapshot only.  Do not use this as a source of truth;
# it cannot follow module-alias injection of MIN_BASELINE_MM after import.
PAIR_CANDIDATE_MIN_BASELINE_MM = MIN_BASELINE_MM + PAIR_CANDIDATE_MIN_BASELINE_MARGIN_MM
MAX_BASELINE_MM = 220.0


def _effective_pair_candidate_min_baseline_mm():
    """Return the runtime candidate-baseline hard gate.

    ``MIN_BASELINE_MM`` is deliberately read on every call so module-alias
    injection performed by Zebra immediately affects pair enumeration, logs and
    diagnostics.  ``PAIR_CANDIDATE_MIN_BASELINE_MM`` is retained only for source
    compatibility and is never consulted by the production gate.
    """
    return float(MIN_BASELINE_MM) + float(PAIR_CANDIDATE_MIN_BASELINE_MARGIN_MM)
IDEAL_BASELINE_MM = 30.0
PAIR_SCORE_REPROJ_W = 1.00
PAIR_SCORE_DEPTH_UNCERTAINTY_W = 0.25
PAIR_SCORE_BLUR_W = 0.18
PAIR_SCORE_COVER_W = 0.12
PAIR_SCORE_MARKER_W = 0.08
PAIR_MATCH_SIGMA_PX = 1.0
PAIR_TARGET_DEPTH_SIGMA_MM = 5.0  # Legacy score used only when pattern_guided=False.

# ---- 特徵極線驗證與混合 RT 精修 ----
PAIR_SCORE_EPI_W = 0.5              # 配對評分: 特徵極線殘差權重 (px)
PAIR_EPI_TOPK = 2                   # 只讓兩組 ArUco 最佳候選進入較昂貴的 SIFT 重排
PAIR_SECOND_SCORE_MARGIN = 0.16     # 第二候選必須與第一名足夠接近才值得做 SIFT
PAIR_RERANK_MIN_IMPROVEMENT = 0.05 # 避免 SIFT 分數微小波動造成不必要換幀
PAIR_SECOND_ROT_TRIGGER_DEG = 5.0
PAIR_SECOND_MARKER_EPI_TRIGGER_PX = 6.0
PAIR_SECOND_PARALLAX_TRIGGER_DEG = 5.0
PAIR_EPI_OK_PX = 0.8                # 配對提前收斂的特徵極線殘差門檻 (px, 全模式)
PAIR_EPI_EXTRA_PX = 1.5             # 次佳對接受的特徵極線殘差上限 (px, 全模式)
PAIR_TOPK_MAX_PER_START = 2         # top-K 多樣性: 同一起始幀最多幀對數
PAIR_TOPK_MAX_PER_END = 1           # top-K 多樣性: 同一結尾幀最多幀對數
ENABLE_FEATURE_RT_REFINE = True     # 用 marker 雙向重投影硬門檻 + SIFT robust residual 聯合精修 RT
FEATURE_MATCH_RATIO = 0.75          # SIFT ratio test 閾值
FEATURE_MIN_MATCHES = 11            # 低紋理影片仍須有分散且可保留 holdout 的幾何支持
FEATURE_E_RANSAC_THRESH_PX = 0.75   # findEssentialMat RANSAC 極線距離閾值 (px)
FEATURE_ROT_DIFF_MAX_DEG = 10.0     # 特徵解與 ArUco 解允許的最大旋轉差 (超過視為異常，保留 ArUco)
FEATURE_MAX_KEYPOINTS = 800         # 特徵精修用 SIFT keypoint 上限 (控制匹配耗時)
FEATURE_IMAGE_SCALE = 1
FEATURE_MARKER_MASK_MARGIN_PX = 8.0 # Do not let marker texture dominate the independent feature check
FEATURE_GRID_COLS = 6
FEATURE_GRID_ROWS = 4
FEATURE_MAX_MATCHES_PER_CELL = 24
FEATURE_MIN_INLIER_RATIO = 0.28
FEATURE_MIN_GRID_COVERAGE = 1.0 / 6.0
FEATURE_MIN_HULL_COVERAGE = 0.02
FEATURE_MIN_PARALLAX_DEG = 1.0
FEATURE_STRONG_INLIERS = 40
FEATURE_STRONG_INLIER_RATIO = 0.35
FEATURE_STRONG_GRID_COVERAGE = 0.25
FEATURE_STRONG_HULL_COVERAGE = 0.04
FEATURE_STRONG_PARALLAX_DEG = 0.10
FEATURE_SCALE_MAX_EDGE_CV = 0.30
FEATURE_SCALE_MAX_MARKER_REL_MAD = 0.25
FEATURE_FINAL_INLIER_PX = 1.5
FEATURE_FINAL_P90_MAX_PX = 1.25
FEATURE_JOINT_GROUP_WEIGHT = 1.0
MARKER_JOINT_GROUP_WEIGHT = 12.0
JOINT_MARKER_WEIGHT_LEVELS = (12.0, 4.0, 1.0)
MARKER_BIDIR_RMS_MAX_PX = 1.5
MARKER_BIDIR_MAX_MAX_PX = 2.0
MARKER_CANDIDATE_RMS_MAX_PX = 3.0
MARKER_CANDIDATE_MAX_MAX_PX = 6.0
MARKER_BIDIR_RMS_MARGIN_PX = 0.75
MARKER_BIDIR_MAX_MARGIN_PX = 1.50
MARKER_SELECTION_RMS_BAND_PX = 0.05
FINAL_PAIR_MARKER_RMS_BAND_PX = 0.25
JOINT_RT_MAX_NFEV = 120
JOINT_RT_TOL = 1e-7
ARUCO_USE_CLAHE = True
ANALYSIS_WORKERS = min(4, max(1, os.cpu_count() or 1))
PAIR_ADD_PRIORITY_EXTRA = False
ADAPTIVE_START_RANGE_FRACTIONS = (0.80, 1.00)
ADAPTIVE_END_RANGE_FRACTIONS = (0.10, 0.22, 0.40)

# ---- lightweight temporal pattern-pose front end ----
# The five existing ArUco probes already form the temporal path.  Extra +/-1
# probes are opt-in because decoding/detection, not the exact SE(3) DP, is the
# measurable cost in this pipeline.
ENABLE_TEMPORAL_NEIGHBOR_PROBES = False
TEMPORAL_NEIGHBOR_RADIUS = 1
TEMPORAL_MAX_FRAME_CANDIDATES = 8
TEMPORAL_REPROJECTION_WEIGHT = 1.0
TEMPORAL_ROTATION_JUMP_WEIGHT = 0.035       # cost / (degree / frame)
TEMPORAL_CENTER_JUMP_WEIGHT = 0.025         # cost / (mm / frame)
TEMPORAL_ROTATION_ACCEL_WEIGHT = 0.08
TEMPORAL_CENTER_ACCEL_WEIGHT = 0.04
TEMPORAL_RELATION_ROTATION_GATE_DEG = 8.0
TEMPORAL_RELATION_TRANSLATION_GATE_MM = 10.0
TEMPORAL_MARKER_GROUP_RMS_MAX_PX = 2.5
TEMPORAL_OUTLIER_MARKER_PENALTY = 0.35
TEMPORAL_BRANCH_PRIOR_MAX_COST = 0.16
TEMPORAL_BRANCH_CONFIDENCE_MARGIN = 0.25
TEMPORAL_PLANAR_RHO_LOW = 0.015
TEMPORAL_PLANAR_RHO_HIGH = 0.05
TEMPORAL_PLANAR_KAPPA_MIN = 0.05
TEMPORAL_MARKER_GROUP_MAX_PX = 5.0
TEMPORAL_HYPOTHESIS_DEDUP_ROT_DEG = 0.05
TEMPORAL_HYPOTHESIS_DEDUP_CENTER_MM = 0.10
TEMPORAL_MAX_GATE_REFINE_ROUNDS = 3
TEMPORAL_LARGE_GAP_MULTIPLIER = 4.0
TEMPORAL_LARGE_GAP_ABS_FRAMES = 120

# ---- unified single/dual endpoint geometry ----
TEMPORAL_SINGLE_FALLBACK_PENALTY = 0.60
PAIR_NOMINAL_DEPTH_MM = 200.0
PAIR_NOMINAL_RAY_COLS = 5
PAIR_NOMINAL_RAY_ROWS = 4
PAIR_SCORE_IDEAL_BASELINE_TIE_W = 0.025
PAIR_SCORE_ANGLE_W = 0.30
PAIR_SCORE_OVERLAP_W = 0.25
PAIR_SCORE_MEASUREMENT_W = 0.20
PAIR_TARGET_TRIANGULATION_ANGLE_DEG = 5.0  # Legacy score used only when pattern_guided=False.

# ---- bounded pattern-guided frame-pair proposal/ranking ----
# These are soft engineering defaults, not hard geometric truths.  They can be
# overridden per call through pattern_guided_config.  The default 0.75 mm depth
# sigma is based on the existing 1 px first-order proxy Z/(f*tan(theta)): at
# Z=200 mm, f=1570 px it corresponds to roughly 9.6 deg conservative parallax.
PATTERN_GUIDED_DEFAULT_CONFIG = {
    'enabled': True,
    'adaptive_extra_probes': True,
    'max_total_aruco_probes': 9,
    'max_extra_probes': 4,
    'extra_probe_deadline_s': 999.0,#1.20,
    'fallback_nominal_depth_mm': 200.0,
    'depth_sigma_target_mm': 0.75,
    'distance_full_mm': (180.0, 220.0),
    'distance_outer_mm': (160.0, 250.0),
    'incidence_sweet_deg': (20.0, 25.0),
    'incidence_preferred_deg': (10.0, 35.0),
    'incidence_outer_deg': (0.0, 55.0),
    'triangulation_sweet_deg': (12.0, 18.0),
    'triangulation_outer_deg': (8.0, 22.0),
    'triangulation_p10_target_deg': 10.0,
    'triangulation_p10_outer_deg': 8.0,
    'bz_sweet': (0.20, 0.35),
    'bz_outer': (0.15, 0.42),
    'range_mismatch_full': 0.10,
    'range_mismatch_outer': 0.20,
    'marker_short_edge_full_px': 50.0,
    'marker_short_edge_outer_px': 35.0,
    'basic_overlap_min': 0.50,
    'basic_triangulation_median_min_deg': 8.0,
    'basic_triangulation_median_max_deg': 22.0,
    'basic_triangulation_p10_min_deg': 6.0,
    'basic_bz_min': 0.15,
    'basic_bz_max': 0.42,
    'basic_depth_sigma_max_mm': 1.50,
    'target_bz': 0.275,
    'min_center_trend_mm_per_frame': 0.02,
    'weights': {
        'depth_sigma': 0.22,
        'triangulation': 0.34,
        'p10': 0.16,
        'overlap': 0.24,
        'bz': 0.18,
        'distance': 0.08,
        'incidence': 0.07,
        'range_mismatch': 0.08,
        'marker_size': 0.07,
        'ideal_baseline_tie': 0.010,
    },
}

# ---- optional angle-targeted endpoint proposal ----
# The optimizer uses chronological start/end endpoints.  In auto reverse mode,
# final outputs are normalized so Frame A remains Zebra-right (target A) and
# Frame B remains Zebra-left (target B), regardless of recording direction.
# This front end only proposes ArUco candidates.  Metric RT, SIFT validation and
# final endpoint refinement remain unchanged and the original sampler is the
# explicit fallback whenever the requested angles are not observable.
ANGLE_GUIDED_DEFAULT_CONFIG = {
    'enabled': False,
    # auto evaluates both start=right/end=left and start=left/end=right.
    # The expensive ArUco detections are shared by the two hypotheses.
    'direction_mode': 'auto',
    'normalize_output_roles': True,
    'target_frame_A_deg': 15.0,
    'target_frame_B_deg': 35.0,
    'target_tolerance_deg': 6.0,
    'coarse_samples_per_segment': 12,
    'max_scan_frames_per_segment': 18,
    'candidates_per_side': 3,
    'min_candidate_separation_frames': 2,
    'pair_score_weight': 0.35,
}


def _angle_guided_resolve_config(overrides=None):
    config = dict(ANGLE_GUIDED_DEFAULT_CONFIG)
    if overrides:
        config.update(dict(overrides))
    config['enabled'] = bool(config.get('enabled', False))
    direction_mode = str(config.get('direction_mode', 'auto')).strip().lower()
    direction_aliases = {
        '15_to_35': 'forward',
        '35_to_15': 'reverse',
    }
    direction_mode = direction_aliases.get(direction_mode, direction_mode)
    if direction_mode not in ('auto', 'forward', 'reverse'):
        raise ValueError(
            "angle-guided direction_mode must be 'auto', 'forward', "
            "'reverse', '15_to_35' or '35_to_15'")
    config['direction_mode'] = direction_mode
    config['normalize_output_roles'] = bool(
        config.get('normalize_output_roles', True))
    config['target_frame_A_deg'] = float(config['target_frame_A_deg'])
    config['target_frame_B_deg'] = float(config['target_frame_B_deg'])
    config['target_tolerance_deg'] = max(
        0.1, float(config['target_tolerance_deg']))
    config['coarse_samples_per_segment'] = max(
        3, int(config['coarse_samples_per_segment']))
    config['max_scan_frames_per_segment'] = max(
        config['coarse_samples_per_segment'],
        int(config['max_scan_frames_per_segment']))
    config['candidates_per_side'] = max(
        1, int(config['candidates_per_side']))
    config['min_candidate_separation_frames'] = max(
        0, int(config['min_candidate_separation_frames']))
    config['pair_score_weight'] = max(
        0.0, float(config['pair_score_weight']))
    return config


def _angle_guided_marker_measurement(
        corners_dict, marker_id, camera_matrix, distortion, marker_size_mm):
    """Return a cheap single-marker distance/incidence estimate for scanning."""
    marker_id = int(marker_id)
    if marker_id not in corners_dict:
        return None
    branches = _temporal_marker_pose_branches(
        corners_dict[marker_id], camera_matrix, distortion, marker_size_mm)
    if not branches:
        return None
    pose = min(branches, key=lambda entry: (
        float(entry.get('reprojection_rms_px', float('inf'))),
        int(entry.get('branch', 0))))
    rotation = np.asarray(pose['R'], np.float64).reshape(3, 3)
    translation = np.asarray(pose['t'], np.float64).reshape(3)
    range_mm = float(np.linalg.norm(translation))
    if range_mm <= 1e-12:
        return None
    marker_normal_camera = rotation[:, 2]
    marker_to_camera = -translation / range_mm
    cosine = abs(float(np.dot(marker_normal_camera, marker_to_camera)))
    incidence_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    points = np.asarray(corners_dict[marker_id], np.float64).reshape(4, 2)
    return {
        'marker_id': marker_id,
        'incidence_deg': incidence_deg,
        'range_mm': range_mm,
        'reprojection_rms_px': float(
            pose.get('reprojection_rms_px', float('inf'))),
        'area_px2': abs(float(cv2.contourArea(points.astype(np.float32)))),
        'branch': int(pose.get('branch', 0)),
    }


def _angle_guided_rank_measurements(measurements, target_deg, config):
    """Select separated frames nearest an incidence target, best first."""
    cfg = _angle_guided_resolve_config(config)
    target = float(target_deg)
    ranked = sorted(
        [dict(entry) for entry in measurements
         if (entry is not None
             and np.isfinite(entry.get('incidence_deg', np.nan))
             and abs(float(entry['incidence_deg']) - target)
                 <= float(cfg['target_tolerance_deg']))],
        key=lambda entry: (
            abs(float(entry['incidence_deg']) - target),
            float(entry.get('reprojection_rms_px', float('inf'))),
            -float(entry.get('area_px2', 0.0)),
            int(entry['idx'])))
    selected = []
    separation = int(cfg['min_candidate_separation_frames'])
    for entry in ranked:
        if any(abs(int(entry['idx']) - int(old['idx'])) < separation
               for old in selected):
            continue
        selected.append(entry)
        if len(selected) >= int(cfg['candidates_per_side']):
            break
    return selected


def _angle_guided_normalize_reversed_output(result):
    """Keep UI right=A(target A), left=B(target B) after a reverse scan.

    The temporal solver always operates in chronological order.  Its relative
    pose maps the later/end (UI-left) camera into the earlier/start (UI-right)
    camera.  When the target angles occur in the opposite temporal order, swap
    all view-labelled outputs and invert the pose only after optimization.
    """
    diagnostics = result.get('angle_guided_diagnostics') or {}
    if not diagnostics.get('output_roles_swapped', False):
        return result

    old_R = np.asarray(result['R_rel'], np.float64).reshape(3, 3).copy()
    old_t_shape = np.asarray(result['t_rel']).shape
    old_t = np.asarray(result['t_rel'], np.float64).reshape(3).copy()
    new_R = old_R.T
    new_t = -new_R @ old_t

    for key_A, key_B in (
            ('frame_A', 'frame_B'), ('idx_A', 'idx_B'),
            ('cornersA', 'cornersB'),
            ('rt_sift_points_right', 'rt_sift_points_left')):
        result[key_A], result[key_B] = result.get(key_B), result.get(key_A)
    result['R_rel'] = new_R
    result['t_rel'] = new_t.reshape(old_t_shape)

    plane_n = result.get('global_plane_n')
    plane_c = result.get('global_plane_c')
    if plane_n is not None and plane_c is not None:
        result['global_plane_n'] = (
            old_R @ np.asarray(plane_n, np.float64).reshape(3))
        result['global_plane_c'] = (
            old_R @ np.asarray(plane_c, np.float64).reshape(3) + old_t)

    for role_key in (
            'detection_roi_bounds_by_role',
            'feature_roi_bounds_by_role',
            'pair_geometry_roi_bounds_by_role'):
        roles = result.get(role_key)
        if isinstance(roles, dict):
            roles['frame_A'], roles['frame_B'] = (
                roles.get('frame_B'), roles.get('frame_A'))
    detection_roles = result.get('detection_roi_bounds_by_role')
    if isinstance(detection_roles, dict):
        result['detection_roi_bounds'] = detection_roles.get('frame_A')

    def swap_angle_measurement(measurement):
        if not isinstance(measurement, dict):
            return measurement
        swapped = dict(measurement)
        for stem in ('incidence', 'target', 'error'):
            key_A = f'{stem}_A_deg'
            key_B = f'{stem}_B_deg'
            if key_A in swapped or key_B in swapped:
                swapped[key_A], swapped[key_B] = (
                    swapped.get(key_B), swapped.get(key_A))
        return swapped

    quality = result.get('rt_quality')
    if isinstance(quality, dict) and quality.get('angle_guided') is not None:
        quality['angle_guided'] = swap_angle_measurement(
            quality.get('angle_guided'))
    final_pair = diagnostics.get('final_selected_pair')
    if isinstance(final_pair, dict):
        final_pair['idx_A'], final_pair['idx_B'] = (
            result.get('idx_A'), result.get('idx_B'))
        final_pair['pair_measurement'] = (
            quality.get('angle_guided') if isinstance(quality, dict)
            else swap_angle_measurement(final_pair.get('pair_measurement')))
    diagnostics['output_idx_A'] = int(result['idx_A'])
    diagnostics['output_idx_B'] = int(result['idx_B'])
    diagnostics['output_role_normalized'] = True

    pattern_diag = result.get('pattern_guided_diagnostics')
    if isinstance(pattern_diag, dict):
        selected_pair = pattern_diag.get('selected_pair')
        if isinstance(selected_pair, dict):
            selected_pair['idx_A'] = int(result['idx_A'])
            selected_pair['idx_B'] = int(result['idx_B'])

    # Extra candidates were produced by varying the chronological start while
    # holding the end fixed.  After swapping roles they are not valid alternate
    # right images, so do not expose them with incorrect RT semantics.
    if result.get('extra_candidates'):
        diagnostics['reversed_extra_candidates_suppressed'] = int(
            len(result['extra_candidates']))
        result['extra_candidates'] = []

    result['angle_guided_diagnostics'] = diagnostics
    return result

# ---- bounded selected-endpoint local window ----
# The local stage runs only after a provisional cheap-geometry pair is known.
# It never performs local SIFT and never rebuilds the fixed marker map.
LOCAL_WINDOW_DEFAULT_CONFIG = {
    'enabled': False,
    'radius': 2,
    'stride': 1,
    'max_unique_new_frames': 8,
    'max_endpoint_candidates': 3,
    'local_stage_budget_s': 999.0,#0.65,
    'analysis_elapsed_deadline_s': 999.0,#2.05,
    'aruco_roi_expand': 2.8,
    'allow_full_detection_fallback': True,
    'klt_scale': 0.5,
    'klt_max_points': 160,
    'klt_min_tracks': 24,
    'klt_fb_gate_px': 1.5,
    'klt_grid_cols': 4,
    'klt_grid_rows': 3,
    'klt_max_per_cell': 16,
    'klt_win_size': 21,
    'klt_max_level': 3,
    'klt_rotation_gate_deg': 4.0,
    'klt_rotation_weight': 0.08,
    'anchor_rotation_deg_per_frame': 2.5,
    'anchor_center_mm_per_frame': 3.0,
    'anchor_rotation_weight': 0.05,
    'anchor_center_weight': 0.04,
    'measurement_weight': 0.12,
    'reprojection_weight': 0.05,
    'sharpness_weight': 0.12,
    'frame_offset_tie_weight': 0.004,
    'pose_denoise': True,
    'pose_blend_alpha': 0.20,
    'pose_fit_min_frames': 3,
    'reprojection_worsen_margin_px': 0.15,
    'max_pose_denoise_rotation_deg': 1.5,
    'max_pose_denoise_center_mm': 2.0,
}

UNIFIED_MARKER_DIRECT_RMS_MAX_PX = 1.75
UNIFIED_MARKER_DIRECT_MAX_PX = 3.0
UNIFIED_MARKER_GROUP_WEIGHT = 4.0


def _temporal_marker_object_points(marker_size_mm):
    half = float(marker_size_mm) * 0.5
    return np.asarray([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def _temporal_rotation_distance_deg(first, second):
    delta = np.asarray(first, np.float64) @ np.asarray(second, np.float64).T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _temporal_camera_center(rotation, translation):
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    translation = np.asarray(translation, np.float64).reshape(3, 1)
    return (-rotation.T @ translation).reshape(3)


def _temporal_relation_distance(first, second):
    return (
        _temporal_rotation_distance_deg(first['R'], second['R']),
        float(np.linalg.norm(
            np.asarray(first['t'], np.float64).reshape(3)
            - np.asarray(second['t'], np.float64).reshape(3))),
    )


def _temporal_robust_pose_mean(poses, initial=None, iterations=10):
    """Robust mean for transforms expressed in one fixed output frame."""
    poses = list(poses)
    if not poses:
        return None
    seed = initial or poses[0]
    mean_R = np.asarray(seed['R'], np.float64).reshape(3, 3).copy()
    mean_t = np.asarray(seed['t'], np.float64).reshape(3, 1).copy()
    for _ in range(int(iterations)):
        rotation_residuals = np.asarray([
            cv2.Rodrigues(np.asarray(pose['R'], np.float64) @ mean_R.T)[0].reshape(3)
            for pose in poses
        ])
        translation_residuals = np.asarray([
            np.asarray(pose['t'], np.float64).reshape(3) - mean_t.reshape(3)
            for pose in poses
        ])
        rotation_norm = np.linalg.norm(rotation_residuals, axis=1)
        translation_norm = np.linalg.norm(translation_residuals, axis=1)
        rotation_scale = max(float(np.median(rotation_norm)) * 2.5, math.radians(0.25))
        translation_scale = max(float(np.median(translation_norm)) * 2.5, 0.25)
        quality = np.asarray([
            1.0 / max(float(pose.get('emission', 0.0)) + 0.10, 0.10)
            for pose in poses
        ])
        robust = 1.0 / np.maximum(
            1.0,
            rotation_norm / rotation_scale + translation_norm / translation_scale,
        )
        weights = quality * robust
        delta_rotation = np.average(rotation_residuals, axis=0, weights=weights)
        delta_translation = np.average(translation_residuals, axis=0, weights=weights)
        mean_R = cv2.Rodrigues(delta_rotation)[0] @ mean_R
        mean_t = mean_t + delta_translation.reshape(3, 1)
        if (np.linalg.norm(delta_rotation) < 1e-8
                and np.linalg.norm(delta_translation) < 1e-5):
            break
    return {'R': mean_R, 't': mean_t, 'emission': 0.0}


def _temporal_marker_pose_branches(corners, camera_matrix, distortion, marker_size_mm):
    object_points = _temporal_marker_object_points(marker_size_mm)
    image_points = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    try:
        _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_points.astype(np.float32), image_points,
            np.asarray(camera_matrix, np.float64), distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    results = []
    for branch, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
        translation = np.asarray(tvec, np.float64).reshape(3, 1)
        points_camera = (rotation @ object_points.T + translation).T
        if np.any(points_camera[:, 2] <= 1e-6):
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, translation, camera_matrix, distortion)
        residual = projected.reshape(4, 2) - np.asarray(corners, np.float64).reshape(4, 2)
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        results.append({
            'R': rotation,
            't': translation,
            'branch': int(branch),
            'reprojection_rms_px': rms,
            'emission': rms,
        })
    return results


def _estimate_temporal_marker_relation(
        frame_infos, reference_id, marker_id, camera_matrix, distortion,
        marker_size_mm):
    """Estimate T_reference<-marker from all co-visible IPPE combinations."""
    candidates_by_frame = {}
    for item in frame_infos:
        corners = item.get('corners', {})
        if reference_id not in corners or marker_id not in corners:
            continue
        reference_branches = _temporal_marker_pose_branches(
            corners[reference_id], camera_matrix, distortion, marker_size_mm)
        marker_branches = _temporal_marker_pose_branches(
            corners[marker_id], camera_matrix, distortion, marker_size_mm)
        combinations = []
        for reference in reference_branches:
            for marker in marker_branches:
                relation_R = reference['R'].T @ marker['R']
                relation_t = reference['R'].T @ (marker['t'] - reference['t'])
                combinations.append({
                    'R': relation_R,
                    't': relation_t,
                    'emission': (
                        reference['reprojection_rms_px']
                        + marker['reprojection_rms_px']),
                    'reference_branch': reference['branch'],
                    'marker_branch': marker['branch'],
                    'frame_index': int(item['idx']),
                })
        if combinations:
            candidates_by_frame[int(item['idx'])] = combinations

    candidate_frames = len(candidates_by_frame)
    diagnostics = {
        'status': 'NO_COVISIBLE_FRAMES',
        'candidate_frames': candidate_frames,
        'minimum_support_frames': 3,
        'support_frames': 0,
        'support_frame_indices': [],
        'outlier_frame_indices': sorted(candidates_by_frame),
        'selected_branches_by_frame': {},
        'rotation_residual_median_deg': None,
        'rotation_residual_p95_deg': None,
        'translation_residual_median_mm': None,
        'translation_residual_p95_mm': None,
    }
    if not candidates_by_frame:
        return None, diagnostics
    if candidate_frames < 3:
        diagnostics['status'] = 'INSUFFICIENT_COVISIBILITY'
        return None, diagnostics

    seeds = [candidate for values in candidates_by_frame.values() for candidate in values]
    best = None
    for seed in seeds:
        accepted = []
        normalized = []
        for combinations in candidates_by_frame.values():
            ranked = []
            for candidate in combinations:
                rotation, translation = _temporal_relation_distance(candidate, seed)
                score = (
                    rotation / TEMPORAL_RELATION_ROTATION_GATE_DEG
                    + translation / TEMPORAL_RELATION_TRANSLATION_GATE_MM
                    + 0.02 * candidate['emission'])
                ranked.append((score, rotation, translation, candidate))
            score, rotation, translation, candidate = min(ranked, key=lambda entry: entry[0])
            if (rotation <= TEMPORAL_RELATION_ROTATION_GATE_DEG
                    and translation <= TEMPORAL_RELATION_TRANSLATION_GATE_MM):
                accepted.append(candidate)
                normalized.append(score)
        seed_score = (
            -len(accepted),
            float(np.median(normalized)) if normalized else float('inf'),
            float(seed['emission']),
            int(seed['frame_index']),
            int(seed['reference_branch']),
            int(seed['marker_branch']),
        )
        if best is None or seed_score < best[0]:
            best = (seed_score, seed, accepted)

    relation = _temporal_robust_pose_mean(best[2], initial=best[1])
    selected = {}
    for _ in range(6):
        accepted = []
        selected = {}
        for frame_index, combinations in candidates_by_frame.items():
            ranked = []
            for candidate in combinations:
                rotation, translation = _temporal_relation_distance(candidate, relation)
                score = (
                    rotation / TEMPORAL_RELATION_ROTATION_GATE_DEG
                    + translation / TEMPORAL_RELATION_TRANSLATION_GATE_MM
                    + 0.02 * candidate['emission'])
                ranked.append((score, rotation, translation, candidate))
            _score, rotation, translation, candidate = min(
                ranked, key=lambda entry: entry[0])
            if (rotation <= TEMPORAL_RELATION_ROTATION_GATE_DEG
                    and translation <= TEMPORAL_RELATION_TRANSLATION_GATE_MM):
                selected[frame_index] = candidate
                accepted.append(candidate)
        updated = _temporal_robust_pose_mean(accepted, initial=relation)
        if updated is None:
            break
        rotation_delta, translation_delta = _temporal_relation_distance(updated, relation)
        relation = updated
        if rotation_delta < 1e-5 and translation_delta < 1e-4:
            break

    support = len(selected)
    minimum_support = max(3, math.ceil(0.60 * candidate_frames))
    rotation_residuals = []
    translation_residuals = []
    for candidate in selected.values():
        rotation, translation = _temporal_relation_distance(candidate, relation)
        rotation_residuals.append(rotation)
        translation_residuals.append(translation)
    diagnostics.update({
        'status': 'OK_ROBUST_CONSENSUS' if support >= minimum_support else 'LOW_SUPPORT_FALLBACK',
        'minimum_support_frames': int(minimum_support),
        'support_frames': support,
        'support_frame_indices': sorted(selected),
        'outlier_frame_indices': sorted(set(candidates_by_frame) - set(selected)),
        'selected_branches_by_frame': {
            int(frame_index): {
                'reference_branch': int(candidate['reference_branch']),
                'marker_branch': int(candidate['marker_branch']),
            }
            for frame_index, candidate in selected.items()
        },
        'rotation_residual_median_deg': (
            float(np.median(rotation_residuals)) if rotation_residuals else None),
        'rotation_residual_p95_deg': (
            float(np.percentile(rotation_residuals, 95)) if rotation_residuals else None),
        'translation_residual_median_mm': (
            float(np.median(translation_residuals)) if translation_residuals else None),
        'translation_residual_p95_mm': (
            float(np.percentile(translation_residuals, 95)) if translation_residuals else None),
    })
    if support < minimum_support:
        return None, diagnostics
    return (relation['R'], relation['t']), diagnostics


def _temporal_inverse_transform(rotation, translation):
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    translation = np.asarray(translation, np.float64).reshape(3, 1)
    inverse_rotation = rotation.T
    return inverse_rotation, -inverse_rotation @ translation


def _temporal_compose_transform(first, second):
    """Compose T_A<-B and T_B<-C into T_A<-C."""
    R_ab, t_ab = first
    R_bc, t_bc = second
    R_ab = np.asarray(R_ab, np.float64).reshape(3, 3)
    t_ab = np.asarray(t_ab, np.float64).reshape(3, 1)
    R_bc = np.asarray(R_bc, np.float64).reshape(3, 3)
    t_bc = np.asarray(t_bc, np.float64).reshape(3, 1)
    return R_ab @ R_bc, R_ab @ t_bc + t_ab


def _temporal_relation_uncertainty_cost(diagnostics, marker_size_mm):
    if not diagnostics:
        return float('inf')
    t95 = diagnostics.get('translation_residual_p95_mm')
    r95 = diagnostics.get('rotation_residual_p95_deg')
    if t95 is None or r95 is None:
        return float('inf')
    radius = float(marker_size_mm) / math.sqrt(2.0)
    return float(t95) + radius * math.radians(float(r95))


def _build_temporal_marker_map_graph(
        frame_infos, marker_ids, camera_matrix, distortion, marker_size_mm,
        start_marker_ids=None, end_marker_ids=None):
    """Build one rigid marker map through robust co-visibility graph edges.

    Returns ``(marker_map, diagnostics, reference_id)``.  A component is usable
    only when it contains at least one marker observed by each endpoint segment.
    This permits A/B to observe different IDs while safely rejecting disconnected
    layouts.
    """
    marker_ids = sorted(set(int(x) for x in marker_ids))
    start_marker_ids = set(int(x) for x in (start_marker_ids or marker_ids))
    end_marker_ids = set(int(x) for x in (end_marker_ids or marker_ids))
    visibility = {mid: 0 for mid in marker_ids}
    for item in frame_infos:
        for mid in set(item.get('corners', {})) & set(marker_ids):
            visibility[int(mid)] += 1

    adjacency = {mid: [] for mid in marker_ids}
    edge_diagnostics = {}
    for pos, first in enumerate(marker_ids):
        for second in marker_ids[pos + 1:]:
            relation, diag = _estimate_temporal_marker_relation(
                frame_infos, first, second, camera_matrix, distortion, marker_size_mm)
            edge_diagnostics[(first, second)] = diag
            if relation is None:
                continue
            R_first_second = np.asarray(relation[0], np.float64).reshape(3, 3)
            t_first_second = np.asarray(relation[1], np.float64).reshape(3, 1)
            uncertainty = _temporal_relation_uncertainty_cost(diag, marker_size_mm)
            adjacency[first].append((second, R_first_second, t_first_second, uncertainty, diag))
            R_second_first, t_second_first = _temporal_inverse_transform(
                R_first_second, t_first_second)
            adjacency[second].append((first, R_second_first, t_second_first, uncertainty, diag))

    components = []
    remaining = set(marker_ids)
    while remaining:
        root = min(remaining)
        stack = [root]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbor for neighbor, *_rest in adjacency[current])
        remaining -= component
        components.append(component)

    viable = [
        component for component in components
        if component & start_marker_ids and component & end_marker_ids
    ]
    graph_diag = {
        'status': 'DISCONNECTED_ENDPOINT_MARKERS' if not viable else 'OK',
        'components': [sorted(component) for component in components],
        'start_marker_ids': sorted(start_marker_ids),
        'end_marker_ids': sorted(end_marker_ids),
        'edge_diagnostics': {f'{a}-{b}': d for (a, b), d in edge_diagnostics.items()},
    }
    if not viable:
        return {}, {'_graph': graph_diag}, None

    def component_rank(component):
        return (
            len(component),
            sum(visibility.get(mid, 0) for mid in component),
            sum(len(adjacency[mid]) for mid in component),
            -min(component),
        )
    component = max(viable, key=component_rank)
    reference_id = min(
        component,
        key=lambda mid: (-visibility.get(mid, 0), -len(adjacency[mid]), mid))

    # Dijkstra favours the lowest accumulated map-uncertainty chain.
    import heapq
    distances = {reference_id: 0.0}
    transforms = {reference_id: (np.eye(3), np.zeros((3, 1)))}
    paths = {reference_id: [reference_id]}
    uncertainty_sums = {reference_id: (0.0, 0.0)}
    queue = [(0.0, reference_id)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost > distances.get(current, float('inf')) + 1e-12:
            continue
        for neighbor, R_current_neighbor, t_current_neighbor, edge_cost, diag in adjacency[current]:
            if neighbor not in component:
                continue
            safe_edge_cost = edge_cost if np.isfinite(edge_cost) else 1e6
            next_cost = cost + safe_edge_cost
            if next_cost + 1e-12 >= distances.get(neighbor, float('inf')):
                continue
            transforms[neighbor] = _temporal_compose_transform(
                transforms[current], (R_current_neighbor, t_current_neighbor))
            distances[neighbor] = next_cost
            paths[neighbor] = paths[current] + [neighbor]
            prev_t, prev_r = uncertainty_sums[current]
            edge_t = diag.get('translation_residual_p95_mm')
            edge_r = diag.get('rotation_residual_p95_deg')
            uncertainty_sums[neighbor] = (
                prev_t + (float(edge_t) if edge_t is not None else float(marker_size_mm)),
                prev_r + (float(edge_r) if edge_r is not None else 15.0),
            )
            heapq.heappush(queue, (next_cost, neighbor))

    marker_map = {}
    diagnostics = {'_graph': graph_diag}
    for marker_id, transform in transforms.items():
        marker_map[int(marker_id)] = (
            np.asarray(transform[0], np.float64),
            np.asarray(transform[1], np.float64).reshape(3, 1))
        if marker_id == reference_id:
            diagnostics[int(marker_id)] = {
                'status': 'REFERENCE_IDENTITY',
                'candidate_frames': int(visibility.get(marker_id, 0)),
                'translation_residual_p95_mm': 0.0,
                'rotation_residual_p95_deg': 0.0,
                'graph_path': [int(reference_id)],
            }
        else:
            t95, r95 = uncertainty_sums[marker_id]
            diagnostics[int(marker_id)] = {
                'status': 'GRAPH_CHAINED',
                'candidate_frames': int(visibility.get(marker_id, 0)),
                'translation_residual_p95_mm': float(t95),
                'rotation_residual_p95_deg': float(r95),
                'graph_path': [int(x) for x in paths[marker_id]],
                'graph_uncertainty_cost': float(distances[marker_id]),
            }
    graph_diag['status'] = 'OK'
    graph_diag['selected_component'] = sorted(component)
    graph_diag['reference_marker_id'] = int(reference_id)
    graph_diag['mapped_marker_ids'] = sorted(marker_map)
    return marker_map, diagnostics, int(reference_id)


def _build_temporal_marker_map(
        frame_infos, marker_ids, reference_id, camera_matrix, distortion,
        marker_size_mm):
    marker_map = {
        int(reference_id): (
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64)),
    }
    diagnostics = {
        int(reference_id): {
            'status': 'REFERENCE_IDENTITY',
            'candidate_frames': sum(
                reference_id in item.get('corners', {}) for item in frame_infos),
        }
    }
    for marker_id in sorted(set(marker_ids) - {reference_id}):
        relation, relation_diagnostics = _estimate_temporal_marker_relation(
            frame_infos, reference_id, marker_id, camera_matrix, distortion,
            marker_size_mm)
        diagnostics[int(marker_id)] = relation_diagnostics
        if relation is not None:
            marker_map[int(marker_id)] = (
                np.asarray(relation[0], np.float64),
                np.asarray(relation[1], np.float64).reshape(3, 1),
            )
    return marker_map, diagnostics


def _temporal_anchor_pose(marker_pose, marker_to_reference):
    """Convert T_camera<-marker into T_camera<-reference."""
    marker_R = np.asarray(marker_pose['R'], np.float64).reshape(3, 3)
    marker_t = np.asarray(marker_pose['t'], np.float64).reshape(3, 1)
    map_R = np.asarray(marker_to_reference[0], np.float64).reshape(3, 3)
    map_t = np.asarray(marker_to_reference[1], np.float64).reshape(3, 1)
    anchor_R = marker_R @ map_R.T
    anchor_t = marker_t - anchor_R @ map_t
    return anchor_R, anchor_t


def _temporal_marker_reference_points(marker_id, marker_map, marker_size_mm):
    object_local = _temporal_marker_object_points(marker_size_mm)
    map_R, map_t = marker_map[int(marker_id)]
    return (
        np.asarray(map_R, np.float64).reshape(3, 3) @ object_local.T
        + np.asarray(map_t, np.float64).reshape(3, 1)
    ).T


def _temporal_pose_marker_metrics(
        rotation, translation, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm):
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    translation = np.asarray(translation, np.float64).reshape(3, 1)
    rvec = cv2.Rodrigues(rotation)[0]
    rms_errors = {}
    max_errors = {}
    depths_ok = {}
    for marker_id, observed in corners_dict.items():
        marker_id = int(marker_id)
        if marker_id not in marker_map:
            continue
        object_reference = _temporal_marker_reference_points(
            marker_id, marker_map, marker_size_mm)
        points_camera = (rotation @ object_reference.T + translation).T
        depths_ok[marker_id] = bool(np.all(points_camera[:, 2] > 1e-6))
        projected, _ = cv2.projectPoints(
            object_reference, rvec, translation, camera_matrix, distortion)
        residual = (
            projected.reshape(4, 2)
            - np.asarray(observed, np.float64).reshape(4, 2))
        corner_error = np.linalg.norm(residual, axis=1)
        rms_errors[marker_id] = float(np.sqrt(np.mean(corner_error ** 2)))
        max_errors[marker_id] = float(np.max(corner_error))
    return rms_errors, max_errors, depths_ok


def _temporal_pose_marker_errors(
        rotation, translation, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm):
    rms_errors, _max_errors, depths_ok = _temporal_pose_marker_metrics(
        rotation, translation, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm)
    return rms_errors, depths_ok


def _temporal_refine_pose_with_marker_groups(
        seed_R, seed_t, inlier_ids, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm):
    if len(inlier_ids) < 2:
        return np.asarray(seed_R, np.float64), np.asarray(seed_t, np.float64).reshape(3, 1)
    object_points = []
    image_points = []
    for marker_id in inlier_ids:
        object_points.append(_temporal_marker_reference_points(
            marker_id, marker_map, marker_size_mm))
        image_points.append(np.asarray(
            corners_dict[marker_id], np.float64).reshape(4, 2))
    object_points = np.concatenate(object_points, axis=0).astype(np.float64)
    image_points = np.concatenate(image_points, axis=0).astype(np.float64)
    rvec = cv2.Rodrigues(np.asarray(seed_R, np.float64))[0]
    tvec = np.asarray(seed_t, np.float64).reshape(3, 1).copy()
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
    if not ok:
        return np.asarray(seed_R, np.float64), np.asarray(seed_t, np.float64).reshape(3, 1)
    return cv2.Rodrigues(rvec)[0], np.asarray(tvec, np.float64).reshape(3, 1)


def _temporal_plane_frame(marker_ids, marker_map, marker_size_mm):
    """Deterministic right-handed best-fit plane expressed in reference frame."""
    marker_ids = [int(marker_id) for marker_id in sorted(marker_ids)]
    points = np.concatenate([
        _temporal_marker_reference_points(marker_id, marker_map, marker_size_mm)
        for marker_id in marker_ids
    ], axis=0)
    center = np.mean(points, axis=0)
    centered = points - center
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], np.float64)

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_alignment = float(np.dot(normal, world_z))
    if abs(world_alignment) > 1e-9:
        if world_alignment < 0.0:
            normal = -normal
    else:
        normal_hints = []
        for marker_id in marker_ids:
            map_R = np.asarray(marker_map[marker_id][0], np.float64).reshape(3, 3)
            normal_hints.append(map_R[:, 2])
        hint = np.sum(normal_hints, axis=0) if normal_hints else world_z.copy()
        if np.linalg.norm(hint) > 1e-12 and abs(float(np.dot(normal, hint))) > 1e-12:
            if float(np.dot(normal, hint)) < 0.0:
                normal = -normal
        else:
            for value in normal:
                if abs(float(value)) > 1e-12:
                    if value < 0:
                        normal = -normal
                    break
    normal = normal / max(np.linalg.norm(normal), 1e-15)

    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = world_x - float(np.dot(world_x, normal)) * normal
    if np.linalg.norm(x_axis) <= 1e-9:
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_axis = world_y - float(np.dot(world_y, normal)) * normal
        y_axis /= max(np.linalg.norm(y_axis), 1e-15)
        x_axis = np.cross(y_axis, normal)
        x_axis /= max(np.linalg.norm(x_axis), 1e-15)
    else:
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis /= max(np.linalg.norm(y_axis), 1e-15)
    R_RP = np.column_stack([x_axis, y_axis, normal])
    if np.linalg.det(R_RP) < 0.0:
        y_axis = -y_axis
        R_RP = np.column_stack([x_axis, y_axis, normal])
    if abs(np.linalg.det(R_RP) - 1.0) > 1e-8:
        raise ValueError('plane frame is not right-handed')
    return {
        'R_RP': R_RP,
        'c': center.reshape(3, 1),
        'points_reference': points,
        'singular_values': np.asarray(singular, np.float64),
    }


def _temporal_plane_pose_to_reference(R_CP, t_CP, plane_frame):
    R_CP = np.asarray(R_CP, np.float64).reshape(3, 3)
    t_CP = np.asarray(t_CP, np.float64).reshape(3, 1)
    R_RP = np.asarray(plane_frame['R_RP'], np.float64).reshape(3, 3)
    c = np.asarray(plane_frame['c'], np.float64).reshape(3, 1)
    R_CR = R_CP @ R_RP.T
    t_CR = t_CP - R_CR @ c
    return R_CR, t_CR


def _temporal_map_uncertainty_mm(marker_ids, marker_map_diagnostics, marker_size_mm):
    if marker_map_diagnostics is None:
        return None
    radius = float(marker_size_mm) / math.sqrt(2.0)
    values = []
    non_reference = 0
    for marker_id in marker_ids:
        diag = marker_map_diagnostics.get(int(marker_id))
        if not diag:
            return None
        if diag.get('status') == 'REFERENCE_IDENTITY':
            values.append(0.0)
            continue
        non_reference += 1
        t95 = diag.get('translation_residual_p95_mm')
        r95 = diag.get('rotation_residual_p95_deg')
        if t95 is None or r95 is None:
            return None
        values.append(float(t95) + radius * math.radians(float(r95)))
    if non_reference == 0:
        return 0.0
    return float(np.percentile(values, 95))


def _temporal_classify_planarity(
        marker_ids, marker_map, marker_map_diagnostics, marker_size_mm):
    plane = _temporal_plane_frame(marker_ids, marker_map, marker_size_mm)
    singular = plane['singular_values'] / math.sqrt(max(len(plane['points_reference']), 1))
    s1, s2, s3 = [float(value) for value in singular]
    denom = math.sqrt(max(s1 * s2, 0.0)) + 1e-12
    rho_obs = s3 / denom
    kappa = s2 / (s1 + 1e-12)
    uncertainty_mm = _temporal_map_uncertainty_mm(
        marker_ids, marker_map_diagnostics, marker_size_mm)
    u_rho = None if uncertainty_mm is None else float(uncertainty_mm / denom)
    if uncertainty_mm is None:
        classification = 'GRAY'
    elif rho_obs + u_rho <= TEMPORAL_PLANAR_RHO_LOW:
        classification = 'PLANAR'
    elif (rho_obs - u_rho >= TEMPORAL_PLANAR_RHO_HIGH
          and kappa >= TEMPORAL_PLANAR_KAPPA_MIN):
        classification = 'NONPLANAR'
    else:
        classification = 'GRAY'
    return classification, {
        'classification': classification,
        'rho_obs': float(rho_obs),
        'u_rho': u_rho,
        'kappa_2d': float(kappa),
        'map_uncertainty_mm': uncertainty_mm,
        'R_RP': plane['R_RP'],
        'c': plane['c'],
    }


def _temporal_board_planar_ippe_seeds(
        marker_ids, corners_dict, marker_map, camera_matrix, distortion,
        marker_size_mm, plane_frame):
    R_RP = np.asarray(plane_frame['R_RP'], np.float64).reshape(3, 3)
    c = np.asarray(plane_frame['c'], np.float64).reshape(3, 1)
    object_points = []
    image_points = []
    for marker_id in sorted(marker_ids):
        points_reference = _temporal_marker_reference_points(
            marker_id, marker_map, marker_size_mm)
        points_plane = (R_RP.T @ (points_reference.T - c)).T
        points_plane[:, 2] = 0.0
        object_points.append(points_plane)
        image_points.append(np.asarray(corners_dict[marker_id], np.float64).reshape(4, 2))
    object_points = np.concatenate(object_points, axis=0).astype(np.float64)
    image_points = np.concatenate(image_points, axis=0).astype(np.float64)
    try:
        _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    seeds = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R_CP = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
        R_CR, t_CR = _temporal_plane_pose_to_reference(
            R_CP, np.asarray(tvec, np.float64).reshape(3, 1), plane_frame)
        seeds.append({
            'R': R_CR,
            't': t_CR,
            'parent_seed': f'PLANAR_IPPE_{index}',
            'basin': int(index),
            'source': 'board_planar_ippe',
        })
    return seeds[:2]


def _temporal_board_native_3d_seed(
        marker_ids, corners_dict, marker_map, camera_matrix, distortion,
        marker_size_mm):
    object_points = np.concatenate([
        _temporal_marker_reference_points(marker_id, marker_map, marker_size_mm)
        for marker_id in sorted(marker_ids)
    ], axis=0).astype(np.float64)
    image_points = np.concatenate([
        np.asarray(corners_dict[marker_id], np.float64).reshape(4, 2)
        for marker_id in sorted(marker_ids)
    ], axis=0).astype(np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
    if not ok:
        return None
    return {
        'R': cv2.Rodrigues(np.asarray(rvec, np.float64))[0],
        't': np.asarray(tvec, np.float64).reshape(3, 1),
        'parent_seed': 'NATIVE_3D',
        'basin': None,
        'source': 'board_native_3d',
    }


def _temporal_gate_refine_candidate(
        seed, available_ids, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm):
    rotation = np.asarray(seed['R'], np.float64).reshape(3, 3)
    translation = np.asarray(seed['t'], np.float64).reshape(3, 1)
    last_inliers = None
    rounds = 0
    for rounds in range(1, TEMPORAL_MAX_GATE_REFINE_ROUNDS + 1):
        rms, max_error, depth = _temporal_pose_marker_metrics(
            rotation, translation, corners_dict, marker_map,
            camera_matrix, distortion, marker_size_mm)
        inliers = sorted(
            marker_id for marker_id in available_ids
            if depth.get(marker_id, False)
            and rms.get(marker_id, float('inf')) <= TEMPORAL_MARKER_GROUP_RMS_MAX_PX
            and max_error.get(marker_id, float('inf')) <= TEMPORAL_MARKER_GROUP_MAX_PX)
        if not inliers:
            return None
        if inliers == last_inliers:
            break
        last_inliers = inliers
        if len(inliers) >= 2:
            rotation, translation = _temporal_refine_pose_with_marker_groups(
                rotation, translation, inliers, corners_dict, marker_map,
                camera_matrix, distortion, marker_size_mm)
    rms, max_error, depth = _temporal_pose_marker_metrics(
        rotation, translation, corners_dict, marker_map,
        camera_matrix, distortion, marker_size_mm)
    inliers = sorted(
        marker_id for marker_id in available_ids
        if depth.get(marker_id, False)
        and rms.get(marker_id, float('inf')) <= TEMPORAL_MARKER_GROUP_RMS_MAX_PX
        and max_error.get(marker_id, float('inf')) <= TEMPORAL_MARKER_GROUP_MAX_PX)
    if not inliers:
        return None
    single_fallback = bool(
        len(available_ids) >= 2 and len(inliers) < 2
        and seed.get('allow_single_fallback', False))
    if len(available_ids) >= 2 and len(inliers) < 2 and not single_fallback:
        return None
    board_rms = float(np.sqrt(np.mean([rms[mid] ** 2 for mid in inliers])))
    outliers = sorted(set(available_ids) - set(inliers))
    measurement_mode = (
        'SINGLE_FALLBACK' if single_fallback
        else seed.get('measurement_mode', 'SINGLE' if len(available_ids) == 1 else 'DUAL_UNKNOWN'))
    confidence_by_mode = {
        'SINGLE': 0.75,
        'DUAL_PLANAR': 0.85,
        'DUAL_GRAY': 0.90,
        'DUAL_NONPLANAR': 1.00,
        'SINGLE_FALLBACK': 0.55,
    }
    measurement_confidence = float(confidence_by_mode.get(measurement_mode, 0.75))
    emission = (
        TEMPORAL_REPROJECTION_WEIGHT * board_rms / math.sqrt(len(inliers))
        + TEMPORAL_OUTLIER_MARKER_PENALTY * len(outliers)
        + (TEMPORAL_SINGLE_FALLBACK_PENALTY if single_fallback else 0.0))
    return {
        **seed,
        'R': rotation,
        't': translation,
        'camera_center': _temporal_camera_center(rotation, translation),
        'marker_ids': list(available_ids),
        'inlier_marker_ids': inliers,
        'outlier_marker_ids': outliers,
        'per_marker_rms_px': rms,
        'per_marker_max_px': max_error,
        'reprojection_rms_px': board_rms,
        'emission_cost': float(emission),
        'gate_refine_rounds': int(rounds),
        'measurement_mode': measurement_mode,
        'measurement_confidence': measurement_confidence,
        'single_fallback': single_fallback,
    }


def _temporal_candidate_duplicate(first, second):
    return (
        _temporal_rotation_distance_deg(first['R'], second['R'])
        < TEMPORAL_HYPOTHESIS_DEDUP_ROT_DEG
        and np.linalg.norm(
            np.asarray(first['camera_center']) - np.asarray(second['camera_center']))
        < TEMPORAL_HYPOTHESIS_DEDUP_CENTER_MM)


def _temporal_deduplicate_candidates(candidates, limit=TEMPORAL_MAX_FRAME_CANDIDATES):
    accepted = []
    for candidate in sorted(candidates, key=lambda value: (
            value['emission_cost'], str(value.get('label', '')),
            str(value.get('parent_seed', '')))):
        if any(_temporal_candidate_duplicate(candidate, other) for other in accepted):
            continue
        accepted.append(candidate)
        if len(accepted) >= int(limit):
            break
    return accepted


def _build_temporal_frame_candidates(
        item, marker_map, camera_matrix, distortion, marker_size_mm,
        marker_map_diagnostics=None):
    corners_dict = item.get('corners', {})
    available_ids = sorted(set(corners_dict) & set(marker_map))
    if not available_ids:
        return []

    seeds = []
    planarity = None
    if len(available_ids) == 1:
        marker_id = available_ids[0]
        for branch in _temporal_marker_pose_branches(
                corners_dict[marker_id], camera_matrix, distortion, marker_size_mm):
            rotation, translation = _temporal_anchor_pose(branch, marker_map[marker_id])
            seeds.append({
                'R': rotation,
                't': translation,
                'label': f"ID{marker_id}:B{branch['branch']}",
                'parent_seed': f"ID{marker_id}:B{branch['branch']}",
                'basin': int(branch['branch']),
                'source': 'marker_ippe',
                'seed_marker_id': int(marker_id),
                'seed_branch': int(branch['branch']),
                'measurement_mode': 'SINGLE',
                'allow_single_fallback': False,
            })
    else:
        classification, planarity = _temporal_classify_planarity(
            available_ids, marker_map, marker_map_diagnostics, marker_size_mm)
        plane_frame = {'R_RP': planarity['R_RP'], 'c': planarity['c']}
        if classification in ('PLANAR', 'GRAY'):
            seeds.extend(_temporal_board_planar_ippe_seeds(
                available_ids, corners_dict, marker_map, camera_matrix,
                distortion, marker_size_mm, plane_frame))
        if classification in ('GRAY', 'NONPLANAR'):
            native = _temporal_board_native_3d_seed(
                available_ids, corners_dict, marker_map, camera_matrix,
                distortion, marker_size_mm)
            if native is not None:
                seeds.append(native)
        # Clearly non-planar boards must retain per-marker planar ambiguity only
        # as seeds/fallbacks; all surviving dual solutions are re-gated against
        # the original mapped 3-D corners.
        if classification == 'NONPLANAR':
            for marker_id in available_ids:
                for branch in _temporal_marker_pose_branches(
                        corners_dict[marker_id], camera_matrix, distortion, marker_size_mm):
                    rotation, translation = _temporal_anchor_pose(
                        branch, marker_map[marker_id])
                    seeds.append({
                        'R': rotation,
                        't': translation,
                        'parent_seed': f"NONPLANAR_ID{marker_id}:B{branch['branch']}",
                        'basin': int(branch['branch']),
                        'source': 'marker_ippe_joint_seed',
                        'seed_marker_id': int(marker_id),
                        'seed_branch': int(branch['branch']),
                        'allow_single_fallback': True,
                    })

        mode = {
            'PLANAR': 'DUAL_PLANAR',
            'GRAY': 'DUAL_GRAY',
            'NONPLANAR': 'DUAL_NONPLANAR',
        }[classification]
        for seed_index, seed in enumerate(seeds):
            seed.setdefault('label', f'{classification}:H{seed_index}')
            seed.setdefault('seed_marker_id', None)
            seed.setdefault('seed_branch', seed.get('basin'))
            seed.setdefault('allow_single_fallback', False)
            seed['measurement_mode'] = mode
            seed['planarity_class'] = classification
            seed['planarity_rho'] = planarity['rho_obs']
            seed['planarity_u_rho'] = planarity['u_rho']
            seed['planarity_kappa_2d'] = planarity['kappa_2d']

    candidates = []
    for seed in seeds:
        candidate = _temporal_gate_refine_candidate(
            seed, available_ids, corners_dict, marker_map, camera_matrix,
            distortion, marker_size_mm)
        if candidate is not None:
            candidates.append(candidate)

    return _temporal_deduplicate_candidates(
        candidates, limit=TEMPORAL_MAX_FRAME_CANDIDATES)


def _temporal_transition_triplet_cost(before_entry, previous_entry, current_entry):
    """Second-order transition; entries are (frame_index, candidate) or None."""
    frame_index, candidate = current_entry
    previous_index, previous = previous_entry
    gap = int(frame_index) - int(previous_index)
    if gap <= 0:
        return float('inf'), {'forbidden': True}
    rotation_jump = _temporal_rotation_distance_deg(candidate['R'], previous['R'])
    center_jump = float(np.linalg.norm(
        candidate['camera_center'] - previous['camera_center']))
    rotation_rate = rotation_jump / gap
    center_rate = center_jump / gap
    rotation_acceleration = 0.0
    center_acceleration = 0.0
    if before_entry is not None:
        before_index, before = before_entry
        before_gap = int(previous_index) - int(before_index)
        if before_gap <= 0:
            return float('inf'), {'forbidden': True}
        average_interval = 0.5 * (before_gap + gap)
        previous_delta = previous['R'].T @ before['R']
        current_delta = candidate['R'].T @ previous['R']
        previous_velocity = cv2.Rodrigues(previous_delta)[0].reshape(3) / before_gap
        current_velocity = cv2.Rodrigues(current_delta)[0].reshape(3) / gap
        rotation_acceleration = float(np.degrees(np.linalg.norm(
            current_velocity - previous_velocity)) / average_interval)
        previous_center_velocity = (
            previous['camera_center'] - before['camera_center']) / before_gap
        current_center_velocity = (
            candidate['camera_center'] - previous['camera_center']) / gap
        center_acceleration = float(np.linalg.norm(
            current_center_velocity - previous_center_velocity) / average_interval)
    cost = (
        TEMPORAL_ROTATION_JUMP_WEIGHT * rotation_rate
        + TEMPORAL_CENTER_JUMP_WEIGHT * center_rate
        + TEMPORAL_ROTATION_ACCEL_WEIGHT * rotation_acceleration
        + TEMPORAL_CENTER_ACCEL_WEIGHT * center_acceleration)
    return float(cost), {
        'frame_gap': int(gap),
        'rotation_jump_deg': float(rotation_jump),
        'camera_center_jump_mm': float(center_jump),
        'rotation_rate_deg_per_frame': float(rotation_rate),
        'camera_center_rate_mm_per_frame': float(center_rate),
        'rotation_acceleration_deg_per_frame2': float(rotation_acceleration),
        'camera_center_acceleration_mm_per_frame2': float(center_acceleration),
        'transition_cost': float(cost),
    }


def _temporal_transition_cost(path, frame_index, candidate):
    before = path[-2] if len(path) >= 2 else None
    previous = path[-1] if path else None
    if previous is None:
        return 0.0, {}
    return _temporal_transition_triplet_cost(
        before, previous, (int(frame_index), candidate))


def _temporal_nominal_gap(nominal_probe_indices, available_indices):
    indices = sorted(set(int(x) for x in (
        nominal_probe_indices if nominal_probe_indices is not None else available_indices)))
    gaps = [b - a for a, b in zip(indices, indices[1:]) if b > a]
    if not gaps:
        return None
    return float(np.median(gaps))


def _temporal_segment_observations(
        frame_candidates, nominal_probe_indices=None,
        gap_multiplier=TEMPORAL_LARGE_GAP_MULTIPLIER,
        gap_absolute=TEMPORAL_LARGE_GAP_ABS_FRAMES):
    available = [
        (int(frame_index), list(candidates))
        for frame_index, candidates in sorted(frame_candidates.items()) if candidates
    ]
    if not available:
        return [], {'nominal_gap': None, 'split_threshold': None, 'splits': []}
    g_nom = _temporal_nominal_gap(
        nominal_probe_indices, [entry[0] for entry in available])
    threshold = float(gap_absolute) if g_nom is None else max(
        float(gap_multiplier) * float(g_nom), float(gap_absolute))
    segments = [[available[0]]]
    splits = []
    for entry in available[1:]:
        gap = entry[0] - segments[-1][-1][0]
        if gap > threshold:  # explicit: g > max(multiplier * g_nom, absolute)
            splits.append({'after_frame': segments[-1][-1][0],
                           'before_frame': entry[0], 'gap': int(gap)})
            segments.append([entry])
        else:
            segments[-1].append(entry)
    return segments, {
        'nominal_gap': g_nom,
        'split_threshold': float(threshold),
        'gap_multiplier': float(gap_multiplier),
        'gap_absolute_frames': int(gap_absolute),
        'splits': splits,
    }


def _temporal_exact_dp_segment(observations, constraints=None, transition_fn=None):
    """Exact second-order DP for one connected segment.

    constraints maps frame_index -> allowed candidate indices. transition_fn
    may return +inf to forbid a transition. All minimum-cost tied paths are
    retained in ``argmin_set`` for deterministic acceptance testing.
    """
    transition_fn = transition_fn or _temporal_transition_triplet_cost
    constraints = constraints or {}
    observations = [(int(frame), list(cands)) for frame, cands in observations]
    T = len(observations)
    if T == 0:
        return {'status': 'EMPTY', 'cost': float('inf'), 'argmin_set': set(),
                'path_indices': (), 'path': []}

    allowed = []
    for frame, candidates in observations:
        indices = list(range(len(candidates)))
        if frame in constraints:
            permitted = set(int(i) for i in constraints[frame])
            indices = [i for i in indices if i in permitted]
        if not indices:
            return {'status': 'INFEASIBLE', 'cost': float('inf'), 'argmin_set': set(),
                    'path_indices': (), 'path': []}
        allowed.append(indices)

    if T == 1:
        frame, candidates = observations[0]
        ranked = [(float(candidates[i]['emission_cost']), i) for i in allowed[0]]
        best_cost = min(cost for cost, _ in ranked)
        best_indices = sorted(i for cost, i in ranked if abs(cost - best_cost) <= 1e-12)
        argmin_set = {(i,) for i in best_indices}
        best = min(best_indices)
        return {'status': 'OK_T1', 'cost': best_cost, 'argmin_set': argmin_set,
                'path_indices': (best,), 'path': [(frame, candidates[best])]}

    f0, c0 = observations[0]
    f1, c1 = observations[1]
    states = {}
    state_paths = {}
    for i in allowed[0]:
        for j in allowed[1]:
            transition, _ = transition_fn(None, (f0, c0[i]), (f1, c1[j]))
            if not np.isfinite(transition):
                continue
            key = (i, j)
            states[key] = float(c0[i]['emission_cost']) + float(c1[j]['emission_cost']) + transition
            state_paths[key] = {(i, j)}
    if not states:
        return {'status': 'INFEASIBLE', 'cost': float('inf'), 'argmin_set': set(),
                'path_indices': (), 'path': []}

    for k in range(2, T):
        fk, ck = observations[k]
        fprev, cprev = observations[k - 1]
        fbefore, cbefore = observations[k - 2]
        next_states = {}
        next_paths = {}
        for (h, i), prefix_cost in states.items():
            for j in allowed[k]:
                transition, _ = transition_fn(
                    (fbefore, cbefore[h]), (fprev, cprev[i]), (fk, ck[j]))
                if not np.isfinite(transition):
                    continue
                value = prefix_cost + float(ck[j]['emission_cost']) + transition
                key = (i, j)
                candidate_paths = {path + (j,) for path in state_paths[(h, i)]}
                old = next_states.get(key)
                if old is None or value < old - 1e-12:
                    next_states[key] = float(value)
                    next_paths[key] = candidate_paths
                elif abs(value - old) <= 1e-12:
                    next_paths[key].update(candidate_paths)
        states = next_states
        state_paths = next_paths
        if not states:
            return {'status': 'INFEASIBLE', 'cost': float('inf'), 'argmin_set': set(),
                    'path_indices': (), 'path': []}

    best_cost = min(states.values())
    argmin_set = set()
    for key, value in states.items():
        if abs(value - best_cost) <= 1e-12:
            argmin_set.update(state_paths[key])
    best_indices = min(argmin_set)
    path = [(observations[k][0], observations[k][1][best_indices[k]]) for k in range(T)]
    return {'status': 'OK_DP', 'cost': float(best_cost), 'argmin_set': argmin_set,
            'path_indices': best_indices, 'path': path}


def _temporal_constrained_cost(observations, constraints=None, transition_fn=None):
    return _temporal_exact_dp_segment(
        observations, constraints=constraints, transition_fn=transition_fn)['cost']


def _temporal_segment_min_marginals(observations, transition_fn=None):
    base = _temporal_exact_dp_segment(observations, transition_fn=transition_fn)
    result = {}
    for frame, candidates in observations:
        values = []
        for index in range(len(candidates)):
            cost = _temporal_constrained_cost(
                observations, constraints={int(frame): {index}},
                transition_fn=transition_fn)
            values.append(float(cost))
        result[int(frame)] = values
    return base, result


def _temporal_joint_pair_min_marginal(
        dp_model, frame_a, candidate_index_a, frame_b, candidate_index_b):
    frame_a = int(frame_a)
    frame_b = int(frame_b)
    segment_a = dp_model['frame_to_segment'].get(frame_a)
    segment_b = dp_model['frame_to_segment'].get(frame_b)
    if segment_a is None or segment_b is None or segment_a != segment_b:
        return None
    observations = dp_model['segments'][segment_a]['observations']
    if len(observations) < 2:
        return None
    if frame_a == frame_b and int(candidate_index_a) != int(candidate_index_b):
        return None
    constraints = {frame_a: {int(candidate_index_a)}}
    constraints.setdefault(frame_b, set()).add(int(candidate_index_b))
    cost = _temporal_constrained_cost(observations, constraints=constraints)
    if not np.isfinite(cost):
        return None
    base_cost = dp_model['segments'][segment_a]['best_cost']
    delta = max(0.0, float(cost - base_cost))
    normalized = delta / max(len(observations), 1)
    prior = TEMPORAL_BRANCH_PRIOR_MAX_COST * float(np.clip(
        normalized / max(TEMPORAL_BRANCH_CONFIDENCE_MARGIN, 1e-12), 0.0, 1.0))
    return {
        'cost': float(cost),
        'best_cost': float(base_cost),
        'excess_cost': float(delta),
        'normalized_excess_cost': float(normalized),
        'prior_cost': float(prior),
        'segment_id': int(segment_a),
    }


def _select_temporal_pose_path(frame_candidates, nominal_probe_indices=None):
    """Exact second-order DP with large-gap segmentation and exact min-marginals."""
    segments, segmentation = _temporal_segment_observations(
        frame_candidates, nominal_probe_indices=nominal_probe_indices)
    if not segments:
        return {}, {
            'status': 'NO_POSE_CANDIDATES', 'path_cost': None,
            'second_cost': None, 'margin': None, 'normalized_margin': None,
            'frames': {}, 'segments': [], 'segmentation': segmentation,
            '_dp_model': {'segments': [], 'frame_to_segment': {}},
        }

    selected = {}
    frames = {}
    model_segments = []
    frame_to_segment = {}
    total_cost = 0.0
    for segment_id, observations in enumerate(segments):
        best, min_marginals = _temporal_segment_min_marginals(observations)
        model_segments.append({
            'observations': observations,
            'best_cost': float(best['cost']),
            'min_marginals': min_marginals,
        })
        if not np.isfinite(best['cost']):
            continue
        total_cost += float(best['cost'])
        history = []
        for local_index, (frame_index, candidate) in enumerate(best['path']):
            selected[int(frame_index)] = candidate
            frame_to_segment[int(frame_index)] = int(segment_id)
            candidate_index = best['path_indices'][local_index]
            candidate['temporal_candidate_index'] = int(candidate_index)
            transition, details = _temporal_transition_cost(
                history, frame_index, candidate)
            values = min_marginals[int(frame_index)]
            finite_values = sorted(value for value in values if np.isfinite(value))
            gap = None
            if len(finite_values) >= 2:
                gap = float(finite_values[1] - finite_values[0])
            frames[int(frame_index)] = {
                'chosen_label': candidate.get('label'),
                'source': candidate.get('source'),
                'seed_marker_id': candidate.get('seed_marker_id'),
                'seed_branch': candidate.get('seed_branch'),
                'parent_seed': candidate.get('parent_seed'),
                'basin': candidate.get('basin'),
                'candidate_count': len(frame_candidates[frame_index]),
                'candidate_index': int(candidate_index),
                'inlier_marker_ids': list(candidate.get('inlier_marker_ids', [])),
                'outlier_marker_ids': list(candidate.get('outlier_marker_ids', [])),
                'reprojection_rms_px': candidate.get('reprojection_rms_px'),
                'emission_cost': candidate.get('emission_cost'),
                'camera_center_world_mm': candidate['camera_center'].tolist(),
                'individual_min_marginal_costs': [
                    None if not np.isfinite(value) else float(value) for value in values],
                'individual_min_marginal_gap': gap,
                'segment_id': int(segment_id),
                **details,
            }
            history.append((frame_index, candidate))

    dp_model = {'segments': model_segments, 'frame_to_segment': frame_to_segment}
    return selected, {
        'status': 'OK_TEMPORAL_EXACT_DP' if selected else 'NO_FEASIBLE_TEMPORAL_PATH',
        'observation_count': len(selected),
        'path_cost': float(total_cost) if selected else None,
        'second_cost': None,
        'margin': None,
        'normalized_margin': None,
        'frames': frames,
        'segments': [
            {'segment_id': i,
             'frames': [int(frame) for frame, _ in segment['observations']],
             'best_cost': segment['best_cost']}
            for i, segment in enumerate(model_segments)
        ],
        'segmentation': segmentation,
        '_dp_model': dp_model,
    }


def _expand_temporal_probe_indices(core_indices, allowed_range, radius=None):
    radius = TEMPORAL_NEIGHBOR_RADIUS if radius is None else int(radius)
    allowed = set(int(index) for index in allowed_range)
    return sorted({
        neighbor
        for index in core_indices
        for neighbor in range(int(index) - radius, int(index) + radius + 1)
        if neighbor in allowed
    })



def _unified_endpoint_reprojection_stats(
        rotation, translation, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm):
    """Direct per-endpoint reprojection against fixed mapped 3-D marker corners."""
    rms, max_error, depth = _temporal_pose_marker_metrics(
        rotation, translation, corners_dict, marker_map, camera_matrix,
        distortion, marker_size_mm)
    valid = sorted(
        marker_id for marker_id in rms
        if depth.get(marker_id, False) and np.isfinite(rms[marker_id]))
    if not valid:
        return None
    # Recompute actual corner residuals once so each endpoint/marker contributes once.
    residuals = []
    per_marker = []
    rvec = cv2.Rodrigues(np.asarray(rotation, np.float64).reshape(3, 3))[0]
    tvec = np.asarray(translation, np.float64).reshape(3, 1)
    for marker_id in valid:
        obj = _temporal_marker_reference_points(marker_id, marker_map, marker_size_mm)
        projected, _ = cv2.projectPoints(
            obj, rvec, tvec, np.asarray(camera_matrix, np.float64), distortion)
        observed = np.asarray(corners_dict[marker_id], np.float64).reshape(4, 2)
        err = np.linalg.norm(projected.reshape(4, 2) - observed, axis=1)
        residuals.extend(err.tolist())
        per_marker.append({
            'marker_id': int(marker_id),
            'rms_px': float(np.sqrt(np.mean(err ** 2))),
            'max_px': float(np.max(err)),
        })
    values = np.asarray(residuals, np.float64)
    return {
        'marker_count': len(valid),
        'marker_ids': valid,
        'rms_px': float(np.sqrt(np.mean(values ** 2))),
        'mean_px': float(np.mean(values)),
        'median_px': float(np.median(values)),
        'max_px': float(np.max(values)),
        'per_marker': per_marker,
    }


def _unified_pair_reprojection_stats(
        R_A, t_A, corners_A, R_B, t_B, corners_B, marker_map,
        camera_matrix, distortion, marker_size_mm):
    """Branch-specific endpoint reprojection; A/B need not share marker IDs."""
    stats_A = _unified_endpoint_reprojection_stats(
        R_A, t_A, corners_A, marker_map, camera_matrix, distortion, marker_size_mm)
    stats_B = _unified_endpoint_reprojection_stats(
        R_B, t_B, corners_B, marker_map, camera_matrix, distortion, marker_size_mm)
    if stats_A is None or stats_B is None:
        return None
    # Equal endpoint weighting avoids one endpoint silently dominating by marker count.
    mean_px = 0.5 * (stats_A['mean_px'] + stats_B['mean_px'])
    rms_px = math.sqrt(0.5 * (stats_A['rms_px'] ** 2 + stats_B['rms_px'] ** 2))
    return {
        'mean_px': float(mean_px),
        'rms_px': float(rms_px),
        'max_px': float(max(stats_A['max_px'], stats_B['max_px'])),
        'endpoint_A': stats_A,
        'endpoint_B': stats_B,
        'shared_marker_count': int(len(set(stats_A['marker_ids']) & set(stats_B['marker_ids']))),
    }


def _unified_roi_grid_points(frame_width, frame_height, roi_bounds=None, cols=5, rows=4):
    if roi_bounds is None:
        x0, y0, x1, y1 = 0.0, 0.0, float(frame_width), float(frame_height)
    else:
        x0, y0, x1, y1 = [float(v) for v in roi_bounds]
    # Avoid extreme image borders where tiny calibration/cropping errors dominate.
    margin_x = 0.08 * max(x1 - x0, 1.0)
    margin_y = 0.08 * max(y1 - y0, 1.0)
    xs = np.linspace(x0 + margin_x, x1 - margin_x, max(int(cols), 1))
    ys = np.linspace(y0 + margin_y, y1 - margin_y, max(int(rows), 1))
    return np.asarray([(x, y) for y in ys for x in xs], np.float64)


def _unified_nominal_pair_geometry(
        R_A_from_B, t_A_from_B, K, frame_width, frame_height,
        nominal_depth_mm=PAIR_NOMINAL_DEPTH_MM, roi_B=None, roi_A=None,
        cols=PAIR_NOMINAL_RAY_COLS, rows=PAIR_NOMINAL_RAY_ROWS,
        match_sigma_px=PAIR_MATCH_SIGMA_PX):
    """Cheap depth-conditioning proxy from nominal-depth ROI rays.

    Relative pose convention is X_A = R_A_from_B X_B + t_A_from_B.
    """
    R = np.asarray(R_A_from_B, np.float64).reshape(3, 3)
    t = np.asarray(t_A_from_B, np.float64).reshape(3, 1)
    K64 = np.asarray(K, np.float64).reshape(3, 3)
    K_inv = np.linalg.inv(K64)
    pixels_B = _unified_roi_grid_points(
        frame_width, frame_height, roi_B, cols=cols, rows=rows)
    hom = np.column_stack([pixels_B, np.ones(len(pixels_B))])
    rays_B = (K_inv @ hom.T).T
    rays_B /= np.maximum(rays_B[:, 2:3], 1e-12)
    X_B = rays_B * float(nominal_depth_mm)
    X_A = (R @ X_B.T + t).T
    uvw_A = (K64 @ X_A.T).T
    valid_z = X_A[:, 2] > 1e-6
    uv_A = uvw_A[:, :2] / np.maximum(uvw_A[:, 2:3], 1e-12)
    if roi_A is None:
        ax0, ay0, ax1, ay1 = 0.0, 0.0, float(frame_width), float(frame_height)
    else:
        ax0, ay0, ax1, ay1 = [float(v) for v in roi_A]
    inside = (
        valid_z & (uv_A[:, 0] >= ax0) & (uv_A[:, 0] < ax1)
        & (uv_A[:, 1] >= ay0) & (uv_A[:, 1] < ay1))
    overlap = float(np.count_nonzero(inside)) / max(len(inside), 1)

    ray_B_unit = X_B / np.maximum(np.linalg.norm(X_B, axis=1, keepdims=True), 1e-12)
    ray_A_unit = X_A / np.maximum(np.linalg.norm(X_A, axis=1, keepdims=True), 1e-12)
    ray_A_in_B = (R.T @ ray_A_unit.T).T
    cosine = np.sum(ray_B_unit * ray_A_in_B, axis=1)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    usable_angles = angles[inside]
    if not len(usable_angles):
        return {
            'overlap_ratio': overlap,
            'triangulation_median_deg': 0.0,
            'triangulation_p10_deg': 0.0,
            'predicted_depth_sigma_mm': float('inf'),
            'ray_count': int(len(angles)),
            'valid_ray_count': int(np.count_nonzero(inside)),
        }
    median_angle = float(np.median(usable_angles))
    p10_angle = float(np.percentile(usable_angles, 10))
    focal = 0.5 * (float(K64[0, 0]) + float(K64[1, 1]))
    conservative_angle = max(p10_angle, 1e-6)
    depth_sigma = (
        float(nominal_depth_mm) * float(match_sigma_px)
        / max(focal * math.tan(math.radians(conservative_angle)), 1e-9))
    return {
        'overlap_ratio': overlap,
        'triangulation_median_deg': median_angle,
        'triangulation_p10_deg': p10_angle,
        'predicted_depth_sigma_mm': float(depth_sigma),
        'ray_count': int(len(angles)),
        'valid_ray_count': int(np.count_nonzero(inside)),
    }



def _unified_depth_geometry_score(
        geometry, baseline_mm, ideal_baseline_mm=IDEAL_BASELINE_MM,
        target_depth_sigma_mm=PAIR_TARGET_DEPTH_SIGMA_MM,
        target_angle_deg=PAIR_TARGET_TRIANGULATION_ANGLE_DEG):
    """Depth-oriented cheap geometry score; ideal baseline is only a weak tie-breaker."""
    depth_sigma = float(geometry.get('predicted_depth_sigma_mm', float('inf')))
    uncertainty_penalty = min(max(depth_sigma / max(target_depth_sigma_mm, 1e-9) - 1.0, 0.0), 4.0)
    p10 = float(geometry.get('triangulation_p10_deg', 0.0))
    angle_penalty = max(0.0, target_angle_deg - p10) / max(target_angle_deg, 1e-9)
    overlap_penalty = 1.0 - float(np.clip(geometry.get('overlap_ratio', 0.0), 0.0, 1.0))
    baseline_tie = abs(float(baseline_mm) - float(ideal_baseline_mm)) / max(float(ideal_baseline_mm), 1e-9)
    return float(
        PAIR_SCORE_DEPTH_UNCERTAINTY_W * uncertainty_penalty
        + PAIR_SCORE_ANGLE_W * angle_penalty
        + PAIR_SCORE_OVERLAP_W * overlap_penalty
        + PAIR_SCORE_IDEAL_BASELINE_TIE_W * baseline_tie)

def _pattern_guided_resolve_config(overrides=None):
    cfg = dict(PATTERN_GUIDED_DEFAULT_CONFIG)
    cfg['weights'] = dict(PATTERN_GUIDED_DEFAULT_CONFIG['weights'])
    if overrides:
        for key, value in dict(overrides).items():
            if key == 'weights' and value is not None:
                cfg['weights'].update(dict(value))
            else:
                cfg[key] = value
    cfg['max_total_aruco_probes'] = max(5, int(cfg['max_total_aruco_probes']))
    cfg['max_extra_probes'] = max(0, min(
        int(cfg['max_extra_probes']), cfg['max_total_aruco_probes'] - 5))
    return cfg


def _pattern_guided_soft_band_penalty(value, full_band, outer_band):
    """0 inside full band, smooth linear rise through outer band, capped at 2."""
    value = float(value)
    if not np.isfinite(value):
        return 2.0
    fl, fh = [float(x) for x in full_band]
    ol, oh = [float(x) for x in outer_band]
    if fl <= value <= fh:
        return 0.0
    if ol <= value < fl:
        return float((fl - value) / max(fl - ol, 1e-9))
    if fh < value <= oh:
        return float((value - fh) / max(oh - fh, 1e-9))
    if value < ol:
        return float(min(2.0, 1.0 + (ol - value) / max(fl - ol, 1e-9)))
    return float(min(2.0, 1.0 + (value - oh) / max(oh - fh, 1e-9)))


def _pattern_guided_lower_bound_penalty(value, full_min, outer_min):
    value = float(value)
    if not np.isfinite(value):
        return 2.0
    full_min = float(full_min)
    outer_min = float(outer_min)
    if value >= full_min:
        return 0.0
    if value >= outer_min:
        return float((full_min - value) / max(full_min - outer_min, 1e-9))
    return float(min(2.0, 1.0 + (outer_min - value) / max(full_min - outer_min, 1e-9)))


def _pattern_guided_weighted_median(values, weights):
    values = np.asarray(values, np.float64).reshape(-1)
    weights = np.asarray(weights, np.float64).reshape(-1)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return None
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(np.sum(weights))
    index = int(np.searchsorted(np.cumsum(weights), cutoff, side='left'))
    return float(values[min(index, len(values) - 1)])


def _pattern_guided_marker_observability(
        candidate, corners_dict, marker_map, marker_size_mm,
        marker_map_diagnostics=None):
    """Pose-derived per-marker observability.  No PnP is executed here.

    marker_map stores T_W<-M.  candidate is T_C<-W.  Incidence is the acute
    angle between the marker's own mapped normal and its viewing line, so each
    non-coplanar marker keeps its own normal and plane.
    """
    R_CW = np.asarray(candidate['R'], np.float64).reshape(3, 3)
    t_CW = np.asarray(candidate['t'], np.float64).reshape(3, 1)
    C_W = np.asarray(candidate.get(
        'camera_center', _temporal_camera_center(R_CW, t_CW)), np.float64).reshape(3)
    measurement_confidence = float(candidate.get('measurement_confidence', 0.75))
    rms_by_id = candidate.get('per_marker_rms_px', {}) or {}
    entries = []
    for marker_id in sorted(set(int(x) for x in corners_dict) & set(marker_map)):
        R_WM, t_WM = marker_map[marker_id]
        R_WM = np.asarray(R_WM, np.float64).reshape(3, 3)
        center_W = np.asarray(t_WM, np.float64).reshape(3)
        normal_W = R_WM[:, 2]
        normal_W /= max(float(np.linalg.norm(normal_W)), 1e-12)
        view_W = center_W - C_W
        range_mm = float(np.linalg.norm(view_W))
        if range_mm <= 1e-12:
            incidence = 90.0
        else:
            # abs() intentionally removes arbitrary normal sign; incidence is
            # observability, not a front/back hard gate.
            cosine = abs(float(np.dot(normal_W, view_W / range_mm)))
            incidence = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        center_C = (R_CW @ center_W.reshape(3, 1) + t_CW).reshape(3)
        axial_depth_mm = float(center_C[2])
        pts = np.asarray(corners_dict[marker_id], np.float64).reshape(4, 2)
        edges = np.asarray([
            np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)
        ], np.float64)
        short_edge_px = float(np.min(edges)) if len(edges) else 0.0
        area_px2 = abs(float(cv2.contourArea(pts.astype(np.float32))))
        reproj = float(rms_by_id.get(marker_id, candidate.get('reprojection_rms_px', 3.0)))
        map_conf = _unified_map_marker_confidence(
            marker_id, marker_map_diagnostics, marker_size_mm)
        weight = (
            max(measurement_confidence, 0.05)
            * max(map_conf, 0.05)
            * math.exp(-0.5 * (reproj / 2.0) ** 2)
            * float(np.clip(short_edge_px / 50.0, 0.20, 1.20)))
        entries.append({
            'marker_id': int(marker_id),
            'range_mm': range_mm,
            'axial_depth_mm': axial_depth_mm,
            'incidence_deg': incidence,
            'projected_short_edge_px': short_edge_px,
            'projected_area_px2': area_px2,
            'reprojection_rms_px': reproj,
            'map_confidence': float(map_conf),
            'weight': float(weight),
            'normal_W': normal_W.copy(),
            'center_W': center_W.copy(),
        })
    if not entries:
        return {
            'per_marker': [], 'range_mm': None, 'axial_depth_mm': None,
            'incidence_deg': None, 'projected_short_edge_px': None,
            'projected_area_px2': None, 'aggregate_weight': 0.0,
        }
    weights = [entry['weight'] for entry in entries]
    def wm(key):
        return _pattern_guided_weighted_median(
            [entry[key] for entry in entries], weights)
    return {
        'per_marker': entries,
        'range_mm': wm('range_mm'),
        'axial_depth_mm': wm('axial_depth_mm'),
        'incidence_deg': wm('incidence_deg'),
        'projected_short_edge_px': wm('projected_short_edge_px'),
        'projected_area_px2': wm('projected_area_px2'),
        'aggregate_weight': float(np.sum(weights)),
    }


def _pattern_guided_orbit_baseline_diagnostic(radius_mm, central_angle_deg):
    """Equal-radius circular diagnostic only; never used for production baseline."""
    return float(2.0 * float(radius_mm) * math.sin(
        0.5 * math.radians(float(central_angle_deg))))


def _pattern_guided_pair_geometry_score(
        geometry, baseline_mm, nominal_depth_mm, metrics_A, metrics_B,
        config, ideal_baseline_mm=IDEAL_BASELINE_MM):
    """Soft pattern-guided cheap score; lower is better.

    Exact baseline must already come from the two common-world camera centres /
    equivalent relative translation.  Incidence is only a soft observability term;
    triangulation and B/Z carry the depth-conditioning preference.
    """
    cfg = config
    w = cfg['weights']
    depth_sigma = float(geometry.get('predicted_depth_sigma_mm', float('inf')))
    depth_pen = max(0.0, depth_sigma / max(float(cfg['depth_sigma_target_mm']), 1e-9) - 1.0)
    depth_pen = min(depth_pen, 2.0)
    median_angle = float(geometry.get('triangulation_median_deg', 0.0))
    p10 = float(geometry.get('triangulation_p10_deg', 0.0))
    tri_pen = _pattern_guided_soft_band_penalty(
        median_angle, cfg['triangulation_sweet_deg'], cfg['triangulation_outer_deg'])
    p10_pen = _pattern_guided_lower_bound_penalty(
        p10, cfg['triangulation_p10_target_deg'], cfg['triangulation_p10_outer_deg'])
    overlap_pen = 1.0 - float(np.clip(geometry.get('overlap_ratio', 0.0), 0.0, 1.0))
    bz = float(baseline_mm) / max(float(nominal_depth_mm), 1e-9)
    bz_pen = _pattern_guided_soft_band_penalty(bz, cfg['bz_sweet'], cfg['bz_outer'])

    distances = [x for x in [metrics_A.get('range_mm'), metrics_B.get('range_mm')] if x is not None]
    distance_pen = float(np.mean([
        _pattern_guided_soft_band_penalty(
            value, cfg['distance_full_mm'], cfg['distance_outer_mm'])
        for value in distances])) if distances else 1.0
    incidences = [x for x in [metrics_A.get('incidence_deg'), metrics_B.get('incidence_deg')] if x is not None]
    incidence_parts = []
    for value in incidences:
        preferred = _pattern_guided_soft_band_penalty(
            value, cfg['incidence_preferred_deg'], cfg['incidence_outer_deg'])
        sweet = _pattern_guided_soft_band_penalty(
            value, cfg['incidence_sweet_deg'], cfg['incidence_preferred_deg'])
        incidence_parts.append(0.75 * preferred + 0.25 * sweet)
    incidence_pen = float(np.mean(incidence_parts)) if incidence_parts else 1.0

    if len(distances) == 2:
        range_mismatch = abs(distances[0] - distances[1]) / max(0.5 * sum(distances), 1e-9)
    else:
        range_mismatch = 0.0
    mismatch_pen = 0.0 if range_mismatch <= cfg['range_mismatch_full'] else min(
        2.0, (range_mismatch - cfg['range_mismatch_full'])
        / max(cfg['range_mismatch_outer'] - cfg['range_mismatch_full'], 1e-9))

    short_edges = [x for x in [metrics_A.get('projected_short_edge_px'), metrics_B.get('projected_short_edge_px')] if x is not None]
    marker_size_pen = float(np.mean([
        _pattern_guided_lower_bound_penalty(
            value, cfg['marker_short_edge_full_px'], cfg['marker_short_edge_outer_px'])
        for value in short_edges])) if short_edges else 1.0
    baseline_tie = abs(float(baseline_mm) - float(ideal_baseline_mm)) / max(float(ideal_baseline_mm), 1e-9)
    score = (
        w['depth_sigma'] * depth_pen
        + w['triangulation'] * tri_pen
        + w['p10'] * p10_pen
        + w['overlap'] * overlap_pen
        + w['bz'] * bz_pen
        + w['distance'] * distance_pen
        + w['incidence'] * incidence_pen
        + w['range_mismatch'] * mismatch_pen
        + w['marker_size'] * marker_size_pen
        + w['ideal_baseline_tie'] * baseline_tie)
    diagnostics = {
        'pattern_guided_score': float(score),
        'nominal_depth_mm': float(nominal_depth_mm),
        'baseline_to_depth': float(bz),
        'range_mismatch_ratio': float(range_mismatch),
        'depth_sigma_penalty': float(depth_pen),
        'triangulation_penalty': float(tri_pen),
        'triangulation_p10_penalty': float(p10_pen),
        'overlap_penalty': float(overlap_pen),
        'bz_penalty': float(bz_pen),
        'distance_penalty': float(distance_pen),
        'incidence_penalty': float(incidence_pen),
        'range_mismatch_penalty': float(mismatch_pen),
        'marker_size_penalty': float(marker_size_pen),
        'ideal_baseline_tie_penalty': float(baseline_tie),
    }
    return float(score), diagnostics


def _pattern_guided_basic_geometry_ok(geometry, baseline_mm, nominal_depth_mm, config):
    bz = float(baseline_mm) / max(float(nominal_depth_mm), 1e-9)
    return bool(
        float(geometry.get('overlap_ratio', 0.0)) >= float(config['basic_overlap_min'])
        and float(config['basic_triangulation_median_min_deg'])
            <= float(geometry.get('triangulation_median_deg', 0.0))
            <= float(config['basic_triangulation_median_max_deg'])
        and float(geometry.get('triangulation_p10_deg', 0.0))
            >= float(config['basic_triangulation_p10_min_deg'])
        and float(config['basic_bz_min']) <= bz <= float(config['basic_bz_max'])
        and float(geometry.get('predicted_depth_sigma_mm', float('inf')))
            <= float(config['basic_depth_sigma_max_mm']))


def _pattern_guided_linear_center_trend(indices, selected_path, min_speed=0.02):
    observations = [
        (float(index), np.asarray(selected_path[index]['camera_center'], np.float64).reshape(3))
        for index in sorted(set(int(x) for x in indices)) if int(index) in selected_path
    ]
    if len(observations) < 2:
        return None
    times = np.asarray([x[0] for x in observations], np.float64)
    centers = np.asarray([x[1] for x in observations], np.float64)
    tc = times - float(np.mean(times))
    denom = float(np.dot(tc, tc))
    if denom <= 1e-12:
        return None
    velocity = (tc[:, None] * centers).sum(axis=0) / denom
    speed = float(np.linalg.norm(velocity))
    if speed < float(min_speed):
        return None
    anchor_t = float(np.mean(times))
    anchor_C = np.mean(centers, axis=0)
    return {'anchor_frame': anchor_t, 'anchor_center': anchor_C, 'velocity': velocity, 'speed_mm_per_frame': speed}


def _pattern_guided_target_index_from_trend(
        trend, opposite_center, target_baseline_mm, frame_min, frame_max,
        already_sampled):
    if trend is None:
        return None
    v = np.asarray(trend['velocity'], np.float64).reshape(3)
    d0 = np.asarray(trend['anchor_center'], np.float64).reshape(3) - np.asarray(opposite_center, np.float64).reshape(3)
    t0 = float(trend['anchor_frame'])
    a = float(np.dot(v, v))
    b = float(2.0 * np.dot(d0, v))
    c = float(np.dot(d0, d0) - float(target_baseline_mm) ** 2)
    roots = []
    if a > 1e-12:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sd = math.sqrt(disc)
            roots = [t0 + (-b - sd) / (2.0 * a), t0 + (-b + sd) / (2.0 * a)]
    candidates = []
    for value in roots:
        index = int(round(value))
        if int(frame_min) <= index <= int(frame_max) and index not in already_sampled:
            candidates.append(index)
    if not candidates:
        return None
    # Prefer a novel target closest to the trend's observed temporal support.
    return int(min(candidates, key=lambda x: abs(float(x) - t0)))


def _pattern_guided_propose_extra_probe_indices(
        sampled_start, sampled_end, selected_path, start_range, end_range,
        nominal_depth_mm, config):
    """Bounded pose-trend proposal.  No image scan and no SIFT is performed."""
    sampled_start = sorted(set(int(x) for x in sampled_start))
    sampled_end = sorted(set(int(x) for x in sampled_end))
    all_existing = set(sampled_start + sampled_end)
    start_valid = [idx for idx in sampled_start if idx in selected_path]
    end_valid = [idx for idx in sampled_end if idx in selected_path]
    if not start_valid or not end_valid:
        return {'start': [], 'end': [], 'status': 'INSUFFICIENT_ENDPOINT_POSES'}
    target_baseline = float(config['target_bz']) * float(nominal_depth_mm)
    # Anchor on the existing pair whose exact centre baseline is closest to target.
    pair_options = []
    for ia in start_valid:
        for ib in end_valid:
            Ca = np.asarray(selected_path[ia]['camera_center'], np.float64).reshape(3)
            Cb = np.asarray(selected_path[ib]['camera_center'], np.float64).reshape(3)
            baseline = float(np.linalg.norm(Ca - Cb))
            pair_options.append((abs(baseline - target_baseline), ia, ib, baseline))
    _, anchor_a, anchor_b, current_baseline = min(pair_options)
    Ca = np.asarray(selected_path[anchor_a]['camera_center'], np.float64).reshape(3)
    Cb = np.asarray(selected_path[anchor_b]['camera_center'], np.float64).reshape(3)
    start_values = list(start_range)
    end_values = list(end_range)
    if not start_values or not end_values:
        return {'start': [], 'end': [], 'status': 'EMPTY_RANGE'}
    trend_start = _pattern_guided_linear_center_trend(
        sampled_start, selected_path, config['min_center_trend_mm_per_frame'])
    trend_end = _pattern_guided_linear_center_trend(
        sampled_end, selected_path, config['min_center_trend_mm_per_frame'])
    proposals = []
    for side, trend, opposite, lo, hi in [
        ('start', trend_start, Cb, start_values[0], start_values[-1]),
        ('end', trend_end, Ca, end_values[0], end_values[-1]),
    ]:
        index = _pattern_guided_target_index_from_trend(
            trend, opposite, target_baseline, lo, hi, all_existing)
        if index is not None:
            proposals.append((side, index))
            all_existing.add(index)
    # If the exact root already coincides with a core probe, a bounded midpoint
    # between the anchor and the segment extreme is a useful second candidate.
    if len(proposals) < int(config['max_extra_probes']):
        if current_baseline < target_baseline:
            fallback_specs = [
                ('start', anchor_a, start_values[0]),
                ('end', anchor_b, end_values[-1]),
            ]
        else:
            fallback_specs = [
                ('start', anchor_a, int(round(np.mean(sampled_start)))),
                ('end', anchor_b, int(round(np.mean(sampled_end)))),
            ]
        for side, anchor, target in fallback_specs:
            if len(proposals) >= int(config['max_extra_probes']):
                break
            index = int(round(0.5 * (float(anchor) + float(target))))
            allowed = start_values if side == 'start' else end_values
            if allowed[0] <= index <= allowed[-1] and index not in all_existing:
                proposals.append((side, index))
                all_existing.add(index)
    proposals = proposals[:int(config['max_extra_probes'])]
    return {
        'start': [idx for side, idx in proposals if side == 'start'],
        'end': [idx for side, idx in proposals if side == 'end'],
        'status': 'OK' if proposals else 'INSUFFICIENT_POSE_TREND',
        'target_baseline_mm': float(target_baseline),
        'anchor_pair': (int(anchor_a), int(anchor_b)),
        'anchor_baseline_mm': float(current_baseline),
    }



def _local_window_resolve_config(overrides=None):
    config = dict(LOCAL_WINDOW_DEFAULT_CONFIG)
    if overrides:
        for key, value in dict(overrides).items():
            if key in config:
                config[key] = value
    config['radius'] = max(0, int(config['radius']))
    config['stride'] = max(1, int(config['stride']))
    config['max_unique_new_frames'] = max(0, int(config['max_unique_new_frames']))
    config['max_endpoint_candidates'] = max(1, int(config['max_endpoint_candidates']))
    config['klt_max_points'] = max(8, int(config['klt_max_points']))
    config['klt_min_tracks'] = max(4, int(config['klt_min_tracks']))
    return config


def _local_window_indices(center_index, allowed_range, total_frames, radius=2, stride=1):
    """Return deterministic clipped local indices inside one endpoint segment."""
    center_index = int(center_index)
    allowed = set(int(x) for x in allowed_range if 0 <= int(x) < int(total_frames))
    radius = max(0, int(radius))
    stride = max(1, int(stride))
    indices = [
        center_index + offset
        for offset in range(-radius * stride, radius * stride + 1, stride)
        if center_index + offset in allowed
    ]
    if center_index in allowed and center_index not in indices:
        indices.append(center_index)
    return sorted(set(indices))


def _local_so3_blend(R_raw, R_target, alpha):
    R_raw = np.asarray(R_raw, np.float64).reshape(3, 3)
    R_target = np.asarray(R_target, np.float64).reshape(3, 3)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    delta = R_target @ R_raw.T
    rvec, _ = cv2.Rodrigues(delta)
    step, _ = cv2.Rodrigues(rvec * alpha)
    return step @ R_raw


def _local_klt_track_pair(gray_prev, gray_curr, camera_matrix, marker_corners=None, config=None):
    """Lightweight adjacent-frame KLT validation. No SIFT/descriptor work occurs."""
    cfg = _local_window_resolve_config(config)
    g0 = np.asarray(gray_prev)
    g1 = np.asarray(gray_curr)
    scale = float(cfg['klt_scale'])
    if scale != 1.0:
        g0s = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        g1s = cv2.resize(g1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        g0s, g1s = g0, g1
    mask = np.full(g0s.shape[:2], 255, np.uint8)
    if marker_corners:
        for points in marker_corners.values():
            quad = np.asarray(points, np.float32).reshape(4, 2) * scale
            center = quad.mean(axis=0)
            expanded = center + 1.35 * (quad - center)
            cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 0)
    pts0 = cv2.goodFeaturesToTrack(
        g0s, maxCorners=int(cfg['klt_max_points']), qualityLevel=0.01,
        minDistance=7.0, blockSize=7, mask=mask)
    result = {
        'available': False, 'initial_count': 0, 'valid_count': 0,
        'fb_median_px': None, 'fb_p90_px': None, 'grid_coverage': 0.0,
        'rotation_deg': None, 'R_curr_from_prev': None,
        'essential_inlier_ratio': None, 'homography_inlier_ratio': None,
        'recoverpose_inlier_count': 0,
        'recoverpose_min_required': max(8, int(cfg['klt_min_tracks']) // 2),
        'planar_degenerate': False, 'reason': 'NO_FEATURES',
    }
    if pts0 is None or len(pts0) < int(cfg['klt_min_tracks']):
        result['initial_count'] = 0 if pts0 is None else int(len(pts0))
        result['reason'] = 'TOO_FEW_INITIAL_TRACKS'
        return result
    result['initial_count'] = int(len(pts0))
    lk = dict(winSize=(int(cfg['klt_win_size']), int(cfg['klt_win_size'])),
              maxLevel=int(cfg['klt_max_level']),
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(g0s, g1s, pts0, None, **lk)
    if pts1 is None:
        result['reason'] = 'FORWARD_LK_FAILED'
        return result
    pts0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1s, g0s, pts1, None, **lk)
    if pts0b is None:
        result['reason'] = 'BACKWARD_LK_FAILED'
        return result
    p0 = pts0.reshape(-1, 2)
    p1 = pts1.reshape(-1, 2)
    p0b = pts0b.reshape(-1, 2)
    status = np.asarray(st1).reshape(-1).astype(bool) & np.asarray(st2).reshape(-1).astype(bool)
    fb_full = np.linalg.norm(p0b - p0, axis=1) / max(scale, 1e-9)
    valid = status & np.isfinite(fb_full) & (fb_full <= float(cfg['klt_fb_gate_px']))
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        result['reason'] = 'FB_GATE_REJECTED_ALL'
        return result
    # Spatial cap/grid balance.
    h, w = g0s.shape[:2]
    counts = {}
    kept = []
    order = ids[np.argsort(fb_full[ids])]
    for idx in order:
        x, y = p0[idx]
        gx = min(int(cfg['klt_grid_cols']) - 1, max(0, int(x * int(cfg['klt_grid_cols']) / max(w, 1))))
        gy = min(int(cfg['klt_grid_rows']) - 1, max(0, int(y * int(cfg['klt_grid_rows']) / max(h, 1))))
        cell = (gx, gy)
        if counts.get(cell, 0) >= int(cfg['klt_max_per_cell']):
            continue
        counts[cell] = counts.get(cell, 0) + 1
        kept.append(int(idx))
    if not kept:
        result['reason'] = 'GRID_BALANCE_REJECTED_ALL'
        return result
    kept = np.asarray(kept, int)
    p0f = p0[kept] / max(scale, 1e-9)
    p1f = p1[kept] / max(scale, 1e-9)
    fberr = fb_full[kept]
    result['valid_count'] = int(len(kept))
    result['fb_median_px'] = float(np.median(fberr))
    result['fb_p90_px'] = float(np.percentile(fberr, 90.0))
    result['grid_coverage'] = float(len(counts) / max(int(cfg['klt_grid_cols']) * int(cfg['klt_grid_rows']), 1))
    if len(kept) < int(cfg['klt_min_tracks']):
        result['reason'] = 'TOO_FEW_FB_TRACKS'
        return result
    # Homography is diagnostic only; a highly planar patch does not invalidate FB tracking.
    H, hm = cv2.findHomography(p0f, p1f, cv2.RANSAC, 2.0)
    if hm is not None and len(hm):
        result['homography_inlier_ratio'] = float(np.mean(np.asarray(hm).reshape(-1) > 0))
    try:
        E, em = cv2.findEssentialMat(
            p0f, p1f, np.asarray(camera_matrix, np.float64),
            method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is not None and em is not None:
            inlier_ratio = float(np.mean(np.asarray(em).reshape(-1) > 0))
            result['essential_inlier_ratio'] = inlier_ratio
            count, R, _t, _mask = cv2.recoverPose(
                E, p0f, p1f, np.asarray(camera_matrix, np.float64), mask=em)
            required_count = int(result['recoverpose_min_required'])
            result['recoverpose_inlier_count'] = int(count)
            if int(count) >= required_count:
                result['R_curr_from_prev'] = np.asarray(R, np.float64)
                result['rotation_deg'] = _temporal_rotation_distance_deg(R, np.eye(3))
                result['available'] = True
                result['reason'] = 'OK'
            else:
                result['reason'] = 'RECOVERPOSE_TOO_FEW_CHEIRALITY_INLIERS'
        else:
            result['reason'] = 'ESSENTIAL_NOT_FOUND'
    except cv2.error:
        result['reason'] = 'ESSENTIAL_FAILED'
    hratio = result.get('homography_inlier_ratio')
    eratio = result.get('essential_inlier_ratio')
    if hratio is not None and eratio is not None and hratio > 0.90 and eratio < 0.55:
        result['planar_degenerate'] = True
        # Keep FB statistics for blur/outlier diagnostics but do not trust rotation.
        result['R_curr_from_prev'] = None
        result['rotation_deg'] = None
        result['available'] = False
        result['reason'] = 'PLANAR_DEGENERATE'
    return result


def _local_window_transition_cost(before_entry, previous_entry, current_entry, klt_cache, config):
    base_cost, details = _temporal_transition_triplet_cost(before_entry, previous_entry, current_entry)
    if previous_entry is None or not np.isfinite(base_cost):
        return base_cost, details
    cfg = _local_window_resolve_config(config)
    prev_frame, prev_candidate = previous_entry
    curr_frame, curr_candidate = current_entry
    klt = klt_cache.get((int(prev_frame), int(curr_frame)))
    klt_cost = 0.0
    rotation_delta = None
    if klt and klt.get('R_curr_from_prev') is not None:
        predicted = np.asarray(klt['R_curr_from_prev'], np.float64)
        candidate_rel = np.asarray(curr_candidate['R'], np.float64) @ np.asarray(prev_candidate['R'], np.float64).T
        rotation_delta = _temporal_rotation_distance_deg(candidate_rel, predicted)
        klt_cost = float(cfg['klt_rotation_weight']) * rotation_delta / max(float(cfg['klt_rotation_gate_deg']), 1e-9)
    details = dict(details)
    details.update({'klt_rotation_delta_deg': rotation_delta, 'klt_transition_cost': float(klt_cost)})
    return float(base_cost + klt_cost), details


def _local_window_path(observations, anchor_frame, anchor_candidate, sharpness_by_frame, klt_cache, config):
    """Exact local branch DP with weak anchor/blur priors and optional KLT rotation validation."""
    cfg = _local_window_resolve_config(config)
    if not observations:
        return {}, {'status': 'NO_LOCAL_CANDIDATES', 'branch_sequence': []}
    max_sharp = max([float(sharpness_by_frame.get(int(f), 0.0)) for f, _ in observations] + [1.0])
    adjusted = []
    for frame, candidates in observations:
        copies = []
        gap = max(abs(int(frame) - int(anchor_frame)), 1)
        for original_index, candidate in enumerate(candidates):
            c = dict(candidate)
            c['_local_original_index'] = int(original_index)
            rot = _temporal_rotation_distance_deg(c['R'], anchor_candidate['R'])
            center = float(np.linalg.norm(np.asarray(c['camera_center']) - np.asarray(anchor_candidate['camera_center'])))
            rot_scale = max(float(cfg['anchor_rotation_deg_per_frame']) * gap, 1.0)
            center_scale = max(float(cfg['anchor_center_mm_per_frame']) * gap, 1.0)
            sharp_pen = 1.0 - min(1.0, float(sharpness_by_frame.get(int(frame), 0.0)) / max(max_sharp, 1e-9))
            confidence_pen = 1.0 - float(c.get('measurement_confidence', 0.75))
            local_extra = (
                float(cfg['anchor_rotation_weight']) * rot / rot_scale
                + float(cfg['anchor_center_weight']) * center / center_scale
                + float(cfg['sharpness_weight']) * sharp_pen
                + float(cfg['measurement_weight']) * confidence_pen
                + float(cfg['reprojection_weight']) * float(c.get('reprojection_rms_px', 0.0))
                + float(cfg['frame_offset_tie_weight']) * abs(int(frame) - int(anchor_frame)))
            c['_base_emission_cost'] = float(c.get('emission_cost', 0.0))
            c['_local_extra_cost'] = float(local_extra)
            c['emission_cost'] = float(c.get('emission_cost', 0.0)) + float(local_extra)
            copies.append(c)
        adjusted.append((int(frame), copies))
    transition = lambda b, p, c: _local_window_transition_cost(b, p, c, klt_cache, cfg)
    best, min_marginals = _temporal_segment_min_marginals(adjusted, transition_fn=transition)
    selected = {int(frame): candidate for frame, candidate in best.get('path', [])}
    sequence = []
    history = []
    for frame, candidate in best.get('path', []):
        transition_cost, details = _temporal_transition_cost(history, frame, candidate)
        values = min_marginals.get(int(frame), [])
        finite = sorted(v for v in values if np.isfinite(v))
        margin = None if len(finite) < 2 else float(finite[1] - finite[0])
        sequence.append({
            'frame': int(frame), 'label': candidate.get('label'),
            'parent_seed': candidate.get('parent_seed'), 'basin': candidate.get('basin'),
            'measurement_mode': candidate.get('measurement_mode'),
            'local_emission_cost': float(candidate.get('emission_cost', 0.0)),
            'individual_margin': margin, 'transition': details,
        })
        history.append((frame, candidate))
    return selected, {
        'status': best.get('status'), 'cost': None if not np.isfinite(best.get('cost', np.inf)) else float(best['cost']),
        'branch_sequence': sequence,
    }


def _local_fit_pose_prior(selected_path, target_frame):
    """Fit bounded constant-velocity Q_WC/C_W priors and evaluate at target."""
    items = sorted((int(f), c) for f, c in selected_path.items())
    if len(items) < 3 or int(target_frame) not in selected_path:
        return None
    times = np.asarray([f for f, _ in items], np.float64)
    tc = times - float(target_frame)
    A = np.column_stack([np.ones_like(tc), tc])
    centers = np.asarray([np.asarray(c['camera_center'], np.float64).reshape(3) for _, c in items])
    coeff_c, *_ = np.linalg.lstsq(A, centers, rcond=None)
    C_pred = coeff_c[0]
    Q0 = np.asarray(selected_path[int(target_frame)]['R'], np.float64).T
    rotvecs = []
    for _frame, cand in items:
        Q = np.asarray(cand['R'], np.float64).T
        rv, _ = cv2.Rodrigues(Q @ Q0.T)
        rotvecs.append(rv.reshape(3))
    coeff_r, *_ = np.linalg.lstsq(A, np.asarray(rotvecs, np.float64), rcond=None)
    delta_Q, _ = cv2.Rodrigues(coeff_r[0].reshape(3, 1))
    Q_pred = delta_Q @ Q0
    return {'Q_WC': Q_pred, 'C_W': C_pred}


def _local_try_pose_denoise(candidate, target_frame, selected_path, corners_dict,
                            marker_map, camera_matrix, distortion, marker_size_mm,
                            config):
    cfg = _local_window_resolve_config(config)
    result = {'attempted': False, 'accepted': False, 'reason': 'DISABLED'}
    if not cfg.get('pose_denoise', True):
        return candidate, result
    if len(selected_path) < int(cfg['pose_fit_min_frames']):
        result['reason'] = 'INSUFFICIENT_PATH'
        return candidate, result
    prior = _local_fit_pose_prior(selected_path, target_frame)
    if prior is None:
        result['reason'] = 'FIT_FAILED'
        return candidate, result
    result['attempted'] = True
    R_raw = np.asarray(candidate['R'], np.float64).reshape(3, 3)
    C_raw = np.asarray(candidate['camera_center'], np.float64).reshape(3)
    R_pred = np.asarray(prior['Q_WC'], np.float64).T
    C_pred = np.asarray(prior['C_W'], np.float64).reshape(3)
    rot_delta = _temporal_rotation_distance_deg(R_raw, R_pred)
    center_delta = float(np.linalg.norm(C_raw - C_pred))
    result['raw_vs_prior_rotation_deg'] = float(rot_delta)
    result['raw_vs_prior_center_mm'] = float(center_delta)
    if rot_delta > float(cfg['max_pose_denoise_rotation_deg']) or center_delta > float(cfg['max_pose_denoise_center_mm']):
        result['reason'] = 'PRIOR_TOO_FAR'
        return candidate, result
    alpha = float(cfg['pose_blend_alpha'])
    R_new = _local_so3_blend(R_raw, R_pred, alpha)
    C_new = (1.0 - alpha) * C_raw + alpha * C_pred
    t_new = (-R_new @ C_new.reshape(3, 1)).reshape(3, 1)
    raw_stats = _unified_endpoint_reprojection_stats(
        R_raw, candidate['t'], corners_dict, marker_map, camera_matrix, distortion, marker_size_mm)
    new_stats = _unified_endpoint_reprojection_stats(
        R_new, t_new, corners_dict, marker_map, camera_matrix, distortion, marker_size_mm)
    if raw_stats is None or new_stats is None:
        result['reason'] = 'REPROJECTION_UNAVAILABLE'
        return candidate, result
    result['raw_rms_px'] = float(raw_stats['rms_px'])
    result['filtered_rms_px'] = float(new_stats['rms_px'])
    result['filtered_max_px'] = float(new_stats['max_px'])
    if new_stats['rms_px'] > TEMPORAL_MARKER_GROUP_RMS_MAX_PX or new_stats['max_px'] > TEMPORAL_MARKER_GROUP_MAX_PX:
        result['reason'] = 'ABSOLUTE_MARKER_GATE'
        return candidate, result
    if new_stats['rms_px'] > raw_stats['rms_px'] + float(cfg['reprojection_worsen_margin_px']):
        result['reason'] = 'REPROJECTION_WORSENED'
        return candidate, result
    out = dict(candidate)
    out.update({
        'R': R_new, 't': t_new, 'camera_center': C_new,
        'reprojection_rms_px': float(new_stats['rms_px']),
        'source': str(candidate.get('source', '')) + '+local_prior',
        'local_pose_prior_applied': True,
    })
    result['accepted'] = True
    result['reason'] = 'ACCEPTED'
    return out, result


def _unified_feature_parallax_stats(points_B, points_A, R_A_from_B, K, mask=None):
    points_B = np.asarray(points_B, np.float64).reshape(-1, 2)
    points_A = np.asarray(points_A, np.float64).reshape(-1, 2)
    if len(points_B) == 0 or len(points_B) != len(points_A):
        return {'median_deg': 0.0, 'p10_deg': 0.0, 'count': 0}
    if mask is not None and len(mask) == len(points_B):
        selected = np.asarray(mask, bool).reshape(-1)
        points_B = points_B[selected]
        points_A = points_A[selected]
    if len(points_B) == 0:
        return {'median_deg': 0.0, 'p10_deg': 0.0, 'count': 0}
    K_inv = np.linalg.inv(np.asarray(K, np.float64))
    rays_B = (K_inv @ np.column_stack([points_B, np.ones(len(points_B))]).T).T
    rays_A = (K_inv @ np.column_stack([points_A, np.ones(len(points_A))]).T).T
    rays_B /= np.maximum(np.linalg.norm(rays_B, axis=1, keepdims=True), 1e-12)
    rays_A /= np.maximum(np.linalg.norm(rays_A, axis=1, keepdims=True), 1e-12)
    rays_A_in_B = (np.asarray(R_A_from_B, np.float64).reshape(3, 3).T @ rays_A.T).T
    dots = np.sum(rays_B * rays_A_in_B, axis=1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return {
        'median_deg': float(np.median(angles)),
        'p10_deg': float(np.percentile(angles, 10)),
        'count': int(len(angles)),
    }


def _unified_map_marker_confidence(marker_id, marker_map_diagnostics, marker_size_mm):
    if not marker_map_diagnostics:
        return 0.75
    diag = marker_map_diagnostics.get(int(marker_id))
    if not diag:
        return 0.75
    if diag.get('status') == 'REFERENCE_IDENTITY':
        return 1.0
    t95 = diag.get('translation_residual_p95_mm')
    r95 = diag.get('rotation_residual_p95_deg')
    if t95 is None or r95 is None:
        return 0.60
    radius = float(marker_size_mm) / math.sqrt(2.0)
    u_mm = float(t95) + radius * math.radians(float(r95))
    ratio = u_mm / max(float(marker_size_mm), 1e-9)
    return float(np.clip(1.0 / (1.0 + ratio * ratio), 0.25, 1.0))


def _unified_signed_sampson(points_B, points_A, R_A_from_B, t_A_from_B, K):
    points_B = np.asarray(points_B, np.float64).reshape(-1, 2)
    points_A = np.asarray(points_A, np.float64).reshape(-1, 2)
    t = np.asarray(t_A_from_B, np.float64).reshape(3)
    if len(points_B) == 0 or len(points_B) != len(points_A) or np.linalg.norm(t) < 1e-9:
        return np.full(len(points_B), 1e3, dtype=np.float64)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    K_inv = np.linalg.inv(np.asarray(K, np.float64).reshape(3, 3))
    F = K_inv.T @ (tx @ np.asarray(R_A_from_B, np.float64).reshape(3, 3)) @ K_inv
    hB = np.column_stack([points_B, np.ones(len(points_B))])
    hA = np.column_stack([points_A, np.ones(len(points_A))])
    lA = hB @ F.T
    lB = hA @ F
    numerator = np.sum(lA * hA, axis=1)
    denominator = np.sqrt(0.5 * (
        lA[:, 0] ** 2 + lA[:, 1] ** 2 + lB[:, 0] ** 2 + lB[:, 1] ** 2))
    return numerator / np.maximum(denominator, 1e-12)


def _unified_optimize_endpoint_world_poses(
        R_A0, t_A0, R_B0, t_B0, corners_A, corners_B, marker_map,
        camera_matrix, distortion, feature_K, marker_size_mm,
        feature_points_B=None, feature_points_A=None, feature_mask=None,
        marker_map_diagnostics=None, marker_group_weight=UNIFIED_MARKER_GROUP_WEIGHT,
        feature_group_weight=FEATURE_JOINT_GROUP_WEIGHT, max_nfev=JOINT_RT_MAX_NFEV):
    """Jointly optimize T_A<-W and T_B<-W using each endpoint's mapped markers.

    Feature points are B(left/end) -> A(right/start), matching the production
    relative-pose convention X_A = R_A<-B X_B + t_A<-B.
    """
    R_A0 = np.asarray(R_A0, np.float64).reshape(3, 3)
    R_B0 = np.asarray(R_B0, np.float64).reshape(3, 3)
    t_A0 = np.asarray(t_A0, np.float64).reshape(3, 1)
    t_B0 = np.asarray(t_B0, np.float64).reshape(3, 1)
    mapped_A = sorted(set(int(x) for x in corners_A) & set(marker_map))
    mapped_B = sorted(set(int(x) for x in corners_B) & set(marker_map))
    if not mapped_A or not mapped_B:
        return None

    feature_B = None if feature_points_B is None else np.asarray(feature_points_B, np.float64).reshape(-1, 2)
    feature_A = None if feature_points_A is None else np.asarray(feature_points_A, np.float64).reshape(-1, 2)
    if feature_B is not None and feature_A is not None and len(feature_B) == len(feature_A):
        if feature_mask is not None and len(feature_mask) == len(feature_B):
            m = np.asarray(feature_mask, bool).reshape(-1)
            feature_B = feature_B[m]
            feature_A = feature_A[m]
    else:
        feature_B = feature_A = None

    rv_A0 = cv2.Rodrigues(R_A0)[0].reshape(3)
    rv_B0 = cv2.Rodrigues(R_B0)[0].reshape(3)
    x0 = np.concatenate([rv_A0, t_A0.reshape(3), rv_B0, t_B0.reshape(3)])
    marker_coordinate_count = max(8 * (len(mapped_A) + len(mapped_B)), 1)
    marker_scale = math.sqrt(float(marker_group_weight) / marker_coordinate_count)
    use_features = feature_B is not None and len(feature_B) >= 5
    feature_scale = math.sqrt(float(feature_group_weight) / max(len(feature_B) if use_features else 1, 1))

    def marker_residual_for_pose(R_CW, t_CW, corners, ids):
        rv = cv2.Rodrigues(R_CW)[0]
        pieces = []
        for marker_id in ids:
            obj = _temporal_marker_reference_points(marker_id, marker_map, marker_size_mm)
            projected, _ = cv2.projectPoints(
                obj, rv, t_CW, np.asarray(camera_matrix, np.float64), distortion)
            observed = np.asarray(corners[marker_id], np.float64).reshape(4, 2)
            confidence = _unified_map_marker_confidence(
                marker_id, marker_map_diagnostics, marker_size_mm)
            pieces.append((projected.reshape(4, 2) - observed).reshape(-1) * math.sqrt(confidence))
        return np.concatenate(pieces) if pieces else np.empty(0, np.float64)

    def unpack(parameters):
        R_A = cv2.Rodrigues(parameters[0:3].reshape(3, 1))[0]
        t_A = parameters[3:6].reshape(3, 1)
        R_B = cv2.Rodrigues(parameters[6:9].reshape(3, 1))[0]
        t_B = parameters[9:12].reshape(3, 1)
        return R_A, t_A, R_B, t_B

    def residual(parameters, include_feature):
        R_A, t_A, R_B, t_B = unpack(parameters)
        parts = [
            marker_residual_for_pose(R_A, t_A, corners_A, mapped_A) * marker_scale,
            marker_residual_for_pose(R_B, t_B, corners_B, mapped_B) * marker_scale,
        ]
        R_AB = R_A @ R_B.T
        t_AB = t_A - R_AB @ t_B
        if include_feature and use_features:
            raw = _unified_signed_sampson(feature_B, feature_A, R_AB, t_AB, feature_K)
            scale = FEATURE_FINAL_INLIER_PX
            pseudo_huber = 2.0 * scale * scale * (
                np.sqrt(1.0 + (raw / scale) ** 2) - 1.0)
            parts.append(np.sign(raw) * np.sqrt(np.maximum(pseudo_huber, 0.0)) * feature_scale)
        baseline = float(np.linalg.norm(t_AB))
        parts.append(np.array([
            max(0.0, MIN_BASELINE_MM - baseline) * 0.1,
            max(0.0, baseline - MAX_BASELINE_MM) * 0.1,
        ], np.float64))
        return np.concatenate(parts)

    try:
        marker_only = least_squares(
            lambda x: residual(x, False), x0, method='trf', loss='huber', f_scale=1.0,
            max_nfev=max_nfev, ftol=JOINT_RT_TOL, xtol=JOINT_RT_TOL, gtol=JOINT_RT_TOL)
        candidates = [('marker_only_world', marker_only.x, False)]
        if use_features:
            joint = least_squares(
                lambda x: residual(x, True), marker_only.x, method='trf', loss='linear',
                max_nfev=max_nfev, ftol=JOINT_RT_TOL, xtol=JOINT_RT_TOL, gtol=JOINT_RT_TOL)
            candidates.append(('joint_world_marker_sift', joint.x, True))
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return None

    evaluated = []
    for role, params, uses_feature in candidates:
        R_A, t_A, R_B, t_B = unpack(params)
        R_AB = R_A @ R_B.T
        t_AB = t_A - R_AB @ t_B
        baseline = float(np.linalg.norm(t_AB))
        if not (MIN_BASELINE_MM <= baseline <= MAX_BASELINE_MM):
            continue
        marker_stats = _unified_pair_reprojection_stats(
            R_A, t_A, corners_A, R_B, t_B, corners_B, marker_map,
            camera_matrix, distortion, marker_size_mm)
        if marker_stats is None:
            continue
        feature_stats = None
        if feature_B is not None:
            values = np.abs(_unified_signed_sampson(
                feature_B, feature_A, R_AB, t_AB, feature_K))
            feature_stats = {
                'inlier_count': int(np.count_nonzero(values <= FEATURE_FINAL_INLIER_PX)),
                'inlier_median_px': float(np.median(values[values <= FEATURE_FINAL_INLIER_PX]))
                    if np.any(values <= FEATURE_FINAL_INLIER_PX) else float('inf'),
                'inlier_p90_px': float(np.percentile(values[values <= FEATURE_FINAL_INLIER_PX], 90))
                    if np.any(values <= FEATURE_FINAL_INLIER_PX) else float('inf'),
                'all_median_px': float(np.median(values)) if len(values) else float('inf'),
                'all_p90_px': float(np.percentile(values, 90)) if len(values) else float('inf'),
            }
        evaluated.append({
            'role': role, 'uses_feature': uses_feature,
            'R_A': R_A, 't_A': t_A, 'R_B': R_B, 't_B': t_B,
            'R_rel': R_AB, 't_rel': t_AB, 'baseline': baseline,
            'marker': marker_stats, 'feature': feature_stats,
        })
    if not evaluated:
        return None

    marker_floor = min(item['marker']['rms_px'] for item in evaluated)
    marker_valid = [
        item for item in evaluated
        if item['marker']['rms_px'] <= min(UNIFIED_MARKER_DIRECT_RMS_MAX_PX, marker_floor + 0.75)
        and item['marker']['max_px'] <= UNIFIED_MARKER_DIRECT_MAX_PX]
    pool = marker_valid or evaluated
    feature_valid = [
        item for item in pool
        if item['uses_feature'] and item['feature'] is not None
        and item['feature']['inlier_count'] >= FEATURE_MIN_MATCHES
        and item['feature']['inlier_p90_px'] <= FEATURE_FINAL_P90_MAX_PX]
    if feature_valid:
        chosen = min(feature_valid, key=lambda item: (
            item['feature']['inlier_p90_px'], item['feature']['inlier_median_px'],
            item['marker']['rms_px']))
    else:
        chosen = min(pool, key=lambda item: item['marker']['rms_px'])
    marker_ok = any(chosen is item for item in marker_valid)
    feature_ok = any(chosen is item for item in feature_valid)
    chosen = dict(chosen)
    chosen['marker_ok'] = bool(marker_ok)
    chosen['feature_ok'] = bool(feature_ok)
    chosen['applied_feature'] = bool(chosen['uses_feature'] and feature_ok)
    return chosen

class LazyVideoFrames:
    """List-like, thread-safe video frames decoded only when first requested."""

    def __init__(self, video_path, frame_count):
        self.video_path = video_path
        self._frame_count = max(0, int(frame_count))
        self._cache = {}
        self._capture = None
        self._lock = threading.Lock()

    def __len__(self):
        return self._frame_count

    def __bool__(self):
        return self._frame_count > 0

    def __iter__(self):
        for idx in range(self._frame_count):
            yield self[idx]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(self._frame_count))]

        index = int(index)
        if index < 0:
            index += self._frame_count
        if not 0 <= index < self._frame_count:
            raise IndexError(index)

        cached = self._cache.get(index)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                return cached

            cap = self._get_capture()
            next_index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if next_index != index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise IndexError(f"Unable to decode frame {index} from {self.video_path}")
            self._cache[index] = frame
            return frame

    def preload(self, indices):
        """Decode requested frames in ascending order to avoid repeated codec seeks."""
        requested = sorted({
            int(index) for index in indices
            if 0 <= int(index) < self._frame_count
            and int(index) not in self._cache
        })
        if not requested:
            return

        with self._lock:
            requested = [index for index in requested if index not in self._cache]
            if not requested:
                return
            cap = self._get_capture()
            position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if position > requested[0] or requested[0] - position > 120:
                cap.set(cv2.CAP_PROP_POS_FRAMES, requested[0])
                position = requested[0]

            for target in requested:
                if position > target:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    position = target
                while position <= target:
                    if not cap.grab():
                        raise IndexError(
                            f"Unable to decode frame {target} from {self.video_path}")
                    if position == target:
                        ok, frame = cap.retrieve()
                        if not ok or frame is None:
                            raise IndexError(
                                f"Unable to retrieve frame {target} from {self.video_path}")
                        self._cache[target] = frame
                    position += 1

    def _get_capture(self):
        if self._capture is None:
            if hasattr(cv2, 'CAP_PROP_N_THREADS'):
                self._capture = cv2.VideoCapture(
                    self.video_path, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_N_THREADS, 8])
            else:
                self._capture = cv2.VideoCapture(self.video_path)
            if not self._capture.isOpened():
                self._capture.release()
                self._capture = None
                raise OSError(f"Unable to open video: {self.video_path}")
        return self._capture

    def close(self):
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def __del__(self):
        capture = getattr(self, '_capture', None)
        if capture is not None:
            capture.release()


def log_and_print(msg):
    print(msg)


def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    return _compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=log_and_print)


def analyze_video_frames(
    video_path,
    start_n,
    end_n,
    K_L,
    dist_L,
    mtx_L,
    marker_size_mm,
    select_mode="average",
    range_mode="fixed",
    progress_callback=None,
    frames_override=None,
    detection_roi_ratio=None,
    feature_roi_ratio=None,
    marker_corners_override=None,
    pattern_guided=True,
    pattern_guided_config=None,
    pair_geometry_roi_ratio=None,
    local_window=True,
    local_window_config=None,
    angle_guided_config=None,
):
    timer = StageTimer("影片分析明細")
    analysis_wall_start = time.perf_counter()
    pg_cfg = _pattern_guided_resolve_config(pattern_guided_config)
    pattern_guided_enabled = bool(pattern_guided and pg_cfg.get('enabled', True))
    lw_cfg = _local_window_resolve_config(local_window_config)
    local_window_enabled = bool(local_window and lw_cfg.get('enabled', True))
    angle_cfg = _angle_guided_resolve_config(angle_guided_config)
    angle_guided_enabled = bool(angle_cfg.get('enabled', False))
    angle_guided_diagnostics = {
        'enabled': angle_guided_enabled,
        'config': dict(angle_cfg),
        'status': 'DISABLED' if not angle_guided_enabled else 'NOT_RUN',
        'fallback_reason': None,
        'reference_marker_id': None,
        'coarse_indices_A': [],
        'coarse_indices_B': [],
        'scanned_indices_A': [],
        'scanned_indices_B': [],
        'selected_indices_A': [],
        'selected_indices_B': [],
        'selected_measurements_A': [],
        'selected_measurements_B': [],
        'direction': None,
        'forward_coarse_error_deg': None,
        'reverse_coarse_error_deg': None,
        'chronological_target_A_deg': None,
        'chronological_target_B_deg': None,
        'output_roles_swapped': False,
        'elapsed_s': 0.0,
    }
    local_window_diagnostics = {
        'enabled': local_window_enabled,
        'config': dict(lw_cfg),
        'status': 'DISABLED' if not local_window_enabled else 'NOT_RUN',
        'provisional_pair': None,
        'endpoints': {},
        'new_frames': [],
        'local_frames_decoded': 0,
        'local_frames_detected': 0,
        'klt_pair_count': 0,
        'sift_pair_count': 0,
        'fallback_reason': None,
    }
    # Resolve derived baseline gating *after* Zebra has injected module globals.
    # Keep this value fixed for the duration of one analysis call so hard gates,
    # logs and diagnostics cannot diverge even if another caller mutates globals.
    runtime_pair_candidate_min_baseline = _effective_pair_candidate_min_baseline_mm()
    pattern_guided_diagnostics = {
        'enabled': pattern_guided_enabled,
        'config': {k: v for k, v in pg_cfg.items() if k != 'weights'},
        'weights': dict(pg_cfg['weights']),
        'adaptive_status': 'DISABLED' if not pattern_guided_enabled else 'NOT_EVALUATED',
        'extra_start_indices': [],
        'extra_end_indices': [],
        'total_aruco_probe_count': 5,
        'deadline_skipped': False,
        'baseline_gate_min_mm': float(runtime_pair_candidate_min_baseline),
        'baseline_gate_margin_mm': float(PAIR_CANDIDATE_MIN_BASELINE_MARGIN_MM),
        'baseline_gate_source': 'runtime_MIN_BASELINE_MM_plus_margin',
    }
    log_and_print(
        f"ℹ️ [Baseline candidate gate] MIN_BASELINE_MM={MIN_BASELINE_MM:.2f} mm, "
        f"margin={PAIR_CANDIDATE_MIN_BASELINE_MARGIN_MM:.2f} mm, "
        f"effective_min={runtime_pair_candidate_min_baseline:.2f} mm, "
        f"max={MAX_BASELINE_MM:.2f} mm")
    if progress_callback:
        progress_callback(2, "階段 1/6：載入影片...")
    if frames_override is None:
        if hasattr(cv2, 'CAP_PROP_N_THREADS'):
            cap = cv2.VideoCapture(
                video_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 8])
        else:
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 無法開啟影片: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log_and_print(f"🎬 載入影片: {video_path}，總影格數: {total_frames} (選幀範圍模式: {range_mode})")

        frames = LazyVideoFrames(video_path, total_frames)
        cap.release()
    else:
        frames = [np.asarray(frame).copy() for frame in frames_override]
        total_frames = len(frames)
        if not frames:
            print("❌ 固定影像對為空")
            return None
        frame_height, frame_width = frames[0].shape[:2]
        if any(frame.shape[:2] != (frame_height, frame_width) for frame in frames):
            print("❌ 固定影像對的影像尺寸不一致")
            return None
        log_and_print(
            f"🎬 載入固定影像對: {video_path}，影像數: {total_frames} "
            f"(選幀範圍模式: {range_mode})")
    if progress_callback:
        progress_callback(12, "階段 1/6：影片索引完成")
    
    if len(frames) == 0:
        print("❌ 影片無有效影格")
        return None
    timer.stage(f"影片索引建立({len(frames)} 幀)")

    mid_idx = len(frames) // 2
    if range_mode == "half_half":
        start_range = range(0, mid_idx)
        end_range = range(mid_idx, len(frames))
    else:
        N = min(start_n, len(frames))
        M = min(end_n, len(frames))
        start_range = range(N)
        end_range = range(len(frames) - M, len(frames))

    def roi_bounds_from_ratio(roi_ratio):
        if roi_ratio is None:
            return None
        if len(roi_ratio) == 2:
            return centered_roi_bounds(
                frame_width,
                frame_height,
                roi_ratio[0],
                roi_ratio[1],
            )
        if len(roi_ratio) == 4:
            return normalized_roi_bounds(
                frame_width,
                frame_height,
                roi_ratio[0],
                roi_ratio[1],
                roi_ratio[2],
                roi_ratio[3],
            )
        raise ValueError(
            "Each detection ROI must contain either width/height or "
            "x/y/width/height ratios")

    if isinstance(detection_roi_ratio, dict):
        start_roi_ratio = detection_roi_ratio.get("frame_A")
        end_roi_ratio = detection_roi_ratio.get("frame_B")
        if start_roi_ratio is None or end_roi_ratio is None:
            raise ValueError(
                "Independent detection ROIs require frame_A and frame_B")
    else:
        start_roi_ratio = detection_roi_ratio
        end_roi_ratio = detection_roi_ratio

    detection_roi_bounds_start = roi_bounds_from_ratio(start_roi_ratio)
    detection_roi_bounds_end = roi_bounds_from_ratio(end_roi_ratio)

    # SIFT ROI is independent from ArUco detection.  For backward
    # compatibility, callers that only supplied detection_roi_ratio retain the
    # historical shared ArUco/SIFT ROI behaviour.  New callers can leave
    # detection_roi_ratio=None and set feature_roi_ratio to restrict SIFT only.
    if isinstance(feature_roi_ratio, dict):
        feature_start_ratio = feature_roi_ratio.get("frame_A")
        feature_end_ratio = feature_roi_ratio.get("frame_B")
        if feature_start_ratio is None or feature_end_ratio is None:
            raise ValueError(
                "Independent SIFT feature ROIs require frame_A and frame_B")
        feature_roi_bounds_start = roi_bounds_from_ratio(feature_start_ratio)
        feature_roi_bounds_end = roi_bounds_from_ratio(feature_end_ratio)
    elif feature_roi_ratio is not None:
        feature_roi_bounds_start = roi_bounds_from_ratio(feature_roi_ratio)
        feature_roi_bounds_end = roi_bounds_from_ratio(feature_roi_ratio)
    else:
        feature_roi_bounds_start = detection_roi_bounds_start
        feature_roi_bounds_end = detection_roi_bounds_end

    # Pair-geometry ROI is independent from detection/SIFT ROI.  If omitted,
    # reuse the detection ROI when available, otherwise fall back to full frame.
    if isinstance(pair_geometry_roi_ratio, dict):
        pg_start_ratio = pair_geometry_roi_ratio.get("frame_A")
        pg_end_ratio = pair_geometry_roi_ratio.get("frame_B")
        if pg_start_ratio is None or pg_end_ratio is None:
            raise ValueError("Independent pair geometry ROIs require frame_A and frame_B")
    else:
        pg_start_ratio = pair_geometry_roi_ratio
        pg_end_ratio = pair_geometry_roi_ratio
    geometry_roi_bounds_start = (
        roi_bounds_from_ratio(pg_start_ratio) if pg_start_ratio is not None
        else detection_roi_bounds_start)
    geometry_roi_bounds_end = (
        roi_bounds_from_ratio(pg_end_ratio) if pg_end_ratio is not None
        else detection_roi_bounds_end)

    def geometry_roi_bounds_for_frame(frame_index):
        if frame_index in end_range:
            return geometry_roi_bounds_end
        return geometry_roi_bounds_start

    def roi_bounds_for_frame(frame_index):
        """ArUco-only detection bounds (legacy name retained internally)."""
        if frame_index in end_range:
            return detection_roi_bounds_end
        return detection_roi_bounds_start

    def feature_roi_bounds_for_frame(frame_index):
        if frame_index in end_range:
            return feature_roi_bounds_end
        return feature_roi_bounds_start

    def scaled_feature_roi_bounds(full_resolution_bounds, scaled_width, scaled_height):
        """Map full-resolution ROI bounds into the resized SIFT image."""
        if full_resolution_bounds is None:
            return 0, 0, int(scaled_width), int(scaled_height)
        scale_x = float(scaled_width) / max(float(frame_width), 1.0)
        scale_y = float(scaled_height) / max(float(frame_height), 1.0)
        x0, y0, x1, y1 = full_resolution_bounds
        sx0 = max(0, min(int(scaled_width), int(round(float(x0) * scale_x))))
        sy0 = max(0, min(int(scaled_height), int(round(float(y0) * scale_y))))
        sx1 = max(0, min(int(scaled_width), int(round(float(x1) * scale_x))))
        sy1 = max(0, min(int(scaled_height), int(round(float(y1) * scale_y))))
        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError(
                "SIFT ROI becomes empty after FEATURE_IMAGE_SCALE resize: "
                f"full={full_resolution_bounds}, scaled={(sx0, sy0, sx1, sy1)}")
        return sx0, sy0, sx1, sy1

    def log_roi_bounds(prefix, label, bounds):
        if bounds is None:
            return
        roi_x0, roi_y0, roi_x1, roi_y1 = bounds
        log_and_print(
            f"🎯 [{prefix}-{label}] "
            f"x={roi_x0}:{roi_x1}, y={roi_y0}:{roi_y1} "
            f"({roi_x1 - roi_x0}x{roi_y1 - roi_y0})")

    if detection_roi_bounds_start == detection_roi_bounds_end:
        log_roi_bounds("RT ArUco ROI", "共用", detection_roi_bounds_start)
    else:
        log_roi_bounds("RT ArUco ROI", "Frame A/右圖", detection_roi_bounds_start)
        log_roi_bounds("RT ArUco ROI", "Frame B/左圖", detection_roi_bounds_end)
    if feature_roi_bounds_start == feature_roi_bounds_end:
        log_roi_bounds("RT SIFT ROI", "共用", feature_roi_bounds_start)
    else:
        log_roi_bounds("RT SIFT ROI", "Frame A/右圖", feature_roi_bounds_start)
        log_roi_bounds("RT SIFT ROI", "Frame B/左圖", feature_roi_bounds_end)
    if feature_roi_bounds_start is not None or feature_roi_bounds_end is not None:
        feature_scale_for_log = float(FEATURE_IMAGE_SCALE)
        if feature_scale_for_log <= 0.0:
            raise ValueError("FEATURE_IMAGE_SCALE must be greater than zero")
        scaled_width_for_log = max(1, int(round(frame_width * feature_scale_for_log)))
        scaled_height_for_log = max(1, int(round(frame_height * feature_scale_for_log)))
        scaled_start_for_log = scaled_feature_roi_bounds(
            feature_roi_bounds_start, scaled_width_for_log, scaled_height_for_log)
        scaled_end_for_log = scaled_feature_roi_bounds(
            feature_roi_bounds_end, scaled_width_for_log, scaled_height_for_log)
        if scaled_start_for_log == scaled_end_for_log:
            log_and_print(
                f"   ↳ [RT SIFT scaled crop] scale={feature_scale_for_log:.3f} | "
                f"image={scaled_width_for_log}x{scaled_height_for_log} | "
                f"crop={scaled_start_for_log}")
        else:
            log_and_print(
                f"   ↳ [RT SIFT scaled crop-A] scale={feature_scale_for_log:.3f} | "
                f"image={scaled_width_for_log}x{scaled_height_for_log} | "
                f"crop={scaled_start_for_log}")
            log_and_print(
                f"   ↳ [RT SIFT scaled crop-B] scale={feature_scale_for_log:.3f} | "
                f"image={scaled_width_for_log}x{scaled_height_for_log} | "
                f"crop={scaled_end_for_log}")
    
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
    else:
        params = cv2.aruco.DetectorParameters_create()
        
    marker_preprocess_state = threading.local()

    def prepare_marker_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ARUCO_USE_CLAHE:
            clahe = getattr(marker_preprocess_state, 'clahe', None)
            if clahe is None:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                marker_preprocess_state.clahe = clahe
            gray = clahe.apply(gray)
        return gray

    def detect_marker_candidates_in_gray(gray, detection_roi_bounds=None):
        detect_gray = gray
        offset_x = 0
        offset_y = 0
        if detection_roi_bounds is not None:
            offset_x, offset_y, roi_x1, roi_y1 = detection_roi_bounds
            detect_gray = gray[offset_y:roi_y1, offset_x:roi_x1]
        if hasattr(cv2.aruco, 'ArucoDetector'):
            local_detector = cv2.aruco.ArucoDetector(
                dict_4x4, cv2.aruco.DetectorParameters())
            corners, ids, _ = local_detector.detectMarkers(detect_gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                detect_gray, dict_4x4, parameters=params)
        return detect_gray, corners, ids, offset_x, offset_y

    def refine_detected_marker_corners(
            detect_gray, corners, ids, offset_x=0, offset_y=0):
        if ids is not None and len(ids) > 0:
            ids_list = [i[0] for i in ids]
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            for c in corners:
                cv2.cornerSubPix(detect_gray, c, (5, 5), (-1, -1), term)
            offset = np.array([offset_x, offset_y], dtype=np.float32)
            raw_corners = [c.reshape(4, 2) + offset for c in corners]
            return dict(zip(ids_list, raw_corners))
        return {}

    def detect_markers_in_gray(gray, detection_roi_bounds=None):
        return refine_detected_marker_corners(
            *detect_marker_candidates_in_gray(gray, detection_roi_bounds))

    def detect_frame_markers(frame_index):
        if marker_corners_override is not None:
            if frame_index < 0 or frame_index >= len(marker_corners_override):
                return {}
            override = marker_corners_override[frame_index] or {}
            return {
                int(marker_id): np.asarray(points, dtype=np.float32).reshape(4, 2).copy()
                for marker_id, points in override.items()
            }
        return detect_markers_in_gray(
            prepare_marker_gray(frames[frame_index]),
            roi_bounds_for_frame(frame_index),
        )

    def ensure_profiled_probe_markers(frame_indices):
        """Populate the marker cache in non-overlapping timed image phases."""
        missing = sorted({
            int(index) for index in frame_indices
            if int(index) not in detected_cache
        })
        if not missing:
            return

        if marker_corners_override is not None:
            phase_start = time.perf_counter()
            detected_cache.update({
                index: detect_frame_markers(index) for index in missing
            })
            add_pair_detail(
                'marker_probe_cache_overhead', time.perf_counter() - phase_start)
            return

        phase_start = time.perf_counter()
        if hasattr(frames, 'preload'):
            frames.preload(missing)
        decoded_frames = {index: frames[index] for index in missing}
        add_pair_detail(
            'marker_probe_video_decode', time.perf_counter() - phase_start)

        def convert_to_gray(index):
            return index, cv2.cvtColor(decoded_frames[index], cv2.COLOR_BGR2GRAY)

        phase_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            gray_by_index = dict(executor.map(convert_to_gray, missing))
        add_pair_detail(
            'marker_probe_gray_convert', time.perf_counter() - phase_start)

        if ARUCO_USE_CLAHE:
            def apply_clahe(index):
                clahe = getattr(marker_preprocess_state, 'clahe', None)
                if clahe is None:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    marker_preprocess_state.clahe = clahe
                return index, clahe.apply(gray_by_index[index])

            phase_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
                prepared_by_index = dict(executor.map(apply_clahe, missing))
            add_pair_detail(
                'marker_probe_clahe', time.perf_counter() - phase_start)
        else:
            prepared_by_index = gray_by_index

        def detect_candidates(index):
            return index, detect_marker_candidates_in_gray(
                prepared_by_index[index], roi_bounds_for_frame(index))

        phase_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            raw_by_index = dict(executor.map(detect_candidates, missing))
        add_pair_detail(
            'marker_probe_aruco_detect', time.perf_counter() - phase_start)

        def refine_corners(index):
            return index, refine_detected_marker_corners(*raw_by_index[index])

        phase_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            refined_by_index = dict(executor.map(refine_corners, missing))
        add_pair_detail(
            'marker_probe_corner_subpix', time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        detected_cache.update(refined_by_index)
        add_pair_detail(
            'marker_probe_cache_overhead', time.perf_counter() - phase_start)

    # 定義輔助工具
    def undistort_corners_dict(corners_dict):
        undist = {}
        for mid, pts in corners_dict.items():
            pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
            pts_undist = cv2.undistortPoints(pts_reshaped, mtx_L, dist_L, P=K_L)
            undist[mid] = pts_undist.reshape(4, 2)
        return undist

    detected_cache = {}
    local_raw_gray_cache = {}
    local_klt_cache = {}
    local_detection_mode = {}
    local_unique_new_frames = set()

    def get_local_raw_gray(frame_index):
        frame_index = int(frame_index)
        if frame_index not in local_raw_gray_cache:
            local_raw_gray_cache[frame_index] = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY)
        return local_raw_gray_cache[frame_index]

    def local_predicted_roi(center_corners):
        if not center_corners:
            return None
        pts = np.vstack([np.asarray(v, np.float32).reshape(-1, 2) for v in center_corners.values()])
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        w = max(x1 - x0, 16.0) * float(lw_cfg['aruco_roi_expand'])
        h = max(y1 - y0, 16.0) * float(lw_cfg['aruco_roi_expand'])
        return (max(0, int(cx - 0.5 * w)), max(0, int(cy - 0.5 * h)),
                min(frame_width, int(cx + 0.5 * w)), min(frame_height, int(cy + 0.5 * h)))

    def detect_local_frame_markers(frame_index, center_corners, stage_deadline):
        frame_index = int(frame_index)
        if frame_index in detected_cache:
            return detected_cache[frame_index]
        if marker_corners_override is not None:
            result = detect_frame_markers(frame_index)
            detected_cache[frame_index] = result
            local_detection_mode[frame_index] = 'override'
            return result
        gray = prepare_marker_gray(frames[frame_index])
        predicted_roi = local_predicted_roi(center_corners)
        result = detect_markers_in_gray(gray, predicted_roi) if predicted_roi is not None else {}
        mode = 'predicted_roi' if result else 'predicted_roi_miss'
        if (not result and bool(lw_cfg['allow_full_detection_fallback'])
                and time.perf_counter() <= stage_deadline):
            result = detect_markers_in_gray(gray, roi_bounds_for_frame(frame_index))
            mode = 'full_fallback' if result else 'full_fallback_miss'
        detected_cache[frame_index] = result
        local_detection_mode[frame_index] = mode
        return result

    def get_frame_info(idxs, stage_idx=0, is_start_segment=True):
        info = []
        seg_name = "開頭段" if is_start_segment else "結尾段"
        missing = [idx for idx in idxs if idx not in detected_cache]
        if missing:
            def detect_index(index):
                return detect_frame_markers(index)

            with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
                detected_cache.update(zip(
                    missing, executor.map(detect_index, missing)))
        for i, idx in enumerate(idxs):
            cd = detected_cache[idx]
            if cd:
                info.append({'idx': idx, 'corners': cd})
            if progress_callback:
                stage_base = 15 + stage_idx * 15
                if is_start_segment:
                    percent = stage_base + (i / len(idxs)) * 7.5
                else:
                    percent = stage_base + 7.5 + (i / len(idxs)) * 7.5
                progress_callback(min(percent, 98.0), f"階段 2/6：分析影像 ({i + 1}/{len(idxs)})...")
        return info

    def select_adaptive_candidate_frames(first_range, second_range):
        """Use the original five probes or angle-targeted ArUco proposals."""
        def select_fractions(frame_range, fractions):
            values = list(frame_range)
            if not values:
                return []
            selected = [
                values[int(round((len(values) - 1) * fraction))]
                for fraction in fractions
            ]
            return list(dict.fromkeys(selected))

        def original_selection():
            return (
                select_fractions(first_range, ADAPTIVE_START_RANGE_FRACTIONS),
                select_fractions(second_range, ADAPTIVE_END_RANGE_FRACTIONS),
            )

        if not angle_guided_enabled:
            return original_selection()

        scan_start = time.perf_counter()

        def uniform_indices(frame_range, count):
            values = list(frame_range)
            if not values:
                return []
            positions = np.linspace(
                0, len(values) - 1, min(int(count), len(values)))
            return list(dict.fromkeys(
                values[int(round(position))] for position in positions))

        def ensure_detected(indices):
            missing = [int(index) for index in indices
                       if int(index) not in detected_cache]
            if not missing:
                return
            detection_start = time.perf_counter()
            if hasattr(frames, 'preload'):
                frames.preload(missing)
            with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
                detected_cache.update(zip(
                    missing, executor.map(detect_frame_markers, missing)))
            angle_guided_diagnostics['frame_decode_aruco_s'] = (
                float(angle_guided_diagnostics.get('frame_decode_aruco_s', 0.0))
                + time.perf_counter() - detection_start)

        def choose_common_marker(indices_A, indices_B):
            counts_A = {}
            counts_B = {}
            areas = {}
            for indices, counts in (
                    (indices_A, counts_A), (indices_B, counts_B)):
                for index in indices:
                    for marker_id, points in detected_cache.get(int(index), {}).items():
                        marker_id = int(marker_id)
                        counts[marker_id] = counts.get(marker_id, 0) + 1
                        area = abs(float(cv2.contourArea(
                            np.asarray(points, np.float32).reshape(4, 2))))
                        areas.setdefault(marker_id, []).append(area)
            common = set(counts_A) & set(counts_B)
            if not common:
                return None
            return max(common, key=lambda marker_id: (
                min(counts_A.get(marker_id, 0), counts_B.get(marker_id, 0)),
                counts_A.get(marker_id, 0) + counts_B.get(marker_id, 0),
                float(np.median(areas.get(marker_id, [0.0]))),
                -int(marker_id)))

        def measurements(indices, marker_id):
            output = []
            for index in indices:
                measurement = _angle_guided_marker_measurement(
                    detected_cache.get(int(index), {}), marker_id,
                    mtx_L, dist_L, marker_size_mm)
                if measurement is not None:
                    measurement = dict(measurement)
                    measurement['idx'] = int(index)
                    output.append(measurement)
            return output

        def refined_indices(frame_range, coarse, coarse_measurements, target):
            values = list(frame_range)
            if not values or not coarse_measurements:
                return list(coarse)
            allowed = set(values)
            maximum = min(int(angle_cfg['max_scan_frames_per_segment']), len(values))
            selected = set(int(index) for index in coarse)
            seeds = sorted(coarse_measurements, key=lambda entry: (
                abs(float(entry['incidence_deg']) - float(target)),
                float(entry.get('reprojection_rms_px', float('inf')))))[:2]
            max_offset = max(2, len(values))
            for offset in range(1, max_offset + 1):
                for seed in seeds:
                    center = int(seed['idx'])
                    for candidate in (center - offset, center + offset):
                        if candidate in allowed:
                            selected.add(candidate)
                        if len(selected) >= maximum:
                            return sorted(selected)
            return sorted(selected)

        def fallback(reason):
            angle_guided_diagnostics['status'] = 'FALLBACK_ORIGINAL'
            angle_guided_diagnostics['fallback_reason'] = str(reason)
            angle_guided_diagnostics['elapsed_s'] = float(
                time.perf_counter() - scan_start)
            log_and_print(
                f"⚠️ [角度導向選幀] {reason}，退回原本 5-frame 取樣 "
                f"({angle_guided_diagnostics['elapsed_s']:.3f}s)")
            return original_selection()

        if progress_callback:
            progress_callback(13, "階段 2/6：掃描 Pattern 角度...")
        coarse_A = uniform_indices(
            first_range, angle_cfg['coarse_samples_per_segment'])
        coarse_B = uniform_indices(
            second_range, angle_cfg['coarse_samples_per_segment'])
        angle_guided_diagnostics['coarse_indices_A'] = [int(x) for x in coarse_A]
        angle_guided_diagnostics['coarse_indices_B'] = [int(x) for x in coarse_B]
        ensure_detected([*coarse_A, *coarse_B])
        reference_marker_id = choose_common_marker(coarse_A, coarse_B)
        if reference_marker_id is None:
            return fallback('NO_COMMON_MARKER_IN_COARSE_SCAN')
        angle_guided_diagnostics['reference_marker_id'] = int(reference_marker_id)

        coarse_measurements_A = measurements(coarse_A, reference_marker_id)
        coarse_measurements_B = measurements(coarse_B, reference_marker_id)
        if not coarse_measurements_A or not coarse_measurements_B:
            return fallback('INSUFFICIENT_COARSE_POSES')

        target_right = float(angle_cfg['target_frame_A_deg'])
        target_left = float(angle_cfg['target_frame_B_deg'])

        def nearest_error(entries, target):
            return min(
                abs(float(entry['incidence_deg']) - float(target))
                for entry in entries)

        forward_error = (
            nearest_error(coarse_measurements_A, target_right)
            + nearest_error(coarse_measurements_B, target_left))
        reverse_error = (
            nearest_error(coarse_measurements_A, target_left)
            + nearest_error(coarse_measurements_B, target_right))
        direction_mode = str(angle_cfg.get('direction_mode', 'auto'))
        if direction_mode == 'forward':
            direction = 'forward'
        elif direction_mode == 'reverse':
            direction = 'reverse'
        else:
            direction = 'forward' if forward_error <= reverse_error else 'reverse'
        if direction == 'forward':
            chronological_target_A = target_right
            chronological_target_B = target_left
        else:
            chronological_target_A = target_left
            chronological_target_B = target_right
        angle_guided_diagnostics['direction'] = direction
        angle_guided_diagnostics['forward_coarse_error_deg'] = float(forward_error)
        angle_guided_diagnostics['reverse_coarse_error_deg'] = float(reverse_error)
        angle_guided_diagnostics['chronological_target_A_deg'] = float(
            chronological_target_A)
        angle_guided_diagnostics['chronological_target_B_deg'] = float(
            chronological_target_B)
        angle_guided_diagnostics['output_roles_swapped'] = bool(
            direction == 'reverse'
            and angle_cfg.get('normalize_output_roles', True))
        log_and_print(
            f"🧭 [角度掃描方向] mode={direction_mode} -> {direction} | "
            f"forward_error={forward_error:.2f}deg | "
            f"reverse_error={reverse_error:.2f}deg")
        scan_A = refined_indices(
            first_range, coarse_A, coarse_measurements_A,
            chronological_target_A)
        scan_B = refined_indices(
            second_range, coarse_B, coarse_measurements_B,
            chronological_target_B)
        ensure_detected([*scan_A, *scan_B])
        final_measurements_A = measurements(scan_A, reference_marker_id)
        final_measurements_B = measurements(scan_B, reference_marker_id)
        selected_A = _angle_guided_rank_measurements(
            final_measurements_A, chronological_target_A, angle_cfg)
        selected_B = _angle_guided_rank_measurements(
            final_measurements_B, chronological_target_B, angle_cfg)
        angle_guided_diagnostics['scanned_indices_A'] = [int(x) for x in scan_A]
        angle_guided_diagnostics['scanned_indices_B'] = [int(x) for x in scan_B]
        if not selected_A or not selected_B:
            return fallback('NO_VALID_TARGET_POSE')
        error_A = abs(float(selected_A[0]['incidence_deg'])
                      - chronological_target_A)
        error_B = abs(float(selected_B[0]['incidence_deg'])
                      - chronological_target_B)
        if max(error_A, error_B) > float(angle_cfg['target_tolerance_deg']):
            return fallback(
                f"TARGET_OUT_OF_TOLERANCE(A={error_A:.2f}deg,B={error_B:.2f}deg)")

        selected_indices_A = sorted(int(entry['idx']) for entry in selected_A)
        selected_indices_B = sorted(int(entry['idx']) for entry in selected_B)
        angle_guided_diagnostics['status'] = 'OK_ANGLE_GUIDED'
        angle_guided_diagnostics['selected_indices_A'] = selected_indices_A
        angle_guided_diagnostics['selected_indices_B'] = selected_indices_B
        angle_guided_diagnostics['selected_measurements_A'] = selected_A
        angle_guided_diagnostics['selected_measurements_B'] = selected_B
        angle_guided_diagnostics['elapsed_s'] = float(
            time.perf_counter() - scan_start)
        log_and_print(
            f"🎯 [角度導向選幀] marker ID={reference_marker_id} | "
            f"前段 F{selected_A[0]['idx']}="
            f"{selected_A[0]['incidence_deg']:.2f}deg "
            f"(target {chronological_target_A:.1f}) | "
            f"後段 F{selected_B[0]['idx']}="
            f"{selected_B[0]['incidence_deg']:.2f}deg "
            f"(target {chronological_target_B:.1f}) | "
            f"scan {angle_guided_diagnostics['elapsed_s']:.3f}s")
        return selected_indices_A, selected_indices_B

    def save_debug_pair_images(item_s, item_e, suffix):
        img_A = frames[item_s['idx']].copy()
        img_B = frames[item_e['idx']].copy()
        corners_s = item_s['corners']
        corners_e = item_e['corners']
        R_s, t_s = item_s['R'], item_s['t']
        R_e, t_e = item_e['R'], item_e['t']
        half = marker_size_mm / 2.0
        canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        for mid, pts in corners_s.items():
            pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_A, [pts_int], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.putText(img_A, f"Obs:{mid}", (pts_int[0][0][0], pts_int[0][0][1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            if mid in marker_map:
                R_m2ref, t_m2ref = marker_map[mid]
                P_w = (R_m2ref @ canon.T).T + t_m2ref.T
                rvec_s, _ = cv2.Rodrigues(R_s)
                pts_s_proj, _ = cv2.projectPoints(P_w.astype(np.float32), rvec_s, t_s, mtx_L, dist_L)
                pts_s_proj = pts_s_proj.reshape(4, 2).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_A, [pts_s_proj], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(img_A, f"Proj:{mid}", (pts_s_proj[0][0][0], pts_s_proj[0][0][1] + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                
        for mid, pts in corners_e.items():
            pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_B, [pts_int], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.putText(img_B, f"Obs:{mid}", (pts_int[0][0][0], pts_int[0][0][1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            if mid in marker_map:
                R_m2ref, t_m2ref = marker_map[mid]
                P_w = (R_m2ref @ canon.T).T + t_m2ref.T
                rvec_e, _ = cv2.Rodrigues(R_e)
                pts_e_proj, _ = cv2.projectPoints(P_w.astype(np.float32), rvec_e, t_e, mtx_L, dist_L)
                pts_e_proj = pts_e_proj.reshape(4, 2).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_B, [pts_e_proj], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(img_B, f"Proj:{mid}", (pts_e_proj[0][0][0], pts_e_proj[0][0][1] + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                
        save_dir = os.path.join(RECORD_SAVE_DIR, "debug_pairs")
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"frame_A_{suffix}.png"), img_A)
        cv2.imwrite(os.path.join(save_dir, f"frame_B_{suffix}.png"), img_B)
        log_and_print(f"✅ 儲存偵錯對圖片至: {save_dir}/frame_A_{suffix}.png 與 frame_B_{suffix}.png")

    def compute_pair_reprojection_error(
            item_s, item_e, mtx_L, dist_L, R_s=None, t_s=None, R_e=None, t_e=None,
            return_stats=False):
        # Branch-specific and endpoint-specific. The two endpoints do not need
        # to share marker IDs; each is checked directly against the fixed map.
        R_s = item_s['R'] if R_s is None else R_s
        t_s = item_s['t'] if t_s is None else t_s
        R_e = item_e['R'] if R_e is None else R_e
        t_e = item_e['t'] if t_e is None else t_e
        stats = _unified_pair_reprojection_stats(
            R_s, t_s, item_s['corners'], R_e, t_e, item_e['corners'],
            marker_map, mtx_L, dist_L, marker_size_mm)
        if stats is None:
            return (float('inf'), None) if return_stats else float('inf')
        value = float(stats['mean_px'])
        return (value, stats) if return_stats else value

    sharpness_cache = {}

    def get_frame_sharpness(idx):
        if idx not in sharpness_cache:
            gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            sharpness_cache[idx] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharpness_cache[idx]

    def marker_coverage_ratio(corners_dict):
        if not corners_dict:
            return 0.0
        pts = np.vstack([np.asarray(v, dtype=np.float32).reshape(-1, 2) for v in corners_dict.values()])
        if len(pts) < 3:
            return 0.0
        hull = cv2.convexHull(pts.astype(np.float32))
        area = float(cv2.contourArea(hull))
        h, w = frame_height, frame_width
        return max(0.0, min(1.0, area / float(w * h)))

    # ---- 特徵極線驗證與混合 RT 精修 ----
    feat_cache = {}

    def feature_point_fullres(keypoint):
        scale = max(float(FEATURE_IMAGE_SCALE), 1e-6)
        return np.asarray(keypoint.pt, dtype=np.float32) / scale

    def get_frame_features(idx):
        if idx not in feat_cache:
            gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            scale = float(FEATURE_IMAGE_SCALE)
            if scale <= 0.0:
                raise ValueError("FEATURE_IMAGE_SCALE must be greater than zero")
            if scale != 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if idx not in detected_cache:
                detected_cache[idx] = detect_frame_markers(idx)
            frame_feature_bounds = feature_roi_bounds_for_frame(idx)
            scaled_x0, scaled_y0, scaled_x1, scaled_y1 = scaled_feature_roi_bounds(
                frame_feature_bounds, gray.shape[1], gray.shape[0])
            # Crop after resizing so the ROI both limits accepted features and
            # avoids building a SIFT pyramid for pixels outside the requested
            # region.  Keypoints are shifted back into the resized full-frame
            # coordinate system before the existing /scale conversion.
            feature_gray = gray[scaled_y0:scaled_y1, scaled_x0:scaled_x1]
            feature_mask = np.full(feature_gray.shape, 255, dtype=np.uint8)
            crop_origin = np.array([scaled_x0, scaled_y0], dtype=np.float32)
            for pts in detected_cache[idx].values():
                quad = (
                    np.asarray(pts, dtype=np.float32).reshape(4, 2) * scale
                    - crop_origin)
                center = quad.mean(axis=0)
                radius = max(float(np.mean(np.linalg.norm(quad - center, axis=1))), 1.0)
                margin = FEATURE_MARKER_MASK_MARGIN_PX * scale
                expanded = center + (quad - center) * (1.0 + margin / radius)
                cv2.fillConvexPoly(feature_mask, np.round(expanded).astype(np.int32), 0)
            extractor = cv2.SIFT_create(
                nfeatures=FEATURE_MAX_KEYPOINTS, contrastThreshold=0.01)
            keypoints, descriptors = extractor.detectAndCompute(
                feature_gray, feature_mask)
            if keypoints and (scaled_x0 != 0 or scaled_y0 != 0):
                keypoints = [
                    cv2.KeyPoint(
                        float(keypoint.pt[0] + scaled_x0),
                        float(keypoint.pt[1] + scaled_y0),
                        float(keypoint.size), float(keypoint.angle),
                        float(keypoint.response), int(keypoint.octave),
                        int(keypoint.class_id))
                    for keypoint in keypoints
                ]
            feat_cache[idx] = (keypoints, descriptors)
        return feat_cache[idx]

    def feature_cell(pt):
        h, w = frame_height, frame_width
        x = min(FEATURE_GRID_COLS - 1, max(0, int(float(pt[0]) * FEATURE_GRID_COLS / max(w, 1))))
        y = min(FEATURE_GRID_ROWS - 1, max(0, int(float(pt[1]) * FEATURE_GRID_ROWS / max(h, 1))))
        return x, y

    def spatially_balance_matches(matches, kp_left, kp_right, max_matches=500):
        """Keep strong matches while preventing one textured patch from owning the pose."""
        counts_left = {}
        counts_right = {}
        selected = []
        for match in sorted(matches, key=lambda m: m.distance):
            cell_left = feature_cell(feature_point_fullres(kp_left[match.queryIdx]))
            cell_right = feature_cell(feature_point_fullres(kp_right[match.trainIdx]))
            if counts_left.get(cell_left, 0) >= FEATURE_MAX_MATCHES_PER_CELL:
                continue
            if counts_right.get(cell_right, 0) >= FEATURE_MAX_MATCHES_PER_CELL:
                continue
            selected.append(match)
            counts_left[cell_left] = counts_left.get(cell_left, 0) + 1
            counts_right[cell_right] = counts_right.get(cell_right, 0) + 1
            if len(selected) >= max_matches:
                break
        return selected

    match_cache = {}
    match_diagnostics_cache = {}

    def get_pair_matches(idx_left, idx_right):
        """左(結尾段)→右(開頭段) 的 SIFT 匹配 (ratio + mutual)，回傳已去畸變至 K_L 座標的點對。"""
        key = (idx_left, idx_right)
        if key in match_cache:
            return match_cache[key]
        kpL, desL = get_frame_features(idx_left)
        kpR, desR = get_frame_features(idx_right)
        diagnostics = {
            'left_keypoint_count': int(len(kpL)),
            'right_keypoint_count': int(len(kpR)),
            'left_descriptor_count': 0 if desL is None else int(len(desL)),
            'right_descriptor_count': 0 if desR is None else int(len(desR)),
            'knn_pair_count': 0,
            'ratio_pass_count': 0,
            'mutual_pass_count': 0,
            'spatially_balanced_count': 0,
        }
        result = None
        if desL is not None and desR is not None and len(desL) >= 8 and len(desR) >= 8:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            knn_lr = bf.knnMatch(desL, desR, k=2)
            knn_rl = bf.knnMatch(desR, desL, k=1)
            diagnostics['knn_pair_count'] = int(len(knn_lr))
            reverse_best = {m[0].queryIdx: m[0].trainIdx for m in knn_rl if m}
            good = []
            for pair in knn_lr:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < FEATURE_MATCH_RATIO * n.distance:
                    diagnostics['ratio_pass_count'] += 1
                    if reverse_best.get(m.trainIdx) == m.queryIdx:
                        good.append(m)
            diagnostics['mutual_pass_count'] = int(len(good))
            if len(good) >= 8:
                good = spatially_balance_matches(good, kpL, kpR)
                diagnostics['spatially_balanced_count'] = int(len(good))
                ptsL = np.float32([feature_point_fullres(kpL[m.queryIdx]) for m in good]).reshape(-1, 1, 2)
                ptsR = np.float32([feature_point_fullres(kpR[m.trainIdx]) for m in good]).reshape(-1, 1, 2)
                ptsL_u = cv2.undistortPoints(ptsL, mtx_L, dist_L, P=K_L).reshape(-1, 2).astype(np.float64)
                ptsR_u = cv2.undistortPoints(ptsR, mtx_L, dist_L, P=K_L).reshape(-1, 2).astype(np.float64)
                result = (ptsL_u, ptsR_u)
        match_diagnostics_cache[key] = diagnostics
        match_cache[key] = result
        return result

    def rt_epipolar_residuals(ptsL_u, ptsR_u, R_rel_c, t_rel_c):
        """Return per-match symmetric epipolar distances for a left-to-right pose."""
        ptsL_u = np.asarray(ptsL_u, dtype=np.float64).reshape(-1, 2)
        ptsR_u = np.asarray(ptsR_u, dtype=np.float64).reshape(-1, 2)
        if len(ptsL_u) == 0 or len(ptsL_u) != len(ptsR_u):
            return np.empty(0, dtype=np.float64)
        t = np.asarray(t_rel_c, dtype=np.float64).flatten()
        if np.linalg.norm(t) < 1e-9:
            return np.full(len(ptsL_u), float('inf'), dtype=np.float64)
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        K_inv = np.linalg.inv(K_L.astype(np.float64))
        F = K_inv.T @ (tx @ np.asarray(R_rel_c, dtype=np.float64)) @ K_inv
        onesL = np.hstack([ptsL_u, np.ones((len(ptsL_u), 1))])
        onesR = np.hstack([ptsR_u, np.ones((len(ptsR_u), 1))])
        lR = onesL @ F.T   # 左點在右圖上的極線
        lL = onesR @ F     # 右點在左圖上的極線
        num = np.abs(np.sum(lR * onesR, axis=1))
        dR = num / np.maximum(np.hypot(lR[:, 0], lR[:, 1]), 1e-12)
        dL = num / np.maximum(np.hypot(lL[:, 0], lL[:, 1]), 1e-12)
        return 0.5 * (dR + dL)

    def rt_epipolar_residual(ptsL_u, ptsR_u, R_rel_c, t_rel_c):
        """給定 左→右 相對位姿，計算點對的中位數對稱極線距離 (px)。"""
        residuals = rt_epipolar_residuals(ptsL_u, ptsR_u, R_rel_c, t_rel_c)
        return float(np.median(residuals)) if len(residuals) else float('inf')

    def signed_sampson_residuals(ptsL_u, ptsR_u, R_rel_c, t_rel_c):
        """Signed first-order geometric residuals in pixels for nonlinear refinement."""
        ptsL_u = np.asarray(ptsL_u, dtype=np.float64).reshape(-1, 2)
        ptsR_u = np.asarray(ptsR_u, dtype=np.float64).reshape(-1, 2)
        t = np.asarray(t_rel_c, dtype=np.float64).reshape(3)
        if len(ptsL_u) == 0 or len(ptsL_u) != len(ptsR_u) or np.linalg.norm(t) < 1e-9:
            return np.full(len(ptsL_u), 1e3, dtype=np.float64)
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        K_inv = np.linalg.inv(K_L.astype(np.float64))
        F = K_inv.T @ (tx @ np.asarray(R_rel_c, dtype=np.float64)) @ K_inv
        onesL = np.hstack([ptsL_u, np.ones((len(ptsL_u), 1))])
        onesR = np.hstack([ptsR_u, np.ones((len(ptsR_u), 1))])
        lR = onesL @ F.T
        lL = onesR @ F
        numerator = np.sum(lR * onesR, axis=1)
        denominator = np.sqrt(
            0.5 * (lR[:, 0] ** 2 + lR[:, 1] ** 2 + lL[:, 0] ** 2 + lL[:, 1] ** 2))
        return numerator / np.maximum(denominator, 1e-12)

    def marker_corner_pairs(corners_left_dict, corners_right_dict):
        shared = set(corners_left_dict.keys()) & set(corners_right_dict.keys())
        if not shared:
            return None
        pts_l = np.vstack([corners_left_dict[mid] for mid in shared]).astype(np.float64)
        pts_r = np.vstack([corners_right_dict[mid] for mid in shared]).astype(np.float64)
        return pts_l, pts_r

    def marker_object_corners():
        half_size = marker_size_mm / 2.0
        return np.array([
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ], dtype=np.float64)

    def project_undistorted_points(points_camera):
        points_camera = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        z = points_camera[:, 2]
        uvw = (K_L.astype(np.float64) @ points_camera.T).T
        projected = uvw[:, :2] / np.maximum(np.abs(uvw[:, 2:3]), 1e-9)
        return projected, z

    marker_pose_branch_cache = {}

    def marker_pose_branches(corners_u):
        cache_key = np.asarray(corners_u, dtype=np.float32).reshape(4, 2).tobytes()
        if cache_key in marker_pose_branch_cache:
            return marker_pose_branch_cache[cache_key]
        object_points = marker_object_corners().astype(np.float32)
        image_points = np.asarray(corners_u, dtype=np.float32).reshape(-1, 1, 2)
        try:
            _n_sol, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
                object_points, image_points, K_L.astype(np.float64), None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return []
        branches = []
        for rvec, tvec in zip(rvecs, tvecs):
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            branches.append((rotation, np.asarray(tvec, dtype=np.float64).reshape(3, 1)))
        marker_pose_branch_cache[cache_key] = branches
        return branches

    def prepare_marker_transfer_models(R_rel_c, t_rel_c, corners_left_u, corners_right_u):
        """Choose source-view IPPE branches and build metric marker points for both directions."""
        shared = sorted(set(corners_left_u.keys()) & set(corners_right_u.keys()))
        if not shared:
            return []
        object_points = marker_object_corners()
        R_rel64 = np.asarray(R_rel_c, dtype=np.float64).reshape(3, 3)
        t_rel64 = np.asarray(t_rel_c, dtype=np.float64).reshape(3, 1)
        models = []
        for marker_id in shared:
            observed_left = np.asarray(corners_left_u[marker_id], dtype=np.float64).reshape(4, 2)
            observed_right = np.asarray(corners_right_u[marker_id], dtype=np.float64).reshape(4, 2)
            left_candidates = []
            for R_marker, t_marker in marker_pose_branches(observed_left):
                points_left = (R_marker @ object_points.T + t_marker).T
                self_projection, z_left = project_undistorted_points(points_left)
                points_right = (R_rel64 @ points_left.T + t_rel64).T
                transfer_projection, z_right = project_undistorted_points(points_right)
                score = float(np.mean(np.linalg.norm(self_projection - observed_left, axis=1)))
                score += float(np.mean(np.linalg.norm(transfer_projection - observed_right, axis=1)))
                if np.any(z_left <= 0) or np.any(z_right <= 0):
                    score += 1e3
                left_candidates.append((score, points_left))

            right_candidates = []
            for R_marker, t_marker in marker_pose_branches(observed_right):
                points_right = (R_marker @ object_points.T + t_marker).T
                self_projection, z_right = project_undistorted_points(points_right)
                points_left = (R_rel64.T @ (points_right.T - t_rel64)).T
                transfer_projection, z_left = project_undistorted_points(points_left)
                score = float(np.mean(np.linalg.norm(self_projection - observed_right, axis=1)))
                score += float(np.mean(np.linalg.norm(transfer_projection - observed_left, axis=1)))
                if np.any(z_left <= 0) or np.any(z_right <= 0):
                    score += 1e3
                right_candidates.append((score, points_right))

            if not left_candidates or not right_candidates:
                continue
            points_left = min(left_candidates, key=lambda item: item[0])[1]
            points_right = min(right_candidates, key=lambda item: item[0])[1]
            models.append({
                'marker_id': int(marker_id),
                'points_left': points_left,
                'points_right': points_right,
                'observed_left': observed_left,
                'observed_right': observed_right,
            })
        return models

    def marker_transfer_residual_vector(R_rel_c, t_rel_c, marker_models):
        R_rel64 = np.asarray(R_rel_c, dtype=np.float64).reshape(3, 3)
        t_rel64 = np.asarray(t_rel_c, dtype=np.float64).reshape(3, 1)
        residuals = []
        for model in marker_models:
            predicted_right_3d = (R_rel64 @ model['points_left'].T + t_rel64).T
            predicted_right, z_right = project_undistorted_points(predicted_right_3d)
            predicted_left_3d = (R_rel64.T @ (model['points_right'].T - t_rel64)).T
            predicted_left, z_left = project_undistorted_points(predicted_left_3d)
            err_right = predicted_right - model['observed_right']
            err_left = predicted_left - model['observed_left']
            if np.any(z_right <= 0):
                err_right[:] = 1e3
            if np.any(z_left <= 0):
                err_left[:] = 1e3
            residuals.extend(err_right.reshape(-1))
            residuals.extend(err_left.reshape(-1))
        return np.asarray(residuals, dtype=np.float64)

    def marker_bidirectional_stats(R_rel_c, t_rel_c, corners_left_u, corners_right_u,
                                   marker_models=None):
        if marker_models is None:
            marker_models = prepare_marker_transfer_models(
                R_rel_c, t_rel_c, corners_left_u, corners_right_u)
        if not marker_models:
            return None
        R_rel64 = np.asarray(R_rel_c, dtype=np.float64).reshape(3, 3)
        t_rel64 = np.asarray(t_rel_c, dtype=np.float64).reshape(3, 1)
        forward = []
        reverse = []
        per_marker = []
        for model in marker_models:
            predicted_right_3d = (R_rel64 @ model['points_left'].T + t_rel64).T
            predicted_right, z_right = project_undistorted_points(predicted_right_3d)
            predicted_left_3d = (R_rel64.T @ (model['points_right'].T - t_rel64)).T
            predicted_left, z_left = project_undistorted_points(predicted_left_3d)
            err_forward = np.linalg.norm(predicted_right - model['observed_right'], axis=1)
            err_reverse = np.linalg.norm(predicted_left - model['observed_left'], axis=1)
            if np.any(z_right <= 0):
                err_forward[:] = 1e3
            if np.any(z_left <= 0):
                err_reverse[:] = 1e3
            forward.extend(err_forward)
            reverse.extend(err_reverse)
            per_marker.append({
                'marker_id': model['marker_id'],
                'left_to_right_rms_px': float(np.sqrt(np.mean(err_forward ** 2))),
                'right_to_left_rms_px': float(np.sqrt(np.mean(err_reverse ** 2))),
                'max_px': float(max(np.max(err_forward), np.max(err_reverse))),
            })
        forward = np.asarray(forward, dtype=np.float64)
        reverse = np.asarray(reverse, dtype=np.float64)
        both = np.concatenate([forward, reverse])
        return {
            'marker_count': len(marker_models),
            'left_to_right_rms_px': float(np.sqrt(np.mean(forward ** 2))),
            'right_to_left_rms_px': float(np.sqrt(np.mean(reverse ** 2))),
            'rms_px': float(np.sqrt(np.mean(both ** 2))),
            'median_px': float(np.median(both)),
            'max_px': float(np.max(both)),
            'per_marker': per_marker,
        }

    def rot_angle_deg(Ra, Rb):
        Rd = np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T
        return float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))))

    def feature_spatial_support(pts_left, pts_right):
        h, w = frame_height, frame_width
        image_area = float(max(h * w, 1))
        grid_total = float(FEATURE_GRID_COLS * FEATURE_GRID_ROWS)
        hull_ratios = []
        grid_ratios = []
        for pts in (pts_left, pts_right):
            pts32 = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            if len(pts32) >= 3:
                hull_ratios.append(float(cv2.contourArea(cv2.convexHull(pts32))) / image_area)
            else:
                hull_ratios.append(0.0)
            occupied = {feature_cell(pt) for pt in pts32}
            grid_ratios.append(len(occupied) / grid_total)
        return float(min(hull_ratios)), float(min(grid_ratios))

    def median_feature_parallax_deg(pts_left, pts_right, R_left_to_right):
        if len(pts_left) == 0:
            return 0.0
        K_inv = np.linalg.inv(K_L.astype(np.float64))
        left_h = np.hstack([pts_left, np.ones((len(pts_left), 1))])
        right_h = np.hstack([pts_right, np.ones((len(pts_right), 1))])
        rays_left = (K_inv @ left_h.T).T
        rays_right = (K_inv @ right_h.T).T
        rays_left /= np.maximum(np.linalg.norm(rays_left, axis=1, keepdims=True), 1e-12)
        rays_right /= np.maximum(np.linalg.norm(rays_right, axis=1, keepdims=True), 1e-12)
        rays_left_rotated = (np.asarray(R_left_to_right, np.float64) @ rays_left.T).T
        rays_left_rotated /= np.maximum(np.linalg.norm(rays_left_rotated, axis=1, keepdims=True), 1e-12)
        dots = np.sum(rays_left_rotated * rays_right, axis=1)
        return float(np.degrees(np.median(np.arccos(np.clip(dots, -1.0, 1.0)))))

    feature_geometry_cache = {}

    def estimate_feature_geometry(idx_left, idx_right):
        """Estimate and grade an Essential-matrix pose from marker-independent image features."""
        key = (idx_left, idx_right)
        if key in feature_geometry_cache:
            return feature_geometry_cache[key]
        matches_lr = get_pair_matches(idx_left, idx_right)
        if matches_lr is None or len(matches_lr[0]) < FEATURE_MIN_MATCHES:
            feature_geometry_cache[key] = None
            return None
        pts_left, pts_right = matches_lr
        K64 = K_L.astype(np.float64)
        E, mask_e = cv2.findEssentialMat(
            pts_left, pts_right, K64, method=cv2.RANSAC,
            prob=0.999, threshold=FEATURE_E_RANSAC_THRESH_PX)
        if E is None or E.ndim != 2 or E.shape[1] != 3 or E.shape[0] % 3 != 0:
            feature_geometry_cache[key] = None
            return None
        essential_candidates = [E[i:i + 3] for i in range(0, E.shape[0], 3)]
        if mask_e is None:
            mask_e = np.ones((len(pts_left), 1), dtype=np.uint8)
        essential_inlier_mask = np.asarray(mask_e).reshape(-1) != 0
        if len(essential_inlier_mask) != len(pts_left):
            essential_inlier_mask = np.ones(len(pts_left), dtype=bool)
            mask_e = essential_inlier_mask.astype(np.uint8).reshape(-1, 1)
        best = None
        for E_cand in essential_candidates:
            try:
                n_in, R_E, t_E, mask_pose = cv2.recoverPose(
                    E_cand, pts_left, pts_right, K64, mask=mask_e.copy())
            except cv2.error:
                continue
            inlier_mask = np.asarray(mask_pose).reshape(-1) != 0
            if len(inlier_mask) != len(pts_left):
                continue
            actual_inliers = int(np.count_nonzero(inlier_mask))
            if actual_inliers == 0:
                continue
            candidate = (actual_inliers, int(n_in), R_E, t_E, inlier_mask)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            feature_geometry_cache[key] = None
            return None

        n_in, _reported_in, R_E, t_E, inlier_mask = best
        in_left = pts_left[inlier_mask]
        in_right = pts_right[inlier_mask]
        inlier_ratio = float(n_in) / max(len(pts_left), 1)
        hull_coverage, grid_coverage = feature_spatial_support(in_left, in_right)
        parallax_deg = median_feature_parallax_deg(in_left, in_right, R_E)
        model_epi = rt_epipolar_residual(in_left, in_right, R_E, t_E)

        homography_ratio = 0.0
        if len(pts_left) >= 4:
            try:
                _H, mask_h = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, 2.0)
                if mask_h is not None:
                    homography_ratio = float(np.count_nonzero(mask_h)) / max(len(pts_left), 1)
            except cv2.error:
                pass
        planar_degenerate = (
            homography_ratio >= max(0.70, 0.90 * inlier_ratio)
            and parallax_deg < FEATURE_STRONG_PARALLAX_DEG)

        quality_ok = (
            n_in >= FEATURE_MIN_MATCHES
            and inlier_ratio >= FEATURE_MIN_INLIER_RATIO
            and grid_coverage >= FEATURE_MIN_GRID_COVERAGE
            and hull_coverage >= FEATURE_MIN_HULL_COVERAGE
            and parallax_deg >= FEATURE_MIN_PARALLAX_DEG
            and not planar_degenerate)
        strong = (
            quality_ok
            and n_in >= FEATURE_STRONG_INLIERS
            and inlier_ratio >= FEATURE_STRONG_INLIER_RATIO
            and grid_coverage >= FEATURE_STRONG_GRID_COVERAGE
            and hull_coverage >= FEATURE_STRONG_HULL_COVERAGE
            and parallax_deg >= FEATURE_STRONG_PARALLAX_DEG)

        support_penalty = max(0.0, FEATURE_STRONG_INLIER_RATIO - inlier_ratio) / FEATURE_STRONG_INLIER_RATIO
        grid_penalty = max(0.0, FEATURE_STRONG_GRID_COVERAGE - grid_coverage) / FEATURE_STRONG_GRID_COVERAGE
        hull_penalty = max(0.0, FEATURE_STRONG_HULL_COVERAGE - hull_coverage) / FEATURE_STRONG_HULL_COVERAGE
        parallax_penalty = max(0.0, FEATURE_STRONG_PARALLAX_DEG - parallax_deg) / FEATURE_STRONG_PARALLAX_DEG
        quality_penalty = (
            0.35 * support_penalty + 0.25 * grid_penalty + 0.15 * hull_penalty
            + 0.25 * parallax_penalty + (0.50 if planar_degenerate else 0.0))
        result = {
            'R': R_E,
            't': t_E.reshape(3, 1),
            'essential_inlier_mask': essential_inlier_mask,
            'essential_inlier_count': int(np.count_nonzero(essential_inlier_mask)),
            'inlier_mask': inlier_mask,
            'match_count': int(len(pts_left)),
            'inlier_count': int(n_in),
            'inlier_ratio': float(inlier_ratio),
            'hull_coverage': float(hull_coverage),
            'grid_coverage': float(grid_coverage),
            'parallax_deg': float(parallax_deg),
            'homography_ratio': float(homography_ratio),
            'planar_degenerate': bool(planar_degenerate),
            'model_epi_px': float(model_epi),
            'quality_penalty': float(quality_penalty),
            'quality_ok': bool(quality_ok),
            'strong': bool(strong),
        }
        feature_geometry_cache[key] = result
        return result

    def build_alt_rotations(item_s_c, item_e_c, R_chosen):
        """同一配對其餘 IPPE 分支組合的 R_rel 清單 (排除與已選解相同者)。"""
        alts = []
        for R_s_b, _t1 in item_s_c.get('branches', []):
            for R_e_b, _t2 in item_e_c.get('branches', []):
                R_c = R_s_b @ R_e_b.T
                if rot_angle_deg(R_c, R_chosen) > 0.5:
                    alts.append(R_c)
        return alts

    def baseline_from_marker_edges(R_rel_c, t_dir, corners_left_u, corners_right_u):
        """Recover metric baseline from every shared marker and reject inconsistent scale."""
        shared = sorted(set(corners_left_u.keys()) & set(corners_right_u.keys()))
        if not shared:
            return None, None
        t_u = np.asarray(t_dir, dtype=np.float64).reshape(3, 1)
        t_norm = float(np.linalg.norm(t_u))
        if t_norm < 1e-9:
            return None, None
        t_u = t_u / t_norm
        K64 = K_L.astype(np.float64)
        P0 = (K64 @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
        P1 = (K64 @ np.hstack([np.asarray(R_rel_c, np.float64), t_u])).astype(np.float32)
        marker_scales = []
        rejected_shape = 0
        for mid in shared:
            pts_left = np.asarray(corners_left_u[mid], np.float32).reshape(4, 2)
            pts_right = np.asarray(corners_right_u[mid], np.float32).reshape(4, 2)
            pts4d = cv2.triangulatePoints(P0, P1, pts_left.T, pts_right.T)
            w = pts4d[3]
            if np.any(np.abs(w) < 1e-12):
                continue
            X_left = pts4d[:3] / w
            X_right = np.asarray(R_rel_c, np.float64) @ X_left + t_u
            if np.count_nonzero(X_left[2] > 0) < 3 or np.count_nonzero(X_right[2] > 0) < 3:
                continue
            edges = np.array([
                np.linalg.norm(X_left[:, (i + 1) % 4] - X_left[:, i])
                for i in range(4)
            ], dtype=np.float64)
            edge_median = float(np.median(edges))
            if edge_median <= 1e-9:
                continue
            edge_cv = float(np.median(np.abs(edges - edge_median)) / edge_median)
            if edge_cv > FEATURE_SCALE_MAX_EDGE_CV:
                rejected_shape += 1
                continue
            marker_scales.append((mid, marker_size_mm / edge_median, edge_cv))
        if not marker_scales:
            return None, {'marker_count': 0, 'rejected_shape': rejected_shape}

        scales = np.array([item[1] for item in marker_scales], dtype=np.float64)
        scale_median = float(np.median(scales))
        rel_mad = float(np.median(np.abs(scales - scale_median)) / max(scale_median, 1e-9))
        diagnostics = {
            'marker_count': len(marker_scales),
            'rejected_shape': rejected_shape,
            'relative_mad': rel_mad,
            'per_marker': marker_scales,
        }
        if len(marker_scales) >= 2 and rel_mad > FEATURE_SCALE_MAX_MARKER_REL_MAD:
            diagnostics['inconsistent'] = True
            return None, diagnostics
        diagnostics['inconsistent'] = False
        return scale_median, diagnostics

    def plane_from_triangulated_corners(R_rel_c, t_rel_c, corners_left_u, corners_right_u):
        """
        用最終 RT 三角化共享標籤角點後 SVD 擬合平面 (左相機座標系)。
        與量測點走同一條幾何鏈，系統誤差在算點到平面距離時可相互抵消。
        """
        mk = marker_corner_pairs(corners_left_u, corners_right_u)
        if mk is None:
            return None, None
        ptsL, ptsR = mk
        K64 = K_L.astype(np.float64)
        P0 = (K64 @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
        P1 = (K64 @ np.hstack([np.asarray(R_rel_c, np.float64),
                               np.asarray(t_rel_c, np.float64).reshape(3, 1)])).astype(np.float32)
        pts4d = cv2.triangulatePoints(P0, P1, ptsL.T.astype(np.float32), ptsR.T.astype(np.float32))
        w = pts4d[3]
        if np.any(np.abs(w) < 1e-12):
            return None, None
        X = (pts4d[:3] / w).T
        if float(np.median(X[:, 2])) <= 0:
            return None, None
        c = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - c)
        n = Vt[-1]
        if np.dot(n, c) > 0:
            n = -n
        resid = float(np.sqrt(np.mean(((X - c) @ n) ** 2)))
        log_and_print(f"📐 [三角化平面] 角點數 {len(X)} | 平面 RMS 殘差 {resid:.3f} mm")
        return n.astype(np.float64), c.astype(np.float64)

    def feature_pose_stats(pts_left, pts_right, R_rel_c, t_rel_c, seed_inlier_mask=None,
                           holdout_mask=None):
        residuals = rt_epipolar_residuals(pts_left, pts_right, R_rel_c, t_rel_c)
        count = len(residuals)
        seed_mask = np.ones(count, dtype=bool)
        if seed_inlier_mask is not None and len(seed_inlier_mask) == count:
            seed_mask = np.asarray(seed_inlier_mask, dtype=bool).reshape(-1)
        final_inlier_mask = seed_mask & np.isfinite(residuals) & (residuals <= FEATURE_FINAL_INLIER_PX)
        inlier_values = residuals[final_inlier_mask]
        seed_values = residuals[seed_mask & np.isfinite(residuals)]
        holdout_values = np.empty(0, dtype=np.float64)
        if holdout_mask is not None and len(holdout_mask) == count:
            holdout_values = residuals[np.asarray(holdout_mask, dtype=bool) & np.isfinite(residuals)]
        return {
            'residuals_px': residuals,
            'final_inlier_mask': final_inlier_mask,
            'inlier_count': int(np.count_nonzero(final_inlier_mask)),
            'outlier_count': int(count - np.count_nonzero(final_inlier_mask)),
            'inlier_ratio': float(np.count_nonzero(final_inlier_mask)) / max(count, 1),
            'inlier_median_px': float(np.median(inlier_values)) if len(inlier_values) else float('inf'),
            'inlier_p90_px': float(np.percentile(inlier_values, 90)) if len(inlier_values) else float('inf'),
            'seed_median_px': float(np.median(seed_values)) if len(seed_values) else float('inf'),
            'seed_p90_px': float(np.percentile(seed_values, 90)) if len(seed_values) else float('inf'),
            'all_median_px': float(np.median(residuals)) if count else float('inf'),
            'all_p90_px': float(np.percentile(residuals, 90)) if count else float('inf'),
            'holdout_count': int(len(holdout_values)),
            'holdout_median_px': float(np.median(holdout_values)) if len(holdout_values) else None,
            'holdout_p90_px': float(np.percentile(holdout_values, 90)) if len(holdout_values) else None,
        }

    def optimize_marker_constrained_rt(R_start, t_start, marker_models,
                                       feature_left=None, feature_right=None,
                                       marker_group_weight=MARKER_JOINT_GROUP_WEIGHT):
        if not marker_models:
            return np.asarray(R_start, dtype=np.float64), np.asarray(t_start, dtype=np.float64).reshape(3, 1)
        rvec_start, _ = cv2.Rodrigues(np.asarray(R_start, dtype=np.float64).reshape(3, 3))
        x0 = np.concatenate([rvec_start.reshape(3), np.asarray(t_start, dtype=np.float64).reshape(3)])
        marker_coordinate_count = max(16 * len(marker_models), 1)
        marker_scale = np.sqrt(float(marker_group_weight) / marker_coordinate_count)
        use_features = (
            feature_left is not None and feature_right is not None
            and len(feature_left) == len(feature_right) and len(feature_left) >= 5)
        feature_scale = np.sqrt(FEATURE_JOINT_GROUP_WEIGHT / max(len(feature_left), 1)) if use_features else 0.0

        def residual_function(parameters):
            rotation, _ = cv2.Rodrigues(parameters[:3].reshape(3, 1))
            translation = parameters[3:6].reshape(3, 1)
            marker_residual = marker_transfer_residual_vector(
                rotation, translation, marker_models) * marker_scale
            residual_parts = [marker_residual]
            if use_features:
                raw_feature = signed_sampson_residuals(
                    feature_left, feature_right, rotation, translation)
                scaled_feature = raw_feature / FEATURE_FINAL_INLIER_PX
                pseudo_huber_cost = 2.0 * FEATURE_FINAL_INLIER_PX ** 2 * (
                    np.sqrt(1.0 + scaled_feature ** 2) - 1.0)
                robust_feature = np.sign(raw_feature) * np.sqrt(
                    np.maximum(pseudo_huber_cost, 0.0))
                residual_parts.append(robust_feature * feature_scale)
            baseline = float(np.linalg.norm(translation))
            residual_parts.append(np.array([
                max(0.0, MIN_BASELINE_MM - baseline) * 0.1,
                max(0.0, baseline - MAX_BASELINE_MM) * 0.1,
            ], dtype=np.float64))
            return np.concatenate(residual_parts)

        try:
            result = least_squares(
                residual_function, x0, method='trf', loss='linear',
                max_nfev=JOINT_RT_MAX_NFEV,
                ftol=JOINT_RT_TOL, xtol=JOINT_RT_TOL, gtol=JOINT_RT_TOL)
            rotation, _ = cv2.Rodrigues(result.x[:3].reshape(3, 1))
            translation = result.x[3:6].reshape(3, 1)
            return rotation, translation
        except (ValueError, np.linalg.LinAlgError) as optimize_error:
            log_and_print(f"⚠️ [RT聯合最佳化] 求解失敗: {optimize_error}")
            return np.asarray(R_start, dtype=np.float64), np.asarray(t_start, dtype=np.float64).reshape(3, 1)

    def refine_rt_with_features(idx_left, idx_right, R_aruco, t_aruco, corners_left_u, corners_right_u,
                                tag="", alt_rotations=None, single_marker=False):
        """Keep marker transfer as a hard constraint and use robust feature inliers to refine RT."""
        del alt_rotations, single_marker
        R_a64 = np.asarray(R_aruco, dtype=np.float64).reshape(3, 3)
        t_a64 = np.asarray(t_aruco, dtype=np.float64).reshape(3, 1)
        matches_lr = get_pair_matches(idx_left, idx_right)
        geometry = estimate_feature_geometry(idx_left, idx_right)
        marker_models = prepare_marker_transfer_models(
            R_a64, t_a64, corners_left_u, corners_right_u)
        empty_metrics = {
            'marker_bidir_ok': False,
            'marker_bidir': None,
            'feature': None,
            'optimization_mask': None,
            'holdout_mask': None,
            'used_feature_count': 0,
            'solution_role': 'aruco_fallback',
        }
        if not marker_models:
            log_and_print(f"⚠️ [RT精修{tag}] 無法建立 marker 雙向投影模型，保留 ArUco RT。")
            return R_aruco, t_aruco, False, empty_metrics

        marker_R, marker_t = optimize_marker_constrained_rt(R_a64, t_a64, marker_models)
        candidate_solutions = [('marker_only', marker_R, marker_t, False)]
        feature_train_left = None
        feature_train_right = None
        optimization_mask = None
        holdout_mask = None

        if ENABLE_FEATURE_RT_REFINE and matches_lr is not None and geometry is not None and geometry['quality_ok']:
            ptsL_u, ptsR_u = matches_lr
            seed_mask = np.asarray(geometry['inlier_mask'], dtype=bool).reshape(-1)
            if len(seed_mask) == len(ptsL_u):
                optimization_mask = seed_mask.copy()
                holdout_mask = np.zeros(len(seed_mask), dtype=bool)
                seed_indices = np.flatnonzero(seed_mask)
                if len(seed_indices) >= 11:
                    holdout_mask[seed_indices[::5]] = True
                    optimization_mask[holdout_mask] = False
                feature_train_left = ptsL_u[optimization_mask]
                feature_train_right = ptsR_u[optimization_mask]
                joint_seed_R, joint_seed_t = marker_R, marker_t
                for marker_weight in JOINT_MARKER_WEIGHT_LEVELS:
                    joint_R, joint_t = optimize_marker_constrained_rt(
                        joint_seed_R, joint_seed_t, marker_models,
                        feature_train_left, feature_train_right,
                        marker_group_weight=marker_weight)
                    candidate_solutions.append((
                        f'joint_from_marker_w{marker_weight:g}', joint_R, joint_t, True))
                    joint_seed_R, joint_seed_t = joint_R, joint_t

        elif matches_lr is None or geometry is None:
            log_and_print(f"ℹ️ [RT精修{tag}] 特徵匹配不足，僅執行 marker 雙向精修。")
        else:
            reason = "平面/低視差退化" if geometry['planar_degenerate'] else "內點或空間覆蓋不足"
            log_and_print(
                f"⚠️ [RT精修{tag}] 特徵幾何不可靠 ({reason})，僅執行 marker 雙向精修。")

        evaluated = []
        seed_mask = None
        ptsL_u = ptsR_u = None
        if matches_lr is not None:
            ptsL_u, ptsR_u = matches_lr
        if geometry is not None and ptsL_u is not None:
            mask_value = np.asarray(geometry.get('inlier_mask', []), dtype=bool).reshape(-1)
            if len(mask_value) == len(ptsL_u):
                seed_mask = mask_value
        for role, rotation, translation, uses_feature in candidate_solutions:
            baseline_value = float(np.linalg.norm(translation))
            if not (MIN_BASELINE_MM <= baseline_value <= MAX_BASELINE_MM):
                continue
            marker_stats = marker_bidirectional_stats(
                rotation, translation, corners_left_u, corners_right_u)
            if marker_stats is None:
                continue
            feature_stats = None
            if ptsL_u is not None:
                feature_stats = feature_pose_stats(
                    ptsL_u, ptsR_u, rotation, translation, seed_mask, holdout_mask)
            evaluated.append({
                'role': role,
                'R': rotation,
                't': translation,
                'uses_feature': uses_feature,
                'marker': marker_stats,
                'feature': feature_stats,
            })

        if not evaluated:
            log_and_print(f"⚠️ [RT精修{tag}] 無有效聯合候選，保留 ArUco RT。")
            return R_aruco, t_aruco, False, empty_metrics

        for candidate in evaluated:
            candidate_feature = candidate['feature'] or {}
            log_and_print(
                f"   [RT候選{tag}] {candidate['role']} | marker={candidate['marker']['rms_px']:.3f}px, "
                f"max={candidate['marker']['max_px']:.3f}px | "
                f"feature_seed_p90={candidate_feature.get('seed_p90_px', float('inf')):.3f}px, "
                f"final_inliers={candidate_feature.get('inlier_count', 0)}, "
                f"final_p90={candidate_feature.get('inlier_p90_px', float('inf')):.3f}px, "
                f"holdout_median={candidate_feature.get('holdout_median_px')}")

        marker_only_evaluated = [item for item in evaluated if not item['uses_feature']]
        marker_floor = min(
            marker_only_evaluated or evaluated,
            key=lambda item: item['marker']['rms_px'])['marker']
        rms_limit = min(
            MARKER_BIDIR_RMS_MAX_PX,
            marker_floor['rms_px'] + MARKER_BIDIR_RMS_MARGIN_PX)
        max_limit = min(
            MARKER_BIDIR_MAX_MAX_PX,
            marker_floor['max_px'] + MARKER_BIDIR_MAX_MARGIN_PX)
        marker_valid = [
            item for item in evaluated
            if item['marker']['rms_px'] <= rms_limit
            and item['marker']['left_to_right_rms_px'] <= MARKER_BIDIR_RMS_MAX_PX
            and item['marker']['right_to_left_rms_px'] <= MARKER_BIDIR_RMS_MAX_PX
            and item['marker']['max_px'] <= max_limit]

        def feature_candidate_ok(item):
            stats = item.get('feature')
            return bool(
                geometry is not None and geometry.get('quality_ok', False)
                and stats is not None
                and stats['inlier_count'] >= FEATURE_MIN_MATCHES
                and stats['inlier_p90_px'] <= FEATURE_FINAL_P90_MAX_PX
                and (stats['holdout_median_px'] is None
                     or stats['holdout_median_px'] <= FEATURE_FINAL_INLIER_PX))

        feature_valid = [item for item in marker_valid if feature_candidate_ok(item)]
        if feature_valid:
            best_feature_valid_marker_rms = min(
                item['marker']['rms_px'] for item in feature_valid)
            marker_priority_band = [
                item for item in feature_valid
                if item['marker']['rms_px']
                <= best_feature_valid_marker_rms + MARKER_SELECTION_RMS_BAND_PX]

            def joint_rank(item):
                feature_stats = item['feature'] or {}
                return (
                    feature_stats.get('seed_p90_px', float('inf')),
                    feature_stats.get('seed_median_px', float('inf')),
                    -feature_stats.get('inlier_count', 0),
                    item['marker']['rms_px'],
                )
            chosen = min(marker_priority_band, key=joint_rank)
        elif marker_valid:
            marker_only_valid = [item for item in marker_valid if not item['uses_feature']]
            chosen = min(
                marker_only_valid or marker_valid,
                key=lambda item: item['marker']['rms_px'])
        else:
            chosen = min(evaluated, key=lambda item: item['marker']['rms_px'])

        marker_ok = any(chosen is item for item in marker_valid)
        feature_stats = chosen['feature']
        feature_ok = feature_candidate_ok(chosen)
        applied = bool(chosen['uses_feature'] and marker_ok and feature_ok)
        solution_metrics = {
            'marker_bidir_ok': bool(marker_ok),
            'marker_bidir': chosen['marker'],
            'marker_floor': marker_floor,
            'marker_rms_limit_px': float(rms_limit),
            'marker_max_limit_px': float(max_limit),
            'feature': feature_stats,
            'optimization_mask': optimization_mask,
            'holdout_mask': holdout_mask,
            'used_feature_count': int(np.count_nonzero(optimization_mask)) if applied and optimization_mask is not None else 0,
            'solution_role': chosen['role'] if marker_ok else 'unreliable_marker_fallback',
            'feature_ok': feature_ok,
        }
        marker_text = chosen['marker']
        feature_text = feature_stats or {}
        log_and_print(
            f"{'✅' if marker_ok else '⚠️'} [RT聯合精修{tag}] role={solution_metrics['solution_role']} | "
            f"marker L→R={marker_text['left_to_right_rms_px']:.3f}px, "
            f"R→L={marker_text['right_to_left_rms_px']:.3f}px, max={marker_text['max_px']:.3f}px | "
            f"feature inliers={feature_text.get('inlier_count', 0)}/{len(ptsL_u) if ptsL_u is not None else 0}, "
            f"median={feature_text.get('inlier_median_px', float('inf')):.3f}px, "
            f"p90={feature_text.get('inlier_p90_px', float('inf')):.3f}px | "
            f"baseline={np.linalg.norm(chosen['t']):.2f}mm")
        return (
            chosen['R'].astype(np.asarray(R_aruco).dtype),
            chosen['t'].astype(np.asarray(t_aruco).dtype),
            applied,
            solution_metrics,
        )

    def refine_world_endpoint_pair(
            idx_left, idx_right, R_A0, t_A0, R_B0, t_B0,
            corners_A_raw, corners_B_raw, tag=""):
        """Primary final optimizer: absolute T_A<-W and T_B<-W.

        ``idx_left``/B is the later/end image; ``idx_right``/A is the earlier/start
        image, matching the existing B->A relative-pose convention.
        """
        matches_lr = get_pair_matches(idx_left, idx_right)
        geometry = estimate_feature_geometry(idx_left, idx_right)
        feature_B = feature_A = None
        feature_mask = None
        if matches_lr is not None and geometry is not None and geometry.get('quality_ok', False):
            feature_B, feature_A = matches_lr
            feature_mask = np.asarray(geometry.get('inlier_mask', []), bool).reshape(-1)
            if len(feature_mask) != len(feature_B):
                feature_mask = None
        result = _unified_optimize_endpoint_world_poses(
            R_A0, t_A0, R_B0, t_B0, corners_A_raw, corners_B_raw, marker_map,
            mtx_L, dist_L, K_L, marker_size_mm,
            feature_points_B=feature_B, feature_points_A=feature_A,
            feature_mask=feature_mask, marker_map_diagnostics=marker_map_diagnostics)
        if result is None:
            R_A = np.asarray(R_A0, np.float64).reshape(3, 3)
            t_A = np.asarray(t_A0, np.float64).reshape(3, 1)
            R_B = np.asarray(R_B0, np.float64).reshape(3, 3)
            t_B = np.asarray(t_B0, np.float64).reshape(3, 1)
            R_rel_local = R_A @ R_B.T
            t_rel_local = t_A - R_rel_local @ t_B
            direct = _unified_pair_reprojection_stats(
                R_A, t_A, corners_A_raw, R_B, t_B, corners_B_raw, marker_map,
                mtx_L, dist_L, marker_size_mm)
            metrics = {
                'marker_bidir_ok': False, 'marker_bidir': None, 'marker_direct': direct,
                'feature': None, 'feature_ok': False, 'optimization_mask': feature_mask,
                'holdout_mask': None, 'used_feature_count': 0,
                'solution_role': 'aruco_world_fallback',
            }
            return R_A, t_A, R_B, t_B, R_rel_local, t_rel_local, False, metrics

        marker = result['marker']
        legacy_marker = {
            'marker_count': marker['endpoint_A']['marker_count'] + marker['endpoint_B']['marker_count'],
            'left_to_right_rms_px': marker['endpoint_B']['rms_px'],
            'right_to_left_rms_px': marker['endpoint_A']['rms_px'],
            'rms_px': marker['rms_px'],
            'median_px': 0.5 * (marker['endpoint_A']['median_px'] + marker['endpoint_B']['median_px']),
            'max_px': marker['max_px'],
            'per_marker': [
                *[{'endpoint': 'A', **entry} for entry in marker['endpoint_A']['per_marker']],
                *[{'endpoint': 'B', **entry} for entry in marker['endpoint_B']['per_marker']],
            ],
        }
        feature_stats = result.get('feature')
        if feature_stats is not None:
            feature_stats = dict(feature_stats)
            feature_stats.setdefault('seed_median_px', feature_stats.get('all_median_px', float('inf')))
            feature_stats.setdefault('seed_p90_px', feature_stats.get('all_p90_px', float('inf')))
            feature_stats.setdefault('holdout_median_px', None)
            feature_stats.setdefault('holdout_p90_px', None)
        metrics = {
            'marker_bidir_ok': bool(result['marker_ok']),
            'marker_bidir': legacy_marker,
            'marker_direct': marker,
            'marker_floor': legacy_marker,
            'marker_rms_limit_px': float(UNIFIED_MARKER_DIRECT_RMS_MAX_PX),
            'marker_max_limit_px': float(UNIFIED_MARKER_DIRECT_MAX_PX),
            'feature': feature_stats,
            'feature_ok': bool(result['feature_ok']),
            'optimization_mask': feature_mask,
            'holdout_mask': None,
            'used_feature_count': int(len(feature_B)) if result['applied_feature'] and feature_B is not None else 0,
            'solution_role': result['role'],
            'world_endpoint_joint': True,
        }
        log_and_print(
            f"{'✅' if result['marker_ok'] else '⚠️'} [World RT聯合精修{tag}] "
            f"role={result['role']} marker={marker['rms_px']:.3f}px "
            f"feature_p90={feature_stats.get('inlier_p90_px', float('inf')) if feature_stats else float('inf'):.3f}px "
            f"baseline={result['baseline']:.2f}mm")
        return (
            result['R_A'], result['t_A'], result['R_B'], result['t_B'],
            result['R_rel'], result['t_rel'], bool(result['applied_feature']), metrics)

    def compute_pair_quality_score(
            err, item_s, item_e, rotation_s_from_e, translation_s_from_e, baseline_mm,
            branch_candidate_s=None, branch_candidate_e=None, reprojection_stats=None):
        sharp_s = get_frame_sharpness(item_s['idx'])
        sharp_e = get_frame_sharpness(item_e['idx'])
        sharp_min = max(min(sharp_s, sharp_e), 1e-6)
        blur_penalty = min(3.0, 120.0 / sharp_min)
        cover_s = marker_coverage_ratio(item_s['corners'])
        cover_e = marker_coverage_ratio(item_e['corners'])
        cover = min(cover_s, cover_e)
        coverage_penalty = max(0.0, 0.08 - cover) / 0.08
        cand_s = branch_candidate_s or {}
        cand_e = branch_candidate_e or {}
        pg_s = cand_s.get('pattern_observability') or {}
        pg_e = cand_e.get('pattern_observability') or {}
        angle_pair_diag = None
        angle_score_penalty = 0.0
        angle_pair_ok = True
        if angle_guided_diagnostics.get('status') == 'OK_ANGLE_GUIDED':
            angle_pair_ok = False
            target_marker_id = angle_guided_diagnostics.get('reference_marker_id')

            def incidence_for_target(observability):
                for entry in observability.get('per_marker', []) or []:
                    if int(entry.get('marker_id', -1)) == int(target_marker_id):
                        return entry.get('incidence_deg')
                return observability.get('incidence_deg')

            incidence_A = incidence_for_target(pg_s)
            incidence_B = incidence_for_target(pg_e)
            target_A = float(angle_guided_diagnostics.get(
                'chronological_target_A_deg', angle_cfg['target_frame_A_deg']))
            target_B = float(angle_guided_diagnostics.get(
                'chronological_target_B_deg', angle_cfg['target_frame_B_deg']))
            tolerance = float(angle_cfg['target_tolerance_deg'])
            if (incidence_A is not None and incidence_B is not None
                    and np.isfinite(incidence_A) and np.isfinite(incidence_B)):
                error_A = abs(float(incidence_A) - target_A)
                error_B = abs(float(incidence_B) - target_B)
                normalized_error = min(
                    3.0, 0.5 * (error_A + error_B) / max(tolerance, 1e-9))
                angle_score_penalty = (
                    float(angle_cfg['pair_score_weight']) * normalized_error)
                angle_pair_ok = bool(
                    error_A <= tolerance and error_B <= tolerance)
                angle_pair_diag = {
                    'marker_id': int(target_marker_id),
                    'incidence_A_deg': float(incidence_A),
                    'incidence_B_deg': float(incidence_B),
                    'target_A_deg': target_A,
                    'target_B_deg': target_B,
                    'error_A_deg': float(error_A),
                    'error_B_deg': float(error_B),
                    'normalized_error': float(normalized_error),
                    'score_penalty': float(angle_score_penalty),
                    'within_tolerance': bool(angle_pair_ok),
                }
        if pattern_guided_enabled:
            nominal_depth = pg_e.get('axial_depth_mm')
            nominal_depth_source = 'marker_axial_depth_B'
            if nominal_depth is None or not np.isfinite(nominal_depth) or nominal_depth <= 1.0:
                nominal_depth = float(pg_cfg['fallback_nominal_depth_mm'])
                nominal_depth_source = 'fallback_config'
            nominal_depth = float(nominal_depth)
            roi_B_for_geometry = geometry_roi_bounds_for_frame(item_e['idx'])
            roi_A_for_geometry = geometry_roi_bounds_for_frame(item_s['idx'])
        else:
            # Compatibility mode intentionally reproduces unified.py's fixed
            # 200 mm / detection-ROI geometry proxy.
            nominal_depth = float(PAIR_NOMINAL_DEPTH_MM)
            nominal_depth_source = 'legacy_fixed_200mm'
            roi_B_for_geometry = roi_bounds_for_frame(item_e['idx'])
            roi_A_for_geometry = roi_bounds_for_frame(item_s['idx'])
        geometry = _unified_nominal_pair_geometry(
            rotation_s_from_e, translation_s_from_e, K_L, frame_width, frame_height,
            nominal_depth_mm=nominal_depth,
            roi_B=roi_B_for_geometry, roi_A=roi_A_for_geometry)
        if pattern_guided_enabled:
            geometry_score, pg_pair_diag = _pattern_guided_pair_geometry_score(
                geometry, baseline_mm, nominal_depth, pg_s, pg_e, pg_cfg,
                ideal_baseline_mm=IDEAL_BASELINE_MM)
        else:
            # Exact legacy unified score for compatibility mode.
            geometry_score = _unified_depth_geometry_score(
                geometry, baseline_mm,
                ideal_baseline_mm=IDEAL_BASELINE_MM,
                target_depth_sigma_mm=PAIR_TARGET_DEPTH_SIGMA_MM,
                target_angle_deg=PAIR_TARGET_TRIANGULATION_ANGLE_DEG)
            pg_pair_diag = None
        conf_s = float(cand_s.get(
            'measurement_confidence', item_s.get('measurement_confidence', 0.75)))
        conf_e = float(cand_e.get(
            'measurement_confidence', item_e.get('measurement_confidence', 0.75)))
        measurement_confidence = min(conf_s, conf_e)
        measurement_penalty = max(0.0, 1.0 - measurement_confidence)
        score = (
            PAIR_SCORE_REPROJ_W * float(err)
            + geometry_score
            + PAIR_SCORE_BLUR_W * blur_penalty
            + PAIR_SCORE_COVER_W * coverage_penalty
            + PAIR_SCORE_MEASUREMENT_W * measurement_penalty
            + angle_score_penalty
        )
        shared_count = len(set(item_s['corners']) & set(item_e['corners']))
        metrics = {
            'score': float(score),
            'err': float(err),
            'baseline': float(baseline_mm),
            'baseline_gate_min_mm': float(runtime_pair_candidate_min_baseline),
            'marker_parallax_deg': float(geometry['triangulation_median_deg']),
            'triangulation_median_deg': float(geometry['triangulation_median_deg']),
            'triangulation_p10_deg': float(geometry['triangulation_p10_deg']),
            'overlap_ratio': float(geometry['overlap_ratio']),
            'predicted_depth_sigma_mm': float(geometry['predicted_depth_sigma_mm']),
            'pair_nominal_depth_mm': nominal_depth,
            'pair_nominal_depth_source': nominal_depth_source,
            'baseline_to_depth': float(baseline_mm / max(nominal_depth, 1e-9)),
            'shared_markers': int(shared_count),
            'sharpness_min': float(sharp_min),
            'coverage': float(cover),
            'measurement_confidence': float(measurement_confidence),
            'measurement_mode_start': cand_s.get(
                'measurement_mode', item_s.get('measurement_mode', 'UNKNOWN')),
            'measurement_mode_end': cand_e.get(
                'measurement_mode', item_e.get('measurement_mode', 'UNKNOWN')),
            'marker_direct': reprojection_stats,
            'pattern_observability_start': pg_s,
            'pattern_observability_end': pg_e,
            'pattern_guided': pg_pair_diag,
            'angle_guided': angle_pair_diag,
            'angle_guided_ok': bool(angle_pair_ok),
        }
        return float(score), metrics

    # 多階段漸進式匹配評估
    best_start = None
    best_end = None
    R_rel = None
    t_rel = None
    baseline = None
    selected_extras = []
    marker_map = {}
    temporal_valid_poses = {}
    temporal_diagnostics = {
        'enabled': True,
        'neighbor_probes_enabled': bool(ENABLE_TEMPORAL_NEIGHBOR_PROBES),
        'status': 'NOT_RUN',
    }
    
    # Five cheap ArUco probes span the useful motion interval; only two pairs reach SIFT.
    stages = [1]
    pair_search_timing = {
        'endpoint_proposal': 0.0,
        'marker_map_temporal': 0.0,
        'pattern_guided_expansion': 0.0,
        'local_window_klt': 0.0,
        'candidate_enumeration': 0.0,
        'sift_essential_rerank': 0.0,
    }
    # These children are intentionally measured as non-overlapping wall-clock
    # blocks.  In particular, ThreadPoolExecutor waits are timed outside worker
    # functions so parallel SIFT/ArUco work is not double-counted.
    pair_search_detail_timing = {
        'endpoint_frame_decode_aruco': 0.0,
        'endpoint_pose_direction_rank': 0.0,
        'marker_probe_decode_detect': 0.0,
        'marker_probe_video_decode': 0.0,
        'marker_probe_gray_convert': 0.0,
        'marker_probe_clahe': 0.0,
        'marker_probe_aruco_detect': 0.0,
        'marker_probe_corner_subpix': 0.0,
        'marker_probe_cache_overhead': 0.0,
        'marker_map_graph': 0.0,
        'temporal_pose_hypotheses': 0.0,
        'temporal_path_dp': 0.0,
        'marker_temporal_overhead': 0.0,
        'pattern_core_geometry': 0.0,
        'pattern_extra_probe_rebuild': 0.0,
        'pattern_overhead': 0.0,
        'local_provisional_pair': 0.0,
        'local_frame_decode_detect': 0.0,
        'local_pose_hypotheses': 0.0,
        'local_klt_tracking': 0.0,
        'local_path_rerank': 0.0,
        'local_global_dp_rebuild': 0.0,
        'local_diagnostic_logging': 0.0,
        'local_overhead': 0.0,
        'candidate_branch_pack': 0.0,
        'candidate_pair_branch_score': 0.0,
        'candidate_topk_budget': 0.0,
        'candidate_overhead': 0.0,
        'sift_feature_extract': 0.0,
        'sift_descriptor_match': 0.0,
        'sift_essential_geometry': 0.0,
        'sift_rerank_and_select': 0.0,
        'sift_overhead': 0.0,
    }

    def add_pair_detail(key, elapsed_s):
        pair_search_detail_timing[key] = (
            float(pair_search_detail_timing.get(key, 0.0))
            + max(0.0, float(elapsed_s)))
    stage_success = False
    best_branch = None
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    for stage_idx, num_samples in enumerate(stages):
        log_and_print("🔄 評估自適應跨段候選幀對...")

        _phase_start = time.perf_counter()
        sampled_start, sampled_end = select_adaptive_candidate_frames(
            start_range, end_range)
        endpoint_elapsed = time.perf_counter() - _phase_start
        pair_search_timing['endpoint_proposal'] += endpoint_elapsed
        endpoint_detect_elapsed = min(
            endpoint_elapsed,
            float(angle_guided_diagnostics.get('frame_decode_aruco_s', 0.0)))
        add_pair_detail('endpoint_frame_decode_aruco', endpoint_detect_elapsed)
        add_pair_detail(
            'endpoint_pose_direction_rank', endpoint_elapsed - endpoint_detect_elapsed)
        _phase_start = time.perf_counter()
        if angle_guided_diagnostics.get('status') == 'OK_ANGLE_GUIDED':
            # Reuse every already-detected scan pose for marker-map/temporal
            # branch continuity, while only the target-nearest frames enter
            # expensive cross-segment pair enumeration and SIFT.
            temporal_start_indices = list(
                angle_guided_diagnostics['scanned_indices_A'])
            temporal_end_indices = list(
                angle_guided_diagnostics['scanned_indices_B'])
        else:
            temporal_start_indices = _expand_temporal_probe_indices(
                sampled_start, start_range,
                radius=TEMPORAL_NEIGHBOR_RADIUS if ENABLE_TEMPORAL_NEIGHBOR_PROBES else 0)
            temporal_end_indices = _expand_temporal_probe_indices(
                sampled_end, end_range,
                radius=TEMPORAL_NEIGHBOR_RADIUS if ENABLE_TEMPORAL_NEIGHBOR_PROBES else 0)
        _detail_start = time.perf_counter()
        marker_probe_child_keys = (
            'marker_probe_video_decode',
            'marker_probe_gray_convert',
            'marker_probe_clahe',
            'marker_probe_aruco_detect',
            'marker_probe_corner_subpix',
            'marker_probe_cache_overhead',
        )
        marker_probe_children_before = sum(
            pair_search_detail_timing[key] for key in marker_probe_child_keys)
        ensure_profiled_probe_markers(
            [*temporal_start_indices, *temporal_end_indices])

        start_info = get_frame_info(sampled_start, stage_idx, is_start_segment=True)
        end_info = get_frame_info(sampled_end, stage_idx, is_start_segment=False)
        temporal_start_info = get_frame_info(
            temporal_start_indices, stage_idx, is_start_segment=True)
        temporal_end_info = get_frame_info(
            temporal_end_indices, stage_idx, is_start_segment=False)
        marker_probe_elapsed = time.perf_counter() - _detail_start
        marker_probe_children_added = (
            sum(pair_search_detail_timing[key] for key in marker_probe_child_keys)
            - marker_probe_children_before)
        add_pair_detail(
            'marker_probe_cache_overhead',
            max(0.0, marker_probe_elapsed - marker_probe_children_added))
        add_pair_detail('marker_probe_decode_detect', marker_probe_elapsed)

        if not start_info or not end_info:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：開頭段或結尾段無有效 ArUco 標籤")
            continue
            
        start_ids = set()
        for item in start_info:
            start_ids.update(item['corners'].keys())
        end_ids = set()
        for item in end_info:
            end_ids.update(item['corners'].keys())

        # Build a fixed rigid marker map from all sparse probes before choosing
        # an endpoint pair.  Endpoints are allowed to observe different marker IDs;
        # only graph connectivity through robust co-visible observations is required.
        temporal_info_by_index = {}
        for item in temporal_start_info + temporal_end_info:
            temporal_info_by_index[int(item['idx'])] = item
        temporal_info = [
            temporal_info_by_index[index]
            for index in sorted(temporal_info_by_index)
        ]
        temporal_marker_ids = set()
        for item in temporal_info:
            temporal_marker_ids.update(item['corners'])
        _detail_start = time.perf_counter()
        marker_map, marker_map_diagnostics, ref_id = _build_temporal_marker_map_graph(
            temporal_info, temporal_marker_ids, mtx_L, dist_L, marker_size_mm,
            start_marker_ids=start_ids, end_marker_ids=end_ids)
        add_pair_detail('marker_map_graph', time.perf_counter() - _detail_start)
        if ref_id is None:
            graph_diag = marker_map_diagnostics.get('_graph', {})
            log_and_print(
                f"⚠️ 第 {stage_idx + 1} 階段：endpoint markers 無法由 robust 共視 marker-map 連通 "
                f"(components={graph_diag.get('components')})")
            continue
        mapped_start_ids = sorted(set(start_ids) & set(marker_map))
        mapped_end_ids = sorted(set(end_ids) & set(marker_map))
        if not mapped_start_ids or not mapped_end_ids:
            log_and_print(
                f"⚠️ 第 {stage_idx + 1} 階段：marker-map 未覆蓋兩端觀測 "
                f"(start={mapped_start_ids}, end={mapped_end_ids})")
            continue
        log_and_print(
            f"📌 [第 {stage_idx + 1} 階段] marker-map reference ID={ref_id}, "
            f"mapped={sorted(marker_map)}, start={mapped_start_ids}, end={mapped_end_ids}")

        # Keep per-frame hypotheses until the complete sparse trajectory is
        # available.  All hypotheses are first converted to T_camera<-reference,
        # so a switch from one visible marker ID to another does not change frame.
        _detail_start = time.perf_counter()
        frame_candidates = {}
        for item in temporal_info:
            candidates = _build_temporal_frame_candidates(
                item, marker_map, mtx_L, dist_L, marker_size_mm,
                marker_map_diagnostics=marker_map_diagnostics)
            if candidates:
                if pattern_guided_enabled or angle_guided_enabled:
                    for candidate in candidates:
                        candidate['pattern_observability'] = _pattern_guided_marker_observability(
                            candidate, item['corners'], marker_map, marker_size_mm,
                            marker_map_diagnostics=marker_map_diagnostics)
                frame_candidates[int(item['idx'])] = candidates
        add_pair_detail(
            'temporal_pose_hypotheses', time.perf_counter() - _detail_start)
        _detail_start = time.perf_counter()
        selected_temporal_path, path_diagnostics = _select_temporal_pose_path(
            frame_candidates,
            nominal_probe_indices=[*temporal_start_indices, *temporal_end_indices])
        add_pair_detail('temporal_path_dp', time.perf_counter() - _detail_start)
        temporal_dp_model = path_diagnostics.get('_dp_model', {
            'segments': [], 'frame_to_segment': {}})
        core_probe_indices = {
            int(index) for index in [*sampled_start, *sampled_end]
        }
        ambiguous_core_indices = sorted(
            frame_index for frame_index, candidate
            in selected_temporal_path.items()
            if frame_index in core_probe_indices
            and candidate.get('measurement_mode') in ('SINGLE', 'SINGLE_FALLBACK'))
        measurement_modes_by_frame = {
            int(frame_index): candidate.get('measurement_mode', 'UNKNOWN')
            for frame_index, candidate in selected_temporal_path.items()
        }
        temporal_valid_poses = {
            int(frame_index): (
                np.asarray(candidate['R'], np.float64),
                np.asarray(candidate['t'], np.float64).reshape(3, 1))
            for frame_index, candidate in selected_temporal_path.items()
        }
        temporal_diagnostics = {
            'enabled': True,
            'neighbor_probes_enabled': bool(ENABLE_TEMPORAL_NEIGHBOR_PROBES),
            'status': path_diagnostics.get('status', 'UNKNOWN'),
            'reference_marker_id': int(ref_id),
            'core_start_indices': [int(index) for index in sampled_start],
            'core_end_indices': [int(index) for index in sampled_end],
            'temporal_start_indices': [int(index) for index in temporal_start_indices],
            'temporal_end_indices': [int(index) for index in temporal_end_indices],
            'marker_map': marker_map_diagnostics,
            'path': {k: v for k, v in path_diagnostics.items() if not k.startswith('_')},
            'ambiguous_core_indices': ambiguous_core_indices,
            'measurement_modes_by_frame': measurement_modes_by_frame,
        }

        log_and_print(
            f"🧭 [時序姿態] observations={len(selected_temporal_path)}, "
            f"segments={len(path_diagnostics.get('segments', []))}, "
            f"map_ids={sorted(marker_map)}")
        marker_temporal_elapsed = time.perf_counter() - _phase_start
        pair_search_timing['marker_map_temporal'] += marker_temporal_elapsed
        marker_children = (
            pair_search_detail_timing['marker_probe_decode_detect']
            + pair_search_detail_timing['marker_map_graph']
            + pair_search_detail_timing['temporal_pose_hypotheses']
            + pair_search_detail_timing['temporal_path_dp'])
        pair_search_detail_timing['marker_temporal_overhead'] = max(
            0.0, pair_search_timing['marker_map_temporal'] - marker_children)

        # Pattern-guided expansion is intentionally bounded: first evaluate the
        # existing five probes.  Only if no selected-path core pair has basic
        # depth geometry do we use camera-centre trends to propose ArUco-only
        # endpoint probes.  These frames are merged into start/end core lists,
        # so they are genuine frame-pair candidates, not temporal-only neighbours.
        _phase_start = time.perf_counter()
        if pattern_guided_enabled and pg_cfg.get('adaptive_extra_probes', True):
            _detail_start = time.perf_counter()
            core_geometry_ok = False
            core_geometry_best = None
            for idx_s in sampled_start:
                cand_s = selected_temporal_path.get(int(idx_s))
                if cand_s is None:
                    continue
                for idx_e in sampled_end:
                    cand_e = selected_temporal_path.get(int(idx_e))
                    if cand_e is None:
                        continue
                    R_s = np.asarray(cand_s['R'], np.float64)
                    t_s = np.asarray(cand_s['t'], np.float64).reshape(3, 1)
                    R_e = np.asarray(cand_e['R'], np.float64)
                    t_e = np.asarray(cand_e['t'], np.float64).reshape(3, 1)
                    R_ab = R_s @ R_e.T
                    t_ab = t_s - R_ab @ t_e
                    exact_baseline = float(np.linalg.norm(
                        np.asarray(cand_s['camera_center']) - np.asarray(cand_e['camera_center'])))
                    depth_proxy = (cand_e.get('pattern_observability') or {}).get('axial_depth_mm')
                    if depth_proxy is None or not np.isfinite(depth_proxy) or depth_proxy <= 1.0:
                        depth_proxy = float(pg_cfg['fallback_nominal_depth_mm'])
                    geom = _unified_nominal_pair_geometry(
                        R_ab, t_ab, K_L, frame_width, frame_height,
                        nominal_depth_mm=float(depth_proxy),
                        roi_B=geometry_roi_bounds_for_frame(int(idx_e)),
                        roi_A=geometry_roi_bounds_for_frame(int(idx_s)))
                    ok = _pattern_guided_basic_geometry_ok(
                        geom, exact_baseline, float(depth_proxy), pg_cfg)
                    rank = (
                        -float(geom.get('overlap_ratio', 0.0)),
                        abs(float(geom.get('triangulation_median_deg', 0.0)) - 15.0),
                        float(geom.get('predicted_depth_sigma_mm', float('inf'))))
                    if core_geometry_best is None or rank < core_geometry_best[0]:
                        core_geometry_best = (rank, int(idx_s), int(idx_e), geom, exact_baseline, float(depth_proxy))
                    core_geometry_ok = core_geometry_ok or ok
            pattern_guided_diagnostics['core_geometry_ok'] = bool(core_geometry_ok)
            if core_geometry_best is not None:
                pattern_guided_diagnostics['core_geometry_best'] = {
                    'frame_A': core_geometry_best[1], 'frame_B': core_geometry_best[2],
                    'baseline_mm': core_geometry_best[4],
                    'nominal_depth_mm': core_geometry_best[5],
                    **core_geometry_best[3],
                }
            add_pair_detail(
                'pattern_core_geometry', time.perf_counter() - _detail_start)
            elapsed = time.perf_counter() - analysis_wall_start
            cap_left = max(0, int(pg_cfg['max_total_aruco_probes']) - len(set(sampled_start + sampled_end)))
            extra_limit = min(int(pg_cfg['max_extra_probes']), cap_left)
            if core_geometry_ok:
                pattern_guided_diagnostics['adaptive_status'] = 'CORE_GEOMETRY_OK'
            elif elapsed > float(pg_cfg['extra_probe_deadline_s']):
                pattern_guided_diagnostics['adaptive_status'] = 'SKIPPED_DEADLINE'
                pattern_guided_diagnostics['deadline_skipped'] = True
            elif extra_limit <= 0:
                pattern_guided_diagnostics['adaptive_status'] = 'PROBE_CAP_REACHED'
            else:
                _detail_start = time.perf_counter()
                proposal_cfg = dict(pg_cfg)
                proposal_cfg['max_extra_probes'] = int(extra_limit)
                proposal_depth = (
                    core_geometry_best[5] if core_geometry_best is not None
                    else float(pg_cfg['fallback_nominal_depth_mm']))
                proposal = _pattern_guided_propose_extra_probe_indices(
                    sampled_start, sampled_end, selected_temporal_path,
                    start_range, end_range, proposal_depth, proposal_cfg)
                extra_start = list(proposal.get('start', []))
                extra_end = list(proposal.get('end', []))
                if extra_start or extra_end:
                    if hasattr(frames, 'preload'):
                        frames.preload([*extra_start, *extra_end])
                    extra_start_info = get_frame_info(
                        extra_start, stage_idx, is_start_segment=True)
                    extra_end_info = get_frame_info(
                        extra_end, stage_idx, is_start_segment=False)
                    start_info.extend(extra_start_info)
                    end_info.extend(extra_end_info)
                    sampled_start = list(dict.fromkeys([*sampled_start, *extra_start]))
                    sampled_end = list(dict.fromkeys([*sampled_end, *extra_end]))
                    temporal_start_indices = list(dict.fromkeys([
                        *temporal_start_indices, *extra_start]))
                    temporal_end_indices = list(dict.fromkeys([
                        *temporal_end_indices, *extra_end]))
                    for item in extra_start_info + extra_end_info:
                        candidates = _build_temporal_frame_candidates(
                            item, marker_map, mtx_L, dist_L, marker_size_mm,
                            marker_map_diagnostics=marker_map_diagnostics)
                        if not candidates:
                            continue
                        for candidate in candidates:
                            candidate['pattern_observability'] = _pattern_guided_marker_observability(
                                candidate, item['corners'], marker_map, marker_size_mm,
                                marker_map_diagnostics=marker_map_diagnostics)
                        frame_candidates[int(item['idx'])] = candidates
                    selected_temporal_path, path_diagnostics = _select_temporal_pose_path(
                        frame_candidates,
                        nominal_probe_indices=[*temporal_start_indices, *temporal_end_indices])
                    temporal_dp_model = path_diagnostics.get('_dp_model', {
                        'segments': [], 'frame_to_segment': {}})
                    temporal_valid_poses = {
                        int(frame_index): (
                            np.asarray(candidate['R'], np.float64),
                            np.asarray(candidate['t'], np.float64).reshape(3, 1))
                        for frame_index, candidate in selected_temporal_path.items()
                    }
                    measurement_modes_by_frame = {
                        int(frame_index): candidate.get('measurement_mode', 'UNKNOWN')
                        for frame_index, candidate in selected_temporal_path.items()
                    }
                    pattern_guided_diagnostics['adaptive_status'] = 'EXTRA_PROBES_ADDED'
                    pattern_guided_diagnostics['extra_start_indices'] = [int(x) for x in extra_start]
                    pattern_guided_diagnostics['extra_end_indices'] = [int(x) for x in extra_end]
                    pattern_guided_diagnostics['proposal'] = proposal
                else:
                    pattern_guided_diagnostics['adaptive_status'] = proposal.get('status', 'NO_PROPOSAL')
                add_pair_detail(
                    'pattern_extra_probe_rebuild',
                    time.perf_counter() - _detail_start)
            pattern_guided_diagnostics['total_aruco_probe_count'] = int(len(set(sampled_start + sampled_end)))
            temporal_diagnostics['pattern_guided'] = pattern_guided_diagnostics
            temporal_diagnostics['core_start_indices'] = [int(index) for index in sampled_start]
            temporal_diagnostics['core_end_indices'] = [int(index) for index in sampled_end]
            temporal_diagnostics['temporal_start_indices'] = [int(index) for index in temporal_start_indices]
            temporal_diagnostics['temporal_end_indices'] = [int(index) for index in temporal_end_indices]
            temporal_diagnostics['measurement_modes_by_frame'] = measurement_modes_by_frame
            temporal_diagnostics['status'] = path_diagnostics.get('status', 'UNKNOWN')
            temporal_diagnostics['path'] = {
                k: v for k, v in path_diagnostics.items() if not k.startswith('_')}
            updated_core_indices = {int(x) for x in [*sampled_start, *sampled_end]}
            temporal_diagnostics['ambiguous_core_indices'] = sorted(
                frame_index for frame_index, candidate in selected_temporal_path.items()
                if frame_index in updated_core_indices
                and candidate.get('measurement_mode') in ('SINGLE', 'SINGLE_FALLBACK'))
        else:
            temporal_diagnostics['pattern_guided'] = pattern_guided_diagnostics
        pattern_elapsed = time.perf_counter() - _phase_start
        pair_search_timing['pattern_guided_expansion'] += pattern_elapsed
        pattern_children = (
            pair_search_detail_timing['pattern_core_geometry']
            + pair_search_detail_timing['pattern_extra_probe_rebuild'])
        pair_search_detail_timing['pattern_overhead'] = max(
            0.0, pair_search_timing['pattern_guided_expansion'] - pattern_children)

        # Selected-endpoint local window: only after the provisional sparse/pattern-guided pair exists.
        # No local SIFT is performed; the fixed marker map is frozen throughout this stage.
        _phase_start = time.perf_counter()
        if local_window_enabled:
            local_stage_start = time.perf_counter()
            local_stage_deadline = min(
                analysis_wall_start + float(lw_cfg['analysis_elapsed_deadline_s']),
                local_stage_start + float(lw_cfg['local_stage_budget_s']))

            def provisional_item(info_item, candidate):
                return {
                    'idx': int(info_item['idx']), 'corners': info_item['corners'],
                    'R': np.asarray(candidate['R'], np.float64),
                    't': np.asarray(candidate['t'], np.float64).reshape(3, 1),
                    'measurement_mode': candidate.get('measurement_mode', 'UNKNOWN'),
                    'measurement_confidence': candidate.get('measurement_confidence', 0.75),
                }

            _detail_start = time.perf_counter()
            provisional_pairs = []
            info_start_by_idx = {int(x['idx']): x for x in start_info}
            info_end_by_idx = {int(x['idx']): x for x in end_info}
            for idx_s, info_s in info_start_by_idx.items():
                cand_s = selected_temporal_path.get(idx_s)
                if cand_s is None:
                    continue
                for idx_e, info_e in info_end_by_idx.items():
                    cand_e = selected_temporal_path.get(idx_e)
                    if cand_e is None:
                        continue
                    R_s = np.asarray(cand_s['R'], np.float64)
                    t_s = np.asarray(cand_s['t'], np.float64).reshape(3, 1)
                    R_e = np.asarray(cand_e['R'], np.float64)
                    t_e = np.asarray(cand_e['t'], np.float64).reshape(3, 1)
                    C_s = _temporal_camera_center(R_s, t_s)
                    C_e = _temporal_camera_center(R_e, t_e)
                    bsl = float(np.linalg.norm(C_s - C_e))
                    if not (runtime_pair_candidate_min_baseline <= bsl <= MAX_BASELINE_MM):
                        continue
                    item_s = provisional_item(info_s, cand_s)
                    item_e = provisional_item(info_e, cand_e)
                    reproj, stats = compute_pair_reprojection_error(
                        item_s, item_e, mtx_L, dist_L, R_s=R_s, t_s=t_s,
                        R_e=R_e, t_e=t_e, return_stats=True)
                    if not np.isfinite(reproj) or stats is None:
                        continue
                    R_ab = R_s @ R_e.T
                    t_ab = t_s - R_ab @ t_e
                    score, metrics = compute_pair_quality_score(
                        reproj, item_s, item_e, R_ab, t_ab, bsl,
                        branch_candidate_s=cand_s, branch_candidate_e=cand_e,
                        reprojection_stats=stats)
                    if not metrics.get('angle_guided_ok', True):
                        continue
                    provisional_pairs.append((float(score), idx_s, idx_e, cand_s, cand_e, metrics))
            provisional_pairs.sort(key=lambda x: x[0])
            add_pair_detail(
                'local_provisional_pair', time.perf_counter() - _detail_start)
            if not provisional_pairs:
                local_window_diagnostics['status'] = 'FALLBACK_NO_PROVISIONAL_PAIR'
                local_window_diagnostics['fallback_reason'] = 'NO_PROVISIONAL_PAIR'
            elif time.perf_counter() > local_stage_deadline:
                local_window_diagnostics['status'] = 'FALLBACK_DEADLINE_BEFORE_WINDOW'
                local_window_diagnostics['fallback_reason'] = 'DEADLINE'
            else:
                _, provisional_A, provisional_B, anchor_A, anchor_B, _ = provisional_pairs[0]
                local_window_diagnostics['provisional_pair'] = {
                    'idx_A': int(provisional_A), 'idx_B': int(provisional_B)}
                log_and_print(
                    f"🔍 [Local window輸入] provisional A=F{provisional_A}, "
                    f"B=F{provisional_B} | radius={int(lw_cfg['radius'])}, "
                    f"stride={int(lw_cfg['stride'])}, "
                    f"max_candidates/side={int(lw_cfg['max_endpoint_candidates'])}")
                requested = {
                    'A': _local_window_indices(
                        provisional_A, start_range, len(frames), lw_cfg['radius'], lw_cfg['stride']),
                    'B': _local_window_indices(
                        provisional_B, end_range, len(frames), lw_cfg['radius'], lw_cfg['stride']),
                }
                centers = {'A': provisional_A, 'B': provisional_B}
                anchors = {'A': anchor_A, 'B': anchor_B}
                side_ranges = {'A': start_range, 'B': end_range}
                side_is_start = {'A': True, 'B': False}
                chosen_items_by_side = {}
                local_all_indices = []
                local_selected_paths = {}
                for side in ('A', 'B'):
                    center_idx = int(centers[side])
                    center_corners = detected_cache.get(center_idx, {})
                    indices = requested[side]
                    allowed_indices = []
                    for idx in indices:
                        if idx == center_idx or idx in detected_cache:
                            allowed_indices.append(idx)
                            continue
                        if len(local_unique_new_frames) >= int(lw_cfg['max_unique_new_frames']):
                            break
                        if time.perf_counter() > local_stage_deadline:
                            break
                        local_unique_new_frames.add(int(idx))
                        allowed_indices.append(int(idx))
                    if center_idx not in allowed_indices:
                        allowed_indices.append(center_idx)
                    allowed_indices = sorted(set(allowed_indices))
                    _detail_start = time.perf_counter()
                    if hasattr(frames, 'preload') and allowed_indices:
                        frames.preload(allowed_indices)
                    for idx in allowed_indices:
                        if idx not in detected_cache:
                            detect_local_frame_markers(idx, center_corners, local_stage_deadline)
                    info = [
                        {'idx': int(idx), 'corners': detected_cache[int(idx)]}
                        for idx in allowed_indices if detected_cache.get(int(idx))
                    ]
                    add_pair_detail(
                        'local_frame_decode_detect',
                        time.perf_counter() - _detail_start)
                    # Build raw local hypotheses in the frozen marker map.
                    _detail_start = time.perf_counter()
                    for item in info:
                        idx = int(item['idx'])
                        if idx not in frame_candidates:
                            cands = _build_temporal_frame_candidates(
                                item, marker_map, mtx_L, dist_L, marker_size_mm,
                                marker_map_diagnostics=marker_map_diagnostics)
                            if cands:
                                for cand in cands:
                                    if pattern_guided_enabled or angle_guided_enabled:
                                        cand['pattern_observability'] = _pattern_guided_marker_observability(
                                            cand, item['corners'], marker_map, marker_size_mm,
                                            marker_map_diagnostics=marker_map_diagnostics)
                                frame_candidates[idx] = cands
                    add_pair_detail(
                        'local_pose_hypotheses',
                        time.perf_counter() - _detail_start)
                    local_obs = [(int(item['idx']), frame_candidates[int(item['idx'])])
                                 for item in info if int(item['idx']) in frame_candidates]
                    local_obs.sort(key=lambda x: x[0])
                    # Adjacent KLT only. Cache is global to both windows so shared pairs are never repeated.
                    _detail_start = time.perf_counter()
                    for (i0, _), (i1, _) in zip(local_obs, local_obs[1:]):
                        key = (int(i0), int(i1))
                        if key not in local_klt_cache:
                            if time.perf_counter() > local_stage_deadline:
                                local_klt_cache[key] = {'available': False, 'reason': 'DEADLINE'}
                            else:
                                local_klt_cache[key] = _local_klt_track_pair(
                                    get_local_raw_gray(i0), get_local_raw_gray(i1), K_L,
                                    marker_corners=detected_cache.get(i0), config=lw_cfg)
                    add_pair_detail(
                        'local_klt_tracking', time.perf_counter() - _detail_start)
                    _detail_start = time.perf_counter()
                    sharpness = {int(f): get_frame_sharpness(int(f)) for f, _ in local_obs}
                    local_path_adjusted, local_path_diag = _local_window_path(
                        local_obs, center_idx, anchors[side], sharpness, local_klt_cache, lw_cfg)
                    # DP evaluates copied candidates with local-only emission terms. Convert the
                    # selected branch back to the raw production candidate so the global DP/pair
                    # enumeration never inherits local scoring as a stale source of truth.
                    local_path = {}
                    for idx, selected_copy in local_path_adjusted.items():
                        original_index = int(selected_copy.get('_local_original_index', 0))
                        originals = frame_candidates.get(int(idx), [])
                        if 0 <= original_index < len(originals):
                            local_path[int(idx)] = originals[original_index]
                    local_selected_paths[side] = local_path
                    # Score real frames, not filtered virtual states. Center offset is deliberately weak.
                    ranked_frames = []
                    ranking_diagnostics = []
                    max_sharp = max(list(sharpness.values()) + [1.0])
                    for idx, cand in local_path.items():
                        reproj = float(cand.get('reprojection_rms_px', float('inf')))
                        sharp_pen = 1.0 - min(1.0, sharpness.get(idx, 0.0) / max(max_sharp, 1e-9))
                        klt_neighbors = [v for (a, b), v in local_klt_cache.items() if idx in (a, b)]
                        valid_klt = [v for v in klt_neighbors if v.get('available')]
                        klt_pen = 0.0
                        if valid_klt:
                            klt_pen = float(np.mean([
                                min(float(v.get('fb_median_px') or 0.0) / max(float(lw_cfg['klt_fb_gate_px']), 1e-9), 2.0)
                                + (1.0 - float(v.get('grid_coverage', 0.0)))
                                for v in valid_klt]))
                        emission_cost = float(cand.get('emission_cost', 0.0))
                        temporal_component = 0.35 * emission_cost
                        sharpness_component = float(lw_cfg['sharpness_weight']) * sharp_pen
                        klt_component = 0.05 * klt_pen
                        offset = int(idx - center_idx)
                        offset_component = (
                            float(lw_cfg['frame_offset_tie_weight']) * abs(offset))
                        score = (
                            reproj + temporal_component + sharpness_component
                            + klt_component + offset_component)
                        ranked_frames.append((float(score), int(idx), cand))
                        ranking_diagnostics.append({
                            'idx': int(idx),
                            'total_score': float(score),
                            'reprojection_component': float(reproj),
                            'temporal_emission_raw': float(emission_cost),
                            'temporal_component': float(temporal_component),
                            'sharpness_raw': float(sharpness.get(idx, 0.0)),
                            'sharpness_penalty': float(sharp_pen),
                            'sharpness_component': float(sharpness_component),
                            'valid_klt_neighbor_count': int(len(valid_klt)),
                            'klt_penalty': float(klt_pen),
                            'klt_component': float(klt_component),
                            'center_offset': int(offset),
                            'offset_component': float(offset_component),
                            'measurement_mode': cand.get('measurement_mode', 'UNKNOWN'),
                            'branch_label': cand.get('label'),
                        })
                    ranked_frames.sort(key=lambda x: x[0])
                    ranking_diagnostics.sort(key=lambda entry: entry['total_score'])
                    for rank, entry in enumerate(ranking_diagnostics, start=1):
                        entry['rank'] = int(rank)
                    protected = []
                    if ranked_frames:
                        protected.append(ranked_frames[0][1])
                    if center_idx in local_path and center_idx not in protected:
                        protected.append(center_idx)
                    for _score, idx, _cand in ranked_frames:
                        if idx not in protected:
                            protected.append(idx)
                        if len(protected) >= int(lw_cfg['max_endpoint_candidates']):
                            break
                    protected = protected[:int(lw_cfg['max_endpoint_candidates'])]
                    for entry in ranking_diagnostics:
                        idx = int(entry['idx'])
                        entry['protected'] = bool(idx in protected)
                        if idx == protected[0] and idx == center_idx:
                            entry['protection_reason'] = 'LOCAL_BEST_AND_CENTER'
                        elif idx == protected[0]:
                            entry['protection_reason'] = 'LOCAL_BEST'
                        elif idx == center_idx and idx in protected:
                            entry['protection_reason'] = 'CENTER_SAFETY_KEEP'
                        elif idx in protected:
                            entry['protection_reason'] = 'RUNNER_UP'
                        else:
                            entry['protection_reason'] = 'NOT_PROTECTED'
                    # Optional tiny pose-prior blend; every accepted result must pass marker reprojection again.
                    smooth_diag = {}
                    for idx in list(protected):
                        raw = local_path.get(idx)
                        if raw is None:
                            continue
                        item = next((x for x in info if int(x['idx']) == int(idx)), None)
                        if item is None:
                            continue
                        filtered, sd = _local_try_pose_denoise(
                            raw, idx, local_path, item['corners'], marker_map,
                            mtx_L, dist_L, marker_size_mm, lw_cfg)
                        smooth_diag[int(idx)] = sd
                        if filtered is not raw and sd.get('accepted'):
                            frame_candidates[idx].append(filtered)
                            local_path[idx] = filtered
                    local_selected_paths[side] = local_path
                    chosen_items = [
                        next(x for x in info if int(x['idx']) == idx)
                        for idx in protected if any(int(x['idx']) == idx for x in info)
                    ]
                    chosen_items_by_side[side] = chosen_items
                    local_all_indices.extend([int(x['idx']) for x in info])
                    best_idx = protected[0] if protected else center_idx
                    endpoint_klt = {
                        f'{a}->{b}': dict(v) for (a, b), v in local_klt_cache.items()
                        if a in allowed_indices and b in allowed_indices
                    }
                    local_window_diagnostics['endpoints'][side] = {
                        'requested_indices': [int(x) for x in requested[side]],
                        'used_indices': [int(x) for x in allowed_indices],
                        'center_index': int(center_idx), 'chosen_index': int(best_idx),
                        'chosen_offset': int(best_idx - center_idx),
                        'protected_indices': [int(x) for x in protected],
                        'valid_aruco_frames': [int(x['idx']) for x in info],
                        'candidate_ranking': ranking_diagnostics,
                        'branch_sequence': local_path_diag.get('branch_sequence', []),
                        'klt': endpoint_klt, 'smooth': smooth_diag,
                    }
                    add_pair_detail(
                        'local_path_rerank', time.perf_counter() - _detail_start)
                    _local_log_start = time.perf_counter()
                    log_and_print(
                        f"🔎 [Local {side}候選排名] center=F{center_idx} | "
                        f"requested={requested[side]} | used={allowed_indices}")
                    for entry in ranking_diagnostics:
                        log_and_print(
                            f"   #{entry['rank']} F{entry['idx']} "
                            f"total={entry['total_score']:.4f} | "
                            f"reproj={entry['reprojection_component']:.4f} + "
                            f"temporal={entry['temporal_component']:.4f} "
                            f"(raw {entry['temporal_emission_raw']:.4f}) + "
                            f"sharp={entry['sharpness_component']:.4f} "
                            f"(raw {entry['sharpness_raw']:.1f}, pen {entry['sharpness_penalty']:.3f}) + "
                            f"KLT={entry['klt_component']:.4f} "
                            f"(valid-neighbors {entry['valid_klt_neighbor_count']}) + "
                            f"offset={entry['offset_component']:.4f} "
                            f"(ΔF {entry['center_offset']:+d}) | "
                            f"pose={entry['measurement_mode']}/{entry['branch_label']} | "
                            f"keep={entry['protection_reason']}")
                    if endpoint_klt:
                        for pair_name, klt_info in endpoint_klt.items():
                            fb_value = klt_info.get('fb_median_px')
                            fb_text = (
                                f"{float(fb_value):.3f}px"
                                if fb_value is not None and np.isfinite(fb_value)
                                else 'N/A')
                            rotation_value = klt_info.get('rotation_deg')
                            rotation_text = (
                                f"{float(rotation_value):.3f}deg"
                                if rotation_value is not None and np.isfinite(rotation_value)
                                else 'N/A')
                            essential_ratio = klt_info.get('essential_inlier_ratio')
                            essential_text = (
                                f"{float(essential_ratio):.3f}"
                                if essential_ratio is not None and np.isfinite(essential_ratio)
                                else 'N/A')
                            homography_ratio = klt_info.get('homography_inlier_ratio')
                            homography_text = (
                                f"{float(homography_ratio):.3f}"
                                if homography_ratio is not None and np.isfinite(homography_ratio)
                                else 'N/A')
                            log_and_print(
                                f"   [KLT {pair_name}] available={bool(klt_info.get('available'))} | "
                                f"reason={klt_info.get('reason')} | "
                                f"tracks={int(klt_info.get('valid_count', 0))}/"
                                f"{int(klt_info.get('initial_count', 0))} | "
                                f"FB median={fb_text} | "
                                f"grid={float(klt_info.get('grid_coverage', 0.0)):.2f} | "
                                f"rotation={rotation_text} | "
                                f"E-inlier={essential_text} | "
                                f"recoverPose={int(klt_info.get('recoverpose_inlier_count', 0))}/"
                                f"{int(klt_info.get('recoverpose_min_required', 0))} | "
                                f"H-inlier={homography_text}")
                    chosen_entry = next((
                        entry for entry in ranking_diagnostics
                        if int(entry['idx']) == int(best_idx)), None)
                    center_entry = next((
                        entry for entry in ranking_diagnostics
                        if int(entry['idx']) == int(center_idx)), None)
                    changed = bool(best_idx != center_idx)
                    log_and_print(
                        f"🎯 [Local {side}選擇] F{center_idx} → F{best_idx} | "
                        f"changed={changed} | protected={protected}")
                    if changed and chosen_entry is not None and center_entry is not None:
                        component_labels = (
                            ('reprojection_component', '重投影'),
                            ('temporal_component', '時序emission'),
                            ('sharpness_component', '清晰度'),
                            ('klt_component', 'KLT'),
                            ('offset_component', '中心距離'),
                        )
                        advantages = []
                        for key, label in component_labels:
                            improvement = float(center_entry[key] - chosen_entry[key])
                            advantages.append((improvement, label))
                        advantages.sort(reverse=True)
                        positive = [
                            f"{label}改善 {improvement:.4f}"
                            for improvement, label in advantages if improvement > 1e-9]
                        disadvantages = [
                            f"{label}變差 {abs(improvement):.4f}"
                            for improvement, label in advantages if improvement < -1e-9]
                        log_and_print(
                            f"   [換幀原因] total改善 "
                            f"{center_entry['total_score'] - chosen_entry['total_score']:.4f} | "
                            f"優勢: {', '.join(positive) if positive else '無單項正改善'} | "
                            f"代價: {', '.join(disadvantages) if disadvantages else '無'}")
                    elif changed and center_entry is None:
                        log_and_print(
                            f"   [換幀原因] 中心F{center_idx}沒有有效的ArUco/IPPE時序姿態，"
                            f"因此改用有效候選F{best_idx}。")
                    else:
                        log_and_print(
                            "   [不換幀原因] 中心Frame仍是Local綜合成本最低；"
                            "鄰幀只保留為後續Pair/IPPE與SIFT備選。")
                    add_pair_detail(
                        'local_diagnostic_logging',
                        time.perf_counter() - _local_log_start)
                # Rebuild a standard DP model including local observations so exact joint min-marginal
                # remains available whenever both endpoints lie in the same connected temporal segment.
                local_nominal = sorted(set([
                    *temporal_start_indices, *temporal_end_indices, *local_all_indices]))
                if local_all_indices:
                    _detail_start = time.perf_counter()
                    dp_selected, dp_diag = _select_temporal_pose_path(
                        frame_candidates, nominal_probe_indices=local_nominal)
                    temporal_dp_model = dp_diag.get('_dp_model', temporal_dp_model)
                    # Preserve the independently anchored local branch path at local indices.
                    selected_temporal_path.update(dp_selected)
                    for side in ('A', 'B'):
                        selected_temporal_path.update(local_selected_paths.get(side, {}))
                    temporal_valid_poses = {
                        int(frame_index): (
                            np.asarray(candidate['R'], np.float64),
                            np.asarray(candidate['t'], np.float64).reshape(3, 1))
                        for frame_index, candidate in selected_temporal_path.items()
                    }
                    add_pair_detail(
                        'local_global_dp_rebuild',
                        time.perf_counter() - _detail_start)
                if chosen_items_by_side.get('A') and chosen_items_by_side.get('B'):
                    start_info = chosen_items_by_side['A']
                    end_info = chosen_items_by_side['B']
                    local_window_diagnostics['status'] = 'OK_LOCAL_RERANK'
                else:
                    local_window_diagnostics['status'] = 'FALLBACK_INSUFFICIENT_LOCAL_DATA'
                    local_window_diagnostics['fallback_reason'] = 'INSUFFICIENT_LOCAL_DATA'
                local_window_diagnostics['new_frames'] = sorted(int(x) for x in local_unique_new_frames)
                local_window_diagnostics['local_frames_decoded'] = int(len(local_unique_new_frames))
                local_window_diagnostics['local_frames_detected'] = int(len([
                    x for x in local_detection_mode if x in local_unique_new_frames]))
                local_window_diagnostics['klt_pair_count'] = int(len(local_klt_cache))
                local_window_diagnostics['stage_elapsed_s'] = float(time.perf_counter() - local_stage_start)
        else:
            local_window_diagnostics['status'] = 'DISABLED'
        temporal_diagnostics['local_window'] = local_window_diagnostics
        local_elapsed = time.perf_counter() - _phase_start
        pair_search_timing['local_window_klt'] += local_elapsed
        local_children = sum(pair_search_detail_timing[key] for key in (
            'local_provisional_pair', 'local_frame_decode_detect',
            'local_pose_hypotheses', 'local_klt_tracking',
            'local_path_rerank', 'local_global_dp_rebuild',
            'local_diagnostic_logging'))
        pair_search_detail_timing['local_overhead'] = max(
            0.0, pair_search_timing['local_window_klt'] - local_children)
        _phase_start = time.perf_counter()

        def build_temporal_valid(core_items):
            valid = []
            for item in core_items:
                frame_index = int(item['idx'])
                candidate = selected_temporal_path.get(frame_index)
                if candidate is None:
                    continue
                all_candidates = list(frame_candidates.get(frame_index, []))
                try:
                    selected_index = next(
                        index for index, value in enumerate(all_candidates)
                        if value is candidate)
                except StopIteration:
                    continue
                ordered = [(selected_index, candidate)]
                ordered.extend(sorted(
                    ((index, value) for index, value in enumerate(all_candidates)
                     if value is not candidate),
                    key=lambda pair: (pair[1]['emission_cost'], pair[1].get('label', ''))))
                branches = [
                    (np.asarray(value['R'], np.float64),
                     np.asarray(value['t'], np.float64).reshape(3, 1))
                    for _index, value in ordered
                ]
                rotation, translation = branches[0]
                valid.append({
                    'idx': frame_index,
                    'R': rotation,
                    't': translation,
                    'branches': branches,
                    'branch_candidates': [value for _index, value in ordered],
                    'branch_candidate_indices': [int(index) for index, _value in ordered],
                    'corners': item['corners'],
                    'temporal_candidate': candidate,
                    'measurement_mode': candidate.get('measurement_mode', 'UNKNOWN'),
                    'measurement_confidence': candidate.get('measurement_confidence', 0.75),
                })
            return valid

        _detail_start = time.perf_counter()
        valid_start = build_temporal_valid(start_info)
        valid_end = build_temporal_valid(end_info)
        add_pair_detail(
            'candidate_branch_pack', time.perf_counter() - _detail_start)

        if not valid_start or not valid_end:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無法計算有效的起點或終點 Joint Pose")
            continue

        # 計算候選對的重投影誤差與 baseline (單標籤模式對每幀的 IPPE 雙解分支展開組合;
        # 注意單標籤時 reproj err 是自我擬合殘差、對分支無鑑別力，真正的裁決在特徵極線重排)
        _detail_start = time.perf_counter()
        sharpness_indices = [item['idx'] for item in valid_start + valid_end]
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            list(executor.map(get_frame_sharpness, sharpness_indices))

        pairs = []
        for item_s in valid_start:
            for item_e in valid_end:
                for bi_s, (R_s, t_s) in enumerate(item_s['branches']):
                    for bi_e, (R_e, t_e) in enumerate(item_e['branches']):
                        R_rel_cand = R_s @ R_e.T
                        t_rel_cand = t_s - R_rel_cand @ t_e
                        # Candidate baseline is explicitly computed from common-world
                        # camera centres C=-R^T t.  ||t_rel|| is mathematically equal
                        # for a rigid common world and remains the final output convention.
                        C_s = _temporal_camera_center(R_s, t_s)
                        C_e = _temporal_camera_center(R_e, t_e)
                        bsl = float(np.linalg.norm(C_s - C_e))
                        pair_candidate_min_baseline = float(runtime_pair_candidate_min_baseline)
                        if not (pair_candidate_min_baseline <= bsl <= MAX_BASELINE_MM):
                            continue
                        err_pair, direct_stats = compute_pair_reprojection_error(
                            item_s, item_e, mtx_L, dist_L,
                            R_s=R_s, t_s=t_s, R_e=R_e, t_e=t_e, return_stats=True)
                        if err_pair == float('inf') or direct_stats is None:
                            continue
                        cand_s = item_s['branch_candidates'][bi_s]
                        cand_e = item_e['branch_candidates'][bi_e]
                        pair_score, pair_metrics = compute_pair_quality_score(
                            err_pair, item_s, item_e, R_rel_cand, t_rel_cand, bsl,
                            branch_candidate_s=cand_s, branch_candidate_e=cand_e,
                            reprojection_stats=direct_stats)
                        if not pair_metrics.get('angle_guided_ok', True):
                            continue
                        pair_metrics['marker_direct_ok'] = bool(
                            direct_stats['rms_px'] <= MARKER_CANDIDATE_RMS_MAX_PX
                            and direct_stats['max_px'] <= MARKER_CANDIDATE_MAX_MAX_PX)
                        # Preserve legacy keys for downstream ranking; semantics are
                        # now direct endpoint reprojection rather than shared-marker transfer.
                        pair_metrics['marker_bidir_ok'] = pair_metrics['marker_direct_ok']
                        pair_metrics['marker_bidir_rms_px'] = direct_stats['rms_px']
                        pair_metrics['marker_bidir_max_px'] = direct_stats['max_px']
                        pair_metrics['marker_left_to_right_rms_px'] = direct_stats['endpoint_B']['rms_px']
                        pair_metrics['marker_right_to_left_rms_px'] = direct_stats['endpoint_A']['rms_px']
                        pair_metrics['branch'] = (bi_s, bi_e)
                        temporal_index_s = item_s['branch_candidate_indices'][bi_s]
                        temporal_index_e = item_e['branch_candidate_indices'][bi_e]
                        pair_metrics['temporal_candidate_indices'] = (
                            int(temporal_index_s), int(temporal_index_e))
                        pair_metrics['temporal_branch_selected'] = (
                            bi_s == 0 and bi_e == 0)
                        pair_metrics['temporal_branch_confidence'] = None
                        pair_metrics['temporal_joint_min_marginal'] = None
                        pair_metrics['temporal_branch_prior_cost'] = 0.0
                        pairs.append((pair_score, err_pair, item_s, item_e, R_rel_cand, t_rel_cand, bsl, pair_metrics))
        add_pair_detail(
            'candidate_pair_branch_score', time.perf_counter() - _detail_start)

        if not pairs:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無合格的匹配對 (Baseline gate: {runtime_pair_candidate_min_baseline:.2f}~{MAX_BASELINE_MM} mm)")
            continue
            
        _detail_start = time.perf_counter()
        # 依誤差由小到大排序
        pairs.sort(key=lambda x: x[0])

        logged_pair_keys = set()
        for candidate in pairs:
            candidate_key = (candidate[2]['idx'], candidate[3]['idx'])
            if candidate_key in logged_pair_keys:
                continue
            logged_pair_keys.add(candidate_key)
            candidate_metrics = candidate[7]
            log_and_print(
                f"   [ArUco候選] A=F{candidate_key[0]} B=F{candidate_key[1]} "
                f"baseline={candidate[6]:.2f}mm "
                f"parallax={candidate_metrics.get('marker_parallax_deg', 0.0):.2f}deg "
                f"depth_sigma={candidate_metrics.get('predicted_depth_sigma_mm', float('inf')):.2f}mm "
                f"score={candidate[0]:.3f}")

        # 前 K 個「幀對」加算特徵極線殘差後重排 (真正的品質裁決)：
        # 自我擬合殘差對極線幾何無鑑別力，收斂與否由特徵極線殘差決定。
        # 以幀對為單位套用多樣性配額，避免名額被相鄰近似幀塞滿；
        # 單標籤模式下同一幀對的所有 IPPE 分支組合全數保留 (共用同一次 SIFT 匹配)。
        _admitted = set()
        _cnt_start, _cnt_end = {}, {}
        topk = []
        first_admitted_key = None
        first_admitted_score = None
        for cand_tuple in pairs:
            _key = (cand_tuple[2]['idx'], cand_tuple[3]['idx'])
            if _key not in _admitted:
                if len(_admitted) >= PAIR_EPI_TOPK:
                    continue
                if (first_admitted_score is not None
                        and cand_tuple[0] > first_admitted_score + PAIR_SECOND_SCORE_MARGIN):
                    continue
                if _cnt_start.get(_key[0], 0) >= PAIR_TOPK_MAX_PER_START:
                    continue
                if _cnt_end.get(_key[1], 0) >= PAIR_TOPK_MAX_PER_END:
                    continue
                _admitted.add(_key)
                if first_admitted_key is None:
                    first_admitted_key = _key
                    first_admitted_score = cand_tuple[0]
                _cnt_start[_key[0]] = _cnt_start.get(_key[0], 0) + 1
                _cnt_end[_key[1]] = _cnt_end.get(_key[1], 0) + 1
            topk.append(cand_tuple)
        if PAIR_ADD_PRIORITY_EXTRA and first_admitted_key is not None:
            priority_end_idx = first_admitted_key[1]
            extra_key = next((
                (candidate[2]['idx'], candidate[3]['idx'])
                for candidate in pairs
                if candidate[3]['idx'] == priority_end_idx
                and (candidate[2]['idx'], candidate[3]['idx']) not in _admitted
            ), None)
            if extra_key is not None:
                _admitted.add(extra_key)
                topk.extend(
                    candidate for candidate in pairs
                    if (candidate[2]['idx'], candidate[3]['idx']) == extra_key)
        topk_pair_keys = list(dict.fromkeys(
            (item[3]['idx'], item[2]['idx']) for item in topk))
        primary_pair_key = topk_pair_keys[0]
        add_pair_detail(
            'candidate_topk_budget', time.perf_counter() - _detail_start)
        candidate_elapsed = time.perf_counter() - _phase_start
        pair_search_timing['candidate_enumeration'] += candidate_elapsed
        candidate_children = sum(pair_search_detail_timing[key] for key in (
            'candidate_branch_pack', 'candidate_pair_branch_score',
            'candidate_topk_budget'))
        pair_search_detail_timing['candidate_overhead'] = max(
            0.0, pair_search_timing['candidate_enumeration'] - candidate_children)
        _phase_start = time.perf_counter()
        primary_feature_indices = list(dict.fromkeys(primary_pair_key))
        _detail_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            list(executor.map(get_frame_features, primary_feature_indices))
        add_pair_detail(
            'sift_feature_extract', time.perf_counter() - _detail_start)
        _detail_start = time.perf_counter()
        primary_matches = get_pair_matches(*primary_pair_key)
        add_pair_detail(
            'sift_descriptor_match', time.perf_counter() - _detail_start)
        _detail_start = time.perf_counter()
        primary_geometry = estimate_feature_geometry(*primary_pair_key)
        add_pair_detail(
            'sift_essential_geometry', time.perf_counter() - _detail_start)

        use_second_pair = primary_geometry is None or not primary_geometry['quality_ok']
        primary_marker_epi = float('inf')
        primary_rotation_delta = float('inf')
        if primary_matches is not None and primary_geometry is not None:
            primary_seed_mask = np.asarray(
                primary_geometry['inlier_mask'], dtype=bool).reshape(-1)
            for candidate in topk:
                candidate_key = (candidate[3]['idx'], candidate[2]['idx'])
                if candidate_key != primary_pair_key:
                    continue
                residuals = rt_epipolar_residuals(
                    primary_matches[0], primary_matches[1],
                    candidate[4], candidate[5])
                if len(primary_seed_mask) == len(residuals) and np.any(primary_seed_mask):
                    residuals = residuals[primary_seed_mask]
                if len(residuals):
                    primary_marker_epi = min(
                        primary_marker_epi, float(np.median(residuals)))
                primary_rotation_delta = min(
                    primary_rotation_delta,
                    rot_angle_deg(primary_geometry['R'], candidate[4]))
            use_second_pair = use_second_pair or (
                primary_marker_epi > PAIR_SECOND_MARKER_EPI_TRIGGER_PX
                or primary_rotation_delta > PAIR_SECOND_ROT_TRIGGER_DEG
                or primary_geometry['parallax_deg'] < PAIR_SECOND_PARALLAX_TRIGGER_DEG)

        if len(topk_pair_keys) > 1 and not use_second_pair:
            log_and_print(
                "   [SIFT預算] 第一候選幾何一致，略過第二候選 "
                f"(marker_epi={primary_marker_epi:.3f}px, "
                f"rotation_delta={primary_rotation_delta:.3f}deg)")
            topk_pair_keys = topk_pair_keys[:1]
            active_pair_keys = set(topk_pair_keys)
            topk = [
                candidate for candidate in topk
                if (candidate[3]['idx'], candidate[2]['idx']) in active_pair_keys
            ]
        topk_feature_indices = list(dict.fromkeys(
            idx for key in topk_pair_keys for idx in key))
        _detail_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            list(executor.map(get_frame_features, topk_feature_indices))
        add_pair_detail(
            'sift_feature_extract', time.perf_counter() - _detail_start)
        _detail_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            list(executor.map(lambda key: get_pair_matches(*key), topk_pair_keys))
        add_pair_detail(
            'sift_descriptor_match', time.perf_counter() - _detail_start)

        _rerank_start = time.perf_counter()
        _rerank_essential_elapsed = 0.0
        reranked = []
        for cand_tuple in topk:
            pair_score, err, item_s, item_e, R_rel_c, t_rel_c, bsl, pair_metrics = cand_tuple
            matches_lr = get_pair_matches(item_e['idx'], item_s['idx'])
            _detail_start = time.perf_counter()
            geometry = estimate_feature_geometry(item_e['idx'], item_s['idx'])
            _essential_elapsed = time.perf_counter() - _detail_start
            _rerank_essential_elapsed += _essential_elapsed
            add_pair_detail('sift_essential_geometry', _essential_elapsed)
            corners_left_u = undistort_corners_dict(item_e['corners'])
            corners_right_u = undistort_corners_dict(item_s['corners'])
            marker_ok = bool(pair_metrics.get('marker_direct_ok', False))
            if matches_lr is not None and geometry is not None:
                feature_residuals = rt_epipolar_residuals(
                    matches_lr[0], matches_lr[1], R_rel_c, t_rel_c)
                seed_mask = np.asarray(geometry['inlier_mask'], dtype=bool).reshape(-1)
                if len(seed_mask) == len(feature_residuals) and np.any(seed_mask):
                    seed_residuals = feature_residuals[seed_mask]
                else:
                    seed_residuals = feature_residuals
                epi_med = float(np.median(seed_residuals))
                epi_p90 = float(np.percentile(seed_residuals, 90))
                rot_agreement = rot_angle_deg(geometry['R'], R_rel_c)
                candidate_parallax = _unified_feature_parallax_stats(
                    matches_lr[0], matches_lr[1], R_rel_c, K_L, mask=seed_mask)
                if pattern_guided_enabled:
                    post_angle_penalty = _pattern_guided_soft_band_penalty(
                        candidate_parallax['median_deg'],
                        pg_cfg['triangulation_sweet_deg'], pg_cfg['triangulation_outer_deg'])
                    post_angle_penalty += 0.5 * _pattern_guided_lower_bound_penalty(
                        candidate_parallax['p10_deg'],
                        pg_cfg['triangulation_p10_target_deg'],
                        pg_cfg['triangulation_p10_outer_deg'])
                else:
                    post_angle_penalty = max(
                        0.0, PAIR_TARGET_TRIANGULATION_ANGLE_DEG - candidate_parallax['p10_deg'])
                    post_angle_penalty /= max(PAIR_TARGET_TRIANGULATION_ANGLE_DEG, 1e-6)
                combined = (
                    0.15 * pair_score
                    + PAIR_SCORE_EPI_W * (
                        0.10 * min(epi_med / 4.0, 2.0)
                        + 0.05 * min(epi_p90 / 4.0, 2.0)
                        + 0.45 * min(geometry['model_epi_px'], 2.0)
                        + geometry['quality_penalty']
                        + 0.30 * min(rot_agreement / FEATURE_ROT_DIFF_MAX_DEG, 3.0)
                        + 0.20 * post_angle_penalty)
                    + (0.0 if marker_ok else 4.0))
                pair_metrics['feat_epi_px'] = float(epi_med)
                pair_metrics['feat_epi_p90_px'] = float(epi_p90)
                pair_metrics['feature_model_epi_px'] = geometry['model_epi_px']
                pair_metrics['feature_quality_ok'] = geometry['quality_ok']
                pair_metrics['feature_strong'] = geometry['strong']
                pair_metrics['feature_inliers'] = geometry['inlier_count']
                pair_metrics['feature_matches'] = geometry['match_count']
                pair_metrics['feature_inlier_ratio'] = geometry['inlier_ratio']
                pair_metrics['feature_grid_coverage'] = geometry['grid_coverage']
                pair_metrics['feature_hull_coverage'] = geometry['hull_coverage']
                pair_metrics['feature_parallax_deg'] = candidate_parallax['median_deg']
                pair_metrics['feature_parallax_p10_deg'] = candidate_parallax['p10_deg']
                pair_metrics['feature_model_parallax_deg'] = geometry['parallax_deg']
                pair_metrics['feature_planar_degenerate'] = geometry['planar_degenerate']
                pair_metrics['feature_rot_agreement_deg'] = float(rot_agreement)
            else:
                combined = 0.20 * pair_score + PAIR_SCORE_EPI_W * 3.0 + (0.0 if marker_ok else 4.0)
                pair_metrics['feat_epi_px'] = None
                pair_metrics['feat_epi_p90_px'] = None
                pair_metrics['feature_model_epi_px'] = None
                pair_metrics['feature_quality_ok'] = False
            temporal_indices = pair_metrics.get('temporal_candidate_indices')
            if temporal_indices is not None:
                joint = _temporal_joint_pair_min_marginal(
                    temporal_dp_model,
                    cand_tuple[2]['idx'], temporal_indices[0],
                    cand_tuple[3]['idx'], temporal_indices[1])
            else:
                joint = None
            pair_metrics['temporal_joint_min_marginal'] = joint
            pair_metrics['temporal_branch_prior_cost'] = (
                0.0 if joint is None else float(joint['prior_cost']))
            combined += pair_metrics['temporal_branch_prior_cost']
            pair_metrics['combined_score'] = float(combined)
            reranked.append((combined, cand_tuple))
        def effective_rerank_score(entry):
            combined, candidate = entry
            candidate_key = (candidate[3]['idx'], candidate[2]['idx'])
            primary_bonus = (
                PAIR_RERANK_MIN_IMPROVEMENT
                if candidate_key == primary_pair_key else 0.0)
            return combined - primary_bonus

        reranked.sort(key=lambda entry: (
            not entry[1][7].get('marker_bidir_ok', False),
            not entry[1][7].get('feature_quality_ok', False),
            effective_rerank_score(entry)))
        if reranked:
            best_combined = effective_rerank_score(reranked[0])
            near_count = sum(
                1 for entry in reranked
                if effective_rerank_score(entry) <= best_combined + 0.01)
            reranked[:near_count] = sorted(
                reranked[:near_count],
                key=lambda entry: (
                    entry[1][7].get('marker_bidir_rms_px', float('inf')),
                    effective_rerank_score(entry)))
        for _c, _t in reranked:
            _fe_t = _t[7].get('feat_epi_px')
            _model_t = _t[7].get('feature_model_epi_px')
            _marker_bidir_t = _t[7].get('marker_bidir_rms_px')
            _marker_bidir_str = f"{_marker_bidir_t:.3f}px" if _marker_bidir_t is not None else "N/A"
            log_and_print(
                f"   [topK] A=F{_t[2]['idx']} B=F{_t[3]['idx']} branch={_t[7].get('branch')} "
                f"marker_bidir={_marker_bidir_str} "
                f"feature_inlier_epi={f'{_fe_t:.3f}px' if _fe_t is not None else 'N/A'} "
                f"E_epi={f'{_model_t:.3f}px' if _model_t is not None else 'N/A'} "
                f"inliers={_t[7].get('feature_inliers', 0)}/{_t[7].get('feature_matches', 0)} "
                f"grid={_t[7].get('feature_grid_coverage', 0.0):.2f} "
                f"parallax={_t[7].get('feature_parallax_deg', 0.0):.3f}° "
                f"score={_t[0]:.3f} combined={_c:.3f}"
            )
        _topk_ids = {id(t) for t in topk}
        pairs = [t for _c, t in reranked] + [t for t in pairs if id(t) not in _topk_ids]

        best_cand = pairs[0]
        best_score = best_cand[0]
        best_err = best_cand[1]
        best_metrics = best_cand[7]
        feat_epi = best_metrics.get('feat_epi_px')
        feature_model_epi = best_metrics.get('feature_model_epi_px')
        best_marker_bidir = best_metrics.get('marker_bidir_rms_px')
        best_marker_bidir_str = f"{best_marker_bidir:.3f}px" if best_marker_bidir is not None else "N/A"
        log_and_print(
            f"🎯 [pair quality] score={best_score:.3f} | reproj={best_err:.3f}px | "
            f"marker_bidir={best_marker_bidir_str} | "
            f"feature_inlier_epi={f'{feat_epi:.3f}px' if feat_epi is not None else 'N/A'} | "
            f"E_epi={f'{feature_model_epi:.3f}px' if feature_model_epi is not None else 'N/A'} | "
            f"baseline={best_metrics['baseline']:.2f}mm | shared={best_metrics['shared_markers']} | "
            f"marker_parallax={best_metrics.get('marker_parallax_deg', 0.0):.2f}deg | "
            f"predicted_depth_sigma={best_metrics.get('predicted_depth_sigma_mm', float('inf')):.2f}mm | "
            f"feature_grid={best_metrics.get('feature_grid_coverage', 0.0):.2f} | "
            f"parallax={best_metrics.get('feature_parallax_deg', 0.0):.3f}°"
        )

        fe_str = f"{feature_model_epi:.3f}" if feature_model_epi is not None else "N/A"
        stage_ok = (
            best_metrics.get('marker_bidir_ok', False)
            and best_metrics.get('feature_quality_ok', False)
            and feature_model_epi is not None
            and feature_model_epi < PAIR_EPI_OK_PX)
        if stage_ok:
            best_start, best_end, R_rel, t_rel, baseline = best_cand[2], best_cand[3], best_cand[4], best_cand[5], best_cand[6]
            best_branch = best_cand[7].get('branch')
            _branch_note = f" | hypothesis {best_branch} | reproj {best_err:.3f} px"
            log_and_print(f"🎉 第 {stage_idx + 1} 階段搜尋成功！特徵極線殘差 {fe_str} px < {PAIR_EPI_OK_PX} px{_branch_note}")
            stage_success = True
        else:
            log_and_print(
                f"ℹ️ 快速橋接幀的特徵極線殘差為 {fe_str} px "
                f"(前置門檻 {PAIR_EPI_OK_PX} px)，交由聯合精修與最終品質檢查。")

        # The bounded search always forwards its best candidate to joint refinement.
        if stage_success or stage_idx == len(stages) - 1:
            if not stage_success:
                _validated = [
                    t for t in pairs
                    if t[7].get('marker_bidir_ok', False)
                    and t[7].get('feature_quality_ok', False)]
                if _validated:
                    best_cand = min(_validated, key=lambda t: t[7].get('combined_score', float('inf')))
                best_start, best_end, R_rel, t_rel, baseline = best_cand[2], best_cand[3], best_cand[4], best_cand[5], best_cand[6]
                best_branch = best_cand[7].get('branch')
                _fe_fb = best_cand[7].get('feature_model_epi_px')
                log_and_print(
                    "⚠️ 快速橋接幀未達前置門檻，仍保留當前最佳候選 "
                    f"(feat_epi {f'{_fe_fb:.3f}' if _fe_fb is not None else 'N/A'} px, "
                    f"reproj {best_cand[1]:.3f} px)，後續由品質旗標決定是否可信。")
                
            # 次佳對選取 (全模式統一)：只收已通過特徵極線驗證 (< PAIR_EPI_EXTRA_PX) 的候選，
            # 同一結尾幀、不同起始幀，每個起始幀只取排序最前 (最佳) 的一組
            candidates_scores = []
            _seen_extra_idx = set()
            for pair_score, err, item_s, item_e, R_rel_c, t_rel_c, bsl, pair_metrics in pairs:
                if item_e['idx'] != best_end['idx'] or item_s['idx'] == best_start['idx']:
                    continue
                if item_s['idx'] in _seen_extra_idx:
                    continue
                fe = pair_metrics.get('feature_model_epi_px')
                if (not pair_metrics.get('marker_bidir_ok', False)
                        or not pair_metrics.get('feature_quality_ok', False)
                        or fe is None or fe >= PAIR_EPI_EXTRA_PX):
                    continue
                _seen_extra_idx.add(item_s['idx'])
                candidates_scores.append((pair_score, err, item_s, R_rel_c, t_rel_c, bsl, pair_metrics))

            selected_extras = candidates_scores[:5]
            rerank_elapsed = time.perf_counter() - _rerank_start
            add_pair_detail(
                'sift_rerank_and_select',
                max(0.0, rerank_elapsed - _rerank_essential_elapsed))
            sift_elapsed = time.perf_counter() - _phase_start
            pair_search_timing['sift_essential_rerank'] += sift_elapsed
            sift_children = sum(pair_search_detail_timing[key] for key in (
                'sift_feature_extract', 'sift_descriptor_match',
                'sift_essential_geometry', 'sift_rerank_and_select'))
            pair_search_detail_timing['sift_overhead'] = max(
                0.0, pair_search_timing['sift_essential_rerank'] - sift_children)
            break

    timer.stage("ArUco偵測+配對搜尋(含極線重排)")
    if best_start is None or best_end is None:
        log_and_print("❌ [漸進式匹配] 無法在該影片中計算出任何影像對，分析失敗。")
        return None
    best_pair_metrics = dict(best_cand[7])

    # Always materialize the selected per-frame hypothesis before continuous refinement.
    if best_branch is not None and 'branches' in best_start and 'branches' in best_end:
        _bi_s, _bi_e = best_branch
        best_start['R'], best_start['t'] = best_start['branches'][_bi_s]
        best_end['R'], best_end['t'] = best_end['branches'][_bi_e]
        best_start['measurement_mode'] = best_start['branch_candidates'][_bi_s].get('measurement_mode', 'UNKNOWN')
        best_end['measurement_mode'] = best_end['branch_candidates'][_bi_e].get('measurement_mode', 'UNKNOWN')
        log_and_print(
            f"ℹ️ [逐幀 hypothesis] 採用 (s={_bi_s}:{best_start['measurement_mode']}, "
            f"e={_bi_e}:{best_end['measurement_mode']})")

    if SAVE_DEBUG_PAIR_IMAGES:
        save_debug_pair_images(best_start, best_end, "best")

    cornersA_undist = undistort_corners_dict(best_start['corners'])
    cornersB_undist = undistort_corners_dict(best_end['corners'])

    (R_A_refined, t_A_refined, R_B_refined, t_B_refined,
     R_rel, t_rel, best_feature_rt_applied, best_joint_metrics) = refine_world_endpoint_pair(
        best_end['idx'], best_start['idx'],
        best_start['R'], best_start['t'], best_end['R'], best_end['t'],
        best_start['corners'], best_end['corners'], tag="-best")
    best_start['R'], best_start['t'] = R_A_refined, t_A_refined
    best_end['R'], best_end['t'] = R_B_refined, t_B_refined
    baseline = float(np.linalg.norm(t_rel))

    # 包裝次優額外右圖組 (同樣做混合 RT 精修)
    extra_candidates_info = []
    for pair_score, err, item_s, R_rel_c, t_rel_c, bsl, pair_metrics in selected_extras:
        cornersA_e_undist = undistort_corners_dict(item_s['corners'])
        _bi_s, _bi_e = pair_metrics.get('branch', (0, 0))
        R_A_seed, t_A_seed = item_s['branches'][_bi_s]
        R_B_seed, t_B_seed = best_end['branches'][_bi_e]
        (R_A_extra, t_A_extra, R_B_extra, t_B_extra,
         R_rel_c, t_rel_c, _extra_feature_rt_applied, _extra_joint_metrics) = refine_world_endpoint_pair(
            best_end['idx'], item_s['idx'], R_A_seed, t_A_seed, R_B_seed, t_B_seed,
            item_s['corners'], best_end['corners'], tag=f"-F{item_s['idx']}")
        _extra_matches = get_pair_matches(best_end['idx'], item_s['idx'])
        if _extra_matches is None:
            log_and_print(f"⚠️ [次佳配對] F{item_s['idx']} 無法做最終特徵驗證，已排除。")
            continue
        _extra_feature_stats = _extra_joint_metrics.get('feature') or {}
        _extra_final_epi = _extra_feature_stats.get(
            'inlier_median_px', rt_epipolar_residual(
                _extra_matches[0], _extra_matches[1], R_rel_c, t_rel_c))
        if (not _extra_joint_metrics.get('marker_bidir_ok', False)
                or not _extra_joint_metrics.get('feature_ok', False)
                or _extra_final_epi >= PAIR_EPI_EXTRA_PX):
            log_and_print(
                f"⚠️ [次佳配對] F{item_s['idx']} 未同時通過 marker 雙向投影與 Feature 內點驗證 "
                f"(feature={_extra_final_epi:.3f}px)，已排除，不參與深度融合。")
            continue
        pair_metrics['final_feature_epi_px'] = float(_extra_final_epi)
        pair_metrics['joint_refine'] = _extra_joint_metrics
        extra_candidates_info.append({
            'idx_A': item_s['idx'],
            'item_start': item_s,
            'frame_A': frames[item_s['idx']],
            'R_rel': R_rel_c,
            't_rel': t_rel_c,
            'baseline': float(np.linalg.norm(t_rel_c)),
            'pair_score': pair_score,
            'pair_metrics': pair_metrics,
            'joint_refine': _extra_joint_metrics,
            'feature_rt_applied': bool(_extra_feature_rt_applied),
            'cornersA': cornersA_e_undist,
            'R_A_abs': R_A_extra, 't_A_abs': t_A_extra,
            'R_B_abs': R_B_extra, 't_B_abs': t_B_extra,
        })
        log_and_print(f"➕ [次佳配對] 額外右圖 (Frame A) 索引: {item_s['idx']} | 重投影誤差: {err:.3f} px | Baseline: {np.linalg.norm(t_rel_c):.2f} mm")

    primary_candidate = {
        'idx_A': best_start['idx'],
        'item_start': best_start,
        'frame_A': frames[best_start['idx']],
        'R_rel': R_rel,
        't_rel': t_rel,
        'baseline': baseline,
        'pair_score': best_pair_metrics.get('score', float('inf')),
        'pair_metrics': best_pair_metrics,
        'joint_refine': best_joint_metrics,
        'feature_rt_applied': bool(best_feature_rt_applied),
        'cornersA': cornersA_undist,
        'R_A_abs': R_A_refined, 't_A_abs': t_A_refined,
        'R_B_abs': R_B_refined, 't_B_abs': t_B_refined,
    }
    valid_final_candidates = [
        candidate for candidate in [primary_candidate] + extra_candidates_info
        if candidate['joint_refine'].get('marker_bidir_ok', False)
        and candidate['joint_refine'].get('feature_ok', False)]
    if valid_final_candidates:
        best_final_marker_rms = min(
            candidate['joint_refine']['marker_bidir']['rms_px']
            for candidate in valid_final_candidates)
        marker_priority_candidates = [
            candidate for candidate in valid_final_candidates
            if candidate['joint_refine']['marker_bidir']['rms_px']
            <= best_final_marker_rms + FINAL_PAIR_MARKER_RMS_BAND_PX]

        def final_candidate_rank(candidate):
            feature_stats = candidate['joint_refine'].get('feature') or {}
            marker_stats = candidate['joint_refine'].get('marker_bidir') or {}
            return (
                feature_stats.get('inlier_p90_px', float('inf')),
                feature_stats.get('inlier_median_px', float('inf')),
                -feature_stats.get('inlier_count', 0),
                marker_stats.get('rms_px', float('inf')),
            )

        best_feature_p90 = min(
            (candidate['joint_refine'].get('feature') or {}).get(
                'inlier_p90_px', float('inf'))
            for candidate in marker_priority_candidates)
        near_feature_candidates = [
            candidate for candidate in marker_priority_candidates
            if (candidate['joint_refine'].get('feature') or {}).get(
                'inlier_p90_px', float('inf')) <= best_feature_p90 + 0.15]
        support_sorted = sorted(
            near_feature_candidates,
            key=lambda candidate: (candidate['joint_refine'].get('feature') or {}).get(
                'inlier_count', 0),
            reverse=True)
        if (len(support_sorted) >= 2
                and (support_sorted[0]['joint_refine'].get('feature') or {}).get(
                    'inlier_count', 0)
                >= 1.35 * max(
                    (support_sorted[1]['joint_refine'].get('feature') or {}).get(
                        'inlier_count', 0), 1)):
            selected_final = support_sorted[0]
        else:
            selected_final = min(
                near_feature_candidates,
                key=lambda candidate: (
                    candidate.get('pair_metrics', {}).get('predicted_depth_sigma_mm', float('inf')),
                    -candidate.get('pair_metrics', {}).get('triangulation_p10_deg', 0.0),
                    -candidate.get('pair_metrics', {}).get('overlap_ratio', 0.0),
                    PAIR_SCORE_IDEAL_BASELINE_TIE_W * abs(
                        candidate['baseline'] - IDEAL_BASELINE_MM) / max(IDEAL_BASELINE_MM, 1e-6),
                    final_candidate_rank(candidate)))
        if selected_final is not primary_candidate:
            old_primary = primary_candidate
            best_start = selected_final['item_start']
            best_start['R'], best_start['t'] = selected_final['R_A_abs'], selected_final['t_A_abs']
            best_end['R'], best_end['t'] = selected_final['R_B_abs'], selected_final['t_B_abs']
            R_rel = selected_final['R_rel']
            t_rel = selected_final['t_rel']
            baseline = selected_final['baseline']
            cornersA_undist = selected_final['cornersA']
            best_pair_metrics = dict(selected_final['pair_metrics'])
            best_joint_metrics = selected_final['joint_refine']
            best_feature_rt_applied = selected_final['feature_rt_applied']
            extra_candidates_info = [
                candidate for candidate in extra_candidates_info
                if candidate is not selected_final]
            if (old_primary['joint_refine'].get('marker_bidir_ok', False)
                    and old_primary['joint_refine'].get('feature_ok', False)):
                extra_candidates_info.append(old_primary)
            if SAVE_DEBUG_PAIR_IMAGES:
                save_debug_pair_images(best_start, best_end, "best_joint_promoted")
            log_and_print(
                f"🔁 [最終配對升格] F{selected_final['idx_A']} 在聯合精修後同時通過 "
                f"marker 與 Feature，且雙重約束品質優於精修前排名第一的影像對。")

    # Keep exactly the Feature points used by the final joint solution. If Feature was
    # validation-only, show the recoverPose inliers but label their role accordingly.
    rt_sift_points_left = np.empty((0, 2), dtype=np.float64)
    rt_sift_points_right = np.empty((0, 2), dtype=np.float64)
    best_feature_matches = get_pair_matches(best_end['idx'], best_start['idx'])
    best_feature_geometry = estimate_feature_geometry(best_end['idx'], best_start['idx'])
    if best_feature_matches is not None and best_feature_geometry is not None:
        inlier_mask = np.asarray(best_feature_geometry['inlier_mask'], dtype=bool).reshape(-1)
        optimization_mask = best_joint_metrics.get('optimization_mask')
        if best_feature_rt_applied and optimization_mask is not None:
            optimization_mask = np.asarray(optimization_mask, dtype=bool).reshape(-1)
            if len(optimization_mask) == len(inlier_mask):
                inlier_mask = optimization_mask
        if len(inlier_mask) == len(best_feature_matches[0]):
            rt_sift_points_left = np.asarray(best_feature_matches[0][inlier_mask], dtype=np.float64)
            rt_sift_points_right = np.asarray(best_feature_matches[1][inlier_mask], dtype=np.float64)
    rt_sift_role = "final_rt" if best_feature_rt_applied else "validation_only"
    log_and_print(
        f"📍 [RT SIFT像素] {len(rt_sift_points_left)}/"
        f"{0 if best_feature_matches is None else len(best_feature_matches[0])} | role={rt_sift_role}")

    if not extra_candidates_info:
        log_and_print(f"ℹ️ [次佳配對] 未找到通過特徵幾何驗證的額外影格 (E 極線門檻 {PAIR_EPI_EXTRA_PX} px, Baseline {MIN_BASELINE_MM}~{MAX_BASELINE_MM} mm)。")

    timer.stage("混合RT精修+次佳打包")
    log_and_print(f"✅ 挑選結果：")
    if angle_guided_diagnostics.get('output_roles_swapped', False):
        log_and_print(f"  - 右圖 (Frame A/15deg) 索引: {best_end['idx']}")
        log_and_print(f"  - 左圖 (Frame B/35deg) 索引: {best_start['idx']}")
        log_and_print(
            f"  - 時序精修順序: F{best_start['idx']} → F{best_end['idx']} "
            "(輸出 RT 將自動反轉)")
    else:
        log_and_print(f"  - 右圖 (Frame A) 索引: {best_start['idx']}")
        log_and_print(f"  - 左圖 (Frame B) 索引: {best_end['idx']}")
    log_and_print(f"  - 計算 Baseline: {baseline:.2f} mm")
    
    # Return the complete temporally selected sparse trajectory, not only the two
    # winning endpoints.  Downstream multi-frame/KLT verification can therefore
    # never silently fall back to IPPE branch zero on the other probe frames.
    valid_poses = dict(temporal_valid_poses)
    for item in valid_start + valid_end:
        valid_poses[item['idx']] = (item['R'], item['t'])
    # Joint marker/SIFT refinement updates a relative pose, while the temporal
    # path stores absolute T_camera<-reference poses.  Keep the selected end
    # pose as the world anchor and recompose every returned refined start pose,
    # so valid_poses and R_rel/t_rel remain exactly consistent downstream.
    anchor_end_R = np.asarray(best_end['R'], np.float64).reshape(3, 3)
    anchor_end_t = np.asarray(best_end['t'], np.float64).reshape(3, 1)
    refined_relative_poses = [
        (int(best_start['idx']), R_rel, t_rel),
        *[
            (int(candidate['idx_A']), candidate['R_rel'], candidate['t_rel'])
            for candidate in extra_candidates_info
        ],
    ]
    reanchored_indices = []
    for frame_index, relative_R, relative_t in refined_relative_poses:
        relative_R = np.asarray(relative_R, np.float64).reshape(3, 3)
        relative_t = np.asarray(relative_t, np.float64).reshape(3, 1)
        valid_poses[frame_index] = (
            relative_R @ anchor_end_R,
            relative_t + relative_R @ anchor_end_t,
        )
        reanchored_indices.append(frame_index)
    temporal_diagnostics['final_rt_reanchored_indices'] = sorted(
        set(reanchored_indices))
    _R_A_closed, _t_A_closed = valid_poses[int(best_start['idx'])]
    _closure_R = _temporal_rotation_distance_deg(_R_A_closed, R_rel @ anchor_end_R)
    _closure_t = float(np.linalg.norm(
        np.asarray(_t_A_closed).reshape(3, 1) - (t_rel + R_rel @ anchor_end_t)))
    temporal_diagnostics['final_rt_closure_rotation_deg'] = float(_closure_R)
    temporal_diagnostics['final_rt_closure_translation_mm'] = float(_closure_t)

    marker_pnp_self_reproj_err = None
    if best_start is not None and best_end is not None:
        marker_pnp_self_reproj_err = compute_pair_reprojection_error(best_start, best_end, mtx_L, dist_L)
    marker_direct_final = _unified_pair_reprojection_stats(
        best_start['R'], best_start['t'], best_start['corners'],
        best_end['R'], best_end['t'], best_end['corners'], marker_map,
        mtx_L, dist_L, marker_size_mm)
    marker_bidir_stats = best_joint_metrics.get('marker_bidir')
    marker_reproj_err = (
        marker_direct_final['rms_px'] if marker_direct_final is not None else None)
    final_feature_stats = best_joint_metrics.get('feature')
    _m_final = get_pair_matches(best_end['idx'], best_start['idx'])
    if _m_final is not None and final_feature_stats is None:
        _seed_mask = None
        if best_feature_geometry is not None:
            _seed_value = np.asarray(best_feature_geometry.get('inlier_mask', []), dtype=bool).reshape(-1)
            if len(_seed_value) == len(_m_final[0]):
                _seed_mask = _seed_value
        final_feature_stats = feature_pose_stats(
            _m_final[0], _m_final[1], R_rel, t_rel, _seed_mask)
    final_feature_epi = (
        final_feature_stats.get('inlier_median_px')
        if final_feature_stats is not None else None)
    quality_components = [value for value in (marker_reproj_err, final_feature_epi)
                          if value is not None and np.isfinite(value)]
    best_reproj_err = max(quality_components) if quality_components else None
    if marker_bidir_stats is not None:
        feature_log = (
            f"inliers={final_feature_stats['inlier_count']}/{len(_m_final[0])}, "
            f"median={final_feature_stats['inlier_median_px']:.3f}px, "
            f"p90={final_feature_stats['inlier_p90_px']:.3f}px"
            if final_feature_stats is not None and _m_final is not None else "N/A")
        log_and_print(
            f"ℹ️ [RT品質] marker 雙向重投影 L→R={marker_bidir_stats['left_to_right_rms_px']:.3f}px, "
            f"R→L={marker_bidir_stats['right_to_left_rms_px']:.3f}px, "
            f"max={marker_bidir_stats['max_px']:.3f}px | Feature {feature_log}")
    if final_feature_epi is not None:
        best_pair_metrics['final_feature_epi_px'] = float(final_feature_epi)
    best_pair_metrics['final_feature_stats'] = final_feature_stats
    best_pair_metrics['marker_bidir'] = marker_bidir_stats
    best_pair_metrics['marker_bidir_ok'] = bool(best_joint_metrics.get('marker_bidir_ok', False))
    best_pair_metrics['marker_reproj_px'] = marker_reproj_err
    best_pair_metrics['marker_pnp_self_reproj_px'] = marker_pnp_self_reproj_err
    best_pair_metrics['feature_final_ok'] = bool(best_joint_metrics.get('feature_ok', False))
    best_pair_metrics['rt_reliable'] = bool(
        best_pair_metrics['marker_bidir_ok']
        and best_pair_metrics['feature_final_ok']
        and best_pair_metrics.get('feature_quality_ok', False))
    best_pair_metrics['temporal_pose'] = {
        'status': temporal_diagnostics.get('status'),
        'reference_marker_id': temporal_diagnostics.get('reference_marker_id'),
        'path_margin': (
            temporal_diagnostics.get('path', {}).get('normalized_margin')),
        'observation_count': (
            temporal_diagnostics.get('path', {}).get('observation_count', 0)),
    }
    timer.stage("最終RT閉環+品質驗證")

    def _diagnostic_grid(points):
        grid = np.zeros((FEATURE_GRID_ROWS, FEATURE_GRID_COLS), dtype=np.int32)
        for point in np.asarray(points, dtype=np.float64).reshape(-1, 2):
            cell_x, cell_y = feature_cell(point)
            grid[cell_y, cell_x] += 1
        return grid

    def _write_grid_section(file_obj, title, points):
        file_obj.write(f"{title}\n")
        for row in _diagnostic_grid(points):
            file_obj.write("  " + " ".join(f"{int(value):4d}" for value in row) + "\n")

    # This file is intentionally separate from the general analysis log so it can be
    # attached as a compact, self-contained report when feature RT behaves unexpectedly.
    rt_sift_diagnostics_path = (
        os.path.splitext(video_path)[0] + "_rt_sift_diagnostics.txt"
        if SAVE_RT_SIFT_DIAGNOSTICS else None
    )
    try:
        if rt_sift_diagnostics_path is None:
            raise RuntimeError("RT SIFT diagnostics disabled")
        kp_diag_left, _des_diag_left = get_frame_features(best_end['idx'])
        kp_diag_right, _des_diag_right = get_frame_features(best_start['idx'])
        raw_keypoints_left = np.asarray(
            [feature_point_fullres(kp) for kp in kp_diag_left],
            dtype=np.float64).reshape(-1, 2)
        raw_keypoints_right = np.asarray(
            [feature_point_fullres(kp) for kp in kp_diag_right],
            dtype=np.float64).reshape(-1, 2)
        if best_feature_matches is None:
            candidate_points_left = np.empty((0, 2), dtype=np.float64)
            candidate_points_right = np.empty((0, 2), dtype=np.float64)
        else:
            candidate_points_left = np.asarray(best_feature_matches[0], dtype=np.float64).reshape(-1, 2)
            candidate_points_right = np.asarray(best_feature_matches[1], dtype=np.float64).reshape(-1, 2)

        candidate_count = min(len(candidate_points_left), len(candidate_points_right))
        candidate_points_left = candidate_points_left[:candidate_count]
        candidate_points_right = candidate_points_right[:candidate_count]
        essential_mask = np.zeros(candidate_count, dtype=bool)
        recover_mask = np.zeros(candidate_count, dtype=bool)
        optimization_mask = np.zeros(candidate_count, dtype=bool)
        holdout_mask = np.zeros(candidate_count, dtype=bool)
        final_feature_inlier_mask = np.zeros(candidate_count, dtype=bool)
        final_feature_residuals = np.full(candidate_count, np.nan, dtype=np.float64)
        if best_feature_geometry is not None:
            essential_mask_src = np.asarray(
                best_feature_geometry.get('essential_inlier_mask', []), dtype=bool).reshape(-1)
            recover_mask_src = np.asarray(
                best_feature_geometry.get('inlier_mask', []), dtype=bool).reshape(-1)
            if len(essential_mask_src) == candidate_count:
                essential_mask = essential_mask_src
            if len(recover_mask_src) == candidate_count:
                recover_mask = recover_mask_src
        optimization_mask_src = best_joint_metrics.get('optimization_mask')
        holdout_mask_src = best_joint_metrics.get('holdout_mask')
        if optimization_mask_src is not None and len(optimization_mask_src) == candidate_count:
            optimization_mask = np.asarray(optimization_mask_src, dtype=bool).reshape(-1)
        if holdout_mask_src is not None and len(holdout_mask_src) == candidate_count:
            holdout_mask = np.asarray(holdout_mask_src, dtype=bool).reshape(-1)
        if final_feature_stats is not None:
            final_mask_src = np.asarray(
                final_feature_stats.get('final_inlier_mask', []), dtype=bool).reshape(-1)
            final_residual_src = np.asarray(
                final_feature_stats.get('residuals_px', []), dtype=np.float64).reshape(-1)
            if len(final_mask_src) == candidate_count:
                final_feature_inlier_mask = final_mask_src
            if len(final_residual_src) == candidate_count:
                final_feature_residuals = final_residual_src

        match_diagnostics = match_diagnostics_cache.get(
            (best_end['idx'], best_start['idx']), {})
        geometry_diagnostics = best_feature_geometry or {}
        with open(rt_sift_diagnostics_path, 'w', encoding='utf-8') as diag_file:
            diag_file.write("=== RT SIFT DIAGNOSTICS ===\n")
            diag_file.write("This report describes the selected best frame pair only.\n")
            diag_file.write("Coordinates are undistorted display pixels: frame_B/UI-left -> frame_A/UI-right.\n\n")

            diag_file.write("[SELECTED_PAIR]\n")
            diag_file.write(f"video={video_path}\n")
            diag_file.write(f"frame_left_B={best_end['idx']}\n")
            diag_file.write(f"frame_right_A={best_start['idx']}\n")
            diag_file.write(f"shared_markers={best_pair_metrics.get('shared_markers', 0)}\n")
            diag_file.write(f"baseline_mm={baseline:.6f}\n")
            diag_file.write(f"rt_sift_role={rt_sift_role}\n")
            diag_file.write(f"rt_sift_applied={bool(best_feature_rt_applied)}\n")
            diag_file.write(f"rt_reliable={bool(best_pair_metrics.get('rt_reliable', False))}\n\n")

            diag_file.write("[PIPELINE_COUNTS]\n")
            for field in (
                    'left_keypoint_count', 'right_keypoint_count',
                    'left_descriptor_count', 'right_descriptor_count',
                    'knn_pair_count', 'ratio_pass_count', 'mutual_pass_count',
                    'spatially_balanced_count'):
                diag_file.write(f"{field}={int(match_diagnostics.get(field, 0))}\n")
            diag_file.write(f"essential_ransac_inlier_count={int(np.count_nonzero(essential_mask))}\n")
            diag_file.write(f"recoverpose_inlier_count={int(np.count_nonzero(recover_mask))}\n")
            diag_file.write(
                f"final_rt_sift_point_count="
                f"{int(np.count_nonzero(optimization_mask)) if best_feature_rt_applied else 0}\n")
            diag_file.write(f"final_rt_feature_inlier_count={int(np.count_nonzero(final_feature_inlier_mask))}\n")
            diag_file.write(f"final_rt_feature_outlier_count={candidate_count - int(np.count_nonzero(final_feature_inlier_mask))}\n\n")

            diag_file.write("[FEATURE_QUALITY]\n")
            quality_fields = (
                'match_count', 'essential_inlier_count', 'inlier_count', 'inlier_ratio',
                'grid_coverage', 'hull_coverage', 'parallax_deg', 'homography_ratio',
                'planar_degenerate', 'model_epi_px', 'quality_penalty', 'quality_ok', 'strong')
            for field in quality_fields:
                diag_file.write(f"{field}={geometry_diagnostics.get(field, 'N/A')}\n")
            diag_file.write(f"marker_rt_feature_epi_px={best_pair_metrics.get('feat_epi_px', 'N/A')}\n")
            diag_file.write(f"final_feature_epi_px={best_pair_metrics.get('final_feature_epi_px', 'N/A')}\n")
            diag_file.write(f"final_feature_inlier_p90_px={final_feature_stats.get('inlier_p90_px', 'N/A') if final_feature_stats else 'N/A'}\n")
            diag_file.write(f"final_feature_all_p90_px={final_feature_stats.get('all_p90_px', 'N/A') if final_feature_stats else 'N/A'}\n")
            diag_file.write(f"final_feature_holdout_median_px={final_feature_stats.get('holdout_median_px', 'N/A') if final_feature_stats else 'N/A'}\n")
            diag_file.write(f"marker_bidir_rms_px={marker_reproj_err if marker_reproj_err is not None else 'N/A'}\n")
            diag_file.write(f"marker_left_to_right_rms_px={marker_bidir_stats.get('left_to_right_rms_px', 'N/A') if marker_bidir_stats else 'N/A'}\n")
            diag_file.write(f"marker_right_to_left_rms_px={marker_bidir_stats.get('right_to_left_rms_px', 'N/A') if marker_bidir_stats else 'N/A'}\n")
            diag_file.write(f"marker_bidir_max_px={marker_bidir_stats.get('max_px', 'N/A') if marker_bidir_stats else 'N/A'}\n")
            diag_file.write(f"marker_pnp_self_reproj_px={marker_pnp_self_reproj_err if marker_pnp_self_reproj_err is not None else 'N/A'}\n")
            diag_file.write(f"joint_solution_role={best_joint_metrics.get('solution_role', 'N/A')}\n")
            diag_file.write(f"feature_rot_agreement_deg={best_pair_metrics.get('feature_rot_agreement_deg', 'N/A')}\n")
            diag_file.write(f"pair_score={best_pair_metrics.get('score', 'N/A')}\n")
            diag_file.write(f"combined_score={best_pair_metrics.get('combined_score', 'N/A')}\n")
            diag_file.write(f"sharpness_min={best_pair_metrics.get('sharpness_min', 'N/A')}\n")
            diag_file.write(f"marker_coverage={best_pair_metrics.get('coverage', 'N/A')}\n\n")

            diag_file.write("[ACTIVE_THRESHOLDS]\n")
            diag_file.write(f"sift_max_keypoints={FEATURE_MAX_KEYPOINTS}\n")
            diag_file.write(f"sift_contrast_threshold=0.01\n")
            diag_file.write(f"match_ratio={FEATURE_MATCH_RATIO}\n")
            diag_file.write(f"ransac_threshold_px={FEATURE_E_RANSAC_THRESH_PX}\n")
            diag_file.write(f"min_matches={FEATURE_MIN_MATCHES}\n")
            diag_file.write(f"min_inlier_ratio={FEATURE_MIN_INLIER_RATIO}\n")
            diag_file.write(f"min_grid_coverage={FEATURE_MIN_GRID_COVERAGE}\n")
            diag_file.write(f"min_hull_coverage={FEATURE_MIN_HULL_COVERAGE}\n")
            diag_file.write(f"min_parallax_deg={FEATURE_MIN_PARALLAX_DEG}\n")
            diag_file.write(f"strong_inliers={FEATURE_STRONG_INLIERS}\n")
            diag_file.write(f"strong_grid_coverage={FEATURE_STRONG_GRID_COVERAGE}\n\n")
            diag_file.write(f"final_feature_inlier_px={FEATURE_FINAL_INLIER_PX}\n")
            diag_file.write(f"final_feature_p90_max_px={FEATURE_FINAL_P90_MAX_PX}\n")
            diag_file.write(f"marker_bidir_rms_max_px={MARKER_BIDIR_RMS_MAX_PX}\n")
            diag_file.write(f"marker_bidir_point_max_px={MARKER_BIDIR_MAX_MAX_PX}\n\n")

            diag_file.write("[FINAL_RT_LEFT_TO_RIGHT]\n")
            for row_idx, row in enumerate(np.asarray(R_rel, dtype=np.float64).reshape(3, 3)):
                diag_file.write(f"R{row_idx}=" + " ".join(f"{value:.12g}" for value in row) + "\n")
            diag_file.write(
                "t_mm=" + " ".join(
                    f"{value:.12g}" for value in np.asarray(t_rel, dtype=np.float64).reshape(3)) + "\n\n")

            diag_file.write(f"[GRID_COUNTS_{FEATURE_GRID_COLS}x{FEATURE_GRID_ROWS}]\n")
            _write_grid_section(diag_file, "raw_keypoints_left", raw_keypoints_left)
            _write_grid_section(diag_file, "raw_keypoints_right", raw_keypoints_right)
            _write_grid_section(diag_file, "balanced_candidates_left", candidate_points_left)
            _write_grid_section(diag_file, "balanced_candidates_right", candidate_points_right)
            _write_grid_section(diag_file, "essential_inliers_left", candidate_points_left[essential_mask])
            _write_grid_section(diag_file, "essential_inliers_right", candidate_points_right[essential_mask])
            _write_grid_section(diag_file, "recoverpose_inliers_left", candidate_points_left[recover_mask])
            _write_grid_section(diag_file, "recoverpose_inliers_right", candidate_points_right[recover_mask])
            _write_grid_section(diag_file, "joint_optimization_left", candidate_points_left[optimization_mask])
            _write_grid_section(diag_file, "joint_optimization_right", candidate_points_right[optimization_mask])
            _write_grid_section(diag_file, "holdout_left", candidate_points_left[holdout_mask])
            _write_grid_section(diag_file, "holdout_right", candidate_points_right[holdout_mask])
            _write_grid_section(diag_file, "final_rt_inliers_left", candidate_points_left[final_feature_inlier_mask])
            _write_grid_section(diag_file, "final_rt_inliers_right", candidate_points_right[final_feature_inlier_mask])
            diag_file.write("\n")

            diag_file.write("[MATCH_TABLE]\n")
            diag_file.write(
                "index,left_x,left_y,right_x,right_y,essential_inlier,"
                "recoverpose_inlier,optimization_used,holdout,final_rt_inlier,"
                "final_epipolar_px\n")
            for index, (point_left, point_right) in enumerate(
                    zip(candidate_points_left, candidate_points_right), start=1):
                match_idx = index - 1
                diag_file.write(
                    f"{index},{point_left[0]:.6f},{point_left[1]:.6f},"
                    f"{point_right[0]:.6f},{point_right[1]:.6f},"
                    f"{int(essential_mask[match_idx])},{int(recover_mask[match_idx])},"
                    f"{int(best_feature_rt_applied and optimization_mask[match_idx])},"
                    f"{int(holdout_mask[match_idx])},{int(final_feature_inlier_mask[match_idx])},"
                    f"{final_feature_residuals[match_idx]:.6f}\n")
        log_and_print(f"🧾 [RT SIFT診斷] 已輸出: {rt_sift_diagnostics_path}")
    except Exception as diag_error:
        if rt_sift_diagnostics_path is not None:
            log_and_print(f"⚠️ [RT SIFT診斷] 輸出失敗: {diag_error}")
        rt_sift_diagnostics_path = None
    timer.stage("RT SIFT診斷檔輸出")

    local_window_diagnostics['final_selected_pair'] = {
        'idx_A': int(best_start['idx']), 'idx_B': int(best_end['idx']),
        'offset_A': None if local_window_diagnostics.get('provisional_pair') is None else int(best_start['idx']) - int(local_window_diagnostics['provisional_pair']['idx_A']),
        'offset_B': None if local_window_diagnostics.get('provisional_pair') is None else int(best_end['idx']) - int(local_window_diagnostics['provisional_pair']['idx_B']),
    }
    provisional_pair = local_window_diagnostics.get('provisional_pair') or {}
    endpoint_A_diag = local_window_diagnostics.get('endpoints', {}).get('A', {})
    endpoint_B_diag = local_window_diagnostics.get('endpoints', {}).get('B', {})
    local_best_pair = {
        'idx_A': endpoint_A_diag.get('chosen_index'),
        'idx_B': endpoint_B_diag.get('chosen_index'),
    }
    final_pair = local_window_diagnostics['final_selected_pair']
    local_changed_A = (
        provisional_pair.get('idx_A') is not None
        and local_best_pair['idx_A'] is not None
        and int(provisional_pair['idx_A']) != int(local_best_pair['idx_A']))
    local_changed_B = (
        provisional_pair.get('idx_B') is not None
        and local_best_pair['idx_B'] is not None
        and int(provisional_pair['idx_B']) != int(local_best_pair['idx_B']))
    final_changed_from_local_A = (
        local_best_pair['idx_A'] is not None
        and int(local_best_pair['idx_A']) != int(final_pair['idx_A']))
    final_changed_from_local_B = (
        local_best_pair['idx_B'] is not None
        and int(local_best_pair['idx_B']) != int(final_pair['idx_B']))

    def _local_pair_candidate_summary(idx_A, idx_B):
        if idx_A is None or idx_B is None:
            return None
        matching = [
            candidate for candidate in pairs
            if int(candidate[2]['idx']) == int(idx_A)
            and int(candidate[3]['idx']) == int(idx_B)
        ]
        if not matching:
            return None

        def candidate_rank(candidate):
            metrics = candidate[7]
            combined = metrics.get('combined_score')
            return (
                0 if combined is not None and np.isfinite(combined) else 1,
                float(combined) if combined is not None and np.isfinite(combined)
                else float(candidate[0]))

        candidate = min(matching, key=candidate_rank)
        metrics = candidate[7]
        return {
            'idx_A': int(idx_A), 'idx_B': int(idx_B),
            'aruco_pair_score': float(candidate[0]),
            'combined_score': metrics.get('combined_score'),
            'marker_rms_px': metrics.get('marker_bidir_rms_px'),
            'feature_marker_epi_px': metrics.get('feat_epi_px'),
            'feature_model_epi_px': metrics.get('feature_model_epi_px'),
            'feature_quality_ok': bool(metrics.get('feature_quality_ok', False)),
            'feature_inliers': int(metrics.get('feature_inliers', 0)),
            'feature_matches': int(metrics.get('feature_matches', 0)),
            'feature_grid_coverage': metrics.get('feature_grid_coverage'),
            'feature_parallax_deg': metrics.get('feature_parallax_deg'),
            'marker_parallax_deg': metrics.get('marker_parallax_deg'),
            'predicted_depth_sigma_mm': metrics.get('predicted_depth_sigma_mm'),
            'baseline_mm': float(candidate[6]),
            'branch': metrics.get('branch'),
            'sift_evaluated': bool(metrics.get('feature_model_epi_px') is not None),
        }

    local_pair_summary = _local_pair_candidate_summary(
        local_best_pair['idx_A'], local_best_pair['idx_B'])
    final_pair_summary = _local_pair_candidate_summary(
        final_pair['idx_A'], final_pair['idx_B'])
    local_window_diagnostics['selection_comparison'] = {
        'provisional_pair': dict(provisional_pair),
        'local_best_pair': dict(local_best_pair),
        'final_pair': dict(final_pair),
        'local_changed_A': bool(local_changed_A),
        'local_changed_B': bool(local_changed_B),
        'final_changed_from_local_A': bool(final_changed_from_local_A),
        'final_changed_from_local_B': bool(final_changed_from_local_B),
        'local_pair_metrics': local_pair_summary,
        'final_pair_metrics': final_pair_summary,
    }
    if provisional_pair:
        log_and_print(
            f"🔁 [Local window前後比較] "
            f"input=(A:F{provisional_pair.get('idx_A')}, B:F{provisional_pair.get('idx_B')}) | "
            f"local_best=(A:F{local_best_pair['idx_A']}, B:F{local_best_pair['idx_B']}) | "
            f"final=(A:F{final_pair['idx_A']}, B:F{final_pair['idx_B']})")
        log_and_print(
            f"   Local換幀: A={bool(local_changed_A)}, B={bool(local_changed_B)} | "
            f"Local後續又換幀: A={bool(final_changed_from_local_A)}, "
            f"B={bool(final_changed_from_local_B)} | "
            f"final offset vs input=({final_pair['offset_A']:+d}, {final_pair['offset_B']:+d})")

    def _format_pair_summary(summary):
        if summary is None:
            return '候選未進入Pair枚舉或已被硬門檻排除'

        def value(name, digits=3, suffix=''):
            raw = summary.get(name)
            if raw is None or not np.isfinite(raw):
                return 'N/A'
            return f"{float(raw):.{digits}f}{suffix}"

        return (
            f"pair_score={value('aruco_pair_score')} | "
            f"combined={value('combined_score')} | "
            f"marker_rms={value('marker_rms_px', suffix='px')} | "
            f"markerRT_epi={value('feature_marker_epi_px', suffix='px')} | "
            f"E_epi={value('feature_model_epi_px', suffix='px')} | "
            f"feature={summary['feature_inliers']}/{summary['feature_matches']} "
            f"quality_ok={summary['feature_quality_ok']} | "
            f"grid={value('feature_grid_coverage', 2)} | "
            f"feature_parallax={value('feature_parallax_deg', suffix='deg')} | "
            f"depth_sigma={value('predicted_depth_sigma_mm', suffix='mm')} | "
            f"baseline={value('baseline_mm', 2, 'mm')} | branch={summary['branch']}")

    if provisional_pair:
        log_and_print(
            f"   [Local最佳pair指標] {_format_pair_summary(local_pair_summary)}")
        log_and_print(
            f"   [最終勝出pair指標] {_format_pair_summary(final_pair_summary)}")
        if final_changed_from_local_A or final_changed_from_local_B:
            observable_reasons = []
            if final_pair_summary is not None:
                if final_pair_summary.get('feature_quality_ok'):
                    observable_reasons.append('最終pair通過Feature幾何品質門檻')
                if local_pair_summary is not None:
                    comparisons = (
                        ('combined_score', 'combined score'),
                        ('marker_rms_px', 'Marker RMS'),
                        ('feature_marker_epi_px', 'Marker-RT極線'),
                        ('feature_model_epi_px', 'Essential極線'),
                        ('predicted_depth_sigma_mm', '預估深度sigma'),
                    )
                    for key, label in comparisons:
                        final_value = final_pair_summary.get(key)
                        local_value = local_pair_summary.get(key)
                        if (final_value is not None and local_value is not None
                                and np.isfinite(final_value) and np.isfinite(local_value)
                                and float(final_value) + 1e-9 < float(local_value)):
                            observable_reasons.append(
                                f"{label}較低({float(local_value):.3f}→{float(final_value):.3f})")
            log_and_print(
                f"   [後續換幀依據] "
                f"{'; '.join(observable_reasons) if observable_reasons else '由後續Pair/IPPE、SIFT與聯合RT有效性排序勝出'}。"
                "未進Top-K的pair不會為了Log額外計算SIFT，因此不虛構不可比較的Feature差值。")
        else:
            log_and_print(
                "   [最終不再換幀] Local最佳pair在後續Pair/IPPE、SIFT與聯合RT檢查後仍勝出。")
    local_window_diagnostics['sift_pair_count'] = int(len(match_cache))
    local_window_diagnostics['elapsed_before_ui_prep_s'] = float(time.perf_counter() - analysis_wall_start)

    pattern_guided_diagnostics['selected_pair'] = {
        'idx_A': int(best_start['idx']),
        'idx_B': int(best_end['idx']),
        'baseline_mm': float(baseline),
        'rt_quality_pattern_guided': best_pair_metrics.get('pattern_guided') if best_pair_metrics else None,
    }
    pattern_guided_diagnostics['elapsed_before_ui_prep_s'] = float(
        time.perf_counter() - analysis_wall_start)
    angle_guided_diagnostics['final_selected_pair'] = {
        'idx_A': int(best_start['idx']),
        'idx_B': int(best_end['idx']),
        'pair_measurement': (
            best_pair_metrics.get('angle_guided') if best_pair_metrics else None),
    }
    timer.stage("診斷狀態封裝")

    if progress_callback:
        progress_callback(92, "階段 3/6：影像校正...")

    # 預先在背景執行去畸變、平面擬合與 SIFT 特徵提取，優化 UI 載入速度
    h_raw, w_raw = frame_height, frame_width
    if frames_override is None:
        newKL_o, _ = cv2.getOptimalNewCameraMatrix(
            mtx_L, dist_L, (w_raw, h_raw), 1, (w_raw, h_raw))
        _map1, _map2 = cv2.initUndistortRectifyMap(
            mtx_L, dist_L, None, newKL_o, (w_raw, h_raw), cv2.CV_16SC2)

        def local_process_view(img):
            return cv2.remap(img, _map1, _map2, cv2.INTER_LINEAR)
    else:
        newKL_o = np.asarray(K_L, dtype=np.float64).copy()

        def local_process_view(img):
            return img.copy()
        
    imgA_bgr = local_process_view(frames[best_end['idx']])  # 結尾最優影格作為左圖 (B)
    imgB_bgr = local_process_view(frames[best_start['idx']])  # 開頭最優影格作為右圖 (A)
    timer.stage("影像去畸變+輸出幀準備")
    
    if progress_callback:
        progress_callback(94, "階段 4/6：基準計算...")
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, True)
    # Reference plane is always the fixed marker-map reference marker plane.
    # Prefer triangulation when the same reference marker is observed by both endpoints.
    # Cross-ID endpoints do not require that observation: T_B<-W already maps the
    # reference plane z=0 into the final left/end camera, so derive it directly.
    _ref_L = {ref_id: cornersB_undist[ref_id]} if ref_id in cornersB_undist else None
    _ref_R = {ref_id: cornersA_undist[ref_id]} if ref_id in cornersA_undist else None
    global_plane_n, global_plane_c = (None, None)
    if _ref_L and _ref_R:
        global_plane_n, global_plane_c = plane_from_triangulated_corners(
            R_rel, t_rel, _ref_L, _ref_R)
        if global_plane_n is not None:
            log_and_print(
                f"✅ [參考平面] 由 ref 標籤 ID:{ref_id} 的三角化角點定義 (其他標籤不參與平面)")
    if global_plane_n is None and best_end is not None:
        _R_B_ref = np.asarray(best_end['R'], dtype=np.float64).reshape(3, 3)
        _t_B_ref = np.asarray(best_end['t'], dtype=np.float64).reshape(3)
        _n_B = _R_B_ref @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _n_norm = float(np.linalg.norm(_n_B))
        if _n_norm > 1e-12 and np.all(np.isfinite(_t_B_ref)):
            global_plane_n = _n_B / _n_norm
            global_plane_c = _t_B_ref.copy()
            log_and_print(
                f"✅ [參考平面] ref ID:{ref_id} 未必雙端共視，改由 final T_B<-W 直接映射固定 reference plane")
    if global_plane_n is None:
        log_and_print("⚠️ [參考平面] marker-map 平面建立失敗，退回 legacy PnP 平面")
        global_plane_n, global_plane_c = compute_global_plane(imgA_gray, K_L, marker_size_mm)
    timer.stage("基準平面建立")
    
    if progress_callback:
        progress_callback(96, "階段 5/6：資料準備...")
    # Depth matching computes local descriptors and does not consume these former
    # full-frame caches, so keep UI startup independent from an unused SIFT pass.
    kb, db = [], None
    timer.stage("深度匹配特徵延後計算")
    analysis_total_elapsed_s = float(time.perf_counter() - analysis_wall_start)
    log_and_print(
        f"⏱️ [選幀/RT耗時] angle_scan="
        f"{float(angle_guided_diagnostics.get('elapsed_s', 0.0)):.3f}s | "
        f"total={analysis_total_elapsed_s:.3f}s | "
        f"mode={'angle_guided' if angle_guided_enabled else 'original'} | "
        f"status={angle_guided_diagnostics.get('status')}")
    timer.report(print_fn=log_and_print)
    pair_stage_elapsed_s = next((
        float(elapsed) for name, elapsed in reversed(timer.stages)
        if name == "ArUco偵測+配對搜尋(含極線重排)"), 0.0)
    accounted_pair_s = float(sum(pair_search_timing.values()))
    pair_search_timing['other_overhead'] = max(
        0.0, pair_stage_elapsed_s - accounted_pair_s)
    # Keep both parent and child rows in the exact data-flow order.  This makes
    # successive runs directly comparable without mentally reordering stages.
    pair_timing_tree = (
        ('endpoint_proposal', '1. 角度掃描/端點提案', (
            ('endpoint_frame_decode_aruco', '1.1 影格解碼+ArUco偵測'),
            ('endpoint_pose_direction_rank', '1.2 PnP角度+方向判斷+候選排序'),
        )),
        ('marker_map_temporal', '2. Marker map+時序姿態DP', (
            ('marker_probe_decode_detect', '2.1 稀疏probe解碼+ArUco偵測', (
                ('marker_probe_video_decode', '2.1.1 影片seek/grab+目標幀解碼'),
                ('marker_probe_gray_convert', '2.1.2 BGR轉灰階'),
                ('marker_probe_clahe', '2.1.3 CLAHE對比增強'),
                ('marker_probe_aruco_detect', '2.1.4 ArUco候選偵測'),
                ('marker_probe_corner_subpix', '2.1.5 cornerSubPix角點精修'),
                ('marker_probe_cache_overhead', '2.1.6 Cache+資料整理'),
            )),
            ('marker_map_graph', '2.2 共視圖建圖+剛體Marker map'),
            ('temporal_pose_hypotheses', '2.3 IPPE逐幀姿態假設'),
            ('temporal_path_dp', '2.4 時序DP最佳路徑'),
            ('marker_temporal_overhead', '2.5 資料整理+診斷輸出'),
        )),
        ('pattern_guided_expansion', '3. Pattern-guided候選擴充', (
            ('pattern_core_geometry', '3.1 Core pair幾何可用性評估'),
            ('pattern_extra_probe_rebuild', '3.2 額外probe+時序路徑重建'),
            ('pattern_overhead', '3.3 狀態更新+控制開銷'),
        )),
        ('local_window_klt', '4. Local window+KLT重排', (
            ('local_provisional_pair', '4.1 暫定端點pair評分'),
            ('local_frame_decode_detect', '4.2 鄰幀解碼+ArUco偵測'),
            ('local_pose_hypotheses', '4.3 鄰幀IPPE姿態假設'),
            ('local_klt_tracking', '4.4 相鄰幀KLT追蹤'),
            ('local_path_rerank', '4.5 Local DP+清晰度/KLT重排'),
            ('local_global_dp_rebuild', '4.6 合併鄰幀後重建全域DP'),
            ('local_diagnostic_logging', '4.7 詳細選幀原因Log輸出'),
            ('local_overhead', '4.8 Window建立+資料整理'),
        )),
        ('candidate_enumeration', '5. Pair/IPPE分支枚舉評分', (
            ('candidate_branch_pack', '5.1 端點時序分支封裝'),
            ('candidate_pair_branch_score', '5.2 Pair×IPPE組合幾何評分'),
            ('candidate_topk_budget', '5.3 排序+Top-K預算配置'),
            ('candidate_overhead', '5.4 控制開銷'),
        )),
        ('sift_essential_rerank', '6. SIFT+Essential極線重排', (
            ('sift_feature_extract', '6.1 SIFT特徵提取'),
            ('sift_descriptor_match', '6.2 ratio+mutual+網格匹配'),
            ('sift_essential_geometry', '6.3 Essential RANSAC+recoverPose'),
            ('sift_rerank_and_select', '6.4 極線/視差評分+最終重排'),
            ('sift_overhead', '6.5 候選控制+快取開銷'),
        )),
    )
    log_and_print(
        "      ↳ [ArUco+配對搜尋細分；依算法執行順序] "
        f"parent={pair_stage_elapsed_s * 1000.0:.1f} ms")
    for parent_key, parent_label, children in pair_timing_tree:
        parent_elapsed = float(pair_search_timing.get(parent_key, 0.0))
        parent_percentage = (
            parent_elapsed / pair_stage_elapsed_s * 100.0
            if pair_stage_elapsed_s > 1e-9 else 0.0)
        log_and_print(
            f"         {parent_label:<34s}{parent_elapsed * 1000.0:9.1f} ms "
            f"({parent_percentage:5.1f}% of search)")
        for child_entry in children:
            child_key, child_label = child_entry[:2]
            grandchildren = child_entry[2] if len(child_entry) >= 3 else ()
            child_elapsed = float(pair_search_detail_timing.get(child_key, 0.0))
            child_percentage = (
                child_elapsed / parent_elapsed * 100.0
                if parent_elapsed > 1e-9 else 0.0)
            log_and_print(
                f"             - {child_label:<31s}{child_elapsed * 1000.0:9.1f} ms "
                f"({child_percentage:5.1f}% of stage)")
            for grandchild_key, grandchild_label in grandchildren:
                grandchild_elapsed = float(
                    pair_search_detail_timing.get(grandchild_key, 0.0))
                grandchild_percentage = (
                    grandchild_elapsed / child_elapsed * 100.0
                    if child_elapsed > 1e-9 else 0.0)
                log_and_print(
                    f"                 · {grandchild_label:<27s}"
                    f"{grandchild_elapsed * 1000.0:9.1f} ms "
                    f"({grandchild_percentage:5.1f}% of 2.1)")
    other_elapsed = float(pair_search_timing.get('other_overhead', 0.0))
    other_percentage = (
        other_elapsed / pair_stage_elapsed_s * 100.0
        if pair_stage_elapsed_s > 1e-9 else 0.0)
    log_and_print(
        f"         7. 其餘跨階段控制與日誌開銷{'':<16s}"
        f"{other_elapsed * 1000.0:9.1f} ms "
        f"({other_percentage:5.1f}% of search)")
    if progress_callback:
        progress_callback(100, "階段 6/6：完成")
    analysis_stage_timing = [
        {'stage': str(name), 'elapsed_s': float(elapsed)}
        for name, elapsed in timer.stages
    ]

    result = {
        'frame_A': frames[best_start['idx']],
        'frame_B': frames[best_end['idx']],
        'idx_A': best_start['idx'],
        'idx_B': best_end['idx'],
        'R_rel': R_rel,
        't_rel': t_rel,
        'baseline': baseline,
        'cornersA': cornersA_undist,
        'cornersB': cornersB_undist,
        'all_frames': frames,
        'valid_poses': valid_poses,
        'marker_map': marker_map,
        'temporal_diagnostics': temporal_diagnostics,
        'pattern_guided_diagnostics': pattern_guided_diagnostics,
        'local_window_diagnostics': local_window_diagnostics,
        'angle_guided_diagnostics': angle_guided_diagnostics,
        'pair_search_timing_s': dict(pair_search_timing),
        'pair_search_detail_timing_s': dict(pair_search_detail_timing),
        'analysis_stage_timing_s': analysis_stage_timing,
        'analysis_total_elapsed_s': analysis_total_elapsed_s,
        'extra_candidates': extra_candidates_info,
        'min_reproj_err': best_reproj_err,
        'marker_reproj_err': marker_reproj_err,
        'marker_pnp_self_reproj_err': marker_pnp_self_reproj_err,
        'marker_bidir_stats': marker_bidir_stats,
        'rt_quality': best_pair_metrics,
        'rt_sift_points_left': rt_sift_points_left,
        'rt_sift_points_right': rt_sift_points_right,
        'rt_sift_match_count': 0 if best_feature_matches is None else int(len(best_feature_matches[0])),
        'rt_sift_inlier_count': int(len(rt_sift_points_left)),
        'rt_sift_applied': bool(best_feature_rt_applied),
        'rt_sift_role': rt_sift_role,
        'rt_sift_diagnostics_path': rt_sift_diagnostics_path,
        'detection_roi_bounds': detection_roi_bounds_start,
        'detection_roi_bounds_by_role': {
            'frame_A': detection_roi_bounds_start,
            'frame_B': detection_roi_bounds_end,
        },
        'feature_roi_bounds_by_role': {
            'frame_A': feature_roi_bounds_start,
            'frame_B': feature_roi_bounds_end,
        },
        'feature_image_scale': float(FEATURE_IMAGE_SCALE),
        'pair_geometry_roi_bounds_by_role': {
            'frame_A': geometry_roi_bounds_start,
            'frame_B': geometry_roi_bounds_end,
        },
        'global_plane_n': global_plane_n,
        'global_plane_c': global_plane_c,
        'best_kpB': kb,
        'best_desB': db
    }
    if (angle_guided_diagnostics.get('status') == 'OK_ANGLE_GUIDED'
            and angle_guided_diagnostics.get('output_roles_swapped', False)):
        log_and_print(
            "🔄 [角度掃描方向] 反向影片：保留早→晚時序精修，"
            "輸出時交換左右圖並反轉 RT")
        result = _angle_guided_normalize_reversed_output(result)
    return result
