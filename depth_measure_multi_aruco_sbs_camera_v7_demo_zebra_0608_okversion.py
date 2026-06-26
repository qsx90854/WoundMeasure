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

import os, sys, glob, json, threading, queue
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

CAMERA_WIDTH          = 1920                       # 相機解析度寬
CAMERA_HEIGHT         = 1080                       # 相機解析度高

PARAMS_JSON_PATH      = "calibration_result_Zebra_1_no_dis.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM = 8.25                       # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
MIN_BASELINE_MM       = 8.0                        # 最小基準線限制 (mm)
AUTO_CALC_INTERVAL_SEC = 0.2                       # 連續計算模式下的計算時間間隔 (秒)
ENFORCE_COPLANAR      = False                      # 強制共面對齊優化
SAVE_ARUCO_DEBUG_IMG  = False                      # 是否存出 ArUco 偵測結果圖片
ENABLE_POSE_SMOOTHING  = True                      # 是否啟用時序平滑濾波 (EMA)
POSE_SMOOTHING_ALPHA   = 0.3                       # 平滑係數
ENABLE_CLAHE_DEFAULT  = False                      # 預設是否啟用 CLAHE
CLAHE_CLIP_LIMIT      = 2.0                        # CLAHE 對比度限制閾值 (數值愈大對比愈強，雜訊也愈大)
CLAHE_TILE_GRID_SIZE  = (8, 8)                     # CLAHE 分塊大小 (8, 8) 代表 8x8 的網格



# ===================================================

def load_json_camera_params(json_path):
    if not os.path.exists(json_path):
        print(f"❌ 找不到參數檔案: {json_path}"); return None, None, None, None, None, None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mtx_L = np.array(data['intrinsic_L']['matrix'], dtype=np.float32)
    dist_L = np.array(data['intrinsic_L']['distortion'], dtype=np.float32)
    mtx_R = np.array(data['intrinsic_R']['matrix'], dtype=np.float32)
    dist_R = np.array(data['intrinsic_R']['distortion'], dtype=np.float32)
    extrinsic = data.get('extrinsic', {})
    R_rel = np.array(extrinsic.get('R', np.eye(3)))
    t_rel = np.array(extrinsic.get('T', np.zeros(3))).reshape(3, 1)
    F_orig = np.array(extrinsic.get('F')) if 'F' in extrinsic else None

    # forvideo,勿修改
    mtx_R = mtx_L
    dist_R = dist_L

    return mtx_L, dist_L, mtx_R, dist_R, extrinsic, F_orig

def compute_fundamental_matrix(K_L, K_R, R_rel, t_rel):
    t = t_rel.flatten()
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    E = tx @ R_rel
    K_R_inv, K_L_inv = np.linalg.inv(K_R.astype(np.float64)), np.linalg.inv(K_L.astype(np.float64))
    return K_R_inv.T @ E @ K_L_inv

def triangulate_point_3d(pt_A, pt_B, K_L, K_R, R_rel, t_rel):
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

def find_precise_match(imgA_gray, imgB_gray, pt_A, F, K_L, K_R, R_rel, t_rel, plane_normal, plane_center):
    if plane_normal is None or plane_center is None: return None
    h, w = imgA_gray.shape
    d = np.dot(plane_normal, plane_center)
    H_3D = R_rel + (t_rel @ plane_normal.reshape(1, 3)) / d
    H_AB = K_R @ H_3D @ np.linalg.inv(K_L.astype(np.float64))
    H_BA = np.linalg.inv(H_AB)
    imgB_warped = cv2.warpPerspective(imgB_gray, H_BA, (w, h), flags=cv2.INTER_LINEAR)
    patch_size = 20
    u, v = int(round(pt_A[0])), int(round(pt_A[1]))
    u0, u1 = max(0, u-patch_size), min(w, u+patch_size+1)
    v0, v1 = max(0, v-patch_size), min(h, v+patch_size+1)
    patch_l = imgA_gray[v0:v1, u0:u1]
    if patch_l.size == 0: return None
    search = 45
    x0, y0 = max(0, u-search-patch_size), max(0, v-search-patch_size)
    x1, y1 = min(w, u+search+patch_size+1), min(h, v+search+patch_size+1)
    roi_r = imgB_warped[y0:y1, x0:x1]
    if roi_r.shape[0] < patch_l.shape[0] or roi_r.shape[1] < patch_l.shape[1]: return None
    res = cv2.matchTemplate(roi_r, patch_l, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.4: return None
    pt_w_h = np.array([x0 + max_loc[0] + patch_size, y0 + max_loc[1] + patch_size, 1.0])
    pt_B_h = H_AB @ pt_w_h
    return (float(pt_B_h[0]/pt_B_h[2]), float(pt_B_h[1]/pt_B_h[2]))


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
                     
    while curr_f_idx != end_f_idx:
        next_f_idx = curr_f_idx + step
        
        img_prev = cv2.cvtColor(all_frames[curr_f_idx], cv2.COLOR_BGR2GRAY)
        img_next = cv2.cvtColor(all_frames[next_f_idx], cv2.COLOR_BGR2GRAY)
        
        img_prev = preprocess_gray(img_prev, True)
        img_next = preprocess_gray(img_next, True)
        
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
    if progress_callback:
        progress_callback(2, "正在開啟影片檔案...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_and_print(f"🎬 載入影片: {video_path}，總影格數: {total_frames} (選幀範圍模式: {range_mode})")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if progress_callback and len(frames) % 30 == 0:
            load_percent = min(2 + (len(frames) / max(1, total_frames)) * 10, 12)
            progress_callback(load_percent, f"正在載入影片影格 ({len(frames)}/{total_frames})...")
            
    cap.release()
    if progress_callback:
        progress_callback(12, f"影片載入完成，共 {len(frames)} 影格。")
    
    if len(frames) == 0:
        print("❌ 影片無有效影格")
        return None
        
    mid_idx = len(frames) // 2
    if range_mode == "half_half":
        start_range = range(0, mid_idx)
        end_range = range(mid_idx, len(frames))
    else:
        N = min(start_n, len(frames))
        M = min(end_n, len(frames))
        start_range = range(N)
        end_range = range(len(frames) - M, len(frames))
    
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
    else:
        params = cv2.aruco.DetectorParameters_create()
        
    def detect_frame_markers(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess_gray(gray, True)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dict_4x4, parameters=params)
        if ids is not None and len(ids) > 0:
            ids_list = [i[0] for i in ids]
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            for c in corners:
                cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), term)
            raw_corners = [c.reshape(4, 2) for c in corners]
            return dict(zip(ids_list, raw_corners))
        return {}

    # 定義輔助工具
    def undistort_corners_dict(corners_dict):
        undist = {}
        for mid, pts in corners_dict.items():
            pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
            pts_undist = cv2.undistortPoints(pts_reshaped, mtx_L, dist_L, P=K_L)
            undist[mid] = pts_undist.reshape(4, 2)
        return undist

    detected_cache = {}
    
    def get_frame_info(idxs, stage_idx=0, is_start_segment=True):
        info = []
        seg_name = "開頭段" if is_start_segment else "結尾段"
        for i, idx in enumerate(idxs):
            if idx not in detected_cache:
                detected_cache[idx] = detect_frame_markers(frames[idx])
            cd = detected_cache[idx]
            if cd:
                info.append({'idx': idx, 'corners': cd})
            if progress_callback:
                stage_base = 15 + stage_idx * 15
                if is_start_segment:
                    percent = stage_base + (i / len(idxs)) * 7.5
                else:
                    percent = stage_base + 7.5 + (i / len(idxs)) * 7.5
                progress_callback(min(percent, 98.0), f"正在分析第 {stage_idx + 1}/5 階段 - {seg_name} 偵測標籤 ({i + 1}/{len(idxs)})...")
        return info

    def sample_range(r, n):
        lst = list(r)
        if len(lst) <= n:
            return lst
        idxs = np.linspace(0, len(lst) - 1, n, dtype=int)
        return [lst[idx] for idx in idxs]

    def save_debug_pair_images(item_s, item_e, suffix):
        img_A = frames[item_s['idx']].copy()
        img_B = frames[item_e['idx']].copy()
        corners_s = item_s['corners']
        corners_e = item_e['corners']
        R_s, t_s = item_s['R'], item_s['t']
        R_e, t_e = item_e['R'], item_e['t']
        shared_mids = set(corners_s.keys()).intersection(set(corners_e.keys()))
        half = marker_size_mm / 2.0
        canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        for mid, pts in corners_s.items():
            pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_A, [pts_int], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.putText(img_A, f"Obs:{mid}", (pts_int[0][0][0], pts_int[0][0][1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            if mid in shared_mids and mid in marker_map:
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
            if mid in shared_mids and mid in marker_map:
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

    def compute_pair_reprojection_error(item_s, item_e, mtx_L, dist_L):
        corners_s = item_s['corners']
        corners_e = item_e['corners']
        R_s, t_s = item_s['R'], item_s['t']
        R_e, t_e = item_e['R'], item_e['t']
        shared_mids = set(corners_s.keys()).intersection(set(corners_e.keys()))
        if not shared_mids:
            return float('inf')
            
        rvec_s, _ = cv2.Rodrigues(R_s)
        rvec_e, _ = cv2.Rodrigues(R_e)
        
        errors = []
        half = marker_size_mm / 2.0
        canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        for mid in shared_mids:
            if mid not in marker_map:
                continue
            R_m2ref, t_m2ref = marker_map[mid]
            P_w = (R_m2ref @ canon.T).T + t_m2ref.T
            
            # 投影至 Frame s (右圖/開頭幀)
            pts_s_proj, _ = cv2.projectPoints(P_w.astype(np.float32), rvec_s, t_s, mtx_L, dist_L)
            pts_s_proj = pts_s_proj.reshape(4, 2)
            err_s = np.linalg.norm(pts_s_proj - corners_s[mid], axis=1)
            errors.extend(err_s)
            
            # 投影至 Frame e (左圖/結尾幀)
            pts_e_proj, _ = cv2.projectPoints(P_w.astype(np.float32), rvec_e, t_e, mtx_L, dist_L)
            pts_e_proj = pts_e_proj.reshape(4, 2)
            err_e = np.linalg.norm(pts_e_proj - corners_e[mid], axis=1)
            errors.extend(err_e)
            
        if not errors:
            return float('inf')
        return np.mean(errors)

    # 多階段漸進式匹配評估
    best_start = None
    best_end = None
    R_rel = None
    t_rel = None
    baseline = None
    selected_extras = []
    marker_map = {}
    
    stages = [10, 20, 30, 40, 50]
    stage_success = False
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    
    for stage_idx, num_samples in enumerate(stages):
        log_and_print(f"🔄 開始第 {stage_idx + 1} 階段抽樣評估 (抽樣張數: 各段最多 {num_samples} 張)...")
        
        sampled_start = sample_range(start_range, num_samples)
        sampled_end = sample_range(end_range, num_samples)
        
        start_info = get_frame_info(sampled_start, stage_idx, is_start_segment=True)
        end_info = get_frame_info(sampled_end, stage_idx, is_start_segment=False)
        
        if not start_info or not end_info:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：開頭段或結尾段無有效 ArUco 標籤")
            continue
            
        start_ids = set()
        for item in start_info: start_ids.update(item['corners'].keys())
        end_ids = set()
        for item in end_info: end_ids.update(item['corners'].keys())
        
        shared_ids = list(start_ids.intersection(end_ids))
        if not shared_ids:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：開頭段與結尾段無共享的 ArUco 標籤，無法建立統一世界座標系")
            continue
            
        ref_id = min(shared_ids)
        log_and_print(f"📌 [第 {stage_idx + 1} 階段] 選定共享參考標籤 ID: {ref_id} 作為世界坐標系原點")
        
        # 1. 全域標籤世界地圖自標定
        marker_map = {}
        marker_map[ref_id] = (np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
        
        relative_poses = {}
        all_info = start_info + end_info
        for item in all_info:
            corners_dict = item['corners']
            if ref_id in corners_dict:
                ok_ref, rvec_ref, tvec_ref = cv2.solvePnP(canon, corners_dict[ref_id], mtx_L, dist_L, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok_ref: continue
                R_ref, _ = cv2.Rodrigues(rvec_ref)
                t_ref = tvec_ref.reshape(3, 1)
                
                for mid, pts in corners_dict.items():
                    if mid == ref_id: continue
                    ok_m, rvec_m, tvec_m = cv2.solvePnP(canon, pts, mtx_L, dist_L, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not ok_m: continue
                    R_m, _ = cv2.Rodrigues(rvec_m)
                    t_m = tvec_m.reshape(3, 1)
                    
                    R_m2ref = R_ref.T @ R_m
                    t_m2ref = R_ref.T @ (t_m - t_ref)
                    
                    if mid not in relative_poses:
                        relative_poses[mid] = []
                    relative_poses[mid].append((R_m2ref, t_m2ref))
                    
        for mid, poses in relative_poses.items():
            R_list = [p[0] for p in poses]
            t_list = [p[1] for p in poses]
            R_avg = average_rotations_svd(R_list)
            t_avg = np.mean(t_list, axis=0)
            marker_map[mid] = (R_avg, t_avg)
            
        def get_joint_pose(corners_dict):
            obj_pts = []
            img_pts = []
            for mid, pts in corners_dict.items():
                if mid in marker_map:
                    R_m2ref, t_m2ref = marker_map[mid]
                    pts_w = (R_m2ref @ canon.T).T + t_m2ref.T
                    obj_pts.append(pts_w)
                    img_pts.append(pts)
            if len(obj_pts) == 0:
                return None, None
            obj_pts = np.vstack(obj_pts).astype(np.float32)
            img_pts = np.vstack(img_pts).astype(np.float32)
            
            rvec_init = np.zeros((3, 1), dtype=np.float32)
            tvec_init = np.zeros((3, 1), dtype=np.float32)
            use_guess = False
            if ref_id in corners_dict:
                ok_init, rv_i, tv_i = cv2.solvePnP(canon, corners_dict[ref_id], mtx_L, dist_L, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok_init:
                    rvec_init = rv_i.copy().astype(np.float32)
                    tvec_init = tv_i.copy().astype(np.float32)
                    use_guess = True
                    
            if use_guess:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx_L, dist_L, rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx_L, dist_L, flags=cv2.SOLVEPNP_ITERATIVE)
                
            if not ok: return None, None
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            return R, t

        valid_start = []
        for item in start_info:
            R, t = get_joint_pose(item['corners'])
            if R is not None:
                valid_start.append({'idx': item['idx'], 'R': R, 't': t, 'corners': item['corners']})
                
        valid_end = []
        for item in end_info:
            R, t = get_joint_pose(item['corners'])
            if R is not None:
                valid_end.append({'idx': item['idx'], 'R': R, 't': t, 'corners': item['corners']})
                
        if not valid_start or not valid_end:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無法計算有效的起點或終點 Joint Pose")
            continue

        # 計算候選對的重投影誤差與 baseline
        pairs = []
        for item_s in valid_start:
            for item_e in valid_end:
                R_s, t_s = item_s['R'], item_s['t']
                R_e, t_e = item_e['R'], item_e['t']
                R_rel_cand = R_s @ R_e.T
                t_rel_cand = t_s - R_rel_cand @ t_e
                bsl = float(np.linalg.norm(t_rel_cand))
                
                if bsl >= MIN_BASELINE_MM:
                    err = compute_pair_reprojection_error(item_s, item_e, mtx_L, dist_L)
                    if err != float('inf'):
                        pairs.append((err, item_s, item_e, R_rel_cand, t_rel_cand, bsl))
                        
        if not pairs:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無合格的匹配對 (Baseline >= {MIN_BASELINE_MM} mm)")
            continue
            
        # 依誤差由小到大排序
        pairs.sort(key=lambda x: x[0])
        best_cand = pairs[0]
        best_err = best_cand[0]
        
        if best_err < 0.2:
            best_start, best_end, R_rel, t_rel, baseline = best_cand[1], best_cand[2], best_cand[3], best_cand[4], best_cand[5]
            log_and_print(f"🎉 第 {stage_idx + 1} 階段搜尋成功！在 {num_samples} 張抽樣下，找到誤差 < 0.2 px 的最佳配對：誤差 {best_err:.3f} px")
            stage_success = True
        else:
            log_and_print(f"ℹ️ 第 {stage_idx + 1} 階段最小誤差為 {best_err:.3f} px (未低於 0.2 px 門檻)")
            
        # 若本階段成功，或這已是最大抽樣張數的第二階段，即固定最佳與次佳解
        if stage_success or num_samples == 50:
            if not stage_success:
                best_start, best_end, R_rel, t_rel, baseline = best_cand[1], best_cand[2], best_cand[3], best_cand[4], best_cand[5]
                log_and_print(f"⚠️ 達到最大抽樣張數 (50 張) 仍未找到低於 0.2 px 的配對。降級使用當前最優對，誤差為: {best_err:.3f} px")
                
            # 次佳對選取：重投影誤差小於 0.5 px 且 baseline 大於門檻的其餘配對，最多取 5 組
            candidates_scores = []
            for err, item_s, item_e, R_rel_c, t_rel_c, bsl in pairs:
                if item_e['idx'] == best_end['idx'] and item_s['idx'] != best_start['idx']:
                    if err < 0.5:
                        candidates_scores.append((err, item_s, R_rel_c, t_rel_c, bsl))
            
            selected_extras = candidates_scores[:5]
            break
            
    if best_start is None or best_end is None:
        log_and_print("❌ [漸進式匹配] 無法在該影片中計算出任何影像對，分析失敗。")
        return None

    # 儲存最佳配對偵錯圖片
    save_debug_pair_images(best_start, best_end, "best")

    # 包裝次優額外右圖組
    extra_candidates_info = []
    for err, item_s, R_rel_c, t_rel_c, bsl in selected_extras:
        extra_candidates_info.append({
            'idx_A': item_s['idx'],
            'frame_A': frames[item_s['idx']],
            'R_rel': R_rel_c,
            't_rel': t_rel_c,
            'baseline': bsl,
            'cornersA': undistort_corners_dict(item_s['corners'])
        })
        log_and_print(f"➕ [次佳配對] 額外右圖 (Frame A) 索引: {item_s['idx']} | 重投影誤差: {err:.3f} px | Baseline: {bsl:.2f} mm")

    if not selected_extras:
        log_and_print(f"ℹ️ [次佳配對] 未找到任何符合條件 of 額外次佳配對影格 (門檻 0.5 px, Baseline >= {MIN_BASELINE_MM} mm)。")

    log_and_print(f"✅ 挑選結果：")
    log_and_print(f"  - 右圖 (Frame A) 索引: {best_start['idx']}")
    log_and_print(f"  - 左圖 (Frame B) 索引: {best_end['idx']}")
    log_and_print(f"  - 計算 Baseline: {baseline:.2f} mm")
    
    valid_poses = {}
    for item in valid_start + valid_end:
        valid_poses[item['idx']] = (item['R'], item['t'])
        
    cornersA_undist = undistort_corners_dict(best_start['corners'])
    cornersB_undist = undistort_corners_dict(best_end['corners'])
    
    best_reproj_err = None
    if best_start is not None and best_end is not None:
        best_reproj_err = compute_pair_reprojection_error(best_start, best_end, mtx_L, dist_L)
        
    if progress_callback:
        progress_callback(92, "正在進行去畸變影像校正...")
        
    # 預先在背景執行去畸變、平面擬合與 SIFT 特徵提取，優化 UI 載入速度
    h_raw, w_raw = frames[0].shape[:2]
    newKL_o, _ = cv2.getOptimalNewCameraMatrix(mtx_L, dist_L, (w_raw, h_raw), 1, (w_raw, h_raw))
    _map1, _map2 = cv2.initUndistortRectifyMap(mtx_L, dist_L, None, newKL_o, (w_raw, h_raw), cv2.CV_16SC2)
    
    def local_process_view(img):
        return cv2.remap(img, _map1, _map2, cv2.INTER_LINEAR)
        
    imgA_bgr = local_process_view(frames[best_end['idx']])  # 結尾最優影格作為左圖 (B)
    imgB_bgr = local_process_view(frames[best_start['idx']])  # 開頭最優影格作為右圖 (A)
    
    if progress_callback:
        progress_callback(94, "正在擬合世界坐標系參考平面...")
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, True)
    global_plane_n, global_plane_c = compute_global_plane(imgA_gray, K_L, marker_size_mm)
    
    if progress_callback:
        progress_callback(96, "正在提取最佳影像組 SIFT 特徵...")
    sift = cv2.SIFT_create(contrastThreshold=0.005)
    imgB_gray = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY)
    kb, db = sift.detectAndCompute(imgB_gray, None)
    
    # 次佳候選影像特徵提取
    for idx_extra, extra in enumerate(extra_candidates_info):
        if progress_callback:
            progress = min(97 + int((idx_extra / max(1, len(extra_candidates_info))) * 3), 99)
            progress_callback(progress, f"正在提取次佳影像 SIFT 特徵 ({idx_extra+1}/{len(extra_candidates_info)})...")
        imgB_extra_bgr = local_process_view(extra['frame_A'])
        imgB_extra_gray = cv2.cvtColor(imgB_extra_bgr, cv2.COLOR_BGR2GRAY)
        kb_e, db_e = sift.detectAndCompute(imgB_extra_gray, None)
        extra['kpB'] = kb_e
        extra['desB'] = db_e
        
    if progress_callback:
        progress_callback(100, "分析完成，即將載入主量測介面...")
        
    return {
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
        'extra_candidates': extra_candidates_info,
        'min_reproj_err': best_reproj_err,
        'global_plane_n': global_plane_n,
        'global_plane_c': global_plane_c,
        'best_kpB': kb,
        'best_desB': db
    }

def preprocess_gray(gray_img, enable_clahe=True):

    if enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        return clahe.apply(gray_img)
    return gray_img

def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cA, idsA, _ = detector.detectMarkers(imgA_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
    if idsA is None or len(idsA) < 1: 
        print("⚠️ [平面擬合] 在左圖中未偵測到任何 ArUco 標籤。")
        return None, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for c in cA: cv2.cornerSubPix(imgA_gray, c, (3, 3), (-1, -1), term)
    
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    objA = []
    for c in cA:
        ok, rv, tv = cv2.solvePnP(canon, c[0], K_L, np.zeros(5))
        if ok:
            R, _ = cv2.Rodrigues(rv)
            objA.append((R @ canon.T).T + tv.T)
    if not objA: 
        print("⚠️ [平面擬合] 對 ArUco 標籤進行 PnP 位姿解算時全部失敗。")
        return None, None
    objA = np.vstack(objA).astype(np.float32)
    c = np.mean(objA, axis=0)
    _, _, Vt = np.linalg.svd(objA - c)
    n = Vt[-1]
    if np.dot(n, c) > 0: n = -n
    log_and_print(f"✅ [平面擬合成功] 偵測到 {len(idsA)} 個標籤，擬合平面法向量: {n.flatten()}，中心點: {c.flatten()}")
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

    if len(shared) == 1:
        ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        if use_guess_rel:
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), rvec=rv_rel_init, tvec=tv_rel_init, useExtrinsicGuess=True)
        else:
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok_rel: return None, map_calibrated
    R_rel, _ = cv2.Rodrigues(rv_rel)
    return (R_rel, tv_rel, float(np.linalg.norm(tv_rel)), objA, shared, cA_dict, cB_dict, curr_marker_poses, rv_rel), map_calibrated


def get_patch(img, pt, size):
    h, w = img.shape; x, y = int(round(pt[0])), int(round(pt[1]))
    x0, x1 = x - size//2, x + size//2 + 1
    y0, y1 = y - size//2, y + size//2 + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h: return None
    return img[y0:y1, x0:x1]

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
    cap = cv2.VideoCapture(1)#, cv2.CAP_MSMF)
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
    
    # 距標記平面深度
    p_dist_str = "None"
    if p3d is not None and cand.get('plane_n') is not None and cand.get('plane_c') is not None:
        p_dist = abs(np.dot(cand['plane_n'], p3d - cand['plane_c']))
        p_dist_str = f"{p_dist:.2f} mm"
        
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

def select_video_source():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.title("互動式量測 - 選擇影片來源")
    
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
    btn_font = ("Microsoft JhengHei", 11, "bold")

    # 回傳變數
    selected_path = {"path": None, "action": None}

    # 標題
    title_label = tk.Label(root, text="請選擇深度量測的影片來源：", font=title_font, fg="#FFFFFF", bg="#2D2D2D", pady=25)
    title_label.pack()

    def on_camera():
        selected_path["action"] = "camera"
        root.destroy()

    def on_file():
        selected_path["action"] = "file"
        file_path = filedialog.askopenfilename(
            parent=root,
            title="選擇 SBS 雙目影片",
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
        text="📷 開啟相機即時錄影", 
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
        text="📁 載入指定影片檔案", 
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
    root.title("影片分析進度")
    
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
    status_var = tk.StringVar(value="準備分析影片...")
    progress_var = tk.DoubleVar(value=0.0)

    # UI 元件
    title_label = tk.Label(root, text="🎥 SBS 雙目影片自動標定中", font=("Microsoft JhengHei", 12, "bold"), fg="#FFFFFF", bg="#2D2D2D", pady=10)
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
        update_queue.put((percent, status_text))

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
    
    # 1. 讀取相機內參
    mtxL_o, distL, mtxR_o, distR, extrinsic, F_orig = load_json_camera_params(PARAMS_JSON_PATH)
    
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
    
    newKL_o, _ = cv2.getOptimalNewCameraMatrix(mtxL_o, distL, (w_alg, h_raw), 1, (w_alg, h_raw))
    KL = newKL_o.copy().astype(np.float64)
    
    # 預先建立去畸變查找表
    _map1, _map2 = cv2.initUndistortRectifyMap(
        mtxL_o, distL, None, newKL_o, (w_alg, h_raw), cv2.CV_16SC2
    )

    def process_view(img, K=None, dist=None, nK=None):
        undist = cv2.remap(img, _map1, _map2, cv2.INTER_LINEAR)
        return undist, 1.0
    log_and_print("🔄 正在分析影片中開頭與結尾影格的 ArUco 標籤與最優姿態對...")
    video_data = analyze_video_with_progress_bar(VIDEO_PATH, START_FRAME_COUNT, END_FRAME_COUNT, KL, distL, mtxL_o, ACTUAL_MARKER_SIZE_MM, POSE_SELECT_MODE, FRAME_RANGE_MODE)
    if video_data is None:
        print("❌ 影片 Pose 分析失敗，無法啟動測量工具")
        sys.exit(1)
        
    # 去畸變處理挑選出的最優左右圖
    imgA_bgr, _ = process_view(video_data['frame_B']) # 結尾最優影格作為左圖 (B)
    imgB_bgr, _ = process_view(video_data['frame_A']) # 開頭最優影格作為右圖 (A)
    
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, ENABLE_CLAHE_DEFAULT)
    h, w = imgA_gray.shape
    
    # 使用分析得到的相對 R, t 和 baseline
    R_r = video_data['R_rel']
    t_r = video_data['t_rel']
    
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
        'pose_info': f"ArUco 多影格平均 (Bsl: {video_data['baseline']:.1f}mm)" if POSE_SELECT_MODE=="average" else f"ArUco 最優對 (Bsl: {video_data['baseline']:.1f}mm)",
        'marker_map': video_data['marker_map'],
        'map_calibrated': True
    }
    
    # 若背景計算因任何理由未獲得特徵，則降級在主線程中計算
    if not current_cand['kpB'] or current_cand['desB'] is None:
        kb, db = sift.detectAndCompute(current_cand['gray'], None)
        current_cand.update({'kpB': kb, 'desB': db})
    
    extra_candidates_list = []
    for extra in video_data.get('extra_candidates', []):
        imgB_extra_bgr, _ = process_view(extra['frame_A'])
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
            'pose_info': f"ArUco 次優對 (Bsl: {extra['baseline']:.1f}mm)"
        }
        # 若背景中未成功提取，才在主線程中提取特徵
        if not extra_cand['kpB'] or extra_cand['desB'] is None:
            kb_e, db_e = sift.detectAndCompute(extra_cand['gray'], None)
            extra_cand.update({'kpB': kb_e, 'desB': db_e})
        extra_candidates_list.append(extra_cand)
        
    candidates = [current_cand]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1E1E1E')
    fig.canvas.manager.set_window_title("MeasureTool")
    try:
        fig.canvas.toolbar.pack_forget() # 隱藏底部的功能條
    except:
        pass
    fig.subplots_adjust(top=0.82, right=0.98, left=0.05, bottom=0.15)
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

    def draw_reprojected_aruco(ax_target, corners_src, R_rel, t_rel, is_left_to_right=True):
        if not hasattr(ax_target, 'reproj_art'): ax_target.reproj_art = []
        for a in ax_target.reproj_art: a.remove()
        ax_target.reproj_art = []
        if not corners_src: return
        
        half = ACTUAL_MARKER_SIZE_MM / 2.0
        canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        if is_left_to_right:
            rvec_rel, _ = cv2.Rodrigues(R_rel)
            t_vec_rel = t_rel.reshape(3, 1)
            color = '#FF00FF' # 洋紅色虛線代表左圖投影至右圖
        else:
            R_rel_back = R_rel.T
            t_vec_rel = -R_rel.T @ t_rel.reshape(3, 1)
            rvec_rel, _ = cv2.Rodrigues(R_rel_back)
            color = '#FF8800' # 橘色虛線代表右圖投影至左圖
            
        label_prefix = "Reproj_L2R" if is_left_to_right else "Reproj_R2L"
        for mid, pts in corners_src.items():
            ok, rvec, tvec = cv2.solvePnP(canon, pts, KL, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok: continue
            R_src, _ = cv2.Rodrigues(rvec)
            P_src = (R_src @ canon.T).T + tvec.T # (4, 3) 3D點在 source 坐標系下
            
            pts_reproj, _ = cv2.projectPoints(P_src, rvec_rel, t_vec_rel, KL, np.zeros(5))
            pts_reproj = pts_reproj.reshape(4, 2)
            
            # 計算與 target 視角真實偵測角點的誤差
            err_str = "N/A"
            if is_left_to_right:
                if mid in current_cand['cornersB']:
                    err = np.linalg.norm(pts_reproj - current_cand['cornersB'][mid], axis=1)
                    err_str = f"{np.mean(err):.2f} px"
            else:
                if mid in current_cand['cornersA']:
                    err = np.linalg.norm(pts_reproj - current_cand['cornersA'][mid], axis=1)
                    err_str = f"{np.mean(err):.2f} px"
            log_and_print(f"📊 [{label_prefix}] 標籤 ID: {mid} | 與真實檢測角點的平均重投影誤差: {err_str}")
            
            p = np.vstack((pts_reproj, pts_reproj[0]))
            l, = ax_target.plot(p[:,0], p[:,1], color=color, linestyle='--', lw=1.5, alpha=0.8, zorder=3)
            ax_target.reproj_art.append(l)
            
            for i in range(4):
                pt, = ax_target.plot(pts_reproj[i,0], pts_reproj[i,1], color=color, marker='+', markersize=5, zorder=3)
                ax_target.reproj_art.append(pt)

    draw_reprojected_aruco(ax_B, current_cand['cornersA'], R_r, t_r, is_left_to_right=True)
    draw_reprojected_aruco(ax_A, current_cand['cornersB'], R_r, t_r, is_left_to_right=False)

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
            f.write("\n==================================================\n\n")
        print(f"💾 已初始化分析日誌至: {init_txt_path}")
    except Exception as e:
        print(f"❌ 初始化日誌失敗: {e}")

    scatter_A = ax_A.scatter([], [], s=80, c='red', marker='x', zorder=5)
    scatter_A_reproj = ax_A.scatter([], [], s=120, facecolors='none', edgecolors='#FF00FF', marker='o', linestyle='--', lw=1.5, zorder=6)
    scatter_B = ax_B.scatter([], [], s=80, c='lime', marker='x', zorder=5)
    scatter_B_reproj = ax_B.scatter([], [], s=120, facecolors='none', edgecolors='#FF00FF', marker='o', linestyle='--', lw=1.5, zorder=6)
    scatter_grad_inject = ax_A.scatter([], [], s=15, c='cyan', alpha=0.6, zorder=4)
    scatter_grad_match = ax_B.scatter([], [], s=15, c='cyan', alpha=0.6, zorder=4)
    scatter_all_sift_B = ax_B.scatter([], [], s=2, c='yellow', alpha=0.3, zorder=3)
    epi_line, = ax_B.plot([], [], 'yellow', lw=1, alpha=0.6, zorder=4)
    sift_rect = Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='magenta', facecolor='none', linestyle='--', alpha=0.8, zorder=4)
    ax_B.add_patch(sift_rect)
    sift_rect.set_visible(False)
    sift_rect_center, = ax_B.plot([], [], '+', color='magenta', markersize=12, markeredgewidth=1.5, zorder=5)
    sift_rect_center.set_visible(False)
    # HUD 風格的文字面板
    depth_text = fig.text(0.53, 0.45, "", transform=fig.transFigure, color='white', fontweight='bold', fontsize=16, bbox=dict(facecolor='#121212', alpha=0.7, edgecolor='#00FFFF', lw=1))
    fps_text = ax_A.text(0.01, 0.97, "FPS: --", transform=ax_A.transAxes,
                         color='#00FF00', fontsize=10, fontweight='bold', va='top',
                         bbox=dict(facecolor='#121212', alpha=0.6, edgecolor='none'), zorder=10)
                         
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
        
    pose_status_text = ax_A.text(0.0, 1.02, pose_status_str, transform=ax_A.transAxes,
                                 color=pose_status_color, fontsize=10, fontweight='bold', va='bottom',
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


    # 勾選框面板 (改成兩行排列，每顆獨立以利排版)
    # 由於 Matplotlib 的 CheckButtons 在不同版本間極難著色，這裡改用標準 Button 來模擬勾選框！
    ax_c1 = fig.add_axes([0.05, 0.92, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c2 = fig.add_axes([0.17, 0.92, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c3 = fig.add_axes([0.29, 0.92, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c4 = fig.add_axes([0.05, 0.86, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c5 = fig.add_axes([0.17, 0.86, 0.11, 0.04], facecolor='#1E1E1E')
    ax_c6 = fig.add_axes([0.29, 0.86, 0.11, 0.04], facecolor='#1E1E1E')
    
    # 建立標準按鈕，文字開頭加上 [X] 或 [ ] 代表勾選狀態
    btn_opt_style = dict(color='#1A1A1A', hovercolor='#333333')
    c1 = Button(ax_c1, "[X] 嚴格精細匹配", **btn_opt_style)
    c2 = Button(ax_c2, "[X] 梯度 SIFT 匹配", **btn_opt_style)
    c3 = Button(ax_c3, "[X] 強制極線對齊", **btn_opt_style)
    c4 = Button(ax_c4, "[X] 啟用 ECC 精修", **btn_opt_style)
    c5 = Button(ax_c5, "[ ] 手動匹配模式", **btn_opt_style)
    c6 = Button(ax_c6, "[X] 啟用 CLAHE 增強" if ENABLE_CLAHE_DEFAULT else "[ ] 啟用 CLAHE 增強", **btn_opt_style)
    
    view_state = {'precise': True, 'grad_sift': True, 'enforce_epi': True, 'ecc': True, 'manual': False,
                  'use_hamming': False, 'enable_clahe': ENABLE_CLAHE_DEFAULT,
                  'manual_pt_A': None, 'lines': [], 'grad_lines': [], 'show_grad_lines': False,
                  'highlighted_grad_line': None, 'highlighted_grad_line_artist': None,
                  'grad_data': None, 'restart': False}  # grad_data = {'ptsA': ndarray, 'ptsB': ndarray}

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
        if last_click:
            do_measure(last_click[0], last_click[1])
            
    radio_mode.on_clicked(on_mode_change)

    # 統一設定文字顏色為白色，並將按鈕外框設為白色
    for c in [c1, c2, c3, c4, c5, c6]:
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
        return _on_opt
        
    c1.on_clicked(make_on_opt(c1, 'precise', "嚴格精細匹配"))
    c2.on_clicked(make_on_opt(c2, 'grad_sift', "梯度 SIFT 匹配"))
    c3.on_clicked(make_on_opt(c3, 'enforce_epi', "強制極線對齊"))
    c4.on_clicked(make_on_opt(c4, 'ecc', "啟用 ECC 精修"))
    c5.on_clicked(make_on_opt(c5, 'manual', "手動匹配模式"))
    c6.on_clicked(make_on_opt(c6, 'enable_clahe', "啟用 CLAHE 增強"))

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
                                  axesA=ax_A, axesB=ax_B, color="cyan", lw=0.8, alpha=0.4, zorder=4)
            ax_B.add_artist(con)
            view_state['grad_lines'].append(con)
        
        if highlight_idx is not None and 0 <= highlight_idx < len(ptsA):
            hl = ConnectionPatch(xyA=ptsA[highlight_idx], xyB=ptsB[highlight_idx], coordsA="data", coordsB="data",
                                 axesA=ax_A, axesB=ax_B, color="red", lw=2.5, alpha=1.0, zorder=10)
            ax_B.add_artist(hl)
            view_state['highlighted_grad_line_artist'] = hl
        view_state['highlighted_grad_line'] = highlight_idx


    # ---- 執行緒通訊佇列 ----
    calc_request_q = queue.Queue(maxsize=1)   # 最多排 1 個請求，避免積壓
    calc_result_q  = queue.Queue()
    calc_busy      = threading.Event()        # 用於標記背景正在計算中

    measure_results = {}
    last_click = None

    # ---- 純計算（可在背景執行緒安全呼叫，不觸碰 Matplotlib）----
    def compute_measure(u, v, snap_cand, snap_imgA_gray, snap_view_state, manual_match_pt=None):
        """純計算版 do_measure，回傳結果 dict，不更新任何 UI 元件。"""
        cand = snap_cand
        m_pt, method, neighbors = None, "", []
        g_ptsA, g_ptsB, g_kptsB, g_rect = None, None, None, None
        trajectory_res = None

        if not cand.get('pose_valid', True):
            print(f"❌ [測量失敗] 當前候選影格位姿無效 (pose_valid == False)，原因: {cand.get('pose_info', '未知')}")
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': None, 'g_rect': None,
                    'fail_reason': '未偵測到 ArUco', 'u': u, 'v': v, 'trajectory': None}
        if cand.get('baseline', 0.0) < MIN_BASELINE_MM:
            print(f"❌ [測量失敗] 基準線視差不足 ({cand.get('baseline', 0.0):.2f} mm < {MIN_BASELINE_MM} mm)")
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': None, 'g_rect': None,
                    'fail_reason': f'視差不足(<{MIN_BASELINE_MM}mm)', 'u': u, 'v': v, 'trajectory': None}

        # 1. 只有在需要匹配點的模式下進行匹配
        if MEASURE_MODE in ("dual_direct", "multi_dedrift"):
            if manual_match_pt is not None:
                m_pt, method = manual_match_pt, "手動點選"

            if m_pt is None:
                for mid, cA in cand['cornersA'].items():
                    d = np.linalg.norm(cA - np.array([u, v]), axis=1)
                    if np.min(d) < 10:
                        best_idx = np.argmin(d)
                        u, v = cA[best_idx] # 🌟 同步校正左圖座標為精確角點
                        m_pt, method = cand['cornersB'][mid][best_idx], "ArUco"
                        break
                if m_pt is None and snap_view_state['grad_sift']:
                    ui, vi = int(round(u)), int(round(v))
                    sobel_range = 18
                    patch_g = snap_imgA_gray[max(0,vi-sobel_range):min(snap_imgA_gray.shape[0],vi+sobel_range+1),
                                             max(0,ui-sobel_range):min(snap_imgA_gray.shape[1],ui+sobel_range+1)]
                    if patch_g.size > 0:
                        gx, gy = cv2.Sobel(patch_g, cv2.CV_32F, 1, 0), cv2.Sobel(patch_g, cv2.CV_32F, 0, 1)
                        mag = cv2.sqrt(gx**2 + gy**2); flat = mag.flatten()
                        idx_g = np.argsort(flat)[-min(len(flat), 100):]
                        kpts_inj = [cv2.KeyPoint(float(max(0,ui-sobel_range)+px), float(max(0,vi-sobel_range)+py), 31.0)
                                    for py, px in [divmod(idx, patch_g.shape[1]) for idx in idx_g]]
                        if snap_view_state.get('use_hamming', False):
                            _, des_inj = orb.compute(snap_imgA_gray, kpts_inj)
                        else:
                            _, des_inj = sift.compute(snap_imgA_gray, kpts_inj)
                        u_exp, v_exp = u, v
                        if cand['plane_n'] is not None:
                            H_AB = cand['K_R'] @ (cand['R_rel'] + (cand['t_rel'] @ cand['plane_n'].reshape(1,3))/np.dot(cand['plane_n'], cand['plane_c'])) @ np.linalg.inv(KL)
                            pt_exp = H_AB @ np.array([u, v, 1.0]); u_exp, v_exp = pt_exp[0]/pt_exp[2], pt_exp[1]/pt_exp[2]
                        rad = 30; uei, vei = int(round(u_exp)), int(round(v_exp))
                        u0, u1, v0, v1 = max(0, uei-rad), min(snap_imgA_gray.shape[1], uei+rad), max(0, vei-rad), min(snap_imgA_gray.shape[0], vei+rad)
                        patch_r = cand['gray'][v0:v1, u0:u1]
                        if patch_r.size > 0:
                            gxr, gyr = cv2.Sobel(patch_r, cv2.CV_32F, 1, 0), cv2.Sobel(patch_r, cv2.CV_32F, 0, 1)
                            magr = cv2.sqrt(gxr**2 + gyr**2); flatr = magr.flatten()
                            idx_gr = np.argsort(flatr)[-min(len(flatr), 400):]
                            kpts_r = [cv2.KeyPoint(float(u0+px), float(v0+py), 31.0)
                                      for py, px in [divmod(idx, patch_r.shape[1]) for idx in idx_gr]]
                            g_kptsB = [kp.pt for kp in kpts_r]
                            g_rect = (u0, v0, u1-u0, v1-v0)
                            if snap_view_state.get('use_hamming', False):
                                _, des_r = orb.compute(cand['gray'], kpts_r)
                                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                                matches = bf.match(des_inj, des_r)
                                good = [m for m in matches if m.distance < 100]
                            else:
                                _, des_r = sift.compute(cand['gray'], kpts_r)
                                bf = cv2.BFMatcher(cv2.NORM_L2)
                                matches = bf.match(des_inj, des_r)
                                good = [m for m in matches if m.distance < 450]
                            
                            pts_info = []
                            for m in good:
                                pL, pR = np.array(kpts_inj[m.queryIdx].pt), np.array(kpts_r[m.trainIdx].pt)
                                if np.linalg.norm(pL - np.array([u, v])) < 50:
                                    pts_info.append({'pL': pL, 'pR': pR, 'off': pR - pL})
                            if len(pts_info) >= 3:
                                offs = np.array([x['off'] for x in pts_info]); med_off = np.median(offs, axis=0)
                                pts_info = [x for x in pts_info if np.linalg.norm(x['off'] - med_off) < 15]
                            if pts_info:
                                ptsA_m = np.array([x['pL'] for x in pts_info], dtype=np.float32)
                                ptsB_m = np.array([x['pR'] for x in pts_info], dtype=np.float32)
                                mapped = None
                                if len(ptsA_m) >= 6:
                                    H_local, _ = cv2.findHomography(ptsA_m, ptsB_m, cv2.RANSAC, 4.0)
                                    if H_local is not None:
                                        pt_h = H_local @ np.array([u, v, 1.0])
                                        mapped = np.array([pt_h[0]/pt_h[2], pt_h[1]/pt_h[2]])
                                if mapped is None and len(ptsA_m) >= 3:
                                    M_local, _ = cv2.estimateAffinePartial2D(ptsA_m, ptsB_m, method=cv2.RANSAC, ransacReprojThreshold=4.0)
                                    if M_local is not None:
                                        pt_a = M_local @ np.array([u, v, 1.0])
                                        mapped = pt_a[:2]
                                if mapped is None:
                                    wts = 1.0 / (np.sum((ptsA_m - np.array([u, v]))**2, axis=1) + 1e-5)
                                    mapped = np.array([u, v]) + np.sum((ptsB_m - ptsA_m) * wts[:, np.newaxis], axis=0) / np.sum(wts)
                                m_pt, method = mapped, "Grad-SIFT"
                                g_ptsA, g_ptsB = ptsA_m, ptsB_m
                if m_pt is None and snap_view_state['precise']:
                    res_p = find_precise_match(snap_imgA_gray, cand['gray'], (u, v), cand['F'],
                                               KL, cand['K_R'], cand['R_rel'], cand['t_rel'],
                                               cand['plane_n'], cand['plane_c'])
                    if res_p: m_pt, method = np.array(res_p), "Precise"
            
            m_pt_raw = m_pt.copy() if m_pt is not None else None

            if m_pt is not None and snap_view_state['enforce_epi'] and method != "ArUco":
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                      m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])
                    method += "+極線對齊"
            if m_pt is not None and snap_view_state['ecc']:
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
            if m_pt is not None and snap_view_state['enforce_epi'] and method != "ArUco":
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                      m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])

        # 2. 分流計算三維點
        d_val, p3d_val, p3d_w_val, fail_reason = None, None, None, ""
        p3d = None
        
        if MEASURE_MODE == "multi_dedrift":
            if m_pt is None:
                print("⚠️ [閉環光流] 雙幀直接匹配失敗，無法取得閉環真值點，退回雙幀直接模式。")
            else:
                trajectory = track_feature_and_verify(
                    video_data['all_frames'], video_data['idx_B'], video_data['idx_A'],
                    (u, v), video_data['valid_poses'], KL, distL
                )
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
            trajectory = track_feature_and_verify(
                video_data['all_frames'], video_data['idx_B'], video_data['idx_A'],
                (u, v), video_data['valid_poses'], KL, distL
            )
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
            p3d = triangulate_point_3d((u, v), m_pt, KL, cand['K_R'], cand['R_rel'], cand['t_rel'])
        elif p3d is None:
            fail_reason = "無匹配點"
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
                    p_dist = abs(np.dot(cand['plane_n'], p3d - cand['plane_c']))
                    p_dist_str = f"{p_dist:.2f} mm"
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
                fail_reason = "無匹配點"
            print(f"👉 [深度計算] 左圖座標: ({u:.1f}, {v:.1f}) | 右圖匹配座標: N/A | 匹配方式: N/A")
            print(f"   [計算失敗] 原因: {fail_reason}")

        depth_z = p3d_val[2] if p3d_val is not None else None
        reproj_err = None
        if p3d_val is not None and m_pt is not None:
            rvec_rel, _ = cv2.Rodrigues(cand['R_rel'])
            pt_reproj_B, _ = cv2.projectPoints(p3d_val.reshape(1, 1, 3), rvec_rel, cand['t_rel'], KL, np.zeros(5))
            pt_reproj_B = pt_reproj_B.reshape(2)
            reproj_err = float(np.linalg.norm(m_pt - pt_reproj_B))

        return {'pt': m_pt, 'pt_raw': m_pt_raw, 'p3d': p3d_val, 'p3d_w': p3d_w_val, 'd': d_val, 'depth': depth_z, 'error': reproj_err, 'method': method, 'neighbors': neighbors,
                'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_kptsB': g_kptsB, 'g_rect': g_rect,
                'fail_reason': fail_reason, 'u': u, 'v': v, 'trajectory': trajectory_res}

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
        nonlocal last_click; last_click = (u, v)
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
        
        all_cands = [current_cand] + extra_candidates_list
        res_list = []
        for cand in all_cands:
            res_c = compute_measure(u, v, cand, left_img_gray, snap_vs, manual_match_pt)
            res_c['cand_idx'] = cand['idx']
            if res_c.get('d') is not None and res_c.get('p3d') is not None:
                res_list.append(res_c)
                
        if not res_list:
            res = compute_measure(u, v, current_cand, left_img_gray, snap_vs, manual_match_pt)
            res['cand_idx'] = current_cand['idx']
        else:
            best_res = next((r for r in res_list if r['cand_idx'] == current_cand['idx']), res_list[0])
            all_p3d = np.array([r['p3d'] for r in res_list])
            all_d = np.array([r['d'] for r in res_list])
            
            avg_p3d = np.mean(all_p3d, axis=0)
            avg_d = np.mean(all_d)
            
            avg_depth = avg_p3d[2]
            avg_error = np.mean([r['error'] for r in res_list if r.get('error') is not None]) if any(r.get('error') is not None for r in res_list) else 0.0
            
            all_p3d_w = [r['p3d_w'] for r in res_list if r.get('p3d_w') is not None]
            avg_p3d_w = np.mean(all_p3d_w, axis=0) if all_p3d_w else None
            
            res = dict(best_res)
            res['p3d'] = avg_p3d
            res['d'] = avg_d
            res['depth'] = avg_depth
            res['error'] = avg_error
            res['p3d_w'] = avg_p3d_w
            res['multi_res'] = res_list
            
            print(f"📊 [多對平均深度] 左圖 F{current_cand['idx']} 與最多 {len(all_cands)} 個右圖進行計算：")
            for r in res_list:
                is_best = " (最優)" if r['cand_idx'] == current_cand['idx'] else ""
                print(f"  - 右圖 F{r['cand_idx']}{is_best}: 深度 = {r['d']:.2f} mm, 3D = [{r['p3d'][0]:.2f}, {r['p3d'][1]:.2f}, {r['p3d'][2]:.2f}]")
            print(f"  ➡️ 融合平均結果 (共 {len(res_list)} 組成功): 深度 = {avg_d:.2f} mm, 3D = [{avg_p3d[0]:.2f}, {avg_p3d[1]:.2f}, {avg_p3d[2]:.2f}]")
        
        if custom_plane_mode:
            if res.get('p3d') is not None:
                custom_plane_pts_3d.append(res['p3d'])
                custom_plane_pts_2d.append((res['u'], res['v']))
                c_pt, = ax_A.plot(res['u'], res['v'], 'mo', markersize=6, zorder=5)
                t_lbl = ax_A.text(res['u'] + 5, res['v'] - 5, f"P{len(custom_plane_pts_3d)}", 
                                  color='magenta', fontsize=9, fontweight='bold', zorder=5)
                custom_plane_artists.extend([c_pt, t_lbl])
                redraw_custom_plane_poly()
                btn_custom_plane.label.set_text(f"完成擬合 ({len(custom_plane_pts_3d)})")
                print(f"🎯 自訂平面已新增點 P{len(custom_plane_pts_3d)}: (u, v)=({res['u']:.1f}, {res['v']:.1f}), 3D={res['p3d']}")
            else:
                print("❌ 點選點之深度計算無效，無法加入自訂平面點！")
                
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
        
        apply_measure_result(res, np.mean(all_d) if all_d else None, summary)


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
        res = measure_results.get(current_cand['idx'], {'pt': None, 'neighbors': [], 'p3d': None, 'g_ptsA': None})
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
        scatter_grad_inject.set_offsets(np.empty((0,2)))
        scatter_grad_match.set_offsets(np.empty((0,2)))
        scatter_all_sift_B.set_offsets(np.empty((0,2)))
        
        if res.get('g_kptsB') is not None:
            scatter_all_sift_B.set_offsets(res['g_kptsB'])
            sift_rect.set_bounds(*res['g_rect'])
            sift_rect.set_visible(True)
            # 更新 Rect 中心標記
            rx, ry, rw, rh = res['g_rect']
            sift_rect_center.set_data([rx + rw/2], [ry + rh/2])
            sift_rect_center.set_visible(True)
            
            if res.get('g_ptsA') is not None:
                scatter_grad_inject.set_offsets(res['g_ptsA'])
                scatter_grad_match.set_offsets(res['g_ptsB'])
                view_state['grad_data'] = {'ptsA': res['g_ptsA'], 'ptsB': res['g_ptsB']}
                redraw_grad_lines(None)  # 初始無高亮
            else:
                scatter_grad_inject.set_offsets(np.empty((0,2)))
                scatter_grad_match.set_offsets(np.empty((0,2)))
        elif current_cand.get('pose_valid', False):
            if current_cand.get('plane_n') is not None:
                d_plane = np.dot(current_cand['plane_n'], current_cand['plane_c'])
                if abs(d_plane) > 1e-6:
                    H_AB = current_cand['K_R'] @ (current_cand['R_rel'] + (current_cand['t_rel'] @ current_cand['plane_n'].reshape(1, 3)) / d_plane) @ np.linalg.inv(KL)
                    pt_exp = H_AB @ np.array([u, v, 1.0])
                    u_exp, v_exp = pt_exp[0]/pt_exp[2], pt_exp[1]/pt_exp[2]
                    
                    if 0 <= u_exp < w and 0 <= v_exp < h:
                        rad = 30
                        sift_rect.set_bounds(u_exp - rad, v_exp - rad, rad*2, rad*2)
                        sift_rect.set_visible(True)
                        sift_rect_center.set_data([u_exp], [v_exp])
                        sift_rect_center.set_visible(True)
                    else:
                        print(f"⚠️ [預估搜尋框繪製失敗] 單應性投影預測點 ({u_exp:.1f}, {v_exp:.1f}) 超出影像邊界。")
                else:
                    print("⚠️ [預估搜尋框繪製失敗] 平面距離 d_plane 趨近於 0，無法計算單應性。")
            else:
                print("⚠️ [預估搜尋框繪製跳過] 因為世界平面參數 (plane_n) 為 None，無法計算平面單應性投影。")
                

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
            if res['p3d'] is not None and current_cand['plane_n'] is not None:
                p_dist = (np.dot(current_cand['plane_n'], res['p3d'] - current_cand['plane_c']))
                if auto_calc_active:
                    plane_dist_history.append(p_dist)
                    p_dist_str = f"\n傷口高度: {np.mean(plane_dist_history):.1f}mm"
                else:
                    plane_dist_history.clear()
                    p_dist_str = f"\n傷口高度: {p_dist:.1f}mm"
            
            # 高度差計算
            h_diff_str = ""
            if res['p3d'] is not None and current_cand.get('plane_n') is not None:
                plane_n = current_cand['plane_n']
                plane_c = current_cand['plane_c']
                signed_dist = np.dot(plane_n, res['p3d'] - plane_c)
                
                # 計算校正法向量後的 Z 座標
                if current_cand.get('R_rect') is not None:
                    R_rect = current_cand['R_rect']
                    # 將 3D 點投射到 rectification 坐標系
                    p3d_rect = R_rect @ res['p3d'].reshape(3, 1)
                    # 傷口表面之高度值 (對 Z 進行負號反向)
                    z_surface = -p3d_rect[2, 0]
                    # 加入補償值
                    z_final = z_surface + wound_z_offset
                    h_diff_str = f"\n3D高度(Z): {z_final:.1f}mm"
                    
            if res['depth'] is not None:
                # 這裡的 res['depth'] 就是左相機坐標系下的 z 座標
                #main_text = f"深度: {res['depth']:.1f}mm{p_dist_str}{h_diff_str}\n誤差: {res['error']:.3f}px\n配對: {res['method']}\n外參來源: {pose_info_str}"
                main_text = f"相機與傷口(點選處)的距離: {res['depth']:.1f}mm{p_dist_str}{h_diff_str}\n"
            
            
            else:
                main_text = f"深度: 計算失敗\n外參來源: {pose_info_str}"
            
            # 連續計算模式下的 pose 繪製
            if current_cand.get('all_aruco_poses') and current_cand.get('main_pose_ref_id') is not None:
                redraw_all_aruco_poses(current_cand['all_aruco_poses'], current_cand['main_pose_ref_id'])
        else:
            scatter_B.set_offsets(np.empty((0,2)))
            scatter_B_reproj.set_offsets(np.empty((0,2)))
            scatter_A_reproj.set_offsets(np.empty((0,2)))
            epi_line.set_data([], [])
            main_text = f"無效點({res.get('fail_reason', '無匹配點')})\n外參來源: {pose_info_str}"
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
                    depth_text.set_text("手動模式：請在右圖極線上點選對應點")
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
    btn_norm_toggle = Button(ax_btn_norm, '切換 HAMMING', **btn_style)
    
    ax_btn_calc = fig.add_axes([0.58, 0.86, 0.08, 0.04])
    btn_calc = Button(ax_btn_calc, "單次計算深度", **btn_style)
    
    ax_btn_auto_calc = fig.add_axes([0.68, 0.86, 0.08, 0.04])
    btn_auto_calc = Button(ax_btn_auto_calc, "連續計算: 關", **btn_style)
    
    ax_btn_grad = fig.add_axes([0.78, 0.86, 0.08, 0.04])
    btn_grad_toggle = Button(ax_btn_grad, '顯示梯度 SIFT 連線', **btn_style)
    
    ax_btn_custom_plane = fig.add_axes([0.88, 0.86, 0.08, 0.04])
    btn_custom_plane = Button(ax_btn_custom_plane, "自訂平面擬合", **btn_style)
    
    ax_btn_return_menu = fig.add_axes([0.88, 0.80, 0.08, 0.04])
    btn_return_menu = Button(ax_btn_return_menu, "返回主選單", **btn_style)
    
    # 建立 TextBox 用於傷口高度補償
    wound_z_offset = 6.0
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
    for b in [btn_lock_L, btn_lock_R, btn_hide_R, btn_norm_toggle, btn_calc, btn_auto_calc, btn_grad_toggle, btn_custom_plane, btn_return_menu]:
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
    for b in [btn_norm_toggle, btn_grad_toggle, btn_custom_plane]:
        b.ax.patch.set_edgecolor('#555555')
        
    # 4. 導覽/返回選單類：使用翡翠綠 (#28A745)
    btn_return_menu.ax.patch.set_edgecolor('#28A745')
        
    btn_grad_toggle.label.set_fontsize(7) # 特長文字微調
    
    def on_grad_toggle(event):
        view_state['show_grad_lines'] = not view_state['show_grad_lines']
        btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線' if view_state['show_grad_lines'] else '顯示梯度 SIFT 連線')
        redraw_grad_lines(view_state.get('highlighted_grad_line'))
        request_blit_refresh()
        
    def on_norm_toggle(event):
        view_state['use_hamming'] = not view_state['use_hamming']
        btn_norm_toggle.label.set_text('使用 L2' if view_state['use_hamming'] else '切換 HAMMING')
        request_blit_refresh()
        
    def on_return_menu(event):
        view_state['restart'] = True
        plt.close(fig)
        print("🔄 正在關閉目前量測介面並返回影片來源選單...")
    
    def on_hide_R(event):
        visible = ax_B.get_visible()
        ax_B.set_visible(not visible)
        btn_hide_R.label.set_text("顯示右圖" if visible else "隱藏右圖")
        if visible:
            # 隱藏右圖時，抬高到原本右圖的中間位置
            depth_text.set_position((0.53, 0.45))
        else:
            # 顯示右圖時，調整到右圖下方的位置
            depth_text.set_position((0.53, 0.05))
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
        nonlocal live_L, locked_L, has_set_L, reset_pose_history
        live_L = not live_L
        if not live_L:
            if len(frame_buffer) > 0:
                locked_L = frame_buffer[-1].copy()
                has_set_L = True
                btn_lock_L.label.set_text("解鎖左圖")
                print("🔒 左圖已鎖定當前畫面")
                if has_set_R and not live_R:
                    on_calc(None)
        else:
            btn_lock_L.label.set_text("鎖定左圖")
            print("🔓 左圖恢復 Live")
        reset_pose_history = True
        request_blit_refresh()
        
    def on_lock_R(event):
        nonlocal live_R, locked_R, has_set_R, reset_pose_history
        live_R = not live_R
        if not live_R:
            if len(frame_buffer) > 0:
                locked_R = frame_buffer[-1].copy()
                has_set_R = True
                btn_lock_R.label.set_text("解鎖右圖")
                print("🔒 右圖已鎖定當前畫面")
                if not live_L:
                    on_calc(None)
        else:
            btn_lock_R.label.set_text("鎖定右圖")
            print("🔓 右圖恢復 Live")
            
            # 解鎖右圖時清除自訂平面與選點
            nonlocal custom_plane_mode, custom_plane_n, custom_plane_c, custom_plane_fitted
            custom_plane_mode = False
            custom_plane_fitted = False
            custom_plane_pts_3d.clear()
            custom_plane_pts_2d.clear()
            custom_plane_n = None
            custom_plane_c = None
            for art in custom_plane_artists:
                try: art.remove()
                except: pass
            custom_plane_artists.clear()
            redraw_custom_plane_poly()
            btn_custom_plane.label.set_text("自訂平面擬合")
            btn_custom_plane.ax.patch.set_facecolor('#1A1A1A')
            btn_custom_plane.ax.patch.set_edgecolor('#555555')
            
        reset_pose_history = True
        request_blit_refresh()
        
    def on_custom_plane(event):
        nonlocal custom_plane_mode, custom_plane_n, custom_plane_c, custom_plane_fitted, auto_calc_active
        
        # 1. 檢查先決條件：必須鎖定右圖，且外參有效
        if live_R:
            print("⚠️ 請先鎖定右圖再開始自訂平面選點！")
            depth_text.set_text("提示：請先鎖定右圖再進行自訂平面選點")
            request_blit_refresh()
            return
        if not current_cand.get('pose_valid', False):
            print("⚠️ 外參/Baseline無效，無法計算3D座標，請先確保 ArUco 定位成功！")
            depth_text.set_text("提示：定位無效，請確保偵測到 ArUco")
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
                
            btn_custom_plane.label.set_text("完成擬合 (0)")
            btn_custom_plane.ax.patch.set_facecolor('#8A2BE2') # 變為紫羅蘭色
            btn_custom_plane.ax.patch.set_edgecolor('#8A2BE2')
            print("🎯 已進入「自訂平面選點模式」，請在左圖點選至少 3 個點...")
            depth_text.set_text("自訂平面模式：請在左圖點選至少 3 個點")
            request_blit_refresh()
        else:
            # 3. 按下按鈕完成或取消擬合
            if len(custom_plane_pts_3d) == 0:
                # 0 個點，取消此模式並清除平面
                custom_plane_mode = False
                custom_plane_fitted = False
                btn_custom_plane.label.set_text("自訂平面擬合")
                btn_custom_plane.ax.patch.set_facecolor('#1A1A1A')
                btn_custom_plane.ax.patch.set_edgecolor('#555555')
                print("❌ 已取消自訂平面擬合，並清除自訂平面。")
                if last_click:
                    do_measure(last_click[0], last_click[1])
                else:
                    depth_text.set_text("已清除自訂平面")
                    request_blit_refresh()
                return
                
            if len(custom_plane_pts_3d) < 3:
                print(f"⚠️ 點數不足 (當前僅 {len(custom_plane_pts_3d)} 個點)，擬合平面至少需要 3 個點！")
                depth_text.set_text(f"錯誤：點數不足 ({len(custom_plane_pts_3d)}/3)，請繼續選點")
                request_blit_refresh()
                return
                
            # 4. SVD 擬合平面
            pts = np.array(custom_plane_pts_3d)
            c = np.mean(pts, axis=0)
            _, _, Vt = np.linalg.svd(pts - c)
            n = Vt[-1]
            if np.dot(n, c) > 0: n = -n # 確保法向量朝向相機
            
            custom_plane_n = n
            custom_plane_c = c
            custom_plane_fitted = True
            custom_plane_mode = False
            
            btn_custom_plane.label.set_text("自訂平面擬合")
            btn_custom_plane.ax.patch.set_facecolor('#1A1A1A')
            btn_custom_plane.ax.patch.set_edgecolor('#555555')
            print(f"✅ 自訂平面擬合成功！")
            print(f"  - 擬合點數: {len(pts)}")
            print(f"  - 平面中心: {c}")
            print(f"  - 平面法向: {n}")
            
            # 重新計算當前選取點，以獲得與新平面的距離
            if last_click:
                do_measure(last_click[0], last_click[1])
            else:
                depth_text.set_text(f"自訂平面擬合成功！點數: {len(pts)}")
                request_blit_refresh()
        
    def on_calc(event):
        if last_click:
            do_measure(last_click[0], last_click[1])
        else:
            do_measure(active_u, active_v)
    btn_lock_L.on_clicked(on_lock_L)
    btn_lock_R.on_clicked(on_lock_R)
    btn_calc.on_clicked(on_calc)
    btn_auto_calc.on_clicked(on_auto_calc)
    btn_hide_R.on_clicked(on_hide_R)
    btn_grad_toggle.on_clicked(on_grad_toggle)
    btn_norm_toggle.on_clicked(on_norm_toggle)
    btn_custom_plane.on_clicked(on_custom_plane)
    btn_return_menu.on_clicked(on_return_menu)

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
        # 依前處理開關，動態切換顯示畫面（使肉眼可見差異）
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
            
        im_A.set_data(disp_A)
        im_B.set_data(disp_B)

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
        ax_A.draw_artist(scatter_grad_inject)
        for a in custom_plane_artists:
            ax_A.draw_artist(a)
        
        if ax_B.get_visible():
            ax_B.draw_artist(im_B)
            ax_B.draw_artist(scatter_B)
            ax_B.draw_artist(scatter_B_reproj)
            ax_B.draw_artist(scatter_grad_match)
            ax_B.draw_artist(scatter_all_sift_B)
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
        ax_A.draw_artist(pose_status_text)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        _time.sleep(0.01)

    return view_state.get('restart', False)

if __name__ == "__main__":
    while True:
        should_restart = main()
        if not should_restart:
            break
        import matplotlib.pyplot as plt
        plt.close('all')
