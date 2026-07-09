import os

import cv2
import numpy as np

from .aruco_pose import average_rotations_svd, compute_global_plane as _compute_global_plane
from .camera_preprocess import preprocess_gray
from .perf_timer import StageTimer

RECORD_SAVE_DIR = "test_video_Zebra"
MIN_BASELINE_MM = 8.0
MAX_BASELINE_MM = 220.0
IDEAL_BASELINE_MM = 45.0
PAIR_SCORE_REPROJ_W = 1.00
PAIR_SCORE_BASELINE_W = 0.18
PAIR_SCORE_BLUR_W = 0.18
PAIR_SCORE_COVER_W = 0.12
PAIR_SCORE_MARKER_W = 0.08

# ---- 特徵極線驗證與混合 RT 精修 ----
PAIR_SCORE_EPI_W = 0.5              # 配對評分: 特徵極線殘差權重 (px)
PAIR_EPI_TOPK = 12                  # 以「幀對」計的特徵極線驗證數量 (每幀對做一次 SIFT 匹配)
PAIR_EPI_OK_PX = 0.8                # 配對提前收斂的特徵極線殘差門檻 (px, 全模式)
PAIR_EPI_EXTRA_PX = 1.5             # 次佳對接受的特徵極線殘差上限 (px, 全模式)
PAIR_TOPK_MAX_PER_START = 2         # top-K 多樣性: 同一起始幀最多幀對數
PAIR_TOPK_MAX_PER_END = 4           # top-K 多樣性: 同一結尾幀最多幀對數
ENABLE_FEATURE_RT_REFINE = True     # 選定配對後用 SIFT + Essential matrix 精修 R 與 t 方向 (尺度 |t| 保留 ArUco 解)
FEATURE_MATCH_RATIO = 0.75          # SIFT ratio test 閾值
FEATURE_MIN_MATCHES = 25            # 精修所需最少匹配數 / recoverPose 內點數
FEATURE_E_RANSAC_THRESH_PX = 0.75   # findEssentialMat RANSAC 極線距離閾值 (px)
FEATURE_ROT_DIFF_MAX_DEG = 10.0     # 特徵解與 ArUco 解允許的最大旋轉差 (超過視為異常，保留 ArUco)
FEATURE_MARKER_EPI_MARGIN_PX = 0.3  # 採用混合解時，標籤角點極線殘差允許的最大退步量 (px)
FEATURE_MAX_KEYPOINTS = 2000        # 特徵精修用 SIFT keypoint 上限 (控制匹配耗時)



def log_and_print(msg):
    print(msg)


def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    return _compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=log_and_print)


def analyze_video_frames(video_path, start_n, end_n, K_L, dist_L, mtx_L, marker_size_mm, select_mode="average", range_mode="fixed", progress_callback=None):
    timer = StageTimer("影片分析明細")
    if progress_callback:
        progress_callback(2, "階段 1/6：載入影片...")
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
            progress_callback(load_percent, f"階段 1/6：載入影片 ({len(frames)}/{total_frames})...")
            
    cap.release()
    if progress_callback:
        progress_callback(12, "階段 1/6：載入完成")
    
    if len(frames) == 0:
        print("❌ 影片無有效影格")
        return None
    timer.stage(f"影格載入({len(frames)} 幀)")
        
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
                progress_callback(min(percent, 98.0), f"階段 2/6：分析影像 ({i + 1}/{len(idxs)})...")
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
            if few_marker_mode and mid != ref_id:
                continue  # 少標籤模式: 標籤不共面且地圖不可靠，只以 ref 標籤自我擬合殘差當參考
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
        h, w = frames[0].shape[:2]
        return max(0.0, min(1.0, area / float(w * h)))

    # ---- 特徵極線驗證與混合 RT 精修 ----
    sift_feat = cv2.SIFT_create(nfeatures=FEATURE_MAX_KEYPOINTS, contrastThreshold=0.01)
    feat_cache = {}

    def get_frame_features(idx):
        if idx not in feat_cache:
            gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            feat_cache[idx] = sift_feat.detectAndCompute(gray, None)
        return feat_cache[idx]

    match_cache = {}

    def get_pair_matches(idx_left, idx_right):
        """左(結尾段)→右(開頭段) 的 SIFT 匹配 (ratio + mutual)，回傳已去畸變至 K_L 座標的點對。"""
        key = (idx_left, idx_right)
        if key in match_cache:
            return match_cache[key]
        kpL, desL = get_frame_features(idx_left)
        kpR, desR = get_frame_features(idx_right)
        result = None
        if desL is not None and desR is not None and len(desL) >= 8 and len(desR) >= 8:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            knn_lr = bf.knnMatch(desL, desR, k=2)
            knn_rl = bf.knnMatch(desR, desL, k=1)
            reverse_best = {m[0].queryIdx: m[0].trainIdx for m in knn_rl if m}
            good = []
            for pair in knn_lr:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < FEATURE_MATCH_RATIO * n.distance and reverse_best.get(m.trainIdx) == m.queryIdx:
                    good.append(m)
            if len(good) >= 8:
                good.sort(key=lambda m: m.distance)
                good = good[:500]
                ptsL = np.float32([kpL[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                ptsR = np.float32([kpR[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                ptsL_u = cv2.undistortPoints(ptsL, mtx_L, dist_L, P=K_L).reshape(-1, 2).astype(np.float64)
                ptsR_u = cv2.undistortPoints(ptsR, mtx_L, dist_L, P=K_L).reshape(-1, 2).astype(np.float64)
                result = (ptsL_u, ptsR_u)
        match_cache[key] = result
        return result

    def rt_epipolar_residual(ptsL_u, ptsR_u, R_rel_c, t_rel_c):
        """給定 左→右 相對位姿，計算點對的中位數對稱極線距離 (px)。"""
        t = np.asarray(t_rel_c, dtype=np.float64).flatten()
        if np.linalg.norm(t) < 1e-9:
            return float('inf')
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
        return float(np.median(0.5 * (dR + dL)))

    def marker_corner_pairs(corners_left_dict, corners_right_dict):
        shared = set(corners_left_dict.keys()) & set(corners_right_dict.keys())
        if not shared:
            return None
        pts_l = np.vstack([corners_left_dict[mid] for mid in shared]).astype(np.float64)
        pts_r = np.vstack([corners_right_dict[mid] for mid in shared]).astype(np.float64)
        return pts_l, pts_r

    def rot_angle_deg(Ra, Rb):
        Rd = np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T
        return float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))))

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
        """
        以單位平移方向三角化共享標籤角點，用已知邊長 (marker_size_mm) 反推 baseline。
        繞開 t_rel = t_s - R_rel @ t_e 的槓桿誤差；失敗回傳 None。
        """
        mk = marker_corner_pairs(corners_left_u, corners_right_u)
        if mk is None:
            return None
        ptsL, ptsR = mk
        t_u = np.asarray(t_dir, dtype=np.float64).reshape(3, 1)
        t_norm = float(np.linalg.norm(t_u))
        if t_norm < 1e-9:
            return None
        t_u = t_u / t_norm
        K64 = K_L.astype(np.float64)
        P0 = (K64 @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
        P1 = (K64 @ np.hstack([np.asarray(R_rel_c, np.float64), t_u])).astype(np.float32)
        pts4d = cv2.triangulatePoints(P0, P1, ptsL.T.astype(np.float32), ptsR.T.astype(np.float32))
        w = pts4d[3]
        if np.any(np.abs(w) < 1e-12):
            return None
        X = pts4d[:3] / w
        if float(np.median(X[2])) <= 0:  # cheirality: 點應在相機前方
            return None
        edges = []
        for m in range(len(ptsL) // 4):
            quad = X[:, m * 4:(m + 1) * 4]
            for i in range(4):
                edges.append(float(np.linalg.norm(quad[:, (i + 1) % 4] - quad[:, i])))
        L_est = float(np.median(edges))
        if L_est <= 1e-9:
            return None
        return marker_size_mm / L_est

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

    def refine_rt_with_features(idx_left, idx_right, R_aruco, t_aruco, corners_left_u, corners_right_u,
                                tag="", alt_rotations=None, single_marker=False):
        """
        混合 RT 精修：R 與 t 方向改用 SIFT + Essential matrix (5-point RANSAC) 的解，
        尺度 |t| 保留 ArUco 解。僅在特徵極線殘差改善、且標籤角點極線殘差未明顯變差時採用。
        """
        if not ENABLE_FEATURE_RT_REFINE:
            return R_aruco, t_aruco
        matches_lr = get_pair_matches(idx_left, idx_right)
        if matches_lr is None or len(matches_lr[0]) < FEATURE_MIN_MATCHES:
            log_and_print(f"ℹ️ [RT精修{tag}] 特徵匹配不足，保留 ArUco RT。")
            return R_aruco, t_aruco
        ptsL_u, ptsR_u = matches_lr
        K64 = K_L.astype(np.float64)
        E, mask_e = cv2.findEssentialMat(ptsL_u, ptsR_u, K64, method=cv2.RANSAC,
                                         prob=0.999, threshold=FEATURE_E_RANSAC_THRESH_PX)
        if E is None or E.shape != (3, 3):
            log_and_print(f"ℹ️ [RT精修{tag}] Essential matrix 求解失敗，保留 ArUco RT。")
            return R_aruco, t_aruco
        n_in, R_E, t_E, _mask_rp = cv2.recoverPose(E, ptsL_u, ptsR_u, K64, mask=mask_e.copy())
        if n_in < FEATURE_MIN_MATCHES:
            log_and_print(f"ℹ️ [RT精修{tag}] recoverPose 內點不足 ({n_in})，保留 ArUco RT。")
            return R_aruco, t_aruco

        # 健全性檢查：特徵解需與 ArUco 解、或其 IPPE 分支替代解大致一致
        R_a64 = np.asarray(R_aruco, dtype=np.float64)
        t_a64 = np.asarray(t_aruco, dtype=np.float64).flatten()
        ang = rot_angle_deg(R_E, R_a64)
        t_dot = float(np.dot(t_E.flatten(), t_a64 / max(np.linalg.norm(t_a64), 1e-9)))
        if ang <= FEATURE_ROT_DIFF_MAX_DEG:
            if t_dot <= 0:
                log_and_print(f"⚠️ [RT精修{tag}] 特徵解平移方向與 ArUco 解相反 (t·t'={t_dot:.2f})，保留 ArUco RT。")
                return R_aruco, t_aruco
        else:
            matched_alt = None
            for R_alt in (alt_rotations or []):
                ang_alt = rot_angle_deg(R_E, R_alt)
                if ang_alt <= FEATURE_ROT_DIFF_MAX_DEG:
                    matched_alt = ang_alt
                    break
            if matched_alt is None:
                log_and_print(f"⚠️ [RT精修{tag}] 特徵解與 ArUco 解及所有分支替代解差異過大 (dR={ang:.1f}°)，保留 ArUco RT。")
                return R_aruco, t_aruco
            log_and_print(f"🔀 [RT精修{tag}] 特徵解與 ArUco 解差 {ang:.1f}°，但與另一 IPPE 分支一致 (dR={matched_alt:.1f}°)——判定 ArUco 落入錯誤分支，交由仲裁決定")

        baseline_val = float(np.linalg.norm(t_a64))
        if single_marker:
            bsl_tri = baseline_from_marker_edges(R_E, t_E, corners_left_u, corners_right_u)
            if bsl_tri is not None and MIN_BASELINE_MM <= bsl_tri <= MAX_BASELINE_MM:
                log_and_print(f"📏 [RT精修{tag}] 標籤已知邊長定尺度: baseline {baseline_val:.2f} → {bsl_tri:.2f} mm")
                baseline_val = bsl_tri
            else:
                log_and_print(f"⚠️ [RT精修{tag}] 邊長定尺度失敗，baseline 沿用 ArUco 值 {baseline_val:.2f} mm")
        t_hybrid = (t_E.flatten() * baseline_val).reshape(3, 1)

        # 仲裁：特徵極線殘差需改善，標籤角點極線殘差不得退步超過容許量
        feat_a = rt_epipolar_residual(ptsL_u, ptsR_u, R_a64, t_a64)
        feat_h = rt_epipolar_residual(ptsL_u, ptsR_u, R_E, t_hybrid)
        mk = marker_corner_pairs(corners_left_u, corners_right_u)
        mk_a = rt_epipolar_residual(mk[0], mk[1], R_a64, t_a64) if mk else None
        mk_h = rt_epipolar_residual(mk[0], mk[1], R_E, t_hybrid) if mk else None
        mk_str = f"{mk_a:.3f}→{mk_h:.3f}" if mk_a is not None else "N/A"
        marker_ok = (mk_a is None) or (mk_h <= mk_a + FEATURE_MARKER_EPI_MARGIN_PX)
        if feat_h < feat_a and marker_ok:
            log_and_print(
                f"🚀 [RT精修{tag}] 採用混合解: 特徵極線 {feat_a:.3f}→{feat_h:.3f}px | "
                f"標籤極線 {mk_str}px | dR={ang:.2f}° | E內點={n_in}/{len(ptsL_u)}"
            )
            return R_E.astype(np.asarray(R_aruco).dtype), t_hybrid.astype(np.asarray(t_aruco).dtype)
        log_and_print(
            f"ℹ️ [RT精修{tag}] 混合解未通過仲裁 (特徵 {feat_a:.3f}→{feat_h:.3f}px, 標籤 {mk_str}px)，保留 ArUco RT。"
        )
        return R_aruco, t_aruco

    def compute_pair_quality_score(err, item_s, item_e, baseline_mm):
        shared_markers = set(item_s['corners'].keys()).intersection(item_e['corners'].keys())
        shared_count = len(shared_markers)
        sharp_s = get_frame_sharpness(item_s['idx'])
        sharp_e = get_frame_sharpness(item_e['idx'])
        sharp_min = max(min(sharp_s, sharp_e), 1e-6)
        blur_penalty = min(3.0, 120.0 / sharp_min)
        cover_s = marker_coverage_ratio(item_s['corners'])
        cover_e = marker_coverage_ratio(item_e['corners'])
        cover = min(cover_s, cover_e)
        coverage_penalty = max(0.0, 0.08 - cover) / 0.08
        marker_penalty = 1.0 / max(shared_count, 1)
        baseline_penalty = abs(baseline_mm - IDEAL_BASELINE_MM) / max(IDEAL_BASELINE_MM, 1e-6)
        score = (
            PAIR_SCORE_REPROJ_W * float(err)
            + PAIR_SCORE_BASELINE_W * baseline_penalty
            + PAIR_SCORE_BLUR_W * blur_penalty
            + PAIR_SCORE_COVER_W * coverage_penalty
            + PAIR_SCORE_MARKER_W * marker_penalty
        )
        metrics = {
            'score': float(score),
            'err': float(err),
            'baseline': float(baseline_mm),
            'shared_markers': int(shared_count),
            'sharpness_min': float(sharp_min),
            'coverage': float(cover),
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
    
    stages = [10, 20, 30, 40, 50]
    stage_success = False
    few_marker_mode = False
    best_branch = None
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
        # 恰好 1 個標籤才需要分支消歧：>=2 個標籤 (即使貼在不同平面) 的 joint PnP
        # 是非共面多點結構，天然沒有 IPPE 鏡像雙解問題，走原地圖路徑即可
        few_marker_mode = (len(shared_ids) == 1)
        if few_marker_mode:
            log_and_print("ℹ️ [單標籤模式] RT 以 ref 標籤 IPPE 雙分支 + 特徵極線消歧解算")

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

        def get_pose_branches(corners_dict):
            """少標籤模式: ref 標籤 IPPE 雙分支 (不假設多標籤共面)；其他: joint pose 單解。"""
            if not few_marker_mode:
                R, t = get_joint_pose(corners_dict)
                return [] if R is None else [(R, t)]
            pts = corners_dict.get(ref_id)
            if pts is None:
                return []
            try:
                n_sol, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
                    canon, pts.reshape(-1, 1, 2).astype(np.float32), mtx_L, dist_L,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error:
                return []
            branches = []
            for rv, tv in zip(rvecs, tvecs):
                R_b, _ = cv2.Rodrigues(rv)
                branches.append((R_b, tv.reshape(3, 1)))
            return branches

        valid_start = []
        for item in start_info:
            branches = get_pose_branches(item['corners'])
            if branches:
                valid_start.append({'idx': item['idx'], 'R': branches[0][0], 't': branches[0][1],
                                    'branches': branches, 'corners': item['corners']})

        valid_end = []
        for item in end_info:
            branches = get_pose_branches(item['corners'])
            if branches:
                valid_end.append({'idx': item['idx'], 'R': branches[0][0], 't': branches[0][1],
                                  'branches': branches, 'corners': item['corners']})

        if not valid_start or not valid_end:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無法計算有效的起點或終點 Joint Pose")
            continue

        # 計算候選對的重投影誤差與 baseline (單標籤模式對每幀的 IPPE 雙解分支展開組合;
        # 注意單標籤時 reproj err 是自我擬合殘差、對分支無鑑別力，真正的裁決在特徵極線重排)
        pairs = []
        for item_s in valid_start:
            for item_e in valid_end:
                err_pair = None
                for bi_s, (R_s, t_s) in enumerate(item_s['branches']):
                    for bi_e, (R_e, t_e) in enumerate(item_e['branches']):
                        R_rel_cand = R_s @ R_e.T
                        t_rel_cand = t_s - R_rel_cand @ t_e
                        bsl = float(np.linalg.norm(t_rel_cand))
                        if not (MIN_BASELINE_MM <= bsl <= MAX_BASELINE_MM):
                            continue
                        if err_pair is None:
                            err_pair = compute_pair_reprojection_error(item_s, item_e, mtx_L, dist_L)
                        if err_pair == float('inf'):
                            continue
                        pair_score, pair_metrics = compute_pair_quality_score(err_pair, item_s, item_e, bsl)
                        pair_metrics['branch'] = (bi_s, bi_e)
                        pairs.append((pair_score, err_pair, item_s, item_e, R_rel_cand, t_rel_cand, bsl, pair_metrics))

        if not pairs:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無合格的匹配對 (Baseline: {MIN_BASELINE_MM}~{MAX_BASELINE_MM} mm)")
            continue
            
        # 依誤差由小到大排序
        pairs.sort(key=lambda x: x[0])

        # 前 K 個「幀對」加算特徵極線殘差後重排 (真正的品質裁決)：
        # 自我擬合殘差對極線幾何無鑑別力，收斂與否由特徵極線殘差決定。
        # 以幀對為單位套用多樣性配額，避免名額被相鄰近似幀塞滿；
        # 單標籤模式下同一幀對的所有 IPPE 分支組合全數保留 (共用同一次 SIFT 匹配)。
        _admitted = set()
        _cnt_start, _cnt_end = {}, {}
        topk = []
        for cand_tuple in pairs:
            _key = (cand_tuple[2]['idx'], cand_tuple[3]['idx'])
            if _key not in _admitted:
                if len(_admitted) >= PAIR_EPI_TOPK:
                    continue
                if _cnt_start.get(_key[0], 0) >= PAIR_TOPK_MAX_PER_START:
                    continue
                if _cnt_end.get(_key[1], 0) >= PAIR_TOPK_MAX_PER_END:
                    continue
                _admitted.add(_key)
                _cnt_start[_key[0]] = _cnt_start.get(_key[0], 0) + 1
                _cnt_end[_key[1]] = _cnt_end.get(_key[1], 0) + 1
            topk.append(cand_tuple)
        reranked = []
        for cand_tuple in topk:
            pair_score, err, item_s, item_e, R_rel_c, t_rel_c, bsl, pair_metrics = cand_tuple
            matches_lr = get_pair_matches(item_e['idx'], item_s['idx'])
            if matches_lr is not None:
                epi_med = rt_epipolar_residual(matches_lr[0], matches_lr[1], R_rel_c, t_rel_c)
                combined = pair_score + PAIR_SCORE_EPI_W * min(epi_med, 10.0)
                pair_metrics['feat_epi_px'] = float(epi_med)
            else:
                # 無足夠特徵可驗證時給中性偏保守的懲罰，避免無驗證的配對反而占優
                combined = pair_score + PAIR_SCORE_EPI_W * 2.0
                pair_metrics['feat_epi_px'] = None
            reranked.append((combined, cand_tuple))
        reranked.sort(key=lambda x: x[0])
        for _c, _t in reranked:
            _fe_t = _t[7].get('feat_epi_px')
            log_and_print(
                f"   [topK] A=F{_t[2]['idx']} B=F{_t[3]['idx']} branch={_t[7].get('branch')} "
                f"feat_epi={f'{_fe_t:.3f}px' if _fe_t is not None else 'N/A'} "
                f"score={_t[0]:.3f} combined={_c:.3f}"
            )
        _topk_ids = {id(t) for t in topk}
        pairs = [t for _c, t in reranked] + [t for t in pairs if id(t) not in _topk_ids]

        best_cand = pairs[0]
        best_cand = pairs[0]
        best_score = best_cand[0]
        best_err = best_cand[1]
        best_metrics = best_cand[7]
        feat_epi = best_metrics.get('feat_epi_px')
        log_and_print(
            f"🎯 [pair quality] score={best_score:.3f} | reproj={best_err:.3f}px | "
            f"feat_epi={f'{feat_epi:.3f}px' if feat_epi is not None else 'N/A'} | "
            f"baseline={best_metrics['baseline']:.2f}mm | shared={best_metrics['shared_markers']} | "
            f"sharp={best_metrics['sharpness_min']:.1f} | coverage={best_metrics['coverage']:.3f}"
        )
        
        fe_str = f"{feat_epi:.3f}" if feat_epi is not None else "N/A"
        stage_ok = feat_epi is not None and feat_epi < PAIR_EPI_OK_PX
        if stage_ok:
            best_start, best_end, R_rel, t_rel, baseline = best_cand[2], best_cand[3], best_cand[4], best_cand[5], best_cand[6]
            best_branch = best_cand[7].get('branch')
            _branch_note = f" (IPPE 分支 {best_branch})" if few_marker_mode else f" | reproj {best_err:.3f} px"
            log_and_print(f"🎉 第 {stage_idx + 1} 階段搜尋成功！特徵極線殘差 {fe_str} px < {PAIR_EPI_OK_PX} px{_branch_note}")
            stage_success = True
        else:
            log_and_print(f"ℹ️ 第 {stage_idx + 1} 階段：最佳特徵極線殘差 {fe_str} px (門檻 {PAIR_EPI_OK_PX} px)，擴大抽樣")

        # 若本階段成功，或這已是最大抽樣張數的第二階段，即固定最佳與次佳解
        if stage_success or num_samples == 50:
            if not stage_success:
                # 降級：從已驗證候選中取特徵極線殘差最小者 (而非分數最小者)
                _validated = [t for t in pairs if t[7].get('feat_epi_px') is not None]
                if _validated:
                    best_cand = min(_validated, key=lambda t: t[7]['feat_epi_px'])
                best_start, best_end, R_rel, t_rel, baseline = best_cand[2], best_cand[3], best_cand[4], best_cand[5], best_cand[6]
                best_branch = best_cand[7].get('branch')
                _fe_fb = best_cand[7].get('feat_epi_px')
                log_and_print(f"⚠️ 達到最大抽樣張數 (50 張) 仍未達特徵極線門檻 {PAIR_EPI_OK_PX} px。"
                              f"降級使用殘差最小候選 (feat_epi {f'{_fe_fb:.3f}' if _fe_fb is not None else 'N/A'} px, "
                              f"reproj {best_cand[1]:.3f} px)——量測品質可能不佳")
                
            # 次佳對選取 (全模式統一)：只收已通過特徵極線驗證 (< PAIR_EPI_EXTRA_PX) 的候選，
            # 同一結尾幀、不同起始幀，每個起始幀只取排序最前 (最佳) 的一組
            candidates_scores = []
            _seen_extra_idx = set()
            for pair_score, err, item_s, item_e, R_rel_c, t_rel_c, bsl, pair_metrics in pairs:
                if item_e['idx'] != best_end['idx'] or item_s['idx'] == best_start['idx']:
                    continue
                if item_s['idx'] in _seen_extra_idx:
                    continue
                fe = pair_metrics.get('feat_epi_px')
                if fe is None or fe >= PAIR_EPI_EXTRA_PX:
                    continue
                _seen_extra_idx.add(item_s['idx'])
                candidates_scores.append((pair_score, err, item_s, R_rel_c, t_rel_c, bsl, pair_metrics))

            selected_extras = candidates_scores[:5]
            selected_extras = candidates_scores[:5]
            break
            
    timer.stage("ArUco偵測+配對搜尋(含極線重排)")
    if best_start is None or best_end is None:
        log_and_print("❌ [漸進式匹配] 無法在該影片中計算出任何影像對，分析失敗。")
        return None

    if few_marker_mode and best_branch is not None and 'branches' in best_start:
        _bi_s, _bi_e = best_branch
        best_start['R'], best_start['t'] = best_start['branches'][_bi_s]
        best_end['R'], best_end['t'] = best_end['branches'][_bi_e]
        log_and_print(f"ℹ️ [單標籤模式] 依特徵消歧採用 IPPE 分支組合 (s={_bi_s}, e={_bi_e})")

    # 儲存最佳配對偵錯圖片
    save_debug_pair_images(best_start, best_end, "best")

    cornersA_undist = undistort_corners_dict(best_start['corners'])
    cornersB_undist = undistort_corners_dict(best_end['corners'])

    # 混合 RT 精修 (最佳對)：R 與 t 方向取特徵解、尺度保留 ArUco，未通過仲裁則維持原值
    _best_alts = build_alt_rotations(best_start, best_end, R_rel) if few_marker_mode else None
    R_rel, t_rel = refine_rt_with_features(
        best_end['idx'], best_start['idx'], R_rel, t_rel,
        cornersB_undist, cornersA_undist, tag="-best",
        alt_rotations=_best_alts, single_marker=few_marker_mode
    )
    baseline = float(np.linalg.norm(t_rel))

    # 包裝次優額外右圖組 (同樣做混合 RT 精修)
    extra_candidates_info = []
    for pair_score, err, item_s, R_rel_c, t_rel_c, bsl, pair_metrics in selected_extras:
        cornersA_e_undist = undistort_corners_dict(item_s['corners'])
        _extra_alts = build_alt_rotations(item_s, best_end, R_rel_c) if few_marker_mode else None
        R_rel_c, t_rel_c = refine_rt_with_features(
            best_end['idx'], item_s['idx'], R_rel_c, t_rel_c,
            cornersB_undist, cornersA_e_undist, tag=f"-F{item_s['idx']}",
            alt_rotations=_extra_alts, single_marker=few_marker_mode
        )
        extra_candidates_info.append({
            'idx_A': item_s['idx'],
            'frame_A': frames[item_s['idx']],
            'R_rel': R_rel_c,
            't_rel': t_rel_c,
            'baseline': float(np.linalg.norm(t_rel_c)),
            'pair_score': pair_score,
            'pair_metrics': pair_metrics,
            'cornersA': cornersA_e_undist
        })
        log_and_print(f"➕ [次佳配對] 額外右圖 (Frame A) 索引: {item_s['idx']} | 重投影誤差: {err:.3f} px | Baseline: {bsl:.2f} mm")

    if not selected_extras:
        log_and_print(f"ℹ️ [次佳配對] 未找到任何符合條件 of 額外次佳配對影格 (門檻 0.5 px, Baseline 介於 {MIN_BASELINE_MM}~{MAX_BASELINE_MM} mm)。")

    timer.stage("混合RT精修+次佳打包")
    log_and_print(f"✅ 挑選結果：")
    log_and_print(f"  - 右圖 (Frame A) 索引: {best_start['idx']}")
    log_and_print(f"  - 左圖 (Frame B) 索引: {best_end['idx']}")
    log_and_print(f"  - 計算 Baseline: {baseline:.2f} mm")
    
    valid_poses = {}
    for item in valid_start + valid_end:
        valid_poses[item['idx']] = (item['R'], item['t'])

    best_reproj_err = None
    if best_start is not None and best_end is not None:
        best_reproj_err = compute_pair_reprojection_error(best_start, best_end, mtx_L, dist_L)
    if few_marker_mode:
        _m_final = get_pair_matches(best_end['idx'], best_start['idx'])
        if _m_final is not None:
            _fe_final = rt_epipolar_residual(_m_final[0], _m_final[1], R_rel, t_rel)
            log_and_print(f"ℹ️ [單標籤模式] 姿態指標改用特徵極線殘差 {_fe_final:.3f} px "
                          f"(自我擬合殘差 {best_reproj_err:.3f} px 無鑑別力)")
            best_reproj_err = _fe_final
        
    if progress_callback:
        progress_callback(92, "階段 3/6：影像校正...")
        
    # 預先在背景執行去畸變、平面擬合與 SIFT 特徵提取，優化 UI 載入速度
    h_raw, w_raw = frames[0].shape[:2]
    newKL_o, _ = cv2.getOptimalNewCameraMatrix(mtx_L, dist_L, (w_raw, h_raw), 1, (w_raw, h_raw))
    _map1, _map2 = cv2.initUndistortRectifyMap(mtx_L, dist_L, None, newKL_o, (w_raw, h_raw), cv2.CV_16SC2)
    
    def local_process_view(img):
        return cv2.remap(img, _map1, _map2, cv2.INTER_LINEAR)
        
    imgA_bgr = local_process_view(frames[best_end['idx']])  # 結尾最優影格作為左圖 (B)
    imgB_bgr = local_process_view(frames[best_start['idx']])  # 開頭最優影格作為右圖 (A)
    
    if progress_callback:
        progress_callback(94, "階段 4/6：基準計算...")
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, True)
    # 參考平面不分模式，一律只由 ref 標籤 (最小 ID) 定義：標籤可能貼在不同平面，
    # 不可合併多標籤擬合。以最終 RT 三角化 ref 標籤角點，與量測點同一幾何鏈 (誤差相消)。
    _ref_L = {ref_id: cornersB_undist[ref_id]} if ref_id in cornersB_undist else None
    _ref_R = {ref_id: cornersA_undist[ref_id]} if ref_id in cornersA_undist else None
    global_plane_n, global_plane_c = (None, None)
    if _ref_L and _ref_R:
        global_plane_n, global_plane_c = plane_from_triangulated_corners(
            R_rel, t_rel, _ref_L, _ref_R)
        if global_plane_n is not None:
            log_and_print(f"✅ [參考平面] 由 ref 標籤 ID:{ref_id} 的三角化角點定義 (其他標籤不參與平面)")
    if global_plane_n is None:
        log_and_print("⚠️ [參考平面] 三角化平面失敗，退回 PnP 平面")
        global_plane_n, global_plane_c = compute_global_plane(imgA_gray, K_L, marker_size_mm)
    timer.stage("去畸變+全域平面擬合")
    
    if progress_callback:
        progress_callback(96, "階段 5/6：資料準備...")
    sift = cv2.SIFT_create(contrastThreshold=0.005)
    imgB_gray = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY)
    kb, db = sift.detectAndCompute(imgB_gray, None)
    
    # 次佳候選影像特徵提取
    for idx_extra, extra in enumerate(extra_candidates_info):
        if progress_callback:
            progress = min(97 + int((idx_extra / max(1, len(extra_candidates_info))) * 3), 99)
            progress_callback(progress, f"階段 5/6：資料準備 ({idx_extra+1}/{len(extra_candidates_info)})...")
        imgB_extra_bgr = local_process_view(extra['frame_A'])
        imgB_extra_gray = cv2.cvtColor(imgB_extra_bgr, cv2.COLOR_BGR2GRAY)
        kb_e, db_e = sift.detectAndCompute(imgB_extra_gray, None)
        extra['kpB'] = kb_e
        extra['desB'] = db_e
        
    timer.stage("SIFT特徵提取(最佳+次佳)")
    timer.report(print_fn=log_and_print)
    if progress_callback:
        progress_callback(100, "階段 6/6：完成")
        
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

