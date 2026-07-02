import os

import cv2
import numpy as np

from .aruco_pose import average_rotations_svd, compute_global_plane as _compute_global_plane
from .camera_preprocess import preprocess_gray

RECORD_SAVE_DIR = "test_video_Zebra"
MIN_BASELINE_MM = 8.0
MAX_BASELINE_MM = 220.0
IDEAL_BASELINE_MM = 45.0
PAIR_SCORE_REPROJ_W = 1.00
PAIR_SCORE_BASELINE_W = 0.18
PAIR_SCORE_BLUR_W = 0.18
PAIR_SCORE_COVER_W = 0.12
PAIR_SCORE_MARKER_W = 0.08


def log_and_print(msg):
    print(msg)


def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    return _compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=log_and_print)


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
                
                if MIN_BASELINE_MM <= bsl <= MAX_BASELINE_MM:
                    err = compute_pair_reprojection_error(item_s, item_e, mtx_L, dist_L)
                    if err != float('inf'):
                        pair_score, pair_metrics = compute_pair_quality_score(err, item_s, item_e, bsl)
                        pairs.append((pair_score, err, item_s, item_e, R_rel_cand, t_rel_cand, bsl, pair_metrics))
                        
        if not pairs:
            log_and_print(f"⚠️ 第 {stage_idx + 1} 階段：無合格的匹配對 (Baseline: {MIN_BASELINE_MM}~{MAX_BASELINE_MM} mm)")
            continue
            
        # 依誤差由小到大排序
        pairs.sort(key=lambda x: x[0])
        best_cand = pairs[0]
        best_score = best_cand[0]
        best_err = best_cand[1]
        best_metrics = best_cand[7]
        log_and_print(
            f"🎯 [pair quality] score={best_score:.3f} | reproj={best_err:.3f}px | "
            f"baseline={best_metrics['baseline']:.2f}mm | shared={best_metrics['shared_markers']} | "
            f"sharp={best_metrics['sharpness_min']:.1f} | coverage={best_metrics['coverage']:.3f}"
        )
        
        if best_err < 0.2:
            best_start, best_end, R_rel, t_rel, baseline = best_cand[2], best_cand[3], best_cand[4], best_cand[5], best_cand[6]
            log_and_print(f"🎉 第 {stage_idx + 1} 階段搜尋成功！在 {num_samples} 張抽樣下，找到誤差 < 0.2 px 的最佳配對：誤差 {best_err:.3f} px")
            stage_success = True
        else:
            log_and_print(f"ℹ️ 第 {stage_idx + 1} 階段最小誤差為 {best_err:.3f} px (未低於 0.2 px 門檻)")
            
        # 若本階段成功，或這已是最大抽樣張數的第二階段，即固定最佳與次佳解
        if stage_success or num_samples == 50:
            if not stage_success:
                best_start, best_end, R_rel, t_rel, baseline = best_cand[2], best_cand[3], best_cand[4], best_cand[5], best_cand[6]
                log_and_print(f"⚠️ 達到最大抽樣張數 (50 張) 仍未找到低於 0.2 px 的配對。降級使用當前最優對，誤差為: {best_err:.3f} px")
                
            # 次佳對選取：重投影誤差小於 0.5 px 且 baseline 大於門檻的其餘配對，最多取 5 組
            candidates_scores = []
            for pair_score, err, item_s, item_e, R_rel_c, t_rel_c, bsl, pair_metrics in pairs:
                if item_e['idx'] == best_end['idx'] and item_s['idx'] != best_start['idx']:
                    if err < 0.8:
                        candidates_scores.append((pair_score, err, item_s, R_rel_c, t_rel_c, bsl, pair_metrics))
            
            selected_extras = candidates_scores[:5]
            break
            
    if best_start is None or best_end is None:
        log_and_print("❌ [漸進式匹配] 無法在該影片中計算出任何影像對，分析失敗。")
        return None

    # 儲存最佳配對偵錯圖片
    save_debug_pair_images(best_start, best_end, "best")

    # 包裝次優額外右圖組
    extra_candidates_info = []
    for pair_score, err, item_s, R_rel_c, t_rel_c, bsl, pair_metrics in selected_extras:
        extra_candidates_info.append({
            'idx_A': item_s['idx'],
            'frame_A': frames[item_s['idx']],
            'R_rel': R_rel_c,
            't_rel': t_rel_c,
            'baseline': bsl,
            'pair_score': pair_score,
            'pair_metrics': pair_metrics,
            'cornersA': undistort_corners_dict(item_s['corners'])
        })
        log_and_print(f"➕ [次佳配對] 額外右圖 (Frame A) 索引: {item_s['idx']} | 重投影誤差: {err:.3f} px | Baseline: {bsl:.2f} mm")

    if not selected_extras:
        log_and_print(f"ℹ️ [次佳配對] 未找到任何符合條件 of 額外次佳配對影格 (門檻 0.5 px, Baseline 介於 {MIN_BASELINE_MM}~{MAX_BASELINE_MM} mm)。")

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

