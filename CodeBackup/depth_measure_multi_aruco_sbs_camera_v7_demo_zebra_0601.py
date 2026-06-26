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
CAMERA_INDEX          = 0                          # 相機索引
FRAME_GAP             = 30                         # Frame B 與 Frame A 的幀數間隔
UPDATE_INTERVAL       = 0.3                     # 畫面更新與計算間隔 (秒)
CAMERA_WIDTH          = 1920                       # 相機解析度寬
CAMERA_HEIGHT         = 1080                        # 相機解析度高


PARAMS_JSON_PATH      = "calibration_result_Zebra_1_no_dis.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM = 8.25#16.5                       # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
MIN_BASELINE_MM       = 8.0                       # 最小基準線限制，低於此值不進行深度計算 (mm)
AUTO_CALC_INTERVAL_SEC = 0.2                       # 連續計算模式下的計算時間間隔 (秒)
ENFORCE_COPLANAR      = False                       # 強制共面對齊優化
SAVE_ARUCO_DEBUG_IMG  = False                      # 是否存出 ArUco 偵測結果圖片
ENABLE_POSE_SMOOTHING  = True                       # 是否啟用時序平滑濾波 (EMA)
POSE_SMOOTHING_ALPHA   = 0.3                        # 平滑係數 (0.0 ~ 1.0)，越小越平滑但延遲越高
ENABLE_CLAHE_DEFAULT  = True                        # 預設是否啟用 CLAHE 直方圖均衡化前處理以改善匹配 (True/False)


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

def preprocess_gray(gray_img, enable_clahe=True):
    if enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
    if idsA is None or len(idsA) < 1: return None, None
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
    if not objA: return None, None
    objA = np.vstack(objA).astype(np.float32)
    c = np.mean(objA, axis=0)
    _, _, Vt = np.linalg.svd(objA - c)
    n = Vt[-1]
    if np.dot(n, c) > 0: n = -n
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
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(canon, cA[idxA][0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
            if ok and len(rvecs) > 0:
                best_idx = 0
                if len(rvecs) == 2:
                    R0, _ = cv2.Rodrigues(rvecs[0])
                    R1, _ = cv2.Rodrigues(rvecs[1])
                    n0, n1 = R0[:, 2], R1[:, 2]
                    dir0 = tvecs[0].flatten() / np.linalg.norm(tvecs[0])
                    dir1 = tvecs[1].flatten() / np.linalg.norm(tvecs[1])
                    tilt0 = abs(np.dot(n0, dir0))
                    tilt1 = abs(np.dot(n1, dir1))
                    best_idx = 0 if tilt0 > tilt1 else 1
                marker_poses_L[mid] = (rvecs[best_idx], tvecs[best_idx])
        
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
            
            ok_L, rv_L, tv_L = cv2.solvePnP(joint_objW, joint_imgA, K_L, np.zeros(5))
            ok_R, rv_R, tv_R = cv2.solvePnP(joint_objW, joint_imgB, K_R, np.zeros(5))
            
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
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(canon, cA[idxA][0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        if ok and len(rvecs) > 0:
            best_idx = 0
            if len(rvecs) == 2:
                if prev_marker_poses is not None and mid in prev_marker_poses:
                    rv_prev, tv_prev = prev_marker_poses[mid]
                    R_prev, _ = cv2.Rodrigues(rv_prev)
                    n_prev = R_prev[:, 2]
                    score0 = np.dot(cv2.Rodrigues(rvecs[0])[0][:, 2], n_prev)
                    score1 = np.dot(cv2.Rodrigues(rvecs[1])[0][:, 2], n_prev)
                    best_idx = 0 if score0 > score1 else 1
                else:
                    R0, _ = cv2.Rodrigues(rvecs[0])
                    R1, _ = cv2.Rodrigues(rvecs[1])
                    n0, n1 = R0[:, 2], R1[:, 2]
                    dir0 = tvecs[0].flatten() / np.linalg.norm(tvecs[0])
                    dir1 = tvecs[1].flatten() / np.linalg.norm(tvecs[1])
                    tilt0 = abs(np.dot(n0, dir0))
                    tilt1 = abs(np.dot(n1, dir1))
                    best_idx = 0 if tilt0 > tilt1 else 1
            
            rv = rvecs[best_idx]
            tv = tvecs[best_idx]
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
        ok_rel, rvecs_rel, tvecs_rel, _ = cv2.solvePnPGeneric(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        if ok_rel and len(rvecs_rel) > 0:
            best_idx = 0
            if len(rvecs_rel) == 2:
                R0, _ = cv2.Rodrigues(rvecs_rel[0])
                R1, _ = cv2.Rodrigues(rvecs_rel[1])
                best_idx = 0 if np.trace(R0) > np.trace(R1) else 1
            rv_rel = rvecs_rel[best_idx]
            tv_rel = tvecs_rel[best_idx]
            ok_rel = True
    else:
        if use_guess_rel:
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), rvec=rv_rel_init, tvec=tv_rel_init, useExtrinsicGuess=True)
        else:
            ok_rel, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5))

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

def main():
    import collections, time
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened(): print(f"❌ 無法開啟相機"); sys.exit(1)
    
    # 讀取第一幀獲取尺寸
    ret, first_frame = cap.read()
    if not ret: print("❌ 無法讀取相機畫面"); sys.exit(1)
    
    h_raw, w_raw = first_frame.shape[:2]
    w_alg = w_raw # 使用完整相機畫面
    active_u = w_alg // 2
    active_v = h_raw // 2
    
    mtxL_o, distL, mtxR_o, distR, extrinsic, F_orig = load_json_camera_params(PARAMS_JSON_PATH)
    newKL_o, _ = cv2.getOptimalNewCameraMatrix(mtxL_o, distL, (w_alg, h_raw), 1, (w_alg, h_raw))
    
    # 假設右圖參照左圖參數 (同一相機)
    newKR_o = newKL_o.copy()
    
    # 預先建立去畸變查找表（只算一次，之後每幀用 remap 即可）
    _map1, _map2 = cv2.initUndistortRectifyMap(
        mtxL_o, distL, None, newKL_o, (w_alg, h_raw), cv2.CV_16SC2
    )

    def process_view(img, K=None, dist=None, nK=None):
        """去畸變。K/dist/nK 保留以維持呼叫介面相容，實際使用預建查找表。"""
        undist = cv2.remap(img, _map1, _map2, cv2.INTER_LINEAR)
        return undist, 1.0
        
    imgA_bgr, scale = process_view(first_frame[:, :w_alg], mtxL_o, distL, newKL_o)
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    imgA_gray = preprocess_gray(imgA_gray, ENABLE_CLAHE_DEFAULT)
    h, w = imgA_gray.shape
    KL = newKL_o.copy().astype(np.float64)
    KL[0,0]*=scale; KL[1,1]*=scale; KL[0,2]*=scale; KL[1,2]*=scale
    
    # 使用 JSON 中的外參，讓深度計算有基準
    R_r = np.array(extrinsic.get('R', np.eye(3)))
    t_r = np.array(extrinsic.get('T', np.zeros(3))).reshape(3, 1)
    
    sift = cv2.SIFT_create(contrastThreshold=0.005)
    orb = cv2.ORB_create(nfeatures=1000)
    
    global_plane_n, global_plane_c = compute_global_plane(imgA_gray, KL, ACTUAL_MARKER_SIZE_MM)
    
    frame_buffer = collections.deque(maxlen=FRAME_GAP + 1)
    
    current_cand = {
        'idx': 0, 'rgb': cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB), 'gray': imgA_gray,
        'K_R': KL, 'R_rel': R_r, 't_rel': t_r, 'F': compute_fundamental_matrix(KL, KL, R_r, t_r),
        'cornersA': {}, 'cornersB': {}, 'kpB': [], 'desB': None,
        'plane_n': global_plane_n, 'plane_c': global_plane_c
    }
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

    scatter_A = ax_A.scatter([], [], s=80, c='red', marker='x', zorder=5)
    scatter_B = ax_B.scatter([], [], s=80, c='lime', marker='x', zorder=5)
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
    depth_text = ax_B.text(0.02, 0.05, "", transform=ax_B.transAxes, color='white', fontweight='bold', fontsize=16, bbox=dict(facecolor='#121212', alpha=0.7, edgecolor='#00FFFF', lw=1))
    fps_text = ax_A.text(0.01, 0.97, "FPS: --", transform=ax_A.transAxes,
                         color='#00FF00', fontsize=10, fontweight='bold', va='top',
                         bbox=dict(facecolor='#121212', alpha=0.6, edgecolor='none'), zorder=10)
    # Blit 最佳化：標記每幀會改變的 artists 為 animated，防止它們被無謂嫚入靜態背景圖
    im_A.set_animated(True)
    im_B.set_animated(True)
    fps_text.set_animated(True)
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
                  'grad_data': None}  # grad_data = {'ptsA': ndarray, 'ptsB': ndarray}

    # 統一設定文字顏色為白色，並將按鈕外框設為白色
    for c in [c1, c2, c3, c4, c5, c6]:
        c.label.set_color('white')
        c.label.set_fontsize(8)
        c.ax.patch.set_edgecolor('white')
        c.ax.patch.set_linewidth(1.0)
            
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

        if not cand.get('pose_valid', True):
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': None, 'g_rect': None,
                    'fail_reason': '未偵測到 ArUco', 'u': u, 'v': v}
        if cand.get('baseline', 0.0) < MIN_BASELINE_MM:
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': None, 'g_rect': None,
                    'fail_reason': f'視差不足(<{MIN_BASELINE_MM}mm)', 'u': u, 'v': v}

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

        d_val, p3d_val, p3d_w_val, fail_reason = None, None, None, ""
        if m_pt is not None:
            print(f"👉 [深度計算] 左圖座標: ({u:.1f}, {v:.1f}) | 右圖匹配座標: ({m_pt[0]:.1f}, {m_pt[1]:.1f}) | 匹配方式: {method}")
            R_str = np.array2string(cand['R_rel'].flatten(), precision=4, suppress_small=True)
            t_str = np.array2string(cand['t_rel'].flatten(), precision=2, suppress_small=True)
            print(f"   [當前外參] R_rel: {R_str} | t_rel: {t_str}")
            p3d = triangulate_point_3d((u, v), m_pt, KL, cand['K_R'], cand['R_rel'], cand['t_rel'])
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
                    # 1. 優先使用背景執行緒中已經過 IPPE 且消解二義性的穩定標籤位姿
                    if 'curr_marker_poses' in cand and min_id in cand['curr_marker_poses']:
                        rv_o, tv_o = cand['curr_marker_poses'][min_id]
                        ok_origin = True
                    else:
                        # 2. 備用方案：使用 IPPE 獨立解算並消解二義性
                        half = ACTUAL_MARKER_SIZE_MM / 2.0
                        canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
                        ok_generic, rvecs_o, tvecs_o, _ = cv2.solvePnPGeneric(
                            canon, cand['cornersA'][min_id], KL, np.zeros(5),
                            flags=cv2.SOLVEPNP_IPPE
                        )
                        if ok_generic and len(rvecs_o) > 0:
                            best_idx = 0
                            if len(rvecs_o) == 2:
                                # 選擇最正面面向相機的解
                                R0, _ = cv2.Rodrigues(rvecs_o[0])
                                R1, _ = cv2.Rodrigues(rvecs_o[1])
                                n0, n1 = R0[:, 2], R1[:, 2]
                                dir0 = tvecs_o[0].flatten() / np.linalg.norm(tvecs_o[0])
                                dir1 = tvecs_o[1].flatten() / np.linalg.norm(tvecs_o[1])
                                best_idx = 0 if abs(np.dot(n0, dir0)) > abs(np.dot(n1, dir1)) else 1
                            rv_o = rvecs_o[best_idx]
                            tv_o = tvecs_o[best_idx]
                            ok_origin = True
                            
                    if ok_origin:
                        R_o, _ = cv2.Rodrigues(rv_o)
                        p3d_w = R_o.T @ (p3d_val.reshape(3, 1) - tv_o.reshape(3, 1))
                        p3d_w_val = p3d_w.flatten()
        else:
            fail_reason = "無匹配點"
            print(f"👉 [深度計算] 左圖座標: ({u:.1f}, {v:.1f}) | 右圖匹配座標: N/A | 匹配方式: N/A")
            print(f"   [計算失敗] 原因: {fail_reason}")

        return {'pt': m_pt, 'p3d': p3d_val, 'p3d_w': p3d_w_val, 'd': d_val, 'method': method, 'neighbors': neighbors,
                'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_kptsB': g_kptsB, 'g_rect': g_rect,
                'fail_reason': fail_reason, 'u': u, 'v': v}

    def do_measure(u, v, manual_match_pt=None):
        """同步計算並立即更新 UI（用於手動點選的即時響應）。"""
        nonlocal last_click; last_click = (u, v)
        if len(plane_dist_history) > 0:
            plane_dist_history.clear()
        for l in view_state['lines']: l.remove()
        view_state['lines'] = []
        view_state['grad_data'] = None
        redraw_grad_lines(None)  # 清除舊連線
        sift_rect.set_visible(False)
        sift_rect_center.set_visible(False)
        snap_vs = dict(view_state)
        if not live_L and locked_L is not None:
            imgA_bgr_snap, _ = process_view(locked_L, mtxL_o, distL, newKL_o)
            left_img_gray = cv2.cvtColor(imgA_bgr_snap, cv2.COLOR_BGR2GRAY)
            left_img_gray = preprocess_gray(left_img_gray, snap_vs['enable_clahe'])
        else:
            left_img_gray = imgA_gray
            
        res = compute_measure(u, v, current_cand, left_img_gray, snap_vs, manual_match_pt)
        
        # 🌟 自訂平面選點攔截邏輯 🌟
        if custom_plane_mode:
            if res.get('p3d') is not None:
                custom_plane_pts_3d.append(res['p3d'])
                custom_plane_pts_2d.append((res['u'], res['v']))
                
                # 繪製洋紅色標記與序號
                c_pt, = ax_A.plot(res['u'], res['v'], 'mo', markersize=6, zorder=5)
                t_lbl = ax_A.text(res['u'] + 5, res['v'] - 5, f"P{len(custom_plane_pts_3d)}", 
                                  color='magenta', fontsize=9, fontweight='bold', zorder=5)
                custom_plane_artists.extend([c_pt, t_lbl])
                redraw_custom_plane_poly()
                
                btn_custom_plane.label.set_text(f"完成擬合 ({len(custom_plane_pts_3d)})")
                print(f"🎯 自訂平面已新增點 P{len(custom_plane_pts_3d)}: (u, v)=({res['u']:.1f}, {res['v']:.1f}), 3D={res['p3d']}")
            else:
                print("❌ 點選點之深度計算無效，無法加入自訂平面點！")
                
        measure_results[current_cand['idx']] = res
        all_d = [r['d'] for r in measure_results.values() if r['d'] is not None]
        summary = [f"F{current_cand['idx']}: {res['d']:.1f}" if res['d'] is not None else f"F{current_cand['idx']}: N/A"]
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
                

        pose_info_str = current_cand.get('pose_info', '')
        if res['pt'] is not None:
            scatter_B.set_offsets([[res['pt'][0], res['pt'][1]]])
            p0, p1 = epipolar_line(current_cand['F'], (u, v), w); epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            p_dist_str = ""
            if res['p3d'] is not None and current_cand['plane_n'] is not None:
                p_dist = abs(np.dot(current_cand['plane_n'], res['p3d'] - current_cand['plane_c']))
                if auto_calc_active:
                    plane_dist_history.append(p_dist)
                    p_dist_str = f"\n距平面: {np.mean(plane_dist_history):.1f}mm"
                else:
                    if len(plane_dist_history) > 0:
                        plane_dist_history.clear()
                    p_dist_str = f"\n距平面: {p_dist:.1f}mm"
            if res['d'] is not None:
                p3d_c = res['p3d']
                p3d_w = res.get('p3d_w')
                c_coords_str = f"({p3d_c[0]:.1f}, {p3d_c[1]:.1f}, {p3d_c[2]:.1f})" if p3d_c is not None else "N/A"
                w_coords_str = f"({p3d_w[0]:.1f}, {p3d_w[1]:.1f}, {p3d_w[2] + wound_z_offset:.1f})" if p3d_w is not None else "N/A"
                
                custom_plane_str = ""
                if custom_plane_fitted and p3d_c is not None and custom_plane_n is not None and custom_plane_c is not None:
                    proj_dist_signed = np.dot(custom_plane_n, p3d_c - custom_plane_c)
                    status = "高於" if proj_dist_signed > 0 else "低於"
                    custom_plane_str = f"\n距自訂平面: {status} {abs(proj_dist_signed):.1f}mm"
                
                main_text = (f"相機與傷口距離(點選處): {res['d']:.1f}mm{custom_plane_str}\n"
                             f"傷口高度(相對於標記): {p3d_w[2] + wound_z_offset:.1f}mm\n" #w_coords_str
                             #相機3D座標: {c_coords_str}\n"
                             #f"世界3D座標: {w_coords_str}{p_dist_str}\n"
                             #f"外參來源: {pose_info_str}"
                             )
            else:
                main_text = f"深度值(距離): 無效({res.get('fail_reason', '原因未知')}){p_dist_str}\n外參來源: {pose_info_str}"
        else: scatter_B.set_offsets(np.empty((0,2))); epi_line.set_data([], []); main_text = f"無效點({res.get('fail_reason', '無匹配點')})\n外參來源: {pose_info_str}"
        depth_text.set_text(main_text)
        request_blit_refresh()



    pan_state = {'pressing': False, 'x': None, 'y': None, 'ax': None, 'dragged': False, 'dragging_hud': False}
    def on_press(event):
        if event.button != 1: return
        
        # 檢查是否點擊在深度數值 HUD 區域內
        try:
            bbox = depth_text.get_window_extent(fig.canvas.get_renderer())
            if bbox.contains(event.x, event.y):
                pan_state['dragging_hud'] = True
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
            if event.inaxes == ax_B:
                inv = ax_B.transAxes.inverted()
                new_pos = inv.transform((event.x, event.y))
                # 限制範圍，避免文字飛出視窗 (x: 0.01~0.75, y: 0.01~0.95)
                new_x = max(0.01, min(0.75, new_pos[0]))
                new_y = max(0.01, min(0.95, new_pos[1]))
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
    locked_L = None
    locked_R = None
    live_L = True
    live_R = True
    has_set_L = False
    has_set_R = False
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
    
    current_cand = {
        'idx': 0,
        'K_R': KL,
        'pose_valid': False,
        'baseline': 0.0,
        'cornersA': {},
        'cornersB': {}
    }
    
    # 建立按鈕 (已統一尺寸、排列，並升級為精緻的「微發光邊框」與「功能分色」設計)
    # 使用更深邃的背景色 (#1A1A1A)，與主背景形成對比
    btn_style = dict(color='#1A1A1A', hovercolor='#333333')
    
    ax_btn_lock_L = fig.add_axes([0.58, 0.92, 0.08, 0.04])
    btn_lock_L = Button(ax_btn_lock_L, "鎖定左圖", **btn_style)
    
    ax_btn_lock_R = fig.add_axes([0.68, 0.92, 0.08, 0.04])
    btn_lock_R = Button(ax_btn_lock_R, "鎖定右圖", **btn_style)
    
    ax_btn_hide_R = fig.add_axes([0.78, 0.92, 0.08, 0.04])
    btn_hide_R = Button(ax_btn_hide_R, "隱藏右圖", **btn_style)
    
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
    for b in [btn_lock_L, btn_lock_R, btn_hide_R, btn_norm_toggle, btn_calc, btn_auto_calc, btn_grad_toggle, btn_custom_plane]:
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
    
    def on_hide_R(event):
        visible = ax_B.get_visible()
        ax_B.set_visible(not visible)
        btn_hide_R.label.set_text("顯示右圖" if visible else "隱藏右圖")
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
        if not has_set_R:
            print("⚠️ 請先鎖定右圖！")
            return
        # 快照計算所需數據，存入 request queue（不阻塞主執行緒）
        current_L_snap   = frame_buffer[-1].copy() if live_L and len(frame_buffer) > 0 else locked_L
        req = {
            'raw_imgA':   current_L_snap,
            'raw_imgB':   locked_R,
            'cand':       dict(current_cand),
            'view_state': dict(view_state),
            'u': active_u, 'v': active_v
        }
        # 丟棄積壓的舊請求，只保留最新一次
        while not calc_request_q.empty():
            try: calc_request_q.get_nowait()
            except: pass
        try:
            calc_request_q.put_nowait(req)
        except queue.Full:
            pass

    # ---- 背景工作執行緒 ----
    def calc_worker():
        nonlocal reset_pose_history
        # 時序狀態變數，用來進行 Extrinsic Guess 初值約束與 EMA 平滑濾波
        history_marker_poses = {}   # marker_id -> (rv, tv)
        history_rel_pose = None     # (rv_rel, tv_rel)
        
        # EMA 平滑濾波歷史變數
        filtered_rv_rel = None
        filtered_tv_rel = None
        
        # 全域標籤地圖自標定狀態 (用作聯合剛體優化)
        global_marker_map = {}      # marker_id -> (R_to_origin, T_to_origin)
        marker_map_calibrated = False
        
        while True:
            req = calc_request_q.get()
            if req is None:   # 結束訊號
                break
            calc_busy.set()
            try:
                # 檢查是否需要重置時序歷史
                if reset_pose_history:
                    history_marker_poses.clear()
                    history_rel_pose = None
                    filtered_rv_rel = None
                    filtered_tv_rel = None
                    global_marker_map.clear()
                    marker_map_calibrated = False
                    reset_pose_history = False
                    print("🧹 時序位姿、平滑濾波歷史與標籤地圖已重置")
                    
                snap_cand      = req['cand']
                snap_vs        = req['view_state']
                u, v           = req['u'], req['v']

                # ---------- 在背景執行緒進行耗時的反畸變與灰階轉換 ----------
                imgA_bgr_, _ = process_view(req['raw_imgA'], mtxL_o, distL, newKL_o)
                snap_imgA_gray = cv2.cvtColor(imgA_bgr_, cv2.COLOR_BGR2GRAY)
                snap_imgA_gray = preprocess_gray(snap_imgA_gray, snap_vs['enable_clahe'])
                
                imgB_bgr_, _ = process_view(req['raw_imgB'], mtxL_o, distL, newKL_o)
                imgB_gray_ = cv2.cvtColor(imgB_bgr_, cv2.COLOR_BGR2GRAY)
                imgB_gray_ = preprocess_gray(imgB_gray_, snap_vs['enable_clahe'])

                # ---------- ArUco 位姿解算 ----------

                res_pose, marker_map_calibrated = get_joint_relative_pose(
                    snap_imgA_gray, imgB_gray_, KL, KL, ACTUAL_MARKER_SIZE_MM,
                    prev_marker_poses=history_marker_poses,
                    prev_rel_pose=history_rel_pose,
                    marker_map=global_marker_map,
                    map_calibrated=marker_map_calibrated
                )
                
                # 檢查算出的原始 baseline 是否在物理合理範圍內 (例如 5.0mm 到 35.0mm 之間)
                if res_pose is not None:
                    raw_baseline = res_pose[2]
                    if not (5.0 <= raw_baseline <= 35.0):
                        print(f"❌ 算出的原始 Baseline 異常 ({raw_baseline:.2f} mm)，判定為無視差退化或定位異常，忽略此幀")
                        res_pose = None
                        
                baseline_val = 0.0
                pose_valid = False
                pose_info = "未偵測到 ArUco"
                R_r, t_r = None, None
                cA_dict, cB_dict = {}, {}

                if res_pose is not None:
                    R_r, t_r, baseline_val, objA, shared, cA_dict, cB_dict, curr_marker_poses, rv_r = res_pose
                    pose_valid = True
                    
                    # 更新歷史標籤位姿
                    history_marker_poses = curr_marker_poses
                    
                    # 進行時序 EMA 平滑濾波
                    if ENABLE_POSE_SMOOTHING:
                        if filtered_rv_rel is None or filtered_tv_rel is None:
                            filtered_rv_rel = rv_r.copy()
                            filtered_tv_rel = t_r.copy()
                        else:
                            alpha = POSE_SMOOTHING_ALPHA
                            filtered_tv_rel = alpha * t_r + (1.0 - alpha) * filtered_tv_rel
                            filtered_rv_rel = alpha * rv_r + (1.0 - alpha) * filtered_rv_rel
                        
                        # 覆蓋 R_r 與 t_r 以便後續三角測量計算
                        t_r = filtered_tv_rel.copy()
                        rv_r_smooth = filtered_rv_rel.copy()
                        R_r, _ = cv2.Rodrigues(rv_r_smooth)
                        baseline_val = float(np.linalg.norm(t_r))
                        
                    # 更新歷史相對位姿供下一幀 extrinsic guess 使用
                    if ENABLE_POSE_SMOOTHING:
                        history_rel_pose = (filtered_rv_rel.copy(), filtered_tv_rel.copy())
                    else:
                        history_rel_pose = (rv_r.copy(), t_r.copy())
                        
                    direction = "向左" if t_r[0] > 0 else "向右"
                    pose_info = f"ArUco 定位(Bsl: {baseline_val:.1f}mm, {direction})"
                    print(f"  - 計算出的時序 Baseline: {baseline_val:.2f} mm")
                    
                    # 🌟 透過立體三角測量或聯合剛體 3D 點計算共享角點的 3D 座標，用來擬合平面
                    if marker_map_calibrated and objA is not None and len(objA) >= 3:
                        all_pts3d = objA
                    else:
                        all_pts3d = []
                        for mid in shared:
                            ptsA = cA_dict[mid] # 4x2
                            ptsB = cB_dict[mid] # 4x2
                            for i in range(4):
                                p3d = triangulate_point_3d(ptsA[i], ptsB[i], KL, KL, R_r, t_r)
                                all_pts3d.append(p3d)
                            
                    if len(all_pts3d) >= 3:
                        all_pts3d = np.array(all_pts3d)
                        c_local = np.mean(all_pts3d, axis=0)
                        _, _, Vt = np.linalg.svd(all_pts3d - c_local)
                        n_local = Vt[-1]
                        if np.dot(n_local, c_local) > 0: n_local = -n_local
                    else:
                        n_local, c_local = global_plane_n, global_plane_c
                else:
                    print("❌ 未偵測到共享 ArUco")
                    n_local, c_local = global_plane_n, global_plane_c
                kb, db = sift.detectAndCompute(imgB_gray_, None)

                # 更新 snap_cand
                snap_cand = dict(snap_cand)
                snap_cand.update({
                    'rgb': cv2.cvtColor(imgB_bgr_, cv2.COLOR_BGR2RGB),
                    'gray': imgB_gray_,
                    'kpB': kb, 'desB': db,
                    'plane_n': n_local, 'plane_c': c_local,
                    'pose_info': pose_info, 'pose_valid': pose_valid,
                    'baseline': baseline_val,
                    'cornersA': cA_dict, 'cornersB': cB_dict,
                    'curr_marker_poses': curr_marker_poses if pose_valid else {}
                })
                if pose_valid:
                    snap_cand.update({
                        'R_rel': R_r, 't_rel': t_r,
                        'F': compute_fundamental_matrix(KL, KL, R_r, t_r)
                    })

                # ---------- 特徵匹配與深度計算 ----------
                measure_res = compute_measure(u, v, snap_cand, snap_imgA_gray, snap_vs)

                # 打包回傳給主執行緒
                calc_result_q.put({
                    'measure_res': measure_res,
                    'snap_cand': snap_cand,
                    'cA_dict': cA_dict, 'cB_dict': cB_dict,
                    'pose_valid': pose_valid, 'pose_info': pose_info,
                    'baseline_val': baseline_val,
                    'imgA_bgr': imgA_bgr_, 'imgB_bgr': imgB_bgr_
                })
            except Exception as e:
                print(f"[calc_worker 錯誤] {e}")
            finally:
                calc_busy.clear()
                calc_request_q.task_done()

    worker_thread = threading.Thread(target=calc_worker, daemon=True)
    worker_thread.start()
    btn_lock_L.on_clicked(on_lock_L)
    btn_lock_R.on_clicked(on_lock_R)
    btn_calc.on_clicked(on_calc)
    btn_auto_calc.on_clicked(on_auto_calc)
    btn_hide_R.on_clicked(on_hide_R)
    btn_grad_toggle.on_clicked(on_grad_toggle)
    btn_norm_toggle.on_clicked(on_norm_toggle)
    btn_custom_plane.on_clicked(on_custom_plane)

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

    while plt.fignum_exists(fig.number):
        _t0 = _time.perf_counter()

        ret, frame = cap.read()
        if not ret: break
        _t1 = _time.perf_counter(); _t_cap += (_t1 - _t0) * 1000

        frame_left = frame[:, :w_alg]
        frame_buffer.append(frame_left)
        if len(frame_buffer) > 30:
            frame_buffer.popleft()
        _t2 = _time.perf_counter(); _t_buf += (_t2 - _t1) * 1000

        # 如果處於連續計算模式，自動觸發（非阻塞），限制計算時間間隔
        _cur_time = _time.perf_counter()
        if auto_calc_active and has_set_R and (_cur_time - _last_auto_calc_time >= AUTO_CALC_INTERVAL_SEC) and calc_request_q.empty() and not calc_busy.is_set():
            _last_auto_calc_time = _cur_time
            on_calc(None)
        _t3 = _time.perf_counter(); _t_calc_q += (_t3 - _t2) * 1000

        # 輪詢 result queue，有結果就在主執行緒更新 UI
        try:
            result = calc_result_q.get_nowait()
            measure_res = result['measure_res']
            snap_cand   = result['snap_cand']
            # 同步更新 current_cand，讓手動點選時可用
            current_cand.update(snap_cand)
            measure_results[current_cand['idx']] = measure_res
            # 更新 ArUco 顯示
            draw_aruco(ax_A, result['cA_dict'])
            draw_aruco(ax_B, result['cB_dict'])
            # 更新 imgA_gray 以便手動點選使用（nonlocal 更新）
            imgA_gray = cv2.cvtColor(result['imgA_bgr'], cv2.COLOR_BGR2GRAY)
            imgA_gray = preprocess_gray(imgA_gray, view_state['enable_clahe'])
            # 用 apply_measure_result 更新所有 UI
            all_d = [r['d'] for r in measure_results.values() if r['d'] is not None]
            summary = [f"F{current_cand['idx']}: {measure_res['d']:.1f}" if measure_res['d'] is not None else f"F{current_cand['idx']}: N/A"]
            apply_measure_result(measure_res, np.mean(all_d) if all_d else None, summary)
        except queue.Empty:
            pass
        _t4 = _time.perf_counter(); _t_result_q += (_t4 - _t3) * 1000

        # 決定顯示內容
        if live_L:
            display_L = frame_left.copy()
        else:
            display_L = locked_L
            
        if live_R:
            display_R = frame_left.copy()
        else:
            display_R = locked_R
            
        imgA_bgr, _ = process_view(display_L, mtxL_o, distL, newKL_o)
        imgB_bgr, _ = process_view(display_R, mtxL_o, distL, newKL_o)
        
        # 依前處理開關，動態切換顯示畫面（使肉眼可見差異）
        if view_state['enable_clahe']:
            gray_A = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
            gray_A_enh = preprocess_gray(gray_A, True)
            disp_A = cv2.cvtColor(gray_A_enh, cv2.COLOR_GRAY2RGB)
            
            gray_B = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY)
            gray_B_enh = preprocess_gray(gray_B, True)
            disp_B = cv2.cvtColor(gray_B_enh, cv2.COLOR_GRAY2RGB)
        else:
            disp_A = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB)
            disp_B = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB)
            
        _t5 = _time.perf_counter(); _t_proc += (_t5 - _t4) * 1000
        
        im_A.set_data(disp_A)
        im_B.set_data(disp_B)
        _t6 = _time.perf_counter(); _t_setdata += (_t6 - _t5) * 1000

        # FPS 計算與更新
        _fps_counter += 1
        _now = _time.perf_counter()
        _elapsed = _now - _fps_t0
        if _elapsed >= 0.5:
            _fps_val = _fps_counter / _elapsed
            fps_text.set_text(f"FPS: {_fps_val:.1f}")
            _fps_t0 = _now
            _fps_counter = 0

        # ---- Blit 渲染 ----
        if blit_state['needs_refresh'] or blit_state['bg'] is None:
            # 有 UI 更新（按鈕文字、散點、深度文字…等），做一次完整全圖繪製
            fig.canvas.draw()
            blit_state['bg'] = fig.canvas.copy_from_bbox(fig.bbox)
            blit_state['needs_refresh'] = False
        else:
            # 只更新影像與 FPS（最快路徑）
            fig.canvas.restore_region(blit_state['bg'])
        # 兩路徑都需要 draw_artist + blit，確保畫面不空白
        ax_A.draw_artist(im_A)
        if custom_plane_poly_artist is not None:
            ax_A.draw_artist(custom_plane_poly_artist)
        ax_A.draw_artist(scatter_A)
        ax_A.draw_artist(scatter_grad_inject)
        for a in custom_plane_artists:
            ax_A.draw_artist(a)
        
        if ax_B.get_visible():
            ax_B.draw_artist(im_B)
            
            # === 手動繪製覆蓋層（解決點、線被影像遮擋的問題） ===
            # 1. 散點與基礎幾何
            ax_B.draw_artist(scatter_B)
            ax_B.draw_artist(scatter_grad_match)
            ax_B.draw_artist(scatter_all_sift_B)
            ax_B.draw_artist(epi_line)
            ax_B.draw_artist(sift_rect)
            ax_B.draw_artist(sift_rect_center)
    
            # 2. 梯度 SIFT 連線
            for line in view_state.get('grad_lines', []):
                ax_B.draw_artist(line)
            if view_state.get('highlighted_grad_line_artist'):
                ax_B.draw_artist(view_state['highlighted_grad_line_artist'])
                
        # 繪製深度資訊文字，即使右圖隱藏也維持顯示
        ax_B.draw_artist(depth_text)

        # 3. ArUco 標記 (儲存在 ax.art 中)
        for ax in [ax_A, ax_B]:
            if hasattr(ax, 'art'):
                if ax == ax_B and not ax_B.get_visible():
                    continue
                for a in ax.art:
                    ax.draw_artist(a)
        # =========================================================

        ax_A.draw_artist(fps_text)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        _t7 = _time.perf_counter(); _t_pause += (_t7 - _t6) * 1000

        # 每 2 秒印出各階段平均耗時
        _perf_frames += 1
        if (_t7 - _perf_t0) >= 2.0:
            n = _perf_frames
            print(
                f"[耗時分析] {n}幀平均 | "
                f"cap.read={_t_cap/n:.1f}ms  "
                f"frame_buf={_t_buf/n:.2f}ms  "
                f"calc_q={_t_calc_q/n:.2f}ms  "
                f"result_q={_t_result_q/n:.2f}ms  "
                f"process_view={_t_proc/n:.1f}ms  "
                f"set_data={_t_setdata/n:.2f}ms  "
                f"plt.pause={_t_pause/n:.1f}ms  "
                f"| FPS={_fps_val:.1f}"
            )
            _t_cap=_t_buf=_t_calc_q=_t_result_q=_t_proc=_t_setdata=_t_pause=0.0
            _perf_frames = 0
            _perf_t0 = _t7
        
    cap.release()

if __name__ == "__main__":
    main()
