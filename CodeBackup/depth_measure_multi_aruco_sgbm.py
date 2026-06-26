"""
depth_measure_multi_aruco_sbs.py
================
互動式深度量測工具 (SBS 併排影像 + JSON 標定參數版本)。

特點：
- 支援 Side-by-Side (SBS) 併排影像輸入，自動從中線分割左右視景
- 整合 JSON 標定參數，支援左右相機不對稱的內參與畸變修正
- 雙內參精確幾何：三角測距、單應性映射與基本矩陣均使用獨立的 KL/KR
- 外參對比報告：自動比對 ArUco 動態估算的 RT 與 JSON 標定值，輸出誤差分析
- 連線視覺化：LightGlue 特徵匹配時，自動在左右圖間繪製洋紅色虛線
"""

import os, sys, glob, configparser, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")          # 可改為 "Qt5Agg" 視環境而定
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button, CheckButtons
import onnxruntime as ort

# ==================== 全局設定區 ====================
IMAGE_FOLDER          = "su01_test_images_1"  # 輸入 SBS 影像資料夾
TARGET_IMAGE_INDEX    = 2                          # 使用資料夾中的第幾張圖 (SBS 影像)
PARAMS_JSON_PATH      = "calibration_result.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM = 8.25#33#8.25                      # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度（影像較短邊同比例縮放）
FEATURE_MATCHING_MODE = 'LightGlue'                # 可選: 'LightGlue' 或 'OpenCV'
LG_ONNX_PATH          = "superpoint_lightglue_pipeline.onnx"
LG_SCORE_THRESH       = 0.5                       # LightGlue 匹配分數門檻
MATCH_SEARCH_RADIUS   = 7                         # 點點在匹配中搜尋的起始半徑（像素）
MATCH_SEARCH_RADIUS_MAX = 30                       # 動態搜尋半徑上限
TEMPLATE_HALF_SIZE    = 20                         # 模板匹配時的半窗大小（像素）
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
ENFORCE_COPLANAR      = True                       # 🌟 強制共面對齊優化
DYNAMIC_RADIUS_EXPANSION = True                    # 點擊若無特徵，是否動態擴大搜尋半徑
STICK_TO_EPIPOLAR     = True                       # 🌟 沿著極線搜尋 , 設成True的話, 會強制沿著極線做LK; False的話會全域做LK再投影回極線上(True: 1D 約束; False: 2D LK + 投影)
# ===================================================


# ─────────────────────────── 工具函式 ───────────────────────────

def load_json_camera_params(json_path):
    """從 JSON 載入 OpenCV 格式的標定參數"""
    if not os.path.exists(json_path):
        print(f"❌ 找不到參數檔案: {json_path}")
        return None, None, None, None, None, None
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
    
    return mtx_L, dist_L, mtx_R, dist_R, extrinsic, F_orig


def get_patch(img, pt, size):
    """安全地從影像中切出給定大小的 patch"""
    h, w = img.shape[:2]
    u, v = int(round(pt[0])), int(round(pt[1]))
    u0, u1 = u - size//2, u + size//2 + 1
    v0, v1 = v - size//2, v + size//2 + 1
    if u0 < 0 or v0 < 0 or u1 > w or v1 > h: return None
    return img[v0:v1, u0:u1]


def find_precise_match(imgA_gray, imgB_gray, pt_A, F, K_L, K_R, R_rel, t_rel, plane_normal, plane_center):
    """
    終極精確匹配流程 (Homography Planar Rectification)：
    1. 計算 3D 基準面的單應性矩陣 H_AB = K_R @ (R_rel + t_rel @ n.T / d) @ K_L_inv
    2. 將右圖透視扭曲回左圖的正視角，完全消滅放射形變
    3. 進行純平移的 2D 匹配與亞像素插值
    4. 將座標使用 H_AB 反向投射回真實的右圖像素點
    """
    if plane_normal is None or plane_center is None:
        return None
        
    h, w = imgA_gray.shape
    d = np.dot(plane_normal, plane_center)
    H_3D = R_rel + (t_rel @ plane_normal.reshape(1, 3)) / d
    K_L_inv = np.linalg.inv(K_L.astype(np.float64))
    H_AB = K_R @ H_3D @ K_L_inv
    H_BA = np.linalg.inv(H_AB)
    
    imgB_warped = cv2.warpPerspective(imgB_gray, H_BA, (w, h), flags=cv2.INTER_LINEAR)
    
    patch_size = 15
    u, v = int(round(pt_A[0])), int(round(pt_A[1]))
    u0_a, u1_a = max(0, u - patch_size), min(w, u + patch_size + 1)
    v0_a, v1_a = max(0, v - patch_size), min(h, v + patch_size + 1)
    patch_left = imgA_gray[v0_a:v1_a, u0_a:u1_a]
    if patch_left.size == 0: return None
    
    search_range = 30
    x_min, x_max = max(0, int(u) - search_range), min(w, int(u) + search_range + 1)
    y_min, y_max = max(0, int(v) - search_range), min(h, int(v) + search_range + 1)
    
    roi_x0, roi_y0 = max(0, x_min - patch_size), max(0, y_min - patch_size)
    roi_x1, roi_y1 = min(w, x_max + patch_size + 1), min(h, y_max + patch_size + 1)
    
    roi_right = imgB_warped[roi_y0:roi_y1, roi_x0:roi_x1]
    if roi_right.shape[0] < patch_left.shape[0] or roi_right.shape[1] < patch_left.shape[1]:
        return None
        
    res = cv2.matchTemplate(roi_right, patch_left, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.4: return None
    
    x_peak, y_peak = max_loc
    sub_x, sub_y = float(x_peak), float(y_peak)
    if 0 < x_peak < res.shape[1] - 1 and 0 < y_peak < res.shape[0] - 1:
        dx = (res[y_peak, x_peak-1] - res[y_peak, x_peak+1]) / (2.0 * (res[y_peak, x_peak-1] - 2*res[y_peak, x_peak] + res[y_peak, x_peak+1]) + 1e-6)
        dy = (res[y_peak-1, x_peak] - res[y_peak+1, x_peak]) / (2.0 * (res[y_peak-1, x_peak] - 2*res[y_peak, x_peak] + res[y_peak+1, x_peak]) + 1e-6)
        if abs(dx) < 1.0: sub_x += dx
        if abs(dy) < 1.0: sub_y += dy
        
    best_x_warped = roi_x0 + sub_x + patch_left.shape[1]//2.0
    best_y_warped = roi_y0 + sub_y + patch_left.shape[0]//2.0
    
    pt_warped_homo = np.array([best_x_warped, best_y_warped, 1.0])
    pt_B_homo = H_AB @ pt_warped_homo
    true_x_B = float(pt_B_homo[0] / pt_B_homo[2])
    true_y_B = float(pt_B_homo[1] / pt_B_homo[2])
    
    return (true_x_B, true_y_B)


def get_joint_relative_pose(imgA_gray, imgB_gray, K_L, K_R, marker_size_mm, active_ids=None):
    """
    偵測兩張影像中所有相同的 ArUco 標籤，
    利用影像 A 估算出標籤的 3D 座標，與影像 B 的 2D 角點做 joint solvePnP。
    """
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cornersA, idsA, _ = detector.detectMarkers(imgA_gray)
        cornersB, idsB, _ = detector.detectMarkers(imgB_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cornersA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
        cornersB, idsB, _ = cv2.aruco.detectMarkers(imgB_gray, dict_4x4, parameters=params)

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if cornersA is not None:
        for c in cornersA: cv2.cornerSubPix(imgA_gray, c, (5, 5), (-1, -1), term)
    if cornersB is not None:
        for c in cornersB: cv2.cornerSubPix(imgB_gray, c, (5, 5), (-1, -1), term)

    if idsA is None or idsB is None: return None
    idsA_list = [i[0] for i in idsA]
    idsB_list = [i[0] for i in idsB]
    shared_ids = list(set(idsA_list).intersection(set(idsB_list)))
    if active_ids is not None: shared_ids = [m for m in shared_ids if m in active_ids]
    if len(shared_ids) == 0: return None

    half = marker_size_mm / 2.0
    canonical_pts = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    all_obj_pts_A = []
    all_img_pts_B = []
    corners_dict_A, corners_dict_B = {}, {}

    for marker_id in shared_ids:
        idx_A, idx_B = idsA_list.index(marker_id), idsB_list.index(marker_id)
        c_A, c_B = cornersA[idx_A][0], cornersB[idx_B][0]
        corners_dict_A[marker_id], corners_dict_B[marker_id] = c_A, c_B
        ok, rvec_A, tvec_A = cv2.solvePnP(canonical_pts, c_A, K_L, np.zeros(5))
        if not ok: continue
        R_A, _ = cv2.Rodrigues(rvec_A)
        pts_3d_A = (R_A @ canonical_pts.T).T + tvec_A.T
        all_obj_pts_A.append(pts_3d_A)
        all_img_pts_B.append(c_B)

    if len(all_obj_pts_A) == 0: return None
    all_obj_pts_A = np.vstack(all_obj_pts_A).astype(np.float32)
    all_img_pts_B = np.vstack(all_img_pts_B).astype(np.float32)

    if ENFORCE_COPLANAR and len(shared_ids) >= 2:
        plane_center = np.mean(all_obj_pts_A, axis=0)
        pts_centered = all_obj_pts_A - plane_center
        _, _, Vt = np.linalg.svd(pts_centered)
        normal = Vt[-1]
        if np.dot(normal, plane_center) > 0: normal = -normal
        d = np.dot(normal, plane_center)
        K_L_inv = np.linalg.inv(K_L.astype(np.float64))
        refined_obj_pts_A = []
        for marker_id in shared_ids:
            c_A = corners_dict_A[marker_id]
            rays = np.hstack([c_A, np.ones((4, 1))]) @ K_L_inv.T
            t_vals = d / (rays @ normal)
            pts_3d_refined = rays * t_vals[:, np.newaxis]
            pts_3d_refined = pts_3d_refined.astype(np.float32)
            refined_obj_pts_A.append(pts_3d_refined)
        all_obj_pts_A = np.vstack(refined_obj_pts_A).astype(np.float32)

    ok, rvec_rel, tvec_rel = cv2.solvePnP(all_obj_pts_A, all_img_pts_B, K_R, np.zeros(5))
    if not ok: return None
    R_rel, _ = cv2.Rodrigues(rvec_rel)
    t_rel = tvec_rel    # (3, 1)
    baseline_mm = float(np.linalg.norm(t_rel))

    return R_rel, t_rel, baseline_mm, all_obj_pts_A, shared_ids, corners_dict_A, corners_dict_B


def run_lightglue(lg_session, imgA_gray, imgB_gray):
    t0, t1 = imgA_gray.astype(np.float32)/255.0, imgB_gray.astype(np.float32)/255.0
    inp = np.expand_dims(np.stack([t0, t1], axis=0), axis=1)
    outputs = lg_session.run(['keypoints', 'matches', 'mscores'], {"images": inp})
    kpts, matches, scores = outputs
    valid = [(kpts[0, int(m[1])], kpts[1, int(m[2])]) for m, s in zip(matches, scores) if s > LG_SCORE_THRESH]
    if len(valid) < 4: return None, None
    return np.array([v[0] for v in valid], dtype=np.float32), np.array([v[1] for v in valid], dtype=np.float32)


def run_opencv_matching(imgA_gray, imgB_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgA_gray, None)
    kp2, des2 = sift.detectAndCompute(imgB_gray, None)
    if des1 is None or len(des1) < 2 or des2 is None or len(des2) < 2: return None, None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    if len(good) < 4: return None, None
    ptsA = np.float32([kp1[m.queryIdx].pt for m in good])
    ptsB = np.float32([kp2[m.trainIdx].pt for m in good])
    return ptsA, ptsB


def compute_fundamental_matrix(K_L, K_R, R_rel, t_rel):
    t = t_rel.flatten()
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    E = tx @ R_rel
    K_R_inv, K_L_inv = np.linalg.inv(K_R.astype(np.float64)), np.linalg.inv(K_L.astype(np.float64))
    return K_R_inv.T @ E @ K_L_inv


def triangulate_point_3d(pt_A, pt_B, K_L, K_R, R_rel, t_rel):
    P0 = (K_L.astype(np.float64) @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
    P1 = (K_R.astype(np.float64) @ np.hstack([R_rel, t_rel])).astype(np.float32)
    pts4d = cv2.triangulatePoints(P0, P1, np.array([[pt_A[0]], [pt_A[1]]]), np.array([[pt_B[0]], [pt_B[1]]]))
    pt3d = (pts4d[:3] / pts4d[3]).flatten()
    if pt3d[2] <= 0 or pt3d[2] > MAX_DEPTH_MM: return None
    return pt3d


def epipolar_line(F, pt, img_w):
    l = F @ np.array([pt[0], pt[1], 1.0])
    a, b, c = l
    if abs(b) > 1e-8: return (0, int(-c/b)), (img_w-1, int(-(a*(img_w-1)+c)/b))
    return (int(-c/a), 0), (int(-c/a), img_w-1)




# ─────────────────────────── 主程式 ───────────────────────────

def main():
    # 1. 載入影像路徑
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))
    if len(image_paths) == 0:
        print(f"❌ 在 {IMAGE_FOLDER} 找不到圖片。"); sys.exit(1)

    # 2. 載入標定參數
    mtx_L_orig, dist_L, mtx_R_orig, dist_R, extrinsic_json, F_from_json = load_json_camera_params(PARAMS_JSON_PATH)
    if mtx_L_orig is None: sys.exit(1)
    
    # 取得原始解析度 (假設為 c_x * 2 左右)
    orig_w = mtx_L_orig[0, 2] * 2.0

    # 3. 處理 SBS 影像
    target_path = image_paths[min(TARGET_IMAGE_INDEX, len(image_paths)-1)]
    print(f"🎬 載入 SBS 影像: {target_path}")
    full_img = cv2.imread(target_path)
    h_raw, w_raw = full_img.shape[:2]
    half_w = w_raw // 2
    imgL_raw, imgR_raw = full_img[:, :half_w], full_img[:, half_w:]

    # 去畸變與縮放
    new_K_L_orig, _ = cv2.getOptimalNewCameraMatrix(mtx_L_orig, dist_L, (half_w, h_raw), 1, (half_w, h_raw))
    new_K_R_orig, _ = cv2.getOptimalNewCameraMatrix(mtx_R_orig, dist_R, (half_w, h_raw), 1, (half_w, h_raw))

    def process_view(img, K_orig, dist, new_K):
        undist = cv2.undistort(img, K_orig, dist, None, new_K)
        scale = TARGET_W / undist.shape[1]
        target_h = int(undist.shape[0] * scale)
        return cv2.resize(undist, (TARGET_W, target_h)), scale

    imgA_bgr, scale = process_view(imgL_raw, mtx_L_orig, dist_L, new_K_L_orig)
    imgB_bgr, _     = process_view(imgR_raw, mtx_R_orig, dist_R, new_K_R_orig)
    imgA_gray, imgB_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY)
    h, w = imgA_gray.shape

    # 縮放 K
    K_L, K_R = new_K_L_orig.copy().astype(np.float64), new_K_R_orig.copy().astype(np.float64)
    for mk in [K_L, K_R]:
        mk[0, 0] *= scale; mk[1, 1] *= scale; mk[0, 2] *= scale; mk[1, 2] *= scale

    # 4. ArUco 姿態估測
    print("偵測 ArUco 中...")
    res = get_joint_relative_pose(imgA_gray, imgB_gray, K_L, K_R, ACTUAL_MARKER_SIZE_MM)
    if res is None:
        print("❌ 無法建立相對位姿，結束。"); sys.exit(1)
        
    R_rel, t_rel, baseline_mm, all_obj_pts_A, shared_ids, corners_dict_A, corners_dict_B = res
    global_shared_ids = shared_ids.copy()
    
    # --- 🌟 恢復動態模式：使用 ArUco 估算出的當前外參來計算基礎矩陣 F ---
    F = compute_fundamental_matrix(K_L, K_R, R_rel, t_rel)
    print(f"✅ 已使用「動態估算外參」計算 F 矩陣 (Baseline: {baseline_mm:.2f}mm)")
    
    # 僅作為參考：對比 JSON 中的標定值
    b_calib = np.linalg.norm(np.array(extrinsic_json['T']))
    print(f"ℹ️ 標定參考值: Baseline = {b_calib:.2f}mm")

    # 5. 外部參數對比報告
    def print_extrinsic_report(R_est, t_est, b_est):
        if extrinsic_json:
            print("\n" + "="*40 + "\n📊 外部參數對比報告 (ArUco 估算 vs 標定 JSON)")
            R_json, T_json = np.array(extrinsic_json['R']), np.array(extrinsic_json['T'])
            b_json = np.linalg.norm(T_json)
            print(f"Baseline: 估算={b_est:.2f} mm | JSON={b_json:.2f} mm | 誤差={abs(b_est - b_json):.2f} mm")
            rod, _ = cv2.Rodrigues(R_est @ R_json.T)
            print(f"旋轉角誤差: {np.linalg.norm(rod)*180/np.pi:.4f} 度")
            cos_sim = np.dot(T_json.flatten()/b_json, t_est.flatten()/b_est)
            print(f"位移向量方向餘弦相似度: {cos_sim:.6f} (越接近 1.0 越好)\n" + "="*40 + "\n")

    print_extrinsic_report(R_rel, t_rel, baseline_mm)

    # 6. 平面擬合與匹配
    plane_normal, plane_center = None, None
    if len(all_obj_pts_A) >= 3:
        plane_center = np.mean(all_obj_pts_A, axis=0)
        _, _, Vt = np.linalg.svd(all_obj_pts_A - plane_center)
        plane_normal = Vt[-1]

    if FEATURE_MATCHING_MODE == 'LightGlue':
        lg_session = ort.InferenceSession(LG_ONNX_PATH, providers=['CPUExecutionProvider'])
        ptsA_lg, ptsB_lg = run_lightglue(lg_session, imgA_gray, imgB_gray)
    else: ptsA_lg, ptsB_lg = run_opencv_matching(imgA_gray, imgB_gray)

    # 6.5 計算 SGBM 全域視差圖
    print("⏳ 計算 SGBM 全域視差圖...")
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*12,  # 192px 範圍
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity_map = stereo.compute(imgA_gray, imgB_gray).astype(np.float32) / 16.0
    print("✅ SGBM 視差圖計算完成")

    # 7. UI
    imgA_rgb = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB)
    imgB_rgb = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB)

    # === 預先產生特徵點與匹配連線圖 ===
    vis_img = np.hstack([imgA_rgb, imgB_rgb]).copy()
    wA = imgA_rgb.shape[1]
    if ptsA_lg is not None and len(ptsA_lg) > 0:
        draw_count = min(len(ptsA_lg), 500)
        np.random.seed(42)
        indices = np.random.choice(len(ptsA_lg), draw_count, replace=False)
        for i in indices:
            pt1 = (int(ptsA_lg[i][0]), int(ptsA_lg[i][1]))
            pt2 = (int(ptsB_lg[i][0] + wA), int(ptsB_lg[i][1]))
            color = tuple(int(x) for x in np.random.randint(50, 255, 3))
            cv2.circle(vis_img, pt1, 3, color, -1)
            cv2.circle(vis_img, pt2, 3, color, -1)
            cv2.line(vis_img, pt1, pt2, color, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"SBS 跨鏡頭深度量測　│　Baseline={baseline_mm:.1f}mm　│　在左圖點擊以量測深度",
        fontsize=13
    )

    ax_A, ax_B = axes
    ax_A.set_title(f"影像 A (左視角)  ← 點擊量測", fontsize=11)
    ax_B.set_title(f"影像 B (右視角)", fontsize=11)
    im_A = ax_A.imshow(imgA_rgb)
    im_B = ax_B.imshow(imgB_rgb)
    for ax in axes:
        ax.axis("off")

    def draw_all_aruco(ax, corners_dict, active_ids=None):
        # 先清除舊的 plot/scatter (簡單作法是清除 axes 上的特定 artists)
        # 這裡為了簡單，我們直接保留舊的，但如果是重繪模式，建議 ax.clear()
        # 不過 ax.clear() 會把影像也清掉。所以我們改用清單管理
        if not hasattr(ax, 'aruco_artists'): ax.aruco_artists = []
        for art in ax.aruco_artists: art.remove()
        ax.aruco_artists = []

        colors = ['cyan', 'magenta', 'yellow', 'orange']
        for i, (marker_id, corners) in enumerate(corners_dict.items()):
            is_active = (active_ids is None or marker_id in active_ids)
            color = colors[i % len(colors)]
            alpha = 0.8 if is_active else 0.2
            pts = np.vstack((corners, corners[0]))
            l, = ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, linestyle='-', alpha=alpha)
            s = ax.scatter(corners[:, 0], corners[:, 1], color=color, s=40, zorder=3, edgecolors='black', alpha=alpha)
            t = ax.text(corners[0, 0], corners[0, 1]-15, f"ID:{marker_id}", color=color, fontsize=9, fontweight='bold', alpha=alpha)
            ax.aruco_artists.extend([l, s, t])

    draw_all_aruco(ax_A, corners_dict_A)
    draw_all_aruco(ax_B, corners_dict_B)

    # 覆蓋層：用來繪製點擊資訊
    scatter_A = ax_A.scatter([], [], s=120, c='red', zorder=5, marker='x', linewidths=2)
    scatter_B = ax_B.scatter([], [], s=120, c='lime', zorder=5, marker='x', linewidths=2)
    epi_line, = ax_B.plot([], [], 'yellow', lw=1.5, alpha=0.7, zorder=4)
    depth_text = ax_B.text(0.02, 0.05, "", transform=ax_B.transAxes,
                           color='white', fontsize=13, fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.6))
    method_text = ax_A.text(0.02, 0.05, "", transform=ax_A.transAxes,
                            color='cyan', fontsize=10,
                            bbox=dict(facecolor='black', alpha=0.5))

    # ====== 切換特徵匹配檢視按鈕與介面 ======
    ax_match = fig.add_axes([0.05, 0.05, 0.9, 0.85])
    ax_match.imshow(vis_img)
    ax_match.axis("off")
    ax_match.set_title(f"特徵匹配檢視 ({FEATURE_MATCHING_MODE})", fontsize=13)
    ax_match.set_visible(False)

    ax_button = fig.add_axes([0.45, 0.01, 0.1, 0.05])
    btn_toggle = Button(ax_button, '看特徵連線圖')

    view_state = {'show_match': False, 'use_sgbm': True, 'manual_mode': False, 'manual_pt_A': None, 'precise_match': True, 'lg_lines': []}
    pan_state = {'pressing': False, 'x': 0.0, 'y': 0.0, 'ax': None, 'dragged': False}
    def on_toggle(event):
        view_state['show_match'] = not view_state['show_match']
        if view_state['show_match']:
            ax_A.set_visible(False); ax_B.set_visible(False); ax_match.set_visible(True)
            btn_toggle.label.set_text('返回量測模式')
        else:
            ax_A.set_visible(True); ax_B.set_visible(True); ax_match.set_visible(False)
            btn_toggle.label.set_text('看特徵連線圖')
        fig.canvas.draw_idle()

    btn_toggle.on_clicked(on_toggle)

    # ====== SGBM 與手動模式控制 ======
    ax_options = fig.add_axes([0.13, 0.82, 0.16, 0.08])
    opt_labels = ["啟用 SGBM 保底", "手動匹配模式"]
    check_opt = CheckButtons(ax_options, opt_labels, [True, False])
    for text in check_opt.labels: text.set_fontsize(9)
        
    def on_opt_change(label):
        opt_states = check_opt.get_status()
        view_state['use_sgbm'], view_state['manual_mode'] = opt_states
        if view_state['manual_mode']:
            depth_text.set_text("手動模式等待中\n請在左圖點選特徵點")
        else:
            view_state['manual_pt_A'] = None
            depth_text.set_text("自動模式：在左圖點擊\n(等待量測)")
        fig.canvas.draw_idle()
        
    check_opt.on_clicked(on_opt_change)

    # ====== 新增：ArUco 擬合平面控制 ======
    ax_check = fig.add_axes([0.01, 0.75, 0.08, min(0.05 * len(global_shared_ids), 0.25)])
    check_labels = [f"ID:{m}" for m in global_shared_ids]
    check = CheckButtons(ax_check, check_labels, [True] * len(global_shared_ids))
    for text in check.labels: text.set_fontsize(9)

    measure_state = {'pt3d': None, 'method': "", 'click_u': None, 'click_v': None, 'manual_match_pt': None}

    def update_plane(label=None):
        nonlocal plane_normal, plane_center, R_rel, t_rel, baseline_mm, all_obj_pts_A, corners_dict_A, corners_dict_B, F
        active_ids = [m for i, m in enumerate(global_shared_ids) if check.get_status()[i]]
        if len(active_ids) == 0:
            plane_normal = plane_center = None
            update_measure_text(); return
            
        res = get_joint_relative_pose(imgA_gray, imgB_gray, K_L, K_R, ACTUAL_MARKER_SIZE_MM, active_ids=active_ids)
        if res is None: return
            
        R_rel, t_rel, baseline_mm, all_obj_pts_A, _, corners_dict_A, corners_dict_B = res
        F = compute_fundamental_matrix(K_L, K_R, R_rel, t_rel)
        
        # 更新報表與標題
        print(f"🔄 重新計算 RT (使用 {len(active_ids)} 個標籤)...")
        print_extrinsic_report(R_rel, t_rel, baseline_mm)
        fig.suptitle(f"SBS 跨鏡頭深度量測　│　Baseline={baseline_mm:.1f}mm　│　在左圖點擊以量測深度", fontsize=13)
        draw_all_aruco(ax_A, corners_dict_A, active_ids)
        draw_all_aruco(ax_B, corners_dict_B, active_ids)

        if len(all_obj_pts_A) >= 3:
            plane_center = np.mean(all_obj_pts_A, axis=0)
            pts_centered = all_obj_pts_A - plane_center
            _, _, Vt = np.linalg.svd(pts_centered); plane_normal = Vt[-1]
            if np.dot(plane_normal, plane_center) > 0: plane_normal = -plane_normal
        else: plane_normal = plane_center = None
            
        if measure_state.get('click_u') is not None:
            do_measure(measure_state['click_u'], measure_state['click_v'], manual_match_pt=measure_state.get('manual_match_pt'))
        else: update_measure_text()
    
    check.on_clicked(update_plane)

    def update_measure_text():
        pt3d = measure_state['pt3d']
        if pt3d is None:
            depth_text.set_text("找不到點或深度無效\n（嘗試點選有紋理的位置）")
            return
        Z = float(pt3d[2])
        dist_str = ""
        if plane_normal is not None and plane_center is not None:
             dist_val = abs(np.dot(plane_normal, pt3d - plane_center))
             dist_str = f"\n距選取平面：{dist_val:.1f} mm"
        depth_text.set_text(f"深度：{Z:.1f} mm\n({Z/10:.1f} cm){dist_str}")
        method_text.set_text(f"方法：{measure_state['method']}")
        fig.canvas.draw_idle()

    # ====== 滑鼠拖曳與縮放 ======
    def snap_to_aruco_corner(x, y, corners_dict, threshold=15):
        best_pt, min_d = (x, y), threshold
        for cid in corners_dict:
            for pt in corners_dict[cid]:
                d = np.linalg.norm(pt - np.array([x, y]))
                if d < min_d: min_d = d; best_pt = (float(pt[0]), float(pt[1]))
        return best_pt

    pan_state = {'pressing': False, 'x': None, 'y': None, 'ax': None, 'dragged': False}

    def on_press(event):
        if event.inaxes not in (ax_A, ax_B, ax_match) or event.button != 1: return
        # 如果點擊的是按鈕或 CheckBox 區域， event.inaxes 會是那些小 axes，不會進到這裡
        pan_state.update({'pressing': True, 'dragged': False, 'x': event.x, 'y': event.y, 'ax': event.inaxes})

    def on_release(event):
        if not pan_state['pressing']: return
        pan_state['pressing'] = False
        if not pan_state['dragged'] and event.xdata is not None:
            if pan_state['ax'] == ax_match: return # 匹配圖不處理量測點擊
            
            # 手動模式或自動模式點擊左圖時，嘗試吸附 ArUco 角點
            click_x, click_y = float(event.xdata), float(event.ydata)
            if pan_state['ax'] == ax_A:
                click_x, click_y = snap_to_aruco_corner(click_x, click_y, corners_dict_A)

            if not view_state['manual_mode']:
                if pan_state['ax'] == ax_A: do_measure(click_x, click_y)
            else:
                if pan_state['ax'] == ax_A:
                    view_state['manual_pt_A'] = (click_x, click_y)
                    scatter_A.set_offsets([[click_x, click_y]])
                    p0, p1 = epipolar_line(F, view_state['manual_pt_A'], w)
                    epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                    depth_text.set_text("手動模式等待中...\n請在右圖黃色外極線上點選對應點")
                    fig.canvas.draw_idle()
                elif pan_state['ax'] == ax_B:
                    if view_state['manual_pt_A']:
                        click_x_B, click_y_B = snap_to_aruco_corner(float(event.xdata), float(event.ydata), corners_dict_B)
                        do_measure(view_state['manual_pt_A'][0], view_state['manual_pt_A'][1], manual_match_pt=(click_x_B, click_y_B))

    def on_motion(event):
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
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in (ax_A, ax_B, ax_match): return
        ax, f = event.inaxes, 1.2 if event.button == 'down' else 1/1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim([x - (x-xl[0])*f, x + (xl[1]-x)*f]); ax.set_ylim([y - (y-yl[0])*f, y + (yl[1]-y)*f])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('button_press_event', on_press)

    def do_measure(click_u, click_v, manual_match_pt=None):
        print(f"\n🖱️ 點擊座標 (左圖): ({click_u:.1f}, {click_v:.1f})")
        measure_state.update({'click_u': click_u, 'click_v': click_v, 'manual_match_pt': manual_match_pt})
        for l in view_state['lg_lines']: l.remove()
        view_state['lg_lines'] = []
        match_pt, method = None, ""
        
        if manual_match_pt: match_pt, method = np.array(manual_match_pt), "手動匹配"
        
        # (0) 優先檢查點擊是否吸附在 ArUco 角點
        if match_pt is None:
            for mid in corners_dict_A:
                cA, cB = corners_dict_A[mid], corners_dict_B[mid]
                dists_c = np.linalg.norm(cA - np.array([click_u, click_v]), axis=1)
                if np.min(dists_c) < 5:
                    idx = np.argmin(dists_c)
                    click_u, click_v = float(cA[idx, 0]), float(cA[idx, 1])
                    match_pt, method = cB[idx], "ArUco角點直連"
                    print(f"  [ArUco Snap] 偵測到角點吸附：ID:{mid}, Corner:{idx}")
                    break

        # (a) 特徵庫匹配
        if match_pt is None and ptsA_lg is not None:
            dists = np.linalg.norm(ptsA_lg - np.array([click_u, click_v]), axis=1)
            curr_r = MATCH_SEARCH_RADIUS
            in_rad = np.where(dists < curr_r)[0]
            
            if DYNAMIC_RADIUS_EXPANSION and len(in_rad) == 0:
                while len(in_rad) == 0 and curr_r < MATCH_SEARCH_RADIUS_MAX:
                    curr_r += 2
                    in_rad = np.where(dists < curr_r)[0]
                if len(in_rad) > 0: print(f"  [Dynamic] 擴展半徑至 {curr_r} 以尋找特徵")

            if len(in_rad) > 0:
                off = ptsB_lg[in_rad] - ptsA_lg[in_rad]
                weights = 1.0 / (dists[in_rad] + 1e-5)**2
                match_pt = np.array([click_u, click_v]) + np.sum(off * weights[:, np.newaxis], axis=0) / np.sum(weights)
                method = FEATURE_MATCHING_MODE + (f"(r={curr_r})" if curr_r > MATCH_SEARCH_RADIUS else "")
                for i in in_rad:
                    con = ConnectionPatch(xyA=ptsA_lg[i], xyB=ptsB_lg[i], coordsA="data", coordsB="data", 
                                          axesA=ax_A, axesB=ax_B, color="magenta", lw=1.2, ls="--")
                    ax_B.add_artist(con); view_state['lg_lines'].append(con)

        # (b) SGBM 視差圖保底
        if match_pt is None:
            if view_state['use_sgbm']:
                v_idx, u_idx = int(round(click_v)), int(round(click_u))
                if 0 <= v_idx < h and 0 <= u_idx < w:
                    disp = disparity_map[v_idx, u_idx]
                    if disp > 0:
                        match_pt = np.array([click_u - disp, click_v])
                        method = f"SGBM(d={disp:.1f})"
                        print(f"  [SGBM] 查找成功: Disparity={disp:.1f}")
                    else:
                        print(f"  [SGBM Failed] 該點視差無效 ({disp:.1f})")
                
        if match_pt is None:
            print(f"  [Search Failed] 特徵點與 SGBM 均無法匹配")
            update_measure_text()
            return

        if match_pt is not None:
             # (c) 極線修正與投影
             l_B = F @ np.array([click_u, click_v, 1.0])
             denom_l = l_B[0]**2 + l_B[1]**2
             dist_l = (l_B[0]*match_pt[0] + l_B[1]*match_pt[1] + l_B[2]) / np.sqrt(denom_l) if denom_l > 1e-9 else 0.0
             
             # 強制修正回極線上 (亞像素對齊)
             proj_pt = np.array([match_pt[0] - l_B[0]/np.sqrt(denom_l)*dist_l, match_pt[1] - l_B[1]/np.sqrt(denom_l)*dist_l])
             
             pt3d = triangulate_point_3d((click_u, click_v), proj_pt, K_L, K_R, R_rel, t_rel)
             measure_state['pt3d'] = pt3d
             measure_state['method'] = method + "+極線對齊"
             
             print(f"📊 匹配資訊:")
             print(f"   - 原始座標 (R): ({match_pt[0]:.2f}, {match_pt[1]:.2f})")
             print(f"   - 垂直極線誤差: {abs(dist_l):.4f} 像素")
             print(f"   - 使用方法: {measure_state['method']}")
             
             if pt3d is not None:
                 scatter_A.set_offsets([[click_u, click_v]])
                 scatter_B.set_offsets([[proj_pt[0], proj_pt[1]]])
                 p0, p1 = epipolar_line(F, (click_u, click_v), w)
                 epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        update_measure_text()

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
