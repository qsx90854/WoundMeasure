"""
depth_measure_multi_aruco_sbs_camera.py
================
互動式深度量測工具 (SBS 併排影片 + JSON 標定參數版本)。

特點：
- 支援影片輸入，可指定左圖幀與多個右圖候選幀
- 整合 JSON 標定參數，支援左右相機不對稱的內參與畸變修正
- 雙內參精確幾何：三角測距、單應性映射與基本矩陣均使用獨立的 KL/KR
- 多幀平均量測：點擊左圖後同時計算所有候選幀深度並平均
- 整合 LightGlue 與 Grad-SIFT 混合匹配演算法
- 動態 UI：提供右圖候選幀切換選單與匹配狀態切換
"""

import os, sys, glob, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import onnxruntime as ort

# ==================== 全局設定區 ====================
VIDEO_PATH            = "video_20260505_155449.mp4"           # 輸入影片路徑
videoframe_leftImage_index = 40                   # 指定哪一幀當左圖
videoframe_total_number    = 3                     # 右圖候選張數
videoframe_left_to_right_gap = 30                  # 左圖與第一張右圖的間距
videoframe_right_to_right_gap = 5                 # 右圖候選幀之間的間距
PARAMS_JSON_PATH      = "calibration_result.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM = 16.5                       # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
ENFORCE_COPLANAR      = True                       # 強制共面對齊優化
SAVE_ARUCO_DEBUG_IMG  = False                      # 是否存出 ArUco 偵測結果圖片

FEATURE_MATCHING_MODE = 'LightGlue'                # 可選: 'LightGlue' 或 'OpenCV'
LG_ONNX_PATH          = "superpoint_lightglue_pipeline.onnx"
LG_SCORE_THRESH       = 0.5                       # LightGlue 匹配分數門檻
MATCH_SEARCH_RADIUS   = 7                         # 點點在匹配中搜尋的起始半徑
MATCH_SEARCH_RADIUS_MAX = 30                       # 動態搜尋半徑上限
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
    pts4d = cv2.triangulatePoints(P0, P1, np.array([[pt_A[0]], [pt_A[1]]]), np.array([[pt_B[0]], [pt_B[1]]]))
    pt3d = (pts4d[:3] / pts4d[3]).flatten()
    if pt3d[2] <= 0 or pt3d[2] > MAX_DEPTH_MM: return None
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
    patch_size = 15
    u, v = int(round(pt_A[0])), int(round(pt_A[1]))
    u0, u1 = max(0, u-patch_size), min(w, u+patch_size+1)
    v0, v1 = max(0, v-patch_size), min(h, v+patch_size+1)
    patch_l = imgA_gray[v0:v1, u0:u1]
    if patch_l.size == 0: return None
    search = 30
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

def compute_global_plane(imgA_gray, K_L, marker_size_mm):
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cA, idsA, _ = detector.detectMarkers(imgA_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
    if idsA is None or len(idsA) < 1: return None, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for c in cA: cv2.cornerSubPix(imgA_gray, c, (5, 5), (-1, -1), term)
    
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

def get_joint_relative_pose(imgA_gray, imgB_gray, K_L, K_R, marker_size_mm, global_plane_n=None, global_plane_c=None):
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cA, idsA, _ = detector.detectMarkers(imgA_gray)
        cB, idsB, _ = detector.detectMarkers(imgB_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
        cB, idsB, _ = cv2.aruco.detectMarkers(imgB_gray, dict_4x4, parameters=params)
    
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if cA is not None:
        for c in cA: cv2.cornerSubPix(imgA_gray, c, (5, 5), (-1, -1), term)
    if cB is not None:
        for c in cB: cv2.cornerSubPix(imgB_gray, c, (5, 5), (-1, -1), term)

    if idsA is None or idsB is None: return None
    idsA_l, idsB_l = [i[0] for i in idsA], [i[0] for i in idsB]
    shared = list(set(idsA_l).intersection(set(idsB_l)))
    if not shared: return None
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    objA, imgB = [], []
    cA_dict, cB_dict = {}, {}
    for mid in shared:
        idxA, idxB = idsA_l.index(mid), idsB_l.index(mid)
        cA_dict[mid], cB_dict[mid] = cA[idxA][0], cB[idxB][0]
        ok, rv, tv = cv2.solvePnP(canon, cA[idxA][0], K_L, np.zeros(5))
        if ok:
            R, _ = cv2.Rodrigues(rv)
            objA.append((R @ canon.T).T + tv.T); imgB.append(cB[idxB][0])
    if not objA: return None
    objA = np.vstack(objA).astype(np.float32)
    imgB = np.vstack(imgB).astype(np.float32)

    # --- 🌟 套用全域共面約束 (Global Coplanar Refinement) ---
    if ENFORCE_COPLANAR and global_plane_n is not None and global_plane_c is not None:
        normal = global_plane_n
        d_val = np.dot(normal, global_plane_c)
        
        K_L_inv = np.linalg.inv(K_L.astype(np.float64))
        refined_objA = []
        for mid in shared:
            pts_2d = cA_dict[mid] # 4x2
            rays = np.hstack([pts_2d, np.ones((4, 1))]) @ K_L_inv.T
            t_vals = d_val / (rays @ normal)
            refined_objA.append(rays * t_vals[:, np.newaxis])
        objA = np.vstack(refined_objA).astype(np.float32)

    ok, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5))
    if not ok: return None
    R_rel, _ = cv2.Rodrigues(rv_rel)
    return R_rel, tv_rel, float(np.linalg.norm(tv_rel)), objA, shared, cA_dict, cB_dict

def run_lightglue(lg_session, imgA, imgB):
    t0, t1 = imgA.astype(np.float32)/255.0, imgB.astype(np.float32)/255.0
    inp = np.expand_dims(np.stack([t0, t1], axis=0), axis=1)
    out = lg_session.run(['keypoints', 'matches', 'mscores'], {"images": inp})
    kpts, matches, scores = out
    valid = [(kpts[0, int(m[1])], kpts[1, int(m[2])]) for m, s in zip(matches, scores) if s > LG_SCORE_THRESH]
    if len(valid) < 4: return None, None
    return np.array([v[0] for v in valid], dtype=np.float32), np.array([v[1] for v in valid], dtype=np.float32)

def run_opencv_matching(imgA, imgB):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgA, None)
    kp2, des2 = sift.detectAndCompute(imgB, None)
    if des1 is None or des2 is None: return None, None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m[0] for m in matches if len(m)==2 and m[0].distance < 0.75*m[1].distance]
    if len(good) < 4: return None, None
    return np.float32([kp1[m.queryIdx].pt for m in good]), np.float32([kp2[m.trainIdx].pt for m in good])

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
        if np.min(dists) < 10:
            return float(corners[np.argmin(dists)][0]), float(corners[np.argmin(dists)][1])
    return x, y

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"❌ 無法開啟影片: {VIDEO_PATH}"); sys.exit(1)
    def get_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read(); return f if ret else None
    rawL = get_frame(videoframe_leftImage_index)
    if rawL is None: print("❌ 無法讀取左圖"); sys.exit(1)
    rawR_cands = []
    for i in range(videoframe_total_number):
        f_idx = videoframe_leftImage_index + videoframe_left_to_right_gap + i * videoframe_right_to_right_gap
        frame = get_frame(f_idx)
        if frame is not None: rawR_cands.append({'idx': f_idx, 'sbs': frame})
    cap.release()

    h_raw, w_raw = rawL.shape[:2]
    mtxL_o, distL, mtxR_o, distR, _, _ = load_json_camera_params(PARAMS_JSON_PATH)
    newKL_o, _ = cv2.getOptimalNewCameraMatrix(mtxL_o, distL, (w_raw, h_raw), 1, (w_raw, h_raw))
    newKR_o, _ = cv2.getOptimalNewCameraMatrix(mtxR_o, distR, (w_raw, h_raw), 1, (w_raw, h_raw))

    def process_view(img, K, dist, nK):
        undist = cv2.undistort(img, K, dist, None, nK); s = TARGET_W / undist.shape[1]
        return cv2.resize(undist, (TARGET_W, int(undist.shape[0]*s))), s

    imgA_bgr, scale = process_view(rawL, mtxL_o, distL, newKL_o)
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY); h, w = imgA_gray.shape
    KL = newKL_o.copy().astype(np.float64)
    KL[0,0]*=scale; KL[1,1]*=scale; KL[0,2]*=scale; KL[1,2]*=scale

    lg_session = None
    if FEATURE_MATCHING_MODE == 'LightGlue':
        lg_session = ort.InferenceSession(LG_ONNX_PATH, providers=['CPUExecutionProvider'])

    sift = cv2.SIFT_create(contrastThreshold=0.005)
    
    # 🌟 計算全域參考平面 (以左圖為基準)
    global_plane_n, global_plane_c = compute_global_plane(imgA_gray, KL, ACTUAL_MARKER_SIZE_MM)
    if global_plane_n is not None:
        print("✅ 已成功建立左圖全域 ArUco 參考平面")

    candidates = []
    first_left_saved = False
    for item in rawR_cands:
        imgB_bgr, _ = process_view(item['sbs'], mtxR_o, distR, newKR_o)
        imgB_gray = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY)
        KR = newKR_o.copy().astype(np.float64)
        KR[0,0]*=scale; KR[1,1]*=scale; KR[0,2]*=scale; KR[1,2]*=scale
        
        if SAVE_ARUCO_DEBUG_IMG:
            d_dir = "debug_aruco"
            if not os.path.exists(d_dir): os.makedirs(d_dir)
            if not first_left_saved:
                dbA = imgA_bgr.copy(); cA_r, iA, _ = cv2.aruco.detectMarkers(imgA_gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
                if iA is not None: cv2.aruco.drawDetectedMarkers(dbA, cA_r, iA)
                cv2.imwrite(os.path.join(d_dir, f"debug_L_{videoframe_leftImage_index}.png"), dbA); first_left_saved = True
            dbB = imgB_bgr.copy(); cB_r, iB, _ = cv2.aruco.detectMarkers(imgB_gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
            if iB is not None: cv2.aruco.drawDetectedMarkers(dbB, cB_r, iB)
            cv2.imwrite(os.path.join(d_dir, f"debug_R_{item['idx']}.png"), dbB)

        res = get_joint_relative_pose(imgA_gray, imgB_gray, KL, KR, ACTUAL_MARKER_SIZE_MM, global_plane_n, global_plane_c)
        if res:
            R_r, t_r, base, objA, ids, cA, cB = res
            print(f"\n[幀 {item['idx']} 姿態估計結果]")
            print(f"  - Baseline: {base:.2f} mm")
            print(f"  - Translation (T):\n{t_r}")
            print(f"  - Rotation (R):\n{R_r}")
            kb, db = sift.detectAndCompute(imgB_gray, None)
            ptsA_lg, ptsB_lg = None, None
            if lg_session: ptsA_lg, ptsB_lg = run_lightglue(lg_session, imgA_gray, imgB_gray)
            else: ptsA_lg, ptsB_lg = run_opencv_matching(imgA_gray, imgB_gray)
            cand = {
                'idx': item['idx'], 'rgb': cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB), 'gray': imgB_gray,
                'K_R': KR, 'R_rel': R_r, 't_rel': t_r, 'F': compute_fundamental_matrix(KL, KR, R_r, t_r),
                'cornersA': cA, 'cornersB': cB, 'kpB': kb, 'desB': db, 'ptsA_lg': ptsA_lg, 'ptsB_lg': ptsB_lg,
                'plane_n': global_plane_n, 'plane_c': global_plane_c
            }
            candidates.append(cand)

    if not candidates: print("❌ 無候選幀"); sys.exit(1)
    current_cand = candidates[0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax_A, ax_B = axes
    im_A = ax_A.imshow(cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB))
    im_B = ax_B.imshow(current_cand['rgb'])
    for ax in axes: ax.axis("off")

    def draw_aruco(ax, corners):
        if not hasattr(ax, 'art'): ax.art = []
        for a in ax.art: a.remove()
        ax.art = []
        for mid, pts in corners.items():
            p = np.vstack((pts, pts[0])); l, = ax.plot(p[:,0], p[:,1], 'cyan', lw=1.5)
            t = ax.text(pts[0,0], pts[0,1]-5, f"ID:{mid}", color='cyan', fontsize=8); ax.art.extend([l, t])

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
    depth_text = ax_B.text(0.02, 0.05, "", transform=ax_B.transAxes, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

    ax_opts = fig.add_axes([0.13, 0.78, 0.16, 0.20])
    opt_labels = ["嚴格精細匹配", "梯度 SIFT 匹配", "強制極線對齊", "啟用 ECC 精修", "手動匹配模式"]
    check_opt = CheckButtons(ax_opts, opt_labels, [True, True, True, True, False])
    for text in check_opt.labels: text.set_fontsize(8)
    view_state = {'precise': True, 'grad_sift': True, 'enforce_epi': True, 'ecc': True, 'manual': False,
                  'manual_pt_A': None, 'lines': [], 'grad_lines': [], 'show_grad_lines': False,
                  'highlighted_grad_line': None, 'highlighted_grad_line_artist': None,
                  'grad_data': None}  # grad_data = {'ptsA': ndarray, 'ptsB': ndarray}
    def on_opt(label):
        s = check_opt.get_status()
        view_state.update({'precise': s[0], 'grad_sift': s[1], 'enforce_epi': s[2], 'ecc': s[3], 'manual': s[4]})
        if label == "手動匹配模式" and not s[4]: view_state['manual_pt_A'] = None
    check_opt.on_clicked(on_opt)

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

    ax_btn_grad = fig.add_axes([0.3, 0.02, 0.15, 0.05])
    btn_grad_toggle = Button(ax_btn_grad, '顯示梯度 SIFT 連線')
    def on_grad_toggle(event):
        view_state['show_grad_lines'] = not view_state['show_grad_lines']
        btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線' if view_state['show_grad_lines'] else '顯示梯度 SIFT 連線')
        redraw_grad_lines(view_state.get('highlighted_grad_line'))
        fig.canvas.draw_idle()
    btn_grad_toggle.on_clicked(on_grad_toggle)

    measure_results = {}
    last_click = None

    def do_measure(u, v, manual_match_pt=None):
        nonlocal last_click; last_click = (u, v)
        for l in view_state['lines']: l.remove()
        view_state['lines'] = []
        view_state['grad_data'] = None
        redraw_grad_lines(None)  # 清除舊連線
        sift_rect.set_visible(False)
        all_d, summary = [], []
        for cand in candidates:
            m_pt, method, neighbors = None, "", []
            g_ptsA, g_ptsB, g_kptsB, g_rect = None, None, None, None
            if manual_match_pt is not None and cand['idx'] == current_cand['idx']:
                m_pt, method = manual_match_pt, "手動點選"
            elif manual_match_pt is not None:
                measure_results[cand['idx']] = {'pt': None, 'p3d': None, 'd': None, 'method': "", 'neighbors': []}
                summary.append(f"F{cand['idx']}: N/A"); continue
            if m_pt is None:
                for mid, cA in cand['cornersA'].items():
                    d = np.linalg.norm(cA - np.array([u, v]), axis=1)
                    if np.min(d) < 10: m_pt, method = cand['cornersB'][mid][np.argmin(d)], "ArUco"; break
                if m_pt is None and view_state['grad_sift']:
                    ui, vi = int(round(u)), int(round(v))
                    patch_g = imgA_gray[max(0,vi-8):min(h,vi+9), max(0,ui-8):min(w,ui+9)]
                    if patch_g.size > 0:
                        gx, gy = cv2.Sobel(patch_g, cv2.CV_32F, 1, 0), cv2.Sobel(patch_g, cv2.CV_32F, 0, 1)
                        mag = cv2.sqrt(gx**2 + gy**2); flat = mag.flatten()
                        idx_g = np.argsort(flat)[-min(len(flat), 100):]
                        kpts_inj = [cv2.KeyPoint(float(max(0,ui-8)+px), float(max(0,vi-8)+py), 5.0) for py, px in [divmod(idx, patch_g.shape[1]) for idx in idx_g]]
                        _, des_inj = sift.compute(imgA_gray, kpts_inj)
                        u_exp, v_exp = u, v
                        if cand['plane_n'] is not None:
                            H_AB = cand['K_R'] @ (cand['R_rel'] + (cand['t_rel'] @ cand['plane_n'].reshape(1,3))/np.dot(cand['plane_n'], cand['plane_c'])) @ np.linalg.inv(KL)
                            pt_exp = H_AB @ np.array([u, v, 1.0]); u_exp, v_exp = pt_exp[0]/pt_exp[2], pt_exp[1]/pt_exp[2]
                        rad = 20; uei, vei = int(round(u_exp)), int(round(v_exp))
                        u0, u1, v0, v1 = max(0, uei-rad), min(w, uei+rad), max(0, vei-rad), min(h, vei+rad)
                        patch_r = cand['gray'][v0:v1, u0:u1]
                        if patch_r.size > 0:
                            gxr, gyr = cv2.Sobel(patch_r, cv2.CV_32F, 1, 0), cv2.Sobel(patch_r, cv2.CV_32F, 0, 1)
                            magr = cv2.sqrt(gxr**2 + gyr**2); flatr = magr.flatten()
                            idx_gr = np.argsort(flatr)[-min(len(flatr), 400):]
                            kpts_r = [cv2.KeyPoint(float(u0+px), float(v0+py), 5.0) for py, px in [divmod(idx, patch_r.shape[1]) for idx in idx_gr]]
                            _, des_r = sift.compute(cand['gray'], kpts_r)
                            if des_inj is not None and des_r is not None:
                                bf = cv2.BFMatcher(cv2.NORM_L2); matches = bf.match(des_inj, des_r)
                                good = [m for m in matches if m.distance < 450]
                                pts_info = []
                                for m in good:
                                    pL, pR = np.array(kpts_inj[m.queryIdx].pt), np.array(kpts_r[m.trainIdx].pt)
                                    if np.linalg.norm(pL - np.array([u, v])) < 50: pts_info.append({'pL': pL, 'pR': pR, 'off': pR - pL})
                                if len(pts_info) >= 3:
                                    offs = np.array([x['off'] for x in pts_info]); med_off = np.median(offs, axis=0)
                                    pts_info = [x for x in pts_info if np.linalg.norm(x['off'] - med_off) < 15]
                                if pts_info:
                                    ptsA_m = np.array([x['pL'] for x in pts_info], dtype=np.float32)
                                    ptsB_m = np.array([x['pR'] for x in pts_info], dtype=np.float32)
                                    mapped = None
                                    if len(ptsA_m) >= 6:
                                        H_local, mask_h = cv2.findHomography(ptsA_m, ptsB_m, cv2.RANSAC, 3.0)
                                        if H_local is not None:
                                            pt_h = H_local @ np.array([u, v, 1.0])
                                            mapped = np.array([pt_h[0]/pt_h[2], pt_h[1]/pt_h[2]])
                                    if mapped is None and len(ptsA_m) >= 3:
                                        M_local, mask_a = cv2.estimateAffinePartial2D(ptsA_m, ptsB_m, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                                        if M_local is not None:
                                            pt_a = M_local @ np.array([u, v, 1.0])
                                            mapped = pt_a[:2]
                                    if mapped is None:
                                        # Fallback: weighted median offset (last resort)
                                        wts = 1.0 / (np.sum((ptsA_m - np.array([u, v]))**2, axis=1) + 1e-5)
                                        mapped = np.array([u, v]) + np.sum((ptsB_m - ptsA_m) * wts[:, np.newaxis], axis=0) / np.sum(wts)
                                    m_pt, method = mapped, "Grad-SIFT"
                                    g_ptsA, g_ptsB, g_kptsB, g_rect = ptsA_m, ptsB_m, [kp.pt for kp in kpts_r], (u0, v0, u1-u0, v1-v0)
                if m_pt is None and view_state['precise']:
                    res_p = find_precise_match(imgA_gray, cand['gray'], (u, v), cand['F'], KL, cand['K_R'], cand['R_rel'], cand['t_rel'], cand['plane_n'], cand['plane_c'])
                    if res_p: m_pt, method = np.array(res_p), "Precise"
                if m_pt is None and cand['ptsA_lg'] is not None:
                    dists = np.linalg.norm(cand['ptsA_lg'] - np.array([u, v]), axis=1)
                    curr_r = MATCH_SEARCH_RADIUS
                    in_rad = np.where(dists < curr_r)[0]
                    if len(in_rad) == 0:
                        while len(in_rad) == 0 and curr_r < MATCH_SEARCH_RADIUS_MAX:
                            curr_r += 5; in_rad = np.where(dists < curr_r)[0]
                    if len(in_rad) > 0:
                        off = cand['ptsB_lg'][in_rad] - cand['ptsA_lg'][in_rad]
                        wts = 1.0 / (dists[in_rad] + 1e-5)**2
                        m_pt, method = np.array([u, v]) + np.sum(off * wts[:, np.newaxis], axis=0) / np.sum(wts), FEATURE_MATCHING_MODE
                        neighbors = list(in_rad)
            if m_pt is not None and view_state['enforce_epi']:
                l_B = cand['F'] @ np.array([u, v, 1.0])
                denom = l_B[0]**2 + l_B[1]**2
                if denom > 1e-9:
                    dist_to_epi = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                    m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_to_epi, m_pt[1] - l_B[1]/np.sqrt(denom)*dist_to_epi])
                    method += "+極線對齊"
            if m_pt is not None and view_state['ecc']:
                tmpl = get_patch(imgA_gray, (u, v), 31)
                roi = get_patch(cand['gray'], m_pt, 61)
                if tmpl is not None and roi is not None:
                    warp = np.eye(2, 3, dtype=np.float32)
                    warp[0, 2] = (61 - 31) / 2.0; warp[1, 2] = (61 - 31) / 2.0
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
                    try:
                        _, warp = cv2.findTransformECC(tmpl, roi, warp, cv2.MOTION_TRANSLATION, criteria)
                        m_pt = np.array([m_pt[0] - 30.5 + warp[0, 2] + 15.5, m_pt[1] - 30.5 + warp[1, 2] + 15.5])
                        method += "+ECC精修"
                    except: method += "+ECC失敗"
            d_val, p3d_val = None, None
            if m_pt is not None:
                p3d = triangulate_point_3d((u, v), m_pt, KL, cand['K_R'], cand['R_rel'], cand['t_rel'])
                if p3d is not None: d_val = p3d[2]; p3d_val = p3d
            measure_results[cand['idx']] = {
                'pt': m_pt, 'p3d': p3d_val, 'd': d_val, 'method': method, 'neighbors': neighbors,
                'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_kptsB': g_kptsB, 'g_rect': g_rect
            }
            summary.append(f"F{cand['idx']}: {d_val:.1f}" if d_val is not None else f"F{cand['idx']}: N/A")
        update_display(np.mean(all_d) if all_d else None, summary)

    def update_display(avg, summary):
        u, v = last_click; res = measure_results.get(current_cand['idx'], {'pt': None, 'neighbors': [], 'p3d': None, 'g_ptsA': None})
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
        scatter_grad_inject.set_offsets(np.empty((0,2)))
        scatter_grad_match.set_offsets(np.empty((0,2)))
        scatter_all_sift_B.set_offsets(np.empty((0,2)))
        
        if res.get('g_ptsA') is not None:
            scatter_grad_inject.set_offsets(res['g_ptsA'])
            scatter_grad_match.set_offsets(res['g_ptsB'])
            scatter_all_sift_B.set_offsets(res['g_kptsB'])
            sift_rect.set_bounds(*res['g_rect'])
            sift_rect.set_visible(True)
            view_state['grad_data'] = {'ptsA': res['g_ptsA'], 'ptsB': res['g_ptsB']}
            redraw_grad_lines(None)  # 初始無高亮
                
        if res.get('neighbors') and current_cand['ptsA_lg'] is not None:
            for i in res['neighbors']:
                con = ConnectionPatch(xyA=current_cand['ptsA_lg'][i], xyB=current_cand['ptsB_lg'][i], 
                                      coordsA="data", coordsB="data", axesA=ax_A, axesB=ax_B, color="magenta", lw=1.2, ls="--")
                ax_B.add_artist(con); view_state['lines'].append(con)
        if res['pt'] is not None:
            scatter_B.set_offsets([[res['pt'][0], res['pt'][1]]])
            p0, p1 = epipolar_line(current_cand['F'], (u, v), w); epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            p_dist_str = ""
            if res['p3d'] is not None and current_cand['plane_n'] is not None:
                p_dist = abs(np.dot(current_cand['plane_n'], res['p3d'] - current_cand['plane_c']))
                p_dist_str = f" | 距平面: {p_dist:.1f}mm"
            main_text = f"深度(Z): {res['d']:.1f}mm{p_dist_str}"
        else: scatter_B.set_offsets(np.empty((0,2))); epi_line.set_data([], []); main_text = "無效點"
        depth_text.set_text(main_text + "\n" + " | ".join(summary) + (f"\n平均深度: {avg:.1f}mm" if avg else "\n平均深度: N/A"))
        fig.canvas.draw_idle()

    ax_radio = plt.axes([0.88, 0.4, 0.1, 0.15], facecolor='#222222')
    radio = RadioButtons(ax_radio, [f"F {c['idx']}" for c in candidates])
    for l in radio.labels: l.set_color('white'); l.set_fontsize(9)
    def change_cand(label):
        nonlocal current_cand; idx = int(label.split(' ')[1])
        current_cand = next(c for c in candidates if c['idx'] == idx)
        ax_B.set_title(f"候選幀: {current_cand['idx']}"); im_B.set_data(current_cand['rgb']); draw_aruco(ax_B, current_cand['cornersB'])
        if last_click:
            all_d = [r['d'] for r in measure_results.values() if r['d'] is not None]
            summary = [f"F{c['idx']}: {measure_results[c['idx']]['d']:.1f}" if measure_results[c['idx']]['d'] is not None else f"F{c['idx']}: N/A" for c in candidates]
            update_display(np.mean(all_d) if all_d else None, summary)
        fig.canvas.draw_idle()

    pan_state = {'pressing': False, 'x': None, 'y': None, 'ax': None, 'dragged': False}
    def on_press(event):
        if event.inaxes not in (ax_A, ax_B) or event.button != 1: return
        pan_state.update({'pressing': True, 'dragged': False, 'x': event.x, 'y': event.y, 'ax': event.inaxes})

    def on_release(event):
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
                    fig.canvas.draw_idle()
                elif pan_state['ax'] == ax_B:
                    # 有 grad_data 時，點擊右圖做高亮（不管連線目前是否顯示）
                    if view_state.get('grad_data') and not view_state['manual_pt_A']:
                        ptsB_arr = view_state['grad_data']['ptsB']
                        dists = np.linalg.norm(ptsB_arr - np.array([ux, vx]), axis=1)
                        nearest_idx = int(np.argmin(dists))
                        view_state['show_grad_lines'] = True
                        btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線')
                        redraw_grad_lines(nearest_idx)
                        fig.canvas.draw_idle()
                    elif view_state['manual_pt_A']:
                        do_measure(view_state['manual_pt_A'][0], view_state['manual_pt_A'][1], manual_match_pt=np.array([ux, vx]))
            else:
                if pan_state['ax'] == ax_A:
                    do_measure(ux, vx)
                elif pan_state['ax'] == ax_B and view_state.get('grad_data'):
                    ptsB_arr = view_state['grad_data']['ptsB']
                    dists = np.linalg.norm(ptsB_arr - np.array([ux, vx]), axis=1)
                    nearest_idx = int(np.argmin(dists))
                    view_state['show_grad_lines'] = True
                    btn_grad_toggle.label.set_text('隱藏梯度 SIFT 連線')
                    redraw_grad_lines(nearest_idx)
                    fig.canvas.draw_idle()
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
        if event.inaxes not in (ax_A, ax_B): return
        ax, f = event.inaxes, 1.2 if event.button == 'down' else 1/1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim([x - (x-xl[0])*f, x + (xl[1]-x)*f]); ax.set_ylim([y - (y-yl[0])*f, y + (yl[1]-y)*f])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('button_press_event', on_press)

    radio.on_clicked(change_cand)
    plt.show()

if __name__ == "__main__":
    main()
