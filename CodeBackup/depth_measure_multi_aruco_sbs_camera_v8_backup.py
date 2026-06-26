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
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import onnxruntime as ort

# ==================== 全局設定區 ====================
CAMERA_INDEX          = 0                          # 相機索引
FRAME_GAP             = 30                         # Frame B 與 Frame A 的幀數間隔
UPDATE_INTERVAL       = 0.0                     # 畫面更新與計算間隔 (秒)
CAMERA_WIDTH          = 2560                       # 相機解析度寬
CAMERA_HEIGHT         = 1024                        # 相機解析度高


PARAMS_JSON_PATH      = "calibration_result.json"  # 標定參數 JSON 檔路徑
ACTUAL_MARKER_SIZE_MM = 33#16.5                       # ArUco 標籤真實邊長 (mm)
TARGET_W              = 1024                       # 統一縮放寬度
MAX_DEPTH_MM          = 2000                       # 深度超過此值視為無效 (mm)
ENFORCE_COPLANAR      = True                       # 強制共面對齊優化
SAVE_ARUCO_DEBUG_IMG  = False                      # 是否存出 ArUco 偵測結果圖片


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
        ok, rv, tv = cv2.solvePnP(canon, c[0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
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
        ok, rv, tv = cv2.solvePnP(canon, cA[idxA][0], K_L, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
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

    ok, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K_R, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    if not ok: return None
    R_rel, _ = cv2.Rodrigues(rv_rel)
    return R_rel, tv_rel, float(np.linalg.norm(tv_rel)), objA, shared, cA_dict, cB_dict





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
    import collections, time
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened(): print(f"❌ 無法開啟相機"); sys.exit(1)
    
    # 讀取第一幀獲取尺寸
    ret, first_frame = cap.read()
    if not ret: print("❌ 無法讀取相機畫面"); sys.exit(1)
    
    h_raw, w_raw = first_frame.shape[:2]
    w_alg = w_raw // 2 # 使用畫面左半邊
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
    imgA_gray = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY); h, w = imgA_gray.shape
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
    sift_rect_center, = ax_B.plot([], [], '+', color='magenta', markersize=12, markeredgewidth=1.5, zorder=5)
    sift_rect_center.set_visible(False)
    depth_text = ax_B.text(0.02, 0.05, "", transform=ax_B.transAxes, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
    fps_text = ax_A.text(0.01, 0.97, "FPS: --", transform=ax_A.transAxes,
                         color='lime', fontsize=10, fontweight='bold', va='top',
                         bbox=dict(facecolor='black', alpha=0.6), zorder=10)
    # Blit 最佳化：標記每幀會改變的 artists 為 animated，防止它們被無謂嫚入靜態背景圖
    im_A.set_animated(True)
    im_B.set_animated(True)
    fps_text.set_animated(True)
    # blit_state: 管理背景圖狀態
    blit_state = {'bg': None, 'needs_refresh': True}

    def request_blit_refresh():
        """UI 元件有治變時呼叫，主迴圈下一幀會重新全圖儲存新背景。"""
        blit_state['needs_refresh'] = True


    ax_opts = fig.add_axes([0.13, 0.74, 0.16, 0.24])
    opt_labels = ["嚴格精細匹配", "梯度 SIFT 匹配", "強制極線對齊", "啟用 ECC 精修", "手動匹配模式", "Farneback匹配"]
    check_opt = CheckButtons(ax_opts, opt_labels, [True, True, True, True, False, False])
    for text in check_opt.labels: text.set_fontsize(8)
    view_state = {'precise': True, 'grad_sift': True, 'enforce_epi': True, 'ecc': True, 'manual': False, 'use_farneback': False,
                  'use_hamming': False,
                  'manual_pt_A': None, 'lines': [], 'grad_lines': [], 'show_grad_lines': False,
                  'highlighted_grad_line': None, 'highlighted_grad_line_artist': None,
                  'grad_data': None}  # grad_data = {'ptsA': ndarray, 'ptsB': ndarray}
    def on_opt(label):
        s = check_opt.get_status()
        view_state.update({'precise': s[0], 'grad_sift': s[1], 'enforce_epi': s[2], 'ecc': s[3], 'manual': s[4], 'use_farneback': s[5]})
        if label == "手動匹配模式" and not s[4]: view_state['manual_pt_A'] = None
        request_blit_refresh()
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
        request_blit_refresh()
    btn_grad_toggle.on_clicked(on_grad_toggle)

    ax_btn_norm = fig.add_axes([0.02, 0.92, 0.10, 0.04]) # 放左上角，避免重疊
    btn_norm_toggle = Button(ax_btn_norm, '切換 HAMMING')
    def on_norm_toggle(event):
        view_state['use_hamming'] = not view_state['use_hamming']
        btn_norm_toggle.label.set_text('使用 L2' if view_state['use_hamming'] else '切換 HAMMING')
        request_blit_refresh()
    btn_norm_toggle.on_clicked(on_norm_toggle)

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
        if cand.get('baseline', 0.0) < 15.0:
            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                    'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': None, 'g_rect': None,
                    'fail_reason': '視差不足(<15mm)', 'u': u, 'v': v}

        if manual_match_pt is not None:
            m_pt, method = manual_match_pt, "手動點選"

        if m_pt is None:
            for mid, cA in cand['cornersA'].items():
                d = np.linalg.norm(cA - np.array([u, v]), axis=1)
                if np.min(d) < 10:
                    m_pt, method = cand['cornersB'][mid][np.argmin(d)], "ArUco"
                    break
            u_exp, v_exp = None, None
            if m_pt is None and (snap_view_state.get('grad_sift') or snap_view_state.get('use_farneback', False)):
                h_raw, w_alg = snap_imgA_gray.shape
                # 1. 執行 Stereo Rectify
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(KL, np.zeros(5), cand['K_R'], np.zeros(5), (w_alg, h_raw), cand['R_rel'], cand['t_rel'], flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
                map1_L, map2_L = cv2.initUndistortRectifyMap(KL, np.zeros(5), R1, P1, (w_alg, h_raw), cv2.CV_16SC2)
                map1_R, map2_R = cv2.initUndistortRectifyMap(cand['K_R'], np.zeros(5), R2, P2, (w_alg, h_raw), cv2.CV_16SC2)
                
                rectA = cv2.remap(snap_imgA_gray, map1_L, map2_L, cv2.INTER_LINEAR)
                rectB = cv2.remap(cand['gray'], map1_R, map2_R, cv2.INTER_LINEAR)
                
                # 2. 將點擊座標轉至 rectA
                pts_u = np.array([[[u, v]]], dtype=np.float32)
                pts_rect_L = cv2.undistortPoints(pts_u, KL, np.zeros(5), R=R1, P=P1)
                u_rect, v_rect = pts_rect_L[0][0]
                
                # 預測右圖中心 uB_exp_rect
                u_exp, v_exp = u, v
                if cand['plane_n'] is not None:
                    H_AB = cand['K_R'] @ (cand['R_rel'] + (cand['t_rel'] @ cand['plane_n'].reshape(1,3))/np.dot(cand['plane_n'], cand['plane_c'])) @ np.linalg.inv(KL)
                    pt_exp = H_AB @ np.array([u, v, 1.0])
                    u_exp, v_exp = pt_exp[0]/pt_exp[2], pt_exp[1]/pt_exp[2]
                
                pts_exp = np.array([[[u_exp, v_exp]]], dtype=np.float32)
                pts_rect_R = cv2.undistortPoints(pts_exp, cand['K_R'], np.zeros(5), R=R2, P=P2)
                uB_exp_rect, _ = pts_rect_R[0][0]

                if snap_view_state.get('grad_sift'):
                    # 3. 提取左圖 patch_g (在 rectA)
                    rad_L = 11
                    ui_rect, vi_rect = int(round(u_rect)), int(round(v_rect))
                    patch_g = rectA[max(0, vi_rect-rad_L):min(h_raw, vi_rect+rad_L+1),
                                    max(0, ui_rect-rad_L):min(w_alg, ui_rect+rad_L+1)]
                    
                    if patch_g.size > 0:
                        corners = cv2.goodFeaturesToTrack(patch_g, maxCorners=100, qualityLevel=0.001, minDistance=2, useHarrisDetector=False)
                        if corners is None or len(corners) < 1:
                            return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                                    'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': None, 'g_rect': None,
                                    'fail_reason': '左圖特徵不足(太平滑或無角點)', 'u': u, 'v': v}
                        kpts_inj = [cv2.KeyPoint(float(max(0,ui_rect-rad_L)+c[0][0]), float(max(0,vi_rect-rad_L)+c[0][1]), 31.0) for c in corners]
                        if snap_view_state.get('use_hamming', False):
                            _, des_inj = orb.compute(rectA, kpts_inj)
                        else:
                            _, des_inj = sift.compute(rectA, kpts_inj)
                        
                        # 5. 在 rectB 取扁平廊道提取 patch_r
                        rad_R_x = 40
                        rad_R_y = 6 # 容許少許垂直誤差
                        u0_R, u1_R = max(0, int(round(uB_exp_rect)) - rad_R_x), min(w_alg, int(round(uB_exp_rect)) + rad_R_x)
                        v0_R, v1_R = max(0, int(round(v_rect)) - rad_R_y), min(h_raw, int(round(v_rect)) + rad_R_y)
                        patch_r = rectB[v0_R:v1_R, u0_R:u1_R]
                        
                        if patch_r.size > 0:
                            corners_r = cv2.goodFeaturesToTrack(patch_r, maxCorners=400, qualityLevel=0.001, minDistance=2, useHarrisDetector=False)
                            if corners_r is None or len(corners_r) < 1:
                                return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                                        'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': [], 'g_rect': None,
                                        'fail_reason': '右圖特徵不足(太平滑或無角點)', 'u': u, 'v': v}
                            kpts_r = [cv2.KeyPoint(float(u0_R+c[0][0]), float(v0_R+c[0][1]), 31.0) for c in corners_r]
                            
                            good = []
                            if snap_view_state.get('use_hamming', False):
                                _, des_r = orb.compute(rectB, kpts_r)
                                if des_inj is not None and des_r is not None and len(des_r) >= 2:
                                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                                    matches = bf.knnMatch(des_inj, des_r, k=2)
                                    for m_n in matches:
                                        if len(m_n) == 2 and m_n[0].distance < 0.8 * m_n[1].distance:
                                            good.append(m_n[0])
                            else:
                                _, des_r = sift.compute(rectB, kpts_r)
                                if des_inj is not None and des_r is not None and len(des_r) >= 2:
                                    bf = cv2.BFMatcher(cv2.NORM_L2)
                                    matches = bf.knnMatch(des_inj, des_r, k=2)
                                    for m_n in matches:
                                        if len(m_n) == 2 and m_n[0].distance < 0.9 * m_n[1].distance:
                                            good.append(m_n[0])
                            
                            print(f"--- 診斷 LOG ---")
                            print(f"左圖實際提取的特徵點數量: {len(kpts_inj)}")
                            print(f"右圖實際提取的特徵點數量: {len(kpts_r)}")
                            print(f"Ratio Test 過濾後剩餘的匹配點數量: {len(good)}")
                            print(f"----------------")
                            
                            pts_info = []
                            for m in good:
                                pL, pR = np.array(kpts_inj[m.queryIdx].pt), np.array(kpts_r[m.trainIdx].pt)
                                if np.linalg.norm(pL - np.array([u_rect, v_rect])) < 50:
                                    pts_info.append({'pL': pL, 'pR': pR, 'off': pR - pL})
                            
                            if len(pts_info) >= 3:
                                offs = np.array([x['off'] for x in pts_info]); med_off = np.median(offs, axis=0)
                                pts_info = [x for x in pts_info if np.linalg.norm(x['off'] - med_off) < 15]
                            
                            if pts_info:
                                ptsA_m = np.array([x['pL'] for x in pts_info], dtype=np.float32)
                                ptsB_m = np.array([x['pR'] for x in pts_info], dtype=np.float32)
                                wts = 1.0 / (np.sum((ptsA_m - np.array([u_rect, v_rect]))**2, axis=1) + 1e-5)
                                mapped_rectB = np.array([u_rect, v_rect]) + np.sum((ptsB_m - ptsA_m) * wts[:, np.newaxis], axis=0) / np.sum(wts)
                                mapped_rectB[1] = v_rect # 強制完美極線對齊
                                
                                # Triangulate in Rectified Space -> 3D point in Camera 1 unrectified space!
                                pts4D = cv2.triangulatePoints(P1, P2, np.array([[u_rect], [v_rect]], dtype=np.float32), np.array([[mapped_rectB[0]], [mapped_rectB[1]]], dtype=np.float32))
                                pts3D = (pts4D[:3] / pts4D[3]).flatten()
                                
                                # 投影回原始右圖 Unrectified 取得精確 m_pt (供 UI 顯示)
                                pts3D_unrect1 = R1.T @ pts3D
                                pt3D_cam2 = cand['R_rel'] @ pts3D_unrect1 + cand['t_rel'].flatten()
                                pt_unrectB = cand['K_R'] @ pt3D_cam2
                                m_pt = pt_unrectB[:2] / pt_unrectB[2]
                                method = "Stereo-Rectify-1D"
                                
                                # 輔助將 rect 上的點轉換回 unrect 供 UI 顯示
                                def unrectify_pts(pts, P_rect, R_un2rect, K_orig):
                                    if len(pts) == 0: return np.empty((0,2))
                                    pts_homo = np.hstack([pts, np.ones((len(pts), 1))]).astype(np.float32)
                                    rays_rect = pts_homo @ np.linalg.inv(P_rect[:3, :3]).T
                                    rays_unrect = rays_rect @ R_un2rect
                                    pts_unrect_homo = rays_unrect @ K_orig.T
                                    return pts_unrect_homo[:, :2] / pts_unrect_homo[:, 2:]
                                    
                                g_ptsA = unrectify_pts(ptsA_m, P1, R1, KL)
                                g_ptsB = unrectify_pts(ptsB_m, P2, R2, cand['K_R'])
                                g_kptsB = [tuple(pt) for pt in unrectify_pts(np.array([kp.pt for kp in kpts_r]), P2, R2, cand['K_R'])]
                                g_rect = None
                elif snap_view_state.get('use_farneback', False):
                    rad = 60
                    ui_rect, vi_rect = int(round(u_rect)), int(round(v_rect))
                    uB_exp_i, vB_exp_i = int(round(uB_exp_rect)), int(round(v_rect))
                    
                    if (vi_rect - rad >= 0 and vi_rect + rad < h_raw and 
                        ui_rect - rad >= 0 and ui_rect + rad < w_alg and
                        vB_exp_i - rad >= 0 and vB_exp_i + rad < h_raw and 
                        uB_exp_i - rad >= 0 and uB_exp_i + rad < w_alg):
                        
                        roi_l = rectA[vi_rect-rad:vi_rect+rad, ui_rect-rad:ui_rect+rad]
                        roi_r = rectB[vB_exp_i-rad:vB_exp_i+rad, uB_exp_i-rad:uB_exp_i+rad]
                        
                        flow = cv2.calcOpticalFlowFarneback(
                            roi_l, roi_r, None,
                            pyr_scale=0.5, levels=5, winsize=31,
                            iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
                        
                        # 取中心 5x5 區域的平均流速，避免單一像素雜訊影響
                        center_flow = flow[rad-2:rad+3, rad-2:rad+3]
                        dx, dy = np.mean(center_flow, axis=(0, 1))
                        
                        mapped_rectB = np.array([uB_exp_i + dx, vB_exp_i + dy])
                        mapped_rectB[1] = v_rect # 強制完美極線對齊
                        
                        pts4D = cv2.triangulatePoints(P1, P2, np.array([[u_rect], [v_rect]], dtype=np.float32), np.array([[mapped_rectB[0]], [mapped_rectB[1]]], dtype=np.float32))
                        pts3D = (pts4D[:3] / pts4D[3]).flatten()
                        
                        pts3D_unrect1 = R1.T @ pts3D
                        pt3D_cam2 = cand['R_rel'] @ pts3D_unrect1 + cand['t_rel'].flatten()
                        pt_unrectB = cand['K_R'] @ pt3D_cam2
                        m_pt = pt_unrectB[:2] / pt_unrectB[2]
                        method = "Farneback-Dense"
                        
                        g_ptsA = None
                        g_ptsB = None
                        g_kptsB = []
                        g_rect = (uB_exp_i - rad, vB_exp_i - rad, 2*rad, 2*rad)
                    else:
                        return {'pt': None, 'p3d': None, 'd': None, 'method': '', 'neighbors': [],
                                'g_ptsA': None, 'g_ptsB': None, 'g_kptsB': [], 'g_rect': None,
                                'fail_reason': 'ROI 超出影像範圍', 'u': u, 'v': v}
            if m_pt is None and snap_view_state['precise']:
                res_p = find_precise_match(snap_imgA_gray, cand['gray'], (u, v), cand['F'],
                                           KL, cand['K_R'], cand['R_rel'], cand['t_rel'],
                                           cand['plane_n'], cand['plane_c'])
                if res_p: m_pt, method = np.array(res_p), "Precise"

        if m_pt is not None and snap_view_state['enforce_epi']:
            l_B = cand['F'] @ np.array([u, v, 1.0])
            denom = l_B[0]**2 + l_B[1]**2
            if denom > 1e-9:
                dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                  m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])
                method += "+極線對齊"
        if m_pt is not None and snap_view_state['ecc']:
            tmpl = get_patch(snap_imgA_gray, (u, v), 31)
            roi = get_patch(cand['gray'], m_pt, 61)
            if tmpl is not None and roi is not None:
                warp = np.eye(2, 3, dtype=np.float32)
                warp[0, 2] = (61 - 31) / 2.0; warp[1, 2] = (61 - 31) / 2.0
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
                try:
                    _, warp = cv2.findTransformECC(tmpl, roi, warp, cv2.MOTION_TRANSLATION, criteria)
                    m_pt = np.array([m_pt[0] - 30.5 + warp[0, 2] + 15.5,
                                     m_pt[1] - 30.5 + warp[1, 2] + 15.5])
                    method += "+ECC精修"
                except: method += "+ECC失敗"
        if m_pt is not None and snap_view_state['enforce_epi']:
            l_B = cand['F'] @ np.array([u, v, 1.0])
            denom = l_B[0]**2 + l_B[1]**2
            if denom > 1e-9:
                dist_e = (l_B[0]*m_pt[0] + l_B[1]*m_pt[1] + l_B[2]) / np.sqrt(denom)
                m_pt = np.array([m_pt[0] - l_B[0]/np.sqrt(denom)*dist_e,
                                  m_pt[1] - l_B[1]/np.sqrt(denom)*dist_e])

        d_val, p3d_val, fail_reason = None, None, ""
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
                d_val = p3d[2]; p3d_val = p3d
                p_dist_str = "N/A"
                if cand['plane_n'] is not None:
                    p_dist = abs(np.dot(cand['plane_n'], p3d - cand['plane_c']))
                    p_dist_str = f"{p_dist:.2f} mm"
                print(f"   [計算結果] 深度(Z): {d_val:.2f} mm | 距平面深度: {p_dist_str}")
        else:
            fail_reason = "無匹配點"
            print(f"👉 [深度計算] 左圖座標: ({u:.1f}, {v:.1f}) | 右圖匹配座標: N/A | 匹配方式: N/A")
            print(f"   [計算失敗] 原因: {fail_reason}")

        return {'pt': m_pt, 'p3d': p3d_val, 'd': d_val, 'method': method, 'neighbors': neighbors,
                'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_kptsB': g_kptsB, 'g_rect': g_rect,
                'fail_reason': fail_reason, 'u': u, 'v': v,
                'pred_pt': (u_exp, v_exp) if u_exp is not None else None}

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
        else:
            left_img_gray = imgA_gray
            
        res = compute_measure(u, v, current_cand, left_img_gray, snap_vs, manual_match_pt)
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
        
        if res.get('pred_pt') is not None:
            px, py = res['pred_pt']
            sift_rect_center.set_data([px], [py])
            sift_rect_center.set_visible(True)
            # 畫出預估範圍 (固定大小 120x120 作為示意)
            sift_rect.set_bounds(px - 60, py - 60, 120, 120)
            sift_rect.set_visible(True)
            
        if res.get('g_kptsB') is not None and len(res['g_kptsB']) > 0:
            try:
                scatter_all_sift_B.set_offsets(res['g_kptsB'])
            except Exception:
                # 預防某些版本 Matplotlib 對於空陣列轉 numpy 仍拋出維度錯誤
                pass
            
            if res.get('g_rect') is not None:
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
                    p_dist_str = f" | 距平面(15幀均): {np.mean(plane_dist_history):.1f}mm"
                else:
                    if len(plane_dist_history) > 0:
                        plane_dist_history.clear()
                    p_dist_str = f" | 距平面: {p_dist:.1f}mm"
            if res['d'] is not None:
                main_text = f"深度(Z): {res['d']:.1f}mm{p_dist_str}\n外參來源: {pose_info_str}"
            else:
                main_text = f"深度(Z): 無效({res.get('fail_reason', '原因未知')}){p_dist_str}\n外參來源: {pose_info_str}"
        else: scatter_B.set_offsets(np.empty((0,2))); epi_line.set_data([], []); main_text = f"無效點({res.get('fail_reason', '無匹配點')})\n外參來源: {pose_info_str}"
        depth_text.set_text(main_text + "\n" + " | ".join(summary) + (f"\n平均深度: {avg:.1f}mm" if avg else "\n平均深度: N/A"))
        request_blit_refresh()

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
        request_blit_refresh()

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
    
    current_cand = {
        'idx': 0,
        'K_R': KL,
        'pose_valid': False,
        'baseline': 0.0,
        'cornersA': {},
        'cornersB': {}
    }
    
    # 建立按鈕
    ax_btn_lock_L = fig.add_axes([0.25, 0.92, 0.15, 0.04])
    btn_lock_L = Button(ax_btn_lock_L, "鎖定左圖")
    
    ax_btn_calc = fig.add_axes([0.35, 0.92, 0.12, 0.04])
    btn_calc = Button(ax_btn_calc, "單次計算深度")
    
    ax_btn_auto_calc = fig.add_axes([0.52, 0.92, 0.15, 0.04])
    btn_auto_calc = Button(ax_btn_auto_calc, "連續計算: 關")
    
    ax_btn_lock_R = fig.add_axes([0.72, 0.92, 0.15, 0.04])
    btn_lock_R = Button(ax_btn_lock_R, "鎖定右圖")
    
    auto_calc_active = False
    
    def on_auto_calc(event):
        nonlocal auto_calc_active
        auto_calc_active = not auto_calc_active
        if auto_calc_active:
            btn_auto_calc.label.set_text("連續計算: 開")
            print("▶️ 開啟連續計算模式")
        else:
            btn_auto_calc.label.set_text("連續計算: 關")
            print("⏸️ 關閉連續計算模式")
        request_blit_refresh()
    
    def on_lock_L(event):
        nonlocal live_L, locked_L, has_set_L
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
        request_blit_refresh()
        
    def on_lock_R(event):
        nonlocal live_R, locked_R, has_set_R
        live_R = not live_R
        if not live_R:
            if len(frame_buffer) > 0:
                locked_R = frame_buffer[-1].copy()
                has_set_R = True
                btn_lock_R.label.set_text("解鎖右圖")
                print("🔒 右圖已鎖定當前畫面")
                if has_set_L and not live_L:
                    on_calc(None)
        else:
            btn_lock_R.label.set_text("鎖定右圖")
            print("🔓 右圖恢復 Live")
        request_blit_refresh()
        
    def on_calc(event):
        if not has_set_L:
            print("⚠️ 請先設定左圖！")
            return
        # 快照計算所需數据，鴮入 request queue（不阻塞主執行緒）
        imgA_bgr_snap, _ = process_view(locked_L, mtxL_o, distL, newKL_o)
        imgA_gray_snap   = cv2.cvtColor(imgA_bgr_snap, cv2.COLOR_BGR2GRAY)
        current_R_snap   = frame_buffer[-1].copy() if live_R and len(frame_buffer) > 0 else locked_R
        imgB_bgr_snap, _ = process_view(current_R_snap, mtxL_o, distL, newKL_o)
        req = {
            'imgA_bgr':   imgA_bgr_snap,
            'imgA_gray':  imgA_gray_snap,
            'imgB_bgr':   imgB_bgr_snap,
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
        while True:
            req = calc_request_q.get()
            if req is None:   # 結束訊號
                break
            calc_busy.set()
            try:
                snap_imgA_gray = req['imgA_gray']
                snap_cand      = req['cand']
                snap_vs        = req['view_state']
                u, v           = req['u'], req['v']

                # ---------- ArUco 位姿解算 ----------
                imgA_bgr_ = req['imgA_bgr']
                imgB_bgr_ = req['imgB_bgr']
                imgB_gray_ = cv2.cvtColor(imgB_bgr_, cv2.COLOR_BGR2GRAY)

                res_pose = get_joint_relative_pose(snap_imgA_gray, imgB_gray_, KL, KL, ACTUAL_MARKER_SIZE_MM)
                baseline_val = 0.0
                pose_valid = False
                pose_info = "未偵測到 ArUco"
                R_r, t_r = None, None
                cA_dict, cB_dict = {}, {}

                if res_pose is not None:
                    R_r, t_r, baseline_val, _, _, cA_dict, cB_dict = res_pose
                    pose_valid = True
                    direction = "向左" if t_r[0] > 0 else "向右"
                    pose_info = f"ArUco 定位(Bsl: {baseline_val:.1f}mm, {direction})"
                    print(f"  - 計算出的時序 Baseline: {baseline_val:.2f} mm")
                else:
                    print("❌ 未偵測到共享 ArUco")

                n_local, c_local = compute_global_plane(snap_imgA_gray, KL, ACTUAL_MARKER_SIZE_MM)
                if n_local is None:
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
                    'cornersA': cA_dict, 'cornersB': cB_dict
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

        # 如果處於連續計算模式，自動觸發（非阻塞）
        if auto_calc_active and has_set_L and calc_request_q.empty() and not calc_busy.is_set():
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
        _t5 = _time.perf_counter(); _t_proc += (_t5 - _t4) * 1000
        
        im_A.set_data(cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2RGB))
        im_B.set_data(cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2RGB))
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
        ax_B.draw_artist(im_B)
        
        # === 手動繪製覆蓋層（解決點、線被影像遮擋的問題） ===
        # 1. 散點與基礎幾何
        ax_A.draw_artist(scatter_A)
        ax_B.draw_artist(scatter_B)
        ax_A.draw_artist(scatter_grad_inject)
        ax_B.draw_artist(scatter_grad_match)
        ax_B.draw_artist(scatter_all_sift_B)
        ax_B.draw_artist(epi_line)
        ax_B.draw_artist(sift_rect)
        ax_B.draw_artist(sift_rect_center)
        # 繪製深度資訊文字，避免被影像遮擋
        ax_B.draw_artist(depth_text)

        # 2. 梯度 SIFT 連線
        for line in view_state.get('grad_lines', []):
            ax_B.draw_artist(line)
        if view_state.get('highlighted_grad_line_artist'):
            ax_B.draw_artist(view_state['highlighted_grad_line_artist'])

        # 3. ArUco 標記 (儲存在 ax.art 中)
        for ax in [ax_A, ax_B]:
            if hasattr(ax, 'art'):
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
