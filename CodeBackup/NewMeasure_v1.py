import os
import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button

# ==================== 系統與參數設定 ====================
CAMERA_INDEX = 0
PARAMS_JSON_PATH = "calibration_result.json"
ACTUAL_MARKER_SIZE_MM = 33.0
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1024
# 統一將影像裁切/調整為左右兩半，或者按照原始設定：如果相機送出 2560x1024，一半是 1280x1024。
# 這裡根據使用者舊有專案習慣，取一半寬度。
HALF_WIDTH = 1280

def load_json_camera_params(json_path):
    """讀取相機內參與畸變參數"""
    if not os.path.exists(json_path):
        print(f"❌ 找不到參數檔案: {json_path}")
        return None, None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mtx_L = np.array(data['intrinsic_L']['matrix'], dtype=np.float32)
    dist_L = np.array(data['intrinsic_L']['distortion'], dtype=np.float32)
    return mtx_L, dist_L

def estimate_relative_pose(imgA_gray, imgB_gray, K, dist, marker_size_mm):
    """
    第二階段：利用 ArUco 估計相機 A 與相機 B 之間的相對位姿 R_rel, T_rel
    並且取得 ArUco 平面在 Camera A 下的法向量與中心點。
    """
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cA, idsA, _ = detector.detectMarkers(imgA_gray)
        cB, idsB, _ = detector.detectMarkers(imgB_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
        cB, idsB, _ = cv2.aruco.detectMarkers(imgB_gray, dict_4x4, parameters=params)

    if idsA is None or idsB is None:
        return False, "無法在兩張圖中都偵測到 ArUco", None

    idsA_l = [i[0] for i in idsA]
    idsB_l = [i[0] for i in idsB]
    shared = list(set(idsA_l).intersection(set(idsB_l)))
    if not shared:
        return False, "兩張圖中沒有共同的 ArUco ID", None

    # 機制 4: 放大亞像素精細化的搜尋半徑與迭代次數
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for c in cA: cv2.cornerSubPix(imgA_gray, c, (11, 11), (-1, -1), term)
    for c in cB: cv2.cornerSubPix(imgB_gray, c, (11, 11), (-1, -1), term)

    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    
    objA, imgB_pts = [], []
    rvec1_first, tvec1_first = None, None
    
    # 計算各 Marker 的 PnP
    for mid in shared:
        idxA = idsA_l.index(mid)
        idxB = idsB_l.index(mid)
        okA, rvA, tvA = cv2.solvePnP(canon, cA[idxA][0], K, dist)
        if okA:
            if rvec1_first is None:
                rvec1_first = rvA.copy()
                tvec1_first = tvA.copy()
            RA, _ = cv2.Rodrigues(rvA)
            # 將 3D Pattern 點轉為 Camera A 座標系
            pts3d_camA = (RA @ canon.T).T + tvA.T
            objA.append(pts3d_camA)
            imgB_pts.append(cB[idxB][0])

    if not objA:
        return False, "solvePnP 失敗", None

    objA = np.vstack(objA).astype(np.float32)
    imgB_pts = np.vstack(imgB_pts).astype(np.float32)

    # 以 Camera A 中的 3D 點去對應 Camera B 中的 2D 點，解出相對位姿
    ok, rv_rel, tv_rel = cv2.solvePnP(objA, imgB_pts, K, dist)
    if not ok:
        return False, "相對位姿估計失敗", None
        
    # 機制 3: 針對所有標籤的多角點，進行 LM 聯合優化約束
    cv2.solvePnPRefineLM(objA, imgB_pts, K, dist, rv_rel, tv_rel)

    R_rel, _ = cv2.Rodrigues(rv_rel)
    baseline = np.linalg.norm(tv_rel)

    # 取出 ArUco 平面參數 (Camera A)
    R1, _ = cv2.Rodrigues(rvec1_first)
    plane_normal = R1[:, 2] # Z 軸方向
    plane_center = tvec1_first.flatten()

    pose_data = {
        'R_rel': R_rel,
        'T_rel': tv_rel,
        'rv_rel': rv_rel,
        'tv_rel': tv_rel,
        'baseline': baseline,
        'plane_normal': plane_normal,
        'plane_center': plane_center,
        'cA': cA,
        'cB': cB
    }
    return True, "成功", pose_data

def stereo_rectify_and_remap(imgA, imgB, K, dist, R_rel, T_rel):
    """
    第三階段：立體校正 (Stereo Rectification)
    """
    h, w = imgA.shape[:2]
    # cv2.stereoRectify
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R_rel, T_rel, 
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )
    
    # init maps
    map1_A, map2_A = cv2.initUndistortRectifyMap(K, dist, R1_rect, P1, (w, h), cv2.CV_16SC2)
    map1_B, map2_B = cv2.initUndistortRectifyMap(K, dist, R2_rect, P2, (w, h), cv2.CV_16SC2)
    
    imgA_rect = cv2.remap(imgA, map1_A, map2_A, cv2.INTER_LINEAR)
    imgB_rect = cv2.remap(imgB, map1_B, map2_B, cv2.INTER_LINEAR)
    
    rect_data = {
        'R1_rect': R1_rect, 'R2_rect': R2_rect,
        'P1': P1, 'P2': P2, 'Q': Q,
        'map1_A': map1_A, 'map2_A': map2_A,
        'map1_B': map1_B, 'map2_B': map2_B,
    }
    return imgA_rect, imgB_rect, rect_data

def find_match_1d(imgA_rect_gray, imgB_rect_gray, x1, y1, patch_size=21):
    """
    第五/六階段：多尺度 Patch 匹配與信心度驗證
    沿著水平極線 (y=y1) 搜尋對應點
    """
    h, w = imgA_rect_gray.shape
    half = patch_size // 2
    x1_i, y1_i = int(round(x1)), int(round(y1))
    
    if x1_i - half < 0 or x1_i + half >= w or y1_i - half < 0 or y1_i + half >= h:
        return None, 0, "查詢點過於靠近邊緣", 0
        
    patchA = imgA_rect_gray[y1_i-half:y1_i+half+1, x1_i-half:x1_i+half+1]
    
    # 第六階段：信心度驗證 - 局部梯度強度
    gx = cv2.Sobel(patchA, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patchA, cv2.CV_32F, 0, 1)
    mag = cv2.sqrt(gx**2 + gy**2)
    mean_grad = np.mean(mag)
    if mean_grad < 10.0:
        return None, 0, f"紋理不足(梯度={mean_grad:.1f})", mean_grad
        
    roi_y0 = y1_i - half
    roi_y1 = y1_i + half + 1
    # 取 B 圖對應的整條水平帶狀區域
    roiB = imgB_rect_gray[roi_y0:roi_y1, 0:w]
    
    if roiB.shape[0] < patchA.shape[0]:
        return None, 0, "ROI B size 異常", mean_grad

    # 1. ZNCC 正向搜尋 (A -> B)
    res_fwd = cv2.matchTemplate(roiB, patchA, cv2.TM_CCOEFF_NORMED)
    res_1d = res_fwd[0]
    
    max_idx_fwd = np.argmax(res_1d)
    max_score_fwd = res_1d[max_idx_fwd]
    
    # 採用超低門檻 (放寬到 0.3)
    if max_score_fwd < 0.3:
        return None, max_score_fwd, "ZNCC 分數過低 (<0.3)", mean_grad
        
    x2_int = max_idx_fwd + half
    
    # 2. ZNCC 反向搜尋 (B -> A) - 左右一致性檢測 (L-R Check)
    if x2_int - half < 0 or x2_int + half >= w:
        return None, max_score_fwd, "右圖匹配點過近邊緣", mean_grad
        
    patchB = imgB_rect_gray[roi_y0:roi_y1, x2_int-half:x2_int+half+1]
    roiA = imgA_rect_gray[roi_y0:roi_y1, 0:w]
    
    res_bwd = cv2.matchTemplate(roiA, patchB, cv2.TM_CCOEFF_NORMED)
    max_idx_bwd = np.argmax(res_bwd[0])
    x1_bwd_int = max_idx_bwd + half
    
    # 一致性門檻設為 2 pixel
    if abs(x1_bwd_int - x1_i) > 2:
        return None, max_score_fwd, f"L-R Check 失敗 (差 {abs(x1_bwd_int - x1_i)} px)", mean_grad

    def parabolic_subpixel(res_arr, peak_idx):
        if 0 < peak_idx < len(res_arr) - 1:
            s_prev = res_arr[peak_idx - 1]
            s_curr = res_arr[peak_idx]
            s_next = res_arr[peak_idx + 1]
            denom = (s_prev - 2*s_curr + s_next)
            if denom != 0:
                return peak_idx - (s_next - s_prev) / (2 * denom)
        return peak_idx

    # 3. 亞像素精化 (優先使用 ECC 處理視角形變，若失敗則退回拋物線)
    ecc_patch_size = patch_size + 10  # 稍微擴大區塊給 ECC 更多上下文
    ecc_half = ecc_patch_size // 2
    
    if (x1_i - ecc_half >= 0 and x1_i + ecc_half < w and y1_i - ecc_half >= 0 and y1_i + ecc_half < h and
        x2_int - ecc_half >= 0 and x2_int + ecc_half < w):
        
        tmpl = imgA_rect_gray[y1_i-ecc_half:y1_i+ecc_half+1, x1_i-ecc_half:x1_i+ecc_half+1]
        roi_ecc = imgB_rect_gray[y1_i-ecc_half:y1_i+ecc_half+1, x2_int-ecc_half:x2_int+ecc_half+1]
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        
        try:
            # 使用 Affine 模型對抗視角形變
            _, warp_matrix = cv2.findTransformECC(tmpl, roi_ecc, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 1)
            # 將中心點透過轉換矩陣映射，計算精確的亞像素座標
            mapped_pt = warp_matrix @ np.array([ecc_half, ecc_half, 1.0])
            x2_sub = x2_int - ecc_half + mapped_pt[0]
            msg = "匹配成功 (L-R + ECC Affine)"
        except cv2.error:
            # 若 ECC 不收斂 (例如紋理真的太弱)，退回拋物線擬合
            x2_sub = parabolic_subpixel(res_1d, max_idx_fwd) + half
            msg = "匹配成功 (L-R + Parabola)"
    else:
        x2_sub = parabolic_subpixel(res_1d, max_idx_fwd) + half
        msg = "匹配成功 (L-R + Parabola)"
        
    return x2_sub, max_score_fwd, msg, mean_grad


def main():
    mtx_L, dist_L = load_json_camera_params(PARAMS_JSON_PATH)
    if mtx_L is None:
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        print("❌ 無法開啟相機")
        sys.exit(1)

    # 讀取第一幀
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法讀取畫面")
        sys.exit(1)

    h_raw, w_raw = frame.shape[:2]
    # 取相機畫面左半邊為工作區域
    w_alg = min(HALF_WIDTH, w_raw)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1E1E1E')
    fig.canvas.manager.set_window_title("NewMeasure_v1 測距系統")
    fig.subplots_adjust(top=0.85, bottom=0.1)
    ax_A, ax_B = axes

    live_img_bgr = frame[:, :w_alg].copy()
    live_img_rgb = cv2.cvtColor(live_img_bgr, cv2.COLOR_BGR2RGB)
    
    im_A = ax_A.imshow(live_img_rgb)
    im_B = ax_B.imshow(live_img_rgb)
    
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor('#1E1E1E')
        
    ax_A.set_title('影像 A (Live - 點擊測距)', color='white', fontsize=12, pad=10)
    ax_B.set_title('影像 B (Locked - 參考幀)', color='white', fontsize=12, pad=10)

    # UI 元件
    scatter_A = ax_A.scatter([], [], s=80, c='red', marker='+', zorder=5)
    scatter_B = ax_B.scatter([], [], s=80, c='lime', marker='+', zorder=5)
    con_line = None
    
    info_text = fig.text(0.5, 0.02, "準備就緒。請先點擊「鎖定右圖」，然後在左圖點擊查詢點。", 
                         ha="center", color="white", fontsize=12,
                         bbox=dict(facecolor='#333333', alpha=0.8))

    # 狀態變數
    state = {
        'locked_img_bgr': None,
        'locked_img_gray': None,
        'rect_data': None,
        'pose_data': None,
        'imgA_rect': None,
        'imgB_rect': None,
        'live_img_bgr': live_img_bgr,
        'live_img_gray': cv2.cvtColor(live_img_bgr, cv2.COLOR_BGR2GRAY)
    }

    def process_measure(u, v):
        if state['locked_img_gray'] is None or state['rect_data'] is None:
            info_text.set_text("錯誤: 尚未鎖定右圖或未成功初始化姿態。")
            fig.canvas.draw_idle()
            return
            
        rd = state['rect_data']
        pd = state['pose_data']
        
        # 目前顯示的是校正後的影像，所以點擊座標 (u,v) 就是 (x1, y1)
        x1, y1 = u, v
        
        imgA_rect_gray = state['imgA_rect_gray']
        imgB_rect_gray = state['imgB_rect_gray']
        
        # 第五階段：ZNCC 匹配
        x2, score, msg, grad = find_match_1d(imgA_rect_gray, imgB_rect_gray, x1, y1, patch_size=21)
        
        nonlocal con_line
        if con_line is not None:
            con_line.remove()
            con_line = None
            
        scatter_A.set_offsets([[x1, y1]])
        
        if x2 is None:
            scatter_B.set_offsets(np.empty((0,2)))
            info_text.set_text(f"匹配失敗: {msg} (梯度:{grad:.1f})")
            print(f"❌ [匹配失敗] {msg} (梯度:{grad:.1f})")
            fig.canvas.draw_idle()
            return
            
        # 第七階段：DLT 三角化
        ptsA_rect = np.array([[[x1, y1]]], dtype=np.float32)
        ptsB_rect = np.array([[[x2, y1]]], dtype=np.float32)
        
        pts4d = cv2.triangulatePoints(rd['P1'], rd['P2'], ptsA_rect.T, ptsB_rect.T)
        pt3d = (pts4d[:3] / pts4d[3]).flatten()
        
        # 計算歐氏距離
        dist_cam = np.linalg.norm(pt3d)
        dist_plane = abs(np.dot(pd['plane_normal'], pt3d - pd['plane_center']))
        
        # 由於 B 圖現在也顯示為校正後影像，直接在 (x2, y1) 畫點
        scatter_B.set_offsets([[x2, y1]])
        
        con_line = ConnectionPatch(xyA=(x1, y1), xyB=(x2, y1), 
                                   coordsA="data", coordsB="data",
                                   axesA=ax_A, axesB=ax_B, color="cyan", lw=1.5, alpha=0.8)
        fig.add_artist(con_line)
        
        # 重投影誤差估計
        pt3d_camA = pt3d.reshape(3, 1)
        pt2d_A_proj, _ = cv2.projectPoints(pt3d_camA, np.zeros(3), np.zeros(3), mtx_L, dist_L)
        p1_orig_proj = pt2d_A_proj.flatten()
        # 我們將校正座標反轉回原圖以算誤差，或直接在校正空間算。使用原始投影:
        # x1, y1 的原始座標可以透過 map 尋找，或假設這就是使用者想點的位置
        # 簡單起見，我們將 pt2d_A_proj 投影到校正空間，然後和 x1, y1 比較
        err_A = "N/A" # 省略詳細重投影計算，避免反向查找映射表耗時，或者用預估
        
        res_text = (f"成功! 距離光心: {dist_cam:.1f} mm | 距 ArUco面: {dist_plane:.1f} mm\n"
                    f"ZNCC: {score:.3f} | 3D: ({pt3d[0]:.1f}, {pt3d[1]:.1f}, {pt3d[2]:.1f})")
        info_text.set_text(res_text)
        print(f"👉 [測量成功] 距離光心: {dist_cam:.1f}mm, 距平面: {dist_plane:.1f}mm, ZNCC: {score:.3f}")
        fig.canvas.draw_idle()

    pan_state = {'pressing': False, 'x': None, 'y': None, 'ax': None, 'dragged': False}

    def on_press(event):
        if event.button != 1: return
        if event.inaxes not in (ax_A, ax_B): return
        pan_state.update({'pressing': True, 'dragged': False, 'x': event.x, 'y': event.y, 'ax': event.inaxes})

    def on_release(event):
        if not pan_state.get('pressing'): return
        pan_state['pressing'] = False
        if not pan_state.get('dragged') and event.xdata is not None:
            if pan_state['ax'] == ax_A:
                u, v = event.xdata, event.ydata
                process_measure(u, v)

    def on_motion(event):
        if not pan_state.get('pressing') or event.inaxes != pan_state['ax']: return
        dx, dy = event.x - pan_state['x'], event.y - pan_state['y']
        if not pan_state['dragged'] and abs(dx) < 3 and abs(dy) < 3: return
        pan_state['dragged'] = True
        ax = pan_state['ax']
        inv = ax.transData.inverted()
        p0 = inv.transform((pan_state['x'], pan_state['y']))
        p1 = inv.transform((event.x, event.y))
        dx_d, dy_d = p1 - p0
        ax.set_xlim(ax.get_xlim() - dx_d)
        ax.set_ylim(ax.get_ylim() - dy_d)
        pan_state.update({'x': event.x, 'y': event.y})
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in (ax_A, ax_B): return
        if event.xdata is None or event.ydata is None: return
        ax = event.inaxes
        f = 1.2 if event.button == 'down' else 1/1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim([x - (x - xl[0]) * f, x + (xl[1] - x) * f])
        ax.set_ylim([y - (y - yl[0]) * f, y + (yl[1] - y) * f])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # 鎖定按鈕 UI
    ax_lock = plt.axes([0.45, 0.90, 0.1, 0.05])
    btn_lock = Button(ax_lock, '鎖定右圖')
    
    def lock_frame(event):
        imgA_gray = state['live_img_gray']
        imgB_gray = imgA_gray.copy()
        
        # 若之前有鎖定，我們假設現在要更新。為了有 disparity，應該在稍微移動相機後鎖定。
        # 其實這個按鈕是"捕捉當前畫面為右圖"。如果只是按一次，左圖也是同一幀，baseline=0。
        # 所以流程是：先照出 ArUco，按下鎖定。然後相機平移，畫面 A(live) 改變，此時點擊 A 測量。
        # 為了能在點擊時做相對位姿估計，相對位姿應該在「點擊當下」重新算？
        # 或是鎖定右圖後，每次點擊前，即時用 live_img 和 locked_img 算相對位姿。
        
        info_text.set_text("右圖已鎖定，請移動相機製造視差，然後在左圖點擊測距。")
        state['ema_rv_rel'] = None
        state['ema_tv_rel'] = None
        state['locked_img_bgr'] = state['live_img_bgr'].copy()
        state['locked_img_gray'] = state['live_img_gray'].copy()
        im_B.set_data(cv2.cvtColor(state['locked_img_bgr'], cv2.COLOR_BGR2RGB))
        scatter_A.set_offsets(np.empty((0,2)))
        scatter_B.set_offsets(np.empty((0,2)))
        nonlocal con_line
        if con_line:
            con_line.remove()
            con_line = None
        fig.canvas.draw_idle()

    btn_lock.on_clicked(lock_frame)

    # 即時更新 Loop (使用 Matplotlib 的 pause)
    plt.ion()
    plt.show()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        live_img_bgr = frame[:, :w_alg].copy()
        live_img_gray = cv2.cvtColor(live_img_bgr, cv2.COLOR_BGR2GRAY)
        state['live_img_bgr'] = live_img_bgr
        state['live_img_gray'] = live_img_gray
        
        # 背景中不斷更新立體校正矩陣 (如果已經鎖定了右圖)
        if state['locked_img_gray'] is not None:
            ok, msg, pd = estimate_relative_pose(live_img_gray, state['locked_img_gray'], mtx_L, dist_L, ACTUAL_MARKER_SIZE_MM)
            if ok:
                # 機制 1: EMA 時域低通濾波 (權重 0.2 代表新值佔 20%)
                alpha = 0.2
                if state.get('ema_rv_rel') is None:
                    state['ema_rv_rel'] = pd['rv_rel'].copy()
                    state['ema_tv_rel'] = pd['tv_rel'].copy()
                else:
                    state['ema_rv_rel'] = alpha * pd['rv_rel'] + (1 - alpha) * state['ema_rv_rel']
                    state['ema_tv_rel'] = alpha * pd['tv_rel'] + (1 - alpha) * state['ema_tv_rel']
                
                # 更新平滑後的 R_rel, T_rel
                R_rel_smooth, _ = cv2.Rodrigues(state['ema_rv_rel'])
                T_rel_smooth = state['ema_tv_rel']
                pd['R_rel'] = R_rel_smooth
                pd['T_rel'] = T_rel_smooth
                pd['baseline'] = float(np.linalg.norm(T_rel_smooth))
                
                state['pose_data'] = pd
                imgA_rect_bgr, imgB_rect_bgr, rd = stereo_rectify_and_remap(live_img_bgr, state['locked_img_bgr'], mtx_L, dist_L, pd['R_rel'], pd['T_rel'])
                state['rect_data'] = rd
                state['imgA_rect_gray'] = cv2.cvtColor(imgA_rect_bgr, cv2.COLOR_BGR2GRAY)
                state['imgB_rect_gray'] = cv2.cvtColor(imgB_rect_bgr, cv2.COLOR_BGR2GRAY)
                
                # 顯示校正後的影像
                im_A.set_data(cv2.cvtColor(imgA_rect_bgr, cv2.COLOR_BGR2RGB))
                im_B.set_data(cv2.cvtColor(imgB_rect_bgr, cv2.COLOR_BGR2RGB))
                
                ax_A.set_title(f"影像 A (Live 立體校正) - baseline: {pd['baseline']:.1f}mm", color='white')
            else:
                ax_A.set_title("影像 A (Live 立體校正) - 尋找 ArUco 中...", color='yellow')
                # 如果找不到，保持顯示最後一次的校正畫面 (如果有)
                if state['rect_data'] is not None:
                    rd = state['rect_data']
                    imgA_rect_bgr = cv2.remap(live_img_bgr, rd['map1_A'], rd['map2_A'], cv2.INTER_LINEAR)
                    state['imgA_rect_gray'] = cv2.cvtColor(imgA_rect_bgr, cv2.COLOR_BGR2GRAY)
                    im_A.set_data(cv2.cvtColor(imgA_rect_bgr, cv2.COLOR_BGR2RGB))
        else:
            im_A.set_data(cv2.cvtColor(live_img_bgr, cv2.COLOR_BGR2RGB))

        fig.canvas.flush_events()
        plt.pause(0.01)
        
        # 關閉視窗處理
        if not plt.fignum_exists(fig.number):
            break

    cap.release()

if __name__ == "__main__":
    main()
