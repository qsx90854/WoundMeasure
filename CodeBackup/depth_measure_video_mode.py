import os, sys, json, threading, queue
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局設定區 ====================
VIDEO_PATH            = "test_video/1.mp4"
PARAMS_JSON_PATH      = "calibration_result_c2.json"
ACTUAL_MARKER_SIZE_MM = 8.25
MARKER_DICT           = cv2.aruco.DICT_4X4_100

def load_calibration(path):
    if not os.path.exists(path):
        print(f"找不到標定檔 {path}，使用預設值")
        return np.eye(3), np.zeros(5)
    with open(path, 'r') as f:
        data = json.load(f)
        
    if 'intrinsic_L' in data:
        # 新格式 (calibration_result_c2.json 等)
        return np.array(data['intrinsic_L']['matrix']), np.array(data['intrinsic_L']['distortion'])
    else:
        # 舊格式相容
        return np.array(data['K']), np.array(data['dist'])

K_cam, dist_cam = load_calibration(PARAMS_JSON_PATH)

# ==================== 核心演算法 ====================

def get_multi_aruco_pose(img_gray, K, dist, marker_size_mm, virtual_board, prev_gray=None, prev_aruco_corners=None):
    """
    實作動態虛擬板 (Dynamic Virtual Board) 演算法，並具備光流補幀能力。
    virtual_board: 傳入參照字典 { marker_id: np.array([[X, Y, Z], ...]) }
    prev_gray, prev_aruco_corners: 用於當原生偵測因模糊失敗時，啟動光流補幀 (Optical Flow Fallback)
    回傳: R_w2c, t_w2c, origin_id, corners, virtual_board
    """
    # 1. 影像前處理：使用 CLAHE 增強局部對比度，大幅減少動態模糊導致的邊界丟失
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(img_gray)
    
    dict_4x4 = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(enhanced_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(enhanced_gray, dict_4x4, parameters=params)
    
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ids is not None and len(ids) > 0:
        ids = [i[0] for i in ids]
        for c in corners:
            cv2.cornerSubPix(enhanced_gray, c, (3, 3), (-1, -1), term)
        corners_dict = dict(zip(ids, [c[0] for c in corners]))
    else:
        ids = []
        corners_dict = {}
        
    # 2. 光流補幀 (Optical Flow Fallback)
    # 如果上一幀有的標籤，這一幀 OpenCV 沒抓到 (被模糊吃掉)，我們用光流法硬追蹤過來！
    if prev_gray is not None and prev_aruco_corners is not None:
        lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        for cid, prev_pts in prev_aruco_corners.items():
            if cid not in ids:
                # 準備光流追蹤 (4x2 points)
                p0 = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, img_gray, p0, None, **lk_params)
                
                # 如果 4 個角點都追蹤成功
                if st is not None and np.sum(st) == 4:
                    tracked_pts = p1.reshape(4, 2)
                    # 進行 SubPix 優化 (雖然是追蹤來的，還是稍微優化一下)
                    cv2.cornerSubPix(enhanced_gray, tracked_pts, (3, 3), (-1, -1), term)
                    corners_dict[cid] = tracked_pts
                    ids.append(cid)
                    
    if len(ids) == 0:
        return None, None, None, None, virtual_board
    
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    
    # 模式一：Virtual Board 為空，進行初始化 (Map Initialization)
    if not virtual_board:
        origin_id = min(ids)
        origin_corners = corners_dict[origin_id]
        
        ok, rvec_w, tvec_w = cv2.solvePnP(canon, origin_corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok: return None, None, None, None, virtual_board
        
        R_w2c, _ = cv2.Rodrigues(rvec_w)
        if np.dot(R_w2c[:, 2], tvec_w.flatten()) > 0:
            return None, None, None, None, virtual_board
            
        # 將 origin 註冊進 Virtual Board
        virtual_board[origin_id] = canon.copy()
        
        # 將其他標籤也註冊進 Virtual Board
        for cid in ids:
            if cid == origin_id: continue
            ok_i, rvec_i, tvec_i = cv2.solvePnP(canon, corners_dict[cid], K, dist, flags=cv2.SOLVEPNP_IPPE)
            if ok_i:
                R_i, _ = cv2.Rodrigues(rvec_i)
                # X_c = R_i * X_canon + t_i
                X_c = (R_i @ canon.T).T + tvec_i.reshape(1, 3)
                # X_w = R_w2c^T * (X_c - t_w)
                X_w = (R_w2c.T @ (X_c - tvec_w.reshape(1, 3)).T).T
                virtual_board[cid] = X_w.astype(np.float32)
                
        return R_w2c, tvec_w, origin_id, corners_dict, virtual_board

    # 模式二：多點 PnP 追蹤 (Multi-Point PnP)
    obj_pts = []
    img_pts = []
    matched_ids = []
    unmatched_ids = []
    
    for cid in ids:
        if cid in virtual_board:
            obj_pts.append(virtual_board[cid])
            img_pts.append(corners_dict[cid])
            matched_ids.append(cid)
        else:
            unmatched_ids.append(cid)
            
    if len(obj_pts) == 0:
        return None, None, None, None, virtual_board
        
    obj_pts = np.vstack(obj_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)
    
    # 使用 EPNP 或 ITERATIVE 解算所有收集到的點
    flags = cv2.SOLVEPNP_EPNP if len(obj_pts) >= 6 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec_w, tvec_w = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags)
    if not ok: return None, None, None, None, virtual_board
    
    R_w2c, _ = cv2.Rodrigues(rvec_w)
    if np.dot(R_w2c[:, 2], tvec_w.flatten()) > 0:
        return None, None, None, None, virtual_board
        
    # 模式三：動態地圖擴充 (Dynamic Mapping)
    for cid in unmatched_ids:
        ok_i, rvec_i, tvec_i = cv2.solvePnP(canon, corners_dict[cid], K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok_i:
            R_i, _ = cv2.Rodrigues(rvec_i)
            X_c = (R_i @ canon.T).T + tvec_i.reshape(1, 3)
            X_w = (R_w2c.T @ (X_c - tvec_w.reshape(1, 3)).T).T
            virtual_board[cid] = X_w.astype(np.float32)
            
    origin_id = min(virtual_board.keys()) if virtual_board else None
    return R_w2c, tvec_w, origin_id, corners_dict, virtual_board

def undistort_points_for_triangulation(pts_2d, K, dist):
    """將原始像素消除畸變，並轉換為相機正規化座標 (Normalized Coordinates, x=(u-cx)/fx)"""
    if len(pts_2d) == 0: return []
    pts_arr = np.array(pts_2d, dtype=np.float32).reshape(-1, 1, 2)
    # 不傳入 P=K，回傳的即是正規化座標 (x, y)
    undistorted = cv2.undistortPoints(pts_arr, K, dist)
    return [pt[0].tolist() for pt in undistorted]

def multi_view_triangulation(P_matrices, points_2d):
    """
    N-View 三角測量演算法 (DLT)
    P_matrices: list of 3x4 projection matrices, [R | t]
    points_2d: list of (x, y) normalized coordinates
    """
    if len(P_matrices) < 2: return None
    A = []
    for P, (x, y) in zip(P_matrices, points_2d):
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    # 避免 W 極小導致除零
    if abs(X[3]) < 1e-6: return None
    return (X[:3] / X[3])

# ==================== UI 與 主迴圈 ====================

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"錯誤：找不到影片 {VIDEO_PATH}")
        return
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0: total_frames = 100
    
    # 狀態管理
    app_state = {
        'playing': False,
        'frame_idx': 0,
        'global_origin_id': None,
        'virtual_board': {},
        'prev_aruco_corners': None,  # 用於光流補幀
        'auto_calc': False,          # 是否連續自動計算
        'manual_mode': False,        # 是否開啟手動匹配模式
        'manual_click_idx': 0,       # 手動模式點擊的是第一點 (0) 還是第二點 (1)
        'current_P': None,           # 暫存當前幀的相機姿態供點擊時提取
        
        # 光流追蹤狀態
        'is_tracking': False,
        'track_pts': [],     # [[u,v], [u,v]] of current frame
        'track_history': {}, # point_idx -> list of {f_idx, pt: [u,v], P: 3x4} (僅存有 ArUco 的幀，供計算用)
        'ui_track_history': {}, # point_idx -> list of [u,v] (所有追蹤過的點，供畫線用)
        'track_colors': ['cyan', 'magenta'],
        
        # 最新畫面與灰階
        'current_bgr': None,
        'current_gray': None,
        'aruco_corners': {},
        
        # UI 狀態
        'pick_mode': False   # True 時等待使用者點擊畫面
    }
    
    # 建立 UI
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.05, right=0.95)
    
    ret, initial_frame = cap.read()
    if not ret: return
    app_state['current_bgr'] = initial_frame
    app_state['current_gray'] = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    
    im_obj = ax.imshow(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    # UI 元件
    scatter_pts = ax.scatter([], [], c=[], s=80, marker='x', zorder=5)
    track_lines = [ax.plot([], [], c=c, lw=1.5, alpha=0.7)[0] for c in app_state['track_colors']]
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color='white', 
                        bbox=dict(facecolor='#121212', alpha=0.7), fontsize=10, va='top')
    depth_text = ax.text(0.02, 0.05, "", transform=ax.transAxes, color='yellow', 
                         bbox=dict(facecolor='#121212', alpha=0.7), fontsize=14, fontweight='bold', va='bottom')
    
    # 底部控制區
    ax_slider = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='#222222')
    slider = Slider(ax_slider, 'Frame', 0, total_frames-1, valinit=0, valstep=1, color='#007ACC')
    
    btn_style = dict(color='#1A1A1A', hovercolor='#333333')
    ax_play = plt.axes([0.1, 0.05, 0.1, 0.06])
    b_play = Button(ax_play, 'Play / Pause', **btn_style)
    b_play.label.set_color('white')
    
    ax_pick = plt.axes([0.22, 0.05, 0.12, 0.06])
    b_pick = Button(ax_pick, 'Select Points', **btn_style)
    b_pick.label.set_color('white')
    b_pick.ax.patch.set_edgecolor('#D83B01')
    b_pick.ax.patch.set_linewidth(1.5)
    
    ax_calc = plt.axes([0.36, 0.05, 0.12, 0.06])
    b_calc = Button(ax_calc, 'Calc Depth', **btn_style)
    b_calc.label.set_color('white')
    b_calc.ax.patch.set_edgecolor('#00FF00')
    b_calc.ax.patch.set_linewidth(1.5)
    
    ax_clear = plt.axes([0.5, 0.05, 0.1, 0.06])
    b_clear = Button(ax_clear, 'Clear Tracks', **btn_style)
    b_clear.label.set_color('white')
    
    ax_auto_calc = plt.axes([0.62, 0.05, 0.12, 0.06])
    b_auto_calc = Button(ax_auto_calc, 'Auto: OFF', **btn_style)
    b_auto_calc.label.set_color('white')
    b_auto_calc.ax.patch.set_edgecolor('#00FFFF')
    b_auto_calc.ax.patch.set_linewidth(1.5)
    
    ax_manual = plt.axes([0.76, 0.05, 0.12, 0.06])
    b_manual = Button(ax_manual, 'Manual: OFF', **btn_style)
    b_manual.label.set_color('white')
    b_manual.ax.patch.set_edgecolor('#FF00FF')
    b_manual.ax.patch.set_linewidth(1.5)
    
    # 事件綁定
    def update_frame(val):
        f_idx = int(slider.val)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: return
        
        old_gray = app_state['current_gray']
        app_state['frame_idx'] = f_idx
        app_state['current_bgr'] = frame
        app_state['current_gray'] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ArUco 姿態估計 (結合光流補幀)
        R_w, t_w, o_id, corners, v_board = get_multi_aruco_pose(
            app_state['current_gray'], K_cam, dist_cam, ACTUAL_MARKER_SIZE_MM, 
            app_state['virtual_board'], app_state.get('prev_gray'), app_state['prev_aruco_corners']
        )
        app_state['aruco_corners'] = corners or {}
        app_state['virtual_board'] = v_board
        
        # 儲存成功偵測到的角點，供下一幀做光流補幀使用
        app_state['prev_aruco_corners'] = app_state['aruco_corners'] if len(app_state['aruco_corners']) > 0 else None
        
        if o_id is not None:
            app_state['global_origin_id'] = o_id
            
        current_P = None
        if R_w is not None and t_w is not None:
            # P = [R | t] 供正規化座標三角測量使用
            current_P = np.hstack((R_w, t_w))
        app_state['current_P'] = current_P
        
        # 光流追蹤
        if app_state['is_tracking'] and len(app_state['track_pts']) > 0 and old_gray is not None:
            p0 = np.array(app_state['track_pts'], dtype=np.float32).reshape(-1, 1, 2)
            # 加大 winSize 與 maxLevel 以增強對抗 Motion Blur (畫面模糊) 的穩定度
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, app_state['current_gray'], p0, None, 
                                                   winSize=(31, 31), maxLevel=3,
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            
            new_track_pts = []
            for i, (pt, status) in enumerate(zip(p1, st)):
                if status[0] == 1:
                    new_pt = pt[0].tolist()
                    new_track_pts.append(new_pt)
                    
                    # 無論 ArUco 是否偵測到，都存入 UI 軌跡供畫線
                    if i not in app_state['ui_track_history']:
                        app_state['ui_track_history'][i] = []
                    app_state['ui_track_history'][i].append(new_pt)
                    
                    # 只有當有計算出相機矩陣 (ArUco 偵測成功) 時，才存入運算用的歷史軌跡
                    if current_P is not None:
                        app_state['track_history'][i].append({
                            'f_idx': f_idx,
                            'pt': new_pt,
                            'P': current_P
                        })
                else:
                    new_track_pts.append(app_state['track_pts'][i]) # 追蹤失敗，留著舊的
            app_state['track_pts'] = new_track_pts
            
        # 如果開啟了自動計算，就每幀更新深度數字
        if app_state['auto_calc'] and app_state['is_tracking'] and len(app_state['track_history'][0]) >= 2:
            calculate_and_display_depth()
            
        redraw_ui()

    slider.on_changed(update_frame)
    
    def on_play(e):
        app_state['playing'] = not app_state['playing']
        if app_state['playing']:
            play_loop()
    b_play.on_clicked(on_play)
    
    def play_loop():
        if app_state['playing'] and plt.fignum_exists(fig.number):
            val = slider.val + 1
            if val >= total_frames:
                # 影片到底時自動暫停，避免光流歷史軌跡受到跳回開頭的影響而損壞
                app_state['playing'] = False
                return
            slider.set_val(val)
            fig.canvas.start_event_loop(0.01) # 短暫讓出控制權
            fig.canvas.callbacks.callbacks.get('idle_event', []) # trigger play_loop again?
            # matplotlib 的 timer 較穩，這裡用簡易遞迴或 timer
            plt.pause(0.03) # roughly 30fps
            play_loop()
            
    def on_pick_btn(e):
        app_state['pick_mode'] = True
        app_state['is_tracking'] = False
        app_state['track_pts'] = []
        app_state['track_history'] = {0: [], 1: []}
        app_state['ui_track_history'] = {0: [], 1: []}
        app_state['manual_mode'] = False
        b_manual.label.set_text('Manual: OFF')
        info_text.set_text("Pick Mode: ON. \nClick 2 points on the image.")
        fig.canvas.draw_idle()
    b_pick.on_clicked(on_pick_btn)
    
    def on_clear(e):
        app_state['track_pts'] = []
        app_state['track_history'] = {0: [], 1: []}
        app_state['ui_track_history'] = {0: [], 1: []}
        app_state['is_tracking'] = False
        app_state['pick_mode'] = False
        app_state['playing'] = False
        app_state['virtual_board'] = {}
        app_state['prev_aruco_corners'] = None
        app_state['auto_calc'] = False
        app_state['manual_mode'] = False
        app_state['manual_click_idx'] = 0
        b_auto_calc.label.set_text('Auto: OFF')
        b_manual.label.set_text('Manual: OFF')
        depth_text.set_text("")
        redraw_ui()
    b_clear.on_clicked(on_clear)
    
    def on_manual_mode(e):
        app_state['manual_mode'] = not app_state['manual_mode']
        b_manual.label.set_text(f"Manual: {'ON' if app_state['manual_mode'] else 'OFF'}")
        if app_state['manual_mode']:
            app_state['is_tracking'] = False
            app_state['pick_mode'] = True
            app_state['manual_click_idx'] = 0
            info_text.set_text("Manual Mode ON.\nClick Pt1 in any frame.")
        else:
            app_state['pick_mode'] = False
            info_text.set_text("Manual Mode OFF.")
        fig.canvas.draw_idle()
    b_manual.on_clicked(on_manual_mode)
    
    def on_auto_calc(e):
        app_state['auto_calc'] = not app_state['auto_calc']
        b_auto_calc.label.set_text(f"Auto: {'ON' if app_state['auto_calc'] else 'OFF'}")
        fig.canvas.draw_idle()
    b_auto_calc.on_clicked(on_auto_calc)
    
    def calculate_and_display_depth():
        if len(app_state['track_history'][0]) < 2 or len(app_state['track_history'][1]) < 2:
            depth_text.set_text("Not enough tracking history!")
            return
            
        pts3d = []
        for i in range(2):
            hist = app_state['track_history'][i]
            
            # 建立數值穩定 (Well-conditioned) 的 P 矩陣
            P_mats = []
            for h in hist:
                P = h['P'].copy()
                # 將 tvec 從 mm 轉為 m，與旋轉矩陣 R 的量級 (~1.0) 匹配，避免 SVD 崩潰
                P[:, 3] /= 1000.0
                P_mats.append(P)
                
            pts2d = [h['pt'] for h in hist]
            
            # 取得「正規化坐標 (Normalized Coordinates)」
            norm_pts = undistort_points_for_triangulation(pts2d, K_cam, dist_cam)
            
            # 進行 DLT 三角測量，此時求出的 3D 座標單位為公尺 (m)
            pt3d_m = multi_view_triangulation(P_mats, norm_pts)
            
            if pt3d_m is not None:
                # 轉回 mm
                pts3d.append(pt3d_m * 1000.0)
            else:
                pts3d.append(None)
            
        # 計算攝影機的最大基準線 (Max Baseline)
        max_baseline = 0
        if len(app_state['track_history'][0]) > 0:
            hist = app_state['track_history'][0]
            centers = []
            for h in hist:
                P = h['P'] # [R | t] (mm level before scaling)
                R = P[:, :3]
                t = P[:, 3]
                C = -R.T @ t
                centers.append(C)
            for c1 in centers:
                for c2 in centers:
                    dist = np.linalg.norm(c1 - c2)
                    if dist > max_baseline:
                        max_baseline = dist
                        
        if pts3d[0] is not None and pts3d[1] is not None:
            dist_3d = np.linalg.norm(pts3d[0] - pts3d[1])
            delta_z = abs(pts3d[0][2] - pts3d[1][2])
            
            p1_str = f"P1: ({pts3d[0][0]:.1f}, {pts3d[0][1]:.1f}, {pts3d[0][2]:.1f})"
            p2_str = f"P2: ({pts3d[1][0]:.1f}, {pts3d[1][1]:.1f}, {pts3d[1][2]:.1f})"
            base_str = f"Max Baseline: {max_baseline:.1f} mm"
            warn_str = "\n[WARN] Baseline < 20mm! Translate camera!" if max_baseline < 20.0 else ""
            
            depth_text.set_text(f"{p1_str}\n{p2_str}\nDelta Z: {delta_z:.1f} mm | Dist: {dist_3d:.1f} mm\n{base_str}{warn_str}")
            if not app_state['auto_calc']:
                print(f"Calculated using {len(app_state['track_history'][0])} frames. Max Baseline: {max_baseline:.1f} mm")
                
    def on_calc_btn(e):
        calculate_and_display_depth()
        fig.canvas.draw_idle()
    b_calc.on_clicked(on_calc_btn)
    
    def on_click(e):
        # 防呆：如果 matplotlib 的縮放/拖曳工具正在使用中，不要觸發選點
        if fig.canvas.manager.toolbar.mode != '': return
        if not app_state['pick_mode']: return
        if e.inaxes != ax: return
        
        if app_state['manual_mode']:
            # 手動模式：允許使用者在不同影格自由點擊
            if app_state['current_P'] is None:
                print("Warning: ArUco tracking lost in this frame. Cannot pick points here!")
                info_text.set_text("Warning: ArUco tracking lost!\nCannot pick points here.")
                fig.canvas.draw_idle()
                return
                
            pt = [e.xdata, e.ydata]
            idx = app_state['manual_click_idx']
            
            # 維護畫面上顯示的 2 個點
            if len(app_state['track_pts']) <= idx:
                app_state['track_pts'].append(pt)
            else:
                app_state['track_pts'][idx] = pt
                
            # 加入歷史紀錄
            if idx not in app_state['ui_track_history']:
                app_state['ui_track_history'][idx] = []
            app_state['ui_track_history'][idx].append(pt)
            
            app_state['track_history'][idx].append({
                'f_idx': app_state['frame_idx'],
                'pt': pt,
                'P': app_state['current_P'].copy()
            })
            
            # 切換下一次點擊的 ID
            app_state['manual_click_idx'] = 1 - app_state['manual_click_idx']
            next_pt = "Pt1" if app_state['manual_click_idx'] == 0 else "Pt2"
            info_text.set_text(f"Manual Pick: Point {1 - app_state['manual_click_idx'] + 1} logged.\nNow click {next_pt} in any frame.")
            
        else:
            # 自動模式：點擊 2 點後開始光流追蹤
            if len(app_state['track_pts']) < 2:
                pt = [e.xdata, e.ydata]
                idx = len(app_state['track_pts'])
                app_state['track_pts'].append(pt)
                
                if idx not in app_state['ui_track_history']:
                    app_state['ui_track_history'][idx] = []
                app_state['ui_track_history'][idx].append(pt)
                
                if app_state['current_P'] is not None:
                    app_state['track_history'][idx].append({
                        'f_idx': app_state['frame_idx'],
                        'pt': pt,
                        'P': app_state['current_P'].copy()
                    })
                    
            if len(app_state['track_pts']) == 2:
                app_state['pick_mode'] = False
                app_state['is_tracking'] = True
                info_text.set_text("Tracking Mode Active.")
                
        redraw_ui()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    def redraw_ui():
        disp_bgr = app_state['current_bgr'].copy()
        
        # 繪製 ArUco
        for cid, corners in app_state['aruco_corners'].items():
            pts = np.vstack((corners, corners[0])).astype(int)
            # 如果這個標籤已經加入了 Virtual Board，畫綠色代表穩固，否則畫黃色
            color = (0, 255, 0) if cid in app_state['virtual_board'] else (0, 255, 255)
            cv2.polylines(disp_bgr, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(disp_bgr, f"ID:{cid}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 在左上角顯示 Virtual Board 狀態
        board_status = f"Virtual Board: {len(app_state['virtual_board'])} markers"
        cv2.putText(disp_bgr, board_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        im_obj.set_data(cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB))
        
        # 繪製選取點
        if len(app_state['track_pts']) > 0:
            xs = [p[0] for p in app_state['track_pts']]
            ys = [p[1] for p in app_state['track_pts']]
            cs = app_state['track_colors'][:len(app_state['track_pts'])]
            scatter_pts.set_offsets(np.c_[xs, ys])
            scatter_pts.set_color(cs)
        else:
            scatter_pts.set_offsets(np.empty((0, 2)))
            
        # 繪製軌跡 (使用 ui_track_history 確保軌跡連續不中斷)
        for i in range(2):
            if i in app_state['ui_track_history'] and len(app_state['ui_track_history'][i]) > 1:
                hist = app_state['ui_track_history'][i]
                tx = [p[0] for p in hist]
                ty = [p[1] for p in hist]
                track_lines[i].set_data(tx, ty)
            else:
                track_lines[i].set_data([], [])
                
        info = f"Frame: {app_state['frame_idx']}/{total_frames}\n"
        if app_state['global_origin_id'] is not None:
            info += f"Global Origin: ArUco ID {app_state['global_origin_id']}\n"
        else:
            info += "Global Origin: NOT FOUND\n"
        info += f"Tracking points: {len(app_state['track_pts'])}/2"
        info_text.set_text(info)
        
        fig.canvas.draw_idle()

    # Initial draw
    update_frame(0)
    plt.show()

if __name__ == "__main__":
    main()
