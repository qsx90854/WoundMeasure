import os
import glob
import cv2
import numpy as np
import onnxruntime as ort
import configparser
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
import time
from scipy.interpolate import Rbf  # 需在檔案頂部 import


# ==================== 全局設定區 ====================
IMAGE_FOLDER = "captured_images_m2"    # 你的圖片資料夾
ACTUAL_MARKER_SIZE_MM = 29.0        # ArUco 標籤真實尺寸
TARGET_MARKER_ID = 12               # ArUco 標籤 ID
TARGET_W = 1024                     # 統一處理寬度

# 流程模式切換：
# 1 = 單張 AI 深度 + 多圖特徵點聯合 RANSAC 修正 (速度快，比例準)
# 2 = 全圖 AI 深度 + 全局 ArUco 點雲空間融合 (細節最好，可消除局部扭曲)
PIPELINE_MODE = 1

# 若為模式 1，選擇哪一張圖作為主要的深度生成基礎 (Index)
REFERENCE_IMAGE_INDEX = 0           
# ====================================================

def get_aruco_pose(image_gray, K, marker_size_mm, target_id):
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dict_4x4, detector_params)
        corners, ids, _ = detector.detectMarkers(image_gray)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image_gray, dict_4x4, parameters=detector_params)

    if ids is None or target_id not in ids:
        return None, None

    idx = np.where(ids == target_id)[0][0]
    marker_corners = corners[idx]
    half_size = marker_size_mm / 2.0
    obj_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners[0], K, np.zeros(5))
    if success:
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec
    return None, None

def load_raw_camera_params(ini_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    fx = float(config['intrinsic1']['fx'])
    fy = float(config['intrinsic1']['fy'])
    cx = float(config['intrinsic1']['cx'])
    cy = float(config['intrinsic1']['cy'])
    k1 = float(config['distortion1']['k1'])
    k2 = float(config['distortion1']['k2'])
    k3 = float(config['distortion1']['k3'])
    p1 = float(config['distortion1']['p1'])
    p2 = float(config['distortion1']['p2'])
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, dist_coeffs

def run_depth_anything(session, image_bgr, target_shape, save_path=None):
    input_size = 518
    img_resized = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    img_tensor = img_norm.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
    
    start_time = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: img_tensor})
    end_time = time.perf_counter()
    print(f"DepthAnthingV2 cost: {(end_time - start_time)*1000:.2f} ms")
    depth_aligned = cv2.resize(outputs[0][0], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    
    
    # ====== 新增：儲存視差視覺化圖 ======
    if save_path:
        # 1. 歸一化到 0-255
        depth_min = depth_aligned.min()
        depth_max = depth_aligned.max()
        depth_norm = (depth_aligned - depth_min) / (depth_max - depth_min + 1e-8) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        
        # 2. 套用偽彩色（可選，讓視覺效果更明顯，像熱點圖）
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        # 3. 存檔
        cv2.imwrite(save_path, depth_color)
        print(f"   [Debug] 已儲存深度視覺化圖: {save_path}")
    # ==================================
    
    return depth_aligned

def compute_sparse_metric_points(imgA_gray, imgB_gray, K_scaled, lg_session, pair_index=0):
    """
    執行 LightGlue 匹配、繪製隨機顏色匹配連線，並回傳 3D 深度點
    """
    t0 = imgA_gray.astype(np.float32) / 255.0
    t1 = imgB_gray.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.stack([t0, t1], axis=0), axis=1)
    start_time = time.perf_counter()
    outputs = lg_session.run(['keypoints', 'matches', 'mscores'], {"images": input_tensor})
    end_time = time.perf_counter()
    print(f"lightglue cost: {(end_time - start_time)*1000:.2f} ms")
    kpts, matches, scores = outputs[0], outputs[1], outputs[2]

    # 過濾高信心度匹配
    valid_matches = [(kpts[0, int(m[1])], kpts[1, int(m[2])]) for m, s in zip(matches, scores) if s > 0.5]
    
    # ====== 新增：繪製匹配點連線圖 ======
    if len(valid_matches) > 0:
        # 建立畫布：將兩張圖左右拼接
        h, w = imgA_gray.shape
        vis_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis_img[:, :w] = cv2.cvtColor(imgA_gray, cv2.COLOR_GRAY2BGR)
        vis_img[:, w:] = cv2.cvtColor(imgB_gray, cv2.COLOR_GRAY2BGR)

        for ptA, ptB in valid_matches:
            # 隨機顏色 (B, G, R)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            p1 = (int(ptA[0]), int(ptA[1]))
            p2 = (int(ptB[0]) + w, int(ptB[1])) # 第二張圖的 X 座標要加上偏移量 w
            cv2.line(vis_img, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(vis_img, p1, 2, color, -1)
            cv2.circle(vis_img, p2, 2, color, -1)

        # 輸出圖片
        os.makedirs("debug_matches", exist_ok=True)
        cv2.imwrite(f"debug_matches/match_pair_{pair_index}.jpg", vis_img)
        print(f"   [Debug] 已儲存匹配連線圖: debug_matches/match_pair_{pair_index}.jpg")
    # ==================================

    if len(valid_matches) < 10: return None, None
    
    ptsA = np.array([m[0] for m in valid_matches], dtype=np.float32)
    ptsB = np.array([m[1] for m in valid_matches], dtype=np.float32)

    E, mask = cv2.findEssentialMat(ptsA, ptsB, K_scaled, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None: return None, None
    _, R, t, mask_pose = cv2.recoverPose(E, ptsA, ptsB, K_scaled, mask=mask)

    R_A, t_A = get_aruco_pose(imgA_gray, K_scaled, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)
    R_B, t_B = get_aruco_pose(imgB_gray, K_scaled, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)

    if R_A is not None and R_B is not None:
        C_A = -R_A.T @ t_A
        C_B = -R_B.T @ t_B
        true_baseline_mm = np.linalg.norm(C_A - C_B)
    else:
        return None, None 

    T_scaled = t * true_baseline_mm
    P0 = np.float32(K_scaled @ np.hstack((np.eye(3), np.zeros((3, 1)))))
    P1 = np.float32(K_scaled @ np.hstack((R, T_scaled)))
    
    valid_mask = mask_pose.ravel() > 0 
    ptsA_valid = np.float32(ptsA[valid_mask]).T
    ptsB_valid = np.float32(ptsB[valid_mask]).T
    
    points_4D = cv2.triangulatePoints(P0, P1, ptsA_valid, ptsB_valid)
    Z_metric = (points_4D[:3, :] / points_4D[3, :])[2, :] 
    
    return ptsA_valid.T, Z_metric

def align_and_generate_pcd(ai_depth_map, pts_2d, Z_metric, K_scaled, img_color, img_gray, return_world_pcd=False):
    TARGET_H, TARGET_W = ai_depth_map.shape
    good_z_mask = (Z_metric > 0) & (Z_metric < 2000) 
    pts_fit = pts_2d[good_z_mask]
    Z_metric_fit = Z_metric[good_z_mask]
    
    # --- Step A: 全域修正 (RANSAC) ---
    inv_Z_metric = (1.0 / Z_metric_fit).reshape(-1, 1)
    ai_samples = []
    for pt in pts_fit:
        u, v = int(np.clip(pt[0], 0, TARGET_W-1)), int(np.clip(pt[1], 0, TARGET_H-1))
        ai_samples.append(ai_depth_map[v, u])
    ai_samples = np.array(ai_samples).reshape(-1, 1)

    if len(ai_samples) < 10: return None

    # 取得全域線性參數
    ransac = RANSACRegressor(min_samples=int(len(ai_samples)*0.3), residual_threshold=0.001)
    ransac.fit(ai_samples, inv_Z_metric)
    scale_factor = ransac.estimator_.coef_[0][0]
    shift = ransac.estimator_.intercept_[0]
    
    # 產生全域修正後的初步逆深度圖
    global_inv_depth = np.clip(scale_factor * ai_depth_map + shift, 1e-6, None)

    # --- Step B: 區域修正 (RBF Local Correction) ---
    # 計算每個特徵點處，全域模型與真實物理值之間的「殘差」
    # 殘差 = 真實逆深度 - 全域預測逆深度
    global_ref_inv = scale_factor * ai_samples.flatten() + shift
    residuals = inv_Z_metric.flatten() - global_ref_inv
    
    # 使用 RBF 建立殘差場 (這能捕捉 AI 沒算出來的局部凹陷)
    # 我們只取一部分點來加速計算，否則 5000 點會運算很久
    step = max(1, len(pts_fit) // 500) 
    rbf = Rbf(pts_fit[::step, 0], pts_fit[::step, 1], residuals[::step], 
              function='thin_plate', smooth=0.1)
    
    # 產生全圖的殘差修正量
    u_grid, v_grid = np.meshgrid(np.linspace(0, TARGET_W-1, TARGET_W//8), 
                                 np.linspace(0, TARGET_H-1, TARGET_H//8))
    res_grid = rbf(u_grid, v_grid)
    # 放大回原圖尺寸
    full_res_map = cv2.resize(res_grid, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    
    # 最終逆深度 = 全域 + 區域殘差
    final_inv_depth = np.clip(global_inv_depth + full_res_map, 1e-6, None)
    final_metric_depth = 1.0 / final_inv_depth

    # --- Step C: 長度約束與投影 (後續邏輯不變) ---
    physical_correction = get_pcd_scale_factor(img_gray, K_scaled, final_metric_depth, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)
    final_metric_depth *= physical_correction 

    # 投影點雲邏輯...
    fx, fy = K_scaled[0, 0], K_scaled[1, 1]
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]
    u, v = np.meshgrid(np.arange(TARGET_W), np.arange(TARGET_H))
    X = (u - cx) * final_metric_depth / fx
    Y = (v - cy) * final_metric_depth / fy
    
    points = np.stack((X, Y, final_metric_depth), axis=-1).reshape(-1, 3)
    colors = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    # ... 其餘代碼與原檔相同

    dist_mask = (points[:, 2] < 1500) & (points[:, 2] > 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[dist_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[dist_mask])
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)

    # 轉換到全局 ArUco 座標系
    if return_world_pcd:
        R_A, t_A = get_aruco_pose(img_gray, K_scaled, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)
        if R_A is not None:
            T_cam_to_world = np.eye(4)
            T_cam_to_world[:3, :3] = R_A.T
            T_cam_to_world[:3, 3] = (-R_A.T @ t_A).flatten()
            pcd.transform(T_cam_to_world)
        else:
            print("⚠️ 警告：此幀找不到 ArUco，無法投影至全局座標。")

    return pcd
def get_pcd_scale_factor(img_gray, K_scaled, final_metric_depth, marker_size_mm, target_id):
    """
    找出點雲中 ArUco 標籤的 3D 長度，並計算與真實尺寸的比例
    """
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dict_4x4, detector_params)
        corners, ids, _ = detector.detectMarkers(img_gray)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(img_gray, dict_4x4, parameters=detector_params)

    if ids is None or target_id not in ids:
        return 1.0  # 找不到標籤則不縮放

    idx = np.where(ids == target_id)[0][0]
    marker_corners = corners[idx][0] # 4 個角點的 (u, v)

    # 取得這 4 個點在深度圖上的深度值
    h, w = final_metric_depth.shape
    depths = []
    for pt in marker_corners:
        u, v = int(np.clip(pt[0], 0, w-1)), int(np.clip(pt[1], 0, h-1))
        depths.append(final_metric_depth[v, u])
    
    # 將 2D 角點還原回 3D 空間點
    fx, fy = K_scaled[0, 0], K_scaled[1, 1]
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]
    pts_3d = []
    for i, pt in enumerate(marker_corners):
        z = depths[i]
        x = (pt[0] - cx) * z / fx
        y = (pt[1] - cy) * z / fy
        pts_3d.append(np.array([x, y, z]))

    # 計算點雲中標籤的平均邊長 (4 條邊取平均)
    side_lengths = []
    for i in range(4):
        side_lengths.append(np.linalg.norm(pts_3d[i] - pts_3d[(i+1)%4]))
    
    pcd_measured_size = np.mean(side_lengths)
    
    # 修正係數 = 真實尺寸 / 點雲測量尺寸
    scale_factor = marker_size_mm / (pcd_measured_size + 1e-6)
    
    print(f"   [Scale Check] 標籤真實尺寸: {marker_size_mm}mm, 點雲內測量: {pcd_measured_size:.2f}mm")
    print(f"   [Scale Check] 修正補償係數: {scale_factor:.4f}")
    
    return scale_factor
def main():
    print("正在載入模型...")
    providers = ['CPUExecutionProvider']
    lg_session = ort.InferenceSession("superpoint_lightglue_pipeline.onnx", providers=providers)
    da_session = ort.InferenceSession("depth_anything_v2_vits.onnx", providers=providers)

    # 載入資料夾內所有圖片
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))
    if len(image_paths) < 2:
        print(f"錯誤：在 {IMAGE_FOLDER} 找不到足夠的圖片。")
        return
    
    print(f"找到 {len(image_paths)} 張圖片，開始批次處理 (模式 {PIPELINE_MODE})...")
    
    K_orig, dist_coeffs = load_raw_camera_params('K.ini')
    processed_imgs_bgr = []
    processed_imgs_gray = []
    
    for path in image_paths:
        img_raw = cv2.imread(path)
        img_undist = cv2.undistort(img_raw, K_orig, dist_coeffs)
        
        orig_h, orig_w = img_raw.shape[:2]
        scale = TARGET_W / orig_w
        TARGET_H = int(orig_h * scale)
        
        img_resized = cv2.resize(img_undist, (TARGET_W, TARGET_H))
        processed_imgs_bgr.append(img_resized)
        processed_imgs_gray.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))

    K_scaled = K_orig.copy()
    K_scaled[0, 0] *= scale; K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale; K_scaled[1, 2] *= scale

    final_pcd = o3d.geometry.PointCloud()

    # ================= 模式 1: 匯集多圖特徵池，單圖修正 =================
    if PIPELINE_MODE == 1:
        ref_idx = min(REFERENCE_IMAGE_INDEX, len(processed_imgs_bgr) - 1)
        ref_bgr = processed_imgs_bgr[ref_idx]
        ref_gray = processed_imgs_gray[ref_idx]
        
        print(f"\n[模式 1] 產生基準圖 (Index {ref_idx}) 的 AI 相對深度...")
        time1 = time.perf_counter()
        ai_depth_map = run_depth_anything(da_session, ref_bgr, (TARGET_H, TARGET_W), save_path="depth_ref_visual.png")
        
        all_pts_2d, all_z_metric = [], []
        
        for i in range(len(processed_imgs_gray)):
            if i == ref_idx: continue
            print(f"-> 正在與圖片 {i} 進行 LightGlue 匹配與三角測距...")
            pts_2d, z_metric = compute_sparse_metric_points(ref_gray, processed_imgs_gray[i], K_scaled, lg_session, pair_index=i)
            if pts_2d is not None:
                all_pts_2d.append(pts_2d)
                all_z_metric.append(z_metric)
        
        if len(all_pts_2d) > 0:
            pooled_pts_2d = np.vstack(all_pts_2d)
            pooled_z_metric = np.concatenate(all_z_metric)
            print(f"總計收集到 {len(pooled_z_metric)} 個有效的深度參考點，開始全域 RANSAC 擬合...")
            
            final_pcd = align_and_generate_pcd(ai_depth_map, pooled_pts_2d, pooled_z_metric, K_scaled, ref_bgr, ref_gray, return_world_pcd=False)
            
            # 加上防護：如果點雲生成失敗，安全退出而不是直接閃退
            if final_pcd is None:
                print("❌ 錯誤：所有深度特徵點都被過濾器清空了，無法生成點雲！請確認拍攝距離。")
                return

            # 翻轉視角供顯示
            final_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # ================= 模式 2: 每張皆產生深度，全局 ArUco 融合 =================
    elif PIPELINE_MODE == 2:
        pcd_list = []
        for i in range(len(processed_imgs_gray) - 1):
            print(f"\n[模式 2] 處理圖片對 {i} 與 {i+1} ...")
            imgA_bgr, imgA_gray = processed_imgs_bgr[i], processed_imgs_gray[i]
            imgB_gray = processed_imgs_gray[i+1]
            
            ai_depth_map = run_depth_anything(da_session, imgA_bgr, (TARGET_H, TARGET_W) , save_path="depth_ref_visual.png")
            pts_2d, z_metric = compute_sparse_metric_points(imgA_gray, imgB_gray, K_scaled, lg_session, pair_index=i)
            
            if pts_2d is not None:
                pcd = align_and_generate_pcd(ai_depth_map, pts_2d, z_metric, K_scaled, imgA_bgr, imgA_gray, return_world_pcd=True)
                if pcd is not None:
                    pcd_list.append(pcd)
                    
        print(f"\n成功生成 {len(pcd_list)} 組局部點雲，開始 Voxel 融合...")
        for pcd in pcd_list:
            final_pcd += pcd
        
        # 進行體素下採樣 (1mm)，能完美消除不同視角的局部變形雜訊
        final_pcd = final_pcd.voxel_down_sample(voxel_size=1.0)
        final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)

    # ================= 匯出與顯示 =================
    o3d.io.write_point_cloud("wound_metric_model_multiview.xyz", final_pcd)
    o3d.io.write_point_cloud("wound_metric_model_multiview.ply", final_pcd)
    print("✅ 已成功匯出多視角對齊點雲: wound_metric_model_multiview.ply / .xyz")

    print("正在開啟 Open3D 視窗顯示點雲...")
    o3d.visualization.draw_geometries([final_pcd], window_name="Multi-View Metric 3D Model")

if __name__ == "__main__":
    main()