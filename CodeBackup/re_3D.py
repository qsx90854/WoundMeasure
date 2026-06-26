import cv2
import numpy as np
import onnxruntime as ort
import configparser
import open3d as o3d
from sklearn.linear_model import RANSACRegressor

# ================= 0. ArUco 姿態估計 (計算絕對物理尺度) =================
def get_aruco_pose(image_gray, K, marker_size_mm, target_id=12):
    """
    偵測指定的 ArUco 標籤，並計算相對於相機的旋轉矩陣 R 與平移向量 t
    """
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    
    # 相容不同版本的 OpenCV ArUco API
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dict_4x4, detector_params)
        corners, ids, _ = detector.detectMarkers(image_gray)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image_gray, dict_4x4, parameters=detector_params)

    if ids is None or target_id not in ids:
        return None, None

    # 找到目標 ID 的標籤
    idx = np.where(ids == target_id)[0][0]
    marker_corners = corners[idx]

    # 定義標籤在 3D 空間中的四個角點座標 (以標籤中心為原點)
    half_size = marker_size_mm / 2.0
    obj_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    # 透過 PnP 演算法算出標籤相對於相機的 3D 姿態
    success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners[0], K, np.zeros(5))
    if success:
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec
    return None, None

# ================= 1. 載入與處理相機參數 =================
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

# ================= 2. Depth Anything V2 前處理與推論 =================
def run_depth_anything(session, image_bgr, target_shape):
    input_size = 518
    img_resized = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    img_tensor = img_norm.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})
    raw_depth = outputs[0][0]
    
    depth_aligned = cv2.resize(raw_depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_aligned

# ================= 3. 主流程開始 =================
def main():
    lg_model_path = "superpoint_lightglue_pipeline.onnx"
    da_model_path = "depth_anything_v2_vits.onnx"
    
    print("正在載入模型...")
    providers = ['CPUExecutionProvider']
    lg_session = ort.InferenceSession(lg_model_path, providers=providers)
    da_session = ort.InferenceSession(da_model_path, providers=providers)

    img0_raw = cv2.imread("captured_images/image_7.png") # Frame A
    img1_raw = cv2.imread("captured_images/image_6.png") # Frame B
    K_orig, dist_coeffs = load_raw_camera_params('K.ini')
    orig_h, orig_w = img0_raw.shape[:2]

    # 去畸變
    img0_undist = cv2.undistort(img0_raw, K_orig, dist_coeffs)
    img1_undist = cv2.undistort(img1_raw, K_orig, dist_coeffs)

    TARGET_W = 1024
    scale = TARGET_W / orig_w
    TARGET_H = int(orig_h * scale)
    
    img0_resized = cv2.resize(img0_undist, (TARGET_W, TARGET_H))
    img1_resized = cv2.resize(img1_undist, (TARGET_W, TARGET_H))

    K_scaled = K_orig.copy()
    K_scaled[0, 0] *= scale; K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale; K_scaled[1, 2] *= scale

    print("\n--- 1. 產生 AI 稠密相對深度 (Depth Anything V2) ---")
    ai_depth_map = run_depth_anything(da_session, img0_resized, (TARGET_H, TARGET_W))


    print("\n--- [Debug] 匯出 AI 原始相對深度的 3D 點雲 ---")
    # 1. Depth Anything 輸出的是「逆深度」(數值越大代表越近)。
    # 為了能看出正確的形狀，我們先把它取倒數變成「相對深度 Z」
    # 加上 1e-6 避免除以 0 導致報錯
    relative_depth = 1.0 / (ai_depth_map + 1e-6)

    # 2. 為了讓匯出的點雲在 CloudCompare 中比較好觀看，不會因為數值太極端而找不到
    # 我們將它線性縮放到一個虛擬的相對範圍 (例如 100 ~ 1000)
    d_min, d_max = relative_depth.min(), relative_depth.max()
    relative_depth_norm = ((relative_depth - d_min) / (d_max - d_min + 1e-6)) * 900.0 + 100.0

    # 3. 反投影到 3D 空間
    fx_d, fy_d = K_scaled[0, 0], K_scaled[1, 1]
    cx_d, cy_d = K_scaled[0, 2], K_scaled[1, 2]
    u_d, v_d = np.meshgrid(np.arange(TARGET_W), np.arange(TARGET_H))

    X_rel = (u_d - cx_d) * relative_depth_norm / fx_d
    Y_rel = (v_d - cy_d) * relative_depth_norm / fy_d
    Z_rel = relative_depth_norm

    points_rel = np.stack((X_rel, Y_rel, Z_rel), axis=-1).reshape(-1, 3)
    colors_rel = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    pcd_rel = o3d.geometry.PointCloud()
    pcd_rel.points = o3d.utility.Vector3dVector(points_rel)
    pcd_rel.colors = o3d.utility.Vector3dVector(colors_rel)

    # 翻轉 180 度以適配外部軟體的視角
    pcd_rel.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.io.write_point_cloud("debug_raw_ai_depth.xyz", pcd_rel)
    o3d.io.write_point_cloud("debug_raw_ai_depth.ply", pcd_rel)
    print("✅ 已匯出 Debug 專用點雲: debug_raw_ai_depth.ply / .xyz\n")






    print("\n--- 2. 擷取特徵與姿態估計 (LightGlue) ---")
    img0_gray = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    t0 = img0_gray.astype(np.float32) / 255.0
    t1 = img1_gray.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.stack([t0, t1], axis=0), axis=1)
    
    outputs = lg_session.run(['keypoints', 'matches', 'mscores'], {"images": input_tensor})
    kpts, matches, scores = outputs[0], outputs[1], outputs[2]

    valid_matches = [(kpts[0, int(m[1])], kpts[1, int(m[2])]) for m, s in zip(matches, scores) if s > 0.5]
    pts0 = np.array([m[0] for m in valid_matches], dtype=np.float32)
    pts1 = np.array([m[1] for m in valid_matches], dtype=np.float32)
    print(f"找到 {len(pts0)} 組有效匹配點。")

    E, mask = cv2.findEssentialMat(pts0, pts1, K_scaled, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, K_scaled, mask=mask)

    print("\n--- 3. 透過 ArUco 計算真實基準線 (True Baseline) ---")
    # 【重點設定】：請用直尺測量你印出來的 ArUco 標籤的「黑色正方形邊長」，填入下方 (單位：毫米)
    ACTUAL_MARKER_SIZE_MM = 29.0 
    TARGET_MARKER_ID = 12
    
    R_A, t_A = get_aruco_pose(img0_gray, K_scaled, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)
    R_B, t_B = get_aruco_pose(img1_gray, K_scaled, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)

    if R_A is not None and R_B is not None:
        # 計算相機光心在真實世界 (ArUco座標系) 中的位置
        C_A = -R_A.T @ t_A
        C_B = -R_B.T @ t_B
        # 兩台相機之間的真實物理距離
        true_baseline_mm = np.linalg.norm(C_A - C_B)
        print(f"✅ 成功從 ArUco 計算出真實相機移動距離: {true_baseline_mm:.2f} mm")
    else:
        print(f"⚠️ 警告: 無法在兩張畫面中同時找到 ArUco 標籤 (ID:{TARGET_MARKER_ID})。退回預設值 50.0 mm")
        true_baseline_mm = 50.0

    # 將 LightGlue 算出的單位向量 t，乘上真實物理距離
    T_scaled = t * true_baseline_mm
    print(f"帶入真實物理尺度的平移向量 (mm):\n{T_scaled.flatten()}")

    # 建立投影矩陣
    P0 = np.float32(K_scaled @ np.hstack((np.eye(3), np.zeros((3, 1)))))
    P1 = np.float32(K_scaled @ np.hstack((R, T_scaled)))
    
    valid_mask = mask_pose.ravel() > 0 
    pts0_valid = np.float32(pts0[valid_mask]).T
    pts1_valid = np.float32(pts1[valid_mask]).T
    
    print(f"準備三角測量的點數量: {pts0_valid.shape[1]}")

    points_4D = cv2.triangulatePoints(P0, P1, pts0_valid, pts1_valid)
    points_3D = points_4D[:3, :] / points_4D[3, :] 
    Z_metric = points_3D[2, :] 

    print("\n--- 4. Sparse-to-Dense 深度對齊 (RANSAC 擬合) ---")
    good_z_mask = (Z_metric > 0) & (Z_metric < 5000) 
    
    pts0_fit = pts0_valid.T[good_z_mask]
    Z_metric_fit = Z_metric[good_z_mask]
    inv_Z_metric = (1.0 / Z_metric_fit).reshape(-1, 1)

    ai_samples = []
    for pt in pts0_fit:
        u, v = int(np.clip(pt[0], 0, TARGET_W-1)), int(np.clip(pt[1], 0, TARGET_H-1))
        ai_samples.append(ai_depth_map[v, u])
    ai_samples = np.array(ai_samples).reshape(-1, 1)

    ransac = RANSACRegressor(min_samples=int(len(ai_samples)*0.3), residual_threshold=0.001)
    ransac.fit(ai_samples, inv_Z_metric)
    scale_factor = ransac.estimator_.coef_[0][0]
    shift = ransac.estimator_.intercept_[0]
    
    inliers = np.sum(ransac.inlier_mask_)
    print(f"RANSAC 對齊完成! 參考點數量: {inliers}/{len(ai_samples)}")
    print(f"映射公式: 1/Z = ({scale_factor:.5e}) * AI_Depth + ({shift:.5e})")

    dense_inv_metric_map = scale_factor * ai_depth_map + shift
    dense_inv_metric_map = np.clip(dense_inv_metric_map, 1e-6, None) 
    
    final_metric_depth = 1.0 / dense_inv_metric_map

    depth_vis = cv2.normalize(final_metric_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("aligned_metric_depth.jpg", cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA))

    print("\n--- 5. 生成帶有物理尺度的 3D 點雲 ---")
    fx, fy = K_scaled[0, 0], K_scaled[1, 1]
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]

    u, v = np.meshgrid(np.arange(TARGET_W), np.arange(TARGET_H))
    X = (u - cx) * final_metric_depth / fx
    Y = (v - cy) * final_metric_depth / fy
    Z = final_metric_depth
    
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    dist_mask = (points[:, 2] < 1500) & (points[:, 2] > 0)
    valid_points = points[dist_mask]
    valid_colors = colors[dist_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

    print("正在清理點雲雜訊...")
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)

    transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcd_clean.transform(transform)


    # ================= 新增：匯出 3D 點雲檔案 =================
    
    # 1. 存成標準的 .xyz 檔 (只包含 X, Y, Z 座標)
    xyz_filename = "wound_metric_model.xyz"
    o3d.io.write_point_cloud(xyz_filename, pcd_clean)
    print(f"✅ 已成功匯出純座標點雲: {xyz_filename}")

    # 2. (強烈推薦) 存成 .ply 檔 (包含 X, Y, Z 以及 RGB 顏色)
    # 大多數測量軟體 (如 CloudCompare) 都能讀取 .ply，有顏色在標記傷口邊緣時會容易非常多！
    ply_filename = "wound_metric_model.ply"
    o3d.io.write_point_cloud(ply_filename, pcd_clean)
    print(f"✅ 已成功匯出帶顏色點雲: {ply_filename}")


    print("正在開啟 Open3D 視窗顯示點雲...")
    o3d.visualization.draw_geometries([pcd_clean], window_name="AI-Metric Fused 3D Point Cloud")

if __name__ == "__main__":
    main()