import cv2
import numpy as np
import onnxruntime as ort
import time
import random
import configparser
import open3d as o3d

# ================= 1. 載入原始 K.ini (不進行縮放) =================
def load_raw_camera_params(ini_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    # 讀取內參
    fx = float(config['intrinsic1']['fx'])
    fy = float(config['intrinsic1']['fy'])
    cx = float(config['intrinsic1']['cx'])
    cy = float(config['intrinsic1']['cy'])
    
    # 讀取畸變參數 (OpenCV 標準順序: k1, k2, p1, p2, k3)
    k1 = float(config['distortion1']['k1'])
    k2 = float(config['distortion1']['k2'])
    k3 = float(config['distortion1']['k3'])
    p1 = float(config['distortion1']['p1'])
    p2 = float(config['distortion1']['p2'])
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K, dist_coeffs

# ================= 2. 初始化模型與資料前處理 =================
model_path = "superpoint_lightglue_pipeline.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def get_input_data(img0, img1):
    t0 = img0.astype(np.float32) / 255.0
    t1 = img1.astype(np.float32) / 255.0
    input_tensor = np.stack([t0, t1], axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=1)
    return input_tensor

# 讀取原始圖片
img0_raw = cv2.imread("captured_images/image_1.png") 
img1_raw = cv2.imread("captured_images/image_0.png")

if img0_raw is None or img1_raw is None:
    print("錯誤：無法讀取圖片，請確認路徑。")
    exit()

orig_h, orig_w = img0_raw.shape[:2]

# 載入原始相機參數
K_orig, dist_coeffs = load_raw_camera_params('K.ini')

# ================= 3. 先去畸變 (Undistort) =================
# 確保後續所有幾何運算都在完美的針孔相機模型下進行
img0_undist = cv2.undistort(img0_raw, K_orig, dist_coeffs)
img1_undist = cv2.undistort(img1_raw, K_orig, dist_coeffs)

# ================= 4. 等比例縮放 (Proportional Resize) =================
TARGET_W = 1024
# 計算等比例的縮放係數
scale = TARGET_W / orig_w
TARGET_H = int(orig_h * scale)

# 進行等比例縮放
img0_resized = cv2.resize(img0_undist, (TARGET_W, TARGET_H))
img1_resized = cv2.resize(img1_undist, (TARGET_W, TARGET_H))

# 根據縮放係數調整新的 K 矩陣
K_scaled = K_orig.copy()
K_scaled[0, 0] *= scale  # fx
K_scaled[1, 1] *= scale  # fy
K_scaled[0, 2] *= scale  # cx
K_scaled[1, 2] *= scale  # cy

print(f"等比例縮放後的尺寸: {TARGET_W}x{TARGET_H}")
print(f"縮放並去畸變後的 K 矩陣:\n{K_scaled}")

# 轉灰階供 LightGlue 使用
img0_gray = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)

# ================= 5. LightGlue 推論 =================
input_data = get_input_data(img0_gray, img1_gray)
outputs = session.run(['keypoints', 'matches', 'mscores'], {"images": input_data})

kpts, matches, scores = outputs[0], outputs[1], outputs[2]

# ================= 6. 視覺化 LightGlue 匹配結果 =================
print("正在生成特徵匹配結果圖...")

# 建立左右拼接的彩色畫布
match_canvas = np.hstack((img0_resized, img1_resized))

match_count = 0
for i in range(len(matches)):
    score = scores[i]
    if score > 0.5:  # 只顯示信心度高於 0.5 的配對
        idx0, idx1 = int(matches[i, 1]), int(matches[i, 2])
        
        # 取得坐標
        pt0 = tuple(kpts[0, idx0].astype(int))
        pt1 = tuple(kpts[1, idx1].astype(int))
        
        # 右圖坐標需加上偏移量 (TARGET_W)
        pt1_offset = (pt1[0] + TARGET_W, pt1[1])
        
        # 隨機顏色
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # 畫點與線
        cv2.circle(match_canvas, pt0, 4, color, -1)
        cv2.circle(match_canvas, pt1_offset, 4, color, -1)
        cv2.line(match_canvas, pt0, pt1_offset, color, 1, cv2.LINE_AA)
        match_count += 1

# 在圖片左上角標註配對數量
cv2.putText(match_canvas, f"Valid Matches: {match_count}", (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imwrite("lightglue_matches.jpg", match_canvas)
print(f"匹配結果圖已儲存為 lightglue_matches.jpg (共 {match_count} 組配對)")

# 縮放顯示 (可選)
cv2.imshow("LightGlue Feature Matches", cv2.resize(match_canvas, (1024, int(512 * (TARGET_H/TARGET_W)))))
cv2.waitKey(1)

# ================= 7. 幾何重建與極線校正 =================
print("\n--- 開始執行極線校正與 3D 密集重建 ---")

valid_matches = []
for i in range(len(matches)):
    if scores[i] > 0.5:
        idx0, idx1 = matches[i, 1], matches[i, 2]
        valid_matches.append((kpts[0, idx0], kpts[1, idx1]))

pts0 = np.array([m[0] for m in valid_matches], dtype=np.float32)
pts1 = np.array([m[1] for m in valid_matches], dtype=np.float32)

# 估計位姿 (使用處理好的 K_scaled，輸入的點也是無畸變的)
E, mask = cv2.findEssentialMat(pts0, pts1, K_scaled, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, K_scaled, mask=mask)

print(f"估計的 R 向量: {R}")
print(f"估計的 T 向量: {t.flatten()}")


# 極線校正
# 注意：因為影像在前面已經去過畸變，所以這裡的 dist_coeffs 傳入 np.zeros(5)
image_size = (TARGET_W, TARGET_H)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_scaled, np.zeros(5), K_scaled, np.zeros(5), image_size, R, t, flags=cv2.CALIB_ZERO_DISPARITY
)

# 生成 Remap 表
map1x, map1y = cv2.initUndistortRectifyMap(K_scaled, np.zeros(5), R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K_scaled, np.zeros(5), R2, P2, image_size, cv2.CV_32FC1)

# 映射灰階圖供 SGBM 使用
img0_rect = cv2.remap(img0_gray, map1x, map1y, cv2.INTER_LINEAR)
img1_rect = cv2.remap(img1_gray, map2x, map2y, cv2.INTER_LINEAR)

# 映射彩色圖供 Open3D 上色使用
img0_color_rect = cv2.remap(img0_resized, map1x, map1y, cv2.INTER_LINEAR)

# ================= 8. 儲存與對齊檢查 =================
cv2.imwrite("rectified_L.jpg", img0_rect)
cv2.imwrite("rectified_R.jpg", img1_rect)

# 建立一個水平拼接的視覺化圖，並畫上水平紅線檢查對齊
canvas_rect = np.hstack((img0_rect, img1_rect))
canvas_rect = cv2.cvtColor(canvas_rect, cv2.COLOR_GRAY2BGR)

# 每隔 40 像素畫一條紅線
for y in range(0, canvas_rect.shape[0], 40):
    cv2.line(canvas_rect, (0, y), (canvas_rect.shape[1], y), (0, 0, 255), 1)

cv2.imwrite("rectified_with_lines.jpg", canvas_rect)
print("對齊後的影像已儲存為 rectified_L.jpg, rectified_R.jpg 與 rectified_with_lines.jpg")

cv2.imshow("Alignment Check (Epipolar Lines)", cv2.resize(canvas_rect, (1024, int(512 * (TARGET_H/TARGET_W)))))
cv2.waitKey(1) 

# ================= 9. SGBM 與 3D 視覺化 =================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,      # 根據你的 baseline 估算
    blockSize=9,
    P1=8 * 3 * 9**2,        # = 1944
    P2=32 * 3 * 9**2,       # = 7776
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(img0_rect, img1_rect).astype(np.float32) / 16.0


# 1. 處理無效的視差值 (通常小於 minDisparity 的就是無效值)
# 將無效值設為 0，方便視覺化
disp_vis = disparity.copy()
min_disp = 0  # 你的 minDisparity 設定
disp_vis[disp_vis < min_disp] = min_disp

# 2. 將數值正規化到 0~255 供顯示
cv2.normalize(disp_vis, disp_vis, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)

# 3. 套用偽色彩 (Color Map) 讓深度更清楚 (紅色近、藍色遠)
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

display_w = 1024
display_h = int(TARGET_H * display_w / TARGET_W)  # 按實際比例算
cv2.imshow("Disparity Map", cv2.resize(disp_color, (display_w, display_h)))
cv2.waitKey(0)  # 卡住程式，直到你按任意鍵
# ----------------------------------



points_3D = cv2.reprojectImageTo3D(disparity, Q)

# 過濾點雲
mask_valid = (disparity > 1) & (disparity < 20) & (points_3D[:, :, 2] < 100.0) # 限制距離
valid_points = points_3D[mask_valid]
valid_colors = cv2.cvtColor(img0_color_rect, cv2.COLOR_BGR2RGB)[mask_valid] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)

# 移除統計噪訊
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

print("正在開啟 Open3D 視窗顯示點雲...")
o3d.visualization.draw_geometries([pcd_clean], window_name="Corrected 3D Reconstruction")