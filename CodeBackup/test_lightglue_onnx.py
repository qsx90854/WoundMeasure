import cv2
import numpy as np
import onnxruntime as ort
import time
import random
# 1. 初始化 Session
model_path = "superpoint_lightglue_pipeline.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# 2. 準備輸入數據 [2, 1, 1024, 1024]
# 這裡假設你有兩張圖 img0_gray, img1_gray (都是 1024x1024)
def get_input_data(img0, img1):
    # 轉為 float32 並正規化
    t0 = img0.astype(np.float32) / 255.0
    t1 = img1.astype(np.float32) / 255.0
    # 堆疊成 [2, 1, 1024, 1024]
    input_tensor = np.stack([t0, t1], axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=1)
    return input_tensor

# 讀取真實圖片測試
img0_raw = cv2.imread("glue_img4_1024.jpg", cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread("glue_img3_1024.jpg", cv2.IMREAD_GRAYSCALE)
img0_gray = cv2.resize(img0_raw, (1024, 1024))
img1_gray = cv2.resize(img1_raw, (1024, 1024))

input_data = get_input_data(img0_gray, img1_gray)

# 3. 執行推論
start_time = time.perf_counter()
# 根據截圖，輸出名稱分別是 'keypoints', 'matches', 'mscores'
outputs = session.run(['keypoints', 'matches', 'mscores'], {"images": input_data})
end_time = time.perf_counter()

kpts = outputs[0]     # Shape: [2, 1024, 2]
matches = outputs[1]  # Shape: [num_matches, 3]
scores = outputs[2]   # Shape: [num_matches]

print(f"推論耗時: {(end_time - start_time)*1000:.2f} ms")
print(f"找到配對數: {len(matches)}")

# 4. 視覺化配對結果
# 建立左右拼接的畫布
canvas = np.hstack((img0_gray, img1_gray))
canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# 遍歷所有配對並畫線
for i in range(len(matches)):
    # 取得配對索引 (注意：matches 的第 0 欄是 batch 資訊，通常忽略)
    idx0 = matches[i, 1]
    idx1 = matches[i, 2]
    score = scores[i]
    
    # 只畫出信心度較高的配對 (例如 > 0.5)
    if score > 0.5:
        # 取得坐標 (x, y)
        pt0 = kpts[0, idx0].astype(int)
        pt1 = kpts[1, idx1].astype(int)
        
        # 右圖坐標要在拼接畫布上向右偏移 1024 像素
        pt1_offset = (pt1[0] + 1024, pt1[1])
        

        # --- 步驟 2 & 3: 生成與使用隨機顏色 ---
        # 生成三個 0~255 之間的隨機整數，分別代表 Blue, Green, Red 通道
        # 我們稍微限制一下範圍 (e.g., 50~255) 可以避免生成太暗的顏色看不清楚
        random_color = (
            random.randint(50, 255),  # Blue
            random.randint(50, 255),  # Green
            random.randint(50, 255)   # Red
        )


        # 畫圓點與連線
        #color = (0, 255, 0) # 綠色
        cv2.circle(canvas, tuple(pt0), 3, random_color, -1)
        cv2.circle(canvas, pt1_offset, 3, random_color, -1)
        cv2.line(canvas, tuple(pt0), pt1_offset, random_color, 1, cv2.LINE_AA)

# 顯示結果
cv2.putText(canvas, f"Matches: {len(matches)}", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("lightglue_onnx_match_result.jpg", canvas)
canvas = cv2.resize(canvas, (900,450))
cv2.imshow("LightGlue Matches", canvas)
#cv2.waitKey(0) # 修正大寫 K
cv2.destroyAllWindows()


import open3d as o3d

print("\n--- 開始執行極線校正與 3D 密集重建 ---")

# 1. 整理 LightGlue 的有效配對點
valid_matches = []
for i in range(len(matches)):
    if scores[i] > 0.5:
        idx0 = matches[i, 1]
        idx1 = matches[i, 2]
        pt0 = kpts[0, idx0].astype(np.float32)
        pt1 = kpts[1, idx1].astype(np.float32)
        valid_matches.append((pt0, pt1))

pts0 = np.array([m[0] for m in valid_matches])
pts1 = np.array([m[1] for m in valid_matches])

# 2. 假設一組相機內參 (K) 與無畸變參數
# 如果你要做精確的傷口檢測，未來這裡必須換成你用棋盤格校正出來的真實 K 矩陣
K = np.array([[800, 0, 512], 
              [0, 800, 512], 
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)

#-先註解------------------------------------------------------------------------------------
# 3. 估計相機位姿 (R, t)
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, K, mask=mask)

#t = t * 15.0
print(f"R: {R}")
print(f"T: {t}")
print("相機位姿估計完成 (t 為相對平移向量)。")

# 4. 極線校正 (Stereo Rectification)
image_size = (1024, 1024)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K, dist_coeffs, K, dist_coeffs, image_size, R, t, flags=cv2.CALIB_ZERO_DISPARITY
)

# 計算映射表並扳正 (Warp) 影像
map1x, map1y = cv2.initUndistortRectifyMap(K, dist_coeffs, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K, dist_coeffs, R2, P2, image_size, cv2.CV_32FC1)

img0_rect = cv2.remap(img0_gray, map1x, map1y, cv2.INTER_LINEAR)
img1_rect = cv2.remap(img1_gray, map2x, map2y, cv2.INTER_LINEAR)
#為了讓 3D 點雲有顏色，我們也把彩色原圖扳正
img0_color_resized = cv2.resize(img0_raw, image_size) # img0_raw 來自你前面的 code
img0_color_rect = cv2.remap(img0_color_resized, map1x, map1y, cv2.INTER_LINEAR)
#-------------------------------------------------------------------------------------------------





# 後面的 SGBM (第 5 步) 照常執行
# 注意：使用 Uncalibrated 算出來的 3D 點雲 (第 6 步) 會產生形變，
# 它無法保持完美的物理比例，但你能明顯看出物件的「前後順序與形狀」了。




# 顯示扳正後的影像 (可選，確認水平是否對齊)
rect_canvas = np.hstack((img0_rect, img1_rect))
cv2.imwrite("rectified_images.jpg", rect_canvas)
print("影像扳正完成，已儲存為 rectified_images.jpg")

# 5. 執行 SGBM 計算視差圖 (Disparity Map)
window_size = 5
min_disp = 0
num_disp = 16 * 24 # 1024 解析度較大，搜尋範圍設為 192 (必須是 16 的倍數)

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 1 * window_size**2,
    P2=32 * 1 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print("正在計算 SGBM 視差圖...")
disparity = stereo.compute(img0_rect, img1_rect).astype(np.float32) / 16.0

# 6. 將視差圖轉為 3D 點雲 (利用 Q 矩陣)
print("正在將視差圖投影至 3D 空間...")
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# 7. 過濾無效的 3D 點與建立 Open3D 點雲物件
# SGBM 算不出來的地方視差會小於等於 0，深度會算錯，直接濾掉
mask_valid = disparity > 0
# 原本可能是 < 50.0，請把它改小，例如 10.0 或 5.0
# 你可以根據視窗中點雲的座標軸，觀察主體大概落在哪個 Z 值
Z_MAX_THRESHOLD = 4.0  # 限制最遠距離
Z_MIN_THRESHOLD = 0.01   # 限制最近距離 (避免相機鏡頭前的超近雜訊)

mask_z_max = points_3D[:, :, 2] < Z_MAX_THRESHOLD
mask_z_min = points_3D[:, :, 2] > Z_MIN_THRESHOLD

# 結合 SGBM 的有效 mask
final_mask = np.logical_and(mask_valid, np.logical_and(mask_z_max, mask_z_min))

valid_points = points_3D[final_mask]
valid_colors = cv2.cvtColor(img0_color_rect, cv2.COLOR_BGR2RGB)[final_mask] / 255.0
#final_mask = np.logical_and(mask_valid, mask_z_limit)

#valid_points = points_3D[final_mask]

# OpenCV 是 BGR，Open3D 吃 RGB
#valid_colors = cv2.cvtColor(img0_color_rect, cv2.COLOR_BGR2RGB)[final_mask] / 255.0

# 建立 Open3D 點雲
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)

# 簡單的點雲降噪 (Statistical Outlier Removal)  
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_clean = pcd.select_by_index(ind)

# 8. 開啟 3D 互動視窗
print("開啟 Open3D 視窗 (可使用滑鼠旋轉、縮放)...")
# 建立座標軸輔助 (紅色 X, 綠色 Y, 藍色 Z)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd_clean, coordinate_frame], 
                                  window_name="SGBM Dense Point Cloud",
                                  width=1024, height=768)