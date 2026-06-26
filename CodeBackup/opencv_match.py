import cv2
import numpy as np
import time
import random

# 1. 讀取圖片並轉為灰階
img1 = cv2.imread('captured_images_m/image_0.png')
img2 = cv2.imread('captured_images_m/image_1.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. 初始化 SIFT 偵測器
# 在較新版本的 OpenCV 中，SIFT 已經專利過期並內建在主庫中
sift = cv2.SIFT_create()

# 3. 偵測關鍵點 (Keypoints) 並計算描述子 (Descriptors)
start_time = time.perf_counter()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
end_time = time.perf_counter()

print(f"SIFT 提取耗時: {(end_time - start_time)*1000:.2f} ms")
print(f"圖1 特徵點數: {len(kp1)}, 圖2 特徵點數: {len(kp2)}")

# 4. 初始化 FLANN 匹配器 (比 Brute-Force 更快，適合 SIFT)
_time1 = time.perf_counter()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # 檢查次數，越高越準但越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
_time2 = time.perf_counter()
# 5. 進行匹配 (k=2 代表對每個點找兩個最近鄰，用於 Ratio Test)
matches = flann.knnMatch(des1, des2, k=2)
_time3 = time.perf_counter()
# 6. 進行 Lowe's Ratio Test (濾除錯誤配對)
# 這是 SIFT 的標準作法：如果最近的距離比次近的距離小很多，才視為好配對

print(f"初始化 FLANN耗時: {(_time2 - _time1)*1000:.2f} ms")
print(f"knnMatch耗時: {(_time3 - _time2)*1000:.2f} ms")


good_matches = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)

print(f"過濾後的優質配對數: {len(good_matches)}")

# 7. 手動繪製隨機顏色的連線 (仿照你之前的 LightGlue 邏輯)
# 建立拼接畫布
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
canvas[:h1, :w1] = img1
canvas[:h2, w1:w1+w2] = img2

for m in good_matches:
    # 取得點的坐標
    pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
    pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
    
    # 右圖坐標偏移
    pt2_offset = (pt2[0] + w1, pt2[1])
    
    # 生成隨機顏色
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    # 畫線與圓點
    cv2.line(canvas, pt1, pt2_offset, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, pt1, 3, color, -1)
    cv2.circle(canvas, pt2_offset, 3, color, -1)

# 8. 顯示結果
cv2.imshow('SIFT Matching with Random Colors', canvas)
cv2.imwrite("opencv_match_result.png", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()