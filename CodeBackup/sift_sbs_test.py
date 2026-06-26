import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # 檢查輸入檔案
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # 預設尋找常見的檔名
        # for f in ['capture.png', 'test.png', 'sbs.png']:
        #     if os.path.exists(f):
        #         img_path = f
        #         break
        img_path = "su01_test_images_1/image_0.png"
        #else:
        #    print("❌ 找不到輸入圖片！請提供路徑，例如：python sift_sbs_test.py my_image.png")
        #    return

    print(f"📂 正在載入圖片: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 無法讀取圖片資料")
        return

    h, w = img.shape[:2]
    # 假設是 SBS 格式，左右平分
    imgL = img[:, :w//2]
    imgR = img[:, w//2:]

    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # --- 改善建議 1: 使用 CLAHE 增強局部對比度 ---
    print("✨ 正在套用 CLAHE 影像增強...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgL_gray = clahe.apply(imgL_gray)
    imgR_gray = clahe.apply(imgR_gray)

    print("🔍 正在進行 SIFT 特徵提取 (增強密度模式)...")
    # 使用之前優化過的參數
    sift = cv2.SIFT_create(contrastThreshold=0.005, edgeThreshold=15)
    
    kp1, des1 = sift.detectAndCompute(imgL_gray, None)
    kp2, des2 = sift.detectAndCompute(imgR_gray, None)

    print(f"📈 偵測結果: 左圖 {len(kp1)} 點, 右圖 {len(kp2)} 點")

    if des1 is None or des2 is None:
        print("❌ 找不到足夠的描述子進行匹配")
        return

    # --- 改善建議 2: 保持嚴謹的 Ratio Test 門檻 (0.7) ---
    ratio_threshold = 0.7
    print(f"🤝 正在進行特徵匹配 (Ratio Test = {ratio_threshold})...")
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_threshold = 0.85
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    # --- 改善建議 3: 使用 RANSAC 過濾誤匹配 (避免亂七八糟的連線) ---
    if len(good_matches) > 10:
        print("🛡️ 正在執行 RANSAC 幾何驗證...")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用 RANSAC 找出符合大多數點規律的基礎矩陣 F
        # 參數 1.0 是容許的像素誤差距離
        _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is not None:
            matches_mask = mask.ravel().tolist()
            final_matches = [m for i, m in enumerate(good_matches) if matches_mask[i]]
            print(f"✅ RANSAC 過濾完成: {len(good_matches)} -> {len(final_matches)} 點 (剔除 {len(good_matches)-len(final_matches)} 點)")
            good_matches = final_matches
    else:
        print("⚠️ 匹配點太少，跳過 RANSAC 驗證")

    print(f"✅ 最終有效匹配數: {len(good_matches)}")

    # 繪製結果
    vis_img = cv2.drawMatches(imgL, kp1, imgR, kp2, good_matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 8))
    plt.imshow(vis_img)
    plt.title(f"SIFT Matching Test: {img_path}\nMatches: {len(good_matches)} (L:{len(kp1)}, R:{len(kp2)})")
    plt.axis("off")
    plt.tight_layout()
    print("🖥️ 顯示結果視窗...")
    plt.show()

if __name__ == "__main__":
    main()
