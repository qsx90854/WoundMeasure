# 修改記錄：depth_measure_multi_aruco_sbs.py
**時間**：2026-05-07 09:15  
**修改原因**：使用者要求「看原始 SIFT 連線」按鈕顯示的圖片移除特徵點連線

## 修改內容

### 第 374–376 行（舊）→ 第 374–379 行（新）

**刪除（-）**：
`
Line 374: # 預畫 SIFT 全局連線圖備用
Line 375: vis_sift_img = cv2.drawMatches(imgA_gray, kpA_glob, imgB_gray, kpB_glob, good_matches_glob, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
Line 376: vis_sift_img = cv2.cvtColor(vis_sift_img, cv2.COLOR_BGR2RGB)
`

**新增（+）**：
`
Line 374: # 預畫 SIFT 全局特徵點圖備用 (只畫點，不畫連線)
Line 375: _vis_sift_A = cv2.drawKeypoints(imgA_gray, kpA_glob, None, color=(0, 255, 0),
Line 376:                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
Line 377: _vis_sift_B = cv2.drawKeypoints(imgB_gray, kpB_glob, None, color=(0, 255, 0),
Line 378:                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
Line 379: vis_sift_img = cv2.cvtColor(np.hstack([_vis_sift_A, _vis_sift_B]), cv2.COLOR_BGR2RGB)
`

## 說明
原本使用 cv2.drawMatches 同時繪製特徵點與匹配連線；  
改為使用 cv2.drawKeypoints 分別在左右圖上只繪製特徵圓圈（含尺度與方向的 rich keypoints），再水平拼接。
