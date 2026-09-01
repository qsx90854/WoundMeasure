# Grad-SIFT/ORB Debug 對話交接

更新日期：2026-09-01  
工作目錄：`D:\Lightglue\lg_env`

## 使用者目標與工作原則

- 使用者正在分析「點擊左圖 A 後，Grad-SIFT/ORB 為什麼會匹配錯」，目前優先是觀察與定位原因，不希望先假設原因後直接改演算法。
- Debug UI 功能只加入：
  - `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py`
- 不要把 Debug UI 加到原版 Zebra、`zebra_0825v2.py` 或共用 matcher。
- 回覆請使用繁體中文。
- 工作區非常髒，含大量使用者修改、未追蹤檔案及既有刪除；不要 reset、restore 或整理不相關檔案。
- `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py` 目前本身是未追蹤檔案，必須保留。

## 主要檔案

- Debug 主程式：
  - `D:\Lightglue\lg_env\depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py`
- 正式 Grad-SIFT/ORB matcher：
  - `D:\Lightglue\lg_env\Algorithm\stereo_matching.py`
- 原版：
  - `D:\Lightglue\lg_env\depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py`
- 0825 版本：
  - `D:\Lightglue\lg_env\zebra_0825v2.py`

## Debug 版本目前重要預設

位於 Debug 主程式頂端：

- `ENABLE_WOUND_AI = False`
  - 不載入傷口 AI 模型，也不執行推論。
- `ENABLE_ECC_REFINEMENT_DEFAULT = False`
  - 點擊匹配後 ECC 預設關閉，方便確認原始匹配偏差。
- `ENABLE_IMPROVED_MATCHING_DEFAULT = False`
  - 預設走原本的 Grad-SIFT/ORB 流程。
- `view_state['use_hamming'] = True`
  - 預設實際 descriptor 是 ORB/Hamming；畫面或舊 log 的「Grad-SIFT」有時只是流程通稱。
- `Reject SpecPts` 預設開啟。

## 已完成的 Debug UI

### 1. 四圖版面

目前為 2×2：

- 上左：原始左圖
- 上右：原始右圖
- 下左：實際 matcher 使用的左灰階 ROI
- 下右：實際 matcher 使用的右灰階 ROI

目前整體大小：

```python
figsize=(14, 10)
```

按鈕已壓縮到頂端：

```python
control_row_y = (0.932, 0.900, 0.868, 0.836, 0.804, 0.772)
control_h = 0.026
```

### 2. Debug 說明文字已移出圖片

左右 Debug 說明現在各自位於圖片下方的獨立資訊框：

- `ax_debug_info_A`
- `ax_debug_info_B`

版面保留左右間距，文字限制在各自資訊框內，不再直接覆蓋 ROI。右圖隱藏時，右資訊框也一起隱藏。

### 3. Debug 圖例

下方 Debug 圖：

- 淺藍小點：所有 High 原始候選
- 淺橘小點：所有 Mid 原始候選
- 候選密集重疊時可能看起來像深色實心點；不代表成功
- 藍色空心大圓環：最後保留的 High 支撐點
- 橘色空心大圓環：最後保留的 Mid 支撐點
- 藍／橘跨圖連線與相同編號：真正成對的最終支撐點
- 白色 `x`：左圖點擊 A
- 紫色 `+`：A 的 RT/plane seed
- 白色菱形：原始 A′
- 綠色 `x`：後處理後最終 A′
- 紫色小圓：Homography 預測位置

Descriptor Audit：

- 黃色星星：目前選取的左參考點
- 綠色圓圈：Global KNN Top-1
- 紅色方框：Global KNN Top-2
- 紫色 `x`：該參考點的 local seed

點下方左圖的候選點，只會切換 Audit 選點，不會重新執行量測或改變 matcher。

## Descriptor Audit 已完成功能

入口：

- `build_grad_descriptor_audit(...)`
- `select_grad_descriptor_audit(...)`
- `update_grad_match_debug_views(...)`

Audit 會依點擊當下 `snap_view_state`，重算與正式 matcher 相同的 Global KNN：

1. High 只配 High，Mid 只配 Mid
2. Absolute distance
3. Lowe ratio
4. Mutual nearest-neighbor

支援：

- ORB/Hamming
- Gray-SIFT/L2
- RGB-SIFT/L2
- Opponent-SIFT/L2

每個點保存並顯示：

- `d1`
- `d2`
- `d1/d2`
- `d2-d1`
- Top-1／Top-2 右座標
- Top-1 與 Top-2 的影像空間距離
- distance／ratio／mutual 判定
- global Top-1 的 epi 與 seed 誤差（若前面已失敗，標成 diagnostic only）

Global 門檻：

- ORB/Hamming absolute：`d1 < 100`
- Gray-SIFT absolute：`d1 < 450`
- RGB/Opponent-SIFT absolute：`d1 < 780`
- Lowe ratio：`d1/d2 < 0.78`（嚴格小於）

Console 會印每組 ratio min／median／p90，以及低於 0.78／0.85／0.90／0.95 的數量；`d2=0` 另行統計。

## Guided fallback 與救回追蹤

正式 matcher 目前常數：

- `GRAD_SIFT_MIN_GROUP_INLIERS = 3`
- `GRAD_SIFT_GUIDED_RADIUS_PX = 10`
- `GRAD_SIFT_GUIDED_RATIO_TEST = 0.95`
- `GRAD_SIFT_EPIPOLAR_TOL_PX = 3`
- `GRAD_SIFT_MAX_RT_ADJUST_PX = 40`
- `GRAD_SIFT_OFFSET_MEDIAN_TOL_PX = 8`
- `GRAD_SIFT_RANSAC_REPROJ_PX = 2.5`

觸發條件：

- High／Mid 各自做完 Global distance→ratio→mutual 後，若整組 `good < 3`，才啟動 guided fallback。
- 不是某一個點 ratio 失敗就單獨啟動。

Guided 流程：

1. 左參考點距離點擊 A 必須小於 50 px
2. `local_seed = A 的 plane/RT seed + (pL - A)`
3. 右候選限制在 local seed 10 px 內
4. 若有 F，再限制極線誤差 ≤3 px
5. 在縮小後候選集合重新計算 descriptor 距離
6. absolute distance 同原模式
7. guided ratio 使用 0.95
8. 多個左點搶同一右點時保留 descriptor distance 較小者
9. 後續仍需通過 epi、seed、optional color、offset median、RANSAC 與 final RT bound

### 支撐點來源 log

Audit 會把每個左參考點與正式 matcher 最後回傳的 `g_ptsA/g_ptsB/g_groups` 比對：

- `support=NOT_USED`
  - 沒有參與最終 Grad-SIFT 內插。
- `support=GLOBAL_KNN`
  - 通過 Global KNN，且最後仍是內插支撐點。
- `support=GUIDED_RESCUE`
  - Global decision 是 DIST／RATIO／MUTUAL，但最後出現在正式支撐點中；依目前 matcher 邏輯，確定是 guided fallback 救回並通過後續幾何/RANSAC。

救回時會額外印：

```text
[Descriptor Audit GUIDED_RESCUE]
#173 MID L=(1209.0,561.0)
global=RATIO but final_support=YES
guided_R=(1255.0,548.0)
this point participated in Grad-SIFT interpolation
```

每組也會印：

```text
[Descriptor Audit support HIGH] final=..., global=..., guided_rescue=...
[Descriptor Audit support MID]  final=..., global=..., guided_rescue=...
```

## 最近確認過的案例

```text
[Descriptor Audit select] #173 MID
L=(1209,561)
Top1=(1255,548)
d1=83.403
d2=85.428
ratio=0.9763
decision=RATIO
support=GUIDED_RESCUE
```

解讀：

- Global 全候選 ratio 0.9763，未通過 0.78。
- Guided 先以 seed 10 px／epi 3 px 縮小候選集合。
- Global Top-2 很可能被幾何條件排除；Top-1 仍是 (1255,548)，但 guided 的新 Top-2 較遠或只剩單一候選，因此通過 guided 0.95。
- 該點後續也通過 offset/RANSAC 等檢查，因此確實參與內插。

## 已知限制／下一步可能需求

1. 目前畫面顯示的 `d1/d2` 是 Global KNN。
2. `GUIDED_RESCUE` 會顯示正式採用的 `guided_R`，但尚未印 guided 子集合自己的：
   - guided d1
   - guided d2
   - guided d1/d2
   - guided 候選數量
3. 如果下一步需要查「為什麼 guided 能救回」，最有價值的增強就是補上述 guided-specific audit；必須明確標示與 Global KNN 是不同候選集合。
4. Debug 資訊框目前使用固定高度與 `clip_on=True`；若再增加很多行文字，應增加資訊框高度、縮短文字，或加入頁籤/切換，不要再覆蓋影像。
5. 「過濾高光反光」只是排除左右局部灰階最亮 20% 並以 3×3 膨脹，主要影響 Grad 候選選取；它不是高光修復。
6. 「進階高光過濾」使用 HSV；若與簡易版同時開啟，進階版優先。
7. `Reject SpecPts` 是另一套預先計算的空間／時間高光遮罩，預設開啟。
8. `filter_specular*` 只作用於原本 Grad-SIFT/ORB 路徑；開啟 Improved matching 時不走這段。

## 驗證狀態

最新修改後已通過：

```powershell
python -m py_compile depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py
```

尚未在此對話內完整操作實際 Tk UI 做視覺驗收；使用者有在本機實際執行並提供 log。

## 下一個對話建議起始訊息

```text
請先閱讀 D:\Lightglue\lg_env\GRADSIFT_DEBUG_HANDOFF.md，
延續 depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py
的 Grad-SIFT/ORB 對位失準診斷。只修改 Debug 版本，保留工作區其他既有變更。
```
