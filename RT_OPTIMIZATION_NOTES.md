# RT Optimization Notes

本檔用來記錄 ArUco、全畫面特徵匹配、影格配對選擇及相對姿態 `R/t` 計算流程的修改。

## 維護規則

- 相關程式每次有行為變更時，都在本檔最上方新增一筆日期紀錄。
- 紀錄需包含：修改目的、實際改動、參數變更、驗證方式、結果與已知限制。
- 實驗數據需註明是幾何一致性測試或 ground-truth 深度測試，不可混為一談。
- 若修改會影響既有輸出欄位、配對選擇或失敗條件，必須明確記錄相容性影響。

---

## 2026-07-11 - ArUco 與全畫面特徵融合 RT 優化

### 修改目的

原本在畫面有兩個以上 ArUco pattern 時，主要依賴 joint PnP 的 8 個角點計算姿態。當 pattern 很小、集中在物件同一側、角點有微小誤差，或 marker map 本身不準時，可能出現以下問題：

- Marker 重投影誤差很小，但全畫面特徵不符合該 RT。
- 兩個 marker 並沒有提供足夠廣的空間分布，RT 仍可能不穩定。
- 2-pattern 模式的 baseline 完全沿用 joint PnP，沒有使用已知 marker 邊長重新驗證尺度。
- Essential matrix 的結果只看極線誤差，沒有充分檢查內點比例、空間覆蓋與退化情況。
- Feature refinement 失敗後，次佳影格仍可能使用不可靠的 marker RT 參與深度融合。

本次目標是讓 1、2 或多個 pattern 都使用 pattern 外的全畫面特徵驗證與優化 RT，並讓 marker 主要提供公制尺度與幾何交叉檢查。

### 影響檔案

- `Algorithm/video_pose_analysis.py`
- `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py` 與 `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_circle.py` 共用上述模組，因此會自動使用新版流程。

### 實作內容

#### 1. 建立獨立於 marker 的特徵證據

- SIFT 偵測時遮蔽 ArUco 區域及外圍 margin，避免大量匹配其實來自 marker 圖案本身。
- SIFT mutual matching 與 ratio test 後，使用影像網格限制每格保留數量。
- 避免高紋理小區域壟斷 Essential matrix，鼓勵匹配分布到較大的畫面範圍。

#### 2. Essential matrix 品質檢查

每一個候選影格對除了 marker 指標，也會計算：

- 匹配總數與 Essential/recoverPose 內點數。
- 內點比例。
- 左右影像的 convex-hull 覆蓋率。
- 6 x 4 網格覆蓋率。
- 去除旋轉後的中位視差角。
- Homography 支持率及低視差平面退化判斷。
- Feature RT 與 marker RT 的旋轉差。
- Feature Essential 模型極線殘差，以及 marker RT 對全畫面特徵的極線殘差。

只有內點、覆蓋與視差符合條件的 feature geometry 才能進入 RT refinement。強特徵解可在有限角度內修正錯誤 marker 解。

#### 3. 候選影格重新排序

候選影格不再只依 marker 重投影誤差排序，新的 combined score 同時考慮：

- Marker pair quality。
- Marker RT 對全畫面特徵的極線誤差。
- Essential 模型自身的極線誤差。
- Feature 內點、空間覆蓋、視差與退化懲罰。
- Feature RT 與 marker RT 的旋轉一致性。

只有 strong feature geometry 才允許提早結束漸進式抽樣；普通合格結果會繼續擴大搜尋。

#### 4. 1/2/多 pattern 共用 feature refinement

- 所有 marker 數量都使用 SIFT + Essential matrix 求 `R` 與 `t` 方向。
- 單 marker 仍保留 IPPE 多分支列舉，利用 feature geometry 選擇較合理的分支。
- 多 marker joint PnP 改為 marker prior，而不是最終 RT 的唯一來源。
- 當全畫面特徵證據夠強時，允許 marker 角點極線誤差有限度退步，避免 4 或 8 個小區域角點否決大量獨立特徵。

#### 5. 所有 marker 數量都重新驗證 baseline

- 使用 unit translation 三角化每個共享 marker 的四個角點。
- 由已知 `marker_size_mm` 反推每個 marker 的 baseline。
- 檢查三角化點的左右相機 cheirality。
- 檢查單一 marker 四條邊的變形程度。
- 多 marker 時檢查各 marker 尺度的 relative MAD。
- 尺度一致且位於 baseline 合法範圍內時，以中位尺度取代原 joint PnP baseline。
- 尺度不一致時保留 marker PnP baseline，並輸出明確診斷訊息。

#### 6. 次佳候選最終驗證

- 每個次佳候選完成 refinement 後，再重新計算最終全畫面特徵極線殘差。
- 最終殘差超過 `PAIR_EPI_EXTRA_PX` 的候選直接排除，不參與多影格深度融合。
- 避免 feature refinement 被拒絕後，原本不可靠的 marker RT 仍被當成次佳參考。

#### 7. 新增診斷輸出

`analyze_video_frames()` 回傳資料新增：

- `marker_reproj_err`：marker PnP 重投影誤差。
- `rt_quality`：完整 feature/marker 品質指標。
- `rt_quality.final_feature_epi_px`：最終 RT 對全畫面特徵的極線殘差。
- `rt_quality.rt_reliable`：最終 RT 是否通過目前的 feature 品質與殘差門檻。

原本的 `min_reproj_err` 在有特徵驗證時改為最終全畫面特徵極線殘差，讓 UI 品質顯示不再只依賴 marker 自我擬合誤差。

### 主要新增門檻

目前主要預設值位於 `Algorithm/video_pose_analysis.py`：

- `FEATURE_MARKER_MASK_MARGIN_PX = 8.0`
- `FEATURE_GRID_COLS = 6`
- `FEATURE_GRID_ROWS = 4`
- `FEATURE_MIN_INLIER_RATIO = 0.30`
- `FEATURE_MIN_GRID_COVERAGE = 0.17`
- `FEATURE_MIN_HULL_COVERAGE = 0.02`
- `FEATURE_MIN_PARALLAX_DEG = 0.05`
- `FEATURE_STRONG_INLIERS = 40`
- `FEATURE_STRONG_INLIER_RATIO = 0.35`
- `FEATURE_STRONG_GRID_COVERAGE = 0.25`
- `FEATURE_STRONG_HULL_COVERAGE = 0.04`
- `FEATURE_STRONG_PARALLAX_DEG = 0.10`
- `FEATURE_ROT_DIFF_HARD_MAX_DEG = 25.0`
- `FEATURE_MARKER_EPI_HARD_MAX_PX = 3.0`
- `FEATURE_SCALE_MAX_EDGE_CV = 0.30`
- `FEATURE_SCALE_MAX_MARKER_REL_MAD = 0.25`

這些值是目前實驗資料上的保守初始值，不代表已完成跨相機、跨距離及跨材質校準。

### 實際影片驗證

#### 2-pattern 測試

影片：`test_video_Zebra/video_20260601_172436.mp4`

- 選定影格：F41 / F56。
- 共享 marker：2 個。
- Marker RT 對全畫面特徵的極線殘差：`2.398 px`。
- Feature refinement 後最終極線殘差：`0.853 px`。
- Joint PnP baseline：`23.97 mm`。
- Feature 方向加 marker 邊長尺度後：`20.93 mm`。
- 兩個 marker 的尺度 relative MAD：`0.018`。
- Essential 內點：`104 / 184`。
- Feature grid coverage：`0.54`。

#### 1-pattern 測試

影片：`test_video_Zebra/Zebra1_只有1Pattern_1.mp4`

- 選定影格：F84 / F141。
- 共享 marker：1 個。
- Marker RT 對全畫面特徵的極線殘差：`3.359 px`。
- Feature refinement 後最終極線殘差：`1.013 px`。
- Marker PnP baseline：`32.67 mm`。
- Feature 方向加 marker 邊長尺度後：`24.51 mm`。
- Essential 內點：`43 / 109`。
- Feature grid coverage：`0.25`。

測試過程曾發現一個 marker 重投影約 `0.55 px`、但全畫面特徵極線殘差約 `20.54 px` 的假好解。新版流程會避免只因 marker 重投影小就將此類解判定為可靠。

### 驗證完成項目

- `Algorithm/video_pose_analysis.py` 語法編譯檢查通過。
- 兩支 Zebra 入口程式語法檢查通過。
- 1-pattern 實際影片端到端分析完成。
- 2-pattern 實際影片端到端分析完成。
- 測試產生的暫存圖片及 Python cache 已清理。

### 尚未解決與理論限制

- Essential matrix 只能估計平移方向，公制尺度仍需依賴已知 marker 邊長或其他尺度來源。
- Marker 都位於同一側時，尺度可以交叉驗證，但物理幾何條件仍弱於 marker 分散在畫面兩側。
- 大量特徵落在同一平面、重複紋理、非剛體物體、強烈反光、motion blur 或 rolling shutter 都可能使 Essential matrix 失準。
- 目前是 pair-wise Essential refinement，尚未做多幀 bundle adjustment。
- 現有測試證明幾何一致性改善，尚未使用已知深度治具量化毫米級 ground-truth 誤差。
- `rt_reliable=True` 只代表通過目前軟體門檻，不等於已獲得絕對精度保證。

### 建議後續驗證

- 使用多個已知深度平面或標準治具建立批次測試集。
- 分別統計 1-pattern、2-pattern 同側、2-pattern 分散及多-pattern 的深度 bias 與標準差。
- 將 `rt_quality` 與每次點測量誤差寫入 CSV，分析哪些門檻最能預測真實深度誤差。
- 有足夠資料後，再以 ground-truth 調整 inlier、coverage、parallax、marker epi 與尺度一致性門檻。
- 下一階段可考慮固定參考幀的多幀 feature tracks 與局部 bundle adjustment。

---

## 後續紀錄範本

```text
## YYYY-MM-DD - 修改標題

### 修改目的
- 問題與預期結果

### 影響檔案
- 檔案路徑

### 實作內容
- 行為、公式、門檻或輸出欄位變更

### 驗證
- 測試影片或資料
- 修改前後數據
- 語法或自動測試結果

### 已知限制
- 尚未處理的風險
```
