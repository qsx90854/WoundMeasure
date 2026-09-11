# Block 高度誤差統計

主程式：`depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py`。

## 操作

1. 展開橘色控制面板，按 **Block 誤差統計**。
2. 在上方左圖依序選 **左上、右上、右下、左下**；6 格方向是橫向，4 格方向是縱向。
3. 檢查預覽網格、黃色真值標籤與紫色取樣點。確認每格九點均位於該格頂面，再按 **開始 216 點統計**。
4. 逐点顯示進度及預估剩餘時間。可按 **取消統計**，於目前點完成後停止；關閉視窗也會保存已完成資料。
5. 統計結束後，按旁邊的 **MAE 著色: Off** 可切換半透明格子填色：`mae_mm < 0.5` 淺綠、`0.5 <= mae_mm <= 1.0` 淺橘、`mae_mm > 1.0` 淺紅（單位 mm）。On 時隱藏各格統計文字與粉紅色取樣點，保留格線及色階圖例；沒有有效量測者不填色。再次按下切回 Off，關閉填色並恢復統計文字與取樣點。

著色使用 **各點絕對誤差的平均 `mae_mm`**，不是有號平均誤差、`abs_mean_height_error_mm` 或 RMSE。此功能只更新顯示，不重新量測、不修改統計或 CSV。取消後的部分結果也可著色，須注意有效點數；開始新一輪時清除舊結果與著色狀態。

批次期間鎖定匹配選項、平面選項、影像切換與其他量測按鈕，避免一輪中途更換條件。
連續計算會關閉；完成後不自動重新啟用。自訂平面尚在選點、手動匹配或 RT Warp 顯示模式時，需先退出該模式。

## 模型與取樣

- 模型：6 欄 × 4 列，寬 120 mm、高 80 mm；每格 20 × 20 mm。
- 真值：逐列由左至右，第一列 `12.5, 12, 11.5, 11, 10.5, 10` mm，最後一列 `3.5, 3, 2.5, 2, 1.5, 1` mm。
- 每格沿兩軸的 25%、50%、75% 交叉取九點，名義上距格邊至少 5 mm。
- 四角 homography 將模型座標投影到左圖。量測中心比照一般左圖點擊四捨五入至像素；CSV 同時記錄原投影中心 `u/v` 與實際量測 `measured_u/measured_v`，不吸附 ArUco 角點。
- `BLOCK_ACCURACY_CONFIG` 可調整格數、格寬高、每軸點數、內縮比例、起始高度及高度差。

**重要限制：** 四角 homography 假設平面格網，但不同高度的 block 頂面並不共平面。斜視角、遮擋或高度視差可能使分格不準；必須先檢查預覽，必要時取消重選或改用較正面的影像。內縮只保證 anchor 中心位於預覽格內，不保證 30×30 anchor 群或 SIFT support 全部位於單格頂面。

## 量測定義

沿用一般點擊的 `do_measure`：前處理、所有右圖候選匹配、三角化與多影格融合均不另寫算法。
批次只略過一般單點 TXT/JSON 存檔及逐點 Debug UI 更新，改寫獨立統計報告。

記錄既有 `height_display_mm`，即未四捨五入的 **Wound Height**，不是相機 Z 距離：

- Custom Plane 已擬合時依既有規則優先，以融合 3D 計算。
- 非 Custom Plane 時依現有 Shared／Pose／Legacy 平面選擇，優先使用同幾何鏈的最佳右圖 3D；套用既有 `DEFAULT_WOUND_HEIGHT_OFFSET_MM`。
- Shared 預設 On；若沒有至少兩個可用共同 pattern 形成平面，依既有邏輯退回 Legacy。此開關仍僅選擇高度基準，不改寫匹配用 homography 或 RT。
- 高度與誤差保留正負號，不取高度絕對值、不另做離群剔除。既有匹配拒絕與融合規則仍生效。

誤差定義：`error_mm = measured Wound Height - true height`。
失敗或無高度的點不代入零，也不納入平均；所有失敗均保留原因與計數。
若最佳右圖匹配失敗、但其他右圖產生有效融合高度，依一般畫面的有效 Wound Height 納入統計；原最佳對失敗原因保留為 `measurement_warning`。

## 輸出

預設目錄：主程式旁的 `measurement_accuracy/<時間戳_唯一識別>/`。

- `points.csv`：每個已執行點的位置、格編號、真值、Wound Height、有號／絕對誤差、成功狀態、失敗原因、平面來源、算法、候選影格、GroupScore、ObjectiveScore 及耗時。每點立即 flush。
- `blocks.csv`：全部 24 格的真值、有效／失敗／未執行數、平均高度、平均高度誤差、平均高度絕對誤差、點誤差 MAE／RMSE、母體標準差。
- `blocks.xlsx`：`Summary` 工作表以 6 組 `GT / Mean_H`、4 列排列 24 格真值與平均量測高度，並包含 `MAE vs True Height`、`STD vs True Height` 兩張可編輯 XY 折線圖；`Chart Data` 工作表保留依真值高度遞增排列的圖表來源數值。
- `summary.json`：整體點誤差統計、各格平均高度誤差的等格權 MAE／RMSE、完整格數、狀態與取消／錯誤原因。
- `run.json`：四角、完整取樣計畫、模型設定、影片、右圖影格與幾何、匹配選項及實際高度基準。

`mae_mm` 是有效點的平均絕對誤差；`abs_mean_height_error_mm` 是單格平均高度的絕對誤差，兩者不同。
CSV 浮點數輸出最多 15 位有效數字，以相容 Excel 的數值精度，避免長數字被匯入為帶單引號標記的文字。負值保留負號、缺值留空；內部計算及 JSON 仍保留原始精度。
`block_mean_mae_mm` 只對至少有一個有效點的格等權平均；須連同 `valid_block_count`、`complete_block_count` 判讀，不能將缺失格視為零誤差。
取消時仍輸出全部格，但未量格的平均為空值。時間戳加唯一識別避免覆寫上一輪資料。

## 驗證

執行 `Scripts/python.exe -B -m unittest tests.test_block_height_accuracy tests.test_block_height_accuracy_ui`。
UI 測試使用實際 callback 的無視窗測試環境與模擬量測，不開啟相機、不執行影片量測。
