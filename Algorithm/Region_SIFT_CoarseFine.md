# Region-SIFT 兩階段搜尋

主程式預設 `two_stage_search=True`；在「SIFT 參數」改成 False，可回到原本完整搜尋。程式庫 `DEFAULT_CONFIG` 保持 False，以維持既有呼叫者行為。

## 流程

1. 依 UI 基礎設定建立 anchor 與 descriptor。粗搜只取極線中心線，預設每 2 px 評分；沿線全長沿用 `search_length_px`（主程式目前 95）。
2. 依 Objective 選最佳位置，但以 GroupScore 的左右下降幅度判斷是否存在明顯低谷。需要足夠有效樣本、兩側肩部、可被細搜涵蓋的谷底，以及與其他分離低谷有足夠差距。
3. 低谷不明確且 `coarse_side_rescue=True` 時，加搜原搜尋帶兩側邊線（5 px 寬時 across = -2、+2）。保留中心線結果，不重算相同候選。
4. 仍不明確時，cell 的寬高各加 15 px、每格點數加 2，重新取點並重搜整段極線；最多擴大 2 次（共 3 輪）。新增點不保證包含前一輪原本的點。各輪不以原始分數大小互相比高低。
5. 找到明確低谷後，以同一輪 anchor 做局部細搜。預設沿線中心 ±10 px、步距 1 px，共 21 個位置；垂直極線寬度 5 px（以原極線為中心），即最多 5×21。細搜不超出原始搜尋帶，且保留整段粗搜候選參與最終評分與模糊性檢查。
6. 最後仍須通過原有 BestG／ObjRatio 等門檻，以及最終低谷檢查。細搜最佳點在沿線局部邊界或之外時拒絕。細搜失敗不自動增加第三次擴大或強制採用候選。

沒有可評分候選、反光或完整 support 不足等失敗，與「曲線模糊」分開記錄，不盲目擴大以繞過遮罩／幾何限制。

## 可調參數（全部在 SIFT 參數視窗）

| 名稱 | 預設 | 意義 |
|---|---:|---|
| two_stage_search | True（主程式） | False 為舊完整搜尋 |
| coarse_along_step_px | 2 | 粗搜沿線步距 |
| coarse_side_rescue | True | 中心線低谷不明確時補搜兩側 |
| fine_half_length_px | 10 | 細搜沿線半長；包含中心與兩端 |
| fine_width_px | 5 | 細搜寬度，限制在原搜尋带 |
| fine_step_px | 1 | 細搜兩方向步距 |
| adaptive_max_expansions | 2 | 可設 0、1、2 |
| adaptive_cell_increment_px | 15 | 每輪 cell 寬、高增量 |
| adaptive_points_increment | 2 | 每輪每格點數增量 |
| valley_min_relative_depth | 0.08 | 谷底低於較低一側肩部至少 8% |
| valley_shoulder_distance_px | 10 | 各側尋找肩部的距離 |
| valley_level_fraction | 0.5 | 谷底至肩部高度的 50% 以下，視為谷底區 |
| valley_min_side_samples | 2 | 每側肩部範圍最少有效樣本 |
| valley_min_valid_fraction | 0.7 | 沿線至少 70% 位置有可評分候選 |
| valley_max_basin_ratio | 0.95 | 最佳與另一分離低谷的 Objective 比值上限 |

低谷门檻是待資料校準的起始值，不是正確率保證。若使用 `keep_best_count` 固定值，擴大點數後仍保留這個數值；希望保留比例隨點數調整時，使用 `keep_best_count=None` 搭配 `keep_best_ratio`。

## Debug 與效能

- `[Region-SIFT Search]` log 列出每輪大小、點數、候選數、新 descriptor rows、候選快取命中、相對低谷深度、分離低谷比值、耗時與判斷原因。
- 新增 `Region-SIFT coarse-to-fine search` 視窗：每輪獨立一張曲線。藍色為中心線粗搜、紫色為含側線補搜、橘色為細搜加全域粗搜；綠色陰影為谷底範圍，橘色陰影為細搜範圍。
- 曲線只顯示已評分樣本。原診斷熱圖的灰色格也可能是「尚未評分」，不是都代表反光／無效。只看局部細搜不能證明全域唯一解。
- 拖曳高度剖面回放會保留這些歷程。即使擴大後無 descriptor 結果，仍可看到之前各輪的曲線及最後失敗原因。
- 同一次影像對的左右金字塔各至多建立一次、右圖 warp 建立一次；同一輪的相同候選重用尺度、角度、descriptor 與評分。擴大後換了一組 anchor，候選快取重新建立。
- 預設無擴大／無補搜時，95 px 粗搜約 49 點，細搜最多 105 點，重疊位置不重算；加速幅度仍取決於影像、有效候選及擴大次數。困難點可能更慢。

這是近似搜尋，不保證與完整搜尋同解；2 px 粗搜可能錯過非常窄或偏離中心線的低谷。擴大 cell 也可能跨越不同深度，違反共同位移近似，因此只能增加資訊，不能保證中心 P 更準。請以同影格、同點的 GT 高度、錯誤接受率及耗時比較兩個模式。

此次未加入次像素影像精修；回轉到右圖後的浮點座標與現有三角化流程保持不變。

## 特徵細分計時

主 UI 量測後會額外印出 `[Fxx Region-SIFT 特徵細分]`，分左／右圖統計：

- `pyramid`：初始模糊、Gaussian 各層、DoG 響應與平滑、梯度圖、support 資料整理。
- `frames`：遮罩積分圖、座標去重、尺度響應取樣、support 檢查、尺度選擇／回退、角度估計、可靠性與輸出整理。
- `angle`：中心與視窗、座標網格／Gaussian 權重、梯度強度／atan2、直方圖累加／平滑、主峰／次峰。
- `descriptor`：輸入準備、KeyPoint 列表、OpenCV `SIFT.compute`、輸出檢查／正規化。

`input_points` 是新候選送進 frame 估計的 anchor 請求數；`unique_points` 是每次呼叫內去重後數量的加總，**不是跨粗搜／細搜／擴大輪次的全域去重數**。`angle calls` 才是實際方向估計呼叫數；`window_pixels` 是这些方向視窗讀取像素數的加總，包含重複讀取。`new candidates` 是通過初步篩選後首次進入 frame 評估的候選數，可能還會在完整 support 檢查被拒絕，因此不一定等於 descriptor rows 除以點數。

所有子計時已包含於原本大項，**不可再次加到總耗時**；`angle` 又是 `frames` 的子項。子項百分比以自己所屬函式總時間為分母。粗搜／細搜／擴大各階段明細另存於 `search_history[*].detail_profile`，總表為它們的加總；候選快取命中不再計入新 frame/descriptor 工作量。

OpenCV 呼叫時間包含其內部前置與 descriptor 計算，Python 層不能進一步拆開，未將此值假稱為純 descriptor 核心耗時。攤提每 row、每 batch、每 unique 點的時間都不是固定單點成本。計時本身有少量額外開銷；本次沒有修改尺度、角度、取樣、匹配及快取算法。
