# 自製遮罩式 SIFT：第一版

主畫面在「拖曳高度剖面」上方新增 `Descriptor: OpenCV / 自製` 按鈕。
僅影響 Region-SIFT，不修改 Grad-SIFT、RT 或其他演算法。
預設仍為 OpenCV。切換與 SIFT 參數套用共用相同配置，下一次量測生效；
Block 統計或拖曳模式進行中禁止切換，避免同批混用算子。

## 自製模式

- 使用現有空間性反光遮罩（不是時間性與空間性聯集），即使 Reject SpecPts 關閉仍套用。
- 原始「去畸變後、未經 CLAHE」灰階輸入；左右尺度／角度及 descriptor 使用遮罩式 maps。
- `specular_check_support` 僅控制 OpenCV，不能用來關閉自製遮罩。
- 左圖點擊 P 在反光中心直接拒絕，左圖參考點仍避開反光中心。
- 右圖反光中心不再整組一票否決：尺度／角度可參考未反光鄰域，覆蓋不足的點對標示無效。
- Gaussian 正規化卷積僅使用有效觀測，無效位置不重新變成有效；梯度 stencil 必須有效。
- Gaussian、尺度、角度、128 維三線性 histogram 均不讀遮罩內亮度。
- 使用全解析度尺度圖、不做 octave 降採樣，不是 OpenCV bitwise 重製版。
- 浮點 descriptor 維持 128 維、預設 L2 norm 512，但分數分布不保證與 OpenCV 相同。

## 比較與保守拒絕

保留未正規化 histogram 與 4×4 子區有效比例。共同覆蓋使用
`max(0, coverage_left + coverage_right - 1)` 保守下界。達到子區門檻才參與
雙邊重新正規化與 L2 比較，整體共同覆蓋不足則拒絕該點對。
這是子區層级，而不是逐像素共同遮罩重新計算；仍可能受到部分遮擋的分布差異影響。

群組要求至少 `ceil(N * masked_min_group_fraction)` 個有效點對，且至少原 best-K 數量，
每個 Region cell 至少 `masked_min_points_per_cell` 個有效點對；預設 28 點至少 21 點、每 cell 至少 1 點。
最多容許 `masked_max_deficient_cells` 個 Region cell 未達到這個每格下限仍放行候選
（預設 1；設為 0 等於恢復「每格皆須達標」的原始規則）——這是為了處理反光範圍剛好
整格覆蓋、擴大取點範圍也無法補救的情況；未達標的 cell 不會被跳過或補值，其列仍以
下方的最大距離懲罰留在 CellAll，缺失代價不會消失。
不刪除矩陣列。無效點距離固定為非負 descriptor 的最大 L2（預設 sqrt(2)*512），
不進 KEEP，仍在 CellAll 中留下缺失代價；無效點不參與尺度／方向一致性。
有效點距離使用 `d + weight*(1-common_fraction)*(max_distance-d)` 加上缺失覆蓋懲罰。
因此自製模式的 debug distance 是含懲罰的有效距離，不再是純 L2。
平坦但有效的區域仍是有效資訊；全零群組仍使用原有拒絕機制。
`left/right_descriptors` 是各自完整有效部分的 descriptor，實際 matching L2 使用共同
子區重新正規化，故不應直接相減 debug 的兩個矩陣來重建 matching 分數。

## SIFT 參數（全部以【自製算子】標示）

| 參數 | 預設 | 意義 |
|---|---:|---|
| use_masked_sift | False | 主按鈕同步的模式開關 |
| masked_extra_margin_px | 0 | 原偵測遮罩之外額外膨脹，原遮罩已經膨脹 |
| masked_min_blur_weight | 0.5 | 平滑有效權重最低值 |
| masked_min_valid_fraction | 0.6 | 每點加權有效覆蓋最低比例 |
| masked_min_cell_fraction | 0.8 | descriptor 子區共同覆蓋最低比例 |
| masked_min_common_fraction | 0.5 | 篩選子區後整體共同覆蓋最低比例 |
| masked_descriptor_clip | 0.2 | 初次單位正規化後截斷值 |
| masked_min_group_fraction | 0.75 | 群組有效點比例（也不可少於原 best-K） |
| masked_min_points_per_cell | 1 | 每個 Region cell 至少的有效點對 |
| masked_max_deficient_cells | 1 | 最多容許幾個 Region cell 未達上一行門檻仍放行候選（0=每格皆須達標）|
| masked_missing_penalty_weight | 0.25 | 有效點部分缺失懲罰權重；無效點仍固定最大距離 |

降低覆蓋門檻會放寬，但資訊不足誤配風險增加。原有 scale/angle、batch、搜尋與
GroupScore 門檻仍共用。BestG=500、ratio=0.95 沒有自動改動，需以實際資料重新驗證。
這版以正確性優先，不承諾比 OpenCV 快。

## Debug / 邊界

終端列印 backend、遮罩／CLAHE 行為、覆蓋拒絕候選數，以及最佳候選每點覆蓋 min/mean。
debug 保留左圖子區覆蓋、左右每點覆蓋及右圖共同覆蓋；固定 N 行序。
快取鍵包含全部配置及遮罩，切換不混用。兩階段重用同次呼叫 context。
新模組要求呼叫者提供對齊遮罩，缺失不會靜默退回 OpenCV。

每個 coarse / side / fine stage 另列獨立候選漏斗：requested、inBounds、warpSupport、
centerMask、cache/new、frameOK、newGroupOK、最後 scoreable。自製模式再列每個新候選的
有效點數與 deficient-cell 數 min/median/max、四類點對無效來源、當前門檻、沿線有效率，
並依實際停止原因列出對應 UI 參數。各 stage 數字是局部統計；後方舊的 `[Masked SIFT]`
數字仍是整次搜尋累計，兩者不可相加。

右圖 warp 之前擴張遮罩以涵蓋插值 footprint，warp 後無效區域同樣排除。
「遮罩內像素不影響輸出」保證針對固定遮罩、固定點及本模組輸入成立。
反光偵測本身、之前的去畸變插值仍在此保證之外；漏偵測像素也無法自動排除。

離線測試：`Scripts/python.exe -m unittest tests.test_masked_sift_descriptor -v`
