# Region-SIFT 即時參數

展開主畫面的藍色控制區，按「Region-SIFT 參數」（位於 U/V 座標重現旁）。
視窗列出目前 `RegionSIFTConfig` 的設定值與中文說明，可捲動修改。

- 按「套用」後從下一次量測生效；請重新點圖或使用「座標重現」。原有結果不會自動重算。
- 「取消」不修改設定。「還原開啟時的數值」只還原表單，須再按套用。
- 本功能只修改本次執行的設定，重新啟動後回到主程式 `REGION_SIFT_CONFIG`。
- Block 誤差統計從選角點到完成／取消期間禁止修改，確保一份報告使用同一組參數。
- 參數變更會印在終端機；Block 報告的設定 metadata 使用套用後的參數。

常用欄位：

| 參數 | 意義 |
|---|---|
| `search_length_px` / `search_width_px` | 沿極線／垂直極線的搜尋範圍 |
| `search_along_step_px` / `search_across_step_px` | 搜尋步長 |
| `points_per_cell` | 每格取點数；總數 = rows × cols × 每格點數 + P |
| `scale_keypoint_sizes_px` | 逗號分隔候選尺度，如 `3.2, 4, 5, 6.4` |
| `max_objective_score_ratio` | ObjRatio 大於此值時拒絕 |
| `max_group_score` | BestG 大於此值時拒絕 |
| `keep_best_count` | 固定保留點數，填 `None` 改用 `keep_best_ratio` |

允許 `None` 的欄位也可以留白。布林參數以 True/False 選單操作。
不合法輸入會顯示錯誤且整組不套用。啟用 descriptor 正規化會改變分數尺度，需同步調整 BestG 上限。
SIFT 計算器會使用新的 octave 層數／sigma；左圖特徵快取包含完整設定作為識別，不會重用不同設定的 descriptor。

## 反光排除

主畫面的 `Reject SpecPts` 開啟（預設）時，Region-SIFT 使用既有左右圖的
空間性與時間性反光聯集遮罩。`Show Spatial`／`Show Temporal` 只控制顯示。

- 左圖每格只從可用位置選高／中／低梯度點，維持每格點數與中心 P。
- P 不可移位；P 被遮罩覆蓋或任何一格取不足點時，回報匹配失敗。
- 右图遮罩經同一個幾何變換轉到 warp 座標。任一 anchor 不合格，淘汰整組候選，
  不使用反光點補數、不降低 28 點數量，也不改變共享位移原則。
- `specular_check_support=True`（預設）：取點先排除連最小 support 都無法避開反光的位置，
  再在各點可用尺度中選擇完整 support 沒有反光的尺度。右圖亦檢查完整 support，
  並保守考量影像 warp 插值所讀取的來源像素。此模式可能提高失敗率。
- `specular_check_support=False`：只避開特徵點中心；descriptor 周圍仍可能包含反光。
  可在即時參數視窗修改此選項，從下次量測生效。
- 關閉 `Reject SpecPts` 完全停用此排除。開關／左圖遮罩內容變更後，舊左圖特徵快取不會命中。

Log 的 `[Region-SIFT Specular]` 顯示開關、support 模式、中心點排除候選數，
以及完整 support 檢查排除數（後者也包含影像邊界／warp 無效位置）。
拒絕時不三角化，沿用原本 UI 的失敗原因顯示。
