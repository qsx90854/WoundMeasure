# Region-SIFT 局部視差斜面微調（實驗功能）

主畫面「視差微調: Off/On」；預設 Off。只作用於 Region-SIFT，
OpenCV 與自製遮罩 descriptor 都使用目前選定的 backend。
修改開關不會重新計算歷史量測；請在相同 frame、相同 u/v 重新量測作 A/B 比較。

## 流程

原本搜尋（完整搜尋，或粗搜＋補兩側＋擴大＋細搜）完成且通過原接受條件後，
只對原最佳解做一次微調。原本被拒絕的解不會被救回，也不改變擴大條件。

每個 anchor 的附加沿線位移為 `c + a*(x-Px) + b*(y-Py)`。
左圖 anchors、descriptors 不改動；右圖尺度與方向仍逐點重新估計。
`c` 是 P 的修正量；橫向偏移保持原值。修正後中心使用既有 H_lr 轉回右圖座標，
交給原三角化、品質評估與量測流程。

這是 warp 後局部平行極線切線上的一階殘餘位移模型，並非精確的三維平面模型，
也未建模每個 anchor 的極線方向差異、深度跳變或 descriptor 內部變形。
後續應以實際已知高度測試驗證，分數改善不保證高度改善。

1. 在原匹配附近逐點取樣距離表，預設 ±2 px、每 1 px 一筆（5 組位置）。
2. 使用 TrimK + CellAll 與斜率懲罰，搜尋 c/a/b；不先各點獨立選最低分。
3. 最多每層 125 個模型、預設 3 層；模型比較只查距離表，不重算 descriptor。
4. 對選定模型的真實座標再算一次 descriptor，以完整 Objective＋形變懲罰驗證改善。
5. 不足、無改善、觸及修正邊界或遮罩/support 不通過時，保留原匹配。

右圖 pyramid/context 在當次量測內重用。候選 descriptor 距離表第一版重新計算，
沒有跨原候選 descriptor 快取重用；28 點預設至多 5×28＋28=168 新 rows，
64 點至多 384 rows。不另外搜尋第二個低谷。
對缺失的整組候選採保守處理，不跨無效取樣插值；可能因此跳過微調。

## 調整（SIFT 參數內標記「局部微調」）

- `local_plane_refine=False`：與主 UI 按鈕同步。
- `refine_radius_px=2`：每個 anchor 最大沿線修正；碰到上限不採用。
- `refine_sample_step_px=1`：取樣間距；半徑須是間距整數倍，總區間數 2..16。
- `refine_model_levels=3`：查表搜尋層數 1..5。
- `refine_slope_penalty=0.5`：正規化 ROI 邊緣位移係數的平方懲罰；不懲罰中心平移。
- `refine_min_gain=0.5`：原 Objective 減去新 Objective 與形變懲罰的最低改善量。

斜率懲罰及改善門檻使用 descriptor 分數單位；若啟用 normalize_descriptors，須相應調小。
取樣間距與模型層數不代表已保證的次像素精度；OpenCV 內部取樣及尺度切換可能讓成本不平滑。

## 比較與 Debug

`[Region-SIFT LocalPlane] applied/skipped` 顯示是否套用、原因、額外 rows、耗時；
另印中心沿線修正、斜率、修正前後 G/Objective 與扣除懲罰後的改善。
method 含 `+LocalPlane` 才表示三角化使用了新座標。

舊 best/second、曲線及逐點資訊保持原剛性搜尋，不與微調後分數混合。
右圖橘色星號為新 P'，橘色加號為新 anchors。diagnostics 標題另列微調後單影格高度，
原 best 高度可作修正前比較。微調資料保存在 `region_debug['refinement']`，
含原/新座標、距離表、模型、實算分數與狀態，可隨既有 debug 結果保存。
