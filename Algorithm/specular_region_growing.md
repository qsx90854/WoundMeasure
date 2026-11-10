# Legacy region growing

實作位於 `specular_detection.py` 的 `compute_specular_mask_bgr_block_adaptive`。
`BlockSpatialSpecularConfig.enable_region_growing` 預設 False；原呼叫介面與 uint8 mask 回傳格式不變。

## 獨立 UI

在 Legacy「反光參數」將 `enable_region_growing=True`，按套用重算。
所有共用這個參數視窗的 UI 都能編輯；設定只保留本次執行。
初始生長參數：radius=3、local_hot_min=5、gray_min=80、s_max=150、whiteness_max=90、color_max=60。
先比較開關差異，再調參；這些生長值尚未經 GT 調參驗證。

## GradSIFTDebug 接入

本次沒有修改 `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py`。
未指定新欄位的既有設定仍關閉生長。日後可在其 `SPATIAL_BLOCK_CONFIG` 建構子加入：

```python
enable_region_growing=True,
grow_radius_px=3,
grow_local_hot_min=5.0,
grow_gray_min=80.0,
grow_s_max=150.0,
grow_whiteness_max=90.0,
grow_color_max=60.0,
```

不需新增呼叫流程。共用的「舊版反光參數」視窗也已可於本次執行啟用。
只對全圖區塊 Legacy 路徑生效；Fixed、V2、有效傷口 AI 遮罩走的 wound-adaptive 路徑不使用這個生長功能。

## 定義與限制

種子是原本亮度／白度／突出條件選出的 morphology 前遮罩。弱候選必須通過灰階下限、局部背景差下限，以及 S 或 RGB 白度條件。
以 8 鄰接最多生長 grow_radius_px 步；不會跳過不合法像素。距離及色差參考原始種子，避免逐步漂移。
色差採各 BGR 通道除以該 pixel 最大通道再乘 255，與最近原始種子比較最大通道絕對差。這不是 Lab DeltaE，也不是 HSV hue。
最近種子的色差檢查若失敗，不會嘗試更遠的種子。所有既有種子保留，生長結果再經原本 opening/closing/dilation。
因此後續 morphology 仍可能刪除細線或向外擴張；細線試驗建議 open=1、close=1、dilate=0。

`return_debug=True` 額外提供 `region_growing_seed_mask`、`region_growing_added_mask`，可供後續 UI 分色呈現。
`selected_before_morphology` 為生長後、morphology 前的遮罩；關閉功能時與舊版完全相同。

## 驗證

新增 5 項 unittest 測試：細線、斜線、距離、斷開區域、色差漂移、邊界、空種子、0 半徑、debug 與參數驗證。
既有 block spatial 12 項、settings 3 項測試通過。
額外以備份版實作比對主程式／平衡設定在真實 F550、隨機及 1x1 輸入的關閉結果與既有 debug 遮罩，完全一致。
