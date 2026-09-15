# 全圖分塊空間性反光偵測

主程式的 `Adaptive Spatial` 預設開啟：有有效傷口遮罩時使用原本傷口動態模式；
沒有遮罩（包含 AI 停用、沒有偵測結果、遮罩不足 20 像素）時改用全圖規則分塊。
關閉 `Adaptive Spatial` 仍使用原本的固定門檻模式。
切換此開關在 `ENABLE_WOUND_AI=False` 時不會要求 AI 推論。

分塊是影像像素網格，與手動圈選的 6×4 實體高度模型無關。
預設每塊 64×64 px；1920×1080 影像為 30 欄×17 列，末列採實際剩餘像素。

每塊分別計算以下候選門檻。表格是原始參考值；實際生效值由主程式
`SPATIAL_BLOCK_CONFIG` 覆寫 class 預設值，使用者調整值優先：

| 門檻 | 百分位 | 下限～上限 |
|---|---:|---:|
| HSV V | 90 | 170～245 |
| RGB 最大值 | 92 | 180～250 |
| 灰階 − Gaussian 局部背景 | 85 | 10～45 |

保留原本三條判定分支，符合任一條件先標為候選（下列數字是原始參考設定）：

1. V 超過當地門檻，且 S ≤ 80。
2. RGB 最大值超過當地門檻，且 RGB 最大值與最小值之差 ≤ 36。
3. 灰階 ≥ max(150, 當地 V 門檻 − 20)，且局部亮度差超過當地門檻。

新增局部亮度突出程度檢查，預設啟用：

```text
D = 灰階 − Gaussian 局部背景
MAD(D) = median(abs(D − median(D)))
每塊突出門檻 = max(prominence_min, median(D) + prominence_mad_multiplier × MAD(D))
保留 = (原始候選 AND D ≥ 當地突出門檻) OR 強反光例外
```

MAD 使用原始中位絕對偏差，沒有乘以 1.4826。
突出門檻與其他門檻一樣，依 `interpolate_thresholds` 設定在區塊間插值。
此檢查套用於三條候選分支，避免「偏白且亮」分支繞過局部亮度檢查。
均勻區域的 D 接近零，通常無法通過；紋理較強的區域 MAD 較大，突出門檻提高。

主程式新增可調參數：

| 參數 | 設定 | 調整效果 |
|---|---:|---|
| `enable_prominence_gate` | True | False 回到原本候選 OR 結果（仍是 64×64） |
| `prominence_min` | 3 | 越低越容易保留微弱亮點 |
| `prominence_mad_multiplier` | 2 | 越低越容易通過紋理區域的檢查 |
| `enable_strong_highlight_exception` | True | 保留接近飽和且近白色的區域 |
| `strong_v_min` | 250 | 強反光例外的 V 下限 |
| `strong_s_max` | 30 | 強反光例外的 S 上限 |
| `strong_whiteness_max` | 20 | 強反光例外的 RGB 最大差上限 |

強反光例外必須同時符合三項門檻；關閉突出檢查時此例外也不作用。
以上是形態學處理前的判定；後續閉運算／膨脹仍可能填補或擴大區域。
`return_debug=True` 額外回傳突出門檻 grid/map、原始候選與形態學前選取結果，供分析。

預設對各塊門檻做雙線性插值，減少方格邊界的門檻跳變。
關閉 `interpolate_thresholds` 則每一塊嚴格共用該塊的門檻。
組合全圖後只執行一次開運算、閉運算與膨脹，不修改原始影像。
這仍是亮度／顏色啟發式偵測，亮白的非反光材質也可能被標記，需以實際影像 tuning。

調參位置：主程式 `SPATIAL_BLOCK_CONFIG`。
所有可用欄位與預設值定義在 `Algorithm/specular_detection.py` 的 `BlockSpatialSpecularConfig`。
這些參數尚未加入 Region-SIFT 的即時參數視窗；修改程式設定後須重新啟動。

使用 `Show Spatial` 查看空間反光遮罩；時間性遮罩仍依既有流程計算，再與空間遮罩聯集。
Region-SIFT 已接入聯集遮罩，透過 `Reject SpecPts` 開關控制取點與匹配排除。
完整 support／僅中心點的選項見 `Region_SIFT_Settings.md`。
