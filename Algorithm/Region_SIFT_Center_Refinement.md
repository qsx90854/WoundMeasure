# 中心精修（CenterRefine）

主 UI「中心精修」或 SIFT 參數 `center_refine_enabled` 控制，預設 False。
啟用按鈕會關閉 LocalPlane；啟用 LocalPlane 也會關閉中心精修。參數視窗若同時設 True，拒絕套用。

## 範圍

先完整執行原本搜尋及匹配接受規則，再只針對原 Top 1 做局部精修。不救回被拒絕的搜尋、不改排名、不覆寫原候選分數。
關閉時不建立中心點組、金字塔或 descriptor。

中心預設在獨立 12×12 px 內分層選 9 個自動點，加 P 共 10 點；分散空間位置並交替選高／中／低梯度，保留最小點距、反光遮罩及有效性限制。
中心區域不隨 coarse-to-fine 的 cell 擴大而擴大。

中心左右圖重新估計 scale/angle 與 descriptor，nominal support radius 上限預設 24px。
此上限與既有 SIFT 尺度設定的定義相同，不含 Gaussian/梯度/DoG halo；log 的 actual_left_support_max_px 才是完整左側讀取半徑。
小 support 減輕跨表面混合，但不保證完全隔離凸起／凹陷。

## 搜尋與評分

- along 初始 ±2px、間距 1px，across 固定。
- 邊界最小值仍在下降且改善量達標，才擴大；每次增加初始半徑，上限預設 ±6px。
- 不越過原全域搜尋帶。重用相同中心點組的候選 cache，已算位置不重算 descriptor。
- 權重 w_i = exp(-distance(P,p_i)^2/(2*sigma^2)) 後正規化。
- 中心分數 = sum(w_i * distance_i)，不是原 TrimK/CellAll Objective。
- 無效點保留最大距離懲罰，不刪點／重正規化權重，避免遮罩較多反而得分更好。
- 至少 max(min_points, ceil(N*min_group_fraction)) 個有效點、足夠有效權重、至少指定象限數，且有效點凸包嚴格包住 P。
- 原 0px 必須可評分。完整區間不得有缺資料；最佳值需在內部，具足夠改善、雙側低谷深度及與排除鄰域後競爭位置的分數差。
- 最後用實際候選位置回轉到原右圖，檢查有限三角化與左右相機前方；主程式再套用其最大深度限制。

原整組 Objective 不必改善，因為精修目標是中心。第一版不擬合中心局部平面、不自動改選 Top 2。

## 結果

- applied：採用修正，method 加上 +CenterRefine。
- fallback：中心證據不足、改善不足、模糊、邊界未解決或幾何失敗，原匹配保留；不表示原結果已被中心驗證。
- skipped：原匹配已被拒絕，不做中心 descriptor。
- discarded：主程式後續原搜尋品質閘門拒絕，不以中心結果救回。

`[CenterRefine]` log 顯示原因、中心分數／位移、額外左右 descriptor rows、耗時及融合前同一基準平面的新舊高度。
原影片同名 txt 的 JSON 新增 center_refinement 摘要；不更改現有 CSV 欄位。

當 ambiguity 視窗開關啟用時，另顯示中心曲線圖：0px 虛線是原候選，綠線是已採用修正。曲線點標示有效點數，缺資料不連線。
既有視差一致性診斷仍是原 rigid Top 1/Top 2，不混用中心分數。

若設定 active 的 confidence_height_span_max_mm，該門檻仍檢驗原 rigid 搜尋候選；log 明確標記不代表中心精修後的不確定度。

## 調參注意

所有新參數均以 center_refine_ 開頭，在「SIFT 參數」內標為【中心精修】。
若改 normalize_descriptors，min_gain/max_score 等絕對分數門檻需一併調整。
此模式改善是否成立必須用已知高度驗證；分數變低不等於高度誤差必然變小。
