# Video Pose RT：算法流程與耗時計時說明

本文對應：

- `Algorithm/video_pose_analysis_temporal_unified_pattern_guided_local_window.py`
- `zebra_0825v2.py` 的 `FRAME_PAIR_SELECTION_MODE = "original"` 或 `"angle_guided"`
- `profile_video_pose_timing.py` 非 UI 實測工具

目的有兩個：第一，說明每個算法區塊實際做什麼；第二，讓耗時 Log 可以直接定位瓶頸，同時避免多執行緒時間被重複計算。

## 一、計時原則

1. 所有數值皆為 wall-clock time（牆鐘時間），不是 CPU time。
2. 平行執行的 ArUco／SIFT 以主執行緒「等待整個工作池完成」的時間計算，不將每條 worker 的執行時間相加。
3. 每個父階段的子項互不重疊；正常完成時，子項加總應等於父階段（只可能有浮點數捨入差）。
4. Log 依實際資料流順序輸出，而非依耗時大小排序。
5. 第一次執行可能包含影像解碼、OpenCV初始化與檔案快取的冷啟動成本；比較版本時應使用同一影片、同一模式、同一設定，至少重複數次。
6. 某項為 `0 ms` 不代表流程失敗。例如 `original` 模式不做角度掃描；Core pair 幾何已合格時也不需要額外 Pattern probe。

## 二、整體流程（依算法邏輯順序）

| 順序 | StageTimer 顯示名稱 | 具體工作 | 主要輸出 |
|---:|---|---|---|
| 0 | 影片索引建立 | 建立可隨機存取的影片影格索引，取得影格數與尺寸 | lazy frame reader |
| 1–6 | ArUco偵測+配對搜尋（含極線重排） | 從稀疏 ArUco pose、時序DP、Local window、IPPE分支與SIFT幾何中選出 frame pair | `best_start`、`best_end`、初始 `R/t/baseline` |
| 7 | 混合RT精修+次佳打包 | 以 Marker雙向投影與SIFT robust residual聯合精修，並檢查次佳候選 | 精修後相對位姿與候選組 |
| 8 | 最終RT閉環+品質驗證 | 將相對位姿重新錨定回共同 reference；計算閉環、Marker與Feature最終品質 | `rt_quality`、`valid_poses` |
| 9 | RT SIFT診斷檔輸出 | 輸出keypoint、匹配、Essential/recoverPose/optimization mask與逐點極線誤差 | `*_rt_sift_diagnostics.txt` |
| 10 | 診斷狀態封裝 | 整理 angle/pattern/local/temporal diagnostics | result diagnostics |
| 11 | 影像去畸變+輸出幀準備 | 對最終左右影格建立/套用remap | UI使用的校正影像 |
| 12 | 基準平面建立 | 優先用reference marker的雙端三角化角點；否則由 `T_B<-W` 映射固定平面；最後才fallback legacy PnP | `global_plane_n/c` |
| 13 | 深度匹配特徵延後計算 | 明確不預先建立深度點擊匹配描述子，避免拖慢UI啟動 | 空的延後計算快取 |

## 三、Frame pair搜尋的六個父階段

### 1. 角度掃描／端點提案

`original` 模式只從前、後兩段依固定fraction取得原本的稀疏候選，所以耗時接近零。

`angle_guided` 模式會：

1. 均勻抽取 coarse frames。
2. 解碼影格並偵測 ArUco。
3. 以已標定內參、8.25 mm Pattern及IPPE/PnP估計相機對Pattern的距離與入射角。
4. 自動比較影片是由15°掃向35°，還是反方向掃描。
5. 在最接近目標角度處做有上限的鄰近補掃，最後依角度誤差、PnP重投影與候選品質排序。
6. 無共同 Marker、pose不足或角度超出容許值時，明確標示 `FALLBACK_ORIGINAL`。

子計時：

- `1.1 影格解碼+ArUco偵測`：只計 coarse/refined scan 新影格的解碼與偵測。
- `1.2 PnP角度+方向判斷+候選排序`：pose量測、掃描方向判斷、目標角度誤差與排序。

### 2. Marker map + 時序姿態DP

這一階段不是逐幀獨立選最小PnP誤差，而是先建立共同世界座標，再選整條時間路徑：

1. 對稀疏 probes 解碼與偵測 ArUco。
2. 由多幀共視關係建立 marker graph，排除不穩定的marker-to-marker關係，固定一個reference marker並形成剛體 Marker map。
3. 每幀保留IPPE多個pose hypotheses，全部轉成共同的 `T_camera<-reference` 座標。
4. 以二階時序DP同時考慮單幀量測成本、相鄰速度和平滑加速度，抑制單一Pattern的IPPE共軛解突然跳支。

子計時：

- `2.1 稀疏probe解碼+ArUco偵測`
  - `2.1.1 影片seek/grab+目標幀解碼`
  - `2.1.2 BGR轉灰階`
  - `2.1.3 CLAHE對比增強`
  - `2.1.4 ArUco候選偵測`
  - `2.1.5 cornerSubPix角點精修`
  - `2.1.6 Cache+資料整理`
- `2.2 共視圖建圖+剛體Marker map`
- `2.3 IPPE逐幀姿態假設`
- `2.4 時序DP最佳路徑`
- `2.5 資料整理+診斷輸出`

單一Pattern與雙Pattern走同一條流程。雙Pattern若同幀可見，會增加共視約束、減少歧義；兩個Pattern不需要共平面，只需要marker map能以實際剛體關係連通。

### 3. Pattern-guided候選擴充

先用現有Core pair快速估計幾何是否適合三角測距：

- camera-center baseline；
- 預估場景深度；
- 影像重疊率；
- 三角化夾角與低分位視差；
- predicted depth sigma。

若Core pair已足夠，流程立即結束；若不足且沒有超過probe上限/時間預算，才根據相機中心運動趨勢提出少量額外ArUco probes，重建對應的pose hypotheses與時序DP。這裡不做全畫面SIFT。

子計時：

- `3.1 Core pair幾何可用性評估`
- `3.2 額外probe+時序路徑重建`
- `3.3 狀態更新+控制開銷`

### 4. Local window + KLT重排

此階段只圍繞暫定左右端點的小範圍鄰幀工作，目的不是重新掃完整支影片，而是避免剛好選到模糊、角點抖動或IPPE分支不穩的那一幀。

1. 對稀疏候選先算暫定pair品質，取得兩個window中心。
2. 解碼有限半徑／stride的鄰幀，以中心Pattern協助ArUco偵測。
3. 依固定 Marker map建立鄰幀IPPE hypotheses。
4. 只在相鄰幀間做KLT forward-backward追蹤；KLT在此提供短時間連續性，不直接決定公制baseline。
5. Local DP結合marker reprojection、pose連續性、清晰度、KLT一致性及很弱的中心距離tie-breaker，保留少量端點候選。
6. 把Local observations併回全域候選後再跑一次全域DP，維持跨階段的branch一致性。

子計時：

- `4.1 暫定端點pair評分`
- `4.2 鄰幀解碼+ArUco偵測`
- `4.3 鄰幀IPPE姿態假設`
- `4.4 相鄰幀KLT追蹤`
- `4.5 Local DP+清晰度/KLT重排`
- `4.6 合併鄰幀後重建全域DP`
- `4.7 詳細選幀原因Log輸出`
- `4.8 Window建立+資料整理`

啟用Local window時，終端會另外列出每側的輸入中心、候選逐項成本、KLT可用性與失敗原因、Local最佳Frame、保留候選，以及後續Pair/IPPE＋SIFT後的最終Frame。`local_window_diagnostics['selection_comparison']`也會保存Local前、Local最佳與最終pair的比較資料。

### 5. Pair / IPPE分支枚舉評分

對Local window保留下來的端點及其pose分支作真正的跨段組合：

1. 封裝時序DP選中分支與其餘IPPE備選分支。
2. 對 `frame_A × frame_B × branch_A × branch_B` 計算相對 `R/t`。
3. baseline由共同世界座標下的camera centers距離計算；這和剛體座標轉換下的 `||t_rel||` 等價。
4. 套用baseline硬門檻、端點Marker直接重投影、重疊率、視差、預估深度誤差、清晰度與Pattern觀測品質。
5. 排序後以「幀對」為單位配置Top-K SIFT預算，避免多個IPPE分支占滿所有名額。

子計時：

- `5.1 端點時序分支封裝`
- `5.2 Pair×IPPE組合幾何評分`
- `5.3 排序+Top-K預算配置`
- `5.4 控制開銷`

### 6. SIFT + Essential極線重排

只有Top-K frame pairs進入較昂貴的SIFT：

1. 在設定ROI內提取SIFT，遮掉Pattern區域，避免Marker本身同時主導公制PnP與獨立Feature驗證。
2. BFMatcher L2 KNN；通過ratio test、mutual best與網格空間均衡。
3. `findEssentialMat(RANSAC)` + `recoverPose`，評估inlier ratio、空間覆蓋、parallax及homography/planar degeneracy。
4. 比較Marker RT與Feature幾何的極線殘差、旋轉一致性及三角化角度，再重排IPPE/frame pair候選。
5. 第一候選已具幾何一致性時可略過第二幀對，但同一幀對的IPPE branches仍共用同一次SIFT匹配接受評分。

子計時：

- `6.1 SIFT特徵提取`
- `6.2 ratio+mutual+網格匹配`
- `6.3 Essential RANSAC+recoverPose`
- `6.4 極線/視差評分+最終重排`
- `6.5 候選控制+快取開銷`

## 四、Frame pair選出後如何得到最終RT與baseline

選出的ArUco相對位姿不是直接原樣輸出。流程會把Marker約束與SIFT內點放入聯合精修：

- Marker使用雙向transfer/reprojection residual，保留8.25 mm Pattern提供的公制尺度。
- SIFT使用robust Sampson/epipolar residual，改善旋轉與平移方向，但不能單獨提供公制尺度。
- least-squares同時微調相對旋轉與公制平移；只有Marker及Feature最終品質均通過才標示 `rt_reliable=True`。
- 最後將精修相對位姿重錨回共同reference，驗證rotation/translation closure。

因此時序性不只是排除共軛解；它也用於建立共同Marker map、抑制逐幀pose抖動、穩定Local端點選擇，並提供IPPE branch prior。最終精度仍需由Marker重投影與獨立Feature極線幾何共同驗證。

## 五、實測範例（original模式）

測試條件：

- 影片：`test_video_Zebra/video_20260825_164357.mp4`
- 257 frames，1920×1080
- Zebra calibration，Pattern 8.25 mm
- `range_mode=half_half`
- `FRAME_PAIR_SELECTION_MODE=original`
- 單次實測；數值只用於展示如何閱讀Log

整體：3.185 s；其中 frame pair搜尋 3063.0 ms（96.2%）。

| Frame pair父階段 | 耗時 | 搜尋占比 | 此次解讀 |
|---|---:|---:|---|
| 1. 端點提案 | 0.1 ms | 0.0% | original模式不掃角度 |
| 2. Marker map+時序DP | 976.5 ms | 31.9% | 其中probe解碼/ArUco佔95.9% |
| 3. Pattern-guided | 3.1 ms | 0.1% | Core geometry可用，未加probe |
| 4. Local window+KLT | 1527.6 ms | 49.9% | 多個子步驟均有成本，並非只有KLT |
| 5. Pair/IPPE枚舉 | 100.6 ms | 3.3% | 幾乎全在組合幾何評分 |
| 6. SIFT+Essential | 455.0 ms | 14.9% | 其中SIFT提取佔68.7% |
| 7. 其餘控制 | 0.3 ms | 0.0% | 可忽略 |

Local window細分：暫定pair 247.5 ms、鄰幀解碼/ArUco 289.2 ms、鄰幀pose 22.3 ms、KLT 263.0 ms、Local重排 368.2 ms、全域DP重建 337.2 ms。

這代表若要降低耗時，第一優先應量測「影格解碼+ArUco快取」及Local window內各重複幾何評分的可重用程度；不能只因SIFT名稱看起來較重，就先降低SIFT品質或放寬幾何門檻。

## 六、Log與result欄位

主程式會依順序印出父階段及縮排子項。程式化分析可直接讀：

- `result['analysis_total_elapsed_s']`
- `result['pair_search_timing_s']`：六個父階段與跨階段overhead
- `result['pair_search_detail_timing_s']`：所有細分子階段

也可執行：

```powershell
.\Scripts\python.exe profile_video_pose_timing.py `
  test_video_Zebra\video_20260825_164357.mp4 `
  --mode original `
  --output rt_timing_profile_original.json
```

若改測角度選幀，將 `--mode original` 改成 `--mode angle_guided`。Profiler只呼叫RT分析，不啟動Zebra量測UI。

### SIFT專用ROI與縮放

兩支Zebra主程式均由下列變數控制RT階段的SIFT範圍：

```python
ENABLE_RT_SIFT_ROI = True
RT_SIFT_ROI_RATIO = (0.10, 0.10, 0.80, 0.80)
RT_SIFT_IMAGE_SCALE = 0.5
```

- `RT_SIFT_ROI_RATIO`格式為原始影像正規化的 `(x, y, width, height)`；預設代表中央80%。
- 實際順序是先依 `RT_SIFT_IMAGE_SCALE` 縮放，再換算並裁切SIFT ROI，SIFT不會替ROI外區域建立金字塔。
- Keypoint在配對前會補回裁切位移並換回原始解析度座標，後續Essential與極線計算仍使用原始相機內參座標系。
- 此ROI只限制RT候選重排及精修所用的SIFT；ArUco偵測與pair geometry維持全畫面。
- 啟用時錄影預覽會畫出ROI與縮放後的crop尺寸；框線只畫在顯示副本，不寫入原始錄影。
- 設定 `ENABLE_RT_SIFT_ROI = False` 會恢復縮放後全畫面SIFT，同時停止顯示ROI框。

Profiler也可單獨驗證此路徑：

```powershell
.\Scripts\python.exe profile_video_pose_timing.py `
  test_video_Zebra\video_20260825_164357.mp4 `
  --mode original `
  --feature-roi 0.10 0.10 0.80 0.80 `
  --feature-scale 0.5
```

## 七、效能調整時不可犧牲的驗證

- 同一批影片比較總耗時、各父/子項時間、所選frame indices與fallback狀態。
- Marker：最終雙向重投影RMS與point max不可惡化。
- Feature：Essential/recoverPose內點數、空間覆蓋、median/p90極線誤差與parallax不可惡化。
- RT：已知ground truth實驗仍需檢查旋轉誤差、baseline相對誤差；不能只用同一組Marker角點的自我PnP重投影判定準確。
- 不應以縮短deadline後靜默fallback來換取較漂亮的耗時；Log必須保留 `status` 與 `fallback_reason`。
