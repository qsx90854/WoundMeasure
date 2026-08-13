# 雙目影片模擬獨立單目時序 Pose 實驗：實作計畫

> 實作狀態（2026-08-11）：第一個可執行版本已完成於
> `analyze_independent_monocular_temporal_pose_charuco_gt.py`。目前包含每眼獨立
> ChArUco GT、ID2／ID5 時序 IPPE 選枝、每眼獨立 ID2↔ID5 relation consensus
> 與 8 點 joint PnP、ID2＋同眼 SIFT 的 Essential／Sampson 局部精修、Grid
> dark-pixel 白化、公式化 Excel 成功率、圖表、CSV 與兩支診斷影片。第一版採
> offline temporal 分析；小 Pattern 外圍 1.6 mm blocks 尚未作為 Pose 點，完整
> 多 landmark 滑動視窗 BA／線上 causal 版則保留為後續比較項。
>
> 最終檢查後已補上：ID2／ID5 專用 APRILTAG corner refinement、固定且不依
> GT／JSON 換幀的 reference N、SE(3) rigid-motion 外推、低視差／低 inlier／
> Sampson P90 SIFT 退化 gate、descriptor 支援範圍排除、ID2／ID5 遮罩完整性
> gate、平滑後 marker 重投影複驗，以及 SIFT-assisted 與 marker fallback 分組統計。

## 1. 實驗目標

建立一支獨立 Python 分析程式，載入：

- Side-by-side（SBS）雙目相機影片。
- 相機標定 JSON（左右內參、畸變、已知左右外參）。
- 大型 ChArUco Grid 規格 metadata。
- 小型 ArUco ID2 與 ID5 規格 metadata。

程式必須把左、右影像視為兩條**完全獨立的單目影片**。每一路只能使用自己過去與目前的影像估計 Pose，解算期間不得使用已知雙目 RT、另一眼特徵或另一眼 Pose。

大型 ChArUco Grid 只負責產生每一眼、每一幀的高品質 Pose ground truth（GT）。已知雙目 JSON 外參只負責複驗 GT 與評估待測結果，不得參與待測方法的選枝、初始化或最佳化。

主要比較四組方法：

1. `ID2_TEMPORAL`：只使用 ID2 與時序性。
2. `ID5_TEMPORAL`：只使用 ID5 與時序性。
3. `ID2_ID5_TEMPORAL`：同時使用 ID2、ID5 與時序性，但不假設兩張 Pattern 共面，也不預先知道兩者相對位置。
4. `ID2_WOUND_SIFT_TEMPORAL`：ID2、傷口區域 SIFT tracks 與時序性；大型 Grid、ID5 與小 Pattern 紋理不得混入 SIFT 證據。

每一組方法都會分別輸出左相機軌跡與右相機軌跡，再進行：

- 各眼 Frame-to-Frame Pose 對 ChArUco GT 的誤差比較。
- 同步左右 Pose 組成的 left-to-right RT 對 JSON 外參的誤差比較。
- 時序方法相對於單幀原始解的改善量與成功率。

---

## 2. 已確認的 Pattern 規格

### 2.1 大型 ChArUco Grid

規格來源：

- `gen_charuco_a3_png.py`
- `charuco_a3_12x8_pattern.json`

目前 metadata 記錄：

| 項目 | 數值 |
|---|---:|
| Dictionary | `DICT_4X4_100` |
| Squares | 12 × 8 |
| ChArUco internal corners | 77 |
| ArUco marker count | 48 |
| Marker ID | 20–67 |
| Square length | 29.0 mm |
| Marker length | 22.0 mm |
| Board physical size | 348 × 232 mm |
| Legacy pattern | false |

分析程式應讀取 metadata JSON 作為權威規格，不應把這些數值再次寫死在分析程式內。如此即使之後重新產生不同尺寸的 ChArUco 板，也能由 metadata 同步。

### 2.2 小型 ID2／ID5 Pattern

規格來源：

- `aruco_id2_12_25mm_corner_blocks.json`
- `aruco_id5_12_25mm_corner_blocks.json`

兩者目前規格相同，只有 ID 不同：

| 項目 | 數值 |
|---|---:|
| Dictionary | `DICT_4X4_100` |
| ArUco ID | 2／5 |
| 中央 ArUco 黑框尺寸 | 8.25 × 8.25 mm |
| 完整印刷 Target 尺寸 | 12.25 × 12.25 mm |
| 外圍角落黑塊 | 4 個，1.6 × 1.6 mm |
| Target DPI | 1200 |

第一版 Pose 解算使用 OpenCV 偵測到的**中央 ArUco 四角點**，其 3D 邊長必須是 8.25 mm，不能誤用完整 Target 的 12.25 mm。

外圍四個黑塊不是 ArUco code 的一部分，`detectMarkers()` 不會回傳它們。若要把外圍黑塊納入角點精修，必須另做 custom detector；此項列為後續擴充，不能假裝現有 ArUco 角點已經使用外圍黑塊。

### 2.3 ID2／ID5 產圖狀態

`gen_aruco_id2_corner_blocks_png.py` 的檔名、module docstring 與部分 log 寫 ID2，但目前程式常數實際為：

```python
ARUCO_ID = 5
OUTPUT_PNG = "aruco_id5_12_25mm_corner_blocks.png"
```

目前 ID2 與 ID5 的 PNG／metadata 已由使用者分別手動修改同一支程式並正確產生，工作區中的兩份 metadata 也已確認各自記錄 ID2、ID5，因此這不構成本次實驗阻礙。上述 source 狀態只保留為日後維護提醒；分析程式以兩份 metadata 的實際內容為準，不要求先修改產生器。

---

## 3. 座標系與核心公式

統一採用：

```text
X_camera = R_camera_from_world @ X_world + t_camera_from_world
```

以某一方法的共同世界座標 `W` 表示，第 `i` 幀左、右相機 Pose 分別為：

```text
T_Li_W
T_Ri_W
```

同步影像推導出的 left-to-right RT：

```text
T_Ri_Li_est = T_Ri_W @ inverse(T_Li_W)
```

對 ID2-only，`W = ID2`；對 ID5-only，`W = ID5`；對雙 Marker 與 ID2+SIFT，固定以 ID2 為 gauge／世界原點。

已知 JSON 外參為：

```text
T_R_L_json
```

每幀 Stereo closure error 由 `T_Ri_Li_est` 與 `T_R_L_json` 比較。

各眼第 `N` 幀到第 `i` 幀的待測相對運動：

```text
T_Ci_CN_test = T_Ci_W_test @ inverse(T_CN_W_test)
```

ChArUco GT 的相對運動：

```text
T_Ci_CN_gt = T_Ci_Grid_gt @ inverse(T_CN_Grid_gt)
```

因此小 Pattern／傷口與大型 Grid 之間不需要預先量測 3D 相對位置；在同一路相機的 Frame-to-Frame 相對運動中，各自世界座標會消去。

---

## 4. 資料隔離原則（不可違反）

程式內部應明確分成三條資料路徑。

### 4.1 GT 路徑

可使用：

- 原始左／右影像。
- ChArUco ID 20–67 與 ChArUco corners。
- 左右內參與畸變。

輸出：

- 每幀獨立的 `T_L_Grid_gt`。
- 每幀獨立的 `T_R_Grid_gt`。
- ChArUco corner 數量、空間覆蓋率、重投影 RMS／P95。
- 由兩個獨立 Grid Pose 組成的 `T_R_L_grid_closure`。

JSON 外參只能在兩個獨立 Grid Pose 都算完後，用於複驗 `T_R_L_grid_closure`，預設不得拿來聯合精修 GT。否則會把待複驗的答案反過來強加到 GT。

### 4.2 待測單目路徑

左、右串流各自建立獨立 estimator state：

- 不共享 IPPE 分支。
- 不共享時序 Pose。
- 不共享 ID2↔ID5 相對 Transform。
- 不做左右 SIFT matching。
- 不使用 JSON Fundamental／Essential matrix。
- 不使用 JSON Baseline、Translation 方向或 Rotation 作為 gate。

### 4.3 評估路徑

只有在某方法的左右 Pose 已經獨立完成後，才允許：

- 組合左右 RT。
- 對 JSON 外參算 Rotation／Translation／Baseline error。
- 對 ChArUco temporal GT 算每一路的 Frame-to-Frame error。

建議在程式架構上讓 evaluator 只接受已完成的結果資料，不把 JSON extrinsic 傳進任何 test estimator constructor，以降低無意間洩漏答案的風險。

---

## 5. 大型 ChArUco GT Pose

每一幀、每一眼獨立執行：

1. 在原始灰階影像偵測 `DICT_4X4_100` Marker。
2. 建立與 metadata 完全一致的 `cv2.aruco.CharucoBoard`（包含 ID 20–67 與 `legacy_pattern=false`）。
3. 使用 `cv2.aruco.CharucoDetector` 偵測／插值 ChArUco corners。
4. 取得 ChArUco 3D internal-corner 座標與影像座標。
5. 使用 `cv2.solvePnPRansac(..., SOLVEPNP_ITERATIVE)` 排除錯誤角點。
6. 使用 `cv2.solvePnPRefineLM()` 做最終 Pose 精修。
7. 用原始內參與 distortion 重新投影，計算 RMS、median、P95 與 max error。
8. 記錄角點在板面與影像中的空間覆蓋率，不只看 corner count。

GT frame 建議具備可參數化 gate：

- 最少 ChArUco corner 數。
- 最少板面 X/Y span 或 occupied-cell coverage。
- 最少影像 quadrant coverage。
- 最大重投影 RMS／P95。
- 所有使用角點須在相機前方。
- 左右獨立 Grid Pose 組成的 Stereo closure 不得嚴重偏離 JSON。

Gate 初值不應視為真理，先由首批影片的分布決定；程式需把被拒絕原因逐幀輸出，而不是靜默丟棄。

GT 的兩種用途：

1. **每一路時間軌跡 GT**：比較同一眼不同 Frame 的相對運動。
2. **Stereo GT 自我檢查**：左右獨立 Grid Pose 組成 RT，與 JSON 外參比較。

若 ChArUco closure 對 JSON 的誤差本身就很大，該 Frame 不應用來評估小 Pattern 方法。

---

## 6. 方法一與二：ID2-only／ID5-only＋時序性

每一路相機分開處理。每幀先執行：

1. 只保留指定 ID。
2. 對中央 ArUco 四角點做動態 window 的 `cornerSubPix`。
3. 使用 8.25 mm 正方形 3D 點。
4. 使用 `solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` 保留所有正深度分支。
5. 計算每個分支的重投影誤差、邊長一致性、角點品質與可見尺寸。

只有單一 Marker 時，各幀的 marker reprojection objective 在數學上彼此獨立；如果只是把所有幀放進同一個 optimizer、卻沒有時序項，結果等同逐幀 PnP，並不會因為「放了很多幀」自動變準。

因此時序優化必須明確加入 trajectory constraint：

- 上一幀／前兩幀的 SE(3) constant-velocity prediction。
- 相鄰幀速度變化成本。
- 二階運動（acceleration）成本。
- IPPE branch 切換成本。
- Marker 消失時的短期 propagation，但須標記為 propagated，不能冒充直接觀測。

建議使用 causal sliding-window 或 beam／Viterbi branch tracking：

```text
cost = marker_reprojection
     + lambda_velocity * SE3_velocity_residual
     + lambda_acceleration * SE3_acceleration_residual
     + lambda_branch_switch * branch_discontinuity
```

必須同時輸出：

- `raw_single_frame` 最佳分支結果。
- `temporal_selected/refined` 結果。
- temporal prior 對結果移動了多少。

如此才能確認改善來自合理消歧，而不是過度平滑。

---

## 7. 方法三：ID2＋ID5＋時序性（不假設共面）

每一張小 Pattern 只在自己的 local coordinate 中假設四角點共面：

```text
ID2 corners: z = 0 in marker-2 coordinates
ID5 corners: z = 0 in marker-5 coordinates
```

不得把八個角點直接放在同一個 `z=0` 平面，也不得假設兩張 Pattern 的中心距離、相對 Rotation 或貼附高度已知。

以 ID2 為世界原點，新增一個固定但未知的 6-DoF 參數：

```text
T_ID2_ID5
```

初始化方式：

1. 在同時看見 ID2、ID5 的 Frame，分別保留 IPPE branches。
2. 列舉合理 branch pair，得到該 Frame 的候選 `T_ID2_ID5`。
3. 跨多幀做 SE(3) robust clustering／medoid，找出一致的初始相對 Transform。
4. 進入 sliding-window joint BA，同時最佳化：
   - 每幀 camera-from-ID2 Pose。
   - 固定的 `T_ID2_ID5`。
   - 每幀 IPPE branch／outlier 權重。
5. ID2 暫時消失但 ID5 可見時，可透過已估出的 `T_ID2_ID5` 維持 ID2 世界座標中的相機 Pose。

左、右 estimator 必須**各自估自己的** `T_ID2_ID5`，不能共享。雖然物理上的 Transform 相同，但左右共享會形成 cross-camera constraint，讓實驗不再等同兩條獨立單目影片。

分析完成後可以額外比較左、右各自估出的 `T_ID2_ID5` 差異，作為演算法穩定性診斷，但不可在解算時互相修正。

---

## 8. 方法四：ID2＋傷口 SIFT＋時序性

### 8.1 證據來源

此方法允許：

- ID2 中央四角點與已知 8.25 mm 尺度。
- 同一路影片中傷口／皮膚表面的 SIFT tracks。
- 同一路過去 Frame 的 causal temporal state。

此方法禁止：

- 大型 ChArUco Grid 的 SIFT 特徵。
- ID5 與其四個外圍黑塊的 SIFT 特徵。
- ID2 紋理本身被再次當成 SIFT 特徵（避免同一證據重複加權）。
- 左右相機跨眼 SIFT matching。

### 8.2 正確的 Grid 消除順序

同一幀必須保留兩份影像：

1. `original_image`：供 ChArUco GT 與 ID2／ID5 偵測。
2. `sift_sanitized_image`：只供 SIFT。

不可先把 Grid 塗白再偵測 GT。

第一版採用使用者建議的 **ChArUco 範圍內黑色像素白化**。由 GT 路徑先在原圖完成 ChArUco Pose 後，建立有效 board 範圍，只把其中屬於 ChArUco 黑格、Marker 黑色 bit 與黑白交界殘留的像素設成 255。白色格原本已接近 255，不需要整板填白。

不能只使用過低的單一灰階 threshold，因為抗鋸齒、失焦、MP4 壓縮與 motion blur 會在黑白交界留下灰色 L 型／十字型輪廓，SIFT 仍可能抓到角點。建議遮罩由以下資訊組成：

```text
board_black_erase_mask
    = valid_projected_charuco_range
    AND expected_or_observed_dark_pixels
```

接著對 `board_black_erase_mask` 膨脹約 1–3 px（依原始解析度參數化），涵蓋灰色邊緣，再執行：

```text
sift_sanitized_image = original_image.copy()
sift_sanitized_image[dilated_board_black_erase_mask] = 255
```

`expected_or_observed_dark_pixels` 可先以 board 內的 adaptive dark-pixel segmentation 實作；較嚴謹版本再由 metadata 建立 ChArUco binary template，依該幀 board homography／Pose 投影黑色區域，並和實際暗像素取交集。後者能降低把非 Grid 區域誤判成 Grid 的機率。

ID2、ID5 與外圍黑塊同樣可能產生 SIFT，因此不論 Grid 黑像素白化是否已碰到它們，仍要另外將兩個完整 12.25 mm target 的投影 polygon 膨脹後排除；Marker 偵測一律使用未修改的 `original_image`。

此方法不會像整板填白那樣必然刪除整個傷口模型，但仍有一項限制：若傷口／皮膚本身存在接近黑色的區域，而且剛好位於 board 範圍內，純 observed-dark threshold 可能把它一起白化。程式需統計白化前後傷口區 SIFT keypoint 保留率；若誤刪明顯，再啟用 wound／skin foreground protection mask：

```text
final_erase_mask
    = dilated_board_black_erase_mask
    AND NOT protected_wound_or_skin_foreground
```

因此 wound segmentation／reference polygon 改為可選的第二層保護，而不是第一版的硬性前置需求。

每幀 sanitization 後必須做 leakage diagnostics：

- 在 sanitized image 上重新跑 Grid ArUco detector，理想結果為 0 個 ID20–67。
- 在 sanitized image 上不得偵測到 ID5。
- 統計 SIFT keypoints 落在 Grid-only 區域的數量，理想結果為 0。
- 統計黑白交界附近（例如原 Grid edge 3–5 px band）的 SIFT keypoint 數量，理想結果為 0。
- 統計傷口／皮膚區域白化前後的 SIFT keypoint 保留率，避免去 Grid 時同時刪掉大部分待測物特徵。
- 保存 mask／sanitized 診斷影片供目視確認。

### 8.3 SIFT 時序 Pose

每一路獨立執行：

1. 在 sanitized image 內偵測 SIFT。
2. 先做相鄰 Frame matching／tracking，再形成跨多幀 feature tracks。
3. 使用 ratio test、mutual check、Essential matrix／Sampson residual 排除錯配。
4. 以 ID2 Pose 提供世界座標與公制尺度。
5. 三角化具足夠 parallax 的傷口特徵。
6. 在 sliding-window BA 中共同最佳化相機 Pose 與傷口 landmark。
7. 使用 Huber／Cauchy robust loss；低 parallax、負深度或高 reprojection track 移除。
8. ID2 暫時不可見時，可由既有 feature map 短期 propagation；須限制最大 gap。

此方法的平移尺度由 ID2 提供，不能只依 `recoverPose()` 的單位 Translation direction 當成毫米。

---

## 9. Ground truth 與評估指標

### 9.1 ChArUco GT 品質

逐眼逐幀記錄：

- detected Grid marker count。
- ChArUco corner count／inlier count。
- board-space X/Y coverage。
- image-space coverage／quadrant count。
- reprojection RMS、median、P95、max。
- GT Pose 是否通過品質 gate。

同步左右 GT closure 記錄：

- Grid-derived vs JSON Rotation error（deg）。
- Translation L2 error（mm）。
- Translation direction error（deg）。
- Grid-derived Baseline（mm）。
- Baseline delta（mm）與 absolute error（%）。

### 9.2 各眼時間軌跡誤差

對每個方法、每一眼、每個 reference-to-current pair：

- relative Rotation error（deg）。
- Translation L2 error（mm）。
- Translation direction error（deg）。
- relative baseline／motion magnitude error（mm、%）。
- SE(3) log translation／rotation norm。
- Relative Pose Error（RPE）。
- 長時間 drift 對 frame gap 的趨勢。

所有方法與左右眼固定使用使用者指定的同一個 reference frame N。若某方法無法解出 N，該方法的 temporal pair 直接標成 unavailable；不可依 ChArUco GT 或 JSON closure 挑 fallback frame，以免用答案挑題。reference row 不列入 requested、GT-valid、estimated 或 comparable 的分母。

### 9.3 每幀 Stereo closure

對每個方法，將獨立左、右 Pose 組成 RT，與 JSON 比較：

- Rotation error。
- Translation L2 error。
- Translation direction error。
- estimated Baseline。
- Baseline delta／absolute error percent。
- closure 的 frame-to-frame 標準差與 jump。

### 9.4 成功率

Excel `Settings` 放可修改門檻，例如：

- Rotation error threshold（預設 2°）。
- Baseline error threshold（預設 5%）。
- GT reprojection gate。
- Test reprojection／epipolar gate。

以公式輸出：

- Rotation pass rate。
- Baseline pass rate。
- Both-pass rate。
- 對全部 requested frames 的成功率。
- 對 GT-valid frames 的成功率。
- 對 estimator-produced frames 的成功率。

三種分母必須分開，避免只看成功解出的少量 Frame 而高估方法能力。

---

## 10. 預計輸出

### 10.1 Excel

建議採 long-format，以 `frame_index + eye + method` 為主要 key：

1. `Settings`
2. `Video Info`
3. `GT Frame Poses`
4. `GT Stereo Check`
5. `Method Frame Poses`
6. `Temporal RPE`
7. `Stereo Closure`
8. `Method Summary`
9. `SIFT Diagnostics`
10. `Mask Diagnostics`
11. `Charts`
12. `Protocol`

Charts 至少包含：

- 四方法的 Rotation error vs frame。
- 四方法的 Baseline error % vs frame。
- 左／右 temporal RPE vs frame gap。
- Both-pass rate 方法比較。
- GT ChArUco reprojection／closure quality。
- SIFT inlier count、parallax、Grid leakage count。

### 10.2 CSV

- GT frame pose CSV。
- Method frame pose long-format CSV。
- Stereo closure CSV。
- Temporal RPE CSV。

### 10.3 診斷影片

1. `*_gt_charuco.mp4`
   - 左右 ChArUco corners、inliers、Pose axes、reprojection RMS。
2. `*_small_patterns.mp4`
   - ID2／ID5 subpixel corners、IPPE branch、marker size。
3. `*_sift_sanitized.mp4`
   - 上／左：original。
   - 下／右：sanitized image、allowed mask、SIFT inliers。
4. `*_pose_comparison.mp4`
   - 四方法與 GT 的每幀誤差、有效／拒絕原因。

所有線與角點應在放大 patch 後以 1 px 線寬繪製，避免遮住真正角點位置。

---

## 11. CLI 與參數規劃

目前可執行介面範例：

```powershell
python analyze_independent_monocular_temporal_pose_charuco_gt.py VIDEO.mp4 `
  --calibration calibration_result_HBVCAM_4M2214HD-2-v11.json `
  --charuco-metadata charuco_a3_12x8_pattern.json `
  --id2-metadata aruco_id2_12_25mm_corner_blocks.json `
  --id5-metadata aruco_id5_12_25mm_corner_blocks.json `
  --reference-frame 50 `
  --start-frame 0 --max-frames 300 --frame-step 1 `
  --sift-roi-left 500,250,800,600 `
  --sift-roi-right 500,250,800,600 `
  --protect-sift-roi-dark `
  --diagnostic-video `
  --output experiment_result.xlsx
```

SIFT 去 Grid 需有獨立參數，例如：

- `--sift-grid-removal-mode black_pixels|black_pixels_with_wound_protection`
- `--grid-black-threshold-mode adaptive|fixed|template_intersection`
- `--grid-black-threshold`
- `--grid-black-dilate-px`
- `--wound-mask-mode none|model|polygon|roi`
- `--wound-mask-dilation-px`
- `--left-wound-polygon`
- `--right-wound-polygon`

程式啟動時應驗證：

- SBS 寬度為偶數。
- JSON 內參尺寸與實際單眼影像尺寸相容。
- ChArUco metadata 與 OpenCV board IDs 完全一致。
- ID2／ID5 dictionary、ID、中央 Marker 尺寸正確。
- Grid IDs 20–67 不與小 Pattern IDs 2／5 衝突。
- reference frame 必須在影片索引範圍內；若該幀缺少 GT 或某方法需要的小 Pattern，該方法會明確標為 unavailable，不自動換 reference。

---

## 12. 實作階段

### 階段 A：規格與偵測快取

- Metadata loader 與一致性檢查。
- SBS 影片順序讀取。
- 每眼每幀以 SUBPIX pass 偵測 ChArUco board，另以 APRILTAG corner refinement pass 專門重抓 ID2／ID5；後者取代小 Pattern 的 SUBPIX 角點，避免外圍黑塊拉偏角點。
- 將偵測結果快取，四種方法不得重複跑 detector。

### 階段 B：獨立 ChArUco GT

- 左右各自 solvePnPRansac＋RefineLM。
- GT 品質 gate。
- Grid Stereo closure 對 JSON 複驗。
- 先輸出短片 preview，確認 GT 可靠後才繼續。

### 階段 C：單 Marker 時序方法

- ID2-only 與 ID5-only 共用 estimator class，但 state 分離。
- 單幀 IPPE branches。
- causal branch tracking／motion prior。
- raw vs temporal 結果並列。

### 階段 D：雙 Marker 非共面方法

- 每一路獨立初始化 `T_ID2_ID5`。
- sliding-window joint BA。
- Marker 缺失與 outlier handling。
- 左右估出的 marker-to-marker Transform 只做事後比較。

### 階段 E：SIFT 隔離與時序 BA

- 由 ChArUco 範圍建立 dark-pixel／template-intersection mask，並膨脹到涵蓋灰色邊緣。
- 將 Grid 黑色像素白化，產生 sanitized image。
- 必要時才加入 wound／skin foreground protection mask。
- Grid／ID5／marker-texture leakage test。
- 傷口 SIFT keypoint retention test。
- SIFT tracks、三角化與 local BA。

### 階段 F：評估與報告

- Temporal GT comparison。
- Stereo closure comparison。
- Excel 公式、Summary 與 Charts。
- CSV 與四類診斷影片。

### 階段 G：驗證

- 合成投影測試：已知 ChArUco、小 Marker、相機軌跡與 Stereo RT。
- Transform composition／inverse 單元測試。
- No-GT-leakage 測試：移除 JSON extrinsic 後，test estimator 結果必須不變。
- SIFT leakage 測試：sanitized image 上 Grid keypoint count 應為 0。
- 短實拍影片（例如 20 Frame）完整 smoke test。
- Excel XML、公式與圖表渲染檢查。
- 再執行完整 300 Frame。

---

## 13. 效能策略

四方法、兩眼與 GT 若各自重跑偵測會非常慢，因此：

- ChArUco／ID2／ID5 每幀只偵測一次並快取。
- SIFT descriptor 每眼每幀只算一次。
- 一般 Frame 使用 KLT／descriptor track；只有 Keyframe 重建 SIFT。
- 每一路 estimator 使用 5–15 個 Keyframe 的滑動視窗。
- BA 使用 sparse Jacobian、robust loss 與前次結果初始化。
- 大型 Grid GT 可以每幀算，但不得放進 test BA。
- 診斷影片最後由快取結果第二次順序讀片產生。

先保證方法正確，再用 profiling 決定預設 `keyframe_stride` 與 window 大小；不可為了速度改成左右共享資料。

---

## 14. 已知不足與風險

### 14.1 ChArUco 是高品質參考，不是無誤差真理

其準確度仍受以下因素影響：

- 印刷縮放不是 100%。
- 紙張／壓克力板不平。
- 鏡面反光、模糊、遮擋。
- Board 只剩局部且角點集中一側。
- 相機內參與 distortion 不準。

因此必須以重投影、coverage 與 Stereo closure gate 篩選 GT Frame。

### 14.2 GT 與待測方法共用同一份內參

若內參有系統誤差，GT 與小 Pattern Pose 可能同方向偏移，實驗無法完全觀察此 common-mode error。可另外用不同標定批次或高精度外部量測交叉檢查。

### 14.3 單 Marker 時序性不是免費幾何資訊

只有 ID2 或 ID5、沒有 SIFT tracks 時，各 Frame PnP 原本彼此獨立。時序改善主要來自 branch continuity 與運動平滑先驗，而不是新增 3D correspondence。先驗過強會把真實快速動作錯誤地平滑掉，必須同時報告 raw 結果與 prior correction。

### 14.4 ID2／ID5 的固定相對關係假設

雙 Marker 方法不假設共面，但仍假設兩張 Pattern 在影片期間剛性固定。如果傷口模型變形、貼紙滑動或皮膚彎曲改變兩者相對 Pose，固定 `T_ID2_ID5` 會失效。程式應輸出 marker-relation residual／drift，不能硬套固定關係。

### 14.5 每張 Pattern 自己的平面假設

中央 8.25 mm ArUco 必須近似平面。貼在明顯曲面上會造成四角點 3D 模型錯誤；Pattern 越小影響可能越低，但仍會直接改變 Pose 與公制尺度。

### 14.6 黑色像素白化可行，但必須處理邊緣殘留與誤刪

將 ChArUco 範圍內的黑色像素改白，原理上會讓主要黑白角點消失，而且比整板填白更能保留傷口模型，是合理的第一版方案。風險在於失焦、抗鋸齒與 MP4 壓縮會留下灰色邊框；若沒有適度 threshold 與 1–3 px 膨脹，SIFT 仍可能抓到殘影。反過來，threshold 太寬又可能把傷口／皮膚的深色紋理一起白化。必須同時檢查 Grid-edge leakage 與 wound-feature retention，必要時才增加 foreground protection mask。

### 14.7 小 Pattern 與外圍黑塊會洩漏進 SIFT

即使不顯式偵測 ID5，SIFT 仍可能把 ID5 code 或四個外圍黑塊當作自然特徵，造成 `ID2+傷口 SIFT` 實際偷用了 ID5。必須將兩個完整 12.25 mm target 的投影區域膨脹後排除。

### 14.8 Extra corner blocks 尚未被 Pose 使用

現有 OpenCV ArUco detector 只回傳中央 Marker 四角點。若期待角落黑塊提高精度，需另行實作與驗證 custom subpixel geometry；第一版數據應明確標示「central ArUco corners only」。

### 14.9 SBS 同步與 rolling shutter

若左右感光時間不同，而拍攝時相機快速移動，即使真正 Stereo RT 固定，左右獨立 Pose 組成的 closure 也會偏離 JSON。實驗應確認硬體同步，並記錄曝光時間、FPS、motion blur；初期先以慢速移動拍攝。

### 14.10 雙相機同步 Pair 仍不完全等於單相機跨時間 Pair

幾何組合公式相同，但左右相機可能有不同焦距、畸變、曝光與感光噪聲；真正單目跨幀則是同一顆 sensor、但場景可能隨時間改變。因此 Stereo closure 是很好的已知答案測試，但仍需 ChArUco temporal GT 來驗證同一眼的 Frame-to-Frame 軌跡。

### 14.11 Stereo closure 可能看不到共同誤差

如果左右 Pose 都帶有相同的世界座標偏差，組成相對 RT 時偏差可能抵消。這就是為什麼必須同時比較每一路 temporal motion 對 ChArUco GT，而不能只看 JSON closure。

### 14.12 JSON 外參也可能漂移

焦距調整、相機外殼受力、溫度或重新插拔都可能讓實際 Stereo RT 與原標定不同。ChArUco closure 若長時間穩定偏離 JSON，需先重新標定或判斷硬體是否改變，不能直接把所有差異算成待測算法錯誤。

### 14.13 傷口 SIFT 的物理限制

高光、濕潤表面、低紋理、非剛性變形與遮擋都會降低 track 品質。Essential matrix 只能提供 Translation direction；公制 scale 必須由 ID2 與 joint BA 維持。Parallax 太小的 Frame 即使 SIFT 數量很多，也未必能準確估 Translation。

---

## 15. 建議的拍攝前檢查

1. ChArUco 以 100% actual size 列印，實測 29 mm square 與整板尺寸。
2. 板面固定平整，避免壓克力反光正對相機。
3. ID2／ID5 中央 Marker 實測為 8.25 mm。
4. ID2、ID5 與傷口模型在整段影片保持剛性固定。
5. 大板至少三個方向都有可見 corners，不要全部被模型遮在同一側。
6. 左右影像同步，先以慢速移動。
7. 拍攝包含位置、距離與傾斜變化，不只重複相同視角。
8. 保存一小段 Grid-only／無傷口模型影像，可協助驗證 board appearance 與 SIFT mask leakage。

---

## 16. 完成定義

程式完成需同時滿足：

- 四種方法、左右兩眼完全獨立解算。
- ChArUco GT 每眼獨立，JSON 只做事後複驗。
- ID2／ID5 雙 Marker 不假設共面或已知相對位置。
- SIFT 證據中無大型 Grid、ID5 與重複 Marker 紋理。
- 每幀可追溯使用了哪些角點、feature、時序 prior 與拒絕原因。
- 同時輸出 individual temporal GT error 與 Stereo closure error。
- Excel 門檻公式可重算，圖表可正常開啟。
- 合成測試、短影片 smoke test、完整影片與 no-GT-leakage test 全部通過。
